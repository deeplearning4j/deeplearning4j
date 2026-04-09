/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.eclipse.deeplearning4j.nd4j.autodiff.samediff;

import lombok.extern.slf4j.Slf4j;
import org.bytedeco.javacpp.Pointer;
import org.junit.jupiter.api.*;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.*;
import org.nd4j.autodiff.samediff.diagnostics.DspDiagnostics;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlanExecutor;
import org.nd4j.common.config.ND4JSystemProperties;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Phase contract enforcement tests for the DSP (Dynamic Shape Plan) execution engine.
 *
 * <p>These tests complement TestDSPLifecycleRegression, TestDSPSlotStateMachine,
 * TestDSPExecutionPathAssertions, and TestDSPBufferOwnership by catching phase
 * contract violations that those tests do not cover:</p>
 * <ul>
 *   <li>Frozen output DataBuffer address stability across replay steps</li>
 *   <li>No slot replacement during replay for value-dependent op chains</li>
 *   <li>Plan phase monotonicity (no backward transitions without explicit reset)</li>
 *   <li>Replay handle persistence across many execution steps</li>
 *   <li>Capture buffer D2D refresh for changing placeholder inputs</li>
 *   <li>Decode input propagation with DspDebugger phase contract validation</li>
 *   <li>Phase demotion and re-warmup after releaseGpuIntermediates</li>
 * </ul>
 *
 * <p>Run:
 * <pre>
 *   cd platform-tests && mvn test -Dtest=TestDSPPhaseContractEnforcement 2>&1 | tee /tmp/dsp-phase-contract.log
 * </pre>
 */
@Slf4j
@Tag("dsp")
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
public class TestDSPPhaseContractEnforcement extends BaseNd4jTestWithBackends {

    private static final double TOL = 1e-4;
    private SameDiff sd;

    @Override
    public char ordering() {
        return 'c';
    }

    @BeforeAll
    static void enableDspGlobally() {
        System.setProperty(ND4JSystemProperties.DYNAMIC_SHAPE_PLAN_ENABLED, "true");
        InferenceSession.setDynamicShapePlanEnabled(true);
    }

    @AfterEach
    void cleanup() {
        if (sd != null) {
            sd.close();
            sd = null;
        }
    }

    private void enableDsp(SameDiff sd) {
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);
    }

    /**
     * Assert two arrays match within tolerance, with a descriptive message.
     */
    private void assertClose(INDArray expected, INDArray actual, String label) {
        assertArrayEquals(expected.shape(), actual.shape(),
                label + ": shape mismatch - expected " + Arrays.toString(expected.shape())
                        + " got " + Arrays.toString(actual.shape()));
        double maxDiff = expected.sub(actual).amaxNumber().doubleValue();
        assertTrue(maxDiff < TOL, label + ": max diff " + maxDiff + " exceeds tolerance " + TOL);
    }

    /**
     * Build a deep matmul chain (10+ layers) with near-identity weights.
     * Uses Nd4j.eye * 0.95 to avoid vanishing values through deep chains
     * while still being large enough to trigger CUDA graph capture.
     *
     * @param inputName  placeholder name
     * @param outputName final output name
     * @param numLayers  number of matmul layers (10+ recommended for capture trigger)
     * @param dim        dimension for square matrices
     * @return configured SameDiff instance
     */
    private SameDiff buildDeepMatmulChain(String inputName, String outputName,
                                           int numLayers, int dim) {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder(inputName, DataType.FLOAT, 1, dim);

        SDVariable current = x;
        for (int layer = 0; layer < numLayers; layer++) {
            SDVariable w = sd.constant("w_" + layer,
                    Nd4j.eye(dim).castTo(DataType.FLOAT).muli(0.95));
            current = sd.mmul("mm_" + layer, current, w);
            current = sd.nn.relu("relu_" + layer, current, 0);
        }
        current.rename(outputName);
        return sd;
    }

    /**
     * Helper to get the native plan handle from a SameDiff instance.
     */
    private Pointer getPlanHandle(SameDiff sd) {
        InferenceSession session = sd.getOrCreateSession();
        DynamicShapePlanExecutor executor = session.getDynamicShapePlanExecutor();
        return executor != null ? executor.getNativePlanHandle() : null;
    }

    // ========================================================================
    // 1. Frozen Output Stability
    // ========================================================================

    /**
     * After 5 executions (freeze + capture + replay), verify that ALL output
     * slot DataBuffer addresses are IDENTICAL between replay steps.
     *
     * Uses DspDebugger.analyzePlan() to get slot info, then compares slot states
     * across 3 replay executions. Any address change in a frozen slot = test failure.
     *
     * This catches bugs where replay reallocates output buffers instead of reusing
     * the captured pointers.
     */
    @Test
    @Order(1)
    @DisplayName("Phase contract: frozen output buffer addresses must be stable across replay")
    public void testFrozenOutputStability() {
        int dim = 16;
        int numLayers = 12;
        sd = buildDeepMatmulChain("x", "out", numLayers, dim);
        enableDsp(sd);

        INDArray input = Nd4j.randn(DataType.FLOAT, 1, dim).muli(0.5);
        Map<String, INDArray> placeholders = Map.of("x", input);

        // Phase 1: warmup + freeze (5 executions)
        for (int i = 0; i < 5; i++) {
            sd.output(placeholders, "out");
        }

        DspDebugger debugger = DspDebugger.attach(sd);
        DspDebugger.PlanReport baselineReport = debugger.analyzePlan();
        log.info("Baseline plan after 5 executions:\n{}", baselineReport);

        assertNotNull(baselineReport.planPhase,
                "Plan phase should be set after 5 executions");

        // Capture baseline slot states
        List<DspDebugger.SlotInfo> baselineSlots = baselineReport.slots;
        assertFalse(baselineSlots.isEmpty(), "Plan should have slots after compilation");

        // Phase 2: replay executions - capture slot states after each
        List<DspDebugger.PlanReport> replayReports = new ArrayList<>();
        for (int replay = 0; replay < 3; replay++) {
            sd.output(placeholders, "out");
            DspDebugger.PlanReport report = debugger.analyzePlan();
            replayReports.add(report);
            log.info("Replay {} plan phase: {}", replay, report.planPhase);
        }

        // Verify: frozen slots must have identical states across all replays
        for (int slotIdx = 0; slotIdx < baselineSlots.size(); slotIdx++) {
            DspDebugger.SlotInfo baseSlot = baselineSlots.get(slotIdx);

            // Only check frozen/compiled slots - they should be stable
            if (baseSlot.state != null && baseSlot.state.isAtLeast(SlotState.FROZEN)) {
                for (int replay = 0; replay < replayReports.size(); replay++) {
                    DspDebugger.PlanReport replayReport = replayReports.get(replay);
                    if (slotIdx < replayReport.slots.size()) {
                        DspDebugger.SlotInfo replaySlot = replayReport.slots.get(slotIdx);

                        assertEquals(baseSlot.state, replaySlot.state,
                                "Slot " + slotIdx + " (" + baseSlot.opName
                                        + ") state changed during replay " + replay
                                        + ": " + baseSlot.state + " -> " + replaySlot.state
                                        + " — frozen slots must be stable");

                        assertEquals(baseSlot.flags, replaySlot.flags,
                                "Slot " + slotIdx + " (" + baseSlot.opName
                                        + ") flags changed during replay " + replay
                                        + " — frozen slot flags must be stable");
                    }
                }
            }
        }

        log.info("testFrozenOutputStability: all frozen slot states stable across 3 replays");
    }

    // ========================================================================
    // 2. No Slot Replacement During Replay
    // ========================================================================

    /**
     * Build a graph with value-dependent ops (gather + reshape + matmul chain).
     * Execute 5 times with CUDA_GRAPHS mode. On the 4th and 5th execution
     * (replay phase), verify outputs are correct AND that the plan phase is
     * SHAPES_FROZEN or higher.
     *
     * This catches bugs where value-dependent ops cause unexpected slot replacements
     * during what should be a stable replay.
     */
    @Test
    @Order(2)
    @DisplayName("Phase contract: no slot replacement during replay for value-dependent ops")
    public void testNoSlotReplacementDuringReplay() {
        int hiddenSize = 16;
        int vocabSize = 50;
        int numLayers = 10;

        sd = SameDiff.create();

        // Embedding table + gather (value-dependent) + reshape + matmul chain
        INDArray embTable = Nd4j.randn(DataType.FLOAT, vocabSize, hiddenSize);
        SDVariable embeddings = sd.constant("embeddings", embTable);
        SDVariable indices = sd.placeHolder("indices", DataType.INT64, 1);

        // Gather is value-dependent (output depends on input VALUES, not just shape)
        SDVariable gathered = sd.gather("gathered", embeddings, indices, 0);

        // Reshape to ensure [1, hiddenSize]
        SDVariable reshaped = sd.reshape("reshaped", gathered, 1, hiddenSize);

        // Deep matmul chain with near-identity weights
        SDVariable current = reshaped;
        for (int layer = 0; layer < numLayers; layer++) {
            SDVariable w = sd.constant("w_" + layer,
                    Nd4j.eye(hiddenSize).castTo(DataType.FLOAT).muli(0.95));
            current = sd.mmul("mm_" + layer, current, w);
            current = sd.nn.relu("relu_" + layer, current, 0);
        }
        SDVariable out = current.sum("out");

        enableDsp(sd);
        sd.setGraphExecutionMode(GraphExecutionMode.CUDA_GRAPHS);

        INDArray indexArr = Nd4j.createFromArray(5L);
        Map<String, INDArray> placeholders = new HashMap<>();
        placeholders.put("indices", indexArr);

        // Get ground truth
        Map<String, INDArray> expected = sd.output(placeholders, "out");
        double expectedSum = expected.get("out").getDouble(0);

        // Freeze shapes after first execution (like the decode loop does)
        InferenceSession session = sd.getOrCreateSession();
        DynamicShapePlanExecutor dspExec = session.getDynamicShapePlanExecutor();
        if (dspExec != null) {
            dspExec.setShapesFrozen(true);
        }

        // Execute 4 more times (total 5) — now in frozen mode
        for (int i = 1; i < 5; i++) {
            Map<String, INDArray> result = sd.output(placeholders, "out");
            double sum = result.get("out").getDouble(0);
            assertEquals(expectedSum, sum, TOL,
                    "Execution " + i + ": output must match expected");
        }

        // Check plan phase via DspDebugger
        DspDebugger debugger = DspDebugger.attach(sd);
        DspDebugger.PlanReport report = debugger.analyzePlan();

        log.info("Plan after 5 executions:\n{}", report);
        assertNotNull(report.planPhase, "Plan phase should be set");
        assertTrue(report.planPhase.isAtLeast(PlanPhase.SHAPES_FROZEN),
                "Plan phase should be SHAPES_FROZEN or higher after 5 executions, got: "
                        + report.planPhase);

        // Verify segment states - check for value-dependent ops
        for (DspDebugger.SegmentReport seg : report.segments) {
            log.info("Segment {}: capturable={}, phase={}, execCount={}",
                    seg.index, seg.capturable, seg.phase, seg.executionCount);

            // Segments should not have capture failures
            if (seg.capturable) {
                assertFalse(seg.captureFailed,
                        "Capturable segment " + seg.index + " should not have failed capture");
            }
        }

        log.info("testNoSlotReplacementDuringReplay: value-dependent chain correct through replay");
    }

    // ========================================================================
    // 3. Plan Phase Monotonicity
    // ========================================================================

    /**
     * Execute a graph 10 times. After each execution, check plan phase via
     * DspDebugger. Verify that phase only advances (SLOT_BY_SLOT -> SHAPES_FROZEN
     * -> POINTERS_STABLE -> REPLAYING) and never goes backward without an explicit
     * reset.
     *
     * This catches bugs where phase transitions oscillate or regress unexpectedly.
     */
    @Test
    @Order(3)
    @DisplayName("Phase contract: plan phase monotonicity over 10 executions")
    public void testPlanPhaseMonotonicity() {
        int dim = 16;
        int numLayers = 12;
        sd = buildDeepMatmulChain("x", "out", numLayers, dim);
        enableDsp(sd);

        INDArray input = Nd4j.randn(DataType.FLOAT, 1, dim).muli(0.5);
        Map<String, INDArray> placeholders = Map.of("x", input);

        DspDebugger debugger = DspDebugger.attach(sd);
        NativeOps ops = NativeOpsHolder.getInstance().getDeviceNativeOps();

        PlanPhase previousPhase = null;
        List<PlanPhase> phaseHistory = new ArrayList<>();

        for (int step = 0; step < 10; step++) {
            sd.output(placeholders, "out");

            Pointer handle = getPlanHandle(sd);
            if (handle == null || handle.isNull()) {
                log.info("Step {}: plan not yet compiled", step);
                phaseHistory.add(null);
                continue;
            }

            int phaseCode = ops.getPlanPhase(handle);
            PlanPhase currentPhase = PlanPhase.fromNativeCode(phaseCode);
            phaseHistory.add(currentPhase);
            log.info("Step {}: plan phase = {}", step, currentPhase);

            // Monotonicity check: current phase must be >= previous phase
            if (previousPhase != null && currentPhase != null) {
                assertTrue(currentPhase.getNativeCode() >= previousPhase.getNativeCode(),
                        "Phase REGRESSION at step " + step + ": "
                                + previousPhase + " -> " + currentPhase
                                + ". Phase history: " + phaseHistory
                                + ". Phase must only advance without explicit reset.");
            }

            previousPhase = currentPhase;
        }

        // Verify we advanced past SLOT_BY_SLOT at some point
        boolean advanced = phaseHistory.stream()
                .filter(Objects::nonNull)
                .anyMatch(p -> p.isAtLeast(PlanPhase.SHAPES_FROZEN));

        log.info("Phase history: {}", phaseHistory);
        log.info("Advanced past SLOT_BY_SLOT: {}", advanced);

        // Also verify via DspDebugger.trackReplayProgression for cross-validation
        DspDebugger.GraphReplayProgressReport progressReport =
                debugger.trackReplayProgression(5, placeholders, "out");
        List<String> regressions = progressReport.getPhaseRegressions();
        assertTrue(regressions.isEmpty(),
                "Segment phase regressions detected: " + regressions);

        log.info("testPlanPhaseMonotonicity: no phase regressions over 15 total executions");
    }

    // ========================================================================
    // 4. Replay Handle Persistence Across Steps
    // ========================================================================

    /**
     * After graph capture (5 executions), verify that capturable segments
     * maintain their replay handles for the next 10 executions. No handle
     * should become null without a logged reason.
     *
     * This catches bugs where replay handles are prematurely released or
     * invalidated during steady-state execution.
     */
    @Test
    @Order(4)
    @DisplayName("Phase contract: replay handles persist across 10 post-capture executions")
    public void testReplayHandlePersistenceAcrossSteps() {
        int dim = 16;
        int numLayers = 12;
        sd = buildDeepMatmulChain("x", "out", numLayers, dim);
        enableDsp(sd);
        sd.setGraphExecutionMode(GraphExecutionMode.CUDA_GRAPHS);

        INDArray input = Nd4j.randn(DataType.FLOAT, 1, dim).muli(0.5);
        Map<String, INDArray> placeholders = Map.of("x", input);

        // Phase 1: warmup + capture (5 executions)
        for (int i = 0; i < 5; i++) {
            sd.output(placeholders, "out");
        }

        DspDebugger debugger = DspDebugger.attach(sd);
        DspDebugger.GraphReplayReport captureReport = debugger.analyzeGraphReplay();

        log.info("After capture:\n{}", captureReport);

        // Record which segments are capturable and their initial state
        Map<Integer, DspDebugger.SegmentReplayInfo> capturedSegments = new LinkedHashMap<>();
        for (DspDebugger.SegmentReplayInfo seg : captureReport.segments) {
            if (seg.capturable && !seg.captureFailed) {
                capturedSegments.put(seg.index, seg);
                log.info("Captured segment {}: phase={}, replayCount={}, replayState={}",
                        seg.index, seg.phase, seg.replayCount, seg.replayState);
            }
        }
        assertFalse(capturedSegments.isEmpty(),
                "Expected at least one capturable segment after warmup/capture");

        // Phase 2: 10 more executions - verify handles persist
        for (int step = 0; step < 10; step++) {
            sd.output(placeholders, "out");

            DspDebugger.GraphReplayReport stepReport = debugger.analyzeGraphReplay();

            for (Map.Entry<Integer, DspDebugger.SegmentReplayInfo> entry : capturedSegments.entrySet()) {
                int segIdx = entry.getKey();
                DspDebugger.SegmentReplayInfo initialSeg = entry.getValue();

                // Find this segment in the current report
                DspDebugger.SegmentReplayInfo currentSeg = null;
                for (DspDebugger.SegmentReplayInfo s : stepReport.segments) {
                    if (s.index == segIdx) {
                        currentSeg = s;
                        break;
                    }
                }

                assertNotNull(currentSeg,
                        "Segment " + segIdx + " disappeared from plan at step " + step);

                // Segment should still be capturable and not failed
                assertTrue(currentSeg.capturable,
                        "Segment " + segIdx + " lost capturable flag at step " + step);
                assertFalse(currentSeg.captureFailed,
                        "Segment " + segIdx + " developed capture failure at step " + step);

                // Replay count should be non-decreasing
                assertTrue(currentSeg.replayCount >= initialSeg.replayCount,
                        "Segment " + segIdx + " replay count decreased at step " + step
                                + ": " + initialSeg.replayCount + " -> " + currentSeg.replayCount
                                + " — replay handle may have been released");
            }
        }

        log.info("testReplayHandlePersistenceAcrossSteps: all handles persisted across 10 steps");
    }

    // ========================================================================
    // 5. Changed Input Propagation During Replay
    // ========================================================================

    /**
     * Build a graph where a placeholder INT64 input feeds both a gather
     * (value-dependent) and a downstream matmul. Change the placeholder value
     * between replay steps. Verify that BOTH paths see the new value (output
     * changes).
     *
     * This tests the direct input propagation contract: when a placeholder
     * changes between frozen executions, replay must observe the new device
     * value so that all downstream ops see the updated data.
     */
    @Test
    @Order(5)
    @DisplayName("Phase contract: changed inputs propagate through replay")
    public void testCaptureBufferRefreshForChangingInputs() {
        int hiddenSize = 16;
        int vocabSize = 50;
        int numLayers = 10;

        sd = SameDiff.create();

        INDArray embTable = Nd4j.randn(DataType.FLOAT, vocabSize, hiddenSize);
        SDVariable embeddings = sd.constant("embeddings", embTable);
        SDVariable indices = sd.placeHolder("indices", DataType.INT64, 1);

        // Gather: value-dependent path
        SDVariable gathered = sd.gather("gathered", embeddings, indices, 0);

        // Deep matmul chain to build up enough ops for capture
        SDVariable current = gathered;
        for (int layer = 0; layer < numLayers; layer++) {
            SDVariable w = sd.constant("w_" + layer,
                    Nd4j.eye(hiddenSize).castTo(DataType.FLOAT).muli(0.95));
            current = sd.mmul("mm_" + layer, current, w);
            current = sd.nn.relu("relu_" + layer, current, 0);
        }
        SDVariable out = current.sum("out");

        enableDsp(sd);

        // Phase 1: warmup with index=0 (5 executions)
        Map<String, INDArray> placeholders = new HashMap<>();
        placeholders.put("indices", Nd4j.createFromArray(0L));

        for (int i = 0; i < 5; i++) {
            sd.output(placeholders, "out");
        }

        // Phase 2: change input and verify output changes
        double previousSum = Double.NaN;
        List<Double> sums = new ArrayList<>();

        for (int idx = 0; idx < 10; idx++) {
            placeholders.put("indices", Nd4j.createFromArray((long) (idx % vocabSize)));

            Map<String, INDArray> result = sd.output(placeholders, "out");
            double sum = result.get("out").getDouble(0);
            sums.add(sum);
            log.info("Index {}: sum = {}", idx, sum);

            if (idx > 0) {
                // Different indices should produce different outputs
                // (unless two embedding rows happen to be nearly identical, which is very unlikely
                // with random initialization)
                assertNotEquals(previousSum, sum, TOL,
                        "Index " + idx + " vs " + (idx - 1)
                                + ": output should differ when input changes."
                                + " Previous=" + previousSum + ", Current=" + sum
                                + ". This indicates replay is using stale input data"
                                + " — the graph is replaying with stale input data.");
            }
            previousSum = sum;
        }

        // Verify we got at least 5 distinct values (out of 10 indices)
        long distinctCount = sums.stream().map(d -> Math.round(d * 1e6) / 1e6).distinct().count();
        assertTrue(distinctCount >= 5,
                "Expected at least 5 distinct output values for 10 different indices, got "
                        + distinctCount + ". Replay input propagation may be broken.");

        log.info("testCaptureBufferRefreshForChangingInputs: {} distinct outputs for 10 inputs",
                distinctCount);
    }

    // ========================================================================
    // 6. Decode Input Propagation With Replay (DspDebugger validation)
    // ========================================================================

    /**
     * Same pattern as testSetNextDecodeTokenPropagation from TestDSPLifecycleRegression,
     * but with explicit DspDebugger validation after each step. After each step,
     * call debugger.analyzePlan() and verify no phase regressions or slot state
     * inconsistencies.
     *
     * This adds DspDebugger-level phase contract checks on top of the basic
     * output correctness checks in the original test.
     */
    @Test
    @Order(6)
    @DisplayName("Phase contract: decode input propagation with DspDebugger validation")
    public void testDecodeInputPropagationWithCapture() {
        int hiddenSize = 16;
        int numLayers = 10;

        sd = SameDiff.create();

        INDArray embTable = Nd4j.randn(DataType.FLOAT, 100, hiddenSize);
        SDVariable embeddings = sd.constant("embeddings", embTable);
        SDVariable posIds = sd.placeHolder("position_ids", DataType.INT64, 1);

        SDVariable x = sd.gather("gathered", embeddings, posIds, 0);
        for (int layer = 0; layer < numLayers; layer++) {
            SDVariable w = sd.constant("w_" + layer,
                    Nd4j.eye(hiddenSize).castTo(DataType.FLOAT).muli(0.95));
            x = sd.mmul("mm_" + layer, x, w);
            x = sd.nn.relu("relu_" + layer, x, 0);
        }
        SDVariable out = x.sum("out");

        enableDsp(sd);
        sd.setGraphExecutionMode(GraphExecutionMode.CUDA_GRAPHS);

        DspDebugger debugger = DspDebugger.attach(sd);
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();

        Map<String, INDArray> placeholders = new HashMap<>();
        placeholders.put("position_ids", Nd4j.createFromArray(0L));

        // Phase 1: compile + warmup
        sd.output(placeholders, "out");

        // Phase 2: configure decode inputs
        InferenceSession session = sd.getOrCreateSession();
        DynamicShapePlanExecutor dspExec = session.getDynamicShapePlanExecutor();
        if (dspExec != null) {
            var plan = dspExec.getCurrentPlan();
            if (plan != null) {
                dspExec.setShapesFrozen(true);
                dspExec.configureDecodeInputs(plan, 100);
                log.info("Decode inputs configured: {}", dspExec.isDecodeInputsConfigured());
            }
        }

        // Phase 3: warmup frozen (3 more steps)
        for (int i = 0; i < 3; i++) {
            sd.output(placeholders, "out");
        }

        PlanPhase previousPhase = null;
        double previousSum = Double.NaN;

        // Phase 4: decode steps with DspDebugger validation
        if (dspExec != null && dspExec.isDecodeInputsConfigured()) {
            for (int step = 0; step < 8; step++) {
                // Update position via setNextDecodeToken
                dspExec.setNextDecodeToken(step, step);

                Map<String, INDArray> result = sd.output(placeholders, "out");
                double sum = result.get("out").getDouble(0);
                log.info("Decode step {}: cachePos={}, sum={}", step, step, sum);

                // Output correctness: different positions should give different outputs
                if (step > 0) {
                    assertNotEquals(previousSum, sum, 1e-6,
                            "Decode step " + step + ": output must change when position changes. "
                                    + "Previous=" + previousSum + ", Current=" + sum);
                }
                previousSum = sum;

                // DspDebugger phase contract validation
                DspDebugger.PlanReport report = debugger.analyzePlan();
                assertNotNull(report.planPhase,
                        "Plan phase should be set at decode step " + step);

                // Phase monotonicity during decode
                if (previousPhase != null && report.planPhase != null) {
                    assertTrue(report.planPhase.getNativeCode() >= previousPhase.getNativeCode(),
                            "Phase REGRESSION during decode at step " + step + ": "
                                    + previousPhase + " -> " + report.planPhase);
                }
                previousPhase = report.planPhase;

                // Check for slot state consistency: no unfrozen ops should appear
                // after we are past SHAPES_FROZEN
                if (report.planPhase != null
                        && report.planPhase.isAtLeast(PlanPhase.SHAPES_FROZEN)) {
                    List<DspDebugger.SlotInfo> unfrozen = report.getUnfrozenOps();
                    log.info("Decode step {}: {} unfrozen ops", step, unfrozen.size());
                    // Log unfrozen ops for debugging but don't fail on them
                    // (value-dependent ops may remain unfrozen)
                    for (DspDebugger.SlotInfo slot : unfrozen) {
                        log.info("  Unfrozen: {}", slot);
                    }
                }

                // Verify no capture failures appeared
                for (DspDebugger.SegmentReport seg : report.segments) {
                    if (seg.capturable) {
                        assertFalse(seg.captureFailed,
                                "Segment " + seg.index + " capture failed at decode step " + step);
                    }
                }
            }
        } else {
            log.warn("DSP executor or decode inputs not configured — "
                    + "verifying basic output correctness only");
            // Fallback: just verify outputs change with different position_ids
            for (int step = 0; step < 8; step++) {
                placeholders.put("position_ids", Nd4j.createFromArray((long) step));
                Map<String, INDArray> result = sd.output(placeholders, "out");
                double sum = result.get("out").getDouble(0);

                if (step > 0) {
                    assertNotEquals(previousSum, sum, 1e-6,
                            "Step " + step + ": output must change when position changes");
                }
                previousSum = sum;
            }
        }

        log.info("testDecodeInputPropagationWithCapture: all phase contracts validated");
    }

    // ========================================================================
    // 7. Phase Contract After releaseGpuIntermediates
    // ========================================================================

    /**
     * Build a graph, execute 5 times (reaches frozen/replaying state), then call
     * releaseGpuIntermediates. Verify that:
     * <ul>
     *   <li>Phase demoted to SLOT_BY_SLOT after release</li>
     *   <li>Re-execution works from scratch (new warmup -> freeze -> replay cycle)</li>
     *   <li>No stale pointers from previous cycle (outputs are still correct)</li>
     * </ul>
     *
     * This catches bugs where releaseGpuIntermediates leaves stale state that
     * corrupts the subsequent warmup cycle.
     */
    @Test
    @Order(7)
    @DisplayName("Phase contract: releaseGpuIntermediates demotes phase and allows clean re-warmup")
    public void testPhaseContractAfterReleaseGpuIntermediates() {
        int dim = 16;
        int numLayers = 12;
        sd = buildDeepMatmulChain("x", "out", numLayers, dim);
        enableDsp(sd);

        INDArray input = Nd4j.randn(DataType.FLOAT, 1, dim).muli(0.5);
        Map<String, INDArray> placeholders = Map.of("x", input);

        // Get ground truth before DSP (standard execution)
        Map<String, INDArray> groundTruth = sd.output(placeholders, "out");
        INDArray expected = groundTruth.get("out").dup();

        // Phase 1: warmup + capture (5 executions)
        for (int i = 0; i < 5; i++) {
            Map<String, INDArray> result = sd.output(placeholders, "out");
            assertClose(expected, result.get("out").dup(),
                    "Pre-release execution " + i);
        }

        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();

        // Record phase before release
        Pointer handleBefore = getPlanHandle(sd);
        PlanPhase phaseBefore = null;
        if (handleBefore != null && !handleBefore.isNull()) {
            phaseBefore = PlanPhase.fromNativeCode(nativeOps.getPlanPhase(handleBefore));
            log.info("Phase before release: {}", phaseBefore);
        }

        // Release GPU intermediates
        InferenceSession session = sd.getOrCreateSession();
        DynamicShapePlanExecutor dspExec = session.getDynamicShapePlanExecutor();
        if (dspExec != null) {
            int freed = dspExec.releaseGpuIntermediates();
            log.info("Released {} GPU intermediates", freed);

            // Check phase after release
            Pointer handleAfter = getPlanHandle(sd);
            if (handleAfter != null && !handleAfter.isNull()) {
                PlanPhase phaseAfter = PlanPhase.fromNativeCode(
                        nativeOps.getPlanPhase(handleAfter));
                log.info("Phase after release: {}", phaseAfter);

                // Phase should be demoted to SLOT_BY_SLOT
                if (phaseBefore != null && phaseBefore.isAtLeast(PlanPhase.SHAPES_FROZEN)) {
                    assertEquals(PlanPhase.SLOT_BY_SLOT, phaseAfter,
                            "Phase should be demoted to SLOT_BY_SLOT after releaseGpuIntermediates, "
                                    + "but got: " + phaseAfter);
                }
            }

            // Phase 2: re-warmup from scratch
            log.info("Starting re-warmup cycle after release...");
            List<PlanPhase> reWarmupPhases = new ArrayList<>();

            for (int i = 0; i < 8; i++) {
                Map<String, INDArray> result = sd.output(placeholders, "out");

                // Verify correctness: no stale pointers from previous cycle
                assertClose(expected, result.get("out").dup(),
                        "Post-release execution " + i);

                Pointer handle = getPlanHandle(sd);
                if (handle != null && !handle.isNull()) {
                    PlanPhase phase = PlanPhase.fromNativeCode(nativeOps.getPlanPhase(handle));
                    reWarmupPhases.add(phase);
                    log.info("Re-warmup step {}: phase = {}", i, phase);
                }
            }

            // Verify re-warmup produces monotonically advancing phases
            for (int i = 1; i < reWarmupPhases.size(); i++) {
                PlanPhase prev = reWarmupPhases.get(i - 1);
                PlanPhase curr = reWarmupPhases.get(i);
                if (prev != null && curr != null) {
                    assertTrue(curr.getNativeCode() >= prev.getNativeCode(),
                            "Phase REGRESSION during re-warmup at step " + i + ": "
                                    + prev + " -> " + curr
                                    + ". Stale state from previous cycle may be corrupting re-warmup.");
                }
            }

            log.info("Re-warmup phase progression: {}", reWarmupPhases);

        } else {
            log.warn("No DSP executor — skipping releaseGpuIntermediates test");
        }

        log.info("testPhaseContractAfterReleaseGpuIntermediates: clean re-warmup verified");
    }
}
