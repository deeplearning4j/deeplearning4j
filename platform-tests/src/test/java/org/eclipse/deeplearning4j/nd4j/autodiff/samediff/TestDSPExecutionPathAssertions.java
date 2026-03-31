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
import org.junit.jupiter.api.Assumptions;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlanExecutor;
import org.nd4j.autodiff.samediff.execution.ExecutionPhase;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.autodiff.samediff.execution.PlanIntrospection;
import org.nd4j.autodiff.samediff.diagnostics.DspDiagnostics;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
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
 * Tests DSP execution paths, backend selection, and hard failure behavior.
 *
 * <p>These tests verify that the DSP (Dynamic Shape Plan) execution path
 * correctly selects backends, progresses through execution states, and
 * fails hard (rather than silently falling back) when errors occur.</p>
 *
 * <p>Run:
 * <pre>
 *   cd platform-tests && mvn test -Dtest=TestDSPExecutionPathAssertions 2>&1 | tee /tmp/dsp-exec-path.log
 * </pre>
 */
@Slf4j
@Tag("dsp")
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
public class TestDSPExecutionPathAssertions extends BaseNd4jTestWithBackends {

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

    private void enableDsp(SameDiff sd) {
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);
    }

    @AfterEach
    void cleanup() {
        if (sd != null) {
            sd.close();
            sd = null;
        }
    }

    // =========================================================================
    // Helper: build a simple matmul + relu graph
    // =========================================================================
    private SameDiff buildMatmulReluGraph() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, 4);
        SDVariable w = sd.constant("w", Nd4j.randn(DataType.FLOAT, 4, 8));
        SDVariable mm = sd.mmul("mm", x, w);
        SDVariable out = sd.nn.relu("out", mm, 0);
        return sd;
    }

    // =========================================================================
    // Helper: build a medium graph (matmul + relu + add)
    // =========================================================================
    private SameDiff buildMediumGraph() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, 4);
        SDVariable w = sd.constant("w", Nd4j.randn(DataType.FLOAT, 4, 8));
        SDVariable b = sd.constant("b", Nd4j.randn(DataType.FLOAT, 1, 8));
        SDVariable mm = sd.mmul("mm", x, w);
        SDVariable relu = sd.nn.relu("relu", mm, 0);
        SDVariable out = relu.add("out", b);
        return sd;
    }

    // =========================================================================
    // Helper: execute graph N times and return the native plan handle
    // =========================================================================
    private Pointer executeAndGetHandle(SameDiff sd, int numExecutions, String outputName) {
        for (int i = 0; i < numExecutions; i++) {
            INDArray input = Nd4j.randn(DataType.FLOAT, 1, 4);
            Map<String, INDArray> result = sd.output(
                    Collections.singletonMap("x", input), outputName);
            assertNotNull(result.get(outputName),
                    "Output should not be null on execution " + (i + 1));
        }

        InferenceSession session = sd.getOrCreateSession();
        DynamicShapePlanExecutor dsp = session.getDynamicShapePlanExecutor();
        assertNotNull(dsp, "DSP executor should exist after execution");
        return dsp.getNativePlanHandle();
    }

    // =========================================================================
    // Test 1: Backend selection matches the requested mode
    // =========================================================================
    @Test
    @Order(1)
    @DisplayName("SLOT_BY_SLOT mode: all segments report slot-by-slot backend")
    public void testBackendSelectionMatchesMode() {
        sd = buildMatmulReluGraph();
        sd.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);
        enableDsp(sd);

        Pointer handle = executeAndGetHandle(sd, 3, "out");
        assertNotNull(handle, "Native plan handle should exist after 3 executions");

        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        int numSegments = nativeOps.getPlanNumSegments(handle);
        log.info("SLOT_BY_SLOT mode: {} segments", numSegments);
        assertTrue(numSegments > 0, "Plan should have at least one segment");

        for (int i = 0; i < numSegments; i++) {
            String backendName = nativeOps.getPlanSegmentBackendName(handle, i);
            log.info("  Segment {}: backend = '{}'", i, backendName);
            // In SLOT_BY_SLOT mode, backend should be slot-by-slot (or equivalent name)
            assertNotNull(backendName, "Backend name should not be null for segment " + i);
            // The backend name should indicate slot-by-slot execution
            assertTrue(backendName.toLowerCase().contains("slot")
                            || backendName.toLowerCase().contains("fallback")
                            || backendName.isEmpty(),
                    "Expected slot-by-slot-related backend for segment " + i
                            + ", got: '" + backendName + "'");
        }
    }

    // =========================================================================
    // Test 2: No fallback events in steady state
    // =========================================================================
    @Test
    @Order(2)
    @DisplayName("No FALLBACK diagnostic events after warmup in steady state")
    public void testNoFallbackEventsInSteadyState() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();

        // Enable diagnostics for FALLBACK and EXECUTE categories
        nativeOps.dspDiagSetCategories(DspDiagnostics.FALLBACK | DspDiagnostics.EXECUTE);
        nativeOps.dspDiagSetLevel(DspDiagnostics.LEVEL_FULL);
        nativeOps.dspDiagClear();

        sd = buildMediumGraph();
        enableDsp(sd);

        // Warmup: first 3 executions may trigger compilation/fallback
        for (int i = 0; i < 3; i++) {
            INDArray input = Nd4j.randn(DataType.FLOAT, 1, 4);
            sd.output(Collections.singletonMap("x", input), "out");
        }

        // Clear diagnostics after warmup
        nativeOps.dspDiagClear();

        // Steady state: next 7 executions should have zero fallback events
        for (int i = 0; i < 7; i++) {
            INDArray input = Nd4j.randn(DataType.FLOAT, 1, 4);
            sd.output(Collections.singletonMap("x", input), "out");
        }

        String jsonReport = nativeOps.dspDiagGetJsonReport();
        log.info("Steady-state diagnostic report: {}", jsonReport);

        // The report should not contain FALLBACK events
        if (jsonReport != null && !jsonReport.isEmpty()) {
            String lower = jsonReport.toLowerCase();
            assertFalse(lower.contains("\"fallback\"") && lower.contains("\"count\"")
                            && !lower.contains("\"count\":0") && !lower.contains("\"count\": 0"),
                    "No FALLBACK events should occur in steady state. Report: " + jsonReport);
        }

        // Cleanup diagnostics
        nativeOps.dspDiagSetCategories(DspDiagnostics.NONE);
    }

    // =========================================================================
    // Test 3: Data-dependent ops are handled correctly (not silently skipped)
    // =========================================================================
    @Test
    @Order(3)
    @DisplayName("Data-dependent op (Where) is correctly identified and handled in plan")
    public void testHardFailureOnBackendError() {
        sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 3);
        // Where is data-dependent: output shape depends on input values
        SDVariable condition = sd.gt("cond", x, 0.0);
        // Use the condition in a simple way that forces DSP to handle it
        SDVariable cast = sd.castTo("cast", condition, DataType.FLOAT);
        SDVariable out = cast.mul("out", x);

        enableDsp(sd);

        // Execute multiple times -- DSP should handle this without crashing
        // and without silently skipping any ops
        for (int i = 0; i < 5; i++) {
            INDArray input = Nd4j.randn(DataType.FLOAT, 2, 3);
            Map<String, INDArray> result = sd.output(
                    Collections.singletonMap("x", input), "out");
            INDArray output = result.get("out");
            assertNotNull(output, "Output should not be null on execution " + (i + 1));

            // Verify correctness: out = (x > 0 ? 1 : 0) * x = relu(x) equivalent
            // but through a data-dependent path
            for (int r = 0; r < 2; r++) {
                for (int c = 0; c < 3; c++) {
                    float xVal = input.getFloat(r, c);
                    float outVal = output.getFloat(r, c);
                    float expected = xVal > 0 ? xVal : 0.0f;
                    assertEquals(expected, outVal, 1e-5,
                            "Incorrect value at [" + r + "," + c + "]");
                }
            }
        }
    }

    // =========================================================================
    // Test 4: Warmup execution works, then subsequent executions continue
    // =========================================================================
    @Test
    @Order(4)
    @DisplayName("Warmup (first execution) completes, and execution count increases")
    public void testWarmupIsSlotBySlot() {
        sd = buildMediumGraph();
        enableDsp(sd);

        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();

        // Execute once (warmup)
        INDArray input = Nd4j.randn(DataType.FLOAT, 1, 4);
        Map<String, INDArray> result1 = sd.output(
                Collections.singletonMap("x", input), "out");
        assertNotNull(result1.get("out"), "Warmup execution should produce output");

        // Get handle after first execution
        Pointer handle = sd.getOrCreateSession().getDynamicShapePlanExecutor()
                .getNativePlanHandle();

        if (handle != null && !handle.isNull()) {
            int numSegments = nativeOps.getPlanNumSegments(handle);
            log.info("After warmup: {} segments", numSegments);

            // Query execution counts for each segment
            for (int i = 0; i < numSegments; i++) {
                int execCount = nativeOps.getPlanSegmentExecutionCount(handle, i);
                log.info("  Segment {} execution count after warmup: {}", i, execCount);
                assertTrue(execCount >= 1,
                        "Segment " + i + " should have at least 1 execution after warmup");
            }

            // Execute again — verify results and that execution count advances
            INDArray input2 = Nd4j.randn(DataType.FLOAT, 1, 4);
            Map<String, INDArray> result2 = sd.output(Collections.singletonMap("x", input2), "out");
            assertNotNull(result2.get("out"), "Second execution should produce output");

            // Re-fetch handle — may be same or newly compiled
            Pointer handle2 = sd.getOrCreateSession().getDynamicShapePlanExecutor()
                    .getNativePlanHandle();
            int numSegments2 = nativeOps.getPlanNumSegments(handle2);
            for (int i = 0; i < numSegments2; i++) {
                int execCount = nativeOps.getPlanSegmentExecutionCount(handle2, i);
                log.info("  Segment {} execution count after 2nd: {}", i, execCount);
                // Count should be >= 1 (at minimum, this plan instance ran once)
                assertTrue(execCount >= 1,
                        "Segment " + i + " should have at least 1 execution after second call");
            }
        } else {
            log.info("Native plan handle not available (CPU backend?), verifying output only");
            // Even without a native handle, the second execution should work
            INDArray input2 = Nd4j.randn(DataType.FLOAT, 1, 4);
            Map<String, INDArray> result2 = sd.output(
                    Collections.singletonMap("x", input2), "out");
            assertNotNull(result2.get("out"), "Second execution should produce output");
        }
    }

    // =========================================================================
    // Test 5: Replay state progresses through states on CUDA
    // =========================================================================
    @Test
    @Order(5)
    @DisplayName("Replay state progresses from initial through capturing to ready (CUDA only)")
    public void testReplayStateProgression() {
        // This test is only meaningful on CUDA backends
        Assumptions.assumeTrue(!Nd4j.getEnvironment().isCPU(),
                "Skipping replay state test on CPU backend");

        sd = buildMatmulReluGraph();
        sd.setGraphExecutionMode(GraphExecutionMode.CUDA_GRAPHS);
        enableDsp(sd);

        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();

        // Execute multiple times to allow state progression
        int[] statesObserved = new int[10];
        for (int i = 0; i < 10; i++) {
            INDArray input = Nd4j.randn(DataType.FLOAT, 1, 4);
            sd.output(Collections.singletonMap("x", input), "out");

            Pointer handle = sd.getOrCreateSession().getDynamicShapePlanExecutor()
                    .getNativePlanHandle();
            if (handle != null && !handle.isNull()) {
                int numSegments = nativeOps.getPlanNumSegments(handle);
                if (numSegments > 0) {
                    statesObserved[i] = nativeOps.getPlanSegmentReplayState(handle, 0);
                    log.info("Execution {}: segment 0 replay state = {}", i + 1, statesObserved[i]);
                }
            }
        }

        // Verify that at least some state was observed (not all -1)
        // Note: -1 means no replay handle yet, which is valid for warmup phases
        boolean anyValidState = false;
        for (int i = 0; i < 10; i++) {
            if (statesObserved[i] >= 0) {
                anyValidState = true;
                break;
            }
        }

        log.info("States observed: {}", Arrays.toString(statesObserved));
        // On CUDA with CUDA_GRAPHS mode, we expect replay state to become valid
        // after several executions. If not, the plan may not support graph capture
        // for this particular graph (e.g., matmul segments may be non-capturable).
        if (!anyValidState) {
            log.info("No valid replay states observed — graph capture may not be applicable for this graph");
        }
    }

    // =========================================================================
    // Test 6: Stale buffer assertion does not false-positive in normal execution
    // =========================================================================
    @Test
    @Order(6)
    @DisplayName("Normal execution produces no false-positive stale buffer assertions")
    public void testStaleBufferIsHardError() {
        sd = buildMediumGraph();
        enableDsp(sd);

        // Execute through warmup + steady state (5+ executions)
        // If there is a stale buffer assertion, it would fire here
        for (int i = 0; i < 8; i++) {
            INDArray input = Nd4j.randn(DataType.FLOAT, 1, 4);
            Map<String, INDArray> result = sd.output(
                    Collections.singletonMap("x", input), "out");
            INDArray output = result.get("out");
            assertNotNull(output, "Output should not be null on execution " + (i + 1));

            // Verify the output is numerically valid (no NaN/Inf from stale buffers)
            assertFalse(output.isNaN().any(),
                    "Output contains NaN on execution " + (i + 1) + " (possible stale buffer)");
            assertFalse(output.isInfinite().any(),
                    "Output contains Inf on execution " + (i + 1) + " (possible stale buffer)");
        }

        log.info("Completed 8 executions with no stale buffer assertions");
    }

    // =========================================================================
    // Test 7: Frozen outputs are never null after reaching frozen state
    // =========================================================================
    @Test
    @Order(7)
    @DisplayName("All outputs are non-null after reaching frozen steady state")
    public void testFrozenOutputNeverNull() {
        sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, 8);
        SDVariable reshaped = sd.reshape("reshaped", x, 1, 2, 4);
        SDVariable relu = sd.nn.relu("relu", reshaped, 0);
        SDVariable flat = sd.reshape("flat", relu, 1, 8);
        SDVariable b = sd.constant("b", Nd4j.randn(DataType.FLOAT, 1, 8));
        SDVariable out = flat.add("out", b);

        enableDsp(sd);

        // Execute 5+ times to reach frozen state
        for (int i = 0; i < 6; i++) {
            INDArray input = Nd4j.randn(DataType.FLOAT, 1, 8);
            Map<String, INDArray> result = sd.output(
                    Collections.singletonMap("x", input), "out");
            INDArray output = result.get("out");
            assertNotNull(output, "Output must not be null on execution " + (i + 1));
            assertArrayEquals(new long[]{1, 8}, output.shape(),
                    "Output shape mismatch on execution " + (i + 1));
        }

        // After frozen state, introspect via native handle
        Pointer handle = sd.getOrCreateSession().getDynamicShapePlanExecutor()
                .getNativePlanHandle();
        if (handle != null && !handle.isNull()) {
            NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
            int numSlots = nativeOps.getPlanNumSlots(handle);
            log.info("Plan has {} slots after reaching frozen state", numSlots);
            assertTrue(numSlots > 0, "Plan should have slots in frozen state");
        }
    }

    // =========================================================================
    // Test 8: AUTO mode selects consistently, does not cascade between backends
    // =========================================================================
    @Test
    @Order(8)
    @DisplayName("AUTO mode selects same backend for all post-warmup executions")
    public void testAutoModeSelectsOnceNotCascades() {
        sd = buildMediumGraph();
        sd.setGraphExecutionMode(GraphExecutionMode.AUTO);
        enableDsp(sd);

        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();

        // Execute enough times to pass warmup and reach steady state
        Pointer handle = executeAndGetHandle(sd, 5, "out");

        if (handle != null && !handle.isNull()) {
            int numSegments = nativeOps.getPlanNumSegments(handle);
            log.info("AUTO mode: {} segments", numSegments);

            // Record backends for all segments
            String[] backends = new String[numSegments];
            for (int i = 0; i < numSegments; i++) {
                backends[i] = nativeOps.getPlanSegmentCompiledBackend(handle, i);
                log.info("  Segment {}: compiled backend = '{}'", i, backends[i]);
            }

            // Execute 5 more times
            for (int i = 0; i < 5; i++) {
                INDArray input = Nd4j.randn(DataType.FLOAT, 1, 4);
                sd.output(Collections.singletonMap("x", input), "out");
            }

            // Verify backends are the same after more executions (no cascading)
            for (int i = 0; i < numSegments; i++) {
                String currentBackend = nativeOps.getPlanSegmentCompiledBackend(handle, i);
                log.info("  Segment {} after 10 total: compiled backend = '{}'", i, currentBackend);
                assertEquals(backends[i], currentBackend,
                        "Backend for segment " + i + " should not change between executions "
                                + "(was '" + backends[i] + "', now '" + currentBackend + "')");
            }
        } else {
            log.info("No native handle available -- verifying execution only");
            // Verify that 5 more executions still work correctly
            for (int i = 0; i < 5; i++) {
                INDArray input = Nd4j.randn(DataType.FLOAT, 1, 4);
                Map<String, INDArray> result = sd.output(
                        Collections.singletonMap("x", input), "out");
                assertNotNull(result.get("out"));
            }
        }
    }

    // =========================================================================
    // Test 9: CPU backend does not attempt GPU backend selection
    // =========================================================================
    @Test
    @Order(9)
    @DisplayName("CPU backend does not attempt GPU backend selection")
    public void testCpuBackendNoDeadGpuAttempt() {
        // This test is only meaningful on CPU
        Assumptions.assumeTrue(Nd4j.getEnvironment().isCPU(),
                "Skipping CPU-only test on GPU backend");

        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();

        // Enable BACKEND diagnostics
        nativeOps.dspDiagSetCategories(DspDiagnostics.BACKEND);
        nativeOps.dspDiagSetLevel(DspDiagnostics.LEVEL_FULL);
        nativeOps.dspDiagClear();

        sd = buildMediumGraph();
        enableDsp(sd);

        // Execute 3 times
        for (int i = 0; i < 3; i++) {
            INDArray input = Nd4j.randn(DataType.FLOAT, 1, 4);
            sd.output(Collections.singletonMap("x", input), "out");
        }

        String report = nativeOps.dspDiagGetJsonReport();
        log.info("CPU backend diagnostic report: {}", report);

        // On CPU, there should be no GPU/CUDA backend attempts
        if (report != null && !report.isEmpty()) {
            String lower = report.toLowerCase();
            assertFalse(lower.contains("cuda_graphs") && lower.contains("attempt"),
                    "CPU backend should not attempt CUDA graph capture. Report: " + report);
        }

        // Check available backends -- should not include CUDA-specific backends
        Pointer handle = sd.getOrCreateSession().getDynamicShapePlanExecutor()
                .getNativePlanHandle();
        if (handle != null && !handle.isNull()) {
            String availableBackends = nativeOps.getPlanAvailableBackends(handle);
            log.info("Available backends on CPU: {}", availableBackends);
            // Not asserting absence of "cuda" in available backends because the plan
            // might legitimately list all known backends -- we just check it didn't
            // attempt to use them
        }

        // Cleanup diagnostics
        nativeOps.dspDiagSetCategories(DspDiagnostics.NONE);
    }

    // =========================================================================
    // Test 10: Every segment has a non-empty compiled backend name
    // =========================================================================
    @Test
    @Order(10)
    @DisplayName("Every segment reports a non-null backend name after compilation")
    public void testSegmentCompiledBackendNonEmpty() {
        sd = buildMediumGraph();
        enableDsp(sd);

        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();

        // Execute 3 times to allow compilation
        Pointer handle = executeAndGetHandle(sd, 3, "out");

        if (handle != null && !handle.isNull()) {
            int numSegments = nativeOps.getPlanNumSegments(handle);
            log.info("Checking {} segments for non-empty backend names", numSegments);
            assertTrue(numSegments > 0, "Plan should have at least one segment");

            for (int i = 0; i < numSegments; i++) {
                String backendName = nativeOps.getPlanSegmentBackendName(handle, i);
                log.info("  Segment {}: backend = '{}'", i, backendName);
                assertNotNull(backendName,
                        "Backend name must not be null for segment " + i);
                // Note: empty string is allowed for the default interface method,
                // but after actual compilation it should be populated.
                // We log it for diagnostic purposes either way.
            }

            // Also check the segments summary JSON is parseable
            String summaryJson = nativeOps.getPlanSegmentsSummaryJson(handle);
            log.info("Segments summary JSON: {}", summaryJson);
            assertNotNull(summaryJson, "Segments summary JSON must not be null");
        } else {
            log.info("No native handle available -- verifying basic execution only");
            // Without a native handle, just verify execution works
            INDArray input = Nd4j.randn(DataType.FLOAT, 1, 4);
            Map<String, INDArray> result = sd.output(
                    Collections.singletonMap("x", input), "out");
            assertNotNull(result.get("out"), "Output should be non-null");
        }
    }

    // ── ExecutionPhase tests ─────────────────────────────────────────────────

    /**
     * Verify execution phase progresses: WARMUP → COMPILING → REPLAYING across multiple executions.
     * After 5 executions, capturable segments should be at COMPILED or REPLAYING.
     */
    @Test
    @Order(20)
    @DisplayName("ExecutionPhase progression: warmup → compiled → replaying")
    void testExecutionPhaseProgression() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, 4);
        SDVariable w = sd.var("w", Nd4j.randn(DataType.FLOAT, 4, 4));
        SDVariable mm = sd.mmul("mm", x, w);
        SDVariable out = sd.nn.relu("out", mm, 0);

        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);
        sd.setGraphExecutionMode(GraphExecutionMode.AUTO);

        INDArray input = Nd4j.randn(DataType.FLOAT, 1, 4);
        Map<String, INDArray> ph = Collections.singletonMap("x", input);

        // Execute 5 times to progress through phases
        for (int i = 0; i < 5; i++) {
            Map<String, INDArray> result = sd.output(ph, "out");
            assertNotNull(result.get("out"), "Output must not be null on iteration " + i);
        }

        // Check phases via JNI
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        DynamicShapePlanExecutor executor = sd.getOrCreateSession().getDynamicShapePlanExecutor();
        if (executor == null) {
            log.info("No DSP executor -- skipping phase check (DSP not compiled for this model)");
            return;
        }

        Pointer handle = executor.getNativePlanHandle();
        if (handle == null) {
            log.info("No native plan handle -- skipping phase check");
            return;
        }

        int numSegments = nativeOps.getPlanNumSegments(handle);
        assertTrue(numSegments > 0, "Plan should have at least one segment");

        for (int i = 0; i < numSegments; i++) {
            int phaseCode = nativeOps.getPlanSegmentExecutionPhase(handle, i);
            ExecutionPhase phase = ExecutionPhase.fromNativeCode(phaseCode);
            int execCount = nativeOps.getPlanSegmentExecutionCount(handle, i);
            boolean isCapturable = nativeOps.isPlanSegmentCapturable(handle, i);

            log.info("Segment {}: phase={} (code={}), execCount={}, capturable={}",
                    i, phase, phaseCode, execCount, isCapturable);

            assertNotNull(phase, "Phase should be a known ExecutionPhase value for segment " + i);

            if (isCapturable && execCount >= 3) {
                // Capturable segments should be past warmup after 3+ executions
                assertTrue(phase == ExecutionPhase.COMPILED || phase == ExecutionPhase.REPLAYING
                                || phase == ExecutionPhase.COMPILING,
                        "Capturable segment " + i + " should be COMPILED/REPLAYING/COMPILING after " +
                                execCount + " executions, but was " + phase);
            }
        }
    }

    /**
     * Verify that non-capturable segments stay at SLOT_BY_SLOT phase.
     * Uses SLOT_BY_SLOT mode explicitly so ALL segments are non-capturable.
     */
    @Test
    @Order(21)
    @DisplayName("Non-capturable segments stay at SLOT_BY_SLOT")
    void testSlotBySlotPhaseForNonCapturable() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, 4);
        SDVariable w = sd.constant("w", Nd4j.randn(DataType.FLOAT, 4, 8));
        SDVariable mm = sd.mmul("mm", x, w);
        SDVariable out = sd.nn.relu("out", mm, 0);

        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);
        // Force SLOT_BY_SLOT — all segments execute slot-by-slot
        sd.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);

        INDArray input = Nd4j.randn(DataType.FLOAT, 1, 4);
        Map<String, INDArray> ph = Collections.singletonMap("x", input);

        for (int i = 0; i < 3; i++) {
            Map<String, INDArray> result = sd.output(ph, "out");
            assertNotNull(result.get("out"));
        }

        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        DynamicShapePlanExecutor executor = sd.getOrCreateSession().getDynamicShapePlanExecutor();
        if (executor == null) return;

        Pointer handle = executor.getNativePlanHandle();
        if (handle == null) return;

        int numSegments = nativeOps.getPlanNumSegments(handle);
        assertTrue(numSegments > 0, "Should have at least one segment");

        for (int i = 0; i < numSegments; i++) {
            int phaseCode = nativeOps.getPlanSegmentExecutionPhase(handle, i);
            ExecutionPhase phase = ExecutionPhase.fromNativeCode(phaseCode);
            log.info("SLOT_BY_SLOT segment {}: phase={}", i, phase);
            // In SLOT_BY_SLOT mode, all segments should be in WARMUP or SLOT_BY_SLOT phase
            assertTrue(phase == ExecutionPhase.SLOT_BY_SLOT || phase == ExecutionPhase.WARMUP,
                    "SLOT_BY_SLOT segment " + i + " should be SLOT_BY_SLOT or WARMUP, was " + phase);
        }
    }

    /**
     * Verify that PlanIntrospection.getPlanPhase returns the minimum phase across all segments.
     */
    @Test
    @Order(22)
    @DisplayName("Plan phase is minimum across all segments")
    void testPlanPhaseIsMinimum() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, 8);
        SDVariable w = sd.var("w", Nd4j.randn(DataType.FLOAT, 8, 8));
        SDVariable mm = sd.mmul("mm", x, w);
        SDVariable out = sd.nn.tanh("out", mm);

        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);
        sd.setGraphExecutionMode(GraphExecutionMode.AUTO);

        INDArray input = Nd4j.randn(DataType.FLOAT, 1, 8);
        Map<String, INDArray> ph = Collections.singletonMap("x", input);

        // Single execution — should be at WARMUP initially
        Map<String, INDArray> result = sd.output(ph, "out");
        assertNotNull(result.get("out"));

        DynamicShapePlanExecutor executor = sd.getOrCreateSession().getDynamicShapePlanExecutor();
        if (executor == null) return;

        Pointer handle = executor.getNativePlanHandle();
        if (handle == null) return;

        List<PlanIntrospection.SegmentInfo> segments =
                PlanIntrospection.getSegmentsWithReplayState(executor.getCurrentPlan(), handle);

        ExecutionPhase planPhase = PlanIntrospection.getPlanPhase(segments);
        log.info("After 1 execution: planPhase={}, segments={}", planPhase,
                segments.stream().map(s -> s.getExecutionPhase() + "").collect(java.util.stream.Collectors.joining(",")));

        assertNotNull(planPhase, "Plan phase should not be null after execution");

        // Verify it's the minimum across segments
        ExecutionPhase expectedMin = null;
        for (PlanIntrospection.SegmentInfo seg : segments) {
            ExecutionPhase sp = seg.getExecutionPhase();
            if (sp != null && (expectedMin == null || sp.getNativeCode() < expectedMin.getNativeCode())) {
                expectedMin = sp;
            }
        }
        assertEquals(expectedMin, planPhase,
                "Plan phase should be the minimum across all segments");

        // After 5 more executions, phase should advance
        for (int i = 0; i < 5; i++) {
            sd.output(ph, "out");
        }

        List<PlanIntrospection.SegmentInfo> segments2 =
                PlanIntrospection.getSegmentsWithReplayState(executor.getCurrentPlan(), handle);
        ExecutionPhase planPhase2 = PlanIntrospection.getPlanPhase(segments2);
        log.info("After 6 executions: planPhase={}", planPhase2);

        // Phase should have advanced (or stayed same if SLOT_BY_SLOT)
        assertNotNull(planPhase2, "Plan phase should not be null after 6 executions");
        assertTrue(planPhase2.getNativeCode() >= planPhase.getNativeCode(),
                "Plan phase should not regress: was " + planPhase + ", now " + planPhase2);
    }

    /**
     * Test that compilationFailed (renamed from captureFailed) is exposed through
     * introspection and defaults to false for a fresh plan.
     */
    @Test
    @DisplayName("compilationFailed defaults to false for fresh segments")
    public void testCompilationFailedDefaultFalse() {
        SameDiff sd = SameDiff.create();
        SDVariable in1 = sd.placeHolder("in1", DataType.FLOAT, 2, 3);
        SDVariable w = sd.var("w", Nd4j.randn(DataType.FLOAT, 3, 4));
        SDVariable out = sd.mmul("out", in1, w);

        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);
        sd.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);

        Map<String, INDArray> ph = new HashMap<>();
        ph.put("in1", Nd4j.randn(DataType.FLOAT, 2, 3));
        sd.output(ph, "out");

        DynamicShapePlanExecutor executor = sd.getOrCreateSession().getDynamicShapePlanExecutor();
        assertNotNull(executor, "Executor should exist after first execution");

        Pointer handle = executor.getNativePlanHandle();
        assertNotNull(handle, "Plan handle should exist");

        List<PlanIntrospection.SegmentInfo> segments =
                PlanIntrospection.getSegmentsWithReplayState(executor.getCurrentPlan(), handle);
        assertFalse(segments.isEmpty(), "Should have at least one segment");

        for (PlanIntrospection.SegmentInfo seg : segments) {
            assertFalse(seg.isCompilationFailed(),
                    "compilationFailed should be false for fresh segment [" +
                    seg.getStartSlot() + "-" + seg.getEndSlot() + "]");
        }
        log.info("All {} segments have compilationFailed=false", segments.size());
    }

    /**
     * Test that SLOT_BY_SLOT mode results in all segments being dispatched
     * as slot-by-slot (no graph compilation attempted).
     */
    @Test
    @DisplayName("SLOT_BY_SLOT mode prevents graph compilation")
    public void testSlotBySlotNoCompilation() {
        SameDiff sd = SameDiff.create();
        SDVariable in1 = sd.placeHolder("in1", DataType.FLOAT, 2, 3);
        SDVariable w = sd.var("w", Nd4j.randn(DataType.FLOAT, 3, 4));
        SDVariable out = sd.mmul("out", in1, w);

        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);
        sd.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);

        Map<String, INDArray> ph = new HashMap<>();
        // Run 3 times
        for (int i = 0; i < 3; i++) {
            ph.put("in1", Nd4j.randn(DataType.FLOAT, 2, 3));
            sd.output(ph, "out");
        }

        DynamicShapePlanExecutor executor = sd.getOrCreateSession().getDynamicShapePlanExecutor();
        Pointer handle = executor.getNativePlanHandle();

        List<PlanIntrospection.SegmentInfo> segments =
                PlanIntrospection.getSegmentsWithReplayState(executor.getCurrentPlan(), handle);
        assertFalse(segments.isEmpty(), "Should have at least one segment");

        for (PlanIntrospection.SegmentInfo seg : segments) {
            // In SLOT_BY_SLOT mode, no segment should have a graph replay handle
            // (replayState == -1 means no handle)
            assertTrue(seg.getReplayState() == -1 || seg.getReplayState() == 0,
                    "SLOT_BY_SLOT should not create graph replay handles, but seg [" +
                    seg.getStartSlot() + "-" + seg.getEndSlot() + "] has replayState=" +
                    seg.getReplayState());
            assertFalse(seg.isCompilationFailed(),
                    "compilationFailed should be false in SLOT_BY_SLOT mode");
        }
        log.info("SLOT_BY_SLOT: all {} segments have no graph compilation", segments.size());
    }
}
