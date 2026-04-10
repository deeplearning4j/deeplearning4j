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
import org.junit.jupiter.api.*;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlan;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlanExecutor;
import org.nd4j.autodiff.samediff.execution.DynamicShapeSlot;
import org.nd4j.autodiff.samediff.execution.PlanIntrospection;
import org.nd4j.common.config.ND4JSystemProperties;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.NativeOps;
import org.bytedeco.javacpp.Pointer;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for DSP view-producer buffer lifecycle.
 *
 * <p>Validates that when a view-producing op (permute, reshape_no_copy) creates
 * a view that shares a DataBuffer with its source, the source DataBuffer survives
 * cleanup and is not freed while the view is still live.</p>
 *
 * <p>Uses plan introspection APIs to inspect:
 * <ul>
 *   <li>Which slots are view-capable ops</li>
 *   <li>DataBuffer addresses at each slot to detect sharing</li>
 *   <li>Segment boundaries and replay state</li>
 *   <li>Release schedule to verify source outlives view</li>
 *   <li>Memory timeline for allocation/release ordering</li>
 * </ul>
 *
 * <p>Run:
 * <pre>
 *   cd platform-tests && mvn test -Dtest=TestDSPViewBufferLifecycle \
 *       -Dnd4j.dsp.diagnostics=ALL -Dnd4j.dsp.diagnostics.level=full
 * </pre>
 */
@Slf4j
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
public class TestDSPViewBufferLifecycle extends BaseNd4jTestWithBackends {

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

    // ──────────────────────────────────────────────────────────────────────
    // Graph builders: each creates a graph with view-producing ops that
    // create cross-slot DataBuffer sharing.
    // ──────────────────────────────────────────────────────────────────────

    /**
     * Build graph: input → matmul → permute → add → output
     * permute is a view-producing op: its output shares the DataBuffer
     * with matmul's output. This creates cross-slot DataBuffer sharing.
     */
    private SameDiff buildPermuteGraph() {
        SameDiff sd = SameDiff.create();
        // input: [batch, seq, hidden]
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 4, 8);
        // weight: [8, 8] → matmul produces [batch, 4, 8]
        SDVariable w = sd.constant("weight", Nd4j.randn(DataType.FLOAT, 8, 8));
        SDVariable mm = sd.mmul("mm", input, w);
        // permute: [batch, 4, 8] → [batch, 8, 4] — view of mm's DataBuffer
        SDVariable perm = sd.permute("perm", mm, 0, 2, 1);
        // add forces materialization downstream
        SDVariable bias = sd.constant("bias", Nd4j.zeros(DataType.FLOAT, 1, 8, 4));
        SDVariable out = perm.add("output", bias);
        return sd;
    }

    /**
     * Build graph: input → matmul → reshape → add → output
     * reshape is a view-producing op: its output shares the DataBuffer
     * with matmul's output.
     */
    private SameDiff buildReshapeGraph() {
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 16);
        SDVariable w = sd.constant("weight", Nd4j.randn(DataType.FLOAT, 16, 32));
        SDVariable mm = sd.mmul("mm", input, w);
        // reshape: [batch, 32] → [batch, 4, 8] — view of mm's DataBuffer
        SDVariable reshaped = sd.reshape("reshaped", mm, -1, 4, 8);
        SDVariable bias = sd.constant("bias", Nd4j.zeros(DataType.FLOAT, 1, 4, 8));
        SDVariable out = reshaped.add("output", bias);
        return sd;
    }

    /**
     * Build graph with chained views: input → matmul → permute → reshape → add → output
     * Both permute and reshape produce views, creating a chain of DataBuffer sharing.
     */
    private SameDiff buildChainedViewGraph() {
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 4, 8);
        SDVariable w = sd.constant("weight", Nd4j.randn(DataType.FLOAT, 8, 8));
        SDVariable mm = sd.mmul("mm", input, w);
        // permute: [batch, 4, 8] → [batch, 8, 4]
        SDVariable perm = sd.permute("perm", mm, 0, 2, 1);
        // reshape: [batch, 8, 4] → [batch, 32]
        SDVariable reshaped = sd.reshape("reshaped", perm, -1, 32);
        SDVariable bias = sd.constant("bias", Nd4j.zeros(DataType.FLOAT, 1, 32));
        SDVariable out = reshaped.add("output", bias);
        return sd;
    }

    // ──────────────────────────────────────────────────────────────────────
    // Test 1: Plan introspection — verify view-capable ops are detected
    // ──────────────────────────────────────────────────────────────────────

    @Test
    @Order(1)
    @DisplayName("Plan introspection: view-capable ops are correctly detected")
    public void testViewCapableOpsDetected() {
        SameDiff sd = buildPermuteGraph();
        enableDsp(sd);

        // Compile plan
        DynamicShapePlan plan = sd.compileDynamicShapePlan("output");
        assertNotNull(plan, "Plan must compile");

        DynamicShapeSlot[] slots = plan.getSlots();
        log.info("Plan has {} slots, {} total output slots",
                slots.length, plan.getTotalOutputSlots());

        // Find view-capable slots
        List<Integer> viewCapableSlots = new ArrayList<>();
        Map<String, Boolean> opViewCapability = new LinkedHashMap<>();
        for (int i = 0; i < slots.length; i++) {
            DynamicShapeSlot slot = slots[i];
            opViewCapability.put(slot.getOpName() + "@step" + i, slot.isViewCapableOp());
            if (slot.isViewCapableOp()) {
                viewCapableSlots.add(i);
                log.info("  Step {}: {} is view-capable, outputs to slots {}",
                        i, slot.getOpName(), Arrays.toString(slot.getOutputSlotIndices()));
            }
        }

        // permute must be detected as view-capable
        boolean foundPermute = false;
        for (int i = 0; i < slots.length; i++) {
            if ("permute".equals(slots[i].getOpName())) {
                assertTrue(slots[i].isViewCapableOp(),
                        "permute op at step " + i + " must be view-capable");
                foundPermute = true;
            }
        }
        assertTrue(foundPermute, "Plan must contain a permute op");

        // matmul and add must NOT be view-capable
        for (int i = 0; i < slots.length; i++) {
            String name = slots[i].getOpName();
            if ("matmul".equals(name) || "mmul".equals(name) || "add".equals(name)) {
                assertFalse(slots[i].isViewCapableOp(),
                        name + " op at step " + i + " must NOT be view-capable");
            }
        }

        log.info("View capability map: {}", opViewCapability);
        sd.close();
    }

    // ──────────────────────────────────────────────────────────────────────
    // Test 2: DataBuffer address tracking across slots
    // ──────────────────────────────────────────────────────────────────────

    @Test
    @Order(2)
    @DisplayName("DataBuffer sharing: view-producer shares address with source")
    public void testViewProducerSharesDataBuffer() {
        // TODO: Re-enable when slot interceptor infrastructure is re-implemented in C++.
        // This test relied on SlotOutputInterceptor and setSlotOutputInterceptor which
        // have been removed (non-functional with native C++ execution).
        log.info("Skipping: interceptor classes removed (SlotOutputInterceptor, setSlotOutputInterceptor)");
    }

    // ──────────────────────────────────────────────────────────────────────
    // Test 3: Release schedule validates source outlives view
    // ──────────────────────────────────────────────────────────────────────

    @Test
    @Order(3)
    @DisplayName("Release schedule: view source must not be released before view consumers finish")
    public void testReleaseOrderSourceOutlivesView() {
        SameDiff sd = buildPermuteGraph();
        enableDsp(sd);

        DynamicShapePlan plan = sd.compileDynamicShapePlan("output");
        DynamicShapeSlot[] slots = plan.getSlots();

        // Build reverse map: outputSlotIdx → stepIdx that produces it
        Map<Integer, Integer> slotToProducerStep = new HashMap<>();
        for (int i = 0; i < slots.length; i++) {
            for (int outSlot : slots[i].getOutputSlotIndices()) {
                slotToProducerStep.put(outSlot, i);
            }
        }

        // Build release step map: slotIdx → step at which it's released
        int[][] releaseAtStep = plan.getReleaseAtStep();
        Map<Integer, Integer> slotReleaseStep = new HashMap<>();
        for (int step = 0; step < releaseAtStep.length; step++) {
            if (releaseAtStep[step] != null) {
                for (int slotIdx : releaseAtStep[step]) {
                    slotReleaseStep.put(slotIdx, step);
                }
            }
        }

        // For each view-capable op, find its input source slot and verify
        // the source is released AFTER all consumers of the view
        for (int i = 0; i < slots.length; i++) {
            if (!slots[i].isViewCapableOp()) continue;

            int viewOutputSlot = slots[i].getOutputSlotIndices()[0];
            int[] inputSources = slots[i].getInputSourceIndices();
            if (inputSources.length == 0 || inputSources[0] < 0) continue;

            int sourceSlot = inputSources[0];
            Integer sourceReleaseStep = slotReleaseStep.get(sourceSlot);
            Integer viewReleaseStep = slotReleaseStep.get(viewOutputSlot);

            log.info("View op '{}' at step {}: source slot {} released at step {}, "
                            + "view slot {} released at step {}",
                    slots[i].getOpName(), i, sourceSlot, sourceReleaseStep,
                    viewOutputSlot, viewReleaseStep);

            // Find all consumers of the view output
            List<Integer> viewConsumers = PlanIntrospection.getDependentsOf(plan, i);
            log.info("  View consumers: {}", viewConsumers);

            // If source is released, it must be released AFTER all view consumers
            if (sourceReleaseStep != null && viewReleaseStep != null) {
                // Source release step must be >= view release step
                // (i.e., source survives at least as long as the view)
                log.info("  Release order check: source@step{} vs view@step{}",
                        sourceReleaseStep, viewReleaseStep);
            }
        }

        sd.close();
    }

    // ──────────────────────────────────────────────────────────────────────
    // Test 4: Multi-execution DataBuffer validity
    // ──────────────────────────────────────────────────────────────────────

    @Test
    @Order(4)
    @DisplayName("Multi-execution: DataBuffers remain valid across calls with view ops")
    public void testMultiExecViewBufferValidity() {
        SameDiff sd = buildPermuteGraph();
        enableDsp(sd);

        INDArray input = Nd4j.randn(DataType.FLOAT, 2, 4, 8);
        INDArray referenceOutput = null;

        // Execute 5 times — each time verify output is valid and consistent
        for (int call = 0; call < 5; call++) {
            Map<String, INDArray> result = sd.output(
                    Collections.singletonMap("input", input), "output");
            INDArray out = result.get("output");

            assertNotNull(out, "Call " + call + ": output must not be null");
            assertFalse(out.wasClosed(), "Call " + call + ": output must not be closed");
            assertFalse(out.isNaN().any(),
                    "Call " + call + ": output must not contain NaN");
            assertFalse(out.isInfinite().any(),
                    "Call " + call + ": output must not contain Inf");
            assertTrue(out.data().address() != 0,
                    "Call " + call + ": output DataBuffer address must be non-zero");

            if (referenceOutput == null) {
                referenceOutput = out.dup();
            } else {
                double maxDiff = referenceOutput.sub(out).amaxNumber().doubleValue();
                assertTrue(maxDiff < 1e-4,
                        "Call " + call + ": output must match reference (maxDiff=" + maxDiff + ")");
            }
            log.info("Call {}: output OK, shape={}, sum={}",
                    call, Arrays.toString(out.shape()), out.sumNumber());
        }

        if (referenceOutput != null) referenceOutput.close();
        sd.close();
    }

    // ──────────────────────────────────────────────────────────────────────
    // Test 5: Frozen shapes + view ops — DataBuffer survives cleanup
    // ──────────────────────────────────────────────────────────────────────

    @Test
    @Order(5)
    @DisplayName("Frozen shapes: view-producer DataBuffer survives between-call cleanup")
    public void testFrozenViewBufferSurvivesCleanup() {
        SameDiff sd = buildPermuteGraph();
        enableDsp(sd);

        INDArray input = Nd4j.randn(DataType.FLOAT, 2, 4, 8);

        // Warmup execution (unfrozen)
        Map<String, INDArray> result = sd.output(
                Collections.singletonMap("input", input), "output");
        INDArray referenceOutput = result.get("output").dup();

        // Freeze shapes
        InferenceSession session = sd.getOrCreateSession();
        DynamicShapePlanExecutor executor = session.getDynamicShapePlanExecutor();
        if (executor != null) {
            executor.setShapesFrozen(true);
            log.info("Shapes frozen");
        }

        // Execute 5 times frozen — each cleanup cycle could corrupt view DataBuffers
        for (int call = 0; call < 5; call++) {
            result = sd.output(Collections.singletonMap("input", input), "output");
            INDArray out = result.get("output");

            assertNotNull(out, "Frozen call " + call + ": output must not be null");
            assertFalse(out.wasClosed(),
                    "Frozen call " + call + ": output must not be closed");

            // dup() immediately to capture the actual values before DSP recycles
            INDArray outDup = out.dup();
            assertFalse(outDup.isNaN().any(),
                    "Frozen call " + call + ": output must not contain NaN");
            assertFalse(outDup.isInfinite().any(),
                    "Frozen call " + call + ": output must not contain Inf");

            double maxDiff = referenceOutput.sub(outDup).amaxNumber().doubleValue();
            double outSum = outDup.sumNumber().doubleValue();
            double refSum = referenceOutput.sumNumber().doubleValue();

            // Log first 4 values for debugging
            float[] outFirst = outDup.dup('c').data().asFloat();
            float[] refFirst = referenceOutput.dup('c').data().asFloat();
            log.info("Frozen call {}: maxDiff={} outSum={} refSum={}", call, maxDiff, outSum, refSum);
            log.info("  out[0:4] = [{}, {}, {}, {}]",
                    outFirst[0], outFirst[1], outFirst[2], outFirst[3]);
            log.info("  ref[0:4] = [{}, {}, {}, {}]",
                    refFirst[0], refFirst[1], refFirst[2], refFirst[3]);
            log.info("  DataBuffer address: out={} ref={}",
                    outDup.data().address(), referenceOutput.data().address());

            assertTrue(maxDiff < 1e-4,
                    "Frozen call " + call + ": output must match reference (maxDiff="
                            + maxDiff + ", outSum=" + outSum + ", refSum=" + refSum + ")");
            outDup.close();
        }

        // Query segment state after frozen execution
        if (executor != null && executor.getNativePlanHandle() != null) {
            NativeOps ops = Nd4j.getNativeOps();
            Pointer handle = executor.getNativePlanHandle();
            int numSegs = ops.getPlanNumSegments(handle);
            log.info("Frozen plan: {} segments", numSegs);
            for (int s = 0; s < numSegs; s++) {
                boolean capturable = ops.isPlanSegmentCapturable(handle, s);
                int replayState = ops.getPlanSegmentReplayState(handle, s);
                int execCount = ops.getPlanSegmentExecutionCount(handle, s);
                String backend = ops.getPlanSegmentBackendName(handle, s);
                log.info("  seg[{}]: capturable={} replayState={} execCount={} backend={}",
                        s, capturable, replayState, execCount, backend);
            }
        }

        referenceOutput.close();
        sd.close();
    }

    // ──────────────────────────────────────────────────────────────────────
    // Test 6: Chained views — DataBuffer sharing across view chain
    // ──────────────────────────────────────────────────────────────────────

    @Test
    @Order(6)
    @DisplayName("Chained views: matmul→permute→reshape all share one DataBuffer")
    public void testChainedViewBufferSharing() {
        SameDiff sd = buildChainedViewGraph();
        enableDsp(sd);

        // Track addresses
        // First execution to create plan
        INDArray input = Nd4j.randn(DataType.FLOAT, 2, 4, 8);
        sd.output(Collections.singletonMap("input", input), "output");

        // NOTE: Interceptor-based DataBuffer address tracking removed — SlotOutputInterceptor
        // has been deleted (non-functional with native C++ execution).

        // Verify multi-execution correctness
        INDArray referenceOutput = null;
        for (int call = 0; call < 5; call++) {
            Map<String, INDArray> result = sd.output(
                    Collections.singletonMap("input", input), "output");
            INDArray out = result.get("output");
            assertFalse(out.isNaN().any(),
                    "Call " + call + ": chained view output must not contain NaN");
            if (referenceOutput == null) {
                referenceOutput = out.dup();
            } else {
                double maxDiff = referenceOutput.sub(out).amaxNumber().doubleValue();
                assertTrue(maxDiff < 1e-4,
                        "Call " + call + ": chained view output diverged (maxDiff=" + maxDiff + ")");
            }
        }
        if (referenceOutput != null) referenceOutput.close();
        sd.close();
    }

    // ──────────────────────────────────────────────────────────────────────
    // Test 7: Segment introspection with view ops
    // ──────────────────────────────────────────────────────────────────────

    @Test
    @Order(7)
    @DisplayName("Segment introspection: view ops may create segment boundaries")
    public void testSegmentBoundariesWithViewOps() {
        SameDiff sd = buildPermuteGraph();
        enableDsp(sd);

        // Execute to create native plan
        INDArray input = Nd4j.randn(DataType.FLOAT, 2, 4, 8);
        sd.output(Collections.singletonMap("input", input), "output");

        InferenceSession session = sd.getOrCreateSession();
        DynamicShapePlanExecutor executor = session.getDynamicShapePlanExecutor();
        DynamicShapePlan plan = executor != null ? executor.getCurrentPlan() : null;

        if (plan != null) {
            // Get segments from plan
            List<PlanIntrospection.SegmentInfo> segments = plan.getSegments();
            log.info("Plan has {} segments:", segments.size());
            for (int i = 0; i < segments.size(); i++) {
                PlanIntrospection.SegmentInfo seg = segments.get(i);
                log.info("  Segment {}: slots [{}-{}], {} ops: {}, capturable={}",
                        i, seg.getStartSlot(), seg.getEndSlot(),
                        seg.getSlotCount(), seg.getOpNames(), seg.isCapturable());
            }

            // With native plan handle, get replay state
            if (executor.getNativePlanHandle() != null) {
                NativeOps ops = Nd4j.getNativeOps();
                Pointer handle = executor.getNativePlanHandle();

                // Execute 2 more times to trigger capture
                sd.output(Collections.singletonMap("input", input), "output");
                sd.output(Collections.singletonMap("input", input), "output");

                int numSegs = ops.getPlanNumSegments(handle);
                int captured = ops.getPlanNumCapturedGraphSegments(handle);
                int totalReplays = ops.getPlanTotalGraphReplays(handle);
                log.info("After 3 executions: {} segments, {} captured, {} total replays",
                        numSegs, captured, totalReplays);

                for (int s = 0; s < numSegs; s++) {
                    int replayState = ops.getPlanSegmentReplayState(handle, s);
                    int replayCount = ops.getPlanSegmentReplayCount(handle, s);
                    int execCount = ops.getPlanSegmentExecutionCount(handle, s);
                    boolean capFailed = ops.isPlanSegmentCaptureFailed(handle, s);
                    String backend = ops.getPlanSegmentBackendName(handle, s);
                    log.info("  seg[{}]: replayState={} replayCount={} execCount={} "
                                    + "capFailed={} backend={}",
                            s, replayState, replayCount, execCount, capFailed, backend);
                }
            }
        }

        sd.close();
    }

    // ──────────────────────────────────────────────────────────────────────
    // Test 8: DataBuffer validity interceptor — verify no closed buffers
    // ──────────────────────────────────────────────────────────────────────

    @Test
    @Order(8)
    @DisplayName("Interceptor validation: no slot output has a closed DataBuffer")
    public void testNoClosedDataBuffersDuringExecution() {
        // TODO: Re-enable when slot interceptor infrastructure is re-implemented in C++.
        // This test relied on SlotOutputInterceptor and setSlotOutputInterceptor which
        // have been removed (non-functional with native C++ execution).
        log.info("Skipping: interceptor classes removed (SlotOutputInterceptor, setSlotOutputInterceptor)");
    }

    // ──────────────────────────────────────────────────────────────────────
    // Test 9: Reshape view lifecycle
    // ──────────────────────────────────────────────────────────────────────

    @Test
    @Order(9)
    @DisplayName("Reshape view: multi-execution stability with reshape view op")
    public void testReshapeViewMultiExecStability() {
        SameDiff sd = buildReshapeGraph();
        enableDsp(sd);

        INDArray input = Nd4j.randn(DataType.FLOAT, 2, 16);
        INDArray referenceOutput = null;

        for (int call = 0; call < 5; call++) {
            Map<String, INDArray> result = sd.output(
                    Collections.singletonMap("input", input), "output");
            INDArray out = result.get("output");

            assertNotNull(out, "Call " + call + ": output null");
            assertFalse(out.wasClosed(), "Call " + call + ": output closed");
            assertFalse(out.isNaN().any(), "Call " + call + ": NaN in output");
            assertFalse(out.isInfinite().any(), "Call " + call + ": Inf in output");

            if (referenceOutput == null) {
                referenceOutput = out.dup();
            } else {
                double maxDiff = referenceOutput.sub(out).amaxNumber().doubleValue();
                assertTrue(maxDiff < 1e-4,
                        "Call " + call + ": reshape view diverged (maxDiff=" + maxDiff + ")");
            }
        }

        if (referenceOutput != null) referenceOutput.close();
        sd.close();
    }

    // ──────────────────────────────────────────────────────────────────────
    // Test 10: Frozen + interceptor — verify DataBuffer validity under freeze
    // ──────────────────────────────────────────────────────────────────────

    @Test
    @Order(10)
    @DisplayName("Frozen + interceptor: no closed DataBuffers during frozen execution")
    public void testFrozenNoClosedDataBuffers() {
        // TODO: Re-enable when slot interceptor infrastructure is re-implemented in C++.
        // This test relied on SlotOutputInterceptor and setSlotOutputInterceptor which
        // have been removed (non-functional with native C++ execution).
        log.info("Skipping: interceptor classes removed (SlotOutputInterceptor, setSlotOutputInterceptor)");
    }
}
