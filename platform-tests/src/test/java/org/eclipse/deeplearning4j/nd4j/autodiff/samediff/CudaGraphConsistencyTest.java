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
import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.Pointer;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlan;
import org.nd4j.autodiff.samediff.execution.DynamicShapeSlot;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.api.shape.options.ArrayOptionsHelper;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;
import org.nd4j.nativeblas.OpaqueContext;
import org.nd4j.nativeblas.OpaqueDataBuffer;
import org.nd4j.nativeblas.OpaqueLaunchContext;
import org.nd4j.nativeblas.OpaqueNDArray;

import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests that validate CUDA graph capture is consistent with the SameDiff graph.
 *
 * When CUDA graphs are enabled, ops in the execution plan are captured into a
 * cudaGraph_t during the second execution. On subsequent executions, the captured
 * graph is replayed instead of re-executing ops individually.
 *
 * The critical invariant is: EVERY op in the SameDiff graph must contribute at
 * least one CUDA graph node. Ops that do host-only work (e.g., shape_of which
 * writes to host via memcpy then syncs to device) may produce zero CUDA graph
 * nodes. Their host work runs during capture but is NOT replayed, causing stale
 * outputs on the 2nd+ execution.
 *
 * IMPORTANT: CUDA graph capture requires segments with at least 10 ops (the
 * minimum threshold). Tests build sufficiently large graphs to trigger capture.
 * Smaller graphs fall back to slot-by-slot execution (no capture/replay).
 */
@Slf4j
@Tag(TagNames.SAMEDIFF)
@NativeTag
public class CudaGraphConsistencyTest extends BaseNd4jTestWithBackends {

    private static final double TOLERANCE = 1e-4;
    // Minimum ops needed to trigger CUDA graph capture
    private static final int MIN_CAPTURE_OPS = 10;

    // ─── Helper methods ──────────────────────────────────────────────────────

    private Pointer compileNativePlan(DynamicShapePlan plan) {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        byte[] serialized = plan.serialize();
        assertNotNull(serialized, "Plan serialization returned null");
        assertTrue(serialized.length > 0, "Plan serialization returned empty");

        BytePointer planBytes = new BytePointer(serialized);
        try {
            return nativeOps.compileDynamicShapePlan(planBytes, serialized.length);
        } catch (UnsupportedOperationException e) {
            log.info("Backend does not support native executor: {}", e.getMessage());
            return null;
        } finally {
            planBytes.close();
        }
    }

    private Map<String, INDArray> executeNativePlan(Pointer planHandle, DynamicShapePlan plan,
                                                     INDArray[] extInputs) {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        int numOutputs = plan.getRequestedOutputs().size();

        OpaqueContext opContext = nativeOps.createGraphContext(1);
        try {
            for (int i = 0; i < extInputs.length; i++) {
                OpaqueNDArray opaqueIn = OpaqueNDArray.fromINDArray(extInputs[i]);
                nativeOps.setGraphContextInputArray(opContext, i, opaqueIn);
            }

            for (int i = 0; i < numOutputs; i++) {
                INDArray dummy = Nd4j.empty(DataType.FLOAT);
                OpaqueNDArray opaqueOut = OpaqueNDArray.fromINDArray(dummy);
                nativeOps.setGraphContextOutputArray(opContext, i, opaqueOut);
            }

            Pointer execStream = null;
            try {
                OpaqueLaunchContext lc = nativeOps.defaultLaunchContext();
                if (lc != null) {
                    execStream = nativeOps.lcExecutionStream(lc);
                    if (execStream != null) execStream.retainReference();
                }
            } catch (Exception e) {
                // CPU backend — stream stays null
            }

            int status = nativeOps.executeDynamicShapePlan(planHandle, opContext, execStream);

            if (status != 0) {
                String errMsg = nativeOps.lastErrorMessage();
                nativeOps.clearLastError();
                fail("Native plan execution failed with status " + status + ": " + errMsg);
            }

            Map<String, INDArray> results = new LinkedHashMap<>();
            List<String> requestedOutputs = new java.util.ArrayList<>(plan.getRequestedOutputs());

            for (int i = 0; i < numOutputs; i++) {
                OpaqueNDArray opaqueOut = nativeOps.getOutputArrayNative(opContext, i);
                assertNotNull(opaqueOut, "Null output at index " + i);
                assertFalse(opaqueOut.isNull(), "Null OpaqueNDArray at index " + i);

                long[] shapeInfo = OpaqueNDArray.getOpaqueNDArrayShapeInfo(opaqueOut);
                long[] shape = Shape.shape(shapeInfo);
                DataType dtype = ArrayOptionsHelper.dataType(shapeInfo);
                long length = OpaqueNDArray.getOpaqueNDArrayLength(opaqueOut);

                INDArray result = Nd4j.createUninitialized(dtype, shape);

                Pointer nativePrimary = nativeOps.getOpaqueNDArrayBuffer(opaqueOut);
                Pointer nativeSpecial = nativeOps.getOpaqueNDArraySpecialBuffer(opaqueOut);

                OpaqueDataBuffer srcOdb = nativeOps.dbCreateExternalDataBuffer(
                        length, dtype.toInt(), nativePrimary, nativeSpecial);
                if (srcOdb != null) {
                    OpaqueDataBuffer dstOdb = result.data().opaqueBuffer();
                    if (dstOdb != null) {
                        nativeOps.copyBuffer(dstOdb, length, srcOdb, 0, 0);
                    }
                }

                results.put(requestedOutputs.get(i), result);
            }

            return results;
        } finally {
            nativeOps.deleteGraphContext(opContext);
        }
    }

    private INDArray[] resolveExternalInputs(DynamicShapePlan plan, SameDiff sd,
                                              Map<String, INDArray> placeholders) {
        String[] extKeys = plan.getExternalInputKeys();
        INDArray[] extInputs = new INDArray[extKeys.length];
        for (int i = 0; i < extKeys.length; i++) {
            String varName = extKeys[i];
            INDArray arr = placeholders != null ? placeholders.get(varName) : null;
            if (arr == null) {
                SDVariable var = sd.getVariable(varName);
                if (var != null) {
                    arr = var.getArr();
                }
            }
            assertNotNull(arr, "Missing external input: " + varName);
            extInputs[i] = arr;
        }
        return extInputs;
    }

    /**
     * Run a CUDA graph consistency test for a given SameDiff graph.
     * Verifies:
     * 1. Plan slot count matches SameDiff op count for requested outputs
     * 2. CUDA graph validates (no host-only ops) after capture
     * 3. Output correctness on warmup, capture, and replay executions
     * 4. Replay count increases (proves graph is being reused)
     */
    private void assertCudaGraphConsistency(SameDiff sd, Map<String, INDArray> placeholders,
                                             String... outputs) {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();

        // NOTE: Do NOT enable debug/verbose mode here! Debug mode causes
        // DebugHelper::checkErrorCode to call cudaStreamSynchronize after every op,
        // which is ILLEGAL during CUDA graph capture (error 900).
        // The C++ capture audit is always-on and doesn't need debug mode.

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, outputs);
        Pointer handle = compileNativePlan(plan);
        if (handle == null || handle.isNull()) {
            log.info("Skipping CUDA graph consistency test — native executor not supported");
            plan.close();
            return;
        }

        try {
            // Log plan structure
            int numSlots = nativeOps.getPlanNumSlots(handle);
            int numSegments = nativeOps.getPlanNumSegments(handle);
            log.info("Plan: {} slots, {} segments, outputs={}", numSlots, numSegments,
                     Arrays.toString(outputs));

            // Log SameDiff ops for comparison
            DynamicShapeSlot[] slots = plan.getSlots();
            log.info("SameDiff plan has {} slots:", slots.length);
            for (int i = 0; i < slots.length; i++) {
                log.info("  slot[{}]: op={}", i, slots[i].getOpName());
            }

            // Verify slot count matches
            assertEquals(slots.length, numSlots,
                    "Plan slot count mismatch between Java (" + slots.length +
                    ") and native (" + numSlots + ")");

            // Enable CUDA Graphs (uses default min segment size of 10)
            nativeOps.setPlanCudaGraphsEnabled(handle, true);

            // Get Java executor results for correctness baseline
            Map<String, INDArray> javaResults = sd.output(placeholders, outputs);

            // Execute 4 times: warmup → capture → replay → replay
            for (int iter = 0; iter < 4; iter++) {
                INDArray[] extInputs = resolveExternalInputs(plan, sd, placeholders);
                Map<String, INDArray> nativeResult = executeNativePlan(handle, plan, extInputs);

                // Verify correctness on EVERY execution
                for (String output : outputs) {
                    INDArray expected = javaResults.get(output);
                    INDArray actual = nativeResult.get(output);
                    assertNotNull(actual, "Missing output '" + output + "' on iter " + iter);

                    assertArrayEquals(expected.shape(), actual.shape(),
                            "Shape mismatch for '" + output + "' on iter " + iter +
                            ": expected " + Arrays.toString(expected.shape()) +
                            " got " + Arrays.toString(actual.shape()));

                    if (expected.dataType().isFPType()) {
                        double maxDiff = expected.sub(actual).amaxNumber().doubleValue();
                        assertTrue(maxDiff <= TOLERANCE,
                                "Value mismatch for '" + output + "' on iter " + iter +
                                ": maxDiff=" + maxDiff + " (tolerance=" + TOLERANCE + ")");
                    } else {
                        // For integer types, exact match
                        assertEquals(expected, actual,
                                "Value mismatch for '" + output + "' on iter " + iter);
                    }
                }

                int captured = nativeOps.getPlanNumCapturedGraphSegments(handle);
                int replays = nativeOps.getPlanTotalGraphReplays(handle);
                log.info("  iter {}: captured={}, replays={}", iter, captured, replays);

                // After iter 1 (capture pass), validate the graph
                if (iter == 1 && captured > 0) {
                    // Print the full CUDA graph debug info
                    nativeOps.printPlanCapturedGraphDebug(handle);

                    // Validate: every op must have contributed CUDA graph nodes
                    boolean valid = nativeOps.validatePlanCapturedGraph(handle);
                    int hostOnlyCount = nativeOps.getPlanNumHostOnlyOps(handle);
                    String hostOnlyNames = nativeOps.getPlanHostOnlyOpNames(handle);

                    log.info("CUDA Graph Validation: valid={}, hostOnlyOps={} [{}]",
                             valid, hostOnlyCount, hostOnlyNames);

                    if (!valid) {
                        fail("CUDA graph capture is INCOMPLETE: " + hostOnlyCount +
                             " ops contributed zero CUDA graph nodes: [" + hostOnlyNames + "]. " +
                             "These ops do host-only work that won't replay on graph re-execution. " +
                             "The CUDA graph is NOT consistent with the SameDiff graph.");
                    }
                }
            }

            // After 4 executions with a graph large enough for capture:
            // - iter 0: warmup (slot-by-slot)
            // - iter 1: capture (graph recorded + launched)
            // - iter 2: replay
            // - iter 3: replay
            // So we expect at least 2 replays if capture succeeded
            int finalReplays = nativeOps.getPlanTotalGraphReplays(handle);
            log.info("Final: {} graph replays", finalReplays);
            if (numSlots >= MIN_CAPTURE_OPS) {
                assertTrue(finalReplays >= 2,
                        "Expected at least 2 graph replays for a " + numSlots +
                        "-slot plan, got " + finalReplays);
            }

        } finally {
            nativeOps.freeDynamicShapePlan(handle);
            plan.close();
        }
    }

    // ─── Graph builders: build large enough graphs (12+ ops) for capture ─────

    /**
     * Build a chain of 12 element-wise ops: add → sigmoid → mul → tanh → add → ...
     * This produces exactly 12 slots to exceed the 10-slot capture threshold.
     */
    private static SameDiff createLargeChainGraph() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable scale = sd.constant("scale", Nd4j.ones(DataType.FLOAT, 1, 4).mul(0.5));
        SDVariable bias = sd.constant("bias", Nd4j.ones(DataType.FLOAT, 1, 4).mul(0.1));

        // Chain of 12 ops to exceed the 10-slot capture threshold
        SDVariable t = x.add("op_01_add", bias);
        t = sd.nn().sigmoid("op_02_sigmoid", t);
        t = t.mul("op_03_mul", scale);
        t = sd.nn().tanh("op_04_tanh", t);
        t = t.add("op_05_add2", bias);
        t = sd.nn().sigmoid("op_06_sigmoid2", t);
        t = t.mul("op_07_mul2", scale);
        t = sd.nn().tanh("op_08_tanh2", t);
        t = t.add("op_09_add3", bias);
        t = sd.nn().sigmoid("op_10_sigmoid3", t);
        t = t.mul("op_11_mul3", scale);
        SDVariable output = sd.nn().tanh("result", t);

        return sd;
    }

    /**
     * Build a diamond graph with 12+ ops: two parallel paths of 6 ops each, merged.
     */
    private static SameDiff createLargeDiamondGraph() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable scale = sd.constant("scale", Nd4j.ones(DataType.FLOAT, 1, 4).mul(0.5));
        SDVariable bias = sd.constant("bias", Nd4j.ones(DataType.FLOAT, 1, 4).mul(0.1));

        // Path A: 6 ops
        SDVariable a1 = x.add("a_01_add", bias);
        SDVariable a2 = sd.nn().sigmoid("a_02_sigmoid", a1);
        SDVariable a3 = a2.mul("a_03_mul", scale);
        SDVariable a4 = sd.nn().tanh("a_04_tanh", a3);
        SDVariable a5 = a4.add("a_05_add", bias);
        SDVariable a6 = sd.nn().sigmoid("a_06_sigmoid", a5);

        // Path B: 6 ops
        SDVariable b1 = x.mul("b_01_mul", scale);
        SDVariable b2 = sd.nn().tanh("b_02_tanh", b1);
        SDVariable b3 = b2.add("b_03_add", bias);
        SDVariable b4 = sd.nn().sigmoid("b_04_sigmoid", b3);
        SDVariable b5 = b4.mul("b_05_mul", scale);
        SDVariable b6 = sd.nn().tanh("b_06_tanh", b5);

        // Merge: 1 op
        SDVariable result = a6.add("result", b6);

        return sd;
    }

    /**
     * Build a graph with shape_of + gather + enough compute ops (12+ total)
     * to trigger capture. The shape_of is the problematic host-only op.
     */
    private static SameDiff createShapeOfPlusComputeGraph() {
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, -1, -1, -1);

        // shape_of op (host-only — problematic for CUDA graphs)
        SDVariable shape = sd.shape("shape_out", input);

        // gather op (reads from shape_of output)
        INDArray scalarIdx = Nd4j.scalar(DataType.INT64, 0);
        SDVariable idx = sd.constant("idx", scalarIdx);
        SDVariable gathered = sd.gather("gather_out", shape, idx, 0);

        // 10+ element-wise ops on the input to exceed capture threshold
        SDVariable scale = sd.constant("scale", Nd4j.ones(DataType.FLOAT, 1, 1, 1, 1).mul(0.5));
        SDVariable bias = sd.constant("bias", Nd4j.ones(DataType.FLOAT, 1, 1, 1, 1).mul(0.1));

        SDVariable t = input.add("op_01_add", bias);
        t = sd.nn().sigmoid("op_02_sigmoid", t);
        t = t.mul("op_03_mul", scale);
        t = sd.nn().tanh("op_04_tanh", t);
        t = t.add("op_05_add2", bias);
        t = sd.nn().sigmoid("op_06_sigmoid2", t);
        t = t.mul("op_07_mul2", scale);
        t = sd.nn().tanh("op_08_tanh2", t);
        t = t.add("op_09_add3", bias);
        SDVariable compute_result = sd.nn().sigmoid("compute_result", t);

        return sd;
    }

    /**
     * Build a multi-output graph with 12+ ops where intermediate results
     * are all requested as outputs.
     */
    private static SameDiff createLargeMultiOutputGraph() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable w1 = sd.constant("w1", Nd4j.ones(DataType.FLOAT, 1, 4).mul(0.5));
        SDVariable w2 = sd.constant("w2", Nd4j.ones(DataType.FLOAT, 1, 4).mul(0.3));
        SDVariable b = sd.constant("b", Nd4j.ones(DataType.FLOAT, 1, 4).mul(0.1));

        // Layer 1
        SDVariable h1 = x.mul("h1_mul", w1);
        SDVariable h2 = h1.add("h1_bias", b);
        SDVariable h3 = sd.nn().sigmoid("h1_act", h2);

        // Layer 2
        SDVariable h4 = h3.mul("h2_mul", w2);
        SDVariable h5 = h4.add("h2_bias", b);
        SDVariable h6 = sd.nn().tanh("h2_act", h5);

        // Layer 3
        SDVariable h7 = h6.mul("h3_mul", w1);
        SDVariable h8 = h7.add("h3_bias", b);
        SDVariable h9 = sd.nn().sigmoid("h3_act", h8);

        // Layer 4
        SDVariable h10 = h9.mul("h4_mul", w2);
        SDVariable h11 = h10.add("h4_bias", b);
        SDVariable result = sd.nn().tanh("result", h11);

        return sd;
    }

    // ─── Test: long chain graph (12 ops → triggers capture) ──────────────────

    @Test
    public void testCudaGraphConsistency_LargeChain() {
        SameDiff sd = createLargeChainGraph();
        INDArray x = Nd4j.randn(DataType.FLOAT, 2, 4);
        Map<String, INDArray> ph = Map.of("x", x);

        assertCudaGraphConsistency(sd, ph, "result");
    }

    // ─── Test: diamond graph (12+ ops → triggers capture) ────────────────────

    @Test
    public void testCudaGraphConsistency_LargeDiamond() {
        SameDiff sd = createLargeDiamondGraph();
        INDArray x = Nd4j.randn(DataType.FLOAT, 2, 4);
        Map<String, INDArray> ph = Map.of("x", x);

        assertCudaGraphConsistency(sd, ph, "result");
    }

    // ─── Test: shape_of + gather + compute (the problematic pattern) ─────────

    @Test
    public void testCudaGraphConsistency_ShapeOfGather() {
        SameDiff sd = createShapeOfPlusComputeGraph();
        INDArray input4d = Nd4j.randn(DataType.FLOAT, 2, 3, 4, 4);
        Map<String, INDArray> ph = Map.of("input", input4d);

        // This tests the exact pattern that fails with CUDA graphs:
        // shape_of does host memcpy → syncToDevice, which produces zero CUDA
        // graph nodes. On graph replay, shape_of's host work doesn't re-execute,
        // so gather reads stale device zeros.
        // The graph has enough ops (12+) to trigger capture.
        assertCudaGraphConsistency(sd, ph, "gather_out", "shape_out", "compute_result");
    }

    // ─── Test: multi-output graph (12 ops, multiple outputs requested) ───────

    @Test
    public void testCudaGraphConsistency_MultiOutput() {
        SameDiff sd = createLargeMultiOutputGraph();
        INDArray xArr = Nd4j.randn(DataType.FLOAT, 2, 4);
        Map<String, INDArray> ph = Map.of("x", xArr);

        // Request intermediate + final outputs
        assertCudaGraphConsistency(sd, ph, "h1_act", "h2_act", "h3_act", "result");
    }

    // ─── Test: replay correctness over many iterations ───────────────────────

    @Test
    public void testCudaGraphReplayCorrectness() {
        // This test verifies that graph REPLAY produces correct results,
        // not just the first capture execution. The bug with host-only ops
        // only manifests on the 2nd+ execution when the graph is replayed.
        SameDiff sd = createLargeChainGraph();
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();

        String[] outputs = {"result"};
        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, outputs);
        Pointer handle = compileNativePlan(plan);
        if (handle == null || handle.isNull()) {
            plan.close();
            return;
        }

        try {
            nativeOps.setPlanCudaGraphsEnabled(handle, true);

            INDArray x = Nd4j.ones(DataType.FLOAT, 3, 4).mul(0.5);
            Map<String, INDArray> ph = Map.of("x", x);

            // Get expected result from Java executor
            Map<String, INDArray> expected = sd.output(ph, outputs);
            INDArray expectedResult = expected.get("result");

            // Execute 10 times — all should produce identical results
            for (int iter = 0; iter < 10; iter++) {
                INDArray[] extInputs = resolveExternalInputs(plan, sd, ph);
                Map<String, INDArray> result = executeNativePlan(handle, plan, extInputs);
                INDArray actual = result.get("result");

                double maxDiff = expectedResult.sub(actual).amaxNumber().doubleValue();
                assertTrue(maxDiff <= TOLERANCE,
                        "Replay iter " + iter + ": maxDiff=" + maxDiff +
                        " (replays=" + nativeOps.getPlanTotalGraphReplays(handle) + ")");
            }

            // Expect graph replays if segment was large enough for capture
            int numSlots = nativeOps.getPlanNumSlots(handle);
            int replays = nativeOps.getPlanTotalGraphReplays(handle);
            log.info("Replay correctness: {} slots, {} replays over 10 executions", numSlots, replays);
            if (numSlots >= MIN_CAPTURE_OPS) {
                assertTrue(replays >= 7,
                        "Expected at least 7 replays over 10 executions with " +
                        numSlots + " slots, got " + replays);
            }

        } finally {
            nativeOps.freeDynamicShapePlan(handle);
            plan.close();
        }
    }

    // ─── Test: plan structure introspection ──────────────────────────────────

    @Test
    public void testPlanIntrospection() {
        // Verify plan introspection APIs return sane values
        SameDiff sd = createLargeChainGraph();
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();

        String[] outputs = {"result"};
        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, outputs);
        Pointer handle = compileNativePlan(plan);
        if (handle == null || handle.isNull()) {
            plan.close();
            return;
        }

        try {
            int numSlots = nativeOps.getPlanNumSlots(handle);
            int numSegments = nativeOps.getPlanNumSegments(handle);
            int numInputs = nativeOps.getPlanNumExternalInputs(handle);
            int numOutputs = nativeOps.getPlanNumRequestedOutputs(handle);

            log.info("Plan introspection: {} slots, {} segments, {} inputs, {} outputs",
                     numSlots, numSegments, numInputs, numOutputs);

            assertTrue(numSlots >= 12, "Expected at least 12 slots, got " + numSlots);
            assertTrue(numSegments >= 1, "Expected at least 1 segment, got " + numSegments);
            // External inputs include placeholders AND constants (scale, bias)
            assertTrue(numInputs >= 1, "Expected at least 1 external input, got " + numInputs);
            assertEquals(1, numOutputs, "Expected 1 requested output (result)");

            // Before any execution, no captures or replays
            assertEquals(0, nativeOps.getPlanNumCapturedGraphSegments(handle));
            assertEquals(0, nativeOps.getPlanTotalGraphReplays(handle));

        } finally {
            nativeOps.freeDynamicShapePlan(handle);
            plan.close();
        }
    }
}
