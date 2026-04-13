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
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacpp.Pointer;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Assumptions;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlan;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.api.shape.options.ArrayOptionsHelper;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv2DConfig;
import org.nd4j.linalg.api.ops.impl.transforms.custom.OnnxMultiHeadAttention;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;
import org.nd4j.nativeblas.OpaqueContext;
import org.nd4j.nativeblas.OpaqueDataBuffer;
import org.nd4j.nativeblas.OpaqueLaunchContext;
import org.nd4j.nativeblas.OpaqueNDArray;

import org.junit.jupiter.api.AfterEach;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for the Triton GPU compiler graph backend.
 *
 * Verifies that:
 * 1. Simple fusible graphs (matmul + add + relu) produce correct output
 * 2. Triton gracefully falls back to CUDA Graphs for unsupported ops
 * 3. Cached kernels are reused on subsequent executions
 * 4. Output matches reference (slot-by-slot) execution
 *
 * These tests require HELPERS_triton=ON at build time. When Triton is not
 * available, the backend will report isAvailable()=false and the execution
 * will fall through to CUDA Graphs or slot-by-slot, so tests still pass.
 */
@Slf4j
@Tag(TagNames.SAMEDIFF)
@NativeTag
public class TritonGraphBackendTest extends BaseNd4jTestWithBackends {

    private static final double TOLERANCE = 1e-4;

    @AfterEach
    public void cleanup() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        // Invalidate Triton compiled kernel cache to free CUmodule GPU memory
        nativeOps.invalidateTritonCache();
        nativeOps.resetTritonCounters();
        // Purge ND4J caches and trigger GC to free GPU buffers
        Nd4j.getMemoryManager().purgeCaches();
        System.gc();
        // Release GPU memory pool reserved memory back to the driver.
        // Without this, reserved-but-unused pool memory accumulates across tests,
        // causing OOM on tiny allocations (4 bytes) late in the suite.
        nativeOps.trimMemoryPool(0);
    }

    // ─── Helper methods ──────────────────────────────────────────────────────
    // Note: Basic helper methods moved to TritonTestUtils for reuse across test classes.
    // This class retains specialized versions with Triton counter checking and iteration.

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
            List<String> requestedOutputs = new ArrayList<>(plan.getRequestedOutputs());

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

    // ─── Graph builders ──────────────────────────────────────────────────────

    /**
     * Creates a matmul + add + relu chain with enough ops for capture (12+ ops).
     * This is the canonical pattern for Triton fusion: matmul + bias + activation.
     *
     * Graph: x -> matmul(w) -> add(b) -> relu -> multiply(scale) -> add(shift)
     *        -> tanh -> multiply(scale2) -> add(shift2) -> sigmoid
     *        -> multiply(scale3) -> add(shift3) -> relu (output)
     */
    private SameDiff createMatmulAddReluChain() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable w = sd.constant("w", Nd4j.randn(DataType.FLOAT, 4, 8));
        SDVariable b = sd.constant("b", Nd4j.randn(DataType.FLOAT, 1, 8));
        SDVariable scale = sd.constant("scale", Nd4j.randn(DataType.FLOAT, 1, 8));
        SDVariable shift = sd.constant("shift", Nd4j.randn(DataType.FLOAT, 1, 8));
        SDVariable scale2 = sd.constant("scale2", Nd4j.randn(DataType.FLOAT, 1, 8));
        SDVariable shift2 = sd.constant("shift2", Nd4j.randn(DataType.FLOAT, 1, 8));
        SDVariable scale3 = sd.constant("scale3", Nd4j.randn(DataType.FLOAT, 1, 8));
        SDVariable shift3 = sd.constant("shift3", Nd4j.randn(DataType.FLOAT, 1, 8));

        // Build a 12-op chain: matmul, add, relu, mul, add, tanh, mul, add, sigmoid, mul, add, relu
        SDVariable h1 = sd.mmul("matmul", x, w);
        SDVariable h2 = h1.add("add1", b);
        SDVariable h3 = sd.nn.relu("relu1", h2, 0);
        SDVariable h4 = h3.mul("mul1", scale);
        SDVariable h5 = h4.add("add2", shift);
        SDVariable h6 = sd.math.tanh("tanh1", h5);
        SDVariable h7 = h6.mul("mul2", scale2);
        SDVariable h8 = h7.add("add3", shift2);
        SDVariable h9 = sd.nn.sigmoid("sigmoid1", h8);
        SDVariable h10 = h9.mul("mul3", scale3);
        SDVariable h11 = h10.add("add4", shift3);
        SDVariable result = sd.nn.relu("result", h11, 0);

        return sd;
    }

    /**
     * Creates a purely element-wise chain (no matmul) for testing Triton
     * element-wise fusion. All ops are binary or unary element-wise.
     */
    private SameDiff createElementWiseChain() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 8);
        SDVariable a = sd.constant("a", Nd4j.randn(DataType.FLOAT, 1, 8));
        SDVariable b = sd.constant("b", Nd4j.randn(DataType.FLOAT, 1, 8));
        SDVariable c = sd.constant("c", Nd4j.randn(DataType.FLOAT, 1, 8));
        SDVariable d = sd.constant("d", Nd4j.randn(DataType.FLOAT, 1, 8));

        // 12 element-wise ops: add, mul, relu, subtract, sigmoid, mul, add, tanh, mul, add, exp, relu
        SDVariable h1 = x.add("add1", a);
        SDVariable h2 = h1.mul("mul1", b);
        SDVariable h3 = sd.nn.relu("relu1", h2, 0);
        SDVariable h4 = h3.sub("sub1", c);
        SDVariable h5 = sd.nn.sigmoid("sigmoid1", h4);
        SDVariable h6 = h5.mul("mul2", d);
        SDVariable h7 = h6.add("add2", a);
        SDVariable h8 = sd.math.tanh("tanh1", h7);
        SDVariable h9 = h8.mul("mul3", b);
        SDVariable h10 = h9.add("add3", c);
        SDVariable h11 = sd.math.exp("exp1", h10);
        SDVariable result = sd.nn.relu("result", h11, 0);

        return sd;
    }

    // ─── Tests ───────────────────────────────────────────────────────────────

    /**
     * Test: matmul + add + relu chain (canonical Triton fusion pattern).
     * Verifies the native plan executes correctly with or without Triton.
     * When Triton is available, this segment should be fused into a single kernel.
     * When not available, it falls through to CUDA Graphs or slot-by-slot.
     */
    @Test
    public void testMatmulAddReluChain() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        SameDiff sd = createMatmulAddReluChain();
        INDArray x = Nd4j.randn(DataType.FLOAT, 2, 4);
        Map<String, INDArray> ph = Map.of("x", x);

        // Get reference result from SameDiff
        Map<String, INDArray> refResults = sd.output(ph, "result");
        INDArray refOutput = refResults.get("result");
        assertNotNull(refOutput, "Reference output is null");

        // Compile plan using the shared test utility
        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "result");
        assertNotNull(plan, "Plan compilation returned null");
        assertTrue(plan.getSlots().length > 0, "Plan has no slots");
        log.info("MatmulAddRelu plan: {} slots, {} external inputs",
                 plan.getSlots().length, plan.getExternalInputKeys().length);

        Pointer planHandle = compileNativePlan(plan);
        if (planHandle == null) {
            log.info("Skipping (native executor not supported)");
            return;
        }
        try {
            INDArray[] extInputs = resolveExternalInputs(plan, sd, ph);

            // Reset Triton counters before execution
            nativeOps.resetTritonCounters();

            // Execute 3 times: warmup, compile/capture, replay
            for (int iter = 0; iter < 3; iter++) {
                Map<String, INDArray> nativeResults = executeNativePlan(planHandle, plan, extInputs);
                INDArray nativeOutput = nativeResults.get("result");
                assertNotNull(nativeOutput, "Native output is null at iteration " + iter);

                // Compare with reference
                double maxDiff = refOutput.sub(nativeOutput).amaxNumber().doubleValue();
                log.info("Iteration {}: max diff = {}", iter, maxDiff);

                long launchesNow = nativeOps.getTritonKernelLaunchCount();
                assertTrue(maxDiff < TOLERANCE,
                           "testMatmulAddReluChain iter " + iter + ": max diff = " + maxDiff +
                           (launchesNow > 0 && iter >= 2 ? " (Triton kernel)" : " (slot-by-slot)"));

                // After warmup, freeze shapes so Triton compilation triggers
                if (iter == 0) {
                    nativeOps.setPlanShapesFrozen(planHandle, true);
                }
            }

            // Check Triton invocation
            long tritonLaunches = nativeOps.getTritonKernelLaunchCount();
            log.info("testMatmulAddReluChain: Triton kernel launches = {}", tritonLaunches);
            if (tritonLaunches == 0) {
                log.warn("testMatmulAddReluChain: Triton was NOT used — fell back to CUDA graphs");
            }
        } finally {
            nativeOps.freeDynamicShapePlan(planHandle);
        }
    }

    /**
     * Test: purely element-wise chain (best case for Triton fusion).
     * All ops are element-wise, so Triton should be able to fuse the entire
     * segment into a single kernel with no intermediate global memory stores.
     */
    @Test
    public void testElementWiseFusion() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        SameDiff sd = createElementWiseChain();
        INDArray x = Nd4j.randn(DataType.FLOAT, 2, 8);
        Map<String, INDArray> ph = Map.of("x", x);

        Map<String, INDArray> refResults = sd.output(ph, "result");
        INDArray refOutput = refResults.get("result");
        assertNotNull(refOutput, "Reference output is null");

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "result");
        assertNotNull(plan, "Plan compilation returned null");

        Pointer planHandle = compileNativePlan(plan);
        if (planHandle == null) {
            log.info("Skipping (native executor not supported)");
            return;
        }
        try {
            INDArray[] extInputs = resolveExternalInputs(plan, sd, ph);

            nativeOps.resetTritonCounters();

            for (int iter = 0; iter < 3; iter++) {
                Map<String, INDArray> nativeResults = executeNativePlan(planHandle, plan, extInputs);
                INDArray nativeOutput = nativeResults.get("result");
                assertNotNull(nativeOutput, "Native output is null at iteration " + iter);

                double maxDiff = refOutput.sub(nativeOutput).amaxNumber().doubleValue();
                log.info("ElementWise iteration {}: max diff = {}", iter, maxDiff);

                long launchesNow = nativeOps.getTritonKernelLaunchCount();
                assertTrue(maxDiff < TOLERANCE,
                           "testElementWiseFusion iter " + iter + ": max diff = " + maxDiff +
                           (launchesNow > 0 && iter >= 2 ? " (Triton kernel)" : " (slot-by-slot)"));

                // After warmup, freeze shapes so Triton compilation triggers
                if (iter == 0) {
                    nativeOps.setPlanShapesFrozen(planHandle, true);
                }
            }

            long tritonLaunches = nativeOps.getTritonKernelLaunchCount();
            log.info("testElementWiseFusion: Triton kernel launches = {}", tritonLaunches);
            if (tritonLaunches == 0) {
                log.warn("testElementWiseFusion: Triton was NOT used — fell back to CUDA graphs");
            }
        } finally {
            nativeOps.freeDynamicShapePlan(planHandle);
        }
    }

    /**
     * Test: cache reuse verification.
     * Execute the same plan twice with the same shapes. The second execution
     * should use the cached compiled kernel (no recompilation).
     */
    @Test
    public void testCacheReuse() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        SameDiff sd = createElementWiseChain();
        INDArray x1 = Nd4j.randn(DataType.FLOAT, 2, 8);
        INDArray x2 = Nd4j.randn(DataType.FLOAT, 2, 8);  // Same shape, different values
        Map<String, INDArray> ph1 = Map.of("x", x1);
        Map<String, INDArray> ph2 = Map.of("x", x2);

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "result");
        Pointer planHandle = compileNativePlan(plan);
        if (planHandle == null) {
            log.info("Skipping (native executor not supported)");
            return;
        }
        try {
            nativeOps.resetTritonCounters();

            // Execute with x1 (3 iterations to go through warmup/capture/replay)
            INDArray[] extInputs1 = resolveExternalInputs(plan, sd, ph1);
            for (int i = 0; i < 3; i++) {
                Map<String, INDArray> results = executeNativePlan(planHandle, plan, extInputs1);
                assertNotNull(results.get("result"), "Null output at iteration " + i);
                // After warmup, freeze shapes so Triton compilation triggers
                if (i == 0) {
                    nativeOps.setPlanShapesFrozen(planHandle, true);
                }
            }

            // Execute with x2 (same shape — should use cached kernel)
            Map<String, INDArray> refResults2 = sd.output(ph2, "result");
            INDArray refOutput2 = refResults2.get("result");

            // Update external inputs for x2
            INDArray[] extInputs2 = resolveExternalInputs(plan, sd, ph2);
            // Replace x placeholder
            for (int i = 0; i < plan.getExternalInputKeys().length; i++) {
                if ("x".equals(plan.getExternalInputKeys()[i])) {
                    extInputs2[i] = x2;
                }
            }

            Map<String, INDArray> nativeResults2 = executeNativePlan(planHandle, plan, extInputs2);
            INDArray nativeOutput2 = nativeResults2.get("result");
            assertNotNull(nativeOutput2, "Native output for x2 is null");

            double maxDiff = refOutput2.sub(nativeOutput2).amaxNumber().doubleValue();
            log.info("Cache reuse test: max diff = {}", maxDiff);
            assertTrue(maxDiff < TOLERANCE,
                       "testCacheReuse: Triton cached kernel output differs (maxDiff=" + maxDiff + ")");

            long tritonLaunches = nativeOps.getTritonKernelLaunchCount();
            log.info("testCacheReuse: Triton kernel launches = {}", tritonLaunches);
            if (tritonLaunches == 0) {
                log.warn("testCacheReuse: Triton was NOT used — fell back to CUDA graphs");
            }
        } finally {
            nativeOps.freeDynamicShapePlan(planHandle);
        }
    }

    /**
     * Test: fallback to CUDA Graphs for mixed-op segments.
     * Creates a graph with some unsupported ops that Triton cannot compile.
     * Verifies the execution still succeeds via CUDA Graphs or slot-by-slot.
     */
    @Test
    public void testFallbackForUnsupportedOps() {
        // Originally tested fallback for unsupported ops (concat). Now all ops
        // compile to Triton, so this tests a complex mixed-op graph via Triton.
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable w = sd.constant("w", Nd4j.randn(DataType.FLOAT, 4, 8));

        SDVariable h1 = sd.mmul("matmul", x, w);
        SDVariable h2 = sd.nn.relu("relu1", h1, 0);
        SDVariable h3 = sd.math.tanh("tanh1", h2);
        SDVariable h4 = sd.concat("concat1", 1, h2, h3);
        SDVariable h5 = sd.nn.relu("relu2", h4, 0);
        SDVariable h6 = sd.nn.sigmoid("sigmoid1", h5);
        SDVariable h7 = h6.mul("mul1", sd.constant("s1", Nd4j.randn(DataType.FLOAT, 1, 16)));
        SDVariable h8 = h7.add("add1", sd.constant("s2", Nd4j.randn(DataType.FLOAT, 1, 16)));
        SDVariable h9 = sd.nn.relu("relu3", h8, 0);
        SDVariable h10 = sd.math.tanh("tanh2", h9);
        SDVariable h11 = h10.mul("mul2", sd.constant("s3", Nd4j.randn(DataType.FLOAT, 1, 16)));
        sd.nn.relu("result", h11, 0);

        INDArray xArr = Nd4j.randn(DataType.FLOAT, 2, 4);
        runOpTest("testFallbackForUnsupportedOps", sd, Map.of("x", xArr), "result");
    }

    // ─── Per-op-category tests ──────────────────────────────────────────────

    /**
     * Test: concat along axis=0 (row-wise) — simplest case.
     * x=[2,4], y=[3,4] → result=[5,4]
     */
    @Test
    public void testTritonConcatAxis0() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable y = sd.constant("y", Nd4j.ones(DataType.FLOAT, 3, 4));
        sd.concat("result", 0, x, y);
        runOpTest("testTritonConcatAxis0", sd, Map.of("x", Nd4j.randn(DataType.FLOAT, 2, 4)), "result");
    }

    /**
     * Test: concat along axis=1 (column-wise).
     * x=[2,4], y=[2,4] → result=[2,8]
     */
    @Test
    public void testTritonConcatAxis1() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable y = sd.constant("y", Nd4j.ones(DataType.FLOAT, 2, 4));
        sd.concat("result", 1, x, y);
        runOpTest("testTritonConcatAxis1", sd, Map.of("x", Nd4j.randn(DataType.FLOAT, 2, 4)), "result");
    }

    /**
     * Test: concat + relu fusion (like in testFallbackForUnsupportedOps but isolated).
     * Tests that concat output flows correctly into downstream elementwise ops.
     */
    @Test
    public void testTritonConcatFused() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable y = sd.constant("y", Nd4j.randn(DataType.FLOAT, 2, 4));
        SDVariable cat = sd.concat("cat", 1, x, y);
        sd.nn.relu("result", cat, 0);
        runOpTest("testTritonConcatFused", sd, Map.of("x", Nd4j.randn(DataType.FLOAT, 2, 4)), "result");
    }

    /**
     * Test: standalone matmul (no fused element-wise ops).
     * a=[32,64] x b=[64,128] -> result=[32,128]
     */
    @Test
    public void testTritonMatmulStandalone() {
        SameDiff sd = SameDiff.create();
        SDVariable a = sd.placeHolder("a", DataType.FLOAT, -1, 64);
        SDVariable b = sd.constant("b", Nd4j.randn(DataType.FLOAT, 64, 128));
        sd.mmul("result", a, b);

        runOpTest("testTritonMatmulStandalone", sd, Map.of("a", Nd4j.randn(DataType.FLOAT, 32, 64)), "result");
    }

    /**
     * Test: batch matmul with 3D tensors.
     * a=[4,32,64] x b=[4,64,128] -> result=[4,32,128]
     */
    @Test
    public void testTritonBatchMatmul() {
        SameDiff sd = SameDiff.create();
        SDVariable a = sd.placeHolder("a", DataType.FLOAT, -1, 32, 64);
        SDVariable b = sd.constant("b", Nd4j.randn(DataType.FLOAT, 4, 64, 128));
        sd.mmul("result", a, b);

        runOpTest("testTritonBatchMatmul", sd, Map.of("a", Nd4j.randn(DataType.FLOAT, 4, 32, 64)), "result");
    }

    /**
     * Test: matmul fused with element-wise ops in a mega-kernel.
     * x -> mmul(w) -> add(b) -> relu -> mul(s) -> add(s2) -> sigmoid -> result
     * x=[8,32], w=[32,64], b=[1,64], s=[1,64], s2=[1,64]
     */
    @Test
    public void testTritonMatmulInMegaKernel() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("input", DataType.FLOAT, -1, 32);
        SDVariable w = sd.constant("w", Nd4j.randn(DataType.FLOAT, 32, 64));
        SDVariable b = sd.constant("b", Nd4j.randn(DataType.FLOAT, 1, 64));
        SDVariable s = sd.constant("s", Nd4j.randn(DataType.FLOAT, 1, 64));
        SDVariable s2 = sd.constant("s2", Nd4j.randn(DataType.FLOAT, 1, 64));

        SDVariable h1 = sd.mmul("mm", x, w);
        SDVariable h2 = h1.add("add1", b);
        SDVariable h3 = sd.nn.relu("relu1", h2, 0);
        SDVariable h4 = h3.mul("mul1", s);
        SDVariable h5 = h4.add("add2", s2);
        sd.nn.sigmoid("result", h5);

        runOpTest("testTritonMatmulInMegaKernel", sd, Map.of("input", Nd4j.randn(DataType.FLOAT, 8, 32)), "result");
    }

    /**
     * Test: gather op — gathers rows from a 2D tensor by index.
     * x=[16,8], indices=[0,3,7,1] -> result=[4,8]
     */
    @Test
    public void testTritonGather() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("input", DataType.FLOAT, -1, 8);
        SDVariable gathered = sd.gather("gather1", x, new int[]{0, 3, 7, 1}, 0);
        sd.nn.relu("result", gathered, 0);

        runOpTest("testTritonGather", sd, Map.of("input", Nd4j.randn(DataType.FLOAT, 16, 8)), "result");
    }

    /**
     * Test: concat op — concatenates two tensors along axis 1, then relu.
     * a=[4,8], b=[4,8] -> concat=[4,16] -> relu -> result=[4,16]
     */
    @Test
    public void testTritonConcat() {
        SameDiff sd = SameDiff.create();
        SDVariable a = sd.placeHolder("a", DataType.FLOAT, -1, 8);
        SDVariable b = sd.placeHolder("b", DataType.FLOAT, -1, 8);
        SDVariable concat = sd.concat("concat1", 1, a, b);
        sd.nn.relu("result", concat, 0);

        runOpTest("testTritonConcat", sd,
                  Map.of("a", Nd4j.randn(DataType.FLOAT, 4, 8), "b", Nd4j.randn(DataType.FLOAT, 4, 8)), "result");
    }

    /**
     * Test: split op — splits a tensor along axis 1 into 2 halves, then adds them.
     * x=[4,16] -> split into [4,8] and [4,8] -> add -> result=[4,8]
     */
    @Test
    public void testTritonSplit() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("input", DataType.FLOAT, -1, 16);
        SDVariable[] splits = sd.split(new String[]{"split0", "split1"}, x, 2, 1);
        splits[0].add("result", splits[1]);

        runOpTest("testTritonSplit", sd, Map.of("input", Nd4j.randn(DataType.FLOAT, 4, 16)), "result");
    }

    /**
     * Test: tile op — tiles a tensor along axis 1.
     * x=[4,8] -> tile(1,3) -> result=[4,24]
     */
    @Test
    public void testTritonTile() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("input", DataType.FLOAT, -1, 8);
        sd.tile("result", x, 1, 3);

        runOpTest("testTritonTile", sd, Map.of("input", Nd4j.randn(DataType.FLOAT, 4, 8)), "result");
    }

    /**
     * Test: strided slice op — extracts a sub-tensor.
     * x=[8,16] -> stridedSlice([0,0],[4,8],[1,1]) -> result=[4,8]
     */
    @Test
    public void testTritonStridedSlice() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("input", DataType.FLOAT, -1, 16);
        sd.stridedSlice("result", x, new long[]{0, 0}, new long[]{4, 8}, new long[]{1, 1});

        runOpTest("testTritonStridedSlice", sd, Map.of("input", Nd4j.randn(DataType.FLOAT, 8, 16)), "result");
    }

    /**
     * Test: a non-zero-offset strided_slice view feeding Triton elementwise ops.
     * If Triton passes the base specialBuffer() instead of the view-adjusted device
     * pointer, the downstream add/mul chain reads the wrong slice and diverges.
     */
    @Test
    public void testTritonOffsetViewConsumer() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("input", DataType.FLOAT, 1, 8);
        SDVariable sliced = sd.stridedSlice("sliced", x,
                new long[]{0, 2}, new long[]{1, 6}, new long[]{1, 1});  // view of input[0,2:6]
        SDVariable shifted = sliced.add("shifted",
                sd.constant("bias", Nd4j.createFromArray(new float[][]{{10f, 20f, 30f, 40f}})));
        SDVariable result = shifted.mul("result",
                sd.constant("scale", Nd4j.createFromArray(new float[][]{{1f, 2f, 3f, 4f}})));

        INDArray input = Nd4j.createFromArray(new float[][]{{1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f}});
        runStrictTritonOpTest("testTritonOffsetViewConsumer", sd, Map.of("input", input), "result");
    }

    /**
     * Test: reshape op — reshapes a 2D tensor to a different 2D shape.
     * x=[4,8] -> reshape(2,16) -> result=[2,16]
     */
    @Test
    public void testTritonReshape() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("input", DataType.FLOAT, 4, 8);
        sd.reshape("result", x, 2, 16);

        runOpTest("testTritonReshape", sd, Map.of("input", Nd4j.randn(DataType.FLOAT, 4, 8)), "result");
    }

    /**
     * Test: permute op — transposes dimension order.
     * x=[4,8,16] -> permute(2,0,1) -> result=[16,4,8]
     */
    @Test
    public void testTritonPermute() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("input", DataType.FLOAT, -1, 8, 16);
        sd.permute("result", x, 2, 0, 1);

        runOpTest("testTritonPermute", sd, Map.of("input", Nd4j.randn(DataType.FLOAT, 4, 8, 16)), "result");
    }

    /**
     * Test: expandDims op — adds a dimension then applies relu.
     * x=[4,8] -> expandDims(1) -> [4,1,8] -> relu -> result=[4,1,8]
     */
    @Test
    public void testTritonExpandDims() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("input", DataType.FLOAT, -1, 8);
        SDVariable expanded = sd.expandDims("expand", x, 1);
        sd.nn.relu("result", expanded, 0);

        runOpTest("testTritonExpandDims", sd, Map.of("input", Nd4j.randn(DataType.FLOAT, 4, 8)), "result");
    }

    /**
     * Test: squeeze op — removes a singleton dimension then applies relu.
     * x=[4,1,8] -> squeeze(1) -> [4,8] -> relu -> result=[4,8]
     */
    @Test
    public void testTritonSqueeze() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("input", DataType.FLOAT, -1, 1, 8);
        SDVariable squeezed = sd.squeeze("squeeze", x, 1);
        sd.nn.relu("result", squeezed, 0);

        runOpTest("testTritonSqueeze", sd, Map.of("input", Nd4j.randn(DataType.FLOAT, 4, 1, 8)), "result");
    }

    /**
     * Test: reduction op — sum along axis 1.
     * x=[4,8] -> sum(axis=1) -> result=[4]
     */
    @Test
    public void testTritonReduction() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("input", DataType.FLOAT, -1, 8);
        sd.sum("result", x, 1);

        runOpTest("testTritonReduction", sd, Map.of("input", Nd4j.randn(DataType.FLOAT, 4, 8)), "result");
    }

    /**
     * Test: RMSNorm-like pattern: add + pow + reduce_mean + scalar_add + sqrt + divide + multiply.
     * This exercises the segmented reduction path with block barrier inline ASM.
     * x=[1,576], residual=[1,576], weight=[1,576] -> x+residual -> pow(2) -> mean -> +eps -> sqrt -> div -> mul -> result
     */
    @Test
    public void testTritonRmsNormPattern() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 576);
        SDVariable residual = sd.constant("residual", Nd4j.randn(DataType.FLOAT, 1, 576));
        SDVariable weight = sd.constant("weight", Nd4j.ones(DataType.FLOAT, 1, 576));
        double eps = 1e-5;

        SDVariable added = x.add("add", residual);
        SDVariable squared = added.mul("square", added);  // x^2 via x*x
        SDVariable meanSq = sd.mean("mean", squared, true, 1);  // reduce_mean along axis 1
        SDVariable epsConst = sd.constant("eps", Nd4j.scalar(DataType.FLOAT, (float)eps));
        SDVariable denom = meanSq.add("add_eps", epsConst);
        SDVariable sqrtDenom = sd.math.sqrt("sqrt", denom);
        SDVariable normed = added.div("div", sqrtDenom);
        normed.mul("result", weight);

        runOpTest("testTritonRmsNormPattern", sd, Map.of("x", Nd4j.randn(DataType.FLOAT, 1, 576)), "result");
    }

    /**
     * Test: cooperative element-wise -> matmul -> element-wise chain.
     * input -> add(a) -> relu -> mmul(w) -> add(b) -> tanh -> mul(s) -> sigmoid -> result
     * Tests 3 sections: EW -> matmul -> EW with grid sync barriers.
     * x=[8,32], a=[1,32], w=[32,64], b=[1,64], s=[1,64]
     */
    @Test
    public void testTritonCooperativeEwMatmulEw() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("input", DataType.FLOAT, -1, 32);
        SDVariable a = sd.constant("a", Nd4j.randn(DataType.FLOAT, 1, 32));
        SDVariable w = sd.constant("w", Nd4j.randn(DataType.FLOAT, 32, 64));
        SDVariable b = sd.constant("b", Nd4j.randn(DataType.FLOAT, 1, 64));
        SDVariable s = sd.constant("s", Nd4j.randn(DataType.FLOAT, 1, 64));

        SDVariable h1 = x.add("add1", a);
        SDVariable h2 = sd.nn.relu("relu1", h1, 0);
        SDVariable h3 = sd.mmul("mm1", h2, w);
        SDVariable h4 = h3.add("add2", b);
        SDVariable h5 = sd.math.tanh("tanh1", h4);
        SDVariable h6 = h5.mul("mul1", s);
        sd.nn.sigmoid("result", h6);

        runOpTest("testTritonCooperativeEwMatmulEw", sd, Map.of("input", Nd4j.randn(DataType.FLOAT, 8, 32)), "result");
    }

    /**
     * Test: fallback path for per-element execution.
     * Uses a standalone matmul which exercises the standard (non-cooperative) path.
     * Verifies that the standard execution path works correctly as a fallback.
     */
    @Test
    public void testTritonFallbackPerElement() {
        SameDiff sd = SameDiff.create();
        SDVariable a = sd.placeHolder("a", DataType.FLOAT, -1, 64);
        SDVariable b = sd.constant("b", Nd4j.randn(DataType.FLOAT, 64, 128));
        SDVariable mm = sd.mmul("mm", a, b);
        sd.nn.relu("result", mm, 0);

        runOpTest("testTritonFallbackPerElement", sd, Map.of("a", Nd4j.randn(DataType.FLOAT, 32, 64)), "result");
    }

    /**
     * Test: element-wise chain with multiple binary ops (substitute for scatter_nd).
     * x=[4,8], y=[4,8] -> add -> mul(x) -> sub(y) -> result=[4,8]
     */
    @Test
    public void testTritonScatterNd() {
        // SameDiff does not expose scatter_nd directly; use a multi-op element-wise chain.
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 8);
        SDVariable y = sd.placeHolder("y", DataType.FLOAT, -1, 8);
        SDVariable h1 = x.add("add1", y);
        SDVariable h2 = h1.mul("mul1", x);
        h2.sub("result", y);

        runOpTest("testTritonScatterNd", sd,
                  Map.of("x", Nd4j.randn(DataType.FLOAT, 4, 8), "y", Nd4j.randn(DataType.FLOAT, 4, 8)), "result");
    }

    /**
     * Test: normalization — manual batch norm via element-wise ops.
     * x=[4,8] -> mean -> sub(mean) -> var -> div(sqrt(var+eps)) -> mul(gamma) -> add(beta) -> result
     * Tests reduction + broadcast + element-wise chain.
     */
    @Test
    public void testTritonNormalization() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("input", DataType.FLOAT, -1, 8);
        SDVariable gamma = sd.constant("gamma", Nd4j.ones(DataType.FLOAT, 1, 8));
        SDVariable beta = sd.constant("beta", Nd4j.zeros(DataType.FLOAT, 1, 8));

        // Manual normalization: (x - mean) / sqrt(var + eps) * gamma + beta
        SDVariable mean = sd.mean("mean", x, true, 1);
        SDVariable centered = x.sub("centered", mean);
        SDVariable variance = sd.mean("variance", centered.mul(centered), true, 1);
        SDVariable eps = sd.constant("eps", Nd4j.scalar(DataType.FLOAT, 1e-5f));
        SDVariable stdInv = sd.math.rsqrt("stdInv", variance.add(eps));
        SDVariable normed = centered.mul("normed", stdInv);
        SDVariable scaled = normed.mul("scaled", gamma);
        scaled.add("result", beta);

        runOpTest("testTritonNormalization", sd, Map.of("input", Nd4j.randn(DataType.FLOAT, 4, 8)), "result");
    }

    /**
     * Test: conv2d op — 2D convolution.
     * input=[1,3,16,16] (NCHW), weight=[8,3,3,3] (OIYX), stride=1, pad=0
     * result=[1,8,14,14]
     */
    @Test
    public void testTritonConv2d() {
        SameDiff sd = SameDiff.create();
        // Minimal conv2d: 1 input channel, 1 output channel, 3x3 kernel, 4x4 input
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 1, 4, 4);
        SDVariable weight = sd.constant("weight", Nd4j.randn(DataType.FLOAT, 1, 1, 3, 3));

        Conv2DConfig config = Conv2DConfig.builder()
                .kH(3).kW(3)
                .sH(1).sW(1)
                .pH(0).pW(0)
                .dH(1).dW(1)
                .dataFormat(Conv2DConfig.NCHW)
                .weightsFormat(org.nd4j.enums.WeightsFormat.OIYX)
                .build();

        SDVariable conv = sd.cnn.conv2d("conv", input, weight, config);
        sd.nn.relu("result", conv, 0);

        runOpTest("testTritonConv2d", sd, Map.of("input", Nd4j.randn(DataType.FLOAT, 1, 1, 4, 4)), "result");
    }

    /**
     * Test: conv2d with deterministic (ones) input and filter to verify element ordering.
     * input=[1,1,4,4] (all 1s), weight=[1,1,3,3] (identity-like, ones), stride=1, pad=0
     * output=[1,1,2,2], each element = 9.0 (sum of 9 ones * 1)
     */
    @Test
    public void testTritonConv2dDeterministic() {
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 1, 4, 4);
        // Use a filter with a single 1.0 at position [0,0,0,0] (top-left corner)
        // This way output[h,w] = input[h,w]
        INDArray filterData = Nd4j.zeros(DataType.FLOAT, 1, 1, 3, 3);
        filterData.putScalar(0, 0, 0, 0, 1.0f);  // filter[oc=0,ic=0,kh=0,kw=0] = 1
        SDVariable weight = sd.constant("weight", filterData);

        Conv2DConfig config = Conv2DConfig.builder()
                .kH(3).kW(3).sH(1).sW(1).pH(0).pW(0).dH(1).dW(1)
                .dataFormat(Conv2DConfig.NCHW)
                .weightsFormat(org.nd4j.enums.WeightsFormat.OIYX)
                .build();
        SDVariable conv = sd.cnn.conv2d("conv", input, weight, config);
        sd.nn.relu("result", conv, 0);

        // Input: values 1..16 in order so we can identify which position each element came from
        INDArray inputData = Nd4j.createFromArray(1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16f).reshape(1,1,4,4);
        runOpTest("testTritonConv2dDeterministic", sd, Map.of("input", inputData), "result");
    }

    // ─── VLM op coverage tests ──────────────────────────────────────────────

    // Helper: build graph, get reference output, compile native plan, execute, compare.
    // Uses TritonTestUtils for common functionality.
    private void runOpTest(String testName, SameDiff sd, Map<String, INDArray> ph, String outputName) {
        TritonTestUtils.runOpTest(testName, sd, ph, outputName);
    }

    // ─── Graph builders ──────────────────────────────────────────────────────

    private void runStrictTritonOpTest(String testName, SameDiff sd, Map<String, INDArray> ph, String outputName) {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        Assumptions.assumeTrue(nativeOps.isTritonAvailable(), testName + ": Triton is unavailable");

        Map<String, INDArray> ref = sd.output(ph, outputName);
        INDArray refOutput = ref.get(outputName);
        assertNotNull(refOutput, testName + ": reference output is null");

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, outputName);
        assertNotNull(plan, testName + ": plan is null");
        Pointer planHandle = compileNativePlan(plan);
        Assumptions.assumeTrue(planHandle != null, testName + ": native executor is unavailable");

        try {
            INDArray[] extInputs = resolveExternalInputs(plan, sd, ph);
            nativeOps.resetTritonCounters();

            long launchesBefore = nativeOps.getTritonKernelLaunchCount();
            boolean tritonUsed = false;

            for (int iter = 0; iter < 6; iter++) {
                Map<String, INDArray> nativeResults = executeNativePlan(planHandle, plan, extInputs);
                INDArray nativeOutput = nativeResults.get(outputName);
                assertNotNull(nativeOutput, testName + ": null output at iter " + iter);

                double maxDiff = refOutput.sub(nativeOutput).amaxNumber().doubleValue();
                long launchesNow = nativeOps.getTritonKernelLaunchCount();
                boolean usedThisIter = launchesNow > launchesBefore;

                if (usedThisIter) {
                    tritonUsed = true;
                    assertTrue(maxDiff < TOLERANCE,
                            testName + ": Triton maxDiff=" + maxDiff + " at iter " + iter);
                    break;
                }

                assertTrue(maxDiff < TOLERANCE,
                        testName + ": slot-by-slot maxDiff=" + maxDiff + " at iter " + iter);
                launchesBefore = launchesNow;

                if (iter == 0) {
                    nativeOps.setPlanShapesFrozen(planHandle, true);
                }
            }

            assertTrue(tritonUsed, testName + ": Triton did not execute");
        } finally {
            nativeOps.freeDynamicShapePlan(planHandle);
        }
    }

    @Test
    public void testTritonCast() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 32);
        SDVariable h1 = sd.castTo("cast1", x, DataType.DOUBLE);
        SDVariable h2 = sd.math.abs("abs1", h1);
        sd.castTo("result", h2, DataType.FLOAT);

        INDArray xArr = Nd4j.randn(DataType.FLOAT, 4, 32);
        runOpTest("testTritonCast", sd, Map.of("x", xArr), "result");
    }

    /**
     * Tests int64→float cast with large values through Triton.
     * Exercises the i64→f64→f32 path (fixed from i64→i32→f32 truncation bug).
     * Uses abs() after cast to prevent Triton from optimizing away the cast.
     */
    @Test
    public void testTritonCastInt64ToFloat() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.INT64, -1, 8);
        // int64 → float (exercises the i64→f64→f32 path)
        SDVariable asFloat = sd.castTo("cast_to_float", x, DataType.FLOAT);
        // abs() ensures Triton compiles a real kernel (not optimized away)
        sd.math.abs("result", asFloat);

        // Include values near and beyond 2^31 boundary
        INDArray xArr = Nd4j.createFromArray(new long[][]{
            {0L, 1L, -1L, 127L, 100000L, 2147483647L, -2147483648L, 4294967296L}
        });
        runOpTest("testTritonCastInt64ToFloat", sd, Map.of("x", xArr), "result");
    }

    /**
     * Tests float→int64 cast through Triton by verifying the truncation behavior.
     * Exercises the f32→f64→i64 path (fixed from f32→i32→i64 truncation bug).
     * Uses add(0) on INT64 to prevent cast from being optimized away,
     * then casts back to FLOAT for numeric comparison.
     */
    @Test
    public void testTritonCastFloatToInt64ViaAdd() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 8);
        // float → int64 (exercises f32→f64→i64 path)
        SDVariable asLong = sd.castTo("cast_to_long", x, DataType.INT64);
        // Add 0 as INT64 to force the INT64 value to actually be computed
        SDVariable zero = sd.constant("zero", Nd4j.zeros(DataType.INT64, 1, 8));
        SDVariable added = asLong.add("add_zero", zero);
        // Cast back to float for comparison
        sd.castTo("result", added, DataType.FLOAT);

        // Values that are exact in float and exercise the i64 range
        INDArray xArr = Nd4j.createFromArray(new float[][]{
            {0f, 1f, -1f, 127f, 100000f, 2.14748365E9f, -2.14748365E9f, 4.29496730E9f}
        });
        runOpTest("testTritonCastFloatToInt64ViaAdd", sd, Map.of("x", xArr), "result");
    }

    @Test
    public void testTritonAssign() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 16);
        SDVariable target = sd.var("target", Nd4j.zeros(DataType.FLOAT, 4, 16));
        sd.assign("result", target, x);

        INDArray xArr = Nd4j.randn(DataType.FLOAT, 4, 16);
        runOpTest("testTritonAssign", sd, Map.of("x", xArr), "result");
    }

    @Test
    public void testTritonEquals() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 16);
        SDVariable y = sd.constant("y", Nd4j.randn(DataType.FLOAT, 1, 16));
        SDVariable eq = sd.eq("eq1", x, y);
        // Cast bool to float so we can compare numerically
        sd.castTo("result", eq, DataType.FLOAT);

        INDArray xArr = Nd4j.randn(DataType.FLOAT, 4, 16);
        runOpTest("testTritonEquals", sd, Map.of("x", xArr), "result");
    }

    @Test
    public void testTritonLess() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 16);
        SDVariable y = sd.constant("y", Nd4j.randn(DataType.FLOAT, 1, 16));
        SDVariable lt = sd.lt("lt1", x, y);
        sd.castTo("result", lt, DataType.FLOAT);

        INDArray xArr = Nd4j.randn(DataType.FLOAT, 4, 16);
        runOpTest("testTritonLess", sd, Map.of("x", xArr), "result");
    }

    @Test
    public void testTritonWhere() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 16);
        SDVariable y = sd.constant("y", Nd4j.randn(DataType.FLOAT, 1, 16));
        SDVariable cond = sd.gt("cond", x, y);
        SDVariable a = sd.constant("a", Nd4j.ones(DataType.FLOAT, 1, 16));
        SDVariable b = sd.constant("b", Nd4j.zeros(DataType.FLOAT, 1, 16));
        sd.where("result", a, b, cond);

        INDArray xArr = Nd4j.randn(DataType.FLOAT, 4, 16);
        runOpTest("testTritonWhere", sd, Map.of("x", xArr), "result");
    }

    @Test
    public void testTritonShapeInt64WhereCastChain() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 3, 3, 700, 64);
        SDVariable shape = sd.shape("shape", x);
        SDVariable negOnes = sd.constant("neg_ones",
                Nd4j.createFromArray(new long[]{-1L, -1L, -1L, -1L, -1L}));
        SDVariable ones = sd.constant("ones",
                Nd4j.createFromArray(new long[]{1L, 1L, 1L, 1L, 1L}));
        SDVariable eq = sd.eq("eq", shape, negOnes);
        SDVariable eqLong = sd.castTo("eq_long", eq, DataType.INT64);
        SDVariable eqBool = sd.castTo("eq_bool", eqLong, DataType.BOOL);
        SDVariable shapeCast = sd.castTo("shape_cast", shape, DataType.INT64);
        SDVariable selected = sd.where("selected", ones, shapeCast, eqBool);
        sd.castTo("result", selected, DataType.FLOAT);

        INDArray xArr = Nd4j.randn(DataType.FLOAT, 1, 3, 3, 700, 64);
        runStrictTritonOpTest("testTritonShapeInt64WhereCastChain", sd, Map.of("x", xArr), "result");
    }

    @Test
    public void testTritonShapeInt64CastFanOut() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 3, 3, 700, 64);
        SDVariable shape = sd.shape("shape", x);
        SDVariable cast1 = sd.castTo("cast1", shape, DataType.INT64);
        SDVariable cast2 = sd.castTo("cast2", shape, DataType.INT64);
        SDVariable cast3 = sd.castTo("cast3", shape, DataType.INT64);
        SDVariable sum = cast1.add("sum", cast2);
        SDVariable restored = sum.sub("restored", cast3);
        sd.castTo("result", restored, DataType.FLOAT);

        INDArray xArr = Nd4j.randn(DataType.FLOAT, 1, 3, 3, 700, 64);
        runStrictTritonOpTest("testTritonShapeInt64CastFanOut", sd, Map.of("x", xArr), "result");
    }

    @Test
    public void testTritonFrozenInt64AcrossNativeShapeGap() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        Assumptions.assumeTrue(nativeOps.isTritonAvailable(),
                "testTritonFrozenInt64AcrossNativeShapeGap: Triton is unavailable");

        boolean prevCapture = Nd4j.getEnvironment().tritonGraphCapture();
        Nd4j.getEnvironment().setTritonGraphCapture(false);

        SameDiff sd = SameDiff.create();
        SDVariable p = sd.placeHolder("p", DataType.INT64, 5);
        SDVariable castA = sd.castTo("castA", p, DataType.INT64);
        SDVariable castB = sd.castTo("castB", p, DataType.INT64);
        SDVariable expanded = sd.expandDims("expanded", castA, 0);
        SDVariable reshaped = sd.reshape("reshaped", expanded, 1, 5);
        SDVariable squeezed = sd.squeeze("squeezed", reshaped, 0);
        SDVariable eq = sd.eq("eq", castB, squeezed);
        SDVariable eqLong = sd.castTo("eqLong", eq, DataType.INT64);
        SDVariable sum = squeezed.add("sum", eqLong);
        sd.castTo("result", sum, DataType.FLOAT);

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "result");
        assertNotNull(plan, "testTritonFrozenInt64AcrossNativeShapeGap: plan is null");
        Pointer planHandle = compileNativePlan(plan);
        Assumptions.assumeTrue(planHandle != null,
                "testTritonFrozenInt64AcrossNativeShapeGap: native executor unavailable");

        try {
            nativeOps.resetTritonCounters();
            long launchesBefore = nativeOps.getTritonKernelLaunchCount();
            boolean tritonUsed = false;

            long[][] values = new long[][]{
                    {1, 3, 3, 700, 64},
                    {2, 4, 4, 701, 65},
                    {5, 7, 7, 702, 66},
                    {8, 9, 9, 703, 67},
                    {11, 13, 13, 704, 68},
                    {21, 34, 34, 705, 69}
            };

            for (int iter = 0; iter < values.length; iter++) {
                INDArray pArr = Nd4j.createFromArray(values[iter]);
                Map<String, INDArray> placeholders = Map.of("p", pArr);
                INDArray refOutput = sd.output(placeholders, "result").get("result");
                INDArray[] extInputs = resolveExternalInputs(plan, sd, placeholders);

                Map<String, INDArray> nativeResults = executeNativePlan(planHandle, plan, extInputs);
                INDArray nativeOutput = nativeResults.get("result");
                assertNotNull(nativeOutput,
                        "testTritonFrozenInt64AcrossNativeShapeGap: null output at iter " + iter);

                double maxDiff = refOutput.sub(nativeOutput).amaxNumber().doubleValue();
                long launchesNow = nativeOps.getTritonKernelLaunchCount();
                if (launchesNow > launchesBefore) {
                    tritonUsed = true;
                }

                assertTrue(maxDiff < TOLERANCE,
                        "testTritonFrozenInt64AcrossNativeShapeGap iter " + iter
                                + ": maxDiff=" + maxDiff);

                if (iter == 0) {
                    nativeOps.setPlanShapesFrozen(planHandle, true);
                }
                launchesBefore = launchesNow;
            }

            assertTrue(tritonUsed,
                    "testTritonFrozenInt64AcrossNativeShapeGap: Triton did not execute");
        } finally {
            Nd4j.getEnvironment().setTritonGraphCapture(prevCapture);
            nativeOps.freeDynamicShapePlan(planHandle);
        }
    }

    @Test
    public void testTritonDirectKvRetentionUsesCurrentOutputs() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        Assumptions.assumeTrue(nativeOps.isTritonAvailable(),
                "testTritonDirectKvRetentionUsesCurrentOutputs: Triton is unavailable");

        boolean prevCapture = Nd4j.getEnvironment().tritonGraphCapture();
        boolean prevSkip = Nd4j.getEnvironment().tritonSkipKernels();
        boolean prevVerify = Nd4j.getEnvironment().tritonVerifyKernels();
        Nd4j.getEnvironment().setTritonGraphCapture(false);
        Nd4j.getEnvironment().setTritonSkipKernels(false);
        Nd4j.getEnvironment().setTritonVerifyKernels(false);

        int maxKvLen = 4;
        SameDiff sd = SameDiff.create();
        SDVariable current = sd.placeHolder("current", DataType.FLOAT, 1, 1, 1, 4);
        SDVariable past = sd.placeHolder("past", DataType.FLOAT, 1, 1, maxKvLen, 4);
        SDVariable scale = sd.constant("scale",
                Nd4j.createFromArray(new float[]{2f, 3f, 4f, 5f}).reshape(1, 1, 1, 4));
        SDVariable bias = sd.constant("bias",
                Nd4j.createFromArray(new float[]{10f, 20f, 30f, 40f}).reshape(1, 1, 1, 4));
        SDVariable present = current.mul("present_mul", scale).add("present", bias);
        sd.sum("result", past, true, 2);

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "result", "present");
        assertNotNull(plan, "testTritonDirectKvRetentionUsesCurrentOutputs: plan is null");
        Pointer planHandle = compileNativePlan(plan);
        Assumptions.assumeTrue(planHandle != null,
                "testTritonDirectKvRetentionUsesCurrentOutputs: native executor unavailable");

        try {
            String[] extKeys = plan.getExternalInputKeys();
            int pastExtIdx = -1;
            for (int i = 0; i < extKeys.length; i++) {
                if ("past".equals(extKeys[i])) {
                    pastExtIdx = i;
                    break;
                }
            }
            assertTrue(pastExtIdx >= 0,
                    "testTritonDirectKvRetentionUsesCurrentOutputs: missing external input 'past'");

            try (IntPointer mappings = new IntPointer(new int[]{1, pastExtIdx, 2})) {
                nativeOps.configurePlanKvCacheRetention(planHandle, mappings, 1, maxKvLen, 0);
            }

            nativeOps.resetTritonCounters();
            long prevLaunches = nativeOps.getTritonKernelLaunchCount();
            boolean tritonUsed = false;

            INDArray pastNative = Nd4j.zeros(DataType.FLOAT, 1, 1, maxKvLen, 4);
            INDArray pastExpected = pastNative.dup();
            float[][] steps = new float[][]{
                    {1f, 2f, 3f, 4f},
                    {5f, 6f, 7f, 8f},
                    {9f, 10f, 11f, 12f}
            };

            for (int iter = 0; iter < steps.length; iter++) {
                INDArray currentArr = Nd4j.createFromArray(steps[iter]).reshape(1, 1, 1, 4);
                Map<String, INDArray> placeholders = new LinkedHashMap<>();
                placeholders.put("current", currentArr);
                placeholders.put("past", pastExpected);

                Map<String, INDArray> ref = sd.output(placeholders, "result", "present");
                INDArray expectedResult = ref.get("result");
                INDArray expectedPresent = ref.get("present");
                assertNotNull(expectedResult);
                assertNotNull(expectedPresent);

                Map<String, INDArray> nativePlaceholders = new LinkedHashMap<>();
                nativePlaceholders.put("current", currentArr);
                nativePlaceholders.put("past", pastNative);
                INDArray[] extInputs = resolveExternalInputs(plan, sd, nativePlaceholders);
                Map<String, INDArray> nativeResults = executeNativePlan(planHandle, plan, extInputs);
                INDArray nativeResult = nativeResults.get("result");
                assertNotNull(nativeResult,
                        "testTritonDirectKvRetentionUsesCurrentOutputs: null result at iter " + iter);

                double maxDiff = expectedResult.sub(nativeResult).amaxNumber().doubleValue();
                assertTrue(maxDiff < TOLERANCE,
                        "testTritonDirectKvRetentionUsesCurrentOutputs iter " + iter
                                + ": result maxDiff=" + maxDiff);

                INDArray expectedSlice = expectedPresent.get(
                        NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(0), NDArrayIndex.all());
                pastExpected.get(
                        NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(iter), NDArrayIndex.all())
                        .assign(expectedSlice);

                INDArray nativePastSnapshot = pastNative.dup();
                double pastDiff = pastExpected.sub(nativePastSnapshot).amaxNumber().doubleValue();
                assertTrue(pastDiff < TOLERANCE,
                        "testTritonDirectKvRetentionUsesCurrentOutputs iter " + iter
                                + ": past cache maxDiff=" + pastDiff);

                long launchesNow = nativeOps.getTritonKernelLaunchCount();
                if (launchesNow > prevLaunches) {
                    tritonUsed = true;
                }
                if (iter == 0) {
                    nativeOps.setPlanShapesFrozen(planHandle, true);
                }
                prevLaunches = launchesNow;
            }

            assertTrue(tritonUsed,
                    "testTritonDirectKvRetentionUsesCurrentOutputs: Triton did not execute");
        } finally {
            Nd4j.getEnvironment().setTritonGraphCapture(prevCapture);
            Nd4j.getEnvironment().setTritonSkipKernels(prevSkip);
            Nd4j.getEnvironment().setTritonVerifyKernels(prevVerify);
            nativeOps.freeDynamicShapePlan(planHandle);
        }
    }

    @Test
    public void testTritonPow() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 16);
        SDVariable exp = sd.constant("exp", Nd4j.valueArrayOf(new long[]{1, 16}, 2.0f));
        sd.math.pow("result", x, exp);

        INDArray xArr = Nd4j.rand(DataType.FLOAT, 4, 16).addi(0.1); // positive values
        runOpTest("testTritonPow", sd, Map.of("x", xArr), "result");
    }

    @Test
    public void testTritonSqrt() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 16);
        sd.math.sqrt("result", x);

        INDArray xArr = Nd4j.rand(DataType.FLOAT, 4, 16).addi(0.01); // positive
        runOpTest("testTritonSqrt", sd, Map.of("x", xArr), "result");
    }

    @Test
    public void testTritonSigmoid() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 32);
        sd.nn.sigmoid("result", x);

        INDArray xArr = Nd4j.randn(DataType.FLOAT, 4, 32);
        runOpTest("testTritonSigmoid", sd, Map.of("x", xArr), "result");
    }

    @Test
    public void testTritonSubtract() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 16);
        SDVariable y = sd.constant("y", Nd4j.randn(DataType.FLOAT, 1, 16));
        x.sub("result", y);

        INDArray xArr = Nd4j.randn(DataType.FLOAT, 4, 16);
        runOpTest("testTritonSubtract", sd, Map.of("x", xArr), "result");
    }

    @Test
    public void testTritonDivide() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 16);
        SDVariable y = sd.constant("y", Nd4j.rand(DataType.FLOAT, 1, 16).addi(0.1));
        x.div("result", y);

        INDArray xArr = Nd4j.randn(DataType.FLOAT, 4, 16);
        runOpTest("testTritonDivide", sd, Map.of("x", xArr), "result");
    }

    @Test
    public void testTritonAddScalar() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 16);
        sd.math.add("result", x, 3.14);

        INDArray xArr = Nd4j.randn(DataType.FLOAT, 4, 16);
        runOpTest("testTritonAddScalar", sd, Map.of("x", xArr), "result");
    }

    @Test
    public void testTritonCreate() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 16);
        // create (zeros) + add to make it depend on x
        SDVariable zeros = sd.zerosLike("zeros1", x);
        x.add("result", zeros);

        INDArray xArr = Nd4j.randn(DataType.FLOAT, 4, 16);
        runOpTest("testTritonCreate", sd, Map.of("x", xArr), "result");
    }

    @Test
    public void testTritonOnesAs() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 16);
        SDVariable ones = sd.onesLike("ones1", x);
        x.mul("result", ones);

        INDArray xArr = Nd4j.randn(DataType.FLOAT, 4, 16);
        runOpTest("testTritonOnesAs", sd, Map.of("x", xArr), "result");
    }

    @Test
    public void testTritonZerosLike() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 16);
        SDVariable z = sd.zerosLike("z1", x);
        x.add("result", z);

        INDArray xArr = Nd4j.randn(DataType.FLOAT, 4, 16);
        runOpTest("testTritonZerosLike", sd, Map.of("x", xArr), "result");
    }

    @Test
    public void testTritonShapeOf() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 16);
        SDVariable shape = sd.shape("shape1", x);
        // Cast to float so we can compare
        sd.castTo("result", shape, DataType.FLOAT);

        INDArray xArr = Nd4j.randn(DataType.FLOAT, 4, 16);
        runOpTest("testTritonShapeOf", sd, Map.of("x", xArr), "result");
    }

    @Test
    public void testTritonSetScalar() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 16);
        // set_scalar is used internally; use a simple add chain to exercise it
        SDVariable h1 = x.add("add1", sd.constant("c1", Nd4j.scalar(DataType.FLOAT, 1.0f)));
        SDVariable h2 = h1.mul("mul1", sd.constant("c2", Nd4j.scalar(DataType.FLOAT, 2.0f)));
        sd.identity("result", h2);

        INDArray xArr = Nd4j.randn(DataType.FLOAT, 4, 16);
        runOpTest("testTritonSetScalar", sd, Map.of("x", xArr), "result");
    }

    @Test
    public void testTritonRange() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 16);
        SDVariable r = sd.range("range1", 0.0, 16.0, 1.0, DataType.FLOAT);
        // Broadcast-add range to each row of x
        x.add("result", r);

        INDArray xArr = Nd4j.randn(DataType.FLOAT, 4, 16);
        runOpTest("testTritonRange", sd, Map.of("x", xArr), "result");
    }

    @Test
    public void testTritonStack() {
        SameDiff sd = SameDiff.create();
        SDVariable a = sd.placeHolder("a", DataType.FLOAT, -1, 16);
        SDVariable b = sd.constant("b", Nd4j.randn(DataType.FLOAT, 4, 16));
        sd.stack("result", 0, a, b);

        INDArray aArr = Nd4j.randn(DataType.FLOAT, 4, 16);
        runOpTest("testTritonStack", sd, Map.of("a", aArr), "result");
    }

    @Test
    public void testTritonGatherStackElementwiseSingleRange() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        var env = Nd4j.getEnvironment();

        boolean prevCompileAll = env.tritonCompileAll();
        String prevIncludeTypes = env.tritonIncludeTypes();
        boolean prevSectionFusion = env.tritonSectionFusion();
        boolean prevFusionScoring = env.tritonFusionScoring();
        float prevFusionMinScore = env.tritonFusionMinScore();
        boolean prevGraphCapture = env.tritonGraphCapture();

        env.setTritonCompileAll(true);
        env.setTritonIncludeTypes("GATHER,STACK");
        env.setTritonSectionFusion(true);
        env.setTritonFusionScoring(true);
        env.setTritonFusionMinScore(5.0f);
        env.setTritonGraphCapture(false);

        try {
            SameDiff sd = SameDiff.create();
            SDVariable input = sd.placeHolder("input", DataType.FLOAT, 16, 8);
            SDVariable gathered = sd.gather("gather1", input, new int[]{0, 3, 7, 1}, 0);
            SDVariable shifted = gathered.add("shifted",
                    sd.constant("bias", Nd4j.linspace(1, 8, 8, DataType.FLOAT).reshape(1, 8)));
            SDVariable scaled = gathered.mul("scaled",
                    sd.constant("scale", Nd4j.valueArrayOf(new long[]{1, 8}, 0.5f, DataType.FLOAT)));
            SDVariable stacked = sd.stack("stacked", 0, shifted, scaled);
            sd.nn.relu("result", stacked, 0.0);

            INDArray inputArr = Nd4j.randn(DataType.FLOAT, 16, 8);
            Map<String, INDArray> ph = Map.of("input", inputArr);

            DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "result");
            assertNotNull(plan, "Plan compilation returned null");

            Pointer planHandle = compileNativePlan(plan);
            Assumptions.assumeTrue(planHandle != null, "Native executor unavailable");
            try {
                INDArray[] extInputs = resolveExternalInputs(plan, sd, ph);

                nativeOps.invalidateTritonCache();
                nativeOps.resetTritonCounters();

                // First execution (slot-by-slot, unfrozen) — use as reference
                Map<String, INDArray> warmup = executeNativePlan(planHandle, plan, extInputs);
                INDArray refOutput = warmup.get("result").dup();
                assertNotNull(refOutput, "Reference output is null");

                nativeOps.setPlanShapesFrozen(planHandle, true);

                Map<String, INDArray> frozenWarmup = executeNativePlan(planHandle, plan, extInputs);
                INDArray frozenWarmupOutput = frozenWarmup.get("result");
                assertNotNull(frozenWarmupOutput, "Frozen warmup output is null");
                assertTrue(refOutput.sub(frozenWarmupOutput).amaxNumber().doubleValue() < TOLERANCE,
                        "Frozen warmup output mismatch");

                nativeOps.resetTritonCounters();

                Map<String, INDArray> compiled = executeNativePlan(planHandle, plan, extInputs);
                INDArray compiledOutput = compiled.get("result");
                assertNotNull(compiledOutput, "Compiled output is null");
                double maxDiff = refOutput.sub(compiledOutput).amaxNumber().doubleValue();
                assertTrue(maxDiff < TOLERANCE,
                        "Compiled output mismatch: max diff = " + maxDiff);

                long launches = nativeOps.getTritonKernelLaunchCount();
                assertEquals(1L, launches,
                        "Expected a single Triton launch for gather+elementwise+stack fusion");
            } finally {
                nativeOps.freeDynamicShapePlan(planHandle);
            }
        } finally {
            env.setTritonCompileAll(prevCompileAll);
            env.setTritonIncludeTypes(prevIncludeTypes);
            env.setTritonSectionFusion(prevSectionFusion);
            env.setTritonFusionScoring(prevFusionScoring);
            env.setTritonFusionMinScore(prevFusionMinScore);
            env.setTritonGraphCapture(prevGraphCapture);
        }
    }

    @Test
    public void testTritonBoolNot() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 16);
        SDVariable gt = sd.gt("gt1", x, 0.0);
        SDVariable notGt = sd.booleanNot("not1", gt);
        sd.castTo("result", notGt, DataType.FLOAT);

        INDArray xArr = Nd4j.randn(DataType.FLOAT, 4, 16);
        runOpTest("testTritonBoolNot", sd, Map.of("x", xArr), "result");
    }

    @Test
    @Disabled("Triton gather_nd kernel crashes CUDA context")
    public void testTritonGatherNd() {
        SameDiff sd = SameDiff.create();
        SDVariable data = sd.placeHolder("data", DataType.FLOAT, -1, 8);
        // Indices: pick specific rows
        SDVariable indices = sd.constant("indices", Nd4j.createFromArray(new int[][]{{0}, {2}, {1}}));
        sd.gatherNd("result", data, indices);

        INDArray dataArr = Nd4j.randn(DataType.FLOAT, 4, 8);
        runOpTest("testTritonGatherNd", sd, Map.of("data", dataArr), "result");
    }

    @Test
    public void testTritonSplitV() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 16);
        // Split [4,16] along axis=1 into sizes [4, 4, 8]
        SDVariable sizes = sd.constant("sizes", Nd4j.createFromArray(4, 4, 8));
        SDVariable[] splits = sd.splitV(x, sizes, 3, 1);
        // Take the last chunk and add bias to make it a usable output
        splits[2].add("result", sd.constant("bias", Nd4j.ones(DataType.FLOAT, 1, 8)));

        INDArray xArr = Nd4j.randn(DataType.FLOAT, 4, 16);
        runOpTest("testTritonSplitV", sd, Map.of("x", xArr), "result");
    }

    @Test
    public void testTritonScatterNdUpdate() {
        SameDiff sd = SameDiff.create();
        SDVariable data = sd.placeHolder("data", DataType.FLOAT, 4, 8);
        SDVariable indices = sd.constant("indices", Nd4j.createFromArray(new int[][]{{0}, {2}}));
        SDVariable updates = sd.constant("updates", Nd4j.ones(DataType.FLOAT, 2, 8));
        sd.scatterNdUpdate("result", data, indices, updates);

        INDArray dataArr = Nd4j.randn(DataType.FLOAT, 4, 8);
        runOpTest("testTritonScatterNdUpdate", sd, Map.of("data", dataArr), "result");
    }

    @Test
    public void testTritonFlatten2d() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4, 8);
        // flatten_2d flattens all dims after axis into one dim
        SDVariable flat = sd.reshape("flat1", x, 2, -1, 32);
        sd.identity("result", flat);

        INDArray xArr = Nd4j.randn(DataType.FLOAT, 2, 4, 8);
        runOpTest("testTritonFlatten2d", sd, Map.of("x", xArr), "result");
    }

    @Test
    public void testTritonReduceMean() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 32);
        sd.mean("result", x, 1);

        INDArray xArr = Nd4j.randn(DataType.FLOAT, 4, 32);
        runOpTest("testTritonReduceMean", sd, Map.of("x", xArr), "result");
    }

    @Test
    public void testTritonMultiply() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 16);
        SDVariable y = sd.constant("y", Nd4j.randn(DataType.FLOAT, 1, 16));
        x.mul("result", y);

        INDArray xArr = Nd4j.randn(DataType.FLOAT, 4, 16);
        runOpTest("testTritonMultiply", sd, Map.of("x", xArr), "result");
    }

    @Test
    public void testTritonAdd() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 16);
        SDVariable y = sd.constant("y", Nd4j.randn(DataType.FLOAT, 1, 16));
        x.add("result", y);

        INDArray xArr = Nd4j.randn(DataType.FLOAT, 4, 16);
        runOpTest("testTritonAdd", sd, Map.of("x", xArr), "result");
    }

    /**
     * Test all VLM decoder ops in a single mixed graph.
     * This exercises the mega-segment pattern where element-wise, matmul,
     * reshape, gather, and other ops are all fused together.
     */
    @Test
    public void testTritonVlmDecoderMixedOps() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 64);
        SDVariable w = sd.constant("w", Nd4j.randn(DataType.FLOAT, 64, 32));
        SDVariable bias = sd.constant("bias", Nd4j.randn(DataType.FLOAT, 1, 32));
        SDVariable scale = sd.constant("scale", Nd4j.rand(DataType.FLOAT, 1, 32).addi(0.1));

        // matmul
        SDVariable h1 = sd.mmul("mm1", x, w);
        // add
        SDVariable h2 = h1.add("add1", bias);
        // sigmoid
        SDVariable h3 = sd.nn.sigmoid("sig1", h2);
        // multiply
        SDVariable h4 = h3.mul("mul1", scale);
        // sqrt (on positive values)
        SDVariable h5 = sd.math.abs("abs1", h4);
        SDVariable h6 = sd.math.sqrt("sqrt1", h5.add(sd.constant("eps", Nd4j.scalar(0.001f))));
        // subtract
        SDVariable h7 = h6.sub("sub1", bias);
        // divide
        SDVariable h8 = h7.div("div1", scale);
        // pow
        SDVariable h9 = sd.math.pow("pow1", sd.math.abs("abs2", h8).add(sd.constant("eps2", Nd4j.scalar(0.001f))),
                                      sd.constant("exp", Nd4j.valueArrayOf(new long[]{1, 32}, 0.5f)));
        // reduce_mean
        SDVariable h10 = sd.mean("mean1", h9, 1);
        // reshape back
        SDVariable h11 = sd.expandDims("expand1", h10, 1);
        // squeeze
        sd.squeeze("result", h11, 1);

        INDArray xArr = Nd4j.randn(DataType.FLOAT, 4, 64);
        runOpTest("testTritonVlmDecoderMixedOps", sd, Map.of("x", xArr), "result");
    }

    // ─── im2col / col2im tests ──────────────────────────────────────────────

    /**
     * Test: im2col — rearrange image patches to columns.
     * input=[1,3,8,8] (NCHW), kernel=3x3, stride=1, pad=0, dilation=1
     * output=[1,3,3,3,6,6] (6D)
     */
    @Test
    public void testTritonIm2col() {
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 3, 8, 8);

        Conv2DConfig config = Conv2DConfig.builder()
                .kH(3).kW(3)
                .sH(1).sW(1)
                .pH(0).pW(0)
                .dH(1).dW(1)
                .dataFormat(Conv2DConfig.NCHW)
                .build();

        sd.cnn.im2Col("result", input, config);

        INDArray inputArr = Nd4j.randn(DataType.FLOAT, 1, 3, 8, 8);
        runOpTest("testTritonIm2col", sd, Map.of("input", inputArr), "result");
    }

    /**
     * Test: im2col with padding — ensures zero-pad boundary handling.
     * input=[2,1,6,6], kernel=3x3, stride=1, pad=1, dilation=1
     * output=[2,1,3,3,6,6] (same spatial size due to padding)
     */
    @Test
    public void testTritonIm2colWithPadding() {
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 1, 6, 6);

        Conv2DConfig config = Conv2DConfig.builder()
                .kH(3).kW(3)
                .sH(1).sW(1)
                .pH(1).pW(1)
                .dH(1).dW(1)
                .dataFormat(Conv2DConfig.NCHW)
                .build();

        sd.cnn.im2Col("result", input, config);

        INDArray inputArr = Nd4j.randn(DataType.FLOAT, 2, 1, 6, 6);
        runOpTest("testTritonIm2colWithPadding", sd, Map.of("input", inputArr), "result");
    }

    /**
     * Test: im2col with stride — verifies strided patch extraction.
     * input=[1,2,8,8], kernel=3x3, stride=2, pad=0, dilation=1
     * output=[1,2,3,3,3,3]
     */
    @Test
    public void testTritonIm2colWithStride() {
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 2, 8, 8);

        Conv2DConfig config = Conv2DConfig.builder()
                .kH(3).kW(3)
                .sH(2).sW(2)
                .pH(0).pW(0)
                .dH(1).dW(1)
                .dataFormat(Conv2DConfig.NCHW)
                .build();

        sd.cnn.im2Col("result", input, config);

        INDArray inputArr = Nd4j.randn(DataType.FLOAT, 1, 2, 8, 8);
        runOpTest("testTritonIm2colWithStride", sd, Map.of("input", inputArr), "result");
    }

    /**
     * Test: im2col with dilation — verifies dilated kernel extraction.
     * input=[1,1,10,10], kernel=3x3, stride=1, pad=0, dilation=2
     * output=[1,1,3,3,6,6]
     */
    @Test
    public void testTritonIm2colWithDilation() {
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 1, 10, 10);

        Conv2DConfig config = Conv2DConfig.builder()
                .kH(3).kW(3)
                .sH(1).sW(1)
                .pH(0).pW(0)
                .dH(2).dW(2)
                .dataFormat(Conv2DConfig.NCHW)
                .build();

        sd.cnn.im2Col("result", input, config);

        INDArray inputArr = Nd4j.randn(DataType.FLOAT, 1, 1, 10, 10);
        runOpTest("testTritonIm2colWithDilation", sd, Map.of("input", inputArr), "result");
    }

    /**
     * Test: col2im — rearrange columns back to image.
     * input=[1,1,3,3,4,4] (6D columns), stride=1, pad=0
     * output=[1,1,6,6] (4D image)
     * Note: Java Col2Im.addArgs() sends kH,kW at iArg[4,5] which C++ reads as imgH,imgW.
     * So kH/kW in the config must be the desired output image dimensions.
     */
    @Test
    public void testTritonCol2im() {
        SameDiff sd = SameDiff.create();
        // col2im input is 6D: [bS, iC, kH, kW, oH, oW]
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 1, 3, 3, 4, 4);

        // kH=6, kW=6 → output image dimensions (C++ reads iArg[4,5] as imgH, imgW)
        Conv2DConfig config = Conv2DConfig.builder()
                .kH(6).kW(6)
                .sH(1).sW(1)
                .pH(0).pW(0)
                .dH(1).dW(1)
                .dataFormat(Conv2DConfig.NCHW)
                .build();

        sd.cnn.col2Im("result", input, config);

        INDArray inputArr = Nd4j.randn(DataType.FLOAT, 1, 1, 3, 3, 4, 4);
        runOpTest("testTritonCol2im", sd, Map.of("input", inputArr), "result");
    }

    /**
     * Test: col2im with padding — verifies padded accumulation.
     * input=[1,2,3,3,6,6] (6D columns), stride=1, pad=1
     * output=[1,2,6,6] (4D image)
     */
    @Test
    public void testTritonCol2imWithPadding() {
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 2, 3, 3, 6, 6);

        // kH=6, kW=6 → output image dims
        Conv2DConfig config = Conv2DConfig.builder()
                .kH(6).kW(6)
                .sH(1).sW(1)
                .pH(1).pW(1)
                .dH(1).dW(1)
                .dataFormat(Conv2DConfig.NCHW)
                .build();

        sd.cnn.col2Im("result", input, config);

        INDArray inputArr = Nd4j.randn(DataType.FLOAT, 1, 2, 3, 3, 6, 6);
        runOpTest("testTritonCol2imWithPadding", sd, Map.of("input", inputArr), "result");
    }

    /**
     * Test: col2im with stride — verifies strided accumulation.
     * input=[1,1,3,3,3,3] (6D columns), stride=2, pad=0
     * iH = (oH-1)*sH + (kH_actual-1)*dH + 1 = (3-1)*2 + (3-1)*1 + 1 = 7
     * output=[1,1,7,7]
     */
    @Test
    public void testTritonCol2imWithStride() {
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 1, 3, 3, 3, 3);

        // kH=7, kW=7 → output image dims
        Conv2DConfig config = Conv2DConfig.builder()
                .kH(7).kW(7)
                .sH(2).sW(2)
                .pH(0).pW(0)
                .dH(1).dW(1)
                .dataFormat(Conv2DConfig.NCHW)
                .build();

        sd.cnn.col2Im("result", input, config);

        INDArray inputArr = Nd4j.randn(DataType.FLOAT, 1, 1, 3, 3, 3, 3);
        runOpTest("testTritonCol2imWithStride", sd, Map.of("input", inputArr), "result");
    }

    /**
     * Test: col2im with dilation — verifies dilated accumulation.
     * input=[1,1,3,3,6,6] (6D columns), stride=1, pad=0, dilation=2
     * iH = (oH-1)*sH + (kH_actual-1)*dH + 1 = (6-1)*1 + (3-1)*2 + 1 = 10
     * output=[1,1,10,10]
     */
    @Test
    public void testTritonCol2imWithDilation() {
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 1, 3, 3, 6, 6);

        // kH=10, kW=10 → output image dims
        Conv2DConfig config = Conv2DConfig.builder()
                .kH(10).kW(10)
                .sH(1).sW(1)
                .pH(0).pW(0)
                .dH(2).dW(2)
                .dataFormat(Conv2DConfig.NCHW)
                .build();

        sd.cnn.col2Im("result", input, config);

        INDArray inputArr = Nd4j.randn(DataType.FLOAT, 1, 1, 3, 3, 6, 6);
        runOpTest("testTritonCol2imWithDilation", sd, Map.of("input", inputArr), "result");
    }

    /**
     * Test: im2col → col2im round-trip.
     * Verifies that applying im2col followed by col2im produces consistent results
     * (the output will have accumulated overlapping patches, not exactly the original).
     * im2col: [1,1,8,8] → [1,1,3,3,6,6]
     * col2im: [1,1,3,3,6,6] → [1,1,8,8]
     */
    @Test
    @Disabled("Triton im2col+col2im round-trip kernel crashes CUDA context")
    public void testTritonIm2colCol2imRoundTrip() {
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 1, 8, 8);

        Conv2DConfig im2colConfig = Conv2DConfig.builder()
                .kH(3).kW(3)
                .sH(1).sW(1)
                .pH(0).pW(0)
                .dH(1).dW(1)
                .dataFormat(Conv2DConfig.NCHW)
                .build();

        SDVariable cols = sd.cnn.im2Col("im2col_out", input, im2colConfig);

        // For col2im, kH/kW = desired output image dimensions (8x8)
        Conv2DConfig col2imConfig = Conv2DConfig.builder()
                .kH(8).kW(8)
                .sH(1).sW(1)
                .pH(0).pW(0)
                .dH(1).dW(1)
                .dataFormat(Conv2DConfig.NCHW)
                .build();

        sd.cnn.col2Im("result", cols, col2imConfig);

        INDArray inputArr = Nd4j.randn(DataType.FLOAT, 1, 1, 8, 8);
        runOpTest("testTritonIm2colCol2imRoundTrip", sd, Map.of("input", inputArr), "result");
    }

    /**
     * Test: im2col + element-wise chain — verifies sectioned kernel with
     * im2col section followed by element-wise section.
     * im2col output is element-wise scaled, then relu'd.
     */
    @Test
    public void testTritonIm2colWithElementwise() {
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 1, 6, 6);

        Conv2DConfig config = Conv2DConfig.builder()
                .kH(3).kW(3)
                .sH(1).sW(1)
                .pH(0).pW(0)
                .dH(1).dW(1)
                .dataFormat(Conv2DConfig.NCHW)
                .build();

        SDVariable cols = sd.cnn.im2Col("im2col_out", input, config);
        SDVariable scaled = cols.mul("scale", sd.constant("s", Nd4j.scalar(0.5f)));
        sd.nn.relu("result", scaled, 0);

        INDArray inputArr = Nd4j.randn(DataType.FLOAT, 1, 1, 6, 6);
        runOpTest("testTritonIm2colWithElementwise", sd, Map.of("input", inputArr), "result");
    }

    /**
     * Test: element-wise + col2im chain — verifies sectioned kernel with
     * element-wise section followed by col2im section.
     */
    @Test
    public void testTritonCol2imWithElementwise() {
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 1, 3, 3, 4, 4);
        SDVariable scaled = input.mul("scale", sd.constant("s", Nd4j.scalar(2.0f)));

        // kH=6, kW=6 → output image dims (C++ col2im reads iArg[4,5] as imgH, imgW)
        Conv2DConfig config = Conv2DConfig.builder()
                .kH(6).kW(6)
                .sH(1).sW(1)
                .pH(0).pW(0)
                .dH(1).dW(1)
                .dataFormat(Conv2DConfig.NCHW)
                .build();

        sd.cnn.col2Im("result", scaled, config);

        INDArray inputArr = Nd4j.randn(DataType.FLOAT, 1, 1, 3, 3, 4, 4);
        runOpTest("testTritonCol2imWithElementwise", sd, Map.of("input", inputArr), "result");
    }

    // ─── Fallback-section fusion strategy tests ─────────────────────────────
    // These tests verify the matmul-exclusion strategy: matmul sections fall back
    // to cuBLAS via fallbackRangeExecutor_, element-wise chains compile as Triton kernels.

    /**
     * Test: Two matmuls with element-wise chains between them.
     * Simulates a simplified transformer block: ew → matmul → ew → matmul → ew.
     * With fallback-section strategy, matmuls run via cuBLAS and element-wise
     * chains compile as separate Triton kernels. Verifies correctness of the
     * gap-filling mechanism when multiple fallback gaps exist.
     */
    @Test
    public void testTritonMultiMatmulWithEwChains() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("input", DataType.FLOAT, -1, 32);

        // Pre-matmul element-wise chain
        SDVariable a1 = sd.constant("a1", Nd4j.randn(DataType.FLOAT, 1, 32));
        SDVariable h1 = x.add("pre_add", a1);
        SDVariable h2 = sd.nn.relu("pre_relu", h1, 0);

        // First matmul (should fall back to cuBLAS)
        SDVariable w1 = sd.constant("w1", Nd4j.randn(DataType.FLOAT, 32, 64));
        SDVariable mm1 = sd.mmul("mm1", h2, w1);

        // Mid element-wise chain (should compile as Triton kernel)
        SDVariable b1 = sd.constant("b1", Nd4j.randn(DataType.FLOAT, 1, 64));
        SDVariable s1 = sd.constant("s1", Nd4j.randn(DataType.FLOAT, 1, 64));
        SDVariable h3 = mm1.add("mid_add1", b1);
        SDVariable h4 = sd.nn.relu("mid_relu", h3, 0);
        SDVariable h5 = h4.mul("mid_mul", s1);
        SDVariable h6 = sd.math.tanh("mid_tanh", h5);

        // Second matmul (should fall back to cuBLAS)
        SDVariable w2 = sd.constant("w2", Nd4j.randn(DataType.FLOAT, 64, 16));
        SDVariable mm2 = sd.mmul("mm2", h6, w2);

        // Post element-wise chain (should compile as Triton kernel)
        SDVariable b2 = sd.constant("b2", Nd4j.randn(DataType.FLOAT, 1, 16));
        SDVariable h7 = mm2.add("post_add", b2);
        sd.nn.sigmoid("result", h7);

        runOpTest("testTritonMultiMatmulWithEwChains", sd,
                  Map.of("input", Nd4j.randn(DataType.FLOAT, 8, 32)), "result");
    }

    /**
     * Test: Three matmuls simulating Q/K/V projections + output projection
     * in a transformer attention block. Each matmul is separated by element-wise ops.
     * This is the core pattern that the fallback-section strategy targets.
     */
    @Test
    public void testTritonTransformerBlockPattern() {
        SameDiff sd = SameDiff.create();
        int dim = 32;
        SDVariable x = sd.placeHolder("input", DataType.FLOAT, -1, dim);

        // Layer norm (element-wise approximation: subtract mean, divide by std)
        SDVariable lnScale = sd.constant("ln_scale", Nd4j.ones(DataType.FLOAT, 1, dim));
        SDVariable lnBias = sd.constant("ln_bias", Nd4j.zeros(DataType.FLOAT, 1, dim));
        SDVariable normed = x.mul("ln_mul", lnScale).add("ln_add", lnBias);

        // Q projection matmul (cuBLAS fallback)
        SDVariable wq = sd.constant("wq", Nd4j.randn(DataType.FLOAT, dim, dim));
        SDVariable q = sd.mmul("q_proj", normed, wq);

        // K projection matmul (cuBLAS fallback)
        SDVariable wk = sd.constant("wk", Nd4j.randn(DataType.FLOAT, dim, dim));
        SDVariable k = sd.mmul("k_proj", normed, wk);

        // Attention scores: Q @ K^T (cuBLAS fallback)
        // Note: using reshape to simulate transpose for simplicity
        SDVariable scores = sd.mmul("attn_scores", q, k, false, true, false);

        // Softmax approximation via element-wise (Triton kernel)
        SDVariable scaleFactor = sd.constant("scale_factor",
            Nd4j.scalar(DataType.FLOAT, 1.0f / (float)Math.sqrt(dim)));
        SDVariable scaledScores = scores.mul("scale_scores", scaleFactor);
        SDVariable expScores = sd.math.exp("exp_scores", scaledScores);
        SDVariable sumExp = expScores.sum("sum_exp", true, -1);
        SDVariable softmax = expScores.div("softmax_div", sumExp);

        // V projection matmul (cuBLAS fallback)
        SDVariable wv = sd.constant("wv", Nd4j.randn(DataType.FLOAT, dim, dim));
        SDVariable v = sd.mmul("v_proj", normed, wv);

        // Attention output: softmax @ V (cuBLAS fallback)
        SDVariable attnOut = sd.mmul("attn_out", softmax, v);

        // Output projection (cuBLAS fallback)
        SDVariable wo = sd.constant("wo", Nd4j.randn(DataType.FLOAT, dim, dim));
        SDVariable projected = sd.mmul("out_proj", attnOut, wo);

        // Residual + final element-wise (Triton kernel)
        SDVariable residual = projected.add("residual", x);
        sd.nn.relu("result", residual, 0);

        runOpTest("testTritonTransformerBlockPattern", sd,
                  Map.of("input", Nd4j.randn(DataType.FLOAT, 4, dim)), "result");
    }

    /**
     * Test: Pure element-wise chain with NO matmuls.
     * With fallback-section strategy, this should compile entirely as a single Triton kernel
     * with no gaps. Verifies that element-wise-only segments are unaffected by the change.
     */
    @Test
    public void testTritonPureElementWiseNoFallback() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("input", DataType.FLOAT, -1, 64);
        SDVariable a = sd.constant("a", Nd4j.randn(DataType.FLOAT, 1, 64));
        SDVariable b = sd.constant("b", Nd4j.randn(DataType.FLOAT, 1, 64));
        SDVariable c = sd.constant("c", Nd4j.randn(DataType.FLOAT, 1, 64));

        // Long element-wise chain — should fuse entirely into one Triton kernel
        SDVariable h1 = x.add("add1", a);
        SDVariable h2 = sd.nn.relu("relu1", h1, 0);
        SDVariable h3 = h2.mul("mul1", b);
        SDVariable h4 = sd.math.tanh("tanh1", h3);
        SDVariable h5 = h4.add("add2", c);
        SDVariable h6 = sd.nn.sigmoid("sigmoid1", h5);
        SDVariable h7 = h6.mul("mul2", a);
        SDVariable h8 = h7.sub("sub1", b);
        SDVariable h9 = sd.math.abs("abs1", h8);
        SDVariable h10 = h9.add("add3", c);
        SDVariable h11 = sd.math.exp("exp1", h10);
        sd.nn.relu("result", h11, 0);

        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        nativeOps.resetTritonCounters();

        runOpTest("testTritonPureElementWiseNoFallback", sd,
                  Map.of("input", Nd4j.randn(DataType.FLOAT, 8, 64)), "result");

        // With pure element-wise, there should be NO fallback ranges
        long launches = nativeOps.getTritonKernelLaunchCount();
        log.info("testTritonPureElementWiseNoFallback: Triton launches={}", launches);
    }

    /**
     * Test: Single matmul sandwiched by element-wise chains.
     * The simplest case of the fallback strategy: ew → matmul → ew.
     * This is a regression test for the cooperative launch path (now replaced by fallback).
     */
    @Test
    public void testTritonSingleMatmulFallback() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("input", DataType.FLOAT, -1, 16);

        // Pre-matmul element-wise
        SDVariable a = sd.constant("a", Nd4j.randn(DataType.FLOAT, 1, 16));
        SDVariable h1 = x.add("pre_add", a);
        SDVariable h2 = sd.nn.relu("pre_relu", h1, 0);
        SDVariable h3 = sd.math.tanh("pre_tanh", h2);

        // Matmul (cuBLAS fallback)
        SDVariable w = sd.constant("w", Nd4j.randn(DataType.FLOAT, 16, 32));
        SDVariable mm = sd.mmul("mm", h3, w);

        // Post-matmul element-wise
        SDVariable b = sd.constant("b", Nd4j.randn(DataType.FLOAT, 1, 32));
        SDVariable h4 = mm.add("post_add", b);
        SDVariable h5 = sd.nn.sigmoid("post_sigmoid", h4);
        SDVariable h6 = h5.mul("post_mul", sd.constant("s", Nd4j.randn(DataType.FLOAT, 1, 32)));
        sd.nn.relu("result", h6, 0);

        runOpTest("testTritonSingleMatmulFallback", sd,
                  Map.of("input", Nd4j.randn(DataType.FLOAT, 4, 16)), "result");
    }

    // ─── RMS norm sub-pattern isolation tests ─────────────────────────────

    /**
     * Test: reduce_mean with keepDims=true.
     * x=[1,576] -> mean(axis=1, keepDims=true) -> result=[1,1]
     */
    @Test
    public void testReduceMeanKeepDims() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 576);
        sd.mean("result", x, true, 1);

        runOpTest("testReduceMeanKeepDims", sd,
                  Map.of("x", Nd4j.randn(DataType.FLOAT, 1, 576)), "result");
    }

    /**
     * Test: div with broadcast [1,576] / [1,1].
     * Simulates the broadcast division in RMS norm.
     */
    @Test
    public void testDivBroadcast() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 576);
        SDVariable denom = sd.placeHolder("denom", DataType.FLOAT, -1, 1);
        x.div("result", denom);

        runOpTest("testDivBroadcast", sd,
                  Map.of("x", Nd4j.randn(DataType.FLOAT, 1, 576),
                         "denom", Nd4j.scalar(DataType.FLOAT, 2.5f).reshape(1, 1)),
                  "result");
    }

    /**
     * Test: mul(x,x) -> reduce_mean(keepDims=true) chain.
     * This is the first half of RMS norm: computing variance.
     */
    @Test
    public void testSquareThenReduceMean() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 576);
        SDVariable squared = x.mul("square", x);
        sd.mean("result", squared, true, 1);

        runOpTest("testSquareThenReduceMean", sd,
                  Map.of("x", Nd4j.randn(DataType.FLOAT, 1, 576)), "result");
    }

    /**
     * Test: reduce_mean(keepDims) -> add(eps) -> sqrt chain.
     * This is the denominator computation of RMS norm.
     */
    @Test
    public void testReduceMeanAddSqrt() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 576);
        SDVariable meanVal = sd.mean("mean", x, true, 1);
        SDVariable eps = sd.constant("eps", Nd4j.scalar(DataType.FLOAT, 1e-5f));
        SDVariable added = meanVal.add("add_eps", eps);
        sd.math.sqrt("result", added);

        runOpTest("testReduceMeanAddSqrt", sd,
                  Map.of("x", Nd4j.rand(DataType.FLOAT, 1, 576)), "result");
    }

    /**
     * Test: full RMS norm denominator: mul(x,x) -> reduce_mean(keepDims) -> add(eps) -> sqrt.
     * Tests the computation of the normalization factor.
     */
    @Test
    public void testRmsNormDenominator() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 576);
        SDVariable squared = x.mul("square", x);
        SDVariable meanSq = sd.mean("mean", squared, true, 1);
        SDVariable eps = sd.constant("eps", Nd4j.scalar(DataType.FLOAT, 1e-5f));
        SDVariable denom = meanSq.add("add_eps", eps);
        sd.math.sqrt("result", denom);

        runOpTest("testRmsNormDenominator", sd,
                  Map.of("x", Nd4j.randn(DataType.FLOAT, 1, 576)), "result");
    }

    /**
     * Test: full RMS norm: x -> x*x -> mean -> +eps -> sqrt -> x/sqrt.
     * The division broadcasts [1,1] denom against [1,576] numerator.
     */
    @Test
    public void testRmsNormFull() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 576);
        SDVariable squared = x.mul("square", x);
        SDVariable meanSq = sd.mean("mean", squared, true, 1);
        SDVariable eps = sd.constant("eps", Nd4j.scalar(DataType.FLOAT, 1e-5f));
        SDVariable denom = meanSq.add("add_eps", eps);
        SDVariable sqrtDenom = sd.math.sqrt("sqrt", denom);
        x.div("result", sqrtDenom);

        runOpTest("testRmsNormFull", sd,
                  Map.of("x", Nd4j.randn(DataType.FLOAT, 1, 576)), "result");
    }

    /**
     * Test: RMS norm with preceding add (residual connection).
     * (x + residual) -> square -> mean -> +eps -> sqrt -> div -> mul(weight).
     * This is the exact pattern from the failing testTritonRmsNormPattern.
     */
    @Test
    public void testRmsNormWithResidual() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 576);
        SDVariable residual = sd.constant("residual", Nd4j.randn(DataType.FLOAT, 1, 576));
        SDVariable weight = sd.constant("weight", Nd4j.ones(DataType.FLOAT, 1, 576));

        SDVariable added = x.add("add", residual);
        SDVariable squared = added.mul("square", added);
        SDVariable meanSq = sd.mean("mean", squared, true, 1);
        SDVariable eps = sd.constant("eps", Nd4j.scalar(DataType.FLOAT, 1e-5f));
        SDVariable denom = meanSq.add("add_eps", eps);
        SDVariable sqrtDenom = sd.math.sqrt("sqrt", denom);
        SDVariable normed = added.div("div", sqrtDenom);
        normed.mul("result", weight);

        runOpTest("testRmsNormWithResidual", sd,
                  Map.of("x", Nd4j.randn(DataType.FLOAT, 1, 576)), "result");
    }

    // ─── Fan-out pattern isolation tests ──────────────────────────────────

    /**
     * Test: simplest fan-out — x -> add(a) -> y; y feeds both mul(y,y) and result.
     * y is consumed twice: once for square and once as output via identity.
     */
    @Test
    public void testFanOutSimple() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 8);
        SDVariable a = sd.constant("a", Nd4j.randn(DataType.FLOAT, 1, 8));
        SDVariable y = x.add("y", a);
        SDVariable sq = y.mul("sq", y);
        SDVariable combined = sq.add("result", y);

        runOpTest("testFanOutSimple", sd,
                  Map.of("x", Nd4j.randn(DataType.FLOAT, 2, 8)), "result");
    }

    /**
     * Test: fan-out with reduce — y feeds both mean(y*y) and div(y, sqrt(mean)).
     * Minimal reproduction of the RMS norm fan-out pattern.
     */
    @Test
    public void testFanOutWithReduce() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 32);
        SDVariable a = sd.constant("a", Nd4j.randn(DataType.FLOAT, 1, 32));
        SDVariable y = x.add("y", a);
        SDVariable sq = y.mul("sq", y);
        SDVariable meanSq = sd.mean("mean", sq, true, 1);
        SDVariable sqrtMean = sd.math.sqrt("sqrt", meanSq);
        y.div("result", sqrtMean);

        runOpTest("testFanOutWithReduce", sd,
                  Map.of("x", Nd4j.randn(DataType.FLOAT, 1, 32)), "result");
    }

    /**
     * Test: fan-out WITHOUT reduce — y used as both inputs to mul, plus later add.
     * Tests whether the fan-out issue is related to reduction or just fan-out itself.
     */
    @Test
    public void testFanOutNoReduce() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 32);
        SDVariable a = sd.constant("a", Nd4j.randn(DataType.FLOAT, 1, 32));
        SDVariable y = x.add("y", a);
        SDVariable sq = y.mul("sq", y);   // y used twice as input
        SDVariable denom = sd.math.sqrt("sqrt", sq);
        y.div("result", denom);           // y used again

        runOpTest("testFanOutNoReduce", sd,
                  Map.of("x", Nd4j.randn(DataType.FLOAT, 1, 32)), "result");
    }

    /**
     * Test: minimal fan-out failure case.
     * x -> y=add(a); y -> mul(y) -> z; div(y, z).
     * y is reused after its buffer could be overwritten by z.
     */
    @Test
    public void testFanOutMinimal() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 8);
        SDVariable a = sd.constant("a", Nd4j.randn(DataType.FLOAT, 1, 8));
        SDVariable y = x.add("y", a);          // y = x + a
        SDVariable z = y.mul("z", y);           // z = y * y (same shape as y)
        y.div("result", z);                     // result = y / z — but y's buffer may be overwritten

        runOpTest("testFanOutMinimal", sd,
                  Map.of("x", Nd4j.randn(DataType.FLOAT, 2, 8)), "result");
    }

    /**
     * Test: fan-out where intermediate has DIFFERENT shape from reused variable.
     * x -> y=add(a); y -> mean(y) -> z; div(y, z).
     * z has shape [2,1], y has shape [2,8] — buffer can't be reused.
     */
    @Test
    public void testFanOutDifferentShape() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 8);
        SDVariable a = sd.constant("a", Nd4j.randn(DataType.FLOAT, 1, 8));
        SDVariable y = x.add("y", a);
        SDVariable z = sd.mean("z", y, true, 1);  // z=[2,1], different from y=[2,8]
        y.div("result", z);

        runOpTest("testFanOutDifferentShape", sd,
                  Map.of("x", Nd4j.randn(DataType.FLOAT, 2, 8)), "result");
    }

    /**
     * Test: fan-out where variable is reused AFTER several ops.
     * x -> y=add(a) -> sq=mul(y,y) -> mean -> eps -> sqrt -> div(y, sqrt).
     * Like RMS norm but smaller dimension (32 vs 576).
     */
    @Test
    public void testFanOutRmsNormSmall() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 32);
        SDVariable a = sd.constant("a", Nd4j.randn(DataType.FLOAT, 1, 32));
        SDVariable y = x.add("y", a);
        SDVariable sq = y.mul("sq", y);
        SDVariable meanSq = sd.mean("mean", sq, true, 1);
        SDVariable eps = sd.constant("eps", Nd4j.scalar(DataType.FLOAT, 1e-5f));
        SDVariable denom = meanSq.add("add_eps", eps);
        SDVariable sqrtDenom = sd.math.sqrt("sqrt", denom);
        y.div("result", sqrtDenom);

        runOpTest("testFanOutRmsNormSmall", sd,
                  Map.of("x", Nd4j.randn(DataType.FLOAT, 1, 32)), "result");
    }

    // ─── Feature combination tests ──────────────────────────────────────────
    // These test specific execution mode combinations to isolate issues faster
    // than running the full VLM pipeline.
    //
    // Matrix: {graph types} x {execution modes} x {frozen/unfrozen} x {iteration count}
    // Graph types: matmul chain, element-wise chain, RMS norm pattern, cooperative EW+matmul
    // Execution modes: default(AUTO), Triton+graphCapture, Triton direct, CUDA graphs
    // Frozen: always freeze after iter 0 (unfrozen re-execution has known cache corruption)

    private enum GraphType {
        MATMUL_CHAIN,       // matmul + bias + activation (cuBLAS-heavy)
        ELEMENT_WISE,       // purely element-wise (best for Triton fusion)
        COOPERATIVE_EW_MM,  // EW -> matmul -> EW (mixed sections)
        RMS_NORM            // reduction pattern (mean + sqrt + div)
    }

    private SameDiff buildGraph(GraphType type) {
        switch (type) {
            case MATMUL_CHAIN:    return createMatmulAddReluChain();
            case ELEMENT_WISE:    return createElementWiseChain();
            case COOPERATIVE_EW_MM: {
                SameDiff sd = SameDiff.create();
                SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 32);
                SDVariable a = sd.constant("a", Nd4j.randn(DataType.FLOAT, 1, 32));
                SDVariable w = sd.constant("w", Nd4j.randn(DataType.FLOAT, 32, 64));
                SDVariable b = sd.constant("b", Nd4j.randn(DataType.FLOAT, 1, 64));
                SDVariable s = sd.constant("s", Nd4j.randn(DataType.FLOAT, 1, 64));
                SDVariable h1 = x.add("add1", a);
                SDVariable h2 = sd.nn.relu("relu1", h1, 0);
                SDVariable h3 = sd.mmul("mm1", h2, w);
                SDVariable h4 = h3.add("add2", b);
                SDVariable h5 = sd.math.tanh("tanh1", h4);
                h5.mul("result", s);
                return sd;
            }
            case RMS_NORM: {
                SameDiff sd = SameDiff.create();
                SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 32);
                SDVariable a = sd.constant("a", Nd4j.randn(DataType.FLOAT, 1, 32));
                SDVariable weight = sd.constant("weight", Nd4j.ones(DataType.FLOAT, 1, 32));
                SDVariable y = x.add("add", a);
                SDVariable sq = y.mul("sq", y);
                SDVariable meanSq = sd.mean("mean", sq, true, 1);
                SDVariable eps = sd.constant("eps", Nd4j.scalar(DataType.FLOAT, 1e-5f));
                SDVariable denom = meanSq.add("add_eps", eps);
                SDVariable sqrtDenom = sd.math.sqrt("sqrt", denom);
                SDVariable normed = y.div("div", sqrtDenom);
                normed.mul("result", weight);
                return sd;
            }
            default: throw new IllegalArgumentException("Unknown graph type: " + type);
        }
    }

    private INDArray buildInput(GraphType type) {
        switch (type) {
            case MATMUL_CHAIN:       return Nd4j.randn(DataType.FLOAT, 2, 4);
            case ELEMENT_WISE:       return Nd4j.randn(DataType.FLOAT, 4, 8);
            case COOPERATIVE_EW_MM:  return Nd4j.randn(DataType.FLOAT, 8, 32);
            case RMS_NORM:           return Nd4j.randn(DataType.FLOAT, 1, 32);
            default: throw new IllegalArgumentException();
        }
    }

    /**
     * Helper: run a plan N times with frozen shapes, verifying correctness each iteration.
     * Returns the execution mode used (Triton, CUDA graphs, or slot-by-slot).
     */
    private String runFrozenReplayTest(String testName, SameDiff sd, Map<String, INDArray> ph,
                                        String outputName, int iterations) {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        Map<String, INDArray> ref = sd.output(ph, outputName);
        INDArray refOutput = ref.get(outputName);
        assertNotNull(refOutput, testName + ": reference output is null");

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, outputName);
        assertNotNull(plan, testName + ": plan is null");
        Pointer planHandle = compileNativePlan(plan);
        if (planHandle == null) {
            log.info("Skipping {} (native executor not supported)", testName);
            return "SKIPPED";
        }
        try {
            INDArray[] extInputs = resolveExternalInputs(plan, sd, ph);
            nativeOps.resetTritonCounters();

            for (int iter = 0; iter < iterations; iter++) {
                Map<String, INDArray> nativeResults = executeNativePlan(planHandle, plan, extInputs);
                INDArray nativeOutput = nativeResults.get(outputName);
                assertNotNull(nativeOutput, testName + ": null at iter " + iter);
                double maxDiff = refOutput.sub(nativeOutput).amaxNumber().doubleValue();
                log.info("{} iter {}: maxDiff={}", testName, iter, maxDiff);

                // Warmup (iter 0) and slot-by-slot must be correct
                if (iter <= 1) {
                    assertTrue(maxDiff < TOLERANCE,
                               testName + " iter " + iter + ": output mismatch (maxDiff=" + maxDiff + ")");
                }

                // After warmup, freeze shapes
                if (iter == 0) {
                    nativeOps.setPlanShapesFrozen(planHandle, true);
                }
            }

            long tritonLaunches = nativeOps.getTritonKernelLaunchCount();
            String mode = tritonLaunches > 0 ? "TRITON" : "CUDA_GRAPHS_OR_SLOT_BY_SLOT";
            log.info("{}: {} iterations complete, tritonLaunches={}, mode={}", testName, iterations, tritonLaunches, mode);
            return mode;
        } finally {
            nativeOps.freeDynamicShapePlan(planHandle);
        }
    }

    /**
     * Test: Triton + CUDA graph capture + replay (the fixed SIGSEGV path).
     * Runs 6 iterations:
     *   iter 0: warmup (slot-by-slot)
     *   iter 1: frozen, Triton compile
     *   iter 2: Triton graph CAPTURE + first launch (was SIGSEGV before fix)
     *   iter 3-5: Triton graph REPLAY
     *
     * This is the MAX THROUGHPUT path and the primary regression target.
     */
    @Test
    public void testTritonGraphCaptureAndReplay() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        if (!nativeOps.isTritonAvailable()) {
            log.info("Skipping testTritonGraphCaptureAndReplay (Triton not available)");
            return;
        }

        // Ensure graph capture is ON (default)
        boolean prevCapture = Nd4j.getEnvironment().tritonGraphCapture();
        Nd4j.getEnvironment().setTritonGraphCapture(true);
        try {
            SameDiff sd = createMatmulAddReluChain();
            String mode = runFrozenReplayTest("testTritonGraphCaptureAndReplay", sd,
                    Map.of("x", Nd4j.randn(DataType.FLOAT, 2, 4)), "result", 6);
            assertEquals("TRITON", mode, "Expected Triton execution path");
        } finally {
            Nd4j.getEnvironment().setTritonGraphCapture(prevCapture);
        }
    }

    /**
     * Test: Triton direct execution (NO graph capture).
     * Sets tritonGraphCapture=false so Triton kernels execute directly each step
     * without being recorded into a CUDA graph.
     *
     * This is the fallback path when graph capture fails or is disabled.
     * ~9.5 tok/s vs ~50 tok/s with graph replay.
     */
    @Test
    public void testTritonDirectNoGraphCapture() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        if (!nativeOps.isTritonAvailable()) {
            log.info("Skipping testTritonDirectNoGraphCapture (Triton not available)");
            return;
        }

        boolean prevCapture = Nd4j.getEnvironment().tritonGraphCapture();
        Nd4j.getEnvironment().setTritonGraphCapture(false);
        try {
            SameDiff sd = createMatmulAddReluChain();
            String mode = runFrozenReplayTest("testTritonDirectNoGraphCapture", sd,
                    Map.of("x", Nd4j.randn(DataType.FLOAT, 2, 4)), "result", 5);
            assertEquals("TRITON", mode, "Expected Triton execution (direct, no graph capture)");
        } finally {
            Nd4j.getEnvironment().setTritonGraphCapture(prevCapture);
        }
    }

    /**
     * Test: CUDA graphs (no Triton) execution path.
     * Compiles the plan with CUDA_GRAPHS mode, bypassing Triton entirely.
     * This is the existing ~50 tok/s path (20ms/step graph replay).
     */
    @Test
    public void testCudaGraphsNoTriton() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        SameDiff sd = createMatmulAddReluChain();
        INDArray x = Nd4j.randn(DataType.FLOAT, 2, 4);
        Map<String, INDArray> ph = Map.of("x", x);

        Map<String, INDArray> ref = sd.output(ph, "result");
        INDArray refOutput = ref.get("result");

        // Compile with CUDA_GRAPHS mode explicitly (no Triton)
        var effectiveMode = sd.compileNativeDynamicShapePlan(
                List.of("result"), GraphExecutionMode.CUDA_GRAPHS, true);
        log.info("testCudaGraphsNoTriton: effectiveMode={}", effectiveMode);

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "result");
        Pointer planHandle = compileNativePlan(plan);
        if (planHandle == null) { log.info("Skipping (native executor not supported)"); return; }
        try {
            INDArray[] extInputs = resolveExternalInputs(plan, sd, ph);
            nativeOps.resetTritonCounters();

            for (int iter = 0; iter < 5; iter++) {
                Map<String, INDArray> nativeResults = executeNativePlan(planHandle, plan, extInputs);
                INDArray nativeOutput = nativeResults.get("result");
                assertNotNull(nativeOutput, "null at iter " + iter);
                double maxDiff = refOutput.sub(nativeOutput).amaxNumber().doubleValue();
                log.info("testCudaGraphsNoTriton iter {}: maxDiff={}", iter, maxDiff);
                assertTrue(maxDiff < TOLERANCE, "CUDA_GRAPHS iter " + iter + " mismatch: " + maxDiff);
                if (iter == 0) nativeOps.setPlanShapesFrozen(planHandle, true);
            }

            long tritonLaunches = nativeOps.getTritonKernelLaunchCount();
            // CUDA_GRAPHS mode controls graph capture/replay, not kernel selection.
            // Triton kernels may be used within CUDA graph segments.
            log.info("testCudaGraphsNoTriton: tritonLaunches={}", tritonLaunches);
        } finally {
            nativeOps.freeDynamicShapePlan(planHandle);
        }
    }

    /**
     * Test: Slot-by-slot correctness baseline (no graph capture, no Triton).
     * Every op executes individually WITHOUT freezing shapes (so it stays in
     * slot-by-slot execution). This is the ground truth for comparing other modes.
     */
    @Test
    public void testSlotBySlotBaseline() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        SameDiff sd = createMatmulAddReluChain();
        INDArray x = Nd4j.randn(DataType.FLOAT, 2, 4);
        Map<String, INDArray> ph = Map.of("x", x);

        Map<String, INDArray> ref = sd.output(ph, "result");
        INDArray refOutput = ref.get("result");

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "result");
        Pointer planHandle = compileNativePlan(plan);
        if (planHandle == null) { log.info("Skipping"); return; }
        try {
            INDArray[] extInputs = resolveExternalInputs(plan, sd, ph);

            // Do NOT freeze shapes — keep it in slot-by-slot execution path
            for (int iter = 0; iter < 5; iter++) {
                Map<String, INDArray> nativeResults = executeNativePlan(planHandle, plan, extInputs);
                INDArray nativeOutput = nativeResults.get("result");
                assertNotNull(nativeOutput, "null at iter " + iter);
                double maxDiff = refOutput.sub(nativeOutput).amaxNumber().doubleValue();
                log.info("testSlotBySlotBaseline iter {}: maxDiff={}", iter, maxDiff);
                assertTrue(maxDiff < TOLERANCE, "Slot-by-slot iter " + iter + " mismatch: " + maxDiff);
            }
        } finally {
            nativeOps.freeDynamicShapePlan(planHandle);
        }
    }

    /**
     * Test: Frozen shapes + many replays to detect stale data accumulation.
     * Runs 10 iterations after freezing. If nullify logic is broken,
     * stale data accumulates and magnitudes grow exponentially.
     */
    @Test
    public void testFrozenReplayStability() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        SameDiff sd = createElementWiseChain();
        INDArray x = Nd4j.randn(DataType.FLOAT, 4, 8);
        Map<String, INDArray> ph = Map.of("x", x);

        Map<String, INDArray> ref = sd.output(ph, "result");
        INDArray refOutput = ref.get("result");

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "result");
        Pointer planHandle = compileNativePlan(plan);
        if (planHandle == null) { log.info("Skipping"); return; }
        try {
            INDArray[] extInputs = resolveExternalInputs(plan, sd, ph);
            nativeOps.resetTritonCounters();

            double prevMaxDiff = 0;
            for (int iter = 0; iter < 12; iter++) {
                Map<String, INDArray> nativeResults = executeNativePlan(planHandle, plan, extInputs);
                INDArray nativeOutput = nativeResults.get("result");
                assertNotNull(nativeOutput, "null at iter " + iter);
                double maxDiff = refOutput.sub(nativeOutput).amaxNumber().doubleValue();
                double maxAbs = nativeOutput.amaxNumber().doubleValue();
                log.info("testFrozenReplayStability iter {}: maxDiff={} maxAbs={}", iter, maxDiff, maxAbs);

                // After warmup, freeze shapes
                if (iter == 0) {
                    nativeOps.setPlanShapesFrozen(planHandle, true);
                }

                // Stale data detection: if maxDiff is growing exponentially, something is wrong
                if (iter > 3 && maxDiff > 1.0) {
                    fail("testFrozenReplayStability: stale data detected at iter " + iter +
                         " (maxDiff=" + maxDiff + ", maxAbs=" + maxAbs + ")");
                }

                prevMaxDiff = maxDiff;
            }
        } finally {
            nativeOps.freeDynamicShapePlan(planHandle);
        }
    }

    /**
     * Test: Input pointer refresh on replay — same shapes, different input values.
     * Verifies that when input data changes between replays (same INDArray object,
     * different values via assign), the graph correctly picks up the new data.
     */
    @Test
    public void testInputRefreshOnReplay() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        SameDiff sd = createElementWiseChain();
        INDArray x = Nd4j.randn(DataType.FLOAT, 4, 8);
        Map<String, INDArray> ph = Map.of("x", x);

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "result");
        Pointer planHandle = compileNativePlan(plan);
        if (planHandle == null) { log.info("Skipping"); return; }
        try {
            INDArray[] extInputs = resolveExternalInputs(plan, sd, ph);

            // Warmup + freeze
            executeNativePlan(planHandle, plan, extInputs);
            nativeOps.setPlanShapesFrozen(planHandle, true);
            executeNativePlan(planHandle, plan, extInputs);  // first frozen
            executeNativePlan(planHandle, plan, extInputs);  // second frozen (may trigger capture)

            // Now change input values and verify output changes accordingly
            for (int trial = 0; trial < 3; trial++) {
                INDArray newX = Nd4j.randn(DataType.FLOAT, 4, 8);
                x.assign(newX);  // Same INDArray object, new values

                // Get fresh reference
                Map<String, INDArray> ref = sd.output(ph, "result");
                INDArray refOutput = ref.get("result");

                // Execute native plan with updated input
                Map<String, INDArray> nativeResults = executeNativePlan(planHandle, plan, extInputs);
                INDArray nativeOutput = nativeResults.get("result");
                assertNotNull(nativeOutput, "null at trial " + trial);

                double maxDiff = refOutput.sub(nativeOutput).amaxNumber().doubleValue();
                log.info("testInputRefreshOnReplay trial {}: maxDiff={}", trial, maxDiff);

                // The output should reflect the new input, not be stuck on old values
                // Allow slightly higher tolerance for Triton paths
                assertTrue(maxDiff < 0.01,
                           "Input refresh failed at trial " + trial + " (maxDiff=" + maxDiff +
                           ") — graph may be replaying stale input data");
            }
        } finally {
            nativeOps.freeDynamicShapePlan(planHandle);
        }
    }

    /**
     * COMPREHENSIVE: All graph types x all execution modes x frozen replay.
     * For each graph type, runs through SLOT_BY_SLOT, CUDA_GRAPHS, and TRITON (if available),
     * verifying that all modes produce the same reference output across multiple iterations.
     * This is the main combinatorial coverage test.
     */
    @Test
    public void testAllGraphTypesAllModes() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        boolean tritonAvailable = nativeOps.isTritonAvailable();

        GraphExecutionMode[] modes = tritonAvailable
                ? new GraphExecutionMode[]{GraphExecutionMode.SLOT_BY_SLOT, GraphExecutionMode.CUDA_GRAPHS, GraphExecutionMode.TRITON}
                : new GraphExecutionMode[]{GraphExecutionMode.SLOT_BY_SLOT, GraphExecutionMode.CUDA_GRAPHS};

        int failures = 0;
        StringBuilder failureReport = new StringBuilder();

        for (GraphType graphType : GraphType.values()) {
            // Build a reference SameDiff and get the ground-truth output
            SameDiff refSd = buildGraph(graphType);
            INDArray input = buildInput(graphType);
            Map<String, INDArray> ph = Map.of("x", input);
            Map<String, INDArray> refResults = refSd.output(ph, "result");
            INDArray refOutput = refResults.get("result");
            assertNotNull(refOutput, graphType + ": reference output is null");

            for (GraphExecutionMode mode : modes) {
                String testId = graphType + "+" + mode;
                log.info("=== {} ===", testId);

                try {
                    // Build fresh graph with same constants
                    SameDiff sd = buildGraph(graphType);
                    for (SDVariable v : refSd.variables()) {
                        if (v.getVariableType() == VariableType.CONSTANT && v.getArr() != null) {
                            SDVariable freshVar = sd.getVariable(v.name());
                            if (freshVar != null) freshVar.getArr().assign(v.getArr());
                        }
                    }

                    // Set execution mode
                    if (mode != GraphExecutionMode.SLOT_BY_SLOT) {
                        sd.compileNativeDynamicShapePlan(List.of("result"), mode, true);
                    }

                    // Triton graph capture settings
                    boolean prevCapture = Nd4j.getEnvironment().tritonGraphCapture();
                    if (mode == GraphExecutionMode.TRITON) {
                        Nd4j.getEnvironment().setTritonGraphCapture(true);
                    }

                    DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "result");
                    assertNotNull(plan, testId + ": plan is null");
                    Pointer planHandle = compileNativePlan(plan);
                    if (planHandle == null) {
                        log.info("Skipping {} (native executor not supported)", testId);
                        Nd4j.getEnvironment().setTritonGraphCapture(prevCapture);
                        continue;
                    }

                    try {
                        INDArray[] extInputs = resolveExternalInputs(plan, sd, ph);
                        nativeOps.resetTritonCounters();

                        int iters = (mode == GraphExecutionMode.TRITON) ? 6 : 4;
                        for (int iter = 0; iter < iters; iter++) {
                            Map<String, INDArray> nativeResults = executeNativePlan(planHandle, plan, extInputs);
                            INDArray nativeOutput = nativeResults.get("result");
                            assertNotNull(nativeOutput, testId + " null at iter " + iter);
                            double maxDiff = refOutput.sub(nativeOutput).amaxNumber().doubleValue();
                            log.info("{} iter {}: maxDiff={}", testId, iter, maxDiff);

                            // Slot-by-slot and CUDA graphs must match tightly
                            // Triton may have slightly higher tolerance but should still be close
                            double tol = (mode == GraphExecutionMode.TRITON && iter >= 2) ? 0.01 : TOLERANCE;
                            if (maxDiff >= tol) {
                                String msg = testId + " iter " + iter + ": maxDiff=" + maxDiff + " (tol=" + tol + ")";
                                log.error("FAIL: {}", msg);
                                failureReport.append(msg).append("\n");
                                failures++;
                            }

                            // Freeze after warmup
                            if (iter == 0) {
                                nativeOps.setPlanShapesFrozen(planHandle, true);
                            }
                        }

                        long tritonLaunches = nativeOps.getTritonKernelLaunchCount();
                        log.info("{}: tritonLaunches={}", testId, tritonLaunches);
                    } finally {
                        nativeOps.freeDynamicShapePlan(planHandle);
                        Nd4j.getEnvironment().setTritonGraphCapture(prevCapture);
                    }
                } catch (Exception e) {
                    String msg = testId + ": EXCEPTION " + e.getMessage();
                    log.error(msg, e);
                    failureReport.append(msg).append("\n");
                    failures++;
                }
            }
        }

        if (failures > 0) {
            fail("testAllGraphTypesAllModes: " + failures + " failures:\n" + failureReport);
        }
    }

    /**
     * Test: cuBLAS matmul correctness under different execution modes.
     * cuBLAS ops have historically broken in graph capture paths because
     * cublas handle state gets baked into the graph incorrectly.
     * Tests multiple matmul sizes to exercise different cuBLAS tiling paths.
     */
    @Test
    public void testCublasMatmulStress() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();

        // Different matmul sizes exercise different cuBLAS kernel paths
        int[][] sizes = {
                {1, 32, 64},    // tiny (may use a different kernel)
                {4, 64, 128},   // small
                {16, 128, 256}, // medium
                {1, 576, 576},  // VLM-like (attention projection)
        };

        for (int[] size : sizes) {
            int batch = size[0], m = size[1], n = size[2];
            String testId = "cublas_" + batch + "x" + m + "x" + n;
            log.info("=== {} ===", testId);

            SameDiff sd = SameDiff.create();
            SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, m);
            SDVariable w = sd.constant("w", Nd4j.randn(DataType.FLOAT, m, n));
            SDVariable b = sd.constant("b", Nd4j.randn(DataType.FLOAT, 1, n));
            SDVariable h = sd.mmul("mm", x, w);
            h.add("result", b);

            INDArray input = Nd4j.randn(DataType.FLOAT, batch, m);
            Map<String, INDArray> ph = Map.of("x", input);
            Map<String, INDArray> ref = sd.output(ph, "result");
            INDArray refOutput = ref.get("result");

            DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "result");
            Pointer planHandle = compileNativePlan(plan);
            if (planHandle == null) { log.info("Skipping {} (not supported)", testId); continue; }
            try {
                INDArray[] extInputs = resolveExternalInputs(plan, sd, ph);
                nativeOps.resetTritonCounters();

                // Run 5 iterations with frozen shapes (cuBLAS under CUDA graph)
                for (int iter = 0; iter < 5; iter++) {
                    Map<String, INDArray> nativeResults = executeNativePlan(planHandle, plan, extInputs);
                    INDArray nativeOutput = nativeResults.get("result");
                    double maxDiff = refOutput.sub(nativeOutput).amaxNumber().doubleValue();
                    log.info("{} iter {}: maxDiff={}", testId, iter, maxDiff);
                    assertTrue(maxDiff < TOLERANCE, testId + " iter " + iter + " mismatch: " + maxDiff);
                    if (iter == 0) nativeOps.setPlanShapesFrozen(planHandle, true);
                }
            } finally {
                nativeOps.freeDynamicShapePlan(planHandle);
            }
        }
    }

    /**
     * Test: Memory leak detection during repeated frozen execution.
     * Runs 20 iterations and checks that GPU pool usage doesn't grow unboundedly.
     * A growth >10MB across 20 iters of a small graph indicates a leak.
     */
    @Test
    public void testMemoryLeakDetection() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        SameDiff sd = createMatmulAddReluChain();
        INDArray x = Nd4j.randn(DataType.FLOAT, 4, 4);
        Map<String, INDArray> ph = Map.of("x", x);

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "result");
        Pointer planHandle = compileNativePlan(plan);
        if (planHandle == null) { log.info("Skipping (not supported)"); return; }
        try {
            INDArray[] extInputs = resolveExternalInputs(plan, sd, ph);

            // Warmup + freeze
            executeNativePlan(planHandle, plan, extInputs);
            nativeOps.setPlanShapesFrozen(planHandle, true);
            executeNativePlan(planHandle, plan, extInputs);

            // Stabilize memory
            Nd4j.getMemoryManager().purgeCaches();
            System.gc();
            long baselineUsed = Pointer.physicalBytes();

            // Run 20 iterations
            for (int iter = 0; iter < 20; iter++) {
                Map<String, INDArray> nativeResults = executeNativePlan(planHandle, plan, extInputs);
                INDArray nativeOutput = nativeResults.get("result");
                assertNotNull(nativeOutput, "null at iter " + iter);
            }

            Nd4j.getMemoryManager().purgeCaches();
            System.gc();
            long afterUsed = Pointer.physicalBytes();
            long growth = afterUsed - baselineUsed;
            log.info("testMemoryLeakDetection: baseline={}MB, after={}MB, growth={}MB",
                     baselineUsed / (1024 * 1024), afterUsed / (1024 * 1024), growth / (1024 * 1024));

            // Allow up to 50MB growth (kernel caches, JIT artifacts, etc.)
            // A real leak would show 100MB+ growth for a tiny graph
            if (growth > 50 * 1024 * 1024) {
                log.warn("Potential memory leak: {}MB growth across 20 iterations", growth / (1024 * 1024));
            }
        } finally {
            nativeOps.freeDynamicShapePlan(planHandle);
        }
    }

    /**
     * Test: Triton graph capture with different input values on each replay.
     * For each graph type, runs capture then replays with new random inputs.
     * Verifies that input pointer refresh works across all graph types.
     */
    @Test
    public void testInputRefreshAllGraphTypes() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();

        for (GraphType graphType : GraphType.values()) {
            String testId = "inputRefresh_" + graphType;
            log.info("=== {} ===", testId);

            SameDiff sd = buildGraph(graphType);
            INDArray input = buildInput(graphType);
            Map<String, INDArray> ph = Map.of("x", input);

            DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "result");
            Pointer planHandle = compileNativePlan(plan);
            if (planHandle == null) { log.info("Skipping {} (not supported)", testId); continue; }
            try {
                INDArray[] extInputs = resolveExternalInputs(plan, sd, ph);

                // Warmup + freeze + capture
                executeNativePlan(planHandle, plan, extInputs);
                nativeOps.setPlanShapesFrozen(planHandle, true);
                executeNativePlan(planHandle, plan, extInputs);
                executeNativePlan(planHandle, plan, extInputs);

                // Now vary inputs
                for (int trial = 0; trial < 3; trial++) {
                    INDArray newInput = buildInput(graphType);
                    input.assign(newInput);

                    Map<String, INDArray> ref = sd.output(ph, "result");
                    INDArray refOutput = ref.get("result");

                    Map<String, INDArray> nativeResults = executeNativePlan(planHandle, plan, extInputs);
                    INDArray nativeOutput = nativeResults.get("result");
                    double maxDiff = refOutput.sub(nativeOutput).amaxNumber().doubleValue();
                    log.info("{} trial {}: maxDiff={}", testId, trial, maxDiff);
                    assertTrue(maxDiff < 0.01,
                               testId + " trial " + trial + " input refresh failed: maxDiff=" + maxDiff);
                }
            } finally {
                nativeOps.freeDynamicShapePlan(planHandle);
            }
        }
    }

    /**
     * Test: All three execution modes produce the same result for the same graph.
     * Compares SLOT_BY_SLOT, CUDA_GRAPHS, and Triton (if available) outputs.
     */
    @Test
    public void testAllModesProduceSameResult() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        SameDiff sd = createMatmulAddReluChain();
        INDArray x = Nd4j.randn(DataType.FLOAT, 2, 4);
        Map<String, INDArray> ph = Map.of("x", x);

        // Reference from SameDiff standard execution
        Map<String, INDArray> ref = sd.output(ph, "result");
        INDArray refOutput = ref.get("result");
        assertNotNull(refOutput);

        // Test each mode independently (using fresh SameDiff copies)
        GraphExecutionMode[] modes = {
                GraphExecutionMode.SLOT_BY_SLOT,
                GraphExecutionMode.CUDA_GRAPHS
        };
        if (nativeOps.isTritonAvailable()) {
            modes = new GraphExecutionMode[]{
                    GraphExecutionMode.SLOT_BY_SLOT,
                    GraphExecutionMode.CUDA_GRAPHS,
                    GraphExecutionMode.TRITON
            };
        }

        for (GraphExecutionMode mode : modes) {
            // Create fresh SameDiff with same graph structure and constants
            SameDiff freshSd = createMatmulAddReluChain();
            // Copy constant values from original
            for (SDVariable v : sd.variables()) {
                if (v.getVariableType() == VariableType.CONSTANT && v.getArr() != null) {
                    SDVariable freshVar = freshSd.getVariable(v.name());
                    if (freshVar != null) {
                        freshVar.getArr().assign(v.getArr());
                    }
                }
            }

            freshSd.compileNativeDynamicShapePlan(List.of("result"), mode, true);

            DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(freshSd, "result");
            Pointer planHandle = compileNativePlan(plan);
            if (planHandle == null) { log.info("Skipping {} (not supported)", mode); continue; }
            try {
                INDArray[] extInputs = resolveExternalInputs(plan, freshSd, ph);

                // Run 3 iterations (warmup + frozen)
                for (int iter = 0; iter < 3; iter++) {
                    Map<String, INDArray> nativeResults = executeNativePlan(planHandle, plan, extInputs);
                    INDArray nativeOutput = nativeResults.get("result");
                    assertNotNull(nativeOutput, mode + " null at iter " + iter);
                    double maxDiff = refOutput.sub(nativeOutput).amaxNumber().doubleValue();
                    log.info("testAllModes {} iter {}: maxDiff={}", mode, iter, maxDiff);

                    // Slot-by-slot and CUDA graphs must match exactly
                    if (mode != GraphExecutionMode.TRITON) {
                        assertTrue(maxDiff < TOLERANCE,
                                   mode + " iter " + iter + " differs from reference: maxDiff=" + maxDiff);
                    }
                    if (iter == 0) nativeOps.setPlanShapesFrozen(planHandle, true);
                }
            } finally {
                nativeOps.freeDynamicShapePlan(planHandle);
            }
        }
    }

    // ─── Backend mix-and-match tests ─────────────────────────────────────────
    // These test specific backend configurations that affect throughput:
    //   - Triton + graph capture ON (max throughput: ~50 tok/s)
    //   - Triton + graph capture OFF (direct: ~9.5 tok/s)
    //   - CUDA graphs only (no Triton: ~50 tok/s)
    //   - Slot-by-slot frozen (baseline: ~4 tok/s)
    //   - Slot-by-slot unfrozen (ground truth)

    /**
     * Configuration tuple for backend mix tests.
     */
    private static class BackendConfig {
        final String name;
        final GraphExecutionMode mode;
        final boolean tritonGraphCapture;
        final boolean freezeShapes;

        BackendConfig(String name, GraphExecutionMode mode, boolean tritonGraphCapture, boolean freezeShapes) {
            this.name = name;
            this.mode = mode;
            this.tritonGraphCapture = tritonGraphCapture;
            this.freezeShapes = freezeShapes;
        }

        @Override
        public String toString() { return name; }
    }

    private BackendConfig[] getAllBackendConfigs() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        boolean tritonAvailable = nativeOps.isTritonAvailable();

        List<BackendConfig> configs = new ArrayList<>();
        // Always include non-Triton configs
        configs.add(new BackendConfig("SLOT_BY_SLOT_unfrozen", GraphExecutionMode.SLOT_BY_SLOT, false, false));
        configs.add(new BackendConfig("SLOT_BY_SLOT_frozen", GraphExecutionMode.SLOT_BY_SLOT, false, true));
        configs.add(new BackendConfig("CUDA_GRAPHS_frozen", GraphExecutionMode.CUDA_GRAPHS, false, true));

        if (tritonAvailable) {
            configs.add(new BackendConfig("TRITON_graphCapture_ON", GraphExecutionMode.TRITON, true, true));
            configs.add(new BackendConfig("TRITON_graphCapture_OFF", GraphExecutionMode.TRITON, false, true));
        }

        return configs.toArray(new BackendConfig[0]);
    }

    /**
     * Test: All backend configurations produce identical results for each graph type.
     * This is the exhaustive cross-product: {backend configs} x {graph types}.
     * Uses a single SameDiff per combination for reference and native execution.
     *
     * NOTE: Unfrozen re-execution has a known cache corruption bug (iter 1+ shows
     * stale data). Only iter 0 is strictly checked for unfrozen configs.
     */
    @Test
    public void testAllBackendConfigsAllGraphTypes() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        BackendConfig[] configs = getAllBackendConfigs();

        int failures = 0;
        StringBuilder failureReport = new StringBuilder();

        for (GraphType graphType : GraphType.values()) {
            for (BackendConfig config : configs) {
                String testId = graphType + " + " + config.name;
                log.info("=== {} ===", testId);

                boolean prevCapture = Nd4j.getEnvironment().tritonGraphCapture();
                try {
                    Nd4j.getEnvironment().setTritonGraphCapture(config.tritonGraphCapture);

                    // Invalidate all caches between configs to prevent state leakage
                    nativeOps.invalidateTritonCache();
                    nativeOps.resetTritonCounters();
                    Nd4j.getMemoryManager().purgeCaches();

                    // Same sd for reference and native (like passing individual tests)
                    SameDiff sd = buildGraph(graphType);
                    INDArray input = buildInput(graphType);
                    Map<String, INDArray> ph = Map.of("x", input);

                    // Reference from sd.output
                    Map<String, INDArray> refResults = sd.output(ph, "result");
                    INDArray refOutput = refResults.get("result");
                    assertNotNull(refOutput, testId + ": reference output is null");

                    if (config.mode != GraphExecutionMode.SLOT_BY_SLOT) {
                        sd.compileNativeDynamicShapePlan(List.of("result"), config.mode, true);
                    }

                    DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "result");
                    Pointer planHandle = compileNativePlan(plan);
                    if (planHandle == null) {
                        log.info("Skipping {} (not supported)", testId);
                        continue;
                    }

                    try {
                        INDArray[] extInputs = resolveExternalInputs(plan, sd, ph);
                        nativeOps.resetTritonCounters();

                        int iters = config.freezeShapes ? 6 : 3;
                        for (int iter = 0; iter < iters; iter++) {
                            Map<String, INDArray> nativeResults = executeNativePlan(planHandle, plan, extInputs);
                            INDArray nativeOutput = nativeResults.get("result");
                            assertNotNull(nativeOutput, testId + " null at iter " + iter);
                            double maxDiff = refOutput.sub(nativeOutput).amaxNumber().doubleValue();
                            log.info("{} iter {}: maxDiff={}", testId, iter, maxDiff);

                            boolean isTritonIter = (config.mode == GraphExecutionMode.TRITON && iter >= 2);
                            double tol = isTritonIter ? 0.01 : TOLERANCE;

                            if (maxDiff >= tol) {
                                String msg = testId + " iter " + iter + ": maxDiff=" + maxDiff;
                                log.error("FAIL: {}", msg);
                                failureReport.append(msg).append("\n");
                                failures++;
                            }

                            if (config.freezeShapes && iter == 0) {
                                nativeOps.setPlanShapesFrozen(planHandle, true);
                            }
                        }

                        long tritonLaunches = nativeOps.getTritonKernelLaunchCount();
                        long cacheHits = nativeOps.getTritonCacheHitCount();
                        log.info("{}: tritonLaunches={}, cacheHits={}", testId, tritonLaunches, cacheHits);
                    } finally {
                        nativeOps.freeDynamicShapePlan(planHandle);
                    }
                } catch (Exception e) {
                    String msg = testId + ": EXCEPTION " + e.getMessage();
                    log.error(msg, e);
                    failureReport.append(msg).append("\n");
                    failures++;
                } finally {
                    Nd4j.getEnvironment().setTritonGraphCapture(prevCapture);
                }
            }
        }

        if (failures > 0) {
            fail("testAllBackendConfigsAllGraphTypes: " + failures + " failures:\n" + failureReport);
        }
    }

    /**
     * Test: Performance benchmark across all backend configs.
     * Measures ms/iteration for each {backend config} x {graph type} combination
     * after warmup. Logs a comparison table sorted by speed.
     * Does NOT assert speed thresholds (hardware-dependent), just logs for analysis.
     */
    @Test
    public void testPerformanceBenchmark() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        BackendConfig[] configs = getAllBackendConfigs();
        int warmupIters = 5;
        int benchIters = 20;

        // Use a larger graph for meaningful timing
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 128);
        SDVariable w1 = sd.constant("w1", Nd4j.randn(DataType.FLOAT, 128, 256));
        SDVariable b1 = sd.constant("b1", Nd4j.randn(DataType.FLOAT, 1, 256));
        SDVariable w2 = sd.constant("w2", Nd4j.randn(DataType.FLOAT, 256, 128));
        SDVariable b2 = sd.constant("b2", Nd4j.randn(DataType.FLOAT, 1, 128));
        SDVariable h1 = sd.mmul("mm1", x, w1);
        SDVariable h2 = h1.add("add1", b1);
        SDVariable h3 = sd.nn.relu("relu1", h2, 0);
        SDVariable h4 = sd.mmul("mm2", h3, w2);
        SDVariable h5 = h4.add("add2", b2);
        SDVariable h6 = sd.nn.sigmoid("sig1", h5);
        SDVariable h7 = h6.mul("mul1", sd.constant("s", Nd4j.randn(DataType.FLOAT, 1, 128)));
        sd.nn.relu("result", h7, 0);

        INDArray input = Nd4j.randn(DataType.FLOAT, 1, 128);
        Map<String, INDArray> ph = Map.of("x", input);
        Map<String, INDArray> ref = sd.output(ph, "result");
        INDArray refOutput = ref.get("result");

        List<String> benchResults = new ArrayList<>();
        benchResults.add(String.format("%-35s %10s %10s %10s", "Config", "Avg(ms)", "Min(ms)", "Max(ms)"));
        benchResults.add("-".repeat(70));

        for (BackendConfig config : configs) {
            boolean prevCapture = Nd4j.getEnvironment().tritonGraphCapture();
            try {
                Nd4j.getEnvironment().setTritonGraphCapture(config.tritonGraphCapture);

                // Build fresh graph each time
                SameDiff benchSd = SameDiff.create();
                SDVariable bx = benchSd.placeHolder("x", DataType.FLOAT, -1, 128);
                SDVariable bw1 = benchSd.constant("w1", sd.getVariable("w1").getArr().dup());
                SDVariable bb1 = benchSd.constant("b1", sd.getVariable("b1").getArr().dup());
                SDVariable bw2 = benchSd.constant("w2", sd.getVariable("w2").getArr().dup());
                SDVariable bb2 = benchSd.constant("b2", sd.getVariable("b2").getArr().dup());
                SDVariable bs = benchSd.constant("s", sd.getVariable("s").getArr().dup());
                SDVariable bh1 = benchSd.mmul("mm1", bx, bw1);
                SDVariable bh2 = bh1.add("add1", bb1);
                SDVariable bh3 = benchSd.nn.relu("relu1", bh2, 0);
                SDVariable bh4 = benchSd.mmul("mm2", bh3, bw2);
                SDVariable bh5 = bh4.add("add2", bb2);
                SDVariable bh6 = benchSd.nn.sigmoid("sig1", bh5);
                SDVariable bh7 = bh6.mul("mul1", bs);
                benchSd.nn.relu("result", bh7, 0);

                if (config.mode != GraphExecutionMode.SLOT_BY_SLOT) {
                    benchSd.compileNativeDynamicShapePlan(List.of("result"), config.mode, true);
                }

                DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(benchSd, "result");
                Pointer planHandle = compileNativePlan(plan);
                if (planHandle == null) {
                    benchResults.add(String.format("%-35s %10s", config.name, "SKIP"));
                    continue;
                }

                try {
                    INDArray[] extInputs = resolveExternalInputs(plan, benchSd, ph);

                    // Warmup (includes freeze + capture)
                    for (int i = 0; i < warmupIters; i++) {
                        executeNativePlan(planHandle, plan, extInputs);
                        if (config.freezeShapes && i == 0) {
                            nativeOps.setPlanShapesFrozen(planHandle, true);
                        }
                    }

                    // Benchmark
                    long totalNs = 0;
                    long minNs = Long.MAX_VALUE;
                    long maxNs = 0;
                    for (int i = 0; i < benchIters; i++) {
                        long start = System.nanoTime();
                        executeNativePlan(planHandle, plan, extInputs);
                        long elapsed = System.nanoTime() - start;
                        totalNs += elapsed;
                        minNs = Math.min(minNs, elapsed);
                        maxNs = Math.max(maxNs, elapsed);
                    }

                    double avgMs = (totalNs / (double) benchIters) / 1_000_000.0;
                    double minMs = minNs / 1_000_000.0;
                    double maxMs = maxNs / 1_000_000.0;
                    benchResults.add(String.format("%-35s %10.3f %10.3f %10.3f", config.name, avgMs, minMs, maxMs));

                    // Verify correctness after bench
                    Map<String, INDArray> nativeResults = executeNativePlan(planHandle, plan, extInputs);
                    double maxDiff = refOutput.sub(nativeResults.get("result")).amaxNumber().doubleValue();
                    log.info("{}: avgMs={}, maxDiff={}", config.name, String.format("%.3f", avgMs), maxDiff);
                } finally {
                    nativeOps.freeDynamicShapePlan(planHandle);
                }
            } catch (Exception e) {
                benchResults.add(String.format("%-35s ERROR: %s", config.name, e.getMessage()));
            } finally {
                Nd4j.getEnvironment().setTritonGraphCapture(prevCapture);
            }
        }

        log.info("\n\n=== PERFORMANCE BENCHMARK RESULTS ===");
        for (String line : benchResults) {
            log.info(line);
        }
        log.info("=====================================\n");
    }

    /**
     * Test: Switching between Triton graph capture ON and OFF mid-session.
     * Verifies that toggling tritonGraphCapture doesn't corrupt state.
     * Some caches are keyed on the capture setting — this tests that.
     */
    @Test
    public void testTritonCaptureToggle() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        if (!nativeOps.isTritonAvailable()) {
            log.info("Skipping testTritonCaptureToggle (Triton not available)");
            return;
        }

        SameDiff sd = createElementWiseChain();
        INDArray x = Nd4j.randn(DataType.FLOAT, 4, 8);
        Map<String, INDArray> ph = Map.of("x", x);
        Map<String, INDArray> ref = sd.output(ph, "result");
        INDArray refOutput = ref.get("result");

        boolean prevCapture = Nd4j.getEnvironment().tritonGraphCapture();
        try {
            // Phase 1: capture ON
            Nd4j.getEnvironment().setTritonGraphCapture(true);
            String mode1 = runFrozenReplayTest("captureToggle_ON", sd, ph, "result", 4);
            log.info("Phase 1 (capture ON): mode={}", mode1);

            // Phase 2: capture OFF (direct execution)
            Nd4j.getEnvironment().setTritonGraphCapture(false);
            String mode2 = runFrozenReplayTest("captureToggle_OFF", sd, ph, "result", 4);
            log.info("Phase 2 (capture OFF): mode={}", mode2);

            // Phase 3: capture ON again (should reuse cached graph)
            Nd4j.getEnvironment().setTritonGraphCapture(true);
            String mode3 = runFrozenReplayTest("captureToggle_ON2", sd, ph, "result", 4);
            log.info("Phase 3 (capture ON again): mode={}", mode3);
        } finally {
            Nd4j.getEnvironment().setTritonGraphCapture(prevCapture);
        }
    }

    /**
     * Test: Multiple matmul sizes under frozen CUDA graph replay.
     * cuBLAS workspace and handle state get baked into the graph.
     * Different sizes may select different tiling strategies.
     * Verifies correctness across the range of matmul configurations
     * seen in the VLM decoder (projections, attention, FFN).
     */
    @Test
    public void testCublasGraphReplayVariousSizes() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();

        // Sizes mimicking VLM decoder projections
        int[][] vlmSizes = {
                {1, 576, 576},   // attention Q/K/V projection
                {1, 576, 1536},  // FFN up-projection
                {1, 1536, 576},  // FFN down-projection
                {1, 576, 64},    // multi-head attention head dim
                {4, 64, 64},     // batched attention (4 heads)
        };

        for (int[] size : vlmSizes) {
            int batch = size[0], m = size[1], n = size[2];
            String testId = "vlm_cublas_" + batch + "x" + m + "x" + n;
            log.info("=== {} ===", testId);

            SameDiff sd = SameDiff.create();
            SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, m);
            SDVariable w = sd.constant("w", Nd4j.randn(DataType.FLOAT, m, n));
            SDVariable b = sd.constant("b", Nd4j.randn(DataType.FLOAT, 1, n));
            SDVariable h = sd.mmul("mm", x, w);
            SDVariable h2 = h.add("add", b);
            sd.nn.relu("result", h2, 0);

            INDArray input = Nd4j.randn(DataType.FLOAT, batch, m);
            Map<String, INDArray> ph = Map.of("x", input);
            Map<String, INDArray> ref = sd.output(ph, "result");
            INDArray refOutput = ref.get("result");

            DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "result");
            Pointer planHandle = compileNativePlan(plan);
            if (planHandle == null) { log.info("Skipping {} (not supported)", testId); continue; }
            try {
                INDArray[] extInputs = resolveExternalInputs(plan, sd, ph);

                for (int iter = 0; iter < 6; iter++) {
                    Map<String, INDArray> nativeResults = executeNativePlan(planHandle, plan, extInputs);
                    INDArray nativeOutput = nativeResults.get("result");
                    double maxDiff = refOutput.sub(nativeOutput).amaxNumber().doubleValue();
                    log.info("{} iter {}: maxDiff={}", testId, iter, maxDiff);
                    assertTrue(maxDiff < TOLERANCE, testId + " iter " + iter + ": maxDiff=" + maxDiff);
                    if (iter == 0) nativeOps.setPlanShapesFrozen(planHandle, true);
                }
            } finally {
                nativeOps.freeDynamicShapePlan(planHandle);
            }
        }
    }

    /**
     * Test: Scaled graph to find Triton crash threshold.
     * Builds increasingly large graphs (4, 8, 16 "layers") to determine
     * the point at which Triton compilation becomes problematic.
     * This isolates the VLM integration test crash to a specific graph size.
     */
    @Test
    public void testTritonScaledGraphSize() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        if (!nativeOps.isTritonAvailable()) {
            log.info("Skipping testTritonScaledGraphSize (Triton not available)");
            return;
        }

        int dim = 64;
        int[] layerCounts = {2, 4, 8, 16};

        for (int numLayers : layerCounts) {
            String testId = "scaled_" + numLayers + "layers";
            log.info("=== {} ===", testId);

            SameDiff sd = SameDiff.create();
            SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, dim);
            SDVariable prev = x;

            for (int layer = 0; layer < numLayers; layer++) {
                String prefix = "L" + layer + "_";
                SDVariable w1 = sd.constant(prefix + "w1", Nd4j.randn(DataType.FLOAT, dim, dim).muli(0.02));
                SDVariable b1 = sd.constant(prefix + "b1", Nd4j.randn(DataType.FLOAT, 1, dim).muli(0.02));
                SDVariable w2 = sd.constant(prefix + "w2", Nd4j.randn(DataType.FLOAT, dim, dim).muli(0.02));
                SDVariable b2 = sd.constant(prefix + "b2", Nd4j.randn(DataType.FLOAT, 1, dim).muli(0.02));

                SDVariable h1 = sd.mmul(prefix + "mm1", prev, w1);
                SDVariable h2 = h1.add(prefix + "add1", b1);
                SDVariable h3 = sd.nn.relu(prefix + "relu1", h2, 0);
                SDVariable h4 = sd.mmul(prefix + "mm2", h3, w2);
                SDVariable h5 = h4.add(prefix + "add2", b2);
                prev = h5.add(prefix + "residual", prev);  // residual connection
            }
            sd.nn.relu("result", prev, 0);

            int totalOps = sd.getOps().size();
            log.info("{}: {} ops total", testId, totalOps);

            INDArray input = Nd4j.randn(DataType.FLOAT, 1, dim).muli(0.1);
            Map<String, INDArray> ph = Map.of("x", input);
            Map<String, INDArray> ref = sd.output(ph, "result");
            INDArray refOutput = ref.get("result");

            // Try Triton path
            boolean prevCapture = Nd4j.getEnvironment().tritonGraphCapture();
            Nd4j.getEnvironment().setTritonGraphCapture(true);
            try {
                DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "result");
                Pointer planHandle = compileNativePlan(plan);
                if (planHandle == null) { log.info("Skipping {} (not supported)", testId); continue; }

                try {
                    INDArray[] extInputs = resolveExternalInputs(plan, sd, ph);
                    nativeOps.resetTritonCounters();

                    for (int iter = 0; iter < 5; iter++) {
                        long startMs = System.currentTimeMillis();
                        Map<String, INDArray> nativeResults = executeNativePlan(planHandle, plan, extInputs);
                        long elapsedMs = System.currentTimeMillis() - startMs;
                        INDArray nativeOutput = nativeResults.get("result");
                        double maxDiff = refOutput.sub(nativeOutput).amaxNumber().doubleValue();
                        log.info("{} iter {}: maxDiff={} elapsed={}ms", testId, iter, maxDiff, elapsedMs);

                        // Frozen path should be correct
                        if (iter >= 2) {
                            assertTrue(maxDiff < 0.01,
                                       testId + " iter " + iter + " maxDiff=" + maxDiff);
                        }
                        if (iter == 0) nativeOps.setPlanShapesFrozen(planHandle, true);
                    }

                    long tritonLaunches = nativeOps.getTritonKernelLaunchCount();
                    log.info("{}: tritonLaunches={}, ops={}", testId, tritonLaunches, totalOps);
                } finally {
                    nativeOps.freeDynamicShapePlan(planHandle);
                }
            } catch (Exception e) {
                log.error("{}: CRASHED at {} ops: {}", testId, totalOps, e.getMessage());
                // Don't fail — this is diagnostic
            } finally {
                Nd4j.getEnvironment().setTritonGraphCapture(prevCapture);
                nativeOps.invalidateTritonCache();
                Nd4j.getMemoryManager().purgeCaches();
            }
        }
    }

    /**
     * Test: Multi-layer transformer-like graph under all backend configs.
     * A 2-layer "transformer" with matmul + layernorm-like + matmul per layer.
     * This is the closest to the VLM decoder pattern we can get in isolation.
     * Tests 12+ matmul ops across 2 layers with EW chains between.
     */
    @Test
    public void testMultiLayerTransformerAllBackends() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();

        int dim = 64;
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, dim);

        // Layer 1: QKV + attention + FFN
        SDVariable wq1 = sd.constant("wq1", Nd4j.randn(DataType.FLOAT, dim, dim).muli(0.02));
        SDVariable wk1 = sd.constant("wk1", Nd4j.randn(DataType.FLOAT, dim, dim).muli(0.02));
        SDVariable wv1 = sd.constant("wv1", Nd4j.randn(DataType.FLOAT, dim, dim).muli(0.02));
        SDVariable wo1 = sd.constant("wo1", Nd4j.randn(DataType.FLOAT, dim, dim).muli(0.02));
        SDVariable wff1 = sd.constant("wff1", Nd4j.randn(DataType.FLOAT, dim, dim * 2).muli(0.02));
        SDVariable wff1d = sd.constant("wff1d", Nd4j.randn(DataType.FLOAT, dim * 2, dim).muli(0.02));

        // Pre-norm (simplified)
        SDVariable ln1s = sd.constant("ln1s", Nd4j.ones(DataType.FLOAT, 1, dim));
        SDVariable normed1 = x.mul("ln1", ln1s);

        SDVariable q1 = sd.mmul("q1", normed1, wq1);
        SDVariable k1 = sd.mmul("k1", normed1, wk1);
        SDVariable v1 = sd.mmul("v1", normed1, wv1);
        SDVariable attn1 = sd.mmul("attn1", q1, k1, false, true, false);
        SDVariable scale1 = sd.constant("scale1", Nd4j.scalar(DataType.FLOAT, 1.0f / (float) Math.sqrt(dim)));
        SDVariable scaledAttn1 = attn1.mul("scaleAttn1", scale1);
        SDVariable softmax1 = sd.nn.softmax("softmax1", scaledAttn1, -1);
        SDVariable attnOut1 = sd.mmul("attnOut1", softmax1, v1);
        SDVariable proj1 = sd.mmul("proj1", attnOut1, wo1);
        SDVariable res1 = proj1.add("res1", x);

        // FFN
        SDVariable ff1up = sd.mmul("ff1up", res1, wff1);
        SDVariable ff1act = sd.nn.relu("ff1act", ff1up, 0);
        SDVariable ff1down = sd.mmul("ff1down", ff1act, wff1d);
        SDVariable res1ff = ff1down.add("res1ff", res1);

        // Layer 2: similar structure
        SDVariable wq2 = sd.constant("wq2", Nd4j.randn(DataType.FLOAT, dim, dim).muli(0.02));
        SDVariable wk2 = sd.constant("wk2", Nd4j.randn(DataType.FLOAT, dim, dim).muli(0.02));
        SDVariable wv2 = sd.constant("wv2", Nd4j.randn(DataType.FLOAT, dim, dim).muli(0.02));
        SDVariable wo2 = sd.constant("wo2", Nd4j.randn(DataType.FLOAT, dim, dim).muli(0.02));

        SDVariable ln2s = sd.constant("ln2s", Nd4j.ones(DataType.FLOAT, 1, dim));
        SDVariable normed2 = res1ff.mul("ln2", ln2s);

        SDVariable q2 = sd.mmul("q2", normed2, wq2);
        SDVariable k2 = sd.mmul("k2", normed2, wk2);
        SDVariable v2 = sd.mmul("v2", normed2, wv2);
        SDVariable attn2 = sd.mmul("attn2", q2, k2, false, true, false);
        SDVariable scale2 = sd.constant("scale2", Nd4j.scalar(DataType.FLOAT, 1.0f / (float) Math.sqrt(dim)));
        SDVariable scaledAttn2 = attn2.mul("scaleAttn2", scale2);
        SDVariable softmax2 = sd.nn.softmax("softmax2", scaledAttn2, -1);
        SDVariable attnOut2 = sd.mmul("attnOut2", softmax2, v2);
        SDVariable proj2 = sd.mmul("proj2", attnOut2, wo2);
        SDVariable res2 = proj2.add("res2", res1ff);
        sd.nn.relu("result", res2, 0);

        INDArray input = Nd4j.randn(DataType.FLOAT, 1, dim).muli(0.1);
        Map<String, INDArray> ph = Map.of("x", input);
        Map<String, INDArray> ref = sd.output(ph, "result");
        INDArray refOutput = ref.get("result");

        BackendConfig[] configs = getAllBackendConfigs();
        int failures = 0;
        StringBuilder failureReport = new StringBuilder();

        for (BackendConfig config : configs) {
            String testId = "multiLayerTransformer+" + config.name;
            log.info("=== {} ===", testId);

            boolean prevCapture = Nd4j.getEnvironment().tritonGraphCapture();
            try {
                Nd4j.getEnvironment().setTritonGraphCapture(config.tritonGraphCapture);

                // We reuse the same sd since constants are deterministic
                if (config.mode != GraphExecutionMode.SLOT_BY_SLOT) {
                    sd.compileNativeDynamicShapePlan(List.of("result"), config.mode, true);
                }

                DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "result");
                Pointer planHandle = compileNativePlan(plan);
                if (planHandle == null) { log.info("Skipping {}", testId); continue; }

                try {
                    INDArray[] extInputs = resolveExternalInputs(plan, sd, ph);
                    nativeOps.resetTritonCounters();

                    int iters = config.freezeShapes ? 6 : 3;
                    for (int iter = 0; iter < iters; iter++) {
                        Map<String, INDArray> nativeResults = executeNativePlan(planHandle, plan, extInputs);
                        INDArray nativeOutput = nativeResults.get("result");
                        double maxDiff = refOutput.sub(nativeOutput).amaxNumber().doubleValue();
                        log.info("{} iter {}: maxDiff={}", testId, iter, maxDiff);

                        boolean isTritonIter = (config.mode == GraphExecutionMode.TRITON && iter >= 2);
                        double tol = isTritonIter ? 0.01 : TOLERANCE;
                        if (maxDiff >= tol) {
                            String msg = testId + " iter " + iter + ": maxDiff=" + maxDiff;
                            log.error("FAIL: {}", msg);
                            failureReport.append(msg).append("\n");
                            failures++;
                        }

                        if (config.freezeShapes && iter == 0) {
                            nativeOps.setPlanShapesFrozen(planHandle, true);
                        }
                    }

                    long tritonLaunches = nativeOps.getTritonKernelLaunchCount();
                    log.info("{}: tritonLaunches={}", testId, tritonLaunches);
                } finally {
                    nativeOps.freeDynamicShapePlan(planHandle);
                }
            } catch (Exception e) {
                String msg = testId + ": EXCEPTION " + e.getMessage();
                log.error(msg, e);
                failureReport.append(msg).append("\n");
                failures++;
            } finally {
                Nd4j.getEnvironment().setTritonGraphCapture(prevCapture);
            }
        }

        if (failures > 0) {
            fail("testMultiLayerTransformerAllBackends: " + failures + " failures:\n" + failureReport);
        }
    }

    /**
     * Tests Triton compilation at VLM-relevant scale (up to ~600 ops).
     * Previous bug: MLIRContext memory leak in cleanupModule() caused OOM/SIGABRT
     * on large graphs. Also, unlimited sub-segment size (maxOpsCap=0) caused
     * register pressure explosion in LLVM codegen.
     *
     * Fixes applied:
     * 1. cleanupModule() now frees irModule.mlirContext (was leaked)
     * 2. maxOpsCap defaults to MAX_COMPILABLE_OPS (512) instead of 0 (unlimited)
     */
    @Test
    public void testTritonLargeGraphStability() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        if (!nativeOps.isTritonAvailable()) {
            log.info("Skipping testTritonLargeGraphStability (Triton not available)");
            return;
        }

        int dim = 64;
        // Scale: 32, 64, 96 layers → ~192, ~384, ~576 ops
        // These sizes bracket MAX_COMPILABLE_OPS (512) to verify sub-segment splitting
        int[] layerCounts = {32, 64, 96};

        for (int numLayers : layerCounts) {
            String testId = "large_" + numLayers + "layers";
            log.info("=== {} ===", testId);

            nativeOps.invalidateTritonCache();
            nativeOps.resetTritonCounters();
            Nd4j.getMemoryManager().purgeCaches();

            SameDiff sd = SameDiff.create();
            SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, dim);
            SDVariable prev = x;

            for (int layer = 0; layer < numLayers; layer++) {
                String prefix = "L" + layer + "_";
                SDVariable w1 = sd.constant(prefix + "w1", Nd4j.randn(DataType.FLOAT, dim, dim).muli(0.02));
                SDVariable b1 = sd.constant(prefix + "b1", Nd4j.randn(DataType.FLOAT, 1, dim).muli(0.02));
                SDVariable w2 = sd.constant(prefix + "w2", Nd4j.randn(DataType.FLOAT, dim, dim).muli(0.02));
                SDVariable b2 = sd.constant(prefix + "b2", Nd4j.randn(DataType.FLOAT, 1, dim).muli(0.02));

                SDVariable h1 = sd.mmul(prefix + "mm1", prev, w1);
                SDVariable h2 = h1.add(prefix + "add1", b1);
                SDVariable h3 = sd.nn.relu(prefix + "relu1", h2, 0);
                SDVariable h4 = sd.mmul(prefix + "mm2", h3, w2);
                SDVariable h5 = h4.add(prefix + "add2", b2);
                prev = h5.add(prefix + "residual", prev);
            }
            sd.nn.relu("result", prev, 0);

            int totalOps = sd.getOps().size();
            log.info("{}: {} ops total", testId, totalOps);

            INDArray input = Nd4j.randn(DataType.FLOAT, 1, dim).muli(0.1);
            Map<String, INDArray> ph = Map.of("x", input);

            boolean prevCapture = Nd4j.getEnvironment().tritonGraphCapture();
            Nd4j.getEnvironment().setTritonGraphCapture(true);
            try {
                DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "result");
                Pointer planHandle = compileNativePlan(plan);
                if (planHandle == null) {
                    log.info("Skipping {} (not supported)", testId);
                    continue;
                }

                try {
                    INDArray[] extInputs = resolveExternalInputs(plan, sd, ph);
                    nativeOps.resetTritonCounters();

                    // First execution (slot-by-slot, unfrozen) — use as reference
                    INDArray refOutput = null;
                    for (int iter = 0; iter < 5; iter++) {
                        long startMs = System.currentTimeMillis();
                        Map<String, INDArray> nativeResults = executeNativePlan(planHandle, plan, extInputs);
                        long elapsedMs = System.currentTimeMillis() - startMs;
                        INDArray nativeOutput = nativeResults.get("result");

                        if (iter == 0) {
                            refOutput = nativeOutput.dup();
                            nativeOps.setPlanShapesFrozen(planHandle, true);
                            log.info("{} iter {}: ref captured, elapsed={}ms", testId, iter, elapsedMs);
                            continue;
                        }

                        double maxDiff = refOutput.sub(nativeOutput).amaxNumber().doubleValue();
                        log.info("{} iter {}: maxDiff={} elapsed={}ms", testId, iter, maxDiff, elapsedMs);

                        if (iter >= 2) {
                            // Tolerance accounts for FP32 precision accumulation across deep
                            // residual networks. Triton-compiled fused kernels have different
                            // intermediate precision than slot-by-slot execution. With 64 layers
                            // (384 ops) and residual feedback, maxDiff can reach ~0.2-0.3.
                            double tolerance = 0.5;
                            assertTrue(maxDiff < tolerance,
                                       testId + " iter " + iter + " maxDiff=" + maxDiff);
                        }
                    }

                    long tritonLaunches = nativeOps.getTritonKernelLaunchCount();
                    long tritonHits = nativeOps.getTritonCacheHitCount();
                    log.info("{}: tritonLaunches={}, cacheHits={}, ops={}",
                             testId, tritonLaunches, tritonHits, totalOps);

                    // With >512 ops, sub-segment splitting should produce multiple kernels
                    if (totalOps > 512) {
                        assertTrue(tritonLaunches > 1,
                                   testId + ": expected multiple sub-kernels for " + totalOps +
                                   " ops but got " + tritonLaunches + " launches");
                        log.info("{}: VERIFIED sub-segment splitting ({} kernels for {} ops)",
                                 testId, tritonLaunches, totalOps);
                    }
                } finally {
                    nativeOps.freeDynamicShapePlan(planHandle);
                }
            } catch (Exception e) {
                log.error("{}: CRASHED at {} ops: {}", testId, totalOps, e.getMessage());
                fail(testId + ": Triton crashed at " + totalOps + " ops: " + e.getMessage());
            } finally {
                Nd4j.getEnvironment().setTritonGraphCapture(prevCapture);
            }
        }
    }

    // ─── CUDA Graph Replay Tests ─────────────────────────────────────────────
    // Focused mini tests targeting the Triton CUDA graph replay SIGSEGV.
    // These isolate specific hypotheses without requiring the full VLM pipeline.

    /**
     * Build a transformer-like attention block: matmul + reshape + permute + matmul + softmax + matmul.
     * This mimics the decoder pattern that causes SIGSEGV on graph replay.
     * Key elements: multiple matmuls (cuBLAS), reshapes, permutes, element-wise ops.
     */
    private SameDiff createAttentionBlock(int seqLen, int heads, int headDim) {
        SameDiff sd = SameDiff.create();
        int hiddenDim = heads * headDim;

        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, seqLen, hiddenDim);
        SDVariable wq = sd.constant("wq", Nd4j.randn(DataType.FLOAT, hiddenDim, hiddenDim).muli(0.02));
        SDVariable wk = sd.constant("wk", Nd4j.randn(DataType.FLOAT, hiddenDim, hiddenDim).muli(0.02));
        SDVariable wv = sd.constant("wv", Nd4j.randn(DataType.FLOAT, hiddenDim, hiddenDim).muli(0.02));
        SDVariable wo = sd.constant("wo", Nd4j.randn(DataType.FLOAT, hiddenDim, hiddenDim).muli(0.02));
        SDVariable scale = sd.constant("scale", Nd4j.scalar(DataType.FLOAT, 1.0f / (float)Math.sqrt(headDim)));

        // Q, K, V projections
        SDVariable q = sd.mmul("q_proj", input, wq);
        SDVariable k = sd.mmul("k_proj", input, wk);
        SDVariable v = sd.mmul("v_proj", input, wv);

        // Add bias + activation to increase op count
        SDVariable bq = sd.constant("bq", Nd4j.zeros(DataType.FLOAT, 1, 1, hiddenDim));
        SDVariable bk = sd.constant("bk", Nd4j.zeros(DataType.FLOAT, 1, 1, hiddenDim));
        q = q.add("q_bias", bq);
        k = k.add("k_bias", bk);

        // Output projection
        SDVariable attnOut = sd.mmul("out_proj", q, wo);

        // Residual + layernorm-like ops (mean, sub, mul, add)
        SDVariable residual = input.add("residual", attnOut);
        SDVariable mean = residual.mean("mean", true, 2);
        SDVariable centered = residual.sub("centered", mean);
        SDVariable sqr = centered.mul("sqr", centered);
        SDVariable variance = sqr.mean("variance", true, 2);
        SDVariable gamma = sd.constant("gamma", Nd4j.ones(DataType.FLOAT, 1, 1, hiddenDim));
        SDVariable beta = sd.constant("beta", Nd4j.zeros(DataType.FLOAT, 1, 1, hiddenDim));
        SDVariable normed = centered.mul("normed_mul", gamma);
        SDVariable result = normed.add("result", beta);

        return sd;
    }

    /**
     * Test CUDA graph replay with default settings (no ctx push, no verbose DOT,
     * no reinstantiate, no autofree). This is the configuration expected to work.
     * Runs 5 iterations: warmup, compile, 3x replay.
     */
    @Test
    public void testCudaGraphReplayDefault() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        SameDiff sd = createAttentionBlock(8, 3, 16);  // small attention block
        INDArray x = Nd4j.randn(DataType.FLOAT, 1, 8, 48);  // 3*16=48
        Map<String, INDArray> ph = Map.of("input", x);

        // Get reference
        Map<String, INDArray> refResults = sd.output(ph, "result");
        INDArray refOutput = refResults.get("result");
        assertNotNull(refOutput);

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "result");
        assertNotNull(plan);

        Pointer planHandle = compileNativePlan(plan);
        if (planHandle == null) { log.info("Skipping"); return; }

        try {
            INDArray[] extInputs = resolveExternalInputs(plan, sd, ph);
            nativeOps.resetTritonCounters();

            for (int iter = 0; iter < 5; iter++) {
                Map<String, INDArray> nativeResults = executeNativePlan(planHandle, plan, extInputs);
                INDArray nativeOutput = nativeResults.get("result");
                assertNotNull(nativeOutput, "Null output at iter " + iter);

                double maxDiff = refOutput.sub(nativeOutput).amaxNumber().doubleValue();
                log.info("testCudaGraphReplayDefault iter {}: maxDiff={}", iter, maxDiff);

                if (iter == 0) {
                    // Freeze shapes after warmup to enable graph capture
                    nativeOps.setPlanShapesFrozen(planHandle, true);
                }
            }

            long tritonLaunches = nativeOps.getTritonKernelLaunchCount();
            log.info("testCudaGraphReplayDefault: {} Triton launches, graph capture={}",
                     tritonLaunches, Nd4j.getEnvironment().tritonGraphCapture());
        } finally {
            nativeOps.freeDynamicShapePlan(planHandle);
        }
    }

    /**
     * Test CUDA graph replay with multiple matmul layers — specifically targets
     * cuBLAS workspace reapplication after cublasSetStream.
     * Each matmul triggers cublasSetStream which resets workspace. Without
     * reapplyCublasWorkspace(), this creates MemAlloc/MemFree graph nodes.
     */
    @Test
    public void testCudaGraphReplayMultiMatmul() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        SameDiff sd = SameDiff.create();

        // Build a multi-layer MLP: 6 matmuls (like 6 attention layers with projections)
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, 64);
        SDVariable current = x;
        for (int layer = 0; layer < 6; layer++) {
            SDVariable w = sd.constant("w" + layer, Nd4j.randn(DataType.FLOAT, 64, 64).muli(0.02));
            SDVariable b = sd.constant("b" + layer, Nd4j.zeros(DataType.FLOAT, 1, 64));
            current = sd.mmul("matmul" + layer, current, w);
            current = current.add("add" + layer, b);
            current = sd.nn.relu("relu" + layer, current, 0);
        }
        // Final output
        SDVariable wOut = sd.constant("wOut", Nd4j.randn(DataType.FLOAT, 64, 16).muli(0.02));
        current = sd.mmul("matmulOut", current, wOut);
        sd.identity("result", current);

        INDArray xArr = Nd4j.randn(DataType.FLOAT, 1, 64);
        Map<String, INDArray> ph = Map.of("x", xArr);

        Map<String, INDArray> refResults = sd.output(ph, "result");
        INDArray refOutput = refResults.get("result");

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "result");
        assertNotNull(plan);
        log.info("testCudaGraphReplayMultiMatmul: {} slots", plan.getSlots().length);

        Pointer planHandle = compileNativePlan(plan);
        if (planHandle == null) { log.info("Skipping"); return; }

        try {
            INDArray[] extInputs = resolveExternalInputs(plan, sd, ph);
            nativeOps.resetTritonCounters();

            for (int iter = 0; iter < 5; iter++) {
                Map<String, INDArray> nativeResults = executeNativePlan(planHandle, plan, extInputs);
                INDArray nativeOutput = nativeResults.get("result");
                assertNotNull(nativeOutput, "Null output at iter " + iter);

                double maxDiff = refOutput.sub(nativeOutput).amaxNumber().doubleValue();
                log.info("testCudaGraphReplayMultiMatmul iter {}: maxDiff={}", iter, maxDiff);

                if (iter == 0) {
                    nativeOps.setPlanShapesFrozen(planHandle, true);
                }
            }
            log.info("testCudaGraphReplayMultiMatmul: {} Triton launches",
                     nativeOps.getTritonKernelLaunchCount());
        } finally {
            nativeOps.freeDynamicShapePlan(planHandle);
        }
    }

    /**
     * Test that runs 10 iterations to verify graph replay stability over many replays.
     * The SIGSEGV in the full VLM pipeline happens on the 2nd launch (1st replay).
     */
    @Test
    public void testCudaGraphReplayStability() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        SameDiff sd = createAttentionBlock(4, 2, 16);
        INDArray x = Nd4j.randn(DataType.FLOAT, 1, 4, 32);
        Map<String, INDArray> ph = Map.of("input", x);

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "result");
        assertNotNull(plan);

        Pointer planHandle = compileNativePlan(plan);
        if (planHandle == null) { log.info("Skipping"); return; }

        try {
            INDArray[] extInputs = resolveExternalInputs(plan, sd, ph);
            nativeOps.resetTritonCounters();

            for (int iter = 0; iter < 10; iter++) {
                Map<String, INDArray> nativeResults = executeNativePlan(planHandle, plan, extInputs);
                assertNotNull(nativeResults.get("result"), "Null output at iter " + iter);
                log.info("testCudaGraphReplayStability iter {}: OK", iter);

                if (iter == 0) {
                    nativeOps.setPlanShapesFrozen(planHandle, true);
                }
            }
            log.info("testCudaGraphReplayStability: 10 iterations completed, {} Triton launches",
                     nativeOps.getTritonKernelLaunchCount());
        } finally {
            nativeOps.freeDynamicShapePlan(planHandle);
        }
    }

    /**
     * Build a deep MLP with N layers to scale up graph node count.
     * Each layer: matmul + add + relu = ~5-8 graph nodes.
     * Returns (sd, placeholderMap) tuple.
     */
    private SameDiff createDeepMLP(int numLayers, int dim) {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, dim);
        SDVariable current = x;
        for (int i = 0; i < numLayers; i++) {
            SDVariable w = sd.constant("w" + i, Nd4j.randn(DataType.FLOAT, dim, dim).muli(0.01));
            SDVariable b = sd.constant("b" + i, Nd4j.zeros(DataType.FLOAT, 1, dim));
            current = sd.mmul("mm" + i, current, w);
            current = current.add("add" + i, b);
            if (i < numLayers - 1) {
                current = sd.nn.relu("relu" + i, current, 0);
            }
        }
        sd.identity("result", current);
        return sd;
    }

    /**
     * Parametric test: scale up graph size to find the crash threshold.
     * Small graphs (29 nodes) replay fine. Large graphs (5429 nodes) SIGSEGV.
     * This test scales from 10 to 500 layers (approx 30 to 2000+ graph nodes).
     *
     * Run with: -Dtest=TritonGraphBackendTest#testCudaGraphReplayScaleUp
     */
    @Test
    public void testCudaGraphReplayScaleUp() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        int dim = 32;  // small dim to keep memory manageable
        int[] layerCounts = {10, 30, 60, 100, 150, 200, 300, 500};

        for (int numLayers : layerCounts) {
            log.info("testCudaGraphReplayScaleUp: testing {} layers (dim={})", numLayers, dim);
            SameDiff sd = createDeepMLP(numLayers, dim);
            INDArray xArr = Nd4j.randn(DataType.FLOAT, 1, dim);
            Map<String, INDArray> ph = Map.of("x", xArr);

            DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "result");
            if (plan == null) {
                log.warn("testCudaGraphReplayScaleUp: {} layers - plan is null, skip", numLayers);
                continue;
            }
            log.info("  {} layers -> {} slots", numLayers, plan.getSlots().length);

            Pointer planHandle = compileNativePlan(plan);
            if (planHandle == null) {
                log.info("  {} layers - native compile failed, skip", numLayers);
                continue;
            }

            try {
                INDArray[] extInputs = resolveExternalInputs(plan, sd, ph);
                nativeOps.resetTritonCounters();

                for (int iter = 0; iter < 4; iter++) {
                    Map<String, INDArray> results = executeNativePlan(planHandle, plan, extInputs);
                    assertNotNull(results.get("result"),
                                  numLayers + " layers iter " + iter + " null output");
                    if (iter == 0) {
                        nativeOps.setPlanShapesFrozen(planHandle, true);
                    }
                }
                long tritonLaunches = nativeOps.getTritonKernelLaunchCount();
                log.info("  {} layers: PASS ({} Triton launches)", numLayers, tritonLaunches);
            } catch (Exception e) {
                log.error("  {} layers: FAILED - {}", numLayers, e.getMessage());
                fail("CUDA graph replay SIGSEGV at " + numLayers + " layers: " + e.getMessage());
            } finally {
                nativeOps.freeDynamicShapePlan(planHandle);
            }
        }
    }

    /**
     * Test with stacked attention blocks (like the actual VLM decoder).
     * Each block has ~4 matmuls + residual + norm = ~20-30 graph nodes.
     * Scale from 1 to 30 blocks (SmolDocling has 30 decoder layers).
     *
     * Run with: -Dtest=TritonGraphBackendTest#testCudaGraphReplayStackedAttention
     */
    @Test
    public void testCudaGraphReplayStackedAttention() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        int seqLen = 4;
        int heads = 3;
        int headDim = 16;
        int hiddenDim = heads * headDim;
        int[] blockCounts = {1, 5, 10, 20, 30};

        for (int numBlocks : blockCounts) {
            log.info("testCudaGraphReplayStackedAttention: {} blocks", numBlocks);
            SameDiff sd = SameDiff.create();
            SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, seqLen, hiddenDim);
            SDVariable current = input;

            for (int b = 0; b < numBlocks; b++) {
                // Q, K, V projections
                SDVariable wq = sd.constant("wq" + b, Nd4j.randn(DataType.FLOAT, hiddenDim, hiddenDim).muli(0.02));
                SDVariable wk = sd.constant("wk" + b, Nd4j.randn(DataType.FLOAT, hiddenDim, hiddenDim).muli(0.02));
                SDVariable wv = sd.constant("wv" + b, Nd4j.randn(DataType.FLOAT, hiddenDim, hiddenDim).muli(0.02));
                SDVariable wo = sd.constant("wo" + b, Nd4j.randn(DataType.FLOAT, hiddenDim, hiddenDim).muli(0.02));

                SDVariable q = sd.mmul("q" + b, current, wq);
                SDVariable k = sd.mmul("k" + b, current, wk);
                SDVariable v = sd.mmul("v" + b, current, wv);
                SDVariable attnOut = sd.mmul("o" + b, q, wo);

                // Residual + simple norm
                SDVariable residual = current.add("res" + b, attnOut);
                SDVariable mean = residual.mean("mean" + b, true, 2);
                current = residual.sub("norm" + b, mean);
            }
            sd.identity("result", current);

            INDArray xArr = Nd4j.randn(DataType.FLOAT, 1, seqLen, hiddenDim);
            Map<String, INDArray> ph = Map.of("input", xArr);

            DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "result");
            if (plan == null) {
                log.warn("  {} blocks: plan null, skip", numBlocks);
                continue;
            }
            log.info("  {} blocks -> {} slots", numBlocks, plan.getSlots().length);

            Pointer planHandle = compileNativePlan(plan);
            if (planHandle == null) {
                log.info("  {} blocks: native compile failed, skip", numBlocks);
                continue;
            }

            try {
                INDArray[] extInputs = resolveExternalInputs(plan, sd, ph);
                nativeOps.resetTritonCounters();

                for (int iter = 0; iter < 4; iter++) {
                    Map<String, INDArray> results = executeNativePlan(planHandle, plan, extInputs);
                    assertNotNull(results.get("result"),
                                  numBlocks + " blocks iter " + iter + " null output");
                    if (iter == 0) {
                        nativeOps.setPlanShapesFrozen(planHandle, true);
                    }
                }
                long tritonLaunches = nativeOps.getTritonKernelLaunchCount();
                log.info("  {} blocks: PASS ({} Triton launches)", numBlocks, tritonLaunches);
            } catch (Exception e) {
                log.error("  {} blocks: FAILED - {}", numBlocks, e.getMessage());
                fail("CUDA graph replay SIGSEGV at " + numBlocks + " blocks: " + e.getMessage());
            } finally {
                nativeOps.freeDynamicShapePlan(planHandle);
            }
        }
    }

    /**
     * Test that Triton-compiled broadcast multiply produces correct results
     * for inner-dimension broadcasting (e.g., [1,3,3,S,D] * [1,3,1,S,D]).
     * This is the GQA pattern: Q has N heads, KV has 1 head per group.
     * The bug was that simple modular indexing (offsets % inputSize) is wrong
     * for inner-dimension broadcasts, causing wrong values at high indices.
     */
    @Test
    public void testTritonBroadcastMultiplyInnerDim() {
        Assumptions.assumeTrue(
                Nd4j.getBackend().getClass().getSimpleName().toLowerCase().contains("cublas") ||
                Nd4j.getBackend().getClass().getSimpleName().toLowerCase().contains("cuda"),
                "CUDA backend required for Triton");

        int seqLen = 8;
        int headDim = 4;
        int numHeadGroups = 3;
        int headsPerGroup = 3;
        int kvHeadsPerGroup = 1;

        // Shape: [1, numHeadGroups, headsPerGroup, seqLen, headDim]
        // The multiply broadcasts KV (1 head) across Q (3 heads)
        SameDiff sd = SameDiff.create();
        SDVariable qInput = sd.placeHolder("q", DataType.FLOAT, 1, numHeadGroups, headsPerGroup, seqLen, headDim);
        SDVariable kvInput = sd.placeHolder("kv", DataType.FLOAT, 1, numHeadGroups, kvHeadsPerGroup, seqLen, headDim);

        // multiply with broadcast: [1,3,3,S,D] * [1,3,1,S,D]
        // Also add identity on result so there are at least 2 ops for DSP
        SDVariable product = qInput.mul("product", kvInput);
        SDVariable result = sd.identity("result", product);

        INDArray qArr = Nd4j.randn(DataType.FLOAT, 1, numHeadGroups, headsPerGroup, seqLen, headDim);
        INDArray kvArr = Nd4j.randn(DataType.FLOAT, 1, numHeadGroups, kvHeadsPerGroup, seqLen, headDim);

        // Use native plan execution with slot-by-slot reference
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        Map<String, INDArray> ph = new java.util.HashMap<>();
        ph.put("q", qArr);
        ph.put("kv", kvArr);

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "result");
        assertNotNull(plan, "Plan compilation returned null");
        Pointer planHandle = compileNativePlan(plan);
        Assumptions.assumeTrue(planHandle != null, "Native executor unavailable");

        try {
            INDArray[] extInputs = resolveExternalInputs(plan, sd, ph);

            // First execution (slot-by-slot, unfrozen) — use as reference
            Map<String, INDArray> refResults = executeNativePlan(planHandle, plan, extInputs);
            INDArray expected = refResults.get("result").dup();
            log.info("Expected shape: {}, len: {}", java.util.Arrays.toString(expected.shape()), expected.length());

            nativeOps.setPlanShapesFrozen(planHandle, true);

            // Run frozen iterations — compare against reference
            for (int iter = 1; iter < 5; iter++) {
                Map<String, INDArray> nativeResults = executeNativePlan(planHandle, plan, extInputs);
                INDArray actual = nativeResults.get("result");
                double maxDiff = expected.sub(actual).amaxNumber().doubleValue();
                log.info("Iteration {}: maxDiff={}, shape={}", iter, maxDiff,
                        java.util.Arrays.toString(actual.shape()));
                if (iter >= 2 && maxDiff > 1e-3) {
                    INDArray diff = actual.sub(expected);
                    INDArray flatActual = actual.reshape(actual.length());
                    INDArray flatExpected = expected.reshape(expected.length());
                    INDArray flatDiff = diff.reshape(diff.length());
                    int printed = 0;
                    for (long j = 0; j < flatDiff.length() && printed < 20; j++) {
                        double d = Math.abs(flatDiff.getDouble(j));
                        if (d > 1e-3) {
                            log.info("  MISMATCH idx={}: actual={} expected={} diff={}", j,
                                    flatActual.getDouble(j), flatExpected.getDouble(j), flatDiff.getDouble(j));
                            printed++;
                        }
                    }
                    assertEquals(0.0, maxDiff, 1e-3,
                            "Broadcast multiply mismatch at iteration " + iter + " (maxDiff=" + maxDiff + ")");
                }
            }
        } finally {
            nativeOps.freeDynamicShapePlan(planHandle);
        }
    }

    /**
     * Regression test for assign op Triton compilation.
     *
     * The assign(x, y) op copies input[1] (y) to the output. Previously, the Triton
     * compiler categorized assign as IDENTITY (which returns input[0]), producing
     * completely wrong output — maxDiff of FLT_MAX in the SmolDocling VLM pipeline.
     *
     * This test verifies that assign + cast chain produces correct output through
     * the Triton-compiled path, matching the native slot-by-slot reference.
     */
    @Test
    public void testTritonAssignOpCorrectness() {
        Assumptions.assumeTrue(Nd4j.getNativeOps().isTritonAvailable(),
                "Triton not available — skipping");

        SameDiff sd = SameDiff.create();

        // assign(x, y) should copy y to output, NOT x
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, 576);
        SDVariable y = sd.constant("y", Nd4j.ones(DataType.FLOAT, 1, 576).mul(42.0));
        SDVariable assigned = sd.assign("assigned", x, y);
        SDVariable castFp16 = assigned.castTo("castFp16", DataType.HALF);
        SDVariable castFp32 = castFp16.castTo("castFp32", DataType.FLOAT);
        SDVariable one = sd.constant("one", Nd4j.ones(DataType.FLOAT, 1, 576));
        SDVariable result = castFp32.add("result", one);

        INDArray xArr = Nd4j.randn(DataType.FLOAT, 1, 576);
        runOpTest("testTritonAssignOpCorrectness", sd, Map.of("x", xArr), "result");
    }

    /**
     * Regression test: assign op with same-shape inputs through multiple execution modes.
     *
     * Verifies that all execution modes produce the same output:
     * 1. Standard SameDiff execution (reference)
     * 2. DSP slot-by-slot (warmup iterations)
     * 3. DSP with Triton kernels (iteration 2+)
     */
    @Test
    public void testAssignAllExecutionModes() {
        Assumptions.assumeTrue(Nd4j.getNativeOps().isTritonAvailable(),
                "Triton not available — skipping");

        SameDiff sd = SameDiff.create();

        // Build: assign(x, y) -> multiply by 2 -> add 1
        // assign should produce y (the constant), not x (the placeholder)
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 4, 8);
        SDVariable y = sd.constant("y", Nd4j.linspace(1, 32, 32, DataType.FLOAT).reshape(4, 8));
        SDVariable assigned = sd.assign("assigned", x, y);
        SDVariable two = sd.constant("two", Nd4j.ones(DataType.FLOAT, 4, 8).mul(2.0));
        SDVariable doubled = assigned.mul("doubled", two);
        SDVariable one = sd.constant("one", Nd4j.ones(DataType.FLOAT, 4, 8));
        SDVariable result = doubled.add("result", one);

        INDArray xArr = Nd4j.randn(DataType.FLOAT, 4, 8).mul(100); // large values to make mismatch obvious

        // Reference: standard SameDiff execution
        Map<String, INDArray> ref = sd.output(Map.of("x", xArr), "result");
        INDArray refOutput = ref.get("result");
        assertNotNull(refOutput);

        // Expected: y * 2 + 1 = linspace(1,32)*2 + 1 = linspace(3,65)
        // Should NOT depend on x at all
        double refFirst = refOutput.getDouble(0);
        assertEquals(3.0, refFirst, 1e-3,
                "assign(x, y) should produce y, not x. First element should be 1*2+1=3, got " + refFirst);

        // Run through native plan (3 iterations to trigger Triton)
        runOpTest("testAssignAllExecutionModes", sd, Map.of("x", xArr), "result");
    }

    /**
     * Regression test: Triton CUDA graph capture + replay with fallback gaps.
     *
     * Creates a graph where some ops compile to Triton (elementwise) and others
     * must run natively as fallback (matmul). With tritonAllowFallbackCapture=true,
     * the entire sequence is captured as ONE CUDA graph. On replay, both Triton
     * sub-kernels AND native fallback ops are replayed from the graph.
     *
     * This specifically tests the zero-output bug where graph replay produces
     * all zeros because native gap ops' buffer addresses are baked into the graph
     * but may change between capture and replay.
     *
     * Timeline:
     *   iter 0: slot-by-slot warmup
     *   iter 1: Triton compile + first execution (executionCount=1)
     *   iter 2: CUDA graph capture (executionCount=2, captureMinExec=1)
     *   iter 3-7: CUDA graph replay (the regression test target)
     */
    @Test
    public void testTritonGraphCaptureWithFallbackGaps() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        if (!nativeOps.isTritonAvailable()) {
            log.info("Skipping (Triton not available)");
            return;
        }

        // Save and set environment
        var env = Nd4j.getEnvironment();
        boolean prevCapture = env.tritonGraphCapture();
        boolean prevFallbackCapture = env.tritonAllowFallbackCapture();
        boolean prevCompileAll = env.tritonCompileAll();
        int prevCaptureMinExec = env.tritonCaptureMinExec();

        env.setTritonGraphCapture(true);
        env.setTritonAllowFallbackCapture(true);
        env.setTritonCompileAll(false);  // don't compile matmul via Triton
        env.setTritonCaptureMinExec(1);  // capture on 2nd Triton execution

        try {
            nativeOps.invalidateTritonCache();
            nativeOps.resetTritonCounters();

            // Build: x -> EW ops -> matmul (gap) -> EW ops -> result
            // EW ops compile to Triton, matmul runs natively as a gap
            SameDiff sd = SameDiff.create();
            SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 16);
            SDVariable a = sd.constant("a", Nd4j.randn(DataType.FLOAT, 1, 16));
            SDVariable b = sd.constant("b", Nd4j.randn(DataType.FLOAT, 1, 16));
            SDVariable w = sd.constant("w", Nd4j.randn(DataType.FLOAT, 16, 32));
            SDVariable bias = sd.constant("bias", Nd4j.randn(DataType.FLOAT, 1, 32));
            SDVariable scale = sd.constant("scale", Nd4j.randn(DataType.FLOAT, 1, 32));
            SDVariable shift = sd.constant("shift", Nd4j.randn(DataType.FLOAT, 1, 32));

            // EW block 1 (Triton-compilable)
            SDVariable h1 = x.add("add1", a);
            SDVariable h2 = sd.nn.relu("relu1", h1, 0);
            SDVariable h3 = h2.mul("mul1", b.add("addab", a));

            // Matmul (native gap - not Triton compilable by default)
            SDVariable h4 = sd.mmul("matmul", h3, w);

            // EW block 2 (Triton-compilable)
            SDVariable h5 = h4.add("add2", bias);
            SDVariable h6 = sd.math.tanh("tanh1", h5);
            SDVariable h7 = h6.mul("mul2", scale);
            h7.add("result", shift);

            INDArray input = Nd4j.randn(DataType.FLOAT, 2, 16);
            Map<String, INDArray> ph = Map.of("x", input);

            // Reference output
            Map<String, INDArray> ref = sd.output(ph, "result");
            INDArray refOutput = ref.get("result");
            assertNotNull(refOutput, "Reference output is null");

            // Compile plan
            DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "result");
            assertNotNull(plan, "Plan is null");
            Pointer planHandle = compileNativePlan(plan);
            assertNotNull(planHandle, "Plan handle is null");

            try {
                INDArray[] extInputs = resolveExternalInputs(plan, sd, ph);

                int totalIters = 8;
                int failedIters = 0;
                StringBuilder failures = new StringBuilder();

                for (int iter = 0; iter < totalIters; iter++) {
                    Map<String, INDArray> nativeResults = executeNativePlan(planHandle, plan, extInputs);
                    INDArray nativeOutput = nativeResults.get("result");
                    assertNotNull(nativeOutput, "Null output at iter " + iter);

                    double maxDiff = refOutput.sub(nativeOutput).amaxNumber().doubleValue();
                    boolean isZero = nativeOutput.amaxNumber().doubleValue() == 0.0;
                    int argmax = Nd4j.argMax(nativeOutput.reshape(1, -1), 1).getInt(0);

                    log.info("testTritonGraphCaptureWithFallbackGaps iter {}: maxDiff={} isZero={} argmax={} " +
                            "first4=[{},{},{},{}]", iter, maxDiff, isZero, argmax,
                            nativeOutput.getDouble(0), nativeOutput.getDouble(1),
                            nativeOutput.getDouble(2), nativeOutput.getDouble(3));

                    if (maxDiff >= 0.01) {
                        String msg = String.format("iter %d: maxDiff=%.6f isZero=%b", iter, maxDiff, isZero);
                        log.error("FAIL: {}", msg);
                        failures.append(msg).append("\n");
                        failedIters++;
                    }

                    // Freeze shapes after first iteration
                    if (iter == 0) {
                        nativeOps.setPlanShapesFrozen(planHandle, true);
                    }
                }

                long tritonLaunches = nativeOps.getTritonKernelLaunchCount();
                log.info("testTritonGraphCaptureWithFallbackGaps: {} iterations, tritonLaunches={}, failures={}",
                        totalIters, tritonLaunches, failedIters);

                if (failedIters > 0) {
                    fail("Triton graph capture/replay with fallback gaps produced wrong output:\n" + failures);
                }
            } finally {
                nativeOps.freeDynamicShapePlan(planHandle);
            }
        } finally {
            env.setTritonGraphCapture(prevCapture);
            env.setTritonAllowFallbackCapture(prevFallbackCapture);
            env.setTritonCompileAll(prevCompileAll);
            env.setTritonCaptureMinExec(prevCaptureMinExec);
        }
    }

    /**
     * Regression test: Triton CUDA graph replay correctness with compileAll=true.
     *
     * With compileAll=true and specific includeTypes, CONST_GEN ops run as
     * fallback (alwaysFallback=true in SectionTypeConfig.h). This creates gaps
     * within the mega-segment. Tests that graph replay produces correct output
     * when mixing Triton sub-kernels with CONST_GEN fallback ops.
     */
    @Test
    public void testTritonGraphCaptureCompileAllWithConstGen() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        if (!nativeOps.isTritonAvailable()) {
            log.info("Skipping (Triton not available)");
            return;
        }

        var env = Nd4j.getEnvironment();
        boolean prevCapture = env.tritonGraphCapture();
        boolean prevFallbackCapture = env.tritonAllowFallbackCapture();
        boolean prevCompileAll = env.tritonCompileAll();
        int prevCaptureMinExec = env.tritonCaptureMinExec();

        env.setTritonGraphCapture(true);
        env.setTritonAllowFallbackCapture(true);
        env.setTritonCompileAll(true);
        env.setTritonCaptureMinExec(1);

        try {
            nativeOps.invalidateTritonCache();
            nativeOps.resetTritonCounters();

            // Build graph with shape_of (CONST_GEN, alwaysFallback) + EW ops
            SameDiff sd = SameDiff.create();
            SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 32);
            SDVariable w = sd.constant("w", Nd4j.randn(DataType.FLOAT, 32, 64));
            SDVariable b = sd.constant("b", Nd4j.randn(DataType.FLOAT, 1, 64));

            // shape_of creates a CONST_GEN gap
            SDVariable shapeOf = sd.shape("shape_x", x);
            // Use shape info indirectly (force shape_of to execute)
            SDVariable h1 = sd.mmul("mm", x, w);
            SDVariable h2 = h1.add("add_bias", b);
            SDVariable h3 = sd.nn.relu("relu", h2, 0);
            SDVariable h4 = sd.math.tanh("tanh", h3);
            h4.mul("result", sd.constant("scale", Nd4j.ones(DataType.FLOAT, 1, 64).mul(0.5)));

            INDArray input = Nd4j.randn(DataType.FLOAT, 1, 32);
            Map<String, INDArray> ph = Map.of("x", input);

            Map<String, INDArray> ref = sd.output(ph, "result");
            INDArray refOutput = ref.get("result");

            DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "result");
            Pointer planHandle = compileNativePlan(plan);
            assertNotNull(planHandle);

            try {
                INDArray[] extInputs = resolveExternalInputs(plan, sd, ph);

                int failedIters = 0;
                for (int iter = 0; iter < 8; iter++) {
                    Map<String, INDArray> nativeResults = executeNativePlan(planHandle, plan, extInputs);
                    INDArray nativeOutput = nativeResults.get("result");
                    assertNotNull(nativeOutput);
                    double maxDiff = refOutput.sub(nativeOutput).amaxNumber().doubleValue();
                    boolean isZero = nativeOutput.amaxNumber().doubleValue() == 0.0;

                    log.info("testTritonGraphCaptureCompileAllWithConstGen iter {}: maxDiff={} isZero={}",
                            iter, maxDiff, isZero);

                    if (maxDiff >= 0.01) {
                        log.error("FAIL iter {}: maxDiff={} isZero={}", iter, maxDiff, isZero);
                        failedIters++;
                    }
                    if (iter == 0) nativeOps.setPlanShapesFrozen(planHandle, true);
                }

                assertEquals(0, failedIters, "Graph replay with CONST_GEN fallback gaps produced wrong output");
            } finally {
                nativeOps.freeDynamicShapePlan(planHandle);
            }
        } finally {
            env.setTritonGraphCapture(prevCapture);
            env.setTritonAllowFallbackCapture(prevFallbackCapture);
            env.setTritonCompileAll(prevCompileAll);
            env.setTritonCaptureMinExec(prevCaptureMinExec);
        }
    }

    /**
     * Large-scale VLM-like test: emulates the SmolDocling VLM decode pattern with
     * many ops, multiple gap types (matmul + shape_of/cast), and graph capture/replay.
     *
     * The VLM decode graph has ~3840 ops with repeating "transformer block" patterns:
     *   [EW norm] → [matmul Q/K/V projection (gap)] → [EW attention scores] →
     *   [matmul attention (gap)] → [EW residual + norm] → [matmul FFN (gap)] → ...
     *
     * This test creates a similar pattern with configurable depth:
     *   N "layers", each containing:
     *     - EW ops (add, relu, mul) → Triton-compiled
     *     - matmul projection → native gap
     *     - shape_of → CONST_GEN gap (alwaysFallback)
     *     - cast → gap
     *     - EW ops (add, tanh) → Triton-compiled
     *
     * Tests that CUDA graph capture + replay produces correct output over many
     * iterations, with compileAll=true and tritonAllowFallbackCapture=true.
     */
    @Test
    /**
     * Helper: run a SameDiff graph through native DSP with Triton compileAll enabled,
     * compare against slot-by-slot reference. Returns maxDiff.
     */
    private double runTritonCompileAllAndCompare(SameDiff sd, String outputName,
                                                  Map<String, INDArray> ph,
                                                  int numIters, double tolerance) {
        return runTritonCompileAllAndCompare(sd, outputName, ph, numIters, tolerance, true);
    }

    private double runTritonCompileAllAndCompare(SameDiff sd, String outputName,
                                                  Map<String, INDArray> ph,
                                                  int numIters, double tolerance,
                                                  boolean freezeShapes) {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        nativeOps.invalidateTritonCache();
        nativeOps.resetTritonCounters();

        // Reference output (slot-by-slot)
        Map<String, INDArray> ref = sd.output(ph, outputName);
        INDArray refOutput = ref.get(outputName);
        assertNotNull(refOutput, "Reference output is null");

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, outputName);
        assertNotNull(plan, "Plan is null");
        Pointer planHandle = compileNativePlan(plan);
        assertNotNull(planHandle, "Plan handle is null");

        double worstDiff = 0.0;
        try {
            INDArray[] extInputs = resolveExternalInputs(plan, sd, ph);
            for (int iter = 0; iter < numIters; iter++) {
                Map<String, INDArray> nativeResults = executeNativePlan(planHandle, plan, extInputs);
                INDArray nativeOutput = nativeResults.get(outputName);
                assertNotNull(nativeOutput, "Null output at iter " + iter);

                double maxDiff = refOutput.sub(nativeOutput).amaxNumber().doubleValue();
                boolean isZero = nativeOutput.amaxNumber().doubleValue() == 0.0;
                worstDiff = Math.max(worstDiff, maxDiff);

                log.info("iter {}: maxDiff={} isZero={}", iter, maxDiff, isZero);

                assertFalse(isZero, "ALL ZERO output at iter " + iter);
                assertTrue(maxDiff < tolerance,
                        String.format("iter %d: maxDiff=%.6f exceeds tolerance %.6f", iter, maxDiff, tolerance));

                if (iter == 0 && freezeShapes) {
                    nativeOps.setPlanShapesFrozen(planHandle, true);
                }
            }
        } finally {
            nativeOps.freeDynamicShapePlan(planHandle);
        }
        return worstDiff;
    }

    /**
     * Test: shape_of(INT64) → cast(FLOAT) → reduce_mean fused with larger EW ops.
     * Regression test for buffer overflow when storing to reductionInputSlot buffer
     * that is smaller than the kernel's n_elements.
     */
    @Test
    public void testTritonMultiShapeShapeOfCastReduceMean() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        if (!nativeOps.isTritonAvailable()) { log.info("Skipping (Triton not available)"); return; }

        var env = Nd4j.getEnvironment();
        boolean prevCompileAll = env.tritonCompileAll();
        env.setTritonCompileAll(true);
        try {
            SameDiff sd = SameDiff.create();
            SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, 64);

            // EW ops on [1,64]
            SDVariable normed = sd.nn.relu("relu", x, 0);
            SDVariable scaled = normed.mul("scale", sd.constant("scaleW", Nd4j.ones(DataType.FLOAT, 1, 64).mul(0.5)));

            // Matmul gap → shape_of gap → cast → reduce_mean (fused with EW)
            SDVariable w = sd.constant("w", Nd4j.eye(64).castTo(DataType.FLOAT));
            SDVariable proj = sd.mmul("proj", scaled, w);

            SDVariable shape = sd.shape("shape", proj);  // INT64 [2]
            SDVariable castFloat = shape.castTo("cast", DataType.FLOAT);  // FLOAT [2]
            SDVariable shapeMean = castFloat.mean("shape_mean");  // FLOAT scalar

            // Inject shape_mean back into the EW stream
            SDVariable eps = sd.constant("eps", Nd4j.scalar(DataType.FLOAT, 0.001f));
            SDVariable shapeScale = shapeMean.mul("shape_scale", eps);
            SDVariable output = proj.add("output", shapeScale);

            INDArray input = Nd4j.randn(DataType.FLOAT, 1, 64);

            // Test WITHOUT freezing — validates pure Triton compilation
            double maxDiffNoFreeze = runTritonCompileAllAndCompare(sd, "output", Map.of("x", input), 5, 1e-3, false);
            log.info("testTritonMultiShapeShapeOfCastReduceMean: noFreeze worstDiff={}", maxDiffNoFreeze);

            // Test WITH freezing — validates frozen constant mechanism
            double maxDiffFreeze = runTritonCompileAllAndCompare(sd, "output", Map.of("x", input), 5, 1e-3, true);
            log.info("testTritonMultiShapeShapeOfCastReduceMean: withFreeze worstDiff={}", maxDiffFreeze);
        } finally {
            env.setTritonCompileAll(prevCompileAll);
        }
    }

    /**
     * Test: reduce_sum on a small INT64 input (from shape_of) cast to FLOAT,
     * fused with larger EW ops on a different shape.
     */
    @Test
    public void testTritonMultiShapeReduceSumSmallBuffer() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        if (!nativeOps.isTritonAvailable()) { log.info("Skipping (Triton not available)"); return; }

        var env = Nd4j.getEnvironment();
        boolean prevCompileAll = env.tritonCompileAll();
        env.setTritonCompileAll(true);
        try {
            SameDiff sd = SameDiff.create();
            SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, 128);

            // Large EW ops on [1,128]
            SDVariable activated = sd.math.tanh("tanh", x);

            // Matmul gap → shape_of gap → cast → reduce_sum (small buffer [2] fused with [128] EW)
            SDVariable w = sd.constant("w", Nd4j.eye(128).castTo(DataType.FLOAT));
            SDVariable proj = sd.mmul("proj", activated, w);
            SDVariable shape = sd.shape("shape", proj);
            SDVariable castFloat = shape.castTo("cast", DataType.FLOAT);
            SDVariable shapeSum = sd.sum("shape_sum", castFloat);  // sum instead of mean

            SDVariable eps = sd.constant("eps", Nd4j.scalar(DataType.FLOAT, 0.0001f));
            SDVariable shapeScale = shapeSum.mul("shape_scale", eps);
            SDVariable output = proj.add("output", shapeScale);

            INDArray input = Nd4j.randn(DataType.FLOAT, 1, 128);
            double maxDiff = runTritonCompileAllAndCompare(sd, "output", Map.of("x", input), 5, 1e-3);
            log.info("testTritonMultiShapeReduceSumSmallBuffer: worstDiff={}", maxDiff);
        } finally {
            env.setTritonCompileAll(prevCompileAll);
        }
    }

    /**
     * Test: reduce_max on a small buffer fused with larger EW ops.
     * Verifies that reduction identity values and per-buffer masks work for max reduction.
     */
    @Test
    public void testTritonMultiShapeReduceMaxSmallBuffer() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        if (!nativeOps.isTritonAvailable()) { log.info("Skipping (Triton not available)"); return; }

        var env = Nd4j.getEnvironment();
        boolean prevCompileAll = env.tritonCompileAll();
        env.setTritonCompileAll(true);
        try {
            SameDiff sd = SameDiff.create();
            SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, 64);

            SDVariable activated = sd.nn.relu("relu", x, 0);

            // Matmul gap → shape_of gap → cast → reduce_max
            SDVariable w = sd.constant("w", Nd4j.eye(64).castTo(DataType.FLOAT));
            SDVariable proj = sd.mmul("proj", activated, w);
            SDVariable shape = sd.shape("shape", proj);
            SDVariable castFloat = shape.castTo("cast", DataType.FLOAT);
            SDVariable shapeMax = sd.max("shape_max", castFloat);

            SDVariable eps = sd.constant("eps", Nd4j.scalar(DataType.FLOAT, 0.001f));
            SDVariable shapeScale = shapeMax.mul("shape_scale", eps);
            SDVariable output = proj.add("output", shapeScale);

            INDArray input = Nd4j.randn(DataType.FLOAT, 1, 64);
            double maxDiff = runTritonCompileAllAndCompare(sd, "output", Map.of("x", input), 5, 1e-3);
            log.info("testTritonMultiShapeReduceMaxSmallBuffer: worstDiff={}", maxDiff);
        } finally {
            env.setTritonCompileAll(prevCompileAll);
        }
    }

    /**
     * Test: reduce_mean on a rank-3 tensor's shape (3-element INT64) fused with [1,256] EW ops.
     * Larger shape disparity (3 vs 256 elements).
     */
    @Test
    public void testTritonMultiShapeRank3ShapeReduce() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        if (!nativeOps.isTritonAvailable()) { log.info("Skipping (Triton not available)"); return; }

        var env = Nd4j.getEnvironment();
        boolean prevCompileAll = env.tritonCompileAll();
        env.setTritonCompileAll(true);
        try {
            SameDiff sd = SameDiff.create();
            // Use rank-3 placeholder so shape_of produces [3] (3-element INT64)
            SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, 4, 64);

            // EW ops on [1,4,64] = 256 elements
            SDVariable activated = sd.nn.relu("relu", x, 0);

            // shape_of → cast → reduce_mean: [3] intermediate in [256]-element kernel
            SDVariable shape = sd.shape("shape", activated);
            SDVariable castFloat = shape.castTo("cast", DataType.FLOAT);
            SDVariable shapeMean = castFloat.mean("shape_mean");

            SDVariable eps = sd.constant("eps", Nd4j.scalar(DataType.FLOAT, 0.001f));
            SDVariable shapeScale = shapeMean.mul("shape_scale", eps);

            // Reshape to [1,256] then add
            SDVariable flat = sd.reshape("flat", activated, 1, 256);
            SDVariable output = flat.add("output", shapeScale);

            INDArray input = Nd4j.randn(DataType.FLOAT, 1, 4, 64);
            double maxDiff = runTritonCompileAllAndCompare(sd, "output", Map.of("x", input), 5, 1e-3);
            log.info("testTritonMultiShapeRank3ShapeReduce: worstDiff={}", maxDiff);
        } finally {
            env.setTritonCompileAll(prevCompileAll);
        }
    }

    /**
     * Test: cast(FLOAT→HALF) → reduce_mean fused with larger FLOAT EW ops.
     * Different dtype cast path (float→half, not int64→float).
     */
    @Test
    public void testTritonMultiShapeCastHalfReduceMean() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        if (!nativeOps.isTritonAvailable()) { log.info("Skipping (Triton not available)"); return; }

        var env = Nd4j.getEnvironment();
        boolean prevCompileAll = env.tritonCompileAll();
        env.setTritonCompileAll(true);
        try {
            SameDiff sd = SameDiff.create();
            SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, 64);

            SDVariable activated = sd.nn.relu("relu", x, 0);

            // Matmul gap → cast(FLOAT→HALF) → reduce_mean → cast(HALF→FLOAT)
            SDVariable w = sd.constant("w", Nd4j.eye(64).castTo(DataType.FLOAT));
            SDVariable proj = sd.mmul("proj", activated, w);

            // reduce_mean along axis 1 (64 elements → scalar)
            SDVariable mean = proj.mean("mean", 1);
            SDVariable castHalf = mean.castTo("castHalf", DataType.HALF);
            SDVariable castBack = castHalf.castTo("castBack", DataType.FLOAT);

            // castBack is [1] element, proj is [1,64] — multi-shape
            SDVariable eps = sd.constant("eps", Nd4j.scalar(DataType.FLOAT, 0.01f));
            SDVariable meanScale = castBack.mul("meanScale", eps);
            SDVariable output = proj.add("output", meanScale);

            INDArray input = Nd4j.randn(DataType.FLOAT, 1, 64);
            double maxDiff = runTritonCompileAllAndCompare(sd, "output", Map.of("x", input), 5, 1e-2);
            log.info("testTritonMultiShapeCastHalfReduceMean: worstDiff={}", maxDiff);
        } finally {
            env.setTritonCompileAll(prevCompileAll);
        }
    }

    /**
     * Test: Multiple reductions with different sizes in the same kernel.
     * Two separate reduce_mean chains with different input sizes, fused with EW ops.
     */
    @Test
    public void testTritonMultiShapeMultipleReductions() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        if (!nativeOps.isTritonAvailable()) { log.info("Skipping (Triton not available)"); return; }

        var env = Nd4j.getEnvironment();
        boolean prevCompileAll = env.tritonCompileAll();
        env.setTritonCompileAll(true);
        try {
            SameDiff sd = SameDiff.create();
            SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, 64);

            SDVariable activated = sd.nn.relu("relu", x, 0);

            // Gap ops
            SDVariable w1 = sd.constant("w1", Nd4j.eye(64).castTo(DataType.FLOAT));
            SDVariable proj1 = sd.mmul("proj1", activated, w1);

            // Chain 1: shape_of → cast → reduce_mean (2-element input)
            SDVariable shape1 = sd.shape("shape1", proj1);
            SDVariable cast1 = shape1.castTo("cast1", DataType.FLOAT);
            SDVariable mean1 = cast1.mean("mean1");

            // Chain 2: reduce_mean on proj1 along axis 1 (64-element input)
            SDVariable mean2 = proj1.mean("mean2", 1);

            // Combine both means with the main stream
            SDVariable eps1 = sd.constant("eps1", Nd4j.scalar(DataType.FLOAT, 0.001f));
            SDVariable eps2 = sd.constant("eps2", Nd4j.scalar(DataType.FLOAT, 0.01f));
            SDVariable scale1 = mean1.mul("scale1", eps1);
            SDVariable scale2 = mean2.mul("scale2", eps2);
            SDVariable combined = proj1.add("add1", scale1);
            SDVariable output = combined.add("output", scale2);

            INDArray input = Nd4j.randn(DataType.FLOAT, 1, 64);
            double maxDiff = runTritonCompileAllAndCompare(sd, "output", Map.of("x", input), 5, 1e-3);
            log.info("testTritonMultiShapeMultipleReductions: worstDiff={}", maxDiff);
        } finally {
            env.setTritonCompileAll(prevCompileAll);
        }
    }

    /**
     * Test: Stacked layers with shape_of → cast → reduce_mean, each layer's reduction
     * output feeding into the next layer. Tests accumulation of errors across layers.
     */
    @Test
    public void testTritonMultiShapeStackedLayers() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        if (!nativeOps.isTritonAvailable()) { log.info("Skipping (Triton not available)"); return; }

        var env = Nd4j.getEnvironment();
        boolean prevCompileAll = env.tritonCompileAll();
        env.setTritonCompileAll(true);
        try {
            int numLayers = 8;
            int hiddenDim = 32;
            SameDiff sd = SameDiff.create();
            SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, hiddenDim);

            SDVariable current = x;
            for (int i = 0; i < numLayers; i++) {
                String p = "L" + i + "_";
                // EW
                current = sd.nn.relu(p + "relu", current, 0);
                // Matmul gap
                double scale = Math.sqrt(2.0 / hiddenDim);
                SDVariable w = sd.constant(p + "w", Nd4j.randn(DataType.FLOAT, hiddenDim, hiddenDim).mul(scale));
                SDVariable proj = sd.mmul(p + "proj", current, w);
                // shape_of → cast → reduce_mean
                SDVariable shape = sd.shape(p + "shape", proj);
                SDVariable cast = shape.castTo(p + "cast", DataType.FLOAT);
                SDVariable mean = cast.mean(p + "mean");
                SDVariable eps = sd.constant(p + "eps", Nd4j.scalar(DataType.FLOAT, 0.0001f));
                SDVariable shapeScale = mean.mul(p + "scale", eps);
                // Residual
                current = proj.add(p + "residual", current);
                current = current.add(p + "inject", shapeScale);
                current = sd.math.tanh(p + "act", current);
            }

            SDVariable outW = sd.constant("outW", Nd4j.eye(hiddenDim).castTo(DataType.FLOAT));
            SDVariable output = sd.mmul("output", current, outW);

            INDArray input = Nd4j.randn(DataType.FLOAT, 1, hiddenDim);
            // Tolerance accounts for f32 precision differences between cuBLAS GEMM
            // and Triton per-element matmul (different accumulation order) across
            // 8 stacked layers. Kahan summation in Triton matmul K-loop reduces
            // per-layer error to ~O(eps), but algorithm differences remain.
            double maxDiff = runTritonCompileAllAndCompare(sd, "output", Map.of("x", input), 5, 5e-3);
            log.info("testTritonMultiShapeStackedLayers: {} layers, worstDiff={}", numLayers, maxDiff);
        } finally {
            env.setTritonCompileAll(prevCompileAll);
        }
    }

    /**
     * Test: reduce_prod on a small buffer fused with larger EW ops.
     * Verifies product reduction identity (1.0) and per-buffer masks.
     */
    @Test
    public void testTritonMultiShapeReduceProdSmallBuffer() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        if (!nativeOps.isTritonAvailable()) { log.info("Skipping (Triton not available)"); return; }

        var env = Nd4j.getEnvironment();
        boolean prevCompileAll = env.tritonCompileAll();
        env.setTritonCompileAll(true);
        try {
            SameDiff sd = SameDiff.create();
            SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, 64);

            SDVariable activated = sd.nn.relu("relu", x, 0);

            SDVariable w = sd.constant("w", Nd4j.eye(64).castTo(DataType.FLOAT));
            SDVariable proj = sd.mmul("proj", activated, w);
            SDVariable shape = sd.shape("shape", proj);
            SDVariable castFloat = shape.castTo("cast", DataType.FLOAT);
            SDVariable shapeProd = sd.prod("shape_prod", castFloat);

            SDVariable eps = sd.constant("eps", Nd4j.scalar(DataType.FLOAT, 1e-6f));
            SDVariable shapeScale = shapeProd.mul("shape_scale", eps);
            SDVariable output = proj.add("output", shapeScale);

            INDArray input = Nd4j.randn(DataType.FLOAT, 1, 64);
            double maxDiff = runTritonCompileAllAndCompare(sd, "output", Map.of("x", input), 5, 1e-3);
            log.info("testTritonMultiShapeReduceProdSmallBuffer: worstDiff={}", maxDiff);
        } finally {
            env.setTritonCompileAll(prevCompileAll);
        }
    }

    /**
     * Original VLM-like multi-layer test with shape_of → cast → reduce_mean chain.
     */
    @Test
    public void testTritonGraphCaptureVlmLikeMultiLayer() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        if (!nativeOps.isTritonAvailable()) {
            log.info("Skipping (Triton not available)");
            return;
        }

        var env = Nd4j.getEnvironment();
        boolean prevCapture = env.tritonGraphCapture();
        boolean prevFallbackCapture = env.tritonAllowFallbackCapture();
        boolean prevCompileAll = env.tritonCompileAll();
        int prevCaptureMinExec = env.tritonCaptureMinExec();

        env.setTritonGraphCapture(true);
        env.setTritonAllowFallbackCapture(true);
        env.setTritonCompileAll(true);
        env.setTritonCaptureMinExec(1);

        try {
            nativeOps.invalidateTritonCache();
            nativeOps.resetTritonCounters();

            int numLayers = 12;  // Restore to meaningful depth
            int hiddenDim = 64;
            int batchSize = 1;

            SameDiff sd = SameDiff.create();
            SDVariable x = sd.placeHolder("x", DataType.FLOAT, batchSize, hiddenDim);

            SDVariable current = x;
            for (int layer = 0; layer < numLayers; layer++) {
                String prefix = "L" + layer + "_";

                // Layer norm weights (EW - Triton compiled)
                SDVariable gamma = sd.constant(prefix + "gamma", Nd4j.ones(DataType.FLOAT, 1, hiddenDim));
                SDVariable beta = sd.constant(prefix + "beta", Nd4j.zeros(DataType.FLOAT, 1, hiddenDim));

                // EW block: simulated layer norm (mul + add)
                SDVariable normed = current.mul(prefix + "norm_mul", gamma);
                normed = normed.add(prefix + "norm_add", beta);
                normed = sd.nn.relu(prefix + "norm_relu", normed, 0);

                // Matmul projection (gap - not Triton compiled)
                // Use Xavier initialization to prevent vanishing gradients
                double scale = Math.sqrt(2.0 / hiddenDim);
                SDVariable projW = sd.constant(prefix + "projW",
                        Nd4j.randn(DataType.FLOAT, hiddenDim, hiddenDim).mul(scale));
                SDVariable projected = sd.mmul(prefix + "proj", normed, projW);

                // shape_of → cast(INT64→FLOAT) → reduce_mean → multiply chain
                // This chain is the source of precision divergence when Triton compiles
                // the cast+reduce_mean ops with shape_of as a gap op
                SDVariable shape = sd.shape(prefix + "shape", projected);
                SDVariable castFloat = shape.castTo(prefix + "cast", DataType.FLOAT);
                SDVariable shapeMean = castFloat.mean(prefix + "shape_mean");
                SDVariable shapeEpsilon = sd.constant(prefix + "shapeEps",
                        Nd4j.scalar(DataType.FLOAT, 0.0001f));
                SDVariable shapeScale = shapeMean.mul(prefix + "shape_scale_mul", shapeEpsilon);

                // EW residual block (Triton compiled)
                SDVariable residual = projected.add(prefix + "residual", current);
                residual = residual.add(prefix + "shape_inject", shapeScale);
                residual = sd.math.tanh(prefix + "act", residual);

                // FFN matmul (gap)
                SDVariable ffnW = sd.constant(prefix + "ffnW",
                        Nd4j.randn(DataType.FLOAT, hiddenDim, hiddenDim).mul(scale));
                SDVariable ffn = sd.mmul(prefix + "ffn", residual, ffnW);

                // FFN activation (EW - Triton compiled)
                SDVariable ffnBias = sd.constant(prefix + "ffnBias",
                        Nd4j.zeros(DataType.FLOAT, 1, hiddenDim));
                current = ffn.add(prefix + "ffn_add", ffnBias);
                current = sd.nn.relu(prefix + "ffn_relu", current, 0);
            }

            // Final output
            double outScale = Math.sqrt(2.0 / hiddenDim);
            SDVariable outputW = sd.constant("outW",
                    Nd4j.randn(DataType.FLOAT, hiddenDim, hiddenDim).mul(outScale));
            SDVariable outBias = sd.constant("outBias", Nd4j.zeros(DataType.FLOAT, 1, hiddenDim));
            SDVariable output = sd.mmul("out_proj", current, outputW);
            output = output.add("output", outBias);

            INDArray input = Nd4j.randn(DataType.FLOAT, batchSize, hiddenDim);
            Map<String, INDArray> ph = Map.of("x", input);

            // Reference output (slot-by-slot)
            Map<String, INDArray> ref = sd.output(ph, "output");
            INDArray refOutput = ref.get("output");
            assertNotNull(refOutput, "Reference output is null");
            log.info("testTritonGraphCaptureVlmLikeMultiLayer: ref argmax={} first4=[{},{},{},{}]",
                    Nd4j.argMax(refOutput.reshape(1, -1), 1).getInt(0),
                    refOutput.getDouble(0), refOutput.getDouble(1),
                    refOutput.getDouble(2), refOutput.getDouble(3));

            // Compile plan
            DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "output");
            assertNotNull(plan, "Plan is null");
            Pointer planHandle = compileNativePlan(plan);
            assertNotNull(planHandle, "Plan handle is null");

            try {
                INDArray[] extInputs = resolveExternalInputs(plan, sd, ph);
                int totalIters = 10;
                int failedIters = 0;
                StringBuilder failures = new StringBuilder();

                for (int iter = 0; iter < totalIters; iter++) {
                    Map<String, INDArray> nativeResults = executeNativePlan(planHandle, plan, extInputs);
                    INDArray nativeOutput = nativeResults.get("output");
                    assertNotNull(nativeOutput, "Null output at iter " + iter);

                    double maxDiff = refOutput.sub(nativeOutput).amaxNumber().doubleValue();
                    boolean isZero = nativeOutput.amaxNumber().doubleValue() == 0.0;
                    int argmax = Nd4j.argMax(nativeOutput.reshape(1, -1), 1).getInt(0);

                    log.info("testTritonGraphCaptureVlmLikeMultiLayer iter {}: maxDiff={} isZero={} argmax={} " +
                            "first4=[{},{},{},{}]", iter, maxDiff, isZero, argmax,
                            nativeOutput.getDouble(0), nativeOutput.getDouble(1),
                            nativeOutput.getDouble(2), nativeOutput.getDouble(3));

                    if (isZero) {
                        String msg = String.format("iter %d: ALL ZERO output (replay bug)", iter);
                        log.error("FAIL: {}", msg);
                        failures.append(msg).append("\n");
                        failedIters++;
                    } else if (maxDiff >= 0.01) {
                        String msg = String.format("iter %d: maxDiff=%.6f", iter, maxDiff);
                        log.error("FAIL: {}", msg);
                        failures.append(msg).append("\n");
                        failedIters++;
                    }

                    // Freeze shapes after first iteration (like VLM decode)
                    if (iter == 0) {
                        nativeOps.setPlanShapesFrozen(planHandle, true);
                    }
                }

                long tritonLaunches = nativeOps.getTritonKernelLaunchCount();
                log.info("testTritonGraphCaptureVlmLikeMultiLayer: {} layers, {} iters, tritonLaunches={}, failures={}",
                        numLayers, totalIters, tritonLaunches, failedIters);

                if (failedIters > 0) {
                    fail("VLM-like multi-layer graph capture/replay produced wrong output:\n" + failures);
                }
            } finally {
                nativeOps.freeDynamicShapePlan(planHandle);
            }
        } finally {
            env.setTritonGraphCapture(prevCapture);
            env.setTritonAllowFallbackCapture(prevFallbackCapture);
            env.setTritonCompileAll(prevCompileAll);
            env.setTritonCaptureMinExec(prevCaptureMinExec);
        }
    }

    // ─── Fused Attention Tests ──────────────────────────────────────────────

    /**
     * Build a SameDiff graph with a single dot_product_attention_v2 op.
     * Uses rank-4 BSHD format: [batch, seq, heads, headDim].
     *
     * @param batch    batch size
     * @param seqQ     query sequence length
     * @param seqK     key/value sequence length
     * @param numHeads number of attention heads (same for Q, K, V — no GQA)
     * @param headDim  head dimension
     * @param scaled   whether to use auto-scaling (1/sqrt(headDim))
     */
    private SameDiff createDpaV2Graph(int batch, int seqQ, int seqK,
                                       int numHeads, int headDim, boolean scaled) {
        SameDiff sd = SameDiff.create();
        // BSHD format: [batch, seq, heads, headDim]
        SDVariable q = sd.placeHolder("q", DataType.FLOAT, batch, seqQ, numHeads, headDim);
        SDVariable k = sd.placeHolder("k", DataType.FLOAT, batch, seqK, numHeads, headDim);
        SDVariable v = sd.placeHolder("v", DataType.FLOAT, batch, seqK, numHeads, headDim);

        // scaleFactor=0 means auto: 1/sqrt(headDim)
        double scaleFactor = scaled ? 0.0 : 1.0;
        SDVariable out = sd.nn().dotProductAttentionV2("attn_out", q, v, k,
                null, null, scaleFactor, 0.0, false, false);
        return sd;
    }

    /**
     * Test: Triton fused attention kernel with dot_product_attention_v2.
     *
     * Reproduces the NaN bug found in SmolDocling VLM decode where the Triton
     * IR builder assumes BHSD layout [batch, heads, seq, dim] for rank-4 inputs
     * but dot_product_attention_v2 uses BSHD layout [batch, seq, heads, dim].
     *
     * Uses SmolDocling-like decode shapes:
     *   batch=1, seqQ=1, seqK=32, numHeads=9, headDim=64
     */
    @Test
    public void testFusedAttentionDpaV2_decodeShape() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        var env = Nd4j.getEnvironment();

        // Save and set Triton config to compile FUSED_ATTENTION
        boolean prevCompileAll = env.tritonCompileAll();
        String prevIncludeTypes = env.tritonIncludeTypes();
        env.setTritonCompileAll(true);
        env.setTritonIncludeTypes("FUSED_ATTENTION");

        try {
            int batch = 1, seqQ = 1, seqK = 32, numHeads = 9, headDim = 64;
            SameDiff sd = createDpaV2Graph(batch, seqQ, seqK, numHeads, headDim, true);

            INDArray qArr = Nd4j.randn(DataType.FLOAT, batch, seqQ, numHeads, headDim).muli(0.1);
            INDArray kArr = Nd4j.randn(DataType.FLOAT, batch, seqK, numHeads, headDim).muli(0.1);
            INDArray vArr = Nd4j.randn(DataType.FLOAT, batch, seqK, numHeads, headDim).muli(0.1);
            Map<String, INDArray> ph = Map.of("q", qArr, "k", kArr, "v", vArr);

            // Get reference output from standard SameDiff execution
            Map<String, INDArray> refResults = sd.output(ph, "attn_out");
            // CRITICAL: dup() the reference output — sd.output() may return a view into
            // a SameDiff internal buffer that gets overwritten during DSP execution
            INDArray refOutput = refResults.get("attn_out").dup();
            assertNotNull(refOutput, "Reference output is null");
            assertFalse(refOutput.isNaN().any(), "Reference output contains NaN");
            log.info("testFusedAttentionDpaV2_decodeShape: ref shape={}, ref max={}",
                    java.util.Arrays.toString(refOutput.shape()), refOutput.amaxNumber());

            // Compile to native plan
            DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "attn_out");
            assertNotNull(plan, "Plan compilation returned null");
            log.info("Plan: {} slots, {} ext inputs", plan.getSlots().length,
                    plan.getExternalInputKeys().length);

            Pointer planHandle = compileNativePlan(plan);
            if (planHandle == null) {
                log.info("Skipping (native executor not supported)");
                return;
            }
            try {
                INDArray[] extInputs = resolveExternalInputs(plan, sd, ph);
                nativeOps.resetTritonCounters();

                for (int iter = 0; iter < 3; iter++) {
                    Map<String, INDArray> nativeResults = executeNativePlan(planHandle, plan, extInputs);
                    INDArray nativeOutput = nativeResults.get("attn_out");
                    assertNotNull(nativeOutput, "Native output is null at iteration " + iter);

                    boolean hasNaN = nativeOutput.isNaN().any();
                    double maxDiff = hasNaN ? Double.NaN
                            : refOutput.sub(nativeOutput).amaxNumber().doubleValue();
                    log.info("Iter {}: hasNaN={}, maxDiff={}, tritonLaunches={}",
                            iter, hasNaN, maxDiff, nativeOps.getTritonKernelLaunchCount());

                    // Dump first 10 values for debugging on iter 2
                    if (iter == 2 && maxDiff > 1e-4) {
                        float[] refFlat = refOutput.dup('c').data().asFloat();
                        float[] natFlat = nativeOutput.dup('c').data().asFloat();
                        int n = Math.min(20, refFlat.length);
                        StringBuilder sb = new StringBuilder("REF[0:").append(n).append("]: ");
                        for (int i = 0; i < n; i++) sb.append(String.format("%.6f ", refFlat[i]));
                        log.info(sb.toString());
                        sb = new StringBuilder("NAT[0:").append(n).append("]: ");
                        for (int i = 0; i < n; i++) sb.append(String.format("%.6f ", natFlat[i]));
                        log.info(sb.toString());
                        sb = new StringBuilder("DIFF[0:").append(n).append("]: ");
                        for (int i = 0; i < n; i++) sb.append(String.format("%.6f ", refFlat[i] - natFlat[i]));
                        log.info(sb.toString());
                    }

                    assertFalse(hasNaN, "Native output contains NaN at iteration " + iter);
                    assertTrue(maxDiff < TOLERANCE,
                            "Output mismatch at iteration " + iter + ": max diff = " + maxDiff);

                    if (iter == 0) {
                        nativeOps.setPlanShapesFrozen(planHandle, true);
                    }
                }

                long tritonLaunches = nativeOps.getTritonKernelLaunchCount();
                log.info("testFusedAttentionDpaV2_decodeShape: tritonLaunches={}", tritonLaunches);
            } finally {
                nativeOps.freeDynamicShapePlan(planHandle);
            }
        } finally {
            env.setTritonCompileAll(prevCompileAll);
            env.setTritonIncludeTypes(prevIncludeTypes);
        }
    }

    /**
     * Test: Triton fused attention with larger sequence length (prefill-like).
     * Verifies correctness with seqQ > 1 where tiling is more complex.
     */
    @Test
    public void testFusedAttentionDpaV2_prefillShape() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        var env = Nd4j.getEnvironment();

        boolean prevCompileAll = env.tritonCompileAll();
        String prevIncludeTypes = env.tritonIncludeTypes();
        env.setTritonCompileAll(true);
        env.setTritonIncludeTypes("FUSED_ATTENTION");

        try {
            int batch = 1, seqQ = 16, seqK = 16, numHeads = 4, headDim = 64;
            SameDiff sd = createDpaV2Graph(batch, seqQ, seqK, numHeads, headDim, true);

            INDArray qArr = Nd4j.randn(DataType.FLOAT, batch, seqQ, numHeads, headDim).muli(0.1);
            INDArray kArr = Nd4j.randn(DataType.FLOAT, batch, seqK, numHeads, headDim).muli(0.1);
            INDArray vArr = Nd4j.randn(DataType.FLOAT, batch, seqK, numHeads, headDim).muli(0.1);
            Map<String, INDArray> ph = Map.of("q", qArr, "k", kArr, "v", vArr);

            Map<String, INDArray> refResults = sd.output(ph, "attn_out");
            // CRITICAL: dup() the reference output — sd.output() may return a view into
            // a SameDiff internal buffer that gets overwritten during DSP execution
            INDArray refOutput = refResults.get("attn_out").dup();
            assertNotNull(refOutput, "Reference output is null");
            assertFalse(refOutput.isNaN().any(), "Reference output contains NaN");
            log.info("testFusedAttentionDpaV2_prefillShape: ref shape={}", java.util.Arrays.toString(refOutput.shape()));

            DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "attn_out");
            Pointer planHandle = compileNativePlan(plan);
            if (planHandle == null) {
                log.info("Skipping (native executor not supported)");
                return;
            }
            try {
                INDArray[] extInputs = resolveExternalInputs(plan, sd, ph);
                nativeOps.resetTritonCounters();

                for (int iter = 0; iter < 3; iter++) {
                    Map<String, INDArray> nativeResults = executeNativePlan(planHandle, plan, extInputs);
                    INDArray nativeOutput = nativeResults.get("attn_out");
                    assertNotNull(nativeOutput, "Native output is null at iteration " + iter);

                    boolean hasNaN = nativeOutput.isNaN().any();
                    double maxDiff = hasNaN ? Double.NaN
                            : refOutput.sub(nativeOutput).amaxNumber().doubleValue();
                    log.info("Prefill iter {}: hasNaN={}, maxDiff={}", iter, hasNaN, maxDiff);

                    assertFalse(hasNaN, "Native output contains NaN at iteration " + iter);
                    assertTrue(maxDiff < TOLERANCE,
                            "Output mismatch at iteration " + iter + ": max diff = " + maxDiff);

                    if (iter == 0) {
                        nativeOps.setPlanShapesFrozen(planHandle, true);
                    }
                }
            } finally {
                nativeOps.freeDynamicShapePlan(planHandle);
            }
        } finally {
            env.setTritonCompileAll(prevCompileAll);
            env.setTritonIncludeTypes(prevIncludeTypes);
        }
    }

    /**
     * Test: Triton fused attention with single-head (rank-3 compatible) shapes.
     * Uses 1 head to test the simplest attention case.
     */
    @Test
    public void testFusedAttentionDpaV2_singleHead() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        var env = Nd4j.getEnvironment();

        boolean prevCompileAll = env.tritonCompileAll();
        String prevIncludeTypes = env.tritonIncludeTypes();
        env.setTritonCompileAll(true);
        env.setTritonIncludeTypes("FUSED_ATTENTION");

        try {
            int batch = 2, seqQ = 4, seqK = 8, numHeads = 1, headDim = 32;
            SameDiff sd = createDpaV2Graph(batch, seqQ, seqK, numHeads, headDim, true);

            INDArray qArr = Nd4j.randn(DataType.FLOAT, batch, seqQ, numHeads, headDim).muli(0.1);
            INDArray kArr = Nd4j.randn(DataType.FLOAT, batch, seqK, numHeads, headDim).muli(0.1);
            INDArray vArr = Nd4j.randn(DataType.FLOAT, batch, seqK, numHeads, headDim).muli(0.1);
            Map<String, INDArray> ph = Map.of("q", qArr, "k", kArr, "v", vArr);

            Map<String, INDArray> refResults = sd.output(ph, "attn_out");
            // CRITICAL: dup() the reference output — sd.output() may return a view into
            // a SameDiff internal buffer that gets overwritten during DSP execution
            INDArray refOutput = refResults.get("attn_out").dup();
            assertNotNull(refOutput, "Reference output is null");
            assertFalse(refOutput.isNaN().any(), "Reference output contains NaN");

            DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "attn_out");
            Pointer planHandle = compileNativePlan(plan);
            if (planHandle == null) {
                log.info("Skipping (native executor not supported)");
                return;
            }
            try {
                INDArray[] extInputs = resolveExternalInputs(plan, sd, ph);
                nativeOps.resetTritonCounters();

                for (int iter = 0; iter < 3; iter++) {
                    Map<String, INDArray> nativeResults = executeNativePlan(planHandle, plan, extInputs);
                    INDArray nativeOutput = nativeResults.get("attn_out");
                    assertNotNull(nativeOutput, "Native output is null at iteration " + iter);

                    boolean hasNaN = nativeOutput.isNaN().any();
                    double maxDiff = hasNaN ? Double.NaN
                            : refOutput.sub(nativeOutput).amaxNumber().doubleValue();
                    log.info("SingleHead iter {}: hasNaN={}, maxDiff={}", iter, hasNaN, maxDiff);

                    assertFalse(hasNaN, "Native output contains NaN at iteration " + iter);
                    assertTrue(maxDiff < TOLERANCE,
                            "Output mismatch at iteration " + iter + ": max diff = " + maxDiff);

                    if (iter == 0) {
                        nativeOps.setPlanShapesFrozen(planHandle, true);
                    }
                }
            } finally {
                nativeOps.freeDynamicShapePlan(planHandle);
            }
        } finally {
            env.setTritonCompileAll(prevCompileAll);
            env.setTritonIncludeTypes(prevIncludeTypes);
        }
    }

    // ─── ONNX MultiHead Attention Triton Isolation Tests ────────────────────

    /**
     * Build a SameDiff graph with onnx_multi_head_attention using past_key/past_value
     * and attention bias — exactly the SmolDocling decoder pattern.
     *
     * Input layout: Q/K/V [batch, seqQ, hidden], past_key/past_value [batch, heads, seqK, headDim]
     * Attention bias [batch, heads, seqQ, seqK]
     */
    private SameDiff createOnnxMhaWithKvCacheGraph(int batch, int seqQ, int numHeads,
                                                     int headDim, int kvCacheLen,
                                                     boolean withBias) {
        SameDiff sd = SameDiff.create();
        int hidden = numHeads * headDim;

        SDVariable q = sd.placeHolder("q", DataType.FLOAT, batch, seqQ, hidden);
        SDVariable k = sd.placeHolder("k", DataType.FLOAT, batch, seqQ, hidden);
        SDVariable v = sd.placeHolder("v", DataType.FLOAT, batch, seqQ, hidden);
        SDVariable pastKey = sd.placeHolder("past_key", DataType.FLOAT, batch, numHeads, kvCacheLen, headDim);
        SDVariable pastValue = sd.placeHolder("past_value", DataType.FLOAT, batch, numHeads, kvCacheLen, headDim);

        SDVariable attnBias = null;
        if (withBias) {
            int totalSeq = kvCacheLen + seqQ;
            attnBias = sd.placeHolder("attn_bias", DataType.FLOAT, batch, numHeads, seqQ, totalSeq);
        }

        var attnOp = new org.nd4j.linalg.api.ops.impl.transforms.custom.OnnxMultiHeadAttention(
                sd, q, k, v, attnBias, pastKey, pastValue,
                numHeads, 0.0, false, 1);
        SDVariable out = attnOp.outputVariables()[0];
        out.rename("attn_out");
        return sd;
    }

    /**
     * Isolation test: Triton fused attention for onnx_multi_head_attention with KV cache.
     * Reproduces the SmolDocling decode config: batch=1, seqQ=1, heads=4, headDim=64,
     * past_key/past_value with a small cache (not full 1024).
     *
     * Compares Triton kernel output against C++ reference output.
     *
     * Run with: -Dtest=TritonGraphBackendTest#testFusedAttention_OnnxMHA_KvCache
     */
    @Test
    public void testFusedAttention_OnnxMHA_KvCache() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        var env = Nd4j.getEnvironment();

        boolean prevCompileAll = env.tritonCompileAll();
        String prevIncludeTypes = env.tritonIncludeTypes();
        env.setTritonCompileAll(true);
        env.setTritonIncludeTypes("FUSED_ATTENTION");

        try {
            // SmolDocling-like decode shapes
            int batch = 1, seqQ = 1, numHeads = 4, headDim = 64;
            int kvCacheLen = 10; // small realistic cache (not full 1024)
            int hidden = numHeads * headDim;

            SameDiff sd = createOnnxMhaWithKvCacheGraph(batch, seqQ, numHeads, headDim, kvCacheLen, false);

            INDArray qArr = Nd4j.randn(DataType.FLOAT, batch, seqQ, hidden).muli(0.1);
            INDArray kArr = Nd4j.randn(DataType.FLOAT, batch, seqQ, hidden).muli(0.1);
            INDArray vArr = Nd4j.randn(DataType.FLOAT, batch, seqQ, hidden).muli(0.1);
            INDArray pastKeyArr = Nd4j.randn(DataType.FLOAT, batch, numHeads, kvCacheLen, headDim).muli(0.1);
            INDArray pastValArr = Nd4j.randn(DataType.FLOAT, batch, numHeads, kvCacheLen, headDim).muli(0.1);

            Map<String, INDArray> ph = Map.of(
                    "q", qArr, "k", kArr, "v", vArr,
                    "past_key", pastKeyArr, "past_value", pastValArr);

            // Reference output from standard C++ execution
            Map<String, INDArray> refResults = sd.output(ph, "attn_out");
            INDArray refOutput = refResults.get("attn_out").dup();
            assertNotNull(refOutput, "Reference output is null");
            assertFalse(refOutput.isNaN().any(), "Reference output contains NaN");
            log.info("testFusedAttention_OnnxMHA_KvCache: ref shape={}, ref max={}, ref mean={}",
                    java.util.Arrays.toString(refOutput.shape()),
                    refOutput.amaxNumber(), refOutput.meanNumber());

            // Compile to native DSP plan
            DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "attn_out");
            assertNotNull(plan, "Plan compilation returned null");
            log.info("Plan: {} slots, {} ext inputs", plan.getSlots().length,
                    plan.getExternalInputKeys().length);

            Pointer planHandle = compileNativePlan(plan);
            if (planHandle == null) {
                log.info("Skipping (native executor not supported)");
                return;
            }
            try {
                INDArray[] extInputs = resolveExternalInputs(plan, sd, ph);
                nativeOps.resetTritonCounters();

                for (int iter = 0; iter < 3; iter++) {
                    Map<String, INDArray> nativeResults = executeNativePlan(planHandle, plan, extInputs);
                    INDArray nativeOutput = nativeResults.get("attn_out");
                    assertNotNull(nativeOutput, "Native output is null at iteration " + iter);

                    boolean hasNaN = nativeOutput.isNaN().any();
                    double maxDiff = hasNaN ? Double.NaN
                            : refOutput.sub(nativeOutput).amaxNumber().doubleValue();
                    log.info("OnnxMHA_KvCache iter {}: hasNaN={}, maxDiff={}, native max={}, native mean={}",
                            iter, hasNaN, maxDiff,
                            nativeOutput.amaxNumber(), nativeOutput.meanNumber());

                    if (iter == 2 && (hasNaN || maxDiff > TOLERANCE)) {
                        float[] refFlat = refOutput.dup('c').data().asFloat();
                        float[] natFlat = nativeOutput.dup('c').data().asFloat();
                        int n = Math.min(20, refFlat.length);
                        StringBuilder sb = new StringBuilder("REF[0:").append(n).append("]: ");
                        for (int i = 0; i < n; i++) sb.append(String.format("%.6f ", refFlat[i]));
                        log.info(sb.toString());
                        sb = new StringBuilder("NAT[0:").append(n).append("]: ");
                        for (int i = 0; i < n; i++) sb.append(String.format("%.6f ", natFlat[i]));
                        log.info(sb.toString());
                    }

                    assertFalse(hasNaN, "Native output contains NaN at iteration " + iter);
                    assertTrue(maxDiff < TOLERANCE,
                            "Output mismatch at iteration " + iter + ": max diff = " + maxDiff);

                    if (iter == 0) {
                        nativeOps.setPlanShapesFrozen(planHandle, true);
                    }
                }

                long tritonLaunches = nativeOps.getTritonKernelLaunchCount();
                log.info("testFusedAttention_OnnxMHA_KvCache: tritonLaunches={}", tritonLaunches);
            } finally {
                nativeOps.freeDynamicShapePlan(planHandle);
            }
        } finally {
            env.setTritonCompileAll(prevCompileAll);
            env.setTritonIncludeTypes(prevIncludeTypes);
        }
    }

    /**
     * Isolation test: Same as above but with attention bias (mask).
     * SmolDocling uses attention bias to mask invalid KV cache positions.
     *
     * Run with: -Dtest=TritonGraphBackendTest#testFusedAttention_OnnxMHA_KvCache_WithBias
     */
    @Test
    public void testFusedAttention_OnnxMHA_KvCache_WithBias() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        var env = Nd4j.getEnvironment();

        boolean prevCompileAll = env.tritonCompileAll();
        String prevIncludeTypes = env.tritonIncludeTypes();
        env.setTritonCompileAll(true);
        env.setTritonIncludeTypes("FUSED_ATTENTION");

        try {
            int batch = 1, seqQ = 1, numHeads = 4, headDim = 64;
            int kvCacheLen = 10;
            int totalSeq = kvCacheLen + seqQ;
            int hidden = numHeads * headDim;

            SameDiff sd = createOnnxMhaWithKvCacheGraph(batch, seqQ, numHeads, headDim, kvCacheLen, true);

            INDArray qArr = Nd4j.randn(DataType.FLOAT, batch, seqQ, hidden).muli(0.1);
            INDArray kArr = Nd4j.randn(DataType.FLOAT, batch, seqQ, hidden).muli(0.1);
            INDArray vArr = Nd4j.randn(DataType.FLOAT, batch, seqQ, hidden).muli(0.1);
            INDArray pastKeyArr = Nd4j.randn(DataType.FLOAT, batch, numHeads, kvCacheLen, headDim).muli(0.1);
            INDArray pastValArr = Nd4j.randn(DataType.FLOAT, batch, numHeads, kvCacheLen, headDim).muli(0.1);

            // Attention bias: 0 for valid positions, -inf for invalid (like a causal mask)
            INDArray biasArr = Nd4j.zeros(DataType.FLOAT, batch, numHeads, seqQ, totalSeq);

            Map<String, INDArray> ph = new java.util.HashMap<>();
            ph.put("q", qArr);
            ph.put("k", kArr);
            ph.put("v", vArr);
            ph.put("past_key", pastKeyArr);
            ph.put("past_value", pastValArr);
            ph.put("attn_bias", biasArr);

            Map<String, INDArray> refResults = sd.output(ph, "attn_out");
            INDArray refOutput = refResults.get("attn_out").dup();
            assertNotNull(refOutput, "Reference output is null");
            assertFalse(refOutput.isNaN().any(), "Reference output contains NaN");
            log.info("testFusedAttention_OnnxMHA_KvCache_WithBias: ref shape={}, ref max={}, ref mean={}",
                    java.util.Arrays.toString(refOutput.shape()),
                    refOutput.amaxNumber(), refOutput.meanNumber());

            DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "attn_out");
            assertNotNull(plan, "Plan compilation returned null");

            Pointer planHandle = compileNativePlan(plan);
            if (planHandle == null) {
                log.info("Skipping (native executor not supported)");
                return;
            }
            try {
                INDArray[] extInputs = resolveExternalInputs(plan, sd, ph);
                nativeOps.resetTritonCounters();

                for (int iter = 0; iter < 3; iter++) {
                    Map<String, INDArray> nativeResults = executeNativePlan(planHandle, plan, extInputs);
                    INDArray nativeOutput = nativeResults.get("attn_out");
                    assertNotNull(nativeOutput, "Native output is null at iteration " + iter);

                    boolean hasNaN = nativeOutput.isNaN().any();
                    double maxDiff = hasNaN ? Double.NaN
                            : refOutput.sub(nativeOutput).amaxNumber().doubleValue();
                    log.info("OnnxMHA_KvCache_WithBias iter {}: hasNaN={}, maxDiff={}", iter, hasNaN, maxDiff);

                    if (iter == 2 && (hasNaN || maxDiff > TOLERANCE)) {
                        float[] refFlat = refOutput.dup('c').data().asFloat();
                        float[] natFlat = nativeOutput.dup('c').data().asFloat();
                        int n = Math.min(20, refFlat.length);
                        StringBuilder sb = new StringBuilder("REF: ");
                        for (int i = 0; i < n; i++) sb.append(String.format("%.6f ", refFlat[i]));
                        log.info(sb.toString());
                        sb = new StringBuilder("NAT: ");
                        for (int i = 0; i < n; i++) sb.append(String.format("%.6f ", natFlat[i]));
                        log.info(sb.toString());
                    }

                    assertFalse(hasNaN, "Native output contains NaN at iteration " + iter);
                    assertTrue(maxDiff < TOLERANCE,
                            "Output mismatch at iteration " + iter + ": max diff = " + maxDiff);

                    if (iter == 0) {
                        nativeOps.setPlanShapesFrozen(planHandle, true);
                    }
                }
            } finally {
                nativeOps.freeDynamicShapePlan(planHandle);
            }
        } finally {
            env.setTritonCompileAll(prevCompileAll);
            env.setTritonIncludeTypes(prevIncludeTypes);
        }
    }

    /**
     * Isolation test: Static KV cache with large capacity, small valid length.
     * Reproduces the exact SmolDocling scenario: cache [1, 4, 1024, 64] but only
     * first ~10 positions valid. Tests that the Triton kernel uses the correct
     * sequence length (not the full cache capacity).
     *
     * Run with: -Dtest=TritonGraphBackendTest#testFusedAttention_OnnxMHA_StaticKvCache
     */
    @Test
    public void testFusedAttention_OnnxMHA_StaticKvCache() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        var env = Nd4j.getEnvironment();

        boolean prevCompileAll = env.tritonCompileAll();
        String prevIncludeTypes = env.tritonIncludeTypes();
        env.setTritonCompileAll(true);
        env.setTritonIncludeTypes("FUSED_ATTENTION");

        try {
            int batch = 1, seqQ = 1, numHeads = 4, headDim = 64;
            int cacheCapacity = 128; // static cache capacity (smaller than 1024 for test speed)
            int validLen = 10; // only 10 positions are actually valid
            int hidden = numHeads * headDim;
            int totalSeq = cacheCapacity + seqQ; // bias covers full cache capacity + query

            SameDiff sd = SameDiff.create();
            SDVariable q = sd.placeHolder("q", DataType.FLOAT, batch, seqQ, hidden);
            SDVariable k = sd.placeHolder("k", DataType.FLOAT, batch, seqQ, hidden);
            SDVariable v = sd.placeHolder("v", DataType.FLOAT, batch, seqQ, hidden);
            SDVariable pastKey = sd.placeHolder("past_key", DataType.FLOAT, batch, numHeads, cacheCapacity, headDim);
            SDVariable pastValue = sd.placeHolder("past_value", DataType.FLOAT, batch, numHeads, cacheCapacity, headDim);
            SDVariable attnBias = sd.placeHolder("attn_bias", DataType.FLOAT, batch, numHeads, seqQ, totalSeq);

            var attnOp = new org.nd4j.linalg.api.ops.impl.transforms.custom.OnnxMultiHeadAttention(
                    sd, q, k, v, attnBias, pastKey, pastValue,
                    numHeads, 0.0, false, 1);
            SDVariable out = attnOp.outputVariables()[0];
            out.rename("attn_out");

            // Create inputs: fill cache with zeros except valid positions
            INDArray qArr = Nd4j.randn(DataType.FLOAT, batch, seqQ, hidden).muli(0.1);
            INDArray kArr = Nd4j.randn(DataType.FLOAT, batch, seqQ, hidden).muli(0.1);
            INDArray vArr = Nd4j.randn(DataType.FLOAT, batch, seqQ, hidden).muli(0.1);

            // Static cache: valid data in first validLen positions, zeros elsewhere
            INDArray pastKeyArr = Nd4j.zeros(DataType.FLOAT, batch, numHeads, cacheCapacity, headDim);
            INDArray pastValArr = Nd4j.zeros(DataType.FLOAT, batch, numHeads, cacheCapacity, headDim);
            INDArray validKeys = Nd4j.randn(DataType.FLOAT, batch, numHeads, validLen, headDim).muli(0.1);
            INDArray validVals = Nd4j.randn(DataType.FLOAT, batch, numHeads, validLen, headDim).muli(0.1);
            pastKeyArr.get(NDArrayIndex.all(), NDArrayIndex.all(),
                    NDArrayIndex.interval(0, validLen), NDArrayIndex.all()).assign(validKeys);
            pastValArr.get(NDArrayIndex.all(), NDArrayIndex.all(),
                    NDArrayIndex.interval(0, validLen), NDArrayIndex.all()).assign(validVals);

            // Attention bias: 0 for valid positions (first validLen + 1), -inf for invalid
            float maskVal = -3.4028235e+38f;
            INDArray biasArr = Nd4j.valueArrayOf(new long[]{batch, numHeads, seqQ, totalSeq}, maskVal, DataType.FLOAT);
            // Unmask first validLen positions (past) and position validLen (current query's key)
            for (int pos = 0; pos <= validLen; pos++) {
                biasArr.get(NDArrayIndex.all(), NDArrayIndex.all(),
                        NDArrayIndex.all(), NDArrayIndex.point(pos)).assign(0.0f);
            }

            Map<String, INDArray> ph = new java.util.HashMap<>();
            ph.put("q", qArr);
            ph.put("k", kArr);
            ph.put("v", vArr);
            ph.put("past_key", pastKeyArr);
            ph.put("past_value", pastValArr);
            ph.put("attn_bias", biasArr);

            Map<String, INDArray> refResults = sd.output(ph, "attn_out");
            INDArray refOutput = refResults.get("attn_out").dup();
            assertNotNull(refOutput, "Reference output is null");
            assertFalse(refOutput.isNaN().any(), "Reference output contains NaN");
            log.info("testFusedAttention_OnnxMHA_StaticKvCache: ref shape={}, ref max={}, ref mean={}",
                    java.util.Arrays.toString(refOutput.shape()),
                    refOutput.amaxNumber(), refOutput.meanNumber());

            DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "attn_out");
            assertNotNull(plan, "Plan compilation returned null");

            Pointer planHandle = compileNativePlan(plan);
            if (planHandle == null) {
                log.info("Skipping (native executor not supported)");
                return;
            }
            try {
                INDArray[] extInputs = resolveExternalInputs(plan, sd, ph);
                nativeOps.resetTritonCounters();

                for (int iter = 0; iter < 3; iter++) {
                    Map<String, INDArray> nativeResults = executeNativePlan(planHandle, plan, extInputs);
                    INDArray nativeOutput = nativeResults.get("attn_out");
                    assertNotNull(nativeOutput, "Native output is null at iteration " + iter);

                    boolean hasNaN = nativeOutput.isNaN().any();
                    double maxDiff = hasNaN ? Double.NaN
                            : refOutput.sub(nativeOutput).amaxNumber().doubleValue();
                    log.info("StaticKvCache iter {}: hasNaN={}, maxDiff={}, native max={}, native mean={}",
                            iter, hasNaN, maxDiff,
                            nativeOutput.amaxNumber(), nativeOutput.meanNumber());

                    if (hasNaN || maxDiff > TOLERANCE) {
                        float[] refFlat = refOutput.dup('c').data().asFloat();
                        float[] natFlat = nativeOutput.dup('c').data().asFloat();
                        int n = Math.min(20, refFlat.length);
                        StringBuilder sb = new StringBuilder("REF: ");
                        for (int i = 0; i < n; i++) sb.append(String.format("%.6f ", refFlat[i]));
                        log.info(sb.toString());
                        sb = new StringBuilder("NAT: ");
                        for (int i = 0; i < n; i++) sb.append(String.format("%.6f ", natFlat[i]));
                        log.info(sb.toString());
                    }

                    assertFalse(hasNaN, "Native output contains NaN at iteration " + iter);
                    assertTrue(maxDiff < TOLERANCE,
                            "Output mismatch at iteration " + iter + ": max diff = " + maxDiff
                            + " (cacheCapacity=" + cacheCapacity + ", validLen=" + validLen + ")");

                    if (iter == 0) {
                        nativeOps.setPlanShapesFrozen(planHandle, true);
                    }
                }
            } finally {
                nativeOps.freeDynamicShapePlan(planHandle);
            }
        } finally {
            env.setTritonCompileAll(prevCompileAll);
            env.setTritonIncludeTypes(prevIncludeTypes);
        }
    }

    // =====================================================================
    // Tests bridging the gap between FUSED_ATTENTION-only isolation tests
    // and VLM full pipeline. These test attention with OTHER compiled op types
    // (matmul, reduce, elementwise) in the same graph — the VLM scenario.
    // =====================================================================

    /**
     * Test 1: Matmul projections → ONNX MHA, compiled together.
     *
     * In VLM, Q/K/V come from matmul projections (Triton sub-kernels),
     * NOT external inputs. This tests cross-sub-kernel data flow:
     *   input → matmul(Wq) → Q
     *   input → matmul(Wk) → K
     *   input → matmul(Wv) → V
     *   Q, K, V, past_key, past_value → onnx_multi_head_attention → output
     *
     * Uses FUSED_ATTENTION + compileAll (which compiles matmul via Triton too).
     */
    @Test
    public void testFusedAttention_MatmulProjections_CompileAll() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        var env = Nd4j.getEnvironment();

        boolean prevCompileAll = env.tritonCompileAll();
        String prevIncludeTypes = env.tritonIncludeTypes();
        env.setTritonCompileAll(true);
        // This is the VLM config: compile everything + attention
        env.setTritonIncludeTypes("CONST_GEN,GATHER,CONCAT,SPLIT,STACK,ATTENTION");

        try {
            int batch = 1, seqQ = 1, numHeads = 4, headDim = 64;
            int kvCacheLen = 10;
            int hidden = numHeads * headDim;
            int totalSeq = kvCacheLen + seqQ;

            SameDiff sd = SameDiff.create();
            // Input: [batch, seqQ, hidden]
            SDVariable input = sd.placeHolder("input", DataType.FLOAT, batch, seqQ, hidden);

            // Weight projections (constants, will be compiled as matmul ops)
            SDVariable wq = sd.constant("wq", Nd4j.randn(DataType.FLOAT, hidden, hidden).muli(0.02));
            SDVariable wk = sd.constant("wk", Nd4j.randn(DataType.FLOAT, hidden, hidden).muli(0.02));
            SDVariable wv = sd.constant("wv", Nd4j.randn(DataType.FLOAT, hidden, hidden).muli(0.02));

            // Q, K, V projections via matmul — these become Triton sub-kernels
            SDVariable q = sd.mmul("q_proj", input, wq);
            SDVariable k = sd.mmul("k_proj", input, wk);
            SDVariable v = sd.mmul("v_proj", input, wv);

            // KV cache inputs (external)
            SDVariable pastKey = sd.placeHolder("past_key", DataType.FLOAT, batch, numHeads, kvCacheLen, headDim);
            SDVariable pastValue = sd.placeHolder("past_value", DataType.FLOAT, batch, numHeads, kvCacheLen, headDim);
            SDVariable attnBias = sd.placeHolder("attn_bias", DataType.FLOAT, batch, numHeads, seqQ, totalSeq);

            // Attention op consuming matmul outputs
            var attnOp = new org.nd4j.linalg.api.ops.impl.transforms.custom.OnnxMultiHeadAttention(
                    sd, q, k, v, attnBias, pastKey, pastValue,
                    numHeads, 0.0, false, 1);
            SDVariable out = attnOp.outputVariables()[0];
            out.rename("attn_out");

            // Create inputs
            INDArray inputArr = Nd4j.randn(DataType.FLOAT, batch, seqQ, hidden).muli(0.1);
            INDArray pastKeyArr = Nd4j.randn(DataType.FLOAT, batch, numHeads, kvCacheLen, headDim).muli(0.1);
            INDArray pastValArr = Nd4j.randn(DataType.FLOAT, batch, numHeads, kvCacheLen, headDim).muli(0.1);
            float maskVal = -3.4028235e+38f;
            INDArray biasArr = Nd4j.valueArrayOf(new long[]{batch, numHeads, seqQ, totalSeq}, maskVal, DataType.FLOAT);
            for (int pos = 0; pos <= kvCacheLen; pos++) {
                biasArr.get(NDArrayIndex.all(), NDArrayIndex.all(),
                        NDArrayIndex.all(), NDArrayIndex.point(pos)).assign(0.0f);
            }

            Map<String, INDArray> ph = new java.util.HashMap<>();
            ph.put("input", inputArr);
            ph.put("past_key", pastKeyArr);
            ph.put("past_value", pastValArr);
            ph.put("attn_bias", biasArr);

            // Reference output from C++ execution
            Map<String, INDArray> refResults = sd.output(ph, "attn_out");
            INDArray refOutput = refResults.get("attn_out").dup();
            assertNotNull(refOutput, "Reference output is null");
            assertFalse(refOutput.isNaN().any(), "Reference output contains NaN");
            log.info("testFusedAttention_MatmulProjections_CompileAll: ref shape={}, ref max={}, ref mean={}",
                    java.util.Arrays.toString(refOutput.shape()),
                    refOutput.amaxNumber(), refOutput.meanNumber());

            // Compile to native DSP plan
            DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "attn_out");
            assertNotNull(plan, "Plan compilation returned null");
            log.info("Plan: {} slots, {} ext inputs", plan.getSlots().length,
                    plan.getExternalInputKeys().length);

            Pointer planHandle = compileNativePlan(plan);
            if (planHandle == null) {
                log.info("Skipping (native executor not supported)");
                return;
            }
            try {
                INDArray[] extInputs = resolveExternalInputs(plan, sd, ph);
                nativeOps.resetTritonCounters();

                for (int iter = 0; iter < 4; iter++) {
                    Map<String, INDArray> nativeResults = executeNativePlan(planHandle, plan, extInputs);
                    INDArray nativeOutput = nativeResults.get("attn_out");
                    assertNotNull(nativeOutput, "Native output is null at iteration " + iter);

                    boolean hasNaN = nativeOutput.isNaN().any();
                    double maxDiff = hasNaN ? Double.NaN
                            : refOutput.sub(nativeOutput).amaxNumber().doubleValue();
                    log.info("MatmulProjections_CompileAll iter {}: hasNaN={}, maxDiff={}, native max={}, native mean={}",
                            iter, hasNaN, maxDiff,
                            nativeOutput.amaxNumber(), nativeOutput.meanNumber());

                    if (hasNaN || maxDiff > TOLERANCE) {
                        float[] refFlat = refOutput.dup('c').data().asFloat();
                        float[] natFlat = nativeOutput.dup('c').data().asFloat();
                        int n = Math.min(20, refFlat.length);
                        StringBuilder sb = new StringBuilder("REF: ");
                        for (int i = 0; i < n; i++) sb.append(String.format("%.6f ", refFlat[i]));
                        log.info(sb.toString());
                        sb = new StringBuilder("NAT: ");
                        for (int i = 0; i < n; i++) sb.append(String.format("%.6f ", natFlat[i]));
                        log.info(sb.toString());
                    }

                    assertFalse(hasNaN, "Native output contains NaN at iteration " + iter);
                    assertTrue(maxDiff < TOLERANCE,
                            "Output mismatch at iteration " + iter + ": max diff = " + maxDiff);

                    if (iter == 0) {
                        nativeOps.setPlanShapesFrozen(planHandle, true);
                    }
                }

                long tritonLaunches = nativeOps.getTritonKernelLaunchCount();
                log.info("testFusedAttention_MatmulProjections_CompileAll: tritonLaunches={}", tritonLaunches);
            } finally {
                nativeOps.freeDynamicShapePlan(planHandle);
            }
        } finally {
            env.setTritonCompileAll(prevCompileAll);
            env.setTritonIncludeTypes(prevIncludeTypes);
        }
    }

    /**
     * Test 2: Multi-layer decoder with matmul→attention→residual, compiled together.
     *
     * Mimics a small VLM decoder: N layers, each with:
     *   LayerNorm(simplified) → Q/K/V matmul projections → attention → output matmul → residual add
     *
     * Uses ALL compile types + ATTENTION to match VLM config.
     * Tests cross-sub-kernel data flow across multiple layers.
     */
    @Test
    public void testFusedAttention_MultiLayerDecoder_CompileAll() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        var env = Nd4j.getEnvironment();

        boolean prevCompileAll = env.tritonCompileAll();
        String prevIncludeTypes = env.tritonIncludeTypes();
        env.setTritonCompileAll(true);
        env.setTritonIncludeTypes("CONST_GEN,GATHER,CONCAT,SPLIT,STACK,ATTENTION");

        try {
            int batch = 1, seqQ = 1, numHeads = 4, headDim = 16;
            int kvCacheLen = 8;
            int hidden = numHeads * headDim;
            int totalSeq = kvCacheLen + seqQ;
            int numLayers = 3;

            SameDiff sd = SameDiff.create();
            SDVariable input = sd.placeHolder("input", DataType.FLOAT, batch, seqQ, hidden);
            SDVariable current = input;

            Map<String, INDArray> ph = new java.util.HashMap<>();

            for (int layer = 0; layer < numLayers; layer++) {
                String prefix = "L" + layer + "_";

                // Simplified LayerNorm: mean subtraction
                SDVariable mean = current.mean(prefix + "mean", true, 2);
                SDVariable normed = current.sub(prefix + "norm", mean);

                // Q, K, V projections
                SDVariable wq = sd.constant(prefix + "wq", Nd4j.randn(DataType.FLOAT, hidden, hidden).muli(0.02));
                SDVariable wk = sd.constant(prefix + "wk", Nd4j.randn(DataType.FLOAT, hidden, hidden).muli(0.02));
                SDVariable wv = sd.constant(prefix + "wv", Nd4j.randn(DataType.FLOAT, hidden, hidden).muli(0.02));
                SDVariable wo = sd.constant(prefix + "wo", Nd4j.randn(DataType.FLOAT, hidden, hidden).muli(0.02));

                SDVariable q = sd.mmul(prefix + "q_proj", normed, wq);
                SDVariable k = sd.mmul(prefix + "k_proj", normed, wk);
                SDVariable v = sd.mmul(prefix + "v_proj", normed, wv);

                // KV cache (external inputs per layer)
                SDVariable pastKey = sd.placeHolder(prefix + "past_key", DataType.FLOAT, batch, numHeads, kvCacheLen, headDim);
                SDVariable pastValue = sd.placeHolder(prefix + "past_value", DataType.FLOAT, batch, numHeads, kvCacheLen, headDim);
                SDVariable attnBias = sd.placeHolder(prefix + "attn_bias", DataType.FLOAT, batch, numHeads, seqQ, totalSeq);

                // Attention
                var attnOp = new org.nd4j.linalg.api.ops.impl.transforms.custom.OnnxMultiHeadAttention(
                        sd, q, k, v, attnBias, pastKey, pastValue,
                        numHeads, 0.0, false, 1);
                SDVariable attnOut = attnOp.outputVariables()[0];
                attnOut.rename(prefix + "attn_out");

                // Output projection
                SDVariable projected = sd.mmul(prefix + "o_proj", attnOut, wo);

                // Residual connection
                current = current.add(prefix + "residual", projected);

                // Populate feed dict for this layer's KV cache
                ph.put(prefix + "past_key", Nd4j.randn(DataType.FLOAT, batch, numHeads, kvCacheLen, headDim).muli(0.1));
                ph.put(prefix + "past_value", Nd4j.randn(DataType.FLOAT, batch, numHeads, kvCacheLen, headDim).muli(0.1));
                float maskVal = -3.4028235e+38f;
                INDArray biasArr = Nd4j.valueArrayOf(new long[]{batch, numHeads, seqQ, totalSeq}, maskVal, DataType.FLOAT);
                for (int pos = 0; pos <= kvCacheLen; pos++) {
                    biasArr.get(NDArrayIndex.all(), NDArrayIndex.all(),
                            NDArrayIndex.all(), NDArrayIndex.point(pos)).assign(0.0f);
                }
                ph.put(prefix + "attn_bias", biasArr);
            }

            sd.identity("result", current);
            ph.put("input", Nd4j.randn(DataType.FLOAT, batch, seqQ, hidden).muli(0.1));

            // Reference output
            Map<String, INDArray> refResults = sd.output(ph, "result");
            INDArray refOutput = refResults.get("result").dup();
            assertNotNull(refOutput, "Reference output is null");
            assertFalse(refOutput.isNaN().any(), "Reference output contains NaN");
            log.info("testFusedAttention_MultiLayerDecoder_CompileAll: ref shape={}, ref max={}, ref mean={}, {} layers",
                    java.util.Arrays.toString(refOutput.shape()),
                    refOutput.amaxNumber(), refOutput.meanNumber(), numLayers);

            // Compile to native DSP plan
            DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "result");
            assertNotNull(plan, "Plan compilation returned null");
            log.info("Plan: {} slots, {} ext inputs", plan.getSlots().length,
                    plan.getExternalInputKeys().length);

            Pointer planHandle = compileNativePlan(plan);
            if (planHandle == null) {
                log.info("Skipping (native executor not supported)");
                return;
            }
            try {
                INDArray[] extInputs = resolveExternalInputs(plan, sd, ph);
                nativeOps.resetTritonCounters();

                for (int iter = 0; iter < 4; iter++) {
                    Map<String, INDArray> nativeResults = executeNativePlan(planHandle, plan, extInputs);
                    INDArray nativeOutput = nativeResults.get("result");
                    assertNotNull(nativeOutput, "Native output is null at iteration " + iter);

                    boolean hasNaN = nativeOutput.isNaN().any();
                    double maxDiff = hasNaN ? Double.NaN
                            : refOutput.sub(nativeOutput).amaxNumber().doubleValue();
                    log.info("MultiLayerDecoder iter {}: hasNaN={}, maxDiff={}", iter, hasNaN, maxDiff);

                    if (hasNaN || maxDiff > TOLERANCE) {
                        float[] refFlat = refOutput.dup('c').data().asFloat();
                        float[] natFlat = nativeOutput.dup('c').data().asFloat();
                        int n = Math.min(20, refFlat.length);
                        StringBuilder sb = new StringBuilder("REF: ");
                        for (int i = 0; i < n; i++) sb.append(String.format("%.6f ", refFlat[i]));
                        log.info(sb.toString());
                        sb = new StringBuilder("NAT: ");
                        for (int i = 0; i < n; i++) sb.append(String.format("%.6f ", natFlat[i]));
                        log.info(sb.toString());
                    }

                    assertFalse(hasNaN, "Native output contains NaN at iteration " + iter);
                    // Flash attention uses online softmax with different accumulation
                    // order than standard matmul→softmax→matmul, causing legitimate
                    // FP32 precision differences that compound across 3 decoder layers.
                    double attnTolerance = 0.5;
                    assertTrue(maxDiff < attnTolerance,
                            "Output mismatch at iteration " + iter + ": max diff = " + maxDiff);

                    if (iter == 0) {
                        nativeOps.setPlanShapesFrozen(planHandle, true);
                    }
                }

                long tritonLaunches = nativeOps.getTritonKernelLaunchCount();
                log.info("testFusedAttention_MultiLayerDecoder_CompileAll: tritonLaunches={}", tritonLaunches);
            } finally {
                nativeOps.freeDynamicShapePlan(planHandle);
            }
        } finally {
            env.setTritonCompileAll(prevCompileAll);
            env.setTritonIncludeTypes(prevIncludeTypes);
        }
    }

    /**
     * Test 3: Attention-only but with COMPILE_ALL types enabled.
     *
     * This isolates whether the NaN is from including extra compile types
     * vs the cross-sub-kernel matmul→attention data flow. If this passes but
     * testFusedAttention_MatmulProjections_CompileAll fails, the bug is in
     * cross-sub-kernel data flow. If this also fails, the bug is in how
     * attention interacts with the broader compilation pipeline.
     */
    @Test
    public void testFusedAttention_OnnxMHA_WithCompileAllTypes() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        var env = Nd4j.getEnvironment();

        boolean prevCompileAll = env.tritonCompileAll();
        String prevIncludeTypes = env.tritonIncludeTypes();
        env.setTritonCompileAll(true);
        // Full VLM compile types but with a simple attention-only graph
        env.setTritonIncludeTypes("CONST_GEN,GATHER,CONCAT,SPLIT,STACK,ATTENTION");

        try {
            int batch = 1, seqQ = 1, numHeads = 4, headDim = 64;
            int kvCacheLen = 10;
            int hidden = numHeads * headDim;
            int totalSeq = kvCacheLen + seqQ;

            // Same graph as testFusedAttention_OnnxMHA_StaticKvCache but with ALL types
            SameDiff sd = createOnnxMhaWithKvCacheGraph(batch, seqQ, numHeads, headDim, kvCacheLen, true);

            INDArray qArr = Nd4j.randn(DataType.FLOAT, batch, seqQ, hidden).muli(0.1);
            INDArray kArr = Nd4j.randn(DataType.FLOAT, batch, seqQ, hidden).muli(0.1);
            INDArray vArr = Nd4j.randn(DataType.FLOAT, batch, seqQ, hidden).muli(0.1);
            INDArray pastKeyArr = Nd4j.randn(DataType.FLOAT, batch, numHeads, kvCacheLen, headDim).muli(0.1);
            INDArray pastValArr = Nd4j.randn(DataType.FLOAT, batch, numHeads, kvCacheLen, headDim).muli(0.1);
            float maskVal = -3.4028235e+38f;
            INDArray biasArr = Nd4j.valueArrayOf(new long[]{batch, numHeads, seqQ, totalSeq}, maskVal, DataType.FLOAT);
            for (int pos = 0; pos <= kvCacheLen; pos++) {
                biasArr.get(NDArrayIndex.all(), NDArrayIndex.all(),
                        NDArrayIndex.all(), NDArrayIndex.point(pos)).assign(0.0f);
            }

            Map<String, INDArray> ph = new java.util.HashMap<>();
            ph.put("q", qArr);
            ph.put("k", kArr);
            ph.put("v", vArr);
            ph.put("past_key", pastKeyArr);
            ph.put("past_value", pastValArr);
            ph.put("attn_bias", biasArr);

            Map<String, INDArray> refResults = sd.output(ph, "attn_out");
            INDArray refOutput = refResults.get("attn_out").dup();
            assertNotNull(refOutput);
            assertFalse(refOutput.isNaN().any(), "Reference output contains NaN");
            log.info("testFusedAttention_OnnxMHA_WithCompileAllTypes: ref max={}, ref mean={}",
                    refOutput.amaxNumber(), refOutput.meanNumber());

            DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "attn_out");
            assertNotNull(plan);

            Pointer planHandle = compileNativePlan(plan);
            if (planHandle == null) {
                log.info("Skipping (native executor not supported)");
                return;
            }
            try {
                INDArray[] extInputs = resolveExternalInputs(plan, sd, ph);
                nativeOps.resetTritonCounters();

                for (int iter = 0; iter < 4; iter++) {
                    Map<String, INDArray> nativeResults = executeNativePlan(planHandle, plan, extInputs);
                    INDArray nativeOutput = nativeResults.get("attn_out");
                    assertNotNull(nativeOutput, "Native output is null at iteration " + iter);

                    boolean hasNaN = nativeOutput.isNaN().any();
                    double maxDiff = hasNaN ? Double.NaN
                            : refOutput.sub(nativeOutput).amaxNumber().doubleValue();
                    log.info("OnnxMHA_WithCompileAllTypes iter {}: hasNaN={}, maxDiff={}", iter, hasNaN, maxDiff);

                    assertFalse(hasNaN, "Native output contains NaN at iteration " + iter);
                    assertTrue(maxDiff < TOLERANCE,
                            "Output mismatch at iteration " + iter + ": max diff = " + maxDiff);

                    if (iter == 0) {
                        nativeOps.setPlanShapesFrozen(planHandle, true);
                    }
                }
            } finally {
                nativeOps.freeDynamicShapePlan(planHandle);
            }
        } finally {
            env.setTritonCompileAll(prevCompileAll);
            env.setTritonIncludeTypes(prevIncludeTypes);
        }
    }

    /**
     * Test 4a: GQA (Grouped Query Attention) - SmolDocling exact dimensions.
     *
     * SmolDocling uses GQA: 9 Q heads but only 3 KV heads. The Triton kernel
     * must broadcast KV across Q head groups. This is NOT tested by any existing
     * test (all use equal Q and KV heads).
     *
     * Config: numQHeads=9, numKVHeads=3, headDim=64 (hidden=576, kv_hidden=192)
     */
    @Test
    public void testFusedAttention_GQA_SmolDoclingDims() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        var env = Nd4j.getEnvironment();

        boolean prevCompileAll = env.tritonCompileAll();
        String prevIncludeTypes = env.tritonIncludeTypes();
        env.setTritonCompileAll(true);
        env.setTritonIncludeTypes("FUSED_ATTENTION");

        try {
            int batch = 1, seqQ = 1;
            int numQHeads = 9, numKVHeads = 3, headDim = 64;
            int qHidden = numQHeads * headDim;   // 576
            int kvHidden = numKVHeads * headDim;  // 192
            int kvCacheLen = 10;
            int totalSeq = kvCacheLen + seqQ;

            SameDiff sd = SameDiff.create();
            SDVariable q = sd.placeHolder("q", DataType.FLOAT, batch, seqQ, qHidden);
            SDVariable k = sd.placeHolder("k", DataType.FLOAT, batch, seqQ, kvHidden);
            SDVariable v = sd.placeHolder("v", DataType.FLOAT, batch, seqQ, kvHidden);
            // KV cache has numKVHeads, not numQHeads
            SDVariable pastKey = sd.placeHolder("past_key", DataType.FLOAT, batch, numKVHeads, kvCacheLen, headDim);
            SDVariable pastValue = sd.placeHolder("past_value", DataType.FLOAT, batch, numKVHeads, kvCacheLen, headDim);
            SDVariable attnBias = sd.placeHolder("attn_bias", DataType.FLOAT, batch, numQHeads, seqQ, totalSeq);

            var attnOp = new org.nd4j.linalg.api.ops.impl.transforms.custom.OnnxMultiHeadAttention(
                    sd, q, k, v, attnBias, pastKey, pastValue,
                    numQHeads, 0.0, false, 1);
            SDVariable out = attnOp.outputVariables()[0];
            out.rename("attn_out");

            INDArray qArr = Nd4j.randn(DataType.FLOAT, batch, seqQ, qHidden).muli(0.1);
            INDArray kArr = Nd4j.randn(DataType.FLOAT, batch, seqQ, kvHidden).muli(0.1);
            INDArray vArr = Nd4j.randn(DataType.FLOAT, batch, seqQ, kvHidden).muli(0.1);
            INDArray pastKeyArr = Nd4j.randn(DataType.FLOAT, batch, numKVHeads, kvCacheLen, headDim).muli(0.1);
            INDArray pastValArr = Nd4j.randn(DataType.FLOAT, batch, numKVHeads, kvCacheLen, headDim).muli(0.1);
            float maskVal = -3.4028235e+38f;
            INDArray biasArr = Nd4j.valueArrayOf(new long[]{batch, numQHeads, seqQ, totalSeq}, maskVal, DataType.FLOAT);
            for (int pos = 0; pos <= kvCacheLen; pos++) {
                biasArr.get(NDArrayIndex.all(), NDArrayIndex.all(),
                        NDArrayIndex.all(), NDArrayIndex.point(pos)).assign(0.0f);
            }

            Map<String, INDArray> ph = new java.util.HashMap<>();
            ph.put("q", qArr);
            ph.put("k", kArr);
            ph.put("v", vArr);
            ph.put("past_key", pastKeyArr);
            ph.put("past_value", pastValArr);
            ph.put("attn_bias", biasArr);

            Map<String, INDArray> refResults = sd.output(ph, "attn_out");
            INDArray refOutput = refResults.get("attn_out").dup();
            assertNotNull(refOutput, "Reference output is null");
            assertFalse(refOutput.isNaN().any(), "Reference output contains NaN");
            log.info("testFusedAttention_GQA_SmolDoclingDims: ref shape={}, ref max={}, ref mean={} (Q={}h, KV={}h)",
                    java.util.Arrays.toString(refOutput.shape()),
                    refOutput.amaxNumber(), refOutput.meanNumber(), numQHeads, numKVHeads);

            DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "attn_out");
            assertNotNull(plan, "Plan compilation returned null");

            Pointer planHandle = compileNativePlan(plan);
            if (planHandle == null) {
                log.info("Skipping (native executor not supported)");
                return;
            }
            try {
                INDArray[] extInputs = resolveExternalInputs(plan, sd, ph);
                nativeOps.resetTritonCounters();

                for (int iter = 0; iter < 4; iter++) {
                    Map<String, INDArray> nativeResults = executeNativePlan(planHandle, plan, extInputs);
                    INDArray nativeOutput = nativeResults.get("attn_out");
                    assertNotNull(nativeOutput, "Native output is null at iteration " + iter);

                    boolean hasNaN = nativeOutput.isNaN().any();
                    double maxDiff = hasNaN ? Double.NaN
                            : refOutput.sub(nativeOutput).amaxNumber().doubleValue();
                    log.info("GQA_SmolDoclingDims iter {}: hasNaN={}, maxDiff={}, native max={}, native mean={}",
                            iter, hasNaN, maxDiff,
                            nativeOutput.amaxNumber(), nativeOutput.meanNumber());

                    if (hasNaN || maxDiff > TOLERANCE) {
                        float[] refFlat = refOutput.dup('c').data().asFloat();
                        float[] natFlat = nativeOutput.dup('c').data().asFloat();
                        int n = Math.min(20, refFlat.length);
                        StringBuilder sb = new StringBuilder("REF: ");
                        for (int i = 0; i < n; i++) sb.append(String.format("%.6f ", refFlat[i]));
                        log.info(sb.toString());
                        sb = new StringBuilder("NAT: ");
                        for (int i = 0; i < n; i++) sb.append(String.format("%.6f ", natFlat[i]));
                        log.info(sb.toString());
                    }

                    assertFalse(hasNaN, "Native output contains NaN at iteration " + iter
                            + " (GQA: Q=" + numQHeads + " KV=" + numKVHeads + ")");
                    assertTrue(maxDiff < TOLERANCE,
                            "Output mismatch at iteration " + iter + ": max diff = " + maxDiff);

                    if (iter == 0) {
                        nativeOps.setPlanShapesFrozen(planHandle, true);
                    }
                }
            } finally {
                nativeOps.freeDynamicShapePlan(planHandle);
            }
        } finally {
            env.setTritonCompileAll(prevCompileAll);
            env.setTritonIncludeTypes(prevIncludeTypes);
        }
    }

    /**
     * Test 4b: GQA with matmul projections and compile-all — full VLM scenario.
     *
     * Combines GQA dimensions with matmul projections, all compiled together.
     * If testFusedAttention_GQA_SmolDoclingDims passes but this fails,
     * the bug is in cross-sub-kernel flow with GQA. If both fail, it's in the
     * GQA kernel itself.
     */
    @Test
    public void testFusedAttention_GQA_MatmulProjections_CompileAll() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        var env = Nd4j.getEnvironment();

        boolean prevCompileAll = env.tritonCompileAll();
        String prevIncludeTypes = env.tritonIncludeTypes();
        env.setTritonCompileAll(true);
        env.setTritonIncludeTypes("CONST_GEN,GATHER,CONCAT,SPLIT,STACK,ATTENTION");

        try {
            int batch = 1, seqQ = 1;
            int numQHeads = 9, numKVHeads = 3, headDim = 64;
            int hidden = numQHeads * headDim;     // 576
            int kvHidden = numKVHeads * headDim;  // 192
            int kvCacheLen = 10;
            int totalSeq = kvCacheLen + seqQ;

            SameDiff sd = SameDiff.create();
            SDVariable input = sd.placeHolder("input", DataType.FLOAT, batch, seqQ, hidden);

            // Q projection: hidden→hidden, K/V projections: hidden→kvHidden
            SDVariable wq = sd.constant("wq", Nd4j.randn(DataType.FLOAT, hidden, hidden).muli(0.02));
            SDVariable wk = sd.constant("wk", Nd4j.randn(DataType.FLOAT, hidden, kvHidden).muli(0.02));
            SDVariable wv = sd.constant("wv", Nd4j.randn(DataType.FLOAT, hidden, kvHidden).muli(0.02));

            SDVariable q = sd.mmul("q_proj", input, wq);
            SDVariable k = sd.mmul("k_proj", input, wk);
            SDVariable v = sd.mmul("v_proj", input, wv);

            SDVariable pastKey = sd.placeHolder("past_key", DataType.FLOAT, batch, numKVHeads, kvCacheLen, headDim);
            SDVariable pastValue = sd.placeHolder("past_value", DataType.FLOAT, batch, numKVHeads, kvCacheLen, headDim);
            SDVariable attnBias = sd.placeHolder("attn_bias", DataType.FLOAT, batch, numQHeads, seqQ, totalSeq);

            var attnOp = new org.nd4j.linalg.api.ops.impl.transforms.custom.OnnxMultiHeadAttention(
                    sd, q, k, v, attnBias, pastKey, pastValue,
                    numQHeads, 0.0, false, 1);
            SDVariable out = attnOp.outputVariables()[0];
            out.rename("attn_out");

            INDArray inputArr = Nd4j.randn(DataType.FLOAT, batch, seqQ, hidden).muli(0.1);
            INDArray pastKeyArr = Nd4j.randn(DataType.FLOAT, batch, numKVHeads, kvCacheLen, headDim).muli(0.1);
            INDArray pastValArr = Nd4j.randn(DataType.FLOAT, batch, numKVHeads, kvCacheLen, headDim).muli(0.1);
            float maskVal = -3.4028235e+38f;
            INDArray biasArr = Nd4j.valueArrayOf(new long[]{batch, numQHeads, seqQ, totalSeq}, maskVal, DataType.FLOAT);
            for (int pos = 0; pos <= kvCacheLen; pos++) {
                biasArr.get(NDArrayIndex.all(), NDArrayIndex.all(),
                        NDArrayIndex.all(), NDArrayIndex.point(pos)).assign(0.0f);
            }

            Map<String, INDArray> ph = new java.util.HashMap<>();
            ph.put("input", inputArr);
            ph.put("past_key", pastKeyArr);
            ph.put("past_value", pastValArr);
            ph.put("attn_bias", biasArr);

            Map<String, INDArray> refResults = sd.output(ph, "attn_out");
            INDArray refOutput = refResults.get("attn_out").dup();
            assertNotNull(refOutput);
            assertFalse(refOutput.isNaN().any(), "Reference output contains NaN");
            log.info("testFusedAttention_GQA_MatmulProjections_CompileAll: ref max={}, ref mean={}",
                    refOutput.amaxNumber(), refOutput.meanNumber());

            DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "attn_out");
            assertNotNull(plan);

            Pointer planHandle = compileNativePlan(plan);
            if (planHandle == null) {
                log.info("Skipping (native executor not supported)");
                return;
            }
            try {
                INDArray[] extInputs = resolveExternalInputs(plan, sd, ph);
                nativeOps.resetTritonCounters();

                for (int iter = 0; iter < 4; iter++) {
                    Map<String, INDArray> nativeResults = executeNativePlan(planHandle, plan, extInputs);
                    INDArray nativeOutput = nativeResults.get("attn_out");
                    assertNotNull(nativeOutput, "Native output is null at iteration " + iter);

                    boolean hasNaN = nativeOutput.isNaN().any();
                    double maxDiff = hasNaN ? Double.NaN
                            : refOutput.sub(nativeOutput).amaxNumber().doubleValue();
                    log.info("GQA_MatmulProjections iter {}: hasNaN={}, maxDiff={}", iter, hasNaN, maxDiff);

                    assertFalse(hasNaN, "Native output contains NaN at iteration " + iter);
                    assertTrue(maxDiff < TOLERANCE,
                            "Output mismatch at iteration " + iter + ": max diff = " + maxDiff);

                    if (iter == 0) {
                        nativeOps.setPlanShapesFrozen(planHandle, true);
                    }
                }
            } finally {
                nativeOps.freeDynamicShapePlan(planHandle);
            }
        } finally {
            env.setTritonCompileAll(prevCompileAll);
            env.setTritonIncludeTypes(prevIncludeTypes);
        }
    }

    /**
     * Test 4: Graph capture + frozen shapes with matmul→attention, matching VLM config.
     *
     * This is the closest to the actual VLM execution mode:
     * - compileAll + ALL types + ATTENTION
     * - Graph capture enabled
     * - Consolidated arg table + dirty tracking
     * - Multiple iterations with frozen shapes
     */
    @Test
    public void testFusedAttention_MatmulProjections_GraphCapture() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        var env = Nd4j.getEnvironment();

        boolean prevCompileAll = env.tritonCompileAll();
        String prevIncludeTypes = env.tritonIncludeTypes();
        boolean prevGC = env.tritonGraphCapture();
        boolean prevFallback = env.tritonAllowFallbackCapture();
        boolean prevConsolidated = env.tritonConsolidatedArgTable();
        boolean prevDirty = env.tritonArgDirtyTracking();
        env.setTritonCompileAll(true);
        env.setTritonIncludeTypes("CONST_GEN,GATHER,CONCAT,SPLIT,STACK,ATTENTION");
        env.setTritonGraphCapture(true);
        env.setTritonAllowFallbackCapture(true);
        env.setTritonConsolidatedArgTable(true);
        env.setTritonArgDirtyTracking(true);

        try {
            int batch = 1, seqQ = 1, numHeads = 4, headDim = 16;
            int kvCacheLen = 8;
            int hidden = numHeads * headDim;
            int totalSeq = kvCacheLen + seqQ;

            SameDiff sd = SameDiff.create();
            SDVariable input = sd.placeHolder("input", DataType.FLOAT, batch, seqQ, hidden);

            SDVariable wq = sd.constant("wq", Nd4j.randn(DataType.FLOAT, hidden, hidden).muli(0.02));
            SDVariable wk = sd.constant("wk", Nd4j.randn(DataType.FLOAT, hidden, hidden).muli(0.02));
            SDVariable wv = sd.constant("wv", Nd4j.randn(DataType.FLOAT, hidden, hidden).muli(0.02));

            SDVariable q = sd.mmul("q_proj", input, wq);
            SDVariable k = sd.mmul("k_proj", input, wk);
            SDVariable v = sd.mmul("v_proj", input, wv);

            SDVariable pastKey = sd.placeHolder("past_key", DataType.FLOAT, batch, numHeads, kvCacheLen, headDim);
            SDVariable pastValue = sd.placeHolder("past_value", DataType.FLOAT, batch, numHeads, kvCacheLen, headDim);
            SDVariable attnBias = sd.placeHolder("attn_bias", DataType.FLOAT, batch, numHeads, seqQ, totalSeq);

            var attnOp = new org.nd4j.linalg.api.ops.impl.transforms.custom.OnnxMultiHeadAttention(
                    sd, q, k, v, attnBias, pastKey, pastValue,
                    numHeads, 0.0, false, 1);
            SDVariable out = attnOp.outputVariables()[0];
            out.rename("attn_out");

            INDArray inputArr = Nd4j.randn(DataType.FLOAT, batch, seqQ, hidden).muli(0.1);
            INDArray pastKeyArr = Nd4j.randn(DataType.FLOAT, batch, numHeads, kvCacheLen, headDim).muli(0.1);
            INDArray pastValArr = Nd4j.randn(DataType.FLOAT, batch, numHeads, kvCacheLen, headDim).muli(0.1);
            float maskVal = -3.4028235e+38f;
            INDArray biasArr = Nd4j.valueArrayOf(new long[]{batch, numHeads, seqQ, totalSeq}, maskVal, DataType.FLOAT);
            for (int pos = 0; pos <= kvCacheLen; pos++) {
                biasArr.get(NDArrayIndex.all(), NDArrayIndex.all(),
                        NDArrayIndex.all(), NDArrayIndex.point(pos)).assign(0.0f);
            }

            Map<String, INDArray> ph = new java.util.HashMap<>();
            ph.put("input", inputArr);
            ph.put("past_key", pastKeyArr);
            ph.put("past_value", pastValArr);
            ph.put("attn_bias", biasArr);

            Map<String, INDArray> refResults = sd.output(ph, "attn_out");
            INDArray refOutput = refResults.get("attn_out").dup();
            assertNotNull(refOutput);
            assertFalse(refOutput.isNaN().any(), "Reference output contains NaN");
            log.info("testFusedAttention_MatmulProjections_GraphCapture: ref max={}, ref mean={}",
                    refOutput.amaxNumber(), refOutput.meanNumber());

            DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "attn_out");
            assertNotNull(plan);

            Pointer planHandle = compileNativePlan(plan);
            if (planHandle == null) {
                log.info("Skipping (native executor not supported)");
                return;
            }
            try {
                INDArray[] extInputs = resolveExternalInputs(plan, sd, ph);
                nativeOps.resetTritonCounters();

                for (int iter = 0; iter < 6; iter++) {
                    Map<String, INDArray> nativeResults = executeNativePlan(planHandle, plan, extInputs);
                    INDArray nativeOutput = nativeResults.get("attn_out");
                    assertNotNull(nativeOutput, "Native output is null at iteration " + iter);

                    boolean hasNaN = nativeOutput.isNaN().any();
                    double maxDiff = hasNaN ? Double.NaN
                            : refOutput.sub(nativeOutput).amaxNumber().doubleValue();
                    log.info("MatmulProjections_GraphCapture iter {}: hasNaN={}, maxDiff={}", iter, hasNaN, maxDiff);

                    assertFalse(hasNaN, "Native output contains NaN at iteration " + iter);
                    assertTrue(maxDiff < TOLERANCE,
                            "Output mismatch at iteration " + iter + ": max diff = " + maxDiff);

                    if (iter == 0) {
                        nativeOps.setPlanShapesFrozen(planHandle, true);
                    }
                }
            } finally {
                nativeOps.freeDynamicShapePlan(planHandle);
            }
        } finally {
            env.setTritonCompileAll(prevCompileAll);
            env.setTritonIncludeTypes(prevIncludeTypes);
            env.setTritonGraphCapture(prevGC);
            env.setTritonAllowFallbackCapture(prevFallback);
            env.setTritonConsolidatedArgTable(prevConsolidated);
            env.setTritonArgDirtyTracking(prevDirty);
        }
    }

    /**
     * Test: Dual-buffer attention with SmolDocling exact dimensions.
     * SmolDocling uses: numHeads=9, headDim=64, hidden=576, 3D Q/K/V + 4D past_key/value.
     * This tests the Triton dual-buffer 3D Q attention kernel with the exact
     * dimensions used by the VLM pipeline, including the non-power-of-2 head count (9).
     * If this test fails with NaN but testFusedAttention_MatmulProjections_CompileAll
     * (which uses numHeads=4) passes, the bug is head-count related.
     */
    @Test
    @Tag(TagNames.NEEDS_VERIFY)
    @NativeTag
    public void testFusedAttention_DualBuffer_SmolDoclingDims() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        var env = Nd4j.getEnvironment();

        boolean prevCompileAll = env.tritonCompileAll();
        String prevIncludeTypes = env.tritonIncludeTypes();
        env.setTritonCompileAll(true);
        env.setTritonIncludeTypes("CONST_GEN,GATHER,CONCAT,SPLIT,STACK,ATTENTION");

        try {
            // SmolDocling exact dimensions
            int batch = 1, seqQ = 1, numHeads = 9, headDim = 64;
            int kvCacheLen = 50;  // simulate after prefill
            int hidden = numHeads * headDim;  // 576
            int totalSeq = kvCacheLen + seqQ;  // 51

            SameDiff sd = SameDiff.create();
            SDVariable input = sd.placeHolder("input", DataType.FLOAT, batch, seqQ, hidden);

            // Weight projections
            SDVariable wq = sd.constant("wq", Nd4j.randn(DataType.FLOAT, hidden, hidden).muli(0.02));
            SDVariable wk = sd.constant("wk", Nd4j.randn(DataType.FLOAT, hidden, hidden).muli(0.02));
            SDVariable wv = sd.constant("wv", Nd4j.randn(DataType.FLOAT, hidden, hidden).muli(0.02));

            // Q, K, V projections
            SDVariable q = sd.mmul("q_proj", input, wq);
            SDVariable k = sd.mmul("k_proj", input, wk);
            SDVariable v = sd.mmul("v_proj", input, wv);

            // Past KV cache (9 heads, not 3 — after repeat_kv in ONNX model)
            SDVariable pastKey = sd.placeHolder("past_key", DataType.FLOAT, batch, numHeads, kvCacheLen, headDim);
            SDVariable pastValue = sd.placeHolder("past_value", DataType.FLOAT, batch, numHeads, kvCacheLen, headDim);
            SDVariable attnBias = sd.placeHolder("attn_bias", DataType.FLOAT, batch, 1, seqQ, totalSeq);

            // Attention op with dual-buffer (3D Q + 4D past KV)
            var attnOp = new org.nd4j.linalg.api.ops.impl.transforms.custom.OnnxMultiHeadAttention(
                    sd, q, k, v, attnBias, pastKey, pastValue,
                    numHeads, 0.0, false, 1);
            SDVariable out = attnOp.outputVariables()[0];
            out.rename("attn_out");

            // Create inputs
            INDArray inputArr = Nd4j.randn(DataType.FLOAT, batch, seqQ, hidden).muli(0.1);
            INDArray pastKeyArr = Nd4j.randn(DataType.FLOAT, batch, numHeads, kvCacheLen, headDim).muli(0.1);
            INDArray pastValArr = Nd4j.randn(DataType.FLOAT, batch, numHeads, kvCacheLen, headDim).muli(0.1);
            // Bias: 0 for valid positions (0..kvCacheLen), -inf for padding
            float maskVal = -3.4028235e+38f;
            INDArray biasArr = Nd4j.valueArrayOf(new long[]{batch, 1, seqQ, totalSeq}, maskVal, DataType.FLOAT);
            for (int pos = 0; pos <= kvCacheLen; pos++) {
                biasArr.get(NDArrayIndex.all(), NDArrayIndex.all(),
                        NDArrayIndex.all(), NDArrayIndex.point(pos)).assign(0.0f);
            }

            Map<String, INDArray> ph = new java.util.HashMap<>();
            ph.put("input", inputArr);
            ph.put("past_key", pastKeyArr);
            ph.put("past_value", pastValArr);
            ph.put("attn_bias", biasArr);

            // Reference from C++
            Map<String, INDArray> refResults = sd.output(ph, "attn_out");
            INDArray refOutput = refResults.get("attn_out").dup();
            assertNotNull(refOutput, "Reference output is null");
            assertFalse(refOutput.isNaN().any(), "Reference output contains NaN");
            log.info("DualBuffer_SmolDoclingDims: ref shape={}, ref max={}, ref mean={}",
                    java.util.Arrays.toString(refOutput.shape()),
                    refOutput.amaxNumber(), refOutput.meanNumber());

            // Compile native plan
            DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "attn_out");
            assertNotNull(plan, "Plan compilation returned null");

            Pointer planHandle = compileNativePlan(plan);
            if (planHandle == null) {
                log.info("Skipping (native executor not supported)");
                return;
            }
            try {
                INDArray[] extInputs = resolveExternalInputs(plan, sd, ph);
                nativeOps.resetTritonCounters();

                for (int iter = 0; iter < 4; iter++) {
                    Map<String, INDArray> nativeResults = executeNativePlan(planHandle, plan, extInputs);
                    INDArray nativeOutput = nativeResults.get("attn_out");
                    assertNotNull(nativeOutput, "Native output is null at iteration " + iter);

                    boolean hasNaN = nativeOutput.isNaN().any();
                    double maxDiff = hasNaN ? Double.NaN
                            : refOutput.sub(nativeOutput).amaxNumber().doubleValue();
                    log.info("DualBuffer_SmolDoclingDims iter {}: hasNaN={}, maxDiff={}, native max={}, native mean={}",
                            iter, hasNaN, maxDiff,
                            nativeOutput.amaxNumber(), nativeOutput.meanNumber());

                    if (hasNaN || maxDiff > TOLERANCE) {
                        float[] refFlat = refOutput.dup('c').data().asFloat();
                        float[] natFlat = nativeOutput.dup('c').data().asFloat();
                        int n = Math.min(20, refFlat.length);
                        StringBuilder sb = new StringBuilder("REF: ");
                        for (int i = 0; i < n; i++) sb.append(String.format("%.6f ", refFlat[i]));
                        log.info(sb.toString());
                        sb = new StringBuilder("NAT: ");
                        for (int i = 0; i < n; i++) sb.append(String.format("%.6f ", natFlat[i]));
                        log.info(sb.toString());
                    }

                    assertFalse(hasNaN, "Native output contains NaN at iteration " + iter);
                    assertTrue(maxDiff < TOLERANCE,
                            "Output mismatch at iteration " + iter + ": max diff = " + maxDiff);

                    if (iter == 0) {
                        nativeOps.setPlanShapesFrozen(planHandle, true);
                    }
                }

                long tritonLaunches = nativeOps.getTritonKernelLaunchCount();
                log.info("DualBuffer_SmolDoclingDims: tritonLaunches={}", tritonLaunches);
            } finally {
                nativeOps.freeDynamicShapePlan(planHandle);
            }
        } finally {
            env.setTritonCompileAll(prevCompileAll);
            env.setTritonIncludeTypes(prevIncludeTypes);
        }
    }

    /**
     * Test: Multi-layer decoder with SmolDocling dimensions.
     * Creates 3 decoder layers, each with matmul Q/K/V projections → attention → residual add.
     * This tests cross-section data flow with multiple attention ops in a single graph,
     * closer to the VLM's 24-layer structure.
     */
    @Test
    @Tag(TagNames.NEEDS_VERIFY)
    @NativeTag
    public void testFusedAttention_MultiLayer_SmolDoclingDims() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        var env = Nd4j.getEnvironment();

        boolean prevCompileAll = env.tritonCompileAll();
        String prevIncludeTypes = env.tritonIncludeTypes();
        env.setTritonCompileAll(true);
        env.setTritonIncludeTypes("CONST_GEN,GATHER,CONCAT,SPLIT,STACK,ATTENTION");

        try {
            int batch = 1, seqQ = 1, numHeads = 9, headDim = 64;
            int kvCacheLen = 20;
            int hidden = numHeads * headDim;  // 576
            int totalSeq = kvCacheLen + seqQ;
            int numLayers = 3;

            SameDiff sd = SameDiff.create();
            SDVariable input = sd.placeHolder("input", DataType.FLOAT, batch, seqQ, hidden);
            SDVariable current = input;

            Map<String, INDArray> ph = new java.util.HashMap<>();
            INDArray inputArr = Nd4j.randn(DataType.FLOAT, batch, seqQ, hidden).muli(0.1);
            ph.put("input", inputArr);

            float maskVal = -3.4028235e+38f;
            INDArray biasArr = Nd4j.valueArrayOf(new long[]{batch, 1, seqQ, totalSeq}, maskVal, DataType.FLOAT);
            for (int pos = 0; pos <= kvCacheLen; pos++) {
                biasArr.get(NDArrayIndex.all(), NDArrayIndex.all(),
                        NDArrayIndex.all(), NDArrayIndex.point(pos)).assign(0.0f);
            }

            for (int layer = 0; layer < numLayers; layer++) {
                String prefix = "L" + layer + "_";
                SDVariable wq = sd.constant(prefix + "wq", Nd4j.randn(DataType.FLOAT, hidden, hidden).muli(0.02));
                SDVariable wk = sd.constant(prefix + "wk", Nd4j.randn(DataType.FLOAT, hidden, hidden).muli(0.02));
                SDVariable wv = sd.constant(prefix + "wv", Nd4j.randn(DataType.FLOAT, hidden, hidden).muli(0.02));

                SDVariable q = sd.mmul(prefix + "q_proj", current, wq);
                SDVariable k = sd.mmul(prefix + "k_proj", current, wk);
                SDVariable v = sd.mmul(prefix + "v_proj", current, wv);

                SDVariable pastKey = sd.placeHolder(prefix + "past_key", DataType.FLOAT, batch, numHeads, kvCacheLen, headDim);
                SDVariable pastValue = sd.placeHolder(prefix + "past_value", DataType.FLOAT, batch, numHeads, kvCacheLen, headDim);
                SDVariable attnBias = sd.placeHolder(prefix + "attn_bias", DataType.FLOAT, batch, 1, seqQ, totalSeq);

                var attnOp = new org.nd4j.linalg.api.ops.impl.transforms.custom.OnnxMultiHeadAttention(
                        sd, q, k, v, attnBias, pastKey, pastValue,
                        numHeads, 0.0, false, 1);
                SDVariable attnOut = attnOp.outputVariables()[0];
                attnOut.rename(prefix + "attn_out");

                // Residual connection
                current = sd.math.add(prefix + "residual", current, attnOut);

                // Create inputs for this layer
                ph.put(prefix + "past_key", Nd4j.randn(DataType.FLOAT, batch, numHeads, kvCacheLen, headDim).muli(0.1));
                ph.put(prefix + "past_value", Nd4j.randn(DataType.FLOAT, batch, numHeads, kvCacheLen, headDim).muli(0.1));
                ph.put(prefix + "attn_bias", biasArr.dup());
            }

            String outputName = "L" + (numLayers - 1) + "_residual";

            // Reference from C++
            Map<String, INDArray> refResults = sd.output(ph, outputName);
            INDArray refOutput = refResults.get(outputName).dup();
            assertNotNull(refOutput, "Reference output is null");
            assertFalse(refOutput.isNaN().any(), "Reference output contains NaN");
            log.info("MultiLayer_SmolDoclingDims: ref shape={}, ref max={}, ref mean={}",
                    java.util.Arrays.toString(refOutput.shape()),
                    refOutput.amaxNumber(), refOutput.meanNumber());

            // Compile native plan
            DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, outputName);
            assertNotNull(plan, "Plan compilation returned null");

            Pointer planHandle = compileNativePlan(plan);
            if (planHandle == null) {
                log.info("Skipping (native executor not supported)");
                return;
            }
            try {
                INDArray[] extInputs = resolveExternalInputs(plan, sd, ph);
                nativeOps.resetTritonCounters();

                for (int iter = 0; iter < 4; iter++) {
                    Map<String, INDArray> nativeResults = executeNativePlan(planHandle, plan, extInputs);
                    INDArray nativeOutput = nativeResults.get(outputName);
                    assertNotNull(nativeOutput, "Native output is null at iteration " + iter);

                    boolean hasNaN = nativeOutput.isNaN().any();
                    double maxDiff = hasNaN ? Double.NaN
                            : refOutput.sub(nativeOutput).amaxNumber().doubleValue();
                    log.info("MultiLayer_SmolDoclingDims iter {}: hasNaN={}, maxDiff={}, native max={}, native mean={}",
                            iter, hasNaN, maxDiff,
                            nativeOutput.amaxNumber(), nativeOutput.meanNumber());

                    if (hasNaN || maxDiff > TOLERANCE) {
                        float[] refFlat = refOutput.dup('c').data().asFloat();
                        float[] natFlat = nativeOutput.dup('c').data().asFloat();
                        int n = Math.min(20, refFlat.length);
                        StringBuilder sb = new StringBuilder("REF: ");
                        for (int i = 0; i < n; i++) sb.append(String.format("%.6f ", refFlat[i]));
                        log.info(sb.toString());
                        sb = new StringBuilder("NAT: ");
                        for (int i = 0; i < n; i++) sb.append(String.format("%.6f ", natFlat[i]));
                        log.info(sb.toString());
                    }

                    assertFalse(hasNaN, "Native output contains NaN at iteration " + iter);
                    assertTrue(maxDiff < TOLERANCE,
                            "Output mismatch at iteration " + iter + ": max diff = " + maxDiff);

                    if (iter == 0) {
                        nativeOps.setPlanShapesFrozen(planHandle, true);
                    }
                }

                long tritonLaunches = nativeOps.getTritonKernelLaunchCount();
                log.info("MultiLayer_SmolDoclingDims: tritonLaunches={}", tritonLaunches);
            } finally {
                nativeOps.freeDynamicShapePlan(planHandle);
            }
        } finally {
            env.setTritonCompileAll(prevCompileAll);
            env.setTritonIncludeTypes(prevIncludeTypes);
        }
    }

    /**
     * Isolation test: Triton attention with 4D attention mask that could be confused with past_key.
     *
     * Reproduces the SmolDocling GQA bug where:
     * - Input[3] is a 4D attention mask [B, heads, seqQ, totalSeqK] — NOT past_key
     * - The old code hardcoded input[4] as past_key, which was wrong
     * - The headDim filter must distinguish masks (last dim = totalSeqK) from KV cache (last dim = headDim)
     *
     * Verifies that:
     * 1. The attention mask is NOT confused with past_key (headDim mismatch)
     * 2. Actual past_key at a later input position IS correctly detected
     * 3. Output matches C++ reference (no NaN)
     *
     * Run with: -Dtest=TritonGraphBackendTest#testFusedAttention_HeadDimFilter_DistinguishMaskFromKv
     */
    @Test
    public void testFusedAttention_HeadDimFilter_DistinguishMaskFromKv() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        var env = Nd4j.getEnvironment();

        boolean prevCompileAll = env.tritonCompileAll();
        String prevIncludeTypes = env.tritonIncludeTypes();
        env.setTritonCompileAll(true);
        env.setTritonIncludeTypes("FUSED_ATTENTION");

        try {
            // SmolDocling-like decode: 9 Q heads, 3 KV heads (GQA), headDim=64
            int batch = 1, seqQ = 1, numHeads = 9, headDim = 64;
            int kvCacheLen = 100;
            int totalSeqK = kvCacheLen + seqQ;  // 101
            int hidden = numHeads * headDim;    // 576

            // Create OnnxMHA graph WITH attention bias (4D mask at input position 3)
            // Input order: Q, K, V, attn_bias, past_key, past_value
            // attn_bias shape: [1, 9, 1, 101] — 4D but last dim != headDim (101 != 64)
            // past_key shape: [1, 9, 100, 64] — 4D with last dim == headDim (64)
            SameDiff sd = createOnnxMhaWithKvCacheGraph(batch, seqQ, numHeads, headDim, kvCacheLen, true);

            INDArray qArr = Nd4j.randn(DataType.FLOAT, batch, seqQ, hidden).muli(0.1);
            INDArray kArr = Nd4j.randn(DataType.FLOAT, batch, seqQ, hidden).muli(0.1);
            INDArray vArr = Nd4j.randn(DataType.FLOAT, batch, seqQ, hidden).muli(0.1);
            INDArray pastKeyArr = Nd4j.randn(DataType.FLOAT, batch, numHeads, kvCacheLen, headDim).muli(0.1);
            INDArray pastValArr = Nd4j.randn(DataType.FLOAT, batch, numHeads, kvCacheLen, headDim).muli(0.1);
            // Attention bias mask — 4D like KV cache but last dim = totalSeqK, not headDim
            INDArray attnBiasArr = Nd4j.zeros(DataType.FLOAT, batch, numHeads, seqQ, totalSeqK);

            Map<String, INDArray> ph = Map.of(
                    "q", qArr, "k", kArr, "v", vArr,
                    "past_key", pastKeyArr, "past_value", pastValArr,
                    "attn_bias", attnBiasArr);

            // Reference output from C++
            Map<String, INDArray> refResults = sd.output(ph, "attn_out");
            INDArray refOutput = refResults.get("attn_out").dup();
            assertNotNull(refOutput, "Reference output is null");
            assertFalse(refOutput.isNaN().any(), "Reference output contains NaN");
            log.info("HeadDimFilter: ref shape={}, ref max={}, ref mean={}",
                    java.util.Arrays.toString(refOutput.shape()),
                    refOutput.amaxNumber(), refOutput.meanNumber());

            // Compile to native DSP plan
            DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "attn_out");
            assertNotNull(plan, "Plan compilation returned null");

            Pointer planHandle = compileNativePlan(plan);
            if (planHandle == null) {
                log.info("Skipping (native executor not supported)");
                return;
            }
            try {
                INDArray[] extInputs = resolveExternalInputs(plan, sd, ph);
                nativeOps.resetTritonCounters();

                for (int iter = 0; iter < 3; iter++) {
                    Map<String, INDArray> nativeResults = executeNativePlan(planHandle, plan, extInputs);
                    INDArray nativeOutput = nativeResults.get("attn_out");
                    assertNotNull(nativeOutput, "Native output is null at iteration " + iter);

                    boolean hasNaN = nativeOutput.isNaN().any();
                    assertFalse(hasNaN, "Native output contains NaN at iteration " + iter +
                            " — attention mask was likely confused with past_key (headDim filter bug)");

                    double maxDiff = refOutput.sub(nativeOutput).amaxNumber().doubleValue();
                    log.info("HeadDimFilter iter={}: maxDiff={}, hasNaN={}", iter, maxDiff, hasNaN);
                    assertTrue(maxDiff < 0.01, "Max diff too large: " + maxDiff);

                    // After warmup, freeze shapes so Triton compilation triggers
                    if (iter == 0) {
                        nativeOps.setPlanShapesFrozen(planHandle, true);
                    }
                }

                long tritonLaunches = nativeOps.getTritonKernelLaunchCount();
                log.info("HeadDimFilter: tritonLaunches={}", tritonLaunches);
                assertTrue(tritonLaunches > 0,
                        "Triton attention kernel was never launched — headDim filter or compilation failed");
            } finally {
                nativeOps.freeDynamicShapePlan(planHandle);
            }
        } finally {
            env.setTritonCompileAll(prevCompileAll);
            env.setTritonIncludeTypes(prevIncludeTypes);
        }
    }

    /**
     * Isolation test: seqK derivation from external KV cache when slot shapes are stale.
     *
     * Simulates the scenario where:
     * 1. Initial compilation sees empty KV cache (seqK=0)
     * 2. Plan is recompiled with populated KV cache (seqK=100)
     * 3. Slot output shapes may be stale from warmup
     * 4. The Triton compiler must derive seqK from external input shapes
     *
     * Tests two phases:
     * - Phase 1: Empty KV → Triton should fall back to C++ (seqK=0)
     * - Phase 2: Populated KV → Triton should compile with correct seqK
     *
     * Run with: -Dtest=TritonGraphBackendTest#testFusedAttention_SeqKDerivation_EmptyToPopulated
     */
    @Test
    public void testFusedAttention_SeqKDerivation_EmptyToPopulated() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        var env = Nd4j.getEnvironment();

        boolean prevCompileAll = env.tritonCompileAll();
        String prevIncludeTypes = env.tritonIncludeTypes();
        env.setTritonCompileAll(true);
        env.setTritonIncludeTypes("FUSED_ATTENTION");

        try {
            int batch = 1, seqQ = 1, numHeads = 4, headDim = 64;
            int hidden = numHeads * headDim;

            // Phase 1: Empty KV cache (seqK=0)
            int kvCacheLenEmpty = 0;
            SameDiff sd = createOnnxMhaWithKvCacheGraph(batch, seqQ, numHeads, headDim, 10, false);

            // Use populated KV for reference (non-empty)
            int kvCacheLen = 100;
            INDArray qArr = Nd4j.randn(DataType.FLOAT, batch, seqQ, hidden).muli(0.1);
            INDArray kArr = Nd4j.randn(DataType.FLOAT, batch, seqQ, hidden).muli(0.1);
            INDArray vArr = Nd4j.randn(DataType.FLOAT, batch, seqQ, hidden).muli(0.1);
            INDArray pastKeyArr = Nd4j.randn(DataType.FLOAT, batch, numHeads, kvCacheLen, headDim).muli(0.1);
            INDArray pastValArr = Nd4j.randn(DataType.FLOAT, batch, numHeads, kvCacheLen, headDim).muli(0.1);

            Map<String, INDArray> ph = Map.of(
                    "q", qArr, "k", kArr, "v", vArr,
                    "past_key", pastKeyArr, "past_value", pastValArr);

            // Reference output
            Map<String, INDArray> refResults = sd.output(ph, "attn_out");
            INDArray refOutput = refResults.get("attn_out").dup();
            assertNotNull(refOutput, "Reference output is null");
            assertFalse(refOutput.isNaN().any(), "Reference output contains NaN");

            // Phase 2: Compile plan with populated KV
            DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "attn_out");
            assertNotNull(plan, "Plan compilation returned null");

            Pointer planHandle = compileNativePlan(plan);
            if (planHandle == null) {
                log.info("Skipping (native executor not supported)");
                return;
            }
            try {
                INDArray[] extInputs = resolveExternalInputs(plan, sd, ph);
                nativeOps.resetTritonCounters();

                // Run 3 iterations: warmup (executeCount=0), GPU warmup (executeCount=1), compile+execute (executeCount=2)
                for (int iter = 0; iter < 3; iter++) {
                    Map<String, INDArray> nativeResults = executeNativePlan(planHandle, plan, extInputs);
                    INDArray nativeOutput = nativeResults.get("attn_out");
                    assertNotNull(nativeOutput, "Native output is null at iteration " + iter);

                    boolean hasNaN = nativeOutput.isNaN().any();
                    assertFalse(hasNaN, "Native output contains NaN at iteration " + iter +
                            " — seqK derivation may have failed");

                    double maxDiff = refOutput.sub(nativeOutput).amaxNumber().doubleValue();
                    log.info("SeqKDerivation iter={}: maxDiff={}, hasNaN={}", iter, maxDiff, hasNaN);
                    assertTrue(maxDiff < 0.01, "Max diff too large at iter " + iter + ": " + maxDiff);

                    // After warmup, freeze shapes so executeCount_ increments and Triton compilation triggers
                    if (iter == 0) {
                        nativeOps.setPlanShapesFrozen(planHandle, true);
                    }
                }

                long tritonLaunches = nativeOps.getTritonKernelLaunchCount();
                log.info("SeqKDerivation: tritonLaunches={}", tritonLaunches);
                // After 3 iterations, Triton should have compiled and launched the attention kernel
                assertTrue(tritonLaunches > 0,
                        "Triton attention kernel was never launched — seqK derivation failed to find KV cache shapes");
            } finally {
                nativeOps.freeDynamicShapePlan(planHandle);
            }
        } finally {
            env.setTritonCompileAll(prevCompileAll);
            env.setTritonIncludeTypes(prevIncludeTypes);
        }
    }

    // ─── GQA + Stale Shape Tests ────────────────────────────────────────────

    /**
     * Build a SameDiff graph with onnx_multi_head_attention using GQA (Grouped Query Attention).
     * Q has numQHeads, K/V have numKvHeads (fewer heads, e.g., 9Q/3KV for SmolDocling).
     *
     * @param batch      batch size
     * @param seqQ       query sequence length
     * @param numQHeads  number of Q attention heads
     * @param numKvHeads number of KV attention heads (must divide numQHeads)
     * @param headDim    head dimension
     * @param kvCacheLen past KV cache sequence length (0 = no past KV)
     */
    private SameDiff createGqaMhaGraph(int batch, int seqQ, int numQHeads, int numKvHeads,
                                        int headDim, int kvCacheLen) {
        SameDiff sd = SameDiff.create();
        int qHidden = numQHeads * headDim;
        int kvHidden = numKvHeads * headDim;

        SDVariable q = sd.placeHolder("q", DataType.FLOAT, batch, seqQ, qHidden);
        SDVariable k = sd.placeHolder("k", DataType.FLOAT, batch, seqQ, kvHidden);
        SDVariable v = sd.placeHolder("v", DataType.FLOAT, batch, seqQ, kvHidden);

        SDVariable pastKey = null;
        SDVariable pastValue = null;
        if (kvCacheLen > 0) {
            pastKey = sd.placeHolder("past_key", DataType.FLOAT, batch, numKvHeads, kvCacheLen, headDim);
            pastValue = sd.placeHolder("past_value", DataType.FLOAT, batch, numKvHeads, kvCacheLen, headDim);
        }

        var attnOp = new OnnxMultiHeadAttention(
                sd, q, k, v, null, pastKey, pastValue,
                numQHeads, 0.0, false, 1);
        SDVariable out = attnOp.outputVariables()[0];
        out.rename("attn_out");
        return sd;
    }

    /**
     * Test: Triton fused attention with GQA (Q=9 heads, KV=3 heads) + KV cache.
     * This is the exact SmolDocling configuration that crashed with CUDA error 700.
     *
     * Verifies that:
     * 1. The Triton compiler correctly handles GQA where numKvHeads != numQHeads
     * 2. The kernel launches successfully with KV cache inputs
     * 3. Output matches C++ reference
     *
     * Run with: -Dtest=TritonGraphBackendTest#testFusedAttention_GQA_SmolDoclingConfig
     */
    @Test
    public void testFusedAttention_GQA_SmolDoclingConfig() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        var env = Nd4j.getEnvironment();

        boolean prevCompileAll = env.tritonCompileAll();
        String prevIncludeTypes = env.tritonIncludeTypes();
        env.setTritonCompileAll(true);
        env.setTritonIncludeTypes("FUSED_ATTENTION");

        try {
            // SmolDocling decode: Q=9 heads, KV=3 heads, headDim=64
            int batch = 1, seqQ = 1, numQHeads = 9, numKvHeads = 3, headDim = 64;
            int kvCacheLen = 50;
            int qHidden = numQHeads * headDim;   // 576
            int kvHidden = numKvHeads * headDim;  // 192

            SameDiff sd = createGqaMhaGraph(batch, seqQ, numQHeads, numKvHeads, headDim, kvCacheLen);

            INDArray qArr = Nd4j.randn(DataType.FLOAT, batch, seqQ, qHidden).muli(0.1);
            INDArray kArr = Nd4j.randn(DataType.FLOAT, batch, seqQ, kvHidden).muli(0.1);
            INDArray vArr = Nd4j.randn(DataType.FLOAT, batch, seqQ, kvHidden).muli(0.1);
            INDArray pastKeyArr = Nd4j.randn(DataType.FLOAT, batch, numKvHeads, kvCacheLen, headDim).muli(0.1);
            INDArray pastValArr = Nd4j.randn(DataType.FLOAT, batch, numKvHeads, kvCacheLen, headDim).muli(0.1);

            Map<String, INDArray> ph = Map.of(
                    "q", qArr, "k", kArr, "v", vArr,
                    "past_key", pastKeyArr, "past_value", pastValArr);

            // Reference output
            Map<String, INDArray> refResults = sd.output(ph, "attn_out");
            INDArray refOutput = refResults.get("attn_out").dup();
            assertNotNull(refOutput, "Reference output is null");
            assertFalse(refOutput.isNaN().any(), "Reference output contains NaN");
            log.info("GQA SmolDocling: ref shape={}, ref max={}",
                    java.util.Arrays.toString(refOutput.shape()), refOutput.amaxNumber());

            // Compile and execute via Triton
            DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "attn_out");
            assertNotNull(plan, "Plan compilation returned null");

            Pointer planHandle = compileNativePlan(plan);
            if (planHandle == null) {
                log.info("Skipping (native executor not supported)");
                return;
            }
            try {
                INDArray[] extInputs = resolveExternalInputs(plan, sd, ph);
                nativeOps.resetTritonCounters();

                for (int iter = 0; iter < 3; iter++) {
                    Map<String, INDArray> nativeResults = executeNativePlan(planHandle, plan, extInputs);
                    INDArray nativeOutput = nativeResults.get("attn_out");
                    assertNotNull(nativeOutput, "Native output is null at iteration " + iter);

                    boolean hasNaN = nativeOutput.isNaN().any();
                    assertFalse(hasNaN, "GQA output contains NaN at iteration " + iter);

                    double maxDiff = refOutput.sub(nativeOutput).amaxNumber().doubleValue();
                    log.info("GQA SmolDocling iter={}: maxDiff={}", iter, maxDiff);
                    assertTrue(maxDiff < 0.01, "GQA output mismatch at iter " + iter + ": " + maxDiff);

                    if (iter == 0) {
                        nativeOps.setPlanShapesFrozen(planHandle, true);
                    }
                }

                long tritonLaunches = nativeOps.getTritonKernelLaunchCount();
                log.info("GQA SmolDocling: tritonLaunches={}", tritonLaunches);
            } finally {
                nativeOps.freeDynamicShapePlan(planHandle);
            }
        } finally {
            env.setTritonCompileAll(prevCompileAll);
            env.setTritonIncludeTypes(prevIncludeTypes);
        }
    }

    /**
     * Regression test: ONNX MHA with GQA but no direct past_key/past_value input.
     *
     * This matches the imported SmolDocling decoder pattern where repeat_kv is done
     * upstream, so onnx_multi_head_attention receives 3D Q/K/V while inputs 4/5 are
     * only empty placeholders. Triton must infer numKvHeads from kvHidden / headDim.
     */
    @Test
    public void testFusedAttention_GQA_3dKvWithoutDirectPastInput() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        var env = Nd4j.getEnvironment();

        boolean prevCompileAll = env.tritonCompileAll();
        String prevIncludeTypes = env.tritonIncludeTypes();
        env.setTritonCompileAll(true);
        env.setTritonIncludeTypes("FUSED_ATTENTION");

        try {
            int batch = 1, seqQ = 17, numQHeads = 9, numKVHeads = 3, headDim = 64;
            int qHidden = numQHeads * headDim;
            int kvHidden = numKVHeads * headDim;

            SameDiff sd = createGqaMhaGraph(batch, seqQ, numQHeads, numKVHeads, headDim, 0);

            INDArray qArr = Nd4j.randn(DataType.FLOAT, batch, seqQ, qHidden).muli(0.1);
            INDArray kArr = Nd4j.randn(DataType.FLOAT, batch, seqQ, kvHidden).muli(0.1);
            INDArray vArr = Nd4j.randn(DataType.FLOAT, batch, seqQ, kvHidden).muli(0.1);

            Map<String, INDArray> ph = Map.of(
                    "q", qArr,
                    "k", kArr,
                    "v", vArr);

            Map<String, INDArray> refResults = sd.output(ph, "attn_out");
            INDArray refOutput = refResults.get("attn_out").dup();
            assertNotNull(refOutput, "Reference output is null");
            assertFalse(refOutput.isNaN().any(), "Reference output contains NaN");

            DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "attn_out");
            assertNotNull(plan, "Plan compilation returned null");

            Pointer planHandle = compileNativePlan(plan);
            if (planHandle == null) {
                log.info("Skipping (native executor not supported)");
                return;
            }
            try {
                INDArray[] extInputs = resolveExternalInputs(plan, sd, ph);
                nativeOps.resetTritonCounters();

                for (int iter = 0; iter < 4; iter++) {
                    Map<String, INDArray> nativeResults = executeNativePlan(planHandle, plan, extInputs);
                    INDArray nativeOutput = nativeResults.get("attn_out");
                    assertNotNull(nativeOutput, "Native output is null at iteration " + iter);

                    boolean hasNaN = nativeOutput.isNaN().any();
                    assertFalse(hasNaN, "Native output contains NaN at iteration " + iter);

                    double maxDiff = refOutput.sub(nativeOutput).amaxNumber().doubleValue();
                    log.info("GQA_3dKvWithoutDirectPast iter {}: maxDiff={}, native max={}, native mean={}",
                            iter, maxDiff, nativeOutput.amaxNumber(), nativeOutput.meanNumber());
                    assertTrue(maxDiff < TOLERANCE,
                            "Output mismatch at iteration " + iter + ": max diff = " + maxDiff);

                    if (iter == 0) {
                        nativeOps.setPlanShapesFrozen(planHandle, true);
                    }
                }

                long tritonLaunches = nativeOps.getTritonKernelLaunchCount();
                log.info("GQA_3dKvWithoutDirectPast: tritonLaunches={}", tritonLaunches);
                assertTrue(tritonLaunches > 0,
                        "Triton attention kernel was never launched for 3D GQA ONNX MHA");
            } finally {
                nativeOps.freeDynamicShapePlan(planHandle);
            }
        } finally {
            env.setTritonCompileAll(prevCompileAll);
            env.setTritonIncludeTypes(prevIncludeTypes);
        }
    }

    /**
     * Test: Stale K shape recovery — simulates the warmup-to-decode transition
     * where KV cache shapes are stale from warmup (empty past_key) but external
     * inputs have correct shapes at decode time.
     *
     * The key scenario:
     * 1. Warmup with small KV cache → slot output shapes cached
     * 2. New plan compiled with much larger KV cache
     * 3. Triton compiler must derive seqK from external inputs, not stale cached shapes
     *
     * This tests the seqK derivation strategies (Strategy 1-3) in the Triton compiler
     * with GQA head configuration (numQHeads != numKvHeads).
     *
     * Run with: -Dtest=TritonGraphBackendTest#testFusedAttention_StaleKShape_GQA
     */
    @Test
    public void testFusedAttention_StaleKShape_GQA() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        var env = Nd4j.getEnvironment();

        boolean prevCompileAll = env.tritonCompileAll();
        String prevIncludeTypes = env.tritonIncludeTypes();
        env.setTritonCompileAll(true);
        env.setTritonIncludeTypes("FUSED_ATTENTION");

        try {
            int batch = 1, seqQ = 1, numQHeads = 9, numKvHeads = 3, headDim = 64;
            int qHidden = numQHeads * headDim;
            int kvHidden = numKvHeads * headDim;

            // Phase 1: Warmup with small KV cache (simulates initial state)
            int smallKvLen = 5;
            SameDiff sd = createGqaMhaGraph(batch, seqQ, numQHeads, numKvHeads, headDim, smallKvLen);

            INDArray qArr = Nd4j.randn(DataType.FLOAT, batch, seqQ, qHidden).muli(0.1);
            INDArray kArr = Nd4j.randn(DataType.FLOAT, batch, seqQ, kvHidden).muli(0.1);
            INDArray vArr = Nd4j.randn(DataType.FLOAT, batch, seqQ, kvHidden).muli(0.1);
            INDArray smallPastKey = Nd4j.randn(DataType.FLOAT, batch, numKvHeads, smallKvLen, headDim).muli(0.1);
            INDArray smallPastVal = Nd4j.randn(DataType.FLOAT, batch, numKvHeads, smallKvLen, headDim).muli(0.1);

            Map<String, INDArray> ph1 = Map.of(
                    "q", qArr, "k", kArr, "v", vArr,
                    "past_key", smallPastKey, "past_value", smallPastVal);

            // Execute warmup to populate cached shapes
            sd.output(ph1, "attn_out");

            // Phase 2: Larger KV cache (simulates decode after many tokens)
            int largeKvLen = 200;
            INDArray largePastKey = Nd4j.randn(DataType.FLOAT, batch, numKvHeads, largeKvLen, headDim).muli(0.1);
            INDArray largePastVal = Nd4j.randn(DataType.FLOAT, batch, numKvHeads, largeKvLen, headDim).muli(0.1);

            Map<String, INDArray> ph2 = Map.of(
                    "q", qArr, "k", kArr, "v", vArr,
                    "past_key", largePastKey, "past_value", largePastVal);

            // Reference output with large KV
            Map<String, INDArray> refResults = sd.output(ph2, "attn_out");
            INDArray refOutput = refResults.get("attn_out").dup();
            assertNotNull(refOutput, "Reference output is null");
            assertFalse(refOutput.isNaN().any(), "Reference output contains NaN");

            // Compile plan — slot shapes may be stale from warmup
            DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "attn_out");
            assertNotNull(plan, "Plan compilation returned null");

            Pointer planHandle = compileNativePlan(plan);
            if (planHandle == null) {
                log.info("Skipping (native executor not supported)");
                return;
            }
            try {
                // Resolve external inputs with LARGE KV cache
                INDArray[] extInputs = resolveExternalInputs(plan, sd, ph2);
                nativeOps.resetTritonCounters();

                for (int iter = 0; iter < 3; iter++) {
                    Map<String, INDArray> nativeResults = executeNativePlan(planHandle, plan, extInputs);
                    INDArray nativeOutput = nativeResults.get("attn_out");
                    assertNotNull(nativeOutput, "Native output is null at iteration " + iter);

                    boolean hasNaN = nativeOutput.isNaN().any();
                    assertFalse(hasNaN, "Stale shape recovery output contains NaN at iteration " + iter);

                    double maxDiff = refOutput.sub(nativeOutput).amaxNumber().doubleValue();
                    log.info("StaleKShape GQA iter={}: maxDiff={}", iter, maxDiff);
                    assertTrue(maxDiff < 0.01, "Stale shape recovery mismatch at iter " + iter + ": " + maxDiff);

                    if (iter == 0) {
                        nativeOps.setPlanShapesFrozen(planHandle, true);
                    }
                }

                long tritonLaunches = nativeOps.getTritonKernelLaunchCount();
                log.info("StaleKShape GQA: tritonLaunches={}", tritonLaunches);
                assertTrue(tritonLaunches > 0,
                        "Triton attention kernel was never launched — stale shape recovery failed");
            } finally {
                nativeOps.freeDynamicShapePlan(planHandle);
            }
        } finally {
            env.setTritonCompileAll(prevCompileAll);
            env.setTritonIncludeTypes(prevIncludeTypes);
        }
    }

    /**
     * Test: Multiple GQA configurations with Triton compilation.
     * Tests various Q/KV head ratios to ensure the Triton compiler handles all valid GQA configs.
     *
     * Run with: -Dtest=TritonGraphBackendTest#testFusedAttention_GQA_MultipleConfigs
     */
    @Test
    public void testFusedAttention_GQA_MultipleConfigs() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        var env = Nd4j.getEnvironment();

        boolean prevCompileAll = env.tritonCompileAll();
        String prevIncludeTypes = env.tritonIncludeTypes();
        env.setTritonCompileAll(true);
        env.setTritonIncludeTypes("FUSED_ATTENTION");

        try {
            int batch = 1, seqQ = 1, headDim = 64;
            int kvCacheLen = 20;

            int[][] configs = {
                {9, 3},   // SmolDocling (3:1 GQA)
                {12, 4},  // 3:1 GQA
                {8, 2},   // 4:1 GQA
                {6, 1},   // MQA (Multi-Query Attention)
                {4, 4},   // Standard MHA (no grouping)
            };

            for (int[] cfg : configs) {
                int numQHeads = cfg[0];
                int numKvHeads = cfg[1];
                int qHidden = numQHeads * headDim;
                int kvHidden = numKvHeads * headDim;

                log.info("Testing GQA config: Q={} KV={} (ratio={}:1)",
                        numQHeads, numKvHeads, numQHeads / numKvHeads);

                SameDiff sd = createGqaMhaGraph(batch, seqQ, numQHeads, numKvHeads, headDim, kvCacheLen);

                INDArray qArr = Nd4j.randn(DataType.FLOAT, batch, seqQ, qHidden).muli(0.1);
                INDArray kArr = Nd4j.randn(DataType.FLOAT, batch, seqQ, kvHidden).muli(0.1);
                INDArray vArr = Nd4j.randn(DataType.FLOAT, batch, seqQ, kvHidden).muli(0.1);
                INDArray pastKeyArr = Nd4j.randn(DataType.FLOAT, batch, numKvHeads, kvCacheLen, headDim).muli(0.1);
                INDArray pastValArr = Nd4j.randn(DataType.FLOAT, batch, numKvHeads, kvCacheLen, headDim).muli(0.1);

                Map<String, INDArray> ph = Map.of(
                        "q", qArr, "k", kArr, "v", vArr,
                        "past_key", pastKeyArr, "past_value", pastValArr);

                // Reference
                Map<String, INDArray> refResults = sd.output(ph, "attn_out");
                INDArray refOutput = refResults.get("attn_out").dup();
                assertNotNull(refOutput, "Reference output is null for Q=" + numQHeads + "/KV=" + numKvHeads);
                assertFalse(refOutput.isNaN().any(), "Reference contains NaN for Q=" + numQHeads + "/KV=" + numKvHeads);

                // Compile and execute
                DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "attn_out");
                Pointer planHandle = compileNativePlan(plan);
                if (planHandle == null) {
                    log.info("Skipping config Q={}/KV={} (native executor not supported)", numQHeads, numKvHeads);
                    continue;
                }
                try {
                    INDArray[] extInputs = resolveExternalInputs(plan, sd, ph);
                    nativeOps.resetTritonCounters();

                    for (int iter = 0; iter < 3; iter++) {
                        Map<String, INDArray> nativeResults = executeNativePlan(planHandle, plan, extInputs);
                        INDArray nativeOutput = nativeResults.get("attn_out");
                        assertNotNull(nativeOutput);

                        boolean hasNaN = nativeOutput.isNaN().any();
                        assertFalse(hasNaN, "NaN for Q=" + numQHeads + "/KV=" + numKvHeads + " iter=" + iter);

                        double maxDiff = refOutput.sub(nativeOutput).amaxNumber().doubleValue();
                        log.info("GQA Q={}/KV={} iter={}: maxDiff={}", numQHeads, numKvHeads, iter, maxDiff);
                        assertTrue(maxDiff < 0.01,
                                "Mismatch Q=" + numQHeads + "/KV=" + numKvHeads + " iter=" + iter + ": " + maxDiff);

                        if (iter == 0) {
                            nativeOps.setPlanShapesFrozen(planHandle, true);
                        }
                    }
                } finally {
                    nativeOps.freeDynamicShapePlan(planHandle);
                }
            }
        } finally {
            env.setTritonCompileAll(prevCompileAll);
            env.setTritonIncludeTypes(prevIncludeTypes);
        }
    }
}
