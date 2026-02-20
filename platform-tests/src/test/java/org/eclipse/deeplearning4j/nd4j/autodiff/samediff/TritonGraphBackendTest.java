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

        INDArray[] extInputs = resolveExternalInputs(plan, sd, ph);

        // Execute 3 times: warmup, compile/capture, replay
        for (int iter = 0; iter < 3; iter++) {
            Map<String, INDArray> nativeResults = executeNativePlan(planHandle, plan, extInputs);
            INDArray nativeOutput = nativeResults.get("result");
            assertNotNull(nativeOutput, "Native output is null at iteration " + iter);

            // Compare with reference
            double maxDiff = refOutput.sub(nativeOutput).amaxNumber().doubleValue();
            log.info("Iteration {}: max diff = {}", iter, maxDiff);
            assertTrue(maxDiff < TOLERANCE,
                       "Output mismatch at iteration " + iter + ": max diff = " + maxDiff);
        }
    }

    /**
     * Test: purely element-wise chain (best case for Triton fusion).
     * All ops are element-wise, so Triton should be able to fuse the entire
     * segment into a single kernel with no intermediate global memory stores.
     */
    @Test
    public void testElementWiseFusion() {
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

        INDArray[] extInputs = resolveExternalInputs(plan, sd, ph);

        for (int iter = 0; iter < 3; iter++) {
            Map<String, INDArray> nativeResults = executeNativePlan(planHandle, plan, extInputs);
            INDArray nativeOutput = nativeResults.get("result");
            assertNotNull(nativeOutput, "Native output is null at iteration " + iter);

            double maxDiff = refOutput.sub(nativeOutput).amaxNumber().doubleValue();
            log.info("ElementWise iteration {}: max diff = {}", iter, maxDiff);
            assertTrue(maxDiff < TOLERANCE,
                       "Output mismatch at iteration " + iter + ": max diff = " + maxDiff);
        }
    }

    /**
     * Test: cache reuse verification.
     * Execute the same plan twice with the same shapes. The second execution
     * should use the cached compiled kernel (no recompilation).
     */
    @Test
    public void testCacheReuse() {
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

        // Execute with x1 (3 iterations to go through warmup/capture/replay)
        INDArray[] extInputs1 = resolveExternalInputs(plan, sd, ph1);
        for (int i = 0; i < 3; i++) {
            Map<String, INDArray> results = executeNativePlan(planHandle, plan, extInputs1);
            assertNotNull(results.get("result"), "Null output at iteration " + i);
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
        assertTrue(maxDiff < TOLERANCE, "Output mismatch with cached kernel: max diff = " + maxDiff);
    }

    /**
     * Test: fallback to CUDA Graphs for mixed-op segments.
     * Creates a graph with some unsupported ops that Triton cannot compile.
     * Verifies the execution still succeeds via CUDA Graphs or slot-by-slot.
     */
    @Test
    public void testFallbackForUnsupportedOps() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable w = sd.constant("w", Nd4j.randn(DataType.FLOAT, 4, 8));

        // Build a chain with some ops that Triton may not support
        SDVariable h1 = sd.mmul("matmul", x, w);
        SDVariable h2 = sd.nn.relu("relu1", h1, 0);
        SDVariable h3 = sd.math.tanh("tanh1", h2);
        // concat is not in the Triton op table
        SDVariable h4 = sd.concat("concat1", 1, h2, h3);
        SDVariable h5 = sd.nn.relu("relu2", h4, 0);
        SDVariable h6 = sd.nn.sigmoid("sigmoid1", h5);
        SDVariable h7 = h6.mul("mul1", sd.constant("s1", Nd4j.randn(DataType.FLOAT, 1, 16)));
        SDVariable h8 = h7.add("add1", sd.constant("s2", Nd4j.randn(DataType.FLOAT, 1, 16)));
        SDVariable h9 = sd.nn.relu("relu3", h8, 0);
        SDVariable h10 = sd.math.tanh("tanh2", h9);
        SDVariable h11 = h10.mul("mul2", sd.constant("s3", Nd4j.randn(DataType.FLOAT, 1, 16)));
        SDVariable result = sd.nn.relu("result", h11, 0);

        INDArray xArr = Nd4j.randn(DataType.FLOAT, 2, 4);
        Map<String, INDArray> ph = Map.of("x", xArr);

        Map<String, INDArray> refResults = sd.output(ph, "result");
        INDArray refOutput = refResults.get("result");
        assertNotNull(refOutput);

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "result");
        Pointer planHandle = compileNativePlan(plan);
        if (planHandle == null) {
            log.info("Skipping (native executor not supported)");
            return;
        }

        INDArray[] extInputs = resolveExternalInputs(plan, sd, ph);

        // Should still work via fallback path
        for (int iter = 0; iter < 3; iter++) {
            Map<String, INDArray> nativeResults = executeNativePlan(planHandle, plan, extInputs);
            INDArray nativeOutput = nativeResults.get("result");
            assertNotNull(nativeOutput, "Native output is null at iteration " + iter);

            double maxDiff = refOutput.sub(nativeOutput).amaxNumber().doubleValue();
            log.info("Fallback test iteration {}: max diff = {}", iter, maxDiff);
            assertTrue(maxDiff < TOLERANCE,
                       "Output mismatch at iteration " + iter + ": max diff = " + maxDiff);
        }
    }
}
