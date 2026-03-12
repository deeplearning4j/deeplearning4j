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
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests that exercise the native C++ graph executor via direct JNI calls.
 *
 * Unlike other NativeExecutor*Test classes that test via sd.output() (Java executor),
 * these tests directly call:
 *   1. compileDynamicShapePlan() - serialize plan and compile in C++
 *   2. executeDynamicShapePlan() - execute the full plan in C++
 *   3. freeDynamicShapePlan() - cleanup
 *
 * This validates the entire end-to-end JNI path: Java serialization → C++ deserialization
 * → C++ op execution → output extraction.
 */
@Slf4j
@Tag(TagNames.SAMEDIFF)
@NativeTag
public class NativeExecutorJNIPathTest extends BaseNd4jTestWithBackends {

    private static final double TOLERANCE = 1e-4;

    /**
     * Helper: compile a SameDiff graph into a native C++ plan handle.
     * Returns the plan handle pointer, or null if the backend doesn't support it.
     */
    private Pointer compileNativePlan(DynamicShapePlan plan) {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        byte[] serialized = plan.serialize();
        assertNotNull(serialized, "Plan serialization returned null");
        assertTrue(serialized.length > 0, "Plan serialization returned empty");

        BytePointer planBytes = new BytePointer(serialized);
        try {
            Pointer handle = nativeOps.compileDynamicShapePlan(planBytes, serialized.length);
            return handle;
        } catch (UnsupportedOperationException e) {
            log.info("Backend does not support native executor: {}", e.getMessage());
            return null;
        } finally {
            planBytes.close();
        }
    }

    /**
     * Helper: execute a compiled native plan with given external inputs.
     * Uses OpaqueContext with per-index setGraphContextInputArray (proven pattern).
     * Returns a map of output name → INDArray.
     */
    private Map<String, INDArray> executeNativePlan(Pointer planHandle, DynamicShapePlan plan,
                                                     INDArray[] extInputs) {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        int numOutputs = plan.getRequestedOutputs().size();

        OpaqueContext opContext = nativeOps.createGraphContext(1);
        try {
            // Set inputs on context one at a time
            for (int i = 0; i < extInputs.length; i++) {
                OpaqueNDArray opaqueIn = OpaqueNDArray.fromINDArray(extInputs[i]);
                nativeOps.setGraphContextInputArray(opContext, i, opaqueIn);
            }

            // Set empty output slots on context (C++ plan allocates actual outputs)
            for (int i = 0; i < numOutputs; i++) {
                INDArray dummy = Nd4j.empty(DataType.FLOAT);
                OpaqueNDArray opaqueOut = OpaqueNDArray.fromINDArray(dummy);
                nativeOps.setGraphContextOutputArray(opContext, i, opaqueOut);
            }

            // Get execution stream (null for CPU)
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

            int status = nativeOps.executeDynamicShapePlan(
                    planHandle,
                    opContext,
                    execStream);

            if (status != 0) {
                String errMsg = nativeOps.lastErrorMessage();
                nativeOps.clearLastError();
                fail("Native plan execution failed with status " + status + ": " + errMsg);
            }

            // Extract outputs from context
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

                // Get raw pointers — primary may be null on CUDA (data only on GPU)
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

    /**
     * Helper: resolve external inputs for a plan from SameDiff variables and placeholders.
     */
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

    // ==================== Basic Operations ====================

    @Test
    public void testAddJNI() {
        SameDiff sd = NativeExecutorTestUtils.createAddGraph();
        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");

        Pointer handle = compileNativePlan(plan);
        if (handle == null || handle.isNull()) {
            log.info("Skipping - backend doesn't support native executor");
            return;
        }

        try {
            INDArray x = Nd4j.ones(DataType.FLOAT, 2, 3).mul(2);
            INDArray y = Nd4j.ones(DataType.FLOAT, 2, 3).mul(3);
            Map<String, INDArray> ph = Map.of("x", x, "y", y);

            INDArray[] extInputs = resolveExternalInputs(plan, sd, ph);
            Map<String, INDArray> nativeResult = executeNativePlan(handle, plan, extInputs);

            // Compare with Java execution
            Map<String, INDArray> javaResult = sd.output(ph, "z");

            assertNotNull(nativeResult.get("z"), "Native output is null");
            NativeExecutorTestUtils.assertArrayEquals(
                    javaResult.get("z"), nativeResult.get("z"), TOLERANCE, "add JNI parity");
        } finally {
            NativeOpsHolder.getInstance().getDeviceNativeOps().freeDynamicShapePlan(handle);
            plan.close();
        }
    }

    @Test
    public void testEinsumJNI() {
        SameDiff sd = SameDiff.create();
        SDVariable left = sd.placeHolder("left", DataType.FLOAT, 2, 3);
        SDVariable right = sd.placeHolder("right", DataType.FLOAT, 4, 3);
        sd.linalg().einsum("z", new SDVariable[]{left, right}, "ik,jk->ij");

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");
        Pointer handle = compileNativePlan(plan);
        if (handle == null || handle.isNull()) {
            log.info("Skipping - backend doesn't support native executor");
            return;
        }

        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        try {
            INDArray leftArr = Nd4j.linspace(1, 6, 6, DataType.FLOAT).reshape(2, 3);
            INDArray rightArr = Nd4j.linspace(1, 12, 12, DataType.FLOAT).reshape(4, 3);
            Map<String, INDArray> placeholders = Map.of("left", leftArr, "right", rightArr);

            INDArray expected = leftArr.mmul(rightArr.transpose());
            INDArray[] extInputs = resolveExternalInputs(plan, sd, placeholders);
            Map<String, INDArray> nativeResult = executeNativePlan(handle, plan, extInputs);
            NativeExecutorTestUtils.assertArrayEquals(expected, nativeResult.get("z"), TOLERANCE, "einsum");
        } finally {
            nativeOps.freeDynamicShapePlan(handle);
            plan.close();
        }
    }

    @Test
    public void testMatMulJNI() {
        SameDiff sd = NativeExecutorTestUtils.createMatMulGraph(4, 8);
        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");

        Pointer handle = compileNativePlan(plan);
        if (handle == null || handle.isNull()) {
            log.info("Skipping - backend doesn't support native executor");
            return;
        }

        try {
            INDArray x = Nd4j.randn(DataType.FLOAT, 2, 4);
            Map<String, INDArray> ph = Map.of("x", x);

            INDArray[] extInputs = resolveExternalInputs(plan, sd, ph);
            Map<String, INDArray> nativeResult = executeNativePlan(handle, plan, extInputs);
            Map<String, INDArray> javaResult = sd.output(ph, "z");

            NativeExecutorTestUtils.assertArrayEquals(
                    javaResult.get("z"), nativeResult.get("z"), TOLERANCE, "matmul JNI parity");
        } finally {
            NativeOpsHolder.getInstance().getDeviceNativeOps().freeDynamicShapePlan(handle);
            plan.close();
        }
    }

    @Test
    public void testJNIViewOutputUsesOffsetAdjustedPointersAndLogicalLength() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        INDArray base = Nd4j.linspace(1, 20, 20, DataType.FLOAT).reshape(5, 4);
        INDArray view = base.get(NDArrayIndex.interval(1, 3), NDArrayIndex.all());
        INDArray expected = view.dup();

        assertTrue(view.offset() > 0, "Test setup requires a non-zero-offset view");

        OpaqueNDArray opaqueView = OpaqueNDArray.fromINDArray(view);
        assertNotNull(opaqueView, "Failed to create opaque view");
        assertFalse(opaqueView.isNull(), "Opaque view was null");

        long[] shapeInfo = OpaqueNDArray.getOpaqueNDArrayShapeInfo(opaqueView);
        long[] shape = Shape.shape(shapeInfo);
        DataType dtype = ArrayOptionsHelper.dataType(shapeInfo);
        long logicalLength = Shape.length(shapeInfo);
        long reportedLength = OpaqueNDArray.getOpaqueNDArrayLength(opaqueView);
        long reportedOffset = OpaqueNDArray.getOpaqueNDArrayOffset(opaqueView);

        assertArrayEquals(expected.shape(), shape, "Unexpected view shape");
        assertEquals(view.offset(), reportedOffset, "JNI wrapper must report the INDArray offset");
        assertEquals(logicalLength, reportedLength,
                "JNI wrapper must report logical array length, not backing-buffer length");

        Pointer nativePrimary = nativeOps.getOpaqueNDArrayBuffer(opaqueView);
        Pointer nativeSpecial = nativeOps.getOpaqueNDArraySpecialBuffer(opaqueView);
        OpaqueDataBuffer srcOdb = nativeOps.dbCreateExternalDataBuffer(
                logicalLength, dtype.toInt(), nativePrimary, nativeSpecial);
        assertNotNull(srcOdb, "Failed to wrap native view buffer");

        INDArray copied = Nd4j.createUninitialized(dtype, shape);
        OpaqueDataBuffer dstOdb = copied.data().opaqueBuffer();
        assertNotNull(dstOdb, "Destination opaque buffer was null");
        nativeOps.copyBuffer(dstOdb, logicalLength, srcOdb, 0, 0);

        NativeExecutorTestUtils.assertArrayEquals(
                expected, copied, TOLERANCE, "JNI output copy must respect view offset");
    }

    @Test
    public void testChainGraphJNI() {
        SameDiff sd = NativeExecutorTestUtils.createChainGraph();
        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");

        Pointer handle = compileNativePlan(plan);
        if (handle == null || handle.isNull()) {
            log.info("Skipping - backend doesn't support native executor");
            return;
        }

        try {
            INDArray x = Nd4j.randn(DataType.FLOAT, 3, 4);
            Map<String, INDArray> ph = Map.of("x", x);

            INDArray[] extInputs = resolveExternalInputs(plan, sd, ph);
            Map<String, INDArray> nativeResult = executeNativePlan(handle, plan, extInputs);
            Map<String, INDArray> javaResult = sd.output(ph, "z");

            NativeExecutorTestUtils.assertArrayEquals(
                    javaResult.get("z"), nativeResult.get("z"), TOLERANCE, "chain JNI parity");
        } finally {
            NativeOpsHolder.getInstance().getDeviceNativeOps().freeDynamicShapePlan(handle);
            plan.close();
        }
    }

    @Test
    public void testDiamondGraphJNI() {
        SameDiff sd = NativeExecutorTestUtils.createDiamondGraph();
        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "c");

        Pointer handle = compileNativePlan(plan);
        if (handle == null || handle.isNull()) {
            log.info("Skipping - backend doesn't support native executor");
            return;
        }

        try {
            INDArray x = Nd4j.randn(DataType.FLOAT, 2, 4);
            Map<String, INDArray> ph = Map.of("x", x);

            INDArray[] extInputs = resolveExternalInputs(plan, sd, ph);
            Map<String, INDArray> nativeResult = executeNativePlan(handle, plan, extInputs);
            Map<String, INDArray> javaResult = sd.output(ph, "c");

            NativeExecutorTestUtils.assertArrayEquals(
                    javaResult.get("c"), nativeResult.get("c"), TOLERANCE, "diamond JNI parity");
        } finally {
            NativeOpsHolder.getInstance().getDeviceNativeOps().freeDynamicShapePlan(handle);
            plan.close();
        }
    }

    // ==================== Plan Metadata Queries ====================

    @Test
    public void testPlanMetadataQueries() {
        SameDiff sd = NativeExecutorTestUtils.createAddGraph();
        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");

        Pointer handle = compileNativePlan(plan);
        if (handle == null || handle.isNull()) {
            log.info("Skipping - backend doesn't support native executor");
            return;
        }

        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        try {
            int numSlots = nativeOps.getPlanNumSlots(handle);
            int numExtInputs = nativeOps.getPlanNumExternalInputs(handle);
            int numOutputs = nativeOps.getPlanNumRequestedOutputs(handle);

            assertEquals(plan.getSlots().length, numSlots,
                    "C++ plan slot count should match Java plan");
            assertEquals(plan.getExternalInputKeys().length, numExtInputs,
                    "C++ external input count should match Java plan");
            assertEquals(plan.getRequestedOutputs().size(), numOutputs,
                    "C++ requested output count should match Java plan");

            log.info("Plan metadata: {} slots, {} external inputs, {} outputs",
                    numSlots, numExtInputs, numOutputs);

            // Segments query
            int numSegments = nativeOps.getPlanNumSegments(handle);
            assertTrue(numSegments >= 0, "Segment count should be non-negative: " + numSegments);
            log.info("Plan segments: {}", numSegments);
        } finally {
            nativeOps.freeDynamicShapePlan(handle);
            plan.close();
        }
    }

    // ==================== Multiple Executions (Shape Cache) ====================

    @Test
    public void testMultipleExecutionsSameShape() {
        SameDiff sd = NativeExecutorTestUtils.createAddGraph();
        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");

        Pointer handle = compileNativePlan(plan);
        if (handle == null || handle.isNull()) {
            log.info("Skipping - backend doesn't support native executor");
            return;
        }

        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        try {
            // Execute 5 times with same shapes but different data
            for (int i = 0; i < 5; i++) {
                INDArray x = Nd4j.randn(DataType.FLOAT, 2, 3);
                INDArray y = Nd4j.randn(DataType.FLOAT, 2, 3);
                Map<String, INDArray> ph = Map.of("x", x, "y", y);

                INDArray[] extInputs = resolveExternalInputs(plan, sd, ph);
                Map<String, INDArray> nativeResult = executeNativePlan(handle, plan, extInputs);
                Map<String, INDArray> javaResult = sd.output(ph, "z");

                NativeExecutorTestUtils.assertArrayEquals(
                        javaResult.get("z"), nativeResult.get("z"), TOLERANCE,
                        "Same-shape iteration " + i);
            }
        } finally {
            nativeOps.freeDynamicShapePlan(handle);
            plan.close();
        }
    }

    @Test
    public void testMultipleExecutionsChangingShapes() {
        SameDiff sd = NativeExecutorTestUtils.createAddGraph();
        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");

        Pointer handle = compileNativePlan(plan);
        if (handle == null || handle.isNull()) {
            log.info("Skipping - backend doesn't support native executor");
            return;
        }

        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        try {
            int[][] shapes = {{2, 3}, {4, 5}, {1, 10}, {8, 2}, {2, 3}};
            for (int i = 0; i < shapes.length; i++) {
                int r = shapes[i][0], c = shapes[i][1];

                // Must clear C++ caches when shapes change
                nativeOps.clearDynamicShapePlanCaches(handle);

                INDArray x = Nd4j.randn(DataType.FLOAT, r, c);
                INDArray y = Nd4j.randn(DataType.FLOAT, r, c);
                Map<String, INDArray> ph = Map.of("x", x, "y", y);

                INDArray[] extInputs = resolveExternalInputs(plan, sd, ph);
                Map<String, INDArray> nativeResult = executeNativePlan(handle, plan, extInputs);
                Map<String, INDArray> javaResult = sd.output(ph, "z");

                NativeExecutorTestUtils.assertArrayEquals(
                        javaResult.get("z"), nativeResult.get("z"), TOLERANCE,
                        "Shape [" + r + "," + c + "] iteration " + i);
            }
        } finally {
            nativeOps.freeDynamicShapePlan(handle);
            plan.close();
        }
    }

    // ==================== Multiple Outputs ====================

    @Test
    public void testMultipleOutputsJNI() {
        SameDiff sd = NativeExecutorTestUtils.createFanOutGraph();
        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "a", "b", "c");

        Pointer handle = compileNativePlan(plan);
        if (handle == null || handle.isNull()) {
            log.info("Skipping - backend doesn't support native executor");
            return;
        }

        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        try {
            INDArray x = Nd4j.randn(DataType.FLOAT, 2, 4);
            Map<String, INDArray> ph = Map.of("x", x);

            INDArray[] extInputs = resolveExternalInputs(plan, sd, ph);
            Map<String, INDArray> nativeResult = executeNativePlan(handle, plan, extInputs);
            Map<String, INDArray> javaResult = sd.output(ph, "a", "b", "c");

            assertEquals(3, nativeResult.size(), "Should have 3 outputs");
            for (String name : new String[]{"a", "b", "c"}) {
                assertNotNull(nativeResult.get(name), "Missing native output: " + name);
                NativeExecutorTestUtils.assertArrayEquals(
                        javaResult.get(name), nativeResult.get(name), TOLERANCE,
                        "multi-output " + name);
            }
        } finally {
            nativeOps.freeDynamicShapePlan(handle);
            plan.close();
        }
    }

    // ==================== Activations & Reductions ====================

    @Test
    public void testSigmoidJNI() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable z = sd.nn().sigmoid("z", x);

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");
        Pointer handle = compileNativePlan(plan);
        if (handle == null || handle.isNull()) return;

        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        try {
            INDArray x1 = Nd4j.randn(DataType.FLOAT, 3, 4);
            Map<String, INDArray> ph = Map.of("x", x1);

            INDArray[] extInputs = resolveExternalInputs(plan, sd, ph);
            Map<String, INDArray> nativeResult = executeNativePlan(handle, plan, extInputs);
            Map<String, INDArray> javaResult = sd.output(ph, "z");

            NativeExecutorTestUtils.assertArrayEquals(
                    javaResult.get("z"), nativeResult.get("z"), TOLERANCE, "sigmoid JNI");
        } finally {
            nativeOps.freeDynamicShapePlan(handle);
            plan.close();
        }
    }

    @Test
    public void testReduceSumJNI() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 3, 4);
        SDVariable z = sd.sum("z", x, 1);

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");
        Pointer handle = compileNativePlan(plan);
        if (handle == null || handle.isNull()) return;

        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        try {
            INDArray x1 = Nd4j.randn(DataType.FLOAT, 3, 4);
            Map<String, INDArray> ph = Map.of("x", x1);

            INDArray[] extInputs = resolveExternalInputs(plan, sd, ph);
            Map<String, INDArray> nativeResult = executeNativePlan(handle, plan, extInputs);
            Map<String, INDArray> javaResult = sd.output(ph, "z");

            NativeExecutorTestUtils.assertArrayEquals(
                    javaResult.get("z"), nativeResult.get("z"), TOLERANCE, "reduce_sum JNI");
        } finally {
            nativeOps.freeDynamicShapePlan(handle);
            plan.close();
        }
    }

    // ==================== Plan Reuse ====================

    @Test
    public void testPlanReuse() {
        SameDiff sd = NativeExecutorTestUtils.createMatMulGraph(8, 4);
        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");

        Pointer handle = compileNativePlan(plan);
        if (handle == null || handle.isNull()) return;

        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        try {
            // Execute multiple times, reusing the compiled C++ plan handle
            for (int iter = 0; iter < 10; iter++) {
                INDArray x = Nd4j.randn(DataType.FLOAT, 2, 8);
                Map<String, INDArray> ph = Map.of("x", x);

                INDArray[] extInputs = resolveExternalInputs(plan, sd, ph);
                Map<String, INDArray> nativeResult = executeNativePlan(handle, plan, extInputs);
                Map<String, INDArray> javaResult = sd.output(ph, "z");

                NativeExecutorTestUtils.assertArrayEquals(
                        javaResult.get("z"), nativeResult.get("z"), TOLERANCE,
                        "plan reuse iter " + iter);
            }
        } finally {
            nativeOps.freeDynamicShapePlan(handle);
            plan.close();
        }
    }

    // ==================== Cache Clear ====================

    @Test
    public void testClearShapeCaches() {
        SameDiff sd = NativeExecutorTestUtils.createAddGraph();
        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");

        Pointer handle = compileNativePlan(plan);
        if (handle == null || handle.isNull()) return;

        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        try {
            INDArray x = Nd4j.ones(DataType.FLOAT, 2, 3);
            INDArray y = Nd4j.ones(DataType.FLOAT, 2, 3);
            Map<String, INDArray> ph = Map.of("x", x, "y", y);

            // Execute once to populate caches
            INDArray[] extInputs = resolveExternalInputs(plan, sd, ph);
            Map<String, INDArray> result1 = executeNativePlan(handle, plan, extInputs);
            assertNotNull(result1.get("z"), "First execution output null");

            // Clear caches
            nativeOps.clearDynamicShapePlanCaches(handle);

            // Execute again — should still produce correct results
            extInputs = resolveExternalInputs(plan, sd, ph);
            Map<String, INDArray> result2 = executeNativePlan(handle, plan, extInputs);
            assertNotNull(result2.get("z"), "Post-clear execution output null");

            NativeExecutorTestUtils.assertArrayEquals(
                    result1.get("z"), result2.get("z"), TOLERANCE, "cache clear parity");
        } finally {
            nativeOps.freeDynamicShapePlan(handle);
            plan.close();
        }
    }

    // ==================== CUDA Graphs (if supported) ====================

    @Test
    public void testCudaGraphsEnableDisable() {
        SameDiff sd = NativeExecutorTestUtils.createAddGraph();
        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");

        Pointer handle = compileNativePlan(plan);
        if (handle == null || handle.isNull()) return;

        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        try {
            // Enable CUDA Graphs
            nativeOps.setPlanCudaGraphsEnabled(handle, true);

            INDArray x = Nd4j.ones(DataType.FLOAT, 2, 3).mul(2);
            INDArray y = Nd4j.ones(DataType.FLOAT, 2, 3).mul(3);
            Map<String, INDArray> ph = Map.of("x", x, "y", y);

            // Execute 3 times to trigger: warm-up → capture → replay
            for (int i = 0; i < 3; i++) {
                INDArray[] extInputs = resolveExternalInputs(plan, sd, ph);
                Map<String, INDArray> nativeResult = executeNativePlan(handle, plan, extInputs);
                Map<String, INDArray> javaResult = sd.output(ph, "z");

                NativeExecutorTestUtils.assertArrayEquals(
                        javaResult.get("z"), nativeResult.get("z"), TOLERANCE,
                        "CUDA graphs iter " + i);
            }

            // Check graph statistics
            int captured = nativeOps.getPlanNumCapturedGraphSegments(handle);
            int replays = nativeOps.getPlanTotalGraphReplays(handle);
            log.info("CUDA Graphs: {} captured segments, {} total replays", captured, replays);

            // Disable CUDA Graphs and verify execution still works
            nativeOps.setPlanCudaGraphsEnabled(handle, false);
            INDArray[] extInputs = resolveExternalInputs(plan, sd, ph);
            Map<String, INDArray> nativeResult = executeNativePlan(handle, plan, extInputs);
            Map<String, INDArray> javaResult = sd.output(ph, "z");

            NativeExecutorTestUtils.assertArrayEquals(
                    javaResult.get("z"), nativeResult.get("z"), TOLERANCE,
                    "post-disable CUDA graphs");
        } finally {
            nativeOps.freeDynamicShapePlan(handle);
            plan.close();
        }
    }

    // ==================== Autoregressive Pattern ====================

    @Test
    public void testAutoRegressiveGrowingKVCache() {
        // Simulates growing KV cache: input shape changes each step
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable w = sd.var("w", Nd4j.randn(DataType.FLOAT, 4, 4).mul(0.01));
        SDVariable out = sd.mmul("out", x, w);

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "out");
        Pointer handle = compileNativePlan(plan);
        if (handle == null || handle.isNull()) return;

        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        try {
            // Simulate 20 autoregressive steps with growing sequence length
            for (int step = 1; step <= 20; step++) {
                nativeOps.clearDynamicShapePlanCaches(handle);

                INDArray x1 = Nd4j.randn(DataType.FLOAT, step, 4);
                Map<String, INDArray> ph = Map.of("x", x1);

                INDArray[] extInputs = resolveExternalInputs(plan, sd, ph);
                Map<String, INDArray> nativeResult = executeNativePlan(handle, plan, extInputs);
                Map<String, INDArray> javaResult = sd.output(ph, "out");

                assertArrayEquals(javaResult.get("out").shape(), nativeResult.get("out").shape(),
                        "Shape mismatch at step " + step);
                NativeExecutorTestUtils.assertArrayEquals(
                        javaResult.get("out"), nativeResult.get("out"), TOLERANCE,
                        "autoregressive step " + step);
            }
        } finally {
            nativeOps.freeDynamicShapePlan(handle);
            plan.close();
        }
    }

    // ==================== Data Type Variety ====================

    @Test
    public void testFloat64JNI() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.DOUBLE, 2, 3);
        SDVariable y = sd.placeHolder("y", DataType.DOUBLE, 2, 3);
        SDVariable z = x.add("z", y);

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");
        Pointer handle = compileNativePlan(plan);
        if (handle == null || handle.isNull()) return;

        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        try {
            INDArray x1 = Nd4j.randn(DataType.DOUBLE, 2, 3);
            INDArray y1 = Nd4j.randn(DataType.DOUBLE, 2, 3);
            Map<String, INDArray> ph = Map.of("x", x1, "y", y1);

            INDArray[] extInputs = resolveExternalInputs(plan, sd, ph);
            Map<String, INDArray> nativeResult = executeNativePlan(handle, plan, extInputs);
            Map<String, INDArray> javaResult = sd.output(ph, "z");

            NativeExecutorTestUtils.assertArrayEquals(
                    javaResult.get("z"), nativeResult.get("z"), 1e-10, "float64 JNI");
        } finally {
            nativeOps.freeDynamicShapePlan(handle);
            plan.close();
        }
    }

    @Test
    public void testFloat16JNI() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.HALF, 2, 3);
        SDVariable y = sd.placeHolder("y", DataType.HALF, 2, 3);
        SDVariable z = x.add("z", y);

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");
        Pointer handle = compileNativePlan(plan);
        if (handle == null || handle.isNull()) return;

        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        try {
            INDArray x1 = Nd4j.ones(DataType.HALF, 2, 3).mul(2);
            INDArray y1 = Nd4j.ones(DataType.HALF, 2, 3).mul(3);
            Map<String, INDArray> ph = Map.of("x", x1, "y", y1);

            INDArray[] extInputs = resolveExternalInputs(plan, sd, ph);
            Map<String, INDArray> nativeResult = executeNativePlan(handle, plan, extInputs);
            Map<String, INDArray> javaResult = sd.output(ph, "z");

            NativeExecutorTestUtils.assertArrayEquals(
                    javaResult.get("z"), nativeResult.get("z"), 1e-2, "float16 JNI");
        } finally {
            nativeOps.freeDynamicShapePlan(handle);
            plan.close();
        }
    }

    // ==================== Larger Graph ====================

    @Test
    public void testLargerGraphJNI() {
        // Build a more complex graph: matmul → bias → relu → sigmoid → reduce
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 8);
        SDVariable w = sd.var("w", Nd4j.randn(DataType.FLOAT, 8, 16).mul(0.01));
        SDVariable b = sd.var("b", Nd4j.zeros(DataType.FLOAT, 1, 16));
        SDVariable mm = sd.mmul("mm", x, w);
        SDVariable biased = mm.add("biased", b);
        SDVariable act = sd.nn().relu("relu", biased, 0.0);
        SDVariable sig = sd.nn().sigmoid("sig", act);
        SDVariable out = sd.mean("out", sig, 1);

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "out");
        Pointer handle = compileNativePlan(plan);
        if (handle == null || handle.isNull()) return;

        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        try {
            int numSlots = nativeOps.getPlanNumSlots(handle);
            log.info("Larger graph: {} slots", numSlots);
            assertTrue(numSlots >= 4, "Expected at least 4 slots for mm→bias→relu→sig→reduce");

            INDArray x1 = Nd4j.randn(DataType.FLOAT, 4, 8);
            Map<String, INDArray> ph = Map.of("x", x1);

            INDArray[] extInputs = resolveExternalInputs(plan, sd, ph);
            Map<String, INDArray> nativeResult = executeNativePlan(handle, plan, extInputs);
            Map<String, INDArray> javaResult = sd.output(ph, "out");

            NativeExecutorTestUtils.assertArrayEquals(
                    javaResult.get("out"), nativeResult.get("out"), TOLERANCE, "larger graph JNI");
        } finally {
            nativeOps.freeDynamicShapePlan(handle);
            plan.close();
        }
    }

    // ==================== Stress Test ====================

    @Test
    public void testStressRepeatedExecution() {
        SameDiff sd = NativeExecutorTestUtils.createChainGraph();
        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");

        Pointer handle = compileNativePlan(plan);
        if (handle == null || handle.isNull()) return;

        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        try {
            // Execute 100 times rapidly
            for (int i = 0; i < 100; i++) {
                INDArray x = Nd4j.randn(DataType.FLOAT, 2, 4);
                Map<String, INDArray> ph = Map.of("x", x);

                INDArray[] extInputs = resolveExternalInputs(plan, sd, ph);
                Map<String, INDArray> nativeResult = executeNativePlan(handle, plan, extInputs);

                assertNotNull(nativeResult.get("z"), "Null output at iteration " + i);
                assertArrayEquals(new long[]{2, 4}, nativeResult.get("z").shape(),
                        "Shape mismatch at iteration " + i);
            }

            // Spot-check final result against Java
            INDArray x = Nd4j.randn(DataType.FLOAT, 2, 4);
            Map<String, INDArray> ph = Map.of("x", x);
            INDArray[] extInputs = resolveExternalInputs(plan, sd, ph);
            Map<String, INDArray> nativeResult = executeNativePlan(handle, plan, extInputs);
            Map<String, INDArray> javaResult = sd.output(ph, "z");

            NativeExecutorTestUtils.assertArrayEquals(
                    javaResult.get("z"), nativeResult.get("z"), TOLERANCE, "stress final parity");
        } finally {
            nativeOps.freeDynamicShapePlan(handle);
            plan.close();
        }
    }

    // ==================== Compile/Free Lifecycle ====================

    @Test
    public void testCompileFreeCycle() {
        SameDiff sd = NativeExecutorTestUtils.createAddGraph();
        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();

        // Compile and free 5 times to test lifecycle
        for (int cycle = 0; cycle < 5; cycle++) {
            Pointer handle = compileNativePlan(plan);
            if (handle == null || handle.isNull()) return;

            INDArray x = Nd4j.ones(DataType.FLOAT, 2, 3);
            INDArray y = Nd4j.ones(DataType.FLOAT, 2, 3);
            Map<String, INDArray> ph = Map.of("x", x, "y", y);

            INDArray[] extInputs = resolveExternalInputs(plan, sd, ph);
            Map<String, INDArray> nativeResult = executeNativePlan(handle, plan, extInputs);
            assertNotNull(nativeResult.get("z"), "Null output in cycle " + cycle);

            nativeOps.freeDynamicShapePlan(handle);
        }

        plan.close();
    }

    // ==================== Scalar Operations ====================

    @Test
    public void testScalarAddJNI() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 3);
        SDVariable scalar = sd.constant("scalar", Nd4j.scalar(DataType.FLOAT, 5.0));
        SDVariable z = x.add("z", scalar);

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");
        Pointer handle = compileNativePlan(plan);
        if (handle == null || handle.isNull()) return;

        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        try {
            INDArray x1 = Nd4j.ones(DataType.FLOAT, 2, 3).mul(3);
            Map<String, INDArray> ph = Map.of("x", x1);

            INDArray[] extInputs = resolveExternalInputs(plan, sd, ph);
            Map<String, INDArray> nativeResult = executeNativePlan(handle, plan, extInputs);
            Map<String, INDArray> javaResult = sd.output(ph, "z");

            NativeExecutorTestUtils.assertArrayEquals(
                    javaResult.get("z"), nativeResult.get("z"), TOLERANCE, "scalar add JNI");
        } finally {
            nativeOps.freeDynamicShapePlan(handle);
            plan.close();
        }
    }

    // ==================== Broadcast Operations ====================

    @Test
    public void testBroadcastAddJNI() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 3);
        SDVariable bias = sd.var("bias", Nd4j.ones(DataType.FLOAT, 1, 3).mul(10));
        SDVariable z = x.add("z", bias);

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");
        Pointer handle = compileNativePlan(plan);
        if (handle == null || handle.isNull()) return;

        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        try {
            INDArray x1 = Nd4j.ones(DataType.FLOAT, 2, 3);
            Map<String, INDArray> ph = Map.of("x", x1);

            INDArray[] extInputs = resolveExternalInputs(plan, sd, ph);
            Map<String, INDArray> nativeResult = executeNativePlan(handle, plan, extInputs);
            Map<String, INDArray> javaResult = sd.output(ph, "z");

            NativeExecutorTestUtils.assertArrayEquals(
                    javaResult.get("z"), nativeResult.get("z"), TOLERANCE, "broadcast add JNI");
        } finally {
            nativeOps.freeDynamicShapePlan(handle);
            plan.close();
        }
    }

    // ==================== Plan Display & Execution Evidence ====================

    /**
     * Helper: print the full plan details — slot-by-slot wiring, args, flags.
     */
    private void printPlanDetails(DynamicShapePlan plan) {
        DynamicShapeSlot[] slots = plan.getSlots();
        String[] extKeys = plan.getExternalInputKeys();
        int[][] releaseAtStep = plan.getReleaseAtStep();

        log.info("╔══════════════════════════════════════════════════════════════╗");
        log.info("║              DYNAMIC SHAPE PLAN DETAILS                     ║");
        log.info("╠══════════════════════════════════════════════════════════════╣");
        log.info("║ Slots: {}  |  Output slots: {}  |  External inputs: {}  |  Outputs: {}",
                slots.length, plan.getTotalOutputSlots(), extKeys.length,
                plan.getRequestedOutputs().size());

        // External inputs
        log.info("║ External inputs:");
        for (int i = 0; i < extKeys.length; i++) {
            log.info("║   ext[{}] = \"{}\"", i, extKeys[i]);
        }

        // Requested outputs
        log.info("║ Requested outputs: {}", plan.getOutputNameToSlotIndex());

        log.info("╠══════════════════════════════════════════════════════════════╣");

        // Per-slot details
        for (int s = 0; s < slots.length; s++) {
            DynamicShapeSlot slot = slots[s];
            log.info("║ Slot[{}]: op=\"{}\"  hash={}  customOp={}  device={}",
                    s, slot.getOpName(), slot.getOpNameHash(), slot.isCustomOp(),
                    slot.getTargetDeviceId());

            // Inputs
            StringBuilder inputDesc = new StringBuilder();
            int[] srcIdx = slot.getInputSourceIndices();
            byte[] srcTypes = slot.getInputSourceTypes();
            String[] inputNames = slot.getInputVarNames();
            for (int i = 0; i < srcIdx.length; i++) {
                String typeStr;
                switch (srcTypes[i]) {
                    case DynamicShapeSlot.SOURCE_CONSTANT: typeStr = "CONST"; break;
                    case DynamicShapeSlot.SOURCE_VARIABLE: typeStr = "VAR"; break;
                    case DynamicShapeSlot.SOURCE_PLACEHOLDER: typeStr = "PH"; break;
                    case DynamicShapeSlot.SOURCE_OP_OUTPUT: typeStr = "SLOT"; break;
                    default: typeStr = "?"; break;
                }
                if (srcIdx[i] >= 0) {
                    inputDesc.append(String.format("  in[%d] = slot[%d] (%s, %s)", i, srcIdx[i], typeStr, inputNames[i]));
                } else {
                    inputDesc.append(String.format("  in[%d] = ext[%d] (%s, %s)", i, -(srcIdx[i] + 1), typeStr, inputNames[i]));
                }
                if (i < srcIdx.length - 1) inputDesc.append("\n║");
            }
            log.info("║   Inputs:  {}", inputDesc);

            // Outputs
            int[] outIdx = slot.getOutputSlotIndices();
            String[] outNames = slot.getOutputVarNames();
            StringBuilder outputDesc = new StringBuilder();
            for (int i = 0; i < outIdx.length; i++) {
                outputDesc.append(String.format("out[%d] → slot[%d] (\"%s\")", i, outIdx[i],
                        i < outNames.length ? outNames[i] : "?"));
                if (i < outIdx.length - 1) outputDesc.append(", ");
            }
            log.info("║   Outputs: {}", outputDesc);

            // Args
            if (slot.getIArgs() != null && slot.getIArgs().length > 0)
                log.info("║   iArgs: {}", Arrays.toString(slot.getIArgs()));
            if (slot.getTArgs() != null && slot.getTArgs().length > 0)
                log.info("║   tArgs: {}", Arrays.toString(slot.getTArgs()));
            if (slot.getBArgs() != null && slot.getBArgs().length > 0)
                log.info("║   bArgs: {}", Arrays.toString(slot.getBArgs()));
            if (slot.getDArgs() != null && slot.getDArgs().length > 0)
                log.info("║   dArgs: {}", Arrays.toString(slot.getDArgs()));

            // Flags
            StringBuilder flags = new StringBuilder();
            if (slot.isNeedsZeroedOutput()) flags.append("ZEROED ");
            if (slot.isDataDependent()) flags.append("DATA_DEP ");
            if (slot.isOutputShapeDependsOnInputValues()) flags.append("SHAPE_DEP_VALUES ");
            if (slot.isNeedsIntLongSync()) flags.append("INT_SYNC ");
            if (flags.length() > 0) log.info("║   Flags: {}", flags.toString().trim());

            // Release schedule
            if (releaseAtStep[s] != null && releaseAtStep[s].length > 0) {
                log.info("║   Release after step: slots {}", Arrays.toString(releaseAtStep[s]));
            }
        }

        log.info("╚══════════════════════════════════════════════════════════════╝");
    }

    /**
     * Helper: print C++ plan handle metadata from JNI queries.
     */
    private void printNativePlanInfo(Pointer handle) {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        int numSlots = nativeOps.getPlanNumSlots(handle);
        int numExtInputs = nativeOps.getPlanNumExternalInputs(handle);
        int numOutputs = nativeOps.getPlanNumRequestedOutputs(handle);
        int numSegments = nativeOps.getPlanNumSegments(handle);
        int capturedGraphs = nativeOps.getPlanNumCapturedGraphSegments(handle);
        int totalReplays = nativeOps.getPlanTotalGraphReplays(handle);

        log.info("┌─ C++ Native Plan Handle ─────────────────────────────────┐");
        log.info("│ Slots:           {}                                      ", numSlots);
        log.info("│ External inputs: {}                                      ", numExtInputs);
        log.info("│ Outputs:         {}                                      ", numOutputs);
        log.info("│ Segments:        {}                                      ", numSegments);
        log.info("│ Captured graphs: {}                                      ", capturedGraphs);
        log.info("│ Graph replays:   {}                                      ", totalReplays);
        log.info("└──────────────────────────────────────────────────────────┘");
    }

    @Test
    public void testPlanDisplayAndExecutionEvidence() {
        // Build a multi-layer graph: matmul → bias → relu → sigmoid → reduce
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 8);
        SDVariable w1 = sd.var("w1", Nd4j.randn(DataType.FLOAT, 8, 16).mul(0.01));
        SDVariable b1 = sd.var("b1", Nd4j.zeros(DataType.FLOAT, 1, 16));
        SDVariable w2 = sd.var("w2", Nd4j.randn(DataType.FLOAT, 16, 4).mul(0.01));
        SDVariable mm1 = sd.mmul("mm1", x, w1);
        SDVariable biased = mm1.add("biased", b1);
        SDVariable act = sd.nn().relu("relu", biased, 0.0);
        SDVariable mm2 = sd.mmul("mm2", act, w2);
        SDVariable sig = sd.nn().sigmoid("sig", mm2);
        SDVariable out = sd.mean("out", sig, 1);

        // ── 1. Compile plan and display details ──
        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "out");
        log.info("=== PLAN COMPILED ===");
        printPlanDetails(plan);

        Pointer handle = compileNativePlan(plan);
        if (handle == null || handle.isNull()) {
            log.info("Skipping - backend doesn't support native executor");
            plan.close();
            return;
        }

        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        try {
            log.info("\n=== NATIVE PLAN HANDLE (before execution) ===");
            printNativePlanInfo(handle);

            INDArray x1 = Nd4j.randn(DataType.FLOAT, 4, 8);
            Map<String, INDArray> ph = Map.of("x", x1);
            INDArray[] extInputs = resolveExternalInputs(plan, sd, ph);

            // ── 2. Time Java execution (warmup + measured) ──
            log.info("\n=== JAVA EXECUTION TIMING ===");
            // Warmup
            for (int i = 0; i < 5; i++) {
                sd.output(ph, "out");
            }
            long javaStart = System.nanoTime();
            int javaIterations = 100;
            Map<String, INDArray> javaResult = null;
            for (int i = 0; i < javaIterations; i++) {
                javaResult = sd.output(ph, "out");
            }
            long javaEnd = System.nanoTime();
            double javaAvgMs = (javaEnd - javaStart) / 1_000_000.0 / javaIterations;
            log.info("Java executor: {} iterations, avg {} ms/exec", javaIterations, String.format("%.3f", javaAvgMs));

            // ── 3. Time native execution WITHOUT CUDA Graphs ──
            log.info("\n=== NATIVE C++ EXECUTION TIMING (no CUDA Graphs) ===");
            nativeOps.setPlanCudaGraphsEnabled(handle, false);
            // Warmup
            for (int i = 0; i < 5; i++) {
                executeNativePlan(handle, plan, extInputs);
            }
            long nativeStart = System.nanoTime();
            Map<String, INDArray> nativeResult = null;
            for (int i = 0; i < javaIterations; i++) {
                nativeResult = executeNativePlan(handle, plan, extInputs);
            }
            long nativeEnd = System.nanoTime();
            double nativeAvgMs = (nativeEnd - nativeStart) / 1_000_000.0 / javaIterations;
            log.info("Native C++ (no graphs): {} iterations, avg {} ms/exec", javaIterations, String.format("%.3f", nativeAvgMs));

            // Verify parity
            NativeExecutorTestUtils.assertArrayEquals(
                    javaResult.get("out"), nativeResult.get("out"), TOLERANCE,
                    "plan display - parity without graphs");

            log.info("\n=== NATIVE PLAN HANDLE (after execution, no graphs) ===");
            printNativePlanInfo(handle);

            // ── 4. Time native execution WITH CUDA Graphs ──
            log.info("\n=== NATIVE C++ EXECUTION TIMING (with CUDA Graphs) ===");
            nativeOps.setPlanCudaGraphsEnabled(handle, true);
            nativeOps.clearDynamicShapePlanCaches(handle);
            // Warmup: first exec = slot-by-slot, second = capture, third+ = replay
            for (int i = 0; i < 5; i++) {
                executeNativePlan(handle, plan, extInputs);
            }

            log.info("After warmup:");
            printNativePlanInfo(handle);

            long graphStart = System.nanoTime();
            Map<String, INDArray> graphResult = null;
            for (int i = 0; i < javaIterations; i++) {
                graphResult = executeNativePlan(handle, plan, extInputs);
            }
            long graphEnd = System.nanoTime();
            double graphAvgMs = (graphEnd - graphStart) / 1_000_000.0 / javaIterations;
            log.info("Native C++ (with graphs): {} iterations, avg {} ms/exec", javaIterations, String.format("%.3f", graphAvgMs));

            // Verify parity
            NativeExecutorTestUtils.assertArrayEquals(
                    javaResult.get("out"), graphResult.get("out"), TOLERANCE,
                    "plan display - parity with graphs");

            log.info("\n=== NATIVE PLAN HANDLE (after {} graph executions) ===", javaIterations);
            printNativePlanInfo(handle);

            // ── 5. Summary comparison ──
            log.info("\n╔══════════════════════════════════════════════════════════════╗");
            log.info("║              TIMING COMPARISON SUMMARY                      ║");
            log.info("╠══════════════════════════════════════════════════════════════╣");
            log.info("║ Graph: {} slots, {} ext inputs → {} output", plan.getSlots().length,
                    plan.getExternalInputKeys().length, plan.getRequestedOutputs().size());
            log.info("║ Input shape: [4, 8]");
            log.info("║                                                              ");
            log.info("║ Java DSP executor:           {} ms/exec", String.format("%.3f", javaAvgMs));
            log.info("║ Native C++ (slot-by-slot):   {} ms/exec", String.format("%.3f", nativeAvgMs));
            log.info("║ Native C++ (CUDA Graphs):    {} ms/exec", String.format("%.3f", graphAvgMs));
            if (javaAvgMs > 0) {
                log.info("║                                                              ");
                log.info("║ Speedup (native vs Java):    {}x", String.format("%.1f", javaAvgMs / nativeAvgMs));
                log.info("║ Speedup (graphs vs Java):    {}x", String.format("%.1f", javaAvgMs / graphAvgMs));
            }
            log.info("║ Graph replays: {}", nativeOps.getPlanTotalGraphReplays(handle));
            log.info("║ Captured segments: {}", nativeOps.getPlanNumCapturedGraphSegments(handle));
            log.info("╚══════════════════════════════════════════════════════════════╝");

        } finally {
            nativeOps.freeDynamicShapePlan(handle);
            plan.close();
        }
    }

    @Test
    public void testCudaGraphReplayEvidence() {
        // This test provides concrete evidence of CUDA graph capture and replay:
        // - Shows segment count and which are capturable
        // - Executes multiple times and counts replays
        // - Verifies replay count matches expected pattern

        SameDiff sd = NativeExecutorTestUtils.createChainGraph();
        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");
        printPlanDetails(plan);

        Pointer handle = compileNativePlan(plan);
        if (handle == null || handle.isNull()) {
            plan.close();
            return;
        }

        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        try {
            nativeOps.setPlanCudaGraphsEnabled(handle, true);

            INDArray x = Nd4j.randn(DataType.FLOAT, 3, 4);
            Map<String, INDArray> ph = Map.of("x", x);
            INDArray[] extInputs = resolveExternalInputs(plan, sd, ph);

            int numSegments = nativeOps.getPlanNumSegments(handle);
            log.info("Plan has {} segments", numSegments);

            // Execute 10 times and track graph capture/replay progression
            for (int i = 0; i < 10; i++) {
                Map<String, INDArray> result = executeNativePlan(handle, plan, extInputs);
                int captured = nativeOps.getPlanNumCapturedGraphSegments(handle);
                int replays = nativeOps.getPlanTotalGraphReplays(handle);
                log.info("  Exec #{}: captured={}, replays={}, output_shape={}",
                        i + 1, captured, replays, Arrays.toString(result.get("z").shape()));

                // Verify correctness every time
                Map<String, INDArray> javaResult = sd.output(ph, "z");
                NativeExecutorTestUtils.assertArrayEquals(
                        javaResult.get("z"), result.get("z"), TOLERANCE,
                        "graph replay exec #" + (i + 1));
            }

            int finalCaptured = nativeOps.getPlanNumCapturedGraphSegments(handle);
            int finalReplays = nativeOps.getPlanTotalGraphReplays(handle);
            log.info("\nFinal: {} captured graphs, {} total replays", finalCaptured, finalReplays);

            // After 10 executions with the same shapes, we expect replays
            // (first exec = warmup/slot-by-slot, second = capture, 3rd+ = replay)
            if (finalCaptured > 0) {
                assertTrue(finalReplays > 0, "Expected graph replays after capture");
                log.info("CUDA Graph replay confirmed: {} replays from {} captured segments",
                        finalReplays, finalCaptured);
            } else {
                log.info("No CUDA graphs captured (may be CPU-only backend)");
            }
        } finally {
            nativeOps.freeDynamicShapePlan(handle);
            plan.close();
        }
    }

    @Test
    public void testTimingScalingWithGraphSize() {
        // Test how execution time scales with graph size for Java vs native
        // Builds progressively larger graphs and compares timings
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        int[] layerCounts = {1, 2, 4, 8};
        int warmup = 10;
        int measured = 50;

        log.info("\n╔══════════════════════════════════════════════════════════════════════╗");
        log.info("║         TIMING vs GRAPH SIZE: Java DSP vs Native C++               ║");
        log.info("╠══════════════════════════════════════════════════════════════════════╣");
        log.info(String.format("║ %6s │ %6s │ %12s │ %12s │ %12s │ %7s ║",
                "Layers", "Slots", "Java (ms)", "Native (ms)", "Graphs (ms)", "Speedup"));
        log.info("╠══════════════════════════════════════════════════════════════════════╣");

        boolean cudaStateHealthy = true;  // Track across iterations
        for (int numLayers : layerCounts) {
            // If CUDA state is corrupted from a previous graph capture failure,
            // skip remaining sizes (the CUDA stream state persists globally)
            if (!cudaStateHealthy) {
                log.info(String.format("║ %6d │ %6s │ %12s │ %12s │ %12s │ %7s ║",
                        numLayers, "?", "skipped", "skipped", "skipped", "N/A"));
                continue;
            }

            // Build a multi-layer MLP: [matmul → bias → relu] × numLayers → sigmoid → mean
            SameDiff sd = SameDiff.create();
            int dim = 16;
            SDVariable current = sd.placeHolder("x", DataType.FLOAT, -1, dim);
            for (int layer = 0; layer < numLayers; layer++) {
                SDVariable w = sd.var("w" + layer, Nd4j.randn(DataType.FLOAT, dim, dim).mul(0.01));
                SDVariable b = sd.var("b" + layer, Nd4j.zeros(DataType.FLOAT, 1, dim));
                current = sd.mmul("mm" + layer, current, w);
                current = current.add("bias" + layer, b);
                current = sd.nn().relu("relu" + layer, current, 0.0);
            }
            current = sd.nn().sigmoid("sig", current);
            SDVariable out = sd.mean("out", current, 1);

            DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "out");
            int totalSlots = plan.getSlots().length;

            INDArray x1 = Nd4j.randn(DataType.FLOAT, 8, dim);
            Map<String, INDArray> ph = Map.of("x", x1);
            INDArray[] extInputs = resolveExternalInputs(plan, sd, ph);

            // Compile native plan
            Pointer handle = compileNativePlan(plan);
            if (handle == null || handle.isNull()) {
                log.info(String.format("║ %6d │ %6d │ N/A - backend doesn't support native executor",
                        numLayers, totalSlots));
                plan.close();
                continue;
            }

            try {
                // ── Java timing ──
                for (int i = 0; i < warmup; i++) sd.output(ph, "out");
                long jStart = System.nanoTime();
                for (int i = 0; i < measured; i++) sd.output(ph, "out");
                double javaMs = (System.nanoTime() - jStart) / 1_000_000.0 / measured;

                // ── Native timing (no graphs) ──
                nativeOps.setPlanCudaGraphsEnabled(handle, false);
                for (int i = 0; i < warmup; i++) executeNativePlan(handle, plan, extInputs);
                long nStart = System.nanoTime();
                for (int i = 0; i < measured; i++) executeNativePlan(handle, plan, extInputs);
                double nativeMs = (System.nanoTime() - nStart) / 1_000_000.0 / measured;

                // ── Native timing (with CUDA graphs) ──
                double graphMs = -1;
                try {
                    nativeOps.setPlanCudaGraphsEnabled(handle, true);
                    nativeOps.clearDynamicShapePlanCaches(handle);
                    for (int i = 0; i < warmup; i++) executeNativePlan(handle, plan, extInputs);
                    long gStart = System.nanoTime();
                    for (int i = 0; i < measured; i++) executeNativePlan(handle, plan, extInputs);
                    graphMs = (System.nanoTime() - gStart) / 1_000_000.0 / measured;
                } catch (Throwable e) {
                    log.info("  CUDA Graphs capture failed for {} layers: {} — disabling for remaining sizes",
                            numLayers, e.getMessage());
                    cudaStateHealthy = false;
                }
                // Always disable CUDA graphs before continuing
                nativeOps.setPlanCudaGraphsEnabled(handle, false);
                nativeOps.clearDynamicShapePlanCaches(handle);

                double speedup;
                if (graphMs > 0) {
                    speedup = javaMs / Math.max(graphMs, 0.001);
                    log.info(String.format("║ %6d │ %6d │ %12.3f │ %12.3f │ %12.3f │ %6.1fx ║",
                            numLayers, totalSlots, javaMs, nativeMs, graphMs, speedup));
                } else {
                    speedup = javaMs / Math.max(nativeMs, 0.001);
                    log.info(String.format("║ %6d │ %6d │ %12.3f │ %12.3f │ %12s │ %6.1fx ║",
                            numLayers, totalSlots, javaMs, nativeMs, "N/A", speedup));
                }

                // Verify correctness (only when CUDA stream state is healthy)
                if (cudaStateHealthy) {
                    Map<String, INDArray> javaResult = sd.output(ph, "out");
                    Map<String, INDArray> nativeResult = executeNativePlan(handle, plan, extInputs);
                    NativeExecutorTestUtils.assertArrayEquals(
                            javaResult.get("out"), nativeResult.get("out"), TOLERANCE,
                            "scaling test " + numLayers + " layers");
                }
            } finally {
                nativeOps.freeDynamicShapePlan(handle);
                plan.close();
            }
        }

        log.info("╚══════════════════════════════════════════════════════════════════════╝");
    }

    // ─── Comprehensive Op Coverage ───────────────────────────────────────────────

    /**
     * Helper: build graph, compile plan, run native + Java, assert parity.
     * Returns true if parity passes, false if native execution fails.
     * Logs result for each op. Never silently swallows errors.
     */
    private boolean verifyOpParity(String opLabel, SameDiff sd, Map<String, INDArray> ph, String outputName) {
        try {
            DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, outputName);
            INDArray[] extInputs = resolveExternalInputs(plan, sd, ph);
            Pointer handle = compileNativePlan(plan);
            if (handle == null || handle.isNull()) {
                log.info("  [SKIP] {} — native compile returned null", opLabel);
                plan.close();
                return false;
            }
            try {
                Map<String, INDArray> javaResult = sd.output(ph, outputName);
                Map<String, INDArray> nativeResult = executeNativePlan(handle, plan, extInputs);
                INDArray javaOut = javaResult.get(outputName);
                INDArray nativeOut = nativeResult.get(outputName);
                try {
                    NativeExecutorTestUtils.assertArrayEquals(javaOut, nativeOut, TOLERANCE, opLabel);
                } catch (AssertionError ae) {
                    // Print diagnostic info on failure
                    log.error("  [DIAG] {} — plan has {} slots, {} external inputs",
                            opLabel, plan.getSlots().length,
                            plan.getExternalInputKeys().length);
                    log.error("  [DIAG] {} — external input keys: {}", opLabel,
                            Arrays.toString(plan.getExternalInputKeys()));
                    log.error("  [DIAG] {} — java shape={} native shape={}", opLabel,
                            javaOut != null ? Arrays.toString(javaOut.shape()) : "null",
                            nativeOut != null ? Arrays.toString(nativeOut.shape()) : "null");
                    if (javaOut != null && javaOut.length() <= 20) {
                        log.error("  [DIAG] {} — java values: {}", opLabel, javaOut);
                    }
                    if (nativeOut != null && nativeOut.length() <= 20) {
                        log.error("  [DIAG] {} — native values: {}", opLabel, nativeOut);
                    }
                    for (int i = 0; i < plan.getSlots().length; i++) {
                        var slot = plan.getSlots()[i];
                        log.error("  [DIAG] {} — slot[{}]: op={} legacyType={} legacyNum={} numInputs={} numOutputs={} tArgs={}",
                                opLabel, i, slot.getOpName(), slot.getLegacyOpType(), slot.getLegacyOpNum(),
                                slot.getInputSourceIndices().length, slot.getOutputSlotIndices().length,
                                Arrays.toString(slot.getTArgs()));
                    }
                    throw ae;
                }
                log.info("  [PASS] {} shape={}", opLabel,
                        Arrays.toString(javaOut.shape()));
                return true;
            } finally {
                NativeOpsHolder.getInstance().getDeviceNativeOps().freeDynamicShapePlan(handle);
                plan.close();
            }
        } catch (Throwable e) {
            log.error("  [FAIL] {} — {}", opLabel, e.getMessage());
            return false;
        }
    }

    @Test
    @Tag(TagNames.LONG_TEST)
    public void testComprehensiveOpCoverage() {
        log.info("╔══════════════════════════════════════════════════════════════╗");
        log.info("║           COMPREHENSIVE OP COVERAGE TEST                    ║");
        log.info("╚══════════════════════════════════════════════════════════════╝");

        INDArray a34 = Nd4j.randn(DataType.FLOAT, 3, 4).mul(2);
        INDArray b34 = Nd4j.randn(DataType.FLOAT, 3, 4).mul(2);
        INDArray w48 = Nd4j.randn(DataType.FLOAT, 4, 8).mul(0.1);
        int passed = 0, failed = 0;

        // ── Binary element-wise ops ──
        {
            SameDiff sd = SameDiff.create();
            SDVariable x = sd.placeHolder("x", DataType.FLOAT, 3, 4);
            SDVariable y = sd.placeHolder("y", DataType.FLOAT, 3, 4);
            sd.math().sub("out", x, y);
            if (verifyOpParity("subtract", sd, Map.of("x", a34, "y", b34), "out")) passed++; else failed++;
        }
        {
            SameDiff sd = SameDiff.create();
            SDVariable x = sd.placeHolder("x", DataType.FLOAT, 3, 4);
            SDVariable y = sd.placeHolder("y", DataType.FLOAT, 3, 4);
            sd.math().div("out", x, y);
            if (verifyOpParity("divide", sd, Map.of("x", a34, "y", b34.add(1)), "out")) passed++; else failed++;
        }
        {
            SameDiff sd = SameDiff.create();
            SDVariable x = sd.placeHolder("x", DataType.FLOAT, 3, 4);
            SDVariable y = sd.placeHolder("y", DataType.FLOAT, 3, 4);
            sd.math().max("out", x, y);
            if (verifyOpParity("maximum", sd, Map.of("x", a34, "y", b34), "out")) passed++; else failed++;
        }
        {
            SameDiff sd = SameDiff.create();
            SDVariable x = sd.placeHolder("x", DataType.FLOAT, 3, 4);
            SDVariable y = sd.placeHolder("y", DataType.FLOAT, 3, 4);
            sd.math().min("out", x, y);
            if (verifyOpParity("minimum", sd, Map.of("x", a34, "y", b34), "out")) passed++; else failed++;
        }

        // ── Unary math ops ──
        {
            SameDiff sd = SameDiff.create();
            SDVariable x = sd.placeHolder("x", DataType.FLOAT, 3, 4);
            sd.math().tanh("out", x);
            if (verifyOpParity("tanh", sd, Map.of("x", a34), "out")) passed++; else failed++;
        }
        {
            SameDiff sd = SameDiff.create();
            SDVariable x = sd.placeHolder("x", DataType.FLOAT, 3, 4);
            sd.math().exp("out", x);
            if (verifyOpParity("exp", sd, Map.of("x", a34.mul(0.5)), "out")) passed++; else failed++;
        }
        {
            SameDiff sd = SameDiff.create();
            SDVariable x = sd.placeHolder("x", DataType.FLOAT, 3, 4);
            sd.math().log("out", x);
            if (verifyOpParity("log", sd, Map.of("x", Nd4j.math().abs(a34).add(0.1)), "out")) passed++; else failed++;
        }
        {
            SameDiff sd = SameDiff.create();
            SDVariable x = sd.placeHolder("x", DataType.FLOAT, 3, 4);
            sd.math().abs("out", x);
            if (verifyOpParity("abs", sd, Map.of("x", a34), "out")) passed++; else failed++;
        }
        {
            SameDiff sd = SameDiff.create();
            SDVariable x = sd.placeHolder("x", DataType.FLOAT, 3, 4);
            sd.math().neg("out", x);
            if (verifyOpParity("neg", sd, Map.of("x", a34), "out")) passed++; else failed++;
        }
        {
            SameDiff sd = SameDiff.create();
            SDVariable x = sd.placeHolder("x", DataType.FLOAT, 3, 4);
            sd.math().square("out", x);
            if (verifyOpParity("square", sd, Map.of("x", a34), "out")) passed++; else failed++;
        }
        {
            SameDiff sd = SameDiff.create();
            SDVariable x = sd.placeHolder("x", DataType.FLOAT, 3, 4);
            sd.math().sqrt("out", x);
            if (verifyOpParity("sqrt", sd, Map.of("x", Nd4j.math().abs(a34).add(0.01)), "out")) passed++; else failed++;
        }
        {
            SameDiff sd = SameDiff.create();
            SDVariable x = sd.placeHolder("x", DataType.FLOAT, 3, 4);
            sd.math().reciprocal("out", x);
            if (verifyOpParity("reciprocal", sd, Map.of("x", Nd4j.math().abs(a34).add(0.1)), "out")) passed++; else failed++;
        }
        {
            SameDiff sd = SameDiff.create();
            SDVariable x = sd.placeHolder("x", DataType.FLOAT, 3, 4);
            sd.math().ceil("out", x);
            if (verifyOpParity("ceil", sd, Map.of("x", a34), "out")) passed++; else failed++;
        }
        {
            SameDiff sd = SameDiff.create();
            SDVariable x = sd.placeHolder("x", DataType.FLOAT, 3, 4);
            sd.math().floor("out", x);
            if (verifyOpParity("floor", sd, Map.of("x", a34), "out")) passed++; else failed++;
        }
        {
            SameDiff sd = SameDiff.create();
            SDVariable x = sd.placeHolder("x", DataType.FLOAT, 3, 4);
            sd.math().sign("out", x);
            if (verifyOpParity("sign", sd, Map.of("x", a34), "out")) passed++; else failed++;
        }
        {
            SameDiff sd = SameDiff.create();
            SDVariable x = sd.placeHolder("x", DataType.FLOAT, 3, 4);
            sd.math().sin("out", x);
            if (verifyOpParity("sin", sd, Map.of("x", a34), "out")) passed++; else failed++;
        }
        {
            SameDiff sd = SameDiff.create();
            SDVariable x = sd.placeHolder("x", DataType.FLOAT, 3, 4);
            sd.math().cos("out", x);
            if (verifyOpParity("cos", sd, Map.of("x", a34), "out")) passed++; else failed++;
        }

        // ── Activation functions ──
        {
            SameDiff sd = SameDiff.create();
            SDVariable x = sd.placeHolder("x", DataType.FLOAT, 3, 4);
            sd.nn().softmax("out", x, 1);
            if (verifyOpParity("softmax", sd, Map.of("x", a34), "out")) passed++; else failed++;
        }
        {
            SameDiff sd = SameDiff.create();
            SDVariable x = sd.placeHolder("x", DataType.FLOAT, 3, 4);
            sd.nn().elu("out", x);
            if (verifyOpParity("elu", sd, Map.of("x", a34), "out")) passed++; else failed++;
        }
        {
            SameDiff sd = SameDiff.create();
            SDVariable x = sd.placeHolder("x", DataType.FLOAT, 3, 4);
            sd.nn().leakyRelu("out", x, 0.1);
            if (verifyOpParity("leaky_relu", sd, Map.of("x", a34), "out")) passed++; else failed++;
        }
        {
            SameDiff sd = SameDiff.create();
            SDVariable x = sd.placeHolder("x", DataType.FLOAT, 3, 4);
            sd.nn().swish("out", x);
            if (verifyOpParity("swish", sd, Map.of("x", a34), "out")) passed++; else failed++;
        }
        {
            SameDiff sd = SameDiff.create();
            SDVariable x = sd.placeHolder("x", DataType.FLOAT, 3, 4);
            sd.nn().gelu("out", x);
            if (verifyOpParity("gelu", sd, Map.of("x", a34), "out")) passed++; else failed++;
        }

        // ── Matrix ops ──
        {
            SameDiff sd = SameDiff.create();
            SDVariable x = sd.placeHolder("x", DataType.FLOAT, 3, 4);
            SDVariable w = sd.var("w", w48);
            sd.mmul("out", x, w);
            if (verifyOpParity("matmul", sd, Map.of("x", a34), "out")) passed++; else failed++;
        }

        // ── Reduction ops (keepDims=false) ──
        {
            SameDiff sd = SameDiff.create();
            SDVariable x = sd.placeHolder("x", DataType.FLOAT, 3, 4);
            sd.sum("out", x, 1);
            if (verifyOpParity("reduce_sum(axis=1)", sd, Map.of("x", a34), "out")) passed++; else failed++;
        }
        {
            SameDiff sd = SameDiff.create();
            SDVariable x = sd.placeHolder("x", DataType.FLOAT, 3, 4);
            sd.mean("out", x, 0);
            if (verifyOpParity("reduce_mean(axis=0)", sd, Map.of("x", a34), "out")) passed++; else failed++;
        }
        {
            SameDiff sd = SameDiff.create();
            SDVariable x = sd.placeHolder("x", DataType.FLOAT, 3, 4);
            sd.max("out", x, 1);
            if (verifyOpParity("reduce_max(axis=1)", sd, Map.of("x", a34), "out")) passed++; else failed++;
        }
        {
            SameDiff sd = SameDiff.create();
            SDVariable x = sd.placeHolder("x", DataType.FLOAT, 3, 4);
            sd.min("out", x, 1);
            if (verifyOpParity("reduce_min(axis=1)", sd, Map.of("x", a34), "out")) passed++; else failed++;
        }
        {
            SameDiff sd = SameDiff.create();
            SDVariable x = sd.placeHolder("x", DataType.FLOAT, 3, 4);
            sd.prod("out", x, 1);
            if (verifyOpParity("reduce_prod(axis=1)", sd, Map.of("x", a34.mul(0.5).add(0.5)), "out")) passed++; else failed++;
        }

        // ── Shape manipulation ──
        {
            SameDiff sd = SameDiff.create();
            SDVariable x = sd.placeHolder("x", DataType.FLOAT, 3, 4);
            sd.reshape("out", x, 12);
            if (verifyOpParity("reshape", sd, Map.of("x", a34), "out")) passed++; else failed++;
        }
        {
            SameDiff sd = SameDiff.create();
            SDVariable x = sd.placeHolder("x", DataType.FLOAT, 3, 4);
            sd.permute("out", x, 1, 0);
            if (verifyOpParity("transpose/permute", sd, Map.of("x", a34), "out")) passed++; else failed++;
        }
        {
            SameDiff sd = SameDiff.create();
            SDVariable x = sd.placeHolder("x", DataType.FLOAT, 3, 4);
            sd.expandDims("out", x, 0);
            if (verifyOpParity("expand_dims", sd, Map.of("x", a34), "out")) passed++; else failed++;
        }
        {
            SameDiff sd = SameDiff.create();
            SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, 3, 4);
            sd.squeeze("out", x, 0);
            if (verifyOpParity("squeeze", sd, Map.of("x", Nd4j.randn(DataType.FLOAT, 1, 3, 4)), "out")) passed++; else failed++;
        }

        // ── Concat / stack ──
        {
            SameDiff sd = SameDiff.create();
            SDVariable x = sd.placeHolder("x", DataType.FLOAT, 3, 4);
            SDVariable y = sd.placeHolder("y", DataType.FLOAT, 3, 4);
            sd.concat("out", 0, x, y);
            if (verifyOpParity("concat(axis=0)", sd, Map.of("x", a34, "y", b34), "out")) passed++; else failed++;
        }

        // ── Cast ──
        {
            SameDiff sd = SameDiff.create();
            SDVariable x = sd.placeHolder("x", DataType.FLOAT, 3, 4);
            x.castTo("out", DataType.DOUBLE);
            if (verifyOpParity("cast(FLOAT->DOUBLE)", sd, Map.of("x", a34), "out")) passed++; else failed++;
        }

        // ── Power ──
        {
            SameDiff sd = SameDiff.create();
            SDVariable x = sd.placeHolder("x", DataType.FLOAT, 3, 4);
            sd.math().pow("out", x, 2.0);
            if (verifyOpParity("pow", sd, Map.of("x", Nd4j.math().abs(a34).add(0.1)), "out")) passed++; else failed++;
        }

        // ── Clip ──
        {
            SameDiff sd = SameDiff.create();
            SDVariable x = sd.placeHolder("x", DataType.FLOAT, 3, 4);
            sd.math().clipByValue("out", x, -1.0, 1.0);
            if (verifyOpParity("clip_by_value", sd, Map.of("x", a34), "out")) passed++; else failed++;
        }

        // ── Multi-op chains (test arg extraction for chained ops) ──
        {
            SameDiff sd = SameDiff.create();
            SDVariable x = sd.placeHolder("x", DataType.FLOAT, 3, 4);
            SDVariable w = sd.var("w", w48);
            SDVariable b = sd.var("b", Nd4j.zeros(DataType.FLOAT, 1, 8));
            SDVariable mm = sd.mmul("mm", x, w);
            SDVariable biased = mm.add("biased", b);
            SDVariable act = sd.nn().gelu("act", biased);
            sd.nn().softmax("out", act, 1);
            if (verifyOpParity("matmul→add→gelu→softmax", sd, Map.of("x", a34), "out")) passed++; else failed++;
        }
        // ── rms_norm sub-chain tests: isolate which op in the chain fails ──
        // Sub-chain 1: square → mean (2 ops)
        {
            SameDiff sd = SameDiff.create();
            SDVariable x = sd.placeHolder("x", DataType.FLOAT, 3, 4);
            SDVariable sq = sd.math().square("sq", x);
            sd.mean("out", sq, 1);
            if (verifyOpParity("rms_sq_mean", sd, Map.of("x", a34), "out")) passed++; else failed++;
        }
        // Sub-chain 2: standalone add_scalar (legacy ScalarAdd in chain)
        {
            SameDiff sd = SameDiff.create();
            SDVariable x = sd.placeHolder("x", DataType.FLOAT, 3);
            x.add("out", 1e-6);
            INDArray input3 = Nd4j.create(new float[]{0.5f, 1.0f, 2.0f});
            if (verifyOpParity("add_scalar_1d", sd, Map.of("x", input3), "out")) passed++; else failed++;
        }
        // Sub-chain 3: add_scalar → sqrt (2 ops with legacy ScalarAdd)
        {
            SameDiff sd = SameDiff.create();
            SDVariable x = sd.placeHolder("x", DataType.FLOAT, 3);
            SDVariable added = x.add(1e-6);
            sd.math().sqrt("out", added);
            INDArray input3 = Nd4j.create(new float[]{0.5f, 1.0f, 2.0f});
            if (verifyOpParity("add_scalar_sqrt", sd, Map.of("x", input3), "out")) passed++; else failed++;
        }
        // Sub-chain 4: square → mean → add_scalar (3 ops)
        {
            SameDiff sd = SameDiff.create();
            SDVariable x = sd.placeHolder("x", DataType.FLOAT, 3, 4);
            SDVariable sq = sd.math().square("sq", x);
            SDVariable mn = sd.mean("mn", sq, 1);
            mn.add("out", 1e-6);
            if (verifyOpParity("sq_mean_addscalar", sd, Map.of("x", a34), "out")) passed++; else failed++;
        }
        // Sub-chain 5: square → mean → add_scalar → sqrt (4 ops)
        {
            SameDiff sd = SameDiff.create();
            SDVariable x = sd.placeHolder("x", DataType.FLOAT, 3, 4);
            SDVariable sq = sd.math().square("sq", x);
            SDVariable mn = sd.mean("mn", sq, 1);
            sd.math().sqrt("out", mn.add(1e-6));
            if (verifyOpParity("sq_mean_addscalar_sqrt", sd, Map.of("x", a34), "out")) passed++; else failed++;
        }
        // Sub-chain 6: square → mean → add_scalar → sqrt → expandDims (5 ops)
        {
            SameDiff sd = SameDiff.create();
            SDVariable x = sd.placeHolder("x", DataType.FLOAT, 3, 4);
            SDVariable sq = sd.math().square("sq", x);
            SDVariable mn = sd.mean("mn", sq, 1);
            SDVariable rms = sd.math().sqrt("rms", mn.add(1e-6));
            sd.expandDims("out", rms, 1);
            if (verifyOpParity("sq_mean_addscalar_sqrt_expand", sd, Map.of("x", a34), "out")) passed++; else failed++;
        }
        // Full rms_norm_chain
        {
            SameDiff sd = SameDiff.create();
            SDVariable x = sd.placeHolder("x", DataType.FLOAT, 3, 4);
            SDVariable sq = sd.math().square("sq", x);
            SDVariable mn = sd.mean("mn", sq, 1);
            SDVariable rms = sd.math().sqrt("rms", mn.add(1e-6));
            sd.math().div("out", x, sd.expandDims("rms_exp", rms, 1));
            if (verifyOpParity("rms_norm_chain", sd, Map.of("x", a34), "out")) passed++; else failed++;
        }

        log.info("╔══════════════════════════════════════════════════════════════╗");
        log.info("║           OP COVERAGE RESULTS: {} passed, {} failed           ║",
                String.format("%2d", passed), String.format("%2d", failed));
        log.info("╚══════════════════════════════════════════════════════════════╝");

        assertEquals(0, failed, failed + " ops failed native execution parity check");
    }
}
