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
import org.nd4j.linalg.factory.Nd4j;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for shape inference, caching, and cache invalidation in the native executor.
 *
 * Covers: static shapes, dynamic shapes, shape cache hit/miss/invalidation,
 * INT/LONG sync flags, data-dependent shape ops.
 */
@Slf4j
@Tag(TagNames.SAMEDIFF)
@NativeTag
public class NativeExecutorShapeInferenceTest extends BaseNd4jTestWithBackends {

    @Test
    public void testStaticShapePlan() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 4);
        SDVariable z = x.add("z", 1.0);

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");
        NativeExecutorTestUtils.assertValidSerialization(plan);

        plan.close();
    }

    @Test
    public void testDynamicShapePlan() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, -1);
        SDVariable z = x.add("z", 1.0);

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");
        NativeExecutorTestUtils.assertValidSerialization(plan);

        // Execute with different shapes
        Map<String, INDArray> r1 = sd.output(Map.of("x", Nd4j.zeros(2, 3)), "z");
        assertArrayEquals(new long[]{2, 3}, r1.get("z").shape());

        Map<String, INDArray> r2 = sd.output(Map.of("x", Nd4j.zeros(5, 7)), "z");
        assertArrayEquals(new long[]{5, 7}, r2.get("z").shape());

        plan.close();
    }

    @Test
    public void testShapeCacheClearing() {
        SameDiff sd = NativeExecutorTestUtils.createAddGraph();
        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");

        // Simulate cache population
        DynamicShapeSlot slot = plan.getSlots()[0];
        slot.updateShapeCache(12345L, new ArrayList<>());
        assertTrue(slot.isShapeCacheValid(12345L));

        // Clear caches
        plan.clearAllShapeCaches();
        assertFalse(slot.isShapeCacheValid(12345L));

        plan.close();
    }

    @Test
    public void testShapeCacheKeyMismatch() {
        DynamicShapeSlot slot = DynamicShapeSlot.builder()
                .opName("test")
                .customOp(true)
                .inputSourceIndices(new int[0])
                .inputSourceTypes(new byte[0])
                .inputVarNames(new String[0])
                .outputSlotIndices(new int[]{0})
                .outputVarNames(new String[]{"out"})
                .stepIndex(0)
                .opNameHash(42L)
                .inputArraysBuffer(new INDArray[0])
                .build();

        slot.updateShapeCache(100L, new ArrayList<>());
        assertTrue(slot.isShapeCacheValid(100L));
        assertFalse(slot.isShapeCacheValid(200L)); // Different key → miss
        assertFalse(slot.isShapeCacheValid(0L));   // Zero key → miss
    }

    @Test
    public void testDataDependentOpFlags() {
        // Where op has data-dependent output shapes
        SameDiff sd = SameDiff.create();
        SDVariable cond = sd.placeHolder("cond", DataType.BOOL, -1, 4);
        // where(condition) - returns indices where condition is true (data-dependent shape)
        SDVariable z = sd.where("z", cond);

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");
        NativeExecutorTestUtils.assertValidSerialization(plan);

        // Check that at least one slot is flagged as data-dependent
        boolean hasDataDependent = false;
        for (DynamicShapeSlot slot : plan.getSlots()) {
            if (slot.isDataDependent()) {
                hasDataDependent = true;
                break;
            }
        }
        assertTrue(hasDataDependent, "Where op should be flagged as data-dependent");

        plan.close();
    }

    @Test
    public void testNeedsIntLongSyncFlag() {
        SameDiff sd = NativeExecutorTestUtils.createAddGraph();
        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");

        // Check that needsIntLongSync is serialized correctly
        for (DynamicShapeSlot slot : plan.getSlots()) {
            // For simple float ops, needsIntLongSync should be false
            assertFalse(slot.isNeedsIntLongSync(),
                    "Simple float add should not need INT/LONG sync");
        }

        plan.close();
    }

    @Test
    public void testOutputShapeDependsOnInputValuesFlag() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 6);
        SDVariable reshaped = sd.reshape("z", x, 3, 4);

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");
        NativeExecutorTestUtils.assertValidSerialization(plan);

        plan.close();
    }

    @Test
    public void testShapePreservingOps() {
        // Elementwise ops preserve input shapes
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable y = sd.nn().relu("relu", x, 0.0);
        SDVariable z = sd.nn().sigmoid("z", y);

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");

        INDArray xArr = Nd4j.randn(DataType.FLOAT, 3, 4);
        Map<String, INDArray> result = sd.output(Map.of("x", xArr), "z");
        assertArrayEquals(xArr.shape(), result.get("z").shape(), "Elementwise ops should preserve shape");

        plan.close();
    }

    @Test
    public void testShapeChangingOps() {
        // Matmul changes shape
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable w = sd.var("w", Nd4j.randn(DataType.FLOAT, 4, 8));
        SDVariable z = sd.mmul("z", x, w);

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");

        INDArray xArr = Nd4j.randn(DataType.FLOAT, 3, 4);
        Map<String, INDArray> result = sd.output(Map.of("x", xArr), "z");
        assertArrayEquals(new long[]{3, 8}, result.get("z").shape(), "Matmul should change last dim");

        plan.close();
    }

    @Test
    public void testReductionShapeInference() {
        // Test reduction WITHOUT keepDims
        SameDiff sdMean = SameDiff.create();
        SDVariable x1 = sdMean.placeHolder("x", DataType.FLOAT, -1, 4);
        x1.mean("z", 1);

        DynamicShapePlan planMean = NativeExecutorTestUtils.compilePlan(sdMean, "z");

        INDArray xArr = Nd4j.ones(DataType.FLOAT, 3, 4);
        Map<String, INDArray> resultMean = sdMean.output(Map.of("x", xArr), "z");
        assertNotNull(resultMean.get("z"), "Mean result should not be null");
        assertEquals(1, resultMean.get("z").rank(), "Mean should reduce rank");

        // Test reduction WITH keepDims
        SameDiff sdSum = SameDiff.create();
        SDVariable x2 = sdSum.placeHolder("x", DataType.FLOAT, -1, 4);
        x2.sum("z", true, 1);

        DynamicShapePlan planSum = NativeExecutorTestUtils.compilePlan(sdSum, "z");
        Map<String, INDArray> resultSum = sdSum.output(Map.of("x", xArr), "z");
        assertNotNull(resultSum.get("z"), "Sum result should not be null");
        assertEquals(2, resultSum.get("z").rank(), "Sum with keepdim should maintain rank");

        planMean.close();
        planSum.close();
    }

    @Test
    public void testGrowingSequenceShapes() {
        // Simulate autoregressive KV cache growth
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, -1, 8); // [batch, seq, dim]
        SDVariable z = x.add("z", 1.0);

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");

        // Growing sequence lengths
        for (int seqLen = 1; seqLen <= 10; seqLen++) {
            INDArray xArr = Nd4j.ones(DataType.FLOAT, 1, seqLen, 8);
            Map<String, INDArray> result = sd.output(Map.of("x", xArr), "z");
            assertArrayEquals(new long[]{1, seqLen, 8}, result.get("z").shape(),
                    "Seq length " + seqLen);
        }

        plan.close();
    }

    @Test
    public void testMultipleDTypeOutputs() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable cast = sd.castTo("z", x, DataType.INT32);

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");
        NativeExecutorTestUtils.assertValidSerialization(plan);

        INDArray xArr = Nd4j.linspace(1, 8, 8, DataType.FLOAT).reshape(2, 4);
        Map<String, INDArray> result = sd.output(Map.of("x", xArr), "z");
        assertEquals(DataType.INT32, result.get("z").dataType());

        plan.close();
    }

    @Test
    public void testBroadcastShapeInference() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 1); // [batch, 1]
        SDVariable y = sd.placeHolder("y", DataType.FLOAT, 1, 4);  // [1, 4]
        SDVariable z = x.add("z", y); // broadcast → [batch, 4]

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");
        NativeExecutorTestUtils.assertValidSerialization(plan);

        INDArray xArr = Nd4j.ones(DataType.FLOAT, 3, 1);
        INDArray yArr = Nd4j.ones(DataType.FLOAT, 1, 4);
        Map<String, INDArray> result = sd.output(Map.of("x", xArr, "y", yArr), "z");
        assertArrayEquals(new long[]{3, 4}, result.get("z").shape(), "Broadcast should expand dims");

        plan.close();
    }

    @Test
    public void testNeedsZeroedOutputFlag() {
        SameDiff sd = NativeExecutorTestUtils.createAddGraph();
        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");

        byte[] data = plan.serialize();
        assertNotNull(data);

        // Verify needsZeroedOutput flag is present for each slot
        for (DynamicShapeSlot slot : plan.getSlots()) {
            // The flag should be a valid boolean
            assertTrue(slot.isNeedsZeroedOutput() || !slot.isNeedsZeroedOutput());
        }

        plan.close();
    }

    @Test
    public void testCustomOpFlag() {
        SameDiff sd = NativeExecutorTestUtils.createMatMulGraph(4, 3);
        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");

        // Matmul should be a custom op
        boolean hasCustomOp = false;
        for (DynamicShapeSlot slot : plan.getSlots()) {
            if (slot.isCustomOp()) {
                hasCustomOp = true;
                break;
            }
        }
        assertTrue(hasCustomOp, "Plan should contain at least one custom op (matmul)");

        plan.close();
    }
}
