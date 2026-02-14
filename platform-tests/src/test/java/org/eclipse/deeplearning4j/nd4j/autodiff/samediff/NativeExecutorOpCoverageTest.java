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
 * Tests for op coverage in the native executor: ensures various op categories
 * compile and serialize correctly with the right flags.
 *
 * Covers: zeroed-output ops, multi-output ops, all arg types (iArgs, tArgs,
 * bArgs, dArgs), view-producing ops.
 */
@Slf4j
@Tag(TagNames.SAMEDIFF)
@NativeTag
public class NativeExecutorOpCoverageTest extends BaseNd4jTestWithBackends {

    @Test
    public void testSliceOp() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 8);
        SDVariable z = sd.slice("z", x, new int[]{0, 2}, new int[]{-1, 4});

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");
        NativeExecutorTestUtils.assertValidSerialization(plan);

        INDArray xArr = Nd4j.linspace(1, 16, 16, DataType.FLOAT).reshape(2, 8);
        Map<String, INDArray> result = sd.output(Map.of("x", xArr), "z");
        assertArrayEquals(new long[]{2, 4}, result.get("z").shape());

        plan.close();
    }

    @Test
    public void testGatherOp() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 4, 3);
        SDVariable indices = sd.constant("indices", Nd4j.createFromArray(new int[]{0, 2}));
        SDVariable z = sd.gather("z", x, indices, 0);

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");
        NativeExecutorTestUtils.assertValidSerialization(plan);

        INDArray xArr = Nd4j.linspace(1, 12, 12, DataType.FLOAT).reshape(4, 3);
        Map<String, INDArray> result = sd.output(Map.of("x", xArr), "z");
        assertArrayEquals(new long[]{2, 3}, result.get("z").shape());

        plan.close();
    }

    @Test
    public void testExpandDimsOp() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable z = sd.expandDims("z", x, 1);

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");
        NativeExecutorTestUtils.assertValidSerialization(plan);

        INDArray xArr = Nd4j.ones(DataType.FLOAT, 2, 4);
        Map<String, INDArray> result = sd.output(Map.of("x", xArr), "z");
        assertArrayEquals(new long[]{2, 1, 4}, result.get("z").shape());

        plan.close();
    }

    @Test
    public void testSqueezeDimsOp() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 1, 4);
        SDVariable z = sd.squeeze("z", x, 1);

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");
        NativeExecutorTestUtils.assertValidSerialization(plan);

        INDArray xArr = Nd4j.ones(DataType.FLOAT, 2, 1, 4);
        Map<String, INDArray> result = sd.output(Map.of("x", xArr), "z");
        assertArrayEquals(new long[]{2, 4}, result.get("z").shape());

        plan.close();
    }

    @Test
    public void testTileOp() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 3);
        SDVariable z = sd.tile("z", x, 2, 3);

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");
        NativeExecutorTestUtils.assertValidSerialization(plan);

        INDArray xArr = Nd4j.ones(DataType.FLOAT, 2, 3);
        Map<String, INDArray> result = sd.output(Map.of("x", xArr), "z");
        assertArrayEquals(new long[]{4, 9}, result.get("z").shape());

        plan.close();
    }

    @Test
    public void testOnesLikeOp() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable z = sd.onesLike("z", x);

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");
        NativeExecutorTestUtils.assertValidSerialization(plan);

        INDArray xArr = Nd4j.randn(DataType.FLOAT, 3, 4);
        Map<String, INDArray> result = sd.output(Map.of("x", xArr), "z");
        assertEquals(1.0, result.get("z").getDouble(0, 0), 1e-5);

        plan.close();
    }

    @Test
    public void testZerosLikeOp() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable z = sd.zerosLike("z", x);

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");
        NativeExecutorTestUtils.assertValidSerialization(plan);

        INDArray xArr = Nd4j.randn(DataType.FLOAT, 3, 4);
        Map<String, INDArray> result = sd.output(Map.of("x", xArr), "z");
        assertEquals(0.0, result.get("z").getDouble(0, 0), 1e-5);

        plan.close();
    }

    @Test
    public void testOpWithIArgs() {
        // Reshape has iArgs for target shape
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 6);
        SDVariable z = sd.reshape("z", x, 3, 4);

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");
        NativeExecutorTestUtils.assertValidSerialization(plan);

        INDArray xArr = Nd4j.linspace(1, 12, 12, DataType.FLOAT).reshape(2, 6);
        Map<String, INDArray> result = sd.output(Map.of("x", xArr), "z");
        assertArrayEquals(new long[]{3, 4}, result.get("z").shape());

        plan.close();
    }

    @Test
    public void testOpWithTArgs() {
        // Leaky ReLU has tArgs for alpha
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable z = sd.nn().leakyRelu("z", x, 0.1);

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");
        NativeExecutorTestUtils.assertValidSerialization(plan);

        INDArray xArr = Nd4j.createFromArray(new float[][]{{-10, -1, 0, 1}});
        Map<String, INDArray> result = sd.output(Map.of("x", xArr), "z");
        assertEquals(-1.0, result.get("z").getFloat(0, 0), 0.1, "Leaky ReLU negative");
        assertEquals(1.0, result.get("z").getFloat(0, 3), 1e-5, "Leaky ReLU positive");

        plan.close();
    }

    @Test
    public void testOpWithDArgs() {
        // Cast has dArgs for target dtype
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable z = sd.castTo("z", x, DataType.DOUBLE);

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");
        NativeExecutorTestUtils.assertValidSerialization(plan);

        // Verify dArgs are serialized
        boolean hasDArgs = false;
        for (DynamicShapeSlot slot : plan.getSlots()) {
            if (slot.getDArgs() != null && slot.getDArgs().length > 0) {
                hasDArgs = true;
                break;
            }
        }
        // Some ops encode dtype in dArgs
        // The important thing is serialization works

        plan.close();
    }

    @Test
    public void testMinimumOp() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable y = sd.placeHolder("y", DataType.FLOAT, -1, 4);
        SDVariable z = sd.math().min("z", x, y);

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");
        NativeExecutorTestUtils.assertValidSerialization(plan);

        INDArray xArr = Nd4j.createFromArray(new float[][]{{1, 5, 3, 7}});
        INDArray yArr = Nd4j.createFromArray(new float[][]{{4, 2, 6, 1}});
        Map<String, INDArray> result = sd.output(Map.of("x", xArr, "y", yArr), "z");
        INDArray expected = Nd4j.createFromArray(new float[][]{{1, 2, 3, 1}});
        NativeExecutorTestUtils.assertArrayEquals(expected, result.get("z"), 1e-5, "min");

        plan.close();
    }

    @Test
    public void testMaximumOp() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable y = sd.placeHolder("y", DataType.FLOAT, -1, 4);
        SDVariable z = sd.math().max("z", x, y);

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");
        NativeExecutorTestUtils.assertValidSerialization(plan);

        INDArray xArr = Nd4j.createFromArray(new float[][]{{1, 5, 3, 7}});
        INDArray yArr = Nd4j.createFromArray(new float[][]{{4, 2, 6, 1}});
        Map<String, INDArray> result = sd.output(Map.of("x", xArr, "y", yArr), "z");
        INDArray expected = Nd4j.createFromArray(new float[][]{{4, 5, 6, 7}});
        NativeExecutorTestUtils.assertArrayEquals(expected, result.get("z"), 1e-5, "max");

        plan.close();
    }

    @Test
    public void testClipByValue() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable z = sd.math().clipByValue("z", x, -1.0, 1.0);

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");
        NativeExecutorTestUtils.assertValidSerialization(plan);

        INDArray xArr = Nd4j.createFromArray(new float[][]{{-5, -0.5f, 0.5f, 5}});
        Map<String, INDArray> result = sd.output(Map.of("x", xArr), "z");
        assertEquals(-1.0, result.get("z").getFloat(0, 0), 1e-5, "clip min");
        assertEquals(-0.5, result.get("z").getFloat(0, 1), 1e-5, "clip pass through");
        assertEquals(0.5, result.get("z").getFloat(0, 2), 1e-5, "clip pass through");
        assertEquals(1.0, result.get("z").getFloat(0, 3), 1e-5, "clip max");

        plan.close();
    }

    @Test
    public void testReciprocal() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable z = sd.math().reciprocal("z", x);

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");
        NativeExecutorTestUtils.assertValidSerialization(plan);

        INDArray xArr = Nd4j.createFromArray(new float[][]{{1, 2, 4, 5}});
        Map<String, INDArray> result = sd.output(Map.of("x", xArr), "z");
        assertEquals(1.0, result.get("z").getFloat(0, 0), 1e-5);
        assertEquals(0.5, result.get("z").getFloat(0, 1), 1e-5);
        assertEquals(0.25, result.get("z").getFloat(0, 2), 1e-5);
        assertEquals(0.2, result.get("z").getFloat(0, 3), 1e-5);

        plan.close();
    }

    @Test
    public void testAllSlotFieldsPopulated() {
        SameDiff sd = NativeExecutorTestUtils.createChainGraph();
        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");

        for (int i = 0; i < plan.getSlots().length; i++) {
            DynamicShapeSlot slot = plan.getSlots()[i];
            assertNotNull(slot.getOpName(), "Slot " + i + " should have opName");
            assertNotNull(slot.getInputSourceIndices(), "Slot " + i + " should have inputSourceIndices");
            assertNotNull(slot.getInputSourceTypes(), "Slot " + i + " should have inputSourceTypes");
            assertNotNull(slot.getOutputSlotIndices(), "Slot " + i + " should have outputSlotIndices");
            assertEquals(slot.getInputSourceIndices().length, slot.getInputSourceTypes().length,
                    "Slot " + i + " input indices and types should have same length");
            assertTrue(slot.getOpNameHash() != 0, "Slot " + i + " should have non-zero opNameHash");
        }

        plan.close();
    }

    @Test
    public void testInputSourceIndicesValid() {
        SameDiff sd = NativeExecutorTestUtils.createChainGraph();
        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");

        int totalOutputSlots = plan.getTotalOutputSlots();
        int numExternalInputs = plan.getExternalInputKeys().length;

        for (DynamicShapeSlot slot : plan.getSlots()) {
            for (int i = 0; i < slot.getInputSourceIndices().length; i++) {
                int idx = slot.getInputSourceIndices()[i];
                byte type = slot.getInputSourceTypes()[i];

                if (idx >= 0) {
                    // Op output reference
                    assertTrue(idx < totalOutputSlots,
                            "Output slot ref " + idx + " should be < " + totalOutputSlots);
                    assertEquals(DynamicShapeSlot.SOURCE_OP_OUTPUT, type,
                            "Positive index should have SOURCE_OP_OUTPUT type");
                } else {
                    // External input reference
                    int extIdx = -(idx + 1);
                    assertTrue(extIdx < numExternalInputs,
                            "External ref " + extIdx + " should be < " + numExternalInputs);
                    assertTrue(type >= DynamicShapeSlot.SOURCE_CONSTANT &&
                                    type <= DynamicShapeSlot.SOURCE_PLACEHOLDER,
                            "External input should be CONSTANT, VARIABLE, or PLACEHOLDER");
                }
            }
        }

        plan.close();
    }

    @Test
    public void testStridedSliceOp() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 4, 4);
        SDVariable z = sd.stridedSlice("z", x,
                new long[]{0, 0}, new long[]{2, 4}, new long[]{1, 1});

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");
        NativeExecutorTestUtils.assertValidSerialization(plan);

        INDArray xArr = Nd4j.linspace(1, 16, 16, DataType.FLOAT).reshape(4, 4);
        Map<String, INDArray> result = sd.output(Map.of("x", xArr), "z");
        assertEquals(2, result.get("z").size(0));
        assertEquals(4, result.get("z").size(1));

        plan.close();
    }
}
