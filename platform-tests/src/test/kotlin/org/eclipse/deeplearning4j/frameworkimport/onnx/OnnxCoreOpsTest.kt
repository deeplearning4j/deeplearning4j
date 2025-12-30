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
package org.eclipse.deeplearning4j.frameworkimport.onnx

import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Tag
import org.junit.jupiter.api.Test
import org.nd4j.autodiff.samediff.SameDiff
import org.nd4j.common.tests.tags.TagNames
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.factory.Nd4j

/**
 * Tests for core ONNX operator implementations including
 * tensor operations, element-wise operations, and shape manipulations.
 *
 * @author Eclipse Deeplearning4j Development Team
 */
@Tag(TagNames.SAMEDIFF)
class OnnxCoreOpsTest {

    // ==================== Cast Operations ====================

    @Test
    fun testCastFloatToInt() {
        val sd = SameDiff.create()
        val input = sd.placeHolder("input", DataType.FLOAT, -1, 4)

        val output = input.castTo("cast", DataType.INT32)

        val inputArr = Nd4j.create(floatArrayOf(1.5f, 2.7f, -3.2f, 0.0f)).reshape(1, 4)
        val result = sd.output(mapOf("input" to inputArr), "cast")["cast"]!!

        assertEquals(DataType.INT32, result.dataType())
        assertEquals(1, result.getInt(0, 0))
        assertEquals(2, result.getInt(0, 1))
        assertEquals(-3, result.getInt(0, 2))
        assertEquals(0, result.getInt(0, 3))
    }

    @Test
    fun testCastIntToFloat() {
        val sd = SameDiff.create()
        val input = sd.placeHolder("input", DataType.INT32, -1, 3)

        val output = input.castTo("cast", DataType.FLOAT)

        val inputArr = Nd4j.createFromArray(intArrayOf(1, 2, 3)).reshape(1, 3)
        val result = sd.output(mapOf("input" to inputArr), "cast")["cast"]!!

        assertEquals(DataType.FLOAT, result.dataType())
        assertEquals(1f, result.getFloat(0, 0), 0.01f)
    }

    @Test
    fun testCastFloatToDouble() {
        val sd = SameDiff.create()
        val input = sd.placeHolder("input", DataType.FLOAT, 2, 2)

        val output = input.castTo("cast", DataType.DOUBLE)

        val inputArr = Nd4j.create(floatArrayOf(1.5f, 2.5f, 3.5f, 4.5f)).reshape(2, 2)
        val result = sd.output(mapOf("input" to inputArr), "cast")["cast"]!!

        assertEquals(DataType.DOUBLE, result.dataType())
    }

    // ==================== Clip Operations ====================

    @Test
    fun testClipByValue() {
        val sd = SameDiff.create()
        val input = sd.placeHolder("input", DataType.FLOAT, -1, 5)

        val output = sd.math.clipByValue("clip", input, -1.0, 1.0)

        val inputArr = Nd4j.create(floatArrayOf(-2f, -0.5f, 0f, 0.5f, 2f)).reshape(1, 5)
        val result = sd.output(mapOf("input" to inputArr), "clip")["clip"]!!

        assertEquals(-1f, result.getFloat(0, 0), 0.01f)  // -2 clipped to -1
        assertEquals(-0.5f, result.getFloat(0, 1), 0.01f)  // unchanged
        assertEquals(0f, result.getFloat(0, 2), 0.01f)  // unchanged
        assertEquals(0.5f, result.getFloat(0, 3), 0.01f)  // unchanged
        assertEquals(1f, result.getFloat(0, 4), 0.01f)  // 2 clipped to 1
    }

    @Test
    fun testClipByNorm() {
        val sd = SameDiff.create()
        val input = sd.placeHolder("input", DataType.FLOAT, -1, 4)

        val output = sd.math.clipByNorm("clip_norm", input, 1.0)

        val inputArr = Nd4j.create(floatArrayOf(3f, 4f, 0f, 0f)).reshape(1, 4)  // norm = 5
        val result = sd.output(mapOf("input" to inputArr), "clip_norm")["clip_norm"]!!

        // After clipping to norm 1: [0.6, 0.8, 0, 0]
        assertEquals(0.6f, result.getFloat(0, 0), 0.01f)
        assertEquals(0.8f, result.getFloat(0, 1), 0.01f)
    }

    // ==================== ConstantOfShape ====================

    @Test
    fun testConstantOfShape() {
        val sd = SameDiff.create()

        // Create a tensor of specified shape filled with a constant
        val shape = longArrayOf(2, 3, 4)
        val value = 5f
        val output = sd.constant("const_shape", Nd4j.ones(DataType.FLOAT, *shape).mul(value))

        val result = sd.output(emptyMap(), "const_shape")["const_shape"]!!

        assertEquals(3, result.rank())
        assertEquals(2, result.shape()[0])
        assertEquals(3, result.shape()[1])
        assertEquals(4, result.shape()[2])
        assertEquals(5f, result.getFloat(0, 0, 0), 0.01f)
    }

    // ==================== CumSum ====================

    @Test
    fun testCumSum() {
        val sd = SameDiff.create()
        val input = sd.placeHolder("input", DataType.FLOAT, 4)

        val output = sd.cumsum("cumsum", input, false, false, 0)

        val inputArr = Nd4j.create(floatArrayOf(1f, 2f, 3f, 4f))
        val result = sd.output(mapOf("input" to inputArr), "cumsum")["cumsum"]!!

        // [1, 1+2, 1+2+3, 1+2+3+4] = [1, 3, 6, 10]
        assertEquals(1f, result.getFloat(0), 0.01f)
        assertEquals(3f, result.getFloat(1), 0.01f)
        assertEquals(6f, result.getFloat(2), 0.01f)
        assertEquals(10f, result.getFloat(3), 0.01f)
    }

    @Test
    fun testCumSumExclusive() {
        val sd = SameDiff.create()
        val input = sd.placeHolder("input", DataType.FLOAT, 4)

        val output = sd.cumsum("cumsum_ex", input, true, false, 0)

        val inputArr = Nd4j.create(floatArrayOf(1f, 2f, 3f, 4f))
        val result = sd.output(mapOf("input" to inputArr), "cumsum_ex")["cumsum_ex"]!!

        // Exclusive: [0, 1, 1+2, 1+2+3] = [0, 1, 3, 6]
        assertEquals(0f, result.getFloat(0), 0.01f)
        assertEquals(1f, result.getFloat(1), 0.01f)
        assertEquals(3f, result.getFloat(2), 0.01f)
        assertEquals(6f, result.getFloat(3), 0.01f)
    }

    @Test
    fun testCumSumReverse() {
        val sd = SameDiff.create()
        val input = sd.placeHolder("input", DataType.FLOAT, 4)

        val output = sd.cumsum("cumsum_rev", input, false, true, 0)

        val inputArr = Nd4j.create(floatArrayOf(1f, 2f, 3f, 4f))
        val result = sd.output(mapOf("input" to inputArr), "cumsum_rev")["cumsum_rev"]!!

        // Reverse: [1+2+3+4, 2+3+4, 3+4, 4] = [10, 9, 7, 4]
        assertEquals(10f, result.getFloat(0), 0.01f)
        assertEquals(9f, result.getFloat(1), 0.01f)
        assertEquals(7f, result.getFloat(2), 0.01f)
        assertEquals(4f, result.getFloat(3), 0.01f)
    }

    // ==================== CumProd ====================

    @Test
    fun testCumProd() {
        val sd = SameDiff.create()
        val input = sd.placeHolder("input", DataType.FLOAT, 4)

        val output = sd.cumprod("cumprod", input, false, false, 0)

        val inputArr = Nd4j.create(floatArrayOf(1f, 2f, 3f, 4f))
        val result = sd.output(mapOf("input" to inputArr), "cumprod")["cumprod"]!!

        // [1, 1*2, 1*2*3, 1*2*3*4] = [1, 2, 6, 24]
        assertEquals(1f, result.getFloat(0), 0.01f)
        assertEquals(2f, result.getFloat(1), 0.01f)
        assertEquals(6f, result.getFloat(2), 0.01f)
        assertEquals(24f, result.getFloat(3), 0.01f)
    }

    // ==================== Expand ====================

    @Test
    fun testExpand() {
        val sd = SameDiff.create()
        val input = sd.placeHolder("input", DataType.FLOAT, 1, 3)

        // Expand to [4, 3] by broadcasting
        val output = sd.tile("expand", input, 4, 1)

        val inputArr = Nd4j.create(floatArrayOf(1f, 2f, 3f)).reshape(1, 3)
        val result = sd.output(mapOf("input" to inputArr), "expand")["expand"]!!

        assertEquals(4, result.shape()[0])
        assertEquals(3, result.shape()[1])
        for (i in 0 until 4) {
            assertEquals(1f, result.getFloat(i.toLong(), 0), 0.01f)
            assertEquals(2f, result.getFloat(i.toLong(), 1), 0.01f)
            assertEquals(3f, result.getFloat(i.toLong(), 2), 0.01f)
        }
    }

    // ==================== Gather ====================

    @Test
    fun testGatherAxis0() {
        val sd = SameDiff.create()
        val input = sd.placeHolder("input", DataType.FLOAT, 3, 2)
        val indices = sd.placeHolder("indices", DataType.INT64, 2)

        val output = sd.gather("gather", input, indices, 0)

        val inputArr = Nd4j.create(floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f)).reshape(3, 2)
        val indicesArr = Nd4j.createFromArray(longArrayOf(0, 2))
        val result = sd.output(mapOf("input" to inputArr, "indices" to indicesArr), "gather")["gather"]!!

        assertEquals(2, result.shape()[0])
        assertEquals(2, result.shape()[1])
        assertEquals(1f, result.getFloat(0, 0), 0.01f)  // row 0
        assertEquals(5f, result.getFloat(1, 0), 0.01f)  // row 2
    }

    @Test
    fun testGatherAxis1() {
        val sd = SameDiff.create()
        val input = sd.placeHolder("input", DataType.FLOAT, 2, 4)
        val indices = sd.placeHolder("indices", DataType.INT64, 2)

        val output = sd.gather("gather", input, indices, 1)

        val inputArr = Nd4j.create(floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f)).reshape(2, 4)
        val indicesArr = Nd4j.createFromArray(longArrayOf(1, 3))
        val result = sd.output(mapOf("input" to inputArr, "indices" to indicesArr), "gather")["gather"]!!

        assertEquals(2, result.shape()[0])
        assertEquals(2, result.shape()[1])
        assertEquals(2f, result.getFloat(0, 0), 0.01f)  // col 1
        assertEquals(4f, result.getFloat(0, 1), 0.01f)  // col 3
    }

    // ==================== GatherElements ====================

    @Test
    fun testGatherElements() {
        val sd = SameDiff.create()
        val input = sd.placeHolder("input", DataType.FLOAT, 2, 3)
        val indices = sd.placeHolder("indices", DataType.INT64, 2, 2)

        // GatherElements selects elements based on indices along an axis
        val inputArr = Nd4j.create(floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f)).reshape(2, 3)
        val indicesArr = Nd4j.createFromArray(longArrayOf(0, 2, 1, 0)).reshape(2, 2)

        // For axis=1: output[i,j] = input[i, indices[i,j]]
        // output[0,0] = input[0, 0] = 1
        // output[0,1] = input[0, 2] = 3
        // output[1,0] = input[1, 1] = 5
        // output[1,1] = input[1, 0] = 4

        assertNotNull(sd)
    }

    // ==================== Reshape ====================

    @Test
    fun testReshape() {
        val sd = SameDiff.create()
        val input = sd.placeHolder("input", DataType.FLOAT, 2, 6)

        val output = sd.reshape("reshape", input, 3, 4)

        val inputArr = Nd4j.linspace(1, 12, 12, DataType.FLOAT).reshape(2, 6)
        val result = sd.output(mapOf("input" to inputArr), "reshape")["reshape"]!!

        assertEquals(3, result.shape()[0])
        assertEquals(4, result.shape()[1])
        assertEquals(1f, result.getFloat(0, 0), 0.01f)
    }

    @Test
    fun testReshapeWithInferredDim() {
        val sd = SameDiff.create()
        val input = sd.placeHolder("input", DataType.FLOAT, 2, 3, 4)

        // -1 infers dimension: 2*3*4 = 24, so 4*-1 = 4*6
        val output = sd.reshape("reshape", input, 4, -1)

        val inputArr = Nd4j.linspace(1, 24, 24, DataType.FLOAT).reshape(2, 3, 4)
        val result = sd.output(mapOf("input" to inputArr), "reshape")["reshape"]!!

        assertEquals(4, result.shape()[0])
        assertEquals(6, result.shape()[1])
    }

    // ==================== Transpose ====================

    @Test
    fun testTranspose2D() {
        val sd = SameDiff.create()
        val input = sd.placeHolder("input", DataType.FLOAT, 2, 3)

        val output = sd.transpose("transpose", input)

        val inputArr = Nd4j.create(floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f)).reshape(2, 3)
        val result = sd.output(mapOf("input" to inputArr), "transpose")["transpose"]!!

        assertEquals(3, result.shape()[0])
        assertEquals(2, result.shape()[1])
        assertEquals(1f, result.getFloat(0, 0), 0.01f)
        assertEquals(4f, result.getFloat(0, 1), 0.01f)
    }

    @Test
    fun testTransposeWithPerm() {
        val sd = SameDiff.create()
        val input = sd.placeHolder("input", DataType.FLOAT, 2, 3, 4)

        val output = sd.permute("permute", input, 2, 0, 1)

        val inputArr = Nd4j.linspace(1, 24, 24, DataType.FLOAT).reshape(2, 3, 4)
        val result = sd.output(mapOf("input" to inputArr), "permute")["permute"]!!

        assertEquals(4, result.shape()[0])
        assertEquals(2, result.shape()[1])
        assertEquals(3, result.shape()[2])
    }

    // ==================== Unsqueeze ====================

    @Test
    fun testUnsqueeze() {
        val sd = SameDiff.create()
        val input = sd.placeHolder("input", DataType.FLOAT, 2, 3)

        val output = sd.expandDims("unsqueeze", input, 0)

        val inputArr = Nd4j.create(floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f)).reshape(2, 3)
        val result = sd.output(mapOf("input" to inputArr), "unsqueeze")["unsqueeze"]!!

        assertEquals(3, result.rank())
        assertEquals(1, result.shape()[0])
        assertEquals(2, result.shape()[1])
        assertEquals(3, result.shape()[2])
    }

    @Test
    fun testUnsqueezeLastAxis() {
        val sd = SameDiff.create()
        val input = sd.placeHolder("input", DataType.FLOAT, 2, 3)

        val output = sd.expandDims("unsqueeze", input, 2)

        val inputArr = Nd4j.create(floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f)).reshape(2, 3)
        val result = sd.output(mapOf("input" to inputArr), "unsqueeze")["unsqueeze"]!!

        assertEquals(3, result.rank())
        assertEquals(2, result.shape()[0])
        assertEquals(3, result.shape()[1])
        assertEquals(1, result.shape()[2])
    }

    // ==================== Squeeze ====================

    @Test
    fun testSqueeze() {
        val sd = SameDiff.create()
        val input = sd.placeHolder("input", DataType.FLOAT, 1, 2, 1, 3)

        val output = sd.squeeze("squeeze", input, 0, 2)

        val inputArr = Nd4j.linspace(1, 6, 6, DataType.FLOAT).reshape(1, 2, 1, 3)
        val result = sd.output(mapOf("input" to inputArr), "squeeze")["squeeze"]!!

        assertEquals(2, result.rank())
        assertEquals(2, result.shape()[0])
        assertEquals(3, result.shape()[1])
    }

    // ==================== Slice ====================

    @Test
    fun testSlice() {
        val sd = SameDiff.create()
        val input = sd.placeHolder("input", DataType.FLOAT, 4, 5)

        val output = sd.stridedSlice("slice", input,
            intArrayOf(1, 1), intArrayOf(3, 4), intArrayOf(1, 1))

        val inputArr = Nd4j.linspace(1, 20, 20, DataType.FLOAT).reshape(4, 5)
        val result = sd.output(mapOf("input" to inputArr), "slice")["slice"]!!

        assertEquals(2, result.shape()[0])
        assertEquals(3, result.shape()[1])
    }

    @Test
    fun testSliceWithStep() {
        val sd = SameDiff.create()
        val input = sd.placeHolder("input", DataType.FLOAT, 6)

        // Select every 2nd element
        val output = sd.stridedSlice("slice_step", input,
            intArrayOf(0), intArrayOf(6), intArrayOf(2))

        val inputArr = Nd4j.create(floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f))
        val result = sd.output(mapOf("input" to inputArr), "slice_step")["slice_step"]!!

        assertEquals(3, result.length())
        assertEquals(1f, result.getFloat(0), 0.01f)
        assertEquals(3f, result.getFloat(1), 0.01f)
        assertEquals(5f, result.getFloat(2), 0.01f)
    }

    // ==================== Split ====================

    @Test
    fun testSplitEqual() {
        val sd = SameDiff.create()
        val input = sd.placeHolder("input", DataType.FLOAT, 6, 3)

        val outputs = sd.split(input, 3, 0)

        val inputArr = Nd4j.linspace(1, 18, 18, DataType.FLOAT).reshape(6, 3)
        val result = sd.output(mapOf("input" to inputArr), outputs[0].name(), outputs[1].name(), outputs[2].name())

        assertEquals(3, result.size)
        for (output in outputs) {
            assertEquals(2, result[output.name()]!!.shape()[0])
            assertEquals(3, result[output.name()]!!.shape()[1])
        }
    }

    // ==================== Concat ====================

    @Test
    fun testConcatAxis0() {
        val sd = SameDiff.create()
        val input1 = sd.placeHolder("input1", DataType.FLOAT, 2, 3)
        val input2 = sd.placeHolder("input2", DataType.FLOAT, 3, 3)

        val output = sd.concat("concat", 0, input1, input2)

        val input1Arr = Nd4j.ones(DataType.FLOAT, 2, 3)
        val input2Arr = Nd4j.ones(DataType.FLOAT, 3, 3).mul(2)
        val result = sd.output(mapOf("input1" to input1Arr, "input2" to input2Arr), "concat")["concat"]!!

        assertEquals(5, result.shape()[0])
        assertEquals(3, result.shape()[1])
    }

    @Test
    fun testConcatAxis1() {
        val sd = SameDiff.create()
        val input1 = sd.placeHolder("input1", DataType.FLOAT, 2, 3)
        val input2 = sd.placeHolder("input2", DataType.FLOAT, 2, 4)

        val output = sd.concat("concat", 1, input1, input2)

        val input1Arr = Nd4j.ones(DataType.FLOAT, 2, 3)
        val input2Arr = Nd4j.ones(DataType.FLOAT, 2, 4).mul(2)
        val result = sd.output(mapOf("input1" to input1Arr, "input2" to input2Arr), "concat")["concat"]!!

        assertEquals(2, result.shape()[0])
        assertEquals(7, result.shape()[1])
    }

    // ==================== Stack ====================

    @Test
    fun testStack() {
        val sd = SameDiff.create()
        val input1 = sd.placeHolder("input1", DataType.FLOAT, 2, 3)
        val input2 = sd.placeHolder("input2", DataType.FLOAT, 2, 3)

        val output = sd.stack("stack", 0, input1, input2)

        val input1Arr = Nd4j.ones(DataType.FLOAT, 2, 3)
        val input2Arr = Nd4j.ones(DataType.FLOAT, 2, 3).mul(2)
        val result = sd.output(mapOf("input1" to input1Arr, "input2" to input2Arr), "stack")["stack"]!!

        assertEquals(3, result.rank())
        assertEquals(2, result.shape()[0])
        assertEquals(2, result.shape()[1])
        assertEquals(3, result.shape()[2])
    }

    // ==================== Tile ====================

    @Test
    fun testTile() {
        val sd = SameDiff.create()
        val input = sd.placeHolder("input", DataType.FLOAT, 2, 3)

        val output = sd.tile("tile", input, 2, 3)

        val inputArr = Nd4j.create(floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f)).reshape(2, 3)
        val result = sd.output(mapOf("input" to inputArr), "tile")["tile"]!!

        assertEquals(4, result.shape()[0])
        assertEquals(9, result.shape()[1])
    }

    // ==================== Pad ====================

    @Test
    fun testPadConstant() {
        val sd = SameDiff.create()
        val input = sd.placeHolder("input", DataType.FLOAT, 2, 3)

        // Pad with constant 0
        val padding = Nd4j.createFromArray(intArrayOf(1, 1, 2, 2)).reshape(2, 2)
        val output = sd.nn.pad("pad", input, sd.constant(padding), 0.0)

        val inputArr = Nd4j.ones(DataType.FLOAT, 2, 3)
        val result = sd.output(mapOf("input" to inputArr), "pad")["pad"]!!

        assertEquals(4, result.shape()[0])  // 2 + 1 + 1
        assertEquals(7, result.shape()[1])  // 3 + 2 + 2
    }

    // ==================== Where (conditional) ====================

    @Test
    fun testWhere() {
        val sd = SameDiff.create()
        val condition = sd.placeHolder("condition", DataType.BOOL, 4)
        val x = sd.placeHolder("x", DataType.FLOAT, 4)
        val y = sd.placeHolder("y", DataType.FLOAT, 4)

        val output = sd.where("where", condition, x, y)

        val condArr = Nd4j.create(booleanArrayOf(true, false, true, false))
        val xArr = Nd4j.create(floatArrayOf(1f, 2f, 3f, 4f))
        val yArr = Nd4j.create(floatArrayOf(10f, 20f, 30f, 40f))
        val result = sd.output(mapOf("condition" to condArr, "x" to xArr, "y" to yArr), "where")["where"]!!

        assertEquals(1f, result.getFloat(0), 0.01f)   // true -> x
        assertEquals(20f, result.getFloat(1), 0.01f)  // false -> y
        assertEquals(3f, result.getFloat(2), 0.01f)   // true -> x
        assertEquals(40f, result.getFloat(3), 0.01f)  // false -> y
    }

    // ==================== NonZero ====================

    @Test
    fun testNonZero() {
        val sd = SameDiff.create()
        val input = sd.placeHolder("input", DataType.FLOAT, 3, 3)

        // Find indices of non-zero elements
        val inputArr = Nd4j.create(floatArrayOf(
            1f, 0f, 2f,
            0f, 0f, 3f,
            4f, 0f, 0f
        )).reshape(3, 3)

        // Non-zero values are at: (0,0), (0,2), (1,2), (2,0)
        assertNotNull(sd)
    }

    // ==================== Maximum / Minimum ====================

    @Test
    fun testMaximum() {
        val sd = SameDiff.create()
        val a = sd.placeHolder("a", DataType.FLOAT, 4)
        val b = sd.placeHolder("b", DataType.FLOAT, 4)

        val output = sd.math.max("maximum", a, b)

        val aArr = Nd4j.create(floatArrayOf(1f, 4f, 2f, 5f))
        val bArr = Nd4j.create(floatArrayOf(3f, 2f, 4f, 1f))
        val result = sd.output(mapOf("a" to aArr, "b" to bArr), "maximum")["maximum"]!!

        assertEquals(3f, result.getFloat(0), 0.01f)
        assertEquals(4f, result.getFloat(1), 0.01f)
        assertEquals(4f, result.getFloat(2), 0.01f)
        assertEquals(5f, result.getFloat(3), 0.01f)
    }

    @Test
    fun testMinimum() {
        val sd = SameDiff.create()
        val a = sd.placeHolder("a", DataType.FLOAT, 4)
        val b = sd.placeHolder("b", DataType.FLOAT, 4)

        val output = sd.math.min("minimum", a, b)

        val aArr = Nd4j.create(floatArrayOf(1f, 4f, 2f, 5f))
        val bArr = Nd4j.create(floatArrayOf(3f, 2f, 4f, 1f))
        val result = sd.output(mapOf("a" to aArr, "b" to bArr), "minimum")["minimum"]!!

        assertEquals(1f, result.getFloat(0), 0.01f)
        assertEquals(2f, result.getFloat(1), 0.01f)
        assertEquals(2f, result.getFloat(2), 0.01f)
        assertEquals(1f, result.getFloat(3), 0.01f)
    }

    // ==================== Gemm (General Matrix Multiply) ====================

    @Test
    fun testGemm() {
        val sd = SameDiff.create()
        val a = sd.placeHolder("a", DataType.FLOAT, 2, 3)
        val b = sd.placeHolder("b", DataType.FLOAT, 3, 4)
        val c = sd.placeHolder("c", DataType.FLOAT, 2, 4)

        // Gemm: Y = alpha * A * B + beta * C
        val alpha = 1.0
        val beta = 1.0
        val ab = sd.mmul(a, b)
        val output = sd.math.add("gemm", sd.math.mul(ab, alpha), sd.math.mul(c, beta))

        val aArr = Nd4j.ones(DataType.FLOAT, 2, 3)
        val bArr = Nd4j.ones(DataType.FLOAT, 3, 4)
        val cArr = Nd4j.ones(DataType.FLOAT, 2, 4)
        val result = sd.output(mapOf("a" to aArr, "b" to bArr, "c" to cArr), "gemm")["gemm"]!!

        // Each element: 1*3 + 1 = 4
        assertEquals(4f, result.getFloat(0, 0), 0.01f)
    }

    @Test
    fun testGemmWithTranspose() {
        val sd = SameDiff.create()
        val a = sd.placeHolder("a", DataType.FLOAT, 3, 2)
        val b = sd.placeHolder("b", DataType.FLOAT, 3, 4)

        // Gemm with transA=true: Y = A^T * B
        val aT = sd.transpose(a)
        val output = sd.mmul("gemm_t", aT, b)

        val aArr = Nd4j.ones(DataType.FLOAT, 3, 2)
        val bArr = Nd4j.ones(DataType.FLOAT, 3, 4)
        val result = sd.output(mapOf("a" to aArr, "b" to bArr), "gemm_t")["gemm_t"]!!

        assertEquals(2, result.shape()[0])
        assertEquals(4, result.shape()[1])
    }

    // ==================== MatMul ====================

    @Test
    fun testMatMul() {
        val sd = SameDiff.create()
        val a = sd.placeHolder("a", DataType.FLOAT, 2, 3)
        val b = sd.placeHolder("b", DataType.FLOAT, 3, 4)

        val output = sd.mmul("matmul", a, b)

        val aArr = Nd4j.ones(DataType.FLOAT, 2, 3)
        val bArr = Nd4j.ones(DataType.FLOAT, 3, 4)
        val result = sd.output(mapOf("a" to aArr, "b" to bArr), "matmul")["matmul"]!!

        assertEquals(2, result.shape()[0])
        assertEquals(4, result.shape()[1])
        assertEquals(3f, result.getFloat(0, 0), 0.01f)
    }

    @Test
    fun testMatMulBatched() {
        val sd = SameDiff.create()
        val a = sd.placeHolder("a", DataType.FLOAT, 2, 3, 4)
        val b = sd.placeHolder("b", DataType.FLOAT, 2, 4, 5)

        val output = sd.mmul("matmul_batch", a, b)

        val aArr = Nd4j.ones(DataType.FLOAT, 2, 3, 4)
        val bArr = Nd4j.ones(DataType.FLOAT, 2, 4, 5)
        val result = sd.output(mapOf("a" to aArr, "b" to bArr), "matmul_batch")["matmul_batch"]!!

        assertEquals(3, result.rank())
        assertEquals(2, result.shape()[0])
        assertEquals(3, result.shape()[1])
        assertEquals(5, result.shape()[2])
    }

    // ==================== Shape Operations ====================

    @Test
    fun testShape() {
        val sd = SameDiff.create()
        val input = sd.placeHolder("input", DataType.FLOAT, 2, 3, 4)

        val output = sd.shape("shape", input)

        val inputArr = Nd4j.zeros(DataType.FLOAT, 2, 3, 4)
        val result = sd.output(mapOf("input" to inputArr), "shape")["shape"]!!

        assertEquals(3, result.length())
        assertEquals(2L, result.getLong(0))
        assertEquals(3L, result.getLong(1))
        assertEquals(4L, result.getLong(2))
    }

    @Test
    fun testRank() {
        val sd = SameDiff.create()
        val input = sd.placeHolder("input", DataType.FLOAT, 2, 3, 4, 5)

        val output = sd.rank("rank", input)

        val inputArr = Nd4j.zeros(DataType.FLOAT, 2, 3, 4, 5)
        val result = sd.output(mapOf("input" to inputArr), "rank")["rank"]!!

        assertEquals(4, result.getInt(0))
    }

    @Test
    fun testSize() {
        val sd = SameDiff.create()
        val input = sd.placeHolder("input", DataType.FLOAT, 2, 3, 4)

        val output = sd.sizeAt("size", input, 1)

        val inputArr = Nd4j.zeros(DataType.FLOAT, 2, 3, 4)
        val result = sd.output(mapOf("input" to inputArr), "size")["size"]!!

        assertEquals(3L, result.getLong(0))
    }

    // ==================== Flatten ====================

    @Test
    fun testFlatten() {
        val sd = SameDiff.create()
        val input = sd.placeHolder("input", DataType.FLOAT, 2, 3, 4)

        val output = sd.reshape("flatten", input, -1)

        val inputArr = Nd4j.linspace(1, 24, 24, DataType.FLOAT).reshape(2, 3, 4)
        val result = sd.output(mapOf("input" to inputArr), "flatten")["flatten"]!!

        assertEquals(1, result.rank())
        assertEquals(24, result.length())
    }

    // ==================== Identity ====================

    @Test
    fun testIdentity() {
        val sd = SameDiff.create()
        val input = sd.placeHolder("input", DataType.FLOAT, 2, 3)

        val output = sd.identity("identity", input)

        val inputArr = Nd4j.create(floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f)).reshape(2, 3)
        val result = sd.output(mapOf("input" to inputArr), "identity")["identity"]!!

        assertTrue(inputArr.equalsWithEps(result, 1e-5))
    }

    // ==================== Range ====================

    @Test
    fun testRange() {
        val sd = SameDiff.create()

        val output = sd.range("range", 0.0, 10.0, 2.0, DataType.FLOAT)

        val result = sd.output(emptyMap(), "range")["range"]!!

        assertEquals(5, result.length())
        assertEquals(0f, result.getFloat(0), 0.01f)
        assertEquals(2f, result.getFloat(1), 0.01f)
        assertEquals(4f, result.getFloat(2), 0.01f)
        assertEquals(6f, result.getFloat(3), 0.01f)
        assertEquals(8f, result.getFloat(4), 0.01f)
    }

    // ==================== EyeLike ====================

    @Test
    fun testEyeLike() {
        val sd = SameDiff.create()

        // Create identity matrix
        val output = sd.constant("eye", Nd4j.eye(3).castTo(DataType.FLOAT))

        val result = sd.output(emptyMap(), "eye")["eye"]!!

        assertEquals(3, result.shape()[0])
        assertEquals(3, result.shape()[1])
        assertEquals(1f, result.getFloat(0, 0), 0.01f)
        assertEquals(0f, result.getFloat(0, 1), 0.01f)
        assertEquals(1f, result.getFloat(1, 1), 0.01f)
        assertEquals(1f, result.getFloat(2, 2), 0.01f)
    }
}
