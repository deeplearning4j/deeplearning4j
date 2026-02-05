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

package org.eclipse.deeplearning4j.nd4j.linalg.api;

import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.util.Collections;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test to reproduce ReduceSum on INT64 issue.
 *
 * The CUDA backend does not support reduction operations on integer types.
 * This causes memory corruption when trying to execute ReduceSum on INT64.
 *
 * Error: "NativeOpExecutioner::execReduce3Scalar requires Z operand to have floating point data type. Z type: INT64"
 */
public class TestReduceSumInt64 extends BaseNd4jTestWithBackends {

    @Override
    public char ordering() {
        return 'c';
    }

    /**
     * Test direct INDArray.sum() on INT64 - this is the raw operation
     */
    @Test
    public void testReduceSumInt64Direct() {
        // Create INT64 tensor
        INDArray input = Nd4j.ones(DataType.INT64, 1, 3, 4, 4);
        System.out.println("Input shape: " + java.util.Arrays.toString(input.shape()));
        System.out.println("Input dtype: " + input.dataType());
        System.out.println("Input sum expected: " + (1 * 3 * 4 * 4)); // 48

        // Try to sum - this should fail or corrupt memory on CUDA with INT64
        INDArray result = input.sum();

        System.out.println("Result shape: " + java.util.Arrays.toString(result.shape()));
        System.out.println("Result dtype: " + result.dataType());
        System.out.println("Result value: " + result.getDouble(0));

        assertEquals(48.0, result.getDouble(0), 1e-5, "Sum should be 48");
    }

    /**
     * Test SameDiff sum on INT64 - this is closer to how ONNX import uses it
     */
    @Test
    public void testReduceSumInt64SameDiff() {
        SameDiff sd = SameDiff.create();

        // Create INT64 input
        INDArray inputArr = Nd4j.ones(DataType.INT64, 1, 3, 4, 4);
        SDVariable input = sd.var("input", inputArr);

        System.out.println("SameDiff Input dtype: " + input.dataType());

        // Sum all dimensions - keepdims=false produces scalar
        SDVariable sum = sd.sum("sum_result", input, false);

        System.out.println("SameDiff Sum output dtype: " + sum.dataType());

        // Execute
        Map<String, INDArray> result = sd.output(Collections.<String, INDArray>emptyMap(), "sum_result");
        INDArray sumResult = result.get("sum_result");

        System.out.println("Result shape: " + java.util.Arrays.toString(sumResult.shape()));
        System.out.println("Result dtype: " + sumResult.dataType());
        System.out.println("Result value: " + sumResult.getDouble(0));

        assertEquals(48.0, sumResult.getDouble(0), 1e-5, "Sum should be 48");
    }

    /**
     * Test SameDiff sum on INT64 with keepdims=true
     */
    @Test
    public void testReduceSumInt64SameDiffKeepDims() {
        SameDiff sd = SameDiff.create();

        // Create INT64 input
        INDArray inputArr = Nd4j.ones(DataType.INT64, 1, 3, 4, 4);
        SDVariable input = sd.var("input", inputArr);

        // Sum all dimensions with keepdims
        SDVariable sum = sd.sum("sum_result", input, true);

        // Execute
        Map<String, INDArray> result = sd.output(Collections.<String, INDArray>emptyMap(), "sum_result");
        INDArray sumResult = result.get("sum_result");

        System.out.println("Result shape: " + java.util.Arrays.toString(sumResult.shape()));
        System.out.println("Result dtype: " + sumResult.dataType());
        System.out.println("Result value: " + sumResult.getDouble(0));

        assertEquals(48.0, sumResult.getDouble(0), 1e-5, "Sum should be 48");
    }

    /**
     * Test the exact scenario from SmolDocling: Cast BOOL to INT64, then ReduceSum
     */
    @Test
    public void testCastBoolToInt64ThenReduceSum() {
        SameDiff sd = SameDiff.create();

        // Create BOOL input (like pixel_attention_mask)
        INDArray boolArr = Nd4j.ones(DataType.BOOL, 1, 3, 512, 512);
        SDVariable boolInput = sd.var("bool_input", boolArr);

        System.out.println("Bool input dtype: " + boolInput.dataType());

        // Cast to INT64 (this is what the ONNX Cast op does)
        SDVariable int64Input = sd.castTo("cast_to_int64", boolInput, DataType.INT64);

        System.out.println("After cast dtype: " + int64Input.dataType());

        // ReduceSum all dimensions
        SDVariable sum = sd.sum("reduce_sum", int64Input, false);

        System.out.println("Sum output dtype: " + sum.dataType());

        // Compare with a constant (like the ONNX model does with equals_1)
        long expectedSum = 1L * 3 * 512 * 512; // 786432
        SDVariable constant = sd.constant("expected", Nd4j.scalar(DataType.INT64, expectedSum));
        SDVariable equals = sd.eq("equals_check", sum, constant);

        // Execute
        Map<String, INDArray> result = sd.output(Collections.<String, INDArray>emptyMap(), "reduce_sum", "equals_check");

        INDArray sumResult = result.get("reduce_sum");
        INDArray equalsResult = result.get("equals_check");

        System.out.println("Sum result shape: " + java.util.Arrays.toString(sumResult.shape()));
        System.out.println("Sum result dtype: " + sumResult.dataType());
        System.out.println("Sum result value: " + sumResult.getDouble(0));
        System.out.println("Expected value: " + expectedSum);
        System.out.println("Equals result: " + equalsResult.getInt(0));

        assertEquals((double) expectedSum, sumResult.getDouble(0), 1e-5,
            "Sum should be " + expectedSum);
        assertEquals(1, equalsResult.getInt(0), "Equals should be true (1)");
    }

    /**
     * Test workaround: Cast INT64 to DOUBLE before sum, then cast back
     */
    @Test
    public void testReduceSumInt64Workaround() {
        SameDiff sd = SameDiff.create();

        // Create INT64 input
        INDArray inputArr = Nd4j.ones(DataType.INT64, 1, 3, 4, 4);
        SDVariable input = sd.var("input", inputArr);

        // WORKAROUND: Cast to DOUBLE first
        SDVariable asDouble = sd.castTo("as_double", input, DataType.DOUBLE);

        // Sum on DOUBLE (this should work)
        SDVariable sum = sd.sum("sum_double", asDouble, false);

        // Cast back to INT64
        SDVariable result = sd.castTo("sum_result", sum, DataType.INT64);

        // Execute
        Map<String, INDArray> output = sd.output(Collections.<String, INDArray>emptyMap(), "sum_result");
        INDArray sumResult = output.get("sum_result");

        System.out.println("Workaround result shape: " + java.util.Arrays.toString(sumResult.shape()));
        System.out.println("Workaround result dtype: " + sumResult.dataType());
        System.out.println("Workaround result value: " + sumResult.getDouble(0));

        assertEquals(48.0, sumResult.getDouble(0), 1e-5, "Sum should be 48");
    }
}
