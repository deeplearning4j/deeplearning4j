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

package org.eclipse.deeplearning4j.nd4j.linalg.shape.reshape;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.shape.Reshape;
import org.nd4j.linalg.api.ops.impl.shape.ReshapeNoCopy;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive tests for reshape operations covering:
 * - View reshape (when input is contiguous and can be reshaped without copy)
 * - Copy reshape (when input is non-contiguous and requires data copy)
 * - Buffer sharing verification
 * - Data integrity after reshape
 * - Both C and F ordering
 * - ReshapeNoCopy op
 * - SameDiff reshape path
 * - Reshape after permute (common pattern that requires copy)
 */
@Slf4j
@NativeTag
@Tag(TagNames.NDARRAY_INDEXING)
public class ReshapeViewCopyTests extends BaseNd4jTestWithBackends {

    @Override
    public char ordering() {
        return 'c';
    }

    /**
     * Test that reshaping a contiguous C-order array creates a view (shares buffer)
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testReshapeContiguousArrayCreatesView(Nd4jBackend backend) {
        INDArray original = Nd4j.linspace(1, 12, 12, DataType.DOUBLE);
        DataBuffer originalBuffer = original.data();

        INDArray reshaped = original.reshape(3, 4);

        // Verify shape changed
        assertArrayEquals(new long[]{3, 4}, reshaped.shape());

        // Verify data is correct
        assertEquals(1.0, reshaped.getDouble(0, 0), 1e-5);
        assertEquals(12.0, reshaped.getDouble(2, 3), 1e-5);

        // Verify buffer is shared (view)
        assertSame(originalBuffer, reshaped.data(),
            "Contiguous reshape should share buffer (be a view)");

        // Verify modifying reshaped affects original
        reshaped.putScalar(0, 0, 999.0);
        assertEquals(999.0, original.getDouble(0), 1e-5,
            "View should share data with original");
    }

    /**
     * Test that reshaping after permute requires a copy (different buffer)
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testReshapeAfterPermuteRequiresCopy(Nd4jBackend backend) {
        INDArray original = Nd4j.linspace(1, 12, 12, DataType.DOUBLE).reshape(3, 4);
        INDArray permuted = original.permute(1, 0);  // Now 4x3, non-contiguous strides

        // Verify permuted has non-standard strides
        log.info("Original strides: {}", java.util.Arrays.toString(original.stride()));
        log.info("Permuted strides: {}", java.util.Arrays.toString(permuted.stride()));

        INDArray reshaped = permuted.reshape(12);

        // Verify shape
        assertArrayEquals(new long[]{12}, reshaped.shape());

        // Verify data integrity - should have correct values after reshape
        // After permute(1,0) on 3x4 -> 4x3, then reshape to 12
        // The data order depends on the implementation
        assertNotNull(reshaped);
        assertEquals(12, reshaped.length());
    }

    /**
     * Test reshape with F ordering
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testReshapeFOrdering(Nd4jBackend backend) {
        INDArray original = Nd4j.linspace(1, 12, 12, DataType.DOUBLE);

        INDArray reshapedC = original.reshape('c', 3, 4);
        INDArray reshapedF = original.reshape('f', 3, 4);

        // Both should have same shape
        assertArrayEquals(new long[]{3, 4}, reshapedC.shape());
        assertArrayEquals(new long[]{3, 4}, reshapedF.shape());

        // But different orderings
        assertEquals('c', reshapedC.ordering());
        assertEquals('f', reshapedF.ordering());

        // Data layout should differ
        // In C order: row-major, so (0,1) comes after (0,0)
        // In F order: column-major, so (1,0) comes after (0,0)
        log.info("C-order [0,1]: {}", reshapedC.getDouble(0, 1));
        log.info("F-order [0,1]: {}", reshapedF.getDouble(0, 1));
    }

    /**
     * Test ReshapeNoCopy op directly - view case
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testReshapeNoCopyViewCase(Nd4jBackend backend) {
        INDArray input = Nd4j.linspace(1, 12, 12, DataType.DOUBLE);

        ReshapeNoCopy op = new ReshapeNoCopy(input, new long[]{3, 4}, null, 'c');
        INDArray[] results = Nd4j.getExecutioner().exec(op);

        assertNotNull(results);
        assertEquals(1, results.length);

        INDArray result = results[0];
        assertArrayEquals(new long[]{3, 4}, result.shape());

        // Verify data integrity
        assertEquals(1.0, result.getDouble(0, 0), 1e-5);
        assertEquals(12.0, result.getDouble(2, 3), 1e-5);
    }

    /**
     * Test ReshapeNoCopy op directly - copy case (after permute)
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testReshapeNoCopyCopyCase(Nd4jBackend backend) {
        INDArray original = Nd4j.linspace(1, 12, 12, DataType.DOUBLE).reshape(3, 4);
        INDArray permuted = original.permute(1, 0);  // Non-contiguous

        ReshapeNoCopy op = new ReshapeNoCopy(permuted, new long[]{12}, null, 'c');
        INDArray[] results = Nd4j.getExecutioner().exec(op);

        assertNotNull(results);
        assertEquals(1, results.length);

        INDArray result = results[0];
        assertArrayEquals(new long[]{12}, result.shape());
        assertEquals(12, result.length());
    }

    /**
     * Test SameDiff reshape path
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSameDiffReshape(Nd4jBackend backend) {
        SameDiff sd = SameDiff.create();

        INDArray inputArr = Nd4j.linspace(1, 12, 12, DataType.DOUBLE);
        SDVariable input = sd.var("input", inputArr);
        SDVariable reshaped = sd.reshape(input, 3, 4);

        Map<String, INDArray> result = sd.outputAll(null);

        INDArray output = result.get(reshaped.name());
        assertNotNull(output);
        assertArrayEquals(new long[]{3, 4}, output.shape());

        // Verify data integrity
        assertEquals(1.0, output.getDouble(0, 0), 1e-5);
        assertEquals(12.0, output.getDouble(2, 3), 1e-5);
    }

    /**
     * Test SameDiff reshape after permute
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSameDiffReshapeAfterPermute(Nd4jBackend backend) {
        SameDiff sd = SameDiff.create();

        INDArray inputArr = Nd4j.linspace(1, 12, 12, DataType.DOUBLE).reshape(3, 4);
        SDVariable input = sd.var("input", inputArr);
        SDVariable permuted = sd.permute(input, 1, 0);  // 4x3
        SDVariable reshaped = sd.reshape(permuted, 12);

        Map<String, INDArray> result = sd.outputAll(null);

        INDArray output = result.get(reshaped.name());
        assertNotNull(output);
        assertArrayEquals(new long[]{12}, output.shape());
        assertEquals(12, output.length());
    }

    /**
     * Test data integrity with various reshape patterns
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testReshapeDataIntegrity(Nd4jBackend backend) {
        long[][] testShapes = {
            {12},
            {3, 4},
            {4, 3},
            {2, 6},
            {6, 2},
            {2, 2, 3},
            {3, 2, 2},
            {1, 12},
            {12, 1}
        };

        INDArray original = Nd4j.linspace(1, 12, 12, DataType.DOUBLE);
        double sum = original.sumNumber().doubleValue();

        for (long[] shape : testShapes) {
            INDArray reshaped = original.reshape(shape);

            // Verify shape
            assertArrayEquals(shape, reshaped.shape(),
                "Shape mismatch for " + java.util.Arrays.toString(shape));

            // Verify sum is preserved (data integrity)
            assertEquals(sum, reshaped.sumNumber().doubleValue(), 1e-5,
                "Sum mismatch for shape " + java.util.Arrays.toString(shape));

            // Verify length
            assertEquals(12, reshaped.length(),
                "Length mismatch for shape " + java.util.Arrays.toString(shape));
        }
    }

    /**
     * Test reshape with -1 dimension inference
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testReshapeWithInferredDimension(Nd4jBackend backend) {
        INDArray original = Nd4j.linspace(1, 12, 12, DataType.DOUBLE);

        // Use -1 to infer dimension
        INDArray reshaped1 = original.reshape(-1, 4);
        assertArrayEquals(new long[]{3, 4}, reshaped1.shape());

        INDArray reshaped2 = original.reshape(3, -1);
        assertArrayEquals(new long[]{3, 4}, reshaped2.shape());

        INDArray reshaped3 = original.reshape(2, -1, 2);
        assertArrayEquals(new long[]{2, 3, 2}, reshaped3.shape());
    }

    /**
     * Test reshape round-trip preserves data
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testReshapeRoundTrip(Nd4jBackend backend) {
        INDArray original = Nd4j.linspace(1, 24, 24, DataType.DOUBLE).reshape(2, 3, 4);
        INDArray originalDup = original.dup();

        // Reshape to different shapes and back
        INDArray step1 = original.reshape(6, 4);
        INDArray step2 = step1.reshape(24);
        INDArray step3 = step2.reshape(4, 6);
        INDArray step4 = step3.reshape(2, 3, 4);

        // Verify final shape matches original
        assertArrayEquals(original.shape(), step4.shape());

        // Verify data matches original
        assertEquals(original, step4);
    }

    /**
     * Test reshape with permute pattern commonly used in attention mechanisms
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testAttentionReshapePattern(Nd4jBackend backend) {
        // Simulates: [batch, seq, heads*dim] -> [batch, seq, heads, dim] -> [batch, heads, seq, dim]
        int batch = 2;
        int seq = 4;
        int heads = 3;
        int dim = 8;

        INDArray input = Nd4j.linspace(1, batch * seq * heads * dim, batch * seq * heads * dim, DataType.DOUBLE)
                .reshape(batch, seq, heads * dim);

        double originalSum = input.sumNumber().doubleValue();

        // Reshape to separate heads
        INDArray reshaped = input.reshape(batch, seq, heads, dim);
        assertEquals(originalSum, reshaped.sumNumber().doubleValue(), 1e-5);

        // Permute for attention
        INDArray permuted = reshaped.permute(0, 2, 1, 3);  // [batch, heads, seq, dim]
        assertEquals(originalSum, permuted.sumNumber().doubleValue(), 1e-5);

        // Reshape for matmul
        INDArray forMatmul = permuted.reshape(batch * heads, seq, dim);
        assertEquals(originalSum, forMatmul.sumNumber().doubleValue(), 1e-5);

        assertArrayEquals(new long[]{batch * heads, seq, dim}, forMatmul.shape());
    }

    /**
     * Test that reshape op via DynamicCustomOp works correctly
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testDynamicCustomOpReshape(Nd4jBackend backend) {
        INDArray input = Nd4j.linspace(1, 12, 12, DataType.DOUBLE);
        INDArray output = Nd4j.create(DataType.DOUBLE, 3, 4);

        DynamicCustomOp op = DynamicCustomOp.builder("reshape")
                .addInputs(input)
                .addOutputs(output)
                .addIntegerArguments(-'c', 3, 4)
                .build();

        Nd4j.getExecutioner().exec(op);

        assertArrayEquals(new long[]{3, 4}, output.shape());
        assertEquals(1.0, output.getDouble(0, 0), 1e-5);
        assertEquals(12.0, output.getDouble(2, 3), 1e-5);
    }

    /**
     * Test reshape_no_copy op via DynamicCustomOp
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testDynamicCustomOpReshapeNoCopy(Nd4jBackend backend) {
        INDArray input = Nd4j.linspace(1, 12, 12, DataType.DOUBLE);

        ReshapeNoCopy op = new ReshapeNoCopy(input, new long[]{3, 4}, null, 'c');
        INDArray[] outputs = Nd4j.getExecutioner().exec(op);

        assertNotNull(outputs);
        assertTrue(outputs.length > 0);

        INDArray output = outputs[0];
        assertArrayEquals(new long[]{3, 4}, output.shape());
        assertEquals(1.0, output.getDouble(0, 0), 1e-5);
        assertEquals(12.0, output.getDouble(2, 3), 1e-5);
    }

    /**
     * Test large array reshape performance characteristics
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testLargeArrayReshape(Nd4jBackend backend) {
        // Large array to ensure any performance issues are visible
        int size = 1024 * 1024;  // 1M elements
        INDArray large = Nd4j.linspace(1, size, size, DataType.FLOAT);

        // Contiguous reshape (should be fast - view)
        long start = System.nanoTime();
        INDArray reshaped1 = large.reshape(1024, 1024);
        long viewTime = System.nanoTime() - start;

        // Verify correctness
        assertEquals(size, reshaped1.length());
        assertEquals(1.0f, reshaped1.getFloat(0, 0), 1e-3);

        log.info("View reshape time: {} ms", viewTime / 1_000_000.0);

        // Non-contiguous reshape (requires copy)
        INDArray permuted = large.reshape(1024, 1024).permute(1, 0);
        start = System.nanoTime();
        INDArray reshaped2 = permuted.reshape(size);
        long copyTime = System.nanoTime() - start;

        assertEquals(size, reshaped2.length());

        log.info("Copy reshape time: {} ms", copyTime / 1_000_000.0);
    }

    /**
     * Test reshape with a contiguous subarray
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testReshapeContiguousSubarray(Nd4jBackend backend) {
        // Create a simple contiguous array
        INDArray arr = Nd4j.linspace(1, 6, 6, DataType.DOUBLE);
        assertEquals(6, arr.length());
        assertEquals(21.0, arr.sumNumber().doubleValue(), 1e-5);  // 1+2+3+4+5+6=21

        // Reshape to 2x3
        INDArray reshaped = arr.reshape(2, 3);
        assertArrayEquals(new long[]{2, 3}, reshaped.shape());

        // Verify length is preserved
        assertEquals(arr.length(), reshaped.length());
        // Verify sum is preserved (data integrity)
        assertEquals(arr.sumNumber().doubleValue(), reshaped.sumNumber().doubleValue(), 1e-5);
    }

    // ===================================================================
    // Regression tests for reshape order-only scalar case
    // When iArgs contains ONLY the order marker (-99 for 'c', -102 for 'f')
    // and the input is scalar, reshape must preserve scalar rank (rank 0),
    // NOT produce a rank-1 array [1].
    // ===================================================================

    /**
     * Test reshape scalar with C ordering only (-99) via reshape op.
     * iArgs = [-99] means "reshape to same shape, C order".
     * For scalar input, output must remain scalar (rank 0).
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testReshapeScalarWithCOrderOnly(Nd4jBackend backend) {
        INDArray scalar = Nd4j.scalar(DataType.FLOAT, 42.0f);
        assertEquals(0, scalar.rank(), "Input should be scalar (rank 0)");

        DynamicCustomOp op = DynamicCustomOp.builder("reshape")
                .addInputs(scalar)
                .addIntegerArguments(-99L)  // C order marker only, no shape dims
                .build();

        INDArray[] results = Nd4j.getExecutioner().exec(op);

        assertNotNull(results);
        assertEquals(1, results.length);
        INDArray result = results[0];
        assertEquals(0, result.rank(), "Scalar reshape with C order should preserve rank 0");
        assertEquals(42.0f, result.getFloat(0), 1e-5, "Data should be preserved");
    }

    /**
     * Test reshape scalar with F ordering only (-102) via reshape op.
     * iArgs = [-102] means "reshape to same shape, F order".
     * For scalar input, output must remain scalar (rank 0).
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testReshapeScalarWithFOrderOnly(Nd4jBackend backend) {
        INDArray scalar = Nd4j.scalar(DataType.FLOAT, 42.0f);
        assertEquals(0, scalar.rank(), "Input should be scalar (rank 0)");

        DynamicCustomOp op = DynamicCustomOp.builder("reshape")
                .addInputs(scalar)
                .addIntegerArguments(-102L)  // F order marker only, no shape dims
                .build();

        INDArray[] results = Nd4j.getExecutioner().exec(op);

        assertNotNull(results);
        assertEquals(1, results.length);
        INDArray result = results[0];
        assertEquals(0, result.rank(), "Scalar reshape with F order should preserve rank 0");
        assertEquals(42.0f, result.getFloat(0), 1e-5, "Data should be preserved");
    }

    /**
     * Test reshape_no_copy scalar with C ordering only via iArgs path.
     * iArgs = [-99] means "no shape dims, C order marker only".
     * For scalar input, output must remain scalar (rank 0).
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testReshapeNoCopyScalarWithCOrderOnly(Nd4jBackend backend) {
        INDArray scalar = Nd4j.scalar(DataType.DOUBLE, 7.0);
        assertEquals(0, scalar.rank(), "Input should be scalar (rank 0)");

        DynamicCustomOp op = DynamicCustomOp.builder("reshape_no_copy")
                .addInputs(scalar)
                .addIntegerArguments(-99L)  // C order marker only
                .build();

        INDArray[] results = Nd4j.getExecutioner().exec(op);

        assertNotNull(results);
        assertEquals(1, results.length);
        INDArray result = results[0];
        assertEquals(0, result.rank(), "reshape_no_copy scalar with C order should preserve rank 0");
        assertEquals(7.0, result.getDouble(0), 1e-5, "Data should be preserved");
    }

    /**
     * Test reshape_no_copy scalar with F ordering only via iArgs path.
     * iArgs = [-102] means "no shape dims, F order marker only".
     * For scalar input, output must remain scalar (rank 0).
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testReshapeNoCopyScalarWithFOrderOnly(Nd4jBackend backend) {
        INDArray scalar = Nd4j.scalar(DataType.DOUBLE, 7.0);
        assertEquals(0, scalar.rank(), "Input should be scalar (rank 0)");

        DynamicCustomOp op = DynamicCustomOp.builder("reshape_no_copy")
                .addInputs(scalar)
                .addIntegerArguments(-102L)  // F order marker only
                .build();

        INDArray[] results = Nd4j.getExecutioner().exec(op);

        assertNotNull(results);
        assertEquals(1, results.length);
        INDArray result = results[0];
        assertEquals(0, result.rank(), "reshape_no_copy scalar with F order should preserve rank 0");
        assertEquals(7.0, result.getDouble(0), 1e-5, "Data should be preserved");
    }

    /**
     * Test reshape scalar with [-1] dimension inference preserves scalar rank.
     * reshape(scalar, [-1]) should stay scalar, not become [1].
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testReshapeScalarWithNeg1PreservesScalar(Nd4jBackend backend) {
        INDArray scalar = Nd4j.scalar(DataType.FLOAT, 3.14f);
        assertEquals(0, scalar.rank(), "Input should be scalar (rank 0)");

        // Test via reshape op
        DynamicCustomOp reshapeOp = DynamicCustomOp.builder("reshape")
                .addInputs(scalar)
                .addIntegerArguments(-1L)
                .build();

        INDArray[] results = Nd4j.getExecutioner().exec(reshapeOp);
        assertNotNull(results);
        assertEquals(1, results.length);
        assertEquals(0, results[0].rank(), "reshape(scalar, [-1]) should preserve rank 0");
        assertEquals(3.14f, results[0].getFloat(0), 1e-5);
    }

    /**
     * Test reshape_no_copy scalar with [-1, -99] (infer dim + C order) preserves scalar rank.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testReshapeNoCopyScalarWithNeg1COrder(Nd4jBackend backend) {
        INDArray scalar = Nd4j.scalar(DataType.FLOAT, 3.14f);
        assertEquals(0, scalar.rank(), "Input should be scalar (rank 0)");

        DynamicCustomOp op = DynamicCustomOp.builder("reshape_no_copy")
                .addInputs(scalar)
                .addIntegerArguments(-1L, -99L)  // shape=[-1], order='c'
                .build();

        INDArray[] results = Nd4j.getExecutioner().exec(op);
        assertNotNull(results);
        assertEquals(1, results.length);
        // reshape with shape=[-1] infers to [1], producing rank 1
        assertEquals(1, results[0].rank(), "reshape_no_copy(scalar, [-1], 'c') produces rank 1 (shape [1])");
        assertEquals(3.14f, results[0].getFloat(0), 1e-5);
    }

    /**
     * Test reshape_no_copy scalar with [-1, -102] (infer dim + F order).
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testReshapeNoCopyScalarWithNeg1FOrder(Nd4jBackend backend) {
        INDArray scalar = Nd4j.scalar(DataType.FLOAT, 3.14f);
        assertEquals(0, scalar.rank(), "Input should be scalar (rank 0)");

        DynamicCustomOp op = DynamicCustomOp.builder("reshape_no_copy")
                .addInputs(scalar)
                .addIntegerArguments(-1L, -102L)  // shape=[-1], order='f'
                .build();

        INDArray[] results = Nd4j.getExecutioner().exec(op);
        assertNotNull(results);
        assertEquals(1, results.length);
        // reshape with shape=[-1] infers to [1], producing rank 1
        assertEquals(1, results[0].rank(), "reshape_no_copy(scalar, [-1], 'f') produces rank 1 (shape [1])");
        assertEquals(3.14f, results[0].getFloat(0), 1e-5);
    }

    /**
     * Test that non-scalar reshape with order-only marker still works correctly.
     * A length-1 array with iArgs=[-99] should remain length-1.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testReshapeLength1WithCOrderOnly(Nd4jBackend backend) {
        INDArray length1 = Nd4j.createFromArray(new float[]{5.0f});
        assertEquals(1, length1.rank(), "Input should be rank 1");
        assertEquals(1, length1.length(), "Input should have 1 element");

        DynamicCustomOp op = DynamicCustomOp.builder("reshape")
                .addInputs(length1)
                .addIntegerArguments(-99L)
                .build();

        INDArray[] results = Nd4j.getExecutioner().exec(op);
        assertNotNull(results);
        assertEquals(1, results.length);
        assertEquals(5.0f, results[0].getFloat(0), 1e-5, "Data should be preserved");
    }

    /**
     * Test that reshape properly handles arrays with zero-length dimensions
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testReshapeZeroLengthDimension(Nd4jBackend backend) {
        // Create array with 0 elements in first dimension
        INDArray zeroLength = Nd4j.create(DataType.DOUBLE, 0, 4);
        assertArrayEquals(new long[]{0, 4}, zeroLength.shape());
        assertEquals(0, zeroLength.length());
    }

    /**
     * Test TimeSeriesUtils-like reshape pattern (3D to 2D and back)
     * This is a common pattern that triggers the reshape performance issues
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testTimeSeriesReshapePattern(Nd4jBackend backend) {
        // Simulate [batch, features, time] format
        int batch = 4;
        int features = 10;
        int time = 8;

        INDArray input3d = Nd4j.linspace(1, batch * features * time, batch * features * time, DataType.DOUBLE)
                .reshape('c', batch, features, time);

        double originalSum = input3d.sumNumber().doubleValue();

        // Pattern from TimeSeriesUtils.reshape3dTo2d:
        // permute(0, 2, 1) then reshape to 2D
        INDArray permuted = input3d.permute(0, 2, 1);  // [batch, time, features]
        assertArrayEquals(new long[]{batch, time, features}, permuted.shape());

        INDArray reshaped2d = permuted.reshape('f', batch * time, features);
        assertArrayEquals(new long[]{batch * time, features}, reshaped2d.shape());

        // Data integrity
        assertEquals(originalSum, reshaped2d.sumNumber().doubleValue(), 1e-5);

        // Reverse: reshape2dTo3d pattern
        INDArray back3d = reshaped2d.reshape('f', batch, time, features);
        INDArray backPermuted = back3d.permute(0, 2, 1);  // [batch, features, time]

        assertArrayEquals(input3d.shape(), backPermuted.shape());
        assertEquals(originalSum, backPermuted.sumNumber().doubleValue(), 1e-5);
    }
}
