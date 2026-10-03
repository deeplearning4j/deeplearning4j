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
package org.eclipse.deeplearning4j.nd4j.linalg.ops;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.Arrays;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * A matrix-vector product over mixed float storage (a HALF or BFLOAT16 weight against a FLOAT
 * activation in a decode step) runs as MmulHelper's mixed GEMV: every operand is read in place in
 * its storage type and the sums are FLOAT, or DOUBLE when an operand is DOUBLE. Results equal a
 * DOUBLE product of the stored values within the accumulation and output rounding, for one output
 * row (M == 1), one output column (N == 1) and a matrix times a rank-1 vector, through contiguous,
 * transposed and strided operands, with alpha and beta.
 */
@NativeTag
public class MixedDtypeGemvTest extends BaseNd4jTestWithBackends {

    private static final DataType[] FLOAT_TYPES = {DataType.HALF, DataType.BFLOAT16, DataType.FLOAT, DataType.DOUBLE};

    /** The products that reduce to a GEMV: [1,K] x [K,N], [M,K] x [K,1] and [M,K] x [K]. */
    enum Product { ROW, COLUMN, VECTOR }

    /** How the operands sit in memory. */
    enum Layout { C, F, TRANSPOSED, STRIDED }

    public static Stream<Arguments> mixedPairs() {
        return configs().flatMap(backend -> Arrays.stream(FLOAT_TYPES)
                .flatMap(matrix -> Arrays.stream(FLOAT_TYPES)
                        .filter(vector -> vector != matrix)
                        .map(vector -> Arguments.of(backend.get()[0], matrix, vector))));
    }

    /** Every product and layout, into an output of either input type, with alpha and beta. */
    @ParameterizedTest
    @MethodSource("mixedPairs")
    public void everyProductAndLayout(Nd4jBackend backend, DataType matrixType, DataType vectorType) {
        for (Product product : Product.values())
            for (Layout layout : Layout.values())
                for (DataType outputType : new DataType[]{matrixType, vectorType})
                    checkProduct(product, layout, matrixType, vectorType, outputType, 37, 45, 0.75, 0.5);
    }

    /**
     * Output and depth counts on both sides of a warp, and a depth of 4096, through both kernels:
     * a depth-major matrix ('f' weights of [1,K] x [K,N]) and a row-major one ('c').
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void sizesAroundTheWarp(Nd4jBackend backend) {
        for (long outputs : new long[]{1, 31, 32, 33, 1537})
            for (long depth : new long[]{1, 31, 32, 33, 4096})
                for (Layout layout : new Layout[]{Layout.C, Layout.F})
                    checkProduct(Product.ROW, layout, DataType.HALF, DataType.FLOAT, DataType.FLOAT,
                            outputs, depth, 1.0, 0.0);
    }

    /** More outputs than the launch's blocks hold: both kernels loop over their grid. */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void moreOutputsThanTheGrid(Nd4jBackend backend) {
        for (Layout layout : new Layout[]{Layout.C, Layout.F})
            checkProduct(Product.ROW, layout, DataType.BFLOAT16, DataType.FLOAT, DataType.FLOAT,
                    40_000, 3, 1.0, 0.25);
    }

    /**
     * One storage type for matrix and vector into an output of another: cuBLAS takes HALF into
     * FLOAT on CUDA, the mixed GEMV everything on CPU.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void sharedStorageIntoAnotherOutputType(Nd4jBackend backend) {
        DataType[][] cases = {
                {DataType.HALF, DataType.FLOAT}, {DataType.BFLOAT16, DataType.FLOAT},
                {DataType.FLOAT, DataType.HALF}, {DataType.DOUBLE, DataType.FLOAT}};
        for (DataType[] c : cases)
            for (Product product : Product.values())
                checkProduct(product, Layout.C, c[0], c[0], c[1], 37, 45, 0.75, 0.5);
    }

    /** Integer storage against floats computes in FLOAT; small integers keep the result exact. */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void integerMatrixAgainstFloatVector(Nd4jBackend backend) {
        int[][] matrixValues = new int[12][7];
        for (int r = 0; r < 12; r++)
            for (int k = 0; k < 7; k++)
                matrixValues[r][k] = (r * 7 + k) % 13 - 6;
        for (Product product : Product.values()) {
            INDArray matrix = Nd4j.createFromArray(matrixValues);
            INDArray vector = Nd4j.createFromArray(-3f, -2f, -1f, 0f, 1f, 2f, 3f);
            INDArray expected;
            INDArray actual;
            switch (product) {
                case ROW:
                    expected = vector.reshape(1, 7).mmul(matrix.castTo(DataType.FLOAT).transpose());
                    actual = matmul(vector.reshape(1, 7), matrix.transpose().dup('c'), Nd4j.create(DataType.FLOAT, 1, 12), 1, 0);
                    break;
                case COLUMN:
                    expected = matrix.castTo(DataType.FLOAT).mmul(vector.reshape(7, 1));
                    actual = matmul(matrix, vector.reshape(7, 1), Nd4j.create(DataType.FLOAT, 12, 1), 1, 0);
                    break;
                default:
                    expected = matrix.castTo(DataType.FLOAT).mmul(vector.reshape(7, 1)).reshape(12);
                    actual = matmul(matrix, vector, Nd4j.create(DataType.FLOAT, 12), 1, 0);
            }
            assertEquals(expected, actual, product.name());
        }
    }

    /** An output the op allocates has the wider input type, the first on a tie, as Java's Mmul. */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void allocatedOutputHasTheWiderInputType(Nd4jBackend backend) {
        DataType[][] cases = {
                {DataType.FLOAT, DataType.BFLOAT16, DataType.FLOAT},
                {DataType.BFLOAT16, DataType.FLOAT, DataType.FLOAT},
                {DataType.HALF, DataType.DOUBLE, DataType.DOUBLE},
                {DataType.HALF, DataType.BFLOAT16, DataType.HALF},
                {DataType.BFLOAT16, DataType.HALF, DataType.BFLOAT16}};
        for (DataType[] c : cases) {
            INDArray x = Nd4j.ones(c[0], 1, 8);
            INDArray y = Nd4j.ones(c[1], 8, 5);
            INDArray[] out = Nd4j.exec(DynamicCustomOp.builder("matmul").addInputs(x, y).build());
            assertEquals(c[2], out[0].dataType(), c[0] + " x " + c[1]);
            assertEquals(Nd4j.valueArrayOf(new long[]{1, 5}, 8.0, c[2]), out[0], c[0] + " x " + c[1]);
        }
    }

    private static void checkProduct(Product product, Layout layout, DataType matrixType, DataType vectorType,
                                     DataType outputType, long outputs, long depth, double alpha, double beta) {
        String label = product + " " + layout + " " + matrixType + " x " + vectorType + " -> " + outputType
                + " (" + outputs + " outputs, depth " + depth + ")";
        long seed = 31L * outputs + depth + 7L * product.ordinal() + layout.ordinal();
        boolean strided = layout == Layout.STRIDED;
        INDArray matrix;
        INDArray vector;
        INDArray output;
        switch (product) {
            case ROW:
                matrix = matrix(matrixType, depth, outputs, layout, seed);
                vector = rowVector(vectorType, depth, strided, seed + 1);
                output = rowVector(outputType, outputs, strided, seed + 2);
                break;
            case COLUMN:
                matrix = matrix(matrixType, outputs, depth, layout, seed);
                vector = columnVector(vectorType, depth, strided, seed + 1);
                output = columnVector(outputType, outputs, strided, seed + 2);
                break;
            default:
                matrix = matrix(matrixType, outputs, depth, layout, seed);
                vector = flatVector(vectorType, depth, strided, seed + 1);
                output = flatVector(outputType, outputs, strided, seed + 2);
        }
        INDArray before = output.castTo(DataType.DOUBLE).dup();
        INDArray w = matrix.castTo(DataType.DOUBLE);
        INDArray x = vector.castTo(DataType.DOUBLE);

        INDArray product64;
        INDArray magnitude;
        switch (product) {
            case ROW:
                product64 = x.mmul(w);
                magnitude = Transforms.abs(x, true).mmul(Transforms.abs(w, true));
                break;
            case COLUMN:
                product64 = w.mmul(x);
                magnitude = Transforms.abs(w, true).mmul(Transforms.abs(x, true));
                break;
            default:
                product64 = w.mmul(x.reshape(depth, 1)).reshape(outputs);
                magnitude = Transforms.abs(w, true).mmul(Transforms.abs(x, true).reshape(depth, 1)).reshape(outputs);
        }
        double[] expected = product64.mul(alpha).addi(before.mul(beta)).toDoubleVector();
        double[] bound = magnitude.mul(Math.abs(alpha)).addi(Transforms.abs(before, true).muli(Math.abs(beta)))
                .toDoubleVector();

        matmul(vector, matrix, output, product, alpha, beta);
        assertEquals(outputType, output.dataType(), label);
        double[] actual = output.castTo(DataType.DOUBLE).toDoubleVector();

        boolean doubleSums = matrixType == DataType.DOUBLE || vectorType == DataType.DOUBLE
                || outputType == DataType.DOUBLE;
        double accumulation = (depth + 3) * roundoff(doubleSums ? DataType.DOUBLE : DataType.FLOAT);
        double rounding = roundoff(outputType);
        for (int i = 0; i < expected.length; i++) {
            double tolerance = bound[i] * (accumulation + rounding) + Double.MIN_NORMAL;
            assertTrue(Math.abs(actual[i] - expected[i]) <= tolerance,
                    label + ": output " + i + " is " + actual[i] + ", expected " + expected[i] + " +- " + tolerance);
        }
    }

    private static void matmul(INDArray vector, INDArray matrix, INDArray output, Product product,
                               double alpha, double beta) {
        if (product == Product.ROW)
            matmul(vector, matrix, output, alpha, beta);
        else
            matmul(matrix, vector, output, alpha, beta);
    }

    private static INDArray matmul(INDArray x, INDArray y, INDArray z, double alpha, double beta) {
        Nd4j.exec(DynamicCustomOp.builder("matmul")
                .addInputs(x, y)
                .addOutputs(z)
                .addFloatingPointArguments(alpha, beta)
                .addIntegerArguments(0, 0, 0)
                .build());
        return z;
    }

    /** Values in [-1, 1) of type, generated from seed. */
    private static INDArray uniform(DataType type, long seed, long... shape) {
        Nd4j.getRandom().setSeed(seed);
        return Nd4j.rand(DataType.DOUBLE, shape).muli(2).subi(1).castTo(type);
    }

    /** A [rows, cols] matrix laid out as layout. */
    private static INDArray matrix(DataType type, long rows, long cols, Layout layout, long seed) {
        switch (layout) {
            case C:
                return uniform(type, seed, rows, cols);
            case F:
                return uniform(type, seed, rows, cols).dup('f');
            case TRANSPOSED:
                return uniform(type, seed, cols, rows).transpose();
            default:
                return uniform(type, seed, rows + 3, 2 * cols + 1)
                        .get(NDArrayIndex.interval(2, rows + 2), NDArrayIndex.interval(1, 2, 2 * cols + 1));
        }
    }

    /** A [1, length] vector, contiguous or every other element of a wider row. */
    private static INDArray rowVector(DataType type, long length, boolean strided, long seed) {
        if (!strided) return uniform(type, seed, 1, length);
        return uniform(type, seed, 3, 2 * length + 1)
                .get(NDArrayIndex.interval(1, 2), NDArrayIndex.interval(1, 2, 2 * length + 1));
    }

    /** A [length, 1] vector, contiguous or every other element of a taller column. */
    private static INDArray columnVector(DataType type, long length, boolean strided, long seed) {
        if (!strided) return uniform(type, seed, length, 1);
        return uniform(type, seed, 2 * length + 1, 3)
                .get(NDArrayIndex.interval(1, 2, 2 * length + 1), NDArrayIndex.interval(1, 2));
    }

    /** A [length] vector, contiguous or every other element of a longer one. */
    private static INDArray flatVector(DataType type, long length, boolean strided, long seed) {
        if (!strided) return uniform(type, seed, length);
        return uniform(type, seed, 2 * length + 1).get(NDArrayIndex.interval(1, 2, 2 * length + 1));
    }

    /** The unit roundoff of a float type: half the distance from 1 to the next value. */
    private static double roundoff(DataType type) {
        switch (type) {
            case HALF:
                return Math.pow(2, -11);
            case BFLOAT16:
                return Math.pow(2, -8);
            case FLOAT:
                return Math.pow(2, -24);
            default:
                return Math.pow(2, -53);
        }
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
