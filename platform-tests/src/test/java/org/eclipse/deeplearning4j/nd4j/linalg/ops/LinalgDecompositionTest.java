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

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.function.ToDoubleFunction;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The decomposition ops (logdet, cholesky, matrix_inverse, matrix_determinant, log_matrix_determinant) against
 * double-precision references, for single matrices and batches of rank 3 and 4, C and F order, sizes beyond a warp
 * (CUDA kernels launched a thread per element of a 32 x 32 block, or computed rows in parallel that depend on each
 * other) and matrices whose LU factorization needs row exchanges (the CUDA inverse and determinants factorized
 * without pivoting). Every op leaves its input as it was.
 */
@NativeTag
@Tag(TagNames.CUSTOM_FUNCTIONALITY)
public class LinalgDecompositionTest extends BaseNd4jTestWithBackends {

    private static INDArray[] exec(DynamicCustomOp op) {
        INDArray[] out = Nd4j.exec(op);
        Nd4j.getExecutioner().commit();
        return out;
    }

    /** The values in a C-order array of the given shape; an empty shape is a scalar. */
    private static INDArray shaped(double[] values, long[] shape) {
        return shape.length == 0 ? Nd4j.scalar(values[0]) : Nd4j.createFromArray(values).reshape('c', shape);
    }

    /** A symmetric positive definite n x n matrix: B * B^T + n * I. */
    private static double[][] spd(int n, Random rng) {
        double[][] bm = new double[n][n];
        for (double[] row : bm)
            for (int j = 0; j < n; j++)
                row[j] = rng.nextDouble() - 0.5;
        double[][] a = new double[n][n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) {
                double s = i == j ? n : 0;
                for (int k = 0; k < n; k++)
                    s += bm[i][k] * bm[j][k];
                a[i][j] = s;
            }
        return a;
    }

    /** An n x n matrix with random entries in [-0.5, 0.5) plus 2 on the diagonal. */
    private static double[][] general(int n, Random rng) {
        double[][] a = new double[n][n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                a[i][j] = rng.nextDouble() - 0.5 + (i == j ? 2 : 0);
        return a;
    }

    /** log det A = 2 * sum log L_ii for A's Cholesky factor L. */
    private static double logdetReference(double[][] a) {
        int n = a.length;
        double[][] l = new double[n][n];
        double logdet = 0;
        for (int j = 0; j < n; j++) {
            double d = a[j][j];
            for (int k = 0; k < j; k++)
                d -= l[j][k] * l[j][k];
            l[j][j] = Math.sqrt(d);
            logdet += 2 * Math.log(l[j][j]);
            for (int i = j + 1; i < n; i++) {
                double s = a[i][j];
                for (int k = 0; k < j; k++)
                    s -= l[i][k] * l[j][k];
                l[i][j] = s / l[j][j];
            }
        }
        return logdet;
    }

    /** det A by Gaussian elimination with partial pivoting. */
    private static double determinantReference(double[][] matrix) {
        int n = matrix.length;
        double[][] a = new double[n][];
        for (int i = 0; i < n; i++)
            a[i] = matrix[i].clone();
        double det = 1;
        for (int c = 0; c < n; c++) {
            int p = c;
            for (int r = c + 1; r < n; r++)
                if (Math.abs(a[r][c]) > Math.abs(a[p][c]))
                    p = r;
            if (a[p][c] == 0)
                return 0;
            if (p != c) {
                double[] t = a[p];
                a[p] = a[c];
                a[c] = t;
                det = -det;
            }
            det *= a[c][c];
            for (int r = c + 1; r < n; r++) {
                double f = a[r][c] / a[c][c];
                for (int k = c; k < n; k++)
                    a[r][k] -= f * a[c][k];
            }
        }
        return det;
    }

    /** A batch of matrices of the given batch shape, flattened in C order, with a reference value per matrix. */
    private static final class Batch {
        final long[] shape;
        final double[] flat;
        final List<double[][]> matrices = new ArrayList<>();

        Batch(long[] batchShape, int n, Random rng, boolean spd) {
            long count = 1;
            for (long d : batchShape)
                count *= d;
            shape = Arrays.copyOf(batchShape, batchShape.length + 2);
            shape[batchShape.length] = n;
            shape[batchShape.length + 1] = n;
            flat = new double[(int) (count * n * n)];
            for (int m = 0; m < count; m++) {
                double[][] a = spd ? spd(n, rng) : general(n, rng);
                matrices.add(a);
                for (int i = 0; i < n; i++)
                    System.arraycopy(a[i], 0, flat, (m * n + i) * n, n);
            }
        }

        INDArray array(char order) {
            return Nd4j.createFromArray(flat).reshape('c', shape).dup(order);
        }

        long[] batchShape() {
            return Arrays.copyOf(shape, shape.length - 2);
        }
    }

    private static final long[][] BATCH_SHAPES = {{}, {3}, {2, 3}};

    /** Runs a reduction of each matrix to a value (logdet, determinants) over batches and both orders. */
    private static void checkMatrixValues(List<String> failures, String opName, boolean spd, int[] sizes,
                                          ToDoubleFunction<double[][]> reference, Random rng) {
        for (long[] batchShape : BATCH_SHAPES) {
            for (int n : sizes) {
                Batch batch = new Batch(batchShape, n, rng, spd);
                double[] expected = batch.matrices.stream().mapToDouble(reference).toArray();
                INDArray expectedArr = shaped(expected, batch.batchShape());
                for (char order : new char[]{'c', 'f'}) {
                    INDArray x = batch.array(order);
                    INDArray before = x.dup();
                    String what = opName + " " + Arrays.toString(batch.shape) + " order " + order;
                    try {
                        INDArray[] out = exec(DynamicCustomOp.builder(opName).addInputs(x).build());
                        if (!before.equalsWithEps(x, 0.0))
                            failures.add(what + ": the input changed");
                        // in C order: getDouble(i) reads an F-ordered array (the output of an F-ordered input) in F order
                        INDArray actual = out[0].castTo(DataType.DOUBLE).dup('c');
                        if (!Arrays.equals(expectedArr.shape(), actual.shape()))
                            failures.add(what + ": shape " + Arrays.toString(actual.shape()));
                        for (int i = 0; i < expected.length; i++) {
                            double e = expected[i], a = actual.getDouble(i);
                            if (!(Math.abs(a - e) <= 1e-9 * Math.max(1, Math.abs(e))))
                                failures.add(what + ", matrix " + i + ": " + a + " vs " + e);
                        }
                    } catch (RuntimeException e) {
                        failures.add(what + ": " + e);
                    }
                }
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void logdetOfBatchesOfAnySize(Nd4jBackend backend) {
        // n = 40: beyond the 32 rows a block of the CUDA Cholesky's result kernel had a thread per element for
        List<String> failures = new ArrayList<>();
        checkMatrixValues(failures, "logdet", true, new int[]{3, 40}, LinalgDecompositionTest::logdetReference,
                new Random(17));
        assertTrue(failures.isEmpty(), failures.size() + " failures:\n" + String.join("\n", failures));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void determinantsOfBatchesOfAnySize(Nd4jBackend backend) {
        List<String> failures = new ArrayList<>();
        checkMatrixValues(failures, "matrix_determinant", false, new int[]{3, 33},
                LinalgDecompositionTest::determinantReference, new Random(37));
        checkMatrixValues(failures, "log_matrix_determinant", false, new int[]{3, 33},
                a -> Math.log(Math.abs(determinantReference(a))), new Random(41));
        assertTrue(failures.isEmpty(), failures.size() + " failures:\n" + String.join("\n", failures));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void determinantsNeedingRowExchanges(Nd4jBackend backend) {
        // without pivoting the first two have a zero pivot; the signs follow the row exchanges
        double[][][] matrices = {{{0, 1}, {1, 0}}, {{0, 2, 1}, {1, 0, 0}, {0, 1, 3}}, {{1, 2}, {3, 4}},
                {{0, 0, 1}, {0, 1, 0}, {1, 0, 0}}};
        for (double[][] a : matrices) {
            double expected = determinantReference(a);
            for (DataType dtype : new DataType[]{DataType.DOUBLE, DataType.FLOAT, DataType.HALF}) {
                INDArray x = Nd4j.createFromArray(a).castTo(dtype);
                double tol = dtype == DataType.HALF ? 1e-2 : 1e-5;
                INDArray[] det = exec(DynamicCustomOp.builder("matrix_determinant").addInputs(x).build());
                assertEquals(expected, det[0].getDouble(0), tol * Math.max(1, Math.abs(expected)),
                        "matrix_determinant " + Arrays.deepToString(a) + " " + dtype);
                INDArray[] logDet = exec(DynamicCustomOp.builder("log_matrix_determinant").addInputs(x).build());
                assertEquals(Math.log(Math.abs(expected)), logDet[0].getDouble(0), tol,
                        "log_matrix_determinant " + Arrays.deepToString(a) + " " + dtype);
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void logDeterminantOfASingularMatrixIsMinusInfinity(Nd4jBackend backend) {
        INDArray x = Nd4j.createFromArray(new double[][]{{1, 2}, {2, 4}});
        INDArray[] logDet = exec(DynamicCustomOp.builder("log_matrix_determinant").addInputs(x).build());
        assertEquals(Double.NEGATIVE_INFINITY, logDet[0].getDouble(0), 0.0);
        INDArray[] det = exec(DynamicCustomOp.builder("matrix_determinant").addInputs(x).build());
        assertEquals(0.0, det[0].getDouble(0), 1e-12);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void choleskyFactorsMatricesOfAnySize(Nd4jBackend backend) {
        Random rng = new Random(19);
        for (int n : new int[]{3, 33, 40}) {
            double[][] a = spd(n, rng);
            INDArray x = Nd4j.createFromArray(a);
            INDArray before = x.dup();
            INDArray[] out = exec(DynamicCustomOp.builder("cholesky").addInputs(x).build());
            String what = "cholesky " + n + "x" + n;
            assertTrue(before.equalsWithEps(x, 0.0), what + ": the input changed");
            INDArray l = out[0];
            for (int i = 0; i < n; i++)
                for (int j = i + 1; j < n; j++)
                    assertEquals(0.0, l.getDouble(i, j), 0.0, what + ": L[" + i + ", " + j + "] above the diagonal");
            double maxDiff = x.sub(l.mmul(l.transpose())).amaxNumber().doubleValue();
            assertTrue(maxDiff <= 1e-9 * n, what + ": max |A - L * L^T| " + maxDiff);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void matrixInverseOfMatricesOfAnySize(Nd4jBackend backend) {
        Random rng = new Random(23);
        List<double[][]> matrices = new ArrayList<>();
        // matrices whose LU factorization needs row exchanges, and one a determinant threshold called singular
        matrices.add(new double[][]{{0, 1}, {1, 0}});
        matrices.add(new double[][]{{0, 2, 1}, {1, 0, 0}, {0, 1, 3}});
        double[][] small = new double[8][8];
        for (int i = 0; i < 8; i++)
            small[i][i] = 0.1;
        matrices.add(small);
        for (int n : new int[]{3, 8, 33, 64})
            matrices.add(general(n, rng));
        List<String> failures = new ArrayList<>();
        for (double[][] a : matrices) {
            int n = a.length;
            for (char order : new char[]{'c', 'f'}) {
                INDArray x = Nd4j.createFromArray(a).dup(order);
                INDArray before = x.dup();
                String what = "matrix_inverse " + n + "x" + n + " order " + order;
                try {
                    INDArray[] out = exec(DynamicCustomOp.builder("matrix_inverse").addInputs(x).build());
                    if (!before.equalsWithEps(x, 0.0))
                        failures.add(what + ": the input changed");
                    double maxDiff = x.mmul(out[0]).sub(Nd4j.eye(n).castTo(DataType.DOUBLE)).amaxNumber().doubleValue();
                    if (!(maxDiff <= 1e-9 * n))
                        failures.add(what + ": max |A * inverse(A) - I| " + maxDiff);
                } catch (RuntimeException e) {
                    failures.add(what + ": " + e);
                }
            }
        }
        // a batch of rank 4
        Batch batch = new Batch(new long[]{2, 3}, 5, rng, false);
        INDArray x = batch.array('c');
        INDArray[] out = exec(DynamicCustomOp.builder("matrix_inverse").addInputs(x).build());
        INDArray product = Nd4j.linalg().matmul(x, out[0]);
        INDArray identities = Nd4j.tile(Nd4j.eye(5).castTo(DataType.DOUBLE).reshape(1, 1, 5, 5), 2, 3, 1, 1);
        double maxDiff = product.sub(identities).amaxNumber().doubleValue();
        if (!(maxDiff <= 1e-9 * 5))
            failures.add("matrix_inverse of a [2, 3, 5, 5] batch: max |A * inverse(A) - I| " + maxDiff);
        assertTrue(failures.isEmpty(), failures.size() + " failures:\n" + String.join("\n", failures));
    }
}
