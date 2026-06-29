/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.eclipse.deeplearning4j.nd4j.linalg.sparse;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.blas.SparseLevel2;
import org.nd4j.linalg.api.blas.SparseLevel3;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ndarray.SparseFormat;
import org.nd4j.linalg.api.ndarray.SparseNDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertSame;

/**
 * Tests for the sparse BLAS API surface exposed via
 * {@link org.nd4j.linalg.factory.BlasWrapper#sparseLevel2()} and
 * {@link org.nd4j.linalg.factory.BlasWrapper#sparseLevel3()}.
 *
 * <p>Strategy: for each shape / dtype combination, construct a dense matrix A
 * (with an explicit mix of zeros and non-zeros), convert it to CSR via
 * {@link Nd4j#toSparse}, call the BLAS-surface method, and compare the result
 * to the reference dense matrix multiply.  Both FLOAT and DOUBLE dtypes are
 * exercised, and transpose variants are included.
 */
@DisplayName("Sparse BLAS API Surface Tests")
public class SparseBlasSurfaceTest {

    private static final double TOL_FLOAT  = 1e-4;
    private static final double TOL_DOUBLE = 1e-9;

    // -----------------------------------------------------------------------
    // Utility
    // -----------------------------------------------------------------------

    /** Assert element-wise equality within tolerance. */
    private static void assertClose(INDArray expected, INDArray actual, double tol, String msg) {
        assertEquals(expected.length(), actual.length(), msg + ": length mismatch");
        for (long i = 0; i < expected.length(); i++) {
            double e = expected.getDouble(i);
            double a = actual.getDouble(i);
            assertEquals(e, a, tol, msg + ": mismatch at linear index " + i);
        }
    }

    /**
     * Build a small dense matrix with an explicit sparsity pattern (every other
     * element is zero) so the CSR representation is non-trivial.
     */
    private static INDArray makeSparseMatrix(int rows, int cols, DataType dtype) {
        int n = rows * cols;
        double[] data = new double[n];
        int v = 1;
        for (int i = 0; i < n; i++) {
            data[i] = (i % 2 == 0) ? v++ : 0.0;
        }
        return Nd4j.create(data, new long[]{rows, cols}).castTo(dtype);
    }

    // -----------------------------------------------------------------------
    // BlasWrapper accessor tests
    // -----------------------------------------------------------------------

    @Test
    @DisplayName("sparseLevel2() returns non-null SparseLevel2 singleton")
    public void testSparseLevel2Accessor() {
        SparseLevel2 sl2 = Nd4j.getBlasWrapper().sparseLevel2();
        assertNotNull(sl2, "sparseLevel2() must not return null");
        // calling it twice should return the same singleton
        assertSame(sl2, Nd4j.getBlasWrapper().sparseLevel2(), "sparseLevel2() should return the same singleton");
    }

    @Test
    @DisplayName("sparseLevel3() returns non-null SparseLevel3 singleton")
    public void testSparseLevel3Accessor() {
        SparseLevel3 sl3 = Nd4j.getBlasWrapper().sparseLevel3();
        assertNotNull(sl3, "sparseLevel3() must not return null");
        assertSame(sl3, Nd4j.getBlasWrapper().sparseLevel3(), "sparseLevel3() should return the same singleton");
    }

    // -----------------------------------------------------------------------
    // SparseLevel3.spmm — FLOAT
    // -----------------------------------------------------------------------

    @Test
    @DisplayName("spmm 3x4 * 4x2 FLOAT ≈ A.mmul(B)")
    public void testSpmmFloat3x4x2() {
        INDArray A = makeSparseMatrix(3, 4, DataType.FLOAT);
        INDArray B = Nd4j.rand(DataType.FLOAT, 4, 2);
        INDArray expected = A.mmul(B);

        SparseNDArray csr = Nd4j.toSparse(A, SparseFormat.CSR);
        INDArray actual = Nd4j.getBlasWrapper().sparseLevel3().spmm(csr, B);
        assertClose(expected, actual, TOL_FLOAT, "spmm 3x4*4x2 FLOAT");
    }

    @Test
    @DisplayName("spmm 5x5 * 5x3 FLOAT ≈ A.mmul(B)")
    public void testSpmmFloat5x5x3() {
        INDArray A = makeSparseMatrix(5, 5, DataType.FLOAT);
        INDArray B = Nd4j.rand(DataType.FLOAT, 5, 3);
        INDArray expected = A.mmul(B);

        SparseNDArray csr = Nd4j.toSparse(A, SparseFormat.CSR);
        INDArray actual = Nd4j.getBlasWrapper().sparseLevel3().spmm(csr, B);
        assertClose(expected, actual, TOL_FLOAT, "spmm 5x5*5x3 FLOAT");
    }

    @Test
    @DisplayName("spmm convenience overload (transposeA=false default) FLOAT")
    public void testSpmmConvenienceFloat() {
        INDArray A = makeSparseMatrix(4, 3, DataType.FLOAT);
        INDArray B = Nd4j.rand(DataType.FLOAT, 3, 2);
        INDArray expected = A.mmul(B);

        SparseNDArray csr = Nd4j.toSparse(A, SparseFormat.CSR);
        // The no-transposeA convenience overload
        INDArray actual = Nd4j.getBlasWrapper().sparseLevel3().spmm(csr, B);
        assertClose(expected, actual, TOL_FLOAT, "spmm convenience 4x3*3x2 FLOAT");
    }

    // -----------------------------------------------------------------------
    // SparseLevel3.spmm — DOUBLE
    // -----------------------------------------------------------------------

    @Test
    @DisplayName("spmm 4x4 * 4x3 DOUBLE ≈ A.mmul(B)")
    public void testSpmmDouble4x4x3() {
        INDArray A = makeSparseMatrix(4, 4, DataType.DOUBLE);
        INDArray B = Nd4j.rand(DataType.DOUBLE, 4, 3);
        INDArray expected = A.mmul(B);

        SparseNDArray csr = Nd4j.toSparse(A, SparseFormat.CSR);
        INDArray actual = Nd4j.getBlasWrapper().sparseLevel3().spmm(csr, B);
        assertClose(expected, actual, TOL_DOUBLE, "spmm 4x4*4x3 DOUBLE");
    }

    @Test
    @DisplayName("spmm tall 8x2 * 2x3 DOUBLE ≈ A.mmul(B)")
    public void testSpmmDoubleTall8x2x3() {
        INDArray A = makeSparseMatrix(8, 2, DataType.DOUBLE);
        INDArray B = Nd4j.rand(DataType.DOUBLE, 2, 3);
        INDArray expected = A.mmul(B);

        SparseNDArray csr = Nd4j.toSparse(A, SparseFormat.CSR);
        INDArray actual = Nd4j.getBlasWrapper().sparseLevel3().spmm(csr, B);
        assertClose(expected, actual, TOL_DOUBLE, "spmm tall 8x2*2x3 DOUBLE");
    }

    // -----------------------------------------------------------------------
    // SparseLevel3.spmm — transpose=true
    // -----------------------------------------------------------------------

    @Test
    @DisplayName("spmm transposeA=true: Aᵀ·B ≈ A.transpose().mmul(B) FLOAT")
    public void testSpmmTransposeFloat() {
        // A is [m, k]; with transposeA=true, B must be [m, n]
        int m = 3, k = 4, n = 2;
        INDArray A = makeSparseMatrix(m, k, DataType.FLOAT);
        INDArray B = Nd4j.rand(DataType.FLOAT, m, n);   // [m, n]
        INDArray expected = A.transpose().mmul(B);       // [k, n]

        SparseNDArray csr = Nd4j.toSparse(A, SparseFormat.CSR);
        INDArray actual = Nd4j.getBlasWrapper().sparseLevel3().spmm(csr, B, true);
        assertClose(expected, actual, TOL_FLOAT, "spmm transposeA=true [3x4]ᵀ*[3x2] FLOAT");
    }

    @Test
    @DisplayName("spmm transposeA=true DOUBLE")
    public void testSpmmTransposeDouble() {
        int m = 4, k = 3, n = 2;
        INDArray A = makeSparseMatrix(m, k, DataType.DOUBLE);
        INDArray B = Nd4j.rand(DataType.DOUBLE, m, n);
        INDArray expected = A.transpose().mmul(B);

        SparseNDArray csr = Nd4j.toSparse(A, SparseFormat.CSR);
        INDArray actual = Nd4j.getBlasWrapper().sparseLevel3().spmm(csr, B, true);
        assertClose(expected, actual, TOL_DOUBLE, "spmm transposeA=true [4x3]ᵀ*[4x2] DOUBLE");
    }

    // -----------------------------------------------------------------------
    // SparseLevel2.spmv — FLOAT
    // -----------------------------------------------------------------------

    @Test
    @DisplayName("spmv 3x4 · x[4] FLOAT ≈ A·x")
    public void testSpmvFloat3x4() {
        int rows = 3, cols = 4;
        INDArray A = makeSparseMatrix(rows, cols, DataType.FLOAT);
        INDArray x = Nd4j.rand(DataType.FLOAT, cols);
        INDArray expected = A.mmul(x.reshape(cols, 1)).reshape(rows);

        SparseNDArray csr = Nd4j.toSparse(A, SparseFormat.CSR);
        INDArray actual = Nd4j.getBlasWrapper().sparseLevel2().spmv(csr, x);
        assertClose(expected, actual, TOL_FLOAT, "spmv 3x4*[4] FLOAT");
    }

    @Test
    @DisplayName("spmv 5x5 · x[5] FLOAT ≈ A·x")
    public void testSpmvFloat5x5() {
        int rows = 5, cols = 5;
        INDArray A = makeSparseMatrix(rows, cols, DataType.FLOAT);
        INDArray x = Nd4j.rand(DataType.FLOAT, cols);
        INDArray expected = A.mmul(x.reshape(cols, 1)).reshape(rows);

        SparseNDArray csr = Nd4j.toSparse(A, SparseFormat.CSR);
        INDArray actual = Nd4j.getBlasWrapper().sparseLevel2().spmv(csr, x);
        assertClose(expected, actual, TOL_FLOAT, "spmv 5x5*[5] FLOAT");
    }

    @Test
    @DisplayName("spmv convenience overload (transposeA=false default) FLOAT")
    public void testSpmvConvenienceFloat() {
        int rows = 4, cols = 3;
        INDArray A = makeSparseMatrix(rows, cols, DataType.FLOAT);
        INDArray x = Nd4j.rand(DataType.FLOAT, cols);
        INDArray expected = A.mmul(x.reshape(cols, 1)).reshape(rows);

        SparseNDArray csr = Nd4j.toSparse(A, SparseFormat.CSR);
        INDArray actual = Nd4j.getBlasWrapper().sparseLevel2().spmv(csr, x);
        assertClose(expected, actual, TOL_FLOAT, "spmv convenience 4x3*[3] FLOAT");
    }

    // -----------------------------------------------------------------------
    // SparseLevel2.spmv — DOUBLE
    // -----------------------------------------------------------------------

    @Test
    @DisplayName("spmv 4x4 · x[4] DOUBLE ≈ A·x")
    public void testSpmvDouble4x4() {
        int rows = 4, cols = 4;
        INDArray A = makeSparseMatrix(rows, cols, DataType.DOUBLE);
        INDArray x = Nd4j.rand(DataType.DOUBLE, cols);
        INDArray expected = A.mmul(x.reshape(cols, 1)).reshape(rows);

        SparseNDArray csr = Nd4j.toSparse(A, SparseFormat.CSR);
        INDArray actual = Nd4j.getBlasWrapper().sparseLevel2().spmv(csr, x);
        assertClose(expected, actual, TOL_DOUBLE, "spmv 4x4*[4] DOUBLE");
    }

    @Test
    @DisplayName("spmv tall 6x2 · x[2] DOUBLE ≈ A·x")
    public void testSpmvDoubleTall6x2() {
        int rows = 6, cols = 2;
        INDArray A = makeSparseMatrix(rows, cols, DataType.DOUBLE);
        INDArray x = Nd4j.rand(DataType.DOUBLE, cols);
        INDArray expected = A.mmul(x.reshape(cols, 1)).reshape(rows);

        SparseNDArray csr = Nd4j.toSparse(A, SparseFormat.CSR);
        INDArray actual = Nd4j.getBlasWrapper().sparseLevel2().spmv(csr, x);
        assertClose(expected, actual, TOL_DOUBLE, "spmv tall 6x2*[2] DOUBLE");
    }

    // -----------------------------------------------------------------------
    // SparseLevel2.spmv — transpose=true
    // -----------------------------------------------------------------------

    @Test
    @DisplayName("spmv transposeA=true: Aᵀ·dy ≈ A.transpose().mmul(dy) FLOAT")
    public void testSpmvTransposeFloat() {
        // A is [rows, cols]; Aᵀ·dy where dy is length rows → result is length cols
        int rows = 3, cols = 4;
        INDArray A = makeSparseMatrix(rows, cols, DataType.FLOAT);
        INDArray dy = Nd4j.rand(DataType.FLOAT, rows);
        INDArray expected = A.transpose().mmul(dy.reshape(rows, 1)).reshape(cols);

        SparseNDArray csr = Nd4j.toSparse(A, SparseFormat.CSR);
        INDArray actual = Nd4j.getBlasWrapper().sparseLevel2().spmv(csr, dy, true);
        assertClose(expected, actual, TOL_FLOAT, "spmv transposeA=true [3x4]ᵀ*dy[3] FLOAT");
    }

    @Test
    @DisplayName("spmv transposeA=true DOUBLE")
    public void testSpmvTransposeDouble() {
        int rows = 4, cols = 3;
        INDArray A = makeSparseMatrix(rows, cols, DataType.DOUBLE);
        INDArray dy = Nd4j.rand(DataType.DOUBLE, rows);
        INDArray expected = A.transpose().mmul(dy.reshape(rows, 1)).reshape(cols);

        SparseNDArray csr = Nd4j.toSparse(A, SparseFormat.CSR);
        INDArray actual = Nd4j.getBlasWrapper().sparseLevel2().spmv(csr, dy, true);
        assertClose(expected, actual, TOL_DOUBLE, "spmv transposeA=true [4x3]ᵀ*dy[4] DOUBLE");
    }
}
