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
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ndarray.SparseFormat;
import org.nd4j.linalg.api.ndarray.SparseNDArray;
import org.nd4j.linalg.api.ops.impl.sparse.CsrSpmm;
import org.nd4j.linalg.api.ops.impl.sparse.CsrSpmv;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Forward-correctness tests for CSR SpMV (csr_spmv), CSR SpMM (csr_spmm), and their
 * SparseNDArray ergonomic wrappers {@code mv()} and {@code mmul()}.
 *
 * <p>Strategy: for each shape/dtype combination, build a dense matrix A with an explicit
 * mix of zeros and non-zeros, convert it to CSR via {@link Nd4j#toSparse}, then compare
 * the sparse result to the reference dense matrix multiply.
 */
@DisplayName("CSR SpMV / SpMM Forward Correctness Tests")
public class SparseSpmvSpmmTest {

    private static final double TOL_FLOAT  = 1e-4;
    private static final double TOL_DOUBLE = 1e-9;

    // -----------------------------------------------------------------------
    // Utility
    // -----------------------------------------------------------------------

    /**
     * Assert element-wise equality within tolerance.
     */
    private static void assertClose(INDArray expected, INDArray actual, double tol, String msg) {
        assertEquals(expected.length(), actual.length(), msg + ": length mismatch");
        for (long i = 0; i < expected.length(); i++) {
            double e = expected.getDouble(i);
            double a = actual.getDouble(i);
            assertEquals(e, a, tol, msg + ": mismatch at linear index " + i);
        }
    }

    /**
     * Build a small 2D dense matrix with an explicit sparsity pattern.
     * Every other element (linear index) is set to zero.
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
    // SpMM (csr_spmm) tests — via SparseNDArray.mmul()
    // -----------------------------------------------------------------------

    @Test
    @DisplayName("SpMM 3x4 * 4x2 FLOAT — matches dense mmul")
    public void testSpmmSquare3x4x2Float() {
        INDArray A = makeSparseMatrix(3, 4, DataType.FLOAT);
        INDArray B = Nd4j.rand(DataType.FLOAT, 4, 2);
        INDArray expected = A.mmul(B);
        SparseNDArray csr = Nd4j.toSparse(A, SparseFormat.CSR);
        INDArray actual = csr.mmul(B);
        assertClose(expected, actual, TOL_FLOAT, "SpMM 3x4*4x2 FLOAT");
    }

    @Test
    @DisplayName("SpMM 4x4 * 4x3 DOUBLE — matches dense mmul")
    public void testSpmmSquare4x4Double() {
        INDArray A = makeSparseMatrix(4, 4, DataType.DOUBLE);
        INDArray B = Nd4j.rand(DataType.DOUBLE, 4, 3);
        INDArray expected = A.mmul(B);
        SparseNDArray csr = Nd4j.toSparse(A, SparseFormat.CSR);
        INDArray actual = csr.mmul(B);
        assertClose(expected, actual, TOL_DOUBLE, "SpMM 4x4*4x3 DOUBLE");
    }

    @Test
    @DisplayName("SpMM tall 8x3 * 3x2 FLOAT — matches dense mmul")
    public void testSpmmTall8x3Float() {
        INDArray A = makeSparseMatrix(8, 3, DataType.FLOAT);
        INDArray B = Nd4j.rand(DataType.FLOAT, 3, 2);
        INDArray expected = A.mmul(B);
        SparseNDArray csr = Nd4j.toSparse(A, SparseFormat.CSR);
        INDArray actual = csr.mmul(B);
        assertClose(expected, actual, TOL_FLOAT, "SpMM tall 8x3*3x2 FLOAT");
    }

    @Test
    @DisplayName("SpMM wide 2x6 * 6x4 FLOAT — matches dense mmul")
    public void testSpmmWide2x6Float() {
        INDArray A = makeSparseMatrix(2, 6, DataType.FLOAT);
        INDArray B = Nd4j.rand(DataType.FLOAT, 6, 4);
        INDArray expected = A.mmul(B);
        SparseNDArray csr = Nd4j.toSparse(A, SparseFormat.CSR);
        INDArray actual = csr.mmul(B);
        assertClose(expected, actual, TOL_FLOAT, "SpMM wide 2x6*6x4 FLOAT");
    }

    @Test
    @DisplayName("SpMM 5x5 * 5x1 DOUBLE — matches dense mmul")
    public void testSpmmSingleColumn5x5Double() {
        INDArray A = makeSparseMatrix(5, 5, DataType.DOUBLE);
        INDArray B = Nd4j.rand(DataType.DOUBLE, 5, 1);
        INDArray expected = A.mmul(B);
        SparseNDArray csr = Nd4j.toSparse(A, SparseFormat.CSR);
        INDArray actual = csr.mmul(B);
        assertClose(expected, actual, TOL_DOUBLE, "SpMM 5x5*5x1 DOUBLE");
    }

    @Test
    @DisplayName("SpMM via CsrSpmm op directly (not SparseNDArray helper) FLOAT")
    public void testSpmmViaOpDirectlyFloat() {
        INDArray A = makeSparseMatrix(3, 3, DataType.FLOAT);
        INDArray B = Nd4j.rand(DataType.FLOAT, 3, 2);
        INDArray expected = A.mmul(B);

        SparseNDArray csr = Nd4j.toSparse(A, SparseFormat.CSR);
        CsrSpmm op = new CsrSpmm(csr.getValues(), csr.getColIdx(), csr.getRowPtr(), B,
                csr.rows(), csr.cols(), false);
        INDArray actual = Nd4j.exec(op)[0];
        assertClose(expected, actual, TOL_FLOAT, "SpMM direct op 3x3*3x2 FLOAT");
    }

    // -----------------------------------------------------------------------
    // SpMM transpose test
    // -----------------------------------------------------------------------

    @Test
    @DisplayName("SpMM transpose=true: Aᵀ·B matches A.transpose().mmul(B) FLOAT")
    public void testSpmmTransposeFloat() {
        // A is [m, k], Aᵀ is [k, m]; B must be [m, n] when transposeA=true
        int m = 3, k = 4, n = 2;
        INDArray A = makeSparseMatrix(m, k, DataType.FLOAT);
        // For transposeA=true: B shape is [rows=m, n]
        INDArray B = Nd4j.rand(DataType.FLOAT, m, n);
        INDArray expected = A.transpose().mmul(B);   // [k, n]

        SparseNDArray csr = Nd4j.toSparse(A, SparseFormat.CSR);
        CsrSpmm op = new CsrSpmm(csr.getValues(), csr.getColIdx(), csr.getRowPtr(), B,
                csr.rows(), csr.cols(), true);
        INDArray actual = Nd4j.exec(op)[0];
        assertClose(expected, actual, TOL_FLOAT, "SpMM transpose 3x4 FLOAT");
    }

    @Test
    @DisplayName("SpMM transpose=true DOUBLE")
    public void testSpmmTransposeDouble() {
        int m = 4, k = 3, n = 2;
        INDArray A = makeSparseMatrix(m, k, DataType.DOUBLE);
        INDArray B = Nd4j.rand(DataType.DOUBLE, m, n);
        INDArray expected = A.transpose().mmul(B);

        SparseNDArray csr = Nd4j.toSparse(A, SparseFormat.CSR);
        CsrSpmm op = new CsrSpmm(csr.getValues(), csr.getColIdx(), csr.getRowPtr(), B,
                csr.rows(), csr.cols(), true);
        INDArray actual = Nd4j.exec(op)[0];
        assertClose(expected, actual, TOL_DOUBLE, "SpMM transpose 4x3 DOUBLE");
    }

    // -----------------------------------------------------------------------
    // SpMV (csr_spmv) tests — via SparseNDArray.mv()
    // -----------------------------------------------------------------------

    @Test
    @DisplayName("SpMV 3x4 * x[4] FLOAT — matches dense mmul")
    public void testSpmvFloat() {
        int rows = 3, cols = 4;
        INDArray A = makeSparseMatrix(rows, cols, DataType.FLOAT);
        INDArray x = Nd4j.rand(DataType.FLOAT, cols);
        // Dense reference: A * x as column vector, then flatten
        INDArray expected = A.mmul(x.reshape(cols, 1)).reshape(rows);
        SparseNDArray csr = Nd4j.toSparse(A, SparseFormat.CSR);
        INDArray actual = csr.mv(x);
        assertClose(expected, actual, TOL_FLOAT, "SpMV 3x4*[4] FLOAT");
    }

    @Test
    @DisplayName("SpMV 4x4 * x[4] DOUBLE — matches dense mmul")
    public void testSpmvSquareDouble() {
        int rows = 4, cols = 4;
        INDArray A = makeSparseMatrix(rows, cols, DataType.DOUBLE);
        INDArray x = Nd4j.rand(DataType.DOUBLE, cols);
        INDArray expected = A.mmul(x.reshape(cols, 1)).reshape(rows);
        SparseNDArray csr = Nd4j.toSparse(A, SparseFormat.CSR);
        INDArray actual = csr.mv(x);
        assertClose(expected, actual, TOL_DOUBLE, "SpMV 4x4*[4] DOUBLE");
    }

    @Test
    @DisplayName("SpMV tall 6x2 * x[2] FLOAT — matches dense mmul")
    public void testSpmvTallFloat() {
        int rows = 6, cols = 2;
        INDArray A = makeSparseMatrix(rows, cols, DataType.FLOAT);
        INDArray x = Nd4j.rand(DataType.FLOAT, cols);
        INDArray expected = A.mmul(x.reshape(cols, 1)).reshape(rows);
        SparseNDArray csr = Nd4j.toSparse(A, SparseFormat.CSR);
        INDArray actual = csr.mv(x);
        assertClose(expected, actual, TOL_FLOAT, "SpMV tall 6x2 FLOAT");
    }

    @Test
    @DisplayName("SpMV via CsrSpmv op directly FLOAT")
    public void testSpmvViaOpDirectlyFloat() {
        int rows = 3, cols = 3;
        INDArray A = makeSparseMatrix(rows, cols, DataType.FLOAT);
        INDArray x = Nd4j.rand(DataType.FLOAT, cols);
        INDArray expected = A.mmul(x.reshape(cols, 1)).reshape(rows);

        SparseNDArray csr = Nd4j.toSparse(A, SparseFormat.CSR);
        CsrSpmv op = new CsrSpmv(csr.getValues(), csr.getColIdx(), csr.getRowPtr(), x,
                csr.rows(), csr.cols(), false);
        INDArray actual = Nd4j.exec(op)[0];
        assertClose(expected, actual, TOL_FLOAT, "SpMV direct op 3x3 FLOAT");
    }

    @Test
    @DisplayName("SpMV transpose=true: Aᵀ·dy FLOAT")
    public void testSpmvTransposeFloat() {
        // A is [rows, cols]; Aᵀ·dy where dy is [rows]
        int rows = 3, cols = 4;
        INDArray A = makeSparseMatrix(rows, cols, DataType.FLOAT);
        INDArray dy = Nd4j.rand(DataType.FLOAT, rows);
        // Aᵀ is [cols, rows]; Aᵀ·dy → [cols]
        INDArray expected = A.transpose().mmul(dy.reshape(rows, 1)).reshape(cols);

        SparseNDArray csr = Nd4j.toSparse(A, SparseFormat.CSR);
        CsrSpmv op = new CsrSpmv(csr.getValues(), csr.getColIdx(), csr.getRowPtr(), dy,
                csr.rows(), csr.cols(), true);
        INDArray actual = Nd4j.exec(op)[0];
        assertClose(expected, actual, TOL_FLOAT, "SpMV transpose 3x4 FLOAT");
    }
}
