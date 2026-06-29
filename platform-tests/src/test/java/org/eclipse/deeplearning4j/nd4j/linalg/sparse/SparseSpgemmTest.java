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
import org.nd4j.linalg.api.ops.impl.sparse.CsrSpgemm;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Forward-correctness and structural-validity tests for CSR sparse×sparse matrix multiply
 * (SpGEMM / {@code csr_spgemm}) and the {@link SparseNDArray#mmul(SparseNDArray)} ergonomic
 * wrapper.
 *
 * <p>Strategy: for each shape/dtype combination, build dense matrices with an explicit mix of
 * zeros and non-zeros, convert each to CSR via {@link Nd4j#toSparse}, execute the sparse
 * multiply, and compare the dense materialisation of C against the reference dense mmul.
 * Structural validity of the CSR result (rowPtr monotonic, colIdx sorted/in-range) is checked
 * after each case.
 */
@DisplayName("CSR SpGEMM (sparse×sparse matmul) Tests")
public class SparseSpgemmTest {

    private static final double TOL_FLOAT  = 1e-4;
    private static final double TOL_DOUBLE = 1e-9;

    // -----------------------------------------------------------------------
    // Utilities
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
     * Build a 2D dense matrix where every other element (by linear index) is non-zero.
     * Values are consecutive integers starting from 1.
     */
    private static INDArray makeSparse2D(int rows, int cols, DataType dtype) {
        int n = rows * cols;
        double[] data = new double[n];
        int v = 1;
        for (int i = 0; i < n; i++) {
            data[i] = (i % 2 == 0) ? v++ : 0.0;
        }
        return Nd4j.create(data, new long[]{rows, cols}).castTo(dtype);
    }

    /**
     * Build a dense matrix where a specific density fraction of elements are non-zero.
     * Elements at position (r, c) where (r + c) % stride == 0 are set to (r*cols + c + 1).
     *
     * @param stride controls sparsity: stride=1 → dense, stride=3 → ~33% filled
     */
    private static INDArray makePatternedSparse(int rows, int cols, int stride, DataType dtype) {
        double[] data = new double[rows * cols];
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                if ((r + c) % stride == 0) {
                    data[r * cols + c] = r * cols + c + 1.0;
                }
            }
        }
        return Nd4j.create(data, new long[]{rows, cols}).castTo(dtype);
    }

    /**
     * Verify structural validity of a CSR result: rowPtr is monotonically non-decreasing,
     * each colIdx is in [0, cols), and within each row the colIdx values are sorted ascending.
     */
    private static void assertCsrValid(SparseNDArray csr, long cols, String context) {
        INDArray rowPtr = csr.getRowPtr();
        INDArray colIdx = csr.getColIdx();
        long rows = csr.rows();

        // rowPtr length == rows + 1
        assertEquals(rows + 1, rowPtr.length(), context + ": rowPtr length must be rows+1");

        // rowPtr[0] == 0
        assertEquals(0, rowPtr.getLong(0), context + ": rowPtr[0] must be 0");

        // rowPtr is monotonically non-decreasing and final value == nnz
        long nnz = csr.nnz();
        assertEquals(nnz, rowPtr.getLong(rows), context + ": rowPtr[rows] must equal nnz");
        for (long r = 0; r < rows; r++) {
            long start = rowPtr.getLong(r);
            long end   = rowPtr.getLong(r + 1);
            assertTrue(end >= start,
                    context + ": rowPtr must be non-decreasing at row " + r
                    + " (start=" + start + ", end=" + end + ")");

            // colIdx values within each row: in [0, cols) and sorted ascending
            long prev = -1;
            for (long k = start; k < end; k++) {
                long col = colIdx.getLong(k);
                assertTrue(col >= 0 && col < cols,
                        context + ": colIdx[" + k + "]=" + col + " out of range [0, " + cols + ")");
                assertTrue(col > prev,
                        context + ": colIdx not sorted ascending in row " + r
                        + " (prev=" + prev + ", cur=" + col + ")");
                prev = col;
            }
        }
    }

    // -----------------------------------------------------------------------
    // Core correctness: dense(C) ≈ A_dense · B_dense
    // -----------------------------------------------------------------------

    @Test
    @DisplayName("SpGEMM 3×4 × 4×2 FLOAT — matches dense mmul")
    public void testSpgemm3x4x2Float() {
        INDArray A = makeSparse2D(3, 4, DataType.FLOAT);
        INDArray B = makeSparse2D(4, 2, DataType.FLOAT);
        INDArray expected = A.mmul(B);

        SparseNDArray csrA = Nd4j.toSparse(A, SparseFormat.CSR);
        SparseNDArray csrB = Nd4j.toSparse(B, SparseFormat.CSR);

        SparseNDArray csrC = csrA.mmul(csrB);
        assertCsrValid(csrC, 2, "SpGEMM 3×4×2 FLOAT");
        assertClose(expected, csrC.toDense(), TOL_FLOAT, "SpGEMM 3×4×2 FLOAT");
    }

    @Test
    @DisplayName("SpGEMM 4×4 × 4×4 DOUBLE — matches dense mmul")
    public void testSpgemmSquare4x4Double() {
        INDArray A = makeSparse2D(4, 4, DataType.DOUBLE);
        INDArray B = makeSparse2D(4, 4, DataType.DOUBLE);
        INDArray expected = A.mmul(B);

        SparseNDArray csrA = Nd4j.toSparse(A, SparseFormat.CSR);
        SparseNDArray csrB = Nd4j.toSparse(B, SparseFormat.CSR);

        SparseNDArray csrC = csrA.mmul(csrB);
        assertCsrValid(csrC, 4, "SpGEMM 4×4×4 DOUBLE");
        assertClose(expected, csrC.toDense(), TOL_DOUBLE, "SpGEMM 4×4×4 DOUBLE");
    }

    @Test
    @DisplayName("SpGEMM 5×3 × 3×6 FLOAT — non-square, m≠k≠n")
    public void testSpgemm5x3x6Float() {
        INDArray A = makeSparse2D(5, 3, DataType.FLOAT);
        INDArray B = makeSparse2D(3, 6, DataType.FLOAT);
        INDArray expected = A.mmul(B);

        SparseNDArray csrA = Nd4j.toSparse(A, SparseFormat.CSR);
        SparseNDArray csrB = Nd4j.toSparse(B, SparseFormat.CSR);

        SparseNDArray csrC = csrA.mmul(csrB);
        assertCsrValid(csrC, 6, "SpGEMM 5×3×6 FLOAT");
        assertClose(expected, csrC.toDense(), TOL_FLOAT, "SpGEMM 5×3×6 FLOAT");
    }

    @Test
    @DisplayName("SpGEMM 6×5 × 5×2 DOUBLE — tall × fat")
    public void testSpgemm6x5x2Double() {
        INDArray A = makePatternedSparse(6, 5, 2, DataType.DOUBLE);
        INDArray B = makePatternedSparse(5, 2, 2, DataType.DOUBLE);
        INDArray expected = A.mmul(B);

        SparseNDArray csrA = Nd4j.toSparse(A, SparseFormat.CSR);
        SparseNDArray csrB = Nd4j.toSparse(B, SparseFormat.CSR);

        SparseNDArray csrC = csrA.mmul(csrB);
        assertCsrValid(csrC, 2, "SpGEMM 6×5×2 DOUBLE");
        assertClose(expected, csrC.toDense(), TOL_DOUBLE, "SpGEMM 6×5×2 DOUBLE");
    }

    @Test
    @DisplayName("SpGEMM 2×8 × 8×3 FLOAT — wide A")
    public void testSpgemm2x8x3Float() {
        INDArray A = makeSparse2D(2, 8, DataType.FLOAT);
        INDArray B = makeSparse2D(8, 3, DataType.FLOAT);
        INDArray expected = A.mmul(B);

        SparseNDArray csrA = Nd4j.toSparse(A, SparseFormat.CSR);
        SparseNDArray csrB = Nd4j.toSparse(B, SparseFormat.CSR);

        SparseNDArray csrC = csrA.mmul(csrB);
        assertCsrValid(csrC, 3, "SpGEMM 2×8×3 FLOAT");
        assertClose(expected, csrC.toDense(), TOL_FLOAT, "SpGEMM 2×8×3 FLOAT");
    }

    // -----------------------------------------------------------------------
    // Identity matrix sanity: I × B == B
    // -----------------------------------------------------------------------

    @Test
    @DisplayName("SpGEMM identity × B == B (FLOAT)")
    public void testSpgemmIdentityFloat() {
        int n = 4;
        // Build identity in CSR manually
        INDArray identityDense = Nd4j.eye(n).castTo(DataType.FLOAT);
        INDArray B = makeSparse2D(n, 3, DataType.FLOAT);
        INDArray expected = B.dup();   // I * B == B

        SparseNDArray csrI = Nd4j.toSparse(identityDense, SparseFormat.CSR);
        SparseNDArray csrB = Nd4j.toSparse(B, SparseFormat.CSR);

        SparseNDArray csrC = csrI.mmul(csrB);
        assertCsrValid(csrC, 3, "SpGEMM I×B FLOAT");
        assertClose(expected, csrC.toDense(), TOL_FLOAT, "SpGEMM identity×B FLOAT");
    }

    @Test
    @DisplayName("SpGEMM identity × B == B (DOUBLE)")
    public void testSpgemmIdentityDouble() {
        int n = 5;
        INDArray identityDense = Nd4j.eye(n).castTo(DataType.DOUBLE);
        INDArray B = makePatternedSparse(n, 4, 2, DataType.DOUBLE);
        INDArray expected = B.dup();

        SparseNDArray csrI = Nd4j.toSparse(identityDense, SparseFormat.CSR);
        SparseNDArray csrB = Nd4j.toSparse(B, SparseFormat.CSR);

        SparseNDArray csrC = csrI.mmul(csrB);
        assertCsrValid(csrC, 4, "SpGEMM I×B DOUBLE");
        assertClose(expected, csrC.toDense(), TOL_DOUBLE, "SpGEMM identity×B DOUBLE");
    }

    // -----------------------------------------------------------------------
    // Zero-row in A → corresponding row of C is all zeros
    // -----------------------------------------------------------------------

    @Test
    @DisplayName("SpGEMM: A with an all-zero row produces zero row in C (FLOAT)")
    public void testSpgemmZeroRowInA() {
        // Build A 4×3 with row 2 forced to all zeros
        double[] aData = {
                1, 0, 2,
                0, 3, 0,
                0, 0, 0,   // row 2 all zero
                4, 0, 5
        };
        INDArray A = Nd4j.create(aData, new long[]{4, 3}).castTo(DataType.FLOAT);
        INDArray B = makeSparse2D(3, 2, DataType.FLOAT);
        INDArray expected = A.mmul(B);

        SparseNDArray csrA = Nd4j.toSparse(A, SparseFormat.CSR);
        SparseNDArray csrB = Nd4j.toSparse(B, SparseFormat.CSR);

        SparseNDArray csrC = csrA.mmul(csrB);
        assertCsrValid(csrC, 2, "SpGEMM zero-row-in-A FLOAT");
        INDArray cDense = csrC.toDense();
        assertClose(expected, cDense, TOL_FLOAT, "SpGEMM zero-row-in-A FLOAT");

        // Row 2 of C must be zero
        INDArray row2 = cDense.getRow(2);
        for (long c = 0; c < row2.length(); c++) {
            assertEquals(0.0, row2.getDouble(c), TOL_FLOAT,
                    "C[2," + c + "] must be 0 when A row 2 is all zeros");
        }
    }

    // -----------------------------------------------------------------------
    // All-zero B → C is all zeros (empty result)
    // -----------------------------------------------------------------------

    @Test
    @DisplayName("SpGEMM: all-zero B produces all-zero C (FLOAT)")
    public void testSpgemmZeroB() {
        INDArray A = makeSparse2D(3, 4, DataType.FLOAT);
        INDArray B = Nd4j.zeros(DataType.FLOAT, 4, 3);   // all zeros → nnzB = 0
        INDArray expected = Nd4j.zeros(DataType.FLOAT, 3, 3);

        SparseNDArray csrA = Nd4j.toSparse(A, SparseFormat.CSR);
        SparseNDArray csrB = Nd4j.toSparse(B, SparseFormat.CSR);

        SparseNDArray csrC = csrA.mmul(csrB);
        assertCsrValid(csrC, 3, "SpGEMM zero-B FLOAT");

        // nnz of C must be 0
        assertEquals(0L, csrC.nnz(), "C must have 0 non-zeros when B is all zeros");
        assertClose(expected, csrC.toDense(), TOL_FLOAT, "SpGEMM zero-B FLOAT");
    }

    @Test
    @DisplayName("SpGEMM: all-zero A produces all-zero C (DOUBLE)")
    public void testSpgemmZeroA() {
        INDArray A = Nd4j.zeros(DataType.DOUBLE, 4, 3);  // all zeros → nnzA = 0
        INDArray B = makeSparse2D(3, 5, DataType.DOUBLE);
        INDArray expected = Nd4j.zeros(DataType.DOUBLE, 4, 5);

        SparseNDArray csrA = Nd4j.toSparse(A, SparseFormat.CSR);
        SparseNDArray csrB = Nd4j.toSparse(B, SparseFormat.CSR);

        SparseNDArray csrC = csrA.mmul(csrB);
        assertCsrValid(csrC, 5, "SpGEMM zero-A DOUBLE");
        assertEquals(0L, csrC.nnz(), "C must have 0 non-zeros when A is all zeros");
        assertClose(expected, csrC.toDense(), TOL_DOUBLE, "SpGEMM zero-A DOUBLE");
    }

    // -----------------------------------------------------------------------
    // Direct op usage (not via SparseNDArray helper)
    // -----------------------------------------------------------------------

    @Test
    @DisplayName("SpGEMM via CsrSpgemm op directly (not SparseNDArray helper) FLOAT")
    public void testSpgemmViaOpDirectlyFloat() {
        INDArray A = makeSparse2D(3, 4, DataType.FLOAT);
        INDArray B = makeSparse2D(4, 3, DataType.FLOAT);
        INDArray expected = A.mmul(B);

        SparseNDArray csrA = Nd4j.toSparse(A, SparseFormat.CSR);
        SparseNDArray csrB = Nd4j.toSparse(B, SparseFormat.CSR);

        CsrSpgemm op = new CsrSpgemm(
                csrA.getValues(), csrA.getColIdx(), csrA.getRowPtr(),
                csrB.getValues(), csrB.getColIdx(), csrB.getRowPtr(),
                csrA.rows(), csrA.cols(), csrB.cols());
        INDArray[] results = Nd4j.exec(op);

        // Wrap and verify
        SparseNDArray csrC = new SparseNDArray(results[0], results[1], results[2],
                new long[]{csrA.rows(), csrB.cols()}, SparseFormat.CSR);
        assertCsrValid(csrC, csrB.cols(), "SpGEMM direct op 3×4×3 FLOAT");
        assertClose(expected, csrC.toDense(), TOL_FLOAT, "SpGEMM direct op 3×4×3 FLOAT");
    }

    // -----------------------------------------------------------------------
    // Higher sparsity (stride-3 pattern, ~33% density)
    // -----------------------------------------------------------------------

    @Test
    @DisplayName("SpGEMM 6×6 × 6×6 FLOAT high-sparsity (stride 3)")
    public void testSpgemmHighSparsityFloat() {
        INDArray A = makePatternedSparse(6, 6, 3, DataType.FLOAT);
        INDArray B = makePatternedSparse(6, 6, 3, DataType.FLOAT);
        INDArray expected = A.mmul(B);

        SparseNDArray csrA = Nd4j.toSparse(A, SparseFormat.CSR);
        SparseNDArray csrB = Nd4j.toSparse(B, SparseFormat.CSR);

        SparseNDArray csrC = csrA.mmul(csrB);
        assertCsrValid(csrC, 6, "SpGEMM high-sparsity 6×6 FLOAT");
        assertClose(expected, csrC.toDense(), TOL_FLOAT, "SpGEMM high-sparsity 6×6 FLOAT");
    }

    @Test
    @DisplayName("SpGEMM 8×5 × 5×7 DOUBLE high-sparsity (stride 3)")
    public void testSpgemmHighSparsityDouble() {
        INDArray A = makePatternedSparse(8, 5, 3, DataType.DOUBLE);
        INDArray B = makePatternedSparse(5, 7, 3, DataType.DOUBLE);
        INDArray expected = A.mmul(B);

        SparseNDArray csrA = Nd4j.toSparse(A, SparseFormat.CSR);
        SparseNDArray csrB = Nd4j.toSparse(B, SparseFormat.CSR);

        SparseNDArray csrC = csrA.mmul(csrB);
        assertCsrValid(csrC, 7, "SpGEMM high-sparsity 8×5×7 DOUBLE");
        assertClose(expected, csrC.toDense(), TOL_DOUBLE, "SpGEMM high-sparsity 8×5×7 DOUBLE");
    }

    // -----------------------------------------------------------------------
    // Output shape is correct (not just values)
    // -----------------------------------------------------------------------

    @Test
    @DisplayName("SpGEMM result shape is [m, n]")
    public void testSpgemmOutputShape() {
        int m = 7, k = 3, n = 5;
        INDArray A = makeSparse2D(m, k, DataType.FLOAT);
        INDArray B = makeSparse2D(k, n, DataType.FLOAT);

        SparseNDArray csrA = Nd4j.toSparse(A, SparseFormat.CSR);
        SparseNDArray csrB = Nd4j.toSparse(B, SparseFormat.CSR);

        SparseNDArray csrC = csrA.mmul(csrB);
        assertArrayEquals(new long[]{m, n}, csrC.getShape(),
                "Output logical shape must be [m, n]");
        assertEquals(m, csrC.rows(), "csrC.rows() must equal m");
        assertEquals(n, csrC.cols(), "csrC.cols() must equal n");

        // toDense() shape
        INDArray cDense = csrC.toDense();
        assertArrayEquals(new long[]{m, n}, cDense.shape(),
                "toDense() shape must be [m, n]");
    }

    // -----------------------------------------------------------------------
    // Precondition enforcement
    // -----------------------------------------------------------------------

    @Test
    @DisplayName("SpGEMM throws when this is not CSR")
    public void testSpgemmThrowsIfThisNotCsr() {
        // Build a COO sparse array and try to multiply
        double[] vals = {1.0, 2.0};
        long[][] idx  = {{0, 0}, {1, 1}};
        INDArray indices = Nd4j.create(new double[]{0, 0, 1, 1}, new long[]{2, 2}).castTo(DataType.INT64);
        INDArray values  = Nd4j.create(vals, new long[]{2}).castTo(DataType.FLOAT);

        SparseNDArray coo = new SparseNDArray(indices, values, new long[]{2, 2}, SparseFormat.COO);

        INDArray B = Nd4j.eye(2).castTo(DataType.FLOAT);
        SparseNDArray csrB = Nd4j.toSparse(B, SparseFormat.CSR);

        assertThrows(Exception.class, () -> coo.mmul(csrB),
                "mmul(SparseNDArray) must throw when this is COO");
    }

    @Test
    @DisplayName("SpGEMM throws when other is not CSR")
    public void testSpgemmThrowsIfOtherNotCsr() {
        INDArray A = makeSparse2D(3, 3, DataType.FLOAT);
        SparseNDArray csrA = Nd4j.toSparse(A, SparseFormat.CSR);

        INDArray indices = Nd4j.create(new double[]{0, 0, 1, 1}, new long[]{2, 2}).castTo(DataType.INT64);
        INDArray values  = Nd4j.create(new double[]{1.0, 2.0}, new long[]{2}).castTo(DataType.FLOAT);
        SparseNDArray cooB = new SparseNDArray(indices, values, new long[]{3, 3}, SparseFormat.COO);

        assertThrows(Exception.class, () -> csrA.mmul(cooB),
                "mmul(SparseNDArray) must throw when other is COO");
    }

    @Test
    @DisplayName("SpGEMM throws on inner-dimension mismatch")
    public void testSpgemmThrowsOnDimensionMismatch() {
        INDArray A = makeSparse2D(3, 4, DataType.FLOAT);
        INDArray B = makeSparse2D(5, 3, DataType.FLOAT); // B.rows=5 != A.cols=4

        SparseNDArray csrA = Nd4j.toSparse(A, SparseFormat.CSR);
        SparseNDArray csrB = Nd4j.toSparse(B, SparseFormat.CSR);

        assertThrows(Exception.class, () -> csrA.mmul(csrB),
                "mmul(SparseNDArray) must throw on inner-dimension mismatch");
    }
}
