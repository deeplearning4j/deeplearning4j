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
import org.nd4j.linalg.api.ops.impl.sparse.CsrAdd;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Forward-correctness and structural-validity tests for:
 * <ul>
 *   <li>CSR elementwise addition ({@code csr_add}) and the {@link SparseNDArray#add(SparseNDArray)}
 *       ergonomic wrapper</li>
 *   <li>Container reductions and scale: {@link SparseNDArray#rowSums()},
 *       {@link SparseNDArray#colSums()}, {@link SparseNDArray#sumNumber()},
 *       {@link SparseNDArray#scale(double)}</li>
 * </ul>
 *
 * <p>Strategy: for each test, build a dense reference matrix, convert to CSR via
 * {@link Nd4j#toSparse}, execute the sparse operation, and compare the dense materialisation
 * against the reference. CSR structural validity (rowPtr monotonic, colIdx sorted/in-range)
 * is verified after each {@code csr_add} result.
 */
@DisplayName("CSR Elementwise Add + Container Reductions/Scale Tests")
public class SparseElementwiseTest {

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
     * Dense matrix where elements at (r+c)%2==0 are set to (r*cols+c+1), rest are 0.
     * About 50% sparsity; guaranteed to be non-zero only at even-parity positions.
     */
    private static INDArray makeEvenSparse(int rows, int cols, DataType dtype) {
        double[] data = new double[rows * cols];
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                if ((r + c) % 2 == 0) {
                    data[r * cols + c] = r * cols + c + 1.0;
                }
            }
        }
        return Nd4j.create(data, new long[]{rows, cols}).castTo(dtype);
    }

    /**
     * Dense matrix where elements at (r+c)%2==1 are set to (r*cols+c+1)*10, rest are 0.
     * Complement of {@link #makeEvenSparse}: no position is non-zero in both.
     */
    private static INDArray makeOddSparse(int rows, int cols, DataType dtype) {
        double[] data = new double[rows * cols];
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                if ((r + c) % 2 == 1) {
                    data[r * cols + c] = (r * cols + c + 1.0) * 10.0;
                }
            }
        }
        return Nd4j.create(data, new long[]{rows, cols}).castTo(dtype);
    }

    /**
     * Dense matrix where elements at stride-separated positions are non-zero.
     * (r + c) % stride == 0 → value = r*cols + c + 1.
     */
    private static INDArray makeStrideSparse(int rows, int cols, int stride, DataType dtype) {
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

        assertEquals(rows + 1, rowPtr.length(), context + ": rowPtr length must be rows+1");
        assertEquals(0, rowPtr.getLong(0), context + ": rowPtr[0] must be 0");

        long nnz = csr.nnz();
        assertEquals(nnz, rowPtr.getLong(rows), context + ": rowPtr[rows] must equal nnz");

        for (long r = 0; r < rows; r++) {
            long start = rowPtr.getLong(r);
            long end   = rowPtr.getLong(r + 1);
            assertTrue(end >= start,
                    context + ": rowPtr must be non-decreasing at row " + r
                    + " (start=" + start + ", end=" + end + ")");

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
    // csr_add: correctness dense(C) ≈ A_dense + B_dense
    // -----------------------------------------------------------------------

    @Test
    @DisplayName("csr_add 3×4 FLOAT even-parity sparsity — matches dense add")
    public void testCsrAdd3x4Float() {
        INDArray A = makeEvenSparse(3, 4, DataType.FLOAT);
        INDArray B = makeStrideSparse(3, 4, 3, DataType.FLOAT);
        INDArray expected = A.add(B);

        SparseNDArray csrA = Nd4j.toSparse(A, SparseFormat.CSR);
        SparseNDArray csrB = Nd4j.toSparse(B, SparseFormat.CSR);

        SparseNDArray csrC = csrA.add(csrB);
        assertCsrValid(csrC, 4, "csr_add 3x4 FLOAT");
        assertClose(expected, csrC.toDense(), TOL_FLOAT, "csr_add 3x4 FLOAT");
    }

    @Test
    @DisplayName("csr_add 4×4 DOUBLE square — matches dense add")
    public void testCsrAdd4x4Double() {
        INDArray A = makeEvenSparse(4, 4, DataType.DOUBLE);
        INDArray B = makeStrideSparse(4, 4, 3, DataType.DOUBLE);
        INDArray expected = A.add(B);

        SparseNDArray csrA = Nd4j.toSparse(A, SparseFormat.CSR);
        SparseNDArray csrB = Nd4j.toSparse(B, SparseFormat.CSR);

        SparseNDArray csrC = csrA.add(csrB);
        assertCsrValid(csrC, 4, "csr_add 4x4 DOUBLE");
        assertClose(expected, csrC.toDense(), TOL_DOUBLE, "csr_add 4x4 DOUBLE");
    }

    @Test
    @DisplayName("csr_add 5×3 FLOAT non-square — matches dense add")
    public void testCsrAdd5x3Float() {
        INDArray A = makeEvenSparse(5, 3, DataType.FLOAT);
        INDArray B = makeOddSparse(5, 3, DataType.FLOAT);
        INDArray expected = A.add(B);

        SparseNDArray csrA = Nd4j.toSparse(A, SparseFormat.CSR);
        SparseNDArray csrB = Nd4j.toSparse(B, SparseFormat.CSR);

        SparseNDArray csrC = csrA.add(csrB);
        assertCsrValid(csrC, 3, "csr_add 5x3 FLOAT");
        assertClose(expected, csrC.toDense(), TOL_FLOAT, "csr_add 5x3 FLOAT");
    }

    @Test
    @DisplayName("csr_add 6×5 DOUBLE tall — matches dense add")
    public void testCsrAdd6x5Double() {
        INDArray A = makeStrideSparse(6, 5, 2, DataType.DOUBLE);
        INDArray B = makeStrideSparse(6, 5, 3, DataType.DOUBLE);
        INDArray expected = A.add(B);

        SparseNDArray csrA = Nd4j.toSparse(A, SparseFormat.CSR);
        SparseNDArray csrB = Nd4j.toSparse(B, SparseFormat.CSR);

        SparseNDArray csrC = csrA.add(csrB);
        assertCsrValid(csrC, 5, "csr_add 6x5 DOUBLE");
        assertClose(expected, csrC.toDense(), TOL_DOUBLE, "csr_add 6x5 DOUBLE");
    }

    // -----------------------------------------------------------------------
    // csr_add: disjoint sparsity patterns (A and B have no position in common)
    // -----------------------------------------------------------------------

    @Test
    @DisplayName("csr_add FLOAT disjoint patterns — every position filled in C")
    public void testCsrAddDisjointFloat() {
        // A has elements at even-parity positions, B at odd-parity.
        // After add, every element should be non-zero.
        INDArray A = makeEvenSparse(4, 4, DataType.FLOAT);
        INDArray B = makeOddSparse(4, 4, DataType.FLOAT);
        INDArray expected = A.add(B);

        SparseNDArray csrA = Nd4j.toSparse(A, SparseFormat.CSR);
        SparseNDArray csrB = Nd4j.toSparse(B, SparseFormat.CSR);

        SparseNDArray csrC = csrA.add(csrB);
        assertCsrValid(csrC, 4, "csr_add disjoint FLOAT");
        assertClose(expected, csrC.toDense(), TOL_FLOAT, "csr_add disjoint FLOAT");
        // Every element is non-zero, so nnz should equal total elements
        assertEquals(16L, csrC.nnz(), "disjoint add 4×4: all 16 elements should be non-zero");
    }

    @Test
    @DisplayName("csr_add DOUBLE disjoint patterns 3×6 — every position filled in C")
    public void testCsrAddDisjointDouble() {
        INDArray A = makeEvenSparse(3, 6, DataType.DOUBLE);
        INDArray B = makeOddSparse(3, 6, DataType.DOUBLE);
        INDArray expected = A.add(B);

        SparseNDArray csrA = Nd4j.toSparse(A, SparseFormat.CSR);
        SparseNDArray csrB = Nd4j.toSparse(B, SparseFormat.CSR);

        SparseNDArray csrC = csrA.add(csrB);
        assertCsrValid(csrC, 6, "csr_add disjoint DOUBLE 3x6");
        assertClose(expected, csrC.toDense(), TOL_DOUBLE, "csr_add disjoint DOUBLE 3x6");
        assertEquals(18L, csrC.nnz(), "disjoint add 3×6: all 18 elements should be non-zero");
    }

    // -----------------------------------------------------------------------
    // csr_add: fully overlapping sparsity patterns (A and B share the same pattern)
    // -----------------------------------------------------------------------

    @Test
    @DisplayName("csr_add FLOAT fully overlapping patterns — result is 2*A")
    public void testCsrAddFullyOverlappingFloat() {
        INDArray A = makeEvenSparse(4, 5, DataType.FLOAT);
        INDArray expected = A.add(A);   // 2*A

        SparseNDArray csrA = Nd4j.toSparse(A, SparseFormat.CSR);

        SparseNDArray csrC = csrA.add(csrA);
        assertCsrValid(csrC, 5, "csr_add overlap FLOAT 4x5");
        assertClose(expected, csrC.toDense(), TOL_FLOAT, "csr_add fully overlapping FLOAT 4x5");
        // nnz should equal nnzA (same pattern, values doubled — no cancellation)
        assertEquals(csrA.nnz(), csrC.nnz(), "overlapping add: nnz must equal nnzA");
    }

    @Test
    @DisplayName("csr_add DOUBLE fully overlapping patterns — result is 2*A")
    public void testCsrAddFullyOverlappingDouble() {
        INDArray A = makeStrideSparse(5, 5, 2, DataType.DOUBLE);
        INDArray expected = A.add(A);

        SparseNDArray csrA = Nd4j.toSparse(A, SparseFormat.CSR);

        SparseNDArray csrC = csrA.add(csrA);
        assertCsrValid(csrC, 5, "csr_add overlap DOUBLE 5x5");
        assertClose(expected, csrC.toDense(), TOL_DOUBLE, "csr_add fully overlapping DOUBLE 5x5");
        assertEquals(csrA.nnz(), csrC.nnz(), "overlapping add: nnz must equal nnzA");
    }

    // -----------------------------------------------------------------------
    // csr_add: one operand all-zero
    // -----------------------------------------------------------------------

    @Test
    @DisplayName("csr_add A all-zero: C == B")
    public void testCsrAddAAllZeroFloat() {
        INDArray A = Nd4j.zeros(DataType.FLOAT, 3, 4);   // all zeros
        INDArray B = makeEvenSparse(3, 4, DataType.FLOAT);
        INDArray expected = B.dup();   // 0 + B == B

        SparseNDArray csrA = Nd4j.toSparse(A, SparseFormat.CSR);
        SparseNDArray csrB = Nd4j.toSparse(B, SparseFormat.CSR);

        SparseNDArray csrC = csrA.add(csrB);
        assertCsrValid(csrC, 4, "csr_add A-zero FLOAT 3x4");
        assertClose(expected, csrC.toDense(), TOL_FLOAT, "csr_add A=0 FLOAT 3x4");
    }

    @Test
    @DisplayName("csr_add B all-zero: C == A")
    public void testCsrAddBAllZeroDouble() {
        INDArray A = makeEvenSparse(4, 3, DataType.DOUBLE);
        INDArray B = Nd4j.zeros(DataType.DOUBLE, 4, 3);   // all zeros
        INDArray expected = A.dup();   // A + 0 == A

        SparseNDArray csrA = Nd4j.toSparse(A, SparseFormat.CSR);
        SparseNDArray csrB = Nd4j.toSparse(B, SparseFormat.CSR);

        SparseNDArray csrC = csrA.add(csrB);
        assertCsrValid(csrC, 3, "csr_add B-zero DOUBLE 4x3");
        assertClose(expected, csrC.toDense(), TOL_DOUBLE, "csr_add B=0 DOUBLE 4x3");
    }

    // -----------------------------------------------------------------------
    // csr_add: both operands all-zero → result is all-zero
    // -----------------------------------------------------------------------

    @Test
    @DisplayName("csr_add both all-zero: C is all-zero and nnz==0")
    public void testCsrAddBothZeroFloat() {
        INDArray A = Nd4j.zeros(DataType.FLOAT, 3, 5);
        INDArray B = Nd4j.zeros(DataType.FLOAT, 3, 5);
        INDArray expected = Nd4j.zeros(DataType.FLOAT, 3, 5);

        SparseNDArray csrA = Nd4j.toSparse(A, SparseFormat.CSR);
        SparseNDArray csrB = Nd4j.toSparse(B, SparseFormat.CSR);

        SparseNDArray csrC = csrA.add(csrB);
        assertCsrValid(csrC, 5, "csr_add both-zero FLOAT 3x5");
        assertEquals(0L, csrC.nnz(), "both-zero add: C must have 0 non-zeros");
        assertClose(expected, csrC.toDense(), TOL_FLOAT, "csr_add both-zero FLOAT 3x5");
    }

    @Test
    @DisplayName("csr_add both all-zero DOUBLE: C is all-zero and nnz==0")
    public void testCsrAddBothZeroDouble() {
        INDArray A = Nd4j.zeros(DataType.DOUBLE, 4, 4);
        INDArray B = Nd4j.zeros(DataType.DOUBLE, 4, 4);

        SparseNDArray csrA = Nd4j.toSparse(A, SparseFormat.CSR);
        SparseNDArray csrB = Nd4j.toSparse(B, SparseFormat.CSR);

        SparseNDArray csrC = csrA.add(csrB);
        assertCsrValid(csrC, 4, "csr_add both-zero DOUBLE 4x4");
        assertEquals(0L, csrC.nnz(), "both-zero add: C must have 0 non-zeros (DOUBLE)");
    }

    // -----------------------------------------------------------------------
    // csr_add: direct op usage (not via SparseNDArray helper)
    // -----------------------------------------------------------------------

    @Test
    @DisplayName("csr_add via CsrAdd op directly (not SparseNDArray helper) FLOAT")
    public void testCsrAddViaOpDirectlyFloat() {
        INDArray A = makeEvenSparse(3, 4, DataType.FLOAT);
        INDArray B = makeStrideSparse(3, 4, 3, DataType.FLOAT);
        INDArray expected = A.add(B);

        SparseNDArray csrA = Nd4j.toSparse(A, SparseFormat.CSR);
        SparseNDArray csrB = Nd4j.toSparse(B, SparseFormat.CSR);

        CsrAdd op = new CsrAdd(
                csrA.getValues(), csrA.getColIdx(), csrA.getRowPtr(),
                csrB.getValues(), csrB.getColIdx(), csrB.getRowPtr(),
                csrA.rows(), csrA.cols());
        INDArray[] results = Nd4j.exec(op);

        SparseNDArray csrC = new SparseNDArray(results[0], results[1], results[2],
                new long[]{csrA.rows(), csrA.cols()}, SparseFormat.CSR);
        assertCsrValid(csrC, csrA.cols(), "CsrAdd direct op 3x4 FLOAT");
        assertClose(expected, csrC.toDense(), TOL_FLOAT, "CsrAdd direct op 3x4 FLOAT");
    }

    // -----------------------------------------------------------------------
    // csr_add: precondition enforcement
    // -----------------------------------------------------------------------

    @Test
    @DisplayName("csr_add throws when this is not CSR")
    public void testCsrAddThrowsIfThisNotCsr() {
        INDArray indices = Nd4j.create(new double[]{0, 0, 1, 1}, new long[]{2, 2}).castTo(DataType.INT64);
        INDArray values  = Nd4j.create(new double[]{1.0, 2.0}, new long[]{2}).castTo(DataType.FLOAT);
        SparseNDArray coo = new SparseNDArray(indices, values, new long[]{2, 2}, SparseFormat.COO);

        INDArray B = makeEvenSparse(2, 2, DataType.FLOAT);
        SparseNDArray csrB = Nd4j.toSparse(B, SparseFormat.CSR);

        assertThrows(Exception.class, () -> coo.add(csrB),
                "add() must throw when this is COO");
    }

    @Test
    @DisplayName("csr_add throws when other is not CSR")
    public void testCsrAddThrowsIfOtherNotCsr() {
        INDArray A = makeEvenSparse(3, 3, DataType.FLOAT);
        SparseNDArray csrA = Nd4j.toSparse(A, SparseFormat.CSR);

        INDArray indices = Nd4j.create(new double[]{0, 0, 1, 1}, new long[]{2, 2}).castTo(DataType.INT64);
        INDArray values  = Nd4j.create(new double[]{1.0, 2.0}, new long[]{2}).castTo(DataType.FLOAT);
        SparseNDArray cooB = new SparseNDArray(indices, values, new long[]{3, 3}, SparseFormat.COO);

        assertThrows(Exception.class, () -> csrA.add(cooB),
                "add() must throw when other is COO");
    }

    @Test
    @DisplayName("csr_add throws on shape mismatch")
    public void testCsrAddThrowsOnShapeMismatch() {
        INDArray A = makeEvenSparse(3, 4, DataType.FLOAT);
        INDArray B = makeEvenSparse(3, 5, DataType.FLOAT); // different cols
        SparseNDArray csrA = Nd4j.toSparse(A, SparseFormat.CSR);
        SparseNDArray csrB = Nd4j.toSparse(B, SparseFormat.CSR);

        assertThrows(Exception.class, () -> csrA.add(csrB),
                "add() must throw on shape mismatch");
    }

    // -----------------------------------------------------------------------
    // rowSums: ≈ A_dense.sum(1)
    // -----------------------------------------------------------------------

    @Test
    @DisplayName("rowSums() FLOAT 3×4 — matches dense.sum(1)")
    public void testRowSums3x4Float() {
        INDArray A = makeEvenSparse(3, 4, DataType.FLOAT);
        INDArray expected = A.sum(1);   // [3]

        SparseNDArray csrA = Nd4j.toSparse(A, SparseFormat.CSR);
        INDArray result = csrA.rowSums();

        assertEquals(3, result.length(), "rowSums() length must equal rows");
        assertClose(expected, result, TOL_FLOAT, "rowSums 3x4 FLOAT");
    }

    @Test
    @DisplayName("rowSums() DOUBLE 5×3 — matches dense.sum(1)")
    public void testRowSums5x3Double() {
        INDArray A = makeStrideSparse(5, 3, 2, DataType.DOUBLE);
        INDArray expected = A.sum(1);   // [5]

        SparseNDArray csrA = Nd4j.toSparse(A, SparseFormat.CSR);
        INDArray result = csrA.rowSums();

        assertEquals(5, result.length(), "rowSums() length must equal rows");
        assertClose(expected, result, TOL_DOUBLE, "rowSums 5x3 DOUBLE");
    }

    @Test
    @DisplayName("rowSums() FLOAT with all-zero row — that row sum is 0")
    public void testRowSumsZeroRowFloat() {
        double[] data = {
                1, 0, 2, 0,
                0, 0, 0, 0,   // row 1 all zero
                0, 3, 0, 4
        };
        INDArray A = Nd4j.create(data, new long[]{3, 4}).castTo(DataType.FLOAT);
        INDArray expected = A.sum(1);

        SparseNDArray csrA = Nd4j.toSparse(A, SparseFormat.CSR);
        INDArray result = csrA.rowSums();

        assertClose(expected, result, TOL_FLOAT, "rowSums zero-row FLOAT");
        assertEquals(0.0, result.getDouble(1), TOL_FLOAT, "rowSums[1] must be 0 for all-zero row");
    }

    // -----------------------------------------------------------------------
    // colSums: ≈ A_dense.sum(0)
    // -----------------------------------------------------------------------

    @Test
    @DisplayName("colSums() FLOAT 3×4 — matches dense.sum(0)")
    public void testColSums3x4Float() {
        INDArray A = makeEvenSparse(3, 4, DataType.FLOAT);
        INDArray expected = A.sum(0);   // [4]

        SparseNDArray csrA = Nd4j.toSparse(A, SparseFormat.CSR);
        INDArray result = csrA.colSums();

        assertEquals(4, result.length(), "colSums() length must equal cols");
        assertClose(expected, result, TOL_FLOAT, "colSums 3x4 FLOAT");
    }

    @Test
    @DisplayName("colSums() DOUBLE 5×3 — matches dense.sum(0)")
    public void testColSums5x3Double() {
        INDArray A = makeStrideSparse(5, 3, 2, DataType.DOUBLE);
        INDArray expected = A.sum(0);   // [3]

        SparseNDArray csrA = Nd4j.toSparse(A, SparseFormat.CSR);
        INDArray result = csrA.colSums();

        assertEquals(3, result.length(), "colSums() length must equal cols");
        assertClose(expected, result, TOL_DOUBLE, "colSums 5x3 DOUBLE");
    }

    @Test
    @DisplayName("colSums() DOUBLE disjoint-pattern 4×4 — matches dense.sum(0)")
    public void testColSumsDisjointDouble() {
        INDArray A = makeEvenSparse(4, 4, DataType.DOUBLE);
        INDArray B = makeOddSparse(4, 4, DataType.DOUBLE);
        INDArray combined = A.add(B);    // fully filled; use as the sparse input
        INDArray expected = combined.sum(0);

        SparseNDArray csrCombined = Nd4j.toSparse(combined, SparseFormat.CSR);
        INDArray result = csrCombined.colSums();

        assertEquals(4, result.length(), "colSums() length must equal cols");
        assertClose(expected, result, TOL_DOUBLE, "colSums disjoint DOUBLE 4x4");
    }

    // -----------------------------------------------------------------------
    // sumNumber: ≈ A_dense.sumNumber()
    // -----------------------------------------------------------------------

    @Test
    @DisplayName("sumNumber() FLOAT — equals dense sum")
    public void testSumNumberFloat() {
        INDArray A = makeEvenSparse(4, 4, DataType.FLOAT);
        double expected = A.sumNumber().doubleValue();

        SparseNDArray csrA = Nd4j.toSparse(A, SparseFormat.CSR);
        double result = csrA.sumNumber().doubleValue();

        assertEquals(expected, result, TOL_FLOAT, "sumNumber FLOAT");
    }

    @Test
    @DisplayName("sumNumber() DOUBLE — equals dense sum")
    public void testSumNumberDouble() {
        INDArray A = makeStrideSparse(5, 3, 2, DataType.DOUBLE);
        double expected = A.sumNumber().doubleValue();

        SparseNDArray csrA = Nd4j.toSparse(A, SparseFormat.CSR);
        double result = csrA.sumNumber().doubleValue();

        assertEquals(expected, result, TOL_DOUBLE, "sumNumber DOUBLE");
    }

    @Test
    @DisplayName("sumNumber() all-zero — returns 0")
    public void testSumNumberAllZero() {
        INDArray A = Nd4j.zeros(DataType.FLOAT, 4, 4);
        SparseNDArray csrA = Nd4j.toSparse(A, SparseFormat.CSR);
        assertEquals(0.0, csrA.sumNumber().doubleValue(), TOL_FLOAT, "sumNumber all-zero must be 0");
    }

    // -----------------------------------------------------------------------
    // scale: scale(s).toDense() ≈ A_dense.mul(s)
    // -----------------------------------------------------------------------

    @Test
    @DisplayName("scale(2.0) FLOAT — toDense matches dense.mul(2.0)")
    public void testScaleFloat() {
        INDArray A = makeEvenSparse(3, 4, DataType.FLOAT);
        INDArray expected = A.mul(2.0);

        SparseNDArray csrA = Nd4j.toSparse(A, SparseFormat.CSR);
        SparseNDArray csrScaled = csrA.scale(2.0);

        assertClose(expected, csrScaled.toDense(), TOL_FLOAT, "scale(2.0) FLOAT 3x4");
    }

    @Test
    @DisplayName("scale(0.5) DOUBLE — toDense matches dense.mul(0.5)")
    public void testScaleDouble() {
        INDArray A = makeStrideSparse(5, 3, 2, DataType.DOUBLE);
        INDArray expected = A.mul(0.5);

        SparseNDArray csrA = Nd4j.toSparse(A, SparseFormat.CSR);
        SparseNDArray csrScaled = csrA.scale(0.5);

        assertClose(expected, csrScaled.toDense(), TOL_DOUBLE, "scale(0.5) DOUBLE 5x3");
    }

    @Test
    @DisplayName("scale(0.0) — result is all-zero, structure preserved (same nnz)")
    public void testScaleByZeroFloat() {
        INDArray A = makeEvenSparse(4, 4, DataType.FLOAT);
        long nnzBefore = Nd4j.toSparse(A, SparseFormat.CSR).nnz();

        SparseNDArray csrA = Nd4j.toSparse(A, SparseFormat.CSR);
        SparseNDArray csrScaled = csrA.scale(0.0);

        // scale by 0 keeps the structural arrays (colIdx/rowPtr unchanged), only values go to 0
        assertEquals(nnzBefore, csrScaled.nnz(), "scale(0): nnz unchanged (structure preserved)");
        INDArray dense = csrScaled.toDense();
        for (long i = 0; i < dense.length(); i++) {
            assertEquals(0.0, dense.getDouble(i), TOL_FLOAT, "scale(0): every element must be 0");
        }
    }

    @Test
    @DisplayName("scale(-3.0) DOUBLE — values negated and magnified")
    public void testScaleNegativeDouble() {
        INDArray A = makeEvenSparse(3, 5, DataType.DOUBLE);
        INDArray expected = A.mul(-3.0);

        SparseNDArray csrA = Nd4j.toSparse(A, SparseFormat.CSR);
        SparseNDArray csrScaled = csrA.scale(-3.0);

        assertClose(expected, csrScaled.toDense(), TOL_DOUBLE, "scale(-3.0) DOUBLE 3x5");
    }

    @Test
    @DisplayName("scale: sparsity structure (colIdx, rowPtr) is unchanged after scale")
    public void testScaleStructureUnchanged() {
        INDArray A = makeEvenSparse(4, 4, DataType.FLOAT);
        SparseNDArray csrA = Nd4j.toSparse(A, SparseFormat.CSR);
        SparseNDArray csrScaled = csrA.scale(7.0);

        // rowPtr and colIdx must be identical
        INDArray rp1 = csrA.getRowPtr();
        INDArray rp2 = csrScaled.getRowPtr();
        INDArray ci1 = csrA.getColIdx();
        INDArray ci2 = csrScaled.getColIdx();

        assertEquals(rp1.length(), rp2.length(), "scale: rowPtr length must be unchanged");
        assertEquals(ci1.length(), ci2.length(), "scale: colIdx length must be unchanged");
        for (long i = 0; i < rp1.length(); i++) {
            assertEquals(rp1.getLong(i), rp2.getLong(i),
                    "scale: rowPtr[" + i + "] must be unchanged");
        }
        for (long i = 0; i < ci1.length(); i++) {
            assertEquals(ci1.getLong(i), ci2.getLong(i),
                    "scale: colIdx[" + i + "] must be unchanged");
        }
    }

    // -----------------------------------------------------------------------
    // sumNumber/rowSums/colSums: precondition enforcement
    // -----------------------------------------------------------------------

    @Test
    @DisplayName("rowSums() throws when format is not CSR")
    public void testRowSumsThrowsIfNotCsr() {
        INDArray indices = Nd4j.create(new double[]{0, 0, 1, 1}, new long[]{2, 2}).castTo(DataType.INT64);
        INDArray values  = Nd4j.create(new double[]{1.0, 2.0}, new long[]{2}).castTo(DataType.FLOAT);
        SparseNDArray coo = new SparseNDArray(indices, values, new long[]{2, 2}, SparseFormat.COO);
        assertThrows(Exception.class, coo::rowSums, "rowSums() must throw when format is COO");
    }

    @Test
    @DisplayName("colSums() throws when format is not CSR")
    public void testColSumsThrowsIfNotCsr() {
        INDArray indices = Nd4j.create(new double[]{0, 0, 1, 1}, new long[]{2, 2}).castTo(DataType.INT64);
        INDArray values  = Nd4j.create(new double[]{1.0, 2.0}, new long[]{2}).castTo(DataType.FLOAT);
        SparseNDArray coo = new SparseNDArray(indices, values, new long[]{2, 2}, SparseFormat.COO);
        assertThrows(Exception.class, coo::colSums, "colSums() must throw when format is COO");
    }

    @Test
    @DisplayName("scale() throws when format is not CSR")
    public void testScaleThrowsIfNotCsr() {
        INDArray indices = Nd4j.create(new double[]{0, 0, 1, 1}, new long[]{2, 2}).castTo(DataType.INT64);
        INDArray values  = Nd4j.create(new double[]{1.0, 2.0}, new long[]{2}).castTo(DataType.FLOAT);
        SparseNDArray coo = new SparseNDArray(indices, values, new long[]{2, 2}, SparseFormat.COO);
        assertThrows(Exception.class, () -> coo.scale(2.0), "scale() must throw when format is COO");
    }

    @Test
    @DisplayName("sumNumber() throws when format is not CSR")
    public void testSumNumberThrowsIfNotCsr() {
        INDArray indices = Nd4j.create(new double[]{0, 0, 1, 1}, new long[]{2, 2}).castTo(DataType.INT64);
        INDArray values  = Nd4j.create(new double[]{1.0, 2.0}, new long[]{2}).castTo(DataType.FLOAT);
        SparseNDArray coo = new SparseNDArray(indices, values, new long[]{2, 2}, SparseFormat.COO);
        assertThrows(Exception.class, coo::sumNumber, "sumNumber() must throw when format is COO");
    }
}
