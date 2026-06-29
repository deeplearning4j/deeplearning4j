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
import org.nd4j.linalg.api.ops.impl.sparse.BsrSpmm;
import org.nd4j.linalg.api.ops.impl.sparse.BsrToDense;
import org.nd4j.linalg.api.ops.impl.sparse.CsrToBsr;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Round-trip, correctness, structure, and edge-case tests for BSR (Block Sparse Row)
 * sparse tensor support via {@link SparseNDArray#toBsr(int)} and
 * {@link SparseNDArray#toDense()}.
 *
 * <p>All matrices used for BSR conversion have block-aligned dimensions (rows and cols
 * are both exact multiples of {@code blockDim}).
 *
 * <p>Test coverage:
 * <ul>
 *   <li>(i)  Round-trip: {@code Nd4j.toSparse(A, CSR).toBsr(bd).toDense() ≈ A}</li>
 *   <li>(ii) SpMM: {@code A_bsr.mmul(B) ≈ A_dense.mmul(B)}</li>
 *   <li>(iii) BSR structure validity: bsrRowPtr monotonic, bsrColIdx in range,
 *             bsrValues length = nnzb * bd * bd</li>
 *   <li>(iv)  Edge cases: all-zero, fully-dense, partially-sparse (zero blocks absent)</li>
 *   <li>(v)   Precondition: non-block-aligned dims throw</li>
 * </ul>
 */
@DisplayName("BSR Sparse Round-Trip, SpMM, and Structure Tests")
public class SparseBsrTest {

    private static final double TOL_FLOAT  = 1e-5;
    private static final double TOL_DOUBLE = 1e-10;

    // -----------------------------------------------------------------------
    // Utility helpers
    // -----------------------------------------------------------------------

    /** Element-wise equality check within tol for 2-D matrices. */
    private static void assertMatrixEquals(INDArray expected, INDArray actual,
                                           double tol, String msg) {
        assertEquals(expected.shape()[0], actual.shape()[0], msg + ": rows mismatch");
        assertEquals(expected.shape()[1], actual.shape()[1], msg + ": cols mismatch");
        long rows = expected.shape()[0];
        long cols = expected.shape()[1];
        for (long r = 0; r < rows; r++) {
            for (long c = 0; c < cols; c++) {
                assertEquals(expected.getDouble(r, c), actual.getDouble(r, c), tol,
                        msg + ": mismatch at [" + r + "," + c + "]");
            }
        }
    }

    /**
     * Assert internal consistency of a BSR SparseNDArray:
     * <ul>
     *   <li>format is BSR</li>
     *   <li>bsrRowPtr length == mb+1, bsrRowPtr[0] == 0, non-decreasing</li>
     *   <li>bsrColIdx all in [0, nb) where nb = cols/blockDim</li>
     *   <li>bsrValues length == nnzb * blockDim * blockDim</li>
     * </ul>
     */
    private static void assertBsrValid(SparseNDArray bsr) {
        assertEquals(SparseFormat.BSR, bsr.getFormat(), "format must be BSR");
        int bd     = bsr.getBlockDim();
        long rows  = bsr.rows();
        long cols  = bsr.cols();
        long mb    = rows / bd;   // number of block-rows
        long nb    = cols / bd;   // number of block-cols

        INDArray bsrRowPtr = bsr.getBsrRowPtr();
        INDArray bsrColIdx = bsr.getBsrColIdx();
        INDArray bsrValues = bsr.getBsrValues();

        // bsrRowPtr length == mb+1
        assertEquals(mb + 1, bsrRowPtr.length(),
                "bsrRowPtr length must be mb+1=" + (mb + 1) + " (mb=rows/blockDim)");

        // bsrRowPtr[0] == 0
        assertEquals(0, bsrRowPtr.getInt(0), "bsrRowPtr[0] must be 0");

        // bsrRowPtr non-decreasing
        for (int br = 0; br < mb; br++) {
            assertTrue(bsrRowPtr.getInt(br) <= bsrRowPtr.getInt(br + 1),
                    "bsrRowPtr must be non-decreasing at block-row " + br);
        }

        // nnzb from bsrRowPtr[mb]
        long nnzb = bsrRowPtr.getInt((int) mb);

        // bsrColIdx length == nnzb
        assertEquals(nnzb, bsrColIdx.length(),
                "bsrColIdx length must equal nnzb=" + nnzb);

        // bsrValues length == nnzb * bd * bd
        assertEquals(nnzb * (long) bd * bd, bsrValues.length(),
                "bsrValues length must equal nnzb*bd*bd=" + (nnzb * bd * bd));

        // every bsrColIdx[k] in [0, nb)
        for (long k = 0; k < nnzb; k++) {
            int bc = bsrColIdx.getInt((int) k);
            assertTrue(bc >= 0 && bc < nb,
                    "bsrColIdx[" + k + "]=" + bc + " out of range [0," + nb + ")");
        }
    }

    /**
     * Build a block-sparse dense matrix: blocks are included or excluded at block-granularity.
     * {@code blockInclude[br][bc]} controls whether block (br, bc) is non-zero.
     */
    private static INDArray buildBlockSparse(int rows, int cols, int bd,
                                             boolean[][] blockInclude, DataType dtype) {
        float[] data = new float[rows * cols];
        int mb = rows / bd;
        int nb = cols / bd;
        float val = 1.0f;
        for (int br = 0; br < mb; br++) {
            for (int bc = 0; bc < nb; bc++) {
                if (blockInclude[br][bc]) {
                    // fill all bd*bd elements of this block with distinct values
                    for (int dr = 0; dr < bd; dr++) {
                        for (int dc = 0; dc < bd; dc++) {
                            int r = br * bd + dr;
                            int c = bc * bd + dc;
                            data[r * cols + c] = val++;
                        }
                    }
                }
            }
        }
        INDArray dense = Nd4j.create(data, new long[]{rows, cols}, DataType.FLOAT);
        if (dtype == DataType.DOUBLE) return dense.castTo(DataType.DOUBLE);
        return dense;
    }

    // -----------------------------------------------------------------------
    // (i) Round-trip: CSR → BSR → dense ≈ original
    // -----------------------------------------------------------------------

    @Test
    @DisplayName("(i) round-trip 4×4 bd=2 FLOAT — all blocks non-zero")
    public void testRoundTrip4x4bd2Float_allBlocks() {
        // 4 rows, 4 cols, bd=2 → 2×2 block grid; all 4 blocks occupied
        boolean[][] inc = {{true, true}, {true, true}};
        INDArray dense = buildBlockSparse(4, 4, 2, inc, DataType.FLOAT);

        SparseNDArray bsr = Nd4j.toSparse(dense, SparseFormat.CSR).toBsr(2);
        assertBsrValid(bsr);
        assertMatrixEquals(dense, bsr.toDense(), TOL_FLOAT, "4×4 bd=2 FLOAT all-blocks round-trip");
    }

    @Test
    @DisplayName("(i) round-trip 4×4 bd=2 FLOAT — partial block sparsity")
    public void testRoundTrip4x4bd2Float_partial() {
        // Only blocks (0,1) and (1,0) are non-zero
        boolean[][] inc = {{false, true}, {true, false}};
        INDArray dense = buildBlockSparse(4, 4, 2, inc, DataType.FLOAT);

        SparseNDArray bsr = Nd4j.toSparse(dense, SparseFormat.CSR).toBsr(2);
        assertBsrValid(bsr);
        assertMatrixEquals(dense, bsr.toDense(), TOL_FLOAT, "4×4 bd=2 FLOAT partial round-trip");
    }

    @Test
    @DisplayName("(i) round-trip 6×4 bd=2 DOUBLE")
    public void testRoundTrip6x4bd2Double() {
        // 6 rows, 4 cols, bd=2 → 3×2 block grid
        boolean[][] inc = {{true, false}, {false, true}, {true, true}};
        INDArray dense = buildBlockSparse(6, 4, 2, inc, DataType.DOUBLE);

        SparseNDArray bsr = Nd4j.toSparse(dense, SparseFormat.CSR).toBsr(2);
        assertBsrValid(bsr);
        assertEquals(DataType.DOUBLE, bsr.dataType(), "dataType must be DOUBLE");
        assertMatrixEquals(dense, bsr.toDense(), TOL_DOUBLE, "6×4 bd=2 DOUBLE round-trip");
    }

    @Test
    @DisplayName("(i) round-trip 8×8 bd=4 FLOAT")
    public void testRoundTrip8x8bd4Float() {
        // 8 rows, 8 cols, bd=4 → 2×2 block grid; only (0,0) and (1,1) occupied
        boolean[][] inc = {{true, false}, {false, true}};
        INDArray dense = buildBlockSparse(8, 8, 4, inc, DataType.FLOAT);

        SparseNDArray bsr = Nd4j.toSparse(dense, SparseFormat.CSR).toBsr(4);
        assertBsrValid(bsr);
        assertMatrixEquals(dense, bsr.toDense(), TOL_FLOAT, "8×8 bd=4 FLOAT round-trip");
    }

    // -----------------------------------------------------------------------
    // (ii) SpMM: A_bsr.mmul(B) ≈ A_dense.mmul(B)
    // -----------------------------------------------------------------------

    @Test
    @DisplayName("(ii) SpMM 4×4 bd=2 FLOAT, B[4,2]")
    public void testSpmm4x4bd2Float_n2() {
        boolean[][] inc = {{true, false}, {false, true}};
        INDArray aDense = buildBlockSparse(4, 4, 2, inc, DataType.FLOAT);
        INDArray B = Nd4j.rand(DataType.FLOAT, 4, 2);

        SparseNDArray bsr = Nd4j.toSparse(aDense, SparseFormat.CSR).toBsr(2);
        INDArray cBsr   = bsr.mmul(B);
        INDArray cDense = aDense.mmul(B);

        assertMatrixEquals(cDense, cBsr, TOL_FLOAT, "SpMM 4×4 bd=2 FLOAT n=2");
    }

    @Test
    @DisplayName("(ii) SpMM 6×4 bd=2 FLOAT, B[4,3]")
    public void testSpmm6x4bd2Float_n3() {
        boolean[][] inc = {{true, true}, {false, true}, {true, false}};
        INDArray aDense = buildBlockSparse(6, 4, 2, inc, DataType.FLOAT);
        INDArray B = Nd4j.rand(DataType.FLOAT, 4, 3);

        SparseNDArray bsr = Nd4j.toSparse(aDense, SparseFormat.CSR).toBsr(2);
        INDArray cBsr   = bsr.mmul(B);
        INDArray cDense = aDense.mmul(B);

        assertMatrixEquals(cDense, cBsr, TOL_FLOAT, "SpMM 6×4 bd=2 FLOAT n=3");
    }

    @Test
    @DisplayName("(ii) SpMM 8×8 bd=4 DOUBLE, B[8,2]")
    public void testSpmm8x8bd4Double_n2() {
        boolean[][] inc = {{true, true}, {true, false}};
        INDArray aDense = buildBlockSparse(8, 8, 4, inc, DataType.DOUBLE);
        INDArray B = Nd4j.rand(DataType.DOUBLE, 8, 2);

        SparseNDArray bsr = Nd4j.toSparse(aDense, SparseFormat.CSR).toBsr(4);
        INDArray cBsr   = bsr.mmul(B);
        INDArray cDense = aDense.mmul(B);

        assertMatrixEquals(cDense, cBsr, TOL_DOUBLE, "SpMM 8×8 bd=4 DOUBLE n=2");
    }

    @Test
    @DisplayName("(ii) SpMM 4×4 bd=2 DOUBLE, B[4,3]")
    public void testSpmm4x4bd2Double_n3() {
        boolean[][] inc = {{true, false}, {true, true}};
        INDArray aDense = buildBlockSparse(4, 4, 2, inc, DataType.DOUBLE);
        INDArray B = Nd4j.rand(DataType.DOUBLE, 4, 3);

        SparseNDArray bsr = Nd4j.toSparse(aDense, SparseFormat.CSR).toBsr(2);
        INDArray cBsr   = bsr.mmul(B);
        INDArray cDense = aDense.mmul(B);

        assertMatrixEquals(cDense, cBsr, TOL_DOUBLE, "SpMM 4×4 bd=2 DOUBLE n=3");
    }

    // -----------------------------------------------------------------------
    // (iii) Structure validity: covered by assertBsrValid in every (i)/(ii) test;
    //       these standalone tests make the structural contract explicit.
    // -----------------------------------------------------------------------

    @Test
    @DisplayName("(iii) bsrRowPtr[0]==0, monotonic, length mb+1 — 6×4 bd=2")
    public void testStructureRowPtr6x4bd2() {
        boolean[][] inc = {{true, false}, {true, true}, {false, true}};
        INDArray dense = buildBlockSparse(6, 4, 2, inc, DataType.FLOAT);
        SparseNDArray bsr = Nd4j.toSparse(dense, SparseFormat.CSR).toBsr(2);
        assertBsrValid(bsr);   // asserts all structural invariants

        // mb = 6/2 = 3  → bsrRowPtr length must be 4
        assertEquals(4, bsr.getBsrRowPtr().length(), "bsrRowPtr length for 6×4 bd=2 must be 4");
    }

    @Test
    @DisplayName("(iii) bsrColIdx all in [0, nb) — 8×8 bd=4")
    public void testStructureColIdx8x8bd4() {
        boolean[][] inc = {{false, true}, {true, false}};
        INDArray dense = buildBlockSparse(8, 8, 4, inc, DataType.FLOAT);
        SparseNDArray bsr = Nd4j.toSparse(dense, SparseFormat.CSR).toBsr(4);
        // nb = 8/4 = 2 → every bsrColIdx value in {0, 1}
        assertBsrValid(bsr);
    }

    @Test
    @DisplayName("(iii) bsrValues length == nnzb * bd * bd — 4×4 bd=2")
    public void testStructureValuesLength4x4bd2() {
        // 3 of the 4 blocks are non-zero → nnzb=3, expected values length = 3*2*2=12
        boolean[][] inc = {{true, true}, {true, false}};
        INDArray dense = buildBlockSparse(4, 4, 2, inc, DataType.FLOAT);
        SparseNDArray bsr = Nd4j.toSparse(dense, SparseFormat.CSR).toBsr(2);
        assertBsrValid(bsr);
        long nnzb = bsr.getBsrRowPtr().getInt(2);   // bsrRowPtr[mb] = bsrRowPtr[2]
        assertEquals(nnzb * 2 * 2, bsr.getBsrValues().length(),
                "bsrValues length must equal nnzb*bd*bd=" + (nnzb * 4));
    }

    // -----------------------------------------------------------------------
    // (iv) Edge cases
    // -----------------------------------------------------------------------

    @Test
    @DisplayName("(iv) all-zero 4×4 bd=2 — nnzb=0, round-trip to zeros")
    public void testEdgeAllZero() {
        INDArray dense = Nd4j.zeros(DataType.FLOAT, 4, 4);

        SparseNDArray bsr = Nd4j.toSparse(dense, SparseFormat.CSR).toBsr(2);
        assertBsrValid(bsr);

        // All block-row pointers must be 0 (nnzb=0)
        assertEquals(0, bsr.getBsrRowPtr().getInt(2), "bsrRowPtr[mb] must be 0 for all-zero matrix");
        assertEquals(0, bsr.getBsrColIdx().length(), "bsrColIdx must be empty for all-zero matrix");
        assertEquals(0, bsr.getBsrValues().length(), "bsrValues must be empty for all-zero matrix");

        // Round-trip: toDense recovers the all-zero matrix
        assertMatrixEquals(dense, bsr.toDense(), TOL_FLOAT, "all-zero BSR round-trip");
    }

    @Test
    @DisplayName("(iv) fully-dense 4×4 bd=2 — nnzb=4, all blocks present")
    public void testEdgeFullyDense() {
        // All elements are non-zero → all 4 blocks occupied
        INDArray dense = Nd4j.ones(DataType.FLOAT, 4, 4).add(1.0f);

        SparseNDArray bsr = Nd4j.toSparse(dense, SparseFormat.CSR).toBsr(2);
        assertBsrValid(bsr);

        long nnzb = bsr.getBsrRowPtr().getInt(2);   // bsrRowPtr[mb=2]
        assertEquals(4, nnzb, "fully-dense 4×4 bd=2 must have nnzb=4");
        assertEquals(4L * 2 * 2, bsr.getBsrValues().length(),
                "bsrValues length must be nnzb*bd*bd=16");

        assertMatrixEquals(dense, bsr.toDense(), TOL_FLOAT, "fully-dense BSR round-trip");
    }

    @Test
    @DisplayName("(iv) some zero blocks absent — only non-empty blocks in BSR")
    public void testEdgeSomeZeroBlocksAbsent() {
        // 4×4, bd=2: blocks (0,0) and (1,1) non-zero; blocks (0,1) and (1,0) zero
        boolean[][] inc = {{true, false}, {false, true}};
        INDArray dense = buildBlockSparse(4, 4, 2, inc, DataType.FLOAT);

        SparseNDArray bsr = Nd4j.toSparse(dense, SparseFormat.CSR).toBsr(2);
        assertBsrValid(bsr);

        long nnzb = bsr.getBsrRowPtr().getInt(2);   // bsrRowPtr[mb=2]
        assertEquals(2, nnzb,
                "only the 2 non-empty blocks should be stored (nnzb=2)");

        // Verify bsrColIdx entries: block-col 0 for block-row 0, block-col 1 for block-row 1
        // bsrRowPtr should be [0, 1, 2]
        assertEquals(0, bsr.getBsrRowPtr().getInt(0), "bsrRowPtr[0]=0");
        assertEquals(1, bsr.getBsrRowPtr().getInt(1), "bsrRowPtr[1]=1 (one block in block-row 0)");
        assertEquals(2, bsr.getBsrRowPtr().getInt(2), "bsrRowPtr[2]=2 (one block in block-row 1)");

        // Round-trip recovers original
        assertMatrixEquals(dense, bsr.toDense(), TOL_FLOAT, "partial-block BSR round-trip");
    }

    // -----------------------------------------------------------------------
    // (v) Precondition: non-block-aligned dimensions throw
    // -----------------------------------------------------------------------

    @Test
    @DisplayName("(v) non-block-aligned rows (5×4 bd=2) → toBsr throws")
    public void testPreconditionNonAlignedRows() {
        // 5 rows is NOT divisible by bd=2 → the Java Preconditions.checkArgument in toBsr()
        // must throw before even reaching the native op.
        INDArray dense = Nd4j.rand(DataType.FLOAT, 5, 4);
        SparseNDArray csr = Nd4j.toSparse(dense, SparseFormat.CSR);
        assertThrows(Exception.class, () -> csr.toBsr(2),
                "toBsr with rows=5, bd=2 (non-aligned) must throw");
    }

    @Test
    @DisplayName("(v) non-block-aligned cols (4×5 bd=2) → toBsr throws")
    public void testPreconditionNonAlignedCols() {
        // 5 cols is NOT divisible by bd=2
        INDArray dense = Nd4j.rand(DataType.FLOAT, 4, 5);
        SparseNDArray csr = Nd4j.toSparse(dense, SparseFormat.CSR);
        assertThrows(Exception.class, () -> csr.toBsr(2),
                "toBsr with cols=5, bd=2 (non-aligned) must throw");
    }

    @Test
    @DisplayName("(v) blockDim=0 → toBsr throws")
    public void testPreconditionZeroBlockDim() {
        INDArray dense = Nd4j.rand(DataType.FLOAT, 4, 4);
        SparseNDArray csr = Nd4j.toSparse(dense, SparseFormat.CSR);
        assertThrows(Exception.class, () -> csr.toBsr(0),
                "toBsr with blockDim=0 must throw");
    }

    // -----------------------------------------------------------------------
    // (vi) BSR accessor guards: BSR-only getters throw on non-BSR instances
    // -----------------------------------------------------------------------

    @Test
    @DisplayName("BSR accessor guards: getBsrValues/ColIdx/RowPtr/getBlockDim throw on CSR")
    public void testBsrAccessorGuards() {
        INDArray dense = Nd4j.rand(DataType.FLOAT, 4, 4);
        SparseNDArray csr = Nd4j.toSparse(dense, SparseFormat.CSR);

        assertThrows(Exception.class, csr::getBsrValues,  "getBsrValues() on CSR must throw");
        assertThrows(Exception.class, csr::getBsrColIdx,  "getBsrColIdx() on CSR must throw");
        assertThrows(Exception.class, csr::getBsrRowPtr,  "getBsrRowPtr() on CSR must throw");
        assertThrows(Exception.class, csr::getBlockDim,   "getBlockDim() on CSR must throw");
    }

    // -----------------------------------------------------------------------
    // (vii) Direct op API
    // -----------------------------------------------------------------------

    @Test
    @DisplayName("CsrToBsr op directly — known 4×4 matrix")
    public void testCsrToBsrOpDirectly() {
        // [[1,0,0,0],[0,0,2,0],[0,0,0,3],[4,0,0,0]] — bd=2
        // CSR: values=[1,2,3,4], colIdx=[0,2,3,0], rowPtr=[0,1,2,3,4]
        INDArray values = Nd4j.createFromArray(new float[]{1f, 2f, 3f, 4f});
        INDArray colIdx = Nd4j.createFromArray(new int[]{0, 2, 3, 0});
        INDArray rowPtr = Nd4j.createFromArray(new int[]{0, 1, 2, 3, 4});

        CsrToBsr op = new CsrToBsr(values, colIdx, rowPtr, 4, 4, 2);
        INDArray[] results = Nd4j.exec(op);
        INDArray bsrValues = results[0];
        INDArray bsrColIdx = results[1];
        INDArray bsrRowPtr = results[2];

        // bsrRowPtr length = mb+1 = 3
        assertEquals(3, bsrRowPtr.length(), "bsrRowPtr length must be mb+1=3");
        // bsrRowPtr[0] == 0
        assertEquals(0, bsrRowPtr.getInt(0), "bsrRowPtr[0] must be 0");
        // nnzb = bsrRowPtr[mb=2]
        long nnzb = bsrRowPtr.getInt(2);
        assertTrue(nnzb > 0, "nnzb must be > 0 for a matrix with non-zeros");
        // bsrValues length == nnzb * bd^2
        assertEquals(nnzb * 4, bsrValues.length(), "bsrValues length must be nnzb*4");
    }

    @Test
    @DisplayName("BsrToDense op directly — reconstruct known 4×4 matrix")
    public void testBsrToDenseOpDirectly() {
        // Build a known BSR for [[1,2,0,0],[3,4,0,0],[0,0,5,6],[0,0,7,8]]
        // bd=2: block-row 0 has block at block-col 0 (values [1,2,3,4])
        //        block-row 1 has block at block-col 1 (values [5,6,7,8])
        INDArray bsrValues = Nd4j.createFromArray(new float[]{1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f});
        INDArray bsrColIdx = Nd4j.createFromArray(new int[]{0, 1});
        INDArray bsrRowPtr = Nd4j.createFromArray(new int[]{0, 1, 2});

        BsrToDense op = new BsrToDense(bsrValues, bsrColIdx, bsrRowPtr, 4, 4, 2);
        INDArray[] results = Nd4j.exec(op);
        INDArray dense = results[0];

        assertEquals(4L, dense.shape()[0], "rows");
        assertEquals(4L, dense.shape()[1], "cols");
        assertEquals(1.0, dense.getDouble(0, 0), TOL_FLOAT, "[0,0]");
        assertEquals(2.0, dense.getDouble(0, 1), TOL_FLOAT, "[0,1]");
        assertEquals(3.0, dense.getDouble(1, 0), TOL_FLOAT, "[1,0]");
        assertEquals(4.0, dense.getDouble(1, 1), TOL_FLOAT, "[1,1]");
        assertEquals(0.0, dense.getDouble(0, 2), TOL_FLOAT, "[0,2] must be zero");
        assertEquals(0.0, dense.getDouble(0, 3), TOL_FLOAT, "[0,3] must be zero");
        assertEquals(5.0, dense.getDouble(2, 2), TOL_FLOAT, "[2,2]");
        assertEquals(6.0, dense.getDouble(2, 3), TOL_FLOAT, "[2,3]");
        assertEquals(7.0, dense.getDouble(3, 2), TOL_FLOAT, "[3,2]");
        assertEquals(8.0, dense.getDouble(3, 3), TOL_FLOAT, "[3,3]");
    }

    @Test
    @DisplayName("BsrSpmm op directly — known 4×4 · B[4,1]")
    public void testBsrSpmmOpDirectly() {
        // A = [[1,2,0,0],[3,4,0,0],[0,0,5,6],[0,0,7,8]], B = [[1],[0],[1],[0]]
        // A·B = [[1],[3],[5],[7]]
        INDArray bsrValues = Nd4j.createFromArray(new float[]{1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f});
        INDArray bsrColIdx = Nd4j.createFromArray(new int[]{0, 1});
        INDArray bsrRowPtr = Nd4j.createFromArray(new int[]{0, 1, 2});
        INDArray B = Nd4j.createFromArray(new float[]{1f, 0f, 1f, 0f}).reshape(4, 1);

        BsrSpmm op = new BsrSpmm(bsrValues, bsrColIdx, bsrRowPtr, B, 4, 4, 2);
        INDArray[] results = Nd4j.exec(op);
        INDArray C = results[0];

        assertEquals(4L, C.shape()[0], "C rows");
        assertEquals(1L, C.shape()[1], "C cols");
        assertEquals(1.0, C.getDouble(0, 0), TOL_FLOAT, "C[0,0]");
        assertEquals(3.0, C.getDouble(1, 0), TOL_FLOAT, "C[1,0]");
        assertEquals(5.0, C.getDouble(2, 0), TOL_FLOAT, "C[2,0]");
        assertEquals(7.0, C.getDouble(3, 0), TOL_FLOAT, "C[3,0]");
    }
}
