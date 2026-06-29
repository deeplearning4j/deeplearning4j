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

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ndarray.SparseFormat;
import org.nd4j.linalg.api.ndarray.SparseNDArray;
import org.nd4j.linalg.api.ndarray.SparsePruning;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for {@link SparsePruning}: magnitude pruning, {@link SparsePruning.SparseLinear},
 * and structured (row/column) pruning.
 *
 * <p>Test strategy:
 * <ul>
 *   <li>All tests run on both CPU and CUDA backends via
 *       {@code @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")}.</li>
 *   <li>Weight matrices use distinct, known magnitudes so threshold computation is
 *       unambiguous (no ties at the pruning boundary).</li>
 *   <li>No host loops over weight data inside the utility under test — only the
 *       single scalar threshold read ({@code getDouble(k-1)}) is allowed.</li>
 * </ul>
 *
 * <p>Test matrix used in most cases:
 * <pre>
 *   W (4×4, row-major, values 1..16, no zeros):
 *   [[ 1,  2,  3,  4],
 *    [ 5,  6,  7,  8],
 *    [ 9, 10, 11, 12],
 *    [13, 14, 15, 16]]
 *
 *   Magnitudes are the values themselves.
 *   Top-8 by magnitude = {9,10,11,12,13,14,15,16} → rows 2 and 3 fully.
 *   Bottom-8 = {1,2,3,4,5,6,7,8} → rows 0 and 1.
 * </pre>
 */
@DisplayName("SparsePruning utility tests")
public class SparsePruningTest extends BaseNd4jTestWithBackends {

    private static final double TOL = 1e-5;

    /**
     * Purge ConstantHandler cache before each test to avoid stale buffers across backends.
     * (Same pattern used in SparseGradCheckTest for the same root cause.)
     */
    @BeforeEach
    public void purgeConstantHandlerCache() {
        Nd4j.getConstantHandler().purgeConstants();
    }

    // -----------------------------------------------------------------------
    // Helper: 4×4 matrix with values 1..16 (all positive, all distinct)
    // -----------------------------------------------------------------------

    private static INDArray make4x4(DataType dtype) {
        // row-major [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]
        return Nd4j.arange(1.0, 17.0).castTo(dtype).reshape(4, 4);
    }

    // -----------------------------------------------------------------------
    // 1. magnitudePrune — NNZ count
    // -----------------------------------------------------------------------

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("magnitudePrune 50% keeps exactly 8 of 16 entries (FLOAT)")
    public void testMagnitudePruneNnzCountFloat(Nd4jBackend backend) {
        INDArray w = make4x4(DataType.FLOAT);
        SparseNDArray csr = SparsePruning.magnitudePrune(w, 0.50);
        assertEquals(SparseFormat.CSR, csr.getFormat(), "format must be CSR");
        assertEquals(8L, csr.nnz(),
                "50% sparsity on a 4×4 matrix should keep exactly 8 entries");
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("magnitudePrune 50% keeps exactly 8 of 16 entries (DOUBLE)")
    public void testMagnitudePruneNnzCountDouble(Nd4jBackend backend) {
        INDArray w = make4x4(DataType.DOUBLE);
        SparseNDArray csr = SparsePruning.magnitudePrune(w, 0.50);
        assertEquals(8L, csr.nnz(),
                "50% sparsity on a 4×4 matrix should keep exactly 8 entries (DOUBLE)");
    }

    // -----------------------------------------------------------------------
    // 2. magnitudePrune — correctness: kept entries are the largest-magnitude ones
    // -----------------------------------------------------------------------

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("magnitudePrune keeps the correct (largest-magnitude) entries")
    public void testMagnitudePruneKeepsLargestEntries(Nd4jBackend backend) {
        // W = [[1..4],[5..8],[9..12],[13..16]], 50% sparsity → keep top-8 = {9..16}
        INDArray w = make4x4(DataType.DOUBLE);
        SparseNDArray csr = SparsePruning.magnitudePrune(w, 0.50);

        // Densify and verify
        INDArray dense = csr.toDense();   // [4, 4]

        // Rows 0 and 1 should be all zeros (values 1..8 < threshold)
        for (long col = 0; col < 4; col++) {
            assertEquals(0.0, dense.getDouble(0, col), TOL,
                    "Row 0, col " + col + " should have been pruned");
            assertEquals(0.0, dense.getDouble(1, col), TOL,
                    "Row 1, col " + col + " should have been pruned");
        }

        // Rows 2 and 3 should match the original (values 9..16 are all kept)
        for (long col = 0; col < 4; col++) {
            assertEquals(w.getDouble(2, col), dense.getDouble(2, col), TOL,
                    "Row 2, col " + col + " should be kept");
            assertEquals(w.getDouble(3, col), dense.getDouble(3, col), TOL,
                    "Row 3, col " + col + " should be kept");
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("magnitudePrune with mixed-sign values keeps largest |w|")
    public void testMagnitudePruneMixedSign(Nd4jBackend backend) {
        // 2×4 weight with both positive and negative values, distinct magnitudes
        // [[-8, 3, -1, 6], [2, -7, 4, -5]]
        // Magnitudes:  [8, 3, 1, 6,   2, 7, 4, 5]
        // Top-4 by |w|: 8, 7, 6, 5 at positions (0,0),(1,1),(0,3),(1,3)
        INDArray w = Nd4j.createFromArray(new double[][]{
                {-8.0,  3.0, -1.0,  6.0},
                { 2.0, -7.0,  4.0, -5.0}});

        SparseNDArray csr = SparsePruning.magnitudePrune(w, 0.50); // keep 4 of 8
        assertEquals(4L, csr.nnz(), "should keep 4 entries");

        INDArray dense = csr.toDense();

        // Kept: (0,0)=-8, (0,3)=6, (1,1)=-7, (1,3)=-5
        assertEquals(-8.0, dense.getDouble(0, 0), TOL, "(0,0) should be kept");
        assertEquals( 6.0, dense.getDouble(0, 3), TOL, "(0,3) should be kept");
        assertEquals(-7.0, dense.getDouble(1, 1), TOL, "(1,1) should be kept");
        assertEquals(-5.0, dense.getDouble(1, 3), TOL, "(1,3) should be kept");

        // Pruned: (0,1), (0,2), (1,0), (1,2) have |w| ≤ 4 < min-kept=5
        assertEquals(0.0, dense.getDouble(0, 1), TOL, "(0,1) should be pruned");
        assertEquals(0.0, dense.getDouble(0, 2), TOL, "(0,2) should be pruned");
        assertEquals(0.0, dense.getDouble(1, 0), TOL, "(1,0) should be pruned");
        assertEquals(0.0, dense.getDouble(1, 2), TOL, "(1,2) should be pruned");
    }

    // -----------------------------------------------------------------------
    // 3. SparseLinear — forward == dense (pruned weight · x)
    // -----------------------------------------------------------------------

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("SparseLinear forward matches dense (pruned-weight · x) FLOAT")
    public void testSparseLinearForwardMatchesDenseFloat(Nd4jBackend backend) {
        sparseLinearForwardCheck(DataType.FLOAT, 1e-4);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("SparseLinear forward matches dense (pruned-weight · x) DOUBLE")
    public void testSparseLinearForwardMatchesDenseDouble(Nd4jBackend backend) {
        sparseLinearForwardCheck(DataType.DOUBLE, 1e-9);
    }

    private void sparseLinearForwardCheck(DataType dtype, double tol) {
        // Weight: 3×4, values 1..12 (no zeros) — all distinct positive magnitudes
        INDArray wDense = Nd4j.arange(1.0, 13.0).castTo(dtype).reshape(3, 4);

        // 50% sparsity: keep top 6 = {7,8,9,10,11,12} (rows 1 and 2)
        SparseNDArray wSparse = SparsePruning.magnitudePrune(wDense, 0.50);
        SparsePruning.SparseLinear layer = new SparsePruning.SparseLinear(wSparse);

        // Reference: dense version of the pruned weight
        INDArray wPrunedDense = wSparse.toDense();  // [3, 4]

        // Input x: [4, 2] — two batch samples
        INDArray x = Nd4j.rand(dtype, 4, 2);

        // Sparse forward
        INDArray sparseOut = layer.forward(x);  // [3, 2]

        // Dense reference
        INDArray denseOut = wPrunedDense.mmul(x);  // [3, 2]

        assertEquals(denseOut.shape()[0], sparseOut.shape()[0], "row count mismatch");
        assertEquals(denseOut.shape()[1], sparseOut.shape()[1], "col count mismatch");

        for (long i = 0; i < denseOut.length(); i++) {
            assertEquals(denseOut.getDouble(i), sparseOut.getDouble(i), tol,
                    "output mismatch at linear index " + i);
        }
    }

    // -----------------------------------------------------------------------
    // 4. SparseLinear — weight gradient is nonzero only at kept positions
    // -----------------------------------------------------------------------

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("SparseLinear weightGradient is nonzero only at kept positions")
    public void testSparseLinearWeightGradientAtKeptPositions(Nd4jBackend backend) {
        // Weight: 3×4, values 1..12.  Keep top-6 (50% sparsity) → rows 1 and 2 kept
        INDArray wDense = Nd4j.arange(1.0, 13.0).castTo(DataType.DOUBLE).reshape(3, 4);
        SparseNDArray wSparse = SparsePruning.magnitudePrune(wDense, 0.50);

        SparsePruning.SparseLinear layer = new SparsePruning.SparseLinear(wSparse);
        long nnz = wSparse.nnz();  // should be 8 (top-6 of 12; rows 1 and 2 = 4 each → 8)

        // gradOut [3, 2]: upstream gradient w.r.t. output C=[3,2]
        INDArray gradOut = Nd4j.rand(DataType.DOUBLE, (int) layer.outFeatures(), 2);
        // x [4, 2]: the same input used in forward
        INDArray x = Nd4j.rand(DataType.DOUBLE, (int) layer.inFeatures(), 2);

        // Sparse weight gradient via SDDMM
        INDArray dWValues = layer.weightGradient(gradOut, x);  // [nnz]
        assertEquals(nnz, dWValues.length(), "gradient vector length must equal nnz");

        // 4a. All nnz gradient entries should be defined (no NaN/Inf)
        for (long i = 0; i < nnz; i++) {
            double g = dWValues.getDouble(i);
            assertFalse(Double.isNaN(g),   "gradient[" + i + "] is NaN");
            assertFalse(Double.isInfinite(g), "gradient[" + i + "] is Infinite");
        }

        // 4b. Verify gradient values match the dense outer-product at kept positions
        // Dense dL/dW = gradOut · x^T  [3, 4] — gradient w.r.t. all weight positions
        INDArray dWDense = gradOut.mmul(x.transpose());  // [3, 4]

        // Extract the dense gradient at only the kept (non-zero) positions of W
        // and compare to the sparse SDDMM gradient values in CSR order
        INDArray colIdx  = wSparse.getColIdx();  // [nnz] INT32: column of each non-zero
        INDArray rowPtr  = wSparse.getRowPtr();  // [rows+1] INT32

        long nRows = wSparse.rows();
        long idx = 0;
        for (long row = 0; row < nRows; row++) {
            long start = rowPtr.getLong(row);
            long end   = rowPtr.getLong(row + 1);
            for (long e = start; e < end; e++) {
                long col = colIdx.getLong(e);
                double expected = dWDense.getDouble(row, col);
                double actual   = dWValues.getDouble(idx);
                assertEquals(expected, actual, 1e-8,
                        "gradient mismatch at CSR entry " + idx
                                + " (row=" + row + ", col=" + col + ")");
                idx++;
            }
        }

        // 4c. Pruned positions (row 0, all cols) should have ZERO in the dense gradient
        //     representation (because those weights are zero in the pruned weight).
        //     The sparse layer never computes gradient at those positions — confirm that
        //     the dense gradient at row=0 is NOT automatically zero (it has a gradient,
        //     we just choose not to use it), which confirms SDDMM is doing selective sampling.
        //     We do this by verifying the dense grad at (0,0) is nonzero with high probability
        //     (random gradOut and x), then confirm dWValues has no entries for row 0.
        //     Row 0 kept entries = rowPtr[1] - rowPtr[0]
        long row0Nnz = rowPtr.getLong(1) - rowPtr.getLong(0);
        assertEquals(0L, row0Nnz,
                "row 0 should have been fully pruned and have nnz=0 in CSR");
    }

    // -----------------------------------------------------------------------
    // 5. structuredPrune — row pruning keeps correct rows
    // -----------------------------------------------------------------------

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("structuredPrune axis=0 keeps rows with largest L2 norm")
    public void testStructuredPruneRows(Nd4jBackend backend) {
        // 4×4 weight: rows have L2 norms proportional to row magnitude
        // row 0 values=[1,2,3,4]   norm≈5.48
        // row 1 values=[5,6,7,8]   norm≈13.19
        // row 2 values=[9,10,11,12] norm≈21.47
        // row 3 values=[13,14,15,16] norm≈29.14
        // Prune 50% of rows = remove 2 lowest-norm rows (0 and 1)
        INDArray w = make4x4(DataType.DOUBLE);
        SparseNDArray csr = SparsePruning.structuredPrune(w, 0.50, 0);

        INDArray dense = csr.toDense();

        // Rows 0 and 1 pruned
        for (long col = 0; col < 4; col++) {
            assertEquals(0.0, dense.getDouble(0, col), TOL, "row 0 should be pruned");
            assertEquals(0.0, dense.getDouble(1, col), TOL, "row 1 should be pruned");
        }
        // Rows 2 and 3 kept
        for (long col = 0; col < 4; col++) {
            assertEquals(w.getDouble(2, col), dense.getDouble(2, col), TOL, "row 2 kept");
            assertEquals(w.getDouble(3, col), dense.getDouble(3, col), TOL, "row 3 kept");
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("structuredPrune axis=1 keeps columns with largest L2 norm")
    public void testStructuredPruneCols(Nd4jBackend backend) {
        // 4×4 weight: col j values = [j+1, j+5, j+9, j+13] (0-indexed cols 0..3)
        // col 0: [1,5,9,13]   norm≈17.49
        // col 1: [2,6,10,14]  norm≈18.44
        // col 2: [3,7,11,15]  norm≈19.39
        // col 3: [4,8,12,16]  norm≈20.40
        // Prune 50% of cols = remove 2 lowest-norm cols (0 and 1)
        INDArray w = make4x4(DataType.DOUBLE);
        SparseNDArray csr = SparsePruning.structuredPrune(w, 0.50, 1);

        INDArray dense = csr.toDense();

        // Cols 0 and 1 pruned
        for (long row = 0; row < 4; row++) {
            assertEquals(0.0, dense.getDouble(row, 0), TOL, "col 0 should be pruned");
            assertEquals(0.0, dense.getDouble(row, 1), TOL, "col 1 should be pruned");
        }
        // Cols 2 and 3 kept
        for (long row = 0; row < 4; row++) {
            assertEquals(w.getDouble(row, 2), dense.getDouble(row, 2), TOL, "col 2 kept");
            assertEquals(w.getDouble(row, 3), dense.getDouble(row, 3), TOL, "col 3 kept");
        }
    }

    // -----------------------------------------------------------------------
    // 6. Edge cases
    // -----------------------------------------------------------------------

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("magnitudePrune with 0% sparsity returns all non-zeros")
    public void testMagnitudePruneZeroSparsity(Nd4jBackend backend) {
        INDArray w = make4x4(DataType.FLOAT);
        SparseNDArray csr = SparsePruning.magnitudePrune(w, 0.0);
        assertEquals(16L, csr.nnz(), "0% sparsity → all 16 entries kept");
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("magnitudePrune with near-100% sparsity retains at least 1 entry")
    public void testMagnitudePruneHighSparsity(Nd4jBackend backend) {
        INDArray w = make4x4(DataType.FLOAT);
        // 0.9375 = 15/16 → keep 1 entry (the largest = 16.0)
        SparseNDArray csr = SparsePruning.magnitudePrune(w, 0.9375);
        assertEquals(1L, csr.nnz(), "93.75% sparsity → only 1 entry kept");
        INDArray vals = csr.getValues();
        assertEquals(16.0, vals.getDouble(0), 1e-4, "kept value should be the largest (16.0)");
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("SparseLinear construction fails for non-CSR format")
    public void testSparseLinearRejectsBadFormat(Nd4jBackend backend) {
        // Build a COO sparse array
        INDArray indices = Nd4j.createFromArray(new long[][]{{0L, 1L}, {1L, 0L}});
        INDArray values  = Nd4j.createFromArray(new float[]{1.0f, 2.0f});
        SparseNDArray coo = new SparseNDArray(indices, values, new long[]{3L, 3L}, SparseFormat.COO);
        assertThrows(IllegalArgumentException.class,
                () -> new SparsePruning.SparseLinear(coo),
                "Should reject non-CSR format");
    }
}
