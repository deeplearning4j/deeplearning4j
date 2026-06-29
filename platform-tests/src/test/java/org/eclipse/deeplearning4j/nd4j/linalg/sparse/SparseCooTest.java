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
import org.nd4j.linalg.api.ops.impl.sparse.DenseToCoo;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for COO (Coordinate) sparse tensor support via {@link SparseNDArray}
 * and {@link org.nd4j.linalg.api.ops.impl.sparse.DenseToCoo} /
 * {@link org.nd4j.linalg.api.ops.impl.sparse.CooToCsr}.
 */
@DisplayName("COO Sparse Tests")
public class SparseCooTest {

    private static final double TOL = 1e-5;
    private static final int ROWS = 4;
    private static final int COLS = 5;

    // -----------------------------------------------------------------------
    // Utility helpers
    // -----------------------------------------------------------------------

    /**
     * Assert element-wise equality of two matrices within TOL.
     */
    private static void assertMatrixEquals(INDArray expected, INDArray actual, String msg) {
        assertEquals(expected.shape()[0], actual.shape()[0], msg + ": rows mismatch");
        assertEquals(expected.shape()[1], actual.shape()[1], msg + ": cols mismatch");
        long rows = expected.shape()[0];
        long cols = expected.shape()[1];
        for (long r = 0; r < rows; r++) {
            for (long c = 0; c < cols; c++) {
                double e = expected.getDouble(r, c);
                double a = actual.getDouble(r, c);
                assertEquals(e, a, TOL, msg + ": mismatch at [" + r + "," + c + "]");
            }
        }
    }

    /**
     * Build a 4x5 matrix with ~50% zeros: even-indexed entries (row-major) are non-zero,
     * odd-indexed entries are zero. Returns the dense INDArray.
     */
    private static INDArray buildHalfSparseMatrix(DataType dtype) {
        float[] data = new float[ROWS * COLS];
        for (int i = 0; i < data.length; i++) {
            data[i] = (i % 2 == 0) ? (i + 1.0f) : 0.0f;
        }
        return Nd4j.create(data, new long[]{ROWS, COLS}, DataType.FLOAT).castTo(dtype);
    }

    // -----------------------------------------------------------------------
    // Test 1: COO round-trip for FLOAT and DOUBLE
    // -----------------------------------------------------------------------

    @Test
    @DisplayName("COO round-trip — FLOAT 4x5 ~50% sparse")
    public void testCooRoundTripFloat() {
        INDArray dense = buildHalfSparseMatrix(DataType.FLOAT);
        SparseNDArray coo = Nd4j.toSparse(dense, SparseFormat.COO);
        assertEquals(SparseFormat.COO, coo.getFormat(), "format must be COO");
        INDArray recovered = coo.toDense();
        assertMatrixEquals(dense, recovered, "COO FLOAT round-trip");
    }

    @Test
    @DisplayName("COO round-trip — DOUBLE 4x5 ~50% sparse")
    public void testCooRoundTripDouble() {
        INDArray dense = buildHalfSparseMatrix(DataType.DOUBLE);
        SparseNDArray coo = Nd4j.toSparse(dense, SparseFormat.COO);
        assertEquals(SparseFormat.COO, coo.getFormat(), "format must be COO");
        assertEquals(DataType.DOUBLE, coo.dataType(), "dataType should be DOUBLE");
        INDArray recovered = coo.toDense();
        assertMatrixEquals(dense, recovered, "COO DOUBLE round-trip");
    }

    // -----------------------------------------------------------------------
    // Test 2: COO indices are in-range and values match dense
    // -----------------------------------------------------------------------

    @Test
    @DisplayName("COO indices in-range and values match")
    public void testCooNnzAndIndices() {
        INDArray dense = buildHalfSparseMatrix(DataType.FLOAT);
        SparseNDArray coo = Nd4j.toSparse(dense, SparseFormat.COO);

        long expectedNnz = 0;
        for (long r = 0; r < ROWS; r++) {
            for (long c = 0; c < COLS; c++) {
                if (dense.getDouble(r, c) != 0.0) expectedNnz++;
            }
        }

        assertEquals(expectedNnz, coo.nnz(), "nnz must match count of non-zeros");
        assertEquals(ROWS, coo.rows(), "rows mismatch");
        assertEquals(COLS, coo.cols(), "cols mismatch");

        INDArray indices = coo.getIndices();
        INDArray values  = coo.getValues();

        // Each row in indices is a (row, col) pair — must be in range
        for (long k = 0; k < coo.nnz(); k++) {
            long row = indices.getLong(k, 0);
            long col = indices.getLong(k, 1);
            assertTrue(row >= 0 && row < ROWS,
                    "indices[" + k + ",0]=" + row + " out of row range [0," + ROWS + ")");
            assertTrue(col >= 0 && col < COLS,
                    "indices[" + k + ",1]=" + col + " out of col range [0," + COLS + ")");

            // Value must match the dense matrix at that position
            double expected = dense.getDouble(row, col);
            double actual   = values.getDouble(k);
            assertEquals(expected, actual, TOL,
                    "value mismatch at nnz index " + k + " -> [" + row + "," + col + "]");
        }
    }

    // -----------------------------------------------------------------------
    // Test 3: COO → CSR → Dense round-trip
    // -----------------------------------------------------------------------

    @Test
    @DisplayName("COO→CSR→Dense round-trip")
    public void testCooToCsrToDense() {
        INDArray dense = buildHalfSparseMatrix(DataType.FLOAT);
        SparseNDArray coo = Nd4j.toSparse(dense, SparseFormat.COO);
        SparseNDArray csr = coo.toCsr();

        assertEquals(SparseFormat.CSR, csr.getFormat(), "toCsr() must produce CSR");
        assertEquals(coo.nnz(), csr.nnz(), "nnz must be preserved across COO→CSR");

        INDArray recovered = csr.toDense();
        assertMatrixEquals(dense, recovered, "COO→CSR→Dense round-trip");
    }

    // -----------------------------------------------------------------------
    // Test 4: COO→CSR consistency with direct CSR conversion
    // -----------------------------------------------------------------------

    @Test
    @DisplayName("COO→CSR consistent with direct CSR")
    public void testCooAndCsrConsistency() {
        INDArray dense = buildHalfSparseMatrix(DataType.FLOAT);

        SparseNDArray directCsr = Nd4j.toSparse(dense, SparseFormat.CSR);
        SparseNDArray cooToCsr  = Nd4j.toSparse(dense, SparseFormat.COO).toCsr();

        assertEquals(directCsr.nnz(), cooToCsr.nnz(), "nnz must agree between direct CSR and COO→CSR");
        assertEquals(directCsr.rows(), cooToCsr.rows(), "rows must agree");
        assertEquals(directCsr.cols(), cooToCsr.cols(), "cols must agree");

        // Both should produce the same dense matrix
        INDArray fromDirect = directCsr.toDense();
        INDArray fromCoo    = cooToCsr.toDense();
        assertMatrixEquals(fromDirect, fromCoo, "COO→CSR and direct CSR must produce same dense");
    }

    // -----------------------------------------------------------------------
    // Test 5: All-zero matrix — nnz must be 0
    // -----------------------------------------------------------------------

    @Test
    @DisplayName("All-zero matrix — nnz must be 0")
    public void testAllZeroMatrix() {
        INDArray dense = Nd4j.zeros(DataType.FLOAT, ROWS, COLS);
        SparseNDArray coo = Nd4j.toSparse(dense, SparseFormat.COO);

        assertEquals(0L, coo.nnz(), "all-zero matrix must have nnz=0");
        assertEquals(SparseFormat.COO, coo.getFormat(), "format must be COO");

        // Round-trip must still work
        INDArray recovered = coo.toDense();
        assertMatrixEquals(dense, recovered, "all-zero COO round-trip");
    }

    // -----------------------------------------------------------------------
    // Test 6: Fully non-zero matrix — nnz must equal rows*cols
    // -----------------------------------------------------------------------

    @Test
    @DisplayName("Fully non-zero matrix — nnz must equal rows*cols")
    public void testFullyDenseMatrix() {
        // ones + 1 guarantees no zeros even after dtype cast
        INDArray dense = Nd4j.ones(DataType.FLOAT, ROWS, COLS).add(1.0f);
        long expectedNnz = (long) ROWS * COLS;

        SparseNDArray coo = Nd4j.toSparse(dense, SparseFormat.COO);
        assertEquals(expectedNnz, coo.nnz(),
                "fully non-zero matrix must have nnz=rows*cols=" + expectedNnz);

        INDArray recovered = coo.toDense();
        assertMatrixEquals(dense, recovered, "fully non-zero COO round-trip");
    }

    // -----------------------------------------------------------------------
    // Test 7: DenseToCoo op directly — shape checks
    // -----------------------------------------------------------------------

    @Test
    @DisplayName("DenseToCoo op directly — output shapes")
    public void testDenseToCooOpDirectly() {
        // [[1, 0, 2], [0, 3, 0]] — nnz=3
        float[] data = {1f, 0f, 2f, 0f, 3f, 0f};
        INDArray dense = Nd4j.create(data, new long[]{2, 3}, DataType.FLOAT);

        DenseToCoo op = new DenseToCoo(dense, 0.0);
        INDArray[] results = Nd4j.exec(op);

        INDArray indices = results[0]; // [nnz, 2] INT64
        INDArray values  = results[1]; // [nnz]

        assertEquals(3L, values.length(), "values length must equal nnz=3");
        assertEquals(2, indices.rank(), "indices must be rank-2");
        assertEquals(3L, indices.shape()[0], "indices must have nnz=3 rows");
        assertEquals(2L, indices.shape()[1], "indices must have 2 columns (row,col)");
    }

    // -----------------------------------------------------------------------
    // Test 8: toCsr() on CSR is identity (returns same object)
    // -----------------------------------------------------------------------

    @Test
    @DisplayName("toCsr() on a CSR array returns this")
    public void testToCsrOnCsrIsIdentity() {
        INDArray dense = buildHalfSparseMatrix(DataType.FLOAT);
        SparseNDArray csr = Nd4j.toSparse(dense, SparseFormat.CSR);
        assertSame(csr, csr.toCsr(), "toCsr() on CSR must return this");
    }
}
