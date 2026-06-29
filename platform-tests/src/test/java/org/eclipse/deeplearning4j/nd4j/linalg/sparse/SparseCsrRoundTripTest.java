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
import org.nd4j.linalg.api.ops.impl.sparse.CsrToDense;
import org.nd4j.linalg.api.ops.impl.sparse.DenseToCsr;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Round-trip tests for the CSR sparse tensor support.
 *
 * <p>Each test verifies:
 * <ul>
 *   <li>Nd4j.toSparse(x, CSR).toDense() equals the original within tolerance</li>
 *   <li>nnz is correct</li>
 *   <li>rowPtr is monotonically non-decreasing, rowPtr[0]==0, rowPtr[rows]==nnz</li>
 *   <li>every colIdx value is in [0, cols)</li>
 * </ul>
 */
@DisplayName("CSR Sparse Round-Trip Tests")
public class SparseCsrRoundTripTest {

    private static final double TOL = 1e-5;

    // -----------------------------------------------------------------------
    // Utility methods
    // -----------------------------------------------------------------------

    /**
     * Assert that the CSR metadata is internally consistent.
     */
    private static void assertCsrValid(SparseNDArray csr, long expectedNnz) {
        long rows = csr.rows();
        long cols = csr.cols();
        INDArray rowPtr = csr.getRowPtr();
        INDArray colIdx = csr.getColIdx();
        INDArray values = csr.getValues();

        // Basic shape
        assertEquals(rows + 1, rowPtr.length(),
                "rowPtr length must be rows+1");
        assertEquals(expectedNnz, values.length(),
                "values length must equal expected nnz");
        assertEquals(expectedNnz, colIdx.length(),
                "colIdx length must equal expected nnz");
        assertEquals(expectedNnz, csr.nnz(),
                "nnz() must match values length");

        // rowPtr[0] == 0
        assertEquals(0, rowPtr.getInt(0),
                "rowPtr[0] must be 0");

        // rowPtr[rows] == nnz
        assertEquals((int) expectedNnz, rowPtr.getInt((int) rows),
                "rowPtr[rows] must equal nnz");

        // rowPtr monotonically non-decreasing
        for (int r = 0; r < rows; r++) {
            assertTrue(rowPtr.getInt(r) <= rowPtr.getInt(r + 1),
                    "rowPtr must be non-decreasing at index " + r);
        }

        // colIdx in [0, cols)
        for (long k = 0; k < expectedNnz; k++) {
            int c = colIdx.getInt((int) k);
            assertTrue(c >= 0 && c < cols,
                    "colIdx[" + k + "]=" + c + " out of range [0," + cols + ")");
        }
    }

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
     * Run the full round-trip for a given dense matrix.
     */
    private static void roundTrip(INDArray dense, long expectedNnz) {
        SparseNDArray csr = Nd4j.toSparse(dense, SparseFormat.CSR);
        assertCsrValid(csr, expectedNnz);
        INDArray recovered = csr.toDense();
        assertMatrixEquals(dense, recovered, "round-trip mismatch");
    }

    // -----------------------------------------------------------------------
    // Tests — shapes
    // -----------------------------------------------------------------------

    @Test
    @DisplayName("1x1 matrix round-trip FLOAT")
    public void test1x1Float() {
        INDArray x = Nd4j.create(new float[]{3.14f}, new long[]{1, 1});
        roundTrip(x, 1L);
    }

    @Test
    @DisplayName("4x5 matrix round-trip FLOAT")
    public void test4x5Float() {
        INDArray x = Nd4j.rand(DataType.FLOAT, 4, 5);
        // All random values will be non-zero (extremely high probability)
        SparseNDArray csr = Nd4j.toSparse(x, SparseFormat.CSR);
        assertCsrValid(csr, csr.nnz()); // nnz determined by the op
        INDArray recovered = csr.toDense();
        assertMatrixEquals(x, recovered, "4x5 round-trip");
    }

    @Test
    @DisplayName("64x64 matrix round-trip FLOAT")
    public void test64x64Float() {
        INDArray x = Nd4j.rand(DataType.FLOAT, 64, 64);
        SparseNDArray csr = Nd4j.toSparse(x, SparseFormat.CSR);
        assertCsrValid(csr, csr.nnz());
        INDArray recovered = csr.toDense();
        assertMatrixEquals(x, recovered, "64x64 round-trip");
    }

    @Test
    @DisplayName("100x33 matrix round-trip FLOAT")
    public void test100x33Float() {
        INDArray x = Nd4j.rand(DataType.FLOAT, 100, 33);
        SparseNDArray csr = Nd4j.toSparse(x, SparseFormat.CSR);
        assertCsrValid(csr, csr.nnz());
        assertMatrixEquals(x, csr.toDense(), "100x33 round-trip");
    }

    // -----------------------------------------------------------------------
    // Tests — dtypes
    // -----------------------------------------------------------------------

    @Test
    @DisplayName("4x5 matrix round-trip DOUBLE")
    public void test4x5Double() {
        INDArray x = Nd4j.rand(DataType.DOUBLE, 4, 5);
        SparseNDArray csr = Nd4j.toSparse(x, SparseFormat.CSR);
        assertCsrValid(csr, csr.nnz());
        assertEquals(DataType.DOUBLE, csr.dataType(), "dataType should be DOUBLE");
        assertMatrixEquals(x, csr.toDense(), "4x5 DOUBLE round-trip");
    }

    // -----------------------------------------------------------------------
    // Tests — sparsity levels
    // -----------------------------------------------------------------------

    @Test
    @DisplayName("All-zero matrix (100% zeros) — nnz must be 0")
    public void testAllZeros() {
        INDArray x = Nd4j.zeros(DataType.FLOAT, 4, 5);
        SparseNDArray csr = Nd4j.toSparse(x, SparseFormat.CSR);
        assertCsrValid(csr, 0L);
        assertEquals(0, csr.nnz(), "All-zero matrix must have nnz=0");
        assertMatrixEquals(x, csr.toDense(), "all-zero round-trip");
    }

    @Test
    @DisplayName("Fully dense matrix (0% zeros) round-trip FLOAT")
    public void testFullyDense() {
        // ones + 1 guarantees no zeros even after floating-point rounding
        INDArray x = Nd4j.ones(DataType.FLOAT, 8, 8).add(1.0f);
        long expectedNnz = 8L * 8L;
        SparseNDArray csr = Nd4j.toSparse(x, SparseFormat.CSR);
        assertCsrValid(csr, expectedNnz);
        assertEquals(expectedNnz, csr.nnz(), "Fully-dense nnz must equal rows*cols");
        assertMatrixEquals(x, csr.toDense(), "fully-dense round-trip");
    }

    @Test
    @DisplayName("~50% sparsity round-trip FLOAT")
    public void test50PercentSparsity() {
        // Build a matrix where exactly half the entries are zero
        int rows = 4, cols = 6;
        float[] data = new float[rows * cols];
        int nnz = 0;
        for (int i = 0; i < data.length; i++) {
            if (i % 2 == 0) { data[i] = (float)(i + 1); nnz++; }
        }
        INDArray x = Nd4j.create(data, new long[]{rows, cols}, DataType.FLOAT);
        SparseNDArray csr = Nd4j.toSparse(x, SparseFormat.CSR);
        assertCsrValid(csr, (long) nnz);
        assertEquals(nnz, csr.nnz(), "nnz mismatch for 50% sparse matrix");
        assertMatrixEquals(x, csr.toDense(), "50% sparse round-trip");
    }

    @Test
    @DisplayName("~90% sparsity round-trip FLOAT")
    public void test90PercentSparsity() {
        int rows = 10, cols = 10;
        float[] data = new float[rows * cols];
        int nnz = 0;
        // Only every 10th entry is non-zero
        for (int i = 0; i < data.length; i++) {
            if (i % 10 == 0) { data[i] = (float)(i + 1); nnz++; }
        }
        INDArray x = Nd4j.create(data, new long[]{rows, cols}, DataType.FLOAT);
        SparseNDArray csr = Nd4j.toSparse(x, SparseFormat.CSR);
        assertCsrValid(csr, (long) nnz);
        assertMatrixEquals(x, csr.toDense(), "90% sparse round-trip");
    }

    // -----------------------------------------------------------------------
    // Tests — direct op APIs
    // -----------------------------------------------------------------------

    @Test
    @DisplayName("DenseToCsr op directly — check output shapes")
    public void testDenseToCsrOpDirectly() {
        int rows = 3, cols = 4;
        float[] data = {1f, 0f, 2f, 0f,
                         0f, 3f, 0f, 4f,
                         0f, 0f, 5f, 0f};
        INDArray dense = Nd4j.create(data, new long[]{rows, cols}, DataType.FLOAT);

        DenseToCsr op = new DenseToCsr(dense, 0.0);
        INDArray[] results = Nd4j.exec(op);

        INDArray values = results[0];
        INDArray colIdx = results[1];
        INDArray rowPtr = results[2];

        // expected nnz = 5  (1,2,3,4,5)
        assertEquals(5, values.length(), "values length");
        assertEquals(5, colIdx.length(), "colIdx length");
        assertEquals(rows + 1, rowPtr.length(), "rowPtr length");

        // rowPtr must be [0, 2, 4, 5]
        assertEquals(0, rowPtr.getInt(0));
        assertEquals(2, rowPtr.getInt(1));
        assertEquals(4, rowPtr.getInt(2));
        assertEquals(5, rowPtr.getInt(3));

        // colIdx must be [0, 2, 1, 3, 2]
        assertEquals(0, colIdx.getInt(0));
        assertEquals(2, colIdx.getInt(1));
        assertEquals(1, colIdx.getInt(2));
        assertEquals(3, colIdx.getInt(3));
        assertEquals(2, colIdx.getInt(4));
    }

    @Test
    @DisplayName("CsrToDense op directly — reconstruct known matrix")
    public void testCsrToDenseOpDirectly() {
        // CSR for [[1,0,2],[0,3,0]]
        // values=[1,2,3], colIdx=[0,2,1], rowPtr=[0,2,3]
        int rows = 2, cols = 3;
        INDArray values = Nd4j.createFromArray(new float[]{1f, 2f, 3f});
        INDArray colIdx = Nd4j.createFromArray(new int[]{0, 2, 1});
        INDArray rowPtr = Nd4j.createFromArray(new int[]{0, 2, 3});

        CsrToDense op = new CsrToDense(values, colIdx, rowPtr, rows, cols);
        INDArray[] results = Nd4j.exec(op);
        INDArray dense = results[0];

        assertEquals(rows, dense.shape()[0]);
        assertEquals(cols, dense.shape()[1]);

        assertEquals(1.0, dense.getDouble(0, 0), TOL);
        assertEquals(0.0, dense.getDouble(0, 1), TOL);
        assertEquals(2.0, dense.getDouble(0, 2), TOL);
        assertEquals(0.0, dense.getDouble(1, 0), TOL);
        assertEquals(3.0, dense.getDouble(1, 1), TOL);
        assertEquals(0.0, dense.getDouble(1, 2), TOL);
    }

    @Test
    @DisplayName("SparseNDArray toDense via container")
    public void testSparseNDArrayContainer() {
        INDArray values = Nd4j.createFromArray(new float[]{1f, 2f, 3f});
        INDArray colIdx = Nd4j.createFromArray(new int[]{0, 2, 1});
        INDArray rowPtr = Nd4j.createFromArray(new int[]{0, 2, 3});
        long[] shape    = {2L, 3L};

        SparseNDArray csr = new SparseNDArray(values, colIdx, rowPtr, shape, SparseFormat.CSR);
        assertEquals(3, csr.nnz());
        assertEquals(2, csr.rows());
        assertEquals(3, csr.cols());
        assertEquals(DataType.FLOAT, csr.dataType());

        INDArray dense = csr.toDense();
        assertEquals(1.0, dense.getDouble(0, 0), TOL);
        assertEquals(0.0, dense.getDouble(0, 1), TOL);
        assertEquals(2.0, dense.getDouble(0, 2), TOL);
        assertEquals(0.0, dense.getDouble(1, 0), TOL);
        assertEquals(3.0, dense.getDouble(1, 1), TOL);
        assertEquals(0.0, dense.getDouble(1, 2), TOL);
    }

    @Test
    @DisplayName("1x1 all-zero matrix — empty CSR")
    public void test1x1AllZero() {
        INDArray x = Nd4j.zeros(DataType.FLOAT, 1, 1);
        SparseNDArray csr = Nd4j.toSparse(x, SparseFormat.CSR);
        assertEquals(0, csr.nnz(), "single zero must give nnz=0");
        assertCsrValid(csr, 0L);
        assertMatrixEquals(x, csr.toDense(), "1x1 zero round-trip");
    }

    @Test
    @DisplayName("1x1 non-zero matrix")
    public void test1x1NonZero() {
        INDArray x = Nd4j.scalar(DataType.FLOAT, 42.0f).reshape(1, 1);
        SparseNDArray csr = Nd4j.toSparse(x, SparseFormat.CSR);
        assertEquals(1, csr.nnz(), "single non-zero must give nnz=1");
        assertCsrValid(csr, 1L);
        assertMatrixEquals(x, csr.toDense(), "1x1 non-zero round-trip");
    }

    // -----------------------------------------------------------------------
    // Tests — threshold
    // -----------------------------------------------------------------------

    @Test
    @DisplayName("DenseToCsr with non-zero threshold filters small values")
    public void testThreshold() {
        // Values: 0.1, 0.5, 1.0, 1.5 — threshold=0.4 keeps 3 entries
        float[] data = {0.1f, 0.5f, 1.0f, 1.5f};
        INDArray dense = Nd4j.create(data, new long[]{1, 4}, DataType.FLOAT);

        DenseToCsr op = new DenseToCsr(dense, 0.4);
        INDArray[] results = Nd4j.exec(op);
        INDArray values = results[0];

        // 0.1 <= 0.4 → excluded; 0.5, 1.0, 1.5 → included
        assertEquals(3, values.length(), "threshold should filter out 0.1");
    }
}
