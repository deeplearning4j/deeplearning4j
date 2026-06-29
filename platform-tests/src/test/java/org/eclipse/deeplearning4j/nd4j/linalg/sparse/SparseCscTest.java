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
import org.nd4j.linalg.api.ops.impl.sparse.CscToDense;
import org.nd4j.linalg.api.ops.impl.sparse.CsrToCsc;
import org.nd4j.linalg.api.ops.impl.sparse.DenseToCsc;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Round-trip and correctness tests for CSC (Compressed Sparse Column) sparse tensor support.
 *
 * <p>Test coverage:
 * <ul>
 *   <li>(i)  {@code Nd4j.toSparse(A, CSC).toDense() == A} — dense→CSC→dense round-trip</li>
 *   <li>(ii) {@code toSparse(A, CSR).toCsc().toDense() == A} — CSR→CSC→dense path</li>
 *   <li>(iii) {@code toSparse(A, CSR).transpose().toDense() == A.transpose()} — sparse transpose</li>
 *   <li>(iv) CSC structure: colPtr monotonic, colPtr[cols]==nnz, rowIdx ∈ [0, rows)</li>
 *   <li>Various shapes, sparsity levels (all-zero, fully-dense, ~50%, ~90%), FLOAT and DOUBLE</li>
 *   <li>Direct op APIs: DenseToCsc, CsrToCsc, CscToDense</li>
 * </ul>
 */
@DisplayName("CSC Sparse Round-Trip and Transpose Tests")
public class SparseCscTest {

    private static final double TOL = 1e-5;

    // -----------------------------------------------------------------------
    // Utility helpers
    // -----------------------------------------------------------------------

    /**
     * Assert that a CSC SparseNDArray is internally consistent:
     * <ul>
     *   <li>colPtr length == cols+1</li>
     *   <li>colPtr[0] == 0</li>
     *   <li>colPtr[cols] == nnz</li>
     *   <li>colPtr is monotonically non-decreasing</li>
     *   <li>every rowIdx value is in [0, rows)</li>
     * </ul>
     */
    private static void assertCscValid(SparseNDArray csc, long expectedNnz) {
        assertEquals(SparseFormat.CSC, csc.getFormat(), "format must be CSC");
        long rows = csc.rows();
        long cols = csc.cols();
        INDArray colPtr = csc.getColPtr();
        INDArray rowIdx = csc.getRowIdx();
        INDArray values = csc.getValues();

        // Length checks
        assertEquals(cols + 1, colPtr.length(),
                "colPtr length must be cols+1=" + (cols + 1));
        assertEquals(expectedNnz, values.length(),
                "values length must equal expectedNnz=" + expectedNnz);
        assertEquals(expectedNnz, rowIdx.length(),
                "rowIdx length must equal expectedNnz=" + expectedNnz);
        assertEquals(expectedNnz, csc.nnz(),
                "nnz() must match values length");

        // colPtr[0] == 0
        assertEquals(0, colPtr.getInt(0), "colPtr[0] must be 0");

        // colPtr[cols] == nnz
        assertEquals((int) expectedNnz, colPtr.getInt((int) cols),
                "colPtr[cols] must equal nnz");

        // colPtr non-decreasing
        for (int c = 0; c < cols; c++) {
            assertTrue(colPtr.getInt(c) <= colPtr.getInt(c + 1),
                    "colPtr must be non-decreasing at index " + c);
        }

        // rowIdx in [0, rows)
        for (long k = 0; k < expectedNnz; k++) {
            int r = rowIdx.getInt((int) k);
            assertTrue(r >= 0 && r < rows,
                    "rowIdx[" + k + "]=" + r + " out of range [0," + rows + ")");
        }
    }

    /** Element-wise equality check within TOL. */
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
     * (i) Dense → CSC → dense round-trip via {@code Nd4j.toSparse(dense, CSC)}.
     */
    private static void roundTripCsc(INDArray dense, long expectedNnz) {
        SparseNDArray csc = Nd4j.toSparse(dense, SparseFormat.CSC);
        assertCscValid(csc, expectedNnz);
        INDArray recovered = csc.toDense();
        assertMatrixEquals(dense, recovered, "CSC round-trip mismatch");
    }

    // -----------------------------------------------------------------------
    // (i) Dense → CSC → dense round-trips
    // -----------------------------------------------------------------------

    @Test
    @DisplayName("(i) 1x1 FLOAT — dense→CSC→dense")
    public void testRoundTrip1x1Float() {
        INDArray x = Nd4j.create(new float[]{7.5f}, new long[]{1, 1});
        roundTripCsc(x, 1L);
    }

    @Test
    @DisplayName("(i) 4x5 FLOAT — dense→CSC→dense")
    public void testRoundTrip4x5Float() {
        INDArray x = Nd4j.rand(DataType.FLOAT, 4, 5);
        SparseNDArray csc = Nd4j.toSparse(x, SparseFormat.CSC);
        assertCscValid(csc, csc.nnz());
        assertMatrixEquals(x, csc.toDense(), "4x5 FLOAT CSC round-trip");
    }

    @Test
    @DisplayName("(i) 4x5 DOUBLE — dense→CSC→dense")
    public void testRoundTrip4x5Double() {
        INDArray x = Nd4j.rand(DataType.DOUBLE, 4, 5);
        SparseNDArray csc = Nd4j.toSparse(x, SparseFormat.CSC);
        assertCscValid(csc, csc.nnz());
        assertEquals(DataType.DOUBLE, csc.dataType(), "dataType must be DOUBLE");
        assertMatrixEquals(x, csc.toDense(), "4x5 DOUBLE CSC round-trip");
    }

    @Test
    @DisplayName("(i) 64x64 FLOAT — dense→CSC→dense")
    public void testRoundTrip64x64Float() {
        INDArray x = Nd4j.rand(DataType.FLOAT, 64, 64);
        SparseNDArray csc = Nd4j.toSparse(x, SparseFormat.CSC);
        assertCscValid(csc, csc.nnz());
        assertMatrixEquals(x, csc.toDense(), "64x64 FLOAT CSC round-trip");
    }

    @Test
    @DisplayName("(i) 100x33 FLOAT — dense→CSC→dense")
    public void testRoundTrip100x33Float() {
        INDArray x = Nd4j.rand(DataType.FLOAT, 100, 33);
        SparseNDArray csc = Nd4j.toSparse(x, SparseFormat.CSC);
        assertCscValid(csc, csc.nnz());
        assertMatrixEquals(x, csc.toDense(), "100x33 FLOAT CSC round-trip");
    }

    @Test
    @DisplayName("(i) all-zero 4x5 — nnz must be 0")
    public void testRoundTripAllZeros() {
        INDArray x = Nd4j.zeros(DataType.FLOAT, 4, 5);
        SparseNDArray csc = Nd4j.toSparse(x, SparseFormat.CSC);
        assertCscValid(csc, 0L);
        assertEquals(0, csc.nnz(), "all-zero matrix must have nnz=0");
        assertMatrixEquals(x, csc.toDense(), "all-zero CSC round-trip");
    }

    @Test
    @DisplayName("(i) fully-dense 8x8 FLOAT — nnz must be rows*cols")
    public void testRoundTripFullyDense() {
        INDArray x = Nd4j.ones(DataType.FLOAT, 8, 8).add(1.0f);
        long expectedNnz = 8L * 8L;
        SparseNDArray csc = Nd4j.toSparse(x, SparseFormat.CSC);
        assertCscValid(csc, expectedNnz);
        assertEquals(expectedNnz, csc.nnz(), "fully-dense nnz must equal rows*cols");
        assertMatrixEquals(x, csc.toDense(), "fully-dense CSC round-trip");
    }

    @Test
    @DisplayName("(i) ~50% sparsity 4x6 FLOAT")
    public void testRoundTrip50PercentSparsity() {
        int rows = 4, cols = 6;
        float[] data = new float[rows * cols];
        int nnz = 0;
        for (int i = 0; i < data.length; i++) {
            if (i % 2 == 0) { data[i] = (float)(i + 1); nnz++; }
        }
        INDArray x = Nd4j.create(data, new long[]{rows, cols}, DataType.FLOAT);
        SparseNDArray csc = Nd4j.toSparse(x, SparseFormat.CSC);
        assertCscValid(csc, (long) nnz);
        assertMatrixEquals(x, csc.toDense(), "50% sparse CSC round-trip");
    }

    @Test
    @DisplayName("(i) ~90% sparsity 10x10 FLOAT")
    public void testRoundTrip90PercentSparsity() {
        int rows = 10, cols = 10;
        float[] data = new float[rows * cols];
        int nnz = 0;
        for (int i = 0; i < data.length; i++) {
            if (i % 10 == 0) { data[i] = (float)(i + 1); nnz++; }
        }
        INDArray x = Nd4j.create(data, new long[]{rows, cols}, DataType.FLOAT);
        SparseNDArray csc = Nd4j.toSparse(x, SparseFormat.CSC);
        assertCscValid(csc, (long) nnz);
        assertMatrixEquals(x, csc.toDense(), "90% sparse CSC round-trip");
    }

    @Test
    @DisplayName("(i) 1x1 all-zero — empty CSC")
    public void testRoundTrip1x1AllZero() {
        INDArray x = Nd4j.zeros(DataType.FLOAT, 1, 1);
        SparseNDArray csc = Nd4j.toSparse(x, SparseFormat.CSC);
        assertCscValid(csc, 0L);
        assertMatrixEquals(x, csc.toDense(), "1x1 zero CSC round-trip");
    }

    @Test
    @DisplayName("(i) 1x1 non-zero — single element CSC")
    public void testRoundTrip1x1NonZero() {
        INDArray x = Nd4j.scalar(DataType.FLOAT, 42.0f).reshape(1, 1);
        SparseNDArray csc = Nd4j.toSparse(x, SparseFormat.CSC);
        assertCscValid(csc, 1L);
        assertEquals(1, csc.nnz(), "single non-zero must give nnz=1");
        assertMatrixEquals(x, csc.toDense(), "1x1 non-zero CSC round-trip");
    }

    // -----------------------------------------------------------------------
    // (ii) CSR → CSC → dense
    // -----------------------------------------------------------------------

    @Test
    @DisplayName("(ii) CSR→CSC→dense round-trip 4x5 FLOAT")
    public void testCsrToCscToDense4x5Float() {
        INDArray x = Nd4j.rand(DataType.FLOAT, 4, 5);
        SparseNDArray csr  = Nd4j.toSparse(x, SparseFormat.CSR);
        SparseNDArray csc  = csr.toCsc();
        assertCscValid(csc, csc.nnz());
        assertMatrixEquals(x, csc.toDense(), "CSR→CSC→dense 4x5 FLOAT");
    }

    @Test
    @DisplayName("(ii) CSR→CSC→dense round-trip 4x5 DOUBLE")
    public void testCsrToCscToDense4x5Double() {
        INDArray x = Nd4j.rand(DataType.DOUBLE, 4, 5);
        SparseNDArray csc  = Nd4j.toSparse(x, SparseFormat.CSR).toCsc();
        assertCscValid(csc, csc.nnz());
        assertEquals(DataType.DOUBLE, csc.dataType(), "dtype must be DOUBLE");
        assertMatrixEquals(x, csc.toDense(), "CSR→CSC→dense 4x5 DOUBLE");
    }

    @Test
    @DisplayName("(ii) CSR→CSC is identity when called again (toCsc on CSC)")
    public void testToCscIdempotent() {
        INDArray x = Nd4j.rand(DataType.FLOAT, 3, 4);
        SparseNDArray csc  = Nd4j.toSparse(x, SparseFormat.CSC);
        SparseNDArray csc2 = csc.toCsc();
        assertSame(csc, csc2, "toCsc() on a CSC instance must return this");
    }

    @Test
    @DisplayName("(ii) CSR→CSC preserves nnz")
    public void testCsrToCscPreservesNnz() {
        int rows = 5, cols = 7;
        float[] data = new float[rows * cols];
        long nnz = 0;
        for (int i = 0; i < data.length; i++) {
            if (i % 3 == 0) { data[i] = i + 1.0f; nnz++; }
        }
        INDArray x = Nd4j.create(data, new long[]{rows, cols}, DataType.FLOAT);
        SparseNDArray csr = Nd4j.toSparse(x, SparseFormat.CSR);
        SparseNDArray csc = csr.toCsc();
        assertEquals(csr.nnz(), csc.nnz(), "CSR→CSC must preserve nnz");
    }

    @Test
    @DisplayName("(ii) CSR→CSC has colPtr length cols+1")
    public void testCsrToCscColPtrLength() {
        int rows = 6, cols = 8;
        INDArray x = Nd4j.rand(DataType.FLOAT, rows, cols);
        SparseNDArray csc = Nd4j.toSparse(x, SparseFormat.CSR).toCsc();
        assertEquals(cols + 1, csc.getColPtr().length(),
                "colPtr length must be cols+1=" + (cols + 1));
    }

    // -----------------------------------------------------------------------
    // (iii) Sparse transpose: CSR.transpose().toDense() == dense.transpose()
    // -----------------------------------------------------------------------

    @Test
    @DisplayName("(iii) transpose 4x5 FLOAT")
    public void testTranspose4x5Float() {
        INDArray x = Nd4j.rand(DataType.FLOAT, 4, 5);
        SparseNDArray csr = Nd4j.toSparse(x, SparseFormat.CSR);
        SparseNDArray at  = csr.transpose();

        assertEquals(SparseFormat.CSR, at.getFormat(), "transpose result must be CSR");
        assertEquals(5L, at.rows(),  "transposed rows must be original cols");
        assertEquals(4L, at.cols(),  "transposed cols must be original rows");
        assertEquals(csr.nnz(), at.nnz(), "transpose must preserve nnz");

        INDArray expected = x.transpose();
        assertMatrixEquals(expected, at.toDense(), "transpose 4x5 FLOAT");
    }

    @Test
    @DisplayName("(iii) transpose 4x5 DOUBLE")
    public void testTranspose4x5Double() {
        INDArray x = Nd4j.rand(DataType.DOUBLE, 4, 5);
        SparseNDArray at = Nd4j.toSparse(x, SparseFormat.CSR).transpose();
        assertEquals(DataType.DOUBLE, at.dataType(), "dtype must be DOUBLE");
        assertMatrixEquals(x.transpose(), at.toDense(), "transpose 4x5 DOUBLE");
    }

    @Test
    @DisplayName("(iii) transpose 64x64 FLOAT")
    public void testTranspose64x64Float() {
        INDArray x = Nd4j.rand(DataType.FLOAT, 64, 64);
        SparseNDArray at = Nd4j.toSparse(x, SparseFormat.CSR).transpose();
        assertMatrixEquals(x.transpose(), at.toDense(), "transpose 64x64 FLOAT");
    }

    @Test
    @DisplayName("(iii) transpose all-zero 4x5")
    public void testTransposeAllZeros() {
        INDArray x = Nd4j.zeros(DataType.FLOAT, 4, 5);
        SparseNDArray at = Nd4j.toSparse(x, SparseFormat.CSR).transpose();
        assertEquals(0, at.nnz(), "transpose of all-zero must have nnz=0");
        assertMatrixEquals(x.transpose(), at.toDense(), "transpose all-zero");
    }

    @Test
    @DisplayName("(iii) transpose ~50% sparse 4x6")
    public void testTranspose50PercentSparse() {
        float[] data = new float[4 * 6];
        for (int i = 0; i < data.length; i++) {
            if (i % 2 == 0) data[i] = i + 1.0f;
        }
        INDArray x = Nd4j.create(data, new long[]{4, 6}, DataType.FLOAT);
        SparseNDArray at = Nd4j.toSparse(x, SparseFormat.CSR).transpose();
        assertMatrixEquals(x.transpose(), at.toDense(), "transpose 50% sparse");
    }

    @Test
    @DisplayName("(iii) double transpose recovers original")
    public void testDoubleTranspose() {
        INDArray x = Nd4j.rand(DataType.FLOAT, 5, 7);
        SparseNDArray csr = Nd4j.toSparse(x, SparseFormat.CSR);
        SparseNDArray att = csr.transpose().transpose(); // (Aᵀ)ᵀ = A
        assertMatrixEquals(x, att.toDense(), "double transpose must recover original");
    }

    // -----------------------------------------------------------------------
    // (iv) Direct op API tests
    // -----------------------------------------------------------------------

    @Test
    @DisplayName("DenseToCsc op — check output shapes for known matrix")
    public void testDenseToCscOpDirectly() {
        // [[1, 0, 2], [0, 3, 0]] (2 rows x 3 cols)
        // CSC column order: col0=(1@r0), col1=(3@r1), col2=(2@r0)
        // cscValues=[1,3,2], cscRowIdx=[0,1,0], cscColPtr=[0,1,2,3]
        int rows = 2, cols = 3;
        float[] data = {1f, 0f, 2f, 0f, 3f, 0f};
        INDArray dense = Nd4j.create(data, new long[]{rows, cols}, DataType.FLOAT);

        DenseToCsc op = new DenseToCsc(dense, 0.0);
        INDArray[] results = Nd4j.exec(op);

        INDArray cscValues = results[0];
        INDArray cscRowIdx = results[1];
        INDArray cscColPtr = results[2];

        // nnz = 3
        assertEquals(3, cscValues.length(), "cscValues length must equal nnz=3");
        assertEquals(3, cscRowIdx.length(), "cscRowIdx length must equal nnz=3");
        assertEquals(cols + 1, cscColPtr.length(), "cscColPtr length must be cols+1=" + (cols + 1));

        // cscColPtr must be [0, 1, 2, 3]
        assertEquals(0, cscColPtr.getInt(0));
        assertEquals(1, cscColPtr.getInt(1));
        assertEquals(2, cscColPtr.getInt(2));
        assertEquals(3, cscColPtr.getInt(3));
    }

    @Test
    @DisplayName("CsrToCsc op — known 2x3 matrix")
    public void testCsrToCscOpDirectly() {
        // CSR for [[1,0,2],[0,3,0]]: values=[1,2,3], colIdx=[0,2,1], rowPtr=[0,2,3]
        int rows = 2, cols = 3;
        INDArray values = Nd4j.createFromArray(new float[]{1f, 2f, 3f});
        INDArray colIdx = Nd4j.createFromArray(new int[]{0, 2, 1});
        INDArray rowPtr = Nd4j.createFromArray(new int[]{0, 2, 3});

        CsrToCsc op = new CsrToCsc(values, colIdx, rowPtr, rows, cols);
        INDArray[] results = Nd4j.exec(op);

        INDArray cscValues = results[0];
        INDArray cscRowIdx = results[1];
        INDArray cscColPtr = results[2];

        assertEquals(3, cscValues.length(), "cscValues length must be nnz=3");
        assertEquals(3, cscRowIdx.length(), "cscRowIdx length must be nnz=3");
        assertEquals(cols + 1, cscColPtr.length(),
                "cscColPtr length must be cols+1=" + (cols + 1));

        // Verify cscColPtr structure
        assertEquals(0, cscColPtr.getInt(0), "cscColPtr[0] must be 0");
        assertEquals(3, cscColPtr.getInt(cols), "cscColPtr[cols] must be nnz=3");
        // Each column pointer must be non-decreasing
        for (int c = 0; c < cols; c++) {
            assertTrue(cscColPtr.getInt(c) <= cscColPtr.getInt(c + 1),
                    "cscColPtr must be non-decreasing at " + c);
        }
    }

    @Test
    @DisplayName("CscToDense op — reconstruct [[1,0,2],[0,3,0]]")
    public void testCscToDenseOpDirectly() {
        // CSC for [[1,0,2],[0,3,0]]:
        // col0 has value 1 at row 0
        // col1 has value 3 at row 1
        // col2 has value 2 at row 0
        // cscValues=[1,3,2], cscRowIdx=[0,1,0], cscColPtr=[0,1,2,3]
        int rows = 2, cols = 3;
        INDArray cscValues = Nd4j.createFromArray(new float[]{1f, 3f, 2f});
        INDArray cscRowIdx = Nd4j.createFromArray(new int[]{0, 1, 0});
        INDArray cscColPtr = Nd4j.createFromArray(new int[]{0, 1, 2, 3});

        CscToDense op = new CscToDense(cscValues, cscRowIdx, cscColPtr, rows, cols);
        INDArray[] results = Nd4j.exec(op);
        INDArray dense = results[0];

        assertEquals(rows, dense.shape()[0], "rows");
        assertEquals(cols, dense.shape()[1], "cols");
        assertEquals(1.0, dense.getDouble(0, 0), TOL, "[0,0]");
        assertEquals(0.0, dense.getDouble(0, 1), TOL, "[0,1]");
        assertEquals(2.0, dense.getDouble(0, 2), TOL, "[0,2]");
        assertEquals(0.0, dense.getDouble(1, 0), TOL, "[1,0]");
        assertEquals(3.0, dense.getDouble(1, 1), TOL, "[1,1]");
        assertEquals(0.0, dense.getDouble(1, 2), TOL, "[1,2]");
    }

    @Test
    @DisplayName("SparseNDArray CSC container constructor and toDense")
    public void testSparseNDArrayCscContainer() {
        // CSC for [[1,0,2],[0,3,0]]: see above
        INDArray cscValues = Nd4j.createFromArray(new float[]{1f, 3f, 2f});
        INDArray cscRowIdx = Nd4j.createFromArray(new int[]{0, 1, 0});
        INDArray cscColPtr = Nd4j.createFromArray(new int[]{0, 1, 2, 3});
        long[] shape = {2L, 3L};

        SparseNDArray csc = new SparseNDArray(cscValues, cscRowIdx, cscColPtr, shape, SparseFormat.CSC);
        assertEquals(SparseFormat.CSC, csc.getFormat());
        assertEquals(3, csc.nnz());
        assertEquals(2, csc.rows());
        assertEquals(3, csc.cols());
        assertEquals(DataType.FLOAT, csc.dataType());

        INDArray dense = csc.toDense();
        assertEquals(1.0, dense.getDouble(0, 0), TOL, "[0,0]");
        assertEquals(0.0, dense.getDouble(0, 1), TOL, "[0,1]");
        assertEquals(2.0, dense.getDouble(0, 2), TOL, "[0,2]");
        assertEquals(0.0, dense.getDouble(1, 0), TOL, "[1,0]");
        assertEquals(3.0, dense.getDouble(1, 1), TOL, "[1,1]");
        assertEquals(0.0, dense.getDouble(1, 2), TOL, "[1,2]");
    }

    @Test
    @DisplayName("DenseToCsc with non-zero threshold filters small values")
    public void testDenseToCscThreshold() {
        // Values 0.1, 0.5, 1.0, 1.5 in a 1x4 row — threshold=0.4 keeps 3 entries
        float[] data = {0.1f, 0.5f, 1.0f, 1.5f};
        INDArray dense = Nd4j.create(data, new long[]{1, 4}, DataType.FLOAT);

        DenseToCsc op = new DenseToCsc(dense, 0.4);
        INDArray[] results = Nd4j.exec(op);
        INDArray cscValues = results[0];

        // 0.1 <= 0.4 → excluded; 0.5, 1.0, 1.5 → kept
        assertEquals(3, cscValues.length(), "threshold=0.4 should keep 3 values");
    }

    @Test
    @DisplayName("CSC accessors throw for wrong format")
    public void testCscAccessorGuards() {
        // A CSR instance must reject getRowIdx() / getColPtr()
        INDArray x = Nd4j.rand(DataType.FLOAT, 3, 4);
        SparseNDArray csr = Nd4j.toSparse(x, SparseFormat.CSR);
        assertThrows(Exception.class, csr::getRowIdx, "getRowIdx() on CSR must throw");
        assertThrows(Exception.class, csr::getColPtr, "getColPtr() on CSR must throw");

        // A CSC instance must reject getColIdx() / getRowPtr()
        SparseNDArray csc = csr.toCsc();
        assertThrows(Exception.class, csc::getColIdx, "getColIdx() on CSC must throw");
        assertThrows(Exception.class, csc::getRowPtr, "getRowPtr() on CSC must throw");
    }
}
