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

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.sparse.CsrAddBp;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

/**
 * Standalone eager-mode test for {@code csr_add_bp} that bypasses DSP / SameDiff entirely.
 *
 * <p>Matrix layout (2 rows, 3 cols):
 * <pre>
 *   A = [[1, 0, 2],    aValues=[1.0, 2.0, 3.0], aColIdx=[0,2,1], aRowPtr=[0,2,3]
 *        [0, 3, 0]]
 *
 *   B = [[0, 4, 0],    bValues=[4.0, 5.0],       bColIdx=[1,2],   bRowPtr=[0,1,2]
 *        [0, 0, 5]]
 *
 *   C = A + B = [[1, 4, 2],    cValues=[1,4,2,3,5], cColIdx=[0,1,2,1,2], cRowPtr=[0,3,5]
 *               [0, 3, 5]]
 * </pre>
 *
 * <p>With upstream gradient {@code gradCValues = [1,1,1,1,1]} (all-ones):
 * <ul>
 *   <li>dAValues[0] = grad of A(0,0)=1.0  → position 0 in C row0 → C[row0,col0] → grad=1.0</li>
 *   <li>dAValues[1] = grad of A(0,2)=2.0  → position 2 in C row0 → C[row0,col2] → grad=1.0</li>
 *   <li>dAValues[2] = grad of A(1,1)=3.0  → position 3 in C row1 → C[row1,col1] → grad=1.0</li>
 *   <li>dBValues[0] = grad of B(0,1)=4.0  → position 1 in C row0 → C[row0,col1] → grad=1.0</li>
 *   <li>dBValues[1] = grad of B(1,2)=5.0  → position 4 in C row1 → C[row1,col2] → grad=1.0</li>
 * </ul>
 */
public class CsrAddBpDirectTest extends BaseNd4jTestWithBackends {

    @Override
    public char ordering() {
        return 'c';
    }

    /**
     * Verifies that the eager {@code Nd4j.exec(new CsrAddBp(...))} call:
     * <ol>
     *   <li>Returns exactly 2 non-null outputs.</li>
     *   <li>dAValues has shape [nnzA=3] and dtype matching gradCValues (DOUBLE).</li>
     *   <li>dBValues has shape [nnzB=2] and dtype matching gradCValues (DOUBLE).</li>
     *   <li>Values are all 1.0 for the all-ones upstream gradient.</li>
     * </ol>
     *
     * <p>This test does NOT go through SameDiff or DSP. It exercises the C++ op registration
     * and execute path directly, so a failure here means the op is not properly registered
     * or has a bug in {@code validateAndExecute}, not in plan compilation.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testCsrAddBpEagerDirect(Nd4jBackend backend) {
        // ── Structural (INT32) arrays ─────────────────────────────────────────
        // A: 2x3, nnzA=3
        INDArray aColIdx = Nd4j.createFromArray(new int[]{0, 2, 1}).castTo(DataType.INT32);
        INDArray aRowPtr = Nd4j.createFromArray(new int[]{0, 2, 3}).castTo(DataType.INT32);

        // B: 2x3, nnzB=2
        INDArray bColIdx = Nd4j.createFromArray(new int[]{1, 2}).castTo(DataType.INT32);
        INDArray bRowPtr = Nd4j.createFromArray(new int[]{0, 1, 2}).castTo(DataType.INT32);

        // C = A + B: nnzC=5 (forward op output — provided as inputs to bp)
        INDArray cColIdx = Nd4j.createFromArray(new int[]{0, 1, 2, 1, 2}).castTo(DataType.INT32);
        INDArray cRowPtr = Nd4j.createFromArray(new int[]{0, 3, 5}).castTo(DataType.INT32);

        // ── Float (DOUBLE) upstream gradient ─────────────────────────────────
        INDArray gradCValues = Nd4j.ones(DataType.DOUBLE, 5);  // all-ones upstream grad

        // ── Execute the backward op directly (eager mode, no SameDiff/DSP) ───
        INDArray[] results = Nd4j.exec(new CsrAddBp(
                aColIdx, aRowPtr,
                bColIdx, bRowPtr,
                cColIdx, cRowPtr,
                gradCValues,
                2L, 3L));  // m=2 rows, n=3 cols

        // ── Output count ─────────────────────────────────────────────────────
        assertNotNull(results, "exec() must not return null");
        assertEquals(2, results.length, "csr_add_bp must produce exactly 2 outputs");

        INDArray dAValues = results[0];
        INDArray dBValues = results[1];

        assertNotNull(dAValues, "dAValues (output[0]) must not be null");
        assertNotNull(dBValues, "dBValues (output[1]) must not be null");

        // ── Output shapes ────────────────────────────────────────────────────
        assertArrayEquals(new long[]{3}, dAValues.shape(),
                "dAValues shape must be [nnzA=3]");
        assertArrayEquals(new long[]{2}, dBValues.shape(),
                "dBValues shape must be [nnzB=2]");

        // ── Output dtypes ────────────────────────────────────────────────────
        assertEquals(DataType.DOUBLE, dAValues.dataType(),
                "dAValues dtype must match gradCValues dtype (DOUBLE)");
        assertEquals(DataType.DOUBLE, dBValues.dataType(),
                "dBValues dtype must match gradCValues dtype (DOUBLE)");

        // ── Output values (all upstream grads are 1.0, so every bp grad = 1.0) ──
        double[] expectedA = {1.0, 1.0, 1.0};
        double[] expectedB = {1.0, 1.0};
        double tol = 1e-6;

        for (int i = 0; i < 3; i++) {
            assertEquals(expectedA[i], dAValues.getDouble(i), tol,
                    "dAValues[" + i + "] mismatch");
        }
        for (int i = 0; i < 2; i++) {
            assertEquals(expectedB[i], dBValues.getDouble(i), tol,
                    "dBValues[" + i + "] mismatch");
        }
    }

    /**
     * Sanity-checks a non-uniform upstream gradient to verify the gather logic selects
     * the correct C entry per A/B nonzero.
     *
     * <p>gradCValues = [10, 20, 30, 40, 50]:
     * <ul>
     *   <li>C row0 sorted cols: 0→g=10, 1→g=20, 2→g=30</li>
     *   <li>C row1 sorted cols: 1→g=40, 2→g=50</li>
     *   <li>A(0,0) maps to C row0 col0 → g=10;  A(0,2) → col2 → g=30;  A(1,1) → row1 col1 → g=40</li>
     *   <li>B(0,1) maps to C row0 col1 → g=20;  B(1,2) → row1 col2 → g=50</li>
     * </ul>
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testCsrAddBpNonUniformGrad(Nd4jBackend backend) {
        INDArray aColIdx = Nd4j.createFromArray(new int[]{0, 2, 1}).castTo(DataType.INT32);
        INDArray aRowPtr = Nd4j.createFromArray(new int[]{0, 2, 3}).castTo(DataType.INT32);
        INDArray bColIdx = Nd4j.createFromArray(new int[]{1, 2}).castTo(DataType.INT32);
        INDArray bRowPtr = Nd4j.createFromArray(new int[]{0, 1, 2}).castTo(DataType.INT32);
        INDArray cColIdx = Nd4j.createFromArray(new int[]{0, 1, 2, 1, 2}).castTo(DataType.INT32);
        INDArray cRowPtr = Nd4j.createFromArray(new int[]{0, 3, 5}).castTo(DataType.INT32);

        // Non-uniform upstream gradient
        INDArray gradCValues = Nd4j.createFromArray(new double[]{10.0, 20.0, 30.0, 40.0, 50.0});

        INDArray[] results = Nd4j.exec(new CsrAddBp(
                aColIdx, aRowPtr, bColIdx, bRowPtr,
                cColIdx, cRowPtr, gradCValues,
                2L, 3L));

        assertNotNull(results);
        assertEquals(2, results.length);

        INDArray dAValues = results[0];
        INDArray dBValues = results[1];

        assertArrayEquals(new long[]{3}, dAValues.shape());
        assertArrayEquals(new long[]{2}, dBValues.shape());

        double tol = 1e-6;
        // A(0,0) → C[row0, col0] at C index 0 → grad=10
        assertEquals(10.0, dAValues.getDouble(0), tol, "dAValues[0] A(0,0) grad");
        // A(0,2) → C[row0, col2] at C index 2 → grad=30
        assertEquals(30.0, dAValues.getDouble(1), tol, "dAValues[1] A(0,2) grad");
        // A(1,1) → C[row1, col1] at C index 3 → grad=40
        assertEquals(40.0, dAValues.getDouble(2), tol, "dAValues[2] A(1,1) grad");

        // B(0,1) → C[row0, col1] at C index 1 → grad=20
        assertEquals(20.0, dBValues.getDouble(0), tol, "dBValues[0] B(0,1) grad");
        // B(1,2) → C[row1, col2] at C index 4 → grad=50
        assertEquals(50.0, dBValues.getDouble(1), tol, "dBValues[1] B(1,2) grad");
    }
}
