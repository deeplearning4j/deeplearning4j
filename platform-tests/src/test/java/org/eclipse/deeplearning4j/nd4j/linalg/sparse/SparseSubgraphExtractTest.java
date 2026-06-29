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
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.validation.GradCheckUtil;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.sparse.CsrSubgraphExtract;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for {@link CsrSubgraphExtract} and its backward pass.
 *
 * <h3>Test graph (N=5 nodes, nnz=9)</h3>
 * <pre>
 *   Adjacency:
 *     0 -> 1 (w=1.0),  0 -> 3 (w=2.0)
 *     1 -> 0 (w=3.0),  1 -> 2 (w=4.0)
 *     2 -> 1 (w=5.0),  2 -> 4 (w=6.0)
 *     3 -> 0 (w=7.0)
 *     4 -> 2 (w=8.0),  4 -> 3 (w=9.0)
 *
 *   values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
 *   colIdx = [1, 3, 0, 2, 1, 4, 0, 2, 3]  (INT32)
 *   rowPtr = [0, 2, 4, 6, 7, 9]            (INT32)
 * </pre>
 *
 * <h3>Selected nodes: nodeIdx = [1, 2, 4]  (K=3)</h3>
 * <p>Remapping: 1→0, 2→1, 4→2</p>
 *
 * <p>Kept edges (source AND destination both selected):
 * <pre>
 *   e=3: 1->2 w=4.0  remapped as 0->1  (row 1 of original = selected row 0)
 *   e=4: 2->1 w=5.0  remapped as 1->0  (row 2 of original = selected row 1)
 *   e=5: 2->4 w=6.0  remapped as 1->2  (row 2 of original = selected row 1)
 *   e=7: 4->2 w=8.0  remapped as 2->1  (row 4 of original = selected row 2)
 * </pre>
 *
 * <p>Expected outputs:
 * <pre>
 *   newValues = [4.0, 5.0, 6.0, 8.0]   nnz'=4
 *   newColIdx = [1, 0, 2, 1]            (INT32, remapped)
 *   newRowPtr = [0, 1, 3, 4]            (INT32)
 * </pre>
 *
 * <h3>Gradient-check graph (N=3, K=2, nodeIdx=[0,2])</h3>
 * <pre>
 *   values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
 *   colIdx = [1, 2, 0, 2, 0, 1]  (INT32)
 *   rowPtr = [0, 2, 4, 6]        (INT32)
 *   nodeIdx = [0, 2]             (INT32, sorted)
 *
 *   Kept edges:
 *     e=1: 0->2 w=2.0 remapped as 0->1
 *     e=4: 2->0 w=5.0 remapped as 1->0
 *   newValues=[2.0, 5.0], newColIdx=[1,0], newRowPtr=[0,1,2]
 * </pre>
 */
public class SparseSubgraphExtractTest extends BaseNd4jTestWithBackends {

    private static final double TOL = 1e-5;

    /**
     * Purge the constant-handler cache before every test to avoid stale-pointer
     * crashes when the backend releases constant buffers between tests.
     */
    @BeforeEach
    public void purgeConstantHandlerCache() {
        Nd4j.getConstantHandler().purgeConstants();
    }

    // -----------------------------------------------------------------------
    // Forward correctness test — N=5, K=3, nodeIdx=[1,2,4]
    // -----------------------------------------------------------------------

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testForwardCorrectness(Nd4jBackend backend) {
        // Build the N=5 test graph
        INDArray values  = Nd4j.createFromArray(new double[]{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0});
        INDArray colIdx  = Nd4j.createFromArray(new int[]{1, 3, 0, 2, 1, 4, 0, 2, 3});
        INDArray rowPtr  = Nd4j.createFromArray(new int[]{0, 2, 4, 6, 7, 9});
        INDArray nodeIdx = Nd4j.createFromArray(new int[]{1, 2, 4});

        long N = 5, K = 3;

        // Run the op
        CsrSubgraphExtract op = new CsrSubgraphExtract(values, colIdx, rowPtr, nodeIdx, N, K);
        INDArray[] outputs = Nd4j.exec(op);

        INDArray newValues = outputs[0];
        INDArray newColIdx = outputs[1];
        INDArray newRowPtr = outputs[2];

        // Check nnz' == 4
        assertEquals(4, newValues.length(), "newValues length (nnz')");
        assertEquals(4, newColIdx.length(), "newColIdx length");
        assertEquals(K + 1, newRowPtr.length(), "newRowPtr length");

        // Check newValues (order follows selected-row scan: 1,2,4 → edges sorted by original row)
        double[] expValues = {4.0, 5.0, 6.0, 8.0};
        for (int i = 0; i < 4; i++) {
            assertEquals(expValues[i], newValues.getDouble(i), TOL,
                    "newValues[" + i + "]");
        }

        // Check newColIdx
        int[] expColIdx = {1, 0, 2, 1};
        for (int i = 0; i < 4; i++) {
            assertEquals(expColIdx[i], newColIdx.getInt(i),
                    "newColIdx[" + i + "]");
        }

        // Check newRowPtr
        int[] expRowPtr = {0, 1, 3, 4};
        for (int i = 0; i <= K; i++) {
            assertEquals(expRowPtr[i], newRowPtr.getInt(i),
                    "newRowPtr[" + i + "]");
        }
    }

    // -----------------------------------------------------------------------
    // Empty subgraph: K=0
    // -----------------------------------------------------------------------

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testEmptySubgraph_K0(Nd4jBackend backend) {
        INDArray values  = Nd4j.createFromArray(new double[]{1.0, 2.0, 3.0});
        INDArray colIdx  = Nd4j.createFromArray(new int[]{1, 0, 0});
        INDArray rowPtr  = Nd4j.createFromArray(new int[]{0, 1, 2, 3});
        INDArray nodeIdx = Nd4j.createFromArray(new int[0]);  // K=0

        long N = 3, K = 0;
        CsrSubgraphExtract op = new CsrSubgraphExtract(values, colIdx, rowPtr, nodeIdx, N, K);
        INDArray[] outputs = Nd4j.exec(op);

        assertEquals(0, outputs[0].length(), "newValues must be empty for K=0");
        assertEquals(0, outputs[1].length(), "newColIdx must be empty for K=0");
        assertEquals(1, outputs[2].length(), "newRowPtr length must be K+1=1 for K=0");
    }

    // -----------------------------------------------------------------------
    // No kept edges: selected nodes exist but no cross-selected edges
    // -----------------------------------------------------------------------

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testNoKeptEdges(Nd4jBackend backend) {
        // 3-node graph: only 0->1 and 1->0; selected=[0,2] — no edge between 0 and 2
        INDArray values  = Nd4j.createFromArray(new double[]{1.0, 2.0});
        INDArray colIdx  = Nd4j.createFromArray(new int[]{1, 0});
        INDArray rowPtr  = Nd4j.createFromArray(new int[]{0, 1, 2, 2});  // node 2 has no edges
        INDArray nodeIdx = Nd4j.createFromArray(new int[]{0, 2});

        long N = 3, K = 2;
        CsrSubgraphExtract op = new CsrSubgraphExtract(values, colIdx, rowPtr, nodeIdx, N, K);
        INDArray[] outputs = Nd4j.exec(op);

        assertEquals(0, outputs[0].length(), "newValues must be empty when no cross-selected edges");
        assertEquals(0, outputs[1].length(), "newColIdx must be empty");
        // newRowPtr = [0, 0, 0]
        assertEquals(K + 1, outputs[2].length(), "newRowPtr length");
        assertEquals(0, outputs[2].getInt(0), "newRowPtr[0]");
        assertEquals(0, outputs[2].getInt(1), "newRowPtr[1]");
        assertEquals(0, outputs[2].getInt(2), "newRowPtr[2]");
    }

    // -----------------------------------------------------------------------
    // SameDiff gradient check — N=3, K=2, nodeIdx=[0,2]
    // -----------------------------------------------------------------------

    /**
     * Gradient check for {@code csr_subgraph_extract} w.r.t. the float values input.
     *
     * <p>Graph: N=3, nnz=6, K=2 selected nodes [0,2].
     * Expected kept edges: e=1 (0->2 w=2.0 → 0->1) and e=4 (2->0 w=5.0 → 1->0).
     * Loss = mean(newValues). Gradient check verifies doDiff is correct.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGradCheck(Nd4jBackend backend) {
        // All DOUBLE for numerical-differentiation accuracy
        INDArray valArr  = Nd4j.createFromArray(new double[]{1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
        INDArray ciArr   = Nd4j.createFromArray(new int[]{1, 2, 0, 2, 0, 1});
        INDArray rpArr   = Nd4j.createFromArray(new int[]{0, 2, 4, 6});
        INDArray niArr   = Nd4j.createFromArray(new int[]{0, 2});

        long N = 3, K = 2;

        SameDiff sd = SameDiff.create();
        try {
            // values is the differentiable variable; the rest are constants (INT — skipped by gradcheck)
            SDVariable values  = sd.var("values", valArr);
            SDVariable colIdx  = sd.constant("colIdx", ciArr);
            SDVariable rowPtr  = sd.constant("rowPtr", rpArr);
            SDVariable nodeIdx = sd.constant("nodeIdx", niArr);

            // Forward op — 3 outputs
            CsrSubgraphExtract fwdOp = new CsrSubgraphExtract(sd, values, colIdx, rowPtr, nodeIdx, N, K);
            SDVariable[] fwdOuts = fwdOp.outputVariables();

            // fwdOuts[0] = newValues[nnz'], fwdOuts[1] = newColIdx, fwdOuts[2] = newRowPtr
            // Loss = mean of newValues (scalar)
            sd.mean("loss", fwdOuts[0]);
            // The op has 3 outputs (values/colIdx/rowPtr) so SameDiff cannot auto-infer
            // the single loss terminal — mark it explicitly (multi-output gradcheck rule).
            sd.setLossVariables("loss");

            assertTrue(
                    GradCheckUtil.checkGradients(sd, null),
                    "Gradient check failed for csr_subgraph_extract (values input)");
        } finally {
            sd.close();
        }
    }
}
