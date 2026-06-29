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
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * End-to-end GNN TRAINING test: transductive node classification with a 2-layer GCN.
 *
 * <h3>Graph</h3>
 * 8 nodes split into two highly-connected clusters:
 * <ul>
 *   <li>Cluster 0 (nodes 0-3, class 0): path 0-1-2-3 + diagonal 0-2</li>
 *   <li>Cluster 1 (nodes 4-7, class 1): path 4-5-6-7 + diagonal 5-7</li>
 *   <li>Bridge: edge 3-4</li>
 * </ul>
 * Self-loops are added to all nodes; edge weights are symmetrically normalised
 * (D^{-1/2} (A+I) D^{-1/2}, GCN-style) by the helper {@link #buildGcnAdj}.
 *
 * <h3>Features</h3>
 * Node features are perfectly class-discriminative:
 * nodes 0-3 → [1.0, 0.0], nodes 4-7 → [0.0, 1.0].
 *
 * <h3>Model</h3>
 * <pre>
 *   h1 = gcnConv(X[8,2], W1[2,4], b1[4], A_norm, relu=true)   [8,4]
 *   logits = gcnConv(h1[8,4], W2[4,2], b2[2], A_norm, relu=false) [8,2]
 *   loss = sparseSoftmaxCrossEntropy(logits, intLabels)
 * </pre>
 *
 * <h3>Training</h3>
 * 200 epochs of vanilla SGD (lr = 0.02) via {@link SameDiff#calculateGradients}.
 *
 * <h3>Assertions</h3>
 * <ul>
 *   <li>Final loss &lt; 0.5 * initial loss (loss halved)</li>
 *   <li>Training accuracy &ge; 6/8 nodes classified correctly</li>
 * </ul>
 */
@Tag(TagNames.SAMEDIFF)
@Tag(TagNames.TRAINING)
@Tag("sparse")
@Tag("gnn")
public class GnnNodeClassificationTest extends BaseNd4jTestWithBackends {

    private static final int N  = 8;   // number of nodes
    private static final int F  = 2;   // input feature dim
    private static final int H  = 4;   // hidden dim
    private static final int C  = 2;   // number of classes

    @BeforeEach
    public void purgeConstants() {
        Nd4j.getConstantHandler().purgeConstants();
    }

    // -----------------------------------------------------------------------
    // CSR adjacency builder (A+I, D^{-1/2}·Â·D^{-1/2} normalisation)
    // -----------------------------------------------------------------------

    /**
     * Builds the CSR representation of the symmetrically-normalised adjacency
     * matrix for any undirected graph, with self-loops added.
     *
     * @param nNodes         total number of nodes
     * @param undirectedEdges pairs [src, dst] for the undirected edge set
     *                        (each undirected edge should appear ONCE; both
     *                        directions are added internally)
     * @return [normVals, colIdx(INT32), rowPtr(INT32)]
     */
    private static INDArray[] buildGcnAdj(int nNodes, int[][] undirectedEdges) {
        // Expand to directed + add self-loops
        List<int[]> edges = new ArrayList<>();
        for (int[] e : undirectedEdges) {
            edges.add(new int[]{e[0], e[1]});
            if (e[0] != e[1]) edges.add(new int[]{e[1], e[0]});
        }
        for (int i = 0; i < nNodes; i++) edges.add(new int[]{i, i});

        // Sort by (row, col)
        edges.sort((a, b) -> a[0] != b[0] ? a[0] - b[0] : a[1] - b[1]);

        int nnz = edges.size();
        int[] colIdxArr = new int[nnz];
        int[] rowPtrArr = new int[nNodes + 1];

        for (int k = 0; k < nnz; k++) {
            colIdxArr[k] = edges.get(k)[1];
            rowPtrArr[edges.get(k)[0] + 1]++;
        }
        for (int i = 0; i < nNodes; i++) rowPtrArr[i + 1] += rowPtrArr[i];

        // Degree per node = number of non-zeros in its row (including self-loop)
        double[] deg = new double[nNodes];
        for (int i = 0; i < nNodes; i++) deg[i] = rowPtrArr[i + 1] - rowPtrArr[i];

        // Row index of each nnz entry (for weight computation)
        int[] rowOf = new int[nnz];
        for (int i = 0; i < nNodes; i++)
            for (int k = rowPtrArr[i]; k < rowPtrArr[i + 1]; k++) rowOf[k] = i;

        // Symmetric normalisation: w_ij = 1 / sqrt(d_i * d_j)
        double[] normVals = new double[nnz];
        for (int k = 0; k < nnz; k++)
            normVals[k] = 1.0 / Math.sqrt(deg[rowOf[k]] * deg[colIdxArr[k]]);

        return new INDArray[]{
                Nd4j.createFromArray(normVals).castTo(DataType.DOUBLE),
                Nd4j.createFromArray(colIdxArr),
                Nd4j.createFromArray(rowPtrArr)
        };
    }

    // -----------------------------------------------------------------------
    // SGD helper
    // -----------------------------------------------------------------------

    private static void sgdUpdate(SameDiff sd, double lr, String... paramNames) {
        Map<String, INDArray> grads = sd.calculateGradients(null, paramNames);
        for (String name : paramNames) {
            INDArray g = grads.get(name);
            if (g != null) sd.getVariable(name).getArr().subi(g.muli(lr));
        }
    }

    // -----------------------------------------------------------------------
    // Test
    // -----------------------------------------------------------------------

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGcnNodeClassificationTrains(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(42L);

        // ---- Graph topology (undirected edges) ----
        //   Cluster 0: path 0-1-2-3 + diagonal 0-2
        //   Cluster 1: path 4-5-6-7 + diagonal 5-7
        //   Bridge:    3-4
        int[][] undirectedEdges = {
                {0,1},{1,2},{2,3},{0,2},        // cluster 0
                {4,5},{5,6},{6,7},{5,7},         // cluster 1
                {3,4}                            // bridge
        };
        INDArray[] adj = buildGcnAdj(N, undirectedEdges);
        INDArray normValsArr = adj[0];   // [nnz] DOUBLE
        INDArray colIdxArr   = adj[1];   // [nnz] INT32
        INDArray rowPtrArr   = adj[2];   // [N+1] INT32

        // ---- Node features: class 0 → [1,0], class 1 → [0,1] ----
        INDArray xArr = Nd4j.createFromArray(new double[][]{
                {1.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}, {1.0, 0.0},
                {0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}
        });

        // ---- Integer class labels ----
        INDArray intLabelsArr = Nd4j.createFromArray(new int[]{0, 0, 0, 0, 1, 1, 1, 1});

        // ---- Weight initialisation (small random, offset from zero) ----
        INDArray w1Arr = Nd4j.rand(DataType.DOUBLE, F, H).subi(0.5).muli(0.2);
        INDArray b1Arr = Nd4j.zeros(DataType.DOUBLE, H);
        INDArray w2Arr = Nd4j.rand(DataType.DOUBLE, H, C).subi(0.5).muli(0.2);
        INDArray b2Arr = Nd4j.zeros(DataType.DOUBLE, C);

        SameDiff sd = SameDiff.create();
        try {
            // Structural INT32 constants — skipped by gradient computation
            SDVariable colIdx = sd.constant("colIdx", colIdxArr);
            SDVariable rowPtr = sd.constant("rowPtr", rowPtrArr);

            // Fixed DOUBLE constants (graph + labels)
            SDVariable aVals  = sd.constant("aVals",  normValsArr);
            SDVariable X      = sd.constant("X",      xArr);
            SDVariable labels = sd.constant("labels", intLabelsArr);

            // Trainable parameters
            SDVariable W1 = sd.var("W1", w1Arr);
            SDVariable b1 = sd.var("b1", b1Arr);
            SDVariable W2 = sd.var("W2", w2Arr);
            SDVariable b2 = sd.var("b2", b2Arr);

            // 2-layer GCN
            SDVariable h1     = sd.gnn().gcnConv(X,  W1, b1, aVals, colIdx, rowPtr, N, N, true);
            SDVariable logits = sd.gnn().gcnConv(h1, W2, b2, aVals, colIdx, rowPtr, N, N, false);

            // Loss (marked as loss automatically by the op)
            SDVariable loss = sd.loss().sparseSoftmaxCrossEntropy("loss", logits, labels);

            // --- Capture initial loss ---
            double initialLoss = loss.eval().getDouble(0);

            // --- Training loop: 200 epochs of SGD ---
            final double LR = 0.02;
            final int    EPOCHS = 200;
            String[] params = {"W1", "b1", "W2", "b2"};

            for (int epoch = 0; epoch < EPOCHS; epoch++) {
                sgdUpdate(sd, LR, params);
            }

            double finalLoss = loss.eval().getDouble(0);

            // --- Compute training accuracy ---
            INDArray logitsArr = logits.eval();  // [8, 2]
            int correct = 0;
            int[] expectedLabels = {0, 0, 0, 0, 1, 1, 1, 1};
            for (int i = 0; i < N; i++) {
                int pred = logitsArr.getDouble(i, 0) >= logitsArr.getDouble(i, 1) ? 0 : 1;
                if (pred == expectedLabels[i]) correct++;
            }

            assertTrue(finalLoss < 0.5 * initialLoss,
                    String.format("GCN node-class loss did not halve: initial=%.4f final=%.4f",
                            initialLoss, finalLoss));

            assertTrue(correct >= 6,
                    String.format("GCN node-class accuracy too low: %d/8 (need >=6)", correct));

        } finally {
            sd.close();
        }
    }
}
