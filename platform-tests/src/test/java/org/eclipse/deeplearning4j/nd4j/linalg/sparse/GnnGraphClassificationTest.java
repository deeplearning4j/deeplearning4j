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
import org.nd4j.linalg.api.ops.impl.graph.GraphPooling;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * End-to-end GNN TRAINING test: graph classification with GIN + global sum pooling.
 *
 * <h3>Dataset</h3>
 * 6 tiny graphs batched into a single block-diagonal adjacency:
 * <ul>
 *   <li>3 path-graphs (3 nodes each, class 0): 0-1-2, 6-7-8, 12-13-14</li>
 *   <li>3 star-graphs (3 nodes each, class 1): 3-4/3-5, 9-10/9-11, 15-16/15-17</li>
 * </ul>
 * Graphs are interleaved as path, star, path, star, path, star.
 * Total 18 nodes (6 graphs × 3 nodes/graph).
 *
 * <h3>Features</h3>
 * Features are class-discriminative at the graph level:
 * <ul>
 *   <li>Path-graph nodes: [1.0, 0.0]</li>
 *   <li>Star-graph nodes: [0.0, 1.0]</li>
 * </ul>
 * After sum pooling a 3-node path gives [3,0]; a star gives [0,3] —
 * perfectly linearly separable, ensuring training convergence.
 *
 * <h3>Model</h3>
 * <pre>
 *   ginOut = ginConv(X[18,2], w1[2,4], b1[4], w2[4,4], b2[4], eps, colIdx, rowPtr, 18, 18,
 *                   layerNorm=true)                                            [18, 4]
 *   pooled = globalSumPool(ginOut, graphIds, numGraphs=6)                      [6, 4]
 *   logits = pooled @ W3[4,2] + b3[2]                                         [6, 2]
 *   loss   = sparseSoftmaxCrossEntropy(logits, graphLabels[6])
 * </pre>
 *
 * <h3>Training</h3>
 * 1000 epochs of SGD (lr = 0.02) with per-tensor L2-norm gradient clipping
 * (threshold = 2.0) via {@link SameDiff#calculateGradients}.  LayerNorm in the GIN
 * layer stabilises the gradient magnitude on both CPU and CUDA backends, allowing the
 * clip threshold to be relaxed from the previous tight value of 1.0 while keeping
 * convergence reliable on CUDA.  Global sum pooling amplifies normalised embeddings
 * by up to √3 ≈ 1.73 (3 nodes/graph); the 2.0 threshold is a safety net for
 * early-epoch transients only.
 *
 * <h3>Assertions</h3>
 * <ul>
 *   <li>Final loss &lt; 0.4 * initial loss (loss drops &gt; 60%)</li>
 *   <li>Classification accuracy &ge; 5/6 graphs on BOTH backends</li>
 * </ul>
 */
@Tag(TagNames.SAMEDIFF)
@Tag(TagNames.TRAINING)
@Tag("sparse")
@Tag("gnn")
public class GnnGraphClassificationTest extends BaseNd4jTestWithBackends {

    /** Total nodes across all 6 batched graphs. */
    private static final int TOTAL_NODES = 18;
    /** Number of graphs in the batch. */
    private static final int NUM_GRAPHS  = 6;
    /** Input feature dimension. */
    private static final int F  = 2;
    /** GIN hidden dimension. */
    private static final int H  = 4;
    /** Number of graph classes. */
    private static final int C  = 2;

    @BeforeEach
    public void purgeConstants() {
        Nd4j.getConstantHandler().purgeConstants();
    }

    // -----------------------------------------------------------------------
    // SGD helper
    // -----------------------------------------------------------------------

    /**
     * SGD step with per-tensor L2-norm gradient clipping (threshold = 2.0).
     *
     * <p>LayerNorm in the GIN layer normalises each node's output embedding to
     * zero-mean / unit-variance, which substantially reduces gradient variance on both
     * CPU and CUDA backends.  A clip threshold of 2.0 (2× the original tight 1.0) is
     * sufficient to keep both backends on a stable convergence trajectory: global sum
     * pooling amplifies the normalised embeddings by up to √3 ≈ 1.73 (3 nodes per
     * graph), so typical gradient L2 norms at W3/b3 remain well below 2.0 throughout
     * training; the clip therefore acts only as a safety net for early-epoch transients.
     */
    private static void sgdUpdate(SameDiff sd, double lr, String... paramNames) {
        Map<String, INDArray> grads = sd.calculateGradients(null, paramNames);
        for (String name : paramNames) {
            INDArray g = grads.get(name);
            if (g == null) continue;
            // Per-tensor L2-norm clipping (threshold 2.0).
            // LayerNorm reduces effective gradient norms; 2.0 is loose enough for
            // normal gradient flow but prevents catastrophic early-epoch spikes on CUDA.
            double norm = g.norm2Number().doubleValue();
            if (norm > 2.0) g = g.mul(2.0 / norm);
            sd.getVariable(name).getArr().subi(g.muli(lr));
        }
    }

    // -----------------------------------------------------------------------
    // Test
    // -----------------------------------------------------------------------

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGinGraphClassificationTrains(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(123L);

        // ---------------------------------------------------------------
        // Batch CSR adjacency (directed edges, no self-loops — GIN's
        // (1+eps)*X term handles self-connections)
        //
        // Graph 0 (path, nodes 0-2):  0-1, 1-2
        // Graph 1 (star, nodes 3-5):  3-4, 3-5
        // Graph 2 (path, nodes 6-8):  6-7, 7-8
        // Graph 3 (star, nodes 9-11): 9-10, 9-11
        // Graph 4 (path, nodes 12-14):12-13,13-14
        // Graph 5 (star, nodes 15-17):15-16,15-17
        //
        // Both directions for each undirected edge (adjacency is symmetric).
        // ---------------------------------------------------------------
        int[] colIdxData = {
                // row 0: [1]
                1,
                // row 1: [0, 2]
                0, 2,
                // row 2: [1]
                1,
                // row 3: [4, 5]
                4, 5,
                // row 4: [3]
                3,
                // row 5: [3]
                3,
                // row 6: [7]
                7,
                // row 7: [6, 8]
                6, 8,
                // row 8: [7]
                7,
                // row 9: [10, 11]
                10, 11,
                // row 10: [9]
                9,
                // row 11: [9]
                9,
                // row 12: [13]
                13,
                // row 13: [12, 14]
                12, 14,
                // row 14: [13]
                13,
                // row 15: [16, 17]
                16, 17,
                // row 16: [15]
                15,
                // row 17: [15]
                15
        };
        // rowPtr: cumulative out-degree per row
        // Degrees: 1,2,1, 2,1,1, 1,2,1, 2,1,1, 1,2,1, 2,1,1
        int[] rowPtrData = {0, 1, 3, 4, 6, 7, 8, 9, 11, 12, 14, 15, 16, 17, 19, 20, 22, 23, 24};

        INDArray colIdxArr = Nd4j.createFromArray(colIdxData);
        INDArray rowPtrArr = Nd4j.createFromArray(rowPtrData);

        // ---------------------------------------------------------------
        // Node features [18, 2]:
        //   path-graph nodes (graphs 0,2,4) → [1, 0]
        //   star-graph nodes (graphs 1,3,5) → [0, 1]
        // ---------------------------------------------------------------
        double[][] xData = new double[TOTAL_NODES][F];
        // graphs 0,2,4 = path (nodes 0-2, 6-8, 12-14)
        for (int g = 0; g < NUM_GRAPHS; g += 2) {
            for (int k = 0; k < 3; k++) {
                xData[g * 3 + k][0] = 1.0;
                xData[g * 3 + k][1] = 0.0;
            }
        }
        // graphs 1,3,5 = star (nodes 3-5, 9-11, 15-17)
        for (int g = 1; g < NUM_GRAPHS; g += 2) {
            for (int k = 0; k < 3; k++) {
                xData[g * 3 + k][0] = 0.0;
                xData[g * 3 + k][1] = 1.0;
            }
        }
        INDArray xArr = Nd4j.createFromArray(xData);

        // graphIds: 3 nodes per graph → [0,0,0, 1,1,1, 2,2,2, 3,3,3, 4,4,4, 5,5,5]
        int[] graphIdsData = new int[TOTAL_NODES];
        for (int g = 0; g < NUM_GRAPHS; g++)
            for (int k = 0; k < 3; k++) graphIdsData[g * 3 + k] = g;
        INDArray graphIdsArr = Nd4j.createFromArray(graphIdsData);

        // Graph labels: even graphs (0,2,4) = path = class 0; odd = star = class 1
        INDArray graphLabelsArr = Nd4j.createFromArray(new int[]{0, 1, 0, 1, 0, 1});

        // ---------------------------------------------------------------
        // Parameter initialisation
        // ---------------------------------------------------------------
        INDArray w1Arr  = Nd4j.rand(DataType.DOUBLE, F, H).subi(0.5).muli(0.2);
        INDArray b1Arr  = Nd4j.zeros(DataType.DOUBLE, H);
        INDArray w2Arr  = Nd4j.rand(DataType.DOUBLE, H, H).subi(0.5).muli(0.2);
        INDArray b2Arr  = Nd4j.zeros(DataType.DOUBLE, H);
        INDArray epsArr = Nd4j.scalar(DataType.DOUBLE, 0.0);
        INDArray W3Arr  = Nd4j.rand(DataType.DOUBLE, H, C).subi(0.5).muli(0.2);
        INDArray b3Arr  = Nd4j.zeros(DataType.DOUBLE, C);

        SameDiff sd = SameDiff.create();
        try {
            // Constants
            SDVariable colIdx    = sd.constant("colIdx",    colIdxArr);
            SDVariable rowPtr    = sd.constant("rowPtr",    rowPtrArr);
            SDVariable X         = sd.constant("X",         xArr);
            SDVariable graphIds  = sd.constant("graphIds",  graphIdsArr);
            SDVariable graphLabels = sd.constant("graphLabels", graphLabelsArr);

            // Trainable parameters
            SDVariable w1  = sd.var("w1",  w1Arr);
            SDVariable b1  = sd.var("b1",  b1Arr);
            SDVariable w2  = sd.var("w2",  w2Arr);
            SDVariable b2  = sd.var("b2",  b2Arr);
            SDVariable eps = sd.var("eps", epsArr);
            SDVariable W3  = sd.var("W3",  W3Arr);
            SDVariable b3  = sd.var("b3",  b3Arr);

            // GIN layer with LayerNorm: [18, F] → [18, H]
            // layerNorm=true normalises each node embedding over the feature dim,
            // stabilising gradient magnitudes on both CPU and CUDA backends.
            SDVariable ginOut = sd.gnn().ginConv(
                    X, w1, b1, w2, b2, eps,
                    colIdx, rowPtr,
                    TOTAL_NODES, TOTAL_NODES, true);

            // Global sum pooling: [18, H] → [6, H]
            SDVariable pooled = GraphPooling.globalSumPool(sd, "pooled", ginOut, graphIds, NUM_GRAPHS);

            // Classification head: [6, H] → [6, C]
            SDVariable logits = sd.mmul(pooled, W3).add(b3);

            // Loss
            SDVariable loss = sd.loss().sparseSoftmaxCrossEntropy("loss", logits, graphLabels);

            // --- Capture initial loss ---
            double initialLoss = loss.eval().getDouble(0);

            // --- Training loop: 1000 epochs of SGD ---
            // LayerNorm reduces gradient variance; the clip (2.0) in sgdUpdate
            // is a safety net for early-epoch transients on both backends.
            final double LR     = 0.02;
            final int    EPOCHS = 1000;
            String[] params = {"w1", "b1", "w2", "b2", "eps", "W3", "b3"};

            for (int epoch = 0; epoch < EPOCHS; epoch++) {
                sgdUpdate(sd, LR, params);
            }

            double finalLoss = loss.eval().getDouble(0);

            // --- Compute accuracy ---
            INDArray logitsArr = logits.eval();  // [6, 2]
            int[] expectedGraphLabels = {0, 1, 0, 1, 0, 1};
            int correct = 0;
            for (int g = 0; g < NUM_GRAPHS; g++) {
                int pred = logitsArr.getDouble(g, 0) >= logitsArr.getDouble(g, 1) ? 0 : 1;
                if (pred == expectedGraphLabels[g]) correct++;
            }
            // Training-works assertion (BOTH backends): loss drops >60% (also catches NaN).
            assertTrue(finalLoss < 0.4 * initialLoss,
                    String.format("GIN graph-class loss did not drop 60%%: initial=%.4f final=%.4f",
                            initialLoss, finalLoss));

            // Strict classification accuracy on BOTH backends: LayerNorm stabilises the
            // gradient magnitude on CUDA as well as CPU, eliminating the gradient-explosion
            // feedback loop that previously made CUDA accuracy marginal.
            assertTrue(correct >= 5,
                    String.format("GIN graph-class accuracy too low: %d/6 (need >=5)", correct));

        } finally {
            sd.close();
        }
    }
}
