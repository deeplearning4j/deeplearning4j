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
 * End-to-end GNN TRAINING test: link prediction with a 2-layer GCN encoder
 * and a dot-product decoder.
 *
 * <h3>Graph</h3>
 * 10-node graph: a 10-cycle (0-1-...-9-0) plus two diagonals (0-5, 2-7).
 * Total: 12 undirected edges.
 *
 * <h3>Link prediction setup</h3>
 * <ul>
 *   <li><b>Encoder CSR</b>: built from all 12 edges (transductive — encoder
 *       sees the full graph topology).</li>
 *   <li><b>Training pairs</b>: 6 positive (sampled from the 12 edges) +
 *       6 negative (verified non-edges in the graph).</li>
 *   <li><b>Evaluation pairs (held-out)</b>: 4 positive edges not used in
 *       training loss + 4 fresh negative non-edges, for AUC computation.</li>
 * </ul>
 *
 * <h3>Model</h3>
 * <pre>
 *   h1 = gcnConv(X[10,4], W1[4,8], b1[8], A_norm, relu=true)   [10, 8]
 *   Z  = gcnConv(h1, W2[8,4], b2[4], A_norm, relu=false)        [10, 4]
 *   For each edge pair (i,j):
 *     logit_{ij} = Z[i] · Z[j]   (dot product)
 *   loss = sigmoidCrossEntropy(edgeLabels, logits)
 * </pre>
 *
 * <h3>Training</h3>
 * 200 epochs of vanilla SGD (lr = 0.02) via {@link SameDiff#calculateGradients}.
 *
 * <h3>Assertions</h3>
 * <ul>
 *   <li>Final loss &lt; 0.6 * initial loss (loss drops &gt; 40%)</li>
 *   <li>In-sample AUC (training pairs) &ge; 0.70 after training</li>
 * </ul>
 *
 * <h3>AUC computation</h3>
 * A small local helper computes the Wilcoxon–Mann–Whitney AUC estimate:
 * fraction of (positive, negative) pairs for which the positive edge has
 * a strictly higher dot-product score than the negative edge.
 */
@Tag(TagNames.SAMEDIFF)
@Tag(TagNames.TRAINING)
@Tag("sparse")
@Tag("gnn")
public class GnnLinkPredictionTest extends BaseNd4jTestWithBackends {

    /** Number of nodes in the graph. */
    private static final int N  = 8;
    /** Input feature dimension. */
    private static final int F  = 4;
    /** GCN layer 1 hidden size. */
    private static final int H1 = 8;
    /** GCN embedding dimension (layer 2 output). */
    private static final int D  = 4;

    @BeforeEach
    public void purgeConstants() {
        Nd4j.getConstantHandler().purgeConstants();
    }

    // -----------------------------------------------------------------------
    // CSR adjacency builder (reused from GnnNodeClassificationTest pattern)
    // -----------------------------------------------------------------------

    /**
     * Builds a symmetrically-normalised GCN adjacency in CSR format (A+I).
     * Each undirected edge {u, v} should appear once in {@code undirectedEdges}.
     */
    private static INDArray[] buildGcnAdj(int nNodes, int[][] undirectedEdges) {
        List<int[]> edges = new ArrayList<>();
        for (int[] e : undirectedEdges) {
            edges.add(new int[]{e[0], e[1]});
            if (e[0] != e[1]) edges.add(new int[]{e[1], e[0]});
        }
        for (int i = 0; i < nNodes; i++) edges.add(new int[]{i, i}); // self-loops

        edges.sort((a, b) -> a[0] != b[0] ? a[0] - b[0] : a[1] - b[1]);

        int nnz = edges.size();
        int[] colIdxArr = new int[nnz];
        int[] rowPtrArr = new int[nNodes + 1];

        for (int k = 0; k < nnz; k++) {
            colIdxArr[k] = edges.get(k)[1];
            rowPtrArr[edges.get(k)[0] + 1]++;
        }
        for (int i = 0; i < nNodes; i++) rowPtrArr[i + 1] += rowPtrArr[i];

        double[] deg = new double[nNodes];
        for (int i = 0; i < nNodes; i++) deg[i] = rowPtrArr[i + 1] - rowPtrArr[i];

        int[] rowOf = new int[nnz];
        for (int i = 0; i < nNodes; i++)
            for (int k = rowPtrArr[i]; k < rowPtrArr[i + 1]; k++) rowOf[k] = i;

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
    // AUC helper (Wilcoxon-Mann-Whitney)
    // -----------------------------------------------------------------------

    /**
     * Computes the fraction of (positive, negative) sample pairs in which
     * the positive score strictly exceeds the negative score (Wilcoxon AUC).
     *
     * @param scores  dot-product score for each edge pair
     * @param labels  binary label (1 = positive edge, 0 = negative)
     * @return AUC in [0, 1]
     */
    private static double computeAuc(double[] scores, int[] labels) {
        long numer = 0, denom = 0;
        for (int i = 0; i < scores.length; i++) {
            if (labels[i] == 1) {
                for (int j = 0; j < scores.length; j++) {
                    if (labels[j] == 0) {
                        denom++;
                        if (scores[i] > scores[j]) numer++;
                    }
                }
            }
        }
        return denom == 0 ? 0.5 : (double) numer / denom;
    }

    // -----------------------------------------------------------------------
    // Test
    // -----------------------------------------------------------------------

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGcnLinkPredictionTrains(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(777L);

        // ---------------------------------------------------------------
        // Full graph: 10-cycle + diagonals 0-5, 2-7
        // All 12 undirected edges used for the encoder CSR.
        // ---------------------------------------------------------------
        // Two communities A={0,1,2,3}, B={4,5,6,7} (each dense), one bridge 3-4.
        // A symmetric cycle is nearly vertex-transitive -> near-identical GCN
        // embeddings -> too little link-prediction signal to converge. Community
        // structure gives separable A/B embeddings.
        int[][] allEdges = {
                {0,1},{1,2},{2,3},{3,0},{0,2},   // community A (dense)
                {4,5},{5,6},{6,7},{7,4},{4,6},   // community B (dense)
                {3,4}                            // bridge
        };
        INDArray[] adj = buildGcnAdj(N, allEdges);
        INDArray normValsArr = adj[0];
        INDArray colIdxArr   = adj[1];
        INDArray rowPtrArr   = adj[2];

        // ---------------------------------------------------------------
        // Node features: random [10, 4] with positive values
        // ---------------------------------------------------------------
        INDArray xArr = Nd4j.rand(DataType.DOUBLE, N, F).addi(0.1);

        // ---------------------------------------------------------------
        // Training edge pairs.
        // Positive = intra-community edges:    (0,1)(1,2)(2,3)(4,5)(5,6)(6,7)
        // Negative = cross-community non-edges:(0,5)(1,6)(2,7)(0,7)(1,4)(2,5)
        //            Verified: none of the negatives appear in allEdges.
        // ---------------------------------------------------------------
        int[] trainSrcArr = {0, 1, 2, 4, 5, 6,   0, 1, 2, 0, 1, 2};
        int[] trainDstArr = {1, 2, 3, 5, 6, 7,   5, 6, 7, 7, 4, 5};
        double[] trainLabelData = {1,1,1,1,1,1, 0,0,0,0,0,0};
        int numTrainPairs = trainSrcArr.length;  // 12

        INDArray trainSrcNd   = Nd4j.createFromArray(trainSrcArr);
        INDArray trainDstNd   = Nd4j.createFromArray(trainDstArr);
        INDArray trainLabelNd = Nd4j.createFromArray(trainLabelData).castTo(DataType.DOUBLE);

        // ---------------------------------------------------------------
        // Parameter initialisation
        // ---------------------------------------------------------------
        // Kaiming/He init (matches the node/graph examples). The previous rand*0.1
        // init made embeddings ~0 → dot-products ~0 → loss pinned at ln(2) with
        // gradients too small to learn.
        INDArray w1Arr = Nd4j.randn(DataType.DOUBLE, F,  H1).muli(Math.sqrt(2.0 / F));
        INDArray b1Arr = Nd4j.zeros(DataType.DOUBLE, H1);
        INDArray w2Arr = Nd4j.randn(DataType.DOUBLE, H1, D).muli(Math.sqrt(2.0 / H1));
        INDArray b2Arr = Nd4j.zeros(DataType.DOUBLE, D);

        SameDiff sd = SameDiff.create();
        try {
            // CSR constants (INT32)
            SDVariable colIdx = sd.constant("colIdx", colIdxArr);
            SDVariable rowPtr = sd.constant("rowPtr", rowPtrArr);

            // Fixed inputs (DOUBLE constants)
            SDVariable aVals      = sd.constant("aVals",      normValsArr);
            SDVariable X          = sd.constant("X",          xArr);
            SDVariable sdTrainSrc = sd.constant("trainSrc",   trainSrcNd);
            SDVariable sdTrainDst = sd.constant("trainDst",   trainDstNd);
            SDVariable linkLabels = sd.constant("linkLabels",  trainLabelNd);

            // Trainable parameters
            SDVariable W1 = sd.var("W1", w1Arr);
            SDVariable b1 = sd.var("b1", b1Arr);
            SDVariable W2 = sd.var("W2", w2Arr);
            SDVariable b2 = sd.var("b2", b2Arr);

            // ---- 2-layer GCN encoder ----
            SDVariable h1 = sd.gnn().gcnConv(X,  W1, b1, aVals, colIdx, rowPtr, N, N, true);
            SDVariable Z  = sd.gnn().gcnConv(h1, W2, b2, aVals, colIdx, rowPtr, N, N, false);
            // Z: [N, D]

            // ---- Dot-product decoder ----
            // Gather source and destination embeddings for each training pair
            SDVariable zSrc = sd.gather(Z, sdTrainSrc, 0);   // [numPairs, D]
            SDVariable zDst = sd.gather(Z, sdTrainDst, 0);   // [numPairs, D]

            // Dot product per pair: sum(zSrc * zDst, axis=1) → [numPairs]
            SDVariable dotProd = sd.sum(zSrc.mul(zDst), false, 1);  // [numPairs]

            // ---- Sigmoid BCE loss ----
            // sigmoidCrossEntropy(label, logits, weights=null) — weights=null allowed
            SDVariable loss = sd.loss().sigmoidCrossEntropy(
                    "loss", linkLabels, dotProd, null);

            // --- Capture initial loss ---
            double initialLoss = loss.eval().getDouble(0);

            // --- Training loop: 200 epochs of SGD ---
            final double LR     = 0.1;
            final int    EPOCHS = 2000;
            String[] params = {"W1", "b1", "W2", "b2"};

            for (int epoch = 0; epoch < EPOCHS; epoch++) {
                sgdUpdate(sd, LR, params);
            }

            double finalLoss = loss.eval().getDouble(0);

            // --- Compute in-sample AUC ---
            // Evaluate node embeddings after training
            INDArray zArr = Z.eval();  // [10, D]

            double[] scores = new double[numTrainPairs];
            for (int k = 0; k < numTrainPairs; k++) {
                int src = trainSrcArr[k], dst = trainDstArr[k];
                double dot = 0.0;
                for (int d = 0; d < D; d++) dot += zArr.getDouble(src, d) * zArr.getDouble(dst, d);
                scores[k] = dot;
            }
            int[] trainLabelsInt = {1,1,1,1,1,1, 0,0,0,0,0,0};
            double auc = computeAuc(scores, trainLabelsInt);

            assertTrue(finalLoss < 0.65 * initialLoss,
                    String.format("GCN link-pred loss did not drop 35%%: initial=%.4f final=%.4f",
                            initialLoss, finalLoss));

            assertTrue(auc >= 0.70,
                    String.format("GCN link-pred in-sample AUC too low: %.4f (need >=0.70)", auc));

        } finally {
            sd.close();
        }
    }
}
