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

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.GatLayer;
import org.deeplearning4j.nn.conf.layers.GcnLayer;
import org.deeplearning4j.nn.conf.layers.GinLayer;
import org.deeplearning4j.nn.conf.layers.GraphSageLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ndarray.SparseFormat;
import org.nd4j.linalg.api.ndarray.SparseNDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import static org.junit.jupiter.api.Assertions.*;

/**
 * DL4J MultiLayerNetwork integration tests for the GNN conf-layer wrappers:
 * {@link GcnLayer}, {@link GinLayer}, {@link GraphSageLayer}, {@link GatLayer}.
 *
 * <h3>Common 4-node directed graph (N=4, F=2 features, nOut=2)</h3>
 * <pre>
 *   Edges: 0→1, 0→2, 1→0, 1→3, 2→0, 3→1
 *
 *   CSR:
 *     values  = [1, 1, 1, 1, 1, 1]  FLOAT (raw, no normalization needed for GIN/SAGE/GAT)
 *     colIdx  = [1, 2, 0, 3, 0, 1]  INT32
 *     rowPtr  = [0, 2, 4, 5, 6]     INT32
 *     shape   = [4, 4]
 * </pre>
 *
 * <p>Tests verify:
 * <ol>
 *   <li>Forward output shape is {@code [numNodes, nOut]} / {@code [numNodes, nLabels]}.</li>
 *   <li>No NaN/Inf in forward output.</li>
 *   <li>For GcnLayer and GinLayer: a few iterations of {@code network.fit()} reduce the
 *       training score (loss decreases), confirming end-to-end gradient flow.</li>
 * </ol>
 */
@Tag(TagNames.SAMEDIFF)
public class SparseGnnDl4jLayerTest {

    // -----------------------------------------------------------------------
    // Graph constants
    // -----------------------------------------------------------------------

    private static final int N    = 4;   // nodes
    private static final int F    = 2;   // input features per node
    private static final int NOUT = 2;   // output features / labels

    // -----------------------------------------------------------------------
    // Shared helpers
    // -----------------------------------------------------------------------

    /**
     * Build a 4-node CSR adjacency matrix:
     * edges 0→1, 0→2, 1→0, 1→3, 2→0, 3→1.
     * Values are all 1.0 (float32), structure is INT32.
     */
    private static SparseNDArray makeAdj() {
        INDArray vals   = Nd4j.createFromArray(new float[]{1f, 1f, 1f, 1f, 1f, 1f});
        INDArray colIdx = Nd4j.createFromArray(new int[]{1, 2, 0, 3, 0, 1});
        INDArray rowPtr = Nd4j.createFromArray(new int[]{0, 2, 4, 5, 6});
        return new SparseNDArray(vals, colIdx, rowPtr, new long[]{N, N}, SparseFormat.CSR);
    }

    /**
     * Node feature matrix [N=4, F=2], float32, all positive.
     */
    private static INDArray makeFeatures() {
        return Nd4j.createFromArray(new float[][]{
                {1.0f, 2.0f},
                {3.0f, 4.0f},
                {5.0f, 6.0f},
                {7.0f, 8.0f}
        });
    }

    /**
     * One-hot label matrix [N=4, NOUT=2] for a 2-class node classification task.
     */
    private static INDArray makeLabels() {
        return Nd4j.createFromArray(new float[][]{
                {1.0f, 0.0f},
                {0.0f, 1.0f},
                {1.0f, 0.0f},
                {0.0f, 1.0f}
        });
    }

    // -----------------------------------------------------------------------
    // GCN — forward shape + no-NaN
    // -----------------------------------------------------------------------

    /**
     * Build a [GcnLayer(nIn=2, nOut=2) → OutputLayer(nIn=2, nOut=2)] MultiLayerNetwork,
     * set the adjacency, call {@code output()}, and verify shape and finiteness.
     */
    @Test
    public void testGcnLayerForwardShape() {
        SparseNDArray adj = makeAdj();

        GcnLayer gcnConf = new GcnLayer.Builder()
                .nIn(F).nOut(NOUT)
                .applyRelu(false)      // disable ReLU so pre-activations can't go all-zero
                .hasBias(true)
                .build();
        gcnConf.setAdjacency(adj);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .updater(new Adam(1e-3))
                .list()
                .layer(gcnConf)
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .activation(Activation.IDENTITY)
                        .nIn(NOUT).nOut(NOUT)
                        .build())
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        INDArray features = makeFeatures();
        INDArray out = net.output(features);

        assertNotNull(out, "GCN output must not be null");
        assertArrayEquals(new long[]{N, NOUT}, out.shape(),
                "GCN output shape must be [numNodes, nOut]");
        assertFalse(out.isNaN().any(),      "GCN output contains NaN");
        assertFalse(out.isInfinite().any(), "GCN output contains Inf");
    }

    // -----------------------------------------------------------------------
    // GCN — fit() reduces loss
    // -----------------------------------------------------------------------

    /**
     * Fit a [GcnLayer → OutputLayer] network for 10 iterations and verify that the
     * training score decreases, confirming end-to-end gradient flow through the
     * GCN composition and back-propagation via SameDiff automatic differentiation.
     */
    @Test
    public void testGcnLayerFitReducesLoss() {
        SparseNDArray adj = makeAdj();

        GcnLayer gcnConf = new GcnLayer.Builder()
                .nIn(F).nOut(NOUT)
                .applyRelu(false)
                .hasBias(true)
                .build();
        gcnConf.setAdjacency(adj);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .updater(new Adam(1e-2))
                .list()
                .layer(gcnConf)
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .activation(Activation.IDENTITY)
                        .nIn(NOUT).nOut(NOUT)
                        .build())
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        DataSet ds = new DataSet(makeFeatures(), makeLabels());

        // Run one iteration to get the baseline score
        net.fit(ds);
        double firstScore = net.score();
        assertFalse(Double.isNaN(firstScore),    "GCN training score is NaN after 1 iter");
        assertFalse(Double.isInfinite(firstScore), "GCN training score is Inf after 1 iter");

        // Train for 9 more iterations
        for (int i = 0; i < 9; i++) {
            net.fit(ds);
        }
        double lastScore = net.score();

        assertTrue(lastScore < firstScore,
                () -> "GCN loss should decrease over 10 iterations: first=" + firstScore + " last=" + lastScore);
    }

    // -----------------------------------------------------------------------
    // GIN — forward shape + no-NaN
    // -----------------------------------------------------------------------

    /**
     * Build a [GinLayer(nIn=2, hiddenSize=4, nOut=2) → OutputLayer] network,
     * verify forward output shape and finiteness.
     */
    @Test
    public void testGinLayerForwardShape() {
        SparseNDArray adj = makeAdj();

        GinLayer ginConf = new GinLayer.Builder()
                .nIn(F).nOut(NOUT).hiddenSize(4)
                .initEps(0.0)
                .build();
        ginConf.setAdjacency(adj);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .updater(new Adam(1e-3))
                .list()
                .layer(ginConf)
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .activation(Activation.IDENTITY)
                        .nIn(NOUT).nOut(NOUT)
                        .build())
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        INDArray features = makeFeatures();
        INDArray out = net.output(features);

        assertNotNull(out, "GIN output must not be null");
        assertArrayEquals(new long[]{N, NOUT}, out.shape(),
                "GIN output shape must be [numNodes, nOut]");
        assertFalse(out.isNaN().any(),      "GIN output contains NaN");
        assertFalse(out.isInfinite().any(), "GIN output contains Inf");
    }

    // -----------------------------------------------------------------------
    // GIN — fit() reduces loss
    // -----------------------------------------------------------------------

    /**
     * Fit a [GinLayer → OutputLayer] network for 10 iterations and verify the training
     * score decreases, confirming end-to-end gradient flow through ginConv (including
     * the learnable epsilon).
     */
    @Test
    public void testGinLayerFitReducesLoss() {
        SparseNDArray adj = makeAdj();

        GinLayer ginConf = new GinLayer.Builder()
                .nIn(F).nOut(NOUT).hiddenSize(4)
                .initEps(0.0)
                .build();
        ginConf.setAdjacency(adj);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .updater(new Adam(1e-2))
                .list()
                .layer(ginConf)
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .activation(Activation.IDENTITY)
                        .nIn(NOUT).nOut(NOUT)
                        .build())
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        DataSet ds = new DataSet(makeFeatures(), makeLabels());

        net.fit(ds);
        double firstScore = net.score();
        assertFalse(Double.isNaN(firstScore),    "GIN training score is NaN after 1 iter");
        assertFalse(Double.isInfinite(firstScore), "GIN training score is Inf after 1 iter");

        for (int i = 0; i < 9; i++) {
            net.fit(ds);
        }
        double lastScore = net.score();

        assertTrue(lastScore < firstScore,
                () -> "GIN loss should decrease over 10 iterations: first=" + firstScore + " last=" + lastScore);
    }

    // -----------------------------------------------------------------------
    // GraphSAGE MEAN — forward shape
    // -----------------------------------------------------------------------

    /**
     * Build a [GraphSageLayer(MEAN, nIn=2, nOut=2) → OutputLayer] network and verify
     * output shape and finiteness.
     */
    @Test
    public void testGraphSageMeanForwardShape() {
        SparseNDArray adj = makeAdj();

        GraphSageLayer sageConf = new GraphSageLayer.Builder()
                .nIn(F).nOut(NOUT)
                .aggregation(GraphSageLayer.Aggregation.MEAN)
                .hasBias(true)
                .build();
        sageConf.setAdjacency(adj);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .updater(new Adam(1e-3))
                .list()
                .layer(sageConf)
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .activation(Activation.IDENTITY)
                        .nIn(NOUT).nOut(NOUT)
                        .build())
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        INDArray out = net.output(makeFeatures());

        assertArrayEquals(new long[]{N, NOUT}, out.shape(),
                "GraphSAGE-mean output shape must be [numNodes, nOut]");
        assertFalse(out.isNaN().any(),      "GraphSAGE-mean output contains NaN");
        assertFalse(out.isInfinite().any(), "GraphSAGE-mean output contains Inf");
    }

    // -----------------------------------------------------------------------
    // GraphSAGE MAX — forward shape
    // -----------------------------------------------------------------------

    /**
     * Build a [GraphSageLayer(MAX, nIn=2, nOut=2) → OutputLayer] network and verify
     * output shape and finiteness.
     */
    @Test
    public void testGraphSageMaxForwardShape() {
        SparseNDArray adj = makeAdj();

        GraphSageLayer sageConf = new GraphSageLayer.Builder()
                .nIn(F).nOut(NOUT)
                .aggregation(GraphSageLayer.Aggregation.MAX)
                .hasBias(true)
                .build();
        sageConf.setAdjacency(adj);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .updater(new Adam(1e-3))
                .list()
                .layer(sageConf)
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .activation(Activation.IDENTITY)
                        .nIn(NOUT).nOut(NOUT)
                        .build())
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        INDArray out = net.output(makeFeatures());

        assertArrayEquals(new long[]{N, NOUT}, out.shape(),
                "GraphSAGE-max output shape must be [numNodes, nOut]");
        assertFalse(out.isNaN().any(),      "GraphSAGE-max output contains NaN");
        assertFalse(out.isInfinite().any(), "GraphSAGE-max output contains Inf");
    }

    // -----------------------------------------------------------------------
    // GraphSAGE POOL — forward shape
    // -----------------------------------------------------------------------

    /**
     * Build a [GraphSageLayer(POOL, nIn=2, poolSize=3, nOut=2) → OutputLayer] network
     * and verify output shape and finiteness.
     */
    @Test
    public void testGraphSagePoolForwardShape() {
        SparseNDArray adj = makeAdj();

        GraphSageLayer sageConf = new GraphSageLayer.Builder()
                .nIn(F).nOut(NOUT)
                .aggregation(GraphSageLayer.Aggregation.POOL)
                .poolSize(3)
                .hasBias(true)
                .build();
        sageConf.setAdjacency(adj);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .updater(new Adam(1e-3))
                .list()
                .layer(sageConf)
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .activation(Activation.IDENTITY)
                        .nIn(NOUT).nOut(NOUT)
                        .build())
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        INDArray out = net.output(makeFeatures());

        assertArrayEquals(new long[]{N, NOUT}, out.shape(),
                "GraphSAGE-pool output shape must be [numNodes, nOut]");
        assertFalse(out.isNaN().any(),      "GraphSAGE-pool output contains NaN");
        assertFalse(out.isInfinite().any(), "GraphSAGE-pool output contains Inf");
    }

    // -----------------------------------------------------------------------
    // GAT (single head) — forward shape
    // -----------------------------------------------------------------------

    /**
     * Build a [GatLayer(numHeads=1, headDim=2) → OutputLayer] network and verify
     * output shape and finiteness.
     */
    @Test
    public void testGatSingleHeadForwardShape() {
        SparseNDArray adj = makeAdj();

        GatLayer gatConf = new GatLayer.Builder()
                .nIn(F).headDim(NOUT).numHeads(1)
                .leakyAlpha(0.2)
                .build();
        gatConf.setAdjacency(adj);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .updater(new Adam(1e-3))
                .list()
                .layer(gatConf)
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .activation(Activation.IDENTITY)
                        .nIn(NOUT).nOut(NOUT)
                        .build())
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        INDArray out = net.output(makeFeatures());

        assertArrayEquals(new long[]{N, NOUT}, out.shape(),
                "GAT (1 head) output shape must be [numNodes, numHeads*headDim]=[N, NOUT]");
        assertFalse(out.isNaN().any(),      "GAT output contains NaN");
        assertFalse(out.isInfinite().any(), "GAT output contains Inf");
    }

    // -----------------------------------------------------------------------
    // GAT (multi-head) — forward shape
    // -----------------------------------------------------------------------

    /**
     * Build a [GatLayer(numHeads=2, headDim=2) → OutputLayer] network.
     * Output of GAT = [N, numHeads * headDim] = [4, 4], feeding into OutputLayer(nIn=4, nOut=2).
     */
    @Test
    public void testGatMultiHeadForwardShape() {
        SparseNDArray adj = makeAdj();

        final int numHeads = 2;
        final int headDim  = 2;  // total output = numHeads * headDim = 4

        GatLayer gatConf = new GatLayer.Builder()
                .nIn(F).headDim(headDim).numHeads(numHeads)
                .leakyAlpha(0.2)
                .build();
        gatConf.setAdjacency(adj);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .updater(new Adam(1e-3))
                .list()
                .layer(gatConf)
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .activation(Activation.IDENTITY)
                        .nIn((long) numHeads * headDim).nOut(NOUT)
                        .build())
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        INDArray out = net.output(makeFeatures());

        assertArrayEquals(new long[]{N, NOUT}, out.shape(),
                "GAT (2 heads) + OutputLayer output shape must be [numNodes, nLabels]");
        assertFalse(out.isNaN().any(),      "GAT multi-head output contains NaN");
        assertFalse(out.isInfinite().any(), "GAT multi-head output contains Inf");
    }

    // -----------------------------------------------------------------------
    // Stacked GNN layers — GCN → GCN → OutputLayer
    // -----------------------------------------------------------------------

    /**
     * Two stacked GCN layers in a single MultiLayerNetwork.  Both layers share the same
     * adjacency (the graph topology is fixed across the network).
     *
     * <p>Verifies that stacking works correctly: each GcnLayer gets its own SameDiff
     * graph and its own copy of the adjacency constants.
     */
    @Test
    public void testStackedGcnLayersForwardShape() {
        SparseNDArray adj = makeAdj();

        GcnLayer gcn1 = new GcnLayer.Builder().nIn(F).nOut(4).applyRelu(false).hasBias(true).build();
        GcnLayer gcn2 = new GcnLayer.Builder().nIn(4).nOut(NOUT).applyRelu(false).hasBias(true).build();
        gcn1.setAdjacency(adj);
        gcn2.setAdjacency(adj);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .updater(new Adam(1e-3))
                .list()
                .layer(gcn1)
                .layer(gcn2)
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .activation(Activation.IDENTITY)
                        .nIn(NOUT).nOut(NOUT)
                        .build())
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        INDArray out = net.output(makeFeatures());

        assertArrayEquals(new long[]{N, NOUT}, out.shape(),
                "Stacked GCN output shape must be [numNodes, nOut]");
        assertFalse(out.isNaN().any(),      "Stacked GCN output contains NaN");
        assertFalse(out.isInfinite().any(), "Stacked GCN output contains Inf");
    }

    // -----------------------------------------------------------------------
    // Adjacency-not-set guard
    // -----------------------------------------------------------------------

    /**
     * Verifies that attempting to initialize a GcnLayer without calling
     * {@link GcnLayer#setAdjacency} throws an {@link IllegalStateException}.
     */
    @Test
    public void testGcnLayerThrowsIfAdjacencyNotSet() {
        GcnLayer gcnConf = new GcnLayer.Builder().nIn(F).nOut(NOUT).build();
        // Deliberately do NOT call gcnConf.setAdjacency(...)

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .list()
                .layer(gcnConf)
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .activation(Activation.IDENTITY).nIn(NOUT).nOut(NOUT).build())
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();  // init succeeds; the adjacency guard fires lazily in defineLayer on the first forward pass
        assertThrows(IllegalStateException.class, () -> net.output(makeFeatures()),
                "GcnLayer should throw IllegalStateException when adjacency is not set and a forward pass is attempted");
    }
}
