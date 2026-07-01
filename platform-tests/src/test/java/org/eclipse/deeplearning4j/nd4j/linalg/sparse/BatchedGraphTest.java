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

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.TrainingConfig;
import org.nd4j.autodiff.validation.GradCheckUtil;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.sparse.GraphDisjointUnion;
import org.nd4j.linalg.api.ops.impl.transforms.custom.segment.SegmentSoftmax;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.api.ops.impl.shape.Gather;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Phase D acceptance tests for variable-size graph batching.
 *
 * <h3>Tests</h3>
 * <ol>
 *   <li>{@link #testGraphDisjointUnionCorrectness} — correctness invariant:
 *       batched forward == per-graph forward concatenation for GCN conv</li>
 *   <li>{@link #testGraphDisjointUnionBatchedSegmentPool} — segment pooling produces
 *       same result as per-graph globalMeanPool</li>
 *   <li>{@link #testSegmentSoftmaxForwardCorrectness} — segment_softmax produces
 *       per-segment softmax identical to K separate softmax calls</li>
 *   <li>{@link #testGraphDisjointUnionGradcheck} — gradcheck on graph_disjoint_union
 *       (X and vals differentiable; structural inputs excluded)</li>
 *   <li>{@link #testSegmentSoftmaxGradcheck} — gradcheck on segment_softmax</li>
 *   <li>{@link #testBatchedGcnTrainingConverges} — end-to-end: GCN + segmentMeanPool
 *       + dense → softmax classifier trains on variable-size synthetic graphs;
 *       loss must decrease over 50 steps</li>
 * </ol>
 */
@Tag(TagNames.SAMEDIFF)
public class BatchedGraphTest extends BaseNd4jTestWithBackends {

    // ─────────────────────────────────────────────────────────────────────────
    // Shared test graphs (fixed seed for reproducibility)
    // ─────────────────────────────────────────────────────────────────────────

    /** A small triangle + self-loops, 3 nodes, 6 edges. */
    private static final int N0 = 3, NNZ0 = 6, F = 4;
    /** A path graph 4 nodes 6 directed edges + self-loops. */
    private static final int N1 = 4, NNZ1 = 7;
    /** A single node 2 self-loops. */
    private static final int N2 = 2, NNZ2 = 2;

    // Graph 0: 3 nodes triangle with self-loops, weights = 1.0
    private static float[] g0Vals()    { return new float[]{1,1,1,1,1,1}; }
    private static int[]   g0ColIdx()  { return new int[]{0,1,2,0,1,2}; }
    private static int[]   g0RowPtr()  { return new int[]{0,2,4,6}; }

    // Graph 1: 4-node path 0->0,0->1, 1->1,1->2, 2->2,2->3, 3->3
    private static float[] g1Vals()    { return new float[]{1,1, 1,1, 1,1, 1}; }
    private static int[]   g1ColIdx()  { return new int[]{0,1, 1,2, 2,3, 3}; }
    private static int[]   g1RowPtr()  { return new int[]{0,2, 4, 6, 7}; }

    // Graph 2: 2 nodes, self-loops
    private static float[] g2Vals()    { return new float[]{1,1}; }
    private static int[]   g2ColIdx()  { return new int[]{0,1}; }
    private static int[]   g2RowPtr()  { return new int[]{0,1,2}; }

    /** Feature matrix for graph k (deterministic random). */
    private static INDArray makeX(int N, long seed) {
        Nd4j.getRandom().setSeed(seed);
        return Nd4j.rand(DataType.FLOAT, N, F);
    }

    @Override
    public char ordering() { return 'c'; }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 1: Correctness invariant — batched GCN == per-graph GCN concat
    // ─────────────────────────────────────────────────────────────────────────

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGraphDisjointUnionCorrectness(Nd4jBackend backend) throws Exception {
        // Build per-graph feature matrices
        INDArray X0 = makeX(N0, 42L);
        INDArray X1 = makeX(N1, 43L);
        INDArray X2 = makeX(N2, 44L);

        // ── per-graph GCN conv (sd.gnn().gcnConv) ──────────────────────────
        INDArray[] perGraphOut = new INDArray[3];
        INDArray[] Xs = {X0, X1, X2};
        float[][] valsArr = {g0Vals(), g1Vals(), g2Vals()};
        int[][]   colIdxArr = {g0ColIdx(), g1ColIdx(), g2ColIdx()};
        int[][]   rowPtrArr = {g0RowPtr(), g1RowPtr(), g2RowPtr()};
        int[]     Ns  = {N0, N1, N2};
        int[]     NNZs = {NNZ0, NNZ1, NNZ2};

        // Shared weight matrix
        Nd4j.getRandom().setSeed(99L);
        INDArray W = Nd4j.rand(DataType.FLOAT, F, F);
        INDArray bias = Nd4j.zeros(DataType.FLOAT, F);

        for (int k = 0; k < 3; k++) {
            SameDiff sd = SameDiff.create();
            SDVariable xv   = sd.var("X",   Xs[k]);
            SDVariable wv   = sd.var("W",   W);
            SDVariable bv   = sd.var("bias", bias);
            SDVariable avv  = sd.var("av",  Nd4j.createFromArray(valsArr[k]));
            SDVariable civ  = sd.var("ci",  Nd4j.createFromArray(colIdxArr[k]));
            SDVariable rpv  = sd.var("rp",  Nd4j.createFromArray(rowPtrArr[k]));
            SDVariable out  = sd.gnn().gcnConv(xv, wv, bv, avv, civ, rpv, Ns[k], Ns[k], false);
            Map<String,INDArray> ph = new HashMap<>();
            perGraphOut[k] = sd.output(ph, out.name()).get(out.name());
        }

        // ── batched: build disjoint union, then gcnConv ───────────────────
        SameDiff sdBatch = SameDiff.create();
        INDArray[] allVals    = {Nd4j.createFromArray(g0Vals()), Nd4j.createFromArray(g1Vals()), Nd4j.createFromArray(g2Vals())};
        INDArray[] allColIdx  = {Nd4j.createFromArray(g0ColIdx()), Nd4j.createFromArray(g1ColIdx()), Nd4j.createFromArray(g2ColIdx())};
        INDArray[] allRowPtr  = {Nd4j.createFromArray(g0RowPtr()), Nd4j.createFromArray(g1RowPtr()), Nd4j.createFromArray(g2RowPtr())};

        SDVariable[] sdXs      = {sdBatch.var("X0", X0), sdBatch.var("X1", X1), sdBatch.var("X2", X2)};
        SDVariable[] sdVals    = {sdBatch.var("v0", allVals[0]), sdBatch.var("v1", allVals[1]), sdBatch.var("v2", allVals[2])};
        SDVariable[] sdColIdxs = {sdBatch.var("ci0", allColIdx[0]), sdBatch.var("ci1", allColIdx[1]), sdBatch.var("ci2", allColIdx[2])};
        SDVariable[] sdRowPtrs = {sdBatch.var("rp0", allRowPtr[0]), sdBatch.var("rp1", allRowPtr[1]), sdBatch.var("rp2", allRowPtr[2])};

        GraphDisjointUnion gdu = new GraphDisjointUnion(sdBatch, sdXs, sdVals, sdColIdxs, sdRowPtrs);
        SDVariable[] outs = gdu.outputVariables();
        SDVariable Xc    = outs[0]; // [sumN, F]
        SDVariable valsc = outs[1]; // [sumNnz]
        SDVariable cic   = outs[2]; // [sumNnz]
        SDVariable rpc   = outs[3]; // [sumN+1]
        // outs[4] = batchVec

        int sumN = N0 + N1 + N2;
        SDVariable Wb   = sdBatch.var("W", W);
        SDVariable bv   = sdBatch.var("bias", bias);
        SDVariable batchOut = sdBatch.gnn().gcnConv(Xc, Wb, bv, valsc, cic, rpc, sumN, sumN, false);

        Map<String,INDArray> batchResult = sdBatch.output(new HashMap<>(), batchOut.name());
        INDArray batchedOut = batchResult.get(batchOut.name()); // [sumN, F]

        // ── correctness invariant ─────────────────────────────────────────
        // Concatenate per-graph outputs along axis 0
        INDArray expected = Nd4j.concat(0, perGraphOut[0], perGraphOut[1], perGraphOut[2]);

        assertEquals(expected.shape()[0], batchedOut.shape()[0], "batched node count must match sumN");
        assertEquals(expected.shape()[1], batchedOut.shape()[1], "feature dim must match");

        // Allow tolerance of 1e-5 for floating point
        double maxDiff = expected.sub(batchedOut).amaxNumber().doubleValue();
        assertTrue(maxDiff < 1e-4,
            "Batched GCN output must match per-graph concat to 1e-4, got max diff=" + maxDiff);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 2: Segment pooling correctness
    // ─────────────────────────────────────────────────────────────────────────

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGraphDisjointUnionBatchedSegmentPool(Nd4jBackend backend) {
        INDArray X0 = makeX(N0, 10L);
        INDArray X1 = makeX(N1, 11L);
        INDArray X2 = makeX(N2, 12L);

        // Per-graph mean (expected)
        INDArray mean0 = X0.mean(0); // [F]
        INDArray mean1 = X1.mean(0);
        INDArray mean2 = X2.mean(0);
        INDArray expectedGraphMeans = Nd4j.vstack(mean0, mean1, mean2); // [3, F]

        // Batched: concatenate X then segment mean by batchVec
        int sumN = N0 + N1 + N2;
        INDArray Xcombined = Nd4j.vstack(X0, X1, X2); // [sumN, F]

        // batchVec: [0,0,0, 1,1,1,1, 2,2]
        int[] bvData = new int[sumN];
        int idx = 0;
        for (int i = 0; i < N0; i++) bvData[idx++] = 0;
        for (int i = 0; i < N1; i++) bvData[idx++] = 1;
        for (int i = 0; i < N2; i++) bvData[idx++] = 2;
        INDArray batchVec = Nd4j.createFromArray(bvData);

        SameDiff sd = SameDiff.create();
        SDVariable xv  = sd.var("X", Xcombined);
        SDVariable bv  = sd.var("bv", batchVec);
        SDVariable graphMeans = sd.segmentMean(xv, bv); // [3, F]

        INDArray result = sd.output(new HashMap<>(), graphMeans.name()).get(graphMeans.name());

        assertEquals(3, result.shape()[0], "should have 3 graph embeddings");
        assertEquals(F, result.shape()[1], "feature dim should match");

        double maxDiff = expectedGraphMeans.sub(result).amaxNumber().doubleValue();
        assertTrue(maxDiff < 1e-4,
            "Segment mean pool must match per-graph mean, got max diff=" + maxDiff);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 3: segment_softmax forward correctness
    // ─────────────────────────────────────────────────────────────────────────

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSegmentSoftmaxForwardCorrectness(Nd4jBackend backend) {
        // 3 segments: [0,0,0], [1,1], [2,2,2,2]
        // logits: 9 elements
        float[] logitsData = {1.0f, 2.0f, 3.0f,   // seg 0
                               0.5f, 1.5f,           // seg 1
                               -1.0f, 0.0f, 1.0f, 2.0f}; // seg 2
        int[]   segIds     = {0,0,0, 1,1, 2,2,2,2};

        INDArray logits = Nd4j.createFromArray(logitsData);
        INDArray segIdsArr = Nd4j.createFromArray(segIds);

        // Expected: softmax independently within each segment
        float[] s0 = softmax(new float[]{1.0f, 2.0f, 3.0f});
        float[] s1 = softmax(new float[]{0.5f, 1.5f});
        float[] s2 = softmax(new float[]{-1.0f, 0.0f, 1.0f, 2.0f});
        float[] expected = concat(s0, s1, s2);

        // Eager execution
        INDArray result = Nd4j.exec(new SegmentSoftmax(logits, segIdsArr, 3L))[0];

        assertEquals(9, result.length(), "output length must match input");
        for (int i = 0; i < 9; i++) {
            assertEquals(expected[i], result.getFloat(i), 1e-5f,
                "segment_softmax mismatch at index " + i);
        }

        // Verify each segment sums to 1.0
        double sum0 = result.get(Nd4j.createFromArray(new int[]{0,1,2})).sumNumber().doubleValue();
        double sum1 = result.get(Nd4j.createFromArray(new int[]{3,4})).sumNumber().doubleValue();
        double sum2 = result.get(Nd4j.createFromArray(new int[]{5,6,7,8})).sumNumber().doubleValue();
        assertEquals(1.0, sum0, 1e-5, "segment 0 must sum to 1");
        assertEquals(1.0, sum1, 1e-5, "segment 1 must sum to 1");
        assertEquals(1.0, sum2, 1e-5, "segment 2 must sum to 1");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 4: Gradcheck for graph_disjoint_union
    // ─────────────────────────────────────────────────────────────────────────

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGraphDisjointUnionGradcheck(Nd4jBackend backend) throws Exception {
        // Use tiny graphs for gradcheck speed
        int n0 = 2, nnz0 = 3, n1 = 3, nnz1 = 4, f = 2;

        Nd4j.getRandom().setSeed(1L);
        INDArray X0v    = Nd4j.rand(DataType.DOUBLE, n0, f);
        INDArray X1v    = Nd4j.rand(DataType.DOUBLE, n1, f);
        INDArray vals0  = Nd4j.rand(DataType.DOUBLE, nnz0);
        INDArray vals1  = Nd4j.rand(DataType.DOUBLE, nnz1);
        // Simple star topology: node 0 connected to nodes 1,2 (and self)
        INDArray ci0    = Nd4j.createFromArray(new int[]{0,0,1});
        INDArray rp0    = Nd4j.createFromArray(new int[]{0,2,3});
        INDArray ci1    = Nd4j.createFromArray(new int[]{0,1,2,0});
        INDArray rp1    = Nd4j.createFromArray(new int[]{0,2,3,4});

        SameDiff sd = SameDiff.create();
        SDVariable sdX0  = sd.var("X0",   X0v);
        SDVariable sdX1  = sd.var("X1",   X1v);
        SDVariable sdV0  = sd.var("v0",   vals0);
        SDVariable sdV1  = sd.var("v1",   vals1);
        SDVariable sdCi0 = sd.constant("ci0", ci0);
        SDVariable sdRp0 = sd.constant("rp0", rp0);
        SDVariable sdCi1 = sd.constant("ci1", ci1);
        SDVariable sdRp1 = sd.constant("rp1", rp1);

        GraphDisjointUnion gdu = new GraphDisjointUnion(sd,
            new SDVariable[]{sdX0, sdX1},
            new SDVariable[]{sdV0, sdV1},
            new SDVariable[]{sdCi0, sdCi1},
            new SDVariable[]{sdRp0, sdRp1});

        SDVariable[] gduOuts = gdu.outputVariables();
        // Loss = sum of X_combined (tests gradient through X concat)
        SDVariable loss = gduOuts[0].sum();
        sd.setLossVariables(loss.name());

        boolean ok = GradCheckUtil.checkGradients(sd, new HashMap<>(),
            1e-5, 1e-4, 1e-4, true, true);  // pre-existing: was checking specific vars X0/X1
        assertTrue(ok, "graph_disjoint_union X gradcheck must pass");

        // Also gradcheck vals
        SameDiff sd2 = SameDiff.create();
        SDVariable sdX0b  = sd2.constant("X0",  X0v);
        SDVariable sdX1b  = sd2.constant("X1",  X1v);
        SDVariable sdV0b  = sd2.var("v0",  vals0);
        SDVariable sdV1b  = sd2.var("v1",  vals1);
        SDVariable sdCi0b = sd2.constant("ci0", ci0);
        SDVariable sdRp0b = sd2.constant("rp0", rp0);
        SDVariable sdCi1b = sd2.constant("ci1", ci1);
        SDVariable sdRp1b = sd2.constant("rp1", rp1);

        GraphDisjointUnion gdu2 = new GraphDisjointUnion(sd2,
            new SDVariable[]{sdX0b, sdX1b},
            new SDVariable[]{sdV0b, sdV1b},
            new SDVariable[]{sdCi0b, sdCi1b},
            new SDVariable[]{sdRp0b, sdRp1b});

        SDVariable[] outs2 = gdu2.outputVariables();
        SDVariable loss2 = outs2[1].sum(); // sum of vals_combined
        sd2.setLossVariables(loss2.name());

        boolean okVals = GradCheckUtil.checkGradients(sd2, new HashMap<>(),
            1e-5, 1e-4, 1e-4, true, true);
        assertTrue(okVals, "graph_disjoint_union vals gradcheck must pass");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 5: Gradcheck for segment_softmax
    // ─────────────────────────────────────────────────────────────────────────

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSegmentSoftmaxGradcheck(Nd4jBackend backend) throws Exception {
        // 3 segments
        Nd4j.getRandom().setSeed(5L);
        INDArray logitsArr = Nd4j.rand(DataType.DOUBLE, 7); // [7]
        int[] segIdsData = {0,0,0, 1,1, 2,2};
        INDArray segIdsArr = Nd4j.createFromArray(segIdsData);

        SameDiff sd = SameDiff.create();
        SDVariable logits  = sd.var("logits", logitsArr);
        SDVariable segIds  = sd.constant("segIds", segIdsArr);
        SDVariable ssOut   = new SegmentSoftmax(sd, logits, segIds, 3L).outputVariable();
        SDVariable loss    = ssOut.sum();
        sd.setLossVariables(loss.name());

        boolean ok = GradCheckUtil.checkGradients(sd, new HashMap<>(),
            1e-5, 1e-4, 1e-4, true, true);
        assertTrue(ok, "segment_softmax gradcheck must pass");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 6: End-to-end GCN graph classifier training on variable-size graphs
    // ─────────────────────────────────────────────────────────────────────────

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testBatchedGcnTrainingConverges(Nd4jBackend backend) throws Exception {
        // Synthetic binary graph classification:
        //   class 0 = dense graphs (triangle-like), class 1 = sparse paths
        // 3 graphs per mini-batch (variable size)
        final int numClasses = 2;
        final int hiddenF = 8;
        final int steps = 50;

        // ── build a fixed mini-batch of 3 graphs ──────────────────────────
        INDArray X0 = makeX(N0, 101L);
        INDArray X1 = makeX(N1, 102L);
        INDArray X2 = makeX(N2, 103L);
        int sumN = N0 + N1 + N2;

        INDArray av0 = Nd4j.createFromArray(g0Vals());
        INDArray ci0 = Nd4j.createFromArray(g0ColIdx());
        INDArray rp0 = Nd4j.createFromArray(g0RowPtr());
        INDArray av1 = Nd4j.createFromArray(g1Vals());
        INDArray ci1 = Nd4j.createFromArray(g1ColIdx());
        INDArray rp1 = Nd4j.createFromArray(g1RowPtr());
        INDArray av2 = Nd4j.createFromArray(g2Vals());
        INDArray ci2 = Nd4j.createFromArray(g2ColIdx());
        INDArray rp2 = Nd4j.createFromArray(g2RowPtr());

        // Labels [3, numClasses] one-hot
        INDArray labels = Nd4j.zeros(DataType.FLOAT, 3, numClasses);
        labels.putScalar(new int[]{0, 0}, 1.0f); // graph 0: class 0
        labels.putScalar(new int[]{1, 1}, 1.0f); // graph 1: class 1
        labels.putScalar(new int[]{2, 0}, 1.0f); // graph 2: class 0

        // ── build SameDiff model ──────────────────────────────────────────
        SameDiff sd = SameDiff.create();

        // Trainable parameters
        Nd4j.getRandom().setSeed(200L);
        SDVariable W1   = sd.var("W1", Nd4j.rand(DataType.FLOAT, F, hiddenF).mul(0.1f));
        SDVariable b1   = sd.var("b1", Nd4j.zeros(DataType.FLOAT, hiddenF));
        SDVariable W2   = sd.var("W2", Nd4j.rand(DataType.FLOAT, hiddenF, numClasses).mul(0.1f));
        SDVariable b2   = sd.var("b2", Nd4j.zeros(DataType.FLOAT, numClasses));

        // Placeholders for batched graph
        SDVariable sdX0   = sd.placeHolder("X0",   DataType.FLOAT, N0, F);
        SDVariable sdX1   = sd.placeHolder("X1",   DataType.FLOAT, N1, F);
        SDVariable sdX2   = sd.placeHolder("X2",   DataType.FLOAT, N2, F);
        SDVariable sdAv0  = sd.constant("av0",  av0);
        SDVariable sdCi0  = sd.constant("ci0",  ci0);
        SDVariable sdRp0  = sd.constant("rp0",  rp0);
        SDVariable sdAv1  = sd.constant("av1",  av1);
        SDVariable sdCi1  = sd.constant("ci1",  ci1);
        SDVariable sdRp1  = sd.constant("rp1",  rp1);
        SDVariable sdAv2  = sd.constant("av2",  av2);
        SDVariable sdCi2  = sd.constant("ci2",  ci2);
        SDVariable sdRp2  = sd.constant("rp2",  rp2);
        SDVariable sdLabels = sd.placeHolder("labels", DataType.FLOAT, 3, numClasses);

        // Build block-diagonal batch
        GraphDisjointUnion gdu = new GraphDisjointUnion(sd,
            new SDVariable[]{sdX0, sdX1, sdX2},
            new SDVariable[]{sdAv0, sdAv1, sdAv2},
            new SDVariable[]{sdCi0, sdCi1, sdCi2},
            new SDVariable[]{sdRp0, sdRp1, sdRp2});
        SDVariable[] gduOuts = gdu.outputVariables();
        SDVariable Xc    = gduOuts[0]; // [sumN, F]
        SDVariable valsc = gduOuts[1]; // [sumNnz]
        SDVariable cic   = gduOuts[2]; // [sumNnz]
        SDVariable rpc   = gduOuts[3]; // [sumN+1]
        SDVariable bv    = gduOuts[4]; // [sumN] INT32 batch vector

        // GCN layer 1: [sumN, F] → [sumN, hiddenF]
        SDVariable h = sd.gnn().gcnConv(Xc, W1, b1, valsc, cic, rpc, sumN, sumN, true);

        // Segment mean pooling: [sumN, hiddenF] → [3, hiddenF]
        SDVariable graphEmb = sd.segmentMean(h, bv);

        // Classifier: [3, hiddenF] → [3, numClasses]
        SDVariable logits = sd.nn().softmax(sd.mmul(graphEmb, W2).add(b2), -1);

        // Cross-entropy loss
        SDVariable loss = sd.math().neg(sdLabels.mul(sd.math().log(logits.add(1e-8f))).sum(true, 1).mean());
        sd.setLossVariables(loss.name());

        TrainingConfig cfg = TrainingConfig.builder()
            .updater(new Adam(0.01))
            .dataSetFeatureMapping("X0", "X1", "X2")
            .dataSetLabelMapping("labels")
            .build();
        sd.setTrainingConfig(cfg);

        // Manually run gradient descent steps
        Map<String, INDArray> ph = new HashMap<>();
        ph.put("X0", X0);
        ph.put("X1", X1);
        ph.put("X2", X2);
        ph.put("labels", labels);

        // Compute initial loss
        Map<String, INDArray> fwdResult = sd.output(ph, loss.name());
        double initialLoss = fwdResult.get(loss.name()).getDouble(0);

        // Run gradient-descent steps on the fixed minibatch via fit(Map<String,INDArray>)
        for (int step = 0; step < steps; step++) {
            sd.fit(ph);
        }

        // Compute final loss
        fwdResult = sd.output(ph, loss.name());
        double finalLoss = fwdResult.get(loss.name()).getDouble(0);

        assertTrue(finalLoss < initialLoss,
            "Loss must decrease over " + steps + " gradient steps. Initial=" + initialLoss + " Final=" + finalLoss);
        assertTrue(finalLoss < initialLoss * 0.9,
            "Loss must decrease by at least 10%. Initial=" + initialLoss + " Final=" + finalLoss);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Utilities
    // ─────────────────────────────────────────────────────────────────────────

    private static float[] softmax(float[] x) {
        float max = x[0];
        for (float v : x) if (v > max) max = v;
        float sum = 0;
        float[] e = new float[x.length];
        for (int i = 0; i < x.length; i++) { e[i] = (float) Math.exp(x[i] - max); sum += e[i]; }
        for (int i = 0; i < x.length; i++) e[i] /= sum;
        return e;
    }

    private static float[] concat(float[]... arrs) {
        int total = 0;
        for (float[] a : arrs) total += a.length;
        float[] out = new float[total];
        int idx = 0;
        for (float[] a : arrs) { System.arraycopy(a, 0, out, idx, a.length); idx += a.length; }
        return out;
    }
}
