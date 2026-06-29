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
import org.nd4j.autodiff.validation.GradCheckUtil;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.util.function.BiFunction;

import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Gradient checks for the codegen-generated {@code sd.graph()} graph-construction / graph-learning
 * compositions, plus equivalence checks that the generated eager {@code Nd4j.graph()} forms match
 * their SameDiff counterparts (validating the composition codegen path that emits both).
 */
@Tag(TagNames.SAMEDIFF)
public class SDGraphTest extends BaseNd4jTestWithBackends {

    private static final int N = 4, D = 3;

    /** Scalar loss for gradchecking (the methods are validated for non-uniform upstream in isolation). */
    private static void scalarLoss(SameDiff sd, SDVariable out) {
        sd.sum("loss", out);
        // Explicitly mark the loss so inference does not fail when there are unconsumed intermediate
        // outputs (e.g. topK returns [values, indices] but sortPool only uses indices; the unused
        // values output is a terminal node that confuses the single-output inference heuristic).
        sd.setLossVariables("loss");
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testCosineSimilarityGradCheck(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(1L);
        SameDiff sd = SameDiff.create();
        try {
            SDVariable x = sd.var("x", Nd4j.randn(DataType.DOUBLE, N, D));
            scalarLoss(sd, sd.graph().cosineSimilarity(x));
            assertTrue(GradCheckUtil.checkGradients(sd, null), "cosineSimilarity grad check failed");
        } finally {
            sd.close();
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testCorrelationMatrixGradCheck(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(2L);
        SameDiff sd = SameDiff.create();
        try {
            SDVariable data = sd.var("data", Nd4j.randn(DataType.DOUBLE, 5, D));
            scalarLoss(sd, sd.graph().correlationMatrix(data));
            assertTrue(GradCheckUtil.checkGradients(sd, null), "correlationMatrix grad check failed");
        } finally {
            sd.close();
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGaussianSimilarityGradCheck(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(3L);
        SameDiff sd = SameDiff.create();
        try {
            SDVariable x = sd.var("x", Nd4j.randn(DataType.DOUBLE, N, D));
            scalarLoss(sd, sd.graph().gaussianSimilarity(x, 1.0));
            assertTrue(GradCheckUtil.checkGradients(sd, null), "gaussianSimilarity grad check failed");
        } finally {
            sd.close();
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testKnnGraphGradCheck(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(24L);
        SameDiff sd = SameDiff.create();
        try {
            // well-spread similarities so the per-row top-k selection is stable under the 1e-5 perturbation
            SDVariable sim = sd.var("sim", Nd4j.rand(DataType.DOUBLE, 5, 5).mul(10.0).add(0.5));
            scalarLoss(sd, sd.graph().knnGraph(sim, 2, 5));   // keep top-2 neighbors of 5 nodes
            assertTrue(GradCheckUtil.checkGradients(sd, null), "knnGraph grad check failed");
        } finally {
            sd.close();
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testDgiLossGradCheck(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(4L);
        SameDiff sd = SameDiff.create();
        try {
            SDVariable h    = sd.var("h", Nd4j.randn(DataType.DOUBLE, N, D).muli(0.3));
            SDVariable hneg = sd.var("hneg", Nd4j.randn(DataType.DOUBLE, N, D).muli(0.3));
            SDVariable w    = sd.var("w", Nd4j.randn(DataType.DOUBLE, D, D).muli(0.3));
            sd.sum("loss", sd.graph().dgiLoss(h, hneg, w));   // dgiLoss is already scalar; sum keeps it a named scalar loss
            assertTrue(GradCheckUtil.checkGradients(sd, null), "dgiLoss grad check failed");
        } finally {
            sd.close();
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testLabelPropagationGradCheck(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(5L);
        SameDiff sd = SameDiff.create();
        try {
            // small symmetric row-normalized adjacency on 3 nodes -> CSR via the sparse namespace
            INDArray adj = Nd4j.create(new double[][]{
                    {0.5, 0.5, 0.0},
                    {0.25, 0.5, 0.25},
                    {0.0, 0.5, 0.5}}).castTo(DataType.DOUBLE);
            SDVariable a = sd.constant(adj);
            SDVariable[] csr = sd.sparse().denseToCsr(a, 0.0);   // values, colIdx, rowPtr
            SDVariable seedY = sd.var("seedY", Nd4j.randn(DataType.DOUBLE, 3, 2));
            SDVariable labels = sd.graph().labelPropagation(seedY, csr[0], csr[1], csr[2], 3, 3, 3, 0.1);
            scalarLoss(sd, labels);
            assertTrue(GradCheckUtil.checkGradients(sd, null), "labelPropagation grad check failed");
        } finally {
            sd.close();
        }
    }

    // -------------------------------------------------------------------------
    // Link-prediction heuristics (topology edge scores over an adjacency matrix)
    // -------------------------------------------------------------------------

    /** Gradcheck a heuristic over a positive adjacency (deg > 1 so Adamic-Adar's log(deg) > 0). */
    private static void heuristicGradCheck(long seed, BiFunction<SameDiff, SDVariable, SDVariable> op) {
        Nd4j.getRandom().setSeed(seed);
        SameDiff sd = SameDiff.create();
        try {
            SDVariable adj = sd.var("adj", Nd4j.rand(DataType.DOUBLE, N, N).add(0.2)); // entries in [0.2,1.2]
            scalarLoss(sd, op.apply(sd, adj));
            assertTrue(GradCheckUtil.checkGradients(sd, null), "heuristic grad check failed");
        } finally {
            sd.close();
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testCommonNeighborsGradCheck(Nd4jBackend backend) {
        heuristicGradCheck(10L, (sd, adj) -> sd.graph().commonNeighbors(adj));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testAdamicAdarGradCheck(Nd4jBackend backend) {
        heuristicGradCheck(11L, (sd, adj) -> sd.graph().adamicAdar(adj));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testResourceAllocationGradCheck(Nd4jBackend backend) {
        heuristicGradCheck(12L, (sd, adj) -> sd.graph().resourceAllocation(adj));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPreferentialAttachmentGradCheck(Nd4jBackend backend) {
        heuristicGradCheck(13L, (sd, adj) -> sd.graph().preferentialAttachment(adj));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testJaccardTopologyGradCheck(Nd4jBackend backend) {
        heuristicGradCheck(14L, (sd, adj) -> sd.graph().jaccardTopology(adj));
    }

    // -------------------------------------------------------------------------
    // Diffusion / propagation ops
    // -------------------------------------------------------------------------

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testCorrectAndSmoothGradCheck(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(20L);
        SameDiff sd = SameDiff.create();
        try {
            INDArray aNormArr = Nd4j.rand(DataType.DOUBLE, 4, 4).add(0.1);
            SDVariable basePreds = sd.var("basePreds", Nd4j.randn(DataType.DOUBLE, 4, 2).muli(0.1));
            SDVariable aNorm     = sd.var("aNorm",     aNormArr);
            SDVariable residuals = sd.var("residuals", Nd4j.randn(DataType.DOUBLE, 4, 2).muli(0.1));
            scalarLoss(sd, sd.graph().correctAndSmooth(basePreds, aNorm, residuals, 0.5, 0.5, 2, 2));
            assertTrue(GradCheckUtil.checkGradients(sd, null), "correctAndSmooth grad check failed");
        } finally {
            sd.close();
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testKatzIndexGradCheck(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(21L);
        SameDiff sd = SameDiff.create();
        try {
            SDVariable adj = sd.var("adj", Nd4j.rand(DataType.DOUBLE, 4, 4).muli(0.3));
            scalarLoss(sd, sd.graph().katzIndex(adj, 0.1, 3));
            assertTrue(GradCheckUtil.checkGradients(sd, null), "katzIndex grad check failed");
        } finally {
            sd.close();
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSimRankGradCheck(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(22L);
        SameDiff sd = SameDiff.create();
        try {
            INDArray eye4 = Nd4j.eye(4).castTo(DataType.DOUBLE);
            SDVariable W        = sd.var("W",        Nd4j.rand(DataType.DOUBLE, 4, 4).add(0.1));
            SDVariable identity = sd.constant("identity", eye4);  // constant -- gradient not checked
            scalarLoss(sd, sd.graph().simRank(W, identity, 0.5, 2));
            assertTrue(GradCheckUtil.checkGradients(sd, null), "simRank grad check failed");
        } finally {
            sd.close();
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPersonalizedPageRankGradCheck(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(23L);
        SameDiff sd = SameDiff.create();
        try {
            SDVariable aNorm = sd.var("aNorm", Nd4j.rand(DataType.DOUBLE, 4, 4).add(0.1));
            SDVariable seed  = sd.var("seed",  Nd4j.rand(DataType.DOUBLE, 4, 2).add(0.05));
            scalarLoss(sd, sd.graph().personalizedPageRank(aNorm, seed, 0.5, 3));
            assertTrue(GradCheckUtil.checkGradients(sd, null), "personalizedPageRank grad check failed");
        } finally {
            sd.close();
        }
    }

    // -------------------------------------------------------------------------
    // Pooling / self-supervision / centrality
    // -------------------------------------------------------------------------

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testTopKPoolGradCheck(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(30L);
        SameDiff sd = SameDiff.create();
        try {
            // well-spread scores so the top-k selection is stable under the 1e-5 perturbation
            SDVariable scores = sd.var("scores", Nd4j.rand(DataType.DOUBLE, N).muli(10.0).add(0.5));
            SDVariable feats  = sd.var("feats",  Nd4j.randn(DataType.DOUBLE, N, D));
            scalarLoss(sd, sd.graph().topKPool(scores, feats, 2, N));   // keep top-2 of N nodes
            assertTrue(GradCheckUtil.checkGradients(sd, null), "topKPool grad check failed");
        } finally {
            sd.close();
        }
    }

    /**
     * SortPool (DGCNN, Zhang et al. 2018): sort nodes by a scalar key, keep top-k rows.
     * Differentiable w.r.t. features through the oneHot@features gather. Well-spread sort keys
     * ensure the top-k selection is stable under the 1e-5 gradcheck perturbation.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSortPoolGradCheck(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(42L);
        SameDiff sd = SameDiff.create();
        try {
            // well-spread sort keys so the top-k selection is stable under the 1e-5 perturbation
            SDVariable feats   = sd.var("feats",   Nd4j.randn(DataType.DOUBLE, N, D));
            // sortKey is declared as a constant because sortPool's gradient w.r.t. the key is
            // zero (piecewise-constant index selection); only feats has a meaningful gradient.
            SDVariable sortKey = sd.constant("sortKey", Nd4j.rand(DataType.DOUBLE, N).muli(10.0).add(0.5));
            scalarLoss(sd, sd.graph().sortPool(feats, sortKey, 2, N));  // keep top-2 of N nodes
            assertTrue(GradCheckUtil.checkGradients(sd, null), "sortPool grad check failed");
        } finally {
            sd.close();
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGraceLossGradCheck(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(31L);
        SameDiff sd = SameDiff.create();
        try {
            SDVariable z1 = sd.var("z1", Nd4j.randn(DataType.DOUBLE, N, D));
            SDVariable z2 = sd.var("z2", Nd4j.randn(DataType.DOUBLE, N, D));
            SDVariable id = sd.constant("id", Nd4j.eye(N).castTo(DataType.DOUBLE));
            sd.sum("loss", sd.graph().graceLoss(z1, z2, id, 0.5));   // already scalar; keep a named scalar loss
            assertTrue(GradCheckUtil.checkGradients(sd, null), "graceLoss grad check failed");
        } finally {
            sd.close();
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testClusteringCoefficientGradCheck(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(32L);
        SameDiff sd = SameDiff.create();
        try {
            SDVariable adj = sd.var("adj", Nd4j.rand(DataType.DOUBLE, N, N).add(0.2)); // positive, deg > 1
            SDVariable id  = sd.constant("id", Nd4j.eye(N).castTo(DataType.DOUBLE));
            scalarLoss(sd, sd.graph().clusteringCoefficient(adj, id));
            assertTrue(GradCheckUtil.checkGradients(sd, null), "clusteringCoefficient grad check failed");
        } finally {
            sd.close();
        }
    }

    /** The generated eager Nd4j.graph().X must match the SameDiff sd.graph().X (validates ND-wrap codegen). */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testEagerMatchesSameDiff(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(6L);
        INDArray h = Nd4j.randn(DataType.DOUBLE, N, D);
        INDArray r = Nd4j.randn(DataType.DOUBLE, N, D);
        INDArray t = Nd4j.randn(DataType.DOUBLE, N, D);

        // transE -- dup the inputs into the SameDiff constants so eval()/cleanup doesn't free h/r/t,
        // which the eager call (and the next check) reuse.
        INDArray eagerTransE = Nd4j.graph().transE(h, r, t);
        SameDiff sd1 = SameDiff.create();
        INDArray sdTransE = sd1.graph().transE(sd1.constant(h.dup()), sd1.constant(r.dup()), sd1.constant(t.dup())).eval();
        assertTrue(eagerTransE.equalsWithEps(sdTransE, 1e-9), "eager transE != SameDiff transE");

        // cosineSimilarity
        INDArray eagerCos = Nd4j.graph().cosineSimilarity(h);
        SameDiff sd2 = SameDiff.create();
        INDArray sdCos = sd2.graph().cosineSimilarity(sd2.constant(h.dup())).eval();
        assertTrue(eagerCos.equalsWithEps(sdCos, 1e-9), "eager cosineSimilarity != SameDiff cosineSimilarity");

        // adamicAdar -- a multi-step heuristic (sum/log/reshape/div/mmul) exercising the ND-wrap translation
        INDArray adj = Nd4j.rand(DataType.DOUBLE, N, N).add(0.2);
        INDArray eagerAA = Nd4j.graph().adamicAdar(adj);
        SameDiff sd3 = SameDiff.create();
        INDArray sdAA = sd3.graph().adamicAdar(sd3.constant(adj.dup())).eval();
        assertTrue(eagerAA.equalsWithEps(sdAA, 1e-9), "eager adamicAdar != SameDiff adamicAdar");
    }
}
