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
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.validation.GradCheckUtil;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.graph.KgeEvaluation;
import org.nd4j.linalg.api.ops.impl.graph.KgeTripleSampler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Gradient checks for the {@code sd.graph()} scoring functions + {@code sd.gnn().compGcnConv}, and
 * unit tests for the knowledge-graph training/eval utilities ({@link KgeTripleSampler},
 * {@link KgeEvaluation}).
 */
@Tag(TagNames.SAMEDIFF)
public class SparseKgeTest extends BaseNd4jTestWithBackends {

    private static final int BATCH = 4;
    private static final int DIM = 3;

    @BeforeEach
    public void purge() {
        Nd4j.getConstantHandler().purgeConstants();
    }

    private static SDVariable emb(SameDiff sd, String name) {
        return sd.var(name, Nd4j.randn(DataType.DOUBLE, BATCH, DIM));
    }

    // ---- scoring functions ----

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testTransEGradCheck(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(1L);
        SameDiff sd = SameDiff.create();
        try {
            SDVariable h = emb(sd, "h"), r = emb(sd, "r"), t = emb(sd, "t");
            sd.mean("loss", sd.graph().transE(h, r, t));
            assertTrue(GradCheckUtil.checkGradients(sd, null), "transE grad check failed");
        } finally {
            sd.close();
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testHolEGradCheck(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(1L);
        SameDiff sd = SameDiff.create();
        try {
            SDVariable h = emb(sd, "h"), r = emb(sd, "r"), t = emb(sd, "t");
            // HolE = relation . circular-correlation(head, tail), composed from the differentiable DFT op
            sd.mean("loss", sd.graph().holE(h, r, t));
            assertTrue(GradCheckUtil.checkGradients(sd, null), "holE grad check failed");
        } finally {
            sd.close();
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testDistMultGradCheck(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(2L);
        SameDiff sd = SameDiff.create();
        try {
            SDVariable h = emb(sd, "h"), r = emb(sd, "r"), t = emb(sd, "t");
            sd.mean("loss", sd.graph().distMult(h, r, t));
            assertTrue(GradCheckUtil.checkGradients(sd, null), "distMult grad check failed");
        } finally {
            sd.close();
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testComplExGradCheck(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(3L);
        SameDiff sd = SameDiff.create();
        try {
            SDVariable hRe = emb(sd, "hRe"), hIm = emb(sd, "hIm");
            SDVariable rRe = emb(sd, "rRe"), rIm = emb(sd, "rIm");
            SDVariable tRe = emb(sd, "tRe"), tIm = emb(sd, "tIm");
            sd.mean("loss", sd.graph().complEx(hRe, hIm, rRe, rIm, tRe, tIm));
            assertTrue(GradCheckUtil.checkGradients(sd, null), "complEx grad check failed");
        } finally {
            sd.close();
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testRotatEGradCheck(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(4L);
        SameDiff sd = SameDiff.create();
        try {
            SDVariable hRe = emb(sd, "hRe"), hIm = emb(sd, "hIm");
            SDVariable phase = emb(sd, "phase");
            SDVariable tRe = emb(sd, "tRe"), tIm = emb(sd, "tIm");
            sd.mean("loss", sd.graph().rotatE(hRe, hIm, phase, tRe, tIm));
            assertTrue(GradCheckUtil.checkGradients(sd, null), "rotatE grad check failed");
        } finally {
            sd.close();
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMarginRankingLossGradCheck(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(5L);
        SameDiff sd = SameDiff.create();
        try {
            SDVariable h = emb(sd, "h"), r = emb(sd, "r"), tPos = emb(sd, "tPos"), tNeg = emb(sd, "tNeg");
            SDVariable pos = sd.graph().distMult(h, r, tPos);
            SDVariable neg = sd.graph().distMult(h, r, tNeg);
            SDVariable loss = sd.graph().marginRankingLoss(pos, neg, 1.0);
            sd.setLossVariables(loss);
            assertTrue(GradCheckUtil.checkGradients(sd, null), "marginRankingLoss grad check failed");
        } finally {
            sd.close();
        }
    }

    // ---- CompGCN ----

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testCompGcnConvGradCheck(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(6L);
        // common 4-node graph: colIdx=[1,2,0,3,0,1], rowPtr=[0,2,4,5,6], nnz=6
        int n = 4, nnz = 6, numRel = 2, dim = 3;
        SameDiff sd = SameDiff.create();
        try {
            SDVariable colIdx = sd.constant("colIdx", Nd4j.createFromArray(new int[]{1, 2, 0, 3, 0, 1}));
            SDVariable rowPtr = sd.constant("rowPtr", Nd4j.createFromArray(new int[]{0, 2, 4, 5, 6}));
            SDVariable edgeRel = sd.constant("edgeRel", Nd4j.createFromArray(new int[]{0, 1, 0, 1, 0, 1}));

            SDVariable X      = sd.var("X", Nd4j.randn(DataType.DOUBLE, n, dim));
            SDVariable relEmb = sd.var("relEmb", Nd4j.randn(DataType.DOUBLE, numRel, dim));
            SDVariable W      = sd.var("W", Nd4j.randn(DataType.DOUBLE, dim, dim).muli(0.5));

            // compOp=1 (element-wise multiply, DistMult-style); applyRelu=false for a clean Jacobian
            SDVariable out = sd.gnn().compGcnConv(X, relEmb, edgeRel, W, colIdx, rowPtr, n, n, 1, false);
            sd.mean("loss", out);
            assertTrue(GradCheckUtil.checkGradients(sd, null), "compGcnConv grad check failed");
        } finally {
            sd.close();
        }
    }

    // ---- utilities (pure JVM) ----

    @Test
    public void testNegativeSamplingValidAndFiltered() {
        int[][] triples = {{0, 0, 1}, {1, 1, 2}, {2, 0, 3}, {3, 1, 0}};
        int numEntities = 4, numRel = 2, negPerPos = 3;
        Set<Long> known = KgeTripleSampler.knownSet(triples, numEntities, numRel);
        int[][] negs = KgeTripleSampler.corrupt(triples, numEntities, numRel, negPerPos, known, 123L);

        assertEquals(triples.length * negPerPos, negs.length, "negatives count");
        for (int i = 0; i < negs.length; i++) {
            int[] pos = triples[i / negPerPos];
            int[] neg = negs[i];
            assertEquals(pos[1], neg[1], "relation is never corrupted");
            assertTrue(neg[0] != pos[0] || neg[2] != pos[2], "head or tail must differ from positive");
            for (int e = 0; e < 3; e++) assertTrue(neg[e] >= 0, "valid entity/relation id");
            assertTrue(neg[0] < numEntities && neg[2] < numEntities, "entity ids in range");
            // filtered: a generated negative must not be a known true triple
            assertFalse(known.contains(((long) neg[0] * numRel + neg[1]) * numEntities + neg[2]),
                    "filtered negative is not a true triple");
        }
        assertArrayEquals(new int[]{0, 1, 2, 3}, KgeTripleSampler.column(triples, 0), "head column");
    }

    @Test
    public void testEvaluationMrrAndHits() {
        // test 0: true tail (idx 2) has the highest score -> rank 1
        // test 1: true tail (idx 2, score 0.5) beaten only by idx 0 (0.9) -> rank 2
        double[][] scores = {
                {0.1, 0.5, 0.9, 0.3},
                {0.9, 0.1, 0.5, 0.3}
        };
        int[] trueTails = {2, 2};
        KgeEvaluation.Metrics m = KgeEvaluation.evaluate(scores, trueTails, new int[]{1, 3});

        assertEquals(0.75, m.mrr, 1e-9, "MRR = (1/1 + 1/2)/2");
        assertEquals(0.5, m.hitsAtK[0], 1e-9, "Hits@1");
        assertEquals(1.0, m.hitsAtK[1], 1e-9, "Hits@3");
    }

    // ---- additional scorers ----

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testTransHGradCheck(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(7L);
        SameDiff sd = SameDiff.create();
        try {
            SDVariable h = emb(sd, "h"), wr = emb(sd, "wr"), r = emb(sd, "r"), t = emb(sd, "t");
            sd.mean("loss", sd.graph().transH(h, wr, r, t));
            assertTrue(GradCheckUtil.checkGradients(sd, null), "transH grad check failed");
        } finally {
            sd.close();
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testTransETGradCheck(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(8L);
        SameDiff sd = SameDiff.create();
        try {
            SDVariable h = emb(sd, "h"), r = emb(sd, "r"), tau = emb(sd, "tau"), t = emb(sd, "t");
            sd.mean("loss", sd.graph().transET(h, r, tau, t));
            assertTrue(GradCheckUtil.checkGradients(sd, null), "transET grad check failed");
        } finally {
            sd.close();
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testTuckERGradCheck(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(9L);
        final int de = 3, dr = 2;
        SameDiff sd = SameDiff.create();
        try {
            SDVariable h    = sd.var("h", Nd4j.randn(DataType.DOUBLE, BATCH, de));
            SDVariable r    = sd.var("r", Nd4j.randn(DataType.DOUBLE, BATCH, dr));
            SDVariable t    = sd.var("t", Nd4j.randn(DataType.DOUBLE, BATCH, de));
            SDVariable core = sd.var("core", Nd4j.randn(DataType.DOUBLE, de, dr, de).muli(0.3));
            sd.mean("loss", sd.graph().tuckER(h, r, t, core, de, dr));
            assertTrue(GradCheckUtil.checkGradients(sd, null), "tuckER grad check failed");
        } finally {
            sd.close();
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testConvEGradCheck(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(10L);
        final int embH = 2, embW = 4, de = embH * embW, channels = 2;
        SameDiff sd = SameDiff.create();
        try {
            SDVariable h     = sd.var("h", Nd4j.randn(DataType.DOUBLE, BATCH, de));
            SDVariable r     = sd.var("r", Nd4j.randn(DataType.DOUBLE, BATCH, de));
            SDVariable t     = sd.var("t", Nd4j.randn(DataType.DOUBLE, BATCH, de));
            // Positive biases (and small weights) keep both ReLUs in their linear region, so the
            // finite-difference gradient check is stable -- with zero biases the conv outputs
            // straddle 0 and ReLU kinks make the numerical gradient of the conv params unreliable.
            SDVariable convW = sd.var("convW", Nd4j.randn(DataType.DOUBLE, 3, 3, 1, channels).muli(0.2));
            SDVariable convB = sd.var("convB", Nd4j.ones(DataType.DOUBLE, channels).muli(5.0));
            long flatDim     = (long) channels * (2L * embH - 2) * (embW - 2);
            SDVariable fcW   = sd.var("fcW", Nd4j.randn(DataType.DOUBLE, flatDim, de).muli(0.2));
            SDVariable fcB   = sd.var("fcB", Nd4j.ones(DataType.DOUBLE, de).muli(5.0));
            sd.mean("loss", sd.graph().convE(h, r, t, convW, convB, fcW, fcB, embH, embW, channels));
            assertTrue(GradCheckUtil.checkGradients(sd, null), "convE grad check failed");
        } finally {
            sd.close();
        }
    }
}
