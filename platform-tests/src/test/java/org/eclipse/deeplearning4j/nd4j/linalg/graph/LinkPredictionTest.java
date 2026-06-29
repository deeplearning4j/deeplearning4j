/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.eclipse.deeplearning4j.nd4j.linalg.graph;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.graph.EdgeSplitter;
import org.nd4j.linalg.api.ops.impl.graph.LinkPredictionEvaluation;
import org.nd4j.linalg.api.ops.impl.graph.NegativeEdgeSampler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Correctness tests for link-prediction evaluation and data-prep utilities.
 *
 * All expected values are computed by hand and documented inline.
 */
@NativeTag
public class LinkPredictionTest extends BaseNd4jTestWithBackends {

    private static final double EPS = 1e-6;

    // =========================================================================
    // LinkPredictionEvaluation — AUC-ROC
    // =========================================================================

    /**
     * Perfect ranking: all 3 positives are ranked above all 2 negatives.
     *
     * scores = [0.9, 0.8, 0.7, 0.4, 0.3]
     * labels = [ 1,   1,   1,   0,   0 ]
     *
     * Ascending ranks (lowest score = rank 1):
     *   0.3 → rank 1 (neg), 0.4 → rank 2 (neg), 0.7 → rank 3 (pos),
     *   0.8 → rank 4 (pos), 0.9 → rank 5 (pos)
     *
     * R_pos_asc = 3+4+5 = 12
     * AUC = (12 - 3*4/2) / (3*2) = (12-6)/6 = 6/6 = 1.0
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testAucRocPerfect(Nd4jBackend backend) {
        INDArray scores = Nd4j.createFromArray(new double[]{0.9, 0.8, 0.7, 0.4, 0.3});
        INDArray labels = Nd4j.createFromArray(new double[]{1.0, 1.0, 1.0, 0.0, 0.0});

        LinkPredictionEvaluation.Result r = LinkPredictionEvaluation.eval(scores, labels);
        assertEquals(1.0, r.getAucRoc(), EPS, "Perfect ranking should give AUC=1.0");
        assertEquals(1.0, r.getAveragePrecision(), EPS, "Perfect ranking should give AP=1.0");
    }

    /**
     * Worst ranking: all 3 positives are ranked below all 2 negatives.
     *
     * scores = [0.9, 0.8, 0.7, 0.4, 0.3]
     * labels = [ 0,   0,   0,   1,   1 ]
     *
     * Ascending ranks: 0.3→1(pos), 0.4→2(pos), 0.7→3(neg), 0.8→4(neg), 0.9→5(neg)
     * R_pos_asc = 1+2 = 3
     * AUC = (3 - 2*3/2) / (2*3) = (3-3)/6 = 0.0
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testAucRocWorst(Nd4jBackend backend) {
        INDArray scores = Nd4j.createFromArray(new double[]{0.9, 0.8, 0.7, 0.4, 0.3});
        INDArray labels = Nd4j.createFromArray(new double[]{0.0, 0.0, 0.0, 1.0, 1.0});

        LinkPredictionEvaluation.Result r = LinkPredictionEvaluation.eval(scores, labels);
        assertEquals(0.0, r.getAucRoc(), EPS, "Worst ranking should give AUC=0.0");
    }

    /**
     * All-tied scores: AUC = 0.5.
     *
     * All 5 examples have the same score 0.5. Tie group covers positions 1..5 (desc),
     * so each gets average ascending rank = (1+5)/2 = 3.
     *
     * P=2, N=3, n=5
     * R_pos_asc = 3+3 = 6
     * AUC = (6 - 2*3/2) / (2*3) = (6-3)/6 = 3/6 = 0.5
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testAucRocAllTied(Nd4jBackend backend) {
        INDArray scores = Nd4j.createFromArray(new double[]{0.5, 0.5, 0.5, 0.5, 0.5});
        INDArray labels = Nd4j.createFromArray(new double[]{1.0, 1.0, 0.0, 0.0, 0.0});

        LinkPredictionEvaluation.Result r = LinkPredictionEvaluation.eval(scores, labels);
        assertEquals(0.5, r.getAucRoc(), EPS, "All-tied scores should give AUC=0.5");
    }

    /**
     * Small hand-computed mixed example.
     *
     * scores = [0.9, 0.8, 0.7, 0.6, 0.5]
     * labels = [ 1,   1,   0,   0,   1 ]
     *
     * Concordant pairs (positive score > negative score):
     *   pos=0.9 vs neg=0.7: 0.9>0.7 ✓
     *   pos=0.9 vs neg=0.6: 0.9>0.6 ✓
     *   pos=0.8 vs neg=0.7: 0.8>0.7 ✓
     *   pos=0.8 vs neg=0.6: 0.8>0.6 ✓
     *   pos=0.5 vs neg=0.7: 0.5<0.7 ✗
     *   pos=0.5 vs neg=0.6: 0.5<0.6 ✗
     * AUC = 4 / (3*2) = 4/6 = 2/3 ≈ 0.66667
     *
     * AP: iterate sorted descending positions:
     *   k=1 (score=0.9, label=1): TP=1, P@1=1/1=1.0
     *   k=2 (score=0.8, label=1): TP=2, P@2=2/2=1.0
     *   k=3 (score=0.7, label=0): skip
     *   k=4 (score=0.6, label=0): skip
     *   k=5 (score=0.5, label=1): TP=3, P@5=3/5=0.6
     * AP = (1.0 + 1.0 + 0.6) / 3 = 2.6/3 ≈ 0.86667
     *
     * MRR: positives have descending ranks 1, 2, 5
     * MRR = (1/1 + 1/2 + 1/5) / 3 = (1.0 + 0.5 + 0.2) / 3 = 1.7/3 ≈ 0.56667
     *
     * Hits@1: top-1 contains 1 positive → 1/3
     * Hits@2: top-2 contains 2 positives → 2/3
     * Hits@5: top-5 contains all 3 positives → 3/3 = 1.0
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMetricsMixedHandComputed(Nd4jBackend backend) {
        INDArray scores = Nd4j.createFromArray(new double[]{0.9, 0.8, 0.7, 0.6, 0.5});
        INDArray labels = Nd4j.createFromArray(new double[]{1.0, 1.0, 0.0, 0.0, 1.0});

        LinkPredictionEvaluation.Result r = LinkPredictionEvaluation.eval(scores, labels);

        assertEquals(2.0 / 3.0, r.getAucRoc(), EPS, "AUC-ROC");
        assertEquals(2.6 / 3.0, r.getAveragePrecision(), EPS, "Average Precision");
        assertEquals(1.7 / 3.0, r.getMrr(), EPS, "MRR");

        // Hits@K
        assertEquals(1.0 / 3.0, r.hitsAtK(1), EPS, "Hits@1");
        assertEquals(2.0 / 3.0, r.hitsAtK(2), EPS, "Hits@2");
        assertEquals(1.0,       r.hitsAtK(5), EPS, "Hits@5");
    }

    /**
     * Verify that scores passed in an arbitrary (non-sorted) order give the same
     * metrics as the same data in sorted order.
     *
     * We reuse the same logical example as testMetricsMixedHandComputed but pass
     * the arrays in a shuffled order.
     * scores_shuffled = [0.5, 0.7, 0.9, 0.6, 0.8]
     * labels_shuffled = [ 1,   0,   1,   0,   1 ]
     * (same pairs as before: 0.9→1, 0.8→1, 0.7→0, 0.6→0, 0.5→1)
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMetricsOrderIndependent(Nd4jBackend backend) {
        INDArray scores = Nd4j.createFromArray(new double[]{0.5, 0.7, 0.9, 0.6, 0.8});
        INDArray labels = Nd4j.createFromArray(new double[]{1.0, 0.0, 1.0, 0.0, 1.0});

        LinkPredictionEvaluation.Result r = LinkPredictionEvaluation.eval(scores, labels);

        assertEquals(2.0 / 3.0, r.getAucRoc(), EPS, "AUC-ROC (order independent)");
        assertEquals(2.6 / 3.0, r.getAveragePrecision(), EPS, "AP (order independent)");
        assertEquals(1.7 / 3.0, r.getMrr(), EPS, "MRR (order independent)");
    }

    /**
     * Tie-group AUC test: two tied score groups.
     *
     * scores = [0.8, 0.8, 0.4, 0.4]
     * labels = [1,   0,   1,   0  ]
     *
     * Descending sort (stable): positions 0..3 → scores 0.8, 0.8, 0.4, 0.4
     * Two tie groups:
     *   group A: positions 0,1 (score=0.8), descending ranks 1,2 → avg desc rank = 1.5
     *   group B: positions 2,3 (score=0.4), descending ranks 3,4 → avg desc rank = 3.5
     *
     * Ascending avg ranks:
     *   group A: asc avg rank = n+1 - desc_avg = 5 - 1.5 = 3.5
     *   group B: asc avg rank = 5 - 3.5 = 1.5
     *
     * Positives are at positions 0 (group A, asc rank 3.5) and 2 (group B, asc rank 1.5).
     * R_pos_asc = 3.5 + 1.5 = 5
     * P=2, N=2, n=4
     * AUC = (5 - 2*3/2) / (2*2) = (5-3)/4 = 2/4 = 0.5
     *
     * Concordant-pairs check:
     *   pair (pos=0.8, neg=0.8): tied → contributes 0.5 concordant
     *   pair (pos=0.8, neg=0.4): concordant → 1.0
     *   pair (pos=0.4, neg=0.8): discordant → 0.0
     *   pair (pos=0.4, neg=0.4): tied → 0.5
     * AUC = (0.5 + 1.0 + 0.0 + 0.5) / 4 = 2.0/4 = 0.5 ✓
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testAucRocWithTies(Nd4jBackend backend) {
        INDArray scores = Nd4j.createFromArray(new double[]{0.8, 0.8, 0.4, 0.4});
        INDArray labels = Nd4j.createFromArray(new double[]{1.0, 0.0, 1.0, 0.0});

        LinkPredictionEvaluation.Result r = LinkPredictionEvaluation.eval(scores, labels);
        assertEquals(0.5, r.getAucRoc(), EPS, "Symmetric tie groups should give AUC=0.5");
    }

    // =========================================================================
    // EdgeSplitter
    // =========================================================================

    /**
     * Basic split: 10 edges, train=0.6, val=0.2 → sizes 6, 2, 2.
     * Verify: sizes match, union = input, sets pairwise disjoint.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testEdgeSplitterSizesAndDisjoint(Nd4jBackend backend) {
        // Build a 10-edge edge list: edges (0,1),(1,2),...,(9,0)
        long[] data = new long[20];
        for (int i = 0; i < 10; i++) {
            data[i * 2]     = i;
            data[i * 2 + 1] = (i + 1) % 10;
        }
        INDArray edgeList = Nd4j.createFromArray(data).reshape(10, 2);

        EdgeSplitter splitter = new EdgeSplitter(42L);
        EdgeSplitter.Split split = splitter.split(edgeList, 0.6, 0.2);

        int trainSize = (int) split.getTrain().size(0);
        int valSize   = (int) split.getVal().size(0);
        int testSize  = (int) split.getTest().size(0);

        assertEquals(6, trainSize, "train size");
        assertEquals(2, valSize,   "val size");
        assertEquals(2, testSize,  "test size");
        assertEquals(10, trainSize + valSize + testSize, "sizes must sum to E");

        // Encode each split's edges and verify pairwise disjoint + union = input.
        Set<String> trainSet = edgeSetOf(split.getTrain());
        Set<String> valSet   = edgeSetOf(split.getVal());
        Set<String> testSet  = edgeSetOf(split.getTest());
        Set<String> inputSet = edgeSetOf(edgeList);

        // No overlaps
        Set<String> tvOverlap = overlap(trainSet, valSet);
        assertTrue(tvOverlap.isEmpty(), "train and val must be disjoint, overlap: " + tvOverlap);
        Set<String> ttOverlap = overlap(trainSet, testSet);
        assertTrue(ttOverlap.isEmpty(), "train and test must be disjoint, overlap: " + ttOverlap);
        Set<String> vtOverlap = overlap(valSet, testSet);
        assertTrue(vtOverlap.isEmpty(), "val and test must be disjoint, overlap: " + vtOverlap);

        // Union = input
        Set<String> union = new HashSet<>(trainSet);
        union.addAll(valSet);
        union.addAll(testSet);
        assertEquals(inputSet, union, "union of splits must equal input edge set");
    }

    /**
     * Determinism: the same seed always produces the same split.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testEdgeSplitterDeterministic(Nd4jBackend backend) {
        long[] data = new long[40];
        for (int i = 0; i < 20; i++) {
            data[i * 2]     = i;
            data[i * 2 + 1] = (i + 3) % 20;
        }
        INDArray edgeList = Nd4j.createFromArray(data).reshape(20, 2);

        EdgeSplitter s1 = new EdgeSplitter(123L);
        EdgeSplitter s2 = new EdgeSplitter(123L);

        EdgeSplitter.Split split1 = s1.split(edgeList, 0.7, 0.15);
        EdgeSplitter.Split split2 = s2.split(edgeList, 0.7, 0.15);

        assertEquals(edgeSetOf(split1.getTrain()), edgeSetOf(split2.getTrain()), "train sets must match for same seed");
        assertEquals(edgeSetOf(split1.getVal()),   edgeSetOf(split2.getVal()),   "val sets must match for same seed");
        assertEquals(edgeSetOf(split1.getTest()),  edgeSetOf(split2.getTest()),  "test sets must match for same seed");
    }

    /**
     * Different seeds produce different splits (with high probability on non-trivial data).
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testEdgeSplitterDifferentSeeds(Nd4jBackend backend) {
        long[] data = new long[40];
        for (int i = 0; i < 20; i++) {
            data[i * 2]     = i;
            data[i * 2 + 1] = (i + 3) % 20;
        }
        INDArray edgeList = Nd4j.createFromArray(data).reshape(20, 2);

        EdgeSplitter s1 = new EdgeSplitter(1L);
        EdgeSplitter s2 = new EdgeSplitter(999L);

        EdgeSplitter.Split split1 = s1.split(edgeList, 0.6, 0.2);
        EdgeSplitter.Split split2 = s2.split(edgeList, 0.6, 0.2);

        // Different seeds should produce different train sets with overwhelming probability.
        assertNotEquals(edgeSetOf(split1.getTrain()), edgeSetOf(split2.getTrain()),
                "Different seeds should (almost certainly) produce different splits");
    }

    /**
     * val fraction = 0: edge list split only into train + test, val is empty [0,2].
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testEdgeSplitterZeroValFraction(Nd4jBackend backend) {
        long[] data = new long[20];
        for (int i = 0; i < 10; i++) {
            data[i * 2]     = i;
            data[i * 2 + 1] = (i + 1) % 10;
        }
        INDArray edgeList = Nd4j.createFromArray(data).reshape(10, 2);

        EdgeSplitter splitter = new EdgeSplitter(7L);
        EdgeSplitter.Split split = splitter.split(edgeList, 0.7, 0.0);

        assertEquals(7,  (int) split.getTrain().size(0), "train size");
        assertEquals(0,  (int) split.getVal().size(0),   "val size should be 0");
        assertEquals(3,  (int) split.getTest().size(0),  "test size");
    }

    // =========================================================================
    // NegativeEdgeSampler
    // =========================================================================

    /**
     * Basic contract: sample n=10 negatives from a 6-node graph with 5 edges,
     * no self-loops.  Verify count, no duplicates, no self-loops, none in existing edges.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testNegativeSamplerContract(Nd4jBackend backend) {
        long numNodes = 6;
        // Existing edges: (0,1),(1,2),(2,3),(3,4),(4,5)
        long[] edgeData = {0,1, 1,2, 2,3, 3,4, 4,5};
        INDArray edgeList = Nd4j.createFromArray(edgeData).reshape(5, 2);

        NegativeEdgeSampler sampler =
                NegativeEdgeSampler.fromEdgeList(edgeList, numNodes, /*allowSelfLoops=*/false, 42L);

        INDArray neg = sampler.sample(10);

        // Shape
        assertEquals(2, neg.rank(), "output rank");
        assertEquals(10, neg.size(0), "row count");
        assertEquals(2,  neg.size(1), "col count");

        // No duplicates, no self-loops, none in existing edges
        Set<String> seen = new HashSet<>();
        for (int i = 0; i < 10; i++) {
            long src = neg.getLong(i, 0);
            long dst = neg.getLong(i, 1);

            assertNotEquals(src, dst, "self-loops not allowed (row " + i + ")");
            assertFalse(sampler.isExistingEdge(src, dst),
                    "sampled edge (" + src + "," + dst + ") must not be an existing edge");

            String key = src + "," + dst;
            assertTrue(seen.add(key), "duplicate sampled edge: " + key);
        }
    }

    /**
     * Determinism: same seed → same samples.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testNegativeSamplerDeterministic(Nd4jBackend backend) {
        long numNodes = 10;
        long[] edgeData = {0,1, 2,3, 4,5, 6,7, 8,9};
        INDArray edgeList = Nd4j.createFromArray(edgeData).reshape(5, 2);

        NegativeEdgeSampler s1 =
                NegativeEdgeSampler.fromEdgeList(edgeList, numNodes, false, 77L);
        NegativeEdgeSampler s2 =
                NegativeEdgeSampler.fromEdgeList(edgeList, numNodes, false, 77L);

        INDArray neg1 = s1.sample(20);
        INDArray neg2 = s2.sample(20);

        assertEquals(neg1, neg2, "Same seed must produce identical samples");
    }

    /**
     * Self-loop allowance: when allowSelfLoops=true, no self-loop filtering is applied.
     * When false, verify that no self-loops appear over many samples.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testNegativeSamplerNoSelfLoops(Nd4jBackend backend) {
        long numNodes = 5;
        // Only 2 existing edges — plenty of negatives.
        long[] edgeData = {0, 1, 2, 3};
        INDArray edgeList = Nd4j.createFromArray(edgeData).reshape(2, 2);

        NegativeEdgeSampler sampler =
                NegativeEdgeSampler.fromEdgeList(edgeList, numNodes, /*allowSelfLoops=*/false, 13L);

        // 5-node graph, no self-loops: 5*4=20 directed edges minus 2 existing = 18 max.
        INDArray neg = sampler.sample(10);
        for (int i = 0; i < 10; i++) {
            long src = neg.getLong(i, 0);
            long dst = neg.getLong(i, 1);
            assertNotEquals(src, dst,
                    "Self-loops must not appear when allowSelfLoops=false (row " + i + ")");
        }
    }

    /**
     * Verify correct count: sample exactly n=1 and exactly n=50.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testNegativeSamplerCount(Nd4jBackend backend) {
        long numNodes = 20;
        long[] edgeData = {0,1, 5,10};
        INDArray edgeList = Nd4j.createFromArray(edgeData).reshape(2, 2);

        NegativeEdgeSampler sampler =
                NegativeEdgeSampler.fromEdgeList(edgeList, numNodes, false, 99L);

        assertEquals(1,  sampler.sample(1).size(0),  "sample(1) row count");
        assertEquals(50, sampler.sample(50).size(0), "sample(50) row count");
    }

    // =========================================================================
    // Helpers
    // =========================================================================

    /** Encode edge list rows as "src,dst" strings for set membership checks. */
    private static Set<String> edgeSetOf(INDArray edgeList) {
        Set<String> set = new HashSet<>();
        int E = (int) edgeList.size(0);
        for (int i = 0; i < E; i++) {
            set.add(edgeList.getLong(i, 0) + "," + edgeList.getLong(i, 1));
        }
        return set;
    }

    /** Return the intersection of two sets (non-destructive). */
    private static Set<String> overlap(Set<String> a, Set<String> b) {
        Set<String> result = new HashSet<>(a);
        result.retainAll(b);
        return result;
    }
}
