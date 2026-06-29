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

package org.nd4j.linalg.api.ops.impl.graph;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Arrays;

/**
 * Link-prediction evaluation utilities.
 *
 * <p>Given predicted edge scores (higher = more likely to be a true edge) and binary
 * ground-truth labels (1 = true edge, 0 = non-edge), computes:
 * <ul>
 *   <li>AUC-ROC — area under the receiver-operating-characteristic curve (Wilcoxon rank-sum)</li>
 *   <li>Average Precision — area under the precision-recall curve (sklearn-compatible step AP)</li>
 *   <li>Hits@K — fraction of true edges whose score ranks in the top-K over ALL candidates</li>
 *   <li>MRR — mean reciprocal rank of true edges (using average rank for tied scores)</li>
 * </ul>
 *
 * <p>Usage:
 * <pre>{@code
 *   LinkPredictionEvaluation.Result r = LinkPredictionEvaluation.eval(scores, labels);
 *   double auc  = r.getAucRoc();
 *   double ap   = r.getAveragePrecision();
 *   double h10  = r.hitsAtK(10);
 *   double mrr  = r.getMrr();
 * }</pre>
 *
 * <p>All computation is pure Java; no backend GPU calls are made.
 */
public class LinkPredictionEvaluation {

    private LinkPredictionEvaluation() {}

    // -------------------------------------------------------------------------
    // Result holder
    // -------------------------------------------------------------------------

    /**
     * Immutable holder for all link-prediction metrics from a single {@link #eval} call.
     */
    public static final class Result {
        private final double aucRoc;
        private final double averagePrecision;
        private final double mrr;
        /** Sorted descending scores (index 0 = highest) used internally for Hits@K. */
        private final double[] sortedScores;
        /** Labels in the same sorted order. */
        private final double[] sortedLabels;
        private final int numPositives;

        private Result(double aucRoc, double averagePrecision, double mrr,
                       double[] sortedScores, double[] sortedLabels, int numPositives) {
            this.aucRoc = aucRoc;
            this.averagePrecision = averagePrecision;
            this.mrr = mrr;
            this.sortedScores = sortedScores;
            this.sortedLabels = sortedLabels;
            this.numPositives = numPositives;
        }

        /** Area under the ROC curve, in [0, 1].  0.5 = random, 1.0 = perfect. */
        public double getAucRoc() { return aucRoc; }

        /** Area under the precision-recall curve (Average Precision), in [0, 1]. */
        public double getAveragePrecision() { return averagePrecision; }

        /**
         * Mean Reciprocal Rank of true edges.
         * Each true edge is assigned its average rank (1-indexed) among all candidates
         * sorted by score descending. MRR = mean of (1 / rank).
         */
        public double getMrr() { return mrr; }

        /**
         * Fraction of true edges whose predicted score is in the top-K over ALL candidates.
         *
         * @param k cut-off (must be &ge; 1)
         * @return Hits@K in [0, 1]; 1.0 means every true edge is in the top K.
         */
        public double hitsAtK(int k) {
            if (k < 1) throw new IllegalArgumentException("k must be >= 1, got " + k);
            if (numPositives == 0) return Double.NaN;
            int n = sortedLabels.length;
            int limit = Math.min(k, n);
            int hitsInTopK = 0;
            for (int i = 0; i < limit; i++) {
                if (sortedLabels[i] == 1.0) hitsInTopK++;
            }
            return (double) hitsInTopK / numPositives;
        }

        @Override
        public String toString() {
            return String.format("LinkPredictionResult{AUC-ROC=%.4f, AP=%.4f, MRR=%.4f}",
                    aucRoc, averagePrecision, mrr);
        }
    }

    // -------------------------------------------------------------------------
    // Main evaluation entry point
    // -------------------------------------------------------------------------

    /**
     * Evaluate link-prediction quality.
     *
     * @param scores 1-D INDArray of predicted edge scores (float or double).
     *               Higher scores should indicate higher probability of a true edge.
     * @param labels 1-D INDArray of binary ground-truth labels (1 = true edge, 0 = non-edge).
     *               Must have the same length as {@code scores}.
     * @return a {@link Result} containing AUC-ROC, AP, MRR, and Hits@K accessors.
     * @throws IllegalArgumentException if arrays have different lengths or are empty.
     */
    public static Result eval(INDArray scores, INDArray labels) {
        int n = (int) scores.length();
        if (n != (int) labels.length()) {
            throw new IllegalArgumentException(
                    "scores and labels must have the same length, got " + n + " vs " + labels.length());
        }
        if (n == 0) throw new IllegalArgumentException("scores/labels must be non-empty");

        // Extract to primitive arrays for fast sorting.
        double[] s = new double[n];
        double[] l = new double[n];
        for (int i = 0; i < n; i++) {
            s[i] = scores.getDouble(i);
            l[i] = labels.getDouble(i);
        }

        // Sort indices by score descending; ties broken stably (preserving insertion order).
        Integer[] idx = new Integer[n];
        for (int i = 0; i < n; i++) idx[i] = i;
        Arrays.sort(idx, (a, b) -> {
            int c = Double.compare(s[b], s[a]); // descending
            return c != 0 ? c : Integer.compare(a, b); // stable
        });

        double[] sortedScores  = new double[n];
        double[] sortedLabels  = new double[n];
        for (int i = 0; i < n; i++) {
            sortedScores[i] = s[idx[i]];
            sortedLabels[i] = l[idx[i]];
        }

        int numPos = 0;
        for (double lv : sortedLabels) if (lv == 1.0) numPos++;
        int numNeg = n - numPos;

        double aucRoc          = computeAucRoc(sortedScores, sortedLabels, numPos, numNeg, n);
        double averagePrecision = computeAveragePrecision(sortedLabels, numPos, n);
        double mrr             = computeMrr(sortedScores, sortedLabels, numPos);

        return new Result(aucRoc, averagePrecision, mrr, sortedScores, sortedLabels, numPos);
    }

    // -------------------------------------------------------------------------
    // AUC-ROC via Wilcoxon rank-sum statistic
    // -------------------------------------------------------------------------

    /**
     * Computes AUC-ROC using the Wilcoxon rank-sum identity.
     *
     * <p>The standard formula uses <em>ascending</em> ranks (rank 1 = lowest score).
     * Because our array is already sorted <em>descending</em> (rank 1-desc = highest score),
     * ascending rank for position i (0-indexed, descending sort) is {@code n - i}.
     * Ties are handled by assigning the average rank over the tied group.
     *
     * <p>AUC = (R_pos_asc - numPos*(numPos+1)/2) / (numPos * numNeg)
     * where R_pos_asc is the sum of ascending average ranks of positive examples.
     *
     * <p>Equivalently, using descending ranks (R_pos_desc, where rank 1 = highest score):
     * R_pos_asc = numPos*(n+1) - R_pos_desc
     * AUC = (numPos*(n+1) - R_pos_desc - numPos*(numPos+1)/2) / (numPos*numNeg)
     */
    private static double computeAucRoc(double[] sortedScores, double[] sortedLabels,
                                         int numPos, int numNeg, int n) {
        if (numPos == 0 || numNeg == 0) return Double.NaN;

        // Compute sum of descending average ranks for positive examples.
        // Sorted descending: index 0 = highest score, descending rank 1.
        // For a tie group occupying 0-indexed positions [i, j), the
        // 1-indexed descending ranks are i+1 .. j; average = (i+1+j)/2.0
        double rankSumDesc = 0.0;
        int i = 0;
        while (i < n) {
            int j = i;
            while (j < n && sortedScores[j] == sortedScores[i]) j++;
            double avgRankDesc = (i + 1 + j) / 2.0; // average of i+1..j
            for (int k = i; k < j; k++) {
                if (sortedLabels[k] == 1.0) {
                    rankSumDesc += avgRankDesc;
                }
            }
            i = j;
        }

        // Convert to ascending rank sum, then apply Wilcoxon formula.
        double rankSumAsc = (double) numPos * (n + 1) - rankSumDesc;
        return (rankSumAsc - (double) numPos * (numPos + 1) / 2.0) / ((double) numPos * numNeg);
    }

    // -------------------------------------------------------------------------
    // Average Precision (area under precision-recall curve, step method)
    // -------------------------------------------------------------------------

    /**
     * Computes Average Precision using the step method (sklearn-compatible).
     *
     * <p>AP = Σ_k P(k) * ΔR(k), where the sum is over all k where label[k]=1,
     * P(k) = (true positives in top-k) / k, ΔR(k) = 1 / numPos.
     *
     * <p>Equivalently: AP = (1/numPos) * Σ_{k: label[k]=1} precision@k
     */
    private static double computeAveragePrecision(double[] sortedLabels, int numPos, int n) {
        if (numPos == 0) return Double.NaN;
        double apSum = 0.0;
        int truePositives = 0;
        for (int k = 0; k < n; k++) {
            if (sortedLabels[k] == 1.0) {
                truePositives++;
                // precision at position k (1-indexed = k+1)
                double precisionAtK = (double) truePositives / (k + 1);
                apSum += precisionAtK;
            }
        }
        return apSum / numPos;
    }

    // -------------------------------------------------------------------------
    // MRR — mean reciprocal rank of positive examples
    // -------------------------------------------------------------------------

    /**
     * Computes MRR.  For each positive example, its rank is the average rank of its
     * tied score group (same tie-breaking as AUC-ROC, 1-indexed).
     * MRR = mean over positives of (1 / rank).
     */
    private static double computeMrr(double[] sortedScores, double[] sortedLabels, int numPos) {
        if (numPos == 0) return Double.NaN;
        int n = sortedScores.length;

        // Pre-compute the average rank for every position.
        double[] avgRank = new double[n];
        int i = 0;
        while (i < n) {
            int j = i;
            while (j < n && sortedScores[j] == sortedScores[i]) j++;
            double ar = (i + 1 + j) / 2.0; // average of 1-indexed ranks i+1..j
            for (int k = i; k < j; k++) avgRank[k] = ar;
            i = j;
        }

        double reciprocalSum = 0.0;
        for (int k = 0; k < n; k++) {
            if (sortedLabels[k] == 1.0) {
                reciprocalSum += 1.0 / avgRank[k];
            }
        }
        return reciprocalSum / numPos;
    }
}
