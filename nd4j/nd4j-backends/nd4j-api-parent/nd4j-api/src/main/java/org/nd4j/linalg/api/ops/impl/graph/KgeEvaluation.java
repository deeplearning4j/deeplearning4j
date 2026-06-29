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

package org.nd4j.linalg.api.ops.impl.graph;

import java.util.Set;

/**
 * Standard knowledge-graph completion ranking metrics: Mean Reciprocal Rank (MRR) and Hits@K.
 *
 * <p>For each test triple {@code (h, r, ?)} the model scores every candidate tail; the rank of the
 * true tail (1 = best) gives {@code 1/rank} for MRR and a hit if {@code rank <= K}. The
 * <em>filtered</em> setting (Bordes et al. 2013) excludes <em>other</em> known-true tails so they
 * don't push the held-out answer down the ranking.
 */
public final class KgeEvaluation {

    private KgeEvaluation() { }

    /** Computed ranking metrics. */
    public static final class Metrics {
        public final double mrr;
        public final int[] ks;
        public final double[] hitsAtK;

        public Metrics(double mrr, int[] ks, double[] hitsAtK) {
            this.mrr = mrr;
            this.ks = ks;
            this.hitsAtK = hitsAtK;
        }

        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder(String.format("MRR=%.4f", mrr));
            for (int i = 0; i < ks.length; i++) {
                sb.append(String.format("  Hits@%d=%.4f", ks[i], hitsAtK[i]));
            }
            return sb.toString();
        }
    }

    /**
     * Raw ranking metrics.
     *
     * @param tailScores [numTest][numEntities] — model score of every candidate tail per test triple
     * @param trueTails  [numTest] — the held-out true tail index per test triple
     * @param ks         the K values for Hits@K
     */
    public static Metrics evaluate(double[][] tailScores, int[] trueTails, int[] ks) {
        return evaluateFiltered(tailScores, trueTails, null, ks);
    }

    /**
     * Filtered ranking metrics.
     *
     * @param tailScores  [numTest][numEntities] — model score of every candidate tail per test triple
     * @param trueTails   [numTest] — the held-out true tail index per test triple
     * @param filterTails [numTest] — other known-true tail indices for each test (h,r) to exclude from
     *                    the ranking; entries (or the whole array) may be {@code null} for raw ranking
     * @param ks          the K values for Hits@K
     */
    public static Metrics evaluateFiltered(double[][] tailScores, int[] trueTails,
                                           Set<Integer>[] filterTails, int[] ks) {
        double mrrSum = 0.0;
        int[] hits = new int[ks.length];
        final int n = trueTails.length;
        for (int i = 0; i < n; i++) {
            double trueScore = tailScores[i][trueTails[i]];
            int rank = 1;
            for (int e = 0; e < tailScores[i].length; e++) {
                if (e == trueTails[i]) continue;
                if (filterTails != null && filterTails[i] != null && filterTails[i].contains(e)) continue;
                if (tailScores[i][e] > trueScore) rank++;
            }
            mrrSum += 1.0 / rank;
            for (int k = 0; k < ks.length; k++) {
                if (rank <= ks[k]) hits[k]++;
            }
        }
        double[] hitsAtK = new double[ks.length];
        for (int k = 0; k < ks.length; k++) {
            hitsAtK[k] = (double) hits[k] / n;
        }
        return new Metrics(mrrSum / n, ks, hitsAtK);
    }
}
