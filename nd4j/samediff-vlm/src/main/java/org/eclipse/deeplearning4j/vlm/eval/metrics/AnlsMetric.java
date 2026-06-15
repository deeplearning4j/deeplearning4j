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

package org.eclipse.deeplearning4j.vlm.eval.metrics;

import org.eclipse.deeplearning4j.llm.eval.metrics.EvalMetric;

import java.util.List;

/**
 * Average Normalized Levenshtein Similarity (ANLS) metric.
 * Used as the primary metric for DocVQA evaluation.
 *
 * For each prediction-reference pair, computes the Normalized Levenshtein Similarity (NLS).
 * If NLS < threshold (default 0.5), the score is 0; otherwise it is the NLS value.
 * Returns the maximum score across all reference answers.
 */
public class AnlsMetric implements EvalMetric {

    private final double threshold;

    public AnlsMetric() {
        this(0.5);
    }

    public AnlsMetric(double threshold) {
        this.threshold = threshold;
    }

    @Override
    public String name() {
        return "anls";
    }

    @Override
    public double score(String prediction, List<String> references) {
        if (prediction == null || references == null || references.isEmpty()) return 0.0;
        String normalizedPred = prediction.toLowerCase().trim();
        double maxScore = 0.0;
        for (String ref : references) {
            String normalizedRef = ref.toLowerCase().trim();
            double nls = normalizedLevenshteinSimilarity(normalizedPred, normalizedRef);
            double score = nls < threshold ? 0.0 : nls;
            maxScore = Math.max(maxScore, score);
        }
        return maxScore;
    }

    @Override
    public boolean higherIsBetter() {
        return true;
    }

    /**
     * Compute Normalized Levenshtein Similarity = 1 - (editDistance / max(len1, len2)).
     */
    static double normalizedLevenshteinSimilarity(String s1, String s2) {
        if (s1.equals(s2)) return 1.0;
        int maxLen = Math.max(s1.length(), s2.length());
        if (maxLen == 0) return 1.0;
        int distance = levenshteinDistance(s1, s2);
        return 1.0 - (double) distance / maxLen;
    }

    /**
     * Standard Levenshtein edit distance (insert, delete, substitute).
     */
    public static int levenshteinDistance(String s1, String s2) {
        int m = s1.length();
        int n = s2.length();

        // Use two-row optimization for O(min(m,n)) space
        if (m < n) {
            String temp = s1; s1 = s2; s2 = temp;
            int t = m; m = n; n = t;
        }

        int[] prev = new int[n + 1];
        int[] curr = new int[n + 1];

        for (int j = 0; j <= n; j++) {
            prev[j] = j;
        }

        for (int i = 1; i <= m; i++) {
            curr[0] = i;
            for (int j = 1; j <= n; j++) {
                int cost = s1.charAt(i - 1) == s2.charAt(j - 1) ? 0 : 1;
                curr[j] = Math.min(Math.min(curr[j - 1] + 1, prev[j] + 1), prev[j - 1] + cost);
            }
            int[] temp = prev;
            prev = curr;
            curr = temp;
        }
        return prev[n];
    }
}
