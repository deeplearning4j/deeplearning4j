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

package org.eclipse.deeplearning4j.llm.eval.metrics;

import java.util.*;

/**
 * ROUGE (Recall-Oriented Understudy for Gisting Evaluation) metric.
 * Supports ROUGE-N (unigram/bigram overlap) and ROUGE-L (longest common subsequence).
 * Reports F1 score by default.
 */
public class RougeMetric implements EvalMetric {

    public enum RougeType {
        ROUGE_1, ROUGE_2, ROUGE_L
    }

    public enum ScoreType {
        PRECISION, RECALL, F1
    }

    private final RougeType rougeType;
    private final ScoreType scoreType;

    public RougeMetric() {
        this(RougeType.ROUGE_L, ScoreType.F1);
    }

    public RougeMetric(RougeType rougeType) {
        this(rougeType, ScoreType.F1);
    }

    public RougeMetric(RougeType rougeType, ScoreType scoreType) {
        this.rougeType = rougeType;
        this.scoreType = scoreType;
    }

    @Override
    public String name() {
        return rougeType.name().toLowerCase().replace("_", "-") + "-" + scoreType.name().toLowerCase();
    }

    @Override
    public double score(String prediction, List<String> references) {
        if (prediction == null || references == null || references.isEmpty()) return 0.0;
        double maxScore = 0.0;
        for (String ref : references) {
            maxScore = Math.max(maxScore, computeScore(prediction, ref));
        }
        return maxScore;
    }

    @Override
    public boolean higherIsBetter() {
        return true;
    }

    private double computeScore(String prediction, String reference) {
        List<String> predTokens = tokenize(prediction);
        List<String> refTokens = tokenize(reference);

        if (predTokens.isEmpty() && refTokens.isEmpty()) return 1.0;
        if (predTokens.isEmpty() || refTokens.isEmpty()) return 0.0;

        switch (rougeType) {
            case ROUGE_1: return computeRougeN(predTokens, refTokens, 1);
            case ROUGE_2: return computeRougeN(predTokens, refTokens, 2);
            case ROUGE_L: return computeRougeL(predTokens, refTokens);
            default: throw new IllegalStateException("Unknown ROUGE type: " + rougeType);
        }
    }

    private double computeRougeN(List<String> pred, List<String> ref, int n) {
        Map<String, Integer> predNgrams = BleuMetric.extractNgrams(pred, n);
        Map<String, Integer> refNgrams = BleuMetric.extractNgrams(ref, n);

        if (predNgrams.isEmpty() || refNgrams.isEmpty()) return 0.0;

        int overlap = 0;
        for (Map.Entry<String, Integer> entry : predNgrams.entrySet()) {
            if (refNgrams.containsKey(entry.getKey())) {
                overlap += Math.min(entry.getValue(), refNgrams.get(entry.getKey()));
            }
        }

        int predTotal = pred.size() - n + 1;
        int refTotal = ref.size() - n + 1;

        double precision = predTotal > 0 ? (double) overlap / predTotal : 0.0;
        double recall = refTotal > 0 ? (double) overlap / refTotal : 0.0;

        return selectScore(precision, recall);
    }

    private double computeRougeL(List<String> pred, List<String> ref) {
        int lcsLen = lcsLength(pred, ref);

        double precision = pred.size() > 0 ? (double) lcsLen / pred.size() : 0.0;
        double recall = ref.size() > 0 ? (double) lcsLen / ref.size() : 0.0;

        return selectScore(precision, recall);
    }

    private double selectScore(double precision, double recall) {
        switch (scoreType) {
            case PRECISION: return precision;
            case RECALL: return recall;
            case F1:
                if (precision + recall == 0) return 0.0;
                return 2.0 * precision * recall / (precision + recall);
            default: return 0.0;
        }
    }

    static int lcsLength(List<String> a, List<String> b) {
        int m = a.size();
        int n = b.size();
        int[] prev = new int[n + 1];
        int[] curr = new int[n + 1];

        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (a.get(i - 1).equals(b.get(j - 1))) {
                    curr[j] = prev[j - 1] + 1;
                } else {
                    curr[j] = Math.max(curr[j - 1], prev[j]);
                }
            }
            int[] temp = prev;
            prev = curr;
            curr = temp;
            Arrays.fill(curr, 0);
        }
        return prev[n];
    }

    private static List<String> tokenize(String text) {
        if (text == null || text.isEmpty()) return Collections.emptyList();
        String[] parts = text.toLowerCase().trim().split("\\s+");
        List<String> tokens = new ArrayList<>();
        for (String part : parts) {
            if (!part.isEmpty()) tokens.add(part);
        }
        return tokens;
    }
}
