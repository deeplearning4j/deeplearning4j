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
 * BLEU (Bilingual Evaluation Understudy) metric.
 * Computes BLEU-N score (default N=4) with brevity penalty.
 * Follows the original Papineni et al. (2002) formulation.
 */
public class BleuMetric implements EvalMetric {

    private final int maxN;

    public BleuMetric() {
        this(4);
    }

    public BleuMetric(int maxN) {
        this.maxN = maxN;
    }

    @Override
    public String name() {
        return "bleu-" + maxN;
    }

    @Override
    public double score(String prediction, List<String> references) {
        if (prediction == null || references == null || references.isEmpty()) return 0.0;

        List<String> predTokens = tokenize(prediction);
        if (predTokens.isEmpty()) return 0.0;

        List<List<String>> refTokenLists = new ArrayList<>();
        for (String ref : references) {
            refTokenLists.add(tokenize(ref));
        }

        // Compute modified precision for each n-gram order
        double logBleu = 0.0;
        for (int n = 1; n <= maxN; n++) {
            double precision = modifiedPrecision(predTokens, refTokenLists, n);
            if (precision == 0.0) return 0.0; // If any n-gram precision is 0, BLEU is 0
            logBleu += Math.log(precision) / maxN;
        }

        // Brevity penalty
        int predLen = predTokens.size();
        int closestRefLen = closestReferenceLength(predLen, refTokenLists);
        double bp = predLen >= closestRefLen ? 1.0 : Math.exp(1.0 - (double) closestRefLen / predLen);

        return bp * Math.exp(logBleu);
    }

    @Override
    public boolean higherIsBetter() {
        return true;
    }

    private double modifiedPrecision(List<String> prediction, List<List<String>> references, int n) {
        Map<String, Integer> predNgrams = extractNgrams(prediction, n);
        if (predNgrams.isEmpty()) return 0.0;

        // Clipped counts: for each n-gram, count is min(pred_count, max_ref_count)
        Map<String, Integer> maxRefCounts = new HashMap<>();
        for (List<String> ref : references) {
            Map<String, Integer> refNgrams = extractNgrams(ref, n);
            for (Map.Entry<String, Integer> entry : refNgrams.entrySet()) {
                maxRefCounts.merge(entry.getKey(), entry.getValue(), Math::max);
            }
        }

        int clippedCount = 0;
        int totalCount = 0;
        for (Map.Entry<String, Integer> entry : predNgrams.entrySet()) {
            int predCount = entry.getValue();
            int maxRef = maxRefCounts.getOrDefault(entry.getKey(), 0);
            clippedCount += Math.min(predCount, maxRef);
            totalCount += predCount;
        }

        return totalCount > 0 ? (double) clippedCount / totalCount : 0.0;
    }

    private int closestReferenceLength(int predLen, List<List<String>> references) {
        int closest = Integer.MAX_VALUE;
        int closestDiff = Integer.MAX_VALUE;
        for (List<String> ref : references) {
            int diff = Math.abs(ref.size() - predLen);
            if (diff < closestDiff || (diff == closestDiff && ref.size() < closest)) {
                closestDiff = diff;
                closest = ref.size();
            }
        }
        return closest;
    }

    static Map<String, Integer> extractNgrams(List<String> tokens, int n) {
        Map<String, Integer> ngrams = new HashMap<>();
        for (int i = 0; i <= tokens.size() - n; i++) {
            StringBuilder sb = new StringBuilder();
            for (int j = 0; j < n; j++) {
                if (j > 0) sb.append(" ");
                sb.append(tokens.get(i + j));
            }
            ngrams.merge(sb.toString(), 1, Integer::sum);
        }
        return ngrams;
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
