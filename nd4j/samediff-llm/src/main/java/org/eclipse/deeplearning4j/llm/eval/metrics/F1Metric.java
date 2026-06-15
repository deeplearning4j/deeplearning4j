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
 * Token-level F1 metric following SQuAD evaluation.
 * Computes precision/recall/F1 based on token overlap between prediction and reference.
 * Returns the maximum F1 score across all reference answers.
 */
public class F1Metric implements EvalMetric {

    @Override
    public String name() {
        return "f1";
    }

    @Override
    public double score(String prediction, List<String> references) {
        if (prediction == null || references == null || references.isEmpty()) return 0.0;
        double maxF1 = 0.0;
        for (String ref : references) {
            maxF1 = Math.max(maxF1, computeF1(prediction, ref));
        }
        return maxF1;
    }

    @Override
    public boolean higherIsBetter() {
        return true;
    }

    static double computeF1(String prediction, String reference) {
        List<String> predTokens = tokenize(ExactMatchMetric.normalize(prediction));
        List<String> refTokens = tokenize(ExactMatchMetric.normalize(reference));

        if (predTokens.isEmpty() && refTokens.isEmpty()) return 1.0;
        if (predTokens.isEmpty() || refTokens.isEmpty()) return 0.0;

        // Count common tokens (with multiplicity)
        Map<String, Integer> predCounts = countTokens(predTokens);
        Map<String, Integer> refCounts = countTokens(refTokens);

        int common = 0;
        for (Map.Entry<String, Integer> entry : predCounts.entrySet()) {
            if (refCounts.containsKey(entry.getKey())) {
                common += Math.min(entry.getValue(), refCounts.get(entry.getKey()));
            }
        }

        if (common == 0) return 0.0;

        double precision = (double) common / predTokens.size();
        double recall = (double) common / refTokens.size();
        return 2.0 * precision * recall / (precision + recall);
    }

    static List<String> tokenize(String text) {
        if (text == null || text.isEmpty()) return Collections.emptyList();
        String[] parts = text.split("\\s+");
        List<String> tokens = new ArrayList<>();
        for (String part : parts) {
            if (!part.isEmpty()) {
                tokens.add(part);
            }
        }
        return tokens;
    }

    private static Map<String, Integer> countTokens(List<String> tokens) {
        Map<String, Integer> counts = new HashMap<>();
        for (String token : tokens) {
            counts.merge(token, 1, Integer::sum);
        }
        return counts;
    }
}
