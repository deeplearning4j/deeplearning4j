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

import java.util.List;
import java.util.regex.Pattern;

/**
 * Exact match metric with optional normalization.
 * Supports lowercase normalization, punctuation stripping, and article removal
 * (following SQuAD evaluation conventions).
 */
public class ExactMatchMetric implements EvalMetric {

    private static final Pattern PUNCTUATION = Pattern.compile("[^\\w\\s]");
    private static final Pattern ARTICLES = Pattern.compile("\\b(a|an|the)\\b");
    private static final Pattern WHITESPACE = Pattern.compile("\\s+");

    private final boolean normalize;

    public ExactMatchMetric() {
        this(true);
    }

    public ExactMatchMetric(boolean normalize) {
        this.normalize = normalize;
    }

    @Override
    public String name() {
        return "exact_match";
    }

    @Override
    public double score(String prediction, List<String> references) {
        if (prediction == null || references == null || references.isEmpty()) return 0.0;
        String normalizedPred = normalize ? normalize(prediction) : prediction;
        for (String ref : references) {
            String normalizedRef = normalize ? normalize(ref) : ref;
            if (normalizedPred.equals(normalizedRef)) {
                return 1.0;
            }
        }
        return 0.0;
    }

    @Override
    public boolean higherIsBetter() {
        return true;
    }

    /**
     * Normalize text following SQuAD conventions:
     * lowercase, remove punctuation, remove articles, collapse whitespace.
     */
    public static String normalize(String text) {
        if (text == null) return "";
        String result = text.toLowerCase();
        result = PUNCTUATION.matcher(result).replaceAll("");
        result = ARTICLES.matcher(result).replaceAll("");
        result = WHITESPACE.matcher(result).replaceAll(" ");
        return result.trim();
    }
}
