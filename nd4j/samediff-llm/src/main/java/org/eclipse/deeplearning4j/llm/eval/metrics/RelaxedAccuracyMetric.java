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

/**
 * Relaxed accuracy metric used in ChartQA evaluation.
 * A prediction is correct if:
 * 1. It exactly matches a reference (after normalization), OR
 * 2. Both prediction and reference are numeric and within a tolerance (default 5%)
 */
public class RelaxedAccuracyMetric implements EvalMetric {

    private final double tolerance;

    public RelaxedAccuracyMetric() {
        this(0.05);
    }

    public RelaxedAccuracyMetric(double tolerance) {
        this.tolerance = tolerance;
    }

    @Override
    public String name() {
        return "relaxed_accuracy";
    }

    @Override
    public double score(String prediction, List<String> references) {
        if (prediction == null || references == null || references.isEmpty()) return 0.0;
        String normalizedPred = ExactMatchMetric.normalize(prediction);
        for (String ref : references) {
            String normalizedRef = ExactMatchMetric.normalize(ref);
            // Exact string match
            if (normalizedPred.equals(normalizedRef)) {
                return 1.0;
            }
            // Numeric tolerance check
            Double predNum = tryParseNumber(prediction.trim());
            Double refNum = tryParseNumber(ref.trim());
            if (predNum != null && refNum != null) {
                if (refNum == 0.0) {
                    if (predNum == 0.0) return 1.0;
                } else {
                    double relError = Math.abs(predNum - refNum) / Math.abs(refNum);
                    if (relError <= tolerance) return 1.0;
                }
            }
        }
        return 0.0;
    }

    @Override
    public boolean higherIsBetter() {
        return true;
    }

    private static Double tryParseNumber(String s) {
        if (s == null || s.isEmpty()) return null;
        String cleaned = s.replaceAll("[,$%]", "").trim();
        try {
            return Double.parseDouble(cleaned);
        } catch (NumberFormatException e) {
            return null;
        }
    }
}
