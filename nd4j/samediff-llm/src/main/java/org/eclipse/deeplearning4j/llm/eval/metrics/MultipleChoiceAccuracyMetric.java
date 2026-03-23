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
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Multiple choice accuracy metric for A/B/C/D selection tasks.
 * Extracts the choice letter from the model output and compares to the reference.
 */
public class MultipleChoiceAccuracyMetric implements EvalMetric {

    private static final Pattern ANSWER_IS = Pattern.compile(
            "(?i)(?:the\\s+)?answer\\s+is\\s*[:\\s]*\\(?([A-Ea-e])\\)?");
    private static final Pattern STARTS_WITH = Pattern.compile(
            "^\\(?([A-Ea-e])[.)\\s]");
    private static final Pattern STANDALONE = Pattern.compile(
            "\\b([A-E])\\b");

    @Override
    public String name() {
        return "mc_accuracy";
    }

    @Override
    public double score(String prediction, List<String> references) {
        if (prediction == null || references == null || references.isEmpty()) return 0.0;
        String extractedChoice = extractChoice(prediction);
        if (extractedChoice == null) return 0.0;
        for (String ref : references) {
            if (extractedChoice.equalsIgnoreCase(ref.trim())) {
                return 1.0;
            }
        }
        return 0.0;
    }

    @Override
    public boolean higherIsBetter() {
        return true;
    }

    static String extractChoice(String text) {
        if (text == null || text.isEmpty()) return null;
        String trimmed = text.trim();

        // Single letter answer
        if (trimmed.length() == 1 && Character.isLetter(trimmed.charAt(0))) {
            return trimmed.toUpperCase();
        }

        // "The answer is (X)"
        Matcher m = ANSWER_IS.matcher(trimmed);
        if (m.find()) return m.group(1).toUpperCase();

        // Starts with "X." or "X)" or "(X)"
        m = STARTS_WITH.matcher(trimmed);
        if (m.find()) return m.group(1).toUpperCase();

        // First standalone A-E
        m = STANDALONE.matcher(trimmed);
        if (m.find()) return m.group(1);

        return null;
    }
}
