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

package org.eclipse.deeplearning4j.llm.eval;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Reusable answer extraction utilities for parsing model output.
 */
public class AnswerExtractor {

    private static final Pattern MC_PATTERN = Pattern.compile(
            "(?i)(?:the\\s+)?answer\\s+is\\s*[:\\s]*\\(?([A-Ea-e])\\)?");
    private static final Pattern MC_START_PATTERN = Pattern.compile(
            "^\\(?([A-Ea-e])[.)\\s]");
    private static final Pattern MC_SINGLE_PATTERN = Pattern.compile(
            "\\b([A-E])\\b");
    private static final Pattern NUMBER_PATTERN = Pattern.compile(
            "####\\s*([\\-+]?[\\d,]+\\.?\\d*)");
    private static final Pattern LAST_NUMBER_PATTERN = Pattern.compile(
            "([\\-+]?[\\d,]+\\.?\\d*)\\s*$");
    private static final Pattern YES_NO_PATTERN = Pattern.compile(
            "(?i)\\b(yes|no)\\b");

    private AnswerExtractor() {}

    public static String extractMultipleChoice(String output) {
        if (output == null || output.isEmpty()) return null;
        String trimmed = output.trim();

        if (trimmed.length() == 1 && Character.isLetter(trimmed.charAt(0))) {
            char c = Character.toUpperCase(trimmed.charAt(0));
            if (c >= 'A' && c <= 'E') return String.valueOf(c);
        }

        Matcher m = MC_PATTERN.matcher(trimmed);
        if (m.find()) return m.group(1).toUpperCase();

        m = MC_START_PATTERN.matcher(trimmed);
        if (m.find()) return m.group(1).toUpperCase();

        m = MC_SINGLE_PATTERN.matcher(trimmed);
        if (m.find()) return m.group(1);

        return null;
    }

    public static String extractNumber(String output) {
        if (output == null || output.isEmpty()) return null;

        Matcher m = NUMBER_PATTERN.matcher(output);
        if (m.find()) {
            return m.group(1).replace(",", "");
        }

        m = LAST_NUMBER_PATTERN.matcher(output.trim());
        if (m.find()) {
            return m.group(1).replace(",", "");
        }

        return null;
    }

    public static String normalizeAnswer(String answer) {
        return org.eclipse.deeplearning4j.llm.eval.metrics.ExactMatchMetric.normalize(answer);
    }

    public static String extractYesNo(String output) {
        if (output == null || output.isEmpty()) return null;
        Matcher m = YES_NO_PATTERN.matcher(output.trim());
        if (m.find()) return m.group(1).toLowerCase();
        return null;
    }
}
