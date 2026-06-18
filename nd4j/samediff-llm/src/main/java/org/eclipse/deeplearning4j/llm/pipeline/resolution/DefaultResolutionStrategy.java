/*
 *  ******************************************************************************
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

package org.eclipse.deeplearning4j.llm.pipeline.resolution;

import java.util.regex.Pattern;

/**
 * Default entity resolution strategy for entity types without a specialized strategy.
 * Uses case-insensitive normalized Levenshtein similarity.
 */
public class DefaultResolutionStrategy implements EntityResolutionStrategy {

    private static final Pattern WHITESPACE = Pattern.compile("\\s+");
    private final String type;

    public DefaultResolutionStrategy(String type) {
        this.type = type;
    }

    @Override
    public double computeSimilarity(String mentionA, String mentionB) {
        String normA = normalize(mentionA);
        String normB = normalize(mentionB);
        if (normA.equals(normB)) return 1.0;

        int maxLen = Math.max(normA.length(), normB.length());
        if (maxLen == 0) return 1.0;
        int dist = levenshteinDistance(normA, normB);
        return 1.0 - (double) dist / maxLen;
    }

    @Override
    public String normalize(String mention) {
        String s = mention.trim().toLowerCase();
        s = WHITESPACE.matcher(s).replaceAll(" ");
        return s;
    }

    @Override
    public String entityType() {
        return type;
    }

    @Override
    public double defaultThreshold() {
        return 0.8;
    }

    private static int levenshteinDistance(String s1, String s2) {
        int m = s1.length(), n = s2.length();
        if (m < n) { String t = s1; s1 = s2; s2 = t; int tmp = m; m = n; n = tmp; }
        int[] prev = new int[n + 1];
        int[] curr = new int[n + 1];
        for (int j = 0; j <= n; j++) prev[j] = j;
        for (int i = 1; i <= m; i++) {
            curr[0] = i;
            for (int j = 1; j <= n; j++) {
                int cost = s1.charAt(i - 1) == s2.charAt(j - 1) ? 0 : 1;
                curr[j] = Math.min(Math.min(curr[j - 1] + 1, prev[j] + 1), prev[j - 1] + cost);
            }
            int[] tmp = prev; prev = curr; curr = tmp;
        }
        return prev[n];
    }
}
