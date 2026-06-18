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

import java.util.*;
import java.util.regex.Pattern;

/**
 * Entity resolution strategy for LOCATION entities.
 * Handles common location abbreviations (US/USA/United States),
 * directional prefixes, and containment relationships.
 */
public class LocationResolutionStrategy implements EntityResolutionStrategy {

    private static final Pattern WHITESPACE = Pattern.compile("\\s+");

    // Common location abbreviation mappings
    private static final Map<String, String> ABBREVIATIONS = new HashMap<>();
    static {
        ABBREVIATIONS.put("us", "united states");
        ABBREVIATIONS.put("usa", "united states");
        ABBREVIATIONS.put("u.s.", "united states");
        ABBREVIATIONS.put("u.s.a.", "united states");
        ABBREVIATIONS.put("uk", "united kingdom");
        ABBREVIATIONS.put("u.k.", "united kingdom");
        ABBREVIATIONS.put("uae", "united arab emirates");
        ABBREVIATIONS.put("nyc", "new york city");
        ABBREVIATIONS.put("ny", "new york");
        ABBREVIATIONS.put("la", "los angeles");
        ABBREVIATIONS.put("sf", "san francisco");
        ABBREVIATIONS.put("dc", "washington dc");
        ABBREVIATIONS.put("d.c.", "washington dc");
    }

    @Override
    public double computeSimilarity(String mentionA, String mentionB) {
        String normA = normalize(mentionA);
        String normB = normalize(mentionB);

        if (normA.equals(normB)) return 1.0;

        // Expand abbreviations and re-check
        String expA = expandAbbreviation(normA);
        String expB = expandAbbreviation(normB);
        if (expA.equals(expB)) return 0.95;

        // Containment: "New York" vs "New York City"
        if (expA.contains(expB) || expB.contains(expA)) {
            double ratio = (double) Math.min(expA.length(), expB.length())
                    / Math.max(expA.length(), expB.length());
            return 0.6 + 0.35 * ratio;
        }

        // Levenshtein
        double lev = normalizedLevenshtein(expA, expB);
        return lev * 0.85;
    }

    @Override
    public String normalize(String mention) {
        String s = mention.trim().toLowerCase();
        s = s.replaceAll("[^a-z0-9.\\s]", "").trim();
        s = WHITESPACE.matcher(s).replaceAll(" ");
        return s;
    }

    @Override
    public String entityType() {
        return "LOCATION";
    }

    @Override
    public double defaultThreshold() {
        return 0.75;
    }

    private String expandAbbreviation(String normalized) {
        String expanded = ABBREVIATIONS.get(normalized);
        return expanded != null ? expanded : normalized;
    }

    private static double normalizedLevenshtein(String a, String b) {
        if (a.equals(b)) return 1.0;
        int maxLen = Math.max(a.length(), b.length());
        if (maxLen == 0) return 1.0;
        int dist = levenshteinDistance(a, b);
        return 1.0 - (double) dist / maxLen;
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
