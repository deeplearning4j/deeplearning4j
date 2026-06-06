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
 * Entity resolution strategy for ORGANIZATION entities.
 * Handles legal suffixes (Inc., Corp., Ltd.), abbreviations,
 * acronyms, and "the" prefix stripping.
 */
public class OrganizationResolutionStrategy implements EntityResolutionStrategy {

    private static final Pattern LEGAL_SUFFIX = Pattern.compile(
            "\\s+(inc\\.?|corp\\.?|co\\.?|ltd\\.?|llc\\.?|llp\\.?|plc\\.?|gmbh"
                    + "|ag|sa|s\\.?a\\.?|n\\.?v\\.?|pty\\.?|pvt\\.?)$",
            Pattern.CASE_INSENSITIVE);
    private static final Pattern THE_PREFIX = Pattern.compile(
            "^the\\s+", Pattern.CASE_INSENSITIVE);
    private static final Pattern WHITESPACE = Pattern.compile("\\s+");

    @Override
    public double computeSimilarity(String mentionA, String mentionB) {
        String normA = normalize(mentionA);
        String normB = normalize(mentionB);

        if (normA.equals(normB)) return 1.0;

        // Acronym check: "IBM" vs "international business machines"
        if (isAcronymOf(normA, normB)) return 0.9;
        if (isAcronymOf(normB, normA)) return 0.9;

        // Containment: "Apple" vs "Apple Computer"
        if (normA.contains(normB) || normB.contains(normA)) {
            double ratio = (double) Math.min(normA.length(), normB.length())
                    / Math.max(normA.length(), normB.length());
            return 0.6 + 0.3 * ratio;
        }

        // Levenshtein
        double lev = normalizedLevenshtein(normA, normB);
        return lev * 0.85;
    }

    @Override
    public String normalize(String mention) {
        String s = mention.trim().toLowerCase();
        s = THE_PREFIX.matcher(s).replaceFirst("");
        s = LEGAL_SUFFIX.matcher(s).replaceFirst("");
        s = s.replaceAll("[^a-z0-9\\s]", "").trim();
        s = WHITESPACE.matcher(s).replaceAll(" ");
        return s;
    }

    @Override
    public String entityType() {
        return "ORGANIZATION";
    }

    @Override
    public double defaultThreshold() {
        return 0.75;
    }

    private boolean isAcronymOf(String potential, String full) {
        if (potential.length() < 2) return false;
        String[] words = WHITESPACE.split(full);
        if (words.length < 2 || words.length != potential.length()) return false;
        for (int i = 0; i < words.length; i++) {
            if (words[i].isEmpty() || words[i].charAt(0) != potential.charAt(i)) return false;
        }
        return true;
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
