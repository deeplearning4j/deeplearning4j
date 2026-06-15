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
 * Entity resolution strategy for PERSON entities.
 * Handles honorifics, suffixes, initials, first/last name matching,
 * and nickname/shortened forms.
 */
public class PersonResolutionStrategy implements EntityResolutionStrategy {

    private static final Pattern HONORIFICS = Pattern.compile(
            "^(mr\\.?|mrs\\.?|ms\\.?|dr\\.?|prof\\.?|sir|dame|lord|lady|rev\\.?|hon\\.?)\\s+",
            Pattern.CASE_INSENSITIVE);
    private static final Pattern SUFFIXES = Pattern.compile(
            "\\s+(jr\\.?|sr\\.?|ii|iii|iv|esq\\.?|ph\\.?d\\.?|md|m\\.d\\.)$",
            Pattern.CASE_INSENSITIVE);
    private static final Pattern WHITESPACE = Pattern.compile("\\s+");

    @Override
    public double computeSimilarity(String mentionA, String mentionB) {
        String normA = normalize(mentionA);
        String normB = normalize(mentionB);

        if (normA.equals(normB)) return 1.0;

        String[] partsA = WHITESPACE.split(normA);
        String[] partsB = WHITESPACE.split(normB);

        // Last name match with first initial
        if (partsA.length >= 2 && partsB.length >= 2) {
            String lastA = partsA[partsA.length - 1];
            String lastB = partsB[partsB.length - 1];
            if (lastA.equals(lastB)) {
                // Full first name vs initial: "john smith" vs "j smith"
                if (partsA[0].length() == 1 && partsB[0].startsWith(partsA[0])) return 0.85;
                if (partsB[0].length() == 1 && partsA[0].startsWith(partsB[0])) return 0.85;
                // Same last, different first
                double firstSim = normalizedLevenshtein(partsA[0], partsB[0]);
                return 0.5 + 0.4 * firstSim;
            }
        }

        // One mention is just a last name: "Obama" vs "Barack Obama"
        if (partsA.length == 1 && partsB.length >= 2) {
            if (partsB[partsB.length - 1].equals(partsA[0])) return 0.7;
        }
        if (partsB.length == 1 && partsA.length >= 2) {
            if (partsA[partsA.length - 1].equals(partsB[0])) return 0.7;
        }

        // Levenshtein fallback on full normalized string
        double lev = normalizedLevenshtein(normA, normB);
        return lev * 0.8; // discount: string similarity alone is weaker for names
    }

    @Override
    public String normalize(String mention) {
        String s = mention.trim().toLowerCase();
        s = HONORIFICS.matcher(s).replaceFirst("");
        s = SUFFIXES.matcher(s).replaceFirst("");
        s = s.replaceAll("[^a-z\\s]", "").trim();
        s = WHITESPACE.matcher(s).replaceAll(" ");
        return s;
    }

    @Override
    public String entityType() {
        return "PERSON";
    }

    @Override
    public double defaultThreshold() {
        return 0.7;
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
