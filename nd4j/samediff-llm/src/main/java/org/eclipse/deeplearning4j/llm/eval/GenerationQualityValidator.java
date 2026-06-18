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

import lombok.Builder;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.eclipse.deeplearning4j.llm.generation.GenerationResult;

import java.util.*;

/**
 * Generation quality validation for LLM outputs.
 *
 * Checks token diversity, repetition, coherence, and expected content.
 * Parallel to VLM token diversity validation in TestSmolDoclingOptimizedPipeline.
 */
@Slf4j
public class GenerationQualityValidator {

    @Data
    @Builder
    public static class QualityReport {
        private final double diversityRatio;
        private final double repetitionScore;
        private final double coherenceScore;
        private final boolean eosReached;
        private final boolean passed;
        private final List<String> issues;

        public String summary() {
            return String.format("diversity=%.2f, repetition=%.2f, coherence=%.2f, eos=%s, passed=%s%s",
                    diversityRatio, repetitionScore, coherenceScore, eosReached, passed,
                    issues.isEmpty() ? "" : " issues=" + issues);
        }
    }

    @Data
    @Builder
    public static class ValidationConfig {
        @Builder.Default private double minDiversityRatio = 0.3;
        @Builder.Default private double maxRepetitionScore = 0.5;
        @Builder.Default private double minCoherenceScore = 0.3;
        @Builder.Default private boolean requireEos = false;
        @Builder.Default private List<String> expectedSubstrings = Collections.emptyList();
        @Builder.Default private List<String> forbiddenSubstrings = Collections.emptyList();
    }

    /**
     * Validate generation result quality with default config.
     */
    public static QualityReport validate(GenerationResult result) {
        return validate(result, ValidationConfig.builder().build());
    }

    /**
     * Validate generation result quality.
     */
    public static QualityReport validate(GenerationResult result, ValidationConfig config) {
        List<String> issues = new ArrayList<>();
        String text = result.getText();
        int[] tokenIds = result.getTokenIds();

        // Token diversity: unique tokens / total tokens
        double diversityRatio = computeDiversityRatio(tokenIds);
        if (diversityRatio < config.getMinDiversityRatio()) {
            issues.add(String.format("Low diversity: %.2f < %.2f", diversityRatio, config.getMinDiversityRatio()));
        }

        // Repetition: detect repeated n-grams
        double repetitionScore = computeRepetitionScore(tokenIds);
        if (repetitionScore > config.getMaxRepetitionScore()) {
            issues.add(String.format("High repetition: %.2f > %.2f", repetitionScore, config.getMaxRepetitionScore()));
        }

        // Coherence: check for garbage output patterns
        double coherenceScore = computeCoherenceScore(text);
        if (coherenceScore < config.getMinCoherenceScore()) {
            issues.add(String.format("Low coherence: %.2f < %.2f", coherenceScore, config.getMinCoherenceScore()));
        }

        // EOS check
        boolean eosReached = result.getFinishReason() == GenerationResult.FinishReason.EOS;
        if (config.isRequireEos() && !eosReached) {
            issues.add("EOS not reached");
        }

        // Expected substrings
        for (String expected : config.getExpectedSubstrings()) {
            if (!text.toLowerCase().contains(expected.toLowerCase())) {
                issues.add("Missing expected: '" + expected + "'");
            }
        }

        // Forbidden substrings
        for (String forbidden : config.getForbiddenSubstrings()) {
            if (text.toLowerCase().contains(forbidden.toLowerCase())) {
                issues.add("Contains forbidden: '" + forbidden + "'");
            }
        }

        boolean passed = issues.isEmpty();

        QualityReport report = QualityReport.builder()
                .diversityRatio(diversityRatio)
                .repetitionScore(repetitionScore)
                .coherenceScore(coherenceScore)
                .eosReached(eosReached)
                .passed(passed)
                .issues(issues)
                .build();

        if (passed) {
            log.info("Quality check PASSED: {}", report.summary());
        } else {
            log.warn("Quality check FAILED: {}", report.summary());
        }

        return report;
    }

    /**
     * Validate with expected content check (convenience method for factual tests).
     */
    public static QualityReport validateWithExpected(GenerationResult result, String... expectedSubstrings) {
        return validate(result, ValidationConfig.builder()
                .expectedSubstrings(Arrays.asList(expectedSubstrings))
                .build());
    }

    /**
     * Validate with minimum diversity (convenience method for creative tests).
     */
    public static QualityReport validateDiversity(GenerationResult result, double minDiversity) {
        return validate(result, ValidationConfig.builder()
                .minDiversityRatio(minDiversity)
                .build());
    }

    // ==================== Metrics Computation ====================

    static double computeDiversityRatio(int[] tokenIds) {
        if (tokenIds == null || tokenIds.length == 0) return 0.0;
        Set<Integer> unique = new HashSet<>();
        for (int id : tokenIds) {
            unique.add(id);
        }
        return (double) unique.size() / tokenIds.length;
    }

    static double computeRepetitionScore(int[] tokenIds) {
        if (tokenIds == null || tokenIds.length < 4) return 0.0;

        // Check for repeated bigrams
        Map<Long, Integer> bigramCounts = new HashMap<>();
        for (int i = 0; i < tokenIds.length - 1; i++) {
            long bigram = ((long) tokenIds[i] << 32) | (tokenIds[i + 1] & 0xFFFFFFFFL);
            bigramCounts.merge(bigram, 1, Integer::sum);
        }

        int totalBigrams = tokenIds.length - 1;
        int repeatedBigrams = 0;
        for (int count : bigramCounts.values()) {
            if (count > 1) {
                repeatedBigrams += count - 1;
            }
        }

        return totalBigrams > 0 ? (double) repeatedBigrams / totalBigrams : 0.0;
    }

    static double computeCoherenceScore(String text) {
        if (text == null || text.isEmpty()) return 0.0;

        double score = 1.0;

        // Penalize very short output
        if (text.length() < 5) {
            score *= 0.3;
        }

        // Penalize high ratio of special/control characters
        long specialChars = text.chars().filter(c -> c < 32 && c != '\n' && c != '\t').count();
        double specialRatio = (double) specialChars / text.length();
        if (specialRatio > 0.1) {
            score *= (1.0 - specialRatio);
        }

        // Penalize long runs of the same character
        int maxRun = 1;
        int currentRun = 1;
        for (int i = 1; i < text.length(); i++) {
            if (text.charAt(i) == text.charAt(i - 1)) {
                currentRun++;
                maxRun = Math.max(maxRun, currentRun);
            } else {
                currentRun = 1;
            }
        }
        if (maxRun > 10) {
            score *= 0.5;
        }

        // Penalize if mostly digits/punctuation (likely garbage)
        long alphaCount = text.chars().filter(Character::isLetterOrDigit).count();
        double alphaRatio = (double) alphaCount / text.length();
        if (alphaRatio < 0.3) {
            score *= 0.5;
        }

        // Script mixing detection: count distinct Unicode scripts and transitions.
        // Mixed-script gibberish (e.g. Latin+CJK+Arabic+Cyrillic all in one output)
        // is a strong signal of garbage/random output.
        Set<Character.UnicodeScript> distinctScripts = new LinkedHashSet<>();
        int scriptTransitions = 0;
        Character.UnicodeScript prevScript = null;

        // Approximate token count as whitespace-separated words for threshold scaling.
        String[] words = text.trim().split("\\s+");
        int tokenCount = words.length;

        int[] codePoints = text.codePoints().toArray();
        for (int cp : codePoints) {
            Character.UnicodeScript script = Character.UnicodeScript.of(cp);
            // Skip COMMON (punctuation, digits, symbols) and INHERITED (combining marks).
            if (script == Character.UnicodeScript.COMMON || script == Character.UnicodeScript.INHERITED) {
                continue;
            }
            distinctScripts.add(script);
            if (prevScript != null && script != prevScript) {
                scriptTransitions++;
            }
            prevScript = script;
        }

        int numDistinctScripts = distinctScripts.size();

        // More than 3 distinct scripts in ~20 tokens is a garbage signal.
        // Scale the threshold linearly with token count (floor at 20 tokens).
        int effectiveTokens = Math.max(tokenCount, 1);
        double scriptThreshold = 3.0 + (effectiveTokens - 20) * (1.0 / 20.0);
        // Always allow at least 3 scripts; cap the leniency at 5 for very long text.
        scriptThreshold = Math.max(3.0, Math.min(scriptThreshold, 5.0));

        if (numDistinctScripts > scriptThreshold) {
            score *= 0.2;
        }

        // Word-boundary chaos: high transition rate relative to token count = garbage.
        // More than 5 script transitions per 20 tokens (i.e. > 0.3 * tokenCount) is suspicious.
        // Use 0.3 * tokenCount so that 9 transitions / 17 tokens (ratio ~0.53) clearly exceeds it.
        double transitionRateThreshold = Math.max(5.0, effectiveTokens * 0.3);
        if (scriptTransitions > transitionRateThreshold) {
            // Use 0.25 so that even with no other penalties, score drops to 0.25 < minCoherenceScore=0.3
            score *= 0.25;
        }

        return Math.max(0.0, Math.min(1.0, score));
    }

    // ==================== Reference Test Prompts ====================

    public static final Map<String, String[]> REFERENCE_PROMPTS = new LinkedHashMap<>();
    static {
        REFERENCE_PROMPTS.put("What is the capital of France?", new String[]{"Paris"});
        REFERENCE_PROMPTS.put("Write a short poem about the sea", new String[]{});
        REFERENCE_PROMPTS.put("Explain photosynthesis in one sentence", new String[]{"sun", "light", "plant", "energy"});
    }
}
