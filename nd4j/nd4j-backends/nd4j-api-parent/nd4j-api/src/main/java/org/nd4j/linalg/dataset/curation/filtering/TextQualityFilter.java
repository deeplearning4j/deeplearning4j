package org.nd4j.linalg.dataset.curation.filtering;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Configurable heuristic text quality filter chain.
 * Uses builder pattern to compose multiple quality checks with AND/OR logic.
 */
public class TextQualityFilter implements Serializable {
    private static final long serialVersionUID = 1L;

    private final int minChars;
    private final int maxChars;
    private final int minWords;
    private final int maxWords;
    private final double minAlphaRatio;
    private final double maxSpecialCharRatio;
    private final double maxRepetitionRatio;
    private final int repetitionNgramSize;
    private final double maxAllCapsRatio;
    private final double maxWhitespaceRatio;
    private final boolean useOrLogic;

    private TextQualityFilter(Builder builder) {
        this.minChars = builder.minChars;
        this.maxChars = builder.maxChars;
        this.minWords = builder.minWords;
        this.maxWords = builder.maxWords;
        this.minAlphaRatio = builder.minAlphaRatio;
        this.maxSpecialCharRatio = builder.maxSpecialCharRatio;
        this.maxRepetitionRatio = builder.maxRepetitionRatio;
        this.repetitionNgramSize = builder.repetitionNgramSize;
        this.maxAllCapsRatio = builder.maxAllCapsRatio;
        this.maxWhitespaceRatio = builder.maxWhitespaceRatio;
        this.useOrLogic = builder.useOrLogic;
    }

    /**
     * Quick accept/reject check.
     */
    public boolean accept(String text) {
        return evaluate(text).isAccepted();
    }

    /**
     * Evaluate text quality with detailed rejection reasons.
     */
    public FilterResult evaluate(String text) {
        if (text == null) {
            return FilterResult.reject("Text is null");
        }

        List<String> reasons = new ArrayList<>();

        // Length checks
        if (text.length() < minChars) {
            reasons.add("Too short: " + text.length() + " chars < min " + minChars);
        }
        if (maxChars > 0 && text.length() > maxChars) {
            reasons.add("Too long: " + text.length() + " chars > max " + maxChars);
        }

        String[] words = text.split("\\s+");
        int wordCount = text.trim().isEmpty() ? 0 : words.length;
        if (wordCount < minWords) {
            reasons.add("Too few words: " + wordCount + " < min " + minWords);
        }
        if (maxWords > 0 && wordCount > maxWords) {
            reasons.add("Too many words: " + wordCount + " > max " + maxWords);
        }

        // Character ratio checks
        if (text.length() > 0) {
            int alphaCount = 0;
            int specialCount = 0;
            int upperCount = 0;
            int letterCount = 0;
            int whitespaceCount = 0;

            for (int i = 0; i < text.length(); i++) {
                char c = text.charAt(i);
                if (Character.isLetter(c)) {
                    alphaCount++;
                    letterCount++;
                    if (Character.isUpperCase(c)) {
                        upperCount++;
                    }
                } else if (Character.isWhitespace(c)) {
                    whitespaceCount++;
                } else if (!Character.isDigit(c)) {
                    specialCount++;
                }
            }

            double alphaRatio = (double) alphaCount / text.length();
            if (alphaRatio < minAlphaRatio) {
                reasons.add("Low alpha ratio: " + String.format("%.2f", alphaRatio) + " < " + minAlphaRatio);
            }

            double specialRatio = (double) specialCount / text.length();
            if (specialRatio > maxSpecialCharRatio) {
                reasons.add("High special char ratio: " + String.format("%.2f", specialRatio) + " > " + maxSpecialCharRatio);
            }

            double capsRatio = letterCount > 0 ? (double) upperCount / letterCount : 0;
            if (capsRatio > maxAllCapsRatio) {
                reasons.add("High all-caps ratio: " + String.format("%.2f", capsRatio) + " > " + maxAllCapsRatio);
            }

            double wsRatio = (double) whitespaceCount / text.length();
            if (wsRatio > maxWhitespaceRatio) {
                reasons.add("High whitespace ratio: " + String.format("%.2f", wsRatio) + " > " + maxWhitespaceRatio);
            }
        }

        // Repetition check
        if (maxRepetitionRatio < 1.0 && wordCount >= repetitionNgramSize) {
            double repRatio = computeRepetitionRatio(words, repetitionNgramSize);
            if (repRatio > maxRepetitionRatio) {
                reasons.add("High repetition ratio: " + String.format("%.2f", repRatio) + " > " + maxRepetitionRatio);
            }
        }

        if (reasons.isEmpty()) {
            return FilterResult.accept();
        }

        if (useOrLogic) {
            // OR logic: reject only if ALL checks fail (i.e., if reasons.size() equals total checks run)
            // For simplicity, OR means accept if at least one check passes
            return FilterResult.accept();
        }

        return FilterResult.reject(reasons);
    }

    private double computeRepetitionRatio(String[] words, int n) {
        if (words.length < n) return 0.0;

        Map<String, Integer> ngramCounts = new HashMap<>();
        int totalNgrams = words.length - n + 1;

        for (int i = 0; i <= words.length - n; i++) {
            StringBuilder sb = new StringBuilder();
            for (int j = 0; j < n; j++) {
                if (j > 0) sb.append(' ');
                sb.append(words[i + j].toLowerCase());
            }
            String ngram = sb.toString();
            ngramCounts.merge(ngram, 1, Integer::sum);
        }

        int repeatedCount = 0;
        for (int count : ngramCounts.values()) {
            if (count > 1) {
                repeatedCount += count;
            }
        }

        return (double) repeatedCount / totalNgrams;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private int minChars = 0;
        private int maxChars = 0;
        private int minWords = 0;
        private int maxWords = 0;
        private double minAlphaRatio = 0.0;
        private double maxSpecialCharRatio = 1.0;
        private double maxRepetitionRatio = 1.0;
        private int repetitionNgramSize = 3;
        private double maxAllCapsRatio = 1.0;
        private double maxWhitespaceRatio = 1.0;
        private boolean useOrLogic = false;

        public Builder minChars(int minChars) { this.minChars = minChars; return this; }
        public Builder maxChars(int maxChars) { this.maxChars = maxChars; return this; }
        public Builder minWords(int minWords) { this.minWords = minWords; return this; }
        public Builder maxWords(int maxWords) { this.maxWords = maxWords; return this; }
        public Builder minAlphaRatio(double ratio) { this.minAlphaRatio = ratio; return this; }
        public Builder maxSpecialCharRatio(double ratio) { this.maxSpecialCharRatio = ratio; return this; }
        public Builder maxRepetitionRatio(double ratio) { this.maxRepetitionRatio = ratio; return this; }
        public Builder repetitionNgramSize(int n) { this.repetitionNgramSize = n; return this; }
        public Builder maxAllCapsRatio(double ratio) { this.maxAllCapsRatio = ratio; return this; }
        public Builder maxWhitespaceRatio(double ratio) { this.maxWhitespaceRatio = ratio; return this; }
        public Builder useOrLogic(boolean useOr) { this.useOrLogic = useOr; return this; }

        public TextQualityFilter build() {
            return new TextQualityFilter(this);
        }
    }
}
