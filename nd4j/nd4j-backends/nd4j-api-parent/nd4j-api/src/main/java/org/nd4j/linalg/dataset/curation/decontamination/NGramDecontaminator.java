package org.nd4j.linalg.dataset.curation.decontamination;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Removes training samples that overlap with benchmark/test data using n-gram matching.
 * Uses the GPT-3 standard n-gram size of 13 by default.
 */
public class NGramDecontaminator implements Serializable {
    private static final long serialVersionUID = 1L;

    private final int ngramSize;
    private final boolean characterLevel;

    // Maps n-gram -> set of benchmark indices that contain it
    private final Map<String, Set<Integer>> benchmarkIndex = new HashMap<>();

    public NGramDecontaminator(int ngramSize, boolean characterLevel) {
        this.ngramSize = ngramSize;
        this.characterLevel = characterLevel;
    }

    public NGramDecontaminator(int ngramSize) {
        this(ngramSize, false);
    }

    public NGramDecontaminator() {
        this(13, false);
    }

    /**
     * Index benchmark data for later checking.
     */
    public void indexBenchmark(List<String> benchmarkTexts) {
        for (int i = 0; i < benchmarkTexts.size(); i++) {
            Set<String> ngrams = extractNgrams(benchmarkTexts.get(i));
            for (String ngram : ngrams) {
                benchmarkIndex.computeIfAbsent(ngram, k -> new HashSet<>()).add(i);
            }
        }
    }

    /**
     * Check training texts for contamination against indexed benchmarks.
     */
    public DecontaminationResult check(List<String> trainingTexts) {
        Set<Integer> contaminated = new HashSet<>();
        Map<Integer, List<Integer>> matches = new HashMap<>();

        for (int i = 0; i < trainingTexts.size(); i++) {
            Set<String> ngrams = extractNgrams(trainingTexts.get(i));
            Set<Integer> matchedBenchmarks = new HashSet<>();

            for (String ngram : ngrams) {
                Set<Integer> benchmarkHits = benchmarkIndex.get(ngram);
                if (benchmarkHits != null) {
                    matchedBenchmarks.addAll(benchmarkHits);
                }
            }

            if (!matchedBenchmarks.isEmpty()) {
                contaminated.add(i);
                matches.put(i, new ArrayList<>(matchedBenchmarks));
            }
        }

        return new DecontaminationResult(contaminated, matches, trainingTexts.size());
    }

    /**
     * Remove contaminated samples from training data.
     */
    public List<String> decontaminate(List<String> trainingTexts) {
        DecontaminationResult result = check(trainingTexts);
        List<String> clean = new ArrayList<>();
        for (int i = 0; i < trainingTexts.size(); i++) {
            if (!result.getContaminatedIndices().contains(i)) {
                clean.add(trainingTexts.get(i));
            }
        }
        return clean;
    }

    /**
     * Reset the benchmark index.
     */
    public void reset() {
        benchmarkIndex.clear();
    }

    private Set<String> extractNgrams(String text) {
        Set<String> ngrams = new HashSet<>();
        String normalized = text.toLowerCase().replaceAll("\\s+", " ").trim();

        if (characterLevel) {
            for (int i = 0; i <= normalized.length() - ngramSize; i++) {
                ngrams.add(normalized.substring(i, i + ngramSize));
            }
        } else {
            String[] words = normalized.split("\\s+");
            for (int i = 0; i <= words.length - ngramSize; i++) {
                StringBuilder sb = new StringBuilder();
                for (int j = 0; j < ngramSize; j++) {
                    if (j > 0) sb.append(' ');
                    sb.append(words[i + j]);
                }
                ngrams.add(sb.toString());
            }
        }

        return ngrams;
    }
}
