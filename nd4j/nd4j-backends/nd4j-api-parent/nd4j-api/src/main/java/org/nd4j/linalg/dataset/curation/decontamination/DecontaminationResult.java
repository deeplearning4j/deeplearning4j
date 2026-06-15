package org.nd4j.linalg.dataset.curation.decontamination;

import java.io.Serializable;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Result of decontamination analysis.
 */
public class DecontaminationResult implements Serializable {
    private static final long serialVersionUID = 1L;

    private final Set<Integer> contaminatedIndices;
    private final Map<Integer, List<Integer>> matchedBenchmarkIndices;
    private final int totalChecked;

    public DecontaminationResult(Set<Integer> contaminatedIndices,
                                  Map<Integer, List<Integer>> matchedBenchmarkIndices,
                                  int totalChecked) {
        this.contaminatedIndices = contaminatedIndices;
        this.matchedBenchmarkIndices = matchedBenchmarkIndices;
        this.totalChecked = totalChecked;
    }

    public Set<Integer> getContaminatedIndices() {
        return contaminatedIndices;
    }

    public Map<Integer, List<Integer>> getMatchedBenchmarkIndices() {
        return matchedBenchmarkIndices;
    }

    public int getTotalChecked() {
        return totalChecked;
    }

    public int getContaminatedCount() {
        return contaminatedIndices.size();
    }

    public double getContaminationRate() {
        return totalChecked > 0 ? (double) contaminatedIndices.size() / totalChecked : 0.0;
    }

    @Override
    public String toString() {
        return "DecontaminationResult{contaminated=" + contaminatedIndices.size() +
               "/" + totalChecked + " (" + String.format("%.1f%%", getContaminationRate() * 100) + ")}";
    }
}
