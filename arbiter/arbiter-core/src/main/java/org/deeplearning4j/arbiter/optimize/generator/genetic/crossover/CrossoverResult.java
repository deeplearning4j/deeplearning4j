package org.deeplearning4j.arbiter.optimize.generator.genetic.crossover;

public class CrossoverResult {
    public final boolean hasModification;
    public final double[] genes;

    public CrossoverResult(boolean hasModification, double[] genes) {
        this.hasModification = hasModification;
        this.genes = genes;
    }
}