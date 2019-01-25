package org.deeplearning4j.arbiter.optimize.generator.genetic;

public class Chromosome {
    public final double fitness;
    public final double[] genes;

    public Chromosome(double[] genes, double fitness) {
        this.genes = genes;
        this.fitness = fitness;
    }
}
