package org.deeplearning4j.arbiter.optimize.generator.genetic;

public class ChromosomeFactory {
    private int chromosomeLength;

    public void initializeInstance(int chromosomeLength) {
        this.chromosomeLength = chromosomeLength;
    }

    public Chromosome createChromosome(double[] genes, double fitness) {
        return new Chromosome(genes, fitness);
    }

    public int getChromosomeLength() {
        return chromosomeLength;
    }
}
