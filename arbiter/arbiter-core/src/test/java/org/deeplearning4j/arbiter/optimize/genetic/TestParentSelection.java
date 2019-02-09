package org.deeplearning4j.arbiter.optimize.genetic;

import org.deeplearning4j.arbiter.optimize.generator.genetic.Chromosome;
import org.deeplearning4j.arbiter.optimize.generator.genetic.crossover.parentselection.TwoParentSelection;

import java.util.List;

public class TestParentSelection extends TwoParentSelection {

    public boolean hasBeenInitialized;

    private final double[][] parents;

    public TestParentSelection(double[][] parents) {
        this.parents = parents;
    }

    public TestParentSelection() {
        this(null);
    }

    @Override
    public void initializeInstance(List<Chromosome> population) {
        super.initializeInstance(population);
        hasBeenInitialized = true;
    }

    @Override
    public double[][] selectParents() {
        return parents;
    }

    public List<Chromosome> getPopulation() {
        return population;
    }
}
