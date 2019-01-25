package org.deeplearning4j.arbiter.optimize.generator.genetic.crossover;

import org.deeplearning4j.arbiter.optimize.generator.genetic.Chromosome;

import java.util.List;

public abstract class ParentSelection {
    protected List<Chromosome> population;

    public void initializeInstance(List<Chromosome> population) {
        this.population = population;
    }

    public abstract double[][] selectParents();
}
