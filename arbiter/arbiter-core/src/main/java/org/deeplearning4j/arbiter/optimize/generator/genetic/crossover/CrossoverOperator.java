package org.deeplearning4j.arbiter.optimize.generator.genetic.crossover;

import org.deeplearning4j.arbiter.optimize.generator.genetic.population.PopulationModel;

public abstract class CrossoverOperator {
    protected PopulationModel populationModel;

    public void initializeInstance(PopulationModel populationModel) {
        this.populationModel = populationModel;
    }
    public abstract CrossoverResult crossover();



}
