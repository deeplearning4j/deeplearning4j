package org.deeplearning4j.arbiter.optimize.generator.genetic.crossover;

import org.deeplearning4j.arbiter.optimize.generator.genetic.population.PopulationModel;

public abstract class TwoParentsCrossoverOperator extends CrossoverOperator {

    protected final TwoParentSelection parentSelection;

    protected TwoParentsCrossoverOperator(TwoParentSelection parentSelection) {
        this.parentSelection = parentSelection;
    }

    @Override
    public void initializeInstance(PopulationModel populationModel) {
        super.initializeInstance(populationModel);
        parentSelection.initializeInstance(populationModel.population);
    }
}
