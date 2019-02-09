package org.deeplearning4j.arbiter.optimize.genetic;

import org.deeplearning4j.arbiter.optimize.generator.genetic.crossover.CrossoverOperator;
import org.deeplearning4j.arbiter.optimize.generator.genetic.crossover.CrossoverResult;
import org.deeplearning4j.arbiter.optimize.generator.genetic.population.PopulationModel;

public class TestCrossoverOperator extends CrossoverOperator {

    private final CrossoverResult[] results;
    private int resultIdx = 0;

    public PopulationModel getPopulationModel() {
        return populationModel;
    }

    public TestCrossoverOperator(CrossoverResult[] results) {
        this.results = results;
    }

    @Override
    public CrossoverResult crossover() {
        return results[resultIdx++];
    }
}
