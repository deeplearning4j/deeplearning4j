package org.deeplearning4j.arbiter.optimize.genetic.crossover;

import org.deeplearning4j.arbiter.optimize.generator.genetic.population.PopulationInitializer;
import org.deeplearning4j.arbiter.optimize.generator.genetic.population.PopulationModel;
import org.deeplearning4j.arbiter.optimize.genetic.TestCrossoverOperator;
import org.deeplearning4j.arbiter.optimize.genetic.TestPopulationInitializer;
import org.junit.Assert;
import org.junit.Test;

public class CrossoverOperatorTests {

    @Test
    public void CrossoverOperator_initializeInstance_ShouldInitPopulationModel() throws IllegalAccessException {
        TestCrossoverOperator sut = new TestCrossoverOperator(null);

        PopulationInitializer populationInitializer = new TestPopulationInitializer();

        PopulationModel populationModel =
                        new PopulationModel.Builder().populationInitializer(populationInitializer).build();
        sut.initializeInstance(populationModel);

        Assert.assertSame(populationModel, sut.getPopulationModel());


    }
}
