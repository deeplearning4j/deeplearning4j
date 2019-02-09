package org.deeplearning4j.arbiter.optimize.genetic.crossover;

import org.deeplearning4j.arbiter.optimize.generator.genetic.crossover.CrossoverResult;
import org.deeplearning4j.arbiter.optimize.generator.genetic.crossover.TwoParentsCrossoverOperator;
import org.deeplearning4j.arbiter.optimize.generator.genetic.crossover.parentselection.TwoParentSelection;
import org.deeplearning4j.arbiter.optimize.generator.genetic.population.PopulationInitializer;
import org.deeplearning4j.arbiter.optimize.generator.genetic.population.PopulationModel;
import org.deeplearning4j.arbiter.optimize.genetic.TestParentSelection;
import org.deeplearning4j.arbiter.optimize.genetic.TestPopulationInitializer;
import org.junit.Assert;
import org.junit.Test;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

public class TwoParentsCrossoverOperatorTests {

    class TestTwoParentsCrossoverOperator extends TwoParentsCrossoverOperator {

        public TestTwoParentsCrossoverOperator(TwoParentSelection parentSelection) {
            super(parentSelection);
        }

        public TwoParentSelection getParentSelection() {
            return parentSelection;
        }

        @Override
        public CrossoverResult crossover() {
            throw new NotImplementedException();
        }
    }

    @Test
    public void TwoParentsCrossoverOperator_ctor_ShouldInitParentSelection() {
        TestParentSelection parentSelection = new TestParentSelection();
        TestTwoParentsCrossoverOperator sut = new TestTwoParentsCrossoverOperator(parentSelection);

        Assert.assertSame(parentSelection, sut.getParentSelection());
    }

    @Test
    public void TwoParentsCrossoverOperator_initializeInstanceShouldInitializeParentSelection() {
        TestParentSelection parentSelection = new TestParentSelection();
        TestTwoParentsCrossoverOperator sut = new TestTwoParentsCrossoverOperator(parentSelection);

        PopulationInitializer populationInitializer = new TestPopulationInitializer();
        PopulationModel populationModel =
                        new PopulationModel.Builder().populationInitializer(populationInitializer).build();

        sut.initializeInstance(populationModel);

        Assert.assertTrue(parentSelection.hasBeenInitialized);
    }

}
