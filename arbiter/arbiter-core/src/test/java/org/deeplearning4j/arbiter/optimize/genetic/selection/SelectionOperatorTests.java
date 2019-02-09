package org.deeplearning4j.arbiter.optimize.genetic.selection;

import org.deeplearning4j.arbiter.optimize.generator.genetic.ChromosomeFactory;
import org.deeplearning4j.arbiter.optimize.generator.genetic.population.PopulationInitializer;
import org.deeplearning4j.arbiter.optimize.generator.genetic.population.PopulationModel;
import org.deeplearning4j.arbiter.optimize.generator.genetic.selection.SelectionOperator;
import org.deeplearning4j.arbiter.optimize.genetic.TestPopulationInitializer;
import org.junit.Assert;
import org.junit.Test;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

public class SelectionOperatorTests {
    private class TestSelectionOperator extends SelectionOperator {

        public PopulationModel getPopulationModel() {
            return populationModel;
        }

        public ChromosomeFactory getChromosomeFactory() {
            return chromosomeFactory;
        }

        @Override
        public double[] buildNextGenes() {
            throw new NotImplementedException();
        }
    }

    @Test
    public void SelectionOperator_InitializeInstance_ShouldInitializeFields() {
        TestSelectionOperator sut = new TestSelectionOperator();

        PopulationInitializer populationInitializer = new TestPopulationInitializer();

        PopulationModel populationModel =
                        new PopulationModel.Builder().populationInitializer(populationInitializer).build();
        ChromosomeFactory chromosomeFactory = new ChromosomeFactory();
        sut.initializeInstance(populationModel, chromosomeFactory);

        Assert.assertSame(populationModel, sut.getPopulationModel());
        Assert.assertSame(chromosomeFactory, sut.getChromosomeFactory());
    }
}
