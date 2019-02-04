package org.deeplearning4j.arbiter.optimize.genetic.selection;

import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.arbiter.optimize.generator.genetic.ChromosomeFactory;
import org.deeplearning4j.arbiter.optimize.generator.genetic.crossover.CrossoverResult;
import org.deeplearning4j.arbiter.optimize.generator.genetic.culling.CullOperator;
import org.deeplearning4j.arbiter.optimize.generator.genetic.population.PopulationInitializer;
import org.deeplearning4j.arbiter.optimize.generator.genetic.population.PopulationModel;
import org.deeplearning4j.arbiter.optimize.generator.genetic.selection.GeneticSelectionOperator;
import org.deeplearning4j.arbiter.optimize.genetic.TestCrossoverOperator;
import org.deeplearning4j.arbiter.optimize.genetic.TestMutationOperator;
import org.deeplearning4j.arbiter.optimize.genetic.TestPopulationInitializer;
import org.deeplearning4j.arbiter.optimize.genetic.TestRandomGenerator;
import org.junit.Assert;
import org.junit.Test;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

public class GeneticSelectionOperatorTests {

    private class TestCullOperator implements CullOperator {

        private final int culledSize;

        public TestCullOperator(int culledSize) {

            this.culledSize = culledSize;
        }

        @Override
        public void initializeInstance(PopulationModel populationModel) {

        }

        @Override
        public void cullPopulation() {
            throw new NotImplementedException();
        }

        @Override
        public int getCulledSize() {
            return culledSize;
        }
    }

    @Test
    public void GeneticSelectionOperator_PopulationNotReadyToBreed_ShouldReturnRandomGenes() {
        RandomGenerator rng = new TestRandomGenerator(null, new double[] { 123.0 });

        PopulationInitializer populationInitializer = new TestPopulationInitializer();

        TestCullOperator cullOperator = new TestCullOperator(1000);
        PopulationModel populationModel = new PopulationModel.Builder().populationInitializer(populationInitializer)
                .cullOperator(cullOperator)
                .build();
        ChromosomeFactory chromosomeFactory = new ChromosomeFactory();
        chromosomeFactory.initializeInstance(1);
        GeneticSelectionOperator sut = new GeneticSelectionOperator.Builder()
                .randomGenerator(rng)
                .build();
        sut.initializeInstance(populationModel, chromosomeFactory);

        double[] newGenes = sut.buildNextGenes();

        Assert.assertEquals(1, newGenes.length);
        Assert.assertEquals(123.0, newGenes[0], 0.0);
    }

    @Test
    public void GeneticSelectionOperator_NoModificationOnFirstTry() {
        RandomGenerator rng = new TestRandomGenerator(null, new double[] { 123.0 });

        PopulationInitializer populationInitializer = new TestPopulationInitializer();

        TestCullOperator cullOperator = new TestCullOperator(-1);

        PopulationModel populationModel = new PopulationModel.Builder().populationInitializer(populationInitializer)
                .cullOperator(cullOperator)
                .build();

        ChromosomeFactory chromosomeFactory = new ChromosomeFactory();
        chromosomeFactory.initializeInstance(1);

        CrossoverResult[] crossoverResults = new CrossoverResult[2];
        crossoverResults[0] = new CrossoverResult(false, new double[0]);
        crossoverResults[1] = new CrossoverResult(true, new double[0]);
        TestCrossoverOperator crossoverOperator = new TestCrossoverOperator(crossoverResults);

        boolean[] mutationResults = new boolean[] { false, false };
        TestMutationOperator mutationOperator = new TestMutationOperator(mutationResults);

        GeneticSelectionOperator sut = new GeneticSelectionOperator.Builder()
                .randomGenerator(rng)
                .crossoverOperator(crossoverOperator)
                .mutationOperator(mutationOperator)
                .build();
        sut.initializeInstance(populationModel, chromosomeFactory);

        double[] newGenes = sut.buildNextGenes();

        Assert.assertSame(crossoverResults[1].genes, newGenes);
    }

    @Test
    public void GeneticSelectionOperator_MutationNoModificationOnFirstTry() {
        RandomGenerator rng = new TestRandomGenerator(null, new double[] { 123.0 });

        PopulationInitializer populationInitializer = new TestPopulationInitializer();

        TestCullOperator cullOperator = new TestCullOperator(-1);

        PopulationModel populationModel = new PopulationModel.Builder().populationInitializer(populationInitializer)
                .cullOperator(cullOperator)
                .build();

        ChromosomeFactory chromosomeFactory = new ChromosomeFactory();
        chromosomeFactory.initializeInstance(1);

        CrossoverResult[] crossoverResults = new CrossoverResult[3];
        crossoverResults[0] = new CrossoverResult(false, new double[0]);
        crossoverResults[1] = new CrossoverResult(false, new double[0]);
        crossoverResults[2] = new CrossoverResult(true, new double[0]);
        TestCrossoverOperator crossoverOperator = new TestCrossoverOperator(crossoverResults);

        boolean[] mutationResults = new boolean[] { false, false, true };
        TestMutationOperator mutationOperator = new TestMutationOperator(mutationResults);

        GeneticSelectionOperator sut = new GeneticSelectionOperator.Builder()
                .randomGenerator(rng)
                .crossoverOperator(crossoverOperator)
                .mutationOperator(mutationOperator)
                .build();
        sut.initializeInstance(populationModel, chromosomeFactory);

        double[] newGenes = sut.buildNextGenes();

        Assert.assertSame(crossoverResults[2].genes, newGenes);
    }

    @Test
    public void GeneticSelectionOperator_ShouldNotBuildDuplicates() {
        RandomGenerator rng = new TestRandomGenerator(null, new double[] { 123.0 });

        PopulationInitializer populationInitializer = new TestPopulationInitializer();

        TestCullOperator cullOperator = new TestCullOperator(-1);

        PopulationModel populationModel = new PopulationModel.Builder().populationInitializer(populationInitializer)
                .cullOperator(cullOperator)
                .build();

        ChromosomeFactory chromosomeFactory = new ChromosomeFactory();
        chromosomeFactory.initializeInstance(1);

        CrossoverResult[] crossoverResults = new CrossoverResult[3];
        crossoverResults[0] = new CrossoverResult(true, new double[] { 1.0 });
        crossoverResults[1] = new CrossoverResult(true, new double[] { 1.0 });
        crossoverResults[2] = new CrossoverResult(true, new double[] { 2.0 });
        TestCrossoverOperator crossoverOperator = new TestCrossoverOperator(crossoverResults);

        boolean[] mutationResults = new boolean[] { false, false, false };
        TestMutationOperator mutationOperator = new TestMutationOperator(mutationResults);

        GeneticSelectionOperator sut = new GeneticSelectionOperator.Builder()
                .randomGenerator(rng)
                .crossoverOperator(crossoverOperator)
                .mutationOperator(mutationOperator)
                .build();
        sut.initializeInstance(populationModel, chromosomeFactory);

        double[] newGenes = sut.buildNextGenes();
        Assert.assertSame(crossoverResults[0].genes, newGenes);

        newGenes = sut.buildNextGenes();
        Assert.assertSame(crossoverResults[2].genes, newGenes);
    }

}
