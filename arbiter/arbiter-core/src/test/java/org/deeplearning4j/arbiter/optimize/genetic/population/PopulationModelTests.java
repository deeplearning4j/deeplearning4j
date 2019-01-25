package org.deeplearning4j.arbiter.optimize.genetic.population;

import main.java.org.ab2002.genetic.tests.TestPopulationInitializer;
import org.deeplearning4j.arbiter.optimize.generator.genetic.Chromosome;
import org.deeplearning4j.arbiter.optimize.generator.genetic.culling.CullOperation;
import org.deeplearning4j.arbiter.optimize.generator.genetic.population.PopulationInitializer;
import org.deeplearning4j.arbiter.optimize.generator.genetic.population.PopulationListener;
import org.deeplearning4j.arbiter.optimize.generator.genetic.population.PopulationModel;
import org.junit.Assert;
import org.junit.Test;

import java.util.List;

public class PopulationModelTests {

    private class TestCullOperation implements CullOperation {

        private final int culledSize;
        public boolean hasCulled = false;

        public TestCullOperation(int culledSize) {

            this.culledSize = culledSize;
        }

        @Override
        public void initializeInstance(PopulationModel populationModel) {

        }

        @Override
        public void cullPopulation() {
            hasCulled = true;
        }

        @Override
        public int getCulledSize() {
            return culledSize;
        }
    }

    private class TestPopulationListener implements PopulationListener {

        public List<Chromosome> population;

        @Override
        public void onChanged(List<Chromosome> population) {
            this.population = population;
        }
    }

    @Test
    public void PopulationModel_IsReadyToBreed_NotReadyToBreed_ShouldReturnFalse() {
        PopulationInitializer populationInitializer = new TestPopulationInitializer();

        PopulationModel sut = new PopulationModel.Builder(populationInitializer)
                .populationSize(5)
                .cullingOperation(new TestCullOperation(1))
                .build();

        boolean result = sut.isReadyToBreed();

        Assert.assertFalse(result);
    }

    @Test
    public void PopulationModel_IsReadyToBreed_ReadyToBreed_ShouldReturnTrue() {
        PopulationInitializer populationInitializer = new TestPopulationInitializer();

        PopulationModel sut = new PopulationModel.Builder(populationInitializer)
                .populationSize(5)
                .cullingOperation(new TestCullOperation(1))
                .build();

        sut.population.add(null);

        boolean result = sut.isReadyToBreed();

        Assert.assertTrue(result);
    }

    @Test
    public void PopulationModel_Add_MaximizeScore_ShouldOrderDescendingPopulation() {
        PopulationInitializer populationInitializer = new TestPopulationInitializer();

        PopulationModel sut = new PopulationModel.Builder(populationInitializer)
                .populationSize(5)
                .cullingOperation(new TestCullOperation(1))
                .build();

        sut.initializeInstance(false);

        Chromosome[] chromosomes = new Chromosome[3];
        chromosomes[0] = new Chromosome(new double[0], 1.0);
        chromosomes[1] = new Chromosome(new double[0], 100.0);
        chromosomes[2] = new Chromosome(new double[0], 10.0);
        sut.add(chromosomes[0]);
        sut.add(chromosomes[1]);
        sut.add(chromosomes[2]);

        Assert.assertSame(chromosomes[1], sut.population.get(0));
        Assert.assertSame(chromosomes[2], sut.population.get(1));
        Assert.assertSame(chromosomes[0], sut.population.get(2));
    }

    @Test
    public void PopulationModel_Add_MinimizeScore_ShouldOrderAscendingPopulation() {
        PopulationInitializer populationInitializer = new TestPopulationInitializer();

        PopulationModel sut = new PopulationModel.Builder(populationInitializer)
                .populationSize(5)
                .cullingOperation(new TestCullOperation(1))
                .build();

        sut.initializeInstance(true);

        Chromosome[] chromosomes = new Chromosome[3];
        chromosomes[0] = new Chromosome(new double[0], 100.0);
        chromosomes[1] = new Chromosome(new double[0], 1.0);
        chromosomes[2] = new Chromosome(new double[0], 10.0);
        sut.add(chromosomes[0]);
        sut.add(chromosomes[1]);
        sut.add(chromosomes[2]);

        Assert.assertSame(chromosomes[1], sut.population.get(0));
        Assert.assertSame(chromosomes[2], sut.population.get(1));
        Assert.assertSame(chromosomes[0], sut.population.get(2));
    }

    @Test
    public void PopulationModel_Add_ShouldTriggerPopulationListeners() {
        PopulationInitializer populationInitializer = new TestPopulationInitializer();

        PopulationModel sut = new PopulationModel.Builder(populationInitializer)
                .populationSize(5)
                .cullingOperation(new TestCullOperation(1))
                .build();

        sut.initializeInstance(true);

        TestPopulationListener populationListener = new TestPopulationListener();
        sut.addListener(populationListener);

        sut.add(new Chromosome(new double[0], 100.0));

        Assert.assertSame(sut.population, populationListener.population);
    }

    @Test
    public void PopulationModel_Add_BelowPopulationSize_ShouldNotCull() {
        PopulationInitializer populationInitializer = new TestPopulationInitializer();

        TestCullOperation cullingOperation = new TestCullOperation(3);

        PopulationModel sut = new PopulationModel.Builder(populationInitializer)
                .populationSize(5)
                .cullingOperation(cullingOperation)
                .build();

        sut.initializeInstance(true);

        sut.add(new Chromosome(new double[0], 1.0));
        sut.add(new Chromosome(new double[0], 2.0));
        sut.add(new Chromosome(new double[0], 3.0));
        sut.add(new Chromosome(new double[0], 4.0));
        sut.add(new Chromosome(new double[0], 5.0));

        Assert.assertFalse(cullingOperation.hasCulled);
    }

    @Test
    public void PopulationModel_Add_AbovePopulationSize_ShouldCull() {
        PopulationInitializer populationInitializer = new TestPopulationInitializer();

        TestCullOperation cullingOperation = new TestCullOperation(3);

        PopulationModel sut = new PopulationModel.Builder(populationInitializer)
                .populationSize(5)
                .cullingOperation(cullingOperation)
                .build();

        sut.initializeInstance(true);

        sut.add(new Chromosome(new double[0], 1.0));
        sut.add(new Chromosome(new double[0], 2.0));
        sut.add(new Chromosome(new double[0], 3.0));
        sut.add(new Chromosome(new double[0], 4.0));
        sut.add(new Chromosome(new double[0], 5.0));
        sut.add(new Chromosome(new double[0], 6.0));

        Assert.assertTrue(cullingOperation.hasCulled);
    }

}
