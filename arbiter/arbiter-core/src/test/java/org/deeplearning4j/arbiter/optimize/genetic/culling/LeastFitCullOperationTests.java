package org.deeplearning4j.arbiter.optimize.genetic.culling;

import main.java.org.ab2002.genetic.tests.TestPopulationInitializer;
import org.deeplearning4j.arbiter.optimize.generator.genetic.Chromosome;
import org.deeplearning4j.arbiter.optimize.generator.genetic.culling.LeastFitCullOperation;
import org.deeplearning4j.arbiter.optimize.generator.genetic.population.PopulationInitializer;
import org.deeplearning4j.arbiter.optimize.generator.genetic.population.PopulationModel;
import org.junit.Assert;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

public class LeastFitCullOperationTests {

    @Test
    public void LeastFitCullingOperation_ShouldCullLastElements() {
        LeastFitCullOperation sut = new LeastFitCullOperation(0.50);

        PopulationInitializer populationInitializer = new TestPopulationInitializer();

        PopulationModel populationModel = new PopulationModel.Builder(populationInitializer).populationSize(10).build();
        sut.initializeInstance(populationModel);

        List<Chromosome> originalChromosomes = new ArrayList<>();
        for(int i = 0; i < 10; ++i) {
            originalChromosomes.add(new Chromosome(null, (double)i));
        }

        List<Chromosome> chromosomes = populationModel.population;
        for(int i = 0; i < 10; ++i) {
            chromosomes.add(originalChromosomes.get(i));
        }

        sut.cullPopulation();

        Assert.assertEquals(5, chromosomes.size());
        for(int i = 0; i < 5; ++i) {
            Assert.assertSame(originalChromosomes.get(i), chromosomes.get(i));
        }
    }


}
