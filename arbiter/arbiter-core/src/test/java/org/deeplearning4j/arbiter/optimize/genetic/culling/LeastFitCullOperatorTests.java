package org.deeplearning4j.arbiter.optimize.genetic.culling;

import org.deeplearning4j.arbiter.optimize.generator.genetic.Chromosome;
import org.deeplearning4j.arbiter.optimize.generator.genetic.culling.LeastFitCullOperator;
import org.deeplearning4j.arbiter.optimize.generator.genetic.population.PopulationInitializer;
import org.deeplearning4j.arbiter.optimize.generator.genetic.population.PopulationModel;
import org.deeplearning4j.arbiter.optimize.genetic.TestPopulationInitializer;
import org.junit.Assert;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

public class LeastFitCullOperatorTests {

    @Test
    public void LeastFitCullingOperation_ShouldCullLastElements() {
        LeastFitCullOperator sut = new LeastFitCullOperator(0.50);

        PopulationInitializer populationInitializer = new TestPopulationInitializer();

        PopulationModel populationModel = new PopulationModel.Builder().populationInitializer(populationInitializer)
                        .populationSize(10).build();
        sut.initializeInstance(populationModel);

        List<Chromosome> originalChromosomes = new ArrayList<>();
        for (int i = 0; i < 10; ++i) {
            originalChromosomes.add(new Chromosome(null, (double) i));
        }

        List<Chromosome> chromosomes = populationModel.getPopulation();
        for (int i = 0; i < 10; ++i) {
            chromosomes.add(originalChromosomes.get(i));
        }

        sut.cullPopulation();

        Assert.assertEquals(5, chromosomes.size());
        for (int i = 0; i < 5; ++i) {
            Assert.assertSame(originalChromosomes.get(i), chromosomes.get(i));
        }
    }


}
