package org.deeplearning4j.arbiter.optimize.genetic.culling;

import org.deeplearning4j.arbiter.optimize.generator.genetic.Chromosome;
import org.deeplearning4j.arbiter.optimize.generator.genetic.culling.RatioCullOperator;
import org.deeplearning4j.arbiter.optimize.generator.genetic.population.PopulationInitializer;
import org.deeplearning4j.arbiter.optimize.generator.genetic.population.PopulationModel;
import org.deeplearning4j.arbiter.optimize.genetic.TestPopulationInitializer;
import org.junit.Assert;
import org.junit.Test;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

import java.util.List;

public class RatioCullOperatorTests {

    class TestRatioCullOperator extends RatioCullOperator {

        public TestRatioCullOperator() {
            super();
        }

        public TestRatioCullOperator(double ratio) {
            super(ratio);
        }

        public List<Chromosome> getPopulation() {
            return population;
        }

        @Override
        public void cullPopulation() {
            throw new NotImplementedException();
        }

        public double getCullRatio() {
            return cullRatio;
        }
    }

    @Test
    public void RatioCullingOperation_ctorWithCullRatio_ShouldHaveParamRatio() {
        TestRatioCullOperator sut = new TestRatioCullOperator(0.123);

        Assert.assertEquals(0.123, sut.getCullRatio(), 0.0);
    }

    @Test
    public void RatioCullingOperation_initialize_shouldSetCulledSizeAndPopulation() throws IllegalAccessException {
        TestRatioCullOperator sut = new TestRatioCullOperator(0.50);

        PopulationInitializer populationInitializer = new TestPopulationInitializer();

        PopulationModel populationModel = new PopulationModel.Builder().populationInitializer(populationInitializer)
                        .populationSize(10).build();
        sut.initializeInstance(populationModel);

        Assert.assertSame(populationModel.getPopulation(), sut.getPopulation());
        Assert.assertEquals(5, sut.getCulledSize());
    }

}
