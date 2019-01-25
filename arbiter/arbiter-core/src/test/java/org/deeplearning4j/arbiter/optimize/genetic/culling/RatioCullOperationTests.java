package org.deeplearning4j.arbiter.optimize.genetic.culling;

import main.java.org.ab2002.genetic.tests.TestPopulationInitializer;
import org.deeplearning4j.arbiter.optimize.generator.genetic.Chromosome;
import org.deeplearning4j.arbiter.optimize.generator.genetic.culling.RatioCullOperation;
import org.deeplearning4j.arbiter.optimize.generator.genetic.population.PopulationInitializer;
import org.deeplearning4j.arbiter.optimize.generator.genetic.population.PopulationModel;
import org.junit.Assert;
import org.junit.Test;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

import java.util.List;

public class RatioCullOperationTests {

    class TestRatioCullOperation extends RatioCullOperation {

        public TestRatioCullOperation() {
            super();
        }

        public TestRatioCullOperation(double ratio) {
            super(ratio);
        }

        public List<Chromosome> getPopulation() {
            return population;
        }

        @Override
        public void cullPopulation() {
            throw new NotImplementedException();
        }
    }

    @Test
    public void RatioCullingOperation_getSetCullRatio() {
        TestRatioCullOperation sut = new TestRatioCullOperation();
        sut.setCullRatio(0.123);

        Assert.assertEquals(0.123, sut.getCullRatio(), 0.0);
    }

    @Test
    public void RatioCullingOperation_ctorWithCullRatio_ShouldHaveParamRatio() {
        TestRatioCullOperation sut = new TestRatioCullOperation(0.123);

        Assert.assertEquals(0.123, sut.getCullRatio(), 0.0);
    }

    @Test
    public void RatioCullingOperation_initialize_shouldSetCulledSizeAndPopulation() throws IllegalAccessException {
        TestRatioCullOperation sut = new TestRatioCullOperation(0.50);

        PopulationInitializer populationInitializer = new TestPopulationInitializer();

        PopulationModel populationModel = new PopulationModel.Builder(populationInitializer).populationSize(10).build();
        sut.initializeInstance(populationModel);

        Assert.assertSame(populationModel.population, sut.getPopulation());
        Assert.assertEquals(5, sut.getCulledSize());
    }

}
