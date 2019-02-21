package org.deeplearning4j.arbiter.optimize.genetic.crossover;

import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.arbiter.optimize.generator.genetic.crossover.CrossoverResult;
import org.deeplearning4j.arbiter.optimize.generator.genetic.crossover.SinglePointCrossover;
import org.deeplearning4j.arbiter.optimize.genetic.TestParentSelection;
import org.deeplearning4j.arbiter.optimize.genetic.TestRandomGenerator;
import org.junit.Assert;
import org.junit.Test;

public class SinglePointCrossoverTests {
    @Test
    public void SinglePointCrossover_BelowCrossoverRate_ShouldReturnParent0() {
        RandomGenerator rng = new TestRandomGenerator(null, new double[] {1.0});

        double[][] parents = new double[2][];
        parents[0] = new double[] {1.0, 1.0, 1.0};
        parents[1] = new double[] {2.0, 2.0, 2.0};
        TestParentSelection parentSelection = new TestParentSelection(parents);

        SinglePointCrossover sut = new SinglePointCrossover.Builder().parentSelection(parentSelection)
                        .randomGenerator(rng).crossoverRate(0.0).build();

        CrossoverResult result = sut.crossover();

        Assert.assertFalse(result.isModified());
        Assert.assertSame(parents[0], result.getGenes());
    }

    @Test
    public void SinglePointCrossover_ShouldReturnSingleSplit() {
        RandomGenerator rng = new TestRandomGenerator(new int[] {2}, new double[] {0.1});

        double[][] parents = new double[2][];
        parents[0] = new double[] {1.0, 1.0, 1.0};
        parents[1] = new double[] {2.0, 2.0, 2.0};
        TestParentSelection parentSelection = new TestParentSelection(parents);

        SinglePointCrossover sut = new SinglePointCrossover.Builder().parentSelection(parentSelection)
                        .randomGenerator(rng).crossoverRate(0.5).build();

        CrossoverResult result = sut.crossover();

        Assert.assertTrue(result.isModified());
        Assert.assertEquals(1.0, result.getGenes()[0], 0.0);
        Assert.assertEquals(1.0, result.getGenes()[1], 0.0);
        Assert.assertEquals(2.0, result.getGenes()[2], 0.0);

    }

}
