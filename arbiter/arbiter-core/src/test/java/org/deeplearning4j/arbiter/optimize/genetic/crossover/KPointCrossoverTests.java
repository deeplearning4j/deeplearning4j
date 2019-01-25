package org.deeplearning4j.arbiter.optimize.genetic.crossover;

import main.java.org.ab2002.genetic.tests.TestParentSelection;
import main.java.org.ab2002.genetic.tests.TestRandomGenerator;
import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.arbiter.optimize.generator.genetic.crossover.CrossoverResult;
import org.deeplearning4j.arbiter.optimize.generator.genetic.crossover.KPointCrossover;
import org.deeplearning4j.arbiter.optimize.generator.genetic.crossover.TwoParentSelection;
import org.junit.Assert;
import org.junit.Test;

public class KPointCrossoverTests {

    @Test
    public void KPointCrossover_BelowCrossoverRate_ShouldReturnParent0() {
        RandomGenerator rng = new TestRandomGenerator(null, new double[] { 1.0 });

        double[][] parents = new double[2][];
        parents[0] = new double[] { 0.0 };
        parents[1] = new double[] { 1.0 };
        TwoParentSelection parentSelection = new TestParentSelection(parents);
        KPointCrossover sut = new KPointCrossover.Builder()
                .randomGenerator(rng)
                .crossoverRate(0.0)
                .parentSelection(parentSelection)
                .build();

        CrossoverResult result = sut.crossover();

        Assert.assertFalse(result.hasModification);
        Assert.assertSame(parents[0], result.genes);
    }

    @Test
    public void KPointCrossover_FixedNumberOfCrossovers() {
        RandomGenerator rng = new TestRandomGenerator(new int[] { 0, 1 }, new double[] { 0.0 });

        double[][] parents = new double[3][];
        parents[0] = new double[] { 0.0, 0.0, 0.0, 0.0, 0.0 };
        parents[1] = new double[] { 1.0, 1.0, 1.0, 1.0, 1.0 };
        parents[2] = new double[] { 2.0, 2.0, 2.0, 2.0, 2.0 };
        TwoParentSelection parentSelection = new TestParentSelection(parents);
        KPointCrossover sut = new KPointCrossover.Builder()
                .randomGenerator(rng)
                .crossoverRate(1.0)
                .parentSelection(parentSelection)
                .numCrossovers(2)
                .build();

        CrossoverResult result = sut.crossover();

        Assert.assertTrue(result.hasModification);
        for(double x : result.genes) {
            Assert.assertTrue(x == 0.0 || x == 1.0);
        }
    }
}
