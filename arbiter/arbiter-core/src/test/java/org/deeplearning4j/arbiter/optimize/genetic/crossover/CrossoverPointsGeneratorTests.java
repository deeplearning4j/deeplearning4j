package org.deeplearning4j.arbiter.optimize.genetic.crossover;

import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.arbiter.optimize.generator.genetic.crossover.utils.CrossoverPointsGenerator;
import org.deeplearning4j.arbiter.optimize.genetic.TestRandomGenerator;
import org.junit.Assert;
import org.junit.Test;

import java.util.Deque;

public class CrossoverPointsGeneratorTests {

    @Test
    public void CrossoverPointsGenerator_FixedNumberCrossovers() {
        RandomGenerator rng = new TestRandomGenerator(new int[] {0}, null);
        CrossoverPointsGenerator sut = new CrossoverPointsGenerator(10, 2, 2, rng);

        Deque<Integer> result = sut.getCrossoverPoints();

        Assert.assertEquals(3, result.size());
        int a = result.pop();
        int b = result.pop();
        int c = result.pop();
        Assert.assertTrue(a < b);
        Assert.assertTrue(b < c);
        Assert.assertEquals(Integer.MAX_VALUE, c);
    }
}
