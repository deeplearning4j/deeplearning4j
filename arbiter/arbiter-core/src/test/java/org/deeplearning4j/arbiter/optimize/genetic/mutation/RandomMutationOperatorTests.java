package org.deeplearning4j.arbiter.optimize.genetic.mutation;

import main.java.org.ab2002.genetic.tests.TestRandomGenerator;
import org.deeplearning4j.arbiter.optimize.generator.genetic.mutation.RandomMutationOperator;
import org.junit.Assert;
import org.junit.Test;

import java.util.Arrays;

public class RandomMutationOperatorTests {
    @Test
    public void RandomMutationOperator_DefaultBuild_ShouldNotBeNull() {
        RandomMutationOperator sut = new RandomMutationOperator.Builder().build();
        Assert.assertNotNull(sut);
    }

    @Test
    public void RandomMutationOperator_BuildWithMutationRate_ShouldUseSuppliedRate() {
        RandomMutationOperator sut = new RandomMutationOperator.Builder()
                .mutationRate(0.123)
                .build();
        Assert.assertEquals(0.123, sut.getMutationRate(), 0.0);
    }

    @Test
    public void RandomMutationOperator_BelowMutationRate_ShouldNotMutate() {
        double[] randomNumbers = new double[] { 0.1, 1.0, 1.0 };

        RandomMutationOperator sut = new RandomMutationOperator.Builder()
                .mutationRate(0.1)
                .randomGenerator(new TestRandomGenerator(null, randomNumbers))
                .build();

        double[] genes = new double[] { -1.0, -1.0, -1.0 };
        boolean hasMutated = sut.mutate(genes);

        Assert.assertFalse(hasMutated);
        Assert.assertTrue(Arrays.equals(new double[] { -1.0, -1.0, -1.0 }, genes));
    }

    @Test
    public void RandomMutationOperator_AboveMutationRate_ShouldMutate() {
        double[] randomNumbers = new double[] { 0.099, 0.123, 1.0, 1.0 };

        RandomMutationOperator sut = new RandomMutationOperator.Builder()
                .mutationRate(0.1)
                .randomGenerator(new TestRandomGenerator(null, randomNumbers))
                .build();

        double[] genes = new double[] { -1.0, -1.0, -1.0 };
        boolean hasMutated = sut.mutate(genes);

        Assert.assertTrue(hasMutated);
        Assert.assertTrue(Arrays.equals(new double[] { 0.123, -1.0, -1.0 }, genes));
    }

    @Test
    public void RandomMutationOperator_getSetMutationRate() {
        RandomMutationOperator sut = new RandomMutationOperator.Builder()
                .build();
        sut.setMutationRate(0.123);

        Assert.assertEquals(0.123, sut.getMutationRate(), 0.0);
    }

}
