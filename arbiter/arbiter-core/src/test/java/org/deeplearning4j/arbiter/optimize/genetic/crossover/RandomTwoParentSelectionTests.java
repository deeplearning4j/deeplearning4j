package org.deeplearning4j.arbiter.optimize.genetic.crossover;

import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.arbiter.optimize.generator.genetic.Chromosome;
import org.deeplearning4j.arbiter.optimize.generator.genetic.crossover.parentselection.RandomTwoParentSelection;
import org.deeplearning4j.arbiter.optimize.genetic.TestRandomGenerator;
import org.junit.Assert;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

public class RandomTwoParentSelectionTests {
    @Test
    public void RandomTwoParentSelection_ShouldReturnTwoDifferentParents() {
        RandomGenerator rng = new TestRandomGenerator(new int[] {1, 1, 1, 0}, null);
        RandomTwoParentSelection sut = new RandomTwoParentSelection(rng);

        List<Chromosome> population = new ArrayList<>();
        population.add(new Chromosome(new double[] {1, 1, 1}, 1.0));
        population.add(new Chromosome(new double[] {2, 2, 2}, 2.0));
        population.add(new Chromosome(new double[] {3, 3, 3}, 3.0));
        sut.initializeInstance(population);

        double[][] result = sut.selectParents();

        Assert.assertSame(population.get(1).getGenes(), result[0]);
        Assert.assertSame(population.get(0).getGenes(), result[1]);
    }
}
