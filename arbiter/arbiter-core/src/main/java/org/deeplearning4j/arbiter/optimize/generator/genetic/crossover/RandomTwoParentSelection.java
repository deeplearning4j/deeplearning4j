package org.deeplearning4j.arbiter.optimize.generator.genetic.crossover;

import org.apache.commons.math3.random.JDKRandomGenerator;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.random.SynchronizedRandomGenerator;

public class RandomTwoParentSelection extends TwoParentSelection {

    private final RandomGenerator rng;

    public RandomTwoParentSelection() {
        this(new SynchronizedRandomGenerator(new JDKRandomGenerator()));
    }

    public RandomTwoParentSelection(RandomGenerator rng) {
        this.rng = rng;
    }

    @Override
    public double[][] selectParents() {
        double[][] parents = new double[2][];

        int parent1Idx = rng.nextInt(population.size());
        int parent2Idx;
        do {
            parent2Idx = rng.nextInt(population.size());
        } while (parent1Idx == parent2Idx);

        parents[0] = population.get(parent1Idx).genes;
        parents[1] = population.get(parent2Idx).genes;

        return parents;
    }
}
