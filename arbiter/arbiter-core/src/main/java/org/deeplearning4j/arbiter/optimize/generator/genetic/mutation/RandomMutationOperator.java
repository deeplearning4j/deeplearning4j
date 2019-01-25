package org.deeplearning4j.arbiter.optimize.generator.genetic.mutation;

import org.apache.commons.math3.random.JDKRandomGenerator;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.random.SynchronizedRandomGenerator;

public class RandomMutationOperator implements MutationOperator {
    private static final double DEFAULT_MUTATION_RATE = 0.005;

    public static class Builder {
        private double mutationRate = DEFAULT_MUTATION_RATE;
        private RandomGenerator rng;

        public RandomMutationOperator.Builder mutationRate(double rate) {
            if(rate < 0 || rate> 1.0) {
                throw new IllegalArgumentException("Rate must be within 0.0 and 1.0 range.");
            }

            this.mutationRate = rate;
            return this;
        }

        public RandomMutationOperator.Builder randomGenerator(RandomGenerator rng) {
            this.rng = rng;
            return this;
        }

        public RandomMutationOperator build() {
            if(rng == null) {
                rng = new SynchronizedRandomGenerator(new JDKRandomGenerator());
            }
            return new RandomMutationOperator(this);
        }
    }

    private double mutationRate;
    public double getMutationRate() {
        return mutationRate;
    }
    public void setMutationRate(double mutationRate) {
        this.mutationRate = mutationRate;
    }

    private final RandomGenerator rng;

    private RandomMutationOperator(RandomMutationOperator.Builder builder) {
        this.mutationRate = builder.mutationRate;
        this.rng = builder.rng;
    }

    @Override
    public boolean mutate(double[] genes) {
        boolean hasMutation = false;

        for (int i = 0; i < genes.length; ++i) {
            if (rng.nextDouble() < mutationRate) {
                genes[i] = rng.nextDouble();
                hasMutation = true;
            }
        }

        return hasMutation;
    }
}
