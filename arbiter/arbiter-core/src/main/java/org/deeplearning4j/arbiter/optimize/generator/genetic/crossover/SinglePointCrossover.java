package org.deeplearning4j.arbiter.optimize.generator.genetic.crossover;

import org.apache.commons.math3.random.JDKRandomGenerator;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.random.SynchronizedRandomGenerator;

public class SinglePointCrossover extends TwoParentsCrossoverOperator {
    private static final double DEFAULT_CROSSOVER_RATE = 0.85;

    public static class Builder {
        private double crossoverRate = DEFAULT_CROSSOVER_RATE;
        private RandomGenerator rng;
        private TwoParentSelection parentSelection;

        public SinglePointCrossover.Builder crossoverRate(double rate) {
            if(rate < 0 || rate > 1.0) {
                throw new IllegalArgumentException("Rate must be between 0.0 and 1.0");
            }

            this.crossoverRate = rate;
            return this;
        }

        public SinglePointCrossover.Builder randomGenerator(RandomGenerator rng) {
            this.rng = rng;
            return this;
        }

        public SinglePointCrossover.Builder parentSelection(TwoParentSelection parentSelection) {
            this.parentSelection = parentSelection;
            return this;
        }

        public SinglePointCrossover build() {
            if(rng == null) {
                rng = new SynchronizedRandomGenerator(new JDKRandomGenerator());
            }

            if(parentSelection == null) {
                parentSelection = new RandomTwoParentSelection();
            }

            return new SinglePointCrossover(this);
        }
    }

    private final RandomGenerator rng;
    private final double crossoverRate;

    public SinglePointCrossover(SinglePointCrossover.Builder builder) {
        super(builder.parentSelection);

        this.crossoverRate = builder.crossoverRate;
        this.rng = builder.rng;
    }

    public CrossoverResult crossover()  {
        double[][] parents = parentSelection.selectParents();

        if (rng.nextDouble() < crossoverRate) {
            int chromosomeLength = parents[0].length;

            // Crossover
            double[] offspringValues = new double[chromosomeLength];

            int crossoverPoint = rng.nextInt(chromosomeLength);
            for (int i = 0; i < offspringValues.length; ++i) {
                offspringValues[i] = ((i < crossoverPoint) ? parents[0] : parents[1])[i];
            }
            return new CrossoverResult(true, offspringValues);
        }

        return new CrossoverResult(false, parents[0]);
    }
}
