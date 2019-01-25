package org.deeplearning4j.arbiter.optimize.generator.genetic.crossover;

import org.apache.commons.math3.random.JDKRandomGenerator;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.random.SynchronizedRandomGenerator;

public class ArithmeticCrossover extends TwoParentsCrossoverOperator {
    private static final double DEFAULT_CROSSOVER_RATE = 0.85;

    public static class Builder {
        private double crossoverRate = DEFAULT_CROSSOVER_RATE;
        private RandomGenerator rng;
        private TwoParentSelection parentSelection;

        public ArithmeticCrossover.Builder crossoverRate(double rate) {
            if(rate < 0 || rate > 1.0) {
                throw new IllegalArgumentException("Rate must be between 0.0 and 1.0");
            }

            this.crossoverRate = rate;
            return this;
        }

        public ArithmeticCrossover.Builder randomGenerator(RandomGenerator rng) {
            this.rng = rng;
            return this;
        }

        public ArithmeticCrossover.Builder parentSelection(TwoParentSelection parentSelection) {
            this.parentSelection = parentSelection;
            return this;
        }

        public ArithmeticCrossover build() {
            if(rng == null) {
                rng = new SynchronizedRandomGenerator(new JDKRandomGenerator());
            }

            if(parentSelection == null) {
                parentSelection = new RandomTwoParentSelection();
            }

            return new ArithmeticCrossover(this);
        }
    }

    private double crossoverRate;
    public double getCrossoverRate() {
        return crossoverRate;
    }
    public void setCrossoverRate(double crossoverRate) {
        this.crossoverRate = crossoverRate;
    }

    private RandomGenerator rng;

    public ArithmeticCrossover(ArithmeticCrossover.Builder builder) {
        super(builder.parentSelection);

        this.crossoverRate = builder.crossoverRate;
        this.rng = builder.rng;
    }

    @Override
    public CrossoverResult crossover()  {
        double[][] parents = parentSelection.selectParents();

        double[] offspringValues = new double[parents[0].length];

        for (int i = 0; i < offspringValues.length; ++i) {
            double t = rng.nextDouble();
            offspringValues[i] = t * parents[0][i] + (1.0 - t) * parents[1][i];
        }
        return new CrossoverResult(true, offspringValues);
    }
}
