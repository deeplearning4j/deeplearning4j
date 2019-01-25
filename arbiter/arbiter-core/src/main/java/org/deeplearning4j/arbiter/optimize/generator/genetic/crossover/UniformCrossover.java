package org.deeplearning4j.arbiter.optimize.generator.genetic.crossover;

import org.apache.commons.math3.random.JDKRandomGenerator;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.random.SynchronizedRandomGenerator;

public class UniformCrossover extends TwoParentsCrossoverOperator {
    private static final double DEFAULT_CROSSOVER_RATE = 0.85;
    private static final double DEFAULT_PARENT_BIAS_FACTOR = 0.5;

    public static class Builder {
        private double crossoverRate = DEFAULT_CROSSOVER_RATE;
        private double parentBiasFactor = DEFAULT_PARENT_BIAS_FACTOR;
        private RandomGenerator rng;
        private TwoParentSelection parentSelection;

        public UniformCrossover.Builder crossoverRate(double rate) {
            if(rate < 0 || rate> 1.0) {
                throw new IllegalArgumentException("Rate must be within 0.0 and 1.0 range.");
            }

            this.crossoverRate = rate;
            return this;
        }

        public UniformCrossover.Builder parentBiasFactor(double factor) {
            if(factor < 0 || factor > 1.0) {
                throw new IllegalArgumentException("Factor must be within 0.0 and 1.0 range.");
            }
            this.parentBiasFactor = factor;
            return this;
        }

        public UniformCrossover.Builder randomGenerator(RandomGenerator rng) {
            this.rng = rng;
            return this;
        }

        public UniformCrossover.Builder parentSelection(TwoParentSelection parentSelection) {
            this.parentSelection = parentSelection;
            return this;
        }

        public UniformCrossover build() {
            if(rng == null) {
                rng = new SynchronizedRandomGenerator(new JDKRandomGenerator());
            }
            if(parentSelection == null) {
                parentSelection = new RandomTwoParentSelection();
            }
            return new UniformCrossover(this);
        }
    }

    private final double crossoverRate;
    private final double parentBiasFactor;
    private final RandomGenerator rng;

    private UniformCrossover(UniformCrossover.Builder builder) {
        super(builder.parentSelection);

        this.crossoverRate = builder.crossoverRate;
        this.parentBiasFactor = builder.parentBiasFactor;
        this.rng = builder.rng;
    }

    @Override
    public CrossoverResult crossover()  {
        // select the parents
        double[][] parents = parentSelection.selectParents();

        if (rng.nextDouble() < crossoverRate) {
            // Crossover
            double[] offspringValues = new double[parents[0].length];

            for (int i = 0; i < offspringValues.length; ++i) {
                offspringValues[i] = ((rng.nextDouble() < parentBiasFactor) ? parents[0] : parents[1])[i];
            }
            return new CrossoverResult(true, offspringValues);
        }

        return new CrossoverResult(false, parents[0]);
    }
}
