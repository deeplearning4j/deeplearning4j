package org.deeplearning4j.arbiter.optimize.generator.genetic.crossover;

import org.apache.commons.math3.random.JDKRandomGenerator;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.random.SynchronizedRandomGenerator;

import java.util.*;

public class KPointCrossover extends TwoParentsCrossoverOperator {
    private static final double DEFAULT_CROSSOVER_RATE = 0.85;
    private static final int DEFAULT_MIN_CROSSOVER = 2;
    private static final int DEFAULT_MAX_CROSSOVER = 4;

    public static class Builder {
        private double crossoverRate = DEFAULT_CROSSOVER_RATE;
        private int minCrossovers = DEFAULT_MIN_CROSSOVER;
        private int maxCrossovers = DEFAULT_MAX_CROSSOVER;
        private RandomGenerator rng;
        private TwoParentSelection parentSelection;

        public KPointCrossover.Builder crossoverRate(double rate) {
            if(rate < 0 || rate > 1.0) {
                throw new IllegalArgumentException("Rate must be between 0.0 and 1.0");
            }

            this.crossoverRate = rate;
            return this;
        }

        public KPointCrossover.Builder numCrossovers(int min, int max) {
            if(max < 0 || min < 0) {
                throw new IllegalArgumentException("Min and max must be positive");
            }
            if(max < min) {
                throw new IllegalArgumentException("Max must be greater or equal to min");
            }
            this.minCrossovers = min;
            this.maxCrossovers = max;
            return this;
        }

        public KPointCrossover.Builder numCrossovers(int num) {
            if(num < 0) {
                throw new IllegalArgumentException("Num must be positive");
            }
            this.minCrossovers = num;
            this.maxCrossovers = num;
            return this;
        }

        public KPointCrossover.Builder randomGenerator(RandomGenerator rng) {
            this.rng = rng;
            return this;
        }

        public KPointCrossover.Builder parentSelection(TwoParentSelection parentSelection) {
            this.parentSelection = parentSelection;
            return this;
        }

        public KPointCrossover build()
        {
            if(rng == null) {
                rng = new SynchronizedRandomGenerator(new JDKRandomGenerator());
            }

            if(parentSelection == null){
                parentSelection = new RandomTwoParentSelection();
            }

            return new KPointCrossover(this);
        }
    }

    private final double crossoverRate;
    private final int minCrossovers;
    private final int maxCrossovers;

    private final RandomGenerator rng;

    private CrossoverPointsGenerator crossoverPointsGenerator;

    private KPointCrossover(KPointCrossover.Builder builder) {
        super(builder.parentSelection);

        this.crossoverRate = builder.crossoverRate;
        this.maxCrossovers = builder.maxCrossovers;
        this.minCrossovers = builder.minCrossovers;
        this.rng = builder.rng;
    }

    @Override
    public CrossoverResult crossover()  {
        double[][] parents = parentSelection.selectParents();

        if (rng.nextDouble() < crossoverRate) {
            // Select crossover points
            if(crossoverPointsGenerator == null) {
                crossoverPointsGenerator = new CrossoverPointsGenerator(parents[0].length, minCrossovers, maxCrossovers, rng);
            }
            Deque<Integer> crossoverPoints = crossoverPointsGenerator.getCrossoverPoints();

            // Crossover
            double[] offspringValues = new double[parents[0].length];
            int currentParent = 0;
            int nextCrossover = crossoverPoints.pop();
            for (int i = 0; i < offspringValues.length; ++i) {
                if(i == nextCrossover) {
                    currentParent =  currentParent == 0 ? 1 : 0;
                    nextCrossover = crossoverPoints.pop();
                }
                offspringValues[i] = parents[currentParent][i];
            }
            return new CrossoverResult(true, offspringValues);
        }

        return new CrossoverResult(false, parents[0]);
    }
}
