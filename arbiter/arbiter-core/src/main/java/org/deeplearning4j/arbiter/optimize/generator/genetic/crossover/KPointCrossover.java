/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.arbiter.optimize.generator.genetic.crossover;

import org.apache.commons.math3.random.JDKRandomGenerator;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.random.SynchronizedRandomGenerator;
import org.deeplearning4j.arbiter.optimize.generator.genetic.crossover.parentselection.RandomTwoParentSelection;
import org.deeplearning4j.arbiter.optimize.generator.genetic.crossover.parentselection.TwoParentSelection;
import org.deeplearning4j.arbiter.optimize.generator.genetic.crossover.utils.CrossoverPointsGenerator;

import java.util.*;

/**
* The K-Point crossover will select at random multiple crossover points.<br>
* Each gene comes from one of the two parents. Each time a crossover point is reached, the parent is switched.
*/
public class KPointCrossover extends TwoParentsCrossoverOperator {
    private static final double DEFAULT_CROSSOVER_RATE = 0.85;
    private static final int DEFAULT_MIN_CROSSOVER = 1;
    private static final int DEFAULT_MAX_CROSSOVER = 4;

    public static class Builder {
        private double crossoverRate = DEFAULT_CROSSOVER_RATE;
        private int minCrossovers = DEFAULT_MIN_CROSSOVER;
        private int maxCrossovers = DEFAULT_MAX_CROSSOVER;
        private RandomGenerator rng;
        private TwoParentSelection parentSelection;

        /**
         * The probability that the operator generates a crossover (default 0.85).
         *
         * @param rate A value between 0.0 and 1.0
         */
        public Builder crossoverRate(double rate) {
            if(rate < 0 || rate > 1.0) {
                throw new IllegalArgumentException("Rate must be between 0.0 and 1.0");
            }

            this.crossoverRate = rate;
            return this;
        }

        /**
         * The number of crossovers points (default is min 1, max 4)
         *
         * @param min The minimum number
         * @param max The maximum number
         */
        public Builder numCrossovers(int min, int max) {
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

        /**
         * Use a fixed number of crossover points
         *
         * @param num The number of crossovers
         */
        public Builder numCrossovers(int num) {
            if(num < 0) {
                throw new IllegalArgumentException("Num must be positive");
            }
            this.minCrossovers = num;
            this.maxCrossovers = num;
            return this;
        }

        /**
         * Use a supplied RandomGenerator
         *
         * @param rng An instance of RandomGenerator
         */
        public Builder randomGenerator(RandomGenerator rng) {
            this.rng = rng;
            return this;
        }

        /**
         * The parent selection behavior. Default is random parent selection.
         *
         * @param parentSelection An instance of TwoParentSelection
         */
        public Builder parentSelection(TwoParentSelection parentSelection) {
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

    /**
     * Has a probability <i>crossoverRate</i> of performing the crossover where the operator will select at random multiple crossover points.<br>
     * Each gene comes from one of the two parents. Each time a crossover point is reached, the parent is switched. <br>
     * Otherwise, returns the genes of a random parent.
     *
     * @return The crossover result. See {@link CrossoverResult}.
     */
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
