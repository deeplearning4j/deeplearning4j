/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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
import org.nd4j.common.base.Preconditions;

/**
 * A crossover operator that linearly combines the genes of two parents. <br>
 * When a crossover is generated (with a of probability <i>crossover rate</i>), each genes is a linear combination of the corresponding genes of the parents.
 * <p>
 * <i>t*parentA + (1-t)*parentB, where t is [0, 1] and different for each gene.</i>
 * 
 * @author Alexandre Boulanger
 */
public class ArithmeticCrossover extends TwoParentsCrossoverOperator {
    private static final double DEFAULT_CROSSOVER_RATE = 0.85;

    private final double crossoverRate;
    private final RandomGenerator rng;

    public static class Builder {
        private double crossoverRate = DEFAULT_CROSSOVER_RATE;
        private RandomGenerator rng;
        private TwoParentSelection parentSelection;

        /**
         * The probability that the operator generates a crossover (default 0.85).
         *
         * @param rate A value between 0.0 and 1.0
         */
        public Builder crossoverRate(double rate) {
            Preconditions.checkState(rate >= 0.0 && rate <= 1.0, "Rate must be between 0.0 and 1.0, got %s", rate);

            this.crossoverRate = rate;
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

        public ArithmeticCrossover build() {
            if (rng == null) {
                rng = new SynchronizedRandomGenerator(new JDKRandomGenerator());
            }

            if (parentSelection == null) {
                parentSelection = new RandomTwoParentSelection();
            }

            return new ArithmeticCrossover(this);
        }
    }

    private ArithmeticCrossover(ArithmeticCrossover.Builder builder) {
        super(builder.parentSelection);

        this.crossoverRate = builder.crossoverRate;
        this.rng = builder.rng;
    }

    /**
     * Has a probability <i>crossoverRate</i> of performing the crossover where each gene is a linear combination of:<br>
     *     <i>t*parentA + (1-t)*parentB, where t is [0, 1] and different for each gene.</i><br>
     * Otherwise, returns the genes of a random parent.
     *
     * @return The crossover result. See {@link CrossoverResult}.
     */
    @Override
    public CrossoverResult crossover() {
        double[][] parents = parentSelection.selectParents();

        double[] offspringValues = new double[parents[0].length];

        if (rng.nextDouble() < crossoverRate) {
            for (int i = 0; i < offspringValues.length; ++i) {
                double t = rng.nextDouble();
                offspringValues[i] = t * parents[0][i] + (1.0 - t) * parents[1][i];
            }
            return new CrossoverResult(true, offspringValues);
        }

        return new CrossoverResult(false, parents[0]);
    }
}
