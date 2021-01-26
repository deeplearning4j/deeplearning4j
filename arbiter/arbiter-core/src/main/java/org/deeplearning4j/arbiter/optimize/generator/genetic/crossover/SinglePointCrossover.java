/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.arbiter.optimize.generator.genetic.crossover;

import org.apache.commons.math3.random.JDKRandomGenerator;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.random.SynchronizedRandomGenerator;
import org.deeplearning4j.arbiter.optimize.generator.genetic.crossover.parentselection.RandomTwoParentSelection;
import org.deeplearning4j.arbiter.optimize.generator.genetic.crossover.parentselection.TwoParentSelection;
import org.nd4j.common.base.Preconditions;

/**
 * The single point crossover will select a random point where every genes before that point comes from one parent
 * and after which every genes comes from the other parent.
 *
 * @author Alexandre Boulanger
 */
public class SinglePointCrossover extends TwoParentsCrossoverOperator {
    private static final double DEFAULT_CROSSOVER_RATE = 0.85;

    private final RandomGenerator rng;
    private final double crossoverRate;

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

        public SinglePointCrossover build() {
            if (rng == null) {
                rng = new SynchronizedRandomGenerator(new JDKRandomGenerator());
            }

            if (parentSelection == null) {
                parentSelection = new RandomTwoParentSelection();
            }

            return new SinglePointCrossover(this);
        }
    }

    private SinglePointCrossover(SinglePointCrossover.Builder builder) {
        super(builder.parentSelection);

        this.crossoverRate = builder.crossoverRate;
        this.rng = builder.rng;
    }

    /**
     * Has a probability <i>crossoverRate</i> of performing the crossover where the operator will select a random crossover point.<br>
     * Each gene before this point comes from one of the two parents and each gene at or after this point comes from the other parent.
     * Otherwise, returns the genes of a random parent.
     *
     * @return The crossover result. See {@link CrossoverResult}.
     */
    public CrossoverResult crossover() {
        double[][] parents = parentSelection.selectParents();

        boolean isModified = false;
        double[] resultGenes = parents[0];

        if (rng.nextDouble() < crossoverRate) {
            int chromosomeLength = parents[0].length;

            // Crossover
            resultGenes = new double[chromosomeLength];

            int crossoverPoint = rng.nextInt(chromosomeLength);
            for (int i = 0; i < resultGenes.length; ++i) {
                resultGenes[i] = ((i < crossoverPoint) ? parents[0] : parents[1])[i];
            }
            isModified = true;
        }

        return new CrossoverResult(isModified, resultGenes);
    }
}
