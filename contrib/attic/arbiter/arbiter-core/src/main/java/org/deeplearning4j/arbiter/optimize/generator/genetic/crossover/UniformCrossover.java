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
 * The uniform crossover will, for each gene, randomly select the parent that donates the gene.
 *
 * @author Alexandre Boulanger
 */
public class UniformCrossover extends TwoParentsCrossoverOperator {
    private static final double DEFAULT_CROSSOVER_RATE = 0.85;
    private static final double DEFAULT_PARENT_BIAS_FACTOR = 0.5;

    private final double crossoverRate;
    private final double parentBiasFactor;
    private final RandomGenerator rng;

    public static class Builder {
        private double crossoverRate = DEFAULT_CROSSOVER_RATE;
        private double parentBiasFactor = DEFAULT_PARENT_BIAS_FACTOR;
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
         * A factor that will introduce a bias in the parent selection.<br>
         *
         * @param factor In the range [0, 1]. 0 will only select the first parent while 1 only select the second one. The default is 0.5; no bias.
         */
        public Builder parentBiasFactor(double factor) {
            Preconditions.checkState(factor >= 0.0 && factor <= 1.0, "Factor must be between 0.0 and 1.0, got %s",
                            factor);

            this.parentBiasFactor = factor;
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

        public UniformCrossover build() {
            if (rng == null) {
                rng = new SynchronizedRandomGenerator(new JDKRandomGenerator());
            }
            if (parentSelection == null) {
                parentSelection = new RandomTwoParentSelection();
            }
            return new UniformCrossover(this);
        }
    }

    private UniformCrossover(UniformCrossover.Builder builder) {
        super(builder.parentSelection);

        this.crossoverRate = builder.crossoverRate;
        this.parentBiasFactor = builder.parentBiasFactor;
        this.rng = builder.rng;
    }

    /**
     * Has a probability <i>crossoverRate</i> of performing the crossover where the operator will select randomly which parent donates the gene.<br>
     * One of the parent may be favored if the bias is different than 0.5
     * Otherwise, returns the genes of a random parent.
     *
     * @return The crossover result. See {@link CrossoverResult}.
     */
    @Override
    public CrossoverResult crossover() {
        // select the parents
        double[][] parents = parentSelection.selectParents();

        double[] resultGenes = parents[0];
        boolean isModified = false;

        if (rng.nextDouble() < crossoverRate) {
            // Crossover
            resultGenes = new double[parents[0].length];

            for (int i = 0; i < resultGenes.length; ++i) {
                resultGenes[i] = ((rng.nextDouble() < parentBiasFactor) ? parents[0] : parents[1])[i];
            }
            isModified = true;
        }

        return new CrossoverResult(isModified, resultGenes);
    }
}
