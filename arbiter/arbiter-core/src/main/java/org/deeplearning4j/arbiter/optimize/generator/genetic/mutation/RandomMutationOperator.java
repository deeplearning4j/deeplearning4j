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

package org.deeplearning4j.arbiter.optimize.generator.genetic.mutation;

import org.apache.commons.math3.random.JDKRandomGenerator;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.random.SynchronizedRandomGenerator;
import org.nd4j.common.base.Preconditions;

/**
 * A mutation operator where each gene has a chance of being mutated with a <i>mutation rate</i> probability.
 *
 * @author Alexandre Boulanger
 */
public class RandomMutationOperator implements MutationOperator {
    private static final double DEFAULT_MUTATION_RATE = 0.005;

    private final double mutationRate;
    private final RandomGenerator rng;

    public static class Builder {
        private double mutationRate = DEFAULT_MUTATION_RATE;
        private RandomGenerator rng;

        /**
         * Each gene will have this probability of being mutated.
         *
         * @param rate The mutation rate. (default 0.005)
         */
        public Builder mutationRate(double rate) {
            Preconditions.checkState(rate >= 0.0 && rate <= 1.0, "Rate must be between 0.0 and 1.0, got %s", rate);

            this.mutationRate = rate;
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

        public RandomMutationOperator build() {
            if (rng == null) {
                rng = new SynchronizedRandomGenerator(new JDKRandomGenerator());
            }
            return new RandomMutationOperator(this);
        }
    }

    private RandomMutationOperator(RandomMutationOperator.Builder builder) {
        this.mutationRate = builder.mutationRate;
        this.rng = builder.rng;
    }

    /**
     * Performs the mutation. Each gene has a <i>mutation rate</i> probability of being mutated.
     *
     * @param genes The genes to be mutated
     * @return True if the genes were mutated, otherwise false.
     */
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
