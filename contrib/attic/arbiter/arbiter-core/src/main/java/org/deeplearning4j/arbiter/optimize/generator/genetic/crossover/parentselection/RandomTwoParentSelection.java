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

package org.deeplearning4j.arbiter.optimize.generator.genetic.crossover.parentselection;

import org.apache.commons.math3.random.JDKRandomGenerator;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.random.SynchronizedRandomGenerator;

/**
 * A parent selection behavior that returns two random parents.
 *
 * @author Alexandre Boulanger
 */
public class RandomTwoParentSelection extends TwoParentSelection {

    private final RandomGenerator rng;

    public RandomTwoParentSelection() {
        this(new SynchronizedRandomGenerator(new JDKRandomGenerator()));
    }

    /**
     * Use a supplied RandomGenerator
     *
     * @param rng An instance of RandomGenerator
     */
    public RandomTwoParentSelection(RandomGenerator rng) {
        this.rng = rng;
    }

    /**
     * Selects two random parents
     *
     * @return An array of parents genes. The outer array are the parents, and the inner array are the genes.
     */
    @Override
    public double[][] selectParents() {
        double[][] parents = new double[2][];

        int parent1Idx = rng.nextInt(population.size());
        int parent2Idx;
        do {
            parent2Idx = rng.nextInt(population.size());
        } while (parent1Idx == parent2Idx);

        parents[0] = population.get(parent1Idx).getGenes();
        parents[1] = population.get(parent2Idx).getGenes();

        return parents;
    }
}
