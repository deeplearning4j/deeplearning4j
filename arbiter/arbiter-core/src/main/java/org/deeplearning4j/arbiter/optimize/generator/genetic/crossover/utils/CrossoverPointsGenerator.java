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

package org.deeplearning4j.arbiter.optimize.generator.genetic.crossover.utils;

import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.arbiter.optimize.generator.genetic.crossover.KPointCrossover;

import java.util.*;

/**
 * A helper class used by {@link KPointCrossover} to generate the crossover points
 * 
 * @author Alexandre Boulanger
 */
public class CrossoverPointsGenerator {
    private final int minCrossovers;
    private final int maxCrossovers;
    private final RandomGenerator rng;
    private final List<Integer> parameterIndexes;

    /**
    * Constructor
    *
    * @param chromosomeLength The number of genes
    * @param minCrossovers    The minimum number of crossover points to generate
    * @param maxCrossovers    The maximum number of crossover points to generate
    * @param rng              A RandomGenerator instance
    */
    public CrossoverPointsGenerator(int chromosomeLength, int minCrossovers, int maxCrossovers, RandomGenerator rng) {
        this.minCrossovers = minCrossovers;
        this.maxCrossovers = maxCrossovers;
        this.rng = rng;
        parameterIndexes = new ArrayList<Integer>();
        for (int i = 0; i < chromosomeLength; ++i) {
            parameterIndexes.add(i);
        }
    }

    /**
    * Generate a list of crossover points.
    *
    * @return An ordered list of crossover point indexes and with Integer.MAX_VALUE as the last element
    */
    public Deque<Integer> getCrossoverPoints() {
        Collections.shuffle(parameterIndexes);
        List<Integer> crossoverPointLists =
                        parameterIndexes.subList(0, rng.nextInt(maxCrossovers - minCrossovers) + minCrossovers);
        Collections.sort(crossoverPointLists);
        Deque<Integer> crossoverPoints = new ArrayDeque<Integer>(crossoverPointLists);
        crossoverPoints.add(Integer.MAX_VALUE);

        return crossoverPoints;
    }
}
