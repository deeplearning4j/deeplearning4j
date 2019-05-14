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

package org.deeplearning4j.arbiter.optimize.generator.genetic.population;

import lombok.Getter;
import org.deeplearning4j.arbiter.optimize.generator.genetic.Chromosome;
import org.deeplearning4j.arbiter.optimize.generator.genetic.culling.CullOperator;
import org.deeplearning4j.arbiter.optimize.generator.genetic.culling.LeastFitCullOperator;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

/**
 * The population model handles all aspects of the population (initialization, additions and culling)
 *
 * @author Alexandre Boulanger
 */
public class PopulationModel {
    private static final int DEFAULT_POPULATION_SIZE = 30;

    private final CullOperator cullOperator;
    private final List<PopulationListener> populationListeners = new ArrayList<>();
    private Comparator<Chromosome> chromosomeComparator;

    /**
     * The maximum population size
     */
    @Getter
    private final int populationSize;

    /**
     * The population
     */
    @Getter
    public final List<Chromosome> population;

    /**
     * A comparator used when higher fitness value is better
     */
    public static class MaximizeScoreComparator implements Comparator<Chromosome> {
        @Override
        public int compare(Chromosome lhs, Chromosome rhs) {
            return -Double.compare(lhs.getFitness(), rhs.getFitness());
        }
    }

    /**
     * A comparator used when lower fitness value is better
     */
    public static class MinimizeScoreComparator implements Comparator<Chromosome> {
        @Override
        public int compare(Chromosome lhs, Chromosome rhs) {
            return Double.compare(lhs.getFitness(), rhs.getFitness());
        }
    }

    public static class Builder {
        private int populationSize = DEFAULT_POPULATION_SIZE;
        private PopulationInitializer populationInitializer;
        private CullOperator cullOperator;

        /**
         * Use an alternate population initialization behavior. Default is empty population.
         *
         * @param populationInitializer An instance of PopulationInitializer
         */
        public Builder populationInitializer(PopulationInitializer populationInitializer) {
            this.populationInitializer = populationInitializer;
            return this;
        }

        /**
         * The maximum population size. <br>
         * If using a ratio based culling, using a population with culled size of around 1.5 to 2 times the number of genes generally gives good results.
         * (e.g. For a chromosome having 10 genes, the culled size should be between 15 and 20. And with a cull ratio of 1/3 we should set the population size to 23 to 30. (15 / (1 - 1/3)), rounded up)
         *
         * @param size The maximum size of the population
         */
        public Builder populationSize(int size) {
            populationSize = size;
            return this;
        }

        /**
         * Use an alternate cull operator behavior. Default is least fit culling.
         *
         * @param cullOperator An instance of a CullOperator
         */
        public Builder cullOperator(CullOperator cullOperator) {
            this.cullOperator = cullOperator;
            return this;
        }

        public PopulationModel build() {
            if (cullOperator == null) {
                cullOperator = new LeastFitCullOperator();
            }

            if (populationInitializer == null) {
                populationInitializer = new EmptyPopulationInitializer();
            }

            return new PopulationModel(this);
        }

    }

    public PopulationModel(PopulationModel.Builder builder) {
        populationSize = builder.populationSize;
        population = new ArrayList<>(builder.populationSize);
        PopulationInitializer populationInitializer = builder.populationInitializer;

        List<Chromosome> initializedPopulation = populationInitializer.getInitializedPopulation(populationSize);
        population.clear();
        population.addAll(initializedPopulation);

        cullOperator = builder.cullOperator;
        cullOperator.initializeInstance(this);
    }

    /**
     * Called by the GeneticSearchCandidateGenerator
     */
    public void initializeInstance(boolean minimizeScore) {
        chromosomeComparator = minimizeScore ? new MinimizeScoreComparator() : new MaximizeScoreComparator();
    }

    /**
     * Add a PopulationListener to the list of change listeners
     * @param listener A PopulationListener instance
     */
    public void addListener(PopulationListener listener) {
        populationListeners.add(listener);
    }

    /**
     * Add a Chromosome to the population and call the PopulationListeners. Culling may be triggered.
     *
     * @param element The chromosome to be added
     */
    public void add(Chromosome element) {
        if (population.size() == populationSize) {
            cullOperator.cullPopulation();
        }

        population.add(element);

        Collections.sort(population, chromosomeComparator);

        triggerPopulationChangedListeners(population);
    }

    /**
     * @return Return false when the population is below the culled size, otherwise true. <br>
     * Used by the selection operator to know if the population is still too small and should generate random genes.
     */
    public boolean isReadyToBreed() {
        return population.size() >= cullOperator.getCulledSize();
    }

    private void triggerPopulationChangedListeners(List<Chromosome> population) {
        for (PopulationListener listener : populationListeners) {
            listener.onChanged(population);
        }
    }
}
