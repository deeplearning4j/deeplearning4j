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

package org.deeplearning4j.arbiter.optimize.generator.genetic.selection;

import org.apache.commons.math3.random.JDKRandomGenerator;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.random.SynchronizedRandomGenerator;
import org.deeplearning4j.arbiter.optimize.generator.genetic.ChromosomeFactory;
import org.deeplearning4j.arbiter.optimize.generator.genetic.crossover.CrossoverOperator;
import org.deeplearning4j.arbiter.optimize.generator.genetic.crossover.CrossoverResult;
import org.deeplearning4j.arbiter.optimize.generator.genetic.crossover.SinglePointCrossover;
import org.deeplearning4j.arbiter.optimize.generator.genetic.exceptions.GeneticGenerationException;
import org.deeplearning4j.arbiter.optimize.generator.genetic.mutation.MutationOperator;
import org.deeplearning4j.arbiter.optimize.generator.genetic.mutation.RandomMutationOperator;
import org.deeplearning4j.arbiter.optimize.generator.genetic.population.PopulationModel;

import java.util.Arrays;

/**
 * A selection operator that will generate random genes initially. Once the population has reached the culled size,
 * will start to generate offsprings of parents selected in the population.
 *
 * @author Alexandre Boulanger
 */
public class GeneticSelectionOperator extends SelectionOperator {

    private final static int PREVIOUS_GENES_TO_KEEP = 100;
    private final static int MAX_NUM_GENERATION_ATTEMPTS = 1024;

    private final CrossoverOperator crossoverOperator;
    private final MutationOperator mutationOperator;
    private final RandomGenerator rng;
    private double[][] previousGenes = new double[PREVIOUS_GENES_TO_KEEP][];
    private int previousGenesIdx = 0;

    public static class Builder {
        private ChromosomeFactory chromosomeFactory;
        private PopulationModel populationModel;
        private CrossoverOperator crossoverOperator;
        private MutationOperator mutationOperator;
        private RandomGenerator rng;

        /**
         * Use an alternate crossover behavior. Default is SinglePointCrossover.
         *
         * @param crossoverOperator An instance of CrossoverOperator
         */
        public Builder crossoverOperator(CrossoverOperator crossoverOperator) {
            this.crossoverOperator = crossoverOperator;
            return this;
        }

        /**
         * Use an alternate mutation behavior. Default is RandomMutationOperator.
         *
         * @param mutationOperator An instance of MutationOperator
         */
        public Builder mutationOperator(MutationOperator mutationOperator) {
            this.mutationOperator = mutationOperator;
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

        public GeneticSelectionOperator build() {
            if (crossoverOperator == null) {
                crossoverOperator = new SinglePointCrossover.Builder().build();
            }

            if (mutationOperator == null) {
                mutationOperator = new RandomMutationOperator.Builder().build();
            }

            if (rng == null) {
                rng = new SynchronizedRandomGenerator(new JDKRandomGenerator());
            }

            return new GeneticSelectionOperator(crossoverOperator, mutationOperator, rng);
        }
    }

    private GeneticSelectionOperator(CrossoverOperator crossoverOperator, MutationOperator mutationOperator,
                    RandomGenerator rng) {
        this.crossoverOperator = crossoverOperator;
        this.mutationOperator = mutationOperator;
        this.rng = rng;
    }

    /**
     * Called by GeneticSearchCandidateGenerator
     */
    @Override
    public void initializeInstance(PopulationModel populationModel, ChromosomeFactory chromosomeFactory) {
        super.initializeInstance(populationModel, chromosomeFactory);
        crossoverOperator.initializeInstance(populationModel);
    }

    /**
     * Build a new set of genes. Has two distinct modes of operation
     * <ul>
     * <li>Before the population has reached the culled size: will return a random set of genes.</li>
     * <li>After: Parents will be selected among the population, a crossover will be applied followed by a mutation.</li>
     * </ul>
     * @return Returns the generated set of genes
     * @throws GeneticGenerationException If buildNextGenes() can't generate a set that has not already been tried,
     *                                               or if the crossover and the mutation operators can't generate a set,
     *                                               this exception is thrown.
     */
    @Override
    public double[] buildNextGenes() {
        double[] result;

        boolean hasAlreadyBeenTried;
        int attemptsRemaining = MAX_NUM_GENERATION_ATTEMPTS;
        do {
            if (populationModel.isReadyToBreed()) {
                result = buildOffspring();
            } else {
                result = buildRandomGenes();
            }

            hasAlreadyBeenTried = hasAlreadyBeenTried(result);
            if (hasAlreadyBeenTried && --attemptsRemaining == 0) {
                throw new GeneticGenerationException("Failed to generate a set of genes not already tried.");
            }
        } while (hasAlreadyBeenTried);

        previousGenes[previousGenesIdx] = result;
        previousGenesIdx = ++previousGenesIdx % previousGenes.length;

        return result;
    }

    private boolean hasAlreadyBeenTried(double[] genes) {
        for (int i = 0; i < previousGenes.length; ++i) {
            double[] current = previousGenes[i];
            if (current != null && Arrays.equals(current, genes)) {
                return true;
            }
        }

        return false;
    }

    private double[] buildOffspring() {
        double[] offspringValues;

        boolean isModified;
        int attemptsRemaining = MAX_NUM_GENERATION_ATTEMPTS;
        do {
            CrossoverResult crossoverResult = crossoverOperator.crossover();
            offspringValues = crossoverResult.getGenes();
            isModified = crossoverResult.isModified();
            isModified |= mutationOperator.mutate(offspringValues);

            if (!isModified && --attemptsRemaining == 0) {
                throw new GeneticGenerationException(
                                String.format("Crossover and mutation operators failed to generate a new set of genes after %s attempts.",
                                                MAX_NUM_GENERATION_ATTEMPTS));
            }
        } while (!isModified);

        return offspringValues;
    }

    private double[] buildRandomGenes() {
        double[] randomValues = new double[chromosomeFactory.getChromosomeLength()];
        for (int i = 0; i < randomValues.length; ++i) {
            randomValues[i] = rng.nextDouble();
        }

        return randomValues;
    }

}
