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

package org.deeplearning4j.arbiter.optimize.generator;

import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.arbiter.optimize.api.Candidate;
import org.deeplearning4j.arbiter.optimize.api.OptimizationResult;
import org.deeplearning4j.arbiter.optimize.api.ParameterSpace;
import org.deeplearning4j.arbiter.optimize.api.score.ScoreFunction;
import org.deeplearning4j.arbiter.optimize.generator.genetic.Chromosome;
import org.deeplearning4j.arbiter.optimize.generator.genetic.ChromosomeFactory;
import org.deeplearning4j.arbiter.optimize.generator.genetic.exceptions.GeneticGenerationException;
import org.deeplearning4j.arbiter.optimize.generator.genetic.population.EmptyPopulationInitializer;
import org.deeplearning4j.arbiter.optimize.generator.genetic.population.PopulationInitializer;
import org.deeplearning4j.arbiter.optimize.generator.genetic.population.PopulationModel;
import org.deeplearning4j.arbiter.optimize.generator.genetic.selection.GeneticSelectionOperator;
import org.deeplearning4j.arbiter.optimize.generator.genetic.selection.SelectionOperator;

import java.util.Map;

/**
 * Uses a genetic algorithm to generate candidates.
 *
 * @author Alexandre Boulanger
 */
@Slf4j
public class GeneticSearchCandidateGenerator extends BaseCandidateGenerator {

    @Getter
    protected final PopulationModel populationModel;

    protected final ChromosomeFactory chromosomeFactory;
    protected final SelectionOperator selectionOperator;

    protected boolean hasMoreCandidates = true;

    public static class Builder {
        protected final ParameterSpace<?> parameterSpace;

        protected Map<String, Object> dataParameters;
        protected boolean initDone;
        protected boolean minimizeScore;
        protected PopulationModel populationModel;
        protected ChromosomeFactory chromosomeFactory;
        protected SelectionOperator selectionOperator;

        /**
         * @param parameterSpace ParameterSpace from which to generate candidates
         * @param scoreFunction The score function that will be used in the OptimizationConfiguration
         */
        public Builder(ParameterSpace<?> parameterSpace, ScoreFunction scoreFunction) {
            this.parameterSpace = parameterSpace;
            this.minimizeScore = scoreFunction.minimize();
        }

        /**
         * @param populationModel The PopulationModel instance to use.
         */
        public Builder populationModel(PopulationModel populationModel) {
            this.populationModel = populationModel;
            return this;
        }

        /**
         * @param selectionOperator The SelectionOperator to use. Default is GeneticSelectionOperator
         */
        public Builder selectionOperator(SelectionOperator selectionOperator) {
            this.selectionOperator = selectionOperator;
            return this;
        }

        public Builder dataParameters(Map<String, Object> dataParameters) {

            this.dataParameters = dataParameters;
            return this;
        }

        public GeneticSearchCandidateGenerator.Builder initDone(boolean initDone) {
            this.initDone = initDone;
            return this;
        }

        /**
         * @param chromosomeFactory The ChromosomeFactory to use
         */
        public Builder chromosomeFactory(ChromosomeFactory chromosomeFactory) {
            this.chromosomeFactory = chromosomeFactory;
            return this;
        }

        public GeneticSearchCandidateGenerator build() {
            if (populationModel == null) {
                PopulationInitializer defaultPopulationInitializer = new EmptyPopulationInitializer();
                populationModel = new PopulationModel.Builder().populationInitializer(defaultPopulationInitializer)
                                .build();
            }

            if (chromosomeFactory == null) {
                chromosomeFactory = new ChromosomeFactory();
            }

            if (selectionOperator == null) {
                selectionOperator = new GeneticSelectionOperator.Builder().build();
            }

            return new GeneticSearchCandidateGenerator(this);
        }
    }

    private GeneticSearchCandidateGenerator(Builder builder) {
        super(builder.parameterSpace, builder.dataParameters, builder.initDone);

        initialize();

        chromosomeFactory = builder.chromosomeFactory;
        populationModel = builder.populationModel;
        selectionOperator = builder.selectionOperator;

        chromosomeFactory.initializeInstance(builder.parameterSpace.numParameters());
        populationModel.initializeInstance(builder.minimizeScore);
        selectionOperator.initializeInstance(populationModel, chromosomeFactory);

    }

    @Override
    public boolean hasMoreCandidates() {
        return hasMoreCandidates;
    }

    @Override
    public Candidate getCandidate() {

        double[] values = null;
        Object value = null;
        Exception e = null;

        try {
            values = selectionOperator.buildNextGenes();
            value = parameterSpace.getValue(values);
        } catch (GeneticGenerationException e2) {
            log.warn("Error generating candidate", e2);
            e = e2;
            hasMoreCandidates = false;
        } catch (Exception e2) {
            log.warn("Error getting configuration for candidate", e2);
            e = e2;
        }

        return new Candidate(value, candidateCounter.getAndIncrement(), values, dataParameters, e);
    }

    @Override
    public Class<?> getCandidateType() {
        return null;
    }

    @Override
    public String toString() {
        return "GeneticSearchCandidateGenerator";
    }

    @Override
    public void reportResults(OptimizationResult result) {
        if (result.getScore() == null) {
            return;
        }

        Chromosome newChromosome = chromosomeFactory.createChromosome(result.getCandidate().getFlatParameters(),
                        result.getScore());
        populationModel.add(newChromosome);
    }
}
