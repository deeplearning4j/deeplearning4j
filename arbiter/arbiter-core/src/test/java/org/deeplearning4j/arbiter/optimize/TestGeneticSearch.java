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

package org.deeplearning4j.arbiter.optimize;

import org.deeplearning4j.arbiter.optimize.api.CandidateGenerator;
import org.deeplearning4j.arbiter.optimize.api.data.DataSetIteratorFactoryProvider;
import org.deeplearning4j.arbiter.optimize.api.score.ScoreFunction;
import org.deeplearning4j.arbiter.optimize.api.termination.MaxCandidatesCondition;
import org.deeplearning4j.arbiter.optimize.config.OptimizationConfiguration;
import org.deeplearning4j.arbiter.optimize.generator.GeneticSearchCandidateGenerator;
import org.deeplearning4j.arbiter.optimize.generator.RandomSearchGenerator;
import org.deeplearning4j.arbiter.optimize.generator.genetic.exceptions.GeneticGenerationException;
import org.deeplearning4j.arbiter.optimize.generator.genetic.selection.SelectionOperator;
import org.deeplearning4j.arbiter.optimize.runner.IOptimizationRunner;
import org.deeplearning4j.arbiter.optimize.runner.LocalOptimizationRunner;
import org.deeplearning4j.arbiter.optimize.runner.listener.impl.LoggingStatusListener;
import org.junit.Test;

import java.util.HashMap;
import java.util.Map;

public class TestGeneticSearch {
    public class TestSelectionOperator extends SelectionOperator {

        @Override
        public double[] buildNextGenes() {
            throw new GeneticGenerationException("TEST");
        }
    }

    @Test
    public void test() throws Exception {

        ScoreFunction scoreFunction = new BraninFunction.BraninScoreFunction();

        //Define configuration:
        CandidateGenerator candidateGenerator = new GeneticSearchCandidateGenerator.Builder(new BraninFunction.BraninSpace(), scoreFunction).build();

        OptimizationConfiguration configuration = new OptimizationConfiguration.Builder()
                .candidateGenerator(candidateGenerator)
                .scoreFunction(scoreFunction)
                .terminationConditions(new MaxCandidatesCondition(50))
                .build();

        IOptimizationRunner runner = new LocalOptimizationRunner(configuration, new BraninFunction.BraninTaskCreator());

        runner.addListeners(new LoggingStatusListener());
        runner.execute();

        System.out.println("----- Complete -----");
    }

    @Test
    public void GeneticSearchCandidateGenerator_getCandidate_GeneticExceptionShouldStopGeneration() {

        ScoreFunction scoreFunction = new BraninFunction.BraninScoreFunction();

        //Define configuration:
        CandidateGenerator candidateGenerator = new GeneticSearchCandidateGenerator.Builder(new BraninFunction.BraninSpace(), scoreFunction)
                .selectionOperator(new TestSelectionOperator())
                .build();

        OptimizationConfiguration configuration = new OptimizationConfiguration.Builder()
                .candidateGenerator(candidateGenerator)
                .scoreFunction(scoreFunction)
                .terminationConditions(new MaxCandidatesCondition(50))
                .build();

        IOptimizationRunner runner = new LocalOptimizationRunner(configuration, new BraninFunction.BraninTaskCreator());

        runner.addListeners(new LoggingStatusListener());
        runner.execute();

        System.out.println("----- Complete -----");

    }

}
