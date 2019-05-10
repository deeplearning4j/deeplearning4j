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

package org.deeplearning4j.arbiter.optimize.api;

import org.deeplearning4j.arbiter.optimize.generator.GridSearchCandidateGenerator;
import org.deeplearning4j.arbiter.optimize.generator.RandomSearchGenerator;
import org.nd4j.shade.jackson.annotation.JsonInclude;
import org.nd4j.shade.jackson.annotation.JsonSubTypes;
import org.nd4j.shade.jackson.annotation.JsonTypeInfo;

/**
 * A CandidateGenerator proposes candidates (i.e., hyperparameter configurations) for evaluation.
 * This abstraction allows for different ways of generating the next configuration to test; for example,
 * random search, grid search, Bayesian optimization methods, etc.
 *
 * @author Alex Black
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
@JsonTypeInfo(use = JsonTypeInfo.Id.CLASS, include = JsonTypeInfo.As.PROPERTY, property = "@class")
public interface CandidateGenerator {

    /**
     * Is this candidate generator able to generate more candidates? This will always return true in some
     * cases, but some search strategies have a limit (grid search, for example)
     */
    boolean hasMoreCandidates();

    /**
     * Generate a candidate hyperparameter configuration
     */
    Candidate getCandidate();

    /**
     * Report results for the candidate generator.
     *
     * @param result The results to report
     */
    void reportResults(OptimizationResult result);

    /**
     * @return Get the parameter space for this candidate generator
     */
    ParameterSpace<?> getParameterSpace();

    /**
     * @param rngSeed Set the random number generator seed for the candidate generator
     */
    void setRngSeed(long rngSeed);

    /**
     * @return The type (class) of the generated candidates
     */
    Class<?> getCandidateType();
}
