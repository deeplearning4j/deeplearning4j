/*
 *
 *  * Copyright 2016 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */
package org.deeplearning4j.arbiter.optimize.api;

/**
 * A CandidateGenerator proposes candidates (i.e., hyperparameter configurations) for evaluation.
 * This abstraction allows for different ways of generating the next configuration to test; for example,
 * random search, grid search, Bayesian optimization methods, etc.
 *
 * @param <C> Type of candidate to generate
 * @author Alex Black
 */
public interface CandidateGenerator<C> {

    /**
     * Is this candidate generator able to generate more candidates? This will always return true in some
     * cases, but some search strategies have a limit (grid search, for example)
     */
    boolean hasMoreCandidates();

    /**
     * Generate a candidate hyperparameter configuration
     */
    Candidate<C> getCandidate();

    void reportResults(OptimizationResult<C, ?, ?> result);

    ParameterSpace<C> getParameterSpace();

    void setRngSeed(long rngSeed);

}
