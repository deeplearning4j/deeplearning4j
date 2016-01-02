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
package org.arbiter.optimize.runner;

import org.arbiter.optimize.api.saving.ResultReference;
import org.arbiter.optimize.config.OptimizationConfiguration;
import org.arbiter.optimize.runner.listener.runner.OptimizationRunnerStatusListener;

import java.util.List;

public interface IOptimizationRunner<C,M,A> {

    void execute();

    /** Total number of candidates: created (scheduled), completed and failed */
    int numCandidatesTotal();

    int numCandidatesCompleted();

    int numCandidatesFailed();

    /** Number of candidates running or queued */
    int numCandidatesQueued();

    /** Best score found so far */
    double bestScore();

    /** Time that the best score was found at, or 0 if no jobs have completed successfully */
    long bestScoreTime();

    /** Index of the best scoring candidate, or -1 if no candidate has scored yet*/
    int bestScoreCandidateIndex();

    List<ResultReference<C,M,A>> getResults();

    OptimizationConfiguration<C,M,?,A> getConfiguration();

    void addListeners(OptimizationRunnerStatusListener... listeners);

    void removeListeners(OptimizationRunnerStatusListener... listeners);

    void removeAllListeners();

    List<CandidateStatus> getCandidateStatus();

}
