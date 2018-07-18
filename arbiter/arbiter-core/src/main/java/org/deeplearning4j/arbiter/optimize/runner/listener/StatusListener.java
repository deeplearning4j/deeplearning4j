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

package org.deeplearning4j.arbiter.optimize.runner.listener;

import org.deeplearning4j.arbiter.optimize.api.OptimizationResult;
import org.deeplearning4j.arbiter.optimize.runner.CandidateInfo;
import org.deeplearning4j.arbiter.optimize.runner.IOptimizationRunner;

/**
 * The status Listener interface is used to inspect/track the status of execution, both for individual candidates,
 * and for the optimisation runner overall.
 *
 * @author Alex Black
 */
public interface StatusListener {

    /** Called when optimization runner starts execution */
    void onInitialization(IOptimizationRunner runner);

    /** Called when optimization runner terminates */
    void onShutdown(IOptimizationRunner runner);

    /** Called when any of the summary stats change, for the optimization runner:
     * number scheduled, number completed, number failed, best score, etc. */
    void onRunnerStatusChange(IOptimizationRunner runner);

    /**
     * Called when the status of the candidate is change. For example created, completed, failed.
     *
     * @param candidateInfo Candidate information
     * @param runner        Optimisation runner calling this method
     * @param result        Optimisation result. Maybe null.
     */
    void onCandidateStatusChange(CandidateInfo candidateInfo, IOptimizationRunner runner, OptimizationResult result);

    /**
     *  This method may be called by tasks as they are executing. The intent of this method is to report partial results,
     *  such as different stages of learning, or scores/evaluations so far
     *
     * @param candidateInfo Candidate information
     * @param candidate     Current candidate value/configuration
     * @param iteration     Current iteration number
     */
    void onCandidateIteration(CandidateInfo candidateInfo, Object candidate, int iteration);

}
