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

package org.deeplearning4j.arbiter.optimize.runner;

import org.deeplearning4j.arbiter.optimize.api.saving.ResultReference;
import org.deeplearning4j.arbiter.optimize.config.OptimizationConfiguration;
import org.deeplearning4j.arbiter.optimize.runner.listener.StatusListener;
import org.nd4j.shade.jackson.annotation.JsonTypeInfo;

import java.util.List;

@JsonTypeInfo(use = JsonTypeInfo.Id.CLASS, include = JsonTypeInfo.As.PROPERTY, property = "@class")
public interface IOptimizationRunner {

    void execute();

    /** Total number of candidates: created (scheduled), completed and failed */
    int numCandidatesTotal();

    int numCandidatesCompleted();

    int numCandidatesFailed();

    /** Number of candidates running or queued */
    int numCandidatesQueued();

    /** Best score found so far */
    Double bestScore();

    /** Time that the best score was found at, or 0 if no jobs have completed successfully */
    Long bestScoreTime();

    /** Index of the best scoring candidate, or -1 if no candidate has scored yet*/
    int bestScoreCandidateIndex();

    List<ResultReference> getResults();

    OptimizationConfiguration getConfiguration();

    void addListeners(StatusListener... listeners);

    void removeListeners(StatusListener... listeners);

    void removeAllListeners();

    List<CandidateInfo> getCandidateStatus();

    /**
     * @param awaitCompletion If true: await completion of currently scheduled tasks. If false: shutdown immediately,
     *                        cancelling any currently executing tasks
     */
    void shutdown(boolean awaitCompletion);
}
