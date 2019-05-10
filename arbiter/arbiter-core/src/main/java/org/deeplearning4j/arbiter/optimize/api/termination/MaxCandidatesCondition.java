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

package org.deeplearning4j.arbiter.optimize.api.termination;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.deeplearning4j.arbiter.optimize.runner.IOptimizationRunner;
import org.nd4j.shade.jackson.annotation.JsonProperty;

/**
 * Terminate hyperparameter search when the number of candidates exceeds a specified value.
 * Note that this is counted as number of completed candidates, plus number of failed candidates.
 */
@AllArgsConstructor
@NoArgsConstructor
@Data
public class MaxCandidatesCondition implements TerminationCondition {
    @JsonProperty
    private int maxCandidates;

    @Override
    public void initialize(IOptimizationRunner optimizationRunner) {
        //No op
    }

    @Override
    public boolean terminate(IOptimizationRunner optimizationRunner) {
        return optimizationRunner.numCandidatesCompleted() + optimizationRunner.numCandidatesFailed() >= maxCandidates;
    }

    @Override
    public String toString() {
        return "MaxCandidatesCondition(" + maxCandidates + ")";
    }
}
