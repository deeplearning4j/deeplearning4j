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

package org.deeplearning4j.earlystopping.termination;

import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.shade.jackson.annotation.JsonProperty;

/**
 * Terminate training if best model score does not improve for N epochs
 */
@Slf4j
@Data
public class ScoreImprovementEpochTerminationCondition implements EpochTerminationCondition {
    @JsonProperty
    private int maxEpochsWithNoImprovement;
    @JsonProperty
    private int bestEpoch = -1;
    @JsonProperty
    private double bestScore;
    @JsonProperty
    private double minImprovement = 0.0;

    public ScoreImprovementEpochTerminationCondition(int maxEpochsWithNoImprovement) {
        this.maxEpochsWithNoImprovement = maxEpochsWithNoImprovement;
    }

    public ScoreImprovementEpochTerminationCondition(int maxEpochsWithNoImprovement, double minImprovement) {
        this.maxEpochsWithNoImprovement = maxEpochsWithNoImprovement;
        this.minImprovement = minImprovement;
    }

    @Override
    public void initialize() {
        bestEpoch = -1;
        bestScore = Double.NaN;
    }

    @Override
    public boolean terminate(int epochNum, double score, boolean minimize) {
        if (bestEpoch == -1) {
            bestEpoch = epochNum;
            bestScore = score;
            return false;
        } else {
            double improvement = (minimize ? bestScore - score : score - bestScore);
            if (improvement > minImprovement) {
                if (minImprovement > 0) {
                    log.info("Epoch with score greater than threshold * * *");
                }
                bestScore = score;
                bestEpoch = epochNum;
                return false;
            }

            return epochNum >= bestEpoch + maxEpochsWithNoImprovement;
        }
    }

    @Override
    public String toString() {
        return "ScoreImprovementEpochTerminationCondition(maxEpochsWithNoImprovement=" + maxEpochsWithNoImprovement
                        + ", minImprovement=" + minImprovement + ")";
    }
}
