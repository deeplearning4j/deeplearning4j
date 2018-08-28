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
import org.nd4j.shade.jackson.annotation.JsonProperty;

/**
 * Created by Sadat Anwar on 3/26/16.
 *
 * Stop the training once we achieved an expected score. Normally this will stop if the current score is lower than
 * the initialized score. If you want to stop the training once the score increases the defined score set the
 * lesserBetter flag to false (feel free to give the flag a better name)
 */
@Data
public class BestScoreEpochTerminationCondition implements EpochTerminationCondition {
    @JsonProperty
    private final double bestExpectedScore;

    public BestScoreEpochTerminationCondition(double bestExpectedScore) {
        this.bestExpectedScore = bestExpectedScore;
    }

    /**
     * @deprecated "lessBetter" argument no longer used
     */
    @Deprecated
    public BestScoreEpochTerminationCondition(double bestExpectedScore, boolean lesserBetter) {
        this(bestExpectedScore);
    }

    @Override
    public void initialize() {
        /* No OP */
    }

    @Override
    public boolean terminate(int epochNum, double score, boolean minimize) {
        if (minimize) {
            return score < bestExpectedScore;
        } else {
            return bestExpectedScore < score;
        }
    }

    @Override
    public String toString() {
        return "BestScoreEpochTerminationCondition(" + bestExpectedScore + ")";
    }
}
