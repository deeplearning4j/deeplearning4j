/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.autodiff.samediff.rl;

import lombok.Builder;
import lombok.Data;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.function.BiFunction;

/**
 * Rule-based reward function for format, length, and pattern rewards.
 * Useful for GRPO-style training where rewards can be computed programmatically.
 *
 * @author Eclipse Deeplearning4j Contributors
 */
@Data
@Builder
public class RuleBasedRewardFunction implements RewardFunction {

    /**
     * Custom scoring function. Takes (prompts, completions) and returns rewards.
     */
    private final BiFunction<INDArray, INDArray, INDArray> scoringFunction;

    /**
     * Target completion length. If set, rewards are penalized for deviation.
     */
    @Builder.Default
    private int targetLength = -1;

    /**
     * Weight for length penalty.
     */
    @Builder.Default
    private double lengthPenaltyWeight = 0.0;

    @Override
    public INDArray score(INDArray prompts, INDArray completions) {
        INDArray rewards;
        if (scoringFunction != null) {
            rewards = scoringFunction.apply(prompts, completions);
        } else {
            rewards = Nd4j.ones(prompts.size(0));
        }

        // Apply length penalty if configured
        if (targetLength > 0 && lengthPenaltyWeight > 0) {
            long completionLength = completions.size(1);
            double lengthDiff = Math.abs(completionLength - targetLength) / (double) targetLength;
            double penalty = lengthPenaltyWeight * lengthDiff;
            rewards = rewards.sub(penalty);
        }

        return rewards;
    }

    /**
     * Create a simple constant reward function (useful for testing).
     */
    public static RuleBasedRewardFunction constant(double reward) {
        return RuleBasedRewardFunction.builder()
                .scoringFunction((p, c) -> Nd4j.ones(p.size(0)).muli(reward))
                .build();
    }
}
