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

package org.nd4j.autodiff.samediff.config;

import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import lombok.experimental.SuperBuilder;

/**
 * Configuration for De-biased Reward GRPO (DrGRPO) training.
 *
 * DrGRPO extends GRPO with two key modifications to remove length bias:
 * - Length normalization: divides rewards by completion length to prevent
 *   the model from learning to generate longer responses just for higher rewards
 * - Baseline subtraction: subtracts the mean reward before computing advantages,
 *   providing a stronger baseline that reduces variance
 *
 * @author Eclipse Deeplearning4j Contributors
 */
@Data
@SuperBuilder
@NoArgsConstructor
@EqualsAndHashCode(callSuper = true)
public class DrGRPOConfig extends RLAlignmentConfig {

    /**
     * Whether to normalize rewards by completion length to remove length bias.
     * reward_normalized = reward / completion_length
     * Default: true
     */
    @lombok.Builder.Default
    private boolean lengthNormalization = true;

    /**
     * Whether to subtract the mean reward before computing advantages.
     * Provides a stronger baseline for variance reduction.
     * Default: true
     */
    @lombok.Builder.Default
    private boolean baselineSubtraction = true;

    /**
     * Number of completions to generate per prompt for group-based advantage estimation.
     * Default: 8
     */
    @lombok.Builder.Default
    private int groupSize = 8;

    /**
     * Clipping epsilon for the surrogate objective.
     * Ratio is clipped to [1 - clipEpsilon, 1 + clipEpsilon].
     * Default: 0.2
     */
    @lombok.Builder.Default
    private double clipEpsilon = 0.2;

    /**
     * Maximum number of new tokens to generate per completion.
     * Default: 256
     */
    @lombok.Builder.Default
    private int maxNewTokens = 256;

    @Override
    public String getMethodName() {
        return "DrGRPO";
    }

    @Override
    public void validate() {
        super.validate();
        if (groupSize < 2) {
            throw new IllegalStateException("groupSize must be >= 2, got: " + groupSize);
        }
        if (clipEpsilon <= 0) {
            throw new IllegalStateException("clipEpsilon must be positive, got: " + clipEpsilon);
        }
        if (maxNewTokens <= 0) {
            throw new IllegalStateException("maxNewTokens must be positive, got: " + maxNewTokens);
        }
    }
}
