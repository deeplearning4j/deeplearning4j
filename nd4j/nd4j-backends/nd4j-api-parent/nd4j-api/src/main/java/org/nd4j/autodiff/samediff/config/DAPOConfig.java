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
 * Configuration for Decoupled Alignment Policy Optimization (DAPO) training.
 *
 * DAPO extends GRPO with several improvements:
 * - Asymmetric clipping (different lower/upper bounds) for better exploration
 * - Token-level KL divergence instead of sequence-level for finer control
 * - Dynamic sampling that skips groups with uniform rewards (no signal)
 * - Overlong filtering to remove completions that exceed max length
 *
 * @author Eclipse Deeplearning4j Contributors
 */
@Data
@SuperBuilder
@NoArgsConstructor
@EqualsAndHashCode(callSuper = true)
public class DAPOConfig extends RLAlignmentConfig {

    /**
     * Lower clip bound for the asymmetric clipping.
     * Ratio is clipped to [1 - clipEpsilonLow, 1 + clipEpsilonHigh].
     * Default: 0.1
     */
    @lombok.Builder.Default
    private double clipEpsilonLow = 0.1;

    /**
     * Upper clip bound for the asymmetric clipping.
     * Allows more aggressive upward updates than downward.
     * Default: 0.28
     */
    @lombok.Builder.Default
    private double clipEpsilonHigh = 0.28;

    /**
     * Whether to use token-level KL divergence instead of sequence-level.
     * Token-level KL provides finer-grained regularization.
     * Default: true
     */
    @lombok.Builder.Default
    private boolean tokenLevelKL = true;

    /**
     * Whether to skip groups where all rewards are equal (no learning signal).
     * Default: true
     */
    @lombok.Builder.Default
    private boolean dynamicSampling = true;

    /**
     * Whether to filter out completions that exceed maxNewTokens.
     * Default: true
     */
    @lombok.Builder.Default
    private boolean overlongFiltering = true;

    /**
     * Number of completions to generate per prompt for group-based advantage estimation.
     * Default: 8
     */
    @lombok.Builder.Default
    private int groupSize = 8;

    /**
     * Maximum number of new tokens to generate per completion.
     * Default: 256
     */
    @lombok.Builder.Default
    private int maxNewTokens = 256;

    @Override
    public String getMethodName() {
        return "DAPO";
    }

    @Override
    public void validate() {
        super.validate();
        if (clipEpsilonLow <= 0) {
            throw new IllegalStateException("clipEpsilonLow must be positive, got: " + clipEpsilonLow);
        }
        if (clipEpsilonHigh <= 0) {
            throw new IllegalStateException("clipEpsilonHigh must be positive, got: " + clipEpsilonHigh);
        }
        if (groupSize < 2) {
            throw new IllegalStateException("groupSize must be >= 2, got: " + groupSize);
        }
        if (maxNewTokens <= 0) {
            throw new IllegalStateException("maxNewTokens must be positive, got: " + maxNewTokens);
        }
    }
}
