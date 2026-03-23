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
 * Configuration for Proximal Policy Optimization (PPO) training.
 *
 * PPO uses a clipped surrogate objective with a value function baseline
 * and entropy bonus. It performs multiple epochs of updates on each batch
 * of collected trajectories using Generalized Advantage Estimation (GAE).
 *
 * Total loss = policy_loss + valueLossCoeff * value_loss - entropyCoeff * entropy
 *
 * Adam Gibson
 */
@Data
@SuperBuilder
@NoArgsConstructor
@EqualsAndHashCode(callSuper = true)
public class PPOConfig extends RLAlignmentConfig {

    /**
     * Clipping epsilon for the surrogate objective.
     * Ratio is clipped to [1 - clipEpsilon, 1 + clipEpsilon].
     * Default: 0.2
     */
    @lombok.Builder.Default
    private double clipEpsilon = 0.2;

    /**
     * Coefficient for the value function loss in the total loss.
     * Default: 0.5
     */
    @lombok.Builder.Default
    private double valueLossCoeff = 0.5;

    /**
     * Coefficient for the entropy bonus (subtracted to encourage exploration).
     * Default: 0.01
     */
    @lombok.Builder.Default
    private double entropyCoeff = 0.01;

    /**
     * Number of PPO optimization epochs per batch of collected data.
     * Default: 4
     */
    @lombok.Builder.Default
    private int ppoEpochs = 4;

    /**
     * Lambda parameter for Generalized Advantage Estimation (GAE).
     * Controls the bias-variance tradeoff: 0 = high bias, 1 = high variance.
     * Default: 0.95
     */
    @lombok.Builder.Default
    private double gaeLambda = 0.95;

    /**
     * Maximum number of new tokens to generate per completion.
     * Default: 256
     */
    @lombok.Builder.Default
    private int maxNewTokens = 256;

    /**
     * Name of the SameDiff variable containing value head output.
     * Required for computing the value function loss.
     */
    private String valueVariable;

    @Override
    public String getMethodName() {
        return "PPO";
    }

    @Override
    public void validate() {
        super.validate();
        if (clipEpsilon <= 0) {
            throw new IllegalStateException("clipEpsilon must be positive, got: " + clipEpsilon);
        }
        if (valueLossCoeff <= 0) {
            throw new IllegalStateException("valueLossCoeff must be positive, got: " + valueLossCoeff);
        }
        if (entropyCoeff <= 0) {
            throw new IllegalStateException("entropyCoeff must be positive, got: " + entropyCoeff);
        }
        if (ppoEpochs <= 0) {
            throw new IllegalStateException("ppoEpochs must be positive, got: " + ppoEpochs);
        }
        if (gaeLambda <= 0) {
            throw new IllegalStateException("gaeLambda must be positive, got: " + gaeLambda);
        }
        if (maxNewTokens <= 0) {
            throw new IllegalStateException("maxNewTokens must be positive, got: " + maxNewTokens);
        }
        if (valueVariable == null || valueVariable.isEmpty()) {
            throw new IllegalStateException("valueVariable must be specified for PPO");
        }
    }

    /**
     * Create a standard PPO configuration.
     */
    public static PPOConfig standard(String policyLogitVar, String valueVar) {
        return PPOConfig.builder()
                .policyLogitVariable(policyLogitVar)
                .valueVariable(valueVar)
                .build();
    }
}
