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
 * Configuration for reward model training on preference data.
 *
 * Supports three reward model training objectives:
 * - BRADLEY_TERRY: -log sigmoid(r_chosen - r_rejected) (standard preference learning)
 * - REGRESSION: MSE(r_chosen - r_rejected - margin) (margin-based regression)
 * - CONTRASTIVE: max(0, margin - (r_chosen - r_rejected)) (hinge/contrastive loss)
 *
 * Reward models are typically trained as a preprocessing step before RL-based
 * alignment methods like PPO or GRPO.
 *
 * Adam Gibson
 */
@Data
@SuperBuilder
@NoArgsConstructor
@EqualsAndHashCode(callSuper = true)
public class RewardModelConfig extends RLAlignmentConfig {

    public enum RewardType {
        /** Standard Bradley-Terry preference model: -log sigmoid(r_chosen - r_rejected) */
        BRADLEY_TERRY,
        /** Regression with target margin: MSE(r_chosen - r_rejected - margin) */
        REGRESSION,
        /** Contrastive hinge loss: max(0, margin - (r_chosen - r_rejected)) */
        CONTRASTIVE
    }

    /**
     * Reward model training objective.
     * Default: BRADLEY_TERRY
     */
    @lombok.Builder.Default
    private RewardType rewardType = RewardType.BRADLEY_TERRY;

    /**
     * Name of the variable containing chosen response token IDs.
     */
    private String chosenVariable;

    /**
     * Name of the variable containing rejected response token IDs.
     */
    private String rejectedVariable;

    /**
     * Margin for contrastive and regression loss types.
     * For CONTRASTIVE: minimum desired gap between chosen and rejected rewards.
     * For REGRESSION: target difference between chosen and rejected rewards.
     * Default: 0.0
     */
    @lombok.Builder.Default
    private double margin = 0.0;

    /**
     * Name of the output variable containing the scalar reward score.
     */
    private String rewardOutputVariable;

    @Override
    public String getMethodName() {
        return "RewardModel-" + rewardType;
    }

    @Override
    public void validate() {
        super.validate();
        if (chosenVariable == null || chosenVariable.isEmpty()) {
            throw new IllegalStateException("chosenVariable must be specified for RewardModel");
        }
        if (rejectedVariable == null || rejectedVariable.isEmpty()) {
            throw new IllegalStateException("rejectedVariable must be specified for RewardModel");
        }
        if (rewardOutputVariable == null || rewardOutputVariable.isEmpty()) {
            throw new IllegalStateException("rewardOutputVariable must be specified for RewardModel");
        }
    }

    /**
     * Create a standard Bradley-Terry reward model configuration.
     *
     * @param policyLogitVar    Name of the policy logit output variable
     * @param chosenVar         Name of the chosen response variable
     * @param rejectedVar       Name of the rejected response variable
     * @param rewardOutputVar   Name of the reward head output variable
     * @return A RewardModelConfig with Bradley-Terry objective
     */
    public static RewardModelConfig bradleyTerry(String policyLogitVar, String chosenVar,
                                                  String rejectedVar, String rewardOutputVar) {
        return RewardModelConfig.builder()
                .policyLogitVariable(policyLogitVar)
                .chosenVariable(chosenVar)
                .rejectedVariable(rejectedVar)
                .rewardOutputVariable(rewardOutputVar)
                .rewardType(RewardType.BRADLEY_TERRY)
                .useReferenceModel(false)
                .build();
    }
}
