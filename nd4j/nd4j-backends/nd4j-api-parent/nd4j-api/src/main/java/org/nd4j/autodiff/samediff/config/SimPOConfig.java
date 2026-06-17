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
 * Configuration for Simple Preference Optimization (SimPO) training.
 *
 * SimPO (arXiv:2405.14734) eliminates the need for a reference model by using
 * length-normalized log probabilities and a target reward margin. The loss is:
 * L = -log sigmoid(beta * (avg_log_prob_chosen - avg_log_prob_rejected - gamma))
 *
 * Where:
 * - avg_log_prob = sum(log_probs) / length  (length-normalized)
 * - gamma is the target reward margin (default: 1.0)
 * - beta controls the sharpness of the sigmoid (default: 2.5)
 *
 * Key advantage over DPO: no reference model needed, reducing memory and
 * compute costs while incorporating length normalization for better calibration.
 *
 * Adam Gibson
 */
@Data
@SuperBuilder
@NoArgsConstructor
@EqualsAndHashCode(callSuper = true)
public class SimPOConfig extends RLAlignmentConfig {

    /**
     * Sharpness parameter controlling the sigmoid slope.
     * Higher beta = sharper preference boundary.
     * Recommended range: 2.0-2.5.
     * Default: 2.5
     */
    @lombok.Builder.Default
    private double beta = 2.5;

    /**
     * Target reward margin — the minimum log-probability gap required between
     * chosen and rejected sequences (after length normalization) for zero loss.
     * Recommended range: 0.5-1.4.
     * Default: 1.0
     */
    @lombok.Builder.Default
    private double gamma = 1.0;

    /**
     * Name of the variable containing chosen response token IDs.
     */
    private String chosenVariable;

    /**
     * Name of the variable containing rejected response token IDs.
     */
    private String rejectedVariable;

    @Override
    public String getMethodName() {
        return "SimPO";
    }

    @Override
    public void validate() {
        super.validate();
        if (beta <= 0) {
            throw new IllegalStateException("beta must be positive, got: " + beta);
        }
        if (gamma < 0) {
            throw new IllegalStateException("gamma must be non-negative, got: " + gamma);
        }
        if (chosenVariable == null || chosenVariable.isEmpty()) {
            throw new IllegalStateException("chosenVariable must be specified for SimPO");
        }
        if (rejectedVariable == null || rejectedVariable.isEmpty()) {
            throw new IllegalStateException("rejectedVariable must be specified for SimPO");
        }
    }

    /**
     * Create a standard SimPO configuration.
     *
     * @param policyLogitVar  Name of the policy logit output variable
     * @param chosenVar       Name of the chosen response variable
     * @param rejectedVar     Name of the rejected response variable
     * @return A SimPOConfig with default parameters (beta=2.5, gamma=1.0)
     */
    public static SimPOConfig standard(String policyLogitVar, String chosenVar, String rejectedVar) {
        return SimPOConfig.builder()
                .policyLogitVariable(policyLogitVar)
                .chosenVariable(chosenVar)
                .rejectedVariable(rejectedVar)
                .beta(2.5)
                .gamma(1.0)
                .useReferenceModel(false)
                .build();
    }

    /**
     * Create a standard SimPO configuration with an explicit vocabulary size.
     *
     * @param policyLogitVar  Name of the policy logit output variable
     * @param chosenVar       Name of the chosen response variable
     * @param rejectedVar     Name of the rejected response variable
     * @param vocabSize       Vocabulary size of the model
     * @return A SimPOConfig with default parameters (beta=2.5, gamma=1.0)
     */
    public static SimPOConfig standard(String policyLogitVar, String chosenVar, String rejectedVar, int vocabSize) {
        return SimPOConfig.builder()
                .policyLogitVariable(policyLogitVar)
                .chosenVariable(chosenVar)
                .rejectedVariable(rejectedVar)
                .beta(2.5)
                .gamma(1.0)
                .vocabSize(vocabSize)
                .useReferenceModel(false)
                .build();
    }
}
