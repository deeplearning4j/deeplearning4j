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
 * Configuration for Odds Ratio Preference Optimization (ORPO) training.
 *
 * ORPO combines supervised fine-tuning with an odds ratio penalty to align
 * model preferences without requiring a reference model. The loss is:
 * L = sft_loss - lambda * log_sigmoid(log_odds_ratio)
 * where log_odds_ratio = log(p_chosen/(1-p_chosen)) - log(p_rejected/(1-p_rejected))
 *
 * Key advantage: no reference model needed, reducing memory and compute costs.
 *
 * Adam Gibson
 */
@Data
@SuperBuilder
@NoArgsConstructor
@EqualsAndHashCode(callSuper = true)
public class ORPOConfig extends RLAlignmentConfig {

    /**
     * Weight for the odds ratio penalty term.
     * Higher values increase the preference alignment strength.
     * Default: 0.1
     */
    @lombok.Builder.Default
    private double orpoLambda = 0.1;

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
        return "ORPO";
    }

    @Override
    public void validate() {
        super.validate();
        if (orpoLambda <= 0) {
            throw new IllegalStateException("orpoLambda must be positive, got: " + orpoLambda);
        }
        if (chosenVariable == null || chosenVariable.isEmpty()) {
            throw new IllegalStateException("chosenVariable must be specified for ORPO");
        }
        if (rejectedVariable == null || rejectedVariable.isEmpty()) {
            throw new IllegalStateException("rejectedVariable must be specified for ORPO");
        }
    }

    /**
     * Create a standard ORPO configuration.
     *
     * @param policyLogitVar  Name of the policy logit output variable
     * @param chosenVar       Name of the chosen response variable
     * @param rejectedVar     Name of the rejected response variable
     * @return An ORPOConfig with default parameters
     */
    public static ORPOConfig standard(String policyLogitVar, String chosenVar, String rejectedVar) {
        return ORPOConfig.builder()
                .policyLogitVariable(policyLogitVar)
                .chosenVariable(chosenVar)
                .rejectedVariable(rejectedVar)
                .orpoLambda(0.1)
                .useReferenceModel(false)
                .build();
    }
}
