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
 * Configuration for Kahneman-Tversky Optimization (KTO) training.
 *
 * KTO aligns models using individual examples labeled as desirable or undesirable,
 * rather than requiring explicit preference pairs. The loss applies asymmetric weighting:
 * - For desirable samples: 1 - sigmoid(beta * (r - KL_ref))
 * - For undesirable samples: 1 - sigmoid(beta * (KL_ref - r))
 * where r = log(pi/ref) for each sample.
 *
 * Loss aversion controls the relative weight of undesirable examples, inspired
 * by Kahneman and Tversky's prospect theory.
 *
 * @author Eclipse Deeplearning4j Contributors
 */
@Data
@SuperBuilder
@NoArgsConstructor
@EqualsAndHashCode(callSuper = true)
public class KTOConfig extends RLAlignmentConfig {

    /**
     * Loss aversion weight for undesirable responses.
     * Values > 1.0 penalize undesirable responses more heavily.
     * Default: 1.0
     */
    @lombok.Builder.Default
    private double lossAversion = 1.0;

    /**
     * Beta parameter for desirable samples controlling KL constraint strength.
     * Default: 0.1
     */
    @lombok.Builder.Default
    private double betaDesirable = 0.1;

    /**
     * Beta parameter for undesirable samples controlling KL constraint strength.
     * Default: 0.1
     */
    @lombok.Builder.Default
    private double betaUndesirable = 0.1;

    /**
     * Name of the variable containing desirability labels.
     * Values should be 1 for desirable (good) and 0 for undesirable (bad) samples.
     */
    private String desirabilityVariable;

    @Override
    public String getMethodName() {
        return "KTO";
    }

    @Override
    public void validate() {
        super.validate();
        if (betaDesirable <= 0) {
            throw new IllegalStateException("betaDesirable must be positive, got: " + betaDesirable);
        }
        if (betaUndesirable <= 0) {
            throw new IllegalStateException("betaUndesirable must be positive, got: " + betaUndesirable);
        }
        if (desirabilityVariable == null || desirabilityVariable.isEmpty()) {
            throw new IllegalStateException("desirabilityVariable must be specified for KTO");
        }
    }

    /**
     * Create a standard KTO configuration with default betas.
     *
     * @param policyLogitVar   Name of the policy logit output variable
     * @param desirabilityVar  Name of the desirability label variable (1=good, 0=bad)
     * @return A KTOConfig with default parameters
     */
    public static KTOConfig standard(String policyLogitVar, String desirabilityVar) {
        return KTOConfig.builder()
                .policyLogitVariable(policyLogitVar)
                .desirabilityVariable(desirabilityVar)
                .betaDesirable(0.1)
                .betaUndesirable(0.1)
                .build();
    }
}
