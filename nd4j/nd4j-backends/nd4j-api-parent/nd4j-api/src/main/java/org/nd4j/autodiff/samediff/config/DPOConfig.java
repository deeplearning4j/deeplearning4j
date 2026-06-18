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
 * Configuration for Direct Preference Optimization (DPO) training.
 *
 * DPO directly optimizes the policy using preference pairs (chosen vs rejected)
 * without needing an explicit reward model. The loss is:
 * L = -log sigmoid(beta * ((pi_chosen - pi_rejected) - (ref_chosen - ref_rejected)))
 *
 * Variants:
 * - STANDARD: Original DPO (Rafailov et al., 2023)
 * - IPO: Identity Preference Optimization — uses squared hinge loss instead of sigmoid
 * - RDPO: Robust DPO — adds label smoothing for noisy preferences
 *
 * Adam Gibson
 */
@Data
@SuperBuilder
@NoArgsConstructor
@EqualsAndHashCode(callSuper = true)
public class DPOConfig extends RLAlignmentConfig {

    public enum DPOVariant {
        STANDARD,
        IPO,
        RDPO
    }

    /**
     * DPO beta parameter controlling the strength of the KL constraint.
     * Higher beta = stronger constraint to reference policy.
     * Default: 0.1
     */
    @lombok.Builder.Default
    private double beta = 0.1;

    /**
     * Label smoothing for RDPO variant.
     * 0.0 = no smoothing, 0.1 = 10% label noise tolerance.
     * Default: 0.0
     */
    @lombok.Builder.Default
    private double labelSmoothing = 0.0;

    /**
     * DPO variant to use.
     * Default: STANDARD
     */
    @lombok.Builder.Default
    private DPOVariant variant = DPOVariant.STANDARD;

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
        return "DPO-" + variant.name();
    }

    @Override
    public void validate() {
        super.validate();
        if (beta <= 0) {
            throw new IllegalStateException("beta must be positive, got: " + beta);
        }
        if (labelSmoothing < 0 || labelSmoothing > 0.5) {
            throw new IllegalStateException("labelSmoothing must be in [0, 0.5], got: " + labelSmoothing);
        }
        if (chosenVariable == null || chosenVariable.isEmpty()) {
            throw new IllegalStateException("chosenVariable must be specified for DPO");
        }
        if (rejectedVariable == null || rejectedVariable.isEmpty()) {
            throw new IllegalStateException("rejectedVariable must be specified for DPO");
        }
    }

    /**
     * Create a standard DPO configuration.
     */
    public static DPOConfig standard(String policyLogitVar, String chosenVar, String rejectedVar) {
        return DPOConfig.builder()
                .policyLogitVariable(policyLogitVar)
                .chosenVariable(chosenVar)
                .rejectedVariable(rejectedVar)
                .beta(0.1)
                .variant(DPOVariant.STANDARD)
                .build();
    }
}
