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
import lombok.NoArgsConstructor;
import lombok.experimental.SuperBuilder;
import org.nd4j.shade.jackson.annotation.JsonSubTypes;
import org.nd4j.shade.jackson.annotation.JsonTypeInfo;

import java.io.Serializable;

/**
 * Abstract base configuration for RL alignment training methods.
 * All RL alignment trainers (DPO, KTO, GRPO, PPO, etc.) extend this config.
 *
 * @author Eclipse Deeplearning4j Contributors
 */
@Data
@SuperBuilder
@NoArgsConstructor
@JsonTypeInfo(use = JsonTypeInfo.Id.NAME, property = "type")
@JsonSubTypes({
        @JsonSubTypes.Type(value = DPOConfig.class, name = "dpo"),
        @JsonSubTypes.Type(value = KTOConfig.class, name = "kto"),
        @JsonSubTypes.Type(value = ORPOConfig.class, name = "orpo"),
        @JsonSubTypes.Type(value = RewardModelConfig.class, name = "reward_model"),
        @JsonSubTypes.Type(value = GRPOConfig.class, name = "grpo"),
        @JsonSubTypes.Type(value = PPOConfig.class, name = "ppo"),
        @JsonSubTypes.Type(value = DAPOConfig.class, name = "dapo"),
        @JsonSubTypes.Type(value = GSPOConfig.class, name = "gspo"),
        @JsonSubTypes.Type(value = DrGRPOConfig.class, name = "drgrpo")
})
public abstract class RLAlignmentConfig implements Serializable {

    /**
     * KL divergence coefficient for regularization against reference model.
     * Higher values penalize deviation from the reference more.
     * Default: 0.1
     */
    @lombok.Builder.Default
    protected double klCoefficient = 0.1;

    /**
     * Name of the SameDiff variable containing policy logits.
     */
    protected String policyLogitVariable;

    /**
     * Whether to use a reference model for KL regularization.
     * Default: true (most methods use a reference)
     */
    @lombok.Builder.Default
    protected boolean useReferenceModel = true;

    /**
     * EMA decay rate for updating the reference model from the policy.
     * 0.0 = never update (frozen reference), 1.0 = always copy policy.
     * Typical: 0.999 for slow tracking.
     * Default: 0.0 (frozen reference)
     */
    @lombok.Builder.Default
    protected double referenceModelEmaDecay = 0.0;

    /**
     * Maximum gradient norm for gradient clipping.
     * 0.0 = no clipping.
     * Default: 1.0
     */
    @lombok.Builder.Default
    protected double maxGradNorm = 1.0;

    /**
     * Whether to normalize advantages to zero mean and unit variance.
     * Default: true
     */
    @lombok.Builder.Default
    protected boolean normalizeAdvantages = true;

    /**
     * Vocabulary size for the model. Required for computing log probabilities
     * via one-hot encoding in the SameDiff graph.
     */
    protected int vocabSize;

    /**
     * Name of the model's primary input variable (e.g., "input_ids", "input").
     * Default: "input"
     */
    @lombok.Builder.Default
    protected String inputVariable = "input";

    /**
     * Learning rate for the optimizer.
     * Default: 5e-5
     */
    @lombok.Builder.Default
    protected double learningRate = 5e-5;

    /**
     * Get the RL alignment method type name.
     */
    public abstract String getMethodName();

    /**
     * Validate the configuration.
     * @throws IllegalStateException if configuration is invalid
     */
    public void validate() {
        if (policyLogitVariable == null || policyLogitVariable.isEmpty()) {
            throw new IllegalStateException("policyLogitVariable must be specified");
        }
        if (vocabSize <= 0) {
            throw new IllegalStateException("vocabSize must be positive, got: " + vocabSize);
        }
        if (klCoefficient < 0) {
            throw new IllegalStateException("klCoefficient must be non-negative, got: " + klCoefficient);
        }
        if (maxGradNorm < 0) {
            throw new IllegalStateException("maxGradNorm must be non-negative, got: " + maxGradNorm);
        }
    }
}
