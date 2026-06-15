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
import org.nd4j.common.base.Preconditions;

/**
 * Configuration for Vision-Language Model GRPO (Group Relative Policy Optimization) training.
 *
 * <p>VLM-GRPO extends standard GRPO to handle multimodal inputs. The key additions over
 * {@link GRPOConfig} are:</p>
 * <ul>
 *   <li>A {@link VlmFineTuneConfig} to control how the vision encoder, projector, and
 *       LLM backbone are trained (frozen vs. LoRA vs. full fine-tune)</li>
 *   <li>{@code imageAwareReward}: whether the reward function receives the image as part
 *       of the context or only the text completion</li>
 *   <li>{@code completionsPerImage}: how many text completions are sampled per input image
 *       for group scoring (analogous to GRPO's groupSize but image-centric)</li>
 * </ul>
 *
 * <p>Example usage:</p>
 * <pre>{@code
 * VlmGRPOConfig config = VlmGRPOConfig.builder()
 *     .policyLogitVariable("logits")
 *     .vocabSize(32000)
 *     .groupSize(4)
 *     .vlmConfig(VlmFineTuneConfig.loraOnly())
 *     .completionsPerImage(4)
 *     .imageAwareReward(true)
 *     .build();
 *
 * VlmGRPOTrainer trainer = new VlmGRPOTrainer(policyModel, referenceModel, config,
 *     samplingStrategy, rewardFunction);
 * }</pre>
 *
 * @author Adam Gibson
 * @see GRPOConfig
 * @see VlmFineTuneConfig
 */
@Data
@SuperBuilder
@NoArgsConstructor
@EqualsAndHashCode(callSuper = true)
public class VlmGRPOConfig extends GRPOConfig {

    /**
     * VLM fine-tuning configuration controlling vision encoder freeze/train behavior,
     * projector training, and LoRA configs for each component.
     * When null, VLM-specific behavior uses defaults (frozen vision encoder, trained projector).
     */
    private VlmFineTuneConfig vlmConfig;

    /**
     * Whether to include the image in the reward computation.
     * When true, the reward function receives both the image and the text completion.
     * When false, only the text completion is scored (image-blind reward).
     * Default: true
     */
    @lombok.Builder.Default
    private boolean imageAwareReward = true;

    /**
     * Number of text completions to generate per image for group scoring.
     * Analogous to GRPO's groupSize but framed around image-centric batching.
     * Multiple completions per image allow z-score advantage normalization
     * within each image's group.
     * Default: 4
     */
    @lombok.Builder.Default
    private int completionsPerImage = 4;

    /**
     * Name of the SameDiff placeholder variable holding image tensor inputs.
     * Default: "image_input"
     */
    @lombok.Builder.Default
    private String imageInputVariable = "image_input";

    @Override
    public String getMethodName() {
        return "VLM-GRPO";
    }

    @Override
    public void validate() {
        super.validate();
        Preconditions.checkState(completionsPerImage >= 2,
                "completionsPerImage must be >= 2 for group relative scoring, got: %s",
                completionsPerImage);
        Preconditions.checkState(imageInputVariable != null && !imageInputVariable.isEmpty(),
                "imageInputVariable must not be null or empty");
        if (vlmConfig != null) {
            vlmConfig.validate();
        }
    }

    /**
     * Get the effective VLM fine-tuning config, returning a default if not set.
     *
     * @return the configured VlmFineTuneConfig, or a default loraOnly config if null
     */
    public VlmFineTuneConfig effectiveVlmConfig() {
        return vlmConfig != null ? vlmConfig : VlmFineTuneConfig.loraOnly();
    }
}
