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

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.nd4j.common.base.Preconditions;

import java.io.Serializable;

/**
 * Configuration for Vision-Language Model (VLM) fine-tuning.
 *
 * <p>VLM fine-tuning extends text-only LoRA/QLoRA training to vision-language models
 * by providing control over which components are trained:</p>
 * <ul>
 *   <li>Vision encoder — processes image patches into embeddings (typically frozen)</li>
 *   <li>Projection layer — maps vision embeddings into the LLM's embedding space</li>
 *   <li>LLM backbone — can be fully fine-tuned or adapted via LoRA</li>
 * </ul>
 *
 * <p>Example usage:</p>
 * <pre>{@code
 * // Standard LoRA fine-tuning: frozen vision encoder + LoRA on LLM
 * VlmFineTuneConfig config = VlmFineTuneConfig.loraOnly();
 *
 * // Full fine-tuning of all components
 * VlmFineTuneConfig config = VlmFineTuneConfig.fullFinetune();
 *
 * // Custom config: train projector + LoRA on both vision encoder and LLM
 * VlmFineTuneConfig config = VlmFineTuneConfig.builder()
 *     .freezeVisionEncoder(false)
 *     .trainProjector(true)
 *     .llmLoraConfig(LoraConfig.defaultTransformer())
 *     .visionLoraConfig(LoraConfig.minimal())
 *     .imageResolution(448)
 *     .build();
 * }</pre>
 *
 * @author Adam Gibson
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class VlmFineTuneConfig implements Serializable {
    private static final long serialVersionUID = 1L;

    /**
     * Whether to freeze the vision encoder during training.
     * When true, vision encoder weights receive no gradient updates.
     * This is the standard approach: only train the projector and LLM backbone.
     * Default: true
     */
    @Builder.Default
    private boolean freezeVisionEncoder = true;

    /**
     * Whether to train the vision-language projector.
     * The projector maps vision embeddings to the LLM's embedding dimension.
     * Default: true
     */
    @Builder.Default
    private boolean trainProjector = true;

    /**
     * Whether to share weights between the vision encoder and LLM backbone.
     * Useful in memory-constrained settings when architectures support sharing.
     * Default: false
     */
    @Builder.Default
    private boolean shareWeights = false;

    /**
     * LoRA configuration for the LLM backbone.
     * When null, the LLM backbone is either fully fine-tuned or frozen depending on context.
     */
    private LoraConfig llmLoraConfig;

    /**
     * LoRA configuration for the vision encoder.
     * Only used when freezeVisionEncoder=false.
     * When null and freezeVisionEncoder=false, the vision encoder is fully fine-tuned.
     */
    private LoraConfig visionLoraConfig;

    /**
     * Image resolution for the vision encoder (height and width, assumed square).
     * Must be divisible by patchSize.
     * Default: 384
     */
    @Builder.Default
    private int imageResolution = 384;

    /**
     * Patch size for the vision encoder (ViT-style patch extraction).
     * Must divide evenly into imageResolution.
     * Default: 14
     */
    @Builder.Default
    private int patchSize = 14;

    /**
     * Maximum number of image tokens (patches) per example.
     * Computed as (imageResolution / patchSize)^2 by default.
     * Default: 576 = (384/14 rounded down to 27)^2
     */
    @Builder.Default
    private int maxImageTokens = 576;

    /**
     * Name of the SameDiff variable that holds the vision encoder output features.
     * Default: "vision_features"
     */
    @Builder.Default
    private String visionOutputVariable = "vision_features";

    /**
     * Name of the SameDiff variable that holds the projected vision features
     * (after the projection layer maps to LLM embedding space).
     * Default: "projected_vision_features"
     */
    @Builder.Default
    private String projectorOutputVariable = "projected_vision_features";

    /**
     * Create a standard LoRA-only VLM fine-tuning config.
     * Freezes the vision encoder, trains the projector, and applies LoRA to the LLM backbone.
     * This is the recommended default for most VLM fine-tuning tasks.
     *
     * @return VlmFineTuneConfig with frozen vision encoder and default transformer LoRA
     */
    public static VlmFineTuneConfig loraOnly() {
        return builder()
                .freezeVisionEncoder(true)
                .trainProjector(true)
                .llmLoraConfig(LoraConfig.defaultTransformer())
                .build();
    }

    /**
     * Create a full fine-tuning config.
     * Trains all components: vision encoder, projector, and LLM backbone.
     * Requires significantly more GPU memory than loraOnly().
     *
     * @return VlmFineTuneConfig with all components trainable
     */
    public static VlmFineTuneConfig fullFinetune() {
        return builder()
                .freezeVisionEncoder(false)
                .trainProjector(true)
                .build();
    }

    /**
     * Create a projector-only fine-tuning config.
     * Freezes both the vision encoder and LLM backbone, only trains the projection layer.
     * Fastest and most memory-efficient option.
     *
     * @return VlmFineTuneConfig with only the projector trainable
     */
    public static VlmFineTuneConfig projectorOnly() {
        return builder()
                .freezeVisionEncoder(true)
                .trainProjector(true)
                .llmLoraConfig(null)
                .build();
    }

    /**
     * Validate this configuration.
     *
     * @throws IllegalStateException if any configuration invariant is violated
     */
    public void validate() {
        Preconditions.checkState(imageResolution > 0,
                "imageResolution must be positive, got: %s", imageResolution);
        Preconditions.checkState(patchSize > 0,
                "patchSize must be positive, got: %s", patchSize);
        Preconditions.checkState(imageResolution % patchSize == 0,
                "imageResolution (%s) must be divisible by patchSize (%s)",
                imageResolution, patchSize);
        Preconditions.checkState(maxImageTokens > 0,
                "maxImageTokens must be positive, got: %s", maxImageTokens);
        Preconditions.checkState(visionOutputVariable != null && !visionOutputVariable.isEmpty(),
                "visionOutputVariable must not be null or empty");
        Preconditions.checkState(projectorOutputVariable != null && !projectorOutputVariable.isEmpty(),
                "projectorOutputVariable must not be null or empty");

        if (llmLoraConfig != null) {
            llmLoraConfig.validate();
        }
        if (visionLoraConfig != null) {
            Preconditions.checkState(!freezeVisionEncoder,
                    "visionLoraConfig is set but freezeVisionEncoder=true; "
                    + "set freezeVisionEncoder=false to train vision encoder with LoRA");
            visionLoraConfig.validate();
        }
        if (!trainProjector && llmLoraConfig == null && freezeVisionEncoder) {
            throw new IllegalStateException(
                    "No trainable components: freezeVisionEncoder=true, trainProjector=false, "
                    + "llmLoraConfig=null. At least one component must be trainable.");
        }
    }

    /**
     * Compute the expected number of patches for a square image.
     *
     * @return number of patches per image = (imageResolution / patchSize)^2
     */
    public int computeNumPatches() {
        int patchesPerSide = imageResolution / patchSize;
        return patchesPerSide * patchesPerSide;
    }

    /**
     * Compute the patch feature dimension (raw flattened patch size).
     *
     * @param channels number of image channels (e.g., 3 for RGB)
     * @return patch feature dimension = channels * patchSize * patchSize
     */
    public int computePatchDim(int channels) {
        return channels * patchSize * patchSize;
    }

    @Override
    public String toString() {
        return String.format(
                "VlmFineTuneConfig(freezeVision=%s, trainProjector=%s, shareWeights=%s, "
                + "resolution=%d, patchSize=%d, maxImageTokens=%d, llmLora=%s, visionLora=%s)",
                freezeVisionEncoder, trainProjector, shareWeights,
                imageResolution, patchSize, maxImageTokens,
                llmLoraConfig != null ? llmLoraConfig.getSummary() : "null",
                visionLoraConfig != null ? visionLoraConfig.getSummary() : "null");
    }
}
