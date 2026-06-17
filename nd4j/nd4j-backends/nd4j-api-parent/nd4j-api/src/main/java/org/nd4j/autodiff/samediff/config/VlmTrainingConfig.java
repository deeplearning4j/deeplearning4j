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
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.dataset.curation.format.ChatTemplate;

import java.io.Serializable;

/**
 * Configuration for Vision-Language Model (VLM) supervised fine-tuning.
 *
 * <p>VLM training differs from text-only SFT in that training examples contain both
 * image features (from the vision encoder) and text tokens (prompt + response). The
 * training config controls optimizer, schedule, mixed precision, and vision-specific
 * parameters such as image resolution and patch size.</p>
 *
 * <p>Typical factory method usage:</p>
 * <pre>{@code
 * // LoRA-efficient fine-tuning (recommended default)
 * VlmTrainingConfig config = VlmTrainingConfig.loraTraining();
 *
 * // Full fine-tune of all parameters
 * VlmTrainingConfig config = VlmTrainingConfig.fullFineTune();
 *
 * // Train only the vision-language projector
 * VlmTrainingConfig config = VlmTrainingConfig.projectorOnly();
 *
 * // Custom builder
 * VlmTrainingConfig config = VlmTrainingConfig.builder()
 *     .learningRate(1e-5)
 *     .numEpochs(2)
 *     .imageSize(336)
 *     .patchSize(14)
 *     .build();
 * }</pre>
 *
 * @author Adam Gibson
 * @see VlmFineTuneConfig
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class VlmTrainingConfig implements Serializable {

    private static final long serialVersionUID = 1L;

    // -------------------------------------------------------------------------
    // Optimiser / schedule
    // -------------------------------------------------------------------------

    /**
     * Base learning rate for the AdamW optimiser.
     * Default: 2e-5
     */
    @Builder.Default
    private double learningRate = 2e-5;

    /**
     * Minimum learning rate at the end of cosine decay.
     * Default: 0.0
     */
    @Builder.Default
    private double minLearningRate = 0.0;

    /**
     * Fraction of total training steps used for linear learning-rate warm-up.
     * Default: 0.03
     */
    @Builder.Default
    private double warmupRatio = 0.03;

    /**
     * Number of training epochs.
     * Default: 1  (VLM fine-tuning usually needs fewer epochs than text-only SFT)
     */
    @Builder.Default
    private int numEpochs = 1;

    /**
     * Gradient accumulation micro-steps.
     * Effective batch size = per-device micro-batch × gradientAccumulationSteps.
     * Default: 8
     */
    @Builder.Default
    private int gradientAccumulationSteps = 8;

    /**
     * AdamW weight-decay coefficient.
     * Default: 0.01
     */
    @Builder.Default
    private double weightDecay = 0.01;

    /**
     * Maximum gradient norm for gradient clipping.
     * Default: 1.0
     */
    @Builder.Default
    private double maxGradNorm = 1.0;

    // -------------------------------------------------------------------------
    // Sequence / tokenisation
    // -------------------------------------------------------------------------

    /**
     * Maximum token sequence length for text portions of training examples.
     * Image tokens are prepended before text tokens; the combined length should
     * not exceed the model's context window.
     * Default: 2048
     */
    @Builder.Default
    private int maxSeqLength = 2048;

    // -------------------------------------------------------------------------
    // Vision
    // -------------------------------------------------------------------------

    /**
     * Input image size (height = width, assumed square).
     * The vision encoder receives images resized to {@code imageSize x imageSize}.
     * Default: 224
     */
    @Builder.Default
    private int imageSize = 224;

    /**
     * Patch size used by the ViT-style vision encoder.
     * {@code imageSize} must be divisible by {@code patchSize}.
     * Default: 14
     */
    @Builder.Default
    private int patchSize = 14;

    // -------------------------------------------------------------------------
    // Mixed precision
    // -------------------------------------------------------------------------

    /**
     * Compute data type for forward/backward passes.
     * Use BFLOAT16 for Ampere/Hopper GPUs; FLOAT16 for older NVIDIA GPUs.
     * Default: BFLOAT16
     */
    @Builder.Default
    private DataType computeDataType = DataType.BFLOAT16;

    /**
     * Optional loss-scale configuration for mixed-precision training.
     * Null means no explicit loss scaling (safe for BF16; usually needed for FP16).
     */
    private LossScaleConfig lossScaleConfig;

    // -------------------------------------------------------------------------
    // Chat / formatting
    // -------------------------------------------------------------------------

    /**
     * Chat template used to format the text portion of training examples.
     * Default: CHATML
     */
    @Builder.Default
    private ChatTemplate chatTemplate = ChatTemplate.CHATML;

    // -------------------------------------------------------------------------
    // Validation
    // -------------------------------------------------------------------------

    /**
     * Validate the configuration and throw {@link IllegalStateException} on errors.
     */
    public void validate() {
        if (learningRate <= 0)
            throw new IllegalStateException("learningRate must be positive, got: " + learningRate);
        if (minLearningRate < 0)
            throw new IllegalStateException("minLearningRate must be >= 0, got: " + minLearningRate);
        if (minLearningRate >= learningRate)
            throw new IllegalStateException(
                    "minLearningRate (" + minLearningRate + ") must be < learningRate (" + learningRate + ")");
        if (warmupRatio < 0 || warmupRatio > 1)
            throw new IllegalStateException("warmupRatio must be in [0, 1], got: " + warmupRatio);
        if (numEpochs < 1)
            throw new IllegalStateException("numEpochs must be >= 1, got: " + numEpochs);
        if (maxSeqLength < 1)
            throw new IllegalStateException("maxSeqLength must be >= 1, got: " + maxSeqLength);
        if (gradientAccumulationSteps < 1)
            throw new IllegalStateException(
                    "gradientAccumulationSteps must be >= 1, got: " + gradientAccumulationSteps);
        if (weightDecay < 0)
            throw new IllegalStateException("weightDecay must be >= 0, got: " + weightDecay);
        if (maxGradNorm <= 0)
            throw new IllegalStateException("maxGradNorm must be positive, got: " + maxGradNorm);
        if (imageSize <= 0)
            throw new IllegalStateException("imageSize must be positive, got: " + imageSize);
        if (patchSize <= 0)
            throw new IllegalStateException("patchSize must be positive, got: " + patchSize);
        if (imageSize % patchSize != 0)
            throw new IllegalStateException(
                    "imageSize (" + imageSize + ") must be divisible by patchSize (" + patchSize + ")");
        if (computeDataType == null)
            throw new IllegalStateException("computeDataType must not be null");
        if (chatTemplate == null)
            throw new IllegalStateException("chatTemplate must not be null");
    }

    /**
     * Compute the number of image patch tokens per example.
     *
     * @return (imageSize / patchSize)^2
     */
    public int computeNumPatches() {
        int patchesPerSide = imageSize / patchSize;
        return patchesPerSide * patchesPerSide;
    }

    // -------------------------------------------------------------------------
    // Factory methods
    // -------------------------------------------------------------------------

    /**
     * Create a LoRA-efficient VLM training config.
     * Suitable for fine-tuning with frozen vision encoder and LoRA on the LLM backbone.
     * This is the recommended default for most hardware setups.
     *
     * @return VlmTrainingConfig optimised for LoRA training
     */
    public static VlmTrainingConfig loraTraining() {
        return VlmTrainingConfig.builder()
                .learningRate(2e-5)
                .numEpochs(1)
                .gradientAccumulationSteps(8)
                .computeDataType(DataType.BFLOAT16)
                .build();
    }

    /**
     * Create a full fine-tune VLM training config.
     * All model parameters are trained (vision encoder, projector, LLM backbone).
     * Requires significantly more GPU memory than {@link #loraTraining()}.
     *
     * @return VlmTrainingConfig for full fine-tuning
     */
    public static VlmTrainingConfig fullFineTune() {
        return VlmTrainingConfig.builder()
                .learningRate(5e-6)
                .numEpochs(1)
                .gradientAccumulationSteps(16)
                .computeDataType(DataType.BFLOAT16)
                .build();
    }

    /**
     * Create a projector-only VLM training config.
     * Only the vision-language projector is trained; vision encoder and LLM are frozen.
     * Fastest and most memory-efficient option.
     *
     * @return VlmTrainingConfig for projector-only training
     */
    public static VlmTrainingConfig projectorOnly() {
        return VlmTrainingConfig.builder()
                .learningRate(1e-4)
                .numEpochs(1)
                .gradientAccumulationSteps(4)
                .computeDataType(DataType.BFLOAT16)
                .build();
    }
}
