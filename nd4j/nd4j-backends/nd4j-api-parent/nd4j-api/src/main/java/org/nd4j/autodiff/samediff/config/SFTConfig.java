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
import org.nd4j.linalg.schedule.ISchedule;

import java.io.Serializable;
import java.util.Arrays;

/**
 * Configuration for Supervised Fine-Tuning (SFT).
 *
 * SFT differs from continued pretraining in one critical way: loss is computed only
 * on assistant response tokens, not on prompt or system tokens. The
 * {@link org.nd4j.linalg.dataset.curation.format.InstructionDataFormatter} produces
 * character-level loss masks ({@code 1} = train, {@code 0} = ignore). The
 * {@link org.nd4j.autodiff.samediff.training.SFTTrainingPipeline} converts these
 * character-level masks to token-level masks before constructing dataset batches.
 *
 * <p>Usage:</p>
 * <pre>
 * SFTConfig config = SFTConfig.builder()
 *     .chatTemplate(ChatTemplate.CHATML)
 *     .learningRate(2e-5)
 *     .numEpochs(3)
 *     .gradientAccumulationSteps(4)
 *     .peftConfig(LoraConfig.defaultTransformer())   // optional – null = full FT
 *     .build();
 * </pre>
 *
 * @author Adam Gibson
 * @see org.nd4j.autodiff.samediff.training.SFTTrainingPipeline
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class SFTConfig implements Serializable {

    private static final long serialVersionUID = 1L;

    // -------------------------------------------------------------------------
    // Chat / formatting
    // -------------------------------------------------------------------------

    /**
     * Chat template used to format conversations into training text.
     * Default: CHATML
     */
    @Builder.Default
    private ChatTemplate chatTemplate = ChatTemplate.CHATML;

    /**
     * Optional system message prepended to every conversation.
     * Null means no system message.
     */
    private String systemMessage;

    // -------------------------------------------------------------------------
    // Sequence / tokenisation
    // -------------------------------------------------------------------------

    /**
     * Maximum token sequence length for training examples.
     * Examples longer than this are truncated; shorter ones are padded.
     * Default: 2048
     */
    @Builder.Default
    private int maxSeqLength = 2048;

    /**
     * Whether to pack multiple short sequences into a single training example
     * to avoid padding waste. Uses {@code PaddingFreeBatchCollator} internally.
     * Default: false
     */
    @Builder.Default
    private boolean packSequences = false;

    /**
     * Whether to use the padding-free packing path (sequence packing without
     * attention-mask padding).  Requires a flash-attention-style kernel.
     * Default: false
     */
    @Builder.Default
    private boolean usePaddingFree = false;

    // -------------------------------------------------------------------------
    // Optimiser / schedule
    // -------------------------------------------------------------------------

    /**
     * Base learning rate.
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
     * Override learning-rate schedule.  When non-null this takes precedence over
     * the cosine-warmup schedule built from {@link #learningRate},
     * {@link #minLearningRate}, and {@link #warmupRatio}.
     */
    private ISchedule lrSchedule;

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
    // Training loop
    // -------------------------------------------------------------------------

    /**
     * Number of training epochs.
     * Default: 3
     */
    @Builder.Default
    private int numEpochs = 3;

    /**
     * Gradient accumulation micro-steps.
     * Effective batch size = per-device batch size × gradientAccumulationSteps.
     * Default: 4
     */
    @Builder.Default
    private int gradientAccumulationSteps = 4;

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
     * Null means no loss scaling (safe for BF16; usually needed for FP16).
     */
    private LossScaleConfig lossScaleConfig;

    // -------------------------------------------------------------------------
    // PEFT
    // -------------------------------------------------------------------------

    /**
     * PEFT configuration (e.g., {@link LoraConfig}, {@link QLoraConfig}).
     * When non-null the pipeline wraps the base model with
     * {@link org.nd4j.autodiff.samediff.peft.PeftModel}.
     * When null the entire model is fine-tuned (full SFT).
     * Default: null (full fine-tune)
     */
    private PeftConfig peftConfig;

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
        if (computeDataType == null)
            throw new IllegalStateException("computeDataType must not be null");
        if (chatTemplate == null)
            throw new IllegalStateException("chatTemplate must not be null");
        if (peftConfig != null)
            peftConfig.validate();
    }

    // -------------------------------------------------------------------------
    // Factory methods
    // -------------------------------------------------------------------------

    /**
     * Create a sensible default SFT configuration for full fine-tuning.
     */
    public static SFTConfig defaultSFT() {
        return SFTConfig.builder().build();
    }

    /**
     * Create an SFT configuration with LoRA adapters.
     *
     * @param rank LoRA rank (e.g. 16)
     */
    public static SFTConfig loraDefaults(int rank) {
        LoraConfig lora = LoraConfig.builder()
                .r(rank)
                .loraAlpha(rank * 2)
                .loraDropout(0.05)
                .targetModules(Arrays.asList("q_proj", "k_proj", "v_proj", "o_proj"))
                .taskType(TaskType.CAUSAL_LM)
                .build();
        return SFTConfig.builder()
                .peftConfig(lora)
                .build();
    }

    /**
     * Create an SFT configuration using QLoRA (quantized LoRA) for memory-efficient
     * fine-tuning on consumer hardware.
     */
    public static SFTConfig qloraDefaults() {
        QLoraConfig qlora = QLoraConfig.builder()
                .r(64)
                .loraAlpha(16)
                .loraDropout(0.1)
                .targetModules(Arrays.asList("q_proj", "k_proj", "v_proj", "o_proj"))
                .taskType(TaskType.CAUSAL_LM)
                .build();
        return SFTConfig.builder()
                .peftConfig(qlora)
                .gradientAccumulationSteps(16)   // QLoRA often needs more accum steps
                .build();
    }
}
