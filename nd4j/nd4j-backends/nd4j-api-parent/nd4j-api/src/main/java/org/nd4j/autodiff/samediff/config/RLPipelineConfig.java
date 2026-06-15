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

import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.AllArgsConstructor;
import org.nd4j.linalg.api.buffer.DataType;

/**
 * Configuration for the {@link org.nd4j.autodiff.samediff.training.RLAlignmentPipeline}.
 *
 * <p>Controls the outer training loop: epochs, learning rate schedule, gradient
 * accumulation, PEFT wrapping, and logging/evaluation frequency. The RL-method-specific
 * hyperparameters live in the corresponding {@link RLAlignmentConfig} subclass
 * (e.g. {@link DPOConfig}, {@link GRPOConfig}).
 *
 * <p>Defaults are chosen for RLHF fine-tuning of language models:
 * <ul>
 *   <li>1 epoch (preference data is often small and single-pass)</li>
 *   <li>5e-7 learning rate (lower than SFT to preserve alignment)</li>
 *   <li>BFLOAT16 compute dtype (stable, no loss-scaling needed)</li>
 *   <li>Gradient accumulation = 1 (override for large effective batch sizes)</li>
 * </ul>
 *
 * <p>Example:
 * <pre>
 * RLPipelineConfig pipelineCfg = RLPipelineConfig.builder()
 *     .numEpochs(1)
 *     .learningRate(5e-7)
 *     .gradientAccumulationSteps(4)
 *     .computeDataType(DataType.BFLOAT16)
 *     .logEveryNSteps(10)
 *     .build();
 * </pre>
 *
 * @author Adam Gibson
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class RLPipelineConfig {

    /**
     * Number of passes over the training dataset.
     * Default: 1
     */
    @Builder.Default
    private int numEpochs = 1;

    /**
     * Base learning rate for the Adam optimiser inside each trainer.
     * Overrides the learning rate in the {@link RLAlignmentConfig} when the
     * pipeline creates or reconfigures the trainer.
     * Default: 5e-7
     */
    @Builder.Default
    private double learningRate = 5e-7;

    /**
     * Fraction of total steps used for linear warmup before the main schedule.
     * Default: 0.1 (10% warmup)
     */
    @Builder.Default
    private double warmupRatio = 0.1;

    /**
     * Number of micro-batches to accumulate gradients over before each weight update.
     * Set to > 1 when the per-step batch does not fit in memory.
     * Default: 1
     */
    @Builder.Default
    private int gradientAccumulationSteps = 1;

    /**
     * Compute dtype for training.  BFLOAT16 is recommended for modern GPUs; use
     * FLOAT for CPU-only training or debugging.
     * Default: BFLOAT16
     */
    @Builder.Default
    private DataType computeDataType = DataType.BFLOAT16;

    /**
     * L2 weight decay applied to all trainable parameters.
     * 0.0 disables regularisation.
     * Default: 0.0
     */
    @Builder.Default
    private double weightDecay = 0.0;

    /**
     * Optional PEFT configuration.  When non-null the policy model is wrapped with
     * {@link org.nd4j.autodiff.samediff.peft.PeftModel} before training begins and
     * only the adapter parameters are updated.
     * Default: null (full fine-tune)
     */
    @Builder.Default
    private PeftConfig peftConfig = null;

    /**
     * Emit a log line every N global training steps.
     * Default: 10
     */
    @Builder.Default
    private int logEveryNSteps = 10;

    /**
     * Run evaluation every N global training steps.
     * -1 means never evaluate during training.
     * Default: -1
     */
    @Builder.Default
    private int evaluateEveryNSteps = -1;

    /**
     * Hard cap on the total number of training steps across all epochs.
     * -1 means unlimited (determined by epochs * steps-per-epoch).
     * Default: -1
     */
    @Builder.Default
    private int maxSteps = -1;

    /**
     * Validate the configuration and throw {@link IllegalStateException} if invalid.
     */
    public void validate() {
        if (numEpochs < 1) {
            throw new IllegalStateException("numEpochs must be >= 1, got: " + numEpochs);
        }
        if (learningRate <= 0) {
            throw new IllegalStateException("learningRate must be positive, got: " + learningRate);
        }
        if (warmupRatio < 0 || warmupRatio >= 1.0) {
            throw new IllegalStateException(
                    "warmupRatio must be in [0, 1), got: " + warmupRatio);
        }
        if (gradientAccumulationSteps < 1) {
            throw new IllegalStateException(
                    "gradientAccumulationSteps must be >= 1, got: " + gradientAccumulationSteps);
        }
        if (weightDecay < 0) {
            throw new IllegalStateException(
                    "weightDecay must be non-negative, got: " + weightDecay);
        }
        if (computeDataType == null) {
            throw new IllegalStateException("computeDataType must not be null");
        }
        if (logEveryNSteps < 1) {
            throw new IllegalStateException(
                    "logEveryNSteps must be >= 1, got: " + logEveryNSteps);
        }
        if (peftConfig != null) {
            peftConfig.validate();
        }
    }

    /**
     * Convenience factory for a typical full-fine-tune RL alignment pipeline.
     */
    public static RLPipelineConfig defaults() {
        return RLPipelineConfig.builder().build();
    }
}
