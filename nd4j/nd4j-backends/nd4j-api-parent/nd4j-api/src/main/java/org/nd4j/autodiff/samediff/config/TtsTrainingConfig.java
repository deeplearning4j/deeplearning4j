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
import org.nd4j.linalg.api.buffer.DataType;

import java.io.Serializable;

/**
 * Training configuration for Text-to-Speech (TTS) model fine-tuning.
 *
 * <p>Controls the optimizer schedule, batch configuration, gradient accumulation,
 * and optional LoRA/PEFT integration for TTS decoder adaption.
 *
 * <p>Works in conjunction with {@link TtsFineTuneConfig} which controls the audio
 * preprocessing parameters (mel spectrogram, sample rate, etc.).
 *
 * <p>Example usage:
 * <pre>
 * TtsTrainingConfig config = TtsTrainingConfig.builder()
 *     .learningRate(1e-4)
 *     .warmupSteps(500)
 *     .numEpochs(10)
 *     .batchSize(8)
 *     .build();
 *
 * // With LoRA
 * TtsTrainingConfig loraConfig = TtsTrainingConfig.builder()
 *     .learningRate(2e-4)
 *     .peftConfig(LoraConfig.builder().r(16).loraAlpha(32).build())
 *     .build();
 * </pre>
 *
 * @author Eclipse Deeplearning4j Contributors
 * @see TtsFineTuneConfig
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class TtsTrainingConfig implements Serializable {

    /**
     * Peak learning rate for the optimizer.
     * Default: 1e-4
     */
    @Builder.Default
    private double learningRate = 1e-4;

    /**
     * Minimum learning rate for cosine decay schedule.
     * Default: 1e-6
     */
    @Builder.Default
    private double minLearningRate = 1e-6;

    /**
     * Number of linear warmup steps before cosine decay begins.
     * Default: 500
     */
    @Builder.Default
    private int warmupSteps = 500;

    /**
     * Number of training epochs.
     * Default: 10
     */
    @Builder.Default
    private int numEpochs = 10;

    /**
     * Per-device batch size (number of audio examples per gradient step).
     * Default: 8
     */
    @Builder.Default
    private int batchSize = 8;

    /**
     * Maximum audio duration in seconds.
     * Audio longer than this is truncated during dataset construction.
     * Default: 30.0
     */
    @Builder.Default
    private double maxAudioLengthSec = 30.0;

    /**
     * Number of gradient accumulation steps.
     * Effective batch size = batchSize * gradientAccumulationSteps.
     * Default: 2
     */
    @Builder.Default
    private int gradientAccumulationSteps = 2;

    /**
     * Compute data type for mixed-precision training.
     * Use BFLOAT16 for modern hardware (A100, H100); FLOAT16 for older GPUs.
     * Default: BFLOAT16
     */
    @Builder.Default
    private DataType computeDataType = DataType.BFLOAT16;

    /**
     * L2 weight decay coefficient (AdamW style).
     * Default: 0.01
     */
    @Builder.Default
    private double weightDecay = 0.01;

    /**
     * Optional PEFT (LoRA/adapter) configuration for the TTS decoder.
     * When null, all unfrozen parameters are fully trained.
     * Default: null (no PEFT)
     */
    private PeftConfig peftConfig;

    /**
     * Validate this configuration, throwing {@link IllegalStateException} on invalid values.
     */
    public void validate() {
        Preconditions.checkState(learningRate > 0,
                "learningRate must be positive, got: %s", learningRate);
        Preconditions.checkState(minLearningRate > 0,
                "minLearningRate must be positive, got: %s", minLearningRate);
        Preconditions.checkState(minLearningRate <= learningRate,
                "minLearningRate (%s) must be <= learningRate (%s)", minLearningRate, learningRate);
        Preconditions.checkState(warmupSteps >= 0,
                "warmupSteps must be >= 0, got: %s", warmupSteps);
        Preconditions.checkState(numEpochs > 0,
                "numEpochs must be positive, got: %s", numEpochs);
        Preconditions.checkState(batchSize > 0,
                "batchSize must be positive, got: %s", batchSize);
        Preconditions.checkState(maxAudioLengthSec > 0,
                "maxAudioLengthSec must be positive, got: %s", maxAudioLengthSec);
        Preconditions.checkState(gradientAccumulationSteps >= 1,
                "gradientAccumulationSteps must be >= 1, got: %s", gradientAccumulationSteps);
        Preconditions.checkState(weightDecay >= 0,
                "weightDecay must be >= 0, got: %s", weightDecay);
        if (peftConfig != null) {
            peftConfig.validate();
        }
    }
}
