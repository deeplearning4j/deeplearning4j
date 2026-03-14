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
import org.nd4j.linalg.schedule.ISchedule;

import java.io.Serializable;

/**
 * Configuration for continued pretraining (domain adaptation).
 *
 * Continued pretraining trains on all tokens in the sequence (no masking),
 * distinguishing it from instruction tuning where only response tokens are trained.
 * This is used to adapt a pretrained model to a specific domain (e.g., medical, legal, code).
 *
 * @author Eclipse Deeplearning4j Contributors
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class ContinuedPretrainingConfig implements Serializable {

    /**
     * Chunk size for splitting text into training sequences.
     * Default: 2048
     */
    @Builder.Default
    private int chunkSize = 2048;

    /**
     * Overlap between adjacent chunks to maintain context continuity.
     * Default: 128
     */
    @Builder.Default
    private int chunkOverlap = 128;

    /**
     * Base learning rate for the optimizer.
     * Default: 5e-5
     */
    @Builder.Default
    private double learningRate = 5e-5;

    /**
     * Learning rate schedule. If null, uses cosine warmup schedule with warmupRatio.
     */
    private ISchedule lrSchedule;

    /**
     * Warmup ratio (fraction of total steps for linear warmup).
     * Used when lrSchedule is null.
     * Default: 0.1
     */
    @Builder.Default
    private double warmupRatio = 0.1;

    /**
     * Minimum learning rate for cosine decay.
     * Default: 1e-6
     */
    @Builder.Default
    private double minLearningRate = 1e-6;

    /**
     * Number of gradient accumulation steps.
     * Effective batch size = batchSize * gradientAccumulationSteps.
     * Default: 4
     */
    @Builder.Default
    private int gradientAccumulationSteps = 4;

    /**
     * Compute data type for mixed precision training.
     * Default: BFLOAT16
     */
    @Builder.Default
    private DataType computeDataType = DataType.BFLOAT16;

    /**
     * Gradient checkpointing configuration. Null = disabled.
     */
    private GradientCheckpointConfig gradientCheckpointConfig;

    /**
     * Whether to apply LoRA instead of full finetuning.
     * Default: false (full pretraining)
     */
    @Builder.Default
    private boolean useLoRA = false;

    /**
     * LoRA rank if useLoRA is true.
     * Default: 16
     */
    @Builder.Default
    private int loraRank = 16;

    /**
     * LoRA alpha if useLoRA is true.
     * Default: 32
     */
    @Builder.Default
    private int loraAlpha = 32;

    /**
     * Weight decay coefficient.
     * Default: 0.01
     */
    @Builder.Default
    private double weightDecay = 0.01;

    /**
     * Maximum gradient norm for clipping.
     * Default: 1.0
     */
    @Builder.Default
    private double maxGradNorm = 1.0;

    /**
     * Number of training epochs.
     * Default: 1
     */
    @Builder.Default
    private int numEpochs = 1;

    /**
     * Validate the configuration.
     */
    public void validate() {
        if (chunkSize <= 0) throw new IllegalStateException("chunkSize must be positive");
        if (chunkOverlap < 0) throw new IllegalStateException("chunkOverlap must be non-negative");
        if (chunkOverlap >= chunkSize) throw new IllegalStateException("chunkOverlap must be < chunkSize");
        if (learningRate <= 0) throw new IllegalStateException("learningRate must be positive");
        if (gradientAccumulationSteps < 1) throw new IllegalStateException("gradientAccumulationSteps must be >= 1");
    }
}
