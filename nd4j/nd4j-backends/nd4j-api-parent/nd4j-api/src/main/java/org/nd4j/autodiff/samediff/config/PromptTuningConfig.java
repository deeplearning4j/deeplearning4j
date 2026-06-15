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

import lombok.*;
import lombok.experimental.SuperBuilder;
import org.nd4j.common.base.Preconditions;

/**
 * Configuration for Prompt Tuning.
 * <p>
 * Prompt Tuning adds trainable "soft prompt" tokens to the input embeddings.
 * These virtual tokens have their own learnable parameters that are updated
 * during training while the rest of the model remains frozen.
 * <p>
 * Key benefits:
 * <ul>
 *   <li>Very parameter-efficient: only prompt token embeddings are trained</li>
 *   <li>Simple implementation: only modifies input layer</li>
 *   <li>Performance scales with model size</li>
 * </ul>
 *
 * <p>Example usage:</p>
 * <pre>
 * PromptTuningConfig config = PromptTuningConfig.builder()
 *     .numVirtualTokens(20)
 *     .promptTuningInit(PromptTuningInit.TEXT)
 *     .promptTuningInitText("Classify the following text:")
 *     .taskType(TaskType.SEQ_CLS)
 *     .build();
 * </pre>
 *
 * @author Adam Gibson
 * @see PeftConfig
 * @see <a href="https://arxiv.org/abs/2104.08691">Prompt Tuning Paper</a>
 */
@Data
@SuperBuilder
@NoArgsConstructor
@AllArgsConstructor
@EqualsAndHashCode(callSuper = true)
public class PromptTuningConfig extends PeftConfig {

    /**
     * Number of virtual tokens to prepend to the input.
     * More tokens = more capacity but slower inference.
     * Typical values: 8-100
     * Default: 20
     */
    @Builder.Default
    private int numVirtualTokens = 20;

    /**
     * Initialization method for prompt embeddings.
     */
    @Builder.Default
    private PromptTuningInit promptTuningInit = PromptTuningInit.RANDOM;

    /**
     * Text to use for initializing prompt embeddings when using TEXT initialization.
     * The text is tokenized and its embeddings are used to initialize the soft prompt.
     */
    private String promptTuningInitText;

    /**
     * The embedding dimension (hidden size) of the model.
     * Required for creating the prompt embedding layer.
     */
    private int tokenEmbeddingDim;

    /**
     * Random seed for initialization.
     */
    @Builder.Default
    private long randomSeed = 42;

    /**
     * Initialization methods for prompt embeddings.
     */
    public enum PromptTuningInit {
        /**
         * Initialize prompt embeddings with random values.
         */
        RANDOM,

        /**
         * Initialize prompt embeddings from tokenized text.
         * Provides better starting point if task-relevant text is available.
         */
        TEXT,

        /**
         * Initialize prompt embeddings from vocabulary embeddings.
         * Randomly samples token embeddings from the model's vocabulary.
         */
        VOCAB,

        /**
         * Initialize with zeros.
         */
        ZEROS,

        /**
         * Initialize with Kaiming/He initialization.
         */
        KAIMING
    }

    @Override
    public PeftType getPeftType() {
        return PeftType.PROMPT_TUNING;
    }

    @Override
    public long calculateTrainableParameters(long originalParamCount) {
        // Trainable params = numVirtualTokens * tokenEmbeddingDim
        if (tokenEmbeddingDim <= 0) {
            return numVirtualTokens * 768L; // Default estimate
        }
        return (long) numVirtualTokens * tokenEmbeddingDim;
    }

    @Override
    public void validate() {
        Preconditions.checkState(numVirtualTokens > 0,
            "Number of virtual tokens must be positive. Got: %s", numVirtualTokens);
        Preconditions.checkState(promptTuningInit != null,
            "Prompt tuning initialization method must be specified");
        if (promptTuningInit == PromptTuningInit.TEXT) {
            Preconditions.checkState(promptTuningInitText != null && !promptTuningInitText.isEmpty(),
                "Prompt tuning init text is required when using TEXT initialization");
        }
    }

    @Override
    public String getSummary() {
        return String.format(
            "PromptTuningConfig(numTokens=%d, init=%s, embDim=%d)",
            numVirtualTokens, promptTuningInit, tokenEmbeddingDim);
    }

    /**
     * Create a default prompt tuning configuration.
     */
    public static PromptTuningConfig defaultConfig(int embeddingDim) {
        return PromptTuningConfig.builder()
            .numVirtualTokens(20)
            .promptTuningInit(PromptTuningInit.RANDOM)
            .tokenEmbeddingDim(embeddingDim)
            .taskType(TaskType.CAUSAL_LM)
            .build();
    }

    /**
     * Create a prompt tuning configuration with text initialization.
     */
    public static PromptTuningConfig withTextInit(String initText, int embeddingDim) {
        return PromptTuningConfig.builder()
            .numVirtualTokens(20)
            .promptTuningInit(PromptTuningInit.TEXT)
            .promptTuningInitText(initText)
            .tokenEmbeddingDim(embeddingDim)
            .taskType(TaskType.CAUSAL_LM)
            .build();
    }
}
