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

package org.nd4j.autodiff.samediff.peft;

import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.config.*;

import java.util.Arrays;
import java.util.List;

/**
 * Factory class for creating PEFT models with common configurations.
 * Provides convenient methods for creating LoRA, Prompt Tuning, and other
 * PEFT configurations with sensible defaults.
 *
 * <p>Example usage:</p>
 * <pre>
 * // Quick LoRA for transformer
 * PeftModel model = PeftModelFactory.createLoraModel(baseModel, 16);
 *
 * // LoRA with custom targets
 * PeftModel model = PeftModelFactory.createLoraModel(baseModel, 16, 32,
 *     Arrays.asList("query", "key", "value", "dense"));
 *
 * // Prompt tuning
 * PeftModel model = PeftModelFactory.createPromptTuningModel(baseModel, 20, 768);
 * </pre>
 *
 * @author Adam Gibson
 * @see PeftModel
 */
public class PeftModelFactory {

    private PeftModelFactory() {
        // Utility class
    }

    // ========== LoRA Factory Methods ==========

    /**
     * Create a LoRA model with default configuration.
     * Uses rank=16, alpha=32, and targets common attention modules.
     *
     * @param baseModel The pretrained base model
     * @return A PeftModel with LoRA applied
     */
    public static PeftModel createLoraModel(SameDiff baseModel) {
        return createLoraModel(baseModel, 16, 32, LoraConfig.TRANSFORMER_TARGET_MODULES);
    }

    /**
     * Create a LoRA model with specified rank.
     * Alpha is set to 2*rank by default.
     *
     * @param baseModel The pretrained base model
     * @param rank      The LoRA rank (r)
     * @return A PeftModel with LoRA applied
     */
    public static PeftModel createLoraModel(SameDiff baseModel, int rank) {
        return createLoraModel(baseModel, rank, rank * 2, LoraConfig.TRANSFORMER_TARGET_MODULES);
    }

    /**
     * Create a LoRA model with specified rank and alpha.
     *
     * @param baseModel The pretrained base model
     * @param rank      The LoRA rank (r)
     * @param alpha     The LoRA alpha (scaling factor)
     * @return A PeftModel with LoRA applied
     */
    public static PeftModel createLoraModel(SameDiff baseModel, int rank, int alpha) {
        return createLoraModel(baseModel, rank, alpha, LoraConfig.TRANSFORMER_TARGET_MODULES);
    }

    /**
     * Create a LoRA model with full configuration.
     *
     * @param baseModel     The pretrained base model
     * @param rank          The LoRA rank (r)
     * @param alpha         The LoRA alpha (scaling factor)
     * @param targetModules List of module names/patterns to apply LoRA to
     * @return A PeftModel with LoRA applied
     */
    public static PeftModel createLoraModel(SameDiff baseModel, int rank, int alpha,
                                            List<String> targetModules) {
        LoraConfig config = LoraConfig.builder()
            .r(rank)
            .loraAlpha(alpha)
            .loraDropout(0.05)
            .targetModules(targetModules)
            .taskType(TaskType.GENERAL)
            .build();

        return PeftModel.fromPretrained(baseModel, config);
    }

    /**
     * Create a LoRA model targeting all linear layers.
     * Good for maximum adaptation capacity.
     *
     * @param baseModel The pretrained base model
     * @param rank      The LoRA rank
     * @return A PeftModel with LoRA applied to all linear layers
     */
    public static PeftModel createLoraAllLinear(SameDiff baseModel, int rank) {
        return createLoraModel(baseModel, rank, rank * 2, LoraConfig.TRANSFORMER_ALL_LINEAR);
    }

    /**
     * Create a minimal LoRA model for memory-constrained scenarios.
     * Uses rank=4 and only targets query and value projections.
     *
     * @param baseModel The pretrained base model
     * @return A PeftModel with minimal LoRA
     */
    public static PeftModel createMinimalLora(SameDiff baseModel) {
        LoraConfig config = LoraConfig.minimal();
        return PeftModel.fromPretrained(baseModel, config);
    }

    /**
     * Create a LoRA model with rsLoRA (rank-stabilized LoRA).
     * Uses α/√r scaling instead of α/r for better stability with high ranks.
     *
     * @param baseModel     The pretrained base model
     * @param rank          The LoRA rank
     * @param targetModules Target modules
     * @return A PeftModel with rsLoRA applied
     */
    public static PeftModel createRsLoraModel(SameDiff baseModel, int rank, List<String> targetModules) {
        LoraConfig config = LoraConfig.builder()
            .r(rank)
            .loraAlpha(rank)
            .loraDropout(0.0)
            .useRsLora(true)
            .targetModules(targetModules)
            .taskType(TaskType.GENERAL)
            .build();

        return PeftModel.fromPretrained(baseModel, config);
    }

    // ========== Prompt Tuning Factory Methods ==========

    /**
     * Create a Prompt Tuning model.
     *
     * @param baseModel    The pretrained base model
     * @param numTokens    Number of soft prompt tokens
     * @param embeddingDim Embedding dimension of the model
     * @return A PeftModel with Prompt Tuning applied
     */
    public static PeftModel createPromptTuningModel(SameDiff baseModel, int numTokens, int embeddingDim) {
        PromptTuningConfig config = PromptTuningConfig.builder()
            .numVirtualTokens(numTokens)
            .tokenEmbeddingDim(embeddingDim)
            .promptTuningInit(PromptTuningConfig.PromptTuningInit.RANDOM)
            .taskType(TaskType.CAUSAL_LM)
            .build();

        return PeftModel.fromPretrained(baseModel, config);
    }

    /**
     * Create a Prompt Tuning model with text initialization.
     *
     * @param baseModel    The pretrained base model
     * @param numTokens    Number of soft prompt tokens
     * @param embeddingDim Embedding dimension
     * @param initText     Text to initialize prompts from
     * @return A PeftModel with Prompt Tuning applied
     */
    public static PeftModel createPromptTuningModel(SameDiff baseModel, int numTokens,
                                                    int embeddingDim, String initText) {
        PromptTuningConfig config = PromptTuningConfig.builder()
            .numVirtualTokens(numTokens)
            .tokenEmbeddingDim(embeddingDim)
            .promptTuningInit(PromptTuningConfig.PromptTuningInit.TEXT)
            .promptTuningInitText(initText)
            .taskType(TaskType.CAUSAL_LM)
            .build();

        return PeftModel.fromPretrained(baseModel, config);
    }

    // ========== Prefix Tuning Factory Methods ==========

    /**
     * Create a Prefix Tuning model.
     *
     * @param baseModel    The pretrained base model
     * @param numTokens    Number of prefix tokens
     * @param numLayers    Number of transformer layers
     * @param hiddenSize   Hidden size of the model
     * @return A PeftModel with Prefix Tuning applied
     */
    public static PeftModel createPrefixTuningModel(SameDiff baseModel, int numTokens,
                                                    int numLayers, int hiddenSize) {
        PrefixTuningConfig config = PrefixTuningConfig.builder()
            .numVirtualTokens(numTokens)
            .numLayers(numLayers)
            .hiddenSize(hiddenSize)
            .prefixProjection(true)
            .encoderHiddenSize(hiddenSize / 2)
            .taskType(TaskType.CAUSAL_LM)
            .build();

        return PeftModel.fromPretrained(baseModel, config);
    }

    /**
     * Create a Prefix Tuning model for seq2seq tasks.
     *
     * @param baseModel    The pretrained base model
     * @param numTokens    Number of prefix tokens
     * @param numLayers    Number of transformer layers
     * @param numHeads     Number of attention heads
     * @param hiddenSize   Hidden size of the model
     * @return A PeftModel with Prefix Tuning for seq2seq
     */
    public static PeftModel createPrefixTuningSeq2Seq(SameDiff baseModel, int numTokens,
                                                      int numLayers, int numHeads, int hiddenSize) {
        PrefixTuningConfig config = PrefixTuningConfig.forSeq2Seq(numLayers, numHeads, hiddenSize);
        config.setNumVirtualTokens(numTokens);
        return PeftModel.fromPretrained(baseModel, config);
    }

    // ========== Adapter Factory Methods ==========

    /**
     * Create a model with Adapter layers.
     *
     * @param baseModel   The pretrained base model
     * @param adapterSize Size of the adapter bottleneck
     * @param hiddenSize  Hidden size of the model
     * @param numLayers   Number of layers to add adapters to
     * @return A PeftModel with Adapters applied
     */
    public static PeftModel createAdapterModel(SameDiff baseModel, int adapterSize,
                                               int hiddenSize, int numLayers) {
        AdapterConfig config = AdapterConfig.builder()
            .adapterSize(adapterSize)
            .hiddenSize(hiddenSize)
            .numLayers(numLayers)
            .adapterActivation("relu")
            .adapterAfterAttention(true)
            .adapterAfterFeedforward(true)
            .taskType(TaskType.GENERAL)
            .build();

        return PeftModel.fromPretrained(baseModel, config);
    }

    // ========== IA³ Factory Methods ==========

    /**
     * Create an IA³ model.
     *
     * @param baseModel     The pretrained base model
     * @param attentionDim  Attention hidden dimension
     * @param feedforwardDim Feedforward hidden dimension
     * @return A PeftModel with IA³ applied
     */
    public static PeftModel createIA3Model(SameDiff baseModel, int attentionDim, int feedforwardDim) {
        IA3Config config = IA3Config.defaultConfig(attentionDim, feedforwardDim);
        return PeftModel.fromPretrained(baseModel, config);
    }

    // ========== Convenience Configuration Builders ==========

    /**
     * Create a LoRA configuration builder with sensible defaults.
     */
    public static LoraConfig.LoraConfigBuilder<?, ?> loraConfigBuilder() {
        return LoraConfig.builder()
            .r(16)
            .loraAlpha(32)
            .loraDropout(0.05)
            .targetModules(LoraConfig.TRANSFORMER_TARGET_MODULES)
            .taskType(TaskType.GENERAL);
    }

    /**
     * Create a LoRA configuration for classification tasks.
     */
    public static LoraConfig loraForClassification(int rank, List<String> targetModules) {
        return LoraConfig.builder()
            .r(rank)
            .loraAlpha(rank * 2)
            .loraDropout(0.1)
            .targetModules(targetModules)
            .taskType(TaskType.SEQ_CLS)
            .build();
    }

    /**
     * Create a LoRA configuration for language modeling.
     */
    public static LoraConfig loraForCausalLM(int rank) {
        return LoraConfig.builder()
            .r(rank)
            .loraAlpha(rank * 2)
            .loraDropout(0.05)
            .targetModules(LoraConfig.TRANSFORMER_ALL_LINEAR)
            .taskType(TaskType.CAUSAL_LM)
            .build();
    }

    /**
     * Create a LoRA configuration for question answering.
     */
    public static LoraConfig loraForQuestionAnswering(int rank) {
        return LoraConfig.builder()
            .r(rank)
            .loraAlpha(rank * 2)
            .loraDropout(0.1)
            .targetModules(LoraConfig.TRANSFORMER_TARGET_MODULES)
            .taskType(TaskType.QUESTION_ANS)
            .build();
    }
}
