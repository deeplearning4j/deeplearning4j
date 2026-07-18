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

package org.nd4j.ggml.architecture;

import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.ggml.convert.ConversionOptions;
import org.nd4j.ggml.format.GGMLMetadata;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Map;
import java.util.Set;

/**
 * Interface for model architecture handlers.
 * Each implementation handles a specific model architecture (LLaMA, BERT, etc.)
 * and knows how to build the corresponding SameDiff graph.
 */
public interface ModelArchitecture {

    /**
     * Get the primary name of this architecture (e.g., "llama", "bert")
     */
    String getName();

    /**
     * Get all supported variant names for this architecture.
     * For example, LLaMA architecture might support: ["llama", "llama2", "llama3", "codellama", "mistral"]
     */
    Set<String> getSupportedVariants();

    /**
     * Check if this architecture can handle the given model metadata.
     *
     * @param metadata the model metadata
     * @return true if this architecture can handle the model
     */
    boolean canHandle(GGMLMetadata metadata);

    /**
     * Build the SameDiff graph for this architecture.
     *
     * @param metadata the model metadata
     * @param weights  map of tensor names to weight arrays
     * @param options  conversion options
     * @return the constructed SameDiff graph
     */
    SameDiff buildGraph(GGMLMetadata metadata, Map<String, INDArray> weights, ConversionOptions options);

    /**
     * Get the expected tensor name patterns for this architecture.
     * Used for mapping GGML tensor names to SameDiff variable names.
     *
     * The map keys are GGML-style names (with {layer} placeholder),
     * and values are the corresponding SameDiff names.
     *
     * Example:
     * "blk.{layer}.attn_q.weight" -> "model.layers.{layer}.self_attn.q_proj.weight"
     */
    Map<String, String> getTensorNamePatterns();

    /**
     * Get the default chat template type for this architecture.
     * Used by tests and generation pipelines to format prompts correctly.
     *
     * @return one of: "chatml", "llama2", "vicuna", "alpaca", "gemma", "phi", "plain", "none"
     */
    default String getDefaultChatTemplateType() {
        return "chatml";
    }

    /**
     * Get the system property name for the GGUF model path in tests.
     * Tests are gated on this property: {@code -D<name>=<path-to.gguf>}
     *
     * @return the system property name (e.g. "llama.gguf.path")
     */
    default String getModelSystemProperty() {
        return getName() + ".gguf.path";
    }

    /**
     * Get a reference test prompt appropriate for this architecture.
     * Returns a prompt that should produce a factual, verifiable answer.
     *
     * @return a simple test prompt
     */
    default String getReferencePrompt() {
        return "What is the capital of France?";
    }

    /**
     * Get expected substrings in the reference prompt's answer.
     * Used by degeneracy tests to verify the model produces coherent output.
     *
     * @return expected substrings (case-insensitive)
     */
    default String[] getReferenceExpected() {
        return new String[]{"Paris"};
    }

    /**
     * Get architecture-specific configuration from metadata.
     */
    default ArchitectureConfig getConfig(GGMLMetadata metadata) {
        // Prefer explicit attention.key_length from GGUF metadata for headDim.
        // This handles models where head_dim != hidden_size / num_attention_heads (e.g. Qwen3.5).
        int headDim = metadata.getAttentionKeyLength();
        int numMtpLayers = metadata.getNumMtpLayers();
        int numTargetLayers = metadata.getNumLayers() - numMtpLayers;
        if (numTargetLayers <= 0) {
            throw new IllegalArgumentException("GGUF block_count " + metadata.getNumLayers()
                    + " does not contain a target trunk after subtracting " + numMtpLayers
                    + " MTP predictor layer(s)");
        }
        return ArchitectureConfig.builder()
                .numLayers(numTargetLayers)
                .numMtpLayers(numMtpLayers)
                .hiddenSize(metadata.getHiddenSize())
                .intermediateSize(metadata.getIntermediateSize())
                .numAttentionHeads(metadata.getNumAttentionHeads())
                .numKVHeads(metadata.getNumKVHeads())
                .vocabSize(metadata.getVocabSize())
                .contextLength(metadata.getContextLength())
                .layerNormEpsilon(metadata.getLayerNormEpsilon())
                .ropeFreqBase(metadata.getRopeFreqBase())
                .ropeDimensionCount(metadata.getRopeDimensionCount())
                .headDim(headDim)
                .layerTypes(metadata.getLayerTypes())
                .fullAttentionInterval(metadata.getFullAttentionInterval())
                .ropeType(metadata.getRopeType())
                .build();
    }
}
