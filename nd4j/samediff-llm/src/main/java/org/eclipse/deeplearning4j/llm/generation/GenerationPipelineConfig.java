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

package org.eclipse.deeplearning4j.llm.generation;

import lombok.Builder;
import lombok.Getter;
import org.nd4j.autodiff.samediff.SameDiff;
import org.eclipse.deeplearning4j.llm.config.PreprocessorConfig;
import org.eclipse.deeplearning4j.llm.tokenizer.Tokenizer;

import java.util.Set;


/**
 * Configuration for {@link GenerationPipeline}.
 *
 * <p>Encapsulates all parameters needed to set up an LLM text generation pipeline:
 * the decoder model, embedding model, tokenizer, sampling strategy, KV cache strategy,
 * and optional speculative decoding.</p>
 *
 * <p>Usage:</p>
 * <pre>{@code
 * GenerationPipelineConfig config = GenerationPipelineConfig.builder()
 *     .decoder(decoder)
 *     .embedTokens(embedTokens)
 *     .tokenizer(tokenizer)
 *     .samplingConfig(SamplingConfig.greedy())
 *     .maxNewTokens(256)
 *     .build();
 * }</pre>
 */
@Getter
@Builder
public class GenerationPipelineConfig {

    /** The decoder SameDiff model. Required. */
    private final SameDiff decoder;

    /** The token embedding SameDiff model. Required. */
    private final SameDiff embedTokens;

    /** The tokenizer for encoding/decoding text. Required. */
    private final Tokenizer tokenizer;

    /** Sampling configuration (temperature, top-k, top-p, etc.). */
    @Builder.Default
    private final SamplingConfig samplingConfig = SamplingConfig.greedy();

    /** Maximum number of new tokens to generate. */
    @Builder.Default
    private final int maxNewTokens = 256;

    /** Model hidden size. Auto-detected from embeddings if 0. */
    @Builder.Default
    private final long hiddenSize = 0;

    /** KV cache strategy. */
    @Builder.Default
    private final KvCacheStrategy kvCacheStrategy = KvCacheStrategy.STATIC;

    /** Optional model I/O configuration. Auto-discovered if null. */
    private final ModelIOConfig ioConfig;

    /** Optional speculator for speculative decoding. */
    private final Speculator speculator;

    /** Maximum speculative tokens per step (0 disables speculation). */
    @Builder.Default
    private final int maxSpeculativeTokens = 0;

    /** Additional stop token IDs beyond EOS. */
    private final Set<Integer> additionalStopTokenIds;

    /** Name of the embedding model input (auto-discovered if null). */
    private final String embedInputName;

    /** Names of the embedding model outputs (auto-discovered if null). */
    private final String[] embedOutputNames;

    /** Path to decoder ONNX/SDZ model file. Alternative to providing a pre-loaded decoder. */
    private final String decoderPath;

    /** Path to embed_tokens ONNX/SDZ model file. Alternative to providing a pre-loaded embedTokens. */
    private final String embedTokensPath;

    /** Path to draft decoder ONNX/SDZ model for speculative decoding. */
    private final String draftModelPath;

    /** Pre-loaded draft decoder model for speculative decoding. */
    private final SameDiff draftDecoder;

    /** Whether DSP (DynamicShapePlan) is enabled for this pipeline. */
    @Builder.Default
    private final boolean dspEnabled = true;

    /** Optional model loader for loading models from paths. Uses default loader if null. */
    private final GenerationPipeline.ModelLoader modelLoader;

    /** Optional preprocessor config for VLM image preprocessing. Nullable for text-only pipelines. */
    private final PreprocessorConfig preprocessorConfig;

    /** Optional path to preprocessor_config.json for auto-loading. */
    private final String preprocessorConfigPath;
}
