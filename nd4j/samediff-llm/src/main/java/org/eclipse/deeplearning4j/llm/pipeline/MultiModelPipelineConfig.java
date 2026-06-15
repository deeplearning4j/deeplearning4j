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

package org.eclipse.deeplearning4j.llm.pipeline;

import lombok.Builder;
import lombok.Getter;
import org.eclipse.deeplearning4j.llm.generation.GenerationPipeline;
import org.eclipse.deeplearning4j.llm.generation.kvcache.KvCacheStrategy;
import org.eclipse.deeplearning4j.llm.generation.sampling.SamplingConfig;
import org.eclipse.deeplearning4j.llm.tokenizer.Tokenizer;
import org.nd4j.autodiff.samediff.SameDiff;

import java.util.*;

/**
 * Configuration for a multi-model pipeline.
 *
 * <p>Defines the models, their roles, and execution parameters for a pipeline
 * that chains multiple models together.</p>
 *
 * <p>Usage:</p>
 * <pre>{@code
 * MultiModelPipelineConfig config = MultiModelPipelineConfig.builder()
 *     .tokenizer(tokenizer)
 *     .kvCacheStrategy(KvCacheStrategy.STATIC)
 *     .build();
 *
 * MultiModelPipeline pipeline = new MultiModelPipeline(config);
 * pipeline.registerModel("summarizer", ModelType.SUMMARIZER, summarizerModel);
 * pipeline.registerModel("classifier", ModelType.CLASSIFIER, classifierModel);
 *
 * PipelineStage stage1 = new PipelineStage("summarizer", "summarization");
 * PipelineStage stage2 = new PipelineStage("classifier", "classification");
 *
 * PipelineResult result = pipeline.execute(List.of(stage1, stage2), "Long input text...");
 * }</pre>
 */
@Getter
@Builder
public class MultiModelPipelineConfig {

    /** The tokenizer for encoding/decoding text. Required. */
    private final Tokenizer tokenizer;

    /** Maximum tokens per model invocation. */
    @Builder.Default
    private final int maxTokensPerModel = 512;

    /** Whether to enable DSP (Dynamic Shape Plan) optimizations. */
    @Builder.Default
    private final boolean dspEnabled = true;

    /** Whether to enable CUDA graphs for DSP execution. */
    @Builder.Default
    private final boolean cudaGraphsEnabled = false;

    /** Whether to free model memory after each stage (reduces peak memory). */
    @Builder.Default
    private final boolean freeMemoryPerStage = true;

    /** Whether to trim GPU memory pools between stages. */
    @Builder.Default
    private final boolean trimMemoryBetweenStages = true;

    /** Sampling configuration for all models. */
    @Builder.Default
    private final SamplingConfig samplingConfig = SamplingConfig.greedy();

    /** KV cache strategy for all models (STATIC, PAGED, QUANTIZED, TURBOQUANT). */
    @Builder.Default
    private final KvCacheStrategy kvCacheStrategy = KvCacheStrategy.STATIC;

    /** TurboQuant bit budget per coordinate for TURBOQUANT KV cache strategy. Defaults to 3. */
    @Builder.Default
    private final int turboQuantBits = 3;

    /** Model registry: maps model IDs to model configurations. */
    @Builder.Default
    private final Map<String, RegisteredModel> modelRegistry = new HashMap<>();

    /** Custom prompt templates (overrides ModelType defaults). */
    @Builder.Default
    private final Map<String, String> customPromptTemplates = new HashMap<>();

    /** Additional parameters for specific model roles. */
    @Builder.Default
    private final Map<String, Map<String, String>> roleParameters = new HashMap<>();

    /**
     * Represents a registered model in the pipeline.
     */
    @Getter
    @Builder
    public static class RegisteredModel {
        /** The model ID (unique identifier). */
        private final String modelId;

        /** The model type (defines role and prompt template). */
        private final ModelType modelType;

        /** The loaded SameDiff model. */
        private final SameDiff model;

        /** Optional: path to load the model from (alternative to pre-loaded). */
        private final String modelPath;

        /** Whether this model is owned by the pipeline (should be closed). */
        @Builder.Default
        private final boolean ownedByPipeline = true;

        /** Whether DSP is enabled for this specific model. */
        @Builder.Default
        private final boolean dspEnabled = true;

        /** Optional: custom GenerationPipeline for this model. */
        private final GenerationPipeline generationPipeline;
    }
}
