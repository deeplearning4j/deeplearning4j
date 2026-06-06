/*
 *  ******************************************************************************
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

package org.deeplearning4j.ui.module.modelmanager;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.Map;

/**
 * Represents a model entry in the local registry.
 * Tracks download state, model metadata (architecture, quantization, context length),
 * tokenizer configuration (chat template, special tokens, vocab), and all downloaded files.
 *
 * Borrows patterns from kompile-model-staging's ModelEntry + ModelMetadata.
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class ModelRegistryEntry {

    public enum ModelStatus {
        DOWNLOADING, READY, ERROR, DELETED
    }

    public enum ModelType {
        LLM_GGUF, LLM_SAMEDIFF, ENCODER, CROSS_ENCODER, VLM, OTHER
    }

    // --- Identity & source ---
    private String id;
    private String source;
    private String repo;
    private String format;
    private String localPath;
    private ModelStatus status;
    private long downloadedAt;
    private long fileSizeBytes;
    private String checksum;
    private String filename;
    private String description;

    // --- Cached SDZ path (import result) ---
    private String sdzPath;
    private boolean sdzCached;

    // --- Model type classification ---
    private ModelType modelType;

    // --- Tokenizer configuration ---
    private TokenizerConfig tokenizer;

    // --- Model metadata (architecture, quantization, etc.) ---
    private ModelMetadata metadata;

    // --- Downloaded files map: key ("model","tokenizer","tokenizer_config") → relative path ---
    private Map<String, String> files;

    /**
     * Tokenizer configuration for inference.
     * Populated from tokenizer_config.json, GGUF metadata, or defaults.
     */
    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class TokenizerConfig {
        /** Tokenizer implementation type: "huggingface", "sentencepiece", "wordpiece", "clip" */
        private String tokenizerType;
        /** Relative path to primary vocab file (tokenizer.json, vocab.txt, or tokenizer.model) */
        private String vocabFile;
        /** Jinja2 chat template string from tokenizer_config.json or GGUF metadata */
        private String chatTemplate;
        /** Chat template format name: "chatml", "llama2", "vicuna", "alpaca" */
        private String chatTemplateFormat;
        @Builder.Default
        private int bosTokenId = -1;
        @Builder.Default
        private int eosTokenId = -1;
        @Builder.Default
        private int padTokenId = -1;
        private int vocabSize;
        @Builder.Default
        private boolean addBosToken = true;
        @Builder.Default
        private boolean addEosToken = false;
    }

    /**
     * Model architecture metadata.
     * For GGUF files, populated automatically from GGMLMetadata.
     * For ONNX/SameDiff, populated from graph inspection.
     */
    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class ModelMetadata {
        /** Architecture name: "llama", "qwen2", "mistral", "phi3", "gemma2" */
        private String architecture;
        /** Quantization type name: "Q4_K_M", "Q8_0", "F16", "F32" */
        private String quantizationType;
        private int contextLength;
        private int embeddingDimension;
        private int numLayers;
        private int numAttentionHeads;
        private int numKVHeads;
        private int feedForwardDimension;
        /** 0 for dense models, >0 for Mixture-of-Experts */
        private int expertCount;
        /** Estimated total parameter count */
        private long parameterCount;
        /** Original file format: "gguf", "onnx", "safetensors", "sdz" */
        private String originalFormat;
    }
}
