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

import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.ggml.export.ExportOptions;
import org.nd4j.ggml.format.GGMLDataType;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Export architecture handler for LLaMA and LLaMA-derived models.
 * Handles name mapping from SameDiff variable names to GGML tensor names.
 *
 * Supports: LLaMA, LLaMA 2, LLaMA 3, CodeLLaMA, Mistral, Mixtral, Qwen, etc.
 */
@Slf4j
public class LLaMAExportArchitecture implements ExportArchitecture {

    // SameDiff -> GGML name mapping patterns (reverse of import)
    private static final Map<Pattern, String> NAME_PATTERNS = new LinkedHashMap<>();

    static {
        // Embeddings
        NAME_PATTERNS.put(Pattern.compile("model\\.embed_tokens\\.weight"), "token_embd.weight");
        NAME_PATTERNS.put(Pattern.compile("lm_head\\.weight"), "output.weight");
        NAME_PATTERNS.put(Pattern.compile("model\\.norm\\.weight"), "output_norm.weight");

        // Attention layers (separate Q/K/V)
        NAME_PATTERNS.put(Pattern.compile("model\\.layers\\.(\\d+)\\.self_attn\\.q_proj\\.weight"),
                "blk.{layer}.attn_q.weight");
        NAME_PATTERNS.put(Pattern.compile("model\\.layers\\.(\\d+)\\.self_attn\\.k_proj\\.weight"),
                "blk.{layer}.attn_k.weight");
        NAME_PATTERNS.put(Pattern.compile("model\\.layers\\.(\\d+)\\.self_attn\\.v_proj\\.weight"),
                "blk.{layer}.attn_v.weight");
        NAME_PATTERNS.put(Pattern.compile("model\\.layers\\.(\\d+)\\.self_attn\\.o_proj\\.weight"),
                "blk.{layer}.attn_output.weight");

        // Fused QKV attention (Qwen3.5 and similar)
        NAME_PATTERNS.put(Pattern.compile("model\\.layers\\.(\\d+)\\.self_attn\\.qkv_proj\\.weight"),
                "blk.{layer}.attn_qkv.weight");
        NAME_PATTERNS.put(Pattern.compile("model\\.layers\\.(\\d+)\\.self_attn\\.qkv_proj\\.bias"),
                "blk.{layer}.attn_qkv.bias");
        NAME_PATTERNS.put(Pattern.compile("model\\.layers\\.(\\d+)\\.self_attn\\.gate\\.weight"),
                "blk.{layer}.attn_gate.weight");

        // FFN layers
        NAME_PATTERNS.put(Pattern.compile("model\\.layers\\.(\\d+)\\.mlp\\.gate_proj\\.weight"),
                "blk.{layer}.ffn_gate.weight");
        NAME_PATTERNS.put(Pattern.compile("model\\.layers\\.(\\d+)\\.mlp\\.up_proj\\.weight"),
                "blk.{layer}.ffn_up.weight");
        NAME_PATTERNS.put(Pattern.compile("model\\.layers\\.(\\d+)\\.mlp\\.down_proj\\.weight"),
                "blk.{layer}.ffn_down.weight");

        // Normalization layers
        NAME_PATTERNS.put(Pattern.compile("model\\.layers\\.(\\d+)\\.input_layernorm\\.weight"),
                "blk.{layer}.attn_norm.weight");
        NAME_PATTERNS.put(Pattern.compile("model\\.layers\\.(\\d+)\\.post_attention_layernorm\\.weight"),
                "blk.{layer}.ffn_norm.weight");
    }

    private static final Set<String> RECOGNIZED_PATTERNS = Set.of(
            "model.embed_tokens.weight",
            "model.layers.*.self_attn.q_proj.weight",
            "model.layers.*.mlp.gate_proj.weight",
            "model.norm.weight"
    );

    @Override
    public String getArchitectureName() {
        return "llama";
    }

    @Override
    public Set<String> getRecognizedVariablePatterns() {
        return RECOGNIZED_PATTERNS;
    }

    @Override
    public boolean canHandle(SameDiff model) {
        if (model == null) return false;

        // Check for LLaMA-style variable names
        for (String varName : model.variableNames()) {
            if (varName.contains("self_attn.q_proj") ||
                varName.contains("mlp.gate_proj") ||
                varName.contains("model.embed_tokens")) {
                return true;
            }
        }
        return false;
    }

    @Override
    public String mapVariableName(String sdVarName) {
        for (Map.Entry<Pattern, String> entry : NAME_PATTERNS.entrySet()) {
            Matcher matcher = entry.getKey().matcher(sdVarName);
            if (matcher.matches()) {
                String ggmlName = entry.getValue();
                // Replace {layer} placeholder with captured layer number
                if (matcher.groupCount() > 0) {
                    ggmlName = ggmlName.replace("{layer}", matcher.group(1));
                }
                return ggmlName;
            }
        }
        return null;
    }

    @Override
    public Map<String, String> getNameMappingPatterns() {
        Map<String, String> patterns = new LinkedHashMap<>();
        for (Map.Entry<Pattern, String> entry : NAME_PATTERNS.entrySet()) {
            patterns.put(entry.getKey().pattern(), entry.getValue());
        }
        return patterns;
    }

    @Override
    public Map<String, Object> buildMetadata(SameDiff model, ExportOptions options) {
        Map<String, Object> metadata = new LinkedHashMap<>();

        // General metadata
        metadata.put("general.architecture", "llama");
        metadata.put("general.name", options.getModelName() != null ?
                options.getModelName() : "exported_model");
        metadata.put("general.quantization_version", 2);
        metadata.put("general.alignment", options.getAlignment());

        if (options.getModelAuthor() != null) {
            metadata.put("general.author", options.getModelAuthor());
        }
        if (options.getModelDescription() != null) {
            metadata.put("general.description", options.getModelDescription());
        }

        // Infer architecture config from model
        ArchitectureConfig config = inferConfig(model);

        // Architecture-specific metadata
        if (config.getContextLength() > 0) {
            metadata.put("llama.context_length", config.getContextLength());
        }
        if (config.getHiddenSize() > 0) {
            metadata.put("llama.embedding_length", config.getHiddenSize());
        }
        if (config.getNumLayers() > 0) {
            metadata.put("llama.block_count", config.getNumLayers());
        }
        if (config.getIntermediateSize() > 0) {
            metadata.put("llama.feed_forward_length", config.getIntermediateSize());
        }
        if (config.getNumAttentionHeads() > 0) {
            metadata.put("llama.attention.head_count", config.getNumAttentionHeads());
        }
        if (config.getNumKVHeads() > 0) {
            metadata.put("llama.attention.head_count_kv", config.getNumKVHeads());
        }
        metadata.put("llama.attention.layer_norm_rms_epsilon", config.getLayerNormEpsilon());
        metadata.put("llama.rope.freq_base", config.getRopeFreqBase());
        if (config.getVocabSize() > 0) {
            metadata.put("llama.vocab_size", config.getVocabSize());
        }

        return metadata;
    }

    @Override
    public GGMLDataType getRecommendedQuantType(String tensorName, ExportOptions options) {
        // Keep embeddings and output projections in higher precision
        if (tensorName.contains("token_embd") || tensorName.contains("output.weight")) {
            // For quantized modes, use at least Q8_0 for embeddings
            if (options.getQuantizationType() == ExportOptions.QuantizationType.Q4_0 ||
                options.getQuantizationType() == ExportOptions.QuantizationType.Q4_1 ||
                options.getQuantizationType() == ExportOptions.QuantizationType.Q4_K) {
                return GGMLDataType.GGML_TYPE_Q8_0;
            }
        }

        // Keep normalization weights in FP16/FP32
        if (tensorName.contains("norm")) {
            return GGMLDataType.GGML_TYPE_F32;
        }

        return options.getQuantizationType().toGGMLDataType();
    }

    @Override
    public List<String> validateForExport(SameDiff model) {
        List<String> errors = new ArrayList<>();

        // Check for required tensors
        boolean hasEmbedding = false;
        boolean hasLayers = false;
        boolean hasNorm = false;

        for (String varName : model.variableNames()) {
            SDVariable var = model.getVariable(varName);
            if (var.getVariableType() != VariableType.VARIABLE &&
                var.getVariableType() != VariableType.CONSTANT) {
                continue;
            }

            if (varName.contains("embed_tokens")) {
                hasEmbedding = true;
            }
            if (varName.contains("model.layers.")) {
                hasLayers = true;
            }
            if (varName.equals("model.norm.weight")) {
                hasNorm = true;
            }
        }

        if (!hasEmbedding) {
            errors.add("Missing embedding layer (model.embed_tokens.weight)");
        }
        if (!hasLayers) {
            errors.add("Missing transformer layers (model.layers.*)");
        }
        if (!hasNorm) {
            errors.add("Missing final normalization (model.norm.weight)");
        }

        return errors;
    }

    @Override
    public ArchitectureConfig inferConfig(SameDiff model) {
        int vocabSize = 0;
        int hiddenSize = 0;
        int numLayers = 0;
        int numHeads = 0;
        int numKVHeads = 0;
        int intermediateSize = 0;

        // Infer from tensor shapes
        for (String varName : model.variableNames()) {
            SDVariable var = model.getVariable(varName);
            if (var.getVariableType() != VariableType.VARIABLE &&
                var.getVariableType() != VariableType.CONSTANT) {
                continue;
            }

            INDArray arr = var.getArr();
            if (arr == null) continue;

            long[] shape = arr.shape();

            // Token embeddings: [vocab_size, hidden_size]
            if (varName.equals("model.embed_tokens.weight") && shape.length == 2) {
                vocabSize = (int) shape[0];
                hiddenSize = (int) shape[1];
            }

            // Count layers
            Matcher layerMatcher = Pattern.compile("model\\.layers\\.(\\d+)").matcher(varName);
            if (layerMatcher.find()) {
                int layerNum = Integer.parseInt(layerMatcher.group(1));
                numLayers = Math.max(numLayers, layerNum + 1);
            }

            // Infer head counts from Q/K shapes
            if (varName.matches("model\\.layers\\.0\\.self_attn\\.q_proj\\.weight") && shape.length == 2) {
                int qOutSize = (int) shape[0];
                int headDim = 128; // Default for LLaMA
                if (hiddenSize > 0) {
                    headDim = hiddenSize / (qOutSize / hiddenSize);
                    numHeads = qOutSize / headDim;
                }
            }

            if (varName.matches("model\\.layers\\.0\\.self_attn\\.k_proj\\.weight") && shape.length == 2) {
                int kOutSize = (int) shape[0];
                int headDim = 128;
                if (hiddenSize > 0 && numHeads > 0) {
                    headDim = hiddenSize / numHeads;
                    numKVHeads = kOutSize / headDim;
                }
            }

            // Infer intermediate size from FFN
            if (varName.matches("model\\.layers\\.0\\.mlp\\.gate_proj\\.weight") && shape.length == 2) {
                intermediateSize = (int) shape[0];
            }
        }

        // Use defaults for missing values
        if (numKVHeads == 0) numKVHeads = numHeads;

        return ArchitectureConfig.builder()
                .vocabSize(vocabSize)
                .hiddenSize(hiddenSize)
                .numLayers(numLayers)
                .numAttentionHeads(numHeads)
                .numKVHeads(numKVHeads)
                .intermediateSize(intermediateSize)
                .contextLength(4096) // Default
                .layerNormEpsilon(1e-5f)
                .ropeFreqBase(10000.0f)
                .build();
    }
}
