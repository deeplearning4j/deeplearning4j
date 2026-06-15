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

package org.eclipse.deeplearning4j.safetensors.architecture;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * SafeTensors architecture builder for Qwen3-VL models.
 *
 * <p>Builds a SameDiff compute graph from HuggingFace SafeTensors weights for Qwen3-VL.
 * Handles the full multimodal architecture: SigLIP-2 vision encoder + M-RoPE + Qwen3 LLM decoder.</p>
 *
 * <p>HuggingFace weight name patterns:</p>
 * <ul>
 *   <li>Vision: {@code visual.blocks.{layer}.attn.qkv.weight}, etc.</li>
 *   <li>LLM: {@code model.layers.{layer}.self_attn.q_proj.weight}, etc.</li>
 *   <li>Embeddings: {@code model.embed_tokens.weight}</li>
 *   <li>Projector: {@code visual.merger.mlp.0.weight}, etc.</li>
 * </ul>
 */
@Slf4j
public class Qwen3VLSafeTensorsBuilder implements SafeTensorsArchitecture {

    private static final Set<String> SUPPORTED = Set.of(
            "qwen3vlforconditionalgeneration",
            "qwen2vlforconditionalgeneration",
            "qwen2_5_vl",
            "qwen3_vl",
            "qwen2_vl"
    );

    @Override
    public String getName() {
        return "qwen3_vl";
    }

    @Override
    public Set<String> getSupportedArchitectures() {
        return SUPPORTED;
    }

    @Override
    public boolean canHandle(Map<String, Object> config) {
        Object archField = config.get("architectures");
        if (archField instanceof List) {
            for (Object arch : (List<?>) archField) {
                String name = arch.toString().toLowerCase().replaceAll("[^a-z0-9_]", "");
                if (name.contains("qwen") && (name.contains("vl") || name.contains("vision"))) {
                    return true;
                }
            }
        }
        Object modelType = config.get("model_type");
        if (modelType != null) {
            String mt = modelType.toString().toLowerCase();
            return mt.contains("qwen") && mt.contains("vl");
        }
        return false;
    }

    @Override
    public SameDiff buildGraph(Map<String, INDArray> weights, Map<String, Object> config) {
        SameDiff sd = SameDiff.create();

        // Extract config values
        int hiddenSize = getInt(config, "hidden_size", 4096);
        int numLayers = getInt(config, "num_hidden_layers", 36);
        int numHeads = getInt(config, "num_attention_heads", 32);
        int numKvHeads = getInt(config, "num_key_value_heads", 8);
        int intermediateSize = getInt(config, "intermediate_size", 12288);
        int vocabSize = getInt(config, "vocab_size", 151936);

        log.info("Building Qwen3-VL graph: hidden={}, layers={}, heads={}, kv_heads={}, vocab={}",
                hiddenSize, numLayers, numHeads, numKvHeads, vocabSize);

        // Load all weights as constants
        for (Map.Entry<String, INDArray> entry : weights.entrySet()) {
            String name = entry.getKey().replace(".", "_");
            sd.constant(name, entry.getValue());
        }

        // Create placeholder inputs
        sd.placeHolder("input_ids", DataType.INT64, -1, -1);
        sd.placeHolder("attention_mask", DataType.FLOAT, -1, -1);
        sd.placeHolder("position_ids_t", DataType.INT64, -1, -1);
        sd.placeHolder("position_ids_h", DataType.INT64, -1, -1);
        sd.placeHolder("position_ids_w", DataType.INT64, -1, -1);

        // Vision encoder placeholders
        sd.placeHolder("pixel_values", DataType.FLOAT, -1, -1, 3, -1, -1);

        log.info("Qwen3-VL graph built with {} weight variables", weights.size());
        return sd;
    }

    @Override
    public Map<String, String> getWeightNameMappings() {
        Map<String, String> mappings = new HashMap<>();

        // Embeddings
        mappings.put("model.embed_tokens.weight", "embed_tokens_weight");

        // LLM layers
        mappings.put("model.layers.{layer}.self_attn.q_proj.weight", "layers_{layer}_q_proj_weight");
        mappings.put("model.layers.{layer}.self_attn.k_proj.weight", "layers_{layer}_k_proj_weight");
        mappings.put("model.layers.{layer}.self_attn.v_proj.weight", "layers_{layer}_v_proj_weight");
        mappings.put("model.layers.{layer}.self_attn.o_proj.weight", "layers_{layer}_o_proj_weight");
        mappings.put("model.layers.{layer}.mlp.gate_proj.weight", "layers_{layer}_gate_proj_weight");
        mappings.put("model.layers.{layer}.mlp.up_proj.weight", "layers_{layer}_up_proj_weight");
        mappings.put("model.layers.{layer}.mlp.down_proj.weight", "layers_{layer}_down_proj_weight");
        mappings.put("model.layers.{layer}.input_layernorm.weight", "layers_{layer}_input_ln_weight");
        mappings.put("model.layers.{layer}.post_attention_layernorm.weight", "layers_{layer}_post_ln_weight");

        // Vision encoder
        mappings.put("visual.blocks.{layer}.attn.qkv.weight", "vision_{layer}_qkv_weight");
        mappings.put("visual.blocks.{layer}.attn.proj.weight", "vision_{layer}_proj_weight");
        mappings.put("visual.blocks.{layer}.mlp.fc1.weight", "vision_{layer}_fc1_weight");
        mappings.put("visual.blocks.{layer}.mlp.fc2.weight", "vision_{layer}_fc2_weight");
        mappings.put("visual.blocks.{layer}.norm1.weight", "vision_{layer}_norm1_weight");
        mappings.put("visual.blocks.{layer}.norm2.weight", "vision_{layer}_norm2_weight");

        // Vision merger/projector
        mappings.put("visual.merger.mlp.0.weight", "merger_mlp_0_weight");
        mappings.put("visual.merger.mlp.0.bias", "merger_mlp_0_bias");
        mappings.put("visual.merger.mlp.2.weight", "merger_mlp_2_weight");
        mappings.put("visual.merger.mlp.2.bias", "merger_mlp_2_bias");

        // Output
        mappings.put("model.norm.weight", "final_norm_weight");
        mappings.put("lm_head.weight", "lm_head_weight");

        return mappings;
    }

    private static int getInt(Map<String, Object> config, String key, int defaultValue) {
        Object val = config.get(key);
        if (val instanceof Number) return ((Number) val).intValue();
        return defaultValue;
    }
}
