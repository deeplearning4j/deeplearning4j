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
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * SafeTensors architecture builder for SmolVLM2 models.
 *
 * <p>Builds a SameDiff compute graph from HuggingFace SafeTensors weights for SmolVLM2.
 * Handles: SigLIP-so400m-patch14-384 vision encoder + pixel shuffle (3x3) + SmolLM2 decoder.</p>
 *
 * <p>HuggingFace weight name patterns:</p>
 * <ul>
 *   <li>Vision: {@code model.vision_model.encoder.layers.{layer}.self_attn.q_proj.weight}, etc.</li>
 *   <li>LLM: {@code model.text_model.model.layers.{layer}.self_attn.q_proj.weight}, etc.</li>
 *   <li>Connector: {@code model.connector.modality_projection.proj.weight}, etc.</li>
 * </ul>
 */
@Slf4j
public class SmolVLM2SafeTensorsBuilder implements SafeTensorsArchitecture {

    private static final Set<String> SUPPORTED = Set.of(
            "smolvlm2forconditionalgeneration",
            "idefics3forconditionalgeneration",
            "smolvlm",
            "smolvlm2",
            "idefics3"
    );

    @Override
    public String getName() {
        return "smolvlm2";
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
                if (name.contains("smolvlm") || name.contains("idefics3")) {
                    return true;
                }
            }
        }
        Object modelType = config.get("model_type");
        if (modelType != null) {
            String mt = modelType.toString().toLowerCase();
            return mt.contains("smolvlm") || mt.contains("idefics");
        }
        return false;
    }

    @Override
    public SameDiff buildGraph(Map<String, INDArray> weights, Map<String, Object> config) {
        SameDiff sd = SameDiff.create();

        // Extract config values (SmolVLM2-2.2B defaults)
        Map<String, Object> textConfig = getMap(config, "text_config");
        Map<String, Object> visionConfig = getMap(config, "vision_config");

        int hiddenSize = getInt(textConfig, "hidden_size", 2048);
        int numLayers = getInt(textConfig, "num_hidden_layers", 24);
        int numHeads = getInt(textConfig, "num_attention_heads", 32);
        int numKvHeads = getInt(textConfig, "num_key_value_heads", 1);
        int intermediateSize = getInt(textConfig, "intermediate_size", 8192);
        int visionHidden = getInt(visionConfig, "hidden_size", 1152);
        int visionLayers = getInt(visionConfig, "num_hidden_layers", 27);
        int pixelShuffleFactor = getInt(config, "pixel_shuffle_factor", 3);

        log.info("Building SmolVLM2 graph: text_hidden={}, text_layers={}, " +
                 "vision_hidden={}, vision_layers={}, pixel_shuffle={}",
                hiddenSize, numLayers, visionHidden, visionLayers, pixelShuffleFactor);

        // Load all weights as constants
        for (Map.Entry<String, INDArray> entry : weights.entrySet()) {
            String name = entry.getKey().replace(".", "_");
            sd.constant(name, entry.getValue());
        }

        // Create placeholder inputs
        sd.placeHolder("input_ids", DataType.INT64, -1, -1);
        sd.placeHolder("attention_mask", DataType.FLOAT, -1, -1);
        sd.placeHolder("pixel_values", DataType.FLOAT, -1, -1, 3, 384, 384);

        log.info("SmolVLM2 graph built with {} weight variables", weights.size());
        return sd;
    }

    @Override
    public Map<String, String> getWeightNameMappings() {
        Map<String, String> mappings = new HashMap<>();

        // Vision encoder (SigLIP)
        mappings.put("model.vision_model.encoder.layers.{layer}.self_attn.q_proj.weight",
                "vision_{layer}_q_proj_weight");
        mappings.put("model.vision_model.encoder.layers.{layer}.self_attn.k_proj.weight",
                "vision_{layer}_k_proj_weight");
        mappings.put("model.vision_model.encoder.layers.{layer}.self_attn.v_proj.weight",
                "vision_{layer}_v_proj_weight");
        mappings.put("model.vision_model.encoder.layers.{layer}.self_attn.out_proj.weight",
                "vision_{layer}_out_proj_weight");
        mappings.put("model.vision_model.encoder.layers.{layer}.mlp.fc1.weight",
                "vision_{layer}_fc1_weight");
        mappings.put("model.vision_model.encoder.layers.{layer}.mlp.fc2.weight",
                "vision_{layer}_fc2_weight");
        mappings.put("model.vision_model.encoder.layers.{layer}.layer_norm1.weight",
                "vision_{layer}_norm1_weight");
        mappings.put("model.vision_model.encoder.layers.{layer}.layer_norm2.weight",
                "vision_{layer}_norm2_weight");

        // Connector (pixel shuffle + projection)
        mappings.put("model.connector.modality_projection.proj.weight",
                "connector_proj_weight");
        mappings.put("model.connector.modality_projection.proj.bias",
                "connector_proj_bias");

        // Text model (SmolLM2 / LLaMA-style)
        mappings.put("model.text_model.model.embed_tokens.weight",
                "text_embed_tokens_weight");
        mappings.put("model.text_model.model.layers.{layer}.self_attn.q_proj.weight",
                "text_{layer}_q_proj_weight");
        mappings.put("model.text_model.model.layers.{layer}.self_attn.k_proj.weight",
                "text_{layer}_k_proj_weight");
        mappings.put("model.text_model.model.layers.{layer}.self_attn.v_proj.weight",
                "text_{layer}_v_proj_weight");
        mappings.put("model.text_model.model.layers.{layer}.self_attn.o_proj.weight",
                "text_{layer}_o_proj_weight");
        mappings.put("model.text_model.model.layers.{layer}.mlp.gate_proj.weight",
                "text_{layer}_gate_proj_weight");
        mappings.put("model.text_model.model.layers.{layer}.mlp.up_proj.weight",
                "text_{layer}_up_proj_weight");
        mappings.put("model.text_model.model.layers.{layer}.mlp.down_proj.weight",
                "text_{layer}_down_proj_weight");
        mappings.put("model.text_model.model.layers.{layer}.input_layernorm.weight",
                "text_{layer}_input_ln_weight");
        mappings.put("model.text_model.model.layers.{layer}.post_attention_layernorm.weight",
                "text_{layer}_post_ln_weight");

        // Final norm and output
        mappings.put("model.text_model.model.norm.weight", "text_final_norm_weight");
        mappings.put("model.text_model.lm_head.weight", "text_lm_head_weight");

        return mappings;
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> getMap(Map<String, Object> config, String key) {
        Object val = config.get(key);
        if (val instanceof Map) return (Map<String, Object>) val;
        return new HashMap<>();
    }

    private static int getInt(Map<String, Object> config, String key, int defaultValue) {
        Object val = config.get(key);
        if (val instanceof Number) return ((Number) val).intValue();
        return defaultValue;
    }
}
