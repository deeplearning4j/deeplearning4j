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
import org.nd4j.ggml.convert.ConversionOptions;
import org.nd4j.ggml.format.GGMLMetadata;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.custom.FusedRoPE;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.autodiff.samediff.SDIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

/**
 * Architecture handler for LLaMA and LLaMA-derived models.
 *
 * Handles the broad family of decoder-only transformers that share the
 * LLaMA pattern: RMSNorm, GQA, SwiGLU FFN, RoPE.  This includes LLaMA 1-3,
 * Mistral, Mixtral, Qwen (all versions), Yi, DeepSeek, InternLM, and any
 * future model that follows the same tensor naming convention in GGUF.
 *
 * MoE (Mixture-of-Experts) layers are detected at runtime by the presence
 * of {@code ffn_gate_inp} router weights.  When found, the FFN block is
 * replaced with a router + per-expert SwiGLU + optional shared expert,
 * which covers Mixtral, Qwen-MoE, DeepSeek-MoE, etc. without needing a
 * separate architecture class per model.
 */
@Slf4j
public class LLaMAArchitecture implements ModelArchitecture {

    private static final Set<String> SUPPORTED_VARIANTS = Set.of(
            "llama", "llama2", "llama3", "codellama",
            "mistral", "mixtral", "yi", "deepseek",
            "qwen", "qwen2", "qwen3", "qwen3.5",
            "internlm", "internlm2"
    );

    @Override
    public String getName() {
        return "llama";
    }

    @Override
    public Set<String> getSupportedVariants() {
        return SUPPORTED_VARIANTS;
    }

    @Override
    public boolean canHandle(GGMLMetadata metadata) {
        String arch = metadata.getArchitecture();
        if (arch == null) return false;

        String archLower = arch.toLowerCase();
        return SUPPORTED_VARIANTS.contains(archLower) ||
               archLower.contains("llama") ||
               archLower.contains("mistral") ||
               archLower.contains("qwen");
    }

    @Override
    public SameDiff buildGraph(GGMLMetadata metadata, Map<String, INDArray> weights, ConversionOptions options) {
        SameDiff sd = SameDiff.create();
        ArchitectureConfig config = getConfig(metadata);

        DataType dtype = options.getTargetDataType();
        boolean hasMoE = weights.keySet().stream().anyMatch(k -> k.contains("ffn_gate_inp") || k.contains("ffn_gate.0."));
        log.info("Building LLaMA graph: {} layers, hidden={}, heads={}, kv_heads={}, MoE={}",
                config.getNumLayers(), config.getHiddenSize(),
                config.getNumAttentionHeads(), config.getNumKVHeads(), hasMoE);

        // Input placeholder: [batch, seq_len]
        SDVariable inputIds = sd.placeHolder("input_ids", DataType.INT64, -1, -1);

        // Token embeddings: [vocab_size, hidden_size]
        INDArray tokenEmbedWeight = weights.get("token_embd.weight");
        if (tokenEmbedWeight == null) {
            throw new IllegalStateException("Missing token embedding weights");
        }
        SDVariable tokenEmbed = sd.var("model.embed_tokens.weight", tokenEmbedWeight);

        // Gather embeddings: [batch, seq_len, hidden_size]
        SDVariable hidden = sd.gather("embedded", tokenEmbed, inputIds, 0);

        // Build transformer layers
        for (int layer = 0; layer < config.getNumLayers(); layer++) {
            hidden = buildTransformerBlock(sd, hidden, layer, config, weights, dtype);
        }

        // Final RMS normalization
        hidden = buildRMSNorm(sd, hidden, "model.norm", weights, config, dtype);

        // Output projection (LM head)
        INDArray outputWeight = weights.get("output.weight");
        if (outputWeight == null) {
            // Some models tie weights
            outputWeight = tokenEmbedWeight;
        }
        SDVariable lmHead = sd.var("lm_head.weight", outputWeight);

        // Logits: [batch, seq_len, vocab_size]
        SDVariable logits = sd.mmul("logits", hidden, lmHead.permute(1, 0));

        return sd;
    }

    private SDVariable buildTransformerBlock(SameDiff sd, SDVariable input, int layerIdx,
            ArchitectureConfig config, Map<String, INDArray> weights, DataType dtype) {

        String prefix = "blk." + layerIdx;

        // Pre-attention RMS normalization
        SDVariable normed = buildRMSNorm(sd, input,
                "model.layers." + layerIdx + ".input_layernorm",
                prefix + ".attn_norm", weights, config, dtype);

        // Self-attention
        SDVariable attnOut = buildSelfAttention(sd, normed, layerIdx, config, weights, dtype);

        // Residual connection
        SDVariable postAttn = input.add("post_attn_" + layerIdx, attnOut);

        // Pre-FFN RMS normalization
        SDVariable ffnNormed = buildRMSNorm(sd, postAttn,
                "model.layers." + layerIdx + ".post_attention_layernorm",
                prefix + ".ffn_norm", weights, config, dtype);

        // Feed-forward: MoE if router gate present, otherwise dense SwiGLU
        SDVariable ffnOut;
        INDArray routerGate = weights.get(prefix + ".ffn_gate_inp.weight");
        if (routerGate != null) {
            ffnOut = buildMoEFFN(sd, ffnNormed, layerIdx, config, weights, dtype, routerGate);
        } else {
            ffnOut = buildSwiGLUFFN(sd, ffnNormed, layerIdx, config, weights, dtype);
        }

        // Residual connection
        return postAttn.add("layer_out_" + layerIdx, ffnOut);
    }

    private SDVariable buildRMSNorm(SameDiff sd, SDVariable input, String outputName,
            Map<String, INDArray> weights, ArchitectureConfig config, DataType dtype) {
        return buildRMSNorm(sd, input, outputName, "output_norm", weights, config, dtype);
    }

    private SDVariable buildRMSNorm(SameDiff sd, SDVariable input, String outputName,
            String weightKey, Map<String, INDArray> weights, ArchitectureConfig config, DataType dtype) {

        INDArray normWeight = weights.get(weightKey + ".weight");
        if (normWeight == null) {
            log.warn("Missing RMS norm weight: {}", weightKey);
            return input;
        }

        SDVariable gamma = sd.var(outputName + ".weight", normWeight);

        // RMS normalization: x * rsqrt(mean(x^2) + eps) * gamma
        SDVariable squared = input.mul(input);
        SDVariable meanSquared = squared.mean(true, -1);
        SDVariable rms = sd.math.sqrt(meanSquared.add(config.getLayerNormEpsilon()));
        SDVariable normalized = input.div(rms);

        return normalized.mul(outputName, gamma);
    }

    private SDVariable buildSelfAttention(SameDiff sd, SDVariable input, int layerIdx,
            ArchitectureConfig config, Map<String, INDArray> weights, DataType dtype) {

        String prefix = "blk." + layerIdx;
        int hiddenSize = config.getHiddenSize();
        int numHeads = config.getNumAttentionHeads();
        int numKVHeads = config.getNumKVHeads();
        int headDim = config.getHeadDimension();

        // Q, K, V projections
        INDArray qWeight = weights.get(prefix + ".attn_q.weight");
        INDArray kWeight = weights.get(prefix + ".attn_k.weight");
        INDArray vWeight = weights.get(prefix + ".attn_v.weight");
        INDArray oWeight = weights.get(prefix + ".attn_output.weight");

        if (qWeight == null || kWeight == null || vWeight == null || oWeight == null) {
            log.warn("Missing attention weights for layer {}", layerIdx);
            return input;
        }

        String attnPrefix = "model.layers." + layerIdx + ".self_attn.";
        SDVariable wq = sd.var(attnPrefix + "q_proj.weight", qWeight);
        SDVariable wk = sd.var(attnPrefix + "k_proj.weight", kWeight);
        SDVariable wv = sd.var(attnPrefix + "v_proj.weight", vWeight);
        SDVariable wo = sd.var(attnPrefix + "o_proj.weight", oWeight);

        // Project to Q, K, V: [batch, seq, hidden] -> [batch, seq, proj_dim]
        SDVariable q = sd.mmul("q_" + layerIdx, input, wq.permute(1, 0));
        SDVariable k = sd.mmul("k_" + layerIdx, input, wk.permute(1, 0));
        SDVariable v = sd.mmul("v_" + layerIdx, input, wv.permute(1, 0));

        // Add biases if present (e.g. Qwen models)
        INDArray qBias = weights.get(prefix + ".attn_q.bias");
        INDArray kBias = weights.get(prefix + ".attn_k.bias");
        INDArray vBias = weights.get(prefix + ".attn_v.bias");
        if (qBias != null) q = q.add(sd.var(attnPrefix + "q_proj.bias", qBias));
        if (kBias != null) k = k.add(sd.var(attnPrefix + "k_proj.bias", kBias));
        if (vBias != null) v = v.add(sd.var(attnPrefix + "v_proj.bias", vBias));

        // Reshape to multi-head format: [batch, seq, num_heads, head_dim] (BSHD)
        long[] qShape = new long[]{-1, -1, numHeads, headDim};
        long[] kvShape = new long[]{-1, -1, numKVHeads, headDim};

        q = q.reshape(qShape);
        k = k.reshape(kvShape);
        v = v.reshape(kvShape);

        // Apply Rotary Position Embeddings (RoPE) to Q and K
        if (config.isUseRotaryEmbeddings()) {
            q = new FusedRoPE(sd, q, FusedRoPE.ROPE_TYPE_STANDARD, 0,
                    config.getRopeFreqBase(), 1.0).outputVariable();
            sd.updateVariableNameAndReference(q, "q_rope_" + layerIdx);

            k = new FusedRoPE(sd, k, FusedRoPE.ROPE_TYPE_STANDARD, 0,
                    config.getRopeFreqBase(), 1.0).outputVariable();
            sd.updateVariableNameAndReference(k, "k_rope_" + layerIdx);
        }

        // dot_product_attention_v2 expects BSHD [batch, seq, heads, headDim]
        // It handles GQA internally (numKVHeads != numHeads), causal masking,
        // and flash attention — all in one fused op.
        SDVariable attnOut = sd.nn.dotProductAttentionV2(
                "attn_out_" + layerIdx,
                q,         // queries: [batch, seq, numHeads, headDim]
                v,         // values:  [batch, seq, numKVHeads, headDim]
                k,         // keys:    [batch, seq, numKVHeads, headDim]
                null,      // queryMask
                null,      // valueMask
                0.0,       // scaleFactor: 0 = auto (1/sqrt(headDim))
                0.0,       // dropout
                true,      // useCausalMask
                false      // training
        );

        // Reshape back: [batch, seq, numHeads, headDim] -> [batch, seq, hidden]
        attnOut = attnOut.reshape(-1, -1, hiddenSize);

        // Output projection
        return sd.mmul("attn_proj_" + layerIdx, attnOut, wo.permute(1, 0));
    }

    private SDVariable buildSwiGLUFFN(SameDiff sd, SDVariable input, int layerIdx,
            ArchitectureConfig config, Map<String, INDArray> weights, DataType dtype) {

        String prefix = "blk." + layerIdx;

        INDArray gateWeight = weights.get(prefix + ".ffn_gate.weight");
        INDArray upWeight = weights.get(prefix + ".ffn_up.weight");
        INDArray downWeight = weights.get(prefix + ".ffn_down.weight");

        if (gateWeight == null || upWeight == null || downWeight == null) {
            log.warn("Missing FFN weights for layer {}", layerIdx);
            return input;
        }

        String mlpPrefix = "model.layers." + layerIdx + ".mlp.";
        SDVariable wGate = sd.var(mlpPrefix + "gate_proj.weight", gateWeight);
        SDVariable wUp = sd.var(mlpPrefix + "up_proj.weight", upWeight);
        SDVariable wDown = sd.var(mlpPrefix + "down_proj.weight", downWeight);

        // SwiGLU: silu(gate(x)) * up(x)
        SDVariable gate = sd.mmul("gate_" + layerIdx, input, wGate.permute(1, 0));
        SDVariable up = sd.mmul("up_" + layerIdx, input, wUp.permute(1, 0));

        // SiLU activation on gate (swish = x * sigmoid(x))
        SDVariable silu = sd.nn.swish(gate);

        // Element-wise multiply
        SDVariable hidden = silu.mul("swiglu_" + layerIdx, up);

        // Down projection
        return sd.mmul("down_" + layerIdx, hidden, wDown.permute(1, 0));
    }

    /**
     * Build a Mixture-of-Experts FFN block.
     * Detected at runtime when the layer has a {@code ffn_gate_inp} router weight.
     * Covers Mixtral, Qwen-MoE, DeepSeek-MoE, and similar architectures.
     */
    private SDVariable buildMoEFFN(SameDiff sd, SDVariable input, int layerIdx,
            ArchitectureConfig config, Map<String, INDArray> weights, DataType dtype,
            INDArray routerGateWeight) {

        String prefix = "blk." + layerIdx;
        String mlpPrefix = "model.layers." + layerIdx + ".mlp.";

        // Router: [batch, seq, hidden] x [num_experts, hidden]^T -> [batch, seq, num_experts]
        SDVariable gate = sd.var(mlpPrefix + "gate.weight", routerGateWeight);
        SDVariable routerLogits = sd.mmul("router_logits_" + layerIdx, input, gate.permute(1, 0));
        SDVariable routerWeights = sd.nn.softmax("router_weights_" + layerIdx, routerLogits, -1);

        // Count available experts
        int numExperts = 0;
        while (weights.containsKey(prefix + ".ffn_gate." + numExperts + ".weight")) {
            numExperts++;
        }

        if (numExperts == 0) {
            log.warn("MoE router found but no expert weights for layer {}", layerIdx);
            return input;
        }

        log.debug("Layer {} MoE: {} experts", layerIdx, numExperts);

        // Simplified dense MoE: compute all experts, weighted-sum by router probs
        // (full top-k sparse dispatch would be an optimization for large expert counts)
        SDVariable combined = null;
        for (int e = 0; e < numExperts; e++) {
            SDVariable expertOut = buildExpertSwiGLU(sd, input, layerIdx, e, weights, dtype);

            // Weight by router probability
            SDVariable expertWeight = routerWeights.get(
                    SDIndex.all(),
                    SDIndex.all(),
                    SDIndex.point(e));
            SDVariable weighted = expertOut.mul("weighted_e" + e + "_" + layerIdx, expertWeight);

            combined = (combined == null) ? weighted : combined.add("combine_e" + e + "_" + layerIdx, weighted);
        }

        // Shared expert (some MoE models have an always-active shared expert)
        INDArray sharedGateW = weights.get(prefix + ".ffn_gate_shexp.weight");
        if (sharedGateW != null) {
            SDVariable sharedOut = buildExpertSwiGLU(sd, input, layerIdx, "shared", weights, dtype);
            combined = combined.add("moe_shared_" + layerIdx, sharedOut);
        }

        return combined != null ? combined : input;
    }

    private SDVariable buildExpertSwiGLU(SameDiff sd, SDVariable input, int layerIdx,
            int expertIdx, Map<String, INDArray> weights, DataType dtype) {
        return buildExpertSwiGLU(sd, input, layerIdx, String.valueOf(expertIdx), weights, dtype);
    }

    private SDVariable buildExpertSwiGLU(SameDiff sd, SDVariable input, int layerIdx,
            String expertId, Map<String, INDArray> weights, DataType dtype) {

        String prefix = "blk." + layerIdx;
        String weightSuffix = "shared".equals(expertId) ? "_shexp" : "." + expertId;
        String nameSuffix = "_" + layerIdx + "_e" + expertId;

        INDArray gateW = weights.get(prefix + ".ffn_gate" + weightSuffix + ".weight");
        INDArray upW = weights.get(prefix + ".ffn_up" + weightSuffix + ".weight");
        INDArray downW = weights.get(prefix + ".ffn_down" + weightSuffix + ".weight");

        if (gateW == null || upW == null || downW == null) {
            log.warn("Missing expert FFN weights for layer {} expert {}", layerIdx, expertId);
            return input;
        }

        String mlpPrefix = "model.layers." + layerIdx + ".mlp.expert_" + expertId + ".";
        SDVariable wGate = sd.var(mlpPrefix + "gate_proj.weight", gateW);
        SDVariable wUp = sd.var(mlpPrefix + "up_proj.weight", upW);
        SDVariable wDown = sd.var(mlpPrefix + "down_proj.weight", downW);

        SDVariable g = sd.mmul("gate" + nameSuffix, input, wGate.permute(1, 0));
        SDVariable u = sd.mmul("up" + nameSuffix, input, wUp.permute(1, 0));

        SDVariable silu = sd.nn.swish(g);
        SDVariable h = silu.mul("swiglu" + nameSuffix, u);

        return sd.mmul("down" + nameSuffix, h, wDown.permute(1, 0));
    }

    @Override
    public Map<String, String> getTensorNamePatterns() {
        Map<String, String> patterns = new HashMap<>();

        patterns.put("token_embd.weight", "model.embed_tokens.weight");
        patterns.put("output.weight", "lm_head.weight");
        patterns.put("output_norm.weight", "model.norm.weight");

        // Attention layers
        patterns.put("blk.{layer}.attn_q.weight", "model.layers.{layer}.self_attn.q_proj.weight");
        patterns.put("blk.{layer}.attn_k.weight", "model.layers.{layer}.self_attn.k_proj.weight");
        patterns.put("blk.{layer}.attn_v.weight", "model.layers.{layer}.self_attn.v_proj.weight");
        patterns.put("blk.{layer}.attn_output.weight", "model.layers.{layer}.self_attn.o_proj.weight");

        // Attention biases (Qwen models)
        patterns.put("blk.{layer}.attn_q.bias", "model.layers.{layer}.self_attn.q_proj.bias");
        patterns.put("blk.{layer}.attn_k.bias", "model.layers.{layer}.self_attn.k_proj.bias");
        patterns.put("blk.{layer}.attn_v.bias", "model.layers.{layer}.self_attn.v_proj.bias");

        // Dense FFN layers
        patterns.put("blk.{layer}.ffn_gate.weight", "model.layers.{layer}.mlp.gate_proj.weight");
        patterns.put("blk.{layer}.ffn_up.weight", "model.layers.{layer}.mlp.up_proj.weight");
        patterns.put("blk.{layer}.ffn_down.weight", "model.layers.{layer}.mlp.down_proj.weight");

        // MoE router gate
        patterns.put("blk.{layer}.ffn_gate_inp.weight", "model.layers.{layer}.mlp.gate.weight");

        // Normalization layers
        patterns.put("blk.{layer}.attn_norm.weight", "model.layers.{layer}.input_layernorm.weight");
        patterns.put("blk.{layer}.ffn_norm.weight", "model.layers.{layer}.post_attention_layernorm.weight");

        return patterns;
    }
}
