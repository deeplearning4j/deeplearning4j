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
import org.nd4j.autodiff.samediff.SDIndex;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.ggml.convert.ConversionOptions;
import org.nd4j.ggml.format.GGMLMetadata;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.custom.FusedRoPE;
import org.nd4j.linalg.factory.Nd4j;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

/**
 * Architecture handler for GLM (General Language Model) by Zhipu AI.
 *
 * <p>Handles the GLM/ChatGLM family, covering GLM 4.5 through 5.1 and all
 * ChatGLM variants. These models share the LLaMA-style decoder-only transformer
 * pattern with the following key properties:</p>
 * <ul>
 *   <li><b>RMSNorm</b>: Used for both input and post-attention normalizations.</li>
 *   <li><b>SwiGLU FFN</b>: Standard gate_proj / up_proj / down_proj with SiLU gate.</li>
 *   <li><b>GQA</b>: Grouped-query attention with separate Q/K/V projections.</li>
 *   <li><b>RoPE</b>: Rotary position embeddings. GLM 4+ uses standard interleaved
 *       (NeoX-style) RoPE; older ChatGLM versions used a 2D position encoding scheme
 *       but their GGUF export maps to the same tensor layout.</li>
 *   <li><b>MoE (optional)</b>: GLM-4 dense-MoE variants expose a router gate
 *       ({@code ffn_gate_inp.weight}) per layer; those layers use sparse expert FFNs
 *       while layers without the router key are standard dense SwiGLU.</li>
 *   <li><b>Tensor naming</b>: Follows the standard GGUF convention
 *       ({@code blk.{layer}.attn_q.weight}, etc.) identically to LLaMA.</li>
 * </ul>
 *
 * <p>Supported variant strings (from {@code general.architecture} in GGUF metadata):</p>
 * <ul>
 *   <li>{@code glm}, {@code glm4}, {@code chatglm}, {@code chatglm3},
 *       {@code chatglm4}, {@code glm-4}, {@code codegeex}</li>
 * </ul>
 *
 * @author Eclipse Deeplearning4j Contributors
 */
@Slf4j
public class GLMArchitecture implements ModelArchitecture {

    private static final Set<String> SUPPORTED_VARIANTS = Set.of(
            "glm", "glm4", "chatglm", "chatglm3", "chatglm4", "glm-4", "codegeex"
    );

    /**
     * Default top-k experts for GLM MoE routing (top-1 for GLM dense-MoE).
     */
    private static final int DEFAULT_EXPERT_USED_COUNT = 1;

    @Override
    public String getName() {
        return "glm";
    }

    @Override
    public Set<String> getSupportedVariants() {
        return SUPPORTED_VARIANTS;
    }

    @Override
    public String getDefaultChatTemplateType() {
        return "chatml"; // ChatGLM uses ChatML-compatible format
    }

    @Override
    public String getModelSystemProperty() {
        return "glm.gguf.path";
    }

    @Override
    public boolean canHandle(GGMLMetadata metadata) {
        String arch = metadata.getArchitecture();
        if (arch == null) return false;

        String archLower = arch.toLowerCase();
        return archLower.contains("glm") ||
               archLower.contains("chatglm") ||
               archLower.contains("codegeex");
    }

    @Override
    public ArchitectureConfig getConfig(GGMLMetadata metadata) {
        int headDim = metadata.getAttentionKeyLength();
        return ArchitectureConfig.builder()
                .numLayers(metadata.getNumLayers())
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
                .useRmsNorm(true)
                .useSwiGLU(true)
                .useRotaryEmbeddings(true)
                .decoderOnly(true)
                // GLM 4+ uses interleaved (NeoX) RoPE, type 1
                .ropeType(1)
                .build();
    }

    @Override
    public SameDiff buildGraph(GGMLMetadata metadata, Map<String, INDArray> weights, ConversionOptions options) {
        SameDiff sd = SameDiff.create();
        ArchitectureConfig config = getConfig(metadata);

        DataType dtype = options.getTargetDataType();
        boolean isMoE = hasMoELayers(weights, config.getNumLayers());
        int expertCount = metadata.getExpertCount();
        int expertUsedCount = metadata.getExpertUsedCount() > 0
                ? metadata.getExpertUsedCount() : DEFAULT_EXPERT_USED_COUNT;

        log.info("Building GLM graph: {} layers, hidden={}, heads={}, kv_heads={}, headDim={}, " +
                "ropeFreqBase={}, ropeNDim={}, isMoE={}, experts={}, topK={}, dtype={}",
                config.getNumLayers(), config.getHiddenSize(),
                config.getNumAttentionHeads(), config.getNumKVHeads(), config.getHeadDimension(),
                config.getRopeFreqBase(), config.getRopeDimensionCount(),
                isMoE, expertCount, expertUsedCount, dtype);

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
            hidden = buildTransformerBlock(sd, hidden, layer, config, weights, dtype, metadata);
        }

        // Final RMS normalization
        hidden = buildRMSNorm(sd, hidden, "model.norm", "output_norm", weights, config);

        // Output projection (LM head)
        INDArray outputWeight = weights.get("output.weight");
        if (outputWeight == null) {
            // Some models tie embedding weights
            outputWeight = tokenEmbedWeight;
        }
        SDVariable lmHead = sd.var("lm_head.weight", outputWeight);

        // Logits: [batch, seq_len, vocab_size]
        QuantizedLinear.matMul(sd, "logits", hidden, lmHead, weights, "output.weight", dtype);

        return sd;
    }

    // ========================================================================
    // Transformer block
    // ========================================================================

    private SDVariable buildTransformerBlock(SameDiff sd, SDVariable input, int layerIdx,
            ArchitectureConfig config, Map<String, INDArray> weights, DataType dtype,
            GGMLMetadata metadata) {

        String prefix = "blk." + layerIdx;

        // Pre-attention RMS normalization
        SDVariable normed = buildRMSNorm(sd, input,
                "model.layers." + layerIdx + ".input_layernorm",
                prefix + ".attn_norm", weights, config);

        // Self-attention with GQA + RoPE
        SDVariable attnOut = buildAttention(sd, normed, layerIdx, config, weights, dtype);

        // Residual connection
        SDVariable postAttn = input.add("post_attn_" + layerIdx, attnOut);

        // Pre-FFN RMS normalization
        SDVariable ffnNormed = buildRMSNorm(sd, postAttn,
                "model.layers." + layerIdx + ".post_attention_layernorm",
                prefix + ".ffn_norm", weights, config);

        // Feed-forward network: MoE or dense SwiGLU
        SDVariable ffnOut;
        if (weights.containsKey(prefix + ".ffn_gate_inp.weight")) {
            INDArray routerGate = weights.get(prefix + ".ffn_gate_inp.weight");
            ffnOut = buildMoEFFN(sd, ffnNormed, layerIdx, config, weights, dtype, routerGate, metadata);
        } else if (weights.containsKey(prefix + ".ffn_gate.weight")) {
            ffnOut = buildSwiGLUFFN(sd, ffnNormed, layerIdx, weights);
        } else {
            log.warn("No FFN weights found for layer {}, passing through", layerIdx);
            ffnOut = ffnNormed;
        }

        // Residual connection
        return postAttn.add("layer_out_" + layerIdx, ffnOut);
    }

    // ========================================================================
    // RMS Normalization
    // ========================================================================

    private SDVariable buildRMSNorm(SameDiff sd, SDVariable input, String outputName,
            String weightKey, Map<String, INDArray> weights, ArchitectureConfig config) {

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

    // ========================================================================
    // Self-attention
    // ========================================================================

    /**
     * Build GQA self-attention with RoPE for GLM.
     *
     * <p>GLM 4+ uses standard GQA with separate Q/K/V projections identical to LLaMA.
     * Interleaved (NeoX-style) RoPE is applied to Q and K after head reshaping.</p>
     */
    private SDVariable buildAttention(SameDiff sd, SDVariable input, int layerIdx,
            ArchitectureConfig config, Map<String, INDArray> weights, DataType dtype) {

        String prefix = "blk." + layerIdx;
        int numHeads = config.getNumAttentionHeads();
        int numKVHeads = config.getNumKVHeads();

        INDArray qWeight = weights.get(prefix + ".attn_q.weight");
        INDArray kWeight = weights.get(prefix + ".attn_k.weight");
        INDArray vWeight = weights.get(prefix + ".attn_v.weight");
        INDArray oWeight = weights.get(prefix + ".attn_output.weight");

        if (qWeight == null || kWeight == null || vWeight == null || oWeight == null) {
            log.warn("Missing attention weights for layer {}", layerIdx);
            return input;
        }

        int kOutDim = QuantizedLinear.logicalOutputDim(weights, prefix + ".attn_k.weight", kWeight);
        int qOutDim = QuantizedLinear.logicalOutputDim(weights, prefix + ".attn_q.weight", qWeight);
        int headDim = config.getHeadDimension();
        if (headDim <= 0) {
            headDim = kOutDim / numKVHeads;
        }
        int actualNumHeads = qOutDim / headDim;

        if (layerIdx == 0) {
            log.info("Layer {} GLM attention: qHeads={}, kvHeads={}, headDim={} (Q out={}, K out={})",
                    layerIdx, actualNumHeads, numKVHeads, headDim, qOutDim, kOutDim);
        }

        String attnPrefix = "model.layers." + layerIdx + ".self_attn.";
        SDVariable wq = sd.var(attnPrefix + "q_proj.weight", qWeight);
        SDVariable wk = sd.var(attnPrefix + "k_proj.weight", kWeight);
        SDVariable wv = sd.var(attnPrefix + "v_proj.weight", vWeight);
        SDVariable wo = sd.var(attnPrefix + "o_proj.weight", oWeight);

        // Project to Q, K, V: [batch, seq, hidden] -> [batch, seq, proj_dim]
        SDVariable q = QuantizedLinear.matMul(sd, "q_" + layerIdx, input, wq, weights, prefix + ".attn_q.weight", dtype);
        SDVariable k = QuantizedLinear.matMul(sd, "k_" + layerIdx, input, wk, weights, prefix + ".attn_k.weight", dtype);
        SDVariable v = QuantizedLinear.matMul(sd, "v_" + layerIdx, input, wv, weights, prefix + ".attn_v.weight", dtype);

        // Add biases if present (some GLM variants include Q/K/V biases)
        INDArray qBias = weights.get(prefix + ".attn_q.bias");
        INDArray kBias = weights.get(prefix + ".attn_k.bias");
        INDArray vBias = weights.get(prefix + ".attn_v.bias");
        if (qBias != null) q = q.add(sd.var(attnPrefix + "q_proj.bias", qBias));
        if (kBias != null) k = k.add(sd.var(attnPrefix + "k_proj.bias", kBias));
        if (vBias != null) v = v.add(sd.var(attnPrefix + "v_proj.bias", vBias));

        SDVariable batchDim = sd.sizeAt(input, 0);
        SDVariable seqDim = sd.sizeAt(input, 1);

        SDVariable qShapeVar = sd.stack("q_shape_" + layerIdx, 0,
                batchDim, seqDim,
                sd.constant(Nd4j.scalar((long) actualNumHeads)),
                sd.constant(Nd4j.scalar((long) headDim)));
        SDVariable kvShapeVar = sd.stack("kv_shape_" + layerIdx, 0,
                batchDim, seqDim,
                sd.constant(Nd4j.scalar((long) numKVHeads)),
                sd.constant(Nd4j.scalar((long) headDim)));

        q = sd.reshape("q_heads_" + layerIdx, q, qShapeVar);
        k = sd.reshape("k_heads_" + layerIdx, k, kvShapeVar);
        v = sd.reshape("v_heads_" + layerIdx, v, kvShapeVar);

        // Apply interleaved RoPE (type 1, NeoX-style) to Q and K
        if (config.isUseRotaryEmbeddings()) {
            q = new FusedRoPE(sd, q, config.getRopeType(), 0,
                    config.getRopeFreqBase(), 1.0, config.getRopeDimensionCount()).outputVariable();
            sd.updateVariableNameAndReference(q, "q_rope_" + layerIdx);

            k = new FusedRoPE(sd, k, config.getRopeType(), 0,
                    config.getRopeFreqBase(), 1.0, config.getRopeDimensionCount()).outputVariable();
            sd.updateVariableNameAndReference(k, "k_rope_" + layerIdx);
        }

        // FusedRoPE promotes HALF→FLOAT internally; V must match Q/K dtype
        if (v.dataType() != q.dataType()) {
            v = v.castTo("v_cast_" + layerIdx, q.dataType());
        }

        // Causal dot-product attention
        SDVariable attnOut = sd.nn.dotProductAttentionV2(
                "attn_out_" + layerIdx,
                q, v, k, null, null,
                0.0, 0.0, true, false
        );

        int attnOutDim = actualNumHeads * headDim;
        SDVariable outShapeVar = sd.stack("attn_out_shape_" + layerIdx, 0,
                batchDim, seqDim,
                sd.constant(Nd4j.scalar((long) attnOutDim)));
        SDVariable attnFlat = sd.reshape("attn_flat_" + layerIdx, attnOut, outShapeVar);

        return QuantizedLinear.matMul(sd, "attn_proj_" + layerIdx, attnFlat, wo, weights, prefix + ".attn_output.weight", dtype);
    }

    // ========================================================================
    // FFN variants
    // ========================================================================

    /**
     * Standard dense SwiGLU feed-forward network.
     * SiLU(gate_proj(x)) * up_proj(x), then down_proj.
     */
    private SDVariable buildSwiGLUFFN(SameDiff sd, SDVariable input, int layerIdx,
            Map<String, INDArray> weights) {

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

        SDVariable gate = QuantizedLinear.matMul(sd, "gate_" + layerIdx, input, wGate, weights, prefix + ".ffn_gate.weight", input.dataType());
        SDVariable up = QuantizedLinear.matMul(sd, "up_" + layerIdx, input, wUp, weights, prefix + ".ffn_up.weight", input.dataType());

        SDVariable silu = sd.nn.swish(gate);
        SDVariable hidden = silu.mul("swiglu_" + layerIdx, up);

        return QuantizedLinear.matMul(sd, "down_" + layerIdx, hidden, wDown, weights, prefix + ".ffn_down.weight", input.dataType());
    }

    /**
     * Mixture-of-Experts FFN block for GLM dense-MoE variants.
     *
     * <p>GLM-4 dense-MoE uses top-1 routing. Each expert is a full SwiGLU FFN.
     * Both packed ({@code ffn_gate_exps.weight}) and per-expert separate weight
     * layouts are supported.</p>
     */
    private SDVariable buildMoEFFN(SameDiff sd, SDVariable input, int layerIdx,
            ArchitectureConfig config, Map<String, INDArray> weights, DataType dtype,
            INDArray routerGateWeight, GGMLMetadata metadata) {

        String prefix = "blk." + layerIdx;
        String moePrefix = "model.layers." + layerIdx + ".block_sparse_moe.";

        // Router gate: [hidden_dim, num_experts] -> softmax routing weights
        SDVariable gate = sd.var(moePrefix + "gate.weight", routerGateWeight);
        SDVariable routerLogits = QuantizedLinear.matMul(sd, "router_logits_" + layerIdx, input, gate, weights, prefix + ".ffn_gate_inp.weight", dtype);
        SDVariable routerWeights = sd.nn.softmax("router_weights_" + layerIdx, routerLogits, -1);

        int numExperts = detectExpertCount(weights, prefix, metadata);
        if (numExperts == 0) {
            log.warn("MoE router found but no expert weights for layer {}", layerIdx);
            return input;
        }

        if (layerIdx == 0) {
            log.info("Layer {} GLM MoE: {} experts detected", layerIdx, numExperts);
        }

        // Check for packed expert format
        INDArray packedGateExps = weights.get(prefix + ".ffn_gate_exps.weight");
        if (packedGateExps != null) {
            return buildPackedMoEFFN(sd, input, layerIdx, numExperts, weights, routerWeights, moePrefix);
        }

        // Per-expert separate weights
        SDVariable combined = null;
        for (int e = 0; e < numExperts; e++) {
            SDVariable expertOut = buildExpertSwiGLU(sd, input, layerIdx, e, weights, moePrefix);
            SDVariable expertWeight = routerWeights.get(SDIndex.all(), SDIndex.all(), SDIndex.point(e));
            SDVariable weighted = expertOut.mul("weighted_e" + e + "_" + layerIdx, expertWeight);
            combined = (combined == null) ? weighted : combined.add("combine_e" + e + "_" + layerIdx, weighted);
        }

        return combined != null ? combined : input;
    }

    /**
     * Build MoE with packed expert weight format.
     * Packed format: ffn_gate_exps.weight [num_experts, intermediate_dim, hidden_dim]
     */
    private SDVariable buildPackedMoEFFN(SameDiff sd, SDVariable input, int layerIdx,
            int numExperts, Map<String, INDArray> weights,
            SDVariable routerWeights, String moePrefix) {

        String prefix = "blk." + layerIdx;

        INDArray packedGate = weights.get(prefix + ".ffn_gate_exps.weight");
        INDArray packedUp = weights.get(prefix + ".ffn_up_exps.weight");
        INDArray packedDown = weights.get(prefix + ".ffn_down_exps.weight");

        if (packedGate == null || packedUp == null || packedDown == null) {
            log.warn("Missing packed MoE weights for layer {}, falling back to per-expert", layerIdx);
            SDVariable combined = null;
            for (int e = 0; e < numExperts; e++) {
                SDVariable expertOut = buildExpertSwiGLU(sd, input, layerIdx, e, weights, moePrefix);
                SDVariable expertWeight = routerWeights.get(SDIndex.all(), SDIndex.all(), SDIndex.point(e));
                SDVariable weighted = expertOut.mul("weighted_e" + e + "_" + layerIdx, expertWeight);
                combined = (combined == null) ? weighted : combined.add("combine_e" + e + "_" + layerIdx, weighted);
            }
            return combined != null ? combined : input;
        }

        SDVariable combined = null;
        for (int e = 0; e < numExperts; e++) {
            INDArray expertGateW = packedGate.slice(e, 0);
            INDArray expertUpW = packedUp.slice(e, 0);
            INDArray expertDownW = packedDown.slice(e, 0);

            String expertPrefix = moePrefix + "experts." + e + ".";
            SDVariable wGate = sd.var(expertPrefix + "w1.weight", expertGateW);
            SDVariable wUp = sd.var(expertPrefix + "w3.weight", expertUpW);
            SDVariable wDown = sd.var(expertPrefix + "w2.weight", expertDownW);

            SDVariable g = sd.mmul("gate_e" + e + "_" + layerIdx, input, wGate.permute(1, 0));
            SDVariable u = sd.mmul("up_e" + e + "_" + layerIdx, input, wUp.permute(1, 0));
            SDVariable silu = sd.nn.swish(g);
            SDVariable h = silu.mul("swiglu_e" + e + "_" + layerIdx, u);
            SDVariable expertOut = sd.mmul("down_e" + e + "_" + layerIdx, h, wDown.permute(1, 0));

            SDVariable expertWeight = routerWeights.get(SDIndex.all(), SDIndex.all(), SDIndex.point(e));
            SDVariable weighted = expertOut.mul("weighted_e" + e + "_" + layerIdx, expertWeight);
            combined = (combined == null) ? weighted : combined.add("combine_e" + e + "_" + layerIdx, weighted);
        }

        return combined != null ? combined : input;
    }

    /**
     * Build a single expert's SwiGLU FFN from per-expert separate weights.
     */
    private SDVariable buildExpertSwiGLU(SameDiff sd, SDVariable input, int layerIdx,
            int expertIdx, Map<String, INDArray> weights, String moePrefix) {

        String prefix = "blk." + layerIdx;
        String nameSuffix = "_" + layerIdx + "_e" + expertIdx;

        INDArray gateW = weights.get(prefix + ".ffn_gate." + expertIdx + ".weight");
        INDArray upW = weights.get(prefix + ".ffn_up." + expertIdx + ".weight");
        INDArray downW = weights.get(prefix + ".ffn_down." + expertIdx + ".weight");

        if (gateW == null || upW == null || downW == null) {
            log.warn("Missing expert FFN weights for layer {} expert {}", layerIdx, expertIdx);
            return input;
        }

        String expertPrefix = moePrefix + "experts." + expertIdx + ".";
        SDVariable wGate = sd.var(expertPrefix + "w1.weight", gateW);
        SDVariable wUp = sd.var(expertPrefix + "w3.weight", upW);
        SDVariable wDown = sd.var(expertPrefix + "w2.weight", downW);

        SDVariable g = QuantizedLinear.matMul(sd, "gate" + nameSuffix, input, wGate, weights, prefix + ".ffn_gate." + expertIdx + ".weight", input.dataType());
        SDVariable u = QuantizedLinear.matMul(sd, "up" + nameSuffix, input, wUp, weights, prefix + ".ffn_up." + expertIdx + ".weight", input.dataType());
        SDVariable silu = sd.nn.swish(g);
        SDVariable h = silu.mul("swiglu" + nameSuffix, u);

        return QuantizedLinear.matMul(sd, "down" + nameSuffix, h, wDown, weights, prefix + ".ffn_down." + expertIdx + ".weight", input.dataType());
    }

    // ========================================================================
    // Tensor name patterns
    // ========================================================================

    @Override
    public Map<String, String> getTensorNamePatterns() {
        Map<String, String> patterns = new HashMap<>();

        // Embeddings and output
        patterns.put("token_embd.weight", "model.embed_tokens.weight");
        patterns.put("output.weight", "lm_head.weight");
        patterns.put("output_norm.weight", "model.norm.weight");

        // Attention layers (separate Q/K/V — same as LLaMA)
        patterns.put("blk.{layer}.attn_q.weight", "model.layers.{layer}.self_attn.q_proj.weight");
        patterns.put("blk.{layer}.attn_k.weight", "model.layers.{layer}.self_attn.k_proj.weight");
        patterns.put("blk.{layer}.attn_v.weight", "model.layers.{layer}.self_attn.v_proj.weight");
        patterns.put("blk.{layer}.attn_output.weight", "model.layers.{layer}.self_attn.o_proj.weight");

        // Attention biases (present in some GLM variants)
        patterns.put("blk.{layer}.attn_q.bias", "model.layers.{layer}.self_attn.q_proj.bias");
        patterns.put("blk.{layer}.attn_k.bias", "model.layers.{layer}.self_attn.k_proj.bias");
        patterns.put("blk.{layer}.attn_v.bias", "model.layers.{layer}.self_attn.v_proj.bias");

        // Dense FFN layers
        patterns.put("blk.{layer}.ffn_gate.weight", "model.layers.{layer}.mlp.gate_proj.weight");
        patterns.put("blk.{layer}.ffn_up.weight", "model.layers.{layer}.mlp.up_proj.weight");
        patterns.put("blk.{layer}.ffn_down.weight", "model.layers.{layer}.mlp.down_proj.weight");

        // MoE router gate
        patterns.put("blk.{layer}.ffn_gate_inp.weight", "model.layers.{layer}.block_sparse_moe.gate.weight");

        // MoE per-expert weights (separate format)
        patterns.put("blk.{layer}.ffn_gate.{expert}.weight",
                "model.layers.{layer}.block_sparse_moe.experts.{expert}.w1.weight");
        patterns.put("blk.{layer}.ffn_up.{expert}.weight",
                "model.layers.{layer}.block_sparse_moe.experts.{expert}.w3.weight");
        patterns.put("blk.{layer}.ffn_down.{expert}.weight",
                "model.layers.{layer}.block_sparse_moe.experts.{expert}.w2.weight");

        // MoE packed expert format
        patterns.put("blk.{layer}.ffn_gate_exps.weight",
                "model.layers.{layer}.block_sparse_moe.experts_packed.gate.weight");
        patterns.put("blk.{layer}.ffn_up_exps.weight",
                "model.layers.{layer}.block_sparse_moe.experts_packed.up.weight");
        patterns.put("blk.{layer}.ffn_down_exps.weight",
                "model.layers.{layer}.block_sparse_moe.experts_packed.down.weight");

        // Normalization layers
        patterns.put("blk.{layer}.attn_norm.weight", "model.layers.{layer}.input_layernorm.weight");
        patterns.put("blk.{layer}.ffn_norm.weight", "model.layers.{layer}.post_attention_layernorm.weight");

        return patterns;
    }

    // ========================================================================
    // Utility methods
    // ========================================================================

    /**
     * Check if any layer has MoE (router gate) weights.
     */
    private boolean hasMoELayers(Map<String, INDArray> weights, int numLayers) {
        for (int i = 0; i < numLayers; i++) {
            if (weights.containsKey("blk." + i + ".ffn_gate_inp.weight")) {
                return true;
            }
        }
        return false;
    }

    /**
     * Detect the number of experts from weight keys or metadata.
     */
    private int detectExpertCount(Map<String, INDArray> weights, String prefix, GGMLMetadata metadata) {
        if (metadata.getExpertCount() > 0) {
            return metadata.getExpertCount();
        }

        // Check for packed format
        INDArray packedGate = weights.get(prefix + ".ffn_gate_exps.weight");
        if (packedGate != null) {
            return (int) packedGate.shape()[0];
        }

        // Count per-expert weights
        int count = 0;
        while (weights.containsKey(prefix + ".ffn_gate." + count + ".weight")) {
            count++;
        }
        return count;
    }
}
