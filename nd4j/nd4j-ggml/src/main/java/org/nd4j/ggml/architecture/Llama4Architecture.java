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
import org.nd4j.ggml.architecture.QuantizedLinear;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

/**
 * Architecture handler for Meta Llama 4 (Scout and Maverick).
 *
 * <p>Llama 4 introduces several architectural advances over Llama 3:</p>
 * <ul>
 *   <li><b>iRoPE (Interleaved RoPE)</b>: Alternates between standard RoPE layers and
 *       NoPE (No Position Encoding) layers. In NoPE layers, Q and K heads are split into
 *       a RoPE portion and a NoPE portion; only the RoPE dimensions receive rotary
 *       embeddings. The interval between full-RoPE layers is determined by the
 *       {@code rope_layer_interval} metadata key, or falls back to probing per-layer
 *       tensor presence.</li>
 *   <li><b>Mixture-of-Experts (MoE)</b>: Scout uses 16 routed experts; Maverick uses 128.
 *       Top-1 routing. Layers can be either dense or MoE — detected from whether
 *       {@code ffn_gate_inp.weight} exists for that layer.</li>
 *   <li><b>Shared experts</b>: Some layers include a shared expert whose output is
 *       always added unconditionally alongside the top-1 routed expert output.
 *       Detected from the presence of {@code ffn_gate_shexp.weight}.</li>
 *   <li><b>Chunked local attention</b>: Each full-RoPE attention layer uses local
 *       attention chunks; the chunk size comes from {@code attention.chunk_size}
 *       metadata (default 8192). NoPE layers may use full or local attention.</li>
 *   <li><b>GQA</b>: Grouped-query attention with separate Q/K/V projections.</li>
 *   <li><b>RMSNorm + SwiGLU</b>: Same as Llama 3.</li>
 * </ul>
 *
 * <p>Supported variant strings (from {@code general.architecture} in GGUF metadata):</p>
 * <ul>
 *   <li>{@code llama4}, {@code llama-4}, {@code llama4-scout}, {@code llama4-maverick}</li>
 * </ul>
 *
 * @author Eclipse Deeplearning4j Contributors
 */
@Slf4j
public class Llama4Architecture implements ModelArchitecture {

    private static final Set<String> SUPPORTED_VARIANTS = Set.of(
            "llama4", "llama-4", "llama4-scout", "llama4-maverick"
    );

    /**
     * Default local attention chunk size for Llama 4 (tokens per chunk).
     */
    private static final int DEFAULT_CHUNK_SIZE = 8192;

    /**
     * Default RoPE layer interval when metadata does not specify one.
     * Every 4th layer is a full-RoPE layer; the rest are NoPE.
     */
    private static final int DEFAULT_ROPE_LAYER_INTERVAL = 4;

    @Override
    public String getName() {
        return "llama4";
    }

    @Override
    public Set<String> getSupportedVariants() {
        return SUPPORTED_VARIANTS;
    }

    @Override
    public String getDefaultChatTemplateType() {
        return "chatml";
    }

    @Override
    public String getModelSystemProperty() {
        return "llama4.gguf.path";
    }

    @Override
    public boolean canHandle(GGMLMetadata metadata) {
        String arch = metadata.getArchitecture();
        if (arch == null) return false;

        String archLower = arch.toLowerCase();
        return archLower.contains("llama4") ||
               archLower.equals("llama-4") ||
               archLower.startsWith("llama4");
    }

    @Override
    public ArchitectureConfig getConfig(GGMLMetadata metadata) {
        int headDim = metadata.getAttentionKeyLength();
        int expertCount = metadata.getExpertCount();
        int expertUsedCount = metadata.getExpertUsedCount() > 0 ? metadata.getExpertUsedCount() : 1;
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
                .numExperts(expertCount)
                .numExpertsPerToken(expertUsedCount)
                // Llama 4 uses split-half (standard) RoPE for RoPE layers, same as Llama 3
                .ropeType(0)
                // fullAttentionInterval stores the rope_layer_interval
                .fullAttentionInterval(metadata.getFullAttentionInterval())
                .build();
    }

    @Override
    public SameDiff buildGraph(GGMLMetadata metadata, Map<String, INDArray> weights, ConversionOptions options) {
        SameDiff sd = SameDiff.create();
        ArchitectureConfig config = getConfig(metadata);

        DataType dtype = options.getTargetDataType();
        int numLayers = config.getNumLayers();
        int ropeLayerInterval = getRopeLayerInterval(metadata);
        int chunkSize = getChunkSize(metadata);
        boolean hasMoE = hasMoELayers(weights, numLayers);
        int expertCount = config.getNumExperts();

        log.info("Building Llama4 graph: {} layers, hidden={}, heads={}, kv_heads={}, headDim={}, " +
                "ropeFreqBase={}, ropeNDim={}, ropeInterval={}, chunkSize={}, hasMoE={}, experts={}, dtype={}",
                numLayers, config.getHiddenSize(),
                config.getNumAttentionHeads(), config.getNumKVHeads(), config.getHeadDimension(),
                config.getRopeFreqBase(), config.getRopeDimensionCount(),
                ropeLayerInterval, chunkSize, hasMoE, expertCount, dtype);

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
        for (int layer = 0; layer < numLayers; layer++) {
            boolean isRopeLayer = isRopeLayer(layer, ropeLayerInterval);
            hidden = buildTransformerBlock(sd, hidden, layer, config, weights, dtype,
                    isRopeLayer, metadata);
        }

        // Final RMS normalization
        hidden = buildRMSNorm(sd, hidden, "model.norm", "output_norm", weights, config);

        // Output projection (LM head)
        INDArray outputWeight = weights.get("output.weight");
        if (outputWeight == null) {
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
            boolean isRopeLayer, GGMLMetadata metadata) {

        String prefix = "blk." + layerIdx;

        // Pre-attention RMS normalization
        SDVariable normed = buildRMSNorm(sd, input,
                "model.layers." + layerIdx + ".input_layernorm",
                prefix + ".attn_norm", weights, config);

        // Self-attention: iRoPE selects RoPE vs NoPE per layer
        SDVariable attnOut = buildiRoPEAttention(sd, normed, layerIdx, config, weights, dtype, isRopeLayer);

        // Residual connection
        SDVariable postAttn = input.add("post_attn_" + layerIdx, attnOut);

        // Pre-FFN RMS normalization
        SDVariable ffnNormed = buildRMSNorm(sd, postAttn,
                "model.layers." + layerIdx + ".post_attention_layernorm",
                prefix + ".ffn_norm", weights, config);

        // Feed-forward: MoE or dense SwiGLU
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
    // iRoPE attention (interleaved RoPE / NoPE)
    // ========================================================================

    /**
     * Build GQA self-attention with iRoPE (interleaved RoPE).
     *
     * <p>For RoPE layers: standard split-half RoPE is applied to all Q and K heads.
     * For NoPE layers: Q and K are split into RoPE dimensions and NoPE dimensions.
     * Only the RoPE dimensions ({@code ropeDimensionCount}) receive rotary embeddings;
     * the remaining NoPE dimensions are left positional-encoding-free. This enables
     * attention over arbitrarily long contexts without distance decay in NoPE heads.</p>
     */
    private SDVariable buildiRoPEAttention(SameDiff sd, SDVariable input, int layerIdx,
            ArchitectureConfig config, Map<String, INDArray> weights, DataType dtype,
            boolean isRopeLayer) {

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
            log.info("Layer {} Llama4 attention: qHeads={}, kvHeads={}, headDim={} (Q out={}, K out={}), " +
                    "isRopeLayer={}",
                    layerIdx, actualNumHeads, numKVHeads, headDim, qOutDim, kOutDim, isRopeLayer);
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

        // Apply RoPE: full RoPE for RoPE layers; partial RoPE for NoPE layers
        if (config.isUseRotaryEmbeddings()) {
            int ropeDim = config.getRopeDimensionCount();
            if (isRopeLayer || ropeDim <= 0 || ropeDim >= headDim) {
                // Full RoPE — all head dimensions receive rotary embeddings
                q = new FusedRoPE(sd, q, config.getRopeType(), 0,
                        config.getRopeFreqBase(), 1.0, ropeDim).outputVariable();
                sd.updateVariableNameAndReference(q, "q_rope_" + layerIdx);

                k = new FusedRoPE(sd, k, config.getRopeType(), 0,
                        config.getRopeFreqBase(), 1.0, ropeDim).outputVariable();
                sd.updateVariableNameAndReference(k, "k_rope_" + layerIdx);
            } else {
                // Partial RoPE (NoPE layer): split head dim into RoPE dims and NoPE dims,
                // apply RoPE only to the first ropeDim dimensions, then concatenate back.
                // Q: [batch, seq, numHeads, headDim] -> slice last dim [0:ropeDim] and [ropeDim:headDim]
                SDVariable qRopePart = q.get(SDIndex.all(), SDIndex.all(), SDIndex.all(),
                        SDIndex.interval(0, ropeDim));
                SDVariable qNopePart = q.get(SDIndex.all(), SDIndex.all(), SDIndex.all(),
                        SDIndex.interval(ropeDim, headDim));

                SDVariable kRopePart = k.get(SDIndex.all(), SDIndex.all(), SDIndex.all(),
                        SDIndex.interval(0, ropeDim));
                SDVariable kNopePart = k.get(SDIndex.all(), SDIndex.all(), SDIndex.all(),
                        SDIndex.interval(ropeDim, headDim));

                // Apply RoPE only to the RoPE portions
                qRopePart = new FusedRoPE(sd, qRopePart, config.getRopeType(), 0,
                        config.getRopeFreqBase(), 1.0, ropeDim).outputVariable();
                sd.updateVariableNameAndReference(qRopePart, "q_rope_part_" + layerIdx);

                kRopePart = new FusedRoPE(sd, kRopePart, config.getRopeType(), 0,
                        config.getRopeFreqBase(), 1.0, ropeDim).outputVariable();
                sd.updateVariableNameAndReference(kRopePart, "k_rope_part_" + layerIdx);

                // Concatenate RoPE and NoPE portions back along last axis
                q = sd.concat("q_rope_" + layerIdx, -1, qRopePart, qNopePart);
                k = sd.concat("k_rope_" + layerIdx, -1, kRopePart, kNopePart);
            }
        }

        // FusedRoPE promotes HALF→FLOAT internally; V must match Q/K dtype
        if (v.dataType() != q.dataType()) {
            v = v.castTo("v_cast_" + layerIdx, q.dataType());
        }

        // Causal dot-product attention with local chunking (chunked attention)
        // dotProductAttentionV2 applies causal masking; chunk-level locality is
        // enforced by limiting the key/value context visible to each query position.
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
     * Mixture-of-Experts FFN block for Llama 4.
     *
     * <p>Llama 4 uses top-1 routing. Some layers include a shared expert whose
     * output is unconditionally added to the routed expert output. The shared
     * expert uses separate weight keys ({@code ffn_gate_shexp.weight}, etc.).</p>
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
            log.info("Layer {} Llama4 MoE: {} experts detected, hasSharedExpert={}",
                    layerIdx, numExperts, weights.containsKey(prefix + ".ffn_gate_shexp.weight"));
        }

        // Routed expert contribution (top-1 weighted sum)
        SDVariable routedOut = null;
        INDArray packedGateExps = weights.get(prefix + ".ffn_gate_exps.weight");
        if (packedGateExps != null) {
            routedOut = buildPackedMoEFFN(sd, input, layerIdx, numExperts, weights, routerWeights, moePrefix);
        } else {
            for (int e = 0; e < numExperts; e++) {
                SDVariable expertOut = buildExpertSwiGLU(sd, input, layerIdx, e, weights, moePrefix);
                SDVariable expertWeight = routerWeights.get(SDIndex.all(), SDIndex.all(), SDIndex.point(e));
                SDVariable weighted = expertOut.mul("weighted_e" + e + "_" + layerIdx, expertWeight);
                routedOut = (routedOut == null) ? weighted : routedOut.add("combine_e" + e + "_" + layerIdx, weighted);
            }
        }

        if (routedOut == null) {
            routedOut = input;
        }

        // Shared expert contribution (always active, unconditional)
        boolean hasSharedExpert = weights.containsKey(prefix + ".ffn_gate_shexp.weight");
        if (hasSharedExpert) {
            SDVariable sharedOut = buildSharedExpert(sd, input, layerIdx, weights, moePrefix);
            routedOut = routedOut.add("moe_out_" + layerIdx, sharedOut);
        }

        return routedOut;
    }

    /**
     * Build shared expert FFN (always-active expert in Llama 4 MoE).
     * Uses dedicated weight keys: ffn_gate_shexp, ffn_up_shexp, ffn_down_shexp.
     */
    private SDVariable buildSharedExpert(SameDiff sd, SDVariable input, int layerIdx,
            Map<String, INDArray> weights, String moePrefix) {

        String prefix = "blk." + layerIdx;
        INDArray gateW = weights.get(prefix + ".ffn_gate_shexp.weight");
        INDArray upW = weights.get(prefix + ".ffn_up_shexp.weight");
        INDArray downW = weights.get(prefix + ".ffn_down_shexp.weight");

        if (gateW == null || upW == null || downW == null) {
            log.warn("Shared expert weights incomplete for layer {}, skipping", layerIdx);
            return input;
        }

        String shexpPrefix = moePrefix + "shared_expert.";
        SDVariable wGate = sd.var(shexpPrefix + "gate_proj.weight", gateW);
        SDVariable wUp = sd.var(shexpPrefix + "up_proj.weight", upW);
        SDVariable wDown = sd.var(shexpPrefix + "down_proj.weight", downW);

        SDVariable gate = QuantizedLinear.matMul(sd, "shexp_gate_" + layerIdx, input, wGate, weights, prefix + ".ffn_gate_shexp.weight", input.dataType());
        SDVariable up = QuantizedLinear.matMul(sd, "shexp_up_" + layerIdx, input, wUp, weights, prefix + ".ffn_up_shexp.weight", input.dataType());
        SDVariable silu = sd.nn.swish(gate);
        SDVariable h = silu.mul("shexp_swiglu_" + layerIdx, up);

        return QuantizedLinear.matMul(sd, "shexp_down_" + layerIdx, h, wDown, weights, prefix + ".ffn_down_shexp.weight", input.dataType());
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

        // Attention layers (separate Q/K/V)
        patterns.put("blk.{layer}.attn_q.weight", "model.layers.{layer}.self_attn.q_proj.weight");
        patterns.put("blk.{layer}.attn_k.weight", "model.layers.{layer}.self_attn.k_proj.weight");
        patterns.put("blk.{layer}.attn_v.weight", "model.layers.{layer}.self_attn.v_proj.weight");
        patterns.put("blk.{layer}.attn_output.weight", "model.layers.{layer}.self_attn.o_proj.weight");

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

        // Shared expert (always-active expert in MoE layers)
        patterns.put("blk.{layer}.ffn_gate_shexp.weight",
                "model.layers.{layer}.block_sparse_moe.shared_expert.gate_proj.weight");
        patterns.put("blk.{layer}.ffn_up_shexp.weight",
                "model.layers.{layer}.block_sparse_moe.shared_expert.up_proj.weight");
        patterns.put("blk.{layer}.ffn_down_shexp.weight",
                "model.layers.{layer}.block_sparse_moe.shared_expert.down_proj.weight");

        // Normalization layers
        patterns.put("blk.{layer}.attn_norm.weight", "model.layers.{layer}.input_layernorm.weight");
        patterns.put("blk.{layer}.ffn_norm.weight", "model.layers.{layer}.post_attention_layernorm.weight");

        return patterns;
    }

    // ========================================================================
    // Utility methods
    // ========================================================================

    /**
     * Determine if a given layer index is a RoPE layer.
     *
     * <p>Llama 4 uses iRoPE: layers at positions divisible by {@code ropeLayerInterval}
     * are full-RoPE layers; all other layers are NoPE layers. Index counting starts at 0,
     * so layer 0 is always a NoPE layer unless the interval is 1.</p>
     *
     * @param layerIdx         zero-based layer index
     * @param ropeLayerInterval  spacing between RoPE layers (e.g., 4 means every 4th layer)
     * @return true if this layer should apply full rotary position embeddings
     */
    private boolean isRopeLayer(int layerIdx, int ropeLayerInterval) {
        if (ropeLayerInterval <= 0) {
            return true; // all layers are RoPE
        }
        // RoPE layers are at positions: interval-1, 2*interval-1, 3*interval-1, ...
        // i.e., (layerIdx + 1) % ropeLayerInterval == 0
        return (layerIdx + 1) % ropeLayerInterval == 0;
    }

    /**
     * Get the RoPE layer interval from metadata.
     * Reads {@code rope_layer_interval} from raw metadata, falling back to
     * {@link #DEFAULT_ROPE_LAYER_INTERVAL} if absent.
     */
    private int getRopeLayerInterval(GGMLMetadata metadata) {
        // Check fullAttentionInterval (used for this in the ArchitectureConfig)
        if (metadata.getFullAttentionInterval() > 0) {
            return metadata.getFullAttentionInterval();
        }
        Map<String, Object> raw = metadata.getRawMetadata();
        if (raw != null) {
            String arch = metadata.getArchitecture();
            if (arch != null) {
                Object val = raw.get(arch.toLowerCase() + ".rope_layer_interval");
                if (val instanceof Number) {
                    return ((Number) val).intValue();
                }
            }
            Object val = raw.get("rope_layer_interval");
            if (val instanceof Number) {
                return ((Number) val).intValue();
            }
        }
        return DEFAULT_ROPE_LAYER_INTERVAL;
    }

    /**
     * Get the local attention chunk size from metadata.
     * Reads {@code attention.chunk_size}, falling back to {@link #DEFAULT_CHUNK_SIZE}.
     */
    private int getChunkSize(GGMLMetadata metadata) {
        Map<String, Object> raw = metadata.getRawMetadata();
        if (raw != null) {
            String arch = metadata.getArchitecture();
            if (arch != null) {
                Object val = raw.get(arch.toLowerCase() + ".attention.chunk_size");
                if (val instanceof Number) {
                    return ((Number) val).intValue();
                }
            }
            Object val = raw.get("attention.chunk_size");
            if (val instanceof Number) {
                return ((Number) val).intValue();
            }
        }
        return DEFAULT_CHUNK_SIZE;
    }

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
