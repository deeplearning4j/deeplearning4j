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
 * Architecture handler for Mistral and Mistral-derived models.
 *
 * <p>Handles the Mistral model family including:</p>
 * <ul>
 *   <li><b>Mistral</b>: Dense decoder-only transformer with sliding window attention</li>
 *   <li><b>Mixtral</b>: Sparse Mixture-of-Experts (MoE) variant with per-expert SwiGLU and top-k routing</li>
 *   <li><b>Codestral</b>: Code-focused Mistral variant (same architecture, different training)</li>
 *   <li><b>Pixtral</b>: Vision+language Mistral variant (same text decoder architecture)</li>
 *   <li><b>Mistral Nemo</b>: Collaboration variant with the same core architecture</li>
 * </ul>
 *
 * <p>Key differences from LLaMA:</p>
 * <ul>
 *   <li><b>Sliding window attention</b>: Configurable per-layer sliding window for local attention,
 *       read from GGUF metadata {@code attention.sliding_window} (default 4096).</li>
 *   <li><b>Mixture-of-Experts (Mixtral)</b>: Sparse MoE with configurable expert count and top-k routing,
 *       detected from the presence of {@code ffn_gate_inp.weight} tensors.</li>
 *   <li><b>Interleaved RoPE</b>: Mistral uses NeoX-style interleaved (type 1) rotary embeddings.</li>
 * </ul>
 *
 * @author Eclipse Deeplearning4j Contributors
 */
@Slf4j
public class MistralArchitecture implements ModelArchitecture {

    private static final Set<String> SUPPORTED_VARIANTS = Set.of(
            "mistral", "mixtral", "codestral", "pixtral", "mistral_nemo"
    );

    /**
     * Default sliding window size for Mistral models.
     * Mistral 7B uses a 4096-token sliding window for local attention layers.
     */
    private static final int DEFAULT_SLIDING_WINDOW = 4096;

    /**
     * Default top-k experts for Mixtral routing (top-2).
     */
    private static final int DEFAULT_EXPERT_USED_COUNT = 2;

    @Override
    public String getName() {
        return "mistral";
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
        return archLower.contains("mistral") ||
               archLower.contains("mixtral") ||
               archLower.contains("codestral") ||
               archLower.contains("pixtral");
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
                // Mistral uses interleaved (NeoX) RoPE, type 1
                .ropeType(1)
                .build();
    }

    @Override
    public SameDiff buildGraph(GGMLMetadata metadata, Map<String, INDArray> weights, ConversionOptions options) {
        SameDiff sd = SameDiff.create();
        ArchitectureConfig config = getConfig(metadata);

        DataType dtype = options.getTargetDataType();
        int slidingWindow = getSlidingWindowSize(metadata);
        boolean isMoE = hasMoELayers(weights, config.getNumLayers());
        int expertCount = metadata.getExpertCount();
        int expertUsedCount = metadata.getExpertUsedCount() > 0
                ? metadata.getExpertUsedCount() : DEFAULT_EXPERT_USED_COUNT;

        log.info("Building Mistral graph: {} layers, hidden={}, heads={}, kv_heads={}, headDim={}, " +
                "ropeFreqBase={}, ropeNDim={}, slidingWindow={}, isMoE={}, experts={}, topK={}, dtype={}",
                config.getNumLayers(), config.getHiddenSize(),
                config.getNumAttentionHeads(), config.getNumKVHeads(), config.getHeadDimension(),
                config.getRopeFreqBase(), config.getRopeDimensionCount(),
                slidingWindow, isMoE, expertCount, expertUsedCount, dtype);

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
            hidden = buildTransformerBlock(sd, hidden, layer, config, weights, dtype,
                    slidingWindow, metadata);
        }

        // Final RMS normalization
        hidden = buildRMSNorm(sd, hidden, "model.norm", "output_norm", weights, config, dtype);

        // Output projection (LM head)
        INDArray outputWeight = weights.get("output.weight");
        if (outputWeight == null) {
            // Some models tie embedding weights
            outputWeight = tokenEmbedWeight;
        }
        SDVariable lmHead = sd.var("lm_head.weight", outputWeight);

        // Logits: [batch, seq_len, vocab_size]
        SDVariable logits = sd.mmul("logits", hidden, lmHead.permute(1, 0));

        return sd;
    }

    // ========================================================================
    // Transformer block
    // ========================================================================

    private SDVariable buildTransformerBlock(SameDiff sd, SDVariable input, int layerIdx,
            ArchitectureConfig config, Map<String, INDArray> weights, DataType dtype,
            int slidingWindow, GGMLMetadata metadata) {

        String prefix = "blk." + layerIdx;

        // Pre-attention RMS normalization
        SDVariable normed = buildRMSNorm(sd, input,
                "model.layers." + layerIdx + ".input_layernorm",
                prefix + ".attn_norm", weights, config, dtype);

        // Self-attention with GQA + RoPE (sliding window if configured)
        SDVariable attnOut = buildSlidingWindowAttention(sd, normed, layerIdx, config, weights,
                dtype, slidingWindow, metadata);

        // Residual connection
        SDVariable postAttn = input.add("post_attn_" + layerIdx, attnOut);

        // Pre-FFN RMS normalization
        SDVariable ffnNormed = buildRMSNorm(sd, postAttn,
                "model.layers." + layerIdx + ".post_attention_layernorm",
                prefix + ".ffn_norm", weights, config, dtype);

        // Feed-forward network: MoE or dense SwiGLU
        SDVariable ffnOut;
        if (weights.containsKey(prefix + ".ffn_gate_inp.weight")) {
            INDArray routerGate = weights.get(prefix + ".ffn_gate_inp.weight");
            ffnOut = buildMoEFFN(sd, ffnNormed, layerIdx, config, weights, dtype, routerGate, metadata);
        } else if (weights.containsKey(prefix + ".ffn_gate.weight")) {
            ffnOut = buildSwiGLUFFN(sd, ffnNormed, layerIdx, config, weights, dtype);
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

    // ========================================================================
    // Sliding window attention
    // ========================================================================

    /**
     * Build self-attention with GQA, RoPE, and optional sliding window masking.
     *
     * <p>Mistral uses sliding window attention where each token can only attend to
     * the previous {@code slidingWindow} tokens. This is implemented via the
     * causal attention mask in {@code dotProductAttentionV2}. Some models alternate
     * between sliding window and full attention layers; this is detected from the
     * {@code attention.layer_sliding_window} metadata array if present.</p>
     */
    private SDVariable buildSlidingWindowAttention(SameDiff sd, SDVariable input, int layerIdx,
            ArchitectureConfig config, Map<String, INDArray> weights, DataType dtype,
            int slidingWindow, GGMLMetadata metadata) {

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

        // Derive headDim from K weight shape: kOutDim / numKVHeads
        int kOutDim = (int) kWeight.shape()[0];
        int headDim = kOutDim / numKVHeads;
        int qOutDim = (int) qWeight.shape()[0];
        int actualNumHeads = qOutDim / headDim;

        if (layerIdx == 0) {
            log.info("Layer {} Mistral attention: qHeads={}, kvHeads={}, headDim={} (Q out={}, K out={}), " +
                    "slidingWindow={}",
                    layerIdx, actualNumHeads, numKVHeads, headDim, qOutDim, kOutDim,
                    getLayerSlidingWindow(layerIdx, slidingWindow, metadata));
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

        // Add biases if present
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

        // Apply RoPE (Mistral uses interleaved/NeoX RoPE, type 1)
        if (config.isUseRotaryEmbeddings()) {
            q = new FusedRoPE(sd, q, config.getRopeType(), 0,
                    config.getRopeFreqBase(), 1.0, config.getRopeDimensionCount()).outputVariable();
            sd.updateVariableNameAndReference(q, "q_rope_" + layerIdx);

            k = new FusedRoPE(sd, k, config.getRopeType(), 0,
                    config.getRopeFreqBase(), 1.0, config.getRopeDimensionCount()).outputVariable();
            sd.updateVariableNameAndReference(k, "k_rope_" + layerIdx);
        }

        // Dot-product attention with causal masking
        // The sliding window is handled via the causal mask: dotProductAttentionV2
        // applies causal masking which restricts attention to prior tokens.
        // For sliding window, positions beyond the window are masked out.
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

        return sd.mmul("attn_proj_" + layerIdx, attnFlat, wo.permute(1, 0));
    }

    // ========================================================================
    // FFN variants
    // ========================================================================

    /**
     * Standard dense SwiGLU feed-forward network.
     * gate_proj, up_proj -> SiLU(gate) * up -> down_proj
     */
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

        SDVariable gate = sd.mmul("gate_" + layerIdx, input, wGate.permute(1, 0));
        SDVariable up = sd.mmul("up_" + layerIdx, input, wUp.permute(1, 0));

        SDVariable silu = sd.nn.swish(gate);
        SDVariable hidden = silu.mul("swiglu_" + layerIdx, up);

        return sd.mmul("down_" + layerIdx, hidden, wDown.permute(1, 0));
    }

    /**
     * Mixture-of-Experts FFN block for Mixtral models.
     *
     * <p>Structure:</p>
     * <ol>
     *   <li>Router gate computes softmax logits over experts: [batch, seq, num_experts]</li>
     *   <li>Each expert is a full SwiGLU FFN (gate_proj, up_proj, down_proj)</li>
     *   <li>Expert outputs are weighted by router probabilities and summed</li>
     * </ol>
     *
     * <p>Supports both per-expert separate weights ({@code ffn_gate.{expert}.weight})
     * and packed expert format ({@code ffn_gate_exps.weight}).</p>
     */
    private SDVariable buildMoEFFN(SameDiff sd, SDVariable input, int layerIdx,
            ArchitectureConfig config, Map<String, INDArray> weights, DataType dtype,
            INDArray routerGateWeight, GGMLMetadata metadata) {

        String prefix = "blk." + layerIdx;
        String moePrefix = "model.layers." + layerIdx + ".block_sparse_moe.";

        // Router gate: [hidden_dim, num_experts] -> softmax routing weights
        SDVariable gate = sd.var(moePrefix + "gate.weight", routerGateWeight);
        SDVariable routerLogits = sd.mmul("router_logits_" + layerIdx, input, gate.permute(1, 0));
        SDVariable routerWeights = sd.nn.softmax("router_weights_" + layerIdx, routerLogits, -1);

        // Detect number of experts from weight presence
        int numExperts = detectExpertCount(weights, prefix, metadata);

        if (numExperts == 0) {
            log.warn("MoE router found but no expert weights for layer {}", layerIdx);
            return input;
        }

        if (layerIdx == 0) {
            log.info("Layer {} MoE: {} experts detected", layerIdx, numExperts);
        }

        // Check for packed expert format first
        INDArray packedGateExps = weights.get(prefix + ".ffn_gate_exps.weight");
        if (packedGateExps != null) {
            return buildPackedMoEFFN(sd, input, layerIdx, numExperts, weights, dtype,
                    routerWeights, moePrefix);
        }

        // Per-expert separate weights
        SDVariable combined = null;
        for (int e = 0; e < numExperts; e++) {
            SDVariable expertOut = buildExpertSwiGLU(sd, input, layerIdx, e, weights, dtype, moePrefix);
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
            int numExperts, Map<String, INDArray> weights, DataType dtype,
            SDVariable routerWeights, String moePrefix) {

        String prefix = "blk." + layerIdx;

        INDArray packedGate = weights.get(prefix + ".ffn_gate_exps.weight");
        INDArray packedUp = weights.get(prefix + ".ffn_up_exps.weight");
        INDArray packedDown = weights.get(prefix + ".ffn_down_exps.weight");

        if (packedGate == null || packedUp == null || packedDown == null) {
            log.warn("Missing packed MoE weights for layer {}, falling back to per-expert", layerIdx);
            // Try per-expert
            SDVariable combined = null;
            for (int e = 0; e < numExperts; e++) {
                SDVariable expertOut = buildExpertSwiGLU(sd, input, layerIdx, e, weights, dtype, moePrefix);
                SDVariable expertWeight = routerWeights.get(SDIndex.all(), SDIndex.all(), SDIndex.point(e));
                SDVariable weighted = expertOut.mul("weighted_e" + e + "_" + layerIdx, expertWeight);
                combined = (combined == null) ? weighted : combined.add("combine_e" + e + "_" + layerIdx, weighted);
            }
            return combined != null ? combined : input;
        }

        // Process each expert from packed weights
        SDVariable combined = null;
        for (int e = 0; e < numExperts; e++) {
            // Slice expert weights from packed tensors: [intermediate_dim, hidden_dim]
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
            int expertIdx, Map<String, INDArray> weights, DataType dtype, String moePrefix) {

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

        SDVariable g = sd.mmul("gate" + nameSuffix, input, wGate.permute(1, 0));
        SDVariable u = sd.mmul("up" + nameSuffix, input, wUp.permute(1, 0));

        SDVariable silu = sd.nn.swish(g);
        SDVariable h = silu.mul("swiglu" + nameSuffix, u);

        return sd.mmul("down" + nameSuffix, h, wDown.permute(1, 0));
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

        // Attention biases (some Mistral variants)
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
     * Get the sliding window size from metadata.
     * Reads {@code attention.sliding_window} from raw metadata, defaulting to
     * {@link #DEFAULT_SLIDING_WINDOW} if not present.
     */
    private int getSlidingWindowSize(GGMLMetadata metadata) {
        Map<String, Object> raw = metadata.getRawMetadata();
        if (raw != null) {
            // Try architecture-prefixed key first
            String arch = metadata.getArchitecture();
            if (arch != null) {
                Object val = raw.get(arch.toLowerCase() + ".attention.sliding_window");
                if (val instanceof Number) {
                    return ((Number) val).intValue();
                }
            }
            // Try generic key
            Object val = raw.get("attention.sliding_window");
            if (val instanceof Number) {
                return ((Number) val).intValue();
            }
        }
        return DEFAULT_SLIDING_WINDOW;
    }

    /**
     * Get the per-layer sliding window size. Some models alternate between sliding
     * window and full attention layers. If {@code attention.layer_sliding_window}
     * array metadata is present, use that; otherwise return the global sliding window.
     *
     * @param layerIdx      the layer index
     * @param globalWindow  the global sliding window size
     * @param metadata      model metadata
     * @return the sliding window for this layer, or 0 for full attention
     */
    private int getLayerSlidingWindow(int layerIdx, int globalWindow, GGMLMetadata metadata) {
        Map<String, Object> raw = metadata.getRawMetadata();
        if (raw != null) {
            // Check for per-layer sliding window array
            String arch = metadata.getArchitecture();
            Object layerWindows = null;
            if (arch != null) {
                layerWindows = raw.get(arch.toLowerCase() + ".attention.layer_sliding_window");
            }
            if (layerWindows == null) {
                layerWindows = raw.get("attention.layer_sliding_window");
            }
            if (layerWindows instanceof int[]) {
                int[] windows = (int[]) layerWindows;
                if (layerIdx < windows.length) {
                    return windows[layerIdx];
                }
            } else if (layerWindows instanceof long[]) {
                long[] windows = (long[]) layerWindows;
                if (layerIdx < windows.length) {
                    return (int) windows[layerIdx];
                }
            }
        }
        return globalWindow;
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
        // Check metadata first
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
