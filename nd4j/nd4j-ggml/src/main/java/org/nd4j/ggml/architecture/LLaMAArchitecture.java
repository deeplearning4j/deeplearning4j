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
import org.nd4j.linalg.api.ops.impl.transforms.custom.CausalConv1d;
import org.nd4j.linalg.api.ops.impl.transforms.custom.FusedRoPE;
import org.nd4j.linalg.api.ops.impl.transforms.custom.GatedDeltaRule;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.autodiff.samediff.SDIndex;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Architecture handler for LLaMA and LLaMA-derived models.
 *
 * Handles the broad family of decoder-only transformers that share the
 * LLaMA pattern: RMSNorm, GQA, SwiGLU FFN, RoPE.  This includes LLaMA 1-3,
 * Qwen (all versions), Yi, DeepSeek, InternLM, and any
 * future model that follows the same tensor naming convention in GGUF.
 *
 * <p>Qwen3.5 hybrid models are supported with two layer types:</p>
 * <ul>
 *   <li><b>Full attention layers</b>: Separate Q/K/V with QK norms and output gating</li>
 *   <li><b>GDN (Gated Delta Network) layers</b>: Linear attention with SSM recurrence</li>
 * </ul>
 */
@Slf4j
public class LLaMAArchitecture implements ModelArchitecture {

    private static final Set<String> SUPPORTED_VARIANTS = Set.of(
            "llama", "llama2", "llama3", "codellama",
            "yi", "deepseek",
            "qwen", "qwen2", "qwen3", "qwen3.5", "qwen35",
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
               archLower.contains("qwen");
    }

    @Override
    public SameDiff buildGraph(GGMLMetadata metadata, Map<String, INDArray> weights, ConversionOptions options) {
        SameDiff sd = SameDiff.create();
        ArchitectureConfig config = getConfig(metadata);

        DataType dtype = options.getTargetDataType();
        log.info("Building LLaMA graph: {} layers, hidden={}, heads={}, kv_heads={}, headDim={}, " +
                "ropeFreqBase={}, ropeNDim={}, layerNormEps={}, fullAttnInterval={}, dtype={}",
                config.getNumLayers(), config.getHiddenSize(),
                config.getNumAttentionHeads(), config.getNumKVHeads(), config.getHeadDimension(),
                config.getRopeFreqBase(), config.getRopeDimensionCount(),
                config.getLayerNormEpsilon(), config.getFullAttentionInterval(), dtype);

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

        // Determine layer type from GGUF metadata
        String layerType = getLayerType(config, layerIdx);
        if (layerIdx == 0) {
            log.info("Layer 0 type from metadata: '{}'", layerType);
        }

        // Pre-attention RMS normalization
        SDVariable normed = buildRMSNorm(sd, input,
                "model.layers." + layerIdx + ".input_layernorm",
                prefix + ".attn_norm", weights, config, dtype);

        // Self-attention — dispatch based on layer type from metadata
        SDVariable attnOut;
        switch (layerType) {
            case "linear_attention":
                attnOut = buildGDNAttention(sd, normed, layerIdx, config, weights, dtype);
                break;
            case "full_attention":
                // Full attention with QK norms and output gating (Qwen3.5)
                attnOut = buildGatedAttention(sd, normed, layerIdx, config, weights, dtype);
                break;
            default:
                // Default: standard separate Q/K/V attention (LLaMA, Mistral, etc.)
                attnOut = buildSeparateQKVAttention(sd, normed, layerIdx, config, weights, dtype);
                break;
        }

        // Residual connection
        SDVariable postAttn = input.add("post_attn_" + layerIdx, attnOut);

        // Pre-FFN RMS normalization — try post_attention_norm first, fallback to ffn_norm
        String postAttnNormKey = weights.containsKey(prefix + ".post_attention_norm.weight")
                ? prefix + ".post_attention_norm"
                : prefix + ".ffn_norm";
        SDVariable ffnNormed = buildRMSNorm(sd, postAttn,
                "model.layers." + layerIdx + ".post_attention_layernorm",
                postAttnNormKey, weights, config, dtype);

        // Feed-forward network
        SDVariable ffnOut;
        if (weights.containsKey(prefix + ".ffn_gate_inp.weight")) {
            INDArray routerGate = weights.get(prefix + ".ffn_gate_inp.weight");
            ffnOut = buildMoEFFN(sd, ffnNormed, layerIdx, config, weights, dtype, routerGate);
        } else if (weights.containsKey(prefix + ".ffn_gate.weight")) {
            ffnOut = buildSwiGLUFFN(sd, ffnNormed, layerIdx, config, weights, dtype);
        } else if (weights.containsKey(prefix + ".ffn_up.weight")) {
            ffnOut = buildGELUFFN(sd, ffnNormed, layerIdx, config, weights, dtype);
        } else {
            log.warn("No FFN weights found for layer {}, passing through", layerIdx);
            ffnOut = ffnNormed;
        }

        // Residual connection
        return postAttn.add("layer_out_" + layerIdx, ffnOut);
    }

    /**
     * Get the layer type for the given layer index from GGUF metadata.
     *
     * <p>Resolution order:</p>
     * <ol>
     *   <li>Explicit {@code layer_types} array (if present in metadata)</li>
     *   <li>{@code full_attention_interval} — every Nth layer (0-indexed: N-1, 2N-1, ...) is
     *       "full_attention", all others are "linear_attention"</li>
     *   <li>Default: "default" (standard attention for LLaMA/Mistral etc.)</li>
     * </ol>
     */
    private String getLayerType(ArchitectureConfig config, int layerIdx) {
        // 1. Explicit layer_types array
        List<String> layerTypes = config.getLayerTypes();
        if (layerTypes != null && layerIdx < layerTypes.size()) {
            return layerTypes.get(layerIdx);
        }

        // 2. full_attention_interval from metadata (e.g., Qwen3.5: interval=4 → layers 3,7,11,... are full)
        int interval = config.getFullAttentionInterval();
        if (interval > 0) {
            boolean isFullAttention = ((layerIdx + 1) % interval == 0);
            return isFullAttention ? "full_attention" : "linear_attention";
        }

        return "default";
    }

    // ========================================================================
    // RMS Normalization
    // ========================================================================

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

    /**
     * Apply per-head RMS normalization. Used for QK norms in gated attention.
     * Input shape: [batch, seq, numHeads, headDim]
     * normWeight shape: [headDim]
     */
    private SDVariable applyHeadNorm(SameDiff sd, SDVariable input, String outputName,
            INDArray normWeight, float eps) {
        SDVariable gamma = sd.var(outputName + ".weight", normWeight);
        // Normalize along the last (headDim) dimension
        SDVariable squared = input.mul(input);
        SDVariable meanSquared = squared.mean(true, -1);
        SDVariable rms = sd.math.sqrt(meanSquared.add(eps));
        SDVariable normalized = input.div(rms);
        return normalized.mul(outputName, gamma);
    }

    // ========================================================================
    // Attention variants
    // ========================================================================

    /**
     * Standard separate Q/K/V attention (LLaMA, Mistral, etc.)
     */
    private SDVariable buildSeparateQKVAttention(SameDiff sd, SDVariable input, int layerIdx,
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

        // Derive headDim from K weight shape: kOutDim / numKVHeads
        // This is more reliable than Q which may include gate projections
        int kOutDim = (int) kWeight.shape()[0];
        int headDim = kOutDim / numKVHeads;
        int qOutDim = (int) qWeight.shape()[0];
        int actualNumHeads = qOutDim / headDim;

        if (layerIdx == 0 || layerIdx == 3) {
            log.info("Layer {} separate attention: qHeads={}, kvHeads={}, headDim={} (Q out={}, K out={})",
                    layerIdx, actualNumHeads, numKVHeads, headDim, qOutDim, kOutDim);
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

        // Apply RoPE
        if (config.isUseRotaryEmbeddings()) {
            q = new FusedRoPE(sd, q, config.getRopeType(), 0,
                    config.getRopeFreqBase(), 1.0, config.getRopeDimensionCount()).outputVariable();
            sd.updateVariableNameAndReference(q, "q_rope_" + layerIdx);

            k = new FusedRoPE(sd, k, config.getRopeType(), 0,
                    config.getRopeFreqBase(), 1.0, config.getRopeDimensionCount()).outputVariable();
            sd.updateVariableNameAndReference(k, "k_rope_" + layerIdx);
        }

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

    /**
     * Gated attention for Qwen3.5 full attention layers.
     *
     * <p>Structure:</p>
     * <ol>
     *   <li>Q projection includes both Q and gate: qOutDim = 2 * numHeads * headDim</li>
     *   <li>Q and gate are split; QK norms applied to Q and K</li>
     *   <li>Standard dot-product attention</li>
     *   <li>Output gated: result = attn_out * swish(gate)</li>
     *   <li>Output projection</li>
     * </ol>
     */
    private SDVariable buildGatedAttention(SameDiff sd, SDVariable input, int layerIdx,
            ArchitectureConfig config, Map<String, INDArray> weights, DataType dtype) {

        String prefix = "blk." + layerIdx;
        int numHeads = config.getNumAttentionHeads();
        int numKVHeads = config.getNumKVHeads();
        int headDim = config.getHeadDimension();

        INDArray qWeight = weights.get(prefix + ".attn_q.weight");
        INDArray kWeight = weights.get(prefix + ".attn_k.weight");
        INDArray vWeight = weights.get(prefix + ".attn_v.weight");
        INDArray oWeight = weights.get(prefix + ".attn_output.weight");
        INDArray qNormWeight = weights.get(prefix + ".attn_q_norm.weight");
        INDArray kNormWeight = weights.get(prefix + ".attn_k_norm.weight");

        if (qWeight == null || kWeight == null || vWeight == null || oWeight == null) {
            log.warn("Missing gated attention weights for layer {}", layerIdx);
            return input;
        }

        // Q weight is 2x because it includes Q + gate: [2 * numHeads * headDim, hidden]
        int qOutDim = (int) qWeight.shape()[0];
        int attnDim = numHeads * headDim;  // actual attention Q dimension
        int gateDim = qOutDim - attnDim;    // gate dimension (should equal attnDim)

        if (layerIdx == 3) {
            log.info("Layer {} gated attention: heads={}, kvHeads={}, headDim={}, qOut={}, attnDim={}, gateDim={}",
                    layerIdx, numHeads, numKVHeads, headDim, qOutDim, attnDim, gateDim);
        }

        String attnPrefix = "model.layers." + layerIdx + ".self_attn.";
        SDVariable wq = sd.var(attnPrefix + "q_proj.weight", qWeight);
        SDVariable wk = sd.var(attnPrefix + "k_proj.weight", kWeight);
        SDVariable wv = sd.var(attnPrefix + "v_proj.weight", vWeight);
        SDVariable wo = sd.var(attnPrefix + "o_proj.weight", oWeight);

        // Project Q (includes gate), K, V
        SDVariable qFull = sd.mmul("q_full_" + layerIdx, input, wq.permute(1, 0));
        SDVariable k = sd.mmul("k_" + layerIdx, input, wk.permute(1, 0));
        SDVariable v = sd.mmul("v_" + layerIdx, input, wv.permute(1, 0));

        SDVariable batchDim = sd.sizeAt(input, 0);
        SDVariable seqDim = sd.sizeAt(input, 1);

        // Split Q into attention Q and gate per-head (interleaved layout):
        // Reference: view(B, L, num_heads, head_dim*2) then chunk along last dim
        // Each head has [Q_dims | gate_dims], NOT [all_Q | all_gate]
        SDVariable qgShapeVar = sd.stack("qg_shape_" + layerIdx, 0,
                batchDim, seqDim,
                sd.constant(Nd4j.scalar((long) numHeads)),
                sd.constant(Nd4j.scalar((long) (headDim * 2))));
        SDVariable qgReshaped = sd.reshape("qg_reshaped_" + layerIdx, qFull, qgShapeVar);
        // Split each head's 2*headDim into Q[headDim] and gate[headDim]
        SDVariable q = qgReshaped.get(SDIndex.all(), SDIndex.all(), SDIndex.all(),
                SDIndex.interval(0, headDim));
        sd.updateVariableNameAndReference(q, "q_" + layerIdx);
        SDVariable gatePerHead = qgReshaped.get(SDIndex.all(), SDIndex.all(), SDIndex.all(),
                SDIndex.interval(headDim, headDim * 2));
        // Flatten gate back to [B, L, attnDim] for later use
        SDVariable gateShapeVar = sd.stack("gate_flat_shape_" + layerIdx, 0,
                batchDim, seqDim,
                sd.constant(Nd4j.scalar((long) attnDim)));
        SDVariable gate = sd.reshape("attn_gate_" + layerIdx, gatePerHead, gateShapeVar);

        // Reshape K, V to [batch, seq, kv_heads, headDim]
        SDVariable kvShapeVar = sd.stack("kv_shape_" + layerIdx, 0,
                batchDim, seqDim,
                sd.constant(Nd4j.scalar((long) numKVHeads)),
                sd.constant(Nd4j.scalar((long) headDim)));

        // Q is already [B, L, numHeads, headDim] from the split above
        k = sd.reshape("k_heads_" + layerIdx, k, kvShapeVar);
        v = sd.reshape("v_heads_" + layerIdx, v, kvShapeVar);

        // Apply QK norms (per-head RMS normalization)
        if (qNormWeight != null) {
            q = applyHeadNorm(sd, q, attnPrefix + "q_norm_" + layerIdx, qNormWeight, config.getLayerNormEpsilon());
        }
        if (kNormWeight != null) {
            k = applyHeadNorm(sd, k, attnPrefix + "k_norm_" + layerIdx, kNormWeight, config.getLayerNormEpsilon());
        }

        // Apply RoPE after QK norms
        if (config.isUseRotaryEmbeddings()) {
            q = new FusedRoPE(sd, q, config.getRopeType(), 0,
                    config.getRopeFreqBase(), 1.0, config.getRopeDimensionCount()).outputVariable();
            sd.updateVariableNameAndReference(q, "q_rope_" + layerIdx);

            k = new FusedRoPE(sd, k, config.getRopeType(), 0,
                    config.getRopeFreqBase(), 1.0, config.getRopeDimensionCount()).outputVariable();
            sd.updateVariableNameAndReference(k, "k_rope_" + layerIdx);
        }

        // Dot-product attention
        SDVariable attnOut = sd.nn.dotProductAttentionV2(
                "attn_out_" + layerIdx,
                q, v, k, null, null,
                0.0, 0.0, true, false
        );

        // Reshape: [batch, seq, numHeads, headDim] -> [batch, seq, attnDim]
        SDVariable outShapeVar = sd.stack("attn_out_shape_" + layerIdx, 0,
                batchDim, seqDim,
                sd.constant(Nd4j.scalar((long) attnDim)));
        SDVariable attnFlat = sd.reshape("attn_flat_" + layerIdx, attnOut, outShapeVar);

        // Gate: output = attn_flat * sigmoid(gate)
        // Reference: attn_output = attn_output * torch.sigmoid(gate)
        SDVariable gateActivated = sd.nn.sigmoid("gate_sigmoid_" + layerIdx, gate);
        SDVariable gatedOut = attnFlat.mul("gated_attn_" + layerIdx, gateActivated);

        // Output projection
        return sd.mmul("attn_proj_" + layerIdx, gatedOut, wo.permute(1, 0));
    }

    /**
     * Gated Delta Network (GDN) linear attention for Qwen3.5 hybrid layers.
     *
     * <p>Pipeline:</p>
     * <ol>
     *   <li>QKV projection via fused attn_qkv weight</li>
     *   <li>Causal conv1d with SiLU activation</li>
     *   <li>Split into Q, K, V heads</li>
     *   <li>Compute beta (update gate) and alpha (decay gate)</li>
     *   <li>Gated delta rule recurrence</li>
     *   <li>RMS normalization on output</li>
     *   <li>Swish gating via attn_gate</li>
     *   <li>Output projection via ssm_out</li>
     * </ol>
     */
    private SDVariable buildGDNAttention(SameDiff sd, SDVariable input, int layerIdx,
            ArchitectureConfig config, Map<String, INDArray> weights, DataType dtype) {

        String prefix = "blk." + layerIdx;
        String attnPrefix = "model.layers." + layerIdx + ".gdn.";

        // Load all GDN weights
        INDArray qkvWeight = weights.get(prefix + ".attn_qkv.weight");
        INDArray gateWeight = weights.get(prefix + ".attn_gate.weight");
        INDArray convWeight = weights.get(prefix + ".ssm_conv1d.weight");
        INDArray ssmA = weights.get(prefix + ".ssm_a");
        INDArray alphaWeight = weights.get(prefix + ".ssm_alpha.weight");
        INDArray betaWeight = weights.get(prefix + ".ssm_beta.weight");
        INDArray dtBias = weights.get(prefix + ".ssm_dt.bias");
        INDArray ssmNormWeight = weights.get(prefix + ".ssm_norm.weight");
        INDArray outWeight = weights.get(prefix + ".ssm_out.weight");

        if (qkvWeight == null || outWeight == null) {
            log.warn("Missing GDN weights for layer {}", layerIdx);
            return input;
        }

        // GDN parameters: 16 heads, headDim=128 for both K and V
        int qkvDim = (int) qkvWeight.shape()[0];   // 6144
        int numGdnHeads = (int) ssmA.shape()[0];     // 16
        int headDimKV = qkvDim / (3 * numGdnHeads);  // 128
        int perComponentDim = numGdnHeads * headDimKV; // 2048

        if (layerIdx == 0) {
            log.info("Layer {} GDN: gdnHeads={}, headDimKV={}, qkvDim={}, gateDim={}",
                    layerIdx, numGdnHeads, headDimKV, qkvDim, gateWeight != null ? gateWeight.shape()[0] : 0);
        }

        // 1. QKV projection: [B, L, hidden] -> [B, L, qkvDim]
        SDVariable wqkv = sd.var(attnPrefix + "qkv.weight", qkvWeight);
        SDVariable qkv = sd.mmul("gdn_qkv_" + layerIdx, input, wqkv.permute(1, 0));

        // 2. Causal conv1d with SiLU activation
        if (convWeight != null) {
            SDVariable wConv = sd.var(attnPrefix + "conv.weight", convWeight);
            SDVariable[] convResult = new CausalConv1d(sd, qkv, wConv, null, null, 1).outputVariables();
            qkv = convResult[0];
            sd.updateVariableNameAndReference(qkv, "gdn_conv_" + layerIdx);
        }

        // 3. Split QKV into Q, K, V: each [B, L, perComponentDim]
        SDVariable qProj = qkv.get(SDIndex.all(), SDIndex.all(), SDIndex.interval(0, perComponentDim));
        sd.updateVariableNameAndReference(qProj, "gdn_q_" + layerIdx);
        SDVariable kProj = qkv.get(SDIndex.all(), SDIndex.all(), SDIndex.interval(perComponentDim, 2 * perComponentDim));
        sd.updateVariableNameAndReference(kProj, "gdn_k_" + layerIdx);
        SDVariable vProj = qkv.get(SDIndex.all(), SDIndex.all(), SDIndex.interval(2 * perComponentDim, 3 * perComponentDim));
        sd.updateVariableNameAndReference(vProj, "gdn_v_" + layerIdx);

        // 4. Reshape to [B, L, H, D]
        SDVariable batchDim = sd.sizeAt(input, 0);
        SDVariable seqDim = sd.sizeAt(input, 1);
        SDVariable gdnHeadShape = sd.stack("gdn_head_shape_" + layerIdx, 0,
                batchDim, seqDim,
                sd.constant(Nd4j.scalar((long) numGdnHeads)),
                sd.constant(Nd4j.scalar((long) headDimKV)));

        SDVariable q = sd.reshape("gdn_q_reshaped_" + layerIdx, qProj, gdnHeadShape);
        SDVariable k = sd.reshape("gdn_k_reshaped_" + layerIdx, kProj, gdnHeadShape);
        SDVariable v = sd.reshape("gdn_v_reshaped_" + layerIdx, vProj, gdnHeadShape);

        // 5. L2-normalize Q and K (per head vector, matching use_qk_l2norm_in_kernel=True)
        SDVariable qNormSq = q.mul(q).sum("gdn_q_normsq_" + layerIdx, true, -1);
        q = q.div("gdn_q_l2norm_" + layerIdx, sd.math.sqrt(qNormSq.add(1e-12)));
        SDVariable kNormSq = k.mul(k).sum("gdn_k_normsq_" + layerIdx, true, -1);
        k = k.div("gdn_k_l2norm_" + layerIdx, sd.math.sqrt(kNormSq.add(1e-12)));

        // 5b. Scale Q by 1/sqrt(head_dim) — standard attention scaling
        // Reference: scale = 1 / (query.shape[-1] ** 0.5); query = query * scale
        double qScale = 1.0 / Math.sqrt(headDimKV);
        q = q.mul("gdn_q_scaled_" + layerIdx, qScale);

        // 6. Compute beta (update gate): sigmoid(input @ Wbeta^T) → [B, L, H]
        // Reference: beta = sigmoid(in_proj_b(x))  — NO dt_bias added here
        SDVariable beta;
        if (betaWeight != null) {
            SDVariable wBeta = sd.var(attnPrefix + "beta.weight", betaWeight);
            beta = sd.mmul("gdn_beta_proj_" + layerIdx, input, wBeta.permute(1, 0));
            beta = sd.nn.sigmoid("gdn_beta_" + layerIdx, beta);
        } else {
            beta = q.mean("gdn_beta_" + layerIdx, false, -1);
            beta = beta.mul(0).add(1.0);
        }
        beta = beta.castTo("gdn_beta_cast_" + layerIdx, dtype);

        // 7. Compute gate (decay) in log-domain: g = -exp(A_log) * softplus(a_proj + dt_bias)
        // Reference: g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        // The GatedDeltaRule op applies exp(g) internally, so g must be negative for decay < 1
        SDVariable gateDecay;
        if (ssmA != null && alphaWeight != null) {
            SDVariable aLog = sd.var(attnPrefix + "a", ssmA);           // A_log: [H]
            SDVariable expALog = sd.math.exp(aLog);                      // exp(A_log): [H]
            SDVariable wAlpha = sd.var(attnPrefix + "alpha.weight", alphaWeight);
            SDVariable aProj = sd.mmul("gdn_alpha_proj_" + layerIdx, input, wAlpha.permute(1, 0)); // [B,L,H]
            if (dtBias != null) {
                aProj = aProj.add("gdn_a_plus_bias_" + layerIdx, sd.var(attnPrefix + "dt.bias", dtBias));
            }
            SDVariable sp = sd.nn.softplus("gdn_softplus_" + layerIdx, aProj); // softplus(a + bias): [B,L,H]
            // g = -exp(A_log) * softplus(a + bias), broadcast [H] * [B,L,H] -> [B,L,H]
            gateDecay = sp.mul(expALog).neg("gdn_gate_decay_" + layerIdx);
            gateDecay = gateDecay.castTo("gdn_gate_decay_cast_" + layerIdx, dtype);
        } else if (ssmA != null) {
            // Fallback: use ssmA directly as log-domain gate (negative)
            SDVariable a = sd.var(attnPrefix + "a", ssmA);
            SDVariable negA = a.neg();
            SDVariable onesBlh = sd.onesLike("gdn_ones_blh_" + layerIdx, beta);
            gateDecay = onesBlh.mul("gdn_gate_decay_" + layerIdx, negA);
            gateDecay = gateDecay.castTo("gdn_gate_decay_cast_" + layerIdx, dtype);
        } else {
            // No decay: gate=0 so exp(0)=1 (identity)
            gateDecay = sd.zerosLike("gdn_gate_decay_" + layerIdx, beta);
        }

        // 8. Gated delta rule: [B, L, H, D] -> [B, L, H, D]
        SDVariable[] gdrResult = new GatedDeltaRule(sd, q, k, v, beta, gateDecay).outputVariables();
        SDVariable gdnOut = gdrResult[0];
        sd.updateVariableNameAndReference(gdnOut, "gdn_out_" + layerIdx);

        // 9. Gated RMSNorm per-head: output = RMSNorm(gdnOut) * weight * SiLU(z)
        // Reference: Qwen3_5RMSNormGated — RMSNorm FIRST, then gate with SiLU(z)
        // gdnOut is still [B, L, H, D] here
        if (ssmNormWeight != null) {
            // Per-head RMSNorm: normalize along last dim (headDim=128), weight is [128]
            gdnOut = applyHeadNorm(sd, gdnOut, attnPrefix + "ssm_norm_" + layerIdx,
                    ssmNormWeight, config.getLayerNormEpsilon());
        }
        if (gateWeight != null) {
            SDVariable wGate = sd.var(attnPrefix + "gate.weight", gateWeight);
            SDVariable z = sd.mmul("gdn_gate_proj_" + layerIdx, input, wGate.permute(1, 0)); // [B,L,value_dim]
            SDVariable zReshaped = sd.reshape("gdn_z_reshaped_" + layerIdx, z, gdnHeadShape); // [B,L,H,D]
            SDVariable gateAct = sd.nn.swish("gdn_gate_act_" + layerIdx, zReshaped);
            gdnOut = gdnOut.mul("gdn_gated_" + layerIdx, gateAct);
        }

        // 10. Reshape from [B, L, H, D] to [B, L, H*D]
        SDVariable flatShape = sd.stack("gdn_flat_shape_" + layerIdx, 0,
                batchDim, seqDim,
                sd.constant(Nd4j.scalar((long) perComponentDim)));
        gdnOut = sd.reshape("gdn_flat_" + layerIdx, gdnOut, flatShape);

        // 11. Output projection: [B, L, perComponentDim] -> [B, L, hidden]
        SDVariable wOut = sd.var(attnPrefix + "out.weight", outWeight);
        return sd.mmul("gdn_proj_" + layerIdx, gdnOut, wOut.permute(1, 0));
    }

    /**
     * Simple fused QKV attention (no SSM, no gating).
     */
    private SDVariable buildFusedQKVAttention(SameDiff sd, SDVariable input, int layerIdx,
            ArchitectureConfig config, Map<String, INDArray> weights, DataType dtype) {

        String prefix = "blk." + layerIdx;
        int numHeads = config.getNumAttentionHeads();
        int numKVHeads = config.getNumKVHeads();
        int headDim = config.getHeadDimension();

        INDArray qkvWeight = weights.get(prefix + ".attn_qkv.weight");
        INDArray oWeight = weights.get(prefix + ".attn_output.weight");

        if (qkvWeight == null || oWeight == null) {
            log.warn("Missing fused QKV attention weights for layer {}", layerIdx);
            return input;
        }

        if (headDim <= 0) {
            // Derive from QKV shape
            int qkvOutDim = (int) qkvWeight.shape()[0];
            int totalHeads = numHeads + 2 * numKVHeads;
            headDim = qkvOutDim / totalHeads;
        }

        int qDim = numHeads * headDim;
        int kDim = numKVHeads * headDim;
        int vDim = numKVHeads * headDim;

        String attnPrefix = "model.layers." + layerIdx + ".self_attn.";
        SDVariable wqkv = sd.var(attnPrefix + "qkv_proj.weight", qkvWeight);
        SDVariable wo = sd.var(attnPrefix + "o_proj.weight", oWeight);

        SDVariable qkv = sd.mmul("qkv_" + layerIdx, input, wqkv.permute(1, 0));

        INDArray qkvBias = weights.get(prefix + ".attn_qkv.bias");
        if (qkvBias != null) {
            qkv = qkv.add(sd.var(attnPrefix + "qkv_proj.bias", qkvBias));
        }

        SDVariable q = qkv.get(SDIndex.all(), SDIndex.all(), SDIndex.interval(0, qDim));
        sd.updateVariableNameAndReference(q, "q_split_" + layerIdx);
        SDVariable k = qkv.get(SDIndex.all(), SDIndex.all(), SDIndex.interval(qDim, qDim + kDim));
        sd.updateVariableNameAndReference(k, "k_split_" + layerIdx);
        SDVariable v = qkv.get(SDIndex.all(), SDIndex.all(), SDIndex.interval(qDim + kDim, qDim + kDim + vDim));
        sd.updateVariableNameAndReference(v, "v_split_" + layerIdx);

        SDVariable batchDim = sd.sizeAt(input, 0);
        SDVariable seqDim = sd.sizeAt(input, 1);

        SDVariable qShapeVar = sd.stack("q_shape_" + layerIdx, 0,
                batchDim, seqDim,
                sd.constant(Nd4j.scalar((long) numHeads)),
                sd.constant(Nd4j.scalar((long) headDim)));
        SDVariable kvShapeVar = sd.stack("kv_shape_" + layerIdx, 0,
                batchDim, seqDim,
                sd.constant(Nd4j.scalar((long) numKVHeads)),
                sd.constant(Nd4j.scalar((long) headDim)));

        q = sd.reshape("q_heads_" + layerIdx, q, qShapeVar);
        k = sd.reshape("k_heads_" + layerIdx, k, kvShapeVar);
        v = sd.reshape("v_heads_" + layerIdx, v, kvShapeVar);

        if (config.isUseRotaryEmbeddings()) {
            q = new FusedRoPE(sd, q, config.getRopeType(), 0,
                    config.getRopeFreqBase(), 1.0, config.getRopeDimensionCount()).outputVariable();
            sd.updateVariableNameAndReference(q, "q_rope_" + layerIdx);

            k = new FusedRoPE(sd, k, config.getRopeType(), 0,
                    config.getRopeFreqBase(), 1.0, config.getRopeDimensionCount()).outputVariable();
            sd.updateVariableNameAndReference(k, "k_rope_" + layerIdx);
        }

        SDVariable attnOut = sd.nn.dotProductAttentionV2(
                "attn_out_" + layerIdx,
                q, v, k, null, null,
                0.0, 0.0, true, false
        );

        int attnOutDim = numHeads * headDim;
        SDVariable outShapeVar = sd.stack("attn_out_shape_" + layerIdx, 0,
                batchDim, seqDim,
                sd.constant(Nd4j.scalar((long) attnOutDim)));
        SDVariable attnFlat = sd.reshape("attn_flat_" + layerIdx, attnOut, outShapeVar);

        return sd.mmul("attn_proj_" + layerIdx, attnFlat, wo.permute(1, 0));
    }

    // ========================================================================
    // FFN variants
    // ========================================================================

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

    private SDVariable buildGELUFFN(SameDiff sd, SDVariable input, int layerIdx,
            ArchitectureConfig config, Map<String, INDArray> weights, DataType dtype) {

        String prefix = "blk." + layerIdx;

        INDArray upWeight = weights.get(prefix + ".ffn_up.weight");
        INDArray downWeight = weights.get(prefix + ".ffn_down.weight");

        if (upWeight == null || downWeight == null) {
            log.warn("Missing GELU FFN weights for layer {}", layerIdx);
            return input;
        }

        String mlpPrefix = "model.layers." + layerIdx + ".mlp.";
        SDVariable wUp = sd.var(mlpPrefix + "up_proj.weight", upWeight);
        SDVariable wDown = sd.var(mlpPrefix + "down_proj.weight", downWeight);

        SDVariable up = sd.mmul("up_" + layerIdx, input, wUp.permute(1, 0));
        SDVariable activated = sd.nn.gelu("gelu_" + layerIdx, up);
        return sd.mmul("down_" + layerIdx, activated, wDown.permute(1, 0));
    }

    /**
     * Build a Mixture-of-Experts FFN block.
     */
    private SDVariable buildMoEFFN(SameDiff sd, SDVariable input, int layerIdx,
            ArchitectureConfig config, Map<String, INDArray> weights, DataType dtype,
            INDArray routerGateWeight) {

        String prefix = "blk." + layerIdx;
        String mlpPrefix = "model.layers." + layerIdx + ".mlp.";

        SDVariable gate = sd.var(mlpPrefix + "gate.weight", routerGateWeight);
        SDVariable routerLogits = sd.mmul("router_logits_" + layerIdx, input, gate.permute(1, 0));
        SDVariable routerWeights = sd.nn.softmax("router_weights_" + layerIdx, routerLogits, -1);

        int numExperts = 0;
        while (weights.containsKey(prefix + ".ffn_gate." + numExperts + ".weight")) {
            numExperts++;
        }

        if (numExperts == 0) {
            log.warn("MoE router found but no expert weights for layer {}", layerIdx);
            return input;
        }

        SDVariable combined = null;
        for (int e = 0; e < numExperts; e++) {
            SDVariable expertOut = buildExpertSwiGLU(sd, input, layerIdx, e, weights, dtype);
            SDVariable expertWeight = routerWeights.get(SDIndex.all(), SDIndex.all(), SDIndex.point(e));
            SDVariable weighted = expertOut.mul("weighted_e" + e + "_" + layerIdx, expertWeight);
            combined = (combined == null) ? weighted : combined.add("combine_e" + e + "_" + layerIdx, weighted);
        }

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

    // ========================================================================
    // Tensor name patterns
    // ========================================================================

    @Override
    public Map<String, String> getTensorNamePatterns() {
        Map<String, String> patterns = new HashMap<>();

        patterns.put("token_embd.weight", "model.embed_tokens.weight");
        patterns.put("output.weight", "lm_head.weight");
        patterns.put("output_norm.weight", "model.norm.weight");

        // Attention layers (separate Q/K/V)
        patterns.put("blk.{layer}.attn_q.weight", "model.layers.{layer}.self_attn.q_proj.weight");
        patterns.put("blk.{layer}.attn_k.weight", "model.layers.{layer}.self_attn.k_proj.weight");
        patterns.put("blk.{layer}.attn_v.weight", "model.layers.{layer}.self_attn.v_proj.weight");
        patterns.put("blk.{layer}.attn_output.weight", "model.layers.{layer}.self_attn.o_proj.weight");

        // Fused QKV attention
        patterns.put("blk.{layer}.attn_qkv.weight", "model.layers.{layer}.self_attn.qkv_proj.weight");
        patterns.put("blk.{layer}.attn_qkv.bias", "model.layers.{layer}.self_attn.qkv_proj.bias");
        patterns.put("blk.{layer}.attn_gate.weight", "model.layers.{layer}.self_attn.gate.weight");

        // QK norms (Qwen3.5 gated attention)
        patterns.put("blk.{layer}.attn_q_norm.weight", "model.layers.{layer}.self_attn.q_norm.weight");
        patterns.put("blk.{layer}.attn_k_norm.weight", "model.layers.{layer}.self_attn.k_norm.weight");

        // GDN / SSM tensors
        patterns.put("blk.{layer}.ssm_a", "model.layers.{layer}.gdn.a");
        patterns.put("blk.{layer}.ssm_alpha.weight", "model.layers.{layer}.gdn.alpha.weight");
        patterns.put("blk.{layer}.ssm_beta.weight", "model.layers.{layer}.gdn.beta.weight");
        patterns.put("blk.{layer}.ssm_conv1d.weight", "model.layers.{layer}.gdn.conv.weight");
        patterns.put("blk.{layer}.ssm_dt.bias", "model.layers.{layer}.gdn.dt.bias");
        patterns.put("blk.{layer}.ssm_norm.weight", "model.layers.{layer}.gdn.norm.weight");
        patterns.put("blk.{layer}.ssm_out.weight", "model.layers.{layer}.gdn.out.weight");

        // Attention biases
        patterns.put("blk.{layer}.attn_q.bias", "model.layers.{layer}.self_attn.q_proj.bias");
        patterns.put("blk.{layer}.attn_k.bias", "model.layers.{layer}.self_attn.k_proj.bias");
        patterns.put("blk.{layer}.attn_v.bias", "model.layers.{layer}.self_attn.v_proj.bias");

        // Dense FFN layers
        patterns.put("blk.{layer}.ffn_gate.weight", "model.layers.{layer}.mlp.gate_proj.weight");
        patterns.put("blk.{layer}.ffn_up.weight", "model.layers.{layer}.mlp.up_proj.weight");
        patterns.put("blk.{layer}.ffn_down.weight", "model.layers.{layer}.mlp.down_proj.weight");

        // MoE router gate
        patterns.put("blk.{layer}.ffn_gate_inp.weight", "model.layers.{layer}.mlp.gate.weight");

        // Normalization layers (both standard and Qwen3.5 naming)
        patterns.put("blk.{layer}.attn_norm.weight", "model.layers.{layer}.input_layernorm.weight");
        patterns.put("blk.{layer}.ffn_norm.weight", "model.layers.{layer}.post_attention_layernorm.weight");
        patterns.put("blk.{layer}.post_attention_norm.weight", "model.layers.{layer}.post_attention_layernorm.weight");

        return patterns;
    }
}
