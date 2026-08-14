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
import org.nd4j.linalg.factory.Nd4j;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

/**
 * Architecture handler for OLMo (Open Language Model) v1 and v2.
 *
 * <ul>
 *   <li><b>OLMo v1</b>: Standard transformer with LayerNorm, MHA, SwiGLU FFN.</li>
 *   <li><b>OLMo2</b>: Updated architecture with RMSNorm, GQA, QK norms
 *       (per-head RMSNorm applied to Q and K before attention), post-norm
 *       placement, and RoPE with theta=500000.</li>
 * </ul>
 *
 * <p>The version is auto-detected from metadata: if QK norm weights are present
 * or the architecture string contains "olmo2", the v2 code path is used.</p>
 *
 * @author Eclipse Deeplearning4j Contributors
 */
@Slf4j
public class OLMoArchitecture implements ModelArchitecture {

    private static final Set<String> SUPPORTED_VARIANTS = Set.of(
            "olmo", "olmo2"
    );

    /**
     * Default RoPE theta for OLMo2.
     */
    private static final float OLMO2_ROPE_THETA = 500000.0f;

    @Override
    public String getName() {
        return "olmo";
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
        return "olmo.gguf.path";
    }

    @Override
    public boolean canHandle(GGMLMetadata metadata) {
        String arch = metadata.getArchitecture();
        if (arch == null) return false;
        return arch.toLowerCase().contains("olmo");
    }

    @Override
    public SameDiff buildGraph(GGMLMetadata metadata, Map<String, INDArray> weights, ConversionOptions options) {
        SameDiff sd = SameDiff.create();
        ArchitectureConfig config = getConfig(metadata);
        DataType dtype = options.getTargetDataType();

        boolean isV2 = isOLMo2(metadata, weights);

        int numLayers = config.getNumLayers();
        int hiddenSize = config.getHiddenSize();
        int numHeads = config.getNumAttentionHeads();
        int numKvHeads = config.getNumKVHeads();
        int headDim = config.getHeadDimension();

        log.info("Building OLMo{} graph: {} layers, hidden={}, heads={}, kv_heads={}, headDim={}, dtype={}",
                isV2 ? "2" : "", numLayers, hiddenSize, numHeads, numKvHeads, headDim, dtype);

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
            if (isV2) {
                hidden = buildOLMo2Block(sd, hidden, layer, config, weights, dtype);
            } else {
                hidden = buildOLMoV1Block(sd, hidden, layer, config, weights, dtype);
            }
        }

        // Final normalization
        if (isV2) {
            hidden = buildRMSNorm(sd, hidden, "model.norm", "output_norm", weights, config);
        } else {
            hidden = buildLayerNorm(sd, hidden, "model.norm", "output_norm", weights, config);
        }

        // Output projection (LM head)
        INDArray outputWeight = weights.get("output.weight");
        if (outputWeight == null) {
            outputWeight = tokenEmbedWeight;
        }
        SDVariable lmHead = sd.var("lm_head.weight", outputWeight);
        QuantizedLinear.matMul(sd, "logits", hidden, lmHead, weights, "output.weight", dtype);

        return sd;
    }

    // ========================================================================
    // OLMo v1 block: LayerNorm + MHA + SwiGLU
    // ========================================================================

    private SDVariable buildOLMoV1Block(SameDiff sd, SDVariable input, int layerIdx,
                                         ArchitectureConfig config, Map<String, INDArray> weights,
                                         DataType dtype) {
        // Pre-attention LayerNorm
        SDVariable normed = buildLayerNorm(sd, input,
                "model.layers." + layerIdx + ".input_layernorm",
                "blk." + layerIdx + ".attn_norm", weights, config);

        // Multi-head attention (no GQA for v1, numKvHeads == numHeads)
        SDVariable attnOut = buildAttention(sd, normed, layerIdx, config, weights, dtype, false);

        // Residual
        SDVariable postAttn = input.add("post_attn_" + layerIdx, attnOut);

        // Pre-FFN LayerNorm
        SDVariable ffnNormed = buildLayerNorm(sd, postAttn,
                "model.layers." + layerIdx + ".post_attention_layernorm",
                "blk." + layerIdx + ".ffn_norm", weights, config);

        // SwiGLU FFN
        SDVariable ffnOut = buildSwiGLUFFN(sd, ffnNormed, layerIdx, weights, dtype);

        // Residual
        return postAttn.add("layer_out_" + layerIdx, ffnOut);
    }

    // ========================================================================
    // OLMo2 block: RMSNorm + GQA with QK norms + post-norm
    // ========================================================================

    private SDVariable buildOLMo2Block(SameDiff sd, SDVariable input, int layerIdx,
                                        ArchitectureConfig config, Map<String, INDArray> weights,
                                        DataType dtype) {
        // Pre-attention RMSNorm
        SDVariable normed = buildRMSNorm(sd, input,
                "model.layers." + layerIdx + ".input_layernorm",
                "blk." + layerIdx + ".attn_norm", weights, config);

        // GQA with QK norms
        SDVariable attnOut = buildAttention(sd, normed, layerIdx, config, weights, dtype, true);

        // Post-attention norm (OLMo2 uses post-norm placement)
        SDVariable attnNormed = buildRMSNorm(sd, attnOut,
                "model.layers." + layerIdx + ".post_attn_norm",
                "blk." + layerIdx + ".post_attn_norm", weights, config);

        // Residual
        SDVariable postAttn = input.add("post_attn_" + layerIdx, attnNormed);

        // Pre-FFN RMSNorm
        SDVariable ffnNormed = buildRMSNorm(sd, postAttn,
                "model.layers." + layerIdx + ".post_attention_layernorm",
                "blk." + layerIdx + ".ffn_norm", weights, config);

        // SwiGLU FFN
        SDVariable ffnOut = buildSwiGLUFFN(sd, ffnNormed, layerIdx, weights, dtype);

        // Post-FFN norm (OLMo2 post-norm)
        SDVariable ffnNormedOut = buildRMSNorm(sd, ffnOut,
                "model.layers." + layerIdx + ".post_ffn_norm",
                "blk." + layerIdx + ".post_ffn_norm", weights, config);

        // Residual
        return postAttn.add("layer_out_" + layerIdx, ffnNormedOut);
    }

    // ========================================================================
    // Attention (shared between v1 and v2)
    // ========================================================================

    private SDVariable buildAttention(SameDiff sd, SDVariable input, int layerIdx,
                                       ArchitectureConfig config, Map<String, INDArray> weights,
                                       DataType dtype, boolean useQKNorms) {
        String prefix = "blk." + layerIdx;
        int numHeads = config.getNumAttentionHeads();
        int numKvHeads = config.getNumKVHeads();
        int headDim = config.getHeadDimension();

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

        SDVariable q = QuantizedLinear.matMul(sd, "q_" + layerIdx, input, wq, weights, prefix + ".attn_q.weight", dtype);
        SDVariable k = QuantizedLinear.matMul(sd, "k_" + layerIdx, input, wk, weights, prefix + ".attn_k.weight", dtype);
        SDVariable v = QuantizedLinear.matMul(sd, "v_" + layerIdx, input, wv, weights, prefix + ".attn_v.weight", dtype);

        // Add biases if present (OLMo v1 has biases)
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
                sd.constant(Nd4j.scalar((long) numHeads)),
                sd.constant(Nd4j.scalar((long) headDim)));
        SDVariable kvShapeVar = sd.stack("kv_shape_" + layerIdx, 0,
                batchDim, seqDim,
                sd.constant(Nd4j.scalar((long) numKvHeads)),
                sd.constant(Nd4j.scalar((long) headDim)));

        q = sd.reshape("q_heads_" + layerIdx, q, qShapeVar);
        k = sd.reshape("k_heads_" + layerIdx, k, kvShapeVar);
        v = sd.reshape("v_heads_" + layerIdx, v, kvShapeVar);

        // QK norms: per-head RMSNorm on Q and K (OLMo2)
        if (useQKNorms) {
            INDArray qNormWeight = weights.get(prefix + ".attn_q_norm.weight");
            INDArray kNormWeight = weights.get(prefix + ".attn_k_norm.weight");
            if (qNormWeight != null) {
                q = applyHeadNorm(sd, q, attnPrefix + "q_norm_" + layerIdx,
                        qNormWeight, config.getLayerNormEpsilon());
            }
            if (kNormWeight != null) {
                k = applyHeadNorm(sd, k, attnPrefix + "k_norm_" + layerIdx,
                        kNormWeight, config.getLayerNormEpsilon());
            }
        }

        // Apply RoPE
        if (config.isUseRotaryEmbeddings()) {
            q = sd.nn().fusedRoPE("q_rope_" + layerIdx, q, null,
                    config.getRopeType(), config.getRopeFreqBase(), 1.0,
                    config.getRopeDimensionCount());

            k = sd.nn().fusedRoPE("k_rope_" + layerIdx, k, null,
                    config.getRopeType(), config.getRopeFreqBase(), 1.0,
                    config.getRopeDimensionCount());
        }

        // FusedRoPE promotes HALF→FLOAT internally; V must match Q/K dtype
        v = GGMLDTypePolicy.castTo(v, "v_cast_" + layerIdx, q.dataType());

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

        return QuantizedLinear.matMul(sd, "attn_proj_" + layerIdx, attnFlat, wo, weights, prefix + ".attn_output.weight", dtype);
    }

    // ========================================================================
    // Per-head RMS normalization (QK norms)
    // ========================================================================

    /**
     * Apply per-head RMS normalization.
     * Input shape: [batch, seq, numHeads, headDim]
     * normWeight shape: [headDim]
     */
    private SDVariable applyHeadNorm(SameDiff sd, SDVariable input, String outputName,
                                      INDArray normWeight, float eps) {
        SDVariable gamma = sd.var(outputName + ".weight", normWeight);
        SDVariable squared = input.mul(input);
        SDVariable meanSquared = squared.mean(true, -1);
        SDVariable rms = sd.math.sqrt(meanSquared.add(eps));
        SDVariable normalized = input.div(rms);
        return normalized.mul(outputName, gamma);
    }

    // ========================================================================
    // SwiGLU FFN
    // ========================================================================

    private SDVariable buildSwiGLUFFN(SameDiff sd, SDVariable input, int layerIdx,
                                       Map<String, INDArray> weights, DataType dtype) {
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

        SDVariable gate = QuantizedLinear.matMul(sd, "gate_" + layerIdx, input, wGate, weights, prefix + ".ffn_gate.weight", dtype);
        SDVariable up = QuantizedLinear.matMul(sd, "up_" + layerIdx, input, wUp, weights, prefix + ".ffn_up.weight", dtype);

        SDVariable silu = sd.nn.swish(gate);
        SDVariable gated = silu.mul("swiglu_" + layerIdx, up);

        return QuantizedLinear.matMul(sd, "down_" + layerIdx, gated, wDown, weights, prefix + ".ffn_down.weight", dtype);
    }

    // ========================================================================
    // Normalization helpers
    // ========================================================================

    private SDVariable buildRMSNorm(SameDiff sd, SDVariable input, String outputName,
                                     String weightKey, Map<String, INDArray> weights,
                                     ArchitectureConfig config) {
        INDArray normWeight = weights.get(weightKey + ".weight");
        if (normWeight == null) {
            log.warn("Missing RMS norm weight: {}", weightKey);
            return input;
        }

        SDVariable gamma = sd.var(outputName + ".weight", normWeight);
        SDVariable squared = input.mul(input);
        SDVariable meanSquared = squared.mean(true, -1);
        SDVariable rms = sd.math.sqrt(meanSquared.add(config.getLayerNormEpsilon()));
        SDVariable normalized = input.div(rms);
        return normalized.mul(outputName, gamma);
    }

    /**
     * Standard LayerNorm: (x - mean) / sqrt(var + eps) * gamma + beta
     */
    private SDVariable buildLayerNorm(SameDiff sd, SDVariable input, String outputName,
                                       String weightKey, Map<String, INDArray> weights,
                                       ArchitectureConfig config) {
        INDArray normWeight = weights.get(weightKey + ".weight");
        INDArray normBias = weights.get(weightKey + ".bias");
        if (normWeight == null) {
            log.warn("Missing LayerNorm weight: {}", weightKey);
            return input;
        }

        SDVariable gamma = sd.var(outputName + ".weight", normWeight);
        SDVariable mean = input.mean(true, -1);
        SDVariable centered = input.sub(mean);
        SDVariable variance = centered.mul(centered).mean(true, -1);
        SDVariable std = sd.math.sqrt(variance.add(config.getLayerNormEpsilon()));
        SDVariable normalized = centered.div(std);
        SDVariable scaled = normalized.mul(outputName, gamma);

        if (normBias != null) {
            SDVariable beta = sd.var(outputName + ".bias", normBias);
            scaled = scaled.add(beta);
        }

        return scaled;
    }

    // ========================================================================
    // Version detection
    // ========================================================================

    /**
     * Detect if this is OLMo2 by checking for QK norm weights or architecture name.
     */
    private boolean isOLMo2(GGMLMetadata metadata, Map<String, INDArray> weights) {
        String arch = metadata.getArchitecture();
        if (arch != null && arch.toLowerCase().contains("olmo2")) {
            return true;
        }

        // Check for QK norm weights in layer 0
        return weights.containsKey("blk.0.attn_q_norm.weight") ||
                weights.containsKey("blk.0.attn_k_norm.weight");
    }

    // ========================================================================
    // Tensor name patterns
    // ========================================================================

    @Override
    public Map<String, String> getTensorNamePatterns() {
        Map<String, String> patterns = new HashMap<>();

        // Embeddings
        patterns.put("token_embd.weight", "model.embed_tokens.weight");
        patterns.put("output.weight", "lm_head.weight");
        patterns.put("output_norm.weight", "model.norm.weight");
        patterns.put("output_norm.bias", "model.norm.bias");

        // Attention layers
        patterns.put("blk.{layer}.attn_q.weight", "model.layers.{layer}.self_attn.q_proj.weight");
        patterns.put("blk.{layer}.attn_k.weight", "model.layers.{layer}.self_attn.k_proj.weight");
        patterns.put("blk.{layer}.attn_v.weight", "model.layers.{layer}.self_attn.v_proj.weight");
        patterns.put("blk.{layer}.attn_output.weight", "model.layers.{layer}.self_attn.o_proj.weight");

        // Attention biases (OLMo v1)
        patterns.put("blk.{layer}.attn_q.bias", "model.layers.{layer}.self_attn.q_proj.bias");
        patterns.put("blk.{layer}.attn_k.bias", "model.layers.{layer}.self_attn.k_proj.bias");
        patterns.put("blk.{layer}.attn_v.bias", "model.layers.{layer}.self_attn.v_proj.bias");

        // QK norms (OLMo2)
        patterns.put("blk.{layer}.attn_q_norm.weight", "model.layers.{layer}.self_attn.q_norm.weight");
        patterns.put("blk.{layer}.attn_k_norm.weight", "model.layers.{layer}.self_attn.k_norm.weight");

        // Post-attention norm (OLMo2)
        patterns.put("blk.{layer}.post_attn_norm.weight", "model.layers.{layer}.post_attn_norm.weight");
        patterns.put("blk.{layer}.post_ffn_norm.weight", "model.layers.{layer}.post_ffn_norm.weight");

        // Normalization
        patterns.put("blk.{layer}.attn_norm.weight", "model.layers.{layer}.input_layernorm.weight");
        patterns.put("blk.{layer}.attn_norm.bias", "model.layers.{layer}.input_layernorm.bias");
        patterns.put("blk.{layer}.ffn_norm.weight", "model.layers.{layer}.post_attention_layernorm.weight");
        patterns.put("blk.{layer}.ffn_norm.bias", "model.layers.{layer}.post_attention_layernorm.bias");

        // SwiGLU FFN
        patterns.put("blk.{layer}.ffn_gate.weight", "model.layers.{layer}.mlp.gate_proj.weight");
        patterns.put("blk.{layer}.ffn_up.weight", "model.layers.{layer}.mlp.up_proj.weight");
        patterns.put("blk.{layer}.ffn_down.weight", "model.layers.{layer}.mlp.down_proj.weight");

        return patterns;
    }

    @Override
    public ArchitectureConfig getConfig(GGMLMetadata metadata) {
        int headDim = metadata.getAttentionKeyLength();
        float ropeBase = metadata.getRopeFreqBase();
        // OLMo2 defaults to theta=500000 if not specified
        if (ropeBase <= 0 && isOLMo2Metadata(metadata)) {
            ropeBase = OLMO2_ROPE_THETA;
        }

        return ArchitectureConfig.builder()
                .numLayers(metadata.getNumLayers())
                .hiddenSize(metadata.getHiddenSize())
                .intermediateSize(metadata.getIntermediateSize())
                .numAttentionHeads(metadata.getNumAttentionHeads())
                .numKVHeads(metadata.getNumKVHeads())
                .vocabSize(metadata.getVocabSize())
                .contextLength(metadata.getContextLength())
                .layerNormEpsilon(metadata.getLayerNormEpsilon())
                .ropeFreqBase(ropeBase)
                .ropeDimensionCount(metadata.getRopeDimensionCount())
                .headDim(headDim)
                .ropeType(metadata.getRopeType())
                .useRmsNorm(isOLMo2Metadata(metadata))
                .useSwiGLU(true)
                .useRotaryEmbeddings(true)
                .decoderOnly(true)
                .build();
    }

    private boolean isOLMo2Metadata(GGMLMetadata metadata) {
        String arch = metadata.getArchitecture();
        return arch != null && arch.toLowerCase().contains("olmo2");
    }
}
