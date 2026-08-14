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
import org.nd4j.linalg.factory.Nd4j;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

/**
 * Architecture handler for OpenELM (Apple).
 *
 * <p>OpenELM is a decoder-only transformer with several distinctive features:</p>
 * <ul>
 *   <li><b>Combined QKV projection</b>: A single qkv_proj weight that is split
 *       into Q, K, and V components. The split dimensions are derived from
 *       per-layer head counts stored in metadata.</li>
 *   <li><b>QK norms</b>: Per-head RMSNorm applied to Q and K before attention,
 *       similar to OLMo2.</li>
 *   <li><b>GQA + RoPE</b>: Grouped-query attention with rotary embeddings.</li>
 *   <li><b>SwiGLU FFN</b>: Standard SwiGLU feed-forward network.</li>
 *   <li><b>Tensor prefix</b>: {@code transformer.layers.{bid}.*} in the original
 *       model, mapped to GGUF {@code blk.{layer}.*} naming.</li>
 * </ul>
 *
 * @author Eclipse Deeplearning4j Contributors
 */
@Slf4j
public class OpenELMArchitecture implements ModelArchitecture {

    private static final Set<String> SUPPORTED_VARIANTS = Set.of(
            "openelm"
    );

    @Override
    public String getName() {
        return "openelm";
    }

    @Override
    public Set<String> getSupportedVariants() {
        return SUPPORTED_VARIANTS;
    }

    @Override
    public String getDefaultChatTemplateType() {
        return "plain";
    }

    @Override
    public String getModelSystemProperty() {
        return "openelm.gguf.path";
    }

    @Override
    public boolean canHandle(GGMLMetadata metadata) {
        String arch = metadata.getArchitecture();
        if (arch == null) return false;
        return arch.toLowerCase().contains("openelm");
    }

    @Override
    public SameDiff buildGraph(GGMLMetadata metadata, Map<String, INDArray> weights, ConversionOptions options) {
        SameDiff sd = SameDiff.create();
        ArchitectureConfig config = getConfig(metadata);
        DataType dtype = options.getTargetDataType();

        int numLayers = config.getNumLayers();
        int hiddenSize = config.getHiddenSize();
        int numHeads = config.getNumAttentionHeads();
        int numKvHeads = config.getNumKVHeads();
        int headDim = config.getHeadDimension();

        log.info("Building OpenELM graph: {} layers, hidden={}, heads={}, kv_heads={}, headDim={}, dtype={}",
                numLayers, hiddenSize, numHeads, numKvHeads, headDim, dtype);

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
            hidden = buildTransformerBlock(sd, hidden, layer, config, weights, dtype);
        }

        // Final RMS normalization
        hidden = buildRMSNorm(sd, hidden, "model.norm", "output_norm", weights, config);

        // Output projection (LM head)
        INDArray outputWeight = weights.get("output.weight");
        if (outputWeight == null) {
            outputWeight = tokenEmbedWeight;
        }
        SDVariable lmHead = sd.var("lm_head.weight", outputWeight);
        QuantizedLinear.matMul(sd, "logits", hidden, lmHead, weights, "output.weight", hidden.dataType());

        return sd;
    }

    // ========================================================================
    // Transformer block
    // ========================================================================

    private SDVariable buildTransformerBlock(SameDiff sd, SDVariable input, int layerIdx,
                                              ArchitectureConfig config, Map<String, INDArray> weights,
                                              DataType dtype) {
        // Pre-attention RMSNorm
        SDVariable normed = buildRMSNorm(sd, input,
                "model.layers." + layerIdx + ".input_layernorm",
                "blk." + layerIdx + ".attn_norm", weights, config);

        // Combined QKV attention with QK norms
        SDVariable attnOut = buildCombinedQKVAttention(sd, normed, layerIdx, config, weights, dtype);

        // Residual
        SDVariable postAttn = input.add("post_attn_" + layerIdx, attnOut);

        // Pre-FFN RMSNorm
        SDVariable ffnNormed = buildRMSNorm(sd, postAttn,
                "model.layers." + layerIdx + ".post_attention_layernorm",
                "blk." + layerIdx + ".ffn_norm", weights, config);

        // SwiGLU FFN
        SDVariable ffnOut = buildSwiGLUFFN(sd, ffnNormed, layerIdx, weights, dtype);

        // Residual
        return postAttn.add("layer_out_" + layerIdx, ffnOut);
    }

    // ========================================================================
    // Combined QKV attention with QK norms
    // ========================================================================

    /**
     * Build attention from a combined qkv_proj weight.
     *
     * <p>The combined weight has shape [qDim + kDim + vDim, hidden].
     * Split dimensions are derived from the weight shape and the known
     * number of Q heads and KV heads. After splitting, per-head RMSNorm
     * is applied to Q and K if norm weights are present.</p>
     */
    private SDVariable buildCombinedQKVAttention(SameDiff sd, SDVariable input, int layerIdx,
                                                   ArchitectureConfig config,
                                                   Map<String, INDArray> weights, DataType dtype) {
        String prefix = "blk." + layerIdx;
        int numHeads = config.getNumAttentionHeads();
        int numKvHeads = config.getNumKVHeads();
        int headDim = config.getHeadDimension();

        INDArray qkvWeight = weights.get(prefix + ".attn_qkv.weight");
        INDArray oWeight = weights.get(prefix + ".attn_output.weight");

        if (qkvWeight == null || oWeight == null) {
            log.warn("Missing QKV/output attention weights for layer {}", layerIdx);
            return input;
        }

        // Derive split dimensions from QKV weight shape
        int qkvOutDim = (int) qkvWeight.shape()[0];

        // If headDim is not known, derive it
        if (headDim <= 0) {
            int totalHeads = numHeads + 2 * numKvHeads;
            headDim = qkvOutDim / totalHeads;
        }

        int qDim = numHeads * headDim;
        int kDim = numKvHeads * headDim;
        int vDim = numKvHeads * headDim;

        if (layerIdx == 0) {
            log.info("Layer {} QKV split: qDim={}, kDim={}, vDim={}, headDim={}, totalQKV={}",
                    layerIdx, qDim, kDim, vDim, headDim, qkvOutDim);
        }

        String attnPrefix = "model.layers." + layerIdx + ".self_attn.";
        SDVariable wqkv = sd.var(attnPrefix + "qkv_proj.weight", qkvWeight);
        SDVariable wo = sd.var(attnPrefix + "o_proj.weight", oWeight);

        // Project QKV
        SDVariable qkv = QuantizedLinear.matMul(sd, "qkv_" + layerIdx, input, wqkv, weights, prefix + ".attn_qkv.weight", input.dataType());

        // Split into Q, K, V
        SDVariable q = qkv.get(SDIndex.all(), SDIndex.all(), SDIndex.interval(0, qDim));
        sd.updateVariableNameAndReference(q, "q_split_" + layerIdx);
        SDVariable k = qkv.get(SDIndex.all(), SDIndex.all(), SDIndex.interval(qDim, qDim + kDim));
        sd.updateVariableNameAndReference(k, "k_split_" + layerIdx);
        SDVariable v = qkv.get(SDIndex.all(), SDIndex.all(), SDIndex.interval(qDim + kDim, qDim + kDim + vDim));
        sd.updateVariableNameAndReference(v, "v_split_" + layerIdx);

        SDVariable batchDim = sd.sizeAt(input, 0);
        SDVariable seqDim = sd.sizeAt(input, 1);

        // Reshape to multi-head: [batch, seq, heads, head_dim]
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

        // QK norms: per-head RMSNorm
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

        // Apply RoPE after QK norms
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

        // Dot-product attention
        SDVariable attnOut = sd.nn.dotProductAttentionV2(
                "attn_out_" + layerIdx,
                q, v, k, null, null,
                0.0, 0.0, true, false
        );

        // Reshape back: [batch, seq, numHeads * headDim]
        int attnOutDim = numHeads * headDim;
        SDVariable outShapeVar = sd.stack("attn_out_shape_" + layerIdx, 0,
                batchDim, seqDim,
                sd.constant(Nd4j.scalar((long) attnOutDim)));
        SDVariable attnFlat = sd.reshape("attn_flat_" + layerIdx, attnOut, outShapeVar);

        return QuantizedLinear.matMul(sd, "attn_proj_" + layerIdx, attnFlat, wo, weights, prefix + ".attn_output.weight", attnFlat.dataType());
    }

    // ========================================================================
    // Per-head RMS normalization (QK norms)
    // ========================================================================

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

        SDVariable gate = QuantizedLinear.matMul(sd, "gate_" + layerIdx, input, wGate, weights, prefix + ".ffn_gate.weight", input.dataType());
        SDVariable up = QuantizedLinear.matMul(sd, "up_" + layerIdx, input, wUp, weights, prefix + ".ffn_up.weight", input.dataType());

        SDVariable silu = sd.nn.swish(gate);
        SDVariable gated = silu.mul("swiglu_" + layerIdx, up);

        return QuantizedLinear.matMul(sd, "down_" + layerIdx, gated, wDown, weights, prefix + ".ffn_down.weight", gated.dataType());
    }

    // ========================================================================
    // RMS Normalization
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

        // Combined QKV
        patterns.put("blk.{layer}.attn_qkv.weight", "model.layers.{layer}.self_attn.qkv_proj.weight");
        patterns.put("blk.{layer}.attn_output.weight", "model.layers.{layer}.self_attn.o_proj.weight");

        // QK norms
        patterns.put("blk.{layer}.attn_q_norm.weight", "model.layers.{layer}.self_attn.q_norm.weight");
        patterns.put("blk.{layer}.attn_k_norm.weight", "model.layers.{layer}.self_attn.k_norm.weight");

        // Normalization
        patterns.put("blk.{layer}.attn_norm.weight", "model.layers.{layer}.input_layernorm.weight");
        patterns.put("blk.{layer}.ffn_norm.weight", "model.layers.{layer}.post_attention_layernorm.weight");

        // SwiGLU FFN
        patterns.put("blk.{layer}.ffn_gate.weight", "model.layers.{layer}.mlp.gate_proj.weight");
        patterns.put("blk.{layer}.ffn_up.weight", "model.layers.{layer}.mlp.up_proj.weight");
        patterns.put("blk.{layer}.ffn_down.weight", "model.layers.{layer}.mlp.down_proj.weight");

        return patterns;
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
                .ropeType(metadata.getRopeType())
                .useRmsNorm(true)
                .useSwiGLU(true)
                .useRotaryEmbeddings(true)
                .decoderOnly(true)
                .build();
    }
}
