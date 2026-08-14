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
 * Architecture handler for GPT-OSS (open-source GPT-style) models.
 *
 * <p>GPT-OSS is a standard decoder-only transformer with:</p>
 * <ul>
 *   <li><b>Standard MHA + RoPE</b>: Multi-head attention with rotary embeddings.</li>
 *   <li><b>Sparse MoE FFN</b>: Mixture-of-Experts with top-k routing. MoE is
 *       detected from the presence of {@code ffn_gate_exps.weight} tensors in the
 *       weight map. Dense (non-MoE) layers fall back to standard SwiGLU.</li>
 *   <li><b>RMSNorm</b>: RMS normalization throughout.</li>
 * </ul>
 *
 * <p>MoE expert weights use packed/batched tensor naming:
 * {@code blk.{layer}.ffn_gate_exps.weight} (shape [numExperts, intermed, hidden]),
 * {@code blk.{layer}.ffn_down_exps.weight}, {@code blk.{layer}.ffn_up_exps.weight}.</p>
 *
 * @author Eclipse Deeplearning4j Contributors
 */
@Slf4j
public class GptOssArchitecture implements ModelArchitecture {

    private static final Set<String> SUPPORTED_VARIANTS = Set.of(
            "gpt-oss", "gptoss"
    );

    @Override
    public String getName() {
        return "gpt-oss";
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
        return "gpt-oss.gguf.path";
    }

    @Override
    public boolean canHandle(GGMLMetadata metadata) {
        String arch = metadata.getArchitecture();
        if (arch == null) return false;
        String archLower = arch.toLowerCase().replace("-", "").replace("_", "");
        return archLower.contains("gptoss");
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

        log.info("Building GPT-OSS graph: {} layers, hidden={}, heads={}, kv_heads={}, headDim={}, dtype={}",
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
        QuantizedLinear.matMul(sd, "logits", hidden, lmHead, weights, "output.weight", dtype);

        return sd;
    }

    // ========================================================================
    // Transformer block
    // ========================================================================

    private SDVariable buildTransformerBlock(SameDiff sd, SDVariable input, int layerIdx,
                                              ArchitectureConfig config, Map<String, INDArray> weights,
                                              DataType dtype) {
        String prefix = "blk." + layerIdx;

        // Pre-attention RMSNorm
        SDVariable normed = buildRMSNorm(sd, input,
                "model.layers." + layerIdx + ".input_layernorm",
                prefix + ".attn_norm", weights, config);

        // Multi-head attention with RoPE
        SDVariable attnOut = buildMHAttention(sd, normed, layerIdx, config, weights, dtype);

        // Residual
        SDVariable postAttn = input.add("post_attn_" + layerIdx, attnOut);

        // Pre-FFN RMSNorm
        SDVariable ffnNormed = buildRMSNorm(sd, postAttn,
                "model.layers." + layerIdx + ".post_attention_layernorm",
                prefix + ".ffn_norm", weights, config);

        // FFN: MoE or dense SwiGLU
        SDVariable ffnOut;
        if (weights.containsKey(prefix + ".ffn_gate_exps.weight")) {
            ffnOut = buildMoEFFN(sd, ffnNormed, layerIdx, config, weights, dtype);
        } else if (weights.containsKey(prefix + ".ffn_gate_inp.weight")) {
            ffnOut = buildPerExpertMoEFFN(sd, ffnNormed, layerIdx, config, weights, dtype);
        } else {
            ffnOut = buildSwiGLUFFN(sd, ffnNormed, layerIdx, weights, dtype);
        }

        // Residual
        return postAttn.add("layer_out_" + layerIdx, ffnOut);
    }

    // ========================================================================
    // Multi-head attention with RoPE
    // ========================================================================

    private SDVariable buildMHAttention(SameDiff sd, SDVariable input, int layerIdx,
                                         ArchitectureConfig config, Map<String, INDArray> weights,
                                         DataType dtype) {
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
    // FFN variants
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

    /**
     * Batched/packed MoE FFN using ffn_gate_exps/ffn_up_exps/ffn_down_exps tensors.
     *
     * <p>Expert weights are packed into a single tensor per component:
     * shape [numExperts, intermedSize, hiddenSize]. The router gate selects
     * experts via softmax routing.</p>
     */
    private SDVariable buildMoEFFN(SameDiff sd, SDVariable input, int layerIdx,
                                    ArchitectureConfig config, Map<String, INDArray> weights,
                                    DataType dtype) {
        String prefix = "blk." + layerIdx;
        String mlpPrefix = "model.layers." + layerIdx + ".mlp.";

        // Router
        INDArray routerWeight = weights.get(prefix + ".ffn_gate_inp.weight");
        if (routerWeight == null) {
            log.warn("Missing MoE router weight for layer {}, using uniform routing", layerIdx);
        }

        // Packed expert weights: [numExperts, intermed, hidden]
        INDArray gateExps = weights.get(prefix + ".ffn_gate_exps.weight");
        INDArray upExps = weights.get(prefix + ".ffn_up_exps.weight");
        INDArray downExps = weights.get(prefix + ".ffn_down_exps.weight");

        if (gateExps == null || upExps == null || downExps == null) {
            log.warn("Missing packed MoE expert weights for layer {}", layerIdx);
            return input;
        }

        int numExperts = (int) gateExps.shape()[0];

        // Router logits -> weights
        SDVariable routerWeights;
        if (routerWeight != null) {
            SDVariable gate = sd.var(mlpPrefix + "gate.weight", routerWeight);
            SDVariable routerLogits = QuantizedLinear.matMul(sd, "router_logits_" + layerIdx, input, gate, weights, prefix + ".ffn_gate_inp.weight", dtype);
            routerWeights = sd.nn.softmax("router_weights_" + layerIdx, routerLogits, -1);
        } else {
            // Uniform routing fallback
            routerWeights = sd.constant(Nd4j.ones(DataType.FLOAT, 1, 1, numExperts).div(numExperts));
        }

        // Process each expert by slicing from the packed tensor
        SDVariable combined = null;
        for (int e = 0; e < numExperts; e++) {
            // Slice expert weights: [intermed, hidden] from [numExperts, intermed, hidden]
            INDArray expertGate = gateExps.get(Nd4j.createFromArray(new long[]{e}));
            INDArray expertUp = upExps.get(Nd4j.createFromArray(new long[]{e}));
            INDArray expertDown = downExps.get(Nd4j.createFromArray(new long[]{e}));

            String nameSuffix = "_" + layerIdx + "_e" + e;
            SDVariable wGate = sd.var(mlpPrefix + "expert_" + e + ".gate_proj.weight", expertGate);
            SDVariable wUp = sd.var(mlpPrefix + "expert_" + e + ".up_proj.weight", expertUp);
            SDVariable wDown = sd.var(mlpPrefix + "expert_" + e + ".down_proj.weight", expertDown);

            SDVariable g = sd.mmul("gate" + nameSuffix, input, wGate.permute(1, 0));
            SDVariable u = sd.mmul("up" + nameSuffix, input, wUp.permute(1, 0));

            SDVariable silu = sd.nn.swish(g);
            SDVariable h = silu.mul("swiglu" + nameSuffix, u);
            SDVariable expertOut = sd.mmul("down" + nameSuffix, h, wDown.permute(1, 0));

            SDVariable expertWeight = routerWeights.get(SDIndex.all(), SDIndex.all(), SDIndex.point(e));
            SDVariable weighted = expertOut.mul("weighted_e" + e + "_" + layerIdx, expertWeight);
            combined = (combined == null) ? weighted : combined.add("combine_e" + e + "_" + layerIdx, weighted);
        }

        return combined != null ? combined : input;
    }

    /**
     * Per-expert MoE FFN using individual ffn_gate.{e}/ffn_up.{e}/ffn_down.{e} tensors.
     */
    private SDVariable buildPerExpertMoEFFN(SameDiff sd, SDVariable input, int layerIdx,
                                             ArchitectureConfig config, Map<String, INDArray> weights,
                                             DataType dtype) {
        String prefix = "blk." + layerIdx;
        String mlpPrefix = "model.layers." + layerIdx + ".mlp.";

        INDArray routerWeight = weights.get(prefix + ".ffn_gate_inp.weight");
        SDVariable gate = sd.var(mlpPrefix + "gate.weight", routerWeight);
        SDVariable routerLogits = QuantizedLinear.matMul(sd, "router_logits_" + layerIdx, input, gate, weights, prefix + ".ffn_gate_inp.weight", dtype);
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
            String nameSuffix = "_" + layerIdx + "_e" + e;
            INDArray gateW = weights.get(prefix + ".ffn_gate." + e + ".weight");
            INDArray upW = weights.get(prefix + ".ffn_up." + e + ".weight");
            INDArray downW = weights.get(prefix + ".ffn_down." + e + ".weight");

            if (gateW == null || upW == null || downW == null) {
                log.warn("Missing expert {} FFN weights for layer {}", e, layerIdx);
                continue;
            }

            SDVariable wGate = sd.var(mlpPrefix + "expert_" + e + ".gate_proj.weight", gateW);
            SDVariable wUp = sd.var(mlpPrefix + "expert_" + e + ".up_proj.weight", upW);
            SDVariable wDown = sd.var(mlpPrefix + "expert_" + e + ".down_proj.weight", downW);

            SDVariable g = QuantizedLinear.matMul(sd, "gate" + nameSuffix, input, wGate, weights, prefix + ".ffn_gate." + e + ".weight", dtype);
            SDVariable u = QuantizedLinear.matMul(sd, "up" + nameSuffix, input, wUp, weights, prefix + ".ffn_up." + e + ".weight", dtype);

            SDVariable silu = sd.nn.swish(g);
            SDVariable h = silu.mul("swiglu" + nameSuffix, u);
            SDVariable expertOut = QuantizedLinear.matMul(sd, "down" + nameSuffix, h, wDown, weights, prefix + ".ffn_down." + e + ".weight", dtype);

            SDVariable expertWeight = routerWeights.get(SDIndex.all(), SDIndex.all(), SDIndex.point(e));
            SDVariable weighted = expertOut.mul("weighted_e" + e + "_" + layerIdx, expertWeight);
            combined = (combined == null) ? weighted : combined.add("combine_e" + e + "_" + layerIdx, weighted);
        }

        return combined != null ? combined : input;
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

        // Attention layers
        patterns.put("blk.{layer}.attn_q.weight", "model.layers.{layer}.self_attn.q_proj.weight");
        patterns.put("blk.{layer}.attn_k.weight", "model.layers.{layer}.self_attn.k_proj.weight");
        patterns.put("blk.{layer}.attn_v.weight", "model.layers.{layer}.self_attn.v_proj.weight");
        patterns.put("blk.{layer}.attn_output.weight", "model.layers.{layer}.self_attn.o_proj.weight");

        // Normalization
        patterns.put("blk.{layer}.attn_norm.weight", "model.layers.{layer}.input_layernorm.weight");
        patterns.put("blk.{layer}.ffn_norm.weight", "model.layers.{layer}.post_attention_layernorm.weight");

        // Dense FFN (SwiGLU)
        patterns.put("blk.{layer}.ffn_gate.weight", "model.layers.{layer}.mlp.gate_proj.weight");
        patterns.put("blk.{layer}.ffn_up.weight", "model.layers.{layer}.mlp.up_proj.weight");
        patterns.put("blk.{layer}.ffn_down.weight", "model.layers.{layer}.mlp.down_proj.weight");

        // MoE router
        patterns.put("blk.{layer}.ffn_gate_inp.weight", "model.layers.{layer}.mlp.gate.weight");

        // Packed MoE expert weights
        patterns.put("blk.{layer}.ffn_gate_exps.weight", "model.layers.{layer}.mlp.gate_exps.weight");
        patterns.put("blk.{layer}.ffn_up_exps.weight", "model.layers.{layer}.mlp.up_exps.weight");
        patterns.put("blk.{layer}.ffn_down_exps.weight", "model.layers.{layer}.mlp.down_exps.weight");

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
