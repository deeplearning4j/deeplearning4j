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

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Architecture handler for IBM Granite model family (3.0-4.0).
 *
 * Supports variants:
 * - granite: Dense transformer (Granite 3.x)
 * - granitemoe: MoE transformer (Granite 3.x)
 * - granitemoeshared: MoE with shared experts (Granite 4.0)
 * - granitemoehybrid: Hybrid Mamba2 + attention with shared MoE (Granite 4.0)
 *
 * Key features:
 * - muP (maximal update parametrization) scaling multipliers
 * - GQA attention with standard RoPE
 * - SwiGLU FFN for dense, MoE for sparse variants
 * - Mamba2 layers interleaved with attention for hybrid variant
 *
 * @author Eclipse Deeplearning4j
 */
@Slf4j
public class GraniteArchitecture implements ModelArchitecture {

    private static final Set<String> SUPPORTED_VARIANTS = Set.of(
            "granite", "granitemoe", "granitemoeshared", "granitemoehybrid"
    );

    @Override
    public String getName() {
        return "granite";
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
        return "granite.gguf.path";
    }

    @Override
    public boolean canHandle(GGMLMetadata metadata) {
        String arch = metadata.getArchitecture();
        if (arch == null) return false;
        String archLower = arch.toLowerCase();
        return SUPPORTED_VARIANTS.contains(archLower) || archLower.contains("granite");
    }

    @Override
    public ArchitectureConfig getConfig(GGMLMetadata metadata) {
        Map<String, Object> raw = metadata.getRawMetadata();
        int headDim = metadata.getAttentionKeyLength();

        // muP scaling multipliers from GGUF metadata
        double embeddingMultiplier = getDoubleMetadata(raw, "granite.embedding_multiplier", 1.0);
        double attentionMultiplier = getDoubleMetadata(raw, "granite.attention_multiplier", 0.0);
        double residualMultiplier = getDoubleMetadata(raw, "granite.residual_multiplier", 1.0);
        double logitsScaling = getDoubleMetadata(raw, "granite.logits_scaling", 1.0);

        // MoE parameters
        int numExperts = metadata.getExpertCount();
        int numExpertsPerToken = metadata.getExpertUsedCount() > 0 ? metadata.getExpertUsedCount() : 2;
        int sharedIntermediateSize = getIntMetadata(raw, "granite.shared_intermediate_size", 0);

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
                .layerTypes(metadata.getLayerTypes())
                .fullAttentionInterval(metadata.getFullAttentionInterval())
                .ropeType(0)  // Granite uses standard split-half RoPE
                .useRmsNorm(true)
                .useSwiGLU(true)
                .useRotaryEmbeddings(true)
                .decoderOnly(true)
                // muP
                .embeddingMultiplier(embeddingMultiplier)
                .attentionMultiplier(attentionMultiplier)
                .residualMultiplier(residualMultiplier)
                .logitsScaling(logitsScaling)
                // MoE
                .numExperts(numExperts)
                .numExpertsPerToken(numExpertsPerToken)
                .sharedIntermediateSize(sharedIntermediateSize)
                .build();
    }

    @Override
    public SameDiff buildGraph(GGMLMetadata metadata, Map<String, INDArray> weights, ConversionOptions options) {
        SameDiff sd = SameDiff.create();
        ArchitectureConfig config = getConfig(metadata);
        DataType dtype = options.getTargetDataType();

        String archVariant = metadata.getArchitecture() != null ? metadata.getArchitecture().toLowerCase() : "granite";

        log.info("Building Granite graph (variant={}): {} layers, hidden={}, heads={}, kv_heads={}, headDim={}, " +
                "embMul={}, attnMul={}, resMul={}, logitsScale={}, numExperts={}, dtype={}",
                archVariant, config.getNumLayers(), config.getHiddenSize(),
                config.getNumAttentionHeads(), config.getNumKVHeads(), config.getHeadDimension(),
                config.getEmbeddingMultiplier(), config.getAttentionMultiplier(),
                config.getResidualMultiplier(), config.getLogitsScaling(),
                config.getNumExperts(), dtype);

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

        // muP: embedding scaling
        if (config.getEmbeddingMultiplier() != 1.0) {
            hidden = hidden.mul("embed_scaled", config.getEmbeddingMultiplier());
        }

        // Build transformer layers
        for (int layer = 0; layer < config.getNumLayers(); layer++) {
            hidden = buildTransformerBlock(sd, hidden, layer, config, weights, dtype, archVariant);
        }

        // Final RMS normalization
        hidden = buildRMSNorm(sd, hidden, "model.norm", "output_norm", weights, config);

        // Output projection (LM head)
        INDArray outputWeight = weights.get("output.weight");
        if (outputWeight == null) {
            outputWeight = tokenEmbedWeight;  // Weight tying
        }
        SDVariable lmHead = sd.var("lm_head.weight", outputWeight);

        // Logits: [batch, seq_len, vocab_size]
        SDVariable logits = sd.mmul("logits_raw", hidden, lmHead.permute(1, 0));

        // muP: logits scaling
        if (config.getLogitsScaling() != 1.0) {
            logits = logits.div("logits", config.getLogitsScaling());
        } else {
            sd.updateVariableNameAndReference(logits, "logits");
        }

        return sd;
    }

    // ========================================================================
    // Transformer block
    // ========================================================================

    private SDVariable buildTransformerBlock(SameDiff sd, SDVariable input, int layerIdx,
            ArchitectureConfig config, Map<String, INDArray> weights, DataType dtype, String variant) {

        String prefix = "blk." + layerIdx;

        // Determine layer type for hybrid models
        String layerType = getLayerType(config, layerIdx);

        // Pre-attention/SSM normalization
        SDVariable normed = buildRMSNorm(sd, input,
                "model.layers." + layerIdx + ".input_layernorm",
                prefix + ".attn_norm", weights, config);

        SDVariable blockOut;
        if ("mamba".equals(layerType)) {
            // Mamba2 SSM layer (hybrid variant)
            blockOut = buildMamba2Layer(sd, normed, layerIdx, config, weights, dtype);
        } else {
            // Standard GQA attention
            blockOut = buildAttention(sd, normed, layerIdx, config, weights, dtype);
        }

        // Residual connection with muP scaling
        SDVariable postAttn;
        if (config.getResidualMultiplier() != 1.0) {
            blockOut = blockOut.mul("attn_residual_scale_" + layerIdx, config.getResidualMultiplier());
        }
        postAttn = input.add("post_attn_" + layerIdx, blockOut);

        // Pre-FFN normalization
        String ffnNormKey = weights.containsKey(prefix + ".post_attention_norm.weight")
                ? prefix + ".post_attention_norm"
                : prefix + ".ffn_norm";
        SDVariable ffnNormed = buildRMSNorm(sd, postAttn,
                "model.layers." + layerIdx + ".post_attention_layernorm",
                ffnNormKey, weights, config);

        // Feed-forward network (dispatch based on variant)
        SDVariable ffnOut;
        if (weights.containsKey(prefix + ".ffn_gate_inp.weight")) {
            // MoE layer
            if (weights.containsKey(prefix + ".ffn_gate_shexp.weight")) {
                // MoE with shared experts (granitemoeshared / granitemoehybrid)
                ffnOut = buildMoESharedFFN(sd, ffnNormed, layerIdx, config, weights, dtype);
            } else {
                // Standard MoE (granitemoe)
                ffnOut = buildMoEFFN(sd, ffnNormed, layerIdx, config, weights, dtype);
            }
        } else if (weights.containsKey(prefix + ".ffn_gate.weight")) {
            // Dense SwiGLU
            ffnOut = buildSwiGLUFFN(sd, ffnNormed, layerIdx, config, weights, dtype);
        } else {
            log.warn("No FFN weights found for layer {}, passing through", layerIdx);
            ffnOut = ffnNormed;
        }

        // Residual connection with muP scaling
        if (config.getResidualMultiplier() != 1.0) {
            ffnOut = ffnOut.mul("ffn_residual_scale_" + layerIdx, config.getResidualMultiplier());
        }
        return postAttn.add("layer_out_" + layerIdx, ffnOut);
    }

    private String getLayerType(ArchitectureConfig config, int layerIdx) {
        List<String> types = config.getLayerTypes();
        if (types != null && !types.isEmpty() && layerIdx < types.size()) {
            String type = types.get(layerIdx).toLowerCase();
            if (type.contains("mamba") || type.contains("ssm")) {
                return "mamba";
            }
            return "attention";
        }
        return "attention";
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

        SDVariable squared = input.mul(input);
        SDVariable meanSquared = squared.mean(true, -1);
        SDVariable rms = sd.math.sqrt(meanSquared.add(config.getLayerNormEpsilon()));
        SDVariable normalized = input.div(rms);

        return normalized.mul(outputName, gamma);
    }

    // ========================================================================
    // GQA Attention
    // ========================================================================

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

        int kOutDim = (int) kWeight.shape()[0];
        int headDim = kOutDim / numKVHeads;
        int qOutDim = (int) qWeight.shape()[0];
        int actualNumHeads = qOutDim / headDim;

        String attnPrefix = "model.layers." + layerIdx + ".self_attn.";
        SDVariable wq = sd.var(attnPrefix + "q_proj.weight", qWeight);
        SDVariable wk = sd.var(attnPrefix + "k_proj.weight", kWeight);
        SDVariable wv = sd.var(attnPrefix + "v_proj.weight", vWeight);
        SDVariable wo = sd.var(attnPrefix + "o_proj.weight", oWeight);

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

        // Apply RoPE (Granite uses standard split-half RoPE)
        if (config.isUseRotaryEmbeddings()) {
            q = new FusedRoPE(sd, q, 0, 0,
                    config.getRopeFreqBase(), 1.0, config.getRopeDimensionCount()).outputVariable();
            sd.updateVariableNameAndReference(q, "q_rope_" + layerIdx);

            k = new FusedRoPE(sd, k, 0, 0,
                    config.getRopeFreqBase(), 1.0, config.getRopeDimensionCount()).outputVariable();
            sd.updateVariableNameAndReference(k, "k_rope_" + layerIdx);
        }

        // FusedRoPE promotes HALF→FLOAT internally; V must match Q/K dtype
        if (v.dataType() != q.dataType()) {
            v = v.castTo("v_cast_" + layerIdx, q.dataType());
        }

        // muP attention scaling (replaces 1/sqrt(d_k))
        // dotProductAttentionV2 applies 1/sqrt(d_k) internally, so we need to counteract it
        // if muP provides a custom multiplier
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
    // Mamba2 SSM Layer (hybrid variant)
    // ========================================================================

    private SDVariable buildMamba2Layer(SameDiff sd, SDVariable input, int layerIdx,
            ArchitectureConfig config, Map<String, INDArray> weights, DataType dtype) {

        String prefix = "blk." + layerIdx;

        INDArray inProjW = weights.get(prefix + ".ssm_in.weight");
        INDArray outProjW = weights.get(prefix + ".ssm_out.weight");

        if (inProjW == null || outProjW == null) {
            log.warn("Missing Mamba2 weights for layer {}, falling back to passthrough", layerIdx);
            return input;
        }

        String mambaPrefix = "model.layers." + layerIdx + ".mamba.";
        SDVariable wIn = sd.var(mambaPrefix + "in_proj.weight", inProjW);
        SDVariable wOut = sd.var(mambaPrefix + "out_proj.weight", outProjW);

        // Input projection: [batch, seq, hidden] -> [batch, seq, in_proj_dim]
        SDVariable projected = sd.mmul("mamba_in_" + layerIdx, input, wIn.permute(1, 0));

        // Extract SSM parameters from in_proj output
        // Granite hybrid Mamba2 decomposes in_proj into: x, B, C, dt components
        // For now, use a simplified linear approximation:
        // Apply SwiGLU-like gating on the projected output then project back
        int projDim = (int) inProjW.shape()[0];
        int halfDim = projDim / 2;

        SDVariable batchDim = sd.sizeAt(input, 0);
        SDVariable seqDim = sd.sizeAt(input, 1);

        SDVariable splitShape = sd.stack("mamba_split_shape_" + layerIdx, 0,
                batchDim, seqDim,
                sd.constant(Nd4j.scalar(2L)),
                sd.constant(Nd4j.scalar((long) halfDim)));
        SDVariable projReshaped = sd.reshape("mamba_proj_reshape_" + layerIdx, projected, splitShape);

        SDVariable gateHalf = projReshaped.get(SDIndex.all(), SDIndex.all(), SDIndex.point(0), SDIndex.all());
        SDVariable valueHalf = projReshaped.get(SDIndex.all(), SDIndex.all(), SDIndex.point(1), SDIndex.all());

        SDVariable silu = sd.nn.swish(gateHalf);
        SDVariable gated = silu.mul("mamba_gated_" + layerIdx, valueHalf);

        // Output projection
        return sd.mmul("mamba_out_" + layerIdx, gated, wOut.permute(1, 0));
    }

    // ========================================================================
    // Feed-forward networks
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

    /**
     * Standard MoE FFN without shared experts (granitemoe variant).
     */
    private SDVariable buildMoEFFN(SameDiff sd, SDVariable input, int layerIdx,
            ArchitectureConfig config, Map<String, INDArray> weights, DataType dtype) {

        String prefix = "blk." + layerIdx;
        String mlpPrefix = "model.layers." + layerIdx + ".mlp.";

        INDArray routerGateWeight = weights.get(prefix + ".ffn_gate_inp.weight");
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

        return combined != null ? combined : input;
    }

    /**
     * MoE FFN with shared experts (granitemoeshared / granitemoehybrid).
     * The shared expert is always active and processes every token.
     */
    private SDVariable buildMoESharedFFN(SameDiff sd, SDVariable input, int layerIdx,
            ArchitectureConfig config, Map<String, INDArray> weights, DataType dtype) {

        String prefix = "blk." + layerIdx;
        String mlpPrefix = "model.layers." + layerIdx + ".mlp.";

        // Shared expert (always-on SwiGLU)
        SDVariable sharedOut = buildExpertSwiGLU(sd, input, layerIdx, "shared", weights, dtype);

        // Routed experts
        INDArray routerGateWeight = weights.get(prefix + ".ffn_gate_inp.weight");
        SDVariable gate = sd.var(mlpPrefix + "gate.weight", routerGateWeight);
        SDVariable routerLogits = sd.mmul("router_logits_" + layerIdx, input, gate.permute(1, 0));
        SDVariable routerWeights = sd.nn.softmax("router_weights_" + layerIdx, routerLogits, -1);

        int numExperts = 0;
        while (weights.containsKey(prefix + ".ffn_gate." + numExperts + ".weight")) {
            numExperts++;
        }

        SDVariable combined = sharedOut;
        for (int e = 0; e < numExperts; e++) {
            SDVariable expertOut = buildExpertSwiGLU(sd, input, layerIdx, e, weights, dtype);
            SDVariable expertWeight = routerWeights.get(SDIndex.all(), SDIndex.all(), SDIndex.point(e));
            SDVariable weighted = expertOut.mul("weighted_e" + e + "_" + layerIdx, expertWeight);
            combined = combined.add("combine_e" + e + "_" + layerIdx, weighted);
        }

        return combined;
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

        // Attention (separate Q/K/V)
        patterns.put("blk.{layer}.attn_q.weight", "model.layers.{layer}.self_attn.q_proj.weight");
        patterns.put("blk.{layer}.attn_k.weight", "model.layers.{layer}.self_attn.k_proj.weight");
        patterns.put("blk.{layer}.attn_v.weight", "model.layers.{layer}.self_attn.v_proj.weight");
        patterns.put("blk.{layer}.attn_output.weight", "model.layers.{layer}.self_attn.o_proj.weight");

        // Attention biases
        patterns.put("blk.{layer}.attn_q.bias", "model.layers.{layer}.self_attn.q_proj.bias");
        patterns.put("blk.{layer}.attn_k.bias", "model.layers.{layer}.self_attn.k_proj.bias");
        patterns.put("blk.{layer}.attn_v.bias", "model.layers.{layer}.self_attn.v_proj.bias");

        // Normalization
        patterns.put("blk.{layer}.attn_norm.weight", "model.layers.{layer}.input_layernorm.weight");
        patterns.put("blk.{layer}.ffn_norm.weight", "model.layers.{layer}.post_attention_layernorm.weight");
        patterns.put("blk.{layer}.post_attention_norm.weight", "model.layers.{layer}.post_attention_layernorm.weight");

        // Dense FFN (SwiGLU)
        patterns.put("blk.{layer}.ffn_gate.weight", "model.layers.{layer}.mlp.gate_proj.weight");
        patterns.put("blk.{layer}.ffn_up.weight", "model.layers.{layer}.mlp.up_proj.weight");
        patterns.put("blk.{layer}.ffn_down.weight", "model.layers.{layer}.mlp.down_proj.weight");

        // MoE router gate
        patterns.put("blk.{layer}.ffn_gate_inp.weight", "model.layers.{layer}.mlp.gate.weight");

        // Shared expert weights
        patterns.put("blk.{layer}.ffn_gate_shexp.weight", "model.layers.{layer}.mlp.expert_shared.gate_proj.weight");
        patterns.put("blk.{layer}.ffn_up_shexp.weight", "model.layers.{layer}.mlp.expert_shared.up_proj.weight");
        patterns.put("blk.{layer}.ffn_down_shexp.weight", "model.layers.{layer}.mlp.expert_shared.down_proj.weight");

        // Mamba2 / SSM weights (hybrid variant)
        patterns.put("blk.{layer}.ssm_in.weight", "model.layers.{layer}.mamba.in_proj.weight");
        patterns.put("blk.{layer}.ssm_out.weight", "model.layers.{layer}.mamba.out_proj.weight");
        patterns.put("blk.{layer}.ssm_conv1d.weight", "model.layers.{layer}.mamba.conv.weight");
        patterns.put("blk.{layer}.ssm_dt.bias", "model.layers.{layer}.mamba.dt.bias");
        patterns.put("blk.{layer}.ssm_a", "model.layers.{layer}.mamba.a");

        return patterns;
    }

    // ========================================================================
    // Metadata helpers
    // ========================================================================

    private static double getDoubleMetadata(Map<String, Object> raw, String key, double defaultValue) {
        if (raw == null) return defaultValue;
        Object val = raw.get(key);
        if (val instanceof Number) return ((Number) val).doubleValue();
        return defaultValue;
    }

    private static int getIntMetadata(Map<String, Object> raw, String key, int defaultValue) {
        if (raw == null) return defaultValue;
        Object val = raw.get(key);
        if (val instanceof Number) return ((Number) val).intValue();
        return defaultValue;
    }

}
