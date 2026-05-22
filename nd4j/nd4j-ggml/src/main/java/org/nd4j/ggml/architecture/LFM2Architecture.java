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
import org.nd4j.linalg.api.ops.impl.transforms.custom.DotProductAttentionV2;
import org.nd4j.linalg.api.ops.impl.transforms.custom.FusedRoPE;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Architecture handler for LFM-2 (Liquid Foundation Model 2).
 *
 * <p>Follows the same graph-construction patterns as {@link LLaMAArchitecture}:
 * FP32 upcasting for matmuls, FP16-safe RMSNorm, KV cache with position offset,
 * DotProductAttentionV2 with causal mask.</p>
 *
 * <p>LFM-2 is a hybrid architecture that interleaves short-convolution blocks
 * with GQA attention blocks. Layer types are detected by probing tensor keys
 * (shortconv.* = conv, attn_q.* = attention).</p>
 */
@Slf4j
public class LFM2Architecture implements ModelArchitecture {

    private static final Set<String> SUPPORTED_VARIANTS = Set.of(
            "lfm2", "lfm2moe"
    );

    @Override
    public String getName() {
        return "lfm2";
    }

    @Override
    public Set<String> getSupportedVariants() {
        return SUPPORTED_VARIANTS;
    }

    @Override
    public String getDefaultChatTemplateType() {
        return "chatml"; // LFM-2 uses ChatML format
    }

    @Override
    public String getModelSystemProperty() {
        return "lfm2.gguf.path";
    }

    @Override
    public boolean canHandle(GGMLMetadata metadata) {
        String arch = metadata.getArchitecture();
        if (arch == null) return false;
        String archLower = arch.toLowerCase();
        return archLower.contains("lfm2") || archLower.contains("liquid");
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

        log.info("Building LFM-2 graph: {} layers, hidden={}, heads={}, kv_heads={}{}, headDim={}, dtype={}",
                numLayers, hiddenSize, numHeads, numKvHeads,
                config.getKvHeadsPerLayer() != null ? " (per-layer)" : "",
                config.getHeadDimension(), dtype);

        // Input placeholder: [batch, seq_len]
        SDVariable inputIds = sd.placeHolder("input_ids", DataType.INT64, -1, -1);

        // KV cache placeholders (same as LLaMA)
        SDVariable positionOffset = sd.placeHolder("position_offset", DataType.INT64);
        SDVariable cachePosition = sd.placeHolder("cache_position", DataType.INT64);
        SDVariable causalMask = sd.placeHolder("_causal_mask", DataType.FLOAT, -1, -1, -1, -1);

        // Per-layer state placeholders:
        // - Attention layers: KV cache [batch, seqLen, kvHeads, headDim]
        // - Conv layers: CausalConv1d state [batch, convDim, kernelSize-1]
        int headDim = config.getHeadDimension();
        Map<Integer, SDVariable> keyCachePlaceholders = new HashMap<>();
        Map<Integer, SDVariable> valueCachePlaceholders = new HashMap<>();
        Map<Integer, SDVariable> convStatePlaceholders = new HashMap<>();
        for (int layer = 0; layer < numLayers; layer++) {
            if (isAttentionLayer(config, layer, weights)) {
                int layerKvHeads = config.getNumKVHeadsForLayer(layer);
                if (layerKvHeads > 0) {
                    SDVariable keyCache = sd.placeHolder("past_key_values." + layer + ".key",
                            dtype, -1, -1, layerKvHeads, headDim);
                    SDVariable valueCache = sd.placeHolder("past_key_values." + layer + ".value",
                            dtype, -1, -1, layerKvHeads, headDim);
                    keyCachePlaceholders.put(layer, keyCache);
                    valueCachePlaceholders.put(layer, valueCache);
                }
            } else {
                // CausalConv1d state: [batch, convDim, kernelSize-1]
                SDVariable convStateIn = sd.placeHolder("past_conv_state." + layer, dtype, -1, -1, -1);
                convStatePlaceholders.put(layer, convStateIn);
            }
        }

        // Token embeddings: [vocab_size, hidden_size]
        INDArray tokenEmbedWeight = weights.get("token_embd.weight");
        if (tokenEmbedWeight == null) {
            throw new IllegalStateException("Missing token embedding weights");
        }
        SDVariable tokenEmbed = sd.var("model.embed_tokens.weight", tokenEmbedWeight);

        // Gather embeddings: [batch, seq_len, hidden_size]
        SDVariable hidden = sd.gather("embedded", tokenEmbed, inputIds, 0);

        // Build transformer layers
        List<String> outputNames = new ArrayList<>();
        for (int layer = 0; layer < numLayers; layer++) {
            hidden = buildTransformerBlock(sd, hidden, layer, config, weights, dtype,
                    positionOffset, cachePosition, causalMask,
                    keyCachePlaceholders.get(layer),
                    valueCachePlaceholders.get(layer),
                    convStatePlaceholders.get(layer));

            // Register per-layer outputs
            if (keyCachePlaceholders.containsKey(layer)) {
                outputNames.add("k_rope_" + layer);
                outputNames.add("v_heads_" + layer);
            } else if (convStatePlaceholders.containsKey(layer)) {
                outputNames.add("conv_state_out_" + layer);
            }
        }

        // Final RMS normalization (token_embd_norm in GGUF — misnomer, it's the output norm)
        hidden = buildRMSNorm(sd, hidden, "model.norm", "token_embd_norm", weights, config, dtype);

        // Output projection (LM head) — tied embeddings, no separate output.weight
        INDArray outputWeight = weights.get("output.weight");
        if (outputWeight == null) {
            outputWeight = tokenEmbedWeight;
        }
        SDVariable lmHead = sd.var("lm_head.weight", outputWeight);

        // Logits in FP32 to prevent overflow
        SDVariable logits = fp32Mmul(sd, "lm_logits", hidden, lmHead.permute(1, 0), dtype);
        outputNames.add("lm_logits");
        sd.setOutputs(outputNames);

        return sd;
    }

    // ========================================================================
    // Transformer block
    // ========================================================================

    private SDVariable buildTransformerBlock(SameDiff sd, SDVariable input, int layerIdx,
                                              ArchitectureConfig config, Map<String, INDArray> weights,
                                              DataType dtype, SDVariable positionOffset,
                                              SDVariable cachePosition, SDVariable causalMask,
                                              SDVariable keyCache, SDVariable valueCache,
                                              SDVariable convStateIn) {
        String prefix = "blk." + layerIdx;

        // Pre-block RMS normalization
        SDVariable normed = buildRMSNorm(sd, input,
                "model.layers." + layerIdx + ".input_layernorm",
                prefix + ".attn_norm", weights, config, dtype);

        // Dispatch: short-conv or GQA attention
        SDVariable blockOut;
        if (isAttentionLayer(config, layerIdx, weights)) {
            blockOut = buildGQAAttention(sd, normed, layerIdx, config, weights, dtype,
                    positionOffset, cachePosition, causalMask, keyCache, valueCache);
        } else {
            blockOut = buildGatedShortConvBlock(sd, normed, layerIdx, config, weights, dtype, convStateIn);
        }

        // Residual
        SDVariable postBlock = input.add("post_block_" + layerIdx, blockOut);

        // Pre-FFN RMS normalization
        SDVariable ffnNormed = buildRMSNorm(sd, postBlock,
                "model.layers." + layerIdx + ".post_attention_layernorm",
                prefix + ".ffn_norm", weights, config, dtype);

        // SwiGLU FFN
        SDVariable ffnOut = buildSwiGLUFFN(sd, ffnNormed, layerIdx, weights, dtype);

        // Residual
        return postBlock.add("layer_out_" + layerIdx, ffnOut);
    }

    // ========================================================================
    // Layer type detection
    // ========================================================================

    private boolean isAttentionLayer(ArchitectureConfig config, int layerIdx, Map<String, INDArray> weights) {
        // 1. Check explicit layer_types from metadata
        List<String> layerTypes = config.getLayerTypes();
        if (layerTypes != null && layerIdx < layerTypes.size()) {
            String type = layerTypes.get(layerIdx).toLowerCase();
            if (type.contains("attention") || type.contains("attn")) return true;
            if (type.contains("conv") || type.contains("short_conv")) return false;
        }

        // 2. Probe tensor keys
        String prefix = "blk." + layerIdx;
        if (weights.containsKey(prefix + ".attn_q.weight")) return true;
        if (weights.containsKey(prefix + ".shortconv.conv.weight") ||
                weights.containsKey(prefix + ".shortconv.in_proj.weight")) return false;

        // 3. Per-layer KV heads: 0 = conv, >0 = attention
        int kvHeads = config.getNumKVHeadsForLayer(layerIdx);
        return kvHeads > 0;
    }

    // ========================================================================
    // GQA Attention (copied from LLaMA patterns)
    // ========================================================================

    private SDVariable buildGQAAttention(SameDiff sd, SDVariable input, int layerIdx,
                                          ArchitectureConfig config, Map<String, INDArray> weights,
                                          DataType dtype, SDVariable positionOffset,
                                          SDVariable cachePosition, SDVariable causalMask,
                                          SDVariable keyCache, SDVariable valueCache) {
        String prefix = "blk." + layerIdx;

        INDArray qWeight = weights.get(prefix + ".attn_q.weight");
        INDArray kWeight = weights.get(prefix + ".attn_k.weight");
        INDArray vWeight = weights.get(prefix + ".attn_v.weight");
        INDArray oWeight = weights.get(prefix + ".attn_output.weight");

        if (qWeight == null || kWeight == null || vWeight == null || oWeight == null) {
            log.warn("Missing attention weights for layer {}", layerIdx);
            return input;
        }

        // Derive headDim from K weight shape (more reliable than metadata)
        int numKvHeads = config.getNumKVHeadsForLayer(layerIdx);
        int kOutDim = (int) kWeight.shape()[0];
        int headDim = kOutDim / numKvHeads;
        int qOutDim = (int) qWeight.shape()[0];
        int actualNumHeads = qOutDim / headDim;

        if (layerIdx < 5) {
            log.info("Layer {} attention: qHeads={}, kvHeads={}, headDim={} (Q out={}, K out={})",
                    layerIdx, actualNumHeads, numKvHeads, headDim, qOutDim, kOutDim);
        }

        String attnPrefix = "model.layers." + layerIdx + ".self_attn.";
        SDVariable wq = sd.var(attnPrefix + "q_proj.weight", qWeight);
        SDVariable wk = sd.var(attnPrefix + "k_proj.weight", kWeight);
        SDVariable wv = sd.var(attnPrefix + "v_proj.weight", vWeight);
        SDVariable wo = sd.var(attnPrefix + "o_proj.weight", oWeight);

        // FP32 matmuls to prevent overflow
        SDVariable q = fp32Mmul(sd, "q_" + layerIdx, input, wq.permute(1, 0), dtype);
        SDVariable k = fp32Mmul(sd, "k_" + layerIdx, input, wk.permute(1, 0), dtype);
        SDVariable v = fp32Mmul(sd, "v_" + layerIdx, input, wv.permute(1, 0), dtype);

        SDVariable batchDim = sd.sizeAt(input, 0);
        SDVariable seqDim = sd.sizeAt(input, 1);

        SDVariable qShapeVar = sd.stack("q_shape_" + layerIdx, 0,
                batchDim, seqDim,
                sd.constant(Nd4j.scalar((long) actualNumHeads)),
                sd.constant(Nd4j.scalar((long) headDim)));
        SDVariable kvShapeVar = sd.stack("kv_shape_" + layerIdx, 0,
                batchDim, seqDim,
                sd.constant(Nd4j.scalar((long) numKvHeads)),
                sd.constant(Nd4j.scalar((long) headDim)));

        q = sd.reshape("q_heads_" + layerIdx, q, qShapeVar);
        k = sd.reshape("k_heads_" + layerIdx, k, kvShapeVar);
        v = sd.reshape("v_heads_" + layerIdx, v, kvShapeVar);

        // Per-head QK RMSNorm (LFM-2 specific, applied before RoPE)
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

        // RoPE with dynamic position offset
        if (config.isUseRotaryEmbeddings()) {
            q = new FusedRoPE(sd, q, positionOffset,
                    config.getRopeType(), config.getRopeFreqBase(), 1.0,
                    config.getRopeDimensionCount()).outputVariable();
            sd.updateVariableNameAndReference(q, "q_rope_" + layerIdx);

            k = new FusedRoPE(sd, k, positionOffset,
                    config.getRopeType(), config.getRopeFreqBase(), 1.0,
                    config.getRopeDimensionCount()).outputVariable();
            sd.updateVariableNameAndReference(k, "k_rope_" + layerIdx);
        }

        // FusedRoPE promotes HALF→FLOAT internally; V must match Q/K dtype
        if (v.dataType() != q.dataType()) {
            v = v.castTo("v_cast_" + layerIdx, q.dataType());
        }

        // Attention with KV cache + causal mask
        SDVariable attnOut = new DotProductAttentionV2(sd,
                q, v, k, null, null,
                keyCache, valueCache, cachePosition, causalMask,
                0.0, 0.0, false, false).outputVariable();
        sd.updateVariableNameAndReference(attnOut, "attn_out_" + layerIdx);

        int attnOutDim = actualNumHeads * headDim;
        SDVariable outShapeVar = sd.stack("attn_out_shape_" + layerIdx, 0,
                batchDim, seqDim,
                sd.constant(Nd4j.scalar((long) attnOutDim)));
        SDVariable attnFlat = sd.reshape("attn_flat_" + layerIdx, attnOut, outShapeVar);

        return fp32Mmul(sd, "attn_proj_" + layerIdx, attnFlat, wo.permute(1, 0), dtype);
    }

    // ========================================================================
    // Gated short-convolution block (LFM-2 specific)
    // ========================================================================

    private SDVariable buildGatedShortConvBlock(SameDiff sd, SDVariable input, int layerIdx,
                                                 ArchitectureConfig config,
                                                 Map<String, INDArray> weights, DataType dtype,
                                                 SDVariable convStateIn) {
        String prefix = "blk." + layerIdx;
        String convPrefix = "model.layers." + layerIdx + ".short_conv.";

        INDArray inProjWeight = weights.get(prefix + ".shortconv.in_proj.weight");
        if (inProjWeight == null) {
            log.warn("Missing shortconv.in_proj weight for layer {}", layerIdx);
            return input;
        }

        // Fused input projection: [B, L, hidden] -> [B, L, 3*hidden]
        SDVariable wInProj = sd.var(convPrefix + "in_proj.weight", inProjWeight);
        SDVariable projected = fp32Mmul(sd, "conv_in_proj_" + layerIdx, input, wInProj.permute(1, 0), dtype);

        // Split into 3 equal chunks: B (input gate), C (output gate), x (value)
        SDVariable[] bCx = sd.split(new String[]{
                "conv_split_b_" + layerIdx, "conv_split_c_" + layerIdx, "conv_split_x_" + layerIdx
        }, projected, 3, -1);
        SDVariable bGate = bCx[0];
        SDVariable cGate = bCx[1];
        SDVariable xVal = bCx[2];

        // Input gating: Bx = B * x
        SDVariable bx = bGate.mul("conv_input_gate_" + layerIdx, xVal);

        // Depthwise causal conv1d with state (no activation)
        INDArray convWeight = weights.get(prefix + ".shortconv.conv.weight");
        if (convWeight != null) {
            SDVariable wConv = sd.var(convPrefix + "conv.weight", convWeight);
            SDVariable[] convResult = new CausalConv1d(sd, bx, wConv, null, convStateIn, 0).outputVariables();
            bx = convResult[0];
            sd.updateVariableNameAndReference(bx, "conv_path_" + layerIdx);
            // Name the state output so GenerationPipeline can discover and feed it back
            SDVariable convStateOut = convResult[1];
            sd.updateVariableNameAndReference(convStateOut, "conv_state_out_" + layerIdx);
        }

        // Output gating: y = C * conv_out
        SDVariable y = cGate.mul("conv_output_gate_" + layerIdx, bx);

        // Output projection
        INDArray outProjWeight = weights.get(prefix + ".shortconv.out_proj.weight");
        if (outProjWeight != null) {
            SDVariable wOutProj = sd.var(convPrefix + "out_proj.weight", outProjWeight);
            y = fp32Mmul(sd, "conv_out_proj_" + layerIdx, y, wOutProj.permute(1, 0), dtype);
        }

        return y;
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

        SDVariable gate = fp32Mmul(sd, "gate_" + layerIdx, input, wGate.permute(1, 0), dtype);
        SDVariable up = fp32Mmul(sd, "up_" + layerIdx, input, wUp.permute(1, 0), dtype);

        SDVariable silu = sd.nn.swish(gate);
        SDVariable gated = silu.mul("swiglu_" + layerIdx, up);

        return fp32Mmul(sd, "down_" + layerIdx, gated, wDown.permute(1, 0), dtype);
    }

    // ========================================================================
    // FP32 matmul helper (copied from LLaMA)
    // ========================================================================

    private SDVariable fp32Mmul(SameDiff sd, String name, SDVariable a, SDVariable b, DataType dtype) {
        if (dtype == DataType.HALF || dtype == DataType.BFLOAT16) {
            SDVariable aF32 = a.castTo(name + "_a_f32", DataType.FLOAT);
            SDVariable bF32 = b.castTo(name + "_b_f32", DataType.FLOAT);
            SDVariable result = sd.mmul(name + "_f32", aF32, bF32);
            return result.castTo(name, dtype);
        }
        return sd.mmul(name, a, b);
    }

    // ========================================================================
    // RMS Normalization (FP16-safe, copied from LLaMA)
    // ========================================================================

    private SDVariable buildRMSNorm(SameDiff sd, SDVariable input, String outputName,
                                     String weightKey, Map<String, INDArray> weights,
                                     ArchitectureConfig config, DataType dtype) {
        INDArray normWeight = weights.get(weightKey + ".weight");
        if (normWeight == null) {
            log.warn("Missing RMS norm weight: {}", weightKey);
            return input;
        }

        SDVariable gamma = sd.var(outputName + ".weight", normWeight);

        // Upcast to FLOAT32 for squaring to prevent HALF overflow
        SDVariable computeInput;
        boolean needsCast = (input.dataType() == DataType.HALF || input.dataType() == DataType.BFLOAT16);
        if (needsCast) {
            computeInput = input.castTo(outputName + "_f32", DataType.FLOAT);
        } else {
            computeInput = input;
        }
        SDVariable squared = computeInput.mul(computeInput);
        SDVariable meanSquared = squared.mean(true, -1);
        SDVariable rms = sd.math.sqrt(meanSquared.add(config.getLayerNormEpsilon()));
        SDVariable normalized = computeInput.div(rms);
        SDVariable normalizedOrig;
        if (needsCast) {
            normalizedOrig = normalized.castTo(outputName + "_cast", input.dataType());
        } else {
            normalizedOrig = normalized;
        }

        return normalizedOrig.mul(outputName, gamma);
    }

    /**
     * Per-head RMS normalization for QK norms.
     * Input shape: [batch, seq, numHeads, headDim]
     */
    private SDVariable applyHeadNorm(SameDiff sd, SDVariable input, String outputName,
                                      INDArray normWeight, float eps) {
        SDVariable gamma = sd.var(outputName + ".weight", normWeight);
        SDVariable computeInput;
        boolean needsCast = (input.dataType() == DataType.HALF || input.dataType() == DataType.BFLOAT16);
        if (needsCast) {
            computeInput = input.castTo(outputName + "_f32", DataType.FLOAT);
        } else {
            computeInput = input;
        }
        SDVariable squared = computeInput.mul(computeInput);
        SDVariable meanSquared = squared.mean(true, -1);
        SDVariable rms = sd.math.sqrt(meanSquared.add(eps));
        SDVariable normalized = computeInput.div(rms);
        SDVariable normalizedOrig;
        if (needsCast) {
            normalizedOrig = normalized.castTo(outputName + "_cast", input.dataType());
        } else {
            normalizedOrig = normalized;
        }
        return normalizedOrig.mul(outputName, gamma);
    }

    // ========================================================================
    // Tensor name patterns
    // ========================================================================

    @Override
    public Map<String, String> getTensorNamePatterns() {
        Map<String, String> patterns = new HashMap<>();

        patterns.put("token_embd.weight", "model.embed_tokens.weight");
        patterns.put("token_embd_norm.weight", "model.norm.weight");
        patterns.put("output.weight", "lm_head.weight");

        // Attention layers
        patterns.put("blk.{layer}.attn_q.weight", "model.layers.{layer}.self_attn.q_proj.weight");
        patterns.put("blk.{layer}.attn_k.weight", "model.layers.{layer}.self_attn.k_proj.weight");
        patterns.put("blk.{layer}.attn_v.weight", "model.layers.{layer}.self_attn.v_proj.weight");
        patterns.put("blk.{layer}.attn_output.weight", "model.layers.{layer}.self_attn.o_proj.weight");

        // Per-head QK norms
        patterns.put("blk.{layer}.attn_q_norm.weight", "model.layers.{layer}.self_attn.q_norm.weight");
        patterns.put("blk.{layer}.attn_k_norm.weight", "model.layers.{layer}.self_attn.k_norm.weight");

        // Short-conv block
        patterns.put("blk.{layer}.shortconv.in_proj.weight", "model.layers.{layer}.short_conv.in_proj.weight");
        patterns.put("blk.{layer}.shortconv.out_proj.weight", "model.layers.{layer}.short_conv.out_proj.weight");
        patterns.put("blk.{layer}.shortconv.conv.weight", "model.layers.{layer}.short_conv.conv.weight");

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
                .layerTypes(metadata.getLayerTypes())
                .kvHeadsPerLayer(metadata.getKvHeadsPerLayer())
                .ropeType(metadata.getRopeType())
                .useRmsNorm(true)
                .useSwiGLU(true)
                .useRotaryEmbeddings(true)
                .decoderOnly(true)
                .build();
    }
}
