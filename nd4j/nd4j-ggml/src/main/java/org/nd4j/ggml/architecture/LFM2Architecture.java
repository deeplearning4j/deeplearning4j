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
import org.nd4j.linalg.factory.Nd4j;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Architecture handler for LFM-2 (Liquid Foundation Model 2) and LFM-2 MoE models.
 *
 * <p>LFM-2 is a hybrid architecture that interleaves gated short-convolution blocks
 * with GQA attention blocks:</p>
 * <ul>
 *   <li><b>Short-conv blocks</b>: Double-gated convolution composed from CausalConv1d
 *       with a sigmoid gate and element-wise multiply. Two parallel paths are convolved
 *       and gated together.</li>
 *   <li><b>Attention blocks</b>: Standard GQA with RoPE</li>
 *   <li><b>FFN</b>: SwiGLU throughout</li>
 *   <li><b>Norm</b>: RMSNorm throughout</li>
 * </ul>
 *
 * <p>Layer types are determined from GGUF metadata ({@code layer_types} array)
 * or by probing tensor keys (conv_* keys indicate short-conv, attn_q indicates attention).</p>
 *
 * @author Eclipse Deeplearning4j Contributors
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
        int headDim = config.getHeadDimension();

        log.info("Building LFM-2 graph: {} layers, hidden={}, heads={}, kv_heads={}, headDim={}, dtype={}",
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
            String layerType = detectLayerType(config, layer, weights);

            // Pre-block RMS normalization
            SDVariable normed = buildRMSNorm(sd, hidden,
                    "model.layers." + layer + ".input_layernorm",
                    "blk." + layer + ".attn_norm", weights, config);

            SDVariable blockOut;
            if ("short_conv".equals(layerType)) {
                blockOut = buildGatedShortConvBlock(sd, normed, layer, config, weights, dtype);
            } else {
                blockOut = buildGQAAttention(sd, normed, layer, config, weights, dtype);
            }

            // Post-block residual
            SDVariable postBlock = hidden.add("post_block_" + layer, blockOut);

            // Pre-FFN normalization
            SDVariable ffnNormed = buildRMSNorm(sd, postBlock,
                    "model.layers." + layer + ".post_attention_layernorm",
                    "blk." + layer + ".ffn_norm", weights, config);

            // SwiGLU FFN
            SDVariable ffnOut = buildSwiGLUFFN(sd, ffnNormed, layer, weights, dtype);

            // Post-FFN residual
            hidden = postBlock.add("layer_out_" + layer, ffnOut);
        }

        // Final RMS normalization
        hidden = buildRMSNorm(sd, hidden, "model.norm", "output_norm", weights, config);

        // Output projection (LM head)
        INDArray outputWeight = weights.get("output.weight");
        if (outputWeight == null) {
            outputWeight = tokenEmbedWeight;
        }
        SDVariable lmHead = sd.var("lm_head.weight", outputWeight);
        sd.mmul("logits", hidden, lmHead.permute(1, 0));

        return sd;
    }

    // ========================================================================
    // Layer type detection
    // ========================================================================

    private String detectLayerType(ArchitectureConfig config, int layerIdx, Map<String, INDArray> weights) {
        // 1. Check explicit layer_types from metadata
        List<String> layerTypes = config.getLayerTypes();
        if (layerTypes != null && layerIdx < layerTypes.size()) {
            String type = layerTypes.get(layerIdx).toLowerCase();
            if (type.contains("conv") || type.contains("short_conv") || type.contains("gated_conv")) {
                return "short_conv";
            }
            if (type.contains("attention") || type.contains("attn")) {
                return "attention";
            }
        }

        // 2. Probe tensor keys
        String prefix = "blk." + layerIdx;
        if (weights.containsKey(prefix + ".conv_gate.weight") ||
                weights.containsKey(prefix + ".conv_in.weight")) {
            return "short_conv";
        }
        if (weights.containsKey(prefix + ".attn_q.weight")) {
            return "attention";
        }

        return "attention";
    }

    // ========================================================================
    // Gated short-convolution block
    // ========================================================================

    /**
     * Double-gated convolution block.
     *
     * <p>Two parallel projection paths are each processed through CausalConv1d.
     * One path goes through sigmoid to produce a gate, then the paths are
     * multiplied together. This implements the gated short-convolution pattern
     * used in LFM-2.</p>
     *
     * <p>Structure: out = conv1d(x @ W_in) * sigmoid(conv1d(x @ W_gate))</p>
     */
    private SDVariable buildGatedShortConvBlock(SameDiff sd, SDVariable input, int layerIdx,
                                                 ArchitectureConfig config,
                                                 Map<String, INDArray> weights, DataType dtype) {
        String prefix = "blk." + layerIdx;
        String convPrefix = "model.layers." + layerIdx + ".short_conv.";

        // Input projection: [B, L, hidden] -> [B, L, convDim]
        INDArray convInWeight = weights.get(prefix + ".conv_in.weight");
        if (convInWeight == null) {
            log.warn("Missing conv_in weight for layer {}", layerIdx);
            return input;
        }
        SDVariable wIn = sd.var(convPrefix + "in_proj.weight", convInWeight);
        SDVariable inProjected = sd.mmul("conv_in_proj_" + layerIdx, input, wIn.permute(1, 0));

        // Gate projection: [B, L, hidden] -> [B, L, convDim]
        INDArray convGateWeight = weights.get(prefix + ".conv_gate.weight");
        if (convGateWeight == null) {
            log.warn("Missing conv_gate weight for layer {}", layerIdx);
            return input;
        }
        SDVariable wGate = sd.var(convPrefix + "gate_proj.weight", convGateWeight);
        SDVariable gateProjected = sd.mmul("conv_gate_proj_" + layerIdx, input, wGate.permute(1, 0));

        // CausalConv1d on the input path
        INDArray conv1dWeight = weights.get(prefix + ".conv1d.weight");
        INDArray conv1dBias = weights.get(prefix + ".conv1d.bias");
        if (conv1dWeight != null) {
            SDVariable wConv = sd.var(convPrefix + "conv.weight", conv1dWeight);
            SDVariable bConv = conv1dBias != null ? sd.var(convPrefix + "conv.bias", conv1dBias) : null;
            SDVariable[] convResult = new CausalConv1d(sd, inProjected, wConv, bConv, null, 1).outputVariables();
            inProjected = convResult[0];
            sd.updateVariableNameAndReference(inProjected, "conv_path_" + layerIdx);
        }

        // CausalConv1d on the gate path
        INDArray gateConv1dWeight = weights.get(prefix + ".conv1d_gate.weight");
        INDArray gateConv1dBias = weights.get(prefix + ".conv1d_gate.bias");
        if (gateConv1dWeight != null) {
            SDVariable wGateConv = sd.var(convPrefix + "gate_conv.weight", gateConv1dWeight);
            SDVariable bGateConv = gateConv1dBias != null ? sd.var(convPrefix + "gate_conv.bias", gateConv1dBias) : null;
            SDVariable[] gateConvResult = new CausalConv1d(sd, gateProjected, wGateConv, bGateConv, null, 1).outputVariables();
            gateProjected = gateConvResult[0];
            sd.updateVariableNameAndReference(gateProjected, "conv_gate_path_" + layerIdx);
        }

        // Sigmoid gating
        SDVariable gateActivation = sd.nn.sigmoid("conv_sigmoid_" + layerIdx, gateProjected);
        SDVariable gated = inProjected.mul("conv_gated_" + layerIdx, gateActivation);

        // Output projection
        INDArray convOutWeight = weights.get(prefix + ".conv_out.weight");
        if (convOutWeight != null) {
            SDVariable wOut = sd.var(convPrefix + "out_proj.weight", convOutWeight);
            gated = sd.mmul("conv_out_proj_" + layerIdx, gated, wOut.permute(1, 0));
        }

        return gated;
    }

    // ========================================================================
    // GQA Attention
    // ========================================================================

    private SDVariable buildGQAAttention(SameDiff sd, SDVariable input, int layerIdx,
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

        SDVariable q = sd.mmul("q_" + layerIdx, input, wq.permute(1, 0));
        SDVariable k = sd.mmul("k_" + layerIdx, input, wk.permute(1, 0));
        SDVariable v = sd.mmul("v_" + layerIdx, input, wv.permute(1, 0));

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

        SDVariable gate = sd.mmul("gate_" + layerIdx, input, wGate.permute(1, 0));
        SDVariable up = sd.mmul("up_" + layerIdx, input, wUp.permute(1, 0));

        SDVariable silu = sd.nn.swish(gate);
        SDVariable gated = silu.mul("swiglu_" + layerIdx, up);

        return sd.mmul("down_" + layerIdx, gated, wDown.permute(1, 0));
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

        // Short-conv block
        patterns.put("blk.{layer}.conv_in.weight", "model.layers.{layer}.short_conv.in_proj.weight");
        patterns.put("blk.{layer}.conv_gate.weight", "model.layers.{layer}.short_conv.gate_proj.weight");
        patterns.put("blk.{layer}.conv1d.weight", "model.layers.{layer}.short_conv.conv.weight");
        patterns.put("blk.{layer}.conv1d.bias", "model.layers.{layer}.short_conv.conv.bias");
        patterns.put("blk.{layer}.conv1d_gate.weight", "model.layers.{layer}.short_conv.gate_conv.weight");
        patterns.put("blk.{layer}.conv1d_gate.bias", "model.layers.{layer}.short_conv.gate_conv.bias");
        patterns.put("blk.{layer}.conv_out.weight", "model.layers.{layer}.short_conv.out_proj.weight");

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
                .ropeType(metadata.getRopeType())
                .useRmsNorm(true)
                .useSwiGLU(true)
                .useRotaryEmbeddings(true)
                .decoderOnly(true)
                .build();
    }
}
