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
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Architecture handler for Nemotron and Nemotron-H models (NVIDIA).
 *
 * <p>Nemotron-H is a hybrid architecture that interleaves Mamba-2 SSM layers
 * with sparse attention layers. Layer types are determined from GGUF metadata
 * (the {@code layer_types} array) or detected by probing which tensor keys
 * exist for each layer (ssm_* keys indicate Mamba-2, attn_q indicates attention).</p>
 *
 * <ul>
 *   <li><b>Mamba-2 layers</b>: CausalConv1d followed by Mamba2SSM recurrence</li>
 *   <li><b>Attention layers</b>: Standard GQA with RoPE</li>
 *   <li><b>FFN</b>: SwiGLU with SquaredReLU for Mamba layers, standard SwiGLU
 *       for attention layers</li>
 *   <li><b>MoE</b>: For nemotron_h_moe variant, detected via ffn_gate_inp weights</li>
 *   <li><b>Norm</b>: RMSNorm throughout</li>
 * </ul>
 *
 * @author Eclipse Deeplearning4j Contributors
 */
@Slf4j
public class NemotronArchitecture implements ModelArchitecture {

    private static final Set<String> SUPPORTED_VARIANTS = Set.of(
            "nemotron", "nemotron_h", "nemotron_h_moe"
    );

    @Override
    public String getName() {
        return "nemotron";
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
        return "nemotron.gguf.path";
    }

    @Override
    public boolean canHandle(GGMLMetadata metadata) {
        String arch = metadata.getArchitecture();
        if (arch == null) return false;
        return arch.toLowerCase().contains("nemotron");
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

        log.info("Building Nemotron graph: {} layers, hidden={}, heads={}, kv_heads={}, headDim={}, dtype={}",
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

            // Pre-norm
            SDVariable normed = buildRMSNorm(sd, hidden,
                    "model.layers." + layer + ".input_layernorm",
                    "blk." + layer + ".attn_norm", weights, config);

            SDVariable blockOut;
            if ("mamba2".equals(layerType)) {
                blockOut = buildMamba2Block(sd, normed, layer, config, weights, dtype);
            } else {
                blockOut = buildGQAAttention(sd, normed, layer, config, weights, dtype);
            }

            // Post-attention residual
            SDVariable postAttn = hidden.add("post_attn_" + layer, blockOut);

            // Pre-FFN normalization
            SDVariable ffnNormed = buildRMSNorm(sd, postAttn,
                    "model.layers." + layer + ".post_attention_layernorm",
                    "blk." + layer + ".ffn_norm", weights, config);

            // FFN: check MoE first, then SwiGLU variants
            SDVariable ffnOut;
            String prefix = "blk." + layer;
            if (weights.containsKey(prefix + ".ffn_gate_inp.weight")) {
                ffnOut = buildMoEFFN(sd, ffnNormed, layer, weights, dtype, "mamba2".equals(layerType));
            } else if ("mamba2".equals(layerType)) {
                ffnOut = buildSquaredReLUSwiGLUFFN(sd, ffnNormed, layer, weights, dtype);
            } else {
                ffnOut = buildSwiGLUFFN(sd, ffnNormed, layer, weights, dtype);
            }

            // Post-FFN residual
            hidden = postAttn.add("layer_out_" + layer, ffnOut);
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
    // Layer type detection
    // ========================================================================

    /**
     * Determine the layer type for a given layer index.
     * Checks explicit metadata first, then probes tensor keys.
     * SSM layers have ssm_in/ssm_a/ssm_dt keys; attention layers have attn_q.
     */
    private String detectLayerType(ArchitectureConfig config, int layerIdx, Map<String, INDArray> weights) {
        // 1. Check explicit layer_types from metadata
        List<String> layerTypes = config.getLayerTypes();
        if (layerTypes != null && layerIdx < layerTypes.size()) {
            String type = layerTypes.get(layerIdx).toLowerCase();
            if (type.contains("mamba") || type.contains("ssm")) {
                return "mamba2";
            }
            if (type.contains("attention") || type.contains("attn")) {
                return "attention";
            }
        }

        // 2. Probe tensor keys
        String prefix = "blk." + layerIdx;
        if (weights.containsKey(prefix + ".ssm_in.weight") ||
                weights.containsKey(prefix + ".ssm_a") ||
                weights.containsKey(prefix + ".ssm_dt.weight")) {
            return "mamba2";
        }
        if (weights.containsKey(prefix + ".attn_q.weight")) {
            return "attention";
        }

        // Default to attention
        return "attention";
    }

    // ========================================================================
    // Mamba-2 SSM block
    // ========================================================================

    private SDVariable buildMamba2Block(SameDiff sd, SDVariable input, int layerIdx,
                                        ArchitectureConfig config, Map<String, INDArray> weights,
                                        DataType dtype) {
        String prefix = "blk." + layerIdx;
        String ssmPrefix = "model.layers." + layerIdx + ".mamba2.";

        // SSM input projection: [B, L, hidden] -> [B, L, ssmDim]
        INDArray ssmInWeight = weights.get(prefix + ".ssm_in.weight");
        if (ssmInWeight == null) {
            log.warn("Missing ssm_in weight for Mamba-2 layer {}", layerIdx);
            return input;
        }
        SDVariable wIn = sd.var(ssmPrefix + "in_proj.weight", ssmInWeight);
        SDVariable projected = sd.mmul("ssm_in_proj_" + layerIdx, input, wIn.permute(1, 0));

        // CausalConv1d
        INDArray convWeight = weights.get(prefix + ".ssm_conv1d.weight");
        INDArray convBias = weights.get(prefix + ".ssm_conv1d.bias");
        if (convWeight != null) {
            SDVariable wConv = sd.var(ssmPrefix + "conv.weight", convWeight);
            SDVariable bConv = convBias != null ? sd.var(ssmPrefix + "conv.bias", convBias) : null;
            SDVariable[] convResult = sd.nn().causalConv1d(
                    projected, wConv, bConv, null, null, 1, 0);
            projected = convResult[0];
            sd.updateVariableNameAndReference(projected, "ssm_conv_" + layerIdx);
        }

        // SSM parameters
        INDArray ssmA = weights.get(prefix + ".ssm_a");
        INDArray ssmDtWeight = weights.get(prefix + ".ssm_dt.weight");
        INDArray ssmDtBias = weights.get(prefix + ".ssm_dt.bias");
        INDArray ssmD = weights.get(prefix + ".ssm_d");

        if (ssmA == null || ssmDtWeight == null) {
            log.warn("Missing SSM parameters for layer {}, falling back to passthrough", layerIdx);
            return projected;
        }

        SDVariable aVar = sd.var(ssmPrefix + "a", ssmA);
        SDVariable dtW = sd.var(ssmPrefix + "dt.weight", ssmDtWeight);

        // dt projection
        SDVariable dt = sd.mmul("ssm_dt_proj_" + layerIdx, input, dtW.permute(1, 0));
        if (ssmDtBias != null) {
            dt = dt.add(sd.var(ssmPrefix + "dt.bias", ssmDtBias));
        }
        dt = sd.nn.softplus("ssm_dt_softplus_" + layerIdx, dt);

        // B and C projections for SSM state
        INDArray ssmBWeight = weights.get(prefix + ".ssm_b.weight");
        INDArray ssmCWeight = weights.get(prefix + ".ssm_c.weight");
        int ssmStateSize = 16;
        int ssmNumHeads = config.getNumAttentionHeads() > 0 ? config.getNumAttentionHeads() : 1;
        int ssmHeadDim = config.getHeadDim() > 0 ? config.getHeadDim() : 64;

        SDVariable bVar, cVar;
        if (ssmBWeight != null && ssmCWeight != null) {
            SDVariable wB = sd.var(ssmPrefix + "b.weight", ssmBWeight);
            SDVariable wC = sd.var(ssmPrefix + "c.weight", ssmCWeight);
            bVar = sd.mmul("ssm_b_proj_" + layerIdx, input, wB.permute(1, 0));
            cVar = sd.mmul("ssm_c_proj_" + layerIdx, input, wC.permute(1, 0));
        } else {
            // Default: identity-like B and C (ones)
            bVar = sd.onesLike("ssm_b_default_" + layerIdx, projected);
            cVar = sd.onesLike("ssm_c_default_" + layerIdx, projected);
        }

        // Mamba2SSM op: (x, A, B, C, dt) -> output
        SDVariable ssmOut = sd.nn().mamba2Ssm(
                projected, aVar, bVar, cVar, dt,
                ssmNumHeads, ssmHeadDim, ssmStateSize)[0];
        sd.updateVariableNameAndReference(ssmOut, "ssm_out_" + layerIdx);

        // Apply D skip connection if present
        if (ssmD != null) {
            SDVariable dVar = sd.var(ssmPrefix + "d", ssmD);
            ssmOut = ssmOut.add("ssm_d_skip_" + layerIdx, projected.mul(dVar));
        }

        // SSM output projection
        INDArray ssmOutWeight = weights.get(prefix + ".ssm_out.weight");
        if (ssmOutWeight != null) {
            SDVariable wOut = sd.var(ssmPrefix + "out_proj.weight", ssmOutWeight);
            ssmOut = sd.mmul("ssm_out_proj_" + layerIdx, ssmOut, wOut.permute(1, 0));
        }

        return ssmOut;
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

        // Project Q, K, V
        SDVariable q = QuantizedLinear.matMul(sd, "q_" + layerIdx, input, wq, weights, prefix + ".attn_q.weight", dtype);
        SDVariable k = QuantizedLinear.matMul(sd, "k_" + layerIdx, input, wk, weights, prefix + ".attn_k.weight", dtype);
        SDVariable v = QuantizedLinear.matMul(sd, "v_" + layerIdx, input, wv, weights, prefix + ".attn_v.weight", dtype);

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
     * SwiGLU FFN with SquaredReLU activation, used for Mamba-2 layers.
     * Structure: gate = SquaredReLU(x @ Wgate), up = x @ Wup, out = (gate * up) @ Wdown
     */
    private SDVariable buildSquaredReLUSwiGLUFFN(SameDiff sd, SDVariable input, int layerIdx,
                                                  Map<String, INDArray> weights, DataType dtype) {
        String prefix = "blk." + layerIdx;
        INDArray gateWeight = weights.get(prefix + ".ffn_gate.weight");
        INDArray upWeight = weights.get(prefix + ".ffn_up.weight");
        INDArray downWeight = weights.get(prefix + ".ffn_down.weight");

        if (gateWeight == null || upWeight == null || downWeight == null) {
            log.warn("Missing SquaredReLU FFN weights for layer {}", layerIdx);
            return input;
        }

        String mlpPrefix = "model.layers." + layerIdx + ".mlp.";
        SDVariable wGate = sd.var(mlpPrefix + "gate_proj.weight", gateWeight);
        SDVariable wUp = sd.var(mlpPrefix + "up_proj.weight", upWeight);
        SDVariable wDown = sd.var(mlpPrefix + "down_proj.weight", downWeight);

        SDVariable gate = QuantizedLinear.matMul(sd, "gate_" + layerIdx, input, wGate, weights, prefix + ".ffn_gate.weight", dtype);
        SDVariable up = QuantizedLinear.matMul(sd, "up_" + layerIdx, input, wUp, weights, prefix + ".ffn_up.weight", dtype);

        // SquaredReLU: relu(x)^2
        SDVariable relu = sd.nn.relu("relu_gate_" + layerIdx, gate, 0);
        SDVariable squaredRelu = relu.mul("sqrelu_" + layerIdx, relu);

        SDVariable gated = squaredRelu.mul("sqrelu_gated_" + layerIdx, up);
        return QuantizedLinear.matMul(sd, "down_" + layerIdx, gated, wDown, weights, prefix + ".ffn_down.weight", dtype);
    }

    /**
     * Mixture of Experts FFN for nemotron_h_moe variant.
     */
    private SDVariable buildMoEFFN(SameDiff sd, SDVariable input, int layerIdx,
                                    Map<String, INDArray> weights, DataType dtype,
                                    boolean useSqRelu) {
        String prefix = "blk." + layerIdx;
        String mlpPrefix = "model.layers." + layerIdx + ".mlp.";

        INDArray routerWeight = weights.get(prefix + ".ffn_gate_inp.weight");
        SDVariable gate = sd.var(mlpPrefix + "gate.weight", routerWeight);
        SDVariable routerLogits = QuantizedLinear.matMul(sd, "router_logits_" + layerIdx, input, gate, weights, prefix + ".ffn_gate_inp.weight", dtype);
        SDVariable routerWeights = sd.nn.softmax("router_weights_" + layerIdx, routerLogits, -1);

        // Count experts
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
            SDVariable expertOut = buildExpertFFN(sd, input, layerIdx, e, weights, dtype, useSqRelu);
            SDVariable expertWeight = routerWeights.get(SDIndex.all(), SDIndex.all(), SDIndex.point(e));
            SDVariable weighted = expertOut.mul("weighted_e" + e + "_" + layerIdx, expertWeight);
            combined = (combined == null) ? weighted : combined.add("combine_e" + e + "_" + layerIdx, weighted);
        }

        return combined != null ? combined : input;
    }

    private SDVariable buildExpertFFN(SameDiff sd, SDVariable input, int layerIdx,
                                       int expertIdx, Map<String, INDArray> weights,
                                       DataType dtype, boolean useSqRelu) {
        String prefix = "blk." + layerIdx;
        String suffix = "." + expertIdx;
        String nameSuffix = "_" + layerIdx + "_e" + expertIdx;

        INDArray gateW = weights.get(prefix + ".ffn_gate" + suffix + ".weight");
        INDArray upW = weights.get(prefix + ".ffn_up" + suffix + ".weight");
        INDArray downW = weights.get(prefix + ".ffn_down" + suffix + ".weight");

        if (gateW == null || upW == null || downW == null) {
            log.warn("Missing expert {} FFN weights for layer {}", expertIdx, layerIdx);
            return input;
        }

        String mlpPrefix = "model.layers." + layerIdx + ".mlp.expert_" + expertIdx + ".";
        SDVariable wGate = sd.var(mlpPrefix + "gate_proj.weight", gateW);
        SDVariable wUp = sd.var(mlpPrefix + "up_proj.weight", upW);
        SDVariable wDown = sd.var(mlpPrefix + "down_proj.weight", downW);

        SDVariable g = QuantizedLinear.matMul(sd, "gate" + nameSuffix, input, wGate, weights, prefix + ".ffn_gate." + expertIdx + ".weight", dtype);
        SDVariable u = QuantizedLinear.matMul(sd, "up" + nameSuffix, input, wUp, weights, prefix + ".ffn_up." + expertIdx + ".weight", dtype);

        SDVariable activated;
        if (useSqRelu) {
            SDVariable relu = sd.nn.relu("relu" + nameSuffix, g, 0);
            activated = relu.mul("sqrelu" + nameSuffix, relu);
        } else {
            activated = sd.nn.swish(g);
        }

        SDVariable h = activated.mul("gated" + nameSuffix, u);
        return QuantizedLinear.matMul(sd, "down" + nameSuffix, h, wDown, weights, prefix + ".ffn_down." + expertIdx + ".weight", dtype);
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

        // Mamba-2 SSM layers
        patterns.put("blk.{layer}.ssm_in.weight", "model.layers.{layer}.mamba2.in_proj.weight");
        patterns.put("blk.{layer}.ssm_conv1d.weight", "model.layers.{layer}.mamba2.conv.weight");
        patterns.put("blk.{layer}.ssm_conv1d.bias", "model.layers.{layer}.mamba2.conv.bias");
        patterns.put("blk.{layer}.ssm_a", "model.layers.{layer}.mamba2.a");
        patterns.put("blk.{layer}.ssm_dt.weight", "model.layers.{layer}.mamba2.dt.weight");
        patterns.put("blk.{layer}.ssm_dt.bias", "model.layers.{layer}.mamba2.dt.bias");
        patterns.put("blk.{layer}.ssm_d", "model.layers.{layer}.mamba2.d");
        patterns.put("blk.{layer}.ssm_out.weight", "model.layers.{layer}.mamba2.out_proj.weight");

        // Normalization
        patterns.put("blk.{layer}.attn_norm.weight", "model.layers.{layer}.input_layernorm.weight");
        patterns.put("blk.{layer}.ffn_norm.weight", "model.layers.{layer}.post_attention_layernorm.weight");

        // Dense FFN (SwiGLU)
        patterns.put("blk.{layer}.ffn_gate.weight", "model.layers.{layer}.mlp.gate_proj.weight");
        patterns.put("blk.{layer}.ffn_up.weight", "model.layers.{layer}.mlp.up_proj.weight");
        patterns.put("blk.{layer}.ffn_down.weight", "model.layers.{layer}.mlp.down_proj.weight");

        // MoE router
        patterns.put("blk.{layer}.ffn_gate_inp.weight", "model.layers.{layer}.mlp.gate.weight");

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

    private static int getIntMetadata(ArchitectureConfig config, String key, int defaultValue) {
        return defaultValue;
    }
}
