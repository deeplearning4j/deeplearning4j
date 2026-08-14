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
 * Architecture handler for the Phi model family (Phi-2, Phi-3, Phi-3.5, Phi-4).
 *
 * <ul>
 *   <li><b>Phi-2</b>: LayerNorm, combined QKV projection, partial RoPE, GELU FFN,
 *       parallel residual (attention and FFN run in parallel on the same normed input).</li>
 *   <li><b>Phi-3/3.5/4</b>: RMSNorm, separate Q/K/V projections, SuRoPE (scaled unified RoPE),
 *       SwiGLU FFN, sequential residual (standard LLaMA-style pre-norm blocks), GQA.</li>
 * </ul>
 *
 * <p>The version is auto-detected from weight key patterns: Phi-2 models have combined
 * {@code blk.{layer}.attn_qkv.weight} tensors, while Phi-3+ models have separate
 * {@code blk.{layer}.attn_q.weight} tensors.</p>
 *
 * <p>Phi-3.5-MoE is detected via the presence of {@code ffn_gate_inp.weight} and uses
 * softmax routing with per-expert SwiGLU, following the same MoE pattern as LLaMA/Mixtral.</p>
 *
 * @author Eclipse Deeplearning4j Contributors
 */
@Slf4j
public class PhiArchitecture implements ModelArchitecture {

    private static final Set<String> SUPPORTED_VARIANTS = Set.of(
            "phi", "phi2", "phi3", "phi3.5", "phi4"
    );

    @Override
    public String getName() {
        return "phi";
    }

    @Override
    public Set<String> getSupportedVariants() {
        return SUPPORTED_VARIANTS;
    }

    @Override
    public boolean canHandle(GGMLMetadata metadata) {
        String arch = metadata.getArchitecture();
        if (arch == null) return false;
        return arch.toLowerCase().contains("phi");
    }

    @Override
    public String getDefaultChatTemplateType() {
        return "chatml"; // Phi-3+ uses ChatML format
    }

    @Override
    public String getModelSystemProperty() {
        return "phi.gguf.path";
    }

    @Override
    public SameDiff buildGraph(GGMLMetadata metadata, Map<String, INDArray> weights, ConversionOptions options) {
        SameDiff sd = SameDiff.create();
        ArchitectureConfig config = getConfig(metadata);
        DataType dtype = options.getTargetDataType();

        boolean isPhi2 = isPhi2(weights);

        int numLayers = config.getNumLayers();
        int hiddenSize = config.getHiddenSize();
        int numHeads = config.getNumAttentionHeads();
        int numKvHeads = config.getNumKVHeads();
        int headDim = config.getHeadDimension();

        log.info("Building Phi{} graph: {} layers, hidden={}, heads={}, kv_heads={}, headDim={}, " +
                        "ropeDimCount={}, ropeFreqBase={}, dtype={}",
                isPhi2 ? "-2" : "-3+", numLayers, hiddenSize, numHeads, numKvHeads, headDim,
                config.getRopeDimensionCount(), config.getRopeFreqBase(), dtype);

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
            if (isPhi2) {
                hidden = buildPhi2Block(sd, hidden, layer, config, weights, dtype);
            } else {
                hidden = buildPhi3Block(sd, hidden, layer, config, weights, dtype);
            }
        }

        // Final normalization
        if (isPhi2) {
            hidden = buildLayerNorm(sd, hidden, "model.norm", "output_norm", weights, config);
        } else {
            hidden = buildRMSNorm(sd, hidden, "model.norm", "output_norm", weights, config);
        }

        // Output projection (LM head)
        INDArray outputWeight = weights.get("output.weight");
        if (outputWeight == null) {
            // Some models tie embedding weights
            outputWeight = tokenEmbedWeight;
        }
        SDVariable lmHead = sd.var("lm_head.weight", outputWeight);
        QuantizedLinear.matMul(sd, "logits", hidden, lmHead, weights, "output.weight", dtype);

        return sd;
    }

    // ========================================================================
    // Phi-2 block: Parallel residual with LayerNorm, combined QKV, GELU FFN
    // ========================================================================

    /**
     * Phi-2 transformer block with parallel residual connections.
     * Both attention and FFN operate on the same layer-normed input,
     * and their outputs are added together with the residual:
     * {@code output = input + attention(norm(input)) + ffn(norm(input))}
     */
    private SDVariable buildPhi2Block(SameDiff sd, SDVariable input, int layerIdx,
                                       ArchitectureConfig config, Map<String, INDArray> weights,
                                       DataType dtype) {
        String prefix = "blk." + layerIdx;

        // LayerNorm (shared for both attention and FFN in parallel residual)
        SDVariable normed = buildLayerNorm(sd, input,
                "model.layers." + layerIdx + ".input_layernorm",
                prefix + ".attn_norm", weights, config);

        // Attention with combined QKV
        SDVariable attnOut = buildPhi2Attention(sd, normed, layerIdx, config, weights, dtype);

        // GELU FFN
        SDVariable ffnOut = buildGELUFFN(sd, normed, layerIdx, weights, dtype);

        // Parallel residual: output = input + attn + ffn
        SDVariable attnPlusFfn = attnOut.add("attn_plus_ffn_" + layerIdx, ffnOut);
        return input.add("layer_out_" + layerIdx, attnPlusFfn);
    }

    // ========================================================================
    // Phi-3+ block: Sequential residual with RMSNorm, separate QKV, SwiGLU
    // ========================================================================

    /**
     * Phi-3/3.5/4 transformer block with sequential residual (LLaMA-style).
     * Standard pre-norm: attention with residual, then FFN with residual.
     */
    private SDVariable buildPhi3Block(SameDiff sd, SDVariable input, int layerIdx,
                                       ArchitectureConfig config, Map<String, INDArray> weights,
                                       DataType dtype) {
        String prefix = "blk." + layerIdx;

        // Pre-attention RMSNorm
        SDVariable normed = buildRMSNorm(sd, input,
                "model.layers." + layerIdx + ".input_layernorm",
                prefix + ".attn_norm", weights, config);

        // Separate Q/K/V attention with GQA
        SDVariable attnOut = buildSeparateQKVAttention(sd, normed, layerIdx, config, weights, dtype);

        // Residual
        SDVariable postAttn = input.add("post_attn_" + layerIdx, attnOut);

        // Pre-FFN RMSNorm
        SDVariable ffnNormed = buildRMSNorm(sd, postAttn,
                "model.layers." + layerIdx + ".post_attention_layernorm",
                prefix + ".ffn_norm", weights, config);

        // FFN: MoE or SwiGLU
        SDVariable ffnOut;
        if (weights.containsKey(prefix + ".ffn_gate_inp.weight")) {
            INDArray routerGate = weights.get(prefix + ".ffn_gate_inp.weight");
            ffnOut = buildMoEFFN(sd, ffnNormed, layerIdx, weights, dtype, routerGate);
        } else {
            ffnOut = buildSwiGLUFFN(sd, ffnNormed, layerIdx, weights, dtype);
        }

        // Residual
        return postAttn.add("layer_out_" + layerIdx, ffnOut);
    }

    // ========================================================================
    // Phi-2 combined QKV attention with partial RoPE
    // ========================================================================

    /**
     * Phi-2 attention with combined QKV projection and partial RoPE.
     * Partial RoPE applies rotary embeddings only to the first {@code ropeDimensionCount}
     * dimensions of each head, passing through the remaining dimensions unchanged.
     */
    private SDVariable buildPhi2Attention(SameDiff sd, SDVariable input, int layerIdx,
                                           ArchitectureConfig config, Map<String, INDArray> weights,
                                           DataType dtype) {
        String prefix = "blk." + layerIdx;
        int numHeads = config.getNumAttentionHeads();
        int numKvHeads = config.getNumKVHeads();
        int headDim = config.getHeadDimension();

        INDArray qkvWeight = weights.get(prefix + ".attn_qkv.weight");
        INDArray oWeight = weights.get(prefix + ".attn_output.weight");

        if (qkvWeight == null || oWeight == null) {
            log.warn("Missing Phi-2 attention weights for layer {}", layerIdx);
            return input;
        }

        // Derive dimensions from QKV weight shape
        int qkvOutDim = (int) qkvWeight.shape()[0];
        if (headDim <= 0) {
            int totalHeads = numHeads + 2 * numKvHeads;
            headDim = qkvOutDim / totalHeads;
        }

        int qDim = numHeads * headDim;
        int kDim = numKvHeads * headDim;
        int vDim = numKvHeads * headDim;

        String attnPrefix = "model.layers." + layerIdx + ".self_attn.";
        SDVariable wqkv = sd.var(attnPrefix + "qkv_proj.weight", qkvWeight);
        SDVariable wo = sd.var(attnPrefix + "o_proj.weight", oWeight);

        // Combined QKV projection
        SDVariable qkv = QuantizedLinear.matMul(sd, "qkv_" + layerIdx, input, wqkv, weights, prefix + ".attn_qkv.weight", dtype);

        // Add QKV bias if present (Phi-2 has biases)
        INDArray qkvBias = weights.get(prefix + ".attn_qkv.bias");
        if (qkvBias != null) {
            qkv = qkv.add(sd.var(attnPrefix + "qkv_proj.bias", qkvBias));
        }

        // Split into Q, K, V
        SDVariable q = qkv.get(SDIndex.all(), SDIndex.all(), SDIndex.interval(0, qDim));
        sd.updateVariableNameAndReference(q, "q_split_" + layerIdx);
        SDVariable k = qkv.get(SDIndex.all(), SDIndex.all(), SDIndex.interval(qDim, qDim + kDim));
        sd.updateVariableNameAndReference(k, "k_split_" + layerIdx);
        SDVariable v = qkv.get(SDIndex.all(), SDIndex.all(), SDIndex.interval(qDim + kDim, qDim + kDim + vDim));
        sd.updateVariableNameAndReference(v, "v_split_" + layerIdx);

        SDVariable batchDim = sd.sizeAt(input, 0);
        SDVariable seqDim = sd.sizeAt(input, 1);

        // Reshape to [batch, seq, heads, headDim]
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

        // Partial RoPE: apply only to first ropeDimensionCount dims of each head
        int ropeDimCount = config.getRopeDimensionCount();
        if (config.isUseRotaryEmbeddings() && ropeDimCount > 0 && ropeDimCount < headDim) {
            // Split head dimension into rotary and pass-through parts
            SDVariable qRot = q.get(SDIndex.all(), SDIndex.all(), SDIndex.all(),
                    SDIndex.interval(0, ropeDimCount));
            sd.updateVariableNameAndReference(qRot, "q_rot_part_" + layerIdx);
            SDVariable qPass = q.get(SDIndex.all(), SDIndex.all(), SDIndex.all(),
                    SDIndex.interval(ropeDimCount, headDim));
            sd.updateVariableNameAndReference(qPass, "q_pass_part_" + layerIdx);

            SDVariable kRot = k.get(SDIndex.all(), SDIndex.all(), SDIndex.all(),
                    SDIndex.interval(0, ropeDimCount));
            sd.updateVariableNameAndReference(kRot, "k_rot_part_" + layerIdx);
            SDVariable kPass = k.get(SDIndex.all(), SDIndex.all(), SDIndex.all(),
                    SDIndex.interval(ropeDimCount, headDim));
            sd.updateVariableNameAndReference(kPass, "k_pass_part_" + layerIdx);

            // Apply RoPE to rotary part only
            qRot = sd.nn().fusedRoPE("q_rot_rope_" + layerIdx, qRot, null,
                    config.getRopeType(), config.getRopeFreqBase(), 1.0, ropeDimCount);

            kRot = sd.nn().fusedRoPE("k_rot_rope_" + layerIdx, kRot, null,
                    config.getRopeType(), config.getRopeFreqBase(), 1.0, ropeDimCount);

            // Concatenate rotary and pass-through parts along head dimension
            q = sd.concat("q_rope_" + layerIdx, -1, qRot, qPass);
            k = sd.concat("k_rope_" + layerIdx, -1, kRot, kPass);
        } else if (config.isUseRotaryEmbeddings()) {
            // Full RoPE (ropeDimCount == headDim or not specified)
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

        // Reshape to [batch, seq, numHeads * headDim]
        int attnOutDim = numHeads * headDim;
        SDVariable outShapeVar = sd.stack("attn_out_shape_" + layerIdx, 0,
                batchDim, seqDim,
                sd.constant(Nd4j.scalar((long) attnOutDim)));
        SDVariable attnFlat = sd.reshape("attn_flat_" + layerIdx, attnOut, outShapeVar);

        // Output projection
        return QuantizedLinear.matMul(sd, "attn_proj_" + layerIdx, attnFlat, wo, weights, prefix + ".attn_output.weight", dtype);
    }

    // ========================================================================
    // Phi-3+ separate Q/K/V attention with full RoPE and GQA
    // ========================================================================

    private SDVariable buildSeparateQKVAttention(SameDiff sd, SDVariable input, int layerIdx,
                                                  ArchitectureConfig config, Map<String, INDArray> weights,
                                                  DataType dtype) {
        String prefix = "blk." + layerIdx;
        int numHeads = config.getNumAttentionHeads();
        int numKvHeads = config.getNumKVHeads();

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
            headDim = kOutDim / numKvHeads;
        }
        int actualNumHeads = qOutDim / headDim;

        if (layerIdx == 0) {
            log.info("Layer {} Phi-3+ attention: qHeads={}, kvHeads={}, headDim={} (Q out={}, K out={})",
                    layerIdx, actualNumHeads, numKvHeads, headDim, qOutDim, kOutDim);
        }

        String attnPrefix = "model.layers." + layerIdx + ".self_attn.";
        SDVariable wq = sd.var(attnPrefix + "q_proj.weight", qWeight);
        SDVariable wk = sd.var(attnPrefix + "k_proj.weight", kWeight);
        SDVariable wv = sd.var(attnPrefix + "v_proj.weight", vWeight);
        SDVariable wo = sd.var(attnPrefix + "o_proj.weight", oWeight);

        // Project to Q, K, V
        SDVariable q = QuantizedLinear.matMul(sd, "q_" + layerIdx, input, wq, weights, prefix + ".attn_q.weight", dtype);
        SDVariable k = QuantizedLinear.matMul(sd, "k_" + layerIdx, input, wk, weights, prefix + ".attn_k.weight", dtype);
        SDVariable v = QuantizedLinear.matMul(sd, "v_" + layerIdx, input, wv, weights, prefix + ".attn_v.weight", dtype);

        SDVariable batchDim = sd.sizeAt(input, 0);
        SDVariable seqDim = sd.sizeAt(input, 1);

        // Reshape to [batch, seq, heads, headDim]
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

        // Apply RoPE (SuRoPE for Phi-3+: freq_base and freq_scale from metadata)
        if (config.isUseRotaryEmbeddings()) {
            q = sd.nn().fusedRoPE("q_rope_" + layerIdx, q, null,
                    config.getRopeType(), config.getRopeFreqBase(), 1.0,
                    config.getRopeDimensionCount());

            k = sd.nn().fusedRoPE("k_rope_" + layerIdx, k, null,
                    config.getRopeType(), config.getRopeFreqBase(), 1.0,
                    config.getRopeDimensionCount());
        }

        // FusedRoPE promotes HALF→FLOAT internally; V must match Q/K dtype
        v = GGMLDTypePolicy.castTo(
                v, "v_cast_phi3_" + layerIdx, q.dataType());

        // Dot-product attention
        SDVariable attnOut = sd.nn.dotProductAttentionV2(
                "attn_out_" + layerIdx,
                q, v, k, null, null,
                0.0, 0.0, true, false
        );

        // Reshape to [batch, seq, numHeads * headDim]
        int attnOutDim = actualNumHeads * headDim;
        SDVariable outShapeVar = sd.stack("attn_out_shape_" + layerIdx, 0,
                batchDim, seqDim,
                sd.constant(Nd4j.scalar((long) attnOutDim)));
        SDVariable attnFlat = sd.reshape("attn_flat_" + layerIdx, attnOut, outShapeVar);

        // Output projection
        return QuantizedLinear.matMul(sd, "attn_proj_" + layerIdx, attnFlat, wo, weights, prefix + ".attn_output.weight", dtype);
    }

    // ========================================================================
    // FFN variants
    // ========================================================================

    /**
     * GELU FFN for Phi-2: up_proj -> GELU -> down_proj (no gate).
     */
    private SDVariable buildGELUFFN(SameDiff sd, SDVariable input, int layerIdx,
                                     Map<String, INDArray> weights, DataType dtype) {
        String prefix = "blk." + layerIdx;

        INDArray upWeight = weights.get(prefix + ".ffn_up.weight");
        INDArray downWeight = weights.get(prefix + ".ffn_down.weight");

        if (upWeight == null || downWeight == null) {
            log.warn("Missing GELU FFN weights for layer {}", layerIdx);
            return input;
        }

        String mlpPrefix = "model.layers." + layerIdx + ".mlp.";
        SDVariable wUp = sd.var(mlpPrefix + "fc1.weight", upWeight);
        SDVariable wDown = sd.var(mlpPrefix + "fc2.weight", downWeight);

        SDVariable up = QuantizedLinear.matMul(sd, "up_" + layerIdx, input, wUp, weights, prefix + ".ffn_up.weight", dtype);

        // Add bias if present (Phi-2 FFN has biases)
        INDArray upBias = weights.get(prefix + ".ffn_up.bias");
        if (upBias != null) {
            up = up.add(sd.var(mlpPrefix + "fc1.bias", upBias));
        }

        SDVariable activated = sd.nn.gelu("gelu_" + layerIdx, up);

        SDVariable down = QuantizedLinear.matMul(sd, "down_" + layerIdx, activated, wDown, weights, prefix + ".ffn_down.weight", dtype);

        // Add bias if present
        INDArray downBias = weights.get(prefix + ".ffn_down.bias");
        if (downBias != null) {
            down = down.add(sd.var(mlpPrefix + "fc2.bias", downBias));
        }

        return down;
    }

    /**
     * SwiGLU FFN for Phi-3+: gate_proj -> SiLU, up_proj, gate * up -> down_proj.
     */
    private SDVariable buildSwiGLUFFN(SameDiff sd, SDVariable input, int layerIdx,
                                       Map<String, INDArray> weights, DataType dtype) {
        String prefix = "blk." + layerIdx;

        INDArray gateWeight = weights.get(prefix + ".ffn_gate.weight");
        INDArray upWeight = weights.get(prefix + ".ffn_up.weight");
        INDArray downWeight = weights.get(prefix + ".ffn_down.weight");

        if (gateWeight == null || upWeight == null || downWeight == null) {
            log.warn("Missing SwiGLU FFN weights for layer {}", layerIdx);
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
     * Mixture-of-Experts FFN for Phi-3.5-MoE.
     */
    private SDVariable buildMoEFFN(SameDiff sd, SDVariable input, int layerIdx,
                                    Map<String, INDArray> weights, DataType dtype,
                                    INDArray routerGateWeight) {
        String prefix = "blk." + layerIdx;
        String mlpPrefix = "model.layers." + layerIdx + ".mlp.";

        SDVariable gate = sd.var(mlpPrefix + "gate.weight", routerGateWeight);
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
            SDVariable expertOut = buildExpertSwiGLU(sd, input, layerIdx, e, weights, dtype);
            SDVariable expertWeight = routerWeights.get(SDIndex.all(), SDIndex.all(), SDIndex.point(e));
            SDVariable weighted = expertOut.mul("weighted_e" + e + "_" + layerIdx, expertWeight);
            combined = (combined == null) ? weighted : combined.add("combine_e" + e + "_" + layerIdx, weighted);
        }

        return combined != null ? combined : input;
    }

    private SDVariable buildExpertSwiGLU(SameDiff sd, SDVariable input, int layerIdx,
                                          int expertIdx, Map<String, INDArray> weights, DataType dtype) {
        String prefix = "blk." + layerIdx;
        String nameSuffix = "_" + layerIdx + "_e" + expertIdx;

        INDArray gateW = weights.get(prefix + ".ffn_gate." + expertIdx + ".weight");
        INDArray upW = weights.get(prefix + ".ffn_up." + expertIdx + ".weight");
        INDArray downW = weights.get(prefix + ".ffn_down." + expertIdx + ".weight");

        if (gateW == null || upW == null || downW == null) {
            log.warn("Missing expert FFN weights for layer {} expert {}", layerIdx, expertIdx);
            return input;
        }

        String mlpPrefix = "model.layers." + layerIdx + ".mlp.expert_" + expertIdx + ".";
        SDVariable wGate = sd.var(mlpPrefix + "gate_proj.weight", gateW);
        SDVariable wUp = sd.var(mlpPrefix + "up_proj.weight", upW);
        SDVariable wDown = sd.var(mlpPrefix + "down_proj.weight", downW);

        SDVariable g = QuantizedLinear.matMul(sd, "gate" + nameSuffix, input, wGate, weights, prefix + ".ffn_gate." + expertIdx + ".weight", dtype);
        SDVariable u = QuantizedLinear.matMul(sd, "up" + nameSuffix, input, wUp, weights, prefix + ".ffn_up." + expertIdx + ".weight", dtype);

        SDVariable silu = sd.nn.swish(g);
        SDVariable h = silu.mul("swiglu" + nameSuffix, u);

        return QuantizedLinear.matMul(sd, "down" + nameSuffix, h, wDown, weights, prefix + ".ffn_down." + expertIdx + ".weight", dtype);
    }

    // ========================================================================
    // Normalization helpers
    // ========================================================================

    /**
     * RMS normalization: x * rsqrt(mean(x^2) + eps) * gamma
     */
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
     * Detect Phi-2 by checking for combined QKV weights in layer 0.
     * Phi-2 uses {@code blk.0.attn_qkv.weight}, while Phi-3+ uses separate
     * {@code blk.0.attn_q.weight}.
     */
    private boolean isPhi2(Map<String, INDArray> weights) {
        return weights.containsKey("blk.0.attn_qkv.weight");
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
        patterns.put("output_norm.bias", "model.norm.bias");

        // Phi-2: Combined QKV attention
        patterns.put("blk.{layer}.attn_qkv.weight", "model.layers.{layer}.self_attn.qkv_proj.weight");
        patterns.put("blk.{layer}.attn_qkv.bias", "model.layers.{layer}.self_attn.qkv_proj.bias");

        // Phi-3+: Separate Q/K/V attention
        patterns.put("blk.{layer}.attn_q.weight", "model.layers.{layer}.self_attn.q_proj.weight");
        patterns.put("blk.{layer}.attn_k.weight", "model.layers.{layer}.self_attn.k_proj.weight");
        patterns.put("blk.{layer}.attn_v.weight", "model.layers.{layer}.self_attn.v_proj.weight");

        // Output projection (shared)
        patterns.put("blk.{layer}.attn_output.weight", "model.layers.{layer}.self_attn.o_proj.weight");

        // Normalization (shared, with optional biases for Phi-2)
        patterns.put("blk.{layer}.attn_norm.weight", "model.layers.{layer}.input_layernorm.weight");
        patterns.put("blk.{layer}.attn_norm.bias", "model.layers.{layer}.input_layernorm.bias");
        patterns.put("blk.{layer}.ffn_norm.weight", "model.layers.{layer}.post_attention_layernorm.weight");
        patterns.put("blk.{layer}.ffn_norm.bias", "model.layers.{layer}.post_attention_layernorm.bias");

        // Phi-2: GELU FFN (fc1/fc2 naming)
        patterns.put("blk.{layer}.ffn_up.weight", "model.layers.{layer}.mlp.fc1.weight");
        patterns.put("blk.{layer}.ffn_up.bias", "model.layers.{layer}.mlp.fc1.bias");
        patterns.put("blk.{layer}.ffn_down.weight", "model.layers.{layer}.mlp.fc2.weight");
        patterns.put("blk.{layer}.ffn_down.bias", "model.layers.{layer}.mlp.fc2.bias");

        // Phi-3+: SwiGLU FFN
        patterns.put("blk.{layer}.ffn_gate.weight", "model.layers.{layer}.mlp.gate_proj.weight");

        // MoE router (Phi-3.5-MoE)
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
                .ropeType(metadata.getRopeType())
                .useRotaryEmbeddings(true)
                .decoderOnly(true)
                .build();
    }
}
