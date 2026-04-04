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
import org.nd4j.linalg.api.ops.impl.transforms.custom.DualRoPE;
import org.nd4j.linalg.api.ops.impl.transforms.custom.PerLayerEmbedding;
import org.nd4j.linalg.api.ops.impl.transforms.custom.SharedKvAttention;
import org.nd4j.linalg.factory.Nd4j;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

/**
 * Architecture handler for Gemma and Gemma-derived models (Gemma 1-4).
 *
 * <p>Gemma 4 introduces several architectural innovations:</p>
 * <ul>
 *   <li><b>Per-Layer Embedding (PLE)</b>: Each layer has its own embedding correction
 *       table that adjusts token representations per-layer.</li>
 *   <li><b>Alternating attention patterns</b>: Sliding-window (local) attention
 *       alternates with global (full-context) attention layers.</li>
 *   <li><b>Shared KV</b>: The last N layers share key/value projections from a
 *       designated source layer, reducing parameter count and memory.</li>
 *   <li><b>Dual RoPE</b>: Two independent sets of rotary embeddings with different
 *       frequency bases for local and global attention patterns.</li>
 *   <li><b>RMSNorm + SwiGLU FFN</b>: Standard LLaMA-family normalization and FFN.</li>
 * </ul>
 *
 * <p>Earlier Gemma versions (1, 2, 3) are also supported and handled as standard
 * transformer architectures with GQA, RMSNorm, and SwiGLU.</p>
 *
 * @author Eclipse Deeplearning4j Contributors
 */
@Slf4j
public class GemmaArchitecture implements ModelArchitecture {

    private static final Set<String> SUPPORTED_VARIANTS = Set.of(
            "gemma", "gemma2", "gemma3", "gemma4"
    );

    /**
     * Default sliding window size for local attention layers.
     * Gemma 4 uses 512 by default; this can be overridden from metadata.
     */
    private static final int DEFAULT_SLIDING_WINDOW = 512;

    /**
     * Default local RoPE frequency base.
     */
    private static final double DEFAULT_LOCAL_FREQ_BASE = 10000.0;

    /**
     * Default global RoPE frequency base.
     */
    private static final double DEFAULT_GLOBAL_FREQ_BASE = 1000000.0;

    @Override
    public String getName() {
        return "gemma";
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
        return SUPPORTED_VARIANTS.contains(archLower) || archLower.contains("gemma");
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
        int slidingWindow = getSlidingWindowSize(metadata);
        int sharedKvStartLayer = getSharedKvStartLayer(metadata, numLayers);
        int sharedKvSourceLayer = getSharedKvSourceLayer(metadata, sharedKvStartLayer);
        double localFreqBase = getLocalFreqBase(metadata);
        double globalFreqBase = getGlobalFreqBase(metadata);
        double localFreqScale = getLocalFreqScale(metadata);
        double globalFreqScale = getGlobalFreqScale(metadata);

        log.info("Building Gemma graph: {} layers, hidden={}, heads={}, kv_heads={}, headDim={}, " +
                        "slidingWindow={}, sharedKvStart={}, sharedKvSource={}, " +
                        "localFreqBase={}, globalFreqBase={}, dtype={}",
                numLayers, hiddenSize, numHeads, numKvHeads, headDim,
                slidingWindow, sharedKvStartLayer, sharedKvSourceLayer,
                localFreqBase, globalFreqBase, dtype);

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

        // Gemma models scale embeddings by sqrt(hidden_size)
        double embeddingScale = Math.sqrt(hiddenSize);
        hidden = hidden.mul("embed_scaled", embeddingScale);

        // Track shared K/V outputs for shared-KV layers
        SDVariable sharedKey = null;
        SDVariable sharedValue = null;

        // Build transformer layers
        for (int layer = 0; layer < numLayers; layer++) {
            boolean isGlobalAttention = isGlobalAttentionLayer(config, layer);
            boolean isSharedKvLayer = (layer >= sharedKvStartLayer && sharedKey != null);
            boolean isSharedKvSource = (layer == sharedKvSourceLayer);
            boolean hasPerLayerEmbedding = weights.containsKey("blk." + layer + ".ple.weight");

            // Apply per-layer embedding if present (Gemma 4)
            if (hasPerLayerEmbedding) {
                hidden = applyPerLayerEmbedding(sd, hidden, inputIds, layer, weights);
            }

            // Pre-attention RMS normalization
            SDVariable normed = buildRMSNorm(sd, hidden,
                    "model.layers." + layer + ".input_layernorm",
                    "blk." + layer + ".attn_norm", weights, config, dtype);

            // Attention
            SDVariable attnOut;
            if (isSharedKvLayer) {
                // Use shared K/V from the source layer
                attnOut = buildSharedKvAttention(sd, normed, sharedKey, sharedValue,
                        layer, config, weights, dtype, isGlobalAttention, slidingWindow);
            } else {
                // Standard attention with per-layer Q/K/V
                SDVariable[] qkv = buildQKV(sd, normed, layer, config, weights, dtype, isGlobalAttention,
                        localFreqBase, globalFreqBase, localFreqScale, globalFreqScale);
                SDVariable q = qkv[0];
                SDVariable k = qkv[1];
                SDVariable v = qkv[2];

                // Capture K/V at the source layer for sharing
                if (isSharedKvSource) {
                    sharedKey = k;
                    sharedValue = v;
                }

                attnOut = buildAttention(sd, q, k, v, layer, config, isGlobalAttention, slidingWindow);
            }

            // Output projection
            INDArray oWeight = weights.get("blk." + layer + ".attn_output.weight");
            if (oWeight != null) {
                SDVariable wo = sd.var("model.layers." + layer + ".self_attn.o_proj.weight", oWeight);
                attnOut = sd.mmul("attn_proj_" + layer, attnOut, wo.permute(1, 0));
            }

            // Post-attention residual
            SDVariable postAttn = hidden.add("post_attn_" + layer, attnOut);

            // Pre-FFN RMS normalization
            SDVariable ffnNormed = buildRMSNorm(sd, postAttn,
                    "model.layers." + layer + ".post_attention_layernorm",
                    "blk." + layer + ".ffn_norm", weights, config, dtype);

            // SwiGLU FFN
            SDVariable ffnOut = buildSwiGLUFFN(sd, ffnNormed, layer, weights, dtype);

            // Post-FFN residual
            hidden = postAttn.add("layer_out_" + layer, ffnOut);
        }

        // Final RMS normalization
        hidden = buildRMSNorm(sd, hidden, "model.norm", "output_norm", weights, config, dtype);

        // Output projection (LM head)
        INDArray outputWeight = weights.get("output.weight");
        if (outputWeight == null) {
            // Gemma ties embedding and output weights
            outputWeight = tokenEmbedWeight;
        }
        SDVariable lmHead = sd.var("lm_head.weight", outputWeight);

        // Logits: [batch, seq_len, vocab_size]
        sd.mmul("logits", hidden, lmHead.permute(1, 0));

        return sd;
    }

    // ========================================================================
    // Per-Layer Embedding
    // ========================================================================

    private SDVariable applyPerLayerEmbedding(SameDiff sd, SDVariable hidden, SDVariable tokenIds,
                                              int layerIdx, Map<String, INDArray> weights) {
        String prefix = "blk." + layerIdx;
        INDArray pleWeight = weights.get(prefix + ".ple.weight");
        if (pleWeight == null) {
            return hidden;
        }

        SDVariable pleVar = sd.var("model.layers." + layerIdx + ".ple.weight", pleWeight);
        SDVariable result = new PerLayerEmbedding(sd, hidden, pleVar, tokenIds).outputVariable();
        sd.updateVariableNameAndReference(result, "ple_" + layerIdx);
        return result;
    }

    // ========================================================================
    // Q/K/V projections with Dual RoPE
    // ========================================================================

    private SDVariable[] buildQKV(SameDiff sd, SDVariable input, int layerIdx,
                                  ArchitectureConfig config, Map<String, INDArray> weights, DataType dtype,
                                  boolean isGlobalAttention, double localFreqBase, double globalFreqBase,
                                  double localFreqScale, double globalFreqScale) {
        String prefix = "blk." + layerIdx;
        int numHeads = config.getNumAttentionHeads();
        int numKvHeads = config.getNumKVHeads();
        int headDim = config.getHeadDimension();

        INDArray qWeight = weights.get(prefix + ".attn_q.weight");
        INDArray kWeight = weights.get(prefix + ".attn_k.weight");
        INDArray vWeight = weights.get(prefix + ".attn_v.weight");

        if (qWeight == null || kWeight == null || vWeight == null) {
            throw new IllegalStateException("Missing Q/K/V weights for layer " + layerIdx);
        }

        String attnPrefix = "model.layers." + layerIdx + ".self_attn.";
        SDVariable wq = sd.var(attnPrefix + "q_proj.weight", qWeight);
        SDVariable wk = sd.var(attnPrefix + "k_proj.weight", kWeight);
        SDVariable wv = sd.var(attnPrefix + "v_proj.weight", vWeight);

        // Project: [batch, seq, hidden] -> [batch, seq, proj_dim]
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

        // Reshape to multi-head format: [batch, seq, heads, head_dim]
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

        // Apply Dual RoPE (Gemma 4) or standard RoPE (earlier Gemma)
        int attentionType = isGlobalAttention ? DualRoPE.ATTENTION_TYPE_GLOBAL : DualRoPE.ATTENTION_TYPE_LOCAL;

        q = new DualRoPE(sd, q, attentionType, 0,
                localFreqBase, globalFreqBase, localFreqScale, globalFreqScale).outputVariable();
        sd.updateVariableNameAndReference(q, "q_rope_" + layerIdx);

        k = new DualRoPE(sd, k, attentionType, 0,
                localFreqBase, globalFreqBase, localFreqScale, globalFreqScale).outputVariable();
        sd.updateVariableNameAndReference(k, "k_rope_" + layerIdx);

        return new SDVariable[]{q, k, v};
    }

    // ========================================================================
    // Attention
    // ========================================================================

    private SDVariable buildAttention(SameDiff sd, SDVariable q, SDVariable k, SDVariable v,
                                      int layerIdx, ArchitectureConfig config,
                                      boolean isGlobalAttention, int slidingWindow) {
        int numHeads = config.getNumAttentionHeads();
        int headDim = config.getHeadDimension();

        // Dot-product attention (causal)
        SDVariable attnOut = sd.nn.dotProductAttentionV2(
                "attn_out_" + layerIdx,
                q, v, k, null, null,
                0.0, 0.0, true, false
        );

        // Reshape: [batch, seq, numHeads, headDim] -> [batch, seq, numHeads * headDim]
        SDVariable batchDim = sd.sizeAt(q, 0);
        SDVariable seqDim = sd.sizeAt(q, 1);
        int attnOutDim = numHeads * headDim;
        SDVariable outShapeVar = sd.stack("attn_out_shape_" + layerIdx, 0,
                batchDim, seqDim,
                sd.constant(Nd4j.scalar((long) attnOutDim)));
        return sd.reshape("attn_flat_" + layerIdx, attnOut, outShapeVar);
    }

    private SDVariable buildSharedKvAttention(SameDiff sd, SDVariable input,
                                              SDVariable sharedKey, SDVariable sharedValue,
                                              int layerIdx, ArchitectureConfig config,
                                              Map<String, INDArray> weights, DataType dtype,
                                              boolean isGlobalAttention, int slidingWindow) {
        String prefix = "blk." + layerIdx;
        int numHeads = config.getNumAttentionHeads();
        int numKvHeads = config.getNumKVHeads();
        int headDim = config.getHeadDimension();

        // Only Q projection for shared-KV layers
        INDArray qWeight = weights.get(prefix + ".attn_q.weight");
        if (qWeight == null) {
            throw new IllegalStateException("Missing Q weights for shared-KV layer " + layerIdx);
        }

        String attnPrefix = "model.layers." + layerIdx + ".self_attn.";
        SDVariable wq = sd.var(attnPrefix + "q_proj.weight", qWeight);

        SDVariable q = sd.mmul("q_" + layerIdx, input, wq.permute(1, 0));

        // Reshape Q to multi-head
        SDVariable batchDim = sd.sizeAt(input, 0);
        SDVariable seqDim = sd.sizeAt(input, 1);
        SDVariable qShapeVar = sd.stack("q_shape_" + layerIdx, 0,
                batchDim, seqDim,
                sd.constant(Nd4j.scalar((long) numHeads)),
                sd.constant(Nd4j.scalar((long) headDim)));
        q = sd.reshape("q_heads_" + layerIdx, q, qShapeVar);

        // Apply RoPE to Q only (K/V already have RoPE from source layer)
        q = new DualRoPE(sd, q,
                isGlobalAttention ? DualRoPE.ATTENTION_TYPE_GLOBAL : DualRoPE.ATTENTION_TYPE_LOCAL,
                0, DEFAULT_LOCAL_FREQ_BASE, DEFAULT_GLOBAL_FREQ_BASE, 1.0, 1.0).outputVariable();
        sd.updateVariableNameAndReference(q, "q_rope_" + layerIdx);

        // Use SharedKvAttention op
        double scale = 1.0 / Math.sqrt(headDim);
        SDVariable attnOut = new SharedKvAttention(sd, q, sharedKey, sharedValue,
                numHeads, numKvHeads, 1,
                isGlobalAttention ? 0 : slidingWindow, scale).outputVariable();
        sd.updateVariableNameAndReference(attnOut, "shared_kv_attn_out_" + layerIdx);

        // Reshape: [batch, seq, numHeads, headDim] -> [batch, seq, numHeads * headDim]
        int attnOutDim = numHeads * headDim;
        SDVariable outShapeVar = sd.stack("attn_out_shape_" + layerIdx, 0,
                batchDim, seqDim,
                sd.constant(Nd4j.scalar((long) attnOutDim)));
        return sd.reshape("attn_flat_" + layerIdx, attnOut, outShapeVar);
    }

    // ========================================================================
    // RMS Normalization
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

        // RMS normalization: x * rsqrt(mean(x^2) + eps) * gamma
        SDVariable squared = input.mul(input);
        SDVariable meanSquared = squared.mean(true, -1);
        SDVariable rms = sd.math.sqrt(meanSquared.add(config.getLayerNormEpsilon()));
        SDVariable normalized = input.div(rms);

        // Gemma adds 1.0 to the norm weights: gamma = (1 + weight)
        SDVariable gammaOffset = gamma.add(1.0);
        return normalized.mul(outputName, gammaOffset);
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
        SDVariable gelu = silu.mul("swiglu_" + layerIdx, up);

        return sd.mmul("down_" + layerIdx, gelu, wDown.permute(1, 0));
    }

    // ========================================================================
    // Attention layer type detection
    // ========================================================================

    /**
     * Determine if a layer uses global (full-context) attention.
     * Gemma alternates between sliding-window and global attention layers.
     * Even-numbered layers (0, 2, 4, ...) use sliding-window (local).
     * Odd-numbered layers (1, 3, 5, ...) use global attention.
     * This can be overridden by explicit layer types in metadata.
     */
    private boolean isGlobalAttentionLayer(ArchitectureConfig config, int layerIdx) {
        // Check explicit layer types first
        if (config.getLayerTypes() != null && layerIdx < config.getLayerTypes().size()) {
            String layerType = config.getLayerTypes().get(layerIdx);
            return "global_attention".equals(layerType) || "full_attention".equals(layerType);
        }

        // Check full attention interval
        int interval = config.getFullAttentionInterval();
        if (interval > 0) {
            return ((layerIdx + 1) % interval == 0);
        }

        // Default Gemma pattern: odd layers are global
        return (layerIdx % 2) == 1;
    }

    // ========================================================================
    // Metadata extraction helpers
    // ========================================================================

    private int getSlidingWindowSize(GGMLMetadata metadata) {
        Object val = metadata.getRawMetadata().get("gemma.attention.sliding_window");
        if (val instanceof Number) return ((Number) val).intValue();
        val = metadata.getRawMetadata().get("attention.sliding_window");
        if (val instanceof Number) return ((Number) val).intValue();
        return DEFAULT_SLIDING_WINDOW;
    }

    /**
     * Get the layer index at which shared KV begins.
     * Layers from this index onward reuse K/V from the source layer.
     * Default: no shared KV (returns numLayers, meaning no layer qualifies).
     */
    private int getSharedKvStartLayer(GGMLMetadata metadata, int numLayers) {
        Object val = metadata.getRawMetadata().get("gemma.shared_kv.start_layer");
        if (val instanceof Number) return ((Number) val).intValue();
        val = metadata.getRawMetadata().get("shared_kv.start_layer");
        if (val instanceof Number) return ((Number) val).intValue();
        return numLayers; // No shared KV by default
    }

    /**
     * Get the source layer whose K/V are shared with subsequent layers.
     * Default: the layer just before the shared KV start.
     */
    private int getSharedKvSourceLayer(GGMLMetadata metadata, int sharedKvStartLayer) {
        Object val = metadata.getRawMetadata().get("gemma.shared_kv.source_layer");
        if (val instanceof Number) return ((Number) val).intValue();
        val = metadata.getRawMetadata().get("shared_kv.source_layer");
        if (val instanceof Number) return ((Number) val).intValue();
        return Math.max(0, sharedKvStartLayer - 1);
    }

    private double getLocalFreqBase(GGMLMetadata metadata) {
        Object val = metadata.getRawMetadata().get("gemma.rope.local_freq_base");
        if (val instanceof Number) return ((Number) val).doubleValue();
        return metadata.getRopeFreqBase() > 0 ? metadata.getRopeFreqBase() : DEFAULT_LOCAL_FREQ_BASE;
    }

    private double getGlobalFreqBase(GGMLMetadata metadata) {
        Object val = metadata.getRawMetadata().get("gemma.rope.global_freq_base");
        if (val instanceof Number) return ((Number) val).doubleValue();
        return DEFAULT_GLOBAL_FREQ_BASE;
    }

    private double getLocalFreqScale(GGMLMetadata metadata) {
        Object val = metadata.getRawMetadata().get("gemma.rope.local_freq_scale");
        if (val instanceof Number) return ((Number) val).doubleValue();
        return 1.0;
    }

    private double getGlobalFreqScale(GGMLMetadata metadata) {
        Object val = metadata.getRawMetadata().get("gemma.rope.global_freq_scale");
        if (val instanceof Number) return ((Number) val).doubleValue();
        return 1.0;
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

        // Per-layer embedding (Gemma 4)
        patterns.put("blk.{layer}.ple.weight", "model.layers.{layer}.ple.weight");

        // Attention layers
        patterns.put("blk.{layer}.attn_q.weight", "model.layers.{layer}.self_attn.q_proj.weight");
        patterns.put("blk.{layer}.attn_k.weight", "model.layers.{layer}.self_attn.k_proj.weight");
        patterns.put("blk.{layer}.attn_v.weight", "model.layers.{layer}.self_attn.v_proj.weight");
        patterns.put("blk.{layer}.attn_output.weight", "model.layers.{layer}.self_attn.o_proj.weight");

        // Attention biases (optional)
        patterns.put("blk.{layer}.attn_q.bias", "model.layers.{layer}.self_attn.q_proj.bias");
        patterns.put("blk.{layer}.attn_k.bias", "model.layers.{layer}.self_attn.k_proj.bias");
        patterns.put("blk.{layer}.attn_v.bias", "model.layers.{layer}.self_attn.v_proj.bias");

        // Layer norms
        patterns.put("blk.{layer}.attn_norm.weight", "model.layers.{layer}.input_layernorm.weight");
        patterns.put("blk.{layer}.ffn_norm.weight", "model.layers.{layer}.post_attention_layernorm.weight");

        // FFN (SwiGLU)
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
                .fullAttentionInterval(metadata.getFullAttentionInterval())
                .ropeType(metadata.getRopeType())
                .useRmsNorm(true)
                .useSwiGLU(true)
                .useRotaryEmbeddings(true)
                .decoderOnly(true)
                .build();
    }
}
