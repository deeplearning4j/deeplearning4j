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
import org.nd4j.linalg.api.ops.impl.transforms.custom.DotProductAttentionV2;
import org.nd4j.linalg.api.ops.impl.transforms.custom.FusedMRoPE;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Architecture handler for Qwen3-VL and Qwen2.5-VL vision-language models.
 *
 * <p>Supports the Qwen3-VL multimodal architecture which combines:</p>
 * <ul>
 *   <li><b>Vision encoder</b>: SigLIP-2 ViT with 27 blocks, hidden=1152, 16 heads, patch=16,
 *       temporal_patch_size=2</li>
 *   <li><b>DeepStack merger</b>: Extracts features from ViT layers [8, 16, 24] and routes
 *       them to the LLM via the MLP projector</li>
 *   <li><b>MLP projector</b>: Maps vision features (hidden=1152) to LLM hidden space (4096)</li>
 *   <li><b>LLM decoder</b>: 36 layers, hidden=4096, 32 heads, 8 KV heads, intermediate=12288,
 *       SiLU activation, RMSNorm</li>
 *   <li><b>M-RoPE (Multimodal RoPE)</b>: Interleaved mode with sections [24, 20, 20] for
 *       temporal, height, and width position encodings respectively</li>
 * </ul>
 *
 * <p>GGUF tensor naming conventions:</p>
 * <ul>
 *   <li>Vision blocks: {@code v.blk.{layer}.attn_q.weight}, etc.</li>
 *   <li>LLM blocks: {@code blk.{layer}.attn_q.weight}, etc. (standard Qwen pattern)</li>
 *   <li>MLP projector: {@code mm.0.weight}, {@code mm.2.weight}</li>
 * </ul>
 *
 * <p>For the LLM decoder the implementation follows the LLaMA/Qwen pattern with
 * separate Q/K/V projections and QK norms, but substitutes
 * {@link FusedMRoPE} for standard RoPE to carry temporal, height, and width
 * position IDs through the attention layers.</p>
 */
@Slf4j
public class Qwen3VLArchitecture implements ModelArchitecture {

    // ========================================================================
    // Qwen3-VL specific constants
    // ========================================================================

    /** Temporal (time) section size for M-RoPE head dimension split. */
    public static final int MROPE_SECTION_T = 24;

    /** Height section size for M-RoPE head dimension split. */
    public static final int MROPE_SECTION_H = 20;

    /** Width section size for M-RoPE head dimension split. */
    public static final int MROPE_SECTION_W = 20;

    /** ViT layer indices from which DeepStack extracts intermediate features. */
    public static final int[] DEEPSTACK_LAYERS = new int[]{8, 16, 24};

    /**
     * Number of frames (patches in time) merged per visual token.
     * A value of 2 means 2 consecutive frames are combined into one sequence position.
     */
    public static final int TEMPORAL_PATCH_SIZE = 2;

    // Default vision encoder hyperparameters (SigLIP-2 variant used in Qwen3-VL)
    private static final int VIT_NUM_LAYERS      = 27;
    private static final int VIT_HIDDEN_SIZE     = 1152;
    private static final int VIT_NUM_HEADS       = 16;
    private static final int VIT_HEAD_DIM        = VIT_HIDDEN_SIZE / VIT_NUM_HEADS;  // 72
    private static final int VIT_INTERMEDIATE    = VIT_HIDDEN_SIZE * 4;              // 4608
    private static final float VIT_LAYER_NORM_EPS = 1e-6f;

    // Default LLM decoder hyperparameters for Qwen3-VL 7B
    private static final int LLM_NUM_LAYERS      = 36;
    private static final int LLM_HIDDEN_SIZE     = 4096;
    private static final int LLM_NUM_HEADS       = 32;
    private static final int LLM_NUM_KV_HEADS    = 8;
    private static final int LLM_HEAD_DIM        = 128;  // LLM_HIDDEN_SIZE / LLM_NUM_HEADS
    private static final int LLM_INTERMEDIATE    = 12288;
    private static final float LLM_LAYER_NORM_EPS = 1e-6f;

    // M-RoPE frequency base for Qwen3-VL (vision-language specific)
    private static final double MROPE_FREQ_BASE  = 1000000.0;

    private static final Set<String> SUPPORTED_VARIANTS = Set.of(
            "qwen3vl",
            "qwen3_vl",
            "qwen2_vl",
            "qwen2.5_vl",
            "qwen25_vl",
            "qwen2.5-vl",
            "qwen3-vl"
    );

    @Override
    public String getName() {
        return "qwen3vl";
    }

    @Override
    public Set<String> getSupportedVariants() {
        return SUPPORTED_VARIANTS;
    }

    /**
     * Handles models whose GGUF architecture name contains "qwen" and also indicates
     * vision-language capability (via "vl" in the variant name, or a metadata key that
     * signals a vision encoder is present).
     */
    @Override
    public boolean canHandle(GGMLMetadata metadata) {
        String arch = metadata.getArchitecture();
        if (arch == null) return false;

        String archLower = arch.toLowerCase();

        // Must be a Qwen-family model
        if (!archLower.contains("qwen")) {
            return false;
        }

        // Accept if the architecture string itself signals VL capability
        if (archLower.contains("vl")) {
            return true;
        }

        // Accept if any registered variant name matches exactly
        if (SUPPORTED_VARIANTS.contains(archLower)) {
            return true;
        }

        // Accept if raw metadata contains a vision-specific key indicating this is a VL model
        Map<String, Object> raw = metadata.getRawMetadata();
        if (raw != null) {
            for (String key : raw.keySet()) {
                String kl = key.toLowerCase();
                if (kl.startsWith("vision.") || kl.startsWith("v.") || kl.contains("vision_config")) {
                    return true;
                }
            }
        }

        return false;
    }

    @Override
    public String getDefaultChatTemplateType() {
        return "chatml";
    }

    @Override
    public String getModelSystemProperty() {
        return "qwen3vl.gguf.path";
    }

    @Override
    public String getReferencePrompt() {
        return "Describe what is happening in this video.";
    }

    // ========================================================================
    // Graph construction
    // ========================================================================

    @Override
    public SameDiff buildGraph(GGMLMetadata metadata, Map<String, INDArray> weights,
                               ConversionOptions options) {
        SameDiff sd = SameDiff.create();
        ArchitectureConfig llmConfig = buildLlmConfig(metadata);
        DataType dtype = options.getTargetDataType();

        log.info("Building Qwen3-VL graph: vitLayers={}, llmLayers={}, llmHidden={}, "
                + "llmHeads={}, llmKvHeads={}, llmHeadDim={}, mropeSection=[{},{},{}], dtype={}",
                VIT_NUM_LAYERS, llmConfig.getNumLayers(), llmConfig.getHiddenSize(),
                llmConfig.getNumAttentionHeads(), llmConfig.getNumKVHeads(),
                llmConfig.getHeadDimension(),
                MROPE_SECTION_T, MROPE_SECTION_H, MROPE_SECTION_W,
                dtype);

        // ----------------------------------------------------------------
        // Inputs
        // ----------------------------------------------------------------

        // Text token IDs: [batch, seq_len]
        SDVariable inputIds = sd.placeHolder("input_ids", DataType.INT64, -1, -1);

        // M-RoPE position IDs — three separate INT64 tensors, each [batch, seq_len].
        // Text tokens use standard temporal positions; visual tokens carry spatial coords.
        SDVariable posT = sd.placeHolder("mrope_position_t", DataType.INT64, -1, -1);
        SDVariable posH = sd.placeHolder("mrope_position_h", DataType.INT64, -1, -1);
        SDVariable posW = sd.placeHolder("mrope_position_w", DataType.INT64, -1, -1);

        // Cache write position scalar (enables DSP replay)
        SDVariable cachePosition = sd.placeHolder("cache_position", DataType.INT64);

        // Causal attention bias: [1, 1, Tq, maxKvLen]
        SDVariable causalMask = sd.placeHolder("_causal_mask", DataType.FLOAT, -1, -1, -1, -1);

        // Visual patch tokens (optional — null-equivalent when no vision input present).
        // Shape: [numPatches, vitHiddenSize] after the vision encoder has been run.
        // Callers may skip the vision path and inject pre-computed embeddings directly.
        SDVariable visualEmbeds = sd.placeHolder("visual_embeds", dtype, -1, VIT_HIDDEN_SIZE);

        // ----------------------------------------------------------------
        // Vision encoder (SigLIP-2 ViT, 27 blocks)
        // ----------------------------------------------------------------

        // Pixel values: [batch, numPatches, channels * patchH * patchW * temporalPatchSize]
        SDVariable pixelValues = sd.placeHolder("pixel_values", dtype,
                -1, -1, 3 * 14 * 14 * TEMPORAL_PATCH_SIZE);

        SDVariable vitOut = buildVisionEncoder(sd, pixelValues, weights, dtype);

        // ----------------------------------------------------------------
        // MLP projector (vision hidden -> LLM hidden)
        // ----------------------------------------------------------------

        SDVariable projOut = buildMlpProjector(sd, vitOut, weights, dtype);

        // ----------------------------------------------------------------
        // Token embeddings for text path
        // ----------------------------------------------------------------

        INDArray tokenEmbedWeight = weights.get("token_embd.weight");
        if (tokenEmbedWeight == null) {
            throw new IllegalStateException("Missing token embedding weights (token_embd.weight)");
        }
        SDVariable tokenEmbed = sd.var("model.embed_tokens.weight", tokenEmbedWeight);

        // Gather text embeddings: [batch, seq_len, hiddenSize]
        SDVariable textEmbeds = sd.gather("text_embeds", tokenEmbed, inputIds, 0);

        // NOTE: Merging visual tokens into the text embedding sequence is handled
        // externally before the graph is executed (caller replaces visual token
        // positions in the input_ids embedding with projOut values).
        // The graph accepts a pre-merged hidden state via the combined placeholder path,
        // or processes the text-only path when no visual tokens are present.
        // For the build-graph step we proceed with the text embedding as the initial hidden.
        SDVariable hidden = textEmbeds;

        // ----------------------------------------------------------------
        // Per-layer KV cache placeholders
        // ----------------------------------------------------------------

        int headDim = llmConfig.getHeadDimension();
        int numKVHeads = llmConfig.getNumKVHeads();
        Map<Integer, SDVariable> keyCachePlaceholders = new HashMap<>();
        Map<Integer, SDVariable> valueCachePlaceholders = new HashMap<>();
        for (int layer = 0; layer < llmConfig.getNumLayers(); layer++) {
            SDVariable keyCache = sd.placeHolder("past_key_values." + layer + ".key",
                    dtype, -1, -1, numKVHeads, headDim);
            SDVariable valueCache = sd.placeHolder("past_key_values." + layer + ".value",
                    dtype, -1, -1, numKVHeads, headDim);
            keyCachePlaceholders.put(layer, keyCache);
            valueCachePlaceholders.put(layer, valueCache);
        }

        // ----------------------------------------------------------------
        // LLM decoder (M-RoPE variant of the Qwen transformer)
        // ----------------------------------------------------------------

        List<String> outputNames = new ArrayList<>();
        for (int layer = 0; layer < llmConfig.getNumLayers(); layer++) {
            hidden = buildLlmDecoderBlock(sd, hidden, layer, llmConfig, weights, dtype,
                    posT, posH, posW, cachePosition, causalMask,
                    keyCachePlaceholders.get(layer),
                    valueCachePlaceholders.get(layer));
            outputNames.add("k_rope_" + layer);
            outputNames.add("v_heads_" + layer);
        }

        // Final RMSNorm
        hidden = buildRMSNorm(sd, hidden, "model.norm", "output_norm", weights,
                llmConfig.getLayerNormEpsilon(), dtype);

        // LM head projection
        INDArray outputWeight = weights.get("output.weight");
        if (outputWeight == null) {
            outputWeight = tokenEmbedWeight;
        }
        SDVariable lmHead = sd.var("lm_head.weight", outputWeight);
        SDVariable logits = QuantizedLinear.matMul(sd, "lm_logits", hidden, lmHead, weights, "output.weight", dtype);

        outputNames.add("lm_logits");
        outputNames.add("vision_out");
        outputNames.add("projector_out");
        sd.updateVariableNameAndReference(vitOut, "vision_out");
        sd.updateVariableNameAndReference(projOut, "projector_out");
        sd.setOutputs(outputNames);

        return sd;
    }

    // ========================================================================
    // Vision encoder — SigLIP-2 ViT (27 blocks)
    // ========================================================================

    /**
     * Build the SigLIP-2 vision encoder.
     *
     * <p>Architecture summary:</p>
     * <ul>
     *   <li>Patch embedding (conv-equivalent projection)</li>
     *   <li>27 ViT blocks: LayerNorm -> Self-Attn -> residual -> LayerNorm -> FFN -> residual</li>
     *   <li>DeepStack: intermediate features are extracted at layers [8, 16, 24] and
     *       concatenated channel-wise to the final output</li>
     *   <li>Final LayerNorm</li>
     * </ul>
     *
     * <p>Tensor names follow the GGUF convention:
     * {@code v.blk.{layer}.attn_q.weight}, {@code v.blk.{layer}.ffn_up.weight}, etc.</p>
     *
     * @param sd       SameDiff graph
     * @param input    pixel values [batch, numPatches, patchDim]
     * @param weights  model weight map
     * @param dtype    working data type
     * @return ViT output [batch, numPatches, vitHiddenSize * (1 + numDeepStackLayers)]
     */
    private SDVariable buildVisionEncoder(SameDiff sd, SDVariable input,
                                           Map<String, INDArray> weights, DataType dtype) {
        // Patch embedding: linear projection from patchDim to vitHiddenSize
        INDArray patchEmbedW = weights.get("v.patch_embd.weight");
        INDArray patchEmbedB = weights.get("v.patch_embd.bias");

        SDVariable hidden;
        if (patchEmbedW != null) {
            SDVariable wPatch = sd.var("vision.patch_embed.proj.weight", patchEmbedW);
            hidden = fp32Mmul(sd, "vit_patch_proj", input, wPatch.permute(1, 0), dtype);
            if (patchEmbedB != null) {
                hidden = hidden.add("vit_patch_proj_bias",
                        sd.var("vision.patch_embed.proj.bias", patchEmbedB));
            }
        } else {
            log.warn("Missing vision patch embedding weight (v.patch_embd.weight); "
                    + "using input directly");
            hidden = input;
        }

        // Collect DeepStack intermediate features
        List<SDVariable> deepStackFeatures = new ArrayList<>();

        // 27 ViT blocks
        for (int layer = 0; layer < VIT_NUM_LAYERS; layer++) {
            hidden = buildVitBlock(sd, hidden, layer, weights, dtype);

            // Capture intermediate features for DeepStack merger
            for (int dsLayer : DEEPSTACK_LAYERS) {
                if (layer == dsLayer) {
                    SDVariable captured = hidden;
                    sd.updateVariableNameAndReference(captured, "vit_ds_feat_" + layer);
                    deepStackFeatures.add(captured);
                }
            }
        }

        // Final LayerNorm
        hidden = buildLayerNorm(sd, hidden, "vision.encoder.final_norm",
                "v.post_ln", weights, VIT_LAYER_NORM_EPS, dtype);

        // DeepStack merge: concatenate captured intermediate features with final output
        if (!deepStackFeatures.isEmpty()) {
            // Append final hidden to the list for concatenation
            deepStackFeatures.add(hidden);
            // concat along last (feature) dimension: [..., vitHiddenSize * (1 + numDeepStackLayers)]
            SDVariable[] featArray = deepStackFeatures.toArray(new SDVariable[0]);
            hidden = sd.concat("vit_deepstack_merged", -1, featArray);
        }

        return hidden;
    }

    /**
     * Build a single ViT transformer block.
     *
     * <p>Block structure: pre-norm -> self-attention -> residual -> pre-norm -> FFN -> residual.</p>
     * <p>ViT uses standard LayerNorm (not RMSNorm) and GELU activation.</p>
     */
    private SDVariable buildVitBlock(SameDiff sd, SDVariable input, int layerIdx,
                                      Map<String, INDArray> weights, DataType dtype) {
        String prefix = "v.blk." + layerIdx;

        // Pre-attention LayerNorm
        SDVariable normed1 = buildLayerNorm(sd, input,
                "vision.encoder.layers." + layerIdx + ".norm1",
                prefix + ".ln1", weights, VIT_LAYER_NORM_EPS, dtype);

        // Self-attention (standard ViT attention — no RoPE, no KV cache)
        SDVariable attnOut = buildVitSelfAttention(sd, normed1, layerIdx, weights, dtype);
        SDVariable postAttn = input.add("vit_post_attn_" + layerIdx, attnOut);

        // Pre-FFN LayerNorm
        SDVariable normed2 = buildLayerNorm(sd, postAttn,
                "vision.encoder.layers." + layerIdx + ".norm2",
                prefix + ".ln2", weights, VIT_LAYER_NORM_EPS, dtype);

        // FFN (GELU activation, matching SigLIP-2)
        SDVariable ffnOut = buildVitFFN(sd, normed2, layerIdx, weights, dtype);

        return postAttn.add("vit_block_out_" + layerIdx, ffnOut);
    }

    /**
     * Build ViT self-attention.
     *
     * <p>Standard full-sequence attention without KV cache or position encoding.
     * Q/K/V projections are separate. Output is projected back to vitHiddenSize.</p>
     */
    private SDVariable buildVitSelfAttention(SameDiff sd, SDVariable input, int layerIdx,
                                              Map<String, INDArray> weights, DataType dtype) {
        String prefix = "v.blk." + layerIdx;
        String attnPrefix = "vision.encoder.layers." + layerIdx + ".self_attn.";

        INDArray qW = weights.get(prefix + ".attn_q.weight");
        INDArray kW = weights.get(prefix + ".attn_k.weight");
        INDArray vW = weights.get(prefix + ".attn_v.weight");
        INDArray oW = weights.get(prefix + ".attn_output.weight");

        if (qW == null || kW == null || vW == null || oW == null) {
            log.warn("Missing ViT attention weights for layer {}", layerIdx);
            return input;
        }

        SDVariable wq = sd.var(attnPrefix + "q_proj.weight", qW);
        SDVariable wk = sd.var(attnPrefix + "k_proj.weight", kW);
        SDVariable wv = sd.var(attnPrefix + "v_proj.weight", vW);
        SDVariable wo = sd.var(attnPrefix + "o_proj.weight", oW);

        SDVariable q = fp32Mmul(sd, "vit_q_" + layerIdx, input, wq.permute(1, 0), dtype);
        SDVariable k = fp32Mmul(sd, "vit_k_" + layerIdx, input, wk.permute(1, 0), dtype);
        SDVariable v = fp32Mmul(sd, "vit_v_" + layerIdx, input, wv.permute(1, 0), dtype);

        // Optional biases
        INDArray qb = weights.get(prefix + ".attn_q.bias");
        INDArray kb = weights.get(prefix + ".attn_k.bias");
        INDArray vb = weights.get(prefix + ".attn_v.bias");
        if (qb != null) q = q.add(sd.var(attnPrefix + "q_proj.bias", qb));
        if (kb != null) k = k.add(sd.var(attnPrefix + "k_proj.bias", kb));
        if (vb != null) v = v.add(sd.var(attnPrefix + "v_proj.bias", vb));

        SDVariable batchDim = sd.sizeAt(input, 0);
        SDVariable seqDim   = sd.sizeAt(input, 1);

        SDVariable qShape = sd.stack("vit_q_shape_" + layerIdx, 0,
                batchDim, seqDim,
                sd.constant(Nd4j.scalar((long) VIT_NUM_HEADS)),
                sd.constant(Nd4j.scalar((long) VIT_HEAD_DIM)));
        q = sd.reshape("vit_q_heads_" + layerIdx, q, qShape);
        k = sd.reshape("vit_k_heads_" + layerIdx, k, qShape);
        v = sd.reshape("vit_v_heads_" + layerIdx, v, qShape);

        // Full self-attention (no causal mask, no KV cache)
        SDVariable attnOut = new DotProductAttentionV2(sd,
                q, v, k, null, null,
                null, null, null, null,
                0.0, 0.0, false, false).outputVariable();
        sd.updateVariableNameAndReference(attnOut, "vit_attn_out_" + layerIdx);

        SDVariable outShape = sd.stack("vit_attn_flat_shape_" + layerIdx, 0,
                batchDim, seqDim,
                sd.constant(Nd4j.scalar((long) VIT_HIDDEN_SIZE)));
        SDVariable attnFlat = sd.reshape("vit_attn_flat_" + layerIdx, attnOut, outShape);

        return fp32Mmul(sd, "vit_attn_proj_" + layerIdx, attnFlat, wo.permute(1, 0), dtype);
    }

    /**
     * Build a ViT feed-forward block (GELU activation, matching SigLIP-2).
     *
     * <p>Tensor names: {@code v.blk.{layer}.ffn_up.weight} and {@code v.blk.{layer}.ffn_down.weight}.</p>
     */
    private SDVariable buildVitFFN(SameDiff sd, SDVariable input, int layerIdx,
                                    Map<String, INDArray> weights, DataType dtype) {
        String prefix  = "v.blk." + layerIdx;
        String mlpPfx  = "vision.encoder.layers." + layerIdx + ".mlp.";

        INDArray upW   = weights.get(prefix + ".ffn_up.weight");
        INDArray downW = weights.get(prefix + ".ffn_down.weight");

        if (upW == null || downW == null) {
            log.warn("Missing ViT FFN weights for layer {}", layerIdx);
            return input;
        }

        SDVariable wUp   = sd.var(mlpPfx + "fc1.weight", upW);
        SDVariable wDown = sd.var(mlpPfx + "fc2.weight", downW);

        INDArray upB   = weights.get(prefix + ".ffn_up.bias");
        INDArray downB = weights.get(prefix + ".ffn_down.bias");

        SDVariable up = fp32Mmul(sd, "vit_up_" + layerIdx, input, wUp.permute(1, 0), dtype);
        if (upB != null) up = up.add(sd.var(mlpPfx + "fc1.bias", upB));

        SDVariable activated = sd.nn.gelu("vit_gelu_" + layerIdx, up);

        SDVariable down = fp32Mmul(sd, "vit_down_" + layerIdx, activated, wDown.permute(1, 0), dtype);
        if (downB != null) down = down.add(sd.var(mlpPfx + "fc2.bias", downB));

        return down;
    }

    // ========================================================================
    // MLP projector (vision -> LLM hidden space)
    // ========================================================================

    /**
     * Build the two-layer MLP projector that maps vision features to LLM hidden size.
     *
     * <p>Projector structure: Linear -> GELU -> Linear (matching Qwen-VL reference).</p>
     * <p>Tensor names follow the GGUF convention: {@code mm.0.weight}, {@code mm.2.weight}.</p>
     *
     * @param sd      SameDiff graph
     * @param input   vision encoder output [..., vitHiddenSize * numDeepstackSlots]
     * @param weights model weight map
     * @param dtype   working data type
     * @return projected features [..., llmHiddenSize]
     */
    private SDVariable buildMlpProjector(SameDiff sd, SDVariable input,
                                          Map<String, INDArray> weights, DataType dtype) {
        INDArray fc1W = weights.get("mm.0.weight");
        INDArray fc2W = weights.get("mm.2.weight");

        if (fc1W == null || fc2W == null) {
            log.warn("Missing MLP projector weights (mm.0.weight / mm.2.weight); "
                    + "returning vision features unchanged");
            return input;
        }

        SDVariable w1 = sd.var("projector.fc1.weight", fc1W);
        SDVariable w2 = sd.var("projector.fc2.weight", fc2W);

        INDArray fc1B = weights.get("mm.0.bias");
        INDArray fc2B = weights.get("mm.2.bias");

        SDVariable h = fp32Mmul(sd, "proj_fc1", input, w1.permute(1, 0), dtype);
        if (fc1B != null) h = h.add(sd.var("projector.fc1.bias", fc1B));

        h = sd.nn.gelu("proj_gelu", h);

        SDVariable out = fp32Mmul(sd, "proj_fc2", h, w2.permute(1, 0), dtype);
        if (fc2B != null) out = out.add(sd.var("projector.fc2.bias", fc2B));

        return out;
    }

    // ========================================================================
    // LLM decoder with M-RoPE
    // ========================================================================

    /**
     * Build a single LLM decoder block using M-RoPE.
     *
     * <p>Follows the Qwen/LLaMA pattern (separate Q/K/V, QK norms, SwiGLU FFN) but
     * replaces standard RoPE with {@link FusedMRoPE} to encode temporal, height, and
     * width positions independently.</p>
     */
    private SDVariable buildLlmDecoderBlock(SameDiff sd, SDVariable input, int layerIdx,
                                             ArchitectureConfig config,
                                             Map<String, INDArray> weights, DataType dtype,
                                             SDVariable posT, SDVariable posH, SDVariable posW,
                                             SDVariable cachePosition, SDVariable causalMask,
                                             SDVariable keyCache, SDVariable valueCache) {
        String prefix = "blk." + layerIdx;

        // Pre-attention RMSNorm
        SDVariable normed = buildRMSNorm(sd, input,
                "model.layers." + layerIdx + ".input_layernorm",
                prefix + ".attn_norm", weights,
                config.getLayerNormEpsilon(), dtype);

        // Self-attention with M-RoPE
        SDVariable attnOut = buildMRoPEAttention(sd, normed, layerIdx, config, weights, dtype,
                posT, posH, posW, cachePosition, causalMask, keyCache, valueCache);

        SDVariable postAttn = input.add("post_attn_" + layerIdx, attnOut);

        // Pre-FFN RMSNorm (try post_attention_norm first, fallback to ffn_norm)
        String postAttnNormKey = weights.containsKey(prefix + ".post_attention_norm.weight")
                ? prefix + ".post_attention_norm"
                : prefix + ".ffn_norm";
        SDVariable ffnNormed = buildRMSNorm(sd, postAttn,
                "model.layers." + layerIdx + ".post_attention_layernorm",
                postAttnNormKey, weights, config.getLayerNormEpsilon(), dtype);

        // SwiGLU FFN
        SDVariable ffnOut = buildSwiGLUFFN(sd, ffnNormed, layerIdx, weights, dtype);

        return postAttn.add("layer_out_" + layerIdx, ffnOut);
    }

    /**
     * Build LLM self-attention using FusedMRoPE.
     *
     * <p>Uses separate Q/K/V projections with optional QK norms (present in Qwen3 models).
     * Position encoding is performed via {@link FusedMRoPE} with sections
     * [{@value #MROPE_SECTION_T}, {@value #MROPE_SECTION_H}, {@value #MROPE_SECTION_W}]
     * in interleaved mode.</p>
     */
    private SDVariable buildMRoPEAttention(SameDiff sd, SDVariable input, int layerIdx,
                                            ArchitectureConfig config,
                                            Map<String, INDArray> weights, DataType dtype,
                                            SDVariable posT, SDVariable posH, SDVariable posW,
                                            SDVariable cachePosition, SDVariable causalMask,
                                            SDVariable keyCache, SDVariable valueCache) {
        String prefix     = "blk." + layerIdx;
        String attnPrefix = "model.layers." + layerIdx + ".self_attn.";
        int numHeads   = config.getNumAttentionHeads();
        int numKVHeads = config.getNumKVHeads();
        int headDim    = config.getHeadDimension();

        INDArray qW = weights.get(prefix + ".attn_q.weight");
        INDArray kW = weights.get(prefix + ".attn_k.weight");
        INDArray vW = weights.get(prefix + ".attn_v.weight");
        INDArray oW = weights.get(prefix + ".attn_output.weight");

        if (qW == null || kW == null || vW == null || oW == null) {
            log.warn("Missing LLM attention weights for layer {}", layerIdx);
            return input;
        }

        int kOutDim      = QuantizedLinear.logicalOutputDim(weights, prefix + ".attn_k.weight", kW);
        int qOutDim      = QuantizedLinear.logicalOutputDim(weights, prefix + ".attn_q.weight", qW);
        int actualKHeads = (headDim > 0) ? kOutDim / headDim : numKVHeads;
        int actualQHeads = (headDim > 0) ? qOutDim / headDim : numHeads;
        if (headDim <= 0) headDim = kOutDim / numKVHeads;

        SDVariable wq = sd.var(attnPrefix + "q_proj.weight", qW);
        SDVariable wk = sd.var(attnPrefix + "k_proj.weight", kW);
        SDVariable wv = sd.var(attnPrefix + "v_proj.weight", vW);
        SDVariable wo = sd.var(attnPrefix + "o_proj.weight", oW);

        // Q/K/V projections (FP32 upcast to prevent overflow at large hidden sizes)
        SDVariable q = QuantizedLinear.matMul(sd, "q_" + layerIdx, input, wq, weights, prefix + ".attn_q.weight", dtype);
        SDVariable k = QuantizedLinear.matMul(sd, "k_" + layerIdx, input, wk, weights, prefix + ".attn_k.weight", dtype);
        SDVariable v = QuantizedLinear.matMul(sd, "v_" + layerIdx, input, wv, weights, prefix + ".attn_v.weight", dtype);

        // Optional biases
        INDArray qb = weights.get(prefix + ".attn_q.bias");
        INDArray kb = weights.get(prefix + ".attn_k.bias");
        INDArray vb = weights.get(prefix + ".attn_v.bias");
        if (qb != null) q = q.add(sd.var(attnPrefix + "q_proj.bias", qb));
        if (kb != null) k = k.add(sd.var(attnPrefix + "k_proj.bias", kb));
        if (vb != null) v = v.add(sd.var(attnPrefix + "v_proj.bias", vb));

        SDVariable batchDim = sd.sizeAt(input, 0);
        SDVariable seqDim   = sd.sizeAt(input, 1);
        int fHeadDim = headDim;  // effectively final for lambdas (not used here, just cleaner)

        SDVariable qShape = sd.stack("q_shape_" + layerIdx, 0,
                batchDim, seqDim,
                sd.constant(Nd4j.scalar((long) actualQHeads)),
                sd.constant(Nd4j.scalar((long) fHeadDim)));
        SDVariable kvShape = sd.stack("kv_shape_" + layerIdx, 0,
                batchDim, seqDim,
                sd.constant(Nd4j.scalar((long) actualKHeads)),
                sd.constant(Nd4j.scalar((long) fHeadDim)));

        q = sd.reshape("q_heads_" + layerIdx, q, qShape);
        k = sd.reshape("k_heads_" + layerIdx, k, kvShape);
        v = sd.reshape("v_heads_" + layerIdx, v, kvShape);

        // Optional QK norms (present in Qwen3 models)
        INDArray qNormW = weights.get(prefix + ".attn_q_norm.weight");
        INDArray kNormW = weights.get(prefix + ".attn_k_norm.weight");
        if (qNormW != null) {
            q = applyHeadNorm(sd, q, attnPrefix + "q_norm_" + layerIdx, qNormW,
                    config.getLayerNormEpsilon());
        }
        if (kNormW != null) {
            k = applyHeadNorm(sd, k, attnPrefix + "k_norm_" + layerIdx, kNormW,
                    config.getLayerNormEpsilon());
        }

        // M-RoPE: apply multimodal rotary position embeddings using three position tensors.
        // Interleaved=true and freq_base=1e6 match the Qwen3-VL reference implementation.
        q = FusedMRoPE.forQwen3VL(sd, q, posT, posH, posW).outputVariable();
        sd.updateVariableNameAndReference(q, "q_rope_" + layerIdx);

        k = FusedMRoPE.forQwen3VL(sd, k, posT, posH, posW).outputVariable();
        sd.updateVariableNameAndReference(k, "k_rope_" + layerIdx);

        // FusedMRoPE may promote HALF→FLOAT; V must match Q/K dtype
        if (v.dataType() != q.dataType()) {
            v = v.castTo("v_cast_" + layerIdx, q.dataType());
        }

        // Dot-product attention with KV cache and causal mask
        SDVariable attnOut = new DotProductAttentionV2(sd,
                q, v, k, null, null,
                keyCache, valueCache, cachePosition, causalMask,
                0.0, 0.0, false, false).outputVariable();
        sd.updateVariableNameAndReference(attnOut, "attn_out_" + layerIdx);

        int attnOutDim = actualQHeads * fHeadDim;
        SDVariable outShape = sd.stack("attn_out_shape_" + layerIdx, 0,
                batchDim, seqDim,
                sd.constant(Nd4j.scalar((long) attnOutDim)));
        SDVariable attnFlat = sd.reshape("attn_flat_" + layerIdx, attnOut, outShape);

        return QuantizedLinear.matMul(sd, "attn_proj_" + layerIdx, attnFlat, wo, weights, prefix + ".attn_output.weight", dtype);
    }

    /**
     * Build a SwiGLU feed-forward block for the LLM decoder.
     *
     * <p>Tensor names: {@code blk.{layer}.ffn_gate.weight}, {@code blk.{layer}.ffn_up.weight},
     * {@code blk.{layer}.ffn_down.weight}.</p>
     */
    private SDVariable buildSwiGLUFFN(SameDiff sd, SDVariable input, int layerIdx,
                                       Map<String, INDArray> weights, DataType dtype) {
        String prefix = "blk." + layerIdx;
        String mlpPfx  = "model.layers." + layerIdx + ".mlp.";

        INDArray gateW = weights.get(prefix + ".ffn_gate.weight");
        INDArray upW   = weights.get(prefix + ".ffn_up.weight");
        INDArray downW = weights.get(prefix + ".ffn_down.weight");

        if (gateW == null || upW == null || downW == null) {
            log.warn("Missing SwiGLU FFN weights for layer {}", layerIdx);
            return input;
        }

        SDVariable wGate = sd.var(mlpPfx + "gate_proj.weight", gateW);
        SDVariable wUp   = sd.var(mlpPfx + "up_proj.weight",   upW);
        SDVariable wDown = sd.var(mlpPfx + "down_proj.weight", downW);

        SDVariable gate    = QuantizedLinear.matMul(sd, "gate_" + layerIdx, input, wGate, weights, prefix + ".ffn_gate.weight", dtype);
        SDVariable up      = QuantizedLinear.matMul(sd, "up_" + layerIdx,   input, wUp,   weights, prefix + ".ffn_up.weight",   dtype);
        SDVariable silu    = sd.nn.swish(gate);
        SDVariable gated   = silu.mul("swiglu_" + layerIdx, up);

        return QuantizedLinear.matMul(sd, "down_" + layerIdx, gated, wDown, weights, prefix + ".ffn_down.weight", dtype);
    }

    // ========================================================================
    // Normalization helpers
    // ========================================================================

    /**
     * RMS normalization: {@code x * rsqrt(mean(x^2) + eps) * gamma}.
     *
     * <p>Upcasts to FP32 internally when the input is HALF or BFLOAT16 to avoid
     * squared-value overflow, then casts the result back to the original dtype.</p>
     */
    private SDVariable buildRMSNorm(SameDiff sd, SDVariable input, String outputName,
                                     String weightKey, Map<String, INDArray> weights,
                                     float eps, DataType dtype) {
        INDArray normW = weights.get(weightKey + ".weight");
        if (normW == null) {
            log.warn("Missing RMSNorm weight: {}", weightKey);
            return input;
        }

        SDVariable gamma = sd.var(outputName + ".weight", normW);

        boolean needsCast = (input.dataType() == DataType.HALF
                || input.dataType() == DataType.BFLOAT16);
        SDVariable x = needsCast ? input.castTo(outputName + "_f32", DataType.FLOAT) : input;

        SDVariable squared    = x.mul(x);
        SDVariable meanSq     = squared.mean(true, -1);
        SDVariable rms        = sd.math.sqrt(meanSq.add(eps));
        SDVariable normalized = x.div(rms);

        if (needsCast) {
            normalized = normalized.castTo(outputName + "_cast", input.dataType());
        }

        return normalized.mul(outputName, gamma);
    }

    /**
     * LayerNorm: {@code (x - mean) / sqrt(var + eps) * gamma + beta}.
     *
     * <p>Used by the ViT blocks (SigLIP-2 uses standard LayerNorm, not RMSNorm).</p>
     */
    private SDVariable buildLayerNorm(SameDiff sd, SDVariable input, String outputName,
                                       String weightKey, Map<String, INDArray> weights,
                                       float eps, DataType dtype) {
        INDArray normW = weights.get(weightKey + ".weight");
        INDArray normB = weights.get(weightKey + ".bias");

        if (normW == null) {
            log.warn("Missing LayerNorm weight: {}", weightKey);
            return input;
        }

        SDVariable gamma = sd.var(outputName + ".weight", normW);

        boolean needsCast = (input.dataType() == DataType.HALF
                || input.dataType() == DataType.BFLOAT16);
        SDVariable x = needsCast ? input.castTo(outputName + "_f32", DataType.FLOAT) : input;

        SDVariable mean = x.mean(true, -1);
        SDVariable diff = x.sub(mean);
        SDVariable varV = diff.mul(diff).mean(true, -1);
        SDVariable std  = sd.math.sqrt(varV.add(eps));
        SDVariable norm = diff.div(std);

        if (needsCast) {
            norm = norm.castTo(outputName + "_cast", input.dataType());
        }

        SDVariable result = norm.mul(outputName, gamma);
        if (normB != null) {
            result = result.add(outputName + "_biased",
                    sd.var(outputName + ".bias", normB));
        }
        return result;
    }

    /**
     * Per-head RMSNorm. Input shape: [batch, seq, numHeads, headDim].
     * Normalizes along the last (headDim) dimension.
     */
    private SDVariable applyHeadNorm(SameDiff sd, SDVariable input, String outputName,
                                      INDArray normWeight, float eps) {
        SDVariable gamma = sd.var(outputName + ".weight", normWeight);

        boolean needsCast = (input.dataType() == DataType.HALF
                || input.dataType() == DataType.BFLOAT16);
        SDVariable x = needsCast ? input.castTo(outputName + "_f32", DataType.FLOAT) : input;

        SDVariable squared    = x.mul(x);
        SDVariable meanSq     = squared.mean(true, -1);
        SDVariable rms        = sd.math.sqrt(meanSq.add(eps));
        SDVariable normalized = x.div(rms);

        if (needsCast) {
            normalized = normalized.castTo(outputName + "_cast", input.dataType());
        }

        return normalized.mul(outputName, gamma);
    }

    // ========================================================================
    // FP32 matmul helper
    // ========================================================================

    /**
     * Perform a matrix multiply in FP32, then cast the result back to the input dtype.
     *
     * <p>When {@code dtype == HALF}, FP16 dot products over large feature dimensions
     * easily overflow to ±65504. This helper upcasts both operands before the multiply
     * and casts the result back, matching the "compute in FP32, store in FP16" pattern
     * used by PyTorch and HuggingFace.</p>
     */
    private SDVariable fp32Mmul(SameDiff sd, String name,
                                 SDVariable a, SDVariable b, DataType dtype) {
        if (dtype == DataType.HALF || dtype == DataType.BFLOAT16) {
            SDVariable aF32 = a.castTo(name + "_a_f32", DataType.FLOAT);
            SDVariable bF32 = b.castTo(name + "_b_f32", DataType.FLOAT);
            SDVariable res  = sd.mmul(name + "_f32", aF32, bF32);
            return res.castTo(name, dtype);
        }
        return sd.mmul(name, a, b);
    }

    // ========================================================================
    // Config helpers
    // ========================================================================

    /**
     * Build an {@link ArchitectureConfig} for the LLM decoder from GGUF metadata,
     * falling back to the Qwen3-VL 7B defaults when metadata does not provide values.
     */
    private ArchitectureConfig buildLlmConfig(GGMLMetadata metadata) {
        int numLayers  = metadata.getNumLayers() > 0 ? metadata.getNumLayers()  : LLM_NUM_LAYERS;
        int hiddenSize = metadata.getHiddenSize() > 0 ? metadata.getHiddenSize() : LLM_HIDDEN_SIZE;
        int intermSize = metadata.getIntermediateSize() > 0 ? metadata.getIntermediateSize() : LLM_INTERMEDIATE;
        int numHeads   = metadata.getNumAttentionHeads() > 0 ? metadata.getNumAttentionHeads() : LLM_NUM_HEADS;
        int numKVHeads = metadata.getNumKVHeads() > 0 ? metadata.getNumKVHeads() : LLM_NUM_KV_HEADS;
        int headDim    = metadata.getAttentionKeyLength() > 0 ? metadata.getAttentionKeyLength() : LLM_HEAD_DIM;
        float eps      = metadata.getLayerNormEpsilon() > 0 ? metadata.getLayerNormEpsilon() : LLM_LAYER_NORM_EPS;
        float ropeBase = metadata.getRopeFreqBase() > 0 ? metadata.getRopeFreqBase() : (float) MROPE_FREQ_BASE;

        return ArchitectureConfig.builder()
                .numLayers(numLayers)
                .hiddenSize(hiddenSize)
                .intermediateSize(intermSize)
                .numAttentionHeads(numHeads)
                .numKVHeads(numKVHeads)
                .vocabSize(metadata.getVocabSize())
                .contextLength(metadata.getContextLength())
                .layerNormEpsilon(eps)
                .ropeFreqBase(ropeBase)
                .ropeDimensionCount(metadata.getRopeDimensionCount())
                .headDim(headDim)
                .layerTypes(metadata.getLayerTypes())
                .fullAttentionInterval(metadata.getFullAttentionInterval())
                .ropeType(1)  // Qwen models always use interleaved (NeoX-style) RoPE
                .useRotaryEmbeddings(true)
                .useRmsNorm(true)
                .useSwiGLU(true)
                .decoderOnly(true)
                .build();
    }

    // ========================================================================
    // Tensor name patterns
    // ========================================================================

    @Override
    public Map<String, String> getTensorNamePatterns() {
        Map<String, String> patterns = new HashMap<>();

        // ----------------------------------------------------------------
        // Global embeddings and output
        // ----------------------------------------------------------------
        patterns.put("token_embd.weight", "model.embed_tokens.weight");
        patterns.put("output.weight",      "lm_head.weight");
        patterns.put("output_norm.weight", "model.norm.weight");

        // ----------------------------------------------------------------
        // LLM decoder blocks (standard Qwen/LLaMA naming)
        // ----------------------------------------------------------------

        // Separate Q/K/V attention projections
        patterns.put("blk.{layer}.attn_q.weight",      "model.layers.{layer}.self_attn.q_proj.weight");
        patterns.put("blk.{layer}.attn_k.weight",      "model.layers.{layer}.self_attn.k_proj.weight");
        patterns.put("blk.{layer}.attn_v.weight",      "model.layers.{layer}.self_attn.v_proj.weight");
        patterns.put("blk.{layer}.attn_output.weight", "model.layers.{layer}.self_attn.o_proj.weight");

        // Optional attention biases
        patterns.put("blk.{layer}.attn_q.bias",        "model.layers.{layer}.self_attn.q_proj.bias");
        patterns.put("blk.{layer}.attn_k.bias",        "model.layers.{layer}.self_attn.k_proj.bias");
        patterns.put("blk.{layer}.attn_v.bias",        "model.layers.{layer}.self_attn.v_proj.bias");

        // QK norms (Qwen3 style)
        patterns.put("blk.{layer}.attn_q_norm.weight", "model.layers.{layer}.self_attn.q_norm.weight");
        patterns.put("blk.{layer}.attn_k_norm.weight", "model.layers.{layer}.self_attn.k_norm.weight");

        // FFN
        patterns.put("blk.{layer}.ffn_gate.weight",    "model.layers.{layer}.mlp.gate_proj.weight");
        patterns.put("blk.{layer}.ffn_up.weight",      "model.layers.{layer}.mlp.up_proj.weight");
        patterns.put("blk.{layer}.ffn_down.weight",    "model.layers.{layer}.mlp.down_proj.weight");

        // Layer norms
        patterns.put("blk.{layer}.attn_norm.weight",            "model.layers.{layer}.input_layernorm.weight");
        patterns.put("blk.{layer}.ffn_norm.weight",             "model.layers.{layer}.post_attention_layernorm.weight");
        patterns.put("blk.{layer}.post_attention_norm.weight",  "model.layers.{layer}.post_attention_layernorm.weight");

        // ----------------------------------------------------------------
        // Vision encoder blocks (SigLIP-2 ViT — prefix "v.")
        // ----------------------------------------------------------------

        patterns.put("v.patch_embd.weight",               "vision.patch_embed.proj.weight");
        patterns.put("v.patch_embd.bias",                 "vision.patch_embed.proj.bias");
        patterns.put("v.post_ln.weight",                  "vision.encoder.final_norm.weight");
        patterns.put("v.post_ln.bias",                    "vision.encoder.final_norm.bias");

        patterns.put("v.blk.{layer}.attn_q.weight",       "vision.encoder.layers.{layer}.self_attn.q_proj.weight");
        patterns.put("v.blk.{layer}.attn_k.weight",       "vision.encoder.layers.{layer}.self_attn.k_proj.weight");
        patterns.put("v.blk.{layer}.attn_v.weight",       "vision.encoder.layers.{layer}.self_attn.v_proj.weight");
        patterns.put("v.blk.{layer}.attn_output.weight",  "vision.encoder.layers.{layer}.self_attn.o_proj.weight");

        patterns.put("v.blk.{layer}.attn_q.bias",         "vision.encoder.layers.{layer}.self_attn.q_proj.bias");
        patterns.put("v.blk.{layer}.attn_k.bias",         "vision.encoder.layers.{layer}.self_attn.k_proj.bias");
        patterns.put("v.blk.{layer}.attn_v.bias",         "vision.encoder.layers.{layer}.self_attn.v_proj.bias");

        patterns.put("v.blk.{layer}.ffn_up.weight",       "vision.encoder.layers.{layer}.mlp.fc1.weight");
        patterns.put("v.blk.{layer}.ffn_down.weight",     "vision.encoder.layers.{layer}.mlp.fc2.weight");
        patterns.put("v.blk.{layer}.ffn_up.bias",         "vision.encoder.layers.{layer}.mlp.fc1.bias");
        patterns.put("v.blk.{layer}.ffn_down.bias",       "vision.encoder.layers.{layer}.mlp.fc2.bias");

        patterns.put("v.blk.{layer}.ln1.weight",          "vision.encoder.layers.{layer}.norm1.weight");
        patterns.put("v.blk.{layer}.ln1.bias",            "vision.encoder.layers.{layer}.norm1.bias");
        patterns.put("v.blk.{layer}.ln2.weight",          "vision.encoder.layers.{layer}.norm2.weight");
        patterns.put("v.blk.{layer}.ln2.bias",            "vision.encoder.layers.{layer}.norm2.bias");

        // ----------------------------------------------------------------
        // MLP projector
        // ----------------------------------------------------------------
        patterns.put("mm.0.weight", "projector.fc1.weight");
        patterns.put("mm.0.bias",   "projector.fc1.bias");
        patterns.put("mm.2.weight", "projector.fc2.weight");
        patterns.put("mm.2.bias",   "projector.fc2.bias");

        return patterns;
    }
}
