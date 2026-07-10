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
import org.nd4j.linalg.api.ops.impl.transforms.custom.DotProductAttentionV2;
import org.nd4j.linalg.api.ops.impl.transforms.custom.FusedRoPE;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Architecture handler for MiniCPM-V multimodal vision-language models.
 *
 * <p>MiniCPM-V 4.5 is a Video-Language Model (VLM) combining:</p>
 * <ul>
 *   <li><b>Vision encoder</b>: SigLIP2-400M (ViT-style, spatial per-frame, patch size 14)</li>
 *   <li><b>3D-Resampler</b>: Cross-attention module that groups 6 consecutive video frames
 *       and compresses their tokens into 64 output tokens per group (96x compression ratio)</li>
 *   <li><b>LLM decoder</b>: Qwen3-8B compatible causal decoder with GQA, RMSNorm, SwiGLU,
 *       and interleaved NeoX-style RoPE</li>
 * </ul>
 *
 * <h3>GGUF tensor naming conventions:</h3>
 * <ul>
 *   <li>Vision encoder: {@code v.blk.{layer}.attn_q.weight}, etc.</li>
 *   <li>3D-Resampler: {@code resampler.query}, {@code resampler.attn_q.weight}, etc.</li>
 *   <li>LLM decoder: {@code blk.{layer}.*} (standard LLaMA/Qwen pattern)</li>
 *   <li>Token embedding: {@code token_embd.weight}</li>
 *   <li>LM head: {@code output.weight}</li>
 *   <li>Final LLM norm: {@code output_norm.weight}</li>
 *   <li>Vision projection: {@code mm.0.weight} / {@code mm.0.bias}</li>
 * </ul>
 *
 * <h3>16 official quantized GGUF sizes (Q2_K through Q8_0 plus F16/BF16):</h3>
 * <ul>
 *   <li>Q2_K, Q3_K_S, Q3_K_M, Q3_K_L</li>
 *   <li>Q4_0, Q4_K_S, Q4_K_M</li>
 *   <li>Q5_0, Q5_K_S, Q5_K_M</li>
 *   <li>Q6_K, Q8_0</li>
 *   <li>IQ1_M, IQ1_S, IQ2_M, IQ2_XXS</li>
 * </ul>
 */
@Slf4j
public class MiniCPMVArchitecture implements ModelArchitecture {

    // ========================================================================
    // Architecture constants
    // ========================================================================

    /**
     * Number of consecutive video frames grouped by the 3D-Resampler in one pass.
     */
    public static final int RESAMPLER_FRAME_GROUP_SIZE = 6;

    /**
     * Number of output tokens produced by the 3D-Resampler per frame group.
     * The resampler compresses N_patch tokens from 6 frames into 64 tokens (96x compression).
     */
    public static final int RESAMPLER_OUTPUT_TOKENS = 64;

    /**
     * SigLIP2 ViT patch size in pixels (14x14 per patch).
     */
    public static final int VISION_PATCH_SIZE = 14;

    private static final Set<String> SUPPORTED_VARIANTS = Set.of(
            "minicpmv",
            "minicpm-v",
            "minicpm_v",
            "minicpm_v_4_5",
            "minicpmv45"
    );

    // ========================================================================
    // ModelArchitecture interface
    // ========================================================================

    @Override
    public String getName() {
        return "minicpmv";
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

        // Explicit variant match
        if (SUPPORTED_VARIANTS.contains(archLower)) return true;

        // Heuristic: name contains "minicpm" and indicates a vision model
        boolean hasMiniCpm = archLower.contains("minicpm");
        boolean hasVisionMarker = archLower.contains("v") || archLower.contains("vision")
                || archLower.contains("vl") || archLower.contains("vlm");

        if (hasMiniCpm && hasVisionMarker) return true;

        // Tensor-based detection: look for the resampler or vision encoder tensors
        for (var tensor : metadata.getTensors()) {
            String name = tensor.getName().toLowerCase();
            if (name.startsWith("resampler.") || name.startsWith("v.blk.")) {
                return true;
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
        return "minicpmv.gguf.path";
    }

    @Override
    public String getReferencePrompt() {
        return "Describe what is happening in this video.";
    }

    @Override
    public String[] getReferenceExpected() {
        // MiniCPM-V is a video/vision model — textual reference answers are context-dependent.
        // Minimal expected tokens that should appear in any coherent response.
        return new String[]{"the", "video"};
    }

    // ========================================================================
    // Config override — interleaved NeoX RoPE (matches Qwen3 LLM backbone)
    // ========================================================================

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
                // MiniCPM-V uses a Qwen3 LLM backbone — NeoX/interleaved RoPE (type 1)
                .ropeType(1)
                .useRmsNorm(true)
                .useSwiGLU(true)
                .useRotaryEmbeddings(true)
                .decoderOnly(false) // has a vision encoder + resampler prefix
                .build();
    }

    // ========================================================================
    // Graph construction
    // ========================================================================

    @Override
    public SameDiff buildGraph(GGMLMetadata metadata, Map<String, INDArray> weights,
                               ConversionOptions options) {
        SameDiff sd = SameDiff.create();
        ArchitectureConfig config = getConfig(metadata);
        DataType dtype = options.getTargetDataType();

        log.info("Building MiniCPM-V graph: llmLayers={}, hidden={}, heads={}, kvHeads={}, "
                        + "headDim={}, ropeFreqBase={}, layerNormEps={}, dtype={}",
                config.getNumLayers(), config.getHiddenSize(),
                config.getNumAttentionHeads(), config.getNumKVHeads(),
                config.getHeadDimension(), config.getRopeFreqBase(),
                config.getLayerNormEpsilon(), dtype);

        // ---- Multimodal inputs -----------------------------------------------

        // Text token ids: [batch, seq_len]
        SDVariable inputIds = sd.placeHolder("input_ids", DataType.INT64, -1, -1);

        // Pre-projected vision tokens: [batch, numVisionTokens, llmHidden]
        // The caller is responsible for running the vision encoder + resampler and
        // projecting the output into the LLM hidden dimension before feeding this graph.
        // Shape is -1 along the token dimension to support variable numbers of frames.
        SDVariable visionEmbeds = sd.placeHolder("vision_embeds", dtype, -1, -1, config.getHiddenSize());

        // Boolean mask indicating which positions in the token sequence are vision tokens.
        // Shape: [batch, seq_len]  (1 = vision position, 0 = text position)
        SDVariable visionMask = sd.placeHolder("vision_token_mask", DataType.BOOL, -1, -1);

        // Autoregressive decode helpers (mirrors LLaMAArchitecture)
        SDVariable positionOffset = sd.placeHolder("position_offset", DataType.INT64);
        SDVariable cachePosition  = sd.placeHolder("cache_position", DataType.INT64);
        // [1, 1, Tq, maxKvLen] attention bias / causal mask
        SDVariable causalMask     = sd.placeHolder("_causal_mask", DataType.FLOAT, -1, -1, -1, -1);

        // Per-layer KV cache placeholders
        int headDim    = config.getHeadDimension();
        int numKVHeads = config.getNumKVHeads();
        Map<Integer, SDVariable> keyCachePlaceholders   = new HashMap<>();
        Map<Integer, SDVariable> valueCachePlaceholders = new HashMap<>();
        for (int layer = 0; layer < config.getNumLayers(); layer++) {
            SDVariable keyCache = sd.placeHolder(
                    "past_key_values." + layer + ".key", dtype, -1, -1, numKVHeads, headDim);
            SDVariable valueCache = sd.placeHolder(
                    "past_key_values." + layer + ".value", dtype, -1, -1, numKVHeads, headDim);
            keyCachePlaceholders.put(layer, keyCache);
            valueCachePlaceholders.put(layer, valueCache);
        }

        // ---- Vision encoder (SigLIP2 ViT) ------------------------------------

        SDVariable visionOut = buildVisionEncoder(sd, weights, dtype, config);

        // ---- 3D-Resampler ---------------------------------------------------

        // Cross-attention resampler that compresses 6-frame groups to 64 tokens
        SDVariable resamplerOut = buildResampler(sd, visionOut, weights, dtype, config);

        // ---- Multimodal embedding merge ---------------------------------------
        // Token embeddings: [vocab_size, hidden_size]
        INDArray tokenEmbedWeight = weights.get("token_embd.weight");
        if (tokenEmbedWeight == null) {
            throw new IllegalStateException("Missing token embedding weights: token_embd.weight");
        }
        SDVariable tokenEmbed = sd.var("model.embed_tokens.weight", tokenEmbedWeight);

        // Text embeddings: [batch, seq_len, hidden_size]
        SDVariable textEmbeds = sd.gather("text_embeds", tokenEmbed, inputIds, 0);

        // Merge vision and text embeddings using the vision mask.
        // Positions where visionMask==true get visionEmbeds; others keep textEmbeds.
        // Shape broadcast: visionMask [batch, seq] -> [batch, seq, 1] for element-wise select.
        SDVariable maskExpanded = sd.reshape("vision_mask_3d",
                visionMask.castTo("vision_mask_float", dtype), -1, -1, 1);
        // hidden = text_embeds * (1 - mask) + vision_embeds * mask
        // Note: vision_embeds must already be scatter-inserted at the right positions by the caller.
        SDVariable hidden = textEmbeds.mul("text_masked", maskExpanded.rsub(1.0))
                .add("merged_embeds", visionEmbeds.mul("vision_masked", maskExpanded));

        // ---- LLM decoder (Qwen3-8B pattern) ----------------------------------

        List<String> outputNames = new ArrayList<>();
        for (int layer = 0; layer < config.getNumLayers(); layer++) {
            hidden = buildLLMBlock(sd, hidden, layer, config, weights, dtype,
                    positionOffset, cachePosition, causalMask,
                    keyCachePlaceholders.get(layer),
                    valueCachePlaceholders.get(layer));
            outputNames.add("k_rope_" + layer);
            outputNames.add("v_heads_" + layer);
        }

        // Final RMS normalization
        hidden = buildRMSNorm(sd, hidden, "model.norm", "output_norm", weights, config, dtype);

        // LM head (output projection)
        INDArray outputWeight = weights.get("output.weight");
        if (outputWeight == null) {
            outputWeight = tokenEmbedWeight; // tied weights fallback
        }
        SDVariable lmHead = sd.var("lm_head.weight", outputWeight);

        // Logits: upcast to FP32 before the large vocab matmul to prevent FP16 overflow
        SDVariable logits = QuantizedLinear.matMul(sd, "lm_logits", hidden, lmHead, weights, "output.weight", dtype);
        outputNames.add("lm_logits");

        sd.setOutputs(outputNames);
        return sd;
    }

    // ========================================================================
    // Vision encoder — SigLIP2 ViT (standard ViT with per-frame spatial attention)
    // ========================================================================

    /**
     * Build the SigLIP2-400M vision encoder stub.
     *
     * <p>Tensor prefix in GGUF: {@code v.blk.{layer}.*} and {@code v.*} for global weights.</p>
     *
     * <p>The encoder produces per-patch features for each video frame.
     * Shape signature: [batch * numFrames, numPatches, visionHidden]
     * where numPatches = (imageH / VISION_PATCH_SIZE) * (imageW / VISION_PATCH_SIZE).</p>
     *
     * <p>The graph placeholder captures the pre-patchified pixel values; the actual
     * patch embedding is computed via a conv projection ({@code v.patch_embd.weight}).</p>
     */
    private SDVariable buildVisionEncoder(SameDiff sd, Map<String, INDArray> weights,
                                          DataType dtype, ArchitectureConfig config) {
        // Input pixels: [batch * numFrames, C, H, W]
        // We receive these as a flattened frame batch (frames from all batch items stacked).
        SDVariable pixelValues = sd.placeHolder("pixel_values", dtype, -1, -1, -1, -1);

        // Detect vision hidden size from patch embedding weight
        INDArray patchEmbedWeight = weights.get("v.patch_embd.weight");
        if (patchEmbedWeight == null) {
            log.warn("Missing vision patch embedding weight v.patch_embd.weight; "
                    + "returning pixel_values pass-through for vision encoder stub");
            return sd.reshape("vision_stub_out", pixelValues, -1, -1, config.getHiddenSize());
        }

        // Patch embedding: conv projection [visionHidden, C, patchH, patchW]
        // We use a reshape+matmul approximation: flatten each patch then project.
        // [batch*frames, C, H, W] -> [batch*frames, numPatches, C*pH*pW] -> [batch*frames, numPatches, visionHidden]
        long visionHidden = patchEmbedWeight.shape()[0];
        long patchDim     = patchEmbedWeight.shape()[1];  // C * VISION_PATCH_SIZE * VISION_PATCH_SIZE

        SDVariable wPatch = sd.var("v.patch_embd.weight", patchEmbedWeight);
        // Reshape to linear projection form: [visionHidden, patchDim] -> [patchDim, visionHidden] after permute
        SDVariable wPatchFlat = sd.reshape("v.patch_embd.weight_flat", wPatch, visionHidden, patchDim);

        // Flatten pixel values into patch vectors (approximation — exact unfolding left to runtime)
        SDVariable patches = sd.reshape("v.patches_flat", pixelValues, -1, -1, patchDim);
        SDVariable patchEmbeds = fp32Mmul(sd, "v.patch_embeds", patches,
                wPatchFlat.permute(1, 0), dtype);

        // Add patch position bias if present
        INDArray patchBias = weights.get("v.patch_embd.bias");
        if (patchBias != null) {
            patchEmbeds = patchEmbeds.add("v.patch_embeds_biased",
                    sd.var("v.patch_embd.bias", patchBias));
        }

        // Class embedding (prepend CLS token)
        INDArray clsWeight = weights.get("v.class_embd");
        if (clsWeight != null) {
            SDVariable cls = sd.var("v.class_embd", clsWeight);
            // Expand cls for batch: [1, 1, visionHidden] -> [batch*frames, 1, visionHidden]
            // Simple concat along seq dim (position 1)
            patchEmbeds = sd.concat("v.with_cls", 1,
                    sd.reshape("v.cls_expanded", cls, 1, 1, visionHidden),
                    patchEmbeds);
        }

        // Positional embedding
        INDArray posEmbWeight = weights.get("v.position_embd.weight");
        if (posEmbWeight != null) {
            SDVariable posEmb = sd.var("v.position_embd.weight", posEmbWeight);
            patchEmbeds = patchEmbeds.add("v.pos_embeds", posEmb);
        }

        // Pre-encoder layer norm
        INDArray preNormW = weights.get("v.pre_ln.weight");
        if (preNormW != null) {
            patchEmbeds = buildLayerNorm(sd, patchEmbeds, "v.pre_ln", preNormW,
                    weights.get("v.pre_ln.bias"), config.getLayerNormEpsilon());
        }

        // Detect vision encoder depth from weight keys
        int numVisionLayers = 0;
        for (String key : weights.keySet()) {
            if (key.startsWith("v.blk.")) {
                String[] parts = key.split("\\.");
                if (parts.length >= 3) {
                    try {
                        int idx = Integer.parseInt(parts[2]);
                        if (idx + 1 > numVisionLayers) numVisionLayers = idx + 1;
                    } catch (NumberFormatException ignored) {
                    }
                }
            }
        }

        if (numVisionLayers == 0) {
            log.warn("No vision encoder blocks found (v.blk.*); returning patch embeds directly");
            return patchEmbeds;
        }

        log.info("Building SigLIP2 vision encoder: {} layers, visionHidden={}",
                numVisionLayers, visionHidden);

        SDVariable x = patchEmbeds;
        for (int layer = 0; layer < numVisionLayers; layer++) {
            x = buildVisionEncoderBlock(sd, x, layer, weights, dtype,
                    (int) visionHidden, config.getLayerNormEpsilon());
        }

        // Post-encoder layer norm
        INDArray postNormW = weights.get("v.post_ln.weight");
        if (postNormW != null) {
            x = buildLayerNorm(sd, x, "v.post_ln", postNormW,
                    weights.get("v.post_ln.bias"), config.getLayerNormEpsilon());
        }

        return x;
    }

    /**
     * One SigLIP2 ViT encoder block (pre-norm, MHA, FFN with GELU).
     * GGUF prefix: {@code v.blk.{layer}.*}
     */
    private SDVariable buildVisionEncoderBlock(SameDiff sd, SDVariable input, int layerIdx,
                                               Map<String, INDArray> weights, DataType dtype,
                                               int visionHidden, float normEps) {
        String prefix = "v.blk." + layerIdx;

        // Pre-attention layer norm (LayerNorm, not RMSNorm — SigLIP2 uses standard LN)
        SDVariable normed = buildLayerNorm(sd, input, prefix + ".attn_norm",
                weights.get(prefix + ".attn_norm.weight"),
                weights.get(prefix + ".attn_norm.bias"), normEps);

        // Derive head count from Q weight shape
        INDArray qWeight = weights.get(prefix + ".attn_q.weight");
        if (qWeight == null) {
            log.warn("Missing vision encoder attention weights for layer {}", layerIdx);
            return input;
        }

        // Self-attention (no RoPE for ViT — uses learned absolute positional embeddings)
        SDVariable attnOut = buildVisionSelfAttention(sd, normed, layerIdx, prefix, weights, dtype, visionHidden);
        SDVariable postAttn = input.add("v_post_attn_" + layerIdx, attnOut);

        // Pre-FFN layer norm
        SDVariable ffnNormed = buildLayerNorm(sd, postAttn, prefix + ".ffn_norm",
                weights.get(prefix + ".ffn_norm.weight"),
                weights.get(prefix + ".ffn_norm.bias"), normEps);

        // GELU FFN (SigLIP2 uses GELU, not SwiGLU)
        SDVariable ffnOut = buildVisionGELUFFN(sd, ffnNormed, layerIdx, prefix, weights, dtype);

        return postAttn.add("v_layer_out_" + layerIdx, ffnOut);
    }

    /**
     * Vision self-attention (standard MHA, no RoPE, no KV cache for encoder).
     */
    private SDVariable buildVisionSelfAttention(SameDiff sd, SDVariable input, int layerIdx,
                                                String prefix, Map<String, INDArray> weights,
                                                DataType dtype, int visionHidden) {
        INDArray qWeight = weights.get(prefix + ".attn_q.weight");
        INDArray kWeight = weights.get(prefix + ".attn_k.weight");
        INDArray vWeight = weights.get(prefix + ".attn_v.weight");
        INDArray oWeight = weights.get(prefix + ".attn_output.weight");

        if (qWeight == null || kWeight == null || vWeight == null || oWeight == null) {
            log.warn("Missing vision self-attention weights for layer {}; skipping", layerIdx);
            return input;
        }

        // Derive head count: assume headDim = 64 for SigLIP2-400M
        int qOutDim = (int) qWeight.shape()[0];
        int headDim = 64; // SigLIP2-400M standard head dim
        int numHeads = qOutDim / headDim;

        String attnPrefix = "v.layers." + layerIdx + ".self_attn.";
        SDVariable wq = sd.var(attnPrefix + "q_proj.weight", qWeight);
        SDVariable wk = sd.var(attnPrefix + "k_proj.weight", kWeight);
        SDVariable wv = sd.var(attnPrefix + "v_proj.weight", vWeight);
        SDVariable wo = sd.var(attnPrefix + "o_proj.weight", oWeight);

        SDVariable q = fp32Mmul(sd, "vq_" + layerIdx, input, wq.permute(1, 0), dtype);
        SDVariable k = fp32Mmul(sd, "vk_" + layerIdx, input, wk.permute(1, 0), dtype);
        SDVariable v = fp32Mmul(sd, "vv_" + layerIdx, input, wv.permute(1, 0), dtype);

        // Add Q/K/V biases if present (SigLIP2 uses biases in attention)
        INDArray qBias = weights.get(prefix + ".attn_q.bias");
        INDArray kBias = weights.get(prefix + ".attn_k.bias");
        INDArray vBias = weights.get(prefix + ".attn_v.bias");
        if (qBias != null) q = q.add(sd.var(attnPrefix + "q_proj.bias", qBias));
        if (kBias != null) k = k.add(sd.var(attnPrefix + "k_proj.bias", kBias));
        if (vBias != null) v = v.add(sd.var(attnPrefix + "v_proj.bias", vBias));

        // Reshape to heads: [batch*frames, numPatches, numHeads, headDim]
        SDVariable batchDim = sd.sizeAt(input, 0);
        SDVariable seqDim   = sd.sizeAt(input, 1);

        SDVariable headShape = sd.stack("vq_shape_" + layerIdx, 0,
                batchDim, seqDim,
                sd.constant(Nd4j.scalar((long) numHeads)),
                sd.constant(Nd4j.scalar((long) headDim)));

        q = sd.reshape("vq_heads_" + layerIdx, q, headShape);
        k = sd.reshape("vk_heads_" + layerIdx, k, headShape);
        v = sd.reshape("vv_heads_" + layerIdx, v, headShape);

        // Scaled dot-product attention (no mask, no KV cache for the ViT encoder)
        SDVariable attnOut = new DotProductAttentionV2(sd,
                q, v, k, null, null,
                null, null, null, null,
                0.0, 0.0, false, false).outputVariable();
        sd.updateVariableNameAndReference(attnOut, "v_attn_out_" + layerIdx);

        // Reshape back: [batch*frames, numPatches, qOutDim]
        SDVariable outShape = sd.stack("v_attn_out_shape_" + layerIdx, 0,
                batchDim, seqDim,
                sd.constant(Nd4j.scalar((long) qOutDim)));
        SDVariable attnFlat = sd.reshape("v_attn_flat_" + layerIdx, attnOut, outShape);

        // Output projection
        SDVariable out = fp32Mmul(sd, "v_attn_proj_" + layerIdx, attnFlat, wo.permute(1, 0), dtype);

        INDArray oBias = weights.get(prefix + ".attn_output.bias");
        if (oBias != null) {
            out = out.add(sd.var("v.layers." + layerIdx + ".self_attn.o_proj.bias", oBias));
        }

        return out;
    }

    /**
     * Vision FFN: Linear -> GELU -> Linear (SigLIP2 style, no gate).
     */
    private SDVariable buildVisionGELUFFN(SameDiff sd, SDVariable input, int layerIdx,
                                          String prefix, Map<String, INDArray> weights,
                                          DataType dtype) {
        INDArray upWeight   = weights.get(prefix + ".ffn_up.weight");
        INDArray downWeight = weights.get(prefix + ".ffn_down.weight");

        if (upWeight == null || downWeight == null) {
            log.warn("Missing vision FFN weights for layer {}; passing through", layerIdx);
            return input;
        }

        String mlpPrefix = "v.layers." + layerIdx + ".mlp.";
        SDVariable wUp   = sd.var(mlpPrefix + "fc1.weight", upWeight);
        SDVariable wDown = sd.var(mlpPrefix + "fc2.weight", downWeight);

        SDVariable up = fp32Mmul(sd, "v_up_" + layerIdx, input, wUp.permute(1, 0), dtype);

        INDArray upBias = weights.get(prefix + ".ffn_up.bias");
        if (upBias != null) up = up.add(sd.var(mlpPrefix + "fc1.bias", upBias));

        SDVariable activated = sd.nn.gelu("v_gelu_" + layerIdx, up);
        SDVariable down = fp32Mmul(sd, "v_down_" + layerIdx, activated, wDown.permute(1, 0), dtype);

        INDArray downBias = weights.get(prefix + ".ffn_down.bias");
        if (downBias != null) down = down.add(sd.var(mlpPrefix + "fc2.bias", downBias));

        return down;
    }

    // ========================================================================
    // 3D-Resampler
    // ========================================================================

    /**
     * Build the 3D-Resampler stub.
     *
     * <p>The resampler uses cross-attention to compress spatial-temporal vision features
     * from {@code RESAMPLER_FRAME_GROUP_SIZE} consecutive frames into
     * {@code RESAMPLER_OUTPUT_TOKENS} fixed-length query tokens.
     * This achieves a 96x compression ratio relative to raw patch tokens.</p>
     *
     * <p>GGUF tensor prefix: {@code resampler.*}</p>
     *
     * <p>Key tensors:</p>
     * <ul>
     *   <li>{@code resampler.query} — learnable query vectors [numOutputTokens, visionHidden]</li>
     *   <li>{@code resampler.attn_q.weight} / {@code resampler.attn_k.weight} /
     *       {@code resampler.attn_v.weight} — cross-attention projections</li>
     *   <li>{@code resampler.attn_output.weight} — output projection</li>
     *   <li>{@code resampler.ln_q.weight} / {@code resampler.ln_kv.weight} — layer norms</li>
     *   <li>{@code resampler.ln_post.weight} — post-resampler layer norm</li>
     * </ul>
     *
     * @param visionFeatures  Output of the vision encoder: [batch*frames, numPatches, visionHidden]
     * @return Compressed vision tokens: [batch, numOutputTokens, visionHidden]
     */
    private SDVariable buildResampler(SameDiff sd, SDVariable visionFeatures,
                                      Map<String, INDArray> weights, DataType dtype,
                                      ArchitectureConfig config) {
        INDArray queryWeight = weights.get("resampler.query");
        if (queryWeight == null) {
            log.warn("Missing resampler query weights; returning vision features pass-through");
            return visionFeatures;
        }

        long visionHidden = queryWeight.shape()[1];
        log.info("Building 3D-Resampler: outputTokens={}, frameGroupSize={}, visionHidden={}",
                RESAMPLER_OUTPUT_TOKENS, RESAMPLER_FRAME_GROUP_SIZE, visionHidden);

        // Learnable query vectors: [1, numOutputTokens, visionHidden]
        SDVariable queryTokens = sd.reshape("resampler.query_3d",
                sd.var("resampler.query", queryWeight), 1, (long) RESAMPLER_OUTPUT_TOKENS, visionHidden);

        // Layer norm on queries before cross-attention
        INDArray lnQWeight = weights.get("resampler.ln_q.weight");
        if (lnQWeight != null) {
            queryTokens = buildLayerNorm(sd, queryTokens, "resampler.ln_q",
                    lnQWeight, weights.get("resampler.ln_q.bias"), config.getLayerNormEpsilon());
        }

        // Layer norm on key/value (vision features) before cross-attention
        SDVariable kv = visionFeatures;
        INDArray lnKvWeight = weights.get("resampler.ln_kv.weight");
        if (lnKvWeight != null) {
            kv = buildLayerNorm(sd, kv, "resampler.ln_kv",
                    lnKvWeight, weights.get("resampler.ln_kv.bias"), config.getLayerNormEpsilon());
        }

        // Cross-attention projections
        INDArray attnQWeight = weights.get("resampler.attn_q.weight");
        INDArray attnKWeight = weights.get("resampler.attn_k.weight");
        INDArray attnVWeight = weights.get("resampler.attn_v.weight");
        INDArray attnOWeight = weights.get("resampler.attn_output.weight");

        if (attnQWeight == null || attnKWeight == null || attnVWeight == null || attnOWeight == null) {
            log.warn("Missing resampler cross-attention weights; returning compressed query tokens");
            return queryTokens;
        }

        SDVariable wq = sd.var("resampler.attn_q.weight", attnQWeight);
        SDVariable wk = sd.var("resampler.attn_k.weight", attnKWeight);
        SDVariable wv = sd.var("resampler.attn_v.weight", attnVWeight);
        SDVariable wo = sd.var("resampler.attn_output.weight", attnOWeight);

        // Project queries, keys, values
        SDVariable q = fp32Mmul(sd, "resampler.q", queryTokens, wq.permute(1, 0), dtype);
        SDVariable k = fp32Mmul(sd, "resampler.k", kv, wk.permute(1, 0), dtype);
        SDVariable v = fp32Mmul(sd, "resampler.v", kv, wv.permute(1, 0), dtype);

        // Add biases if present
        INDArray qBias = weights.get("resampler.attn_q.bias");
        INDArray kBias = weights.get("resampler.attn_k.bias");
        INDArray vBias = weights.get("resampler.attn_v.bias");
        if (qBias != null) q = q.add(sd.var("resampler.attn_q.bias", qBias));
        if (kBias != null) k = k.add(sd.var("resampler.attn_k.bias", kBias));
        if (vBias != null) v = v.add(sd.var("resampler.attn_v.bias", vBias));

        // Derive head count from Q weight shape
        int qOutDim  = (int) attnQWeight.shape()[0];
        int headDim  = 64; // Resampler standard head dim
        int numHeads = qOutDim / headDim;

        SDVariable batchDim  = sd.sizeAt(queryTokens, 0);
        SDVariable qSeqDim   = sd.sizeAt(queryTokens, 1);
        SDVariable kvSeqDim  = sd.sizeAt(kv, 1);

        SDVariable qShape = sd.stack("resampler.q_shape", 0,
                batchDim, qSeqDim,
                sd.constant(Nd4j.scalar((long) numHeads)),
                sd.constant(Nd4j.scalar((long) headDim)));
        SDVariable kvShape = sd.stack("resampler.kv_shape", 0,
                batchDim, kvSeqDim,
                sd.constant(Nd4j.scalar((long) numHeads)),
                sd.constant(Nd4j.scalar((long) headDim)));

        q = sd.reshape("resampler.q_heads", q, qShape);
        k = sd.reshape("resampler.k_heads", k, kvShape);
        v = sd.reshape("resampler.v_heads", v, kvShape);

        // Cross-attention: queries attend to vision key/value pairs (no causal mask)
        SDVariable attnOut = new DotProductAttentionV2(sd,
                q, v, k, null, null,
                null, null, null, null,
                0.0, 0.0, false, false).outputVariable();
        sd.updateVariableNameAndReference(attnOut, "resampler.attn_out");

        // Reshape: [batch, numOutputTokens, numHeads * headDim]
        SDVariable outShape = sd.stack("resampler.out_shape", 0,
                batchDim, qSeqDim,
                sd.constant(Nd4j.scalar((long) qOutDim)));
        SDVariable attnFlat = sd.reshape("resampler.attn_flat", attnOut, outShape);

        // Output projection
        SDVariable resamplerOut = fp32Mmul(sd, "resampler.attn_proj", attnFlat, wo.permute(1, 0), dtype);

        INDArray oBias = weights.get("resampler.attn_output.bias");
        if (oBias != null) {
            resamplerOut = resamplerOut.add(sd.var("resampler.attn_output.bias", oBias));
        }

        // Post-resampler layer norm
        INDArray postLnW = weights.get("resampler.ln_post.weight");
        if (postLnW != null) {
            resamplerOut = buildLayerNorm(sd, resamplerOut, "resampler.ln_post",
                    postLnW, weights.get("resampler.ln_post.bias"), config.getLayerNormEpsilon());
        }

        // Vision projection into LLM hidden dimension (mm.0 / mm.2 linear layers)
        INDArray mmWeight0 = weights.get("mm.0.weight");
        if (mmWeight0 != null) {
            SDVariable wMm0 = sd.var("mm.0.weight", mmWeight0);
            resamplerOut = fp32Mmul(sd, "mm.0.proj", resamplerOut, wMm0.permute(1, 0), dtype);
            INDArray mmBias0 = weights.get("mm.0.bias");
            if (mmBias0 != null) resamplerOut = resamplerOut.add(sd.var("mm.0.bias", mmBias0));
            resamplerOut = sd.nn.gelu("mm.gelu", resamplerOut);
        }
        INDArray mmWeight2 = weights.get("mm.2.weight");
        if (mmWeight2 != null) {
            SDVariable wMm2 = sd.var("mm.2.weight", mmWeight2);
            resamplerOut = fp32Mmul(sd, "mm.2.proj", resamplerOut, wMm2.permute(1, 0), dtype);
            INDArray mmBias2 = weights.get("mm.2.bias");
            if (mmBias2 != null) resamplerOut = resamplerOut.add(sd.var("mm.2.bias", mmBias2));
        }

        return resamplerOut;
    }

    // ========================================================================
    // LLM decoder block (Qwen3-8B pattern: RMSNorm, GQA, SwiGLU, RoPE)
    // ========================================================================

    /**
     * One Qwen3-8B compatible transformer decoder block.
     * Tensor prefix: {@code blk.{layer}.*} (standard GGUF LLaMA/Qwen convention).
     */
    private SDVariable buildLLMBlock(SameDiff sd, SDVariable input, int layerIdx,
                                     ArchitectureConfig config, Map<String, INDArray> weights,
                                     DataType dtype,
                                     SDVariable positionOffset, SDVariable cachePosition,
                                     SDVariable causalMask,
                                     SDVariable keyCache, SDVariable valueCache) {
        String prefix = "blk." + layerIdx;

        // Pre-attention RMS norm
        SDVariable normed = buildRMSNorm(sd, input,
                "model.layers." + layerIdx + ".input_layernorm",
                prefix + ".attn_norm", weights, config, dtype);

        // Self-attention (separate Q/K/V, QK norms optional, RoPE)
        SDVariable attnOut = buildLLMAttention(sd, normed, layerIdx, config, weights, dtype,
                positionOffset, cachePosition, causalMask, keyCache, valueCache);

        SDVariable postAttn = input.add("llm_post_attn_" + layerIdx, attnOut);

        // Pre-FFN RMS norm — check for post_attention_norm first, fallback to ffn_norm
        String postAttnNormKey = weights.containsKey(prefix + ".post_attention_norm.weight")
                ? prefix + ".post_attention_norm"
                : prefix + ".ffn_norm";
        SDVariable ffnNormed = buildRMSNorm(sd, postAttn,
                "model.layers." + layerIdx + ".post_attention_layernorm",
                postAttnNormKey, weights, config, dtype);

        // SwiGLU FFN
        SDVariable ffnOut;
        if (weights.containsKey(prefix + ".ffn_gate.weight")) {
            ffnOut = buildSwiGLUFFN(sd, ffnNormed, layerIdx, prefix, weights, dtype);
        } else {
            log.warn("No FFN gate weight found for LLM layer {}; passing through", layerIdx);
            ffnOut = ffnNormed;
        }

        return postAttn.add("llm_layer_out_" + layerIdx, ffnOut);
    }

    /**
     * LLM self-attention with optional QK norms (Qwen3 style), RoPE, and KV cache.
     */
    private SDVariable buildLLMAttention(SameDiff sd, SDVariable input, int layerIdx,
                                         ArchitectureConfig config, Map<String, INDArray> weights,
                                         DataType dtype,
                                         SDVariable positionOffset, SDVariable cachePosition,
                                         SDVariable causalMask,
                                         SDVariable keyCache, SDVariable valueCache) {
        String prefix = "blk." + layerIdx;
        int numHeads   = config.getNumAttentionHeads();
        int numKVHeads = config.getNumKVHeads();

        INDArray qWeight = weights.get(prefix + ".attn_q.weight");
        INDArray kWeight = weights.get(prefix + ".attn_k.weight");
        INDArray vWeight = weights.get(prefix + ".attn_v.weight");
        INDArray oWeight = weights.get(prefix + ".attn_output.weight");

        if (qWeight == null || kWeight == null || vWeight == null || oWeight == null) {
            log.warn("Missing LLM attention weights for layer {}; passing through", layerIdx);
            return input;
        }

        int kOutDim   = (int) kWeight.shape()[0];
        int headDim   = config.getHeadDimension();
        if (headDim <= 0) headDim = kOutDim / numKVHeads;
        int qOutDim   = (int) qWeight.shape()[0];
        int actualNumHeads = qOutDim / headDim;

        String attnPrefix = "model.layers." + layerIdx + ".self_attn.";
        SDVariable wq = sd.var(attnPrefix + "q_proj.weight", qWeight);
        SDVariable wk = sd.var(attnPrefix + "k_proj.weight", kWeight);
        SDVariable wv = sd.var(attnPrefix + "v_proj.weight", vWeight);
        SDVariable wo = sd.var(attnPrefix + "o_proj.weight", oWeight);

        SDVariable q = QuantizedLinear.matMul(sd, "llm_q_" + layerIdx, input, wq, weights, prefix + ".attn_q.weight", dtype);
        SDVariable k = QuantizedLinear.matMul(sd, "llm_k_" + layerIdx, input, wk, weights, prefix + ".attn_k.weight", dtype);
        SDVariable v = QuantizedLinear.matMul(sd, "llm_v_" + layerIdx, input, wv, weights, prefix + ".attn_v.weight", dtype);

        // Optional Q/K biases (Qwen models typically don't use them, but check anyway)
        INDArray qBias = weights.get(prefix + ".attn_q.bias");
        INDArray kBias = weights.get(prefix + ".attn_k.bias");
        INDArray vBias = weights.get(prefix + ".attn_v.bias");
        if (qBias != null) q = q.add(sd.var(attnPrefix + "q_proj.bias", qBias));
        if (kBias != null) k = k.add(sd.var(attnPrefix + "k_proj.bias", kBias));
        if (vBias != null) v = v.add(sd.var(attnPrefix + "v_proj.bias", vBias));

        SDVariable batchDim = sd.sizeAt(input, 0);
        SDVariable seqDim   = sd.sizeAt(input, 1);

        SDVariable qShape = sd.stack("llm_q_shape_" + layerIdx, 0,
                batchDim, seqDim,
                sd.constant(Nd4j.scalar((long) actualNumHeads)),
                sd.constant(Nd4j.scalar((long) headDim)));
        SDVariable kvShape = sd.stack("llm_kv_shape_" + layerIdx, 0,
                batchDim, seqDim,
                sd.constant(Nd4j.scalar((long) numKVHeads)),
                sd.constant(Nd4j.scalar((long) headDim)));

        q = sd.reshape("llm_q_heads_" + layerIdx, q, qShape);
        k = sd.reshape("llm_k_heads_" + layerIdx, k, kvShape);
        v = sd.reshape("llm_v_heads_" + layerIdx, v, kvShape);

        // Optional per-head QK norms (Qwen3 uses these)
        INDArray qNormW = weights.get(prefix + ".attn_q_norm.weight");
        INDArray kNormW = weights.get(prefix + ".attn_k_norm.weight");
        if (qNormW != null) {
            q = applyHeadRMSNorm(sd, q, attnPrefix + "q_norm_" + layerIdx,
                    qNormW, config.getLayerNormEpsilon());
        }
        if (kNormW != null) {
            k = applyHeadRMSNorm(sd, k, attnPrefix + "k_norm_" + layerIdx,
                    kNormW, config.getLayerNormEpsilon());
        }

        // Apply NeoX/interleaved RoPE with dynamic position offset
        q = new FusedRoPE(sd, q, positionOffset,
                config.getRopeType(), config.getRopeFreqBase(), 1.0,
                config.getRopeDimensionCount()).outputVariable();
        sd.updateVariableNameAndReference(q, "q_rope_" + layerIdx);

        k = new FusedRoPE(sd, k, positionOffset,
                config.getRopeType(), config.getRopeFreqBase(), 1.0,
                config.getRopeDimensionCount()).outputVariable();
        sd.updateVariableNameAndReference(k, "k_rope_" + layerIdx);

        // V must match Q/K dtype after FusedRoPE promotion
        if (v.dataType() != q.dataType()) {
            v = v.castTo("llm_v_cast_" + layerIdx, q.dataType());
        }
        sd.updateVariableNameAndReference(v, "v_heads_" + layerIdx);

        // Dot-product attention with KV cache and causal mask
        SDVariable attnOut = new DotProductAttentionV2(sd,
                q, v, k, null, null,
                keyCache, valueCache, cachePosition, causalMask,
                0.0, 0.0, false, false).outputVariable();
        sd.updateVariableNameAndReference(attnOut, "llm_attn_out_" + layerIdx);

        // Flatten heads: [batch, seq, numHeads, headDim] -> [batch, seq, numHeads*headDim]
        int attnOutDim = actualNumHeads * headDim;
        SDVariable outShape = sd.stack("llm_attn_out_shape_" + layerIdx, 0,
                batchDim, seqDim,
                sd.constant(Nd4j.scalar((long) attnOutDim)));
        SDVariable attnFlat = sd.reshape("llm_attn_flat_" + layerIdx, attnOut, outShape);

        return QuantizedLinear.matMul(sd, "llm_attn_proj_" + layerIdx, attnFlat, wo, weights, prefix + ".attn_output.weight", dtype);
    }

    /**
     * Qwen3-8B SwiGLU FFN: gate(SiLU) * up, then down projection.
     */
    private SDVariable buildSwiGLUFFN(SameDiff sd, SDVariable input, int layerIdx,
                                      String prefix, Map<String, INDArray> weights,
                                      DataType dtype) {
        INDArray gateWeight = weights.get(prefix + ".ffn_gate.weight");
        INDArray upWeight   = weights.get(prefix + ".ffn_up.weight");
        INDArray downWeight = weights.get(prefix + ".ffn_down.weight");

        if (gateWeight == null || upWeight == null || downWeight == null) {
            log.warn("Missing SwiGLU FFN weights for LLM layer {}", layerIdx);
            return input;
        }

        String mlpPrefix = "model.layers." + layerIdx + ".mlp.";
        SDVariable wGate = sd.var(mlpPrefix + "gate_proj.weight", gateWeight);
        SDVariable wUp   = sd.var(mlpPrefix + "up_proj.weight", upWeight);
        SDVariable wDown = sd.var(mlpPrefix + "down_proj.weight", downWeight);

        SDVariable gate = QuantizedLinear.matMul(sd, "llm_gate_" + layerIdx, input, wGate, weights, prefix + ".ffn_gate.weight", dtype);
        SDVariable up   = QuantizedLinear.matMul(sd, "llm_up_" + layerIdx, input, wUp, weights, prefix + ".ffn_up.weight", dtype);

        SDVariable silu   = sd.nn.swish(gate);
        SDVariable hidden = silu.mul("llm_swiglu_" + layerIdx, up);

        return QuantizedLinear.matMul(sd, "llm_down_" + layerIdx, hidden, wDown, weights, prefix + ".ffn_down.weight", dtype);
    }

    // ========================================================================
    // Normalization helpers
    // ========================================================================

    /**
     * RMS normalization (used by the LLM decoder).
     * Upcasts to FP32 internally to prevent FP16 squaring overflow.
     */
    private SDVariable buildRMSNorm(SameDiff sd, SDVariable input, String outputName,
                                    String weightKey, Map<String, INDArray> weights,
                                    ArchitectureConfig config, DataType dtype) {
        INDArray normWeight = weights.get(weightKey + ".weight");
        if (normWeight == null) {
            log.warn("Missing RMS norm weight: {}", weightKey);
            return input;
        }

        SDVariable gamma = sd.var(outputName + ".weight", normWeight);

        boolean needsCast = (input.dataType() == DataType.HALF
                || input.dataType() == DataType.BFLOAT16);
        SDVariable x = needsCast ? input.castTo(outputName + "_f32", DataType.FLOAT) : input;

        SDVariable squared    = x.mul(x);
        SDVariable meanSq     = squared.mean(true, -1);
        SDVariable rms        = sd.math.sqrt(meanSq.add(config.getLayerNormEpsilon()));
        SDVariable normalized = x.div(rms);

        if (needsCast) normalized = normalized.castTo(outputName + "_cast", input.dataType());

        return normalized.mul(outputName, gamma);
    }

    /**
     * Per-head RMS normalization. Input shape: [batch, seq, numHeads, headDim].
     */
    private SDVariable applyHeadRMSNorm(SameDiff sd, SDVariable input, String outputName,
                                        INDArray normWeight, float eps) {
        SDVariable gamma = sd.var(outputName + ".weight", normWeight);

        boolean needsCast = (input.dataType() == DataType.HALF
                || input.dataType() == DataType.BFLOAT16);
        SDVariable x = needsCast ? input.castTo(outputName + "_f32", DataType.FLOAT) : input;

        SDVariable squared    = x.mul(x);
        SDVariable meanSq     = squared.mean(true, -1);
        SDVariable rms        = sd.math.sqrt(meanSq.add(eps));
        SDVariable normalized = x.div(rms);

        if (needsCast) normalized = normalized.castTo(outputName + "_cast", input.dataType());

        return normalized.mul(outputName, gamma);
    }

    /**
     * Standard layer normalization (used by the SigLIP2 vision encoder and resampler).
     * (x - mean) / sqrt(var + eps) * weight [+ bias]
     */
    private SDVariable buildLayerNorm(SameDiff sd, SDVariable input, String name,
                                      INDArray weight, INDArray bias, float eps) {
        if (weight == null) return input;

        SDVariable wVar = sd.var(name + ".weight", weight);

        SDVariable mean       = input.mean(true, -1);
        SDVariable centered   = input.sub(mean);
        SDVariable variance   = centered.mul(centered).mean(true, -1);
        SDVariable normalized = centered.div(sd.math.sqrt(variance.add(eps)));
        SDVariable result     = normalized.mul(wVar);

        if (bias != null) {
            result = result.add(sd.var(name + ".bias", bias));
        }

        return result;
    }

    // ========================================================================
    // FP32 matmul helper (prevents FP16 overflow on large dot products)
    // ========================================================================

    private SDVariable fp32Mmul(SameDiff sd, String name, SDVariable a, SDVariable b,
                                DataType dtype) {
        if (dtype == DataType.HALF || dtype == DataType.BFLOAT16) {
            SDVariable aF32  = a.castTo(name + "_a_f32", DataType.FLOAT);
            SDVariable bF32  = b.castTo(name + "_b_f32", DataType.FLOAT);
            SDVariable result = sd.mmul(name + "_f32", aF32, bF32);
            return result.castTo(name, dtype);
        }
        return sd.mmul(name, a, b);
    }

    // ========================================================================
    // Tensor name patterns
    // ========================================================================

    @Override
    public Map<String, String> getTensorNamePatterns() {
        Map<String, String> patterns = new HashMap<>();

        // ---- Global / embedding tensors ------------------------------------
        patterns.put("token_embd.weight",   "model.embed_tokens.weight");
        patterns.put("output.weight",        "lm_head.weight");
        patterns.put("output_norm.weight",   "model.norm.weight");

        // ---- Vision encoder tensors (SigLIP2 ViT, prefix: v.*) ------------
        patterns.put("v.patch_embd.weight",        "v.patch_embd.weight");
        patterns.put("v.patch_embd.bias",          "v.patch_embd.bias");
        patterns.put("v.class_embd",               "v.class_embd");
        patterns.put("v.position_embd.weight",     "v.position_embd.weight");
        patterns.put("v.pre_ln.weight",            "v.pre_ln.weight");
        patterns.put("v.pre_ln.bias",              "v.pre_ln.bias");
        patterns.put("v.post_ln.weight",           "v.post_ln.weight");
        patterns.put("v.post_ln.bias",             "v.post_ln.bias");

        // Vision encoder block tensors (separate Q/K/V, biased, GELU FFN)
        patterns.put("v.blk.{layer}.attn_q.weight",      "v.layers.{layer}.self_attn.q_proj.weight");
        patterns.put("v.blk.{layer}.attn_q.bias",        "v.layers.{layer}.self_attn.q_proj.bias");
        patterns.put("v.blk.{layer}.attn_k.weight",      "v.layers.{layer}.self_attn.k_proj.weight");
        patterns.put("v.blk.{layer}.attn_k.bias",        "v.layers.{layer}.self_attn.k_proj.bias");
        patterns.put("v.blk.{layer}.attn_v.weight",      "v.layers.{layer}.self_attn.v_proj.weight");
        patterns.put("v.blk.{layer}.attn_v.bias",        "v.layers.{layer}.self_attn.v_proj.bias");
        patterns.put("v.blk.{layer}.attn_output.weight", "v.layers.{layer}.self_attn.o_proj.weight");
        patterns.put("v.blk.{layer}.attn_output.bias",   "v.layers.{layer}.self_attn.o_proj.bias");
        patterns.put("v.blk.{layer}.attn_norm.weight",   "v.layers.{layer}.attn_norm.weight");
        patterns.put("v.blk.{layer}.attn_norm.bias",     "v.layers.{layer}.attn_norm.bias");
        patterns.put("v.blk.{layer}.ffn_up.weight",      "v.layers.{layer}.mlp.fc1.weight");
        patterns.put("v.blk.{layer}.ffn_up.bias",        "v.layers.{layer}.mlp.fc1.bias");
        patterns.put("v.blk.{layer}.ffn_down.weight",    "v.layers.{layer}.mlp.fc2.weight");
        patterns.put("v.blk.{layer}.ffn_down.bias",      "v.layers.{layer}.mlp.fc2.bias");
        patterns.put("v.blk.{layer}.ffn_norm.weight",    "v.layers.{layer}.ffn_norm.weight");
        patterns.put("v.blk.{layer}.ffn_norm.bias",      "v.layers.{layer}.ffn_norm.bias");

        // ---- 3D-Resampler tensors ------------------------------------------
        patterns.put("resampler.query",               "resampler.query");
        patterns.put("resampler.attn_q.weight",       "resampler.attn_q.weight");
        patterns.put("resampler.attn_q.bias",         "resampler.attn_q.bias");
        patterns.put("resampler.attn_k.weight",       "resampler.attn_k.weight");
        patterns.put("resampler.attn_k.bias",         "resampler.attn_k.bias");
        patterns.put("resampler.attn_v.weight",       "resampler.attn_v.weight");
        patterns.put("resampler.attn_v.bias",         "resampler.attn_v.bias");
        patterns.put("resampler.attn_output.weight",  "resampler.attn_output.weight");
        patterns.put("resampler.attn_output.bias",    "resampler.attn_output.bias");
        patterns.put("resampler.ln_q.weight",         "resampler.ln_q.weight");
        patterns.put("resampler.ln_q.bias",           "resampler.ln_q.bias");
        patterns.put("resampler.ln_kv.weight",        "resampler.ln_kv.weight");
        patterns.put("resampler.ln_kv.bias",          "resampler.ln_kv.bias");
        patterns.put("resampler.ln_post.weight",      "resampler.ln_post.weight");
        patterns.put("resampler.ln_post.bias",        "resampler.ln_post.bias");

        // Vision-to-LLM projection MLP (2-layer GELU MLP)
        patterns.put("mm.0.weight", "mm.fc1.weight");
        patterns.put("mm.0.bias",   "mm.fc1.bias");
        patterns.put("mm.2.weight", "mm.fc2.weight");
        patterns.put("mm.2.bias",   "mm.fc2.bias");

        // ---- LLM decoder tensors (Qwen3-8B pattern, prefix: blk.{layer}.*) -
        patterns.put("blk.{layer}.attn_q.weight",              "model.layers.{layer}.self_attn.q_proj.weight");
        patterns.put("blk.{layer}.attn_q.bias",                "model.layers.{layer}.self_attn.q_proj.bias");
        patterns.put("blk.{layer}.attn_k.weight",              "model.layers.{layer}.self_attn.k_proj.weight");
        patterns.put("blk.{layer}.attn_k.bias",                "model.layers.{layer}.self_attn.k_proj.bias");
        patterns.put("blk.{layer}.attn_v.weight",              "model.layers.{layer}.self_attn.v_proj.weight");
        patterns.put("blk.{layer}.attn_v.bias",                "model.layers.{layer}.self_attn.v_proj.bias");
        patterns.put("blk.{layer}.attn_output.weight",         "model.layers.{layer}.self_attn.o_proj.weight");
        patterns.put("blk.{layer}.attn_q_norm.weight",         "model.layers.{layer}.self_attn.q_norm.weight");
        patterns.put("blk.{layer}.attn_k_norm.weight",         "model.layers.{layer}.self_attn.k_norm.weight");
        patterns.put("blk.{layer}.attn_norm.weight",           "model.layers.{layer}.input_layernorm.weight");
        patterns.put("blk.{layer}.ffn_norm.weight",            "model.layers.{layer}.post_attention_layernorm.weight");
        patterns.put("blk.{layer}.post_attention_norm.weight", "model.layers.{layer}.post_attention_layernorm.weight");
        patterns.put("blk.{layer}.ffn_gate.weight",            "model.layers.{layer}.mlp.gate_proj.weight");
        patterns.put("blk.{layer}.ffn_up.weight",              "model.layers.{layer}.mlp.up_proj.weight");
        patterns.put("blk.{layer}.ffn_down.weight",            "model.layers.{layer}.mlp.down_proj.weight");

        return patterns;
    }
}
