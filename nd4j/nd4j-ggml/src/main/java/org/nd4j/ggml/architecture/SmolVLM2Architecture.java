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
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Architecture handler for SmolVLM2 and Idefics3 vision-language models.
 *
 * <p>SmolVLM2 is a multimodal VLM combining:</p>
 * <ul>
 *   <li><b>Vision encoder</b>: SigLIP-so400m-patch14-384, 27 transformer layers,
 *       hidden size 1152, 16 heads</li>
 *   <li><b>Pixel shuffle projector</b>: space_to_depth with block_size=3,
 *       providing 9x spatial compression (3x3 pixel shuffle)</li>
 *   <li><b>LLM decoder</b>: SmolLM2-1.7B (VLlama3 arch), 24 layers, hidden size 2048,
 *       32 attention heads, 1 KV head (extreme GQA)</li>
 * </ul>
 *
 * <p>The GGUF file uses a single-file llama-compatible format where:</p>
 * <ul>
 *   <li>Vision tensors are prefixed with {@code v.}</li>
 *   <li>LLM tensors follow standard {@code blk.N.*} / {@code token_embd.*} naming</li>
 * </ul>
 *
 * <p>Supported GGUF architecture strings: {@code smolvlm}, {@code smolvlm2},
 * {@code smolvlm2-video}, {@code idefics3}.</p>
 *
 * <h3>GGUF tensor name conventions</h3>
 * <b>Vision encoder:</b>
 * <pre>
 *   v.patch_embd.weight             — patch embedding conv weight [hidden, c, ph, pw]
 *   v.position_embd.weight          — positional embedding [num_patches, hidden]
 *   v.blk.{layer}.attn_q.weight     — Q projection [hidden, hidden]
 *   v.blk.{layer}.attn_k.weight     — K projection [hidden, hidden]
 *   v.blk.{layer}.attn_v.weight     — V projection [hidden, hidden]
 *   v.blk.{layer}.attn_output.weight — output projection [hidden, hidden]
 *   v.blk.{layer}.ln1.weight / bias — pre-attention layer norm
 *   v.blk.{layer}.ln2.weight / bias — pre-FFN layer norm
 *   v.blk.{layer}.ffn_up.weight     — FFN up projection
 *   v.blk.{layer}.ffn_down.weight   — FFN down projection
 *   v.post_ln.weight / bias         — post-encoder layer norm
 * </pre>
 * <b>Projector (pixel shuffle):</b>
 * <pre>
 *   mm.proj.weight / bias           — linear projection after pixel shuffle
 * </pre>
 * <b>LLM decoder:</b>
 * <pre>
 *   token_embd.weight               — token embedding [vocab, hidden]
 *   blk.{layer}.attn_q.weight       — Q projection
 *   blk.{layer}.attn_k.weight       — K projection
 *   blk.{layer}.attn_v.weight       — V projection
 *   blk.{layer}.attn_output.weight  — output projection
 *   blk.{layer}.attn_norm.weight    — pre-attention RMS norm
 *   blk.{layer}.ffn_gate.weight     — FFN gate projection (SwiGLU)
 *   blk.{layer}.ffn_up.weight       — FFN up projection
 *   blk.{layer}.ffn_down.weight     — FFN down projection
 *   blk.{layer}.ffn_norm.weight     — pre-FFN RMS norm
 *   output_norm.weight              — final RMS norm
 *   output.weight                   — LM head (may be tied to token_embd)
 * </pre>
 */
@Slf4j
public class SmolVLM2Architecture implements ModelArchitecture {

    // Vision encoder constants (SigLIP-so400m-patch14-384)
    static final int VISION_NUM_LAYERS = 27;
    static final int VISION_HIDDEN_SIZE = 1152;
    static final int VISION_NUM_HEADS = 16;
    static final int VISION_PATCH_SIZE = 14;
    static final int VISION_IMAGE_SIZE = 384;

    // Pixel shuffle block size (3x3 → 9x spatial compression)
    static final int PIXEL_SHUFFLE_BLOCK_SIZE = 3;

    // LLM decoder constants (SmolLM2-1.7B / VLlama3)
    static final int LLM_NUM_LAYERS = 24;
    static final int LLM_HIDDEN_SIZE = 2048;
    static final int LLM_NUM_HEADS = 32;
    static final int LLM_NUM_KV_HEADS = 1;

    private static final Set<String> SUPPORTED_VARIANTS = Set.of(
            "smolvlm", "smolvlm2", "smolvlm2-video", "idefics3"
    );

    @Override
    public String getName() {
        return "smolvlm2";
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
        return archLower.contains("smolvlm") || archLower.contains("idefics");
    }

    @Override
    public String getDefaultChatTemplateType() {
        return "chatml";
    }

    @Override
    public String getModelSystemProperty() {
        return "smolvlm2.gguf.path";
    }

    @Override
    public String getReferencePrompt() {
        return "Describe what you see in this image.";
    }

    @Override
    public String[] getReferenceExpected() {
        return new String[]{"image"};
    }

    // ========================================================================
    // Graph construction
    // ========================================================================

    @Override
    public SameDiff buildGraph(GGMLMetadata metadata, Map<String, INDArray> weights, ConversionOptions options) {
        SameDiff sd = SameDiff.create();

        // Derive LLM config from metadata — fall back to SmolLM2-1.7B defaults
        ArchitectureConfig llmConfig = buildLlmConfig(metadata);

        DataType dtype = options.getTargetDataType();
        log.info("Building SmolVLM2 graph: visionLayers={}, visionHidden={}, " +
                        "llmLayers={}, llmHidden={}, llmHeads={}, llmKvHeads={}, dtype={}",
                VISION_NUM_LAYERS, VISION_HIDDEN_SIZE,
                llmConfig.getNumLayers(), llmConfig.getHiddenSize(),
                llmConfig.getNumAttentionHeads(), llmConfig.getNumKVHeads(), dtype);

        // ----------------------------------------------------------------
        // Vision encoder: image pixels -> patch embeddings -> ViT features
        // ----------------------------------------------------------------
        SDVariable visionFeatures = buildVisionEncoder(sd, weights, dtype);

        // ----------------------------------------------------------------
        // Pixel shuffle projector: 9x spatial compression + linear project
        // ----------------------------------------------------------------
        SDVariable projected = buildPixelShuffleProjector(sd, visionFeatures, weights, dtype);

        // ----------------------------------------------------------------
        // LLM decoder: text tokens + projected image features -> logits
        //
        // NOTE: Full multimodal fusion (image token interleaving with text
        // token_embd outputs) requires the inference pipeline to insert the
        // projected vision features at the <image> placeholder positions
        // before the first transformer layer. That interleaving is handled
        // by the generation pipeline, not by the static graph. The graph
        // below wires the LLM decoder assuming its input hidden states
        // already contain both text and image tokens merged by the caller.
        // ----------------------------------------------------------------
        SDVariable llmOutput = buildLlmDecoder(sd, weights, llmConfig, dtype);

        // Register graph outputs
        List<String> outputs = new ArrayList<>();
        outputs.add("lm_logits");
        // Expose per-layer KV cache outputs for autoregressive decoding
        for (int layer = 0; layer < llmConfig.getNumLayers(); layer++) {
            outputs.add("k_rope_" + layer);
            outputs.add("v_heads_" + layer);
        }
        sd.setOutputs(outputs);

        return sd;
    }

    // ========================================================================
    // Vision encoder (SigLIP ViT)
    // ========================================================================

    /**
     * Build the SigLIP-so400m ViT encoder.
     *
     * <p>Architecture: patch embedding → 27 transformer blocks (LayerNorm + MHSA + GELU FFN)
     * → post-encoder LayerNorm.</p>
     *
     * <p>SigLIP uses full LayerNorm (not RMSNorm) and GELU (not SwiGLU).</p>
     *
     * @return image features [batch, num_patches, vision_hidden]
     */
    private SDVariable buildVisionEncoder(SameDiff sd, Map<String, INDArray> weights, DataType dtype) {
        // Pixel values placeholder: [batch, channels, height, width]
        SDVariable pixelValues = sd.placeHolder("pixel_values", dtype, -1, 3,
                VISION_IMAGE_SIZE, VISION_IMAGE_SIZE);

        // Patch embedding — treat as reshape + linear (no conv2d needed for inference)
        // Patches: [batch, num_patches, patch_size*patch_size*channels]
        int numPatchesPerDim = VISION_IMAGE_SIZE / VISION_PATCH_SIZE; // 384/14 = 27 (≈27.4, actual 27)
        int numPatches = numPatchesPerDim * numPatchesPerDim;
        int patchDim = VISION_PATCH_SIZE * VISION_PATCH_SIZE * 3;

        INDArray patchEmbWeight = weights.get("v.patch_embd.weight");
        INDArray posEmbWeight = weights.get("v.position_embd.weight");

        SDVariable hidden;
        if (patchEmbWeight != null) {
            // Flatten spatial dims into patch tokens: [batch, num_patches, patchDim]
            SDVariable batchDim = sd.sizeAt(pixelValues, 0);
            SDVariable patchShape = sd.stack("v_patch_shape", 0,
                    batchDim,
                    sd.constant(Nd4j.scalar((long) numPatches)),
                    sd.constant(Nd4j.scalar((long) patchDim)));
            SDVariable patches = sd.reshape("v_patches", pixelValues, patchShape);

            SDVariable wPatch = sd.var("vision_encoder.patch_embedding.weight", patchEmbWeight);
            hidden = sd.mmul("v_patch_emb", patches, wPatch.permute(1, 0));

            INDArray patchBias = weights.get("v.patch_embd.bias");
            if (patchBias != null) {
                hidden = hidden.add("v_patch_emb_biased",
                        sd.var("vision_encoder.patch_embedding.bias", patchBias));
            }
        } else {
            log.warn("SmolVLM2: Missing v.patch_embd.weight — using zero patch embeddings");
            SDVariable batchDim = sd.sizeAt(pixelValues, 0);
            SDVariable zeroShape = sd.stack("v_zero_shape", 0,
                    batchDim,
                    sd.constant(Nd4j.scalar((long) numPatches)),
                    sd.constant(Nd4j.scalar((long) VISION_HIDDEN_SIZE)));
            hidden = sd.zero("v_patch_emb", DataType.FLOAT).reshape(zeroShape).castTo(dtype);
        }

        // Add positional embeddings
        if (posEmbWeight != null) {
            SDVariable posEmb = sd.var("vision_encoder.position_embedding", posEmbWeight);
            hidden = hidden.add("v_pos_emb", posEmb);
        }

        // 27 vision transformer blocks
        for (int layer = 0; layer < VISION_NUM_LAYERS; layer++) {
            hidden = buildVisionTransformerBlock(sd, hidden, layer, weights, dtype);
        }

        // Post-encoder LayerNorm
        INDArray postLnWeight = weights.get("v.post_ln.weight");
        INDArray postLnBias = weights.get("v.post_ln.bias");
        if (postLnWeight != null) {
            hidden = buildLayerNorm(sd, hidden, "vision_encoder.post_layernorm",
                    postLnWeight, postLnBias, 1e-6f);
        }

        sd.updateVariableNameAndReference(hidden, "vision_features");
        return hidden;
    }

    /**
     * Single SigLIP ViT transformer block: LayerNorm + MHSA + residual + LayerNorm + GELU FFN + residual.
     */
    private SDVariable buildVisionTransformerBlock(SameDiff sd, SDVariable input, int layerIdx,
            Map<String, INDArray> weights, DataType dtype) {

        String ggufPrefix = "v.blk." + layerIdx;
        String sdPrefix = "vision_encoder.blocks." + layerIdx;

        // Pre-attention LayerNorm
        INDArray ln1Weight = weights.get(ggufPrefix + ".ln1.weight");
        INDArray ln1Bias = weights.get(ggufPrefix + ".ln1.bias");
        SDVariable normed1;
        if (ln1Weight != null) {
            normed1 = buildLayerNorm(sd, input, sdPrefix + ".ln1", ln1Weight, ln1Bias, 1e-6f);
        } else {
            log.warn("SmolVLM2 vision layer {}: missing ln1 weights", layerIdx);
            normed1 = input;
        }

        // Multi-head self-attention (full attention — no KV cache for vision encoder)
        SDVariable attnOut = buildVisionAttention(sd, normed1, layerIdx, weights, dtype);

        // Residual
        SDVariable postAttn = input.add("v_post_attn_" + layerIdx, attnOut);

        // Pre-FFN LayerNorm
        INDArray ln2Weight = weights.get(ggufPrefix + ".ln2.weight");
        INDArray ln2Bias = weights.get(ggufPrefix + ".ln2.bias");
        SDVariable normed2;
        if (ln2Weight != null) {
            normed2 = buildLayerNorm(sd, postAttn, sdPrefix + ".ln2", ln2Weight, ln2Bias, 1e-6f);
        } else {
            log.warn("SmolVLM2 vision layer {}: missing ln2 weights", layerIdx);
            normed2 = postAttn;
        }

        // GELU feed-forward (SigLIP uses standard GELU, not SwiGLU)
        SDVariable ffnOut = buildVisionGELUFFN(sd, normed2, layerIdx, weights, dtype);

        // Residual
        return postAttn.add("v_layer_out_" + layerIdx, ffnOut);
    }

    /**
     * Vision encoder MHSA — full (non-causal) attention, no KV cache.
     * SigLIP uses 16 heads with headDim = 1152/16 = 72.
     */
    private SDVariable buildVisionAttention(SameDiff sd, SDVariable input, int layerIdx,
            Map<String, INDArray> weights, DataType dtype) {

        String ggufPrefix = "v.blk." + layerIdx;
        String sdPrefix = "vision_encoder.blocks." + layerIdx + ".attn.";

        int headDim = VISION_HIDDEN_SIZE / VISION_NUM_HEADS; // 72

        INDArray qWeight = weights.get(ggufPrefix + ".attn_q.weight");
        INDArray kWeight = weights.get(ggufPrefix + ".attn_k.weight");
        INDArray vWeight = weights.get(ggufPrefix + ".attn_v.weight");
        INDArray oWeight = weights.get(ggufPrefix + ".attn_output.weight");

        if (qWeight == null || kWeight == null || vWeight == null || oWeight == null) {
            log.warn("SmolVLM2 vision layer {}: missing attention weights, passing through", layerIdx);
            return input;
        }

        SDVariable wq = sd.var(sdPrefix + "q_proj.weight", qWeight);
        SDVariable wk = sd.var(sdPrefix + "k_proj.weight", kWeight);
        SDVariable wv = sd.var(sdPrefix + "v_proj.weight", vWeight);
        SDVariable wo = sd.var(sdPrefix + "o_proj.weight", oWeight);

        // Project Q, K, V — [batch, num_patches, hidden] -> [batch, num_patches, hidden]
        SDVariable q = sd.mmul("v_q_" + layerIdx, input, wq.permute(1, 0));
        SDVariable k = sd.mmul("v_k_" + layerIdx, input, wk.permute(1, 0));
        SDVariable v = sd.mmul("v_v_" + layerIdx, input, wv.permute(1, 0));

        // Add biases if present
        INDArray qBias = weights.get(ggufPrefix + ".attn_q.bias");
        INDArray kBias = weights.get(ggufPrefix + ".attn_k.bias");
        INDArray vBias = weights.get(ggufPrefix + ".attn_v.bias");
        if (qBias != null) q = q.add(sd.var(sdPrefix + "q_proj.bias", qBias));
        if (kBias != null) k = k.add(sd.var(sdPrefix + "k_proj.bias", kBias));
        if (vBias != null) v = v.add(sd.var(sdPrefix + "v_proj.bias", vBias));

        SDVariable batchDim = sd.sizeAt(input, 0);
        SDVariable seqDim = sd.sizeAt(input, 1);

        // Reshape to [batch, seq, num_heads, head_dim]
        SDVariable headShape = sd.stack("v_head_shape_" + layerIdx, 0,
                batchDim, seqDim,
                sd.constant(Nd4j.scalar((long) VISION_NUM_HEADS)),
                sd.constant(Nd4j.scalar((long) headDim)));
        q = sd.reshape("v_q_heads_" + layerIdx, q, headShape);
        k = sd.reshape("v_k_heads_" + layerIdx, k, headShape);
        v = sd.reshape("v_v_heads_" + layerIdx, v, headShape);

        // Full bidirectional attention (no causal mask for vision encoder)
        SDVariable attnOut = sd.nn.dotProductAttentionV2(
                "v_attn_out_" + layerIdx,
                q, v, k, null, null,
                0.0, 0.0, false, false
        );

        // Reshape back to [batch, seq, hidden]
        SDVariable flatShape = sd.stack("v_attn_flat_shape_" + layerIdx, 0,
                batchDim, seqDim,
                sd.constant(Nd4j.scalar((long) VISION_HIDDEN_SIZE)));
        SDVariable attnFlat = sd.reshape("v_attn_flat_" + layerIdx, attnOut, flatShape);

        SDVariable out = sd.mmul("v_attn_proj_" + layerIdx, attnFlat, wo.permute(1, 0));

        INDArray oBias = weights.get(ggufPrefix + ".attn_output.bias");
        if (oBias != null) {
            out = out.add("v_attn_proj_biased_" + layerIdx, sd.var(sdPrefix + "o_proj.bias", oBias));
        }
        return out;
    }

    /**
     * Vision encoder GELU FFN (SigLIP uses standard GELU, not SwiGLU).
     */
    private SDVariable buildVisionGELUFFN(SameDiff sd, SDVariable input, int layerIdx,
            Map<String, INDArray> weights, DataType dtype) {

        String ggufPrefix = "v.blk." + layerIdx;
        String sdPrefix = "vision_encoder.blocks." + layerIdx + ".mlp.";

        INDArray upWeight = weights.get(ggufPrefix + ".ffn_up.weight");
        INDArray downWeight = weights.get(ggufPrefix + ".ffn_down.weight");

        if (upWeight == null || downWeight == null) {
            log.warn("SmolVLM2 vision layer {}: missing FFN weights, passing through", layerIdx);
            return input;
        }

        SDVariable wUp = sd.var(sdPrefix + "fc1.weight", upWeight);
        SDVariable wDown = sd.var(sdPrefix + "fc2.weight", downWeight);

        SDVariable up = sd.mmul("v_up_" + layerIdx, input, wUp.permute(1, 0));

        INDArray upBias = weights.get(ggufPrefix + ".ffn_up.bias");
        if (upBias != null) {
            up = up.add("v_up_biased_" + layerIdx, sd.var(sdPrefix + "fc1.bias", upBias));
        }

        SDVariable activated = sd.nn.gelu("v_gelu_" + layerIdx, up);
        SDVariable down = sd.mmul("v_down_" + layerIdx, activated, wDown.permute(1, 0));

        INDArray downBias = weights.get(ggufPrefix + ".ffn_down.bias");
        if (downBias != null) {
            down = down.add("v_down_biased_" + layerIdx, sd.var(sdPrefix + "fc2.bias", downBias));
        }
        return down;
    }

    // ========================================================================
    // Pixel shuffle projector
    // ========================================================================

    /**
     * Build the pixel shuffle (space_to_depth) projector.
     *
     * <p>SmolVLM2 uses a 3x3 pixel shuffle to compress the spatial resolution by 9x:
     * [batch, H*W, 1152] → reshape to [batch, H/3 * W/3, 1152 * 9] → linear → [batch, H/3 * W/3, llm_hidden].</p>
     *
     * <p>The pixel shuffle is implemented as a reshape + linear projection. The output dimension
     * matches the LLM hidden size (2048 by default).</p>
     *
     * @param visionFeatures [batch, num_patches, vision_hidden]
     * @return projected [batch, num_patches / 9, llm_hidden]
     */
    private SDVariable buildPixelShuffleProjector(SameDiff sd, SDVariable visionFeatures,
            Map<String, INDArray> weights, DataType dtype) {

        // num_patches = 27*27 = 729 (for 384px / 14px patches)
        // After 3x3 pixel shuffle: 729 / 9 = 81 tokens
        int numPatchesPerDim = VISION_IMAGE_SIZE / VISION_PATCH_SIZE; // 27
        int numPatchesCompressed = (numPatchesPerDim / PIXEL_SHUFFLE_BLOCK_SIZE)
                * (numPatchesPerDim / PIXEL_SHUFFLE_BLOCK_SIZE); // 81
        int projectorInputDim = VISION_HIDDEN_SIZE * PIXEL_SHUFFLE_BLOCK_SIZE * PIXEL_SHUFFLE_BLOCK_SIZE; // 1152*9=10368

        SDVariable batchDim = sd.sizeAt(visionFeatures, 0);

        // Space-to-depth reshape: [batch, h*w, c] → [batch, (h/b)*(w/b), c*b*b]
        SDVariable shuffleShape = sd.stack("mm_shuffle_shape", 0,
                batchDim,
                sd.constant(Nd4j.scalar((long) numPatchesCompressed)),
                sd.constant(Nd4j.scalar((long) projectorInputDim)));
        SDVariable shuffled = sd.reshape("mm_pixel_shuffle", visionFeatures, shuffleShape);

        // Linear projection: [batch, num_compressed_patches, projectorInputDim] → [batch, ..., llm_hidden]
        INDArray projWeight = weights.get("mm.proj.weight");
        if (projWeight == null) {
            log.warn("SmolVLM2: Missing mm.proj.weight — vision features will not be projected to LLM hidden size");
            sd.updateVariableNameAndReference(shuffled, "mm_projected");
            return shuffled;
        }

        SDVariable wProj = sd.var("multimodal_projector.proj.weight", projWeight);
        SDVariable projected = sd.mmul("mm_proj", shuffled, wProj.permute(1, 0));

        INDArray projBias = weights.get("mm.proj.bias");
        if (projBias != null) {
            projected = projected.add("mm_proj_biased",
                    sd.var("multimodal_projector.proj.bias", projBias));
        }

        sd.updateVariableNameAndReference(projected, "mm_projected");
        return projected;
    }

    // ========================================================================
    // LLM decoder (SmolLM2-1.7B / VLlama3 style)
    // ========================================================================

    /**
     * Build the SmolLM2-1.7B LLM decoder.
     *
     * <p>Architecture: RMSNorm + GQA (32 Q heads, 1 KV head) + SwiGLU FFN × 24 layers.
     * This delegates to the LLaMA core pattern for the decoder-only transformer.
     * The full multimodal graph is assembled by the generation pipeline which
     * injects projected vision tokens into the input hidden states.</p>
     *
     * <p><b>Implementation note:</b> This method builds the LLM decoder graph. The
     * vision features produced by {@link #buildPixelShuffleProjector} are available
     * as a named variable ({@code mm_projected}) and must be merged with the text
     * embeddings by the inference pipeline before the first transformer layer.
     * A full static fusion graph would require dynamic sequence-position management
     * that is better handled at the pipeline level.</p>
     *
     * @return logits variable [batch, seq_len, vocab_size]
     */
    private SDVariable buildLlmDecoder(SameDiff sd, Map<String, INDArray> weights,
            ArchitectureConfig config, DataType dtype) {

        log.warn("SmolVLM2: Building LLM decoder stub (full vision-text fusion handled by generation pipeline). " +
                "LLM layers={}, hidden={}, heads={}, kvHeads={}",
                config.getNumLayers(), config.getHiddenSize(),
                config.getNumAttentionHeads(), config.getNumKVHeads());

        // --- Input placeholders ---
        // input_ids: text token IDs (after image tokens have been inserted by the pipeline)
        SDVariable inputIds = sd.placeHolder("input_ids", DataType.INT64, -1, -1);
        SDVariable positionOffset = sd.placeHolder("position_offset", DataType.INT64);
        SDVariable cachePosition = sd.placeHolder("cache_position", DataType.INT64);
        SDVariable causalMask = sd.placeHolder("_causal_mask", DataType.FLOAT, -1, -1, -1, -1);

        int headDim = config.getHeadDimension();
        int numKVHeads = config.getNumKVHeads();

        // Per-layer KV cache placeholders
        Map<Integer, SDVariable> keyCachePlaceholders = new HashMap<>();
        Map<Integer, SDVariable> valueCachePlaceholders = new HashMap<>();
        for (int layer = 0; layer < config.getNumLayers(); layer++) {
            SDVariable keyCache = sd.placeHolder("past_key_values." + layer + ".key",
                    dtype, -1, -1, numKVHeads, headDim);
            SDVariable valueCache = sd.placeHolder("past_key_values." + layer + ".value",
                    dtype, -1, -1, numKVHeads, headDim);
            keyCachePlaceholders.put(layer, keyCache);
            valueCachePlaceholders.put(layer, valueCache);
        }

        // Token embedding
        INDArray tokenEmbedWeight = weights.get("token_embd.weight");
        if (tokenEmbedWeight == null) {
            throw new IllegalStateException("SmolVLM2: Missing token_embd.weight");
        }
        SDVariable tokenEmbed = sd.var("model.embed_tokens.weight", tokenEmbedWeight);
        SDVariable hidden = sd.gather("embedded", tokenEmbed, inputIds, 0);

        // 24 LLM transformer blocks (LLaMA pattern: RMSNorm + GQA + SwiGLU)
        for (int layer = 0; layer < config.getNumLayers(); layer++) {
            hidden = buildLlmTransformerBlock(sd, hidden, layer, config, weights, dtype,
                    positionOffset, cachePosition, causalMask,
                    keyCachePlaceholders.get(layer),
                    valueCachePlaceholders.get(layer));
        }

        // Final RMS normalization
        hidden = buildRMSNorm(sd, hidden, "model.norm", "output_norm", weights, config, dtype);

        // LM head
        INDArray outputWeight = weights.get("output.weight");
        if (outputWeight == null) {
            outputWeight = tokenEmbedWeight; // Tied weights
        }
        SDVariable lmHead = sd.var("lm_head.weight", outputWeight);

        // Logits [batch, seq, vocab] — upcast to FP32 to avoid overflow at vocab scale
        SDVariable logits = fp32Mmul(sd, "lm_logits", hidden, lmHead.permute(1, 0), dtype);
        return logits;
    }

    /**
     * Single LLM transformer block: RMSNorm + GQA (RoPE) + residual + RMSNorm + SwiGLU + residual.
     */
    private SDVariable buildLlmTransformerBlock(SameDiff sd, SDVariable input, int layerIdx,
            ArchitectureConfig config, Map<String, INDArray> weights, DataType dtype,
            SDVariable positionOffset, SDVariable cachePosition, SDVariable causalMask,
            SDVariable keyCache, SDVariable valueCache) {

        String ggufPrefix = "blk." + layerIdx;

        // Pre-attention RMS norm
        SDVariable normed = buildRMSNorm(sd, input,
                "model.layers." + layerIdx + ".input_layernorm",
                ggufPrefix + ".attn_norm", weights, config, dtype);

        // GQA attention with RoPE + KV cache
        SDVariable attnOut = buildLlmAttention(sd, normed, layerIdx, config, weights, dtype,
                positionOffset, cachePosition, causalMask, keyCache, valueCache);

        SDVariable postAttn = input.add("post_attn_" + layerIdx, attnOut);

        // Pre-FFN RMS norm
        SDVariable ffnNormed = buildRMSNorm(sd, postAttn,
                "model.layers." + layerIdx + ".post_attention_layernorm",
                ggufPrefix + ".ffn_norm", weights, config, dtype);

        // SwiGLU FFN
        SDVariable ffnOut;
        if (weights.containsKey(ggufPrefix + ".ffn_gate.weight")) {
            ffnOut = buildSwiGLUFFN(sd, ffnNormed, layerIdx, weights, dtype);
        } else {
            log.warn("SmolVLM2 LLM layer {}: missing FFN weights, passing through", layerIdx);
            ffnOut = ffnNormed;
        }

        return postAttn.add("layer_out_" + layerIdx, ffnOut);
    }

    /**
     * LLM GQA attention: 32 Q heads, 1 KV head (extreme GQA), with RoPE and KV cache.
     */
    private SDVariable buildLlmAttention(SameDiff sd, SDVariable input, int layerIdx,
            ArchitectureConfig config, Map<String, INDArray> weights, DataType dtype,
            SDVariable positionOffset, SDVariable cachePosition, SDVariable causalMask,
            SDVariable keyCache, SDVariable valueCache) {

        String ggufPrefix = "blk." + layerIdx;
        String sdPrefix = "model.layers." + layerIdx + ".self_attn.";

        int numHeads = config.getNumAttentionHeads();
        int numKVHeads = config.getNumKVHeads();

        INDArray qWeight = weights.get(ggufPrefix + ".attn_q.weight");
        INDArray kWeight = weights.get(ggufPrefix + ".attn_k.weight");
        INDArray vWeight = weights.get(ggufPrefix + ".attn_v.weight");
        INDArray oWeight = weights.get(ggufPrefix + ".attn_output.weight");

        if (qWeight == null || kWeight == null || vWeight == null || oWeight == null) {
            log.warn("SmolVLM2 LLM layer {}: missing attention weights, passing through", layerIdx);
            return input;
        }

        int kOutDim = (int) kWeight.shape()[0];
        int headDim = (numKVHeads > 0) ? kOutDim / numKVHeads : config.getHeadDimension();
        int qOutDim = (int) qWeight.shape()[0];
        int actualNumHeads = (headDim > 0) ? qOutDim / headDim : numHeads;

        SDVariable wq = sd.var(sdPrefix + "q_proj.weight", qWeight);
        SDVariable wk = sd.var(sdPrefix + "k_proj.weight", kWeight);
        SDVariable wv = sd.var(sdPrefix + "v_proj.weight", vWeight);
        SDVariable wo = sd.var(sdPrefix + "o_proj.weight", oWeight);

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
                sd.constant(Nd4j.scalar((long) numKVHeads)),
                sd.constant(Nd4j.scalar((long) headDim)));

        q = sd.reshape("q_heads_" + layerIdx, q, qShapeVar);
        k = sd.reshape("k_heads_" + layerIdx, k, kvShapeVar);
        v = sd.reshape("v_heads_" + layerIdx, v, kvShapeVar);

        // RoPE positional encoding with dynamic offset for DSP replay
        if (config.isUseRotaryEmbeddings()) {
            org.nd4j.linalg.api.ops.impl.transforms.custom.FusedRoPE ropeQ =
                    new org.nd4j.linalg.api.ops.impl.transforms.custom.FusedRoPE(
                            sd, q, positionOffset,
                            config.getRopeType(), config.getRopeFreqBase(), 1.0,
                            config.getRopeDimensionCount());
            q = ropeQ.outputVariable();
            sd.updateVariableNameAndReference(q, "q_rope_" + layerIdx);

            org.nd4j.linalg.api.ops.impl.transforms.custom.FusedRoPE ropeK =
                    new org.nd4j.linalg.api.ops.impl.transforms.custom.FusedRoPE(
                            sd, k, positionOffset,
                            config.getRopeType(), config.getRopeFreqBase(), 1.0,
                            config.getRopeDimensionCount());
            k = ropeK.outputVariable();
            sd.updateVariableNameAndReference(k, "k_rope_" + layerIdx);
        }

        // V must match Q/K dtype after FusedRoPE promotes HALF → FLOAT
        if (v.dataType() != q.dataType()) {
            v = v.castTo("v_cast_" + layerIdx, q.dataType());
        }

        // Dot-product attention with KV cache and causal mask
        org.nd4j.linalg.api.ops.impl.transforms.custom.DotProductAttentionV2 attnOp =
                new org.nd4j.linalg.api.ops.impl.transforms.custom.DotProductAttentionV2(
                        sd, q, v, k, null, null,
                        keyCache, valueCache, cachePosition, causalMask,
                        0.0, 0.0, false, false);
        SDVariable attnOut = attnOp.outputVariable();
        sd.updateVariableNameAndReference(attnOut, "attn_out_" + layerIdx);

        int attnOutDim = actualNumHeads * headDim;
        SDVariable outShapeVar = sd.stack("attn_out_shape_" + layerIdx, 0,
                batchDim, seqDim,
                sd.constant(Nd4j.scalar((long) attnOutDim)));
        SDVariable attnFlat = sd.reshape("attn_flat_" + layerIdx, attnOut, outShapeVar);

        // Expose K/V outputs for cache update (named for pipeline extraction)
        sd.updateVariableNameAndReference(k, "k_rope_" + layerIdx);
        sd.updateVariableNameAndReference(v, "v_heads_" + layerIdx);

        return fp32Mmul(sd, "attn_proj_" + layerIdx, attnFlat, wo.permute(1, 0), dtype);
    }

    /**
     * SwiGLU FFN: gate * SiLU(up) projected down.
     */
    private SDVariable buildSwiGLUFFN(SameDiff sd, SDVariable input, int layerIdx,
            Map<String, INDArray> weights, DataType dtype) {

        String ggufPrefix = "blk." + layerIdx;
        String sdPrefix = "model.layers." + layerIdx + ".mlp.";

        INDArray gateWeight = weights.get(ggufPrefix + ".ffn_gate.weight");
        INDArray upWeight = weights.get(ggufPrefix + ".ffn_up.weight");
        INDArray downWeight = weights.get(ggufPrefix + ".ffn_down.weight");

        if (gateWeight == null || upWeight == null || downWeight == null) {
            log.warn("SmolVLM2 LLM layer {}: missing SwiGLU weights, passing through", layerIdx);
            return input;
        }

        SDVariable wGate = sd.var(sdPrefix + "gate_proj.weight", gateWeight);
        SDVariable wUp = sd.var(sdPrefix + "up_proj.weight", upWeight);
        SDVariable wDown = sd.var(sdPrefix + "down_proj.weight", downWeight);

        SDVariable gate = fp32Mmul(sd, "gate_" + layerIdx, input, wGate.permute(1, 0), dtype);
        SDVariable up = fp32Mmul(sd, "up_" + layerIdx, input, wUp.permute(1, 0), dtype);

        SDVariable silu = sd.nn.swish(gate);
        SDVariable hidden = silu.mul("swiglu_" + layerIdx, up);

        return fp32Mmul(sd, "down_" + layerIdx, hidden, wDown.permute(1, 0), dtype);
    }

    // ========================================================================
    // Normalization helpers
    // ========================================================================

    /**
     * RMS normalization (LLaMA-style, no bias).
     * Upcasts to FP32 for the squaring step to avoid HALF overflow.
     */
    private SDVariable buildRMSNorm(SameDiff sd, SDVariable input, String outputName,
            String weightKey, Map<String, INDArray> weights, ArchitectureConfig config, DataType dtype) {

        INDArray normWeight = weights.get(weightKey + ".weight");
        if (normWeight == null) {
            log.warn("SmolVLM2: Missing RMS norm weight: {}", weightKey);
            return input;
        }

        SDVariable gamma = sd.var(outputName + ".weight", normWeight);

        boolean needsCast = (input.dataType() == DataType.HALF || input.dataType() == DataType.BFLOAT16);
        SDVariable computeInput = needsCast ? input.castTo(outputName + "_f32", DataType.FLOAT) : input;

        SDVariable squared = computeInput.mul(computeInput);
        SDVariable meanSquared = squared.mean(true, -1);
        SDVariable rms = sd.math.sqrt(meanSquared.add(config.getLayerNormEpsilon()));
        SDVariable normalized = computeInput.div(rms);

        SDVariable normalizedOrig = needsCast
                ? normalized.castTo(outputName + "_cast", input.dataType())
                : normalized;

        return normalizedOrig.mul(outputName, gamma);
    }

    /**
     * Standard LayerNorm with optional bias (used by the SigLIP vision encoder).
     */
    private SDVariable buildLayerNorm(SameDiff sd, SDVariable input, String outputName,
            INDArray normWeight, INDArray normBias, float eps) {

        SDVariable gamma = sd.var(outputName + ".weight", normWeight);

        boolean needsCast = (input.dataType() == DataType.HALF || input.dataType() == DataType.BFLOAT16);
        SDVariable computeInput = needsCast ? input.castTo(outputName + "_f32", DataType.FLOAT) : input;

        SDVariable mean = computeInput.mean(true, -1);
        SDVariable centered = computeInput.sub(mean);
        SDVariable variance = centered.mul(centered).mean(true, -1);
        SDVariable stdDev = sd.math.sqrt(variance.add(eps));
        SDVariable normalized = centered.div(stdDev);

        SDVariable normalizedOrig = needsCast
                ? normalized.castTo(outputName + "_cast", input.dataType())
                : normalized;

        SDVariable scaled = normalizedOrig.mul(outputName + "_scaled", gamma);
        if (normBias != null) {
            SDVariable beta = sd.var(outputName + ".bias", normBias);
            return scaled.add(outputName, beta);
        }
        sd.updateVariableNameAndReference(scaled, outputName);
        return scaled;
    }

    // ========================================================================
    // FP32 matmul helper
    // ========================================================================

    /**
     * Matmul in FP32 to avoid HALF overflow on large dot products, then cast back to dtype.
     */
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
    // Configuration helpers
    // ========================================================================

    /**
     * Build LLM ArchitectureConfig from GGUF metadata, falling back to SmolLM2-1.7B defaults.
     */
    private ArchitectureConfig buildLlmConfig(GGMLMetadata metadata) {
        int numLayers = metadata.getNumLayers() > 0 ? metadata.getNumLayers() : LLM_NUM_LAYERS;
        int hiddenSize = metadata.getHiddenSize() > 0 ? metadata.getHiddenSize() : LLM_HIDDEN_SIZE;
        int numHeads = metadata.getNumAttentionHeads() > 0 ? metadata.getNumAttentionHeads() : LLM_NUM_HEADS;
        int numKVHeads = metadata.getNumKVHeads() > 0 ? metadata.getNumKVHeads() : LLM_NUM_KV_HEADS;
        int headDim = metadata.getAttentionKeyLength();

        return ArchitectureConfig.builder()
                .numLayers(numLayers)
                .hiddenSize(hiddenSize)
                .intermediateSize(metadata.getIntermediateSize())
                .numAttentionHeads(numHeads)
                .numKVHeads(numKVHeads)
                .vocabSize(metadata.getVocabSize())
                .contextLength(metadata.getContextLength())
                .layerNormEpsilon(metadata.getLayerNormEpsilon() > 0 ? metadata.getLayerNormEpsilon() : 1e-5f)
                .ropeFreqBase(metadata.getRopeFreqBase() > 0 ? metadata.getRopeFreqBase() : 10000.0f)
                .ropeDimensionCount(metadata.getRopeDimensionCount())
                .headDim(headDim)
                .ropeType(metadata.getRopeType())
                .layerTypes(metadata.getLayerTypes())
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

        // ---- Vision encoder ----
        patterns.put("v.patch_embd.weight", "vision_encoder.patch_embedding.weight");
        patterns.put("v.patch_embd.bias", "vision_encoder.patch_embedding.bias");
        patterns.put("v.position_embd.weight", "vision_encoder.position_embedding");
        patterns.put("v.post_ln.weight", "vision_encoder.post_layernorm.weight");
        patterns.put("v.post_ln.bias", "vision_encoder.post_layernorm.bias");

        // Vision attention (per-layer)
        patterns.put("v.blk.{layer}.attn_q.weight", "vision_encoder.blocks.{layer}.attn.q_proj.weight");
        patterns.put("v.blk.{layer}.attn_q.bias", "vision_encoder.blocks.{layer}.attn.q_proj.bias");
        patterns.put("v.blk.{layer}.attn_k.weight", "vision_encoder.blocks.{layer}.attn.k_proj.weight");
        patterns.put("v.blk.{layer}.attn_k.bias", "vision_encoder.blocks.{layer}.attn.k_proj.bias");
        patterns.put("v.blk.{layer}.attn_v.weight", "vision_encoder.blocks.{layer}.attn.v_proj.weight");
        patterns.put("v.blk.{layer}.attn_v.bias", "vision_encoder.blocks.{layer}.attn.v_proj.bias");
        patterns.put("v.blk.{layer}.attn_output.weight", "vision_encoder.blocks.{layer}.attn.o_proj.weight");
        patterns.put("v.blk.{layer}.attn_output.bias", "vision_encoder.blocks.{layer}.attn.o_proj.bias");

        // Vision layer norms
        patterns.put("v.blk.{layer}.ln1.weight", "vision_encoder.blocks.{layer}.ln1.weight");
        patterns.put("v.blk.{layer}.ln1.bias", "vision_encoder.blocks.{layer}.ln1.bias");
        patterns.put("v.blk.{layer}.ln2.weight", "vision_encoder.blocks.{layer}.ln2.weight");
        patterns.put("v.blk.{layer}.ln2.bias", "vision_encoder.blocks.{layer}.ln2.bias");

        // Vision FFN
        patterns.put("v.blk.{layer}.ffn_up.weight", "vision_encoder.blocks.{layer}.mlp.fc1.weight");
        patterns.put("v.blk.{layer}.ffn_up.bias", "vision_encoder.blocks.{layer}.mlp.fc1.bias");
        patterns.put("v.blk.{layer}.ffn_down.weight", "vision_encoder.blocks.{layer}.mlp.fc2.weight");
        patterns.put("v.blk.{layer}.ffn_down.bias", "vision_encoder.blocks.{layer}.mlp.fc2.bias");

        // ---- Multimodal projector ----
        patterns.put("mm.proj.weight", "multimodal_projector.proj.weight");
        patterns.put("mm.proj.bias", "multimodal_projector.proj.bias");

        // ---- LLM decoder ----
        patterns.put("token_embd.weight", "model.embed_tokens.weight");
        patterns.put("output.weight", "lm_head.weight");
        patterns.put("output_norm.weight", "model.norm.weight");

        // LLM attention (per-layer)
        patterns.put("blk.{layer}.attn_q.weight", "model.layers.{layer}.self_attn.q_proj.weight");
        patterns.put("blk.{layer}.attn_k.weight", "model.layers.{layer}.self_attn.k_proj.weight");
        patterns.put("blk.{layer}.attn_v.weight", "model.layers.{layer}.self_attn.v_proj.weight");
        patterns.put("blk.{layer}.attn_output.weight", "model.layers.{layer}.self_attn.o_proj.weight");

        // LLM norms
        patterns.put("blk.{layer}.attn_norm.weight", "model.layers.{layer}.input_layernorm.weight");
        patterns.put("blk.{layer}.ffn_norm.weight", "model.layers.{layer}.post_attention_layernorm.weight");

        // LLM FFN (SwiGLU)
        patterns.put("blk.{layer}.ffn_gate.weight", "model.layers.{layer}.mlp.gate_proj.weight");
        patterns.put("blk.{layer}.ffn_up.weight", "model.layers.{layer}.mlp.up_proj.weight");
        patterns.put("blk.{layer}.ffn_down.weight", "model.layers.{layer}.mlp.down_proj.weight");

        return patterns;
    }
}
