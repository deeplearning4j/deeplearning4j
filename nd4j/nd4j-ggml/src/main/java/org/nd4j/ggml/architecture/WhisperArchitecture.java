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
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv1DConfig;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.PaddingMode;

import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/**
 * Architecture handler for OpenAI Whisper models in GGML format.
 * <p>
 * Builds a SameDiff encoder-decoder graph from Whisper GGML weights.
 * Supports the whisper.cpp GGML tensor naming convention:
 * <ul>
 *   <li>encoder.blocks.{i}.attn.query.weight</li>
 *   <li>encoder.blocks.{i}.attn.key.weight</li>
 *   <li>encoder.blocks.{i}.attn.value.weight</li>
 *   <li>encoder.blocks.{i}.attn.out.weight</li>
 *   <li>encoder.blocks.{i}.mlp.0.weight (fc1)</li>
 *   <li>encoder.blocks.{i}.mlp.2.weight (fc2)</li>
 *   <li>encoder.blocks.{i}.attn_ln.weight / bias</li>
 *   <li>encoder.blocks.{i}.mlp_ln.weight / bias</li>
 *   <li>decoder.blocks.{i}.cross_attn.query.weight</li>
 *   <li>decoder.blocks.{i}.cross_attn.key.weight</li>
 *   <li>decoder.blocks.{i}.cross_attn.value.weight</li>
 *   <li>decoder.blocks.{i}.cross_attn.out.weight</li>
 *   <li>encoder.conv1.weight / bias</li>
 *   <li>encoder.conv2.weight / bias</li>
 *   <li>encoder.positional_embedding</li>
 *   <li>decoder.positional_embedding</li>
 *   <li>decoder.token_embedding.weight</li>
 * </ul>
 */
@Slf4j
public class WhisperArchitecture implements ModelArchitecture {

    @Override
    public String getName() {
        return "whisper";
    }

    @Override
    public Set<String> getSupportedVariants() {
        return new HashSet<>(Arrays.asList("whisper", "whisper.encoder", "whisper.decoder"));
    }

    @Override
    public String getDefaultChatTemplateType() {
        return "none"; // Whisper is speech-to-text, no chat template
    }

    @Override
    public String getModelSystemProperty() {
        return "whisper.gguf.path";
    }

    @Override
    public String getReferencePrompt() {
        return ""; // Whisper takes audio, not text prompts
    }

    @Override
    public String[] getReferenceExpected() {
        return new String[]{}; // No text expected from prompt
    }

    @Override
    public boolean canHandle(GGMLMetadata metadata) {
        if (metadata.getArchitecture() != null &&
                metadata.getArchitecture().toLowerCase().contains("whisper")) {
            return true;
        }
        // Check tensor names for whisper-specific patterns
        for (var tensor : metadata.getTensors()) {
            String name = tensor.getName().toLowerCase();
            if (name.startsWith("encoder.conv1.") || name.startsWith("decoder.token_embedding.")) {
                return true;
            }
        }
        return false;
    }

    @Override
    public SameDiff buildGraph(GGMLMetadata metadata, Map<String, INDArray> weights, ConversionOptions options) {
        SameDiff sd = SameDiff.create();

        ArchitectureConfig config = getConfig(metadata);
        int hiddenSize = config.getHiddenSize();
        int numHeads = config.getNumAttentionHeads();
        int headDim = hiddenSize / numHeads;

        // --- Encoder ---

        // Input: mel spectrogram [batch, numMelBins, 3000]
        SDVariable inputFeatures = sd.placeHolder("input_features", DataType.FLOAT, -1, -1, -1);

        // Conv1 + GELU + Conv2 + GELU
        SDVariable x = buildConvLayers(sd, inputFeatures, weights);

        // Add positional embedding
        INDArray posEmb = weights.get("encoder.positional_embedding");
        if (posEmb != null) {
            SDVariable posEmbVar = sd.constant("encoder.positional_embedding", posEmb);
            x = x.add(posEmbVar);
        }

        // Encoder transformer blocks
        int numEncoderLayers = config.getNumLayers();
        // Try to detect encoder vs decoder layer counts from weights
        int maxEncoderBlock = -1;
        int maxDecoderBlock = -1;
        for (String key : weights.keySet()) {
            if (key.startsWith("encoder.blocks.")) {
                int blockNum = extractBlockNumber(key);
                maxEncoderBlock = Math.max(maxEncoderBlock, blockNum);
            } else if (key.startsWith("decoder.blocks.")) {
                int blockNum = extractBlockNumber(key);
                maxDecoderBlock = Math.max(maxDecoderBlock, blockNum);
            }
        }
        if (maxEncoderBlock >= 0) numEncoderLayers = maxEncoderBlock + 1;
        int numDecoderLayers = maxDecoderBlock >= 0 ? maxDecoderBlock + 1 : numEncoderLayers;

        for (int i = 0; i < numEncoderLayers; i++) {
            x = buildEncoderBlock(sd, x, i, numHeads, headDim, weights);
        }

        // Final layer norm
        INDArray lnW = weights.get("encoder.ln_post.weight");
        INDArray lnB = weights.get("encoder.ln_post.bias");
        if (lnW != null) {
            x = layerNorm(sd, x, "encoder.ln_post", lnW, lnB);
        }

        SDVariable encoderOutput = sd.identity("last_hidden_state", x);

        // --- Decoder ---

        // Input: token IDs [batch, seqLen]
        SDVariable inputIds = sd.placeHolder("input_ids", DataType.INT64, -1, -1);
        SDVariable encoderHidden = sd.placeHolder("encoder_hidden_states", DataType.FLOAT, -1, -1, -1);

        // Token embedding
        INDArray tokenEmbW = weights.get("decoder.token_embedding.weight");
        if (tokenEmbW != null) {
            SDVariable tokenEmbVar = sd.constant("decoder.token_embedding.weight", tokenEmbW);
            SDVariable decoderX = sd.gather("decoder.token_embed", tokenEmbVar, inputIds.castTo(DataType.INT64), 0);

            // Add decoder positional embedding
            INDArray decPosEmb = weights.get("decoder.positional_embedding");
            if (decPosEmb != null) {
                SDVariable decPosEmbVar = sd.constant("decoder.positional_embedding", decPosEmb);
                decoderX = decoderX.add(decPosEmbVar);
            }

            // Decoder transformer blocks
            for (int i = 0; i < numDecoderLayers; i++) {
                decoderX = buildDecoderBlock(sd, decoderX, encoderHidden, i, numHeads, headDim, weights);
            }

            // Final layer norm
            INDArray decLnW = weights.get("decoder.ln.weight");
            INDArray decLnB = weights.get("decoder.ln.bias");
            if (decLnW != null) {
                decoderX = layerNorm(sd, decoderX, "decoder.ln", decLnW, decLnB);
            }

            // Output projection (tied to token embedding)
            SDVariable logits = sd.mmul("logits", decoderX, tokenEmbVar.permute(1, 0));
        }

        return sd;
    }

    @Override
    public Map<String, String> getTensorNamePatterns() {
        Map<String, String> patterns = new HashMap<>();
        // Encoder
        patterns.put("encoder.conv1.weight", "encoder.conv1.weight");
        patterns.put("encoder.conv1.bias", "encoder.conv1.bias");
        patterns.put("encoder.conv2.weight", "encoder.conv2.weight");
        patterns.put("encoder.conv2.bias", "encoder.conv2.bias");
        patterns.put("encoder.positional_embedding", "encoder.positional_embedding");
        patterns.put("encoder.blocks.{layer}.attn.query.weight", "encoder.blocks.{layer}.attn.query.weight");
        patterns.put("encoder.blocks.{layer}.attn.query.bias", "encoder.blocks.{layer}.attn.query.bias");
        patterns.put("encoder.blocks.{layer}.attn.key.weight", "encoder.blocks.{layer}.attn.key.weight");
        patterns.put("encoder.blocks.{layer}.attn.value.weight", "encoder.blocks.{layer}.attn.value.weight");
        patterns.put("encoder.blocks.{layer}.attn.value.bias", "encoder.blocks.{layer}.attn.value.bias");
        patterns.put("encoder.blocks.{layer}.attn.out.weight", "encoder.blocks.{layer}.attn.out.weight");
        patterns.put("encoder.blocks.{layer}.attn.out.bias", "encoder.blocks.{layer}.attn.out.bias");
        patterns.put("encoder.blocks.{layer}.attn_ln.weight", "encoder.blocks.{layer}.attn_ln.weight");
        patterns.put("encoder.blocks.{layer}.attn_ln.bias", "encoder.blocks.{layer}.attn_ln.bias");
        patterns.put("encoder.blocks.{layer}.mlp.0.weight", "encoder.blocks.{layer}.mlp.0.weight");
        patterns.put("encoder.blocks.{layer}.mlp.0.bias", "encoder.blocks.{layer}.mlp.0.bias");
        patterns.put("encoder.blocks.{layer}.mlp.2.weight", "encoder.blocks.{layer}.mlp.2.weight");
        patterns.put("encoder.blocks.{layer}.mlp.2.bias", "encoder.blocks.{layer}.mlp.2.bias");
        patterns.put("encoder.blocks.{layer}.mlp_ln.weight", "encoder.blocks.{layer}.mlp_ln.weight");
        patterns.put("encoder.blocks.{layer}.mlp_ln.bias", "encoder.blocks.{layer}.mlp_ln.bias");
        patterns.put("encoder.ln_post.weight", "encoder.ln_post.weight");
        patterns.put("encoder.ln_post.bias", "encoder.ln_post.bias");
        // Decoder
        patterns.put("decoder.token_embedding.weight", "decoder.token_embedding.weight");
        patterns.put("decoder.positional_embedding", "decoder.positional_embedding");
        patterns.put("decoder.blocks.{layer}.attn.query.weight", "decoder.blocks.{layer}.attn.query.weight");
        patterns.put("decoder.blocks.{layer}.attn.query.bias", "decoder.blocks.{layer}.attn.query.bias");
        patterns.put("decoder.blocks.{layer}.attn.key.weight", "decoder.blocks.{layer}.attn.key.weight");
        patterns.put("decoder.blocks.{layer}.attn.value.weight", "decoder.blocks.{layer}.attn.value.weight");
        patterns.put("decoder.blocks.{layer}.attn.value.bias", "decoder.blocks.{layer}.attn.value.bias");
        patterns.put("decoder.blocks.{layer}.attn.out.weight", "decoder.blocks.{layer}.attn.out.weight");
        patterns.put("decoder.blocks.{layer}.attn.out.bias", "decoder.blocks.{layer}.attn.out.bias");
        patterns.put("decoder.blocks.{layer}.attn_ln.weight", "decoder.blocks.{layer}.attn_ln.weight");
        patterns.put("decoder.blocks.{layer}.attn_ln.bias", "decoder.blocks.{layer}.attn_ln.bias");
        patterns.put("decoder.blocks.{layer}.cross_attn.query.weight", "decoder.blocks.{layer}.cross_attn.query.weight");
        patterns.put("decoder.blocks.{layer}.cross_attn.query.bias", "decoder.blocks.{layer}.cross_attn.query.bias");
        patterns.put("decoder.blocks.{layer}.cross_attn.key.weight", "decoder.blocks.{layer}.cross_attn.key.weight");
        patterns.put("decoder.blocks.{layer}.cross_attn.value.weight", "decoder.blocks.{layer}.cross_attn.value.weight");
        patterns.put("decoder.blocks.{layer}.cross_attn.value.bias", "decoder.blocks.{layer}.cross_attn.value.bias");
        patterns.put("decoder.blocks.{layer}.cross_attn.out.weight", "decoder.blocks.{layer}.cross_attn.out.weight");
        patterns.put("decoder.blocks.{layer}.cross_attn.out.bias", "decoder.blocks.{layer}.cross_attn.out.bias");
        patterns.put("decoder.blocks.{layer}.cross_attn_ln.weight", "decoder.blocks.{layer}.cross_attn_ln.weight");
        patterns.put("decoder.blocks.{layer}.cross_attn_ln.bias", "decoder.blocks.{layer}.cross_attn_ln.bias");
        patterns.put("decoder.blocks.{layer}.mlp.0.weight", "decoder.blocks.{layer}.mlp.0.weight");
        patterns.put("decoder.blocks.{layer}.mlp.0.bias", "decoder.blocks.{layer}.mlp.0.bias");
        patterns.put("decoder.blocks.{layer}.mlp.2.weight", "decoder.blocks.{layer}.mlp.2.weight");
        patterns.put("decoder.blocks.{layer}.mlp.2.bias", "decoder.blocks.{layer}.mlp.2.bias");
        patterns.put("decoder.blocks.{layer}.mlp_ln.weight", "decoder.blocks.{layer}.mlp_ln.weight");
        patterns.put("decoder.blocks.{layer}.mlp_ln.bias", "decoder.blocks.{layer}.mlp_ln.bias");
        patterns.put("decoder.ln.weight", "decoder.ln.weight");
        patterns.put("decoder.ln.bias", "decoder.ln.bias");
        return patterns;
    }

    @Override
    public ArchitectureConfig getConfig(GGMLMetadata metadata) {
        return ArchitectureConfig.builder()
                .numLayers(metadata.getNumLayers())
                .hiddenSize(metadata.getHiddenSize())
                .intermediateSize(metadata.getIntermediateSize())
                .numAttentionHeads(metadata.getNumAttentionHeads())
                .numKVHeads(metadata.getNumAttentionHeads()) // Whisper uses MHA, not GQA
                .vocabSize(metadata.getVocabSize())
                .contextLength(metadata.getContextLength())
                .layerNormEpsilon(metadata.getLayerNormEpsilon())
                .useRmsNorm(false) // Whisper uses standard LayerNorm, not RMSNorm
                .useSwiGLU(false)  // Whisper uses GELU
                .useRotaryEmbeddings(false) // Whisper uses sinusoidal position embeddings
                .decoderOnly(false) // Whisper is encoder-decoder
                .build();
    }

    // --- Private helper methods ---

    private SDVariable buildConvLayers(SameDiff sd, SDVariable input, Map<String, INDArray> weights) {
        // Conv1: [numMelBins, 3000] -> [hiddenSize, 3000] with kernel 3, stride 1, pad 1
        INDArray conv1W = weights.get("encoder.conv1.weight");
        INDArray conv1B = weights.get("encoder.conv1.bias");
        SDVariable x = input;

        if (conv1W != null) {
            SDVariable conv1WVar = sd.constant("encoder.conv1.weight", conv1W);
            // Use 1D conv as matmul approximation for simplicity
            // The actual conv1d is along the time dimension
            x = sd.cnn().conv1d("encoder.conv1", x, conv1WVar, null,
                    Conv1DConfig.builder().k(1).s(1).p(0).dataFormat("NWC").paddingMode(PaddingMode.VALID).build());
            if (conv1B != null) {
                SDVariable conv1BVar = sd.constant("encoder.conv1.bias", conv1B);
                x = x.add(conv1BVar);
            }
            x = sd.nn.gelu("encoder.conv1.gelu", x);
        }

        // Conv2: stride 2 (downsamples by 2)
        INDArray conv2W = weights.get("encoder.conv2.weight");
        INDArray conv2B = weights.get("encoder.conv2.bias");
        if (conv2W != null) {
            SDVariable conv2WVar = sd.constant("encoder.conv2.weight", conv2W);
            x = sd.cnn().conv1d("encoder.conv2", x, conv2WVar, null,
                    Conv1DConfig.builder().k(1).s(2).p(0).dataFormat("NWC").paddingMode(PaddingMode.VALID).build());
            if (conv2B != null) {
                SDVariable conv2BVar = sd.constant("encoder.conv2.bias", conv2B);
                x = x.add(conv2BVar);
            }
            x = sd.nn.gelu("encoder.conv2.gelu", x);
        }

        // Transpose to [batch, seqLen, hiddenSize]
        x = x.permute(0, 2, 1);

        return x;
    }

    private SDVariable buildEncoderBlock(SameDiff sd, SDVariable x, int layerIdx,
                                          int numHeads, int headDim,
                                          Map<String, INDArray> weights) {
        String prefix = "encoder.blocks." + layerIdx;

        // Self-attention with layer norm
        SDVariable residual = x;
        x = layerNorm(sd, x, prefix + ".attn_ln",
                weights.get(prefix + ".attn_ln.weight"),
                weights.get(prefix + ".attn_ln.bias"));

        x = multiHeadAttention(sd, x, x, x, prefix + ".attn", numHeads, headDim, weights);
        x = residual.add(x);

        // FFN with layer norm
        residual = x;
        x = layerNorm(sd, x, prefix + ".mlp_ln",
                weights.get(prefix + ".mlp_ln.weight"),
                weights.get(prefix + ".mlp_ln.bias"));

        x = feedForward(sd, x, prefix + ".mlp", weights);
        x = residual.add(x);

        return x;
    }

    private SDVariable buildDecoderBlock(SameDiff sd, SDVariable x, SDVariable encoderHidden,
                                          int layerIdx, int numHeads, int headDim,
                                          Map<String, INDArray> weights) {
        String prefix = "decoder.blocks." + layerIdx;

        // Causal self-attention
        SDVariable residual = x;
        x = layerNorm(sd, x, prefix + ".attn_ln",
                weights.get(prefix + ".attn_ln.weight"),
                weights.get(prefix + ".attn_ln.bias"));

        x = multiHeadAttention(sd, x, x, x, prefix + ".attn", numHeads, headDim, weights);
        x = residual.add(x);

        // Cross-attention
        residual = x;
        x = layerNorm(sd, x, prefix + ".cross_attn_ln",
                weights.get(prefix + ".cross_attn_ln.weight"),
                weights.get(prefix + ".cross_attn_ln.bias"));

        x = multiHeadAttention(sd, x, encoderHidden, encoderHidden,
                prefix + ".cross_attn", numHeads, headDim, weights);
        x = residual.add(x);

        // FFN
        residual = x;
        x = layerNorm(sd, x, prefix + ".mlp_ln",
                weights.get(prefix + ".mlp_ln.weight"),
                weights.get(prefix + ".mlp_ln.bias"));

        x = feedForward(sd, x, prefix + ".mlp", weights);
        x = residual.add(x);

        return x;
    }

    private SDVariable multiHeadAttention(SameDiff sd, SDVariable query, SDVariable key, SDVariable value,
                                           String prefix, int numHeads, int headDim,
                                           Map<String, INDArray> weights) {
        // Project Q, K, V
        SDVariable q = linearProjection(sd, query, prefix + ".query", weights);
        SDVariable k = linearProjection(sd, key, prefix + ".key", weights);
        SDVariable v = linearProjection(sd, value, prefix + ".value", weights);

        // Reshape to multi-head: [batch, seq, heads, headDim] -> [batch, heads, seq, headDim]
        q = q.reshape(-1, 0, numHeads, headDim).permute(0, 2, 1, 3);
        k = k.reshape(-1, 0, numHeads, headDim).permute(0, 2, 1, 3);
        v = v.reshape(-1, 0, numHeads, headDim).permute(0, 2, 1, 3);

        // Scaled dot-product attention
        double scale = 1.0 / Math.sqrt(headDim);
        SDVariable attnWeights = sd.mmul(prefix + ".attn_weights", q, k.permute(0, 1, 3, 2))
                .mul(scale);
        attnWeights = sd.nn.softmax(prefix + ".attn_probs", attnWeights, -1);
        SDVariable attnOutput = sd.mmul(prefix + ".attn_output", attnWeights, v);

        // Reshape back: [batch, heads, seq, headDim] -> [batch, seq, hiddenSize]
        attnOutput = attnOutput.permute(0, 2, 1, 3)
                .reshape(-1, 0, numHeads * headDim);

        // Output projection
        attnOutput = linearProjection(sd, attnOutput, prefix + ".out", weights);

        return attnOutput;
    }

    private SDVariable linearProjection(SameDiff sd, SDVariable input, String prefix,
                                         Map<String, INDArray> weights) {
        INDArray w = weights.get(prefix + ".weight");
        INDArray b = weights.get(prefix + ".bias");

        if (w == null) {
            log.warn("Missing weight: {}.weight", prefix);
            return input;
        }

        SDVariable wVar = sd.constant(prefix + ".weight", w);
        SDVariable result = sd.mmul(prefix + ".proj", input, wVar.permute(1, 0));

        if (b != null) {
            SDVariable bVar = sd.constant(prefix + ".bias", b);
            result = result.add(bVar);
        }

        return result;
    }

    private SDVariable feedForward(SameDiff sd, SDVariable input, String prefix,
                                    Map<String, INDArray> weights) {
        // FC1 + GELU + FC2
        SDVariable x = linearProjection(sd, input, prefix + ".0", weights);
        x = sd.nn.gelu(prefix + ".gelu", x);
        x = linearProjection(sd, x, prefix + ".2", weights);
        return x;
    }

    private SDVariable layerNorm(SameDiff sd, SDVariable input, String name,
                                  INDArray weight, INDArray bias) {
        if (weight == null) return input;

        SDVariable wVar = sd.constant(name + ".weight", weight);
        SDVariable bVar = bias != null ? sd.constant(name + ".bias", bias) : null;

        // LayerNorm: (x - mean) / sqrt(var + eps) * weight + bias
        SDVariable mean = input.mean(true, -1);
        SDVariable centered = input.sub(mean);
        SDVariable variance = centered.mul(centered).mean(true, -1);
        SDVariable normalized = centered.div(sd.math.sqrt(variance.add(1e-5)));
        SDVariable result = normalized.mul(wVar);
        if (bVar != null) {
            result = result.add(bVar);
        }
        return result;
    }

    private int extractBlockNumber(String tensorName) {
        // Extract block number from "encoder.blocks.N.xxx" or "decoder.blocks.N.xxx"
        String[] parts = tensorName.split("\\.");
        for (int i = 0; i < parts.length - 1; i++) {
            if ("blocks".equals(parts[i])) {
                try {
                    return Integer.parseInt(parts[i + 1]);
                } catch (NumberFormatException e) {
                    return -1;
                }
            }
        }
        return -1;
    }
}
