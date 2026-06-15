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

package org.eclipse.deeplearning4j.ggml.architecture;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.ggml.architecture.ArchitectureConfig;
import org.nd4j.ggml.architecture.ArchitectureRegistry;
import org.nd4j.ggml.architecture.ModelArchitecture;
import org.nd4j.ggml.architecture.WhisperArchitecture;
import org.nd4j.ggml.convert.ConversionOptions;
import org.nd4j.ggml.format.GGMLMetadata;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for WhisperArchitecture — the only encoder-decoder architecture.
 * Validates registration, config, tensor patterns, and graph construction.
 */
@Slf4j
@DisplayName("WhisperArchitecture Tests")
public class TestWhisperArchitecture {

    @Test
    @DisplayName("Registered in ArchitectureRegistry")
    void testRegistration() {
        assertTrue(ArchitectureRegistry.hasArchitecture("whisper"));
        assertTrue(ArchitectureRegistry.hasArchitecture("whisper.encoder"));
        assertTrue(ArchitectureRegistry.hasArchitecture("whisper.decoder"));

        ModelArchitecture arch = ArchitectureRegistry.getArchitecture("whisper");
        assertNotNull(arch);
        assertInstanceOf(WhisperArchitecture.class, arch);
        assertEquals("whisper", arch.getName());
    }

    @Test
    @DisplayName("Supported variants")
    void testSupportedVariants() {
        ModelArchitecture arch = ArchitectureRegistry.getArchitecture("whisper");
        Set<String> variants = arch.getSupportedVariants();
        assertTrue(variants.contains("whisper"));
        assertTrue(variants.contains("whisper.encoder"));
        assertTrue(variants.contains("whisper.decoder"));
    }

    @Test
    @DisplayName("canHandle detects whisper from architecture name")
    void testCanHandleByArchName() {
        WhisperArchitecture arch = new WhisperArchitecture();

        GGMLMetadata meta = createMetadata("whisper");
        assertTrue(arch.canHandle(meta));

        meta = createMetadata("whisper.encoder");
        assertTrue(arch.canHandle(meta));

        meta = createMetadata("llama");
        assertFalse(arch.canHandle(meta));

        meta = createMetadata(null);
        assertFalse(arch.canHandle(meta));
    }

    @Test
    @DisplayName("Config: encoder-decoder specific settings")
    void testConfig() {
        WhisperArchitecture arch = new WhisperArchitecture();
        GGMLMetadata meta = createMetadata("whisper", 4, 512, 2048, 8);

        ArchitectureConfig config = arch.getConfig(meta);
        assertEquals(4, config.getNumLayers());
        assertEquals(512, config.getHiddenSize());
        assertEquals(2048, config.getIntermediateSize());
        assertEquals(8, config.getNumAttentionHeads());
        assertEquals(8, config.getNumKVHeads(), "Whisper uses MHA, KV heads = num heads");
        assertFalse(config.isUseRmsNorm(), "Whisper uses LayerNorm, not RMSNorm");
        assertFalse(config.isUseSwiGLU(), "Whisper uses GELU, not SwiGLU");
        assertFalse(config.isUseRotaryEmbeddings(), "Whisper uses sinusoidal pos embeddings");
        assertFalse(config.isDecoderOnly(), "Whisper is encoder-decoder");
    }

    @Test
    @DisplayName("Chat template: none (speech-to-text)")
    void testChatTemplate() {
        WhisperArchitecture arch = new WhisperArchitecture();
        assertEquals("none", arch.getDefaultChatTemplateType());
        assertEquals("whisper.gguf.path", arch.getModelSystemProperty());
        assertEquals("", arch.getReferencePrompt());
        assertEquals(0, arch.getReferenceExpected().length);
    }

    @Test
    @DisplayName("Tensor name patterns include encoder and decoder")
    void testTensorPatterns() {
        WhisperArchitecture arch = new WhisperArchitecture();
        Map<String, String> patterns = arch.getTensorNamePatterns();

        // Encoder patterns
        assertNotNull(patterns.get("encoder.conv1.weight"), "Missing encoder.conv1.weight");
        assertNotNull(patterns.get("encoder.conv2.weight"), "Missing encoder.conv2.weight");
        assertNotNull(patterns.get("encoder.positional_embedding"), "Missing encoder pos emb");
        assertNotNull(patterns.get("encoder.blocks.{layer}.attn.query.weight"), "Missing encoder query");
        assertNotNull(patterns.get("encoder.blocks.{layer}.attn.key.weight"), "Missing encoder key");
        assertNotNull(patterns.get("encoder.blocks.{layer}.attn.value.weight"), "Missing encoder value");
        assertNotNull(patterns.get("encoder.blocks.{layer}.attn.out.weight"), "Missing encoder out");
        assertNotNull(patterns.get("encoder.blocks.{layer}.mlp.0.weight"), "Missing encoder mlp fc1");
        assertNotNull(patterns.get("encoder.blocks.{layer}.mlp.2.weight"), "Missing encoder mlp fc2");
        assertNotNull(patterns.get("encoder.ln_post.weight"), "Missing encoder ln_post");

        // Decoder patterns
        assertNotNull(patterns.get("decoder.token_embedding.weight"), "Missing decoder embedding");
        assertNotNull(patterns.get("decoder.positional_embedding"), "Missing decoder pos emb");
        assertNotNull(patterns.get("decoder.blocks.{layer}.attn.query.weight"), "Missing decoder query");
        assertNotNull(patterns.get("decoder.blocks.{layer}.cross_attn.query.weight"), "Missing cross-attn query");
        assertNotNull(patterns.get("decoder.blocks.{layer}.cross_attn.key.weight"), "Missing cross-attn key");
        assertNotNull(patterns.get("decoder.blocks.{layer}.cross_attn.value.weight"), "Missing cross-attn value");
        assertNotNull(patterns.get("decoder.ln.weight"), "Missing decoder ln");
    }

    @Test
    @DisplayName("buildGraph with synthetic weights produces valid graph")
    void testBuildGraphSyntheticWeights() {
        WhisperArchitecture arch = new WhisperArchitecture();
        int numLayers = 2;
        int hiddenSize = 64;
        int numHeads = 4;
        int headDim = hiddenSize / numHeads;
        int intermediateSize = 256;
        int vocabSize = 100;
        int numMelBins = 80;
        int maxSeqLen = 64;

        Map<String, INDArray> weights = new HashMap<>();

        // Encoder conv layers
        weights.put("encoder.conv1.weight", Nd4j.randn(DataType.FLOAT, hiddenSize, numMelBins, 1));
        weights.put("encoder.conv1.bias", Nd4j.randn(DataType.FLOAT, hiddenSize));
        weights.put("encoder.conv2.weight", Nd4j.randn(DataType.FLOAT, hiddenSize, hiddenSize, 1));
        weights.put("encoder.conv2.bias", Nd4j.randn(DataType.FLOAT, hiddenSize));

        // Positional embeddings
        weights.put("encoder.positional_embedding", Nd4j.randn(DataType.FLOAT, maxSeqLen, hiddenSize));
        weights.put("decoder.positional_embedding", Nd4j.randn(DataType.FLOAT, maxSeqLen, hiddenSize));

        // Token embedding
        weights.put("decoder.token_embedding.weight", Nd4j.randn(DataType.FLOAT, vocabSize, hiddenSize));

        // Encoder and decoder blocks
        for (int i = 0; i < numLayers; i++) {
            // Encoder block
            addAttentionWeights(weights, "encoder.blocks." + i + ".attn", hiddenSize, numHeads, headDim);
            addLayerNormWeights(weights, "encoder.blocks." + i + ".attn_ln", hiddenSize);
            addFFNWeights(weights, "encoder.blocks." + i + ".mlp", hiddenSize, intermediateSize);
            addLayerNormWeights(weights, "encoder.blocks." + i + ".mlp_ln", hiddenSize);

            // Decoder block - self attention
            addAttentionWeights(weights, "decoder.blocks." + i + ".attn", hiddenSize, numHeads, headDim);
            addLayerNormWeights(weights, "decoder.blocks." + i + ".attn_ln", hiddenSize);

            // Decoder block - cross attention
            addAttentionWeights(weights, "decoder.blocks." + i + ".cross_attn", hiddenSize, numHeads, headDim);
            addLayerNormWeights(weights, "decoder.blocks." + i + ".cross_attn_ln", hiddenSize);

            // Decoder FFN
            addFFNWeights(weights, "decoder.blocks." + i + ".mlp", hiddenSize, intermediateSize);
            addLayerNormWeights(weights, "decoder.blocks." + i + ".mlp_ln", hiddenSize);
        }

        // Post-norms
        addLayerNormWeights(weights, "encoder.ln_post", hiddenSize);
        addLayerNormWeights(weights, "decoder.ln", hiddenSize);

        GGMLMetadata meta = createMetadata("whisper", numLayers, hiddenSize, intermediateSize, numHeads);
        ConversionOptions opts = ConversionOptions.forInference();

        SameDiff sd = arch.buildGraph(meta, weights, opts);
        assertNotNull(sd, "buildGraph returned null");

        // Verify placeholders
        assertNotNull(sd.getVariable("input_features"), "Missing input_features placeholder");
        assertNotNull(sd.getVariable("input_ids"), "Missing input_ids placeholder");
        assertNotNull(sd.getVariable("encoder_hidden_states"), "Missing encoder_hidden_states placeholder");

        // Verify encoder output
        assertNotNull(sd.getVariable("last_hidden_state"), "Missing encoder output");

        // Verify logits
        assertNotNull(sd.getVariable("logits"), "Missing decoder logits");

        log.info("Whisper graph built successfully: {} variables", sd.variables().size());
    }

    // ========================================================================
    // Helpers
    // ========================================================================

    private void addAttentionWeights(Map<String, INDArray> weights, String prefix,
                                      int hiddenSize, int numHeads, int headDim) {
        int projSize = numHeads * headDim;
        weights.put(prefix + ".query.weight", Nd4j.randn(DataType.FLOAT, projSize, hiddenSize));
        weights.put(prefix + ".query.bias", Nd4j.randn(DataType.FLOAT, projSize));
        weights.put(prefix + ".key.weight", Nd4j.randn(DataType.FLOAT, projSize, hiddenSize));
        weights.put(prefix + ".value.weight", Nd4j.randn(DataType.FLOAT, projSize, hiddenSize));
        weights.put(prefix + ".value.bias", Nd4j.randn(DataType.FLOAT, projSize));
        weights.put(prefix + ".out.weight", Nd4j.randn(DataType.FLOAT, hiddenSize, projSize));
        weights.put(prefix + ".out.bias", Nd4j.randn(DataType.FLOAT, hiddenSize));
    }

    private void addLayerNormWeights(Map<String, INDArray> weights, String prefix, int dim) {
        weights.put(prefix + ".weight", Nd4j.ones(DataType.FLOAT, dim));
        weights.put(prefix + ".bias", Nd4j.zeros(DataType.FLOAT, dim));
    }

    private void addFFNWeights(Map<String, INDArray> weights, String prefix,
                                int hiddenSize, int intermediateSize) {
        weights.put(prefix + ".0.weight", Nd4j.randn(DataType.FLOAT, intermediateSize, hiddenSize));
        weights.put(prefix + ".0.bias", Nd4j.randn(DataType.FLOAT, intermediateSize));
        weights.put(prefix + ".2.weight", Nd4j.randn(DataType.FLOAT, hiddenSize, intermediateSize));
        weights.put(prefix + ".2.bias", Nd4j.randn(DataType.FLOAT, hiddenSize));
    }

    private GGMLMetadata createMetadata(String architecture) {
        return createMetadata(architecture, 4, 512, 2048, 8);
    }

    private GGMLMetadata createMetadata(String architecture, int numLayers, int hiddenSize,
                                         int intermediateSize, int numHeads) {
        return GGMLMetadata.builder()
                .architecture(architecture)
                .numLayers(numLayers)
                .hiddenSize(hiddenSize)
                .intermediateSize(intermediateSize)
                .numAttentionHeads(numHeads)
                .numKVHeads(numHeads)
                .vocabSize(51865)
                .contextLength(448)
                .layerNormEpsilon(1e-5f)
                .build();
    }
}
