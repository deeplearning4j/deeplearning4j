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

package org.eclipse.deeplearning4j.ggml;

import org.eclipse.deeplearning4j.llm.eval.GenerationQualityValidator;
import org.eclipse.deeplearning4j.llm.eval.GenerationQualityValidator.QualityReport;
import org.eclipse.deeplearning4j.llm.eval.GenerationQualityValidator.ValidationConfig;
import org.eclipse.deeplearning4j.llm.generation.GenerationPipeline;
import org.eclipse.deeplearning4j.llm.generation.GenerationPipelineConfig;
import org.eclipse.deeplearning4j.llm.generation.GenerationResult;
import org.eclipse.deeplearning4j.llm.generation.ModelIOConfig;
import org.eclipse.deeplearning4j.llm.generation.sampling.SamplingConfig;
import org.eclipse.deeplearning4j.llm.tokenizer.HuggingFaceTokenizer;
import org.eclipse.deeplearning4j.llm.tokenizer.Tokenizer;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.serde.SameDiffSerializer;
import org.nd4j.ggml.GGMLModelImport;
import org.nd4j.ggml.architecture.ArchitectureConfig;
import org.nd4j.ggml.architecture.ArchitectureRegistry;
import org.nd4j.ggml.architecture.LFM2Architecture;
import org.nd4j.ggml.architecture.ModelArchitecture;
import org.nd4j.ggml.convert.ConversionOptions;
import org.nd4j.ggml.format.GGMLMetadata;
import org.nd4j.ggml.format.GGMLTensorInfo;
import org.nd4j.ggml.format.GGUFHeader;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Tests for LFM2Architecture: registration, per-layer KV head counts,
 * GQA reshape correctness, short-conv block, QK norms, and full forward pass.
 *
 * <p>The LFM-2.5 1.2B model has these key properties:</p>
 * <ul>
 *   <li>16 layers: 10 short-conv + 6 attention (interleaved)</li>
 *   <li>Per-layer KV heads: [0,0,8,0,0,8,0,0,8,0,0,8,0,0,8,0]</li>
 *   <li>Short-conv uses fused in_proj (3x expansion), split into B,C,x gates</li>
 *   <li>Attention uses per-head QK RMSNorm before RoPE</li>
 *   <li>Output norm is token_embd_norm (misnomer — it's the post-stack norm)</li>
 *   <li>Tied embeddings (no separate output.weight)</li>
 * </ul>
 */
@DisplayName("LFM2 Architecture Test")
class TestLFM2Architecture {

    // LFM-2.5 1.2B dimensions (from GGUF metadata)
    private static final int HIDDEN_SIZE = 2048;
    private static final int NUM_HEADS = 32;
    private static final int NUM_KV_HEADS = 8;
    private static final int HEAD_DIM = HIDDEN_SIZE / NUM_HEADS; // 64
    private static final int NUM_LAYERS = 16;
    private static final int VOCAB_SIZE = 65536;
    private static final int INTERMEDIATE_SIZE = 8192;
    private static final int CONV_KERNEL_SIZE = 3;

    // Per-layer KV head counts: 0 = conv layer, 8 = attention layer
    private static final List<Integer> KV_HEADS_PER_LAYER = Arrays.asList(
            0, 0, 8, 0, 0, 8, 0, 0, 8, 0, 0, 8, 0, 0, 8, 0
    );

    // Per-layer types matching the KV head pattern
    private static final List<String> LAYER_TYPES = Arrays.asList(
            "short_conv", "short_conv", "attention",
            "short_conv", "short_conv", "attention",
            "short_conv", "short_conv", "attention",
            "short_conv", "short_conv", "attention",
            "short_conv", "short_conv", "attention",
            "short_conv"
    );

    // Each @Test builds a SameDiff model + large (512MB) synthetic weight maps as
    // LOCALS that are never explicitly closed. Their off-heap GPU memory lingers in
    // the DeallocatorService refMap (System.gc alone won't reclaim it in time), so the
    // 25-test suite accumulates device memory until later 512MB allocations OOM even
    // though the GPU is otherwise near-empty. Force-flush dead references + trim the
    // device pool between tests — the same robust lifecycle cleanup TestQwen35Pipeline
    // applies between configs. Best-effort: teardown must never fail a test.
    @AfterEach
    void freeGpuBetweenTests() {
        try {
            System.gc();
            Thread.sleep(50);  // let GC enqueue this test's dead locals' PhantomReferences
            var nativeOps = org.nd4j.nativeblas.NativeOpsHolder.getInstance().getDeviceNativeOps();
            Nd4j.getDeallocatorService().forceFlushAll();
            Nd4j.getMemoryManager().purgeCaches();
            Nd4j.getExecutioner().commit();
            int numDevices = org.nd4j.linalg.api.device.DeviceMemoryManager.getInstance()
                    .getContextProvider().getDeviceCount();
            for (int d = 0; d < numDevices; d++) {
                nativeOps.trimMemoryPool(d);
            }
        } catch (Throwable ignored) {
            // best-effort GPU reclaim; never fail a test on teardown
        }
    }

    // =========================================================================
    // Registration and variant detection
    // =========================================================================

    @Test
    @DisplayName("LFM2Architecture is registered in ArchitectureRegistry")
    void testLFM2Registered() {
        ModelArchitecture arch = ArchitectureRegistry.getArchitecture("lfm2");
        assertNotNull(arch, "LFM2 architecture should be registered");
        assertEquals("lfm2", arch.getName());
    }

    @Test
    @DisplayName("LFM2Architecture can be found by all supported variant names")
    void testLFM2VariantLookup() {
        for (String variant : new String[]{"lfm2", "lfm2moe"}) {
            ModelArchitecture arch = ArchitectureRegistry.getArchitecture(variant);
            assertNotNull(arch, "Expected variant '" + variant + "' to be registered");
            assertEquals("lfm2", arch.getName());
        }
    }

    @Test
    @DisplayName("LFM2Architecture.getSupportedVariants() returns expected set")
    void testLFM2SupportedVariants() {
        LFM2Architecture arch = new LFM2Architecture();
        Set<String> variants = arch.getSupportedVariants();
        assertNotNull(variants);
        assertTrue(variants.contains("lfm2"));
        assertTrue(variants.contains("lfm2moe"));
    }

    @Test
    @DisplayName("LFM2Architecture.canHandle() returns true for lfm2 arch")
    void testLFM2CanHandle() {
        LFM2Architecture arch = new LFM2Architecture();
        assertTrue(arch.canHandle(GGMLMetadata.builder().architecture("lfm2").build()));
        assertTrue(arch.canHandle(GGMLMetadata.builder().architecture("LFM2").build()));
        assertTrue(arch.canHandle(GGMLMetadata.builder().architecture("liquid").build()));
    }

    @Test
    @DisplayName("LFM2Architecture.canHandle() returns false for unrelated architectures")
    void testLFM2CanNotHandle() {
        LFM2Architecture arch = new LFM2Architecture();
        assertFalse(arch.canHandle(GGMLMetadata.builder().architecture("llama").build()));
        assertFalse(arch.canHandle(GGMLMetadata.builder().architecture(null).build()));
    }

    // =========================================================================
    // Tensor name patterns
    // =========================================================================

    @Test
    @DisplayName("LFM2Architecture.getTensorNamePatterns() contains correct GGUF keys")
    void testLFM2TensorNamePatterns() {
        LFM2Architecture arch = new LFM2Architecture();
        Map<String, String> patterns = arch.getTensorNamePatterns();

        assertNotNull(patterns);
        assertFalse(patterns.isEmpty());

        // Embeddings (token_embd_norm is the output norm in LFM2 GGUF)
        assertTrue(patterns.containsKey("token_embd.weight"));
        assertTrue(patterns.containsKey("token_embd_norm.weight"));

        // Attention
        assertTrue(patterns.containsKey("blk.{layer}.attn_q.weight"));
        assertTrue(patterns.containsKey("blk.{layer}.attn_k.weight"));
        assertTrue(patterns.containsKey("blk.{layer}.attn_v.weight"));
        assertTrue(patterns.containsKey("blk.{layer}.attn_output.weight"));

        // Per-head QK norms
        assertTrue(patterns.containsKey("blk.{layer}.attn_q_norm.weight"),
                "Missing QK norm pattern");
        assertTrue(patterns.containsKey("blk.{layer}.attn_k_norm.weight"),
                "Missing QK norm pattern");

        // Short-conv block (actual GGUF tensor names with shortconv. prefix)
        assertTrue(patterns.containsKey("blk.{layer}.shortconv.in_proj.weight"),
                "Missing shortconv.in_proj pattern");
        assertTrue(patterns.containsKey("blk.{layer}.shortconv.out_proj.weight"),
                "Missing shortconv.out_proj pattern");
        assertTrue(patterns.containsKey("blk.{layer}.shortconv.conv.weight"),
                "Missing shortconv.conv pattern");

        // FFN
        assertTrue(patterns.containsKey("blk.{layer}.ffn_gate.weight"));
        assertTrue(patterns.containsKey("blk.{layer}.ffn_up.weight"));
        assertTrue(patterns.containsKey("blk.{layer}.ffn_down.weight"));

        // Norms
        assertTrue(patterns.containsKey("blk.{layer}.attn_norm.weight"));
        assertTrue(patterns.containsKey("blk.{layer}.ffn_norm.weight"));
    }

    // =========================================================================
    // GGUFHeader per-layer KV head array parsing
    // =========================================================================

    @Test
    @DisplayName("GGUFHeader.getAttentionHeadCountKV() handles per-layer int[] array")
    void testGGUFHeaderPerLayerKVHeadsIntArray() {
        // GGUFReader.readArray() returns int[] for UINT32/INT32 arrays
        Map<String, Object> metadata = new HashMap<>();
        metadata.put("general.architecture", "lfm2");
        metadata.put("lfm2.attention.head_count", 32);
        metadata.put("lfm2.attention.head_count_kv",
                new int[]{0, 0, 8, 0, 0, 8, 0, 0, 8, 0, 0, 8, 0, 0, 8, 0});

        GGUFHeader header = GGUFHeader.builder().metadata(metadata).build();

        // Scalar getter should return first non-zero value (8)
        assertEquals(8, header.getAttentionHeadCountKV(),
                "Should return first non-zero KV head count from per-layer int[] array");

        // Per-layer getter should return the full array as List<Integer>
        List<Integer> perLayer = header.getAttentionHeadCountKVPerLayer();
        assertNotNull(perLayer);
        assertEquals(16, perLayer.size());
        assertEquals(0, perLayer.get(0));
        assertEquals(8, perLayer.get(2));
        assertEquals(0, perLayer.get(15));
    }

    @Test
    @DisplayName("GGUFHeader.getAttentionHeadCountKV() handles per-layer List (test harness)")
    void testGGUFHeaderPerLayerKVHeadsList() {
        // In test harnesses, metadata may contain List<Integer> instead of int[]
        Map<String, Object> metadata = new HashMap<>();
        metadata.put("general.architecture", "lfm2");
        metadata.put("lfm2.attention.head_count", 32);
        metadata.put("lfm2.attention.head_count_kv",
                Arrays.asList(0, 0, 8, 0, 0, 8, 0, 0, 8, 0, 0, 8, 0, 0, 8, 0));

        GGUFHeader header = GGUFHeader.builder().metadata(metadata).build();
        assertEquals(8, header.getAttentionHeadCountKV());
        assertNotNull(header.getAttentionHeadCountKVPerLayer());
    }

    @Test
    @DisplayName("GGUFHeader.getAttentionHeadCountKV() handles scalar value")
    void testGGUFHeaderScalarKVHeads() {
        Map<String, Object> metadata = new HashMap<>();
        metadata.put("general.architecture", "llama");
        metadata.put("llama.attention.head_count", 32);
        metadata.put("llama.attention.head_count_kv", 8);

        GGUFHeader header = GGUFHeader.builder().metadata(metadata).build();

        assertEquals(8, header.getAttentionHeadCountKV());
        assertNull(header.getAttentionHeadCountKVPerLayer(),
                "Scalar KV head count should return null for per-layer getter");
    }

    @Test
    @DisplayName("GGUFHeader.getAttentionHeadCountKV() falls back when all entries are 0")
    void testGGUFHeaderAllZeroKVHeads() {
        Map<String, Object> metadata = new HashMap<>();
        metadata.put("general.architecture", "lfm2");
        metadata.put("lfm2.attention.head_count", 32);
        metadata.put("lfm2.attention.head_count_kv", new int[]{0, 0, 0, 0});

        GGUFHeader header = GGUFHeader.builder().metadata(metadata).build();
        assertEquals(32, header.getAttentionHeadCountKV());
    }

    // =========================================================================
    // ArchitectureConfig per-layer KV head support
    // =========================================================================

    @Test
    @DisplayName("ArchitectureConfig.getNumKVHeadsForLayer() uses per-layer array")
    void testConfigPerLayerKVHeads() {
        ArchitectureConfig config = ArchitectureConfig.builder()
                .numAttentionHeads(NUM_HEADS)
                .numKVHeads(NUM_KV_HEADS)
                .kvHeadsPerLayer(KV_HEADS_PER_LAYER)
                .build();

        // Conv layers (0 in per-layer array) should fall back to global numKVHeads
        assertEquals(NUM_KV_HEADS, config.getNumKVHeadsForLayer(0));
        assertEquals(NUM_KV_HEADS, config.getNumKVHeadsForLayer(1));

        // Attention layers should use per-layer value
        assertEquals(8, config.getNumKVHeadsForLayer(2));
        assertEquals(8, config.getNumKVHeadsForLayer(5));
    }

    @Test
    @DisplayName("ArchitectureConfig.getNumKVHeadsForLayer() falls back to global when no per-layer array")
    void testConfigGlobalKVHeadsFallback() {
        ArchitectureConfig config = ArchitectureConfig.builder()
                .numAttentionHeads(32)
                .numKVHeads(8)
                .build();

        assertEquals(8, config.getNumKVHeadsForLayer(0));
        assertEquals(8, config.getNumKVHeadsForLayer(100));
    }

    // =========================================================================
    // getConfig() with per-layer KV heads
    // =========================================================================

    @Test
    @DisplayName("LFM2Architecture.getConfig() propagates per-layer KV heads from metadata")
    void testLFM2GetConfigPerLayerKVHeads() {
        LFM2Architecture arch = new LFM2Architecture();

        GGMLMetadata metadata = GGMLMetadata.builder()
                .architecture("lfm2")
                .hiddenSize(HIDDEN_SIZE)
                .numLayers(NUM_LAYERS)
                .numAttentionHeads(NUM_HEADS)
                .numKVHeads(NUM_KV_HEADS)
                .intermediateSize(INTERMEDIATE_SIZE)
                .vocabSize(VOCAB_SIZE)
                .contextLength(128000)
                .layerNormEpsilon(1e-5f)
                .ropeFreqBase(1000000.0f)
                .layerTypes(LAYER_TYPES)
                .kvHeadsPerLayer(KV_HEADS_PER_LAYER)
                .build();

        ArchitectureConfig config = arch.getConfig(metadata);

        assertEquals(HIDDEN_SIZE, config.getHiddenSize());
        assertEquals(NUM_LAYERS, config.getNumLayers());
        assertEquals(NUM_HEADS, config.getNumAttentionHeads());
        assertEquals(NUM_KV_HEADS, config.getNumKVHeads());
        assertEquals(HEAD_DIM, config.getHeadDimension());

        assertNotNull(config.getKvHeadsPerLayer());
        assertEquals(8, config.getNumKVHeadsForLayer(2));
        assertEquals(NUM_KV_HEADS, config.getNumKVHeadsForLayer(0));
    }

    // =========================================================================
    // Graph build + forward pass with synthetic weights
    // =========================================================================

    @Test
    @DisplayName("LFM2 buildGraph: conv state externalized, correct logits shape, no NaN")
    void testBuildGraphHybridConvAttention() {
        // 2 layers: one short-conv, one attention
        int numLayers = 2;
        List<String> layerTypes = Arrays.asList("short_conv", "attention");
        List<Integer> kvHeadsPerLayer = Arrays.asList(0, NUM_KV_HEADS);

        GGMLMetadata metadata = buildMetadata(numLayers, layerTypes, kvHeadsPerLayer);
        Map<String, INDArray> weights = createSyntheticWeights(numLayers, layerTypes);

        LFM2Architecture arch = new LFM2Architecture();
        SameDiff sd = arch.buildGraph(metadata, weights, ConversionOptions.forInference());

        assertNotNull(sd);
        assertNotNull(sd.getVariable("lm_logits"));

        // Verify conv state placeholder exists for the conv layer
        List<String> graphInputs = sd.inputs();
        assertTrue(graphInputs.contains("past_conv_state.0"),
                "Conv layer 0 should have past_conv_state.0 placeholder. Inputs: " + graphInputs);
        assertFalse(graphInputs.contains("past_conv_state.1"),
                "Attention layer 1 should NOT have conv state placeholder");

        // Verify conv_state_out is a registered output
        List<String> graphOutputs = sd.outputs();
        assertTrue(graphOutputs.contains("conv_state_out_0"),
                "Conv layer 0 should produce conv_state_out_0. Outputs: " + graphOutputs);

        // Verify KV cache for attention layer
        assertTrue(graphInputs.contains("past_key_values.1.key"),
                "Attention layer 1 should have KV cache key input");
        assertTrue(graphOutputs.contains("k_rope_1"),
                "Attention layer 1 should produce k_rope_1 output");

        // Forward pass with conv state input
        int seqLen = 4;
        INDArray inputIds = Nd4j.create(DataType.INT64, 1, seqLen);
        for (int i = 0; i < seqLen; i++) inputIds.putScalar(new int[]{0, i}, i + 1);

        Map<String, INDArray> inputs = buildInputs(inputIds, seqLen, layerTypes, kvHeadsPerLayer);
        // Add conv state placeholder input: [batch, convDim, kernelSize-1]
        inputs.put("past_conv_state.0",
                Nd4j.zeros(DataType.FLOAT, 1, HIDDEN_SIZE, CONV_KERNEL_SIZE - 1));

        // Request both logits and conv state output
        Map<String, INDArray> outputs = sd.output(inputs, "lm_logits", "conv_state_out_0");
        INDArray logits = outputs.get("lm_logits");
        INDArray convStateOut = outputs.get("conv_state_out_0");

        assertNotNull(logits, "Logits should not be null");
        assertArrayEquals(new long[]{1, seqLen, VOCAB_SIZE}, logits.shape(),
                "Logits should be [batch, seq, vocab]");
        assertFalse(logits.isNaN().any(),
                "Logits should not contain NaN values");

        // Verify conv state output shape and non-zero content
        assertNotNull(convStateOut, "conv_state_out_0 should not be null");
        assertArrayEquals(new long[]{1, HIDDEN_SIZE, CONV_KERNEL_SIZE - 1}, convStateOut.shape(),
                "Conv state output should be [batch, D, K-1]");
        // After processing 4 tokens, state should be non-zero (it captured the input history)
        assertFalse(convStateOut.eq(0).all(),
                "Conv state output should be non-zero after processing tokens (captures history)");
    }

    @Test
    @DisplayName("LFM2 short-conv block uses fused in_proj with 3-way split")
    void testShortConvBlockStructure() {
        // Single short-conv layer only
        int numLayers = 1;
        List<String> layerTypes = Arrays.asList("short_conv");
        List<Integer> kvHeadsPerLayer = Arrays.asList(0);

        GGMLMetadata metadata = buildMetadata(numLayers, layerTypes, kvHeadsPerLayer);
        Map<String, INDArray> weights = createSyntheticWeights(numLayers, layerTypes);

        LFM2Architecture arch = new LFM2Architecture();
        SameDiff sd = arch.buildGraph(metadata, weights, ConversionOptions.forInference());

        // Verify the fused in_proj variable exists (3x expansion)
        assertNotNull(sd.getVariable("model.layers.0.short_conv.in_proj.weight"),
                "Should have fused in_proj weight (3x hidden expansion)");

        // Verify the split op was created (3 output names: b, c, x)
        assertNotNull(sd.getVariable("conv_split_b_0"),
                "Should have 3-way split op for B, C, x");

        // Forward pass
        int seqLen = 4;
        INDArray inputIds = Nd4j.create(DataType.INT64, 1, seqLen);
        for (int i = 0; i < seqLen; i++) inputIds.putScalar(new int[]{0, i}, i + 1);

        Map<String, INDArray> inputs = buildInputs(inputIds, seqLen, layerTypes, kvHeadsPerLayer);
        Map<String, INDArray> outputs = sd.output(inputs, "lm_logits");
        assertFalse(outputs.get("lm_logits").isNaN().any(),
                "Short-conv layer should not produce NaN");
    }

    @Test
    @DisplayName("LFM2 attention block with QK norms and GQA reshape")
    void testAttentionBlockWithQKNorms() {
        // Single attention layer with QK norms
        int numLayers = 1;
        List<String> layerTypes = Arrays.asList("attention");
        List<Integer> kvHeadsPerLayer = Arrays.asList(NUM_KV_HEADS);

        GGMLMetadata metadata = buildMetadata(numLayers, layerTypes, kvHeadsPerLayer);
        Map<String, INDArray> weights = createSyntheticWeights(numLayers, layerTypes);

        LFM2Architecture arch = new LFM2Architecture();
        SameDiff sd = arch.buildGraph(metadata, weights, ConversionOptions.forInference());

        // Verify QK norm variables exist (named q_norm_<layer>.weight and k_norm_<layer>.weight)
        assertNotNull(sd.getVariable("model.layers.0.self_attn.q_norm_0.weight"),
                "Should have per-head Q norm weight");
        assertNotNull(sd.getVariable("model.layers.0.self_attn.k_norm_0.weight"),
                "Should have per-head K norm weight");

        // Single-token decode (the original failure mode)
        int seqLen = 1;
        INDArray inputIds = Nd4j.create(DataType.INT64, 1, seqLen);
        inputIds.putScalar(new int[]{0, 0}, 42);

        Map<String, INDArray> inputs = buildInputs(inputIds, seqLen, layerTypes, kvHeadsPerLayer);
        Map<String, INDArray> outputs = sd.output(inputs, "lm_logits");
        INDArray logits = outputs.get("lm_logits");

        assertNotNull(logits);
        assertArrayEquals(new long[]{1, 1, VOCAB_SIZE}, logits.shape());
        assertFalse(logits.isNaN().any(), "Attention with QK norms should not produce NaN");
    }

    @Test
    @DisplayName("LFM2 layer type detection probes correct GGUF tensor keys")
    void testLayerTypeDetectionByTensorKeys() {
        // Build weights with actual GGUF tensor names (no layer_types metadata)
        Map<String, INDArray> weights = new HashMap<>();
        weights.put("token_embd.weight", Nd4j.rand(DataType.FLOAT, VOCAB_SIZE, HIDDEN_SIZE));
        weights.put("token_embd_norm.weight", Nd4j.ones(DataType.FLOAT, HIDDEN_SIZE));

        // Layer 0: short-conv (detected by shortconv.conv.weight key)
        String p0 = "blk.0";
        weights.put(p0 + ".shortconv.in_proj.weight", Nd4j.rand(DataType.FLOAT, 3 * HIDDEN_SIZE, HIDDEN_SIZE));
        weights.put(p0 + ".shortconv.out_proj.weight", Nd4j.rand(DataType.FLOAT, HIDDEN_SIZE, HIDDEN_SIZE));
        weights.put(p0 + ".shortconv.conv.weight", Nd4j.rand(DataType.FLOAT, HIDDEN_SIZE, CONV_KERNEL_SIZE));
        weights.put(p0 + ".attn_norm.weight", Nd4j.ones(DataType.FLOAT, HIDDEN_SIZE));
        weights.put(p0 + ".ffn_norm.weight", Nd4j.ones(DataType.FLOAT, HIDDEN_SIZE));
        weights.put(p0 + ".ffn_gate.weight", Nd4j.rand(DataType.FLOAT, INTERMEDIATE_SIZE, HIDDEN_SIZE));
        weights.put(p0 + ".ffn_up.weight", Nd4j.rand(DataType.FLOAT, INTERMEDIATE_SIZE, HIDDEN_SIZE));
        weights.put(p0 + ".ffn_down.weight", Nd4j.rand(DataType.FLOAT, HIDDEN_SIZE, INTERMEDIATE_SIZE));

        // Layer 1: attention (detected by attn_q.weight key)
        String p1 = "blk.1";
        weights.put(p1 + ".attn_q.weight", Nd4j.rand(DataType.FLOAT, NUM_HEADS * HEAD_DIM, HIDDEN_SIZE));
        weights.put(p1 + ".attn_k.weight", Nd4j.rand(DataType.FLOAT, NUM_KV_HEADS * HEAD_DIM, HIDDEN_SIZE));
        weights.put(p1 + ".attn_v.weight", Nd4j.rand(DataType.FLOAT, NUM_KV_HEADS * HEAD_DIM, HIDDEN_SIZE));
        weights.put(p1 + ".attn_output.weight", Nd4j.rand(DataType.FLOAT, HIDDEN_SIZE, NUM_HEADS * HEAD_DIM));
        weights.put(p1 + ".attn_q_norm.weight", Nd4j.ones(DataType.FLOAT, HEAD_DIM));
        weights.put(p1 + ".attn_k_norm.weight", Nd4j.ones(DataType.FLOAT, HEAD_DIM));
        weights.put(p1 + ".attn_norm.weight", Nd4j.ones(DataType.FLOAT, HIDDEN_SIZE));
        weights.put(p1 + ".ffn_norm.weight", Nd4j.ones(DataType.FLOAT, HIDDEN_SIZE));
        weights.put(p1 + ".ffn_gate.weight", Nd4j.rand(DataType.FLOAT, INTERMEDIATE_SIZE, HIDDEN_SIZE));
        weights.put(p1 + ".ffn_up.weight", Nd4j.rand(DataType.FLOAT, INTERMEDIATE_SIZE, HIDDEN_SIZE));
        weights.put(p1 + ".ffn_down.weight", Nd4j.rand(DataType.FLOAT, HIDDEN_SIZE, INTERMEDIATE_SIZE));

        // Build with NO layer_types metadata — detection must use tensor key probing
        GGMLMetadata metadata = GGMLMetadata.builder()
                .architecture("lfm2")
                .hiddenSize(HIDDEN_SIZE)
                .numLayers(2)
                .numAttentionHeads(NUM_HEADS)
                .numKVHeads(NUM_KV_HEADS)
                .intermediateSize(INTERMEDIATE_SIZE)
                .vocabSize(VOCAB_SIZE)
                .contextLength(128000)
                .layerNormEpsilon(1e-5f)
                .ropeFreqBase(1000000.0f)
                .kvHeadsPerLayer(Arrays.asList(0, NUM_KV_HEADS))
                .build();

        LFM2Architecture arch = new LFM2Architecture();
        SameDiff sd = arch.buildGraph(metadata, weights, ConversionOptions.forInference());

        int seqLen = 2;
        INDArray inputIds = Nd4j.create(DataType.INT64, 1, seqLen);
        inputIds.putScalar(new int[]{0, 0}, 1);
        inputIds.putScalar(new int[]{0, 1}, 2);

        List<String> detectedLayerTypes = Arrays.asList("short_conv", "attention");
        List<Integer> detectedKvHeads = Arrays.asList(0, NUM_KV_HEADS);
        Map<String, INDArray> inputs = buildInputs(inputIds, seqLen, detectedLayerTypes, detectedKvHeads);
        Map<String, INDArray> outputs = sd.output(inputs, "lm_logits");
        assertNotNull(outputs.get("lm_logits"));
        assertFalse(outputs.get("lm_logits").isNaN().any(),
                "Tensor-key-based layer detection should produce valid output");
    }

    @Test
    @DisplayName("GGMLMetadata.fromGGUF propagates per-layer KV head array from int[]")
    void testFromGGUFPerLayerKVHeads() {
        // Simulate a real GGUFHeader where readArray() returns int[]
        Map<String, Object> rawMetadata = new HashMap<>();
        rawMetadata.put("general.architecture", "lfm2");
        rawMetadata.put("lfm2.block_count", 16);
        rawMetadata.put("lfm2.embedding_length", 2048);
        rawMetadata.put("lfm2.feed_forward_length", 8192);
        rawMetadata.put("lfm2.attention.head_count", 32);
        rawMetadata.put("lfm2.attention.head_count_kv",
                new int[]{0, 0, 8, 0, 0, 8, 0, 0, 8, 0, 0, 8, 0, 0, 8, 0});
        rawMetadata.put("lfm2.attention.layer_norm_rms_epsilon", 1e-5f);
        rawMetadata.put("lfm2.rope.freq_base", 1000000.0f);
        rawMetadata.put("lfm2.vocab_size", 65536);
        rawMetadata.put("lfm2.context_length", 128000);

        GGUFHeader header = GGUFHeader.builder().metadata(rawMetadata).build();
        GGMLMetadata metadata = GGMLMetadata.fromGGUF(header, null);

        assertEquals(8, metadata.getNumKVHeads());
        assertNotNull(metadata.getKvHeadsPerLayer());
        assertEquals(16, metadata.getKvHeadsPerLayer().size());
        assertEquals(0, metadata.getKvHeadsPerLayer().get(0));
        assertEquals(8, metadata.getKvHeadsPerLayer().get(2));
    }

    // =========================================================================
    // Real GGUF diagnostic tests (require -Dlfm2.gguf.path=...)
    // =========================================================================

    @Test
    @DisplayName("Inspect real GGUF: dump tensor names, shapes, dtypes, and metadata")
    @EnabledIfSystemProperty(named = "lfm2.gguf.path", matches = ".+")
    void testRealGGUFInspection() throws Exception {
        String path = System.getProperty("lfm2.gguf.path");
        File gguf = new File(path);
        assertTrue(gguf.exists(), "GGUF file not found: " + path);

        GGMLMetadata metadata = GGMLModelImport.inspectModel(gguf);
        System.out.println("=== LFM-2 GGUF Metadata ===");
        System.out.println("Architecture: " + metadata.getArchitecture());
        System.out.println("Model name: " + metadata.getModelName());
        System.out.println("Layers: " + metadata.getNumLayers());
        System.out.println("Hidden: " + metadata.getHiddenSize());
        System.out.println("FFN: " + metadata.getIntermediateSize());
        System.out.println("Heads: " + metadata.getNumAttentionHeads());
        System.out.println("KV Heads: " + metadata.getNumKVHeads());
        System.out.println("KV Heads per layer: " + metadata.getKvHeadsPerLayer());
        System.out.println("Layer types: " + metadata.getLayerTypes());
        System.out.println("Full attn interval: " + metadata.getFullAttentionInterval());
        System.out.println("Vocab: " + metadata.getVocabSize());
        System.out.println("Context: " + metadata.getContextLength());
        System.out.println("Epsilon: " + metadata.getLayerNormEpsilon());
        System.out.println("RoPE base: " + metadata.getRopeFreqBase());
        System.out.println("RoPE dim count: " + metadata.getRopeDimensionCount());
        System.out.println("RoPE type: " + metadata.getRopeType());
        System.out.println("Attn key length: " + metadata.getAttentionKeyLength());
        System.out.println("Attn value length: " + metadata.getAttentionValueLength());
        System.out.println("Expert count: " + metadata.getExpertCount());
        System.out.println("Expert used: " + metadata.getExpertUsedCount());

        System.out.println("\n=== Tensor Names & Shapes (" + metadata.getTensors().size() + " tensors) ===");
        for (GGMLTensorInfo t : metadata.getTensors()) {
            System.out.printf("  %-60s shape=%-20s dtype=%s%n",
                    t.getName(), Arrays.toString(t.getShape()), t.getDataType());
        }

        // Print all raw metadata keys
        System.out.println("\n=== Raw Metadata Keys ===");
        for (Map.Entry<String, Object> e : metadata.getRawMetadata().entrySet()) {
            String val = String.valueOf(e.getValue());
            if (val.length() > 100) val = val.substring(0, 100) + "...";
            System.out.printf("  %-50s = %s%n", e.getKey(), val);
        }

        assertNotNull(metadata.getArchitecture());
        assertTrue(metadata.getNumLayers() > 0);
    }

    @Test
    @DisplayName("Import real GGUF and check for NaN in first forward pass")
    @EnabledIfSystemProperty(named = "lfm2.gguf.path", matches = ".+")
    void testRealGGUFImportAndForwardPass() throws Exception {
        String path = System.getProperty("lfm2.gguf.path");

        // Inspect metadata first for layer configuration
        GGMLMetadata metadata = GGMLModelImport.inspectModel(new File(path));
        List<String> layerTypes = metadata.getLayerTypes();
        List<Integer> kvHeadsPerLayer = metadata.getKvHeadsPerLayer();
        int numLayers = metadata.getNumLayers();
        int numKvHeads = metadata.getNumKVHeads();
        int metaHeadDim = metadata.getAttentionKeyLength();
        if (metaHeadDim <= 0) {
            metaHeadDim = metadata.getHiddenSize() / metadata.getNumAttentionHeads();
        }

        System.out.printf("Model: layers=%d, layerTypes=%s, kvPerLayer=%s, headDim=%d%n",
                numLayers, layerTypes, kvHeadsPerLayer, metaHeadDim);

        SameDiff model = GGMLModelImport.importModel(path);

        System.out.println("=== SameDiff Variables ===");
        for (var v : model.variables()) {
            System.out.printf("  %-60s dtype=%-8s shape=%s%n",
                    v.name(), v.dataType(), Arrays.toString(v.getShape()));
        }

        // Build inputs with KV cache placeholders
        int seqLen = 1;
        INDArray inputIds = Nd4j.create(DataType.INT64, 1, seqLen);
        inputIds.putScalar(new int[]{0, 0}, 1); // BOS token

        Map<String, INDArray> inputs = new HashMap<>();
        inputs.put("input_ids", inputIds);
        inputs.put("position_offset", Nd4j.scalar(DataType.INT64, 0));
        inputs.put("cache_position", Nd4j.scalar(DataType.INT64, 0));
        inputs.put("_causal_mask", Nd4j.zeros(DataType.FLOAT, 1, 1, seqLen, seqLen));

        // Add per-layer KV caches for attention layers, conv state for conv layers
        int hiddenSize = metadata.getHiddenSize();
        for (int i = 0; i < numLayers; i++) {
            boolean isAttn = false;
            if (layerTypes != null && i < layerTypes.size()) {
                String lt = layerTypes.get(i).toLowerCase();
                isAttn = lt.contains("attention") || lt.contains("attn");
            }
            int layerKv = (kvHeadsPerLayer != null && i < kvHeadsPerLayer.size())
                    ? kvHeadsPerLayer.get(i) : numKvHeads;
            if (isAttn || layerKv > 0) {
                int kvH = layerKv > 0 ? layerKv : numKvHeads;
                inputs.put("past_key_values." + i + ".key",
                        Nd4j.zeros(DataType.FLOAT, 1, seqLen, kvH, metaHeadDim));
                inputs.put("past_key_values." + i + ".value",
                        Nd4j.zeros(DataType.FLOAT, 1, seqLen, kvH, metaHeadDim));
            } else {
                // Conv layer: state shape [batch, hiddenSize, kernelSize-1]
                // LFM-2 uses kernel_size=3, so state dim = 2
                inputs.put("past_conv_state." + i,
                        Nd4j.zeros(DataType.FLOAT, 1, hiddenSize, 2));
            }
        }

        System.out.println("Input placeholders: " + inputs.keySet());

        Map<String, INDArray> outputs = model.output(inputs, "lm_logits");
        INDArray logits = outputs.get("lm_logits");
        assertNotNull(logits, "Logits should not be null");

        long nanCount = logits.isNaN().castTo(DataType.INT64).sumNumber().longValue();
        long totalElements = logits.length();
        System.out.printf("Logits shape: %s, NaN count: %d / %d%n",
                Arrays.toString(logits.shape()), nanCount, totalElements);

        if (nanCount > 0) {
            // Print first few logit values to help debug
            float[] first10 = new float[Math.min(10, (int) logits.length())];
            for (int i = 0; i < first10.length; i++) {
                first10[i] = logits.getFloat(0, 0, i);
            }
            System.out.println("First 10 logits: " + Arrays.toString(first10));
        }

        assertEquals(0, nanCount, "Forward pass should not produce NaN logits");
    }

    // =========================================================================
    // End-to-end GGUF→SameDiff→GenerationPipeline tests
    // =========================================================================

    @Test
    @DisplayName("GGUF import produces correct graph structure: conv layers skip KV, attention layers have KV")
    @EnabledIfSystemProperty(named = "lfm2.gguf.path", matches = ".+")
    void testRealGGUFGraphStructure() throws Exception {
        String path = System.getProperty("lfm2.gguf.path");

        GGMLMetadata metadata = GGMLModelImport.inspectModel(new File(path));
        SameDiff model = GGMLModelImport.importModel(path);

        // Bug #1 verification: per-layer KV head counts are correctly parsed from int[]
        List<Integer> kvHeads = metadata.getKvHeadsPerLayer();
        assertNotNull(kvHeads, "Per-layer KV head array should not be null");
        assertEquals(16, kvHeads.size(), "Should have 16 per-layer entries");
        assertEquals(0, kvHeads.get(0), "Layer 0 (conv) should have kvHeads=0");
        assertEquals(0, kvHeads.get(1), "Layer 1 (conv) should have kvHeads=0");
        assertEquals(8, kvHeads.get(2), "Layer 2 (attention) should have kvHeads=8");

        // Verify graph has correct KV cache placeholders for attention layers only
        List<String> modelInputs = model.inputs();
        assertTrue(modelInputs.contains("input_ids"), "Missing input_ids placeholder");
        assertTrue(modelInputs.contains("position_offset"), "Missing position_offset placeholder");
        assertTrue(modelInputs.contains("cache_position"), "Missing cache_position placeholder");
        assertTrue(modelInputs.contains("_causal_mask"), "Missing _causal_mask placeholder");

        // Attention layers (2,5,8,10,12,14) should have KV cache inputs
        for (int layer : new int[]{2, 5, 8, 10, 12, 14}) {
            assertTrue(modelInputs.contains("past_key_values." + layer + ".key"),
                    "Layer " + layer + " (attention) should have KV cache key input");
            assertTrue(modelInputs.contains("past_key_values." + layer + ".value"),
                    "Layer " + layer + " (attention) should have KV cache value input");
        }

        // Conv layers (0,1,3,4,6,7,9,11,13,15) should have conv state, NOT KV cache
        for (int layer : new int[]{0, 1, 3, 4, 6, 7, 9, 11, 13, 15}) {
            assertFalse(modelInputs.contains("past_key_values." + layer + ".key"),
                    "Layer " + layer + " (conv) should NOT have KV cache input");
            assertTrue(modelInputs.contains("past_conv_state." + layer),
                    "Layer " + layer + " (conv) should have past_conv_state." + layer + " input");
        }

        // Verify conv state outputs are registered
        List<String> modelOutputs = model.outputs();
        for (int layer : new int[]{0, 1, 3, 4, 6, 7, 9, 11, 13, 15}) {
            assertTrue(modelOutputs.contains("conv_state_out_" + layer),
                    "Layer " + layer + " should have conv_state_out_" + layer + " output");
        }

        // Bug #2 verification: all ops should have consistent dtypes (no FLOAT vs HALF mismatch)
        // Check that attention matmul weights and activation dtypes are consistent
        for (SDVariable v : model.variables()) {
            if (v.name().contains("q_proj.weight") || v.name().contains("k_proj.weight")
                    || v.name().contains("v_proj.weight") || v.name().contains("o_proj.weight")) {
                assertNotEquals(DataType.HALF, v.dataType(),
                        "Weight " + v.name() + " should not be HALF (should be dequantized to FLOAT)");
            }
        }

        // Verify the model is detected as in-graph KV cache (not ONNX-style)
        assertTrue(ModelIOConfig.isInGraphKvCache(model),
                "LFM-2 GGUF should be detected as in-graph KV cache model");
    }

    @Test
    @DisplayName("Full GenerationPipeline: GGUF import → ChatML prompt → coherent text output")
    @EnabledIfSystemProperty(named = "lfm2.gguf.path", matches = ".+")
    void testFullGenerationPipeline() throws Exception {
        String ggufPath = System.getProperty("lfm2.gguf.path");
        File ggufDir = new File(ggufPath).getParentFile();
        File tokenizerFile = new File(ggufDir, "tokenizer.json");
        assertTrue(tokenizerFile.exists(),
                "tokenizer.json required at " + tokenizerFile.getAbsolutePath());

        SameDiff model = GGMLModelImport.importModel(ggufPath);
        model.setDspAutoCompileEnabled(true);
        model.setDspNativeAutoCompileEnabled(true);

        Tokenizer tokenizer = HuggingFaceTokenizer.fromFile(tokenizerFile.getAbsolutePath());
        assertTrue(tokenizer.isValid(), "Tokenizer should be valid");

        // Generate enough tokens to properly evaluate output quality
        int maxTokens = 32;
        String prompt = "<|im_start|>user\nWhat is machine learning?<|im_end|>\n<|im_start|>assistant\n";

        GenerationPipelineConfig pipelineConfig = GenerationPipelineConfig.builder()
                .decoder(model)
                .tokenizer(tokenizer)
                .samplingConfig(SamplingConfig.defaultConfig())
                .maxNewTokens(maxTokens)
                .graphOptimizerEnabled(false)
                .dspEnabled(true)
                .build();

        try (GenerationPipeline pipeline = GenerationPipeline.create(pipelineConfig)) {
            GenerationResult result = pipeline.generate(prompt, maxTokens);

            System.out.printf("Generated %d tokens in %.1f ms (%.1f tok/s)%n",
                    result.getGeneratedTokenCount(),
                    (double) result.getGenerationTimeMs(),
                    result.getTokensPerSecond());
            System.out.println("Output: " + result.getText());

            // Quality validation: diversity, repetition, coherence
            QualityReport quality = GenerationQualityValidator.validate(result,
                    ValidationConfig.builder()
                            .minDiversityRatio(0.3)
                            .maxRepetitionScore(0.65)
                            .minCoherenceScore(0.3)
                            .build());
            System.out.println("Quality: " + quality.summary());

            assertTrue(result.getGeneratedTokenCount() >= 5,
                    "Should generate at least 5 tokens, got " + result.getGeneratedTokenCount());
            assertTrue(quality.getDiversityRatio() >= 0.3,
                    "Token diversity too low (garbage/repetitive): " + quality.getDiversityRatio()
                    + " — output: " + result.getText());
            assertTrue(quality.getCoherenceScore() >= 0.3,
                    "Coherence too low (garbage output): " + quality.getCoherenceScore()
                    + " — output: " + result.getText());
            assertTrue(quality.getRepetitionScore() <= 0.65,
                    "Repetition too high: " + quality.getRepetitionScore()
                    + " — output: " + result.getText());
            assertTrue(quality.isPassed(),
                    "Quality validation failed: " + quality.summary()
                    + " — output: " + result.getText());
        }
    }

    @Test
    @DisplayName("GenerationPipeline detects in-graph KV cache and routes to correct codepath")
    @EnabledIfSystemProperty(named = "lfm2.gguf.path", matches = ".+")
    void testInGraphKvCacheDetection() throws Exception {
        String path = System.getProperty("lfm2.gguf.path");
        SameDiff model = GGMLModelImport.importModel(path);

        // Verify in-graph KV cache detection
        assertTrue(ModelIOConfig.isInGraphKvCache(model),
                "LFM-2 GGUF must be detected as in-graph KV cache");
        assertTrue(ModelIOConfig.hasKvCache(model),
                "LFM-2 GGUF must report hasKvCache=true");

        // Verify KV cache input names are discovered correctly
        ModelIOConfig.KVCacheNames kvNames = ModelIOConfig.findKVCacheInputNames(model);
        assertNotNull(kvNames, "KV cache input names should not be null");
        assertEquals(6, kvNames.keyNames.size(),
                "Should find 6 attention-layer key cache inputs");
        assertEquals(6, kvNames.valueNames.size(),
                "Should find 6 attention-layer value cache inputs");

        // Verify no ONNX-style present outputs exist
        ModelIOConfig.KVCacheNames outputKv = ModelIOConfig.findKVCacheOutputNames(model);
        assertTrue(outputKv.keyNames.isEmpty(),
                "LFM-2 GGUF should not have ONNX-style present_* key outputs");

        // Verify IO auto-discovery
        ModelIOConfig ioConfig = ModelIOConfig.discover(model);
        assertNotNull(ioConfig, "ModelIOConfig.discover should succeed");
        assertEquals("input_ids", ioConfig.getInputIdsName(),
                "Should discover input_ids");
        assertEquals("lm_logits", ioConfig.getLogitsOutputName(),
                "Should discover lm_logits as output");
    }

    // =========================================================================
    // Helpers
    // =========================================================================

    private GGMLMetadata buildMetadata(int numLayers, List<String> layerTypes,
                                        List<Integer> kvHeadsPerLayer) {
        return GGMLMetadata.builder()
                .architecture("lfm2")
                .hiddenSize(HIDDEN_SIZE)
                .numLayers(numLayers)
                .numAttentionHeads(NUM_HEADS)
                .numKVHeads(NUM_KV_HEADS)
                .intermediateSize(INTERMEDIATE_SIZE)
                .vocabSize(VOCAB_SIZE)
                .contextLength(128000)
                .layerNormEpsilon(1e-5f)
                .ropeFreqBase(1000000.0f)
                .layerTypes(layerTypes)
                .kvHeadsPerLayer(kvHeadsPerLayer)
                .build();
    }

    /**
     * Build the full input map for running a forward pass, including KV cache and conv state placeholders.
     */
    private Map<String, INDArray> buildInputs(INDArray inputIds, int seqLen,
                                               List<String> layerTypes, List<Integer> kvHeadsPerLayer) {
        Map<String, INDArray> inputs = new HashMap<>();
        inputs.put("input_ids", inputIds);
        inputs.put("position_offset", Nd4j.scalar(DataType.INT64, 0));
        inputs.put("cache_position", Nd4j.scalar(DataType.INT64, 0));

        // Causal mask: [1, 1, seqLen, seqLen] — all zeros (no masking for prefill)
        inputs.put("_causal_mask", Nd4j.zeros(DataType.FLOAT, 1, 1, seqLen, seqLen));

        for (int i = 0; i < layerTypes.size(); i++) {
            String type = layerTypes.get(i);
            int kvHeads = kvHeadsPerLayer.get(i);
            if (kvHeads > 0 && "attention".equals(type)) {
                inputs.put("past_key_values." + i + ".key",
                        Nd4j.zeros(DataType.FLOAT, 1, seqLen, kvHeads, HEAD_DIM));
                inputs.put("past_key_values." + i + ".value",
                        Nd4j.zeros(DataType.FLOAT, 1, seqLen, kvHeads, HEAD_DIM));
            } else if ("short_conv".equals(type)) {
                // Conv state: [batch, convDim, kernelSize-1]
                inputs.put("past_conv_state." + i,
                        Nd4j.zeros(DataType.FLOAT, 1, HIDDEN_SIZE, CONV_KERNEL_SIZE - 1));
            }
        }
        return inputs;
    }

    /**
     * Creates synthetic weights matching the real LFM-2.5 GGUF tensor names and shapes.
     * Uses actual GGUF tensor keys (blk.N.shortconv.*, blk.N.attn_q_norm.*, etc.).
     */
    private Map<String, INDArray> createSyntheticWeights(int numLayers, List<String> layerTypes) {
        Map<String, INDArray> weights = new HashMap<>();
        DataType dtype = DataType.FLOAT;

        // Token embedding: [vocab_size, hidden_size]
        weights.put("token_embd.weight", Nd4j.rand(dtype, VOCAB_SIZE, HIDDEN_SIZE).muli(0.02));

        // Output norm (token_embd_norm in GGUF — the post-stack norm)
        weights.put("token_embd_norm.weight", Nd4j.ones(dtype, HIDDEN_SIZE));

        for (int layer = 0; layer < numLayers; layer++) {
            String prefix = "blk." + layer;
            String type = layerTypes.get(layer);

            // Pre-block norm (all layers)
            weights.put(prefix + ".attn_norm.weight", Nd4j.ones(dtype, HIDDEN_SIZE));

            if ("attention".equals(type)) {
                int qDim = NUM_HEADS * HEAD_DIM;       // 32 * 64 = 2048
                int kvDim = NUM_KV_HEADS * HEAD_DIM;    // 8 * 64 = 512

                weights.put(prefix + ".attn_q.weight", Nd4j.rand(dtype, qDim, HIDDEN_SIZE).muli(0.02));
                weights.put(prefix + ".attn_k.weight", Nd4j.rand(dtype, kvDim, HIDDEN_SIZE).muli(0.02));
                weights.put(prefix + ".attn_v.weight", Nd4j.rand(dtype, kvDim, HIDDEN_SIZE).muli(0.02));
                weights.put(prefix + ".attn_output.weight", Nd4j.rand(dtype, HIDDEN_SIZE, qDim).muli(0.02));

                // Per-head QK RMSNorm weights: [head_dim]
                weights.put(prefix + ".attn_q_norm.weight", Nd4j.ones(dtype, HEAD_DIM));
                weights.put(prefix + ".attn_k_norm.weight", Nd4j.ones(dtype, HEAD_DIM));
            } else {
                // Short-conv block with actual GGUF tensor names
                // Fused in_proj: [3*hidden, hidden] (3x expansion, splits into B, C, x)
                weights.put(prefix + ".shortconv.in_proj.weight",
                        Nd4j.rand(dtype, 3 * HIDDEN_SIZE, HIDDEN_SIZE).muli(0.02));
                // Out proj: [hidden, hidden]
                weights.put(prefix + ".shortconv.out_proj.weight",
                        Nd4j.rand(dtype, HIDDEN_SIZE, HIDDEN_SIZE).muli(0.02));
                // Depthwise conv kernel: [D, K] after GGUF shape reversal
                weights.put(prefix + ".shortconv.conv.weight",
                        Nd4j.rand(dtype, HIDDEN_SIZE, CONV_KERNEL_SIZE).muli(0.02));
            }

            // Post-block FFN norm (all layers)
            weights.put(prefix + ".ffn_norm.weight", Nd4j.ones(dtype, HIDDEN_SIZE));

            // SwiGLU FFN (all layers)
            weights.put(prefix + ".ffn_gate.weight", Nd4j.rand(dtype, INTERMEDIATE_SIZE, HIDDEN_SIZE).muli(0.02));
            weights.put(prefix + ".ffn_up.weight", Nd4j.rand(dtype, INTERMEDIATE_SIZE, HIDDEN_SIZE).muli(0.02));
            weights.put(prefix + ".ffn_down.weight", Nd4j.rand(dtype, HIDDEN_SIZE, INTERMEDIATE_SIZE).muli(0.02));
        }

        return weights;
    }

    // =========================================================================
    // DSP compilation diagnostic
    // =========================================================================

    @Test
    @DisplayName("LFM2 DSP compilation check — verify plan compiles for the real model")
    void testDspCompilationOnRealModel() throws Exception {
        String ggufPath = System.getProperty("lfm2.gguf.path",
                System.getProperty("user.home") + "/.kompile/models/llm-ggmls/lfm2.5-1.2b-instruct/LFM2.5-1.2B-Instruct-Q4_K_M.gguf");
        File ggufFile = new File(ggufPath);
        assumeTrue(ggufFile.exists(), "GGUF file not found at " + ggufPath
                + ". Set -Dlfm2.gguf.path to provide the model.");

        ConversionOptions options = ConversionOptions.builder()
                .quantizationMode(ConversionOptions.QuantizationMode.DEQUANTIZE_TO_FLOAT16)
                .preserveTokenizerInfo(true)
                .useMemoryMapping(true)
                .build();
        SameDiff sd = GGMLModelImport.importModel(ggufFile, options);
        assertNotNull(sd);

        // Save and reload (mimics the serving server path)
        File outputDir = new File("/tmp/lfm2-dsp-test");
        outputDir.mkdirs();
        SameDiffSerializer.saveAutoShard(sd, new File(outputDir, "model.sdz"), true, Collections.emptyMap());
        sd = SameDiff.load(new File(outputDir, "model.sdz"), true);
        assertNotNull(sd, "Model should survive save/reload");
        System.out.println("Model saved and reloaded. ops=" + sd.getOps().size());
        System.out.println("dspAutoCompile=" + sd.isDspAutoCompileEnabled()
                + " dspNativeAutoCompile=" + sd.isDspNativeAutoCompileEnabled());

        // Build a simple input and do one output() call to trigger DAG construction
        INDArray inputIds = Nd4j.createFromArray(new int[][]{{1, 2, 3, 4}}).castTo(DataType.INT64);
        Map<String, INDArray> inputs = new HashMap<>();
        inputs.put("input_ids", inputIds);
        inputs.put("position_offset", Nd4j.scalar(DataType.INT64, 0));
        inputs.put("cache_position", Nd4j.scalar(DataType.INT64, 0));
        inputs.put("_causal_mask", Nd4j.zeros(DataType.FLOAT, 1, 1, 4, 4));

        // Add KV cache placeholders for all attention layers
        // From inspection: attention layers are at indices 2,5,8,10,12,14
        int[] attnLayers = {2, 5, 8, 10, 12, 14};
        for (int layerIdx : attnLayers) {
            inputs.put("past_key_values." + layerIdx + ".key",
                    Nd4j.zeros(DataType.FLOAT, 1, 4, 8, 64));
            inputs.put("past_key_values." + layerIdx + ".value",
                    Nd4j.zeros(DataType.FLOAT, 1, 4, 8, 64));
        }

        // Check DSP auto-compile is enabled
        assertTrue(sd.isDspAutoCompileEnabled(), "DSP auto-compile should be enabled");
        assertTrue(sd.isDspNativeAutoCompileEnabled(), "DSP native auto-compile should be enabled");

        System.out.println("Model has " + sd.getOps().size() + " ops");
        System.out.println("Attempting sd.output() to trigger DSP compilation...");

        // Run output — this should trigger DSP compilation
        try {
            Map<String, INDArray> result = sd.output(inputs, "lm_logits");
            assertNotNull(result.get("lm_logits"), "lm_logits should be in output");
            System.out.println("Output succeeded. Logits shape: " + Arrays.toString(result.get("lm_logits").shape()));
        } catch (Exception e) {
            System.out.println("Output failed: " + e.getMessage());
            e.printStackTrace();
        }

        // Now check if DSP compiled
        InferenceSession session = sd.getOrCreateSession();
        assertNotNull(session, "Session should exist after output()");

        var executor = session.getDynamicShapePlanExecutor();
        if (executor == null) {
            System.out.println("DSP EXECUTOR IS NULL — DSP compilation did NOT happen");
            System.out.println("This means DynamicShapePlanCompiler.compile() returned null for this graph");
            System.out.println("Check for unsupported ops (invoke, tensor array ops, input-less random ops)");

            // Print all op types to find what's blocking DSP
            Map<String, Integer> opTypeCounts = new HashMap<>();
            for (var entry : sd.getOps().entrySet()) {
                var op = entry.getValue().getOp();
                String opName = op != null ? op.opName() : "null";
                opTypeCounts.merge(opName, 1, Integer::sum);
            }
            System.out.println("Op type counts:");
            opTypeCounts.entrySet().stream()
                    .sorted(Map.Entry.<String, Integer>comparingByValue().reversed())
                    .forEach(e -> System.out.println("  " + e.getKey() + ": " + e.getValue()));
        } else {
            System.out.println("DSP EXECUTOR EXISTS");
            var planHandle = executor.getNativePlanHandle();
            System.out.println("Native plan handle: " + (planHandle != null && !planHandle.isNull() ? "VALID" : "NULL"));
        }

        // Clean up
        inputIds.close();
    }

    // =========================================================================
    // Production model end-to-end (hardcoded path — delete after verification)
    // =========================================================================

    @Test
    @DisplayName("LFM2 GGUF import + save/load + generate coherent text")
    void testLFM2GGUFImportAndGenerate() throws Exception {
        String ggufPath = System.getProperty("lfm2.gguf.path",
                System.getProperty("user.home") + "/.kompile/models/llm-ggmls/lfm2.5-1.2b-instruct/LFM2.5-1.2B-Instruct-Q4_K_M.gguf");
        File ggufFile = new File(ggufPath);
        assumeTrue(ggufFile.exists(), "GGUF file not found at " + ggufPath
                + ". Set -Dlfm2.gguf.path to provide the model.");

        // 1. Import GGUF → SameDiff
        ConversionOptions options = ConversionOptions.builder()
                .quantizationMode(ConversionOptions.QuantizationMode.DEQUANTIZE_TO_FLOAT16)
                .preserveTokenizerInfo(true)
                .useMemoryMapping(true)
                .build();
        SameDiff sd = GGMLModelImport.importModel(ggufFile, options);
        assertNotNull(sd, "SameDiff model should not be null after import");

        // 2. Save/reload round-trip (mimics serving server path)
        File outputDir = new File("/tmp/lfm2-test");
        outputDir.mkdirs();
        SameDiffSerializer.saveAutoShard(sd, new File(outputDir, "model.sdz"), true, Collections.emptyMap());
        sd = SameDiff.load(new File(outputDir, "model.sdz"), true);
        assertNotNull(sd, "SameDiff model should not be null after reload");

        // Verify conv state placeholders survived save/reload
        List<String> inputs = sd.inputs();
        int convStateCount = 0;
        for (String name : inputs) {
            if (name.startsWith("past_conv_state.")) convStateCount++;
        }
        System.out.println("Conv state inputs after reload: " + convStateCount);
        assertTrue(convStateCount >= 10,
                "Should have at least 10 past_conv_state placeholders (one per conv layer), got: " + convStateCount);

        // Verify conv_state_out outputs are registered
        List<String> outputs = sd.outputs();
        int convStateOutCount = 0;
        for (String name : outputs) {
            if (name.startsWith("conv_state_out_")) convStateOutCount++;
        }
        assertTrue(convStateOutCount >= 10,
                "Should have at least 10 conv_state_out outputs, got: " + convStateOutCount);

        // 3. Load tokenizer
        String tokenizerPath = System.getProperty("lfm2.tokenizer.path",
                new File(ggufFile.getParentFile(), "tokenizer.json").getAbsolutePath());
        File tokenizerFile = new File(tokenizerPath);
        assumeTrue(tokenizerFile.exists(), "Tokenizer not found at " + tokenizerPath
                + ". Set -Dlfm2.tokenizer.path to provide it.");
        Tokenizer tokenizer = HuggingFaceTokenizer.fromFile(tokenizerFile.getAbsolutePath());

        // 4. Generate with a factual prompt — enough tokens to detect garbage
        int maxTokens = 32;
        String prompt = "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n";

        try (GenerationPipeline pipeline = GenerationPipeline.create(
                GenerationPipelineConfig.builder()
                        .decoder(sd)
                        .tokenizer(tokenizer)
                        .samplingConfig(SamplingConfig.defaultConfig())
                        .graphOptimizerEnabled(false)
                        .maxNewTokens(maxTokens)
                        .build())) {

            GenerationResult result = pipeline.generate(prompt, maxTokens);

            System.out.println("Generated text: " + result.getText());
            System.out.printf("Tokens: %d, tok/s: %.1f%n",
                    result.getGeneratedTokenCount(), result.getTokensPerSecond());

            // Quality validation — catches the exact failures we've been seeing:
            // garbage output, repetitive tokens, mixed-script gibberish
            QualityReport quality = GenerationQualityValidator.validate(result,
                    ValidationConfig.builder()
                            .minDiversityRatio(0.3)
                            .maxRepetitionScore(0.65)
                            .minCoherenceScore(0.3)
                            .expectedSubstrings(Arrays.asList("Paris"))
                            .build());
            System.out.println("Quality: " + quality.summary());

            assertTrue(result.getGeneratedTokenCount() >= 5,
                    "Should generate at least 5 tokens, got " + result.getGeneratedTokenCount());
            assertTrue(quality.getDiversityRatio() >= 0.3,
                    "Token diversity too low (garbage/repetitive output): " + quality.getDiversityRatio()
                    + " — output: " + result.getText());
            assertTrue(quality.getCoherenceScore() >= 0.3,
                    "Coherence too low (garbage output): " + quality.getCoherenceScore()
                    + " — output: " + result.getText());
            assertTrue(quality.isPassed(),
                    "Output quality validation failed: " + quality.summary()
                    + " — output: " + result.getText());
        }
    }
}
