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

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.ggml.architecture.ArchitectureConfig;
import org.nd4j.ggml.architecture.ExportArchitecture;
import org.nd4j.ggml.architecture.ExportArchitectureRegistry;
import org.nd4j.ggml.architecture.LLaMAExportArchitecture;
import org.nd4j.ggml.export.ExportOptions;
import org.nd4j.ggml.format.GGMLDataType;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

@DisplayName("Export Architecture Test")
class ExportArchitectureTest {

    // ========== Registry Tests ==========

    @Test
    @DisplayName("Test registry has LLaMA architecture")
    void testRegistryHasLLaMA() {
        assertTrue(ExportArchitectureRegistry.hasArchitecture("llama"));
    }

    @Test
    @DisplayName("Test registry get LLaMA architecture")
    void testRegistryGetLLaMA() {
        ExportArchitecture arch = ExportArchitectureRegistry.get("llama");
        assertNotNull(arch);
        assertEquals("llama", arch.getArchitectureName());
    }

    @Test
    @DisplayName("Test registry is case insensitive")
    void testRegistryCaseInsensitive() {
        ExportArchitecture arch1 = ExportArchitectureRegistry.get("llama");
        ExportArchitecture arch2 = ExportArchitectureRegistry.get("LLAMA");
        ExportArchitecture arch3 = ExportArchitectureRegistry.get("LLaMA");

        assertNotNull(arch1);
        assertEquals(arch1, arch2);
        assertEquals(arch1, arch3);
    }

    @Test
    @DisplayName("Test registry returns null for unknown architecture")
    void testRegistryUnknownArchitecture() {
        assertNull(ExportArchitectureRegistry.get("unknown_arch"));
        assertNull(ExportArchitectureRegistry.get(null));
        assertFalse(ExportArchitectureRegistry.hasArchitecture("unknown_arch"));
    }

    @Test
    @DisplayName("Test registry returns registered architectures")
    void testRegistryGetRegisteredArchitectures() {
        Set<String> architectures = ExportArchitectureRegistry.getRegisteredArchitectures();
        assertNotNull(architectures);
        assertTrue(architectures.contains("llama"));
    }

    @Test
    @DisplayName("Test registry custom architecture registration")
    void testRegistryCustomRegistration() {
        // Create a simple custom architecture
        ExportArchitecture customArch = new ExportArchitecture() {
            @Override
            public String getArchitectureName() {
                return "custom_test";
            }

            @Override
            public Set<String> getRecognizedVariablePatterns() {
                return Set.of("custom.*");
            }

            @Override
            public boolean canHandle(SameDiff model) {
                return false;
            }

            @Override
            public String mapVariableName(String sdVarName) {
                return null;
            }

            @Override
            public Map<String, String> getNameMappingPatterns() {
                return Map.of();
            }

            @Override
            public Map<String, Object> buildMetadata(SameDiff model, ExportOptions options) {
                return Map.of();
            }

            @Override
            public java.util.List<String> validateForExport(SameDiff model) {
                return java.util.List.of();
            }

            @Override
            public ArchitectureConfig inferConfig(SameDiff model) {
                return ArchitectureConfig.builder().build();
            }
        };

        ExportArchitectureRegistry.register(customArch);
        assertTrue(ExportArchitectureRegistry.hasArchitecture("custom_test"));
        assertEquals(customArch, ExportArchitectureRegistry.get("custom_test"));
    }

    // ========== LLaMA Architecture Tests ==========

    @Test
    @DisplayName("Test LLaMA architecture name")
    void testLLaMAArchitectureName() {
        ExportArchitecture arch = new LLaMAExportArchitecture();
        assertEquals("llama", arch.getArchitectureName());
    }

    @Test
    @DisplayName("Test LLaMA recognized variable patterns")
    void testLLaMARecognizedPatterns() {
        ExportArchitecture arch = new LLaMAExportArchitecture();
        Set<String> patterns = arch.getRecognizedVariablePatterns();

        assertNotNull(patterns);
        assertFalse(patterns.isEmpty());
        assertTrue(patterns.contains("model.embed_tokens.weight"));
    }

    @Test
    @DisplayName("Test LLaMA name mapping - embeddings")
    void testLLaMANameMappingEmbeddings() {
        ExportArchitecture arch = new LLaMAExportArchitecture();

        assertEquals("token_embd.weight",
                arch.mapVariableName("model.embed_tokens.weight"));
        assertEquals("output.weight",
                arch.mapVariableName("lm_head.weight"));
        assertEquals("output_norm.weight",
                arch.mapVariableName("model.norm.weight"));
    }

    @Test
    @DisplayName("Test LLaMA name mapping - attention layers")
    void testLLaMANameMappingAttention() {
        ExportArchitecture arch = new LLaMAExportArchitecture();

        assertEquals("blk.0.attn_q.weight",
                arch.mapVariableName("model.layers.0.self_attn.q_proj.weight"));
        assertEquals("blk.0.attn_k.weight",
                arch.mapVariableName("model.layers.0.self_attn.k_proj.weight"));
        assertEquals("blk.0.attn_v.weight",
                arch.mapVariableName("model.layers.0.self_attn.v_proj.weight"));
        assertEquals("blk.0.attn_output.weight",
                arch.mapVariableName("model.layers.0.self_attn.o_proj.weight"));

        // Test different layer numbers
        assertEquals("blk.15.attn_q.weight",
                arch.mapVariableName("model.layers.15.self_attn.q_proj.weight"));
        assertEquals("blk.31.attn_k.weight",
                arch.mapVariableName("model.layers.31.self_attn.k_proj.weight"));
    }

    @Test
    @DisplayName("Test LLaMA name mapping - FFN layers")
    void testLLaMANameMappingFFN() {
        ExportArchitecture arch = new LLaMAExportArchitecture();

        assertEquals("blk.0.ffn_gate.weight",
                arch.mapVariableName("model.layers.0.mlp.gate_proj.weight"));
        assertEquals("blk.0.ffn_up.weight",
                arch.mapVariableName("model.layers.0.mlp.up_proj.weight"));
        assertEquals("blk.0.ffn_down.weight",
                arch.mapVariableName("model.layers.0.mlp.down_proj.weight"));

        // Test different layer numbers
        assertEquals("blk.10.ffn_gate.weight",
                arch.mapVariableName("model.layers.10.mlp.gate_proj.weight"));
    }

    @Test
    @DisplayName("Test LLaMA name mapping - normalization layers")
    void testLLaMANameMappingNorm() {
        ExportArchitecture arch = new LLaMAExportArchitecture();

        assertEquals("blk.0.attn_norm.weight",
                arch.mapVariableName("model.layers.0.input_layernorm.weight"));
        assertEquals("blk.0.ffn_norm.weight",
                arch.mapVariableName("model.layers.0.post_attention_layernorm.weight"));
    }

    @Test
    @DisplayName("Test LLaMA name mapping - unknown variable returns null")
    void testLLaMANameMappingUnknown() {
        ExportArchitecture arch = new LLaMAExportArchitecture();

        assertNull(arch.mapVariableName("unknown.variable.name"));
        assertNull(arch.mapVariableName("some_other_name"));
    }

    @Test
    @DisplayName("Test LLaMA get name mapping patterns")
    void testLLaMAGetNameMappingPatterns() {
        ExportArchitecture arch = new LLaMAExportArchitecture();
        Map<String, String> patterns = arch.getNameMappingPatterns();

        assertNotNull(patterns);
        assertFalse(patterns.isEmpty());
    }

    // ========== Quantization Recommendation Tests ==========

    @Test
    @DisplayName("Test LLaMA recommended quant type for embeddings")
    void testLLaMAQuantTypeEmbeddings() {
        ExportArchitecture arch = new LLaMAExportArchitecture();
        ExportOptions options = ExportOptions.builder()
                .quantizationType(ExportOptions.QuantizationType.Q4_0)
                .build();

        // Embeddings should get higher precision
        GGMLDataType embType = arch.getRecommendedQuantType("token_embd.weight", options);
        assertEquals(GGMLDataType.GGML_TYPE_Q8_0, embType);

        GGMLDataType outputType = arch.getRecommendedQuantType("output.weight", options);
        assertEquals(GGMLDataType.GGML_TYPE_Q8_0, outputType);
    }

    @Test
    @DisplayName("Test LLaMA recommended quant type for norm layers")
    void testLLaMAQuantTypeNorm() {
        ExportArchitecture arch = new LLaMAExportArchitecture();
        ExportOptions options = ExportOptions.builder()
                .quantizationType(ExportOptions.QuantizationType.Q4_K)
                .build();

        // Norm weights should stay in F32
        GGMLDataType normType = arch.getRecommendedQuantType("blk.0.attn_norm.weight", options);
        assertEquals(GGMLDataType.GGML_TYPE_F32, normType);

        GGMLDataType ffnNormType = arch.getRecommendedQuantType("blk.5.ffn_norm.weight", options);
        assertEquals(GGMLDataType.GGML_TYPE_F32, ffnNormType);
    }

    @Test
    @DisplayName("Test LLaMA recommended quant type for regular layers")
    void testLLaMAQuantTypeRegular() {
        ExportArchitecture arch = new LLaMAExportArchitecture();
        ExportOptions options = ExportOptions.builder()
                .quantizationType(ExportOptions.QuantizationType.Q4_K)
                .build();

        // Regular layers should use specified quantization
        GGMLDataType attnType = arch.getRecommendedQuantType("blk.0.attn_q.weight", options);
        assertEquals(GGMLDataType.GGML_TYPE_Q4_K, attnType);

        GGMLDataType ffnType = arch.getRecommendedQuantType("blk.0.ffn_gate.weight", options);
        assertEquals(GGMLDataType.GGML_TYPE_Q4_K, ffnType);
    }

    // ========== ArchitectureConfig Tests ==========

    @Test
    @DisplayName("Test ArchitectureConfig default values")
    void testArchitectureConfigDefaults() {
        ArchitectureConfig config = ArchitectureConfig.builder().build();

        assertEquals(0, config.getNumLayers());
        assertEquals(0, config.getHiddenSize());
        assertEquals(0, config.getIntermediateSize());
        assertEquals(0, config.getNumAttentionHeads());
        assertEquals(0, config.getNumKVHeads());
        assertEquals(0, config.getVocabSize());
        assertEquals(0, config.getContextLength());
        assertEquals(1e-5f, config.getLayerNormEpsilon());
        assertEquals(10000.0f, config.getRopeFreqBase());
        assertTrue(config.isUseRmsNorm());
        assertTrue(config.isUseSwiGLU());
        assertTrue(config.isUseRotaryEmbeddings());
        assertTrue(config.isDecoderOnly());
    }

    @Test
    @DisplayName("Test ArchitectureConfig builder with values")
    void testArchitectureConfigBuilder() {
        ArchitectureConfig config = ArchitectureConfig.builder()
                .numLayers(32)
                .hiddenSize(4096)
                .intermediateSize(11008)
                .numAttentionHeads(32)
                .numKVHeads(8)
                .vocabSize(32000)
                .contextLength(4096)
                .layerNormEpsilon(1e-6f)
                .ropeFreqBase(500000.0f)
                .ropeDimensionCount(128)
                .useRmsNorm(true)
                .useSwiGLU(true)
                .useRotaryEmbeddings(true)
                .decoderOnly(true)
                .build();

        assertEquals(32, config.getNumLayers());
        assertEquals(4096, config.getHiddenSize());
        assertEquals(11008, config.getIntermediateSize());
        assertEquals(32, config.getNumAttentionHeads());
        assertEquals(8, config.getNumKVHeads());
        assertEquals(32000, config.getVocabSize());
        assertEquals(4096, config.getContextLength());
        assertEquals(1e-6f, config.getLayerNormEpsilon());
        assertEquals(500000.0f, config.getRopeFreqBase());
        assertEquals(128, config.getRopeDimensionCount());
    }

    @Test
    @DisplayName("Test ArchitectureConfig head dimension calculation")
    void testArchitectureConfigHeadDimension() {
        ArchitectureConfig config = ArchitectureConfig.builder()
                .hiddenSize(4096)
                .numAttentionHeads(32)
                .build();

        assertEquals(128, config.getHeadDimension());
    }

    @Test
    @DisplayName("Test ArchitectureConfig head dimension with zero heads")
    void testArchitectureConfigHeadDimensionZeroHeads() {
        ArchitectureConfig config = ArchitectureConfig.builder()
                .hiddenSize(4096)
                .numAttentionHeads(0)
                .build();

        assertEquals(0, config.getHeadDimension());
    }

    @Test
    @DisplayName("Test ArchitectureConfig grouped query attention detection")
    void testArchitectureConfigGQA() {
        // Standard MHA (num_kv_heads == num_heads)
        ArchitectureConfig mhaConfig = ArchitectureConfig.builder()
                .numAttentionHeads(32)
                .numKVHeads(32)
                .build();
        assertFalse(mhaConfig.hasGroupedQueryAttention());

        // GQA (num_kv_heads < num_heads)
        ArchitectureConfig gqaConfig = ArchitectureConfig.builder()
                .numAttentionHeads(32)
                .numKVHeads(8)
                .build();
        assertTrue(gqaConfig.hasGroupedQueryAttention());

        // MQA (num_kv_heads == 1)
        ArchitectureConfig mqaConfig = ArchitectureConfig.builder()
                .numAttentionHeads(32)
                .numKVHeads(1)
                .build();
        assertTrue(mqaConfig.hasGroupedQueryAttention());

        // No KV heads specified
        ArchitectureConfig noKVConfig = ArchitectureConfig.builder()
                .numAttentionHeads(32)
                .numKVHeads(0)
                .build();
        assertFalse(noKVConfig.hasGroupedQueryAttention());
    }

    // ========== Model Detection Tests ==========

    @Test
    @DisplayName("Test LLaMA can handle LLaMA-style model")
    void testLLaMACanHandleLLaMAModel() {
        ExportArchitecture arch = new LLaMAExportArchitecture();
        SameDiff sd = SameDiff.create();

        // Create a model with LLaMA-style variable names
        sd.var("model.embed_tokens.weight", DataType.FLOAT, 32000, 4096);
        sd.var("model.layers.0.self_attn.q_proj.weight", DataType.FLOAT, 4096, 4096);
        sd.var("model.layers.0.mlp.gate_proj.weight", DataType.FLOAT, 11008, 4096);

        assertTrue(arch.canHandle(sd));
    }

    @Test
    @DisplayName("Test LLaMA cannot handle non-LLaMA model")
    void testLLaMACannotHandleOtherModel() {
        ExportArchitecture arch = new LLaMAExportArchitecture();
        SameDiff sd = SameDiff.create();

        // Create a model with non-LLaMA variable names
        sd.var("encoder.layer.0.attention.weight", DataType.FLOAT, 768, 768);
        sd.var("decoder.layer.0.self_attn.weight", DataType.FLOAT, 768, 768);

        assertFalse(arch.canHandle(sd));
    }

    @Test
    @DisplayName("Test LLaMA cannot handle null model")
    void testLLaMACannotHandleNull() {
        ExportArchitecture arch = new LLaMAExportArchitecture();
        assertFalse(arch.canHandle(null));
    }

    // ========== Registry Detection Tests ==========

    @Test
    @DisplayName("Test registry detect for LLaMA model")
    void testRegistryDetectLLaMA() {
        SameDiff sd = SameDiff.create();
        sd.var("model.embed_tokens.weight", DataType.FLOAT, 32000, 4096);
        sd.var("model.layers.0.self_attn.q_proj.weight", DataType.FLOAT, 4096, 4096);

        ExportArchitecture arch = ExportArchitectureRegistry.detect(sd);
        assertNotNull(arch);
        assertEquals("llama", arch.getArchitectureName());
    }

    @Test
    @DisplayName("Test registry detect returns null for unknown model")
    void testRegistryDetectUnknown() {
        SameDiff sd = SameDiff.create();
        sd.var("unknown.variable", DataType.FLOAT, 100, 100);

        ExportArchitecture arch = ExportArchitectureRegistry.detect(sd);
        assertNull(arch);
    }

    @Test
    @DisplayName("Test registry detect returns null for null model")
    void testRegistryDetectNull() {
        ExportArchitecture arch = ExportArchitectureRegistry.detect(null);
        assertNull(arch);
    }

    @Test
    @DisplayName("Test registry getOrDetect with hint")
    void testRegistryGetOrDetectWithHint() {
        SameDiff sd = SameDiff.create();
        sd.var("some.variable", DataType.FLOAT, 100, 100);

        ExportArchitecture arch = ExportArchitectureRegistry.getOrDetect(sd, "llama");
        assertNotNull(arch);
        assertEquals("llama", arch.getArchitectureName());
    }

    @Test
    @DisplayName("Test registry getOrDetect with invalid hint falls back to detection")
    void testRegistryGetOrDetectWithInvalidHint() {
        SameDiff sd = SameDiff.create();
        sd.var("model.embed_tokens.weight", DataType.FLOAT, 32000, 4096);
        sd.var("model.layers.0.self_attn.q_proj.weight", DataType.FLOAT, 4096, 4096);

        // Invalid hint should fall back to detection
        ExportArchitecture arch = ExportArchitectureRegistry.getOrDetect(sd, "invalid_arch");
        assertNotNull(arch);
        assertEquals("llama", arch.getArchitectureName());
    }

    // ========== Metadata Building Tests ==========

    @Test
    @DisplayName("Test LLaMA build metadata includes architecture")
    void testLLaMABuildMetadataArchitecture() {
        ExportArchitecture arch = new LLaMAExportArchitecture();
        SameDiff sd = createMinimalLLaMAModel();
        ExportOptions options = ExportOptions.builder()
                .modelName("test-llama")
                .build();

        Map<String, Object> metadata = arch.buildMetadata(sd, options);

        assertNotNull(metadata);
        assertEquals("llama", metadata.get("general.architecture"));
        assertEquals("test-llama", metadata.get("general.name"));
    }

    @Test
    @DisplayName("Test LLaMA build metadata includes author and description")
    void testLLaMABuildMetadataAuthorDescription() {
        ExportArchitecture arch = new LLaMAExportArchitecture();
        SameDiff sd = createMinimalLLaMAModel();
        ExportOptions options = ExportOptions.builder()
                .modelName("test-llama")
                .modelAuthor("Test Author")
                .modelDescription("Test Description")
                .build();

        Map<String, Object> metadata = arch.buildMetadata(sd, options);

        assertEquals("Test Author", metadata.get("general.author"));
        assertEquals("Test Description", metadata.get("general.description"));
    }

    // ========== Validation Tests ==========

    @Test
    @DisplayName("Test LLaMA validate complete model")
    void testLLaMAValidateCompleteModel() {
        ExportArchitecture arch = new LLaMAExportArchitecture();
        SameDiff sd = createMinimalLLaMAModel();

        java.util.List<String> errors = arch.validateForExport(sd);
        assertTrue(errors.isEmpty(), "Expected no errors: " + errors);
    }

    @Test
    @DisplayName("Test LLaMA validate missing embeddings")
    void testLLaMAValidateMissingEmbeddings() {
        ExportArchitecture arch = new LLaMAExportArchitecture();
        SameDiff sd = SameDiff.create();

        // Model without embeddings
        sd.var("model.layers.0.self_attn.q_proj.weight", DataType.FLOAT, 4096, 4096);
        sd.var("model.norm.weight", DataType.FLOAT, 4096);

        java.util.List<String> errors = arch.validateForExport(sd);
        assertFalse(errors.isEmpty());
        assertTrue(errors.stream().anyMatch(e -> e.contains("embedding")));
    }

    // ========== Helper Methods ==========

    private SameDiff createMinimalLLaMAModel() {
        SameDiff sd = SameDiff.create();

        // Embeddings
        sd.var("model.embed_tokens.weight", Nd4j.rand(DataType.FLOAT, 32000, 4096));

        // Layer 0
        sd.var("model.layers.0.self_attn.q_proj.weight", Nd4j.rand(DataType.FLOAT, 4096, 4096));
        sd.var("model.layers.0.self_attn.k_proj.weight", Nd4j.rand(DataType.FLOAT, 4096, 4096));
        sd.var("model.layers.0.self_attn.v_proj.weight", Nd4j.rand(DataType.FLOAT, 4096, 4096));
        sd.var("model.layers.0.self_attn.o_proj.weight", Nd4j.rand(DataType.FLOAT, 4096, 4096));
        sd.var("model.layers.0.mlp.gate_proj.weight", Nd4j.rand(DataType.FLOAT, 11008, 4096));
        sd.var("model.layers.0.mlp.up_proj.weight", Nd4j.rand(DataType.FLOAT, 11008, 4096));
        sd.var("model.layers.0.mlp.down_proj.weight", Nd4j.rand(DataType.FLOAT, 4096, 11008));
        sd.var("model.layers.0.input_layernorm.weight", Nd4j.rand(DataType.FLOAT, 4096));
        sd.var("model.layers.0.post_attention_layernorm.weight", Nd4j.rand(DataType.FLOAT, 4096));

        // Final norm
        sd.var("model.norm.weight", Nd4j.rand(DataType.FLOAT, 4096));

        // LM head
        sd.var("lm_head.weight", Nd4j.rand(DataType.FLOAT, 32000, 4096));

        return sd;
    }
}
