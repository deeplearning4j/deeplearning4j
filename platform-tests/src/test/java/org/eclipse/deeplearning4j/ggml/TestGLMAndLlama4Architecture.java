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

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.nd4j.ggml.architecture.ArchitectureRegistry;
import org.nd4j.ggml.architecture.GLMArchitecture;
import org.nd4j.ggml.architecture.Llama4Architecture;
import org.nd4j.ggml.architecture.ModelArchitecture;
import org.nd4j.ggml.format.GGMLMetadata;

import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for GLMArchitecture and Llama4Architecture registration, variant detection,
 * and metadata handling.
 */
@DisplayName("GLM and Llama4 Architecture Test")
class TestGLMAndLlama4Architecture {

    // =========================================================================
    // GLMArchitecture registration and basic properties
    // =========================================================================

    @Test
    @DisplayName("GLMArchitecture is registered in ArchitectureRegistry")
    void testGLMRegistered() {
        ModelArchitecture arch = ArchitectureRegistry.getArchitecture("glm");
        assertNotNull(arch, "GLM architecture should be registered");
        assertEquals("glm", arch.getName());
    }

    @Test
    @DisplayName("GLMArchitecture can be found by all supported variant names")
    void testGLMVariantLookup() {
        String[] variants = {"glm", "glm4", "chatglm", "chatglm3", "chatglm4", "glm-4", "codegeex"};
        for (String variant : variants) {
            ModelArchitecture arch = ArchitectureRegistry.getArchitecture(variant);
            assertNotNull(arch, "Expected variant '" + variant + "' to be registered");
            assertEquals("glm", arch.getName(),
                    "Variant '" + variant + "' should map to GLMArchitecture");
        }
    }

    @Test
    @DisplayName("GLMArchitecture.getSupportedVariants() returns expected set")
    void testGLMSupportedVariants() {
        GLMArchitecture arch = new GLMArchitecture();
        Set<String> variants = arch.getSupportedVariants();

        assertNotNull(variants);
        assertFalse(variants.isEmpty());
        assertTrue(variants.contains("glm"));
        assertTrue(variants.contains("glm4"));
        assertTrue(variants.contains("chatglm"));
        assertTrue(variants.contains("chatglm3"));
        assertTrue(variants.contains("chatglm4"));
        assertTrue(variants.contains("glm-4"));
        assertTrue(variants.contains("codegeex"));
    }

    // =========================================================================
    // GLMArchitecture.canHandle()
    // =========================================================================

    @Test
    @DisplayName("GLMArchitecture.canHandle() returns true for glm arch name")
    void testGLMCanHandleGlm() {
        GLMArchitecture arch = new GLMArchitecture();
        GGMLMetadata metadata = GGMLMetadata.builder().architecture("glm").build();
        assertTrue(arch.canHandle(metadata));
    }

    @Test
    @DisplayName("GLMArchitecture.canHandle() returns true for chatglm arch name")
    void testGLMCanHandleChatGlm() {
        GLMArchitecture arch = new GLMArchitecture();
        GGMLMetadata metadata = GGMLMetadata.builder().architecture("chatglm").build();
        assertTrue(arch.canHandle(metadata));
    }

    @Test
    @DisplayName("GLMArchitecture.canHandle() returns true for chatglm4 arch name")
    void testGLMCanHandleChatGlm4() {
        GLMArchitecture arch = new GLMArchitecture();
        GGMLMetadata metadata = GGMLMetadata.builder().architecture("chatglm4").build();
        assertTrue(arch.canHandle(metadata));
    }

    @Test
    @DisplayName("GLMArchitecture.canHandle() returns true for glm4 arch name")
    void testGLMCanHandleGlm4() {
        GLMArchitecture arch = new GLMArchitecture();
        GGMLMetadata metadata = GGMLMetadata.builder().architecture("glm4").build();
        assertTrue(arch.canHandle(metadata));
    }

    @Test
    @DisplayName("GLMArchitecture.canHandle() returns true for codegeex arch name")
    void testGLMCanHandleCodegeex() {
        GLMArchitecture arch = new GLMArchitecture();
        GGMLMetadata metadata = GGMLMetadata.builder().architecture("codegeex").build();
        assertTrue(arch.canHandle(metadata));
    }

    @Test
    @DisplayName("GLMArchitecture.canHandle() returns false for llama arch name")
    void testGLMCanNotHandleLlama() {
        GLMArchitecture arch = new GLMArchitecture();
        GGMLMetadata metadata = GGMLMetadata.builder().architecture("llama").build();
        assertFalse(arch.canHandle(metadata));
    }

    @Test
    @DisplayName("GLMArchitecture.canHandle() returns false for null arch")
    void testGLMCanNotHandleNull() {
        GLMArchitecture arch = new GLMArchitecture();
        GGMLMetadata metadata = GGMLMetadata.builder().architecture(null).build();
        assertFalse(arch.canHandle(metadata));
    }

    // =========================================================================
    // GLMArchitecture tensor name patterns
    // =========================================================================

    @Test
    @DisplayName("GLMArchitecture.getTensorNamePatterns() returns non-empty map")
    void testGLMTensorNamePatternsNonEmpty() {
        GLMArchitecture arch = new GLMArchitecture();
        Map<String, String> patterns = arch.getTensorNamePatterns();
        assertNotNull(patterns);
        assertFalse(patterns.isEmpty());
    }

    @Test
    @DisplayName("GLMArchitecture.getTensorNamePatterns() contains expected GGUF keys")
    void testGLMTensorNamePatternsKeys() {
        GLMArchitecture arch = new GLMArchitecture();
        Map<String, String> patterns = arch.getTensorNamePatterns();

        // Embedding and output keys
        assertTrue(patterns.containsKey("token_embd.weight"),
                "Missing token embedding key");
        assertTrue(patterns.containsKey("output.weight"),
                "Missing output projection key");
        assertTrue(patterns.containsKey("output_norm.weight"),
                "Missing output norm key");

        // Attention projection keys
        assertTrue(patterns.containsKey("blk.{layer}.attn_q.weight"),
                "Missing Q projection key");
        assertTrue(patterns.containsKey("blk.{layer}.attn_k.weight"),
                "Missing K projection key");
        assertTrue(patterns.containsKey("blk.{layer}.attn_v.weight"),
                "Missing V projection key");
        assertTrue(patterns.containsKey("blk.{layer}.attn_output.weight"),
                "Missing output projection key");

        // FFN keys
        assertTrue(patterns.containsKey("blk.{layer}.ffn_gate.weight"),
                "Missing FFN gate key");
        assertTrue(patterns.containsKey("blk.{layer}.ffn_up.weight"),
                "Missing FFN up key");
        assertTrue(patterns.containsKey("blk.{layer}.ffn_down.weight"),
                "Missing FFN down key");

        // Normalization keys
        assertTrue(patterns.containsKey("blk.{layer}.attn_norm.weight"),
                "Missing attention norm key");
        assertTrue(patterns.containsKey("blk.{layer}.ffn_norm.weight"),
                "Missing FFN norm key");
    }

    @Test
    @DisplayName("GLMArchitecture.getTensorNamePatterns() maps embedding to correct SameDiff name")
    void testGLMTensorPatternValues() {
        GLMArchitecture arch = new GLMArchitecture();
        Map<String, String> patterns = arch.getTensorNamePatterns();

        assertEquals("model.embed_tokens.weight", patterns.get("token_embd.weight"));
        assertEquals("lm_head.weight", patterns.get("output.weight"));
        assertEquals("model.norm.weight", patterns.get("output_norm.weight"));
        assertEquals("model.layers.{layer}.self_attn.q_proj.weight",
                patterns.get("blk.{layer}.attn_q.weight"));
    }

    @Test
    @DisplayName("GLMArchitecture.getTensorNamePatterns() contains MoE router gate key")
    void testGLMTensorPatternsMoEKeys() {
        GLMArchitecture arch = new GLMArchitecture();
        Map<String, String> patterns = arch.getTensorNamePatterns();

        assertTrue(patterns.containsKey("blk.{layer}.ffn_gate_inp.weight"),
                "Missing MoE router gate key");
    }

    // =========================================================================
    // GLMArchitecture detectArchitecture
    // =========================================================================

    @Test
    @DisplayName("ArchitectureRegistry.detectArchitecture() selects GLM for glm metadata")
    void testDetectGLMArchitecture() {
        GGMLMetadata metadata = GGMLMetadata.builder().architecture("glm").build();
        ModelArchitecture detected = ArchitectureRegistry.detectArchitecture(metadata);
        assertNotNull(detected);
        assertEquals("glm", detected.getName());
    }

    @Test
    @DisplayName("ArchitectureRegistry.detectArchitecture() selects GLM for chatglm4 metadata")
    void testDetectGLMForChatGlm4() {
        GGMLMetadata metadata = GGMLMetadata.builder().architecture("chatglm4").build();
        ModelArchitecture detected = ArchitectureRegistry.detectArchitecture(metadata);
        assertNotNull(detected);
        assertEquals("glm", detected.getName());
    }

    // =========================================================================
    // Llama4Architecture registration and basic properties
    // =========================================================================

    @Test
    @DisplayName("Llama4Architecture is registered in ArchitectureRegistry")
    void testLlama4Registered() {
        ModelArchitecture arch = ArchitectureRegistry.getArchitecture("llama4");
        assertNotNull(arch, "Llama4 architecture should be registered");
        assertEquals("llama4", arch.getName());
    }

    @Test
    @DisplayName("Llama4Architecture can be found by all supported variant names")
    void testLlama4VariantLookup() {
        String[] variants = {"llama4", "llama-4", "llama4-scout", "llama4-maverick"};
        for (String variant : variants) {
            ModelArchitecture arch = ArchitectureRegistry.getArchitecture(variant);
            assertNotNull(arch, "Expected variant '" + variant + "' to be registered");
            assertEquals("llama4", arch.getName(),
                    "Variant '" + variant + "' should map to Llama4Architecture");
        }
    }

    @Test
    @DisplayName("Llama4Architecture.getSupportedVariants() returns expected set")
    void testLlama4SupportedVariants() {
        Llama4Architecture arch = new Llama4Architecture();
        Set<String> variants = arch.getSupportedVariants();

        assertNotNull(variants);
        assertFalse(variants.isEmpty());
        assertTrue(variants.contains("llama4"));
        assertTrue(variants.contains("llama-4"));
        assertTrue(variants.contains("llama4-scout"));
        assertTrue(variants.contains("llama4-maverick"));
    }

    // =========================================================================
    // Llama4Architecture.canHandle()
    // =========================================================================

    @Test
    @DisplayName("Llama4Architecture.canHandle() returns true for llama4 arch name")
    void testLlama4CanHandleLlama4() {
        Llama4Architecture arch = new Llama4Architecture();
        GGMLMetadata metadata = GGMLMetadata.builder().architecture("llama4").build();
        assertTrue(arch.canHandle(metadata));
    }

    @Test
    @DisplayName("Llama4Architecture.canHandle() returns true for llama4-scout arch name")
    void testLlama4CanHandleScout() {
        Llama4Architecture arch = new Llama4Architecture();
        GGMLMetadata metadata = GGMLMetadata.builder().architecture("llama4-scout").build();
        assertTrue(arch.canHandle(metadata));
    }

    @Test
    @DisplayName("Llama4Architecture.canHandle() returns true for llama4-maverick arch name")
    void testLlama4CanHandleMaverick() {
        Llama4Architecture arch = new Llama4Architecture();
        GGMLMetadata metadata = GGMLMetadata.builder().architecture("llama4-maverick").build();
        assertTrue(arch.canHandle(metadata));
    }

    @Test
    @DisplayName("Llama4Architecture.canHandle() returns false for llama3 arch name")
    void testLlama4CanNotHandleLlama3() {
        Llama4Architecture arch = new Llama4Architecture();
        GGMLMetadata metadata = GGMLMetadata.builder().architecture("llama3").build();
        assertFalse(arch.canHandle(metadata));
    }

    @Test
    @DisplayName("Llama4Architecture.canHandle() returns false for llama arch name")
    void testLlama4CanNotHandleLlama() {
        Llama4Architecture arch = new Llama4Architecture();
        GGMLMetadata metadata = GGMLMetadata.builder().architecture("llama").build();
        assertFalse(arch.canHandle(metadata));
    }

    @Test
    @DisplayName("Llama4Architecture.canHandle() returns false for null arch")
    void testLlama4CanNotHandleNull() {
        Llama4Architecture arch = new Llama4Architecture();
        GGMLMetadata metadata = GGMLMetadata.builder().architecture(null).build();
        assertFalse(arch.canHandle(metadata));
    }

    // =========================================================================
    // Llama4Architecture tensor name patterns
    // =========================================================================

    @Test
    @DisplayName("Llama4Architecture.getTensorNamePatterns() returns non-empty map")
    void testLlama4TensorNamePatternsNonEmpty() {
        Llama4Architecture arch = new Llama4Architecture();
        Map<String, String> patterns = arch.getTensorNamePatterns();
        assertNotNull(patterns);
        assertFalse(patterns.isEmpty());
    }

    @Test
    @DisplayName("Llama4Architecture.getTensorNamePatterns() contains expected GGUF keys")
    void testLlama4TensorNamePatternsKeys() {
        Llama4Architecture arch = new Llama4Architecture();
        Map<String, String> patterns = arch.getTensorNamePatterns();

        // Core embedding and output
        assertTrue(patterns.containsKey("token_embd.weight"),
                "Missing token embedding key");
        assertTrue(patterns.containsKey("output.weight"),
                "Missing output projection key");
        assertTrue(patterns.containsKey("output_norm.weight"),
                "Missing output norm key");

        // Attention projections
        assertTrue(patterns.containsKey("blk.{layer}.attn_q.weight"),
                "Missing Q projection key");
        assertTrue(patterns.containsKey("blk.{layer}.attn_k.weight"),
                "Missing K projection key");
        assertTrue(patterns.containsKey("blk.{layer}.attn_v.weight"),
                "Missing V projection key");
        assertTrue(patterns.containsKey("blk.{layer}.attn_output.weight"),
                "Missing output projection key");

        // FFN keys
        assertTrue(patterns.containsKey("blk.{layer}.ffn_gate.weight"),
                "Missing FFN gate key");
        assertTrue(patterns.containsKey("blk.{layer}.ffn_up.weight"),
                "Missing FFN up key");
        assertTrue(patterns.containsKey("blk.{layer}.ffn_down.weight"),
                "Missing FFN down key");

        // MoE router gate
        assertTrue(patterns.containsKey("blk.{layer}.ffn_gate_inp.weight"),
                "Missing MoE router gate key");

        // Shared expert (Llama4-specific)
        assertTrue(patterns.containsKey("blk.{layer}.ffn_gate_shexp.weight"),
                "Missing shared expert gate key (Llama4-specific)");
        assertTrue(patterns.containsKey("blk.{layer}.ffn_up_shexp.weight"),
                "Missing shared expert up key (Llama4-specific)");
        assertTrue(patterns.containsKey("blk.{layer}.ffn_down_shexp.weight"),
                "Missing shared expert down key (Llama4-specific)");

        // Normalization keys
        assertTrue(patterns.containsKey("blk.{layer}.attn_norm.weight"),
                "Missing attention norm key");
        assertTrue(patterns.containsKey("blk.{layer}.ffn_norm.weight"),
                "Missing FFN norm key");
    }

    @Test
    @DisplayName("Llama4Architecture.getTensorNamePatterns() maps shared expert to correct SameDiff name")
    void testLlama4SharedExpertPatternValues() {
        Llama4Architecture arch = new Llama4Architecture();
        Map<String, String> patterns = arch.getTensorNamePatterns();

        assertEquals("model.layers.{layer}.block_sparse_moe.shared_expert.gate_proj.weight",
                patterns.get("blk.{layer}.ffn_gate_shexp.weight"));
        assertEquals("model.layers.{layer}.block_sparse_moe.shared_expert.up_proj.weight",
                patterns.get("blk.{layer}.ffn_up_shexp.weight"));
        assertEquals("model.layers.{layer}.block_sparse_moe.shared_expert.down_proj.weight",
                patterns.get("blk.{layer}.ffn_down_shexp.weight"));
    }

    @Test
    @DisplayName("Llama4Architecture.getTensorNamePatterns() maps embedding to correct SameDiff name")
    void testLlama4TensorPatternValues() {
        Llama4Architecture arch = new Llama4Architecture();
        Map<String, String> patterns = arch.getTensorNamePatterns();

        assertEquals("model.embed_tokens.weight", patterns.get("token_embd.weight"));
        assertEquals("lm_head.weight", patterns.get("output.weight"));
        assertEquals("model.norm.weight", patterns.get("output_norm.weight"));
        assertEquals("model.layers.{layer}.self_attn.q_proj.weight",
                patterns.get("blk.{layer}.attn_q.weight"));
    }

    // =========================================================================
    // Llama4Architecture detectArchitecture
    // =========================================================================

    @Test
    @DisplayName("ArchitectureRegistry.detectArchitecture() selects Llama4 for llama4 metadata")
    void testDetectLlama4Architecture() {
        GGMLMetadata metadata = GGMLMetadata.builder().architecture("llama4").build();
        ModelArchitecture detected = ArchitectureRegistry.detectArchitecture(metadata);
        assertNotNull(detected);
        assertEquals("llama4", detected.getName());
    }

    @Test
    @DisplayName("ArchitectureRegistry.detectArchitecture() selects Llama4 for llama4-scout metadata")
    void testDetectLlama4ForScout() {
        GGMLMetadata metadata = GGMLMetadata.builder().architecture("llama4-scout").build();
        ModelArchitecture detected = ArchitectureRegistry.detectArchitecture(metadata);
        assertNotNull(detected);
        assertEquals("llama4", detected.getName());
    }

    // =========================================================================
    // Mock metadata detection
    // =========================================================================

    @Test
    @DisplayName("GLMArchitecture getConfig() reads metadata fields correctly")
    void testGLMGetConfig() {
        GLMArchitecture arch = new GLMArchitecture();
        GGMLMetadata metadata = GGMLMetadata.builder()
                .architecture("chatglm4")
                .hiddenSize(4096)
                .numLayers(28)
                .numAttentionHeads(32)
                .numKVHeads(2)
                .intermediateSize(13696)
                .vocabSize(151552)
                .contextLength(131072)
                .layerNormEpsilon(1e-5f)
                .ropeFreqBase(10000.0f)
                .build();

        var config = arch.getConfig(metadata);
        assertEquals(4096, config.getHiddenSize());
        assertEquals(28, config.getNumLayers());
        assertEquals(32, config.getNumAttentionHeads());
        assertEquals(2, config.getNumKVHeads());
        assertEquals(13696, config.getIntermediateSize());
        assertEquals(151552, config.getVocabSize());
        assertEquals(131072, config.getContextLength());
        assertTrue(config.isUseRmsNorm());
        assertTrue(config.isUseSwiGLU());
        assertTrue(config.isUseRotaryEmbeddings());
        assertEquals(1, config.getRopeType(), "GLM should use interleaved (NeoX) RoPE, type=1");
    }

    @Test
    @DisplayName("Llama4Architecture getConfig() reads metadata fields correctly")
    void testLlama4GetConfig() {
        Llama4Architecture arch = new Llama4Architecture();
        GGMLMetadata metadata = GGMLMetadata.builder()
                .architecture("llama4")
                .hiddenSize(5120)
                .numLayers(48)
                .numAttentionHeads(40)
                .numKVHeads(8)
                .intermediateSize(8192)
                .vocabSize(202048)
                .contextLength(10485760)
                .layerNormEpsilon(1e-5f)
                .ropeFreqBase(500000.0f)
                .expertCount(16)
                .expertUsedCount(1)
                .fullAttentionInterval(4)
                .build();

        var config = arch.getConfig(metadata);
        assertEquals(5120, config.getHiddenSize());
        assertEquals(48, config.getNumLayers());
        assertEquals(40, config.getNumAttentionHeads());
        assertEquals(8, config.getNumKVHeads());
        assertEquals(16, config.getNumExperts());
        assertEquals(1, config.getNumExpertsPerToken());
        assertEquals(4, config.getFullAttentionInterval());
        assertTrue(config.isUseRmsNorm());
        assertTrue(config.isUseSwiGLU());
        assertEquals(0, config.getRopeType(), "Llama4 should use standard (split-half) RoPE, type=0");
    }

    @Test
    @DisplayName("Llama4Architecture getConfig() sets numExpertsPerToken=1 when metadata returns 0")
    void testLlama4GetConfigExpertUsedCountDefault() {
        Llama4Architecture arch = new Llama4Architecture();
        GGMLMetadata metadata = GGMLMetadata.builder()
                .architecture("llama4")
                .expertCount(16)
                .expertUsedCount(0) // not set in metadata
                .build();

        var config = arch.getConfig(metadata);
        assertEquals(16, config.getNumExperts());
        assertEquals(1, config.getNumExpertsPerToken(),
                "Default expertUsedCount should be 1 for Llama4");
    }

    // =========================================================================
    // Registry does not interfere with other architectures
    // =========================================================================

    @Test
    @DisplayName("Adding GLM and Llama4 does not break existing architecture lookups")
    void testExistingArchitecturesUnaffected() {
        assertNotNull(ArchitectureRegistry.getArchitecture("llama"), "LLaMA should still be registered");
        assertNotNull(ArchitectureRegistry.getArchitecture("mistral"), "Mistral should still be registered");
        assertNotNull(ArchitectureRegistry.getArchitecture("gemma"), "Gemma should still be registered");
        assertNotNull(ArchitectureRegistry.getArchitecture("generic"), "Generic should still be registered");
        assertNotNull(ArchitectureRegistry.getArchitecture("granite"), "Granite should still be registered");
    }

    @Test
    @DisplayName("GLM architecture name does not shadow llama3 lookup")
    void testGLMDoesNotShadowLlama3() {
        // llama3 should still resolve to LLaMAArchitecture
        ModelArchitecture llama3 = ArchitectureRegistry.getArchitecture("llama3");
        assertNotNull(llama3);
        assertEquals("llama", llama3.getName());
    }
}
