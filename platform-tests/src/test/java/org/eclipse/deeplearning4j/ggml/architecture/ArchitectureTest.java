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
import org.nd4j.ggml.architecture.ArchitectureConfig;
import org.nd4j.ggml.architecture.ArchitectureRegistry;
import org.nd4j.ggml.architecture.GenericArchitecture;
import org.nd4j.ggml.architecture.LLaMAArchitecture;
import org.nd4j.ggml.architecture.ModelArchitecture;
import org.nd4j.ggml.format.GGMLMetadata;

import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

@DisplayName("Architecture Test")
class ArchitectureTest {

    @Test
    @DisplayName("Test ArchitectureRegistry has LLaMA architecture")
    void testRegistryHasLLaMA() {
        ModelArchitecture arch = ArchitectureRegistry.getArchitecture("llama");
        assertNotNull(arch);
        assertEquals("llama", arch.getName());
    }

    @Test
    @DisplayName("Test ArchitectureRegistry has generic architecture")
    void testRegistryHasGeneric() {
        ModelArchitecture arch = ArchitectureRegistry.getArchitecture("generic");
        assertNotNull(arch);
        assertEquals("generic", arch.getName());
    }

    @Test
    @DisplayName("Test ArchitectureRegistry returns null for unknown")
    void testRegistryUnknown() {
        ModelArchitecture arch = ArchitectureRegistry.getArchitecture("unknown_arch_xyz");
        assertNull(arch);
    }

    @Test
    @DisplayName("Test LLaMAArchitecture supported variants")
    void testLLaMAVariants() {
        LLaMAArchitecture arch = new LLaMAArchitecture();
        Set<String> variants = arch.getSupportedVariants();

        assertTrue(variants.contains("llama"));
        assertTrue(variants.contains("llama2"));
        assertTrue(variants.contains("llama3"));
        assertTrue(variants.contains("mistral"));
        assertTrue(variants.contains("mixtral"));
        assertTrue(variants.contains("codellama"));
    }

    @Test
    @DisplayName("Test LLaMAArchitecture canHandle")
    void testLLaMACanHandle() {
        LLaMAArchitecture arch = new LLaMAArchitecture();

        GGMLMetadata llamaMetadata = GGMLMetadata.builder()
                .architecture("llama")
                .build();
        assertTrue(arch.canHandle(llamaMetadata));

        GGMLMetadata mistralMetadata = GGMLMetadata.builder()
                .architecture("mistral")
                .build();
        assertTrue(arch.canHandle(mistralMetadata));

        GGMLMetadata bertMetadata = GGMLMetadata.builder()
                .architecture("bert")
                .build();
        assertFalse(arch.canHandle(bertMetadata));
    }

    @Test
    @DisplayName("Test LLaMAArchitecture tensor name patterns")
    void testLLaMATensorPatterns() {
        LLaMAArchitecture arch = new LLaMAArchitecture();
        var patterns = arch.getTensorNamePatterns();

        assertNotNull(patterns);
        assertTrue(patterns.containsKey("token_embd.weight"));
        assertTrue(patterns.containsKey("output.weight"));
        assertTrue(patterns.containsKey("blk.{layer}.attn_q.weight"));
    }

    @Test
    @DisplayName("Test GenericArchitecture")
    void testGenericArchitecture() {
        GenericArchitecture arch = new GenericArchitecture();

        assertEquals("generic", arch.getName());
        // Generic architecture has predefined variants
        assertTrue(arch.getSupportedVariants().contains("generic"));

        // Generic should handle any metadata
        GGMLMetadata metadata = GGMLMetadata.builder()
                .architecture("some_unknown_arch")
                .build();
        assertTrue(arch.canHandle(metadata));
    }

    @Test
    @DisplayName("Test ArchitectureConfig builder")
    void testArchitectureConfig() {
        ArchitectureConfig config = ArchitectureConfig.builder()
                .hiddenSize(4096)
                .numLayers(32)
                .numAttentionHeads(32)
                .numKVHeads(8)
                .intermediateSize(11008)
                .vocabSize(32000)
                .contextLength(4096)
                .layerNormEpsilon(1e-5f)
                .ropeFreqBase(10000.0f)
                .build();

        assertEquals(4096, config.getHiddenSize());
        assertEquals(32, config.getNumLayers());
        assertEquals(32, config.getNumAttentionHeads());
        assertEquals(8, config.getNumKVHeads());
        assertEquals(11008, config.getIntermediateSize());
        assertEquals(32000, config.getVocabSize());
        assertEquals(4096, config.getContextLength());
        assertEquals(128, config.getHeadDimension()); // Computed: 4096/32
        assertEquals(1e-5f, config.getLayerNormEpsilon(), 1e-10);
        assertEquals(10000.0f, config.getRopeFreqBase(), 1e-5);
    }

    @Test
    @DisplayName("Test ArchitectureConfig defaults")
    void testArchitectureConfigDefaults() {
        ArchitectureConfig config = ArchitectureConfig.builder().build();

        assertEquals(1e-5f, config.getLayerNormEpsilon(), 1e-10);
        assertEquals(10000.0f, config.getRopeFreqBase(), 1e-5);
    }

    @Test
    @DisplayName("Test detectArchitecture with LLaMA metadata")
    void testDetectArchitectureLLaMA() {
        GGMLMetadata metadata = GGMLMetadata.builder()
                .architecture("llama")
                .build();

        ModelArchitecture detected = ArchitectureRegistry.detectArchitecture(metadata);
        assertNotNull(detected);
        assertEquals("llama", detected.getName());
    }

    @Test
    @DisplayName("Test detectArchitecture with Mistral metadata")
    void testDetectArchitectureMistral() {
        GGMLMetadata metadata = GGMLMetadata.builder()
                .architecture("mistral")
                .build();

        ModelArchitecture detected = ArchitectureRegistry.detectArchitecture(metadata);
        assertNotNull(detected);
        assertEquals("llama", detected.getName()); // Mistral uses LLaMA architecture
    }

    @Test
    @DisplayName("Test detectArchitecture falls back to generic")
    void testDetectArchitectureGeneric() {
        GGMLMetadata metadata = GGMLMetadata.builder()
                .architecture("some_unknown_architecture")
                .build();

        ModelArchitecture detected = ArchitectureRegistry.detectArchitecture(metadata);
        assertNotNull(detected);
        assertEquals("generic", detected.getName());
    }

    @Test
    @DisplayName("Test registry contains expected architectures")
    void testRegistryContainsExpectedArchitectures() {
        // Verify we can get both known architectures
        assertNotNull(ArchitectureRegistry.getArchitecture("llama"));
        assertNotNull(ArchitectureRegistry.getArchitecture("generic"));
    }

    @Test
    @DisplayName("Test ModelArchitecture getConfig from metadata")
    void testGetConfigFromMetadata() {
        GGMLMetadata metadata = GGMLMetadata.builder()
                .architecture("llama")
                .hiddenSize(4096)
                .numLayers(32)
                .numAttentionHeads(32)
                .numKVHeads(8)
                .vocabSize(32000)
                .contextLength(4096)
                .build();

        LLaMAArchitecture arch = new LLaMAArchitecture();
        ArchitectureConfig config = arch.getConfig(metadata);

        assertNotNull(config);
        assertEquals(4096, config.getHiddenSize());
        assertEquals(32, config.getNumLayers());
        assertEquals(32, config.getNumAttentionHeads());
    }
}
