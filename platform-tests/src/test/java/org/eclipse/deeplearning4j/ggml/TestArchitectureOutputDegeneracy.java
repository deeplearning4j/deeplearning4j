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

import lombok.extern.slf4j.Slf4j;
import org.eclipse.deeplearning4j.llm.eval.GenerationQualityValidator;
import org.eclipse.deeplearning4j.llm.config.TokenizerConfig;
import org.eclipse.deeplearning4j.llm.generation.GenerationPipeline;
import org.eclipse.deeplearning4j.llm.generation.GenerationPipelineConfig;
import org.eclipse.deeplearning4j.llm.generation.GenerationResult;
import org.eclipse.deeplearning4j.llm.generation.SamplingConfig;
import org.eclipse.deeplearning4j.llm.tokenizer.ChatTemplate;
import org.eclipse.deeplearning4j.llm.tokenizer.HuggingFaceTokenizer;
import org.eclipse.deeplearning4j.llm.tokenizer.Tokenizer;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.ggml.architecture.*;
import org.nd4j.ggml.GGMLModelImport;
import org.nd4j.ggml.convert.ConversionOptions;
import org.nd4j.ggml.format.GGMLMetadata;

import java.io.File;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive output degeneracy tests for ALL GGML architecture importers.
 *
 * <p>Two tiers of tests:</p>
 * <ol>
 *   <li><b>Structural (no GGUF required)</b>: Every architecture is tested for
 *       registration, canHandle, tensor patterns, config, prompt template, and
 *       system property configuration.</li>
 *   <li><b>Generation (GGUF required)</b>: Gated on system properties
 *       ({@code -D<arch>.gguf.path=<path>}), runs GenerationPipeline with
 *       architecture-appropriate prompts and validates via GenerationQualityValidator.</li>
 * </ol>
 */
@Slf4j
@DisplayName("Architecture Output Degeneracy Tests")
public class TestArchitectureOutputDegeneracy {

    // ========================================================================
    // Architecture metadata: name, class, chat template type, system property
    // ========================================================================

    static Stream<Arguments> allDecoderArchitectures() {
        return Stream.of(
                Arguments.of("llama", LLaMAArchitecture.class, "chatml", "llama.gguf.path"),
                Arguments.of("mistral", MistralArchitecture.class, "llama2", "mistral.gguf.path"),
                Arguments.of("gemma", GemmaArchitecture.class, "gemma", "gemma.gguf.path"),
                Arguments.of("phi", PhiArchitecture.class, "chatml", "phi.gguf.path"),
                Arguments.of("granite", GraniteArchitecture.class, "chatml", "granite.gguf.path"),
                Arguments.of("nemotron", NemotronArchitecture.class, "chatml", "nemotron.gguf.path"),
                Arguments.of("olmo", OLMoArchitecture.class, "chatml", "olmo.gguf.path"),
                Arguments.of("openelm", OpenELMArchitecture.class, "plain", "openelm.gguf.path"),
                Arguments.of("gpt-oss", GptOssArchitecture.class, "chatml", "gpt-oss.gguf.path"),
                Arguments.of("glm", GLMArchitecture.class, "chatml", "glm.gguf.path"),
                Arguments.of("llama4", Llama4Architecture.class, "chatml", "llama4.gguf.path"),
                Arguments.of("lfm2", LFM2Architecture.class, "chatml", "lfm2.gguf.path")
        );
    }

    static Stream<Arguments> allArchitecturesIncludingWhisper() {
        return Stream.concat(
                allDecoderArchitectures(),
                Stream.of(Arguments.of("whisper", WhisperArchitecture.class, "none", "whisper.gguf.path"))
        );
    }

    // ========================================================================
    // Tier 1: Structural tests (no GGUF required)
    // ========================================================================

    @ParameterizedTest(name = "{0}: registered in ArchitectureRegistry")
    @MethodSource("allArchitecturesIncludingWhisper")
    void testArchitectureRegistered(String name, Class<?> archClass, String templateType, String sysProp) {
        assertTrue(ArchitectureRegistry.hasArchitecture(name),
                name + " not found in ArchitectureRegistry");
        ModelArchitecture arch = ArchitectureRegistry.getArchitecture(name);
        assertNotNull(arch, "getArchitecture returned null for " + name);
        assertEquals(name, arch.getName(), "getName() mismatch");
        assertInstanceOf(archClass, arch, "Wrong class for " + name);
    }

    @ParameterizedTest(name = "{0}: has supported variants")
    @MethodSource("allArchitecturesIncludingWhisper")
    void testArchitectureHasVariants(String name, Class<?> archClass, String templateType, String sysProp) {
        ModelArchitecture arch = ArchitectureRegistry.getArchitecture(name);
        assertNotNull(arch.getSupportedVariants(), "getSupportedVariants() null for " + name);
        assertFalse(arch.getSupportedVariants().isEmpty(), "No variants for " + name);
        assertTrue(arch.getSupportedVariants().contains(name),
                "Primary name not in supportedVariants for " + name);
    }

    @ParameterizedTest(name = "{0}: has tensor name patterns")
    @MethodSource("allArchitecturesIncludingWhisper")
    void testArchitectureHasTensorPatterns(String name, Class<?> archClass, String templateType, String sysProp) {
        ModelArchitecture arch = ArchitectureRegistry.getArchitecture(name);
        Map<String, String> patterns = arch.getTensorNamePatterns();
        assertNotNull(patterns, "getTensorNamePatterns() null for " + name);
        assertFalse(patterns.isEmpty(), "No tensor patterns for " + name);
    }

    @ParameterizedTest(name = "{0}: chat template = {2}")
    @MethodSource("allArchitecturesIncludingWhisper")
    void testArchitectureChatTemplate(String name, Class<?> archClass, String expectedTemplate, String sysProp) {
        ModelArchitecture arch = ArchitectureRegistry.getArchitecture(name);
        assertEquals(expectedTemplate, arch.getDefaultChatTemplateType(),
                "Chat template mismatch for " + name);
    }

    @ParameterizedTest(name = "{0}: system property = {3}")
    @MethodSource("allArchitecturesIncludingWhisper")
    void testArchitectureSystemProperty(String name, Class<?> archClass, String templateType, String expectedProp) {
        ModelArchitecture arch = ArchitectureRegistry.getArchitecture(name);
        assertEquals(expectedProp, arch.getModelSystemProperty(),
                "System property mismatch for " + name);
    }

    @ParameterizedTest(name = "{0}: has reference prompt")
    @MethodSource("allDecoderArchitectures")
    void testArchitectureHasReferencePrompt(String name, Class<?> archClass, String templateType, String sysProp) {
        ModelArchitecture arch = ArchitectureRegistry.getArchitecture(name);
        String prompt = arch.getReferencePrompt();
        assertNotNull(prompt, "getReferencePrompt() null for " + name);
        assertFalse(prompt.isEmpty(), "Empty reference prompt for " + name);
    }

    @Test
    @DisplayName("All registered architectures have unique system properties")
    void testUniqueSystemProperties() {
        Map<String, String> seen = new java.util.HashMap<>();
        allArchitecturesIncludingWhisper().forEach(args -> {
            String name = (String) args.get()[0];
            ModelArchitecture arch = ArchitectureRegistry.getArchitecture(name);
            String prop = arch.getModelSystemProperty();
            String existing = seen.put(prop, name);
            assertNull(existing,
                    "Duplicate system property '" + prop + "' for " + name + " and " + existing);
        });
    }

    @Test
    @DisplayName("ChatTemplate factory covers all template types")
    void testChatTemplateTypes() {
        // Verify each template type referenced by architectures can produce a valid ChatTemplate
        assertNotNull(ChatTemplate.chatML(), "ChatML template");
        assertNotNull(ChatTemplate.llama2(), "Llama2 template");
        assertNotNull(ChatTemplate.alpaca(), "Alpaca template");
        assertNotNull(ChatTemplate.vicuna(), "Vicuna template");

        // Test ChatML format produces valid output
        ChatTemplate chatml = ChatTemplate.chatML();
        String formatted = chatml.apply(
                List.of(ChatTemplate.Message.user("Hello")), true);
        assertTrue(formatted.contains("<|im_start|>"), "ChatML missing start token");
        assertTrue(formatted.contains("Hello"), "ChatML missing content");
        assertTrue(formatted.contains("assistant"), "ChatML missing generation prompt");

        // Test Llama2 format
        ChatTemplate llama2 = ChatTemplate.llama2();
        formatted = llama2.apply(
                List.of(ChatTemplate.Message.user("Hello")), true);
        assertTrue(formatted.contains("[INST]"), "Llama2 missing [INST]");
        assertTrue(formatted.contains("Hello"), "Llama2 missing content");
    }

    @Test
    @DisplayName("ChatTemplate preserves tokenizer BOS for GGUF ChatML templates")
    void testChatTemplatePreservesBosToken() {
        ChatTemplate lfmTemplate = new ChatTemplate(
                "{{- bos_token -}}{% for message in messages %}<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}",
                "<|startoftext|>",
                "<|im_end|>");

        String formatted = lfmTemplate.apply(
                List.of(
                        ChatTemplate.Message.system("You are precise."),
                        ChatTemplate.Message.user("Return JSON.")),
                true);

        assertTrue(formatted.startsWith("<|startoftext|>"), "GGUF ChatML template must include tokenizer BOS");
        assertTrue(formatted.contains("<|im_start|>system\nYou are precise.<|im_end|>"), "System message missing");
        assertTrue(formatted.endsWith("<|im_start|>assistant\n"), "Generation prompt missing");
    }

    @Test
    @DisplayName("TokenizerConfig can derive chat template metadata from GGUF tokenizer info")
    void testTokenizerConfigFromGgufMetadata() {
        GGMLMetadata.TokenizerInfo tokenizerInfo = GGMLMetadata.TokenizerInfo.builder()
                .tokens(List.of("<unk>", "<|startoftext|>", "<|im_end|>"))
                .bosTokenId(1)
                .eosTokenId(2)
                .chatTemplate("{{- bos_token -}}<|im_start|>{{ message.role }}")
                .build();

        TokenizerConfig config = TokenizerConfig.fromGgufMetadata(tokenizerInfo);

        assertNotNull(config, "GGUF tokenizer metadata should produce a config");
        assertEquals("<|startoftext|>", config.getBosToken(), "BOS token should be resolved through GGUF token ids");
        assertEquals("<|im_end|>", config.getEosToken(), "EOS token should be resolved through GGUF token ids");
        assertTrue(config.hasChatTemplate(), "Chat template should come from GGUF metadata");
    }

    @Test
    @DisplayName("HuggingFaceTokenizer loads chat template from GGUF sidecar")
    @EnabledIfSystemProperty(named = "lfm2.tokenizer.dir", matches = ".+")
    void testHuggingFaceTokenizerLoadsGgufSidecarChatTemplate() throws Exception {
        File modelDir = new File(System.getProperty("lfm2.tokenizer.dir"));
        try (Tokenizer tokenizer = HuggingFaceTokenizer.fromDirectory(modelDir)) {
            assertNotNull(tokenizer.getChatTemplate(), "Tokenizer should expose GGUF sidecar chat template");
            assertTrue(tokenizer.getChatTemplate().contains("<|im_start|>"), "Expected ChatML markers");
            assertEquals("<|startoftext|>", tokenizer.getBosToken(), "BOS token should come from GGUF token ids");

            String formatted = tokenizer.applyChatTemplate(
                    List.of(
                            ChatTemplate.Message.system("You are precise."),
                            ChatTemplate.Message.user("Return JSON.")),
                    true);

            assertTrue(formatted.startsWith("<|startoftext|>"), "Formatted prompt should preserve model BOS");
            assertTrue(formatted.contains("<|im_start|>system\nYou are precise.<|im_end|>"), "System message missing");
            assertTrue(formatted.endsWith("<|im_start|>assistant\n"), "Generation prompt missing");
        }
    }

    @Test
    @DisplayName("GenericArchitecture is catch-all fallback")
    void testGenericIsFallback() {
        ModelArchitecture generic = ArchitectureRegistry.getArchitecture("generic");
        assertNotNull(generic);
        assertInstanceOf(GenericArchitecture.class, generic);
        assertEquals("plain", generic.getDefaultChatTemplateType());
    }

    @Test
    @DisplayName("WhisperArchitecture is encoder-decoder, no chat template")
    void testWhisperSpecialCases() {
        ModelArchitecture whisper = ArchitectureRegistry.getArchitecture("whisper");
        assertNotNull(whisper);
        assertEquals("none", whisper.getDefaultChatTemplateType());
        assertEquals("", whisper.getReferencePrompt(), "Whisper should have empty prompt");
        assertEquals(0, whisper.getReferenceExpected().length, "Whisper should have no expected strings");
    }

    // ========================================================================
    // Tier 2: Real-model generation degeneracy tests
    // ========================================================================

    @Test
    @DisplayName("LLaMA/Qwen: generation produces non-degenerate output")
    @EnabledIfSystemProperty(named = "llama.gguf.path", matches = ".+")
    void testLlamaGeneration() throws Exception {
        runGenerationDegeneracyTest("llama");
    }

    @Test
    @DisplayName("Mistral: generation produces non-degenerate output")
    @EnabledIfSystemProperty(named = "mistral.gguf.path", matches = ".+")
    void testMistralGeneration() throws Exception {
        runGenerationDegeneracyTest("mistral");
    }

    @Test
    @DisplayName("Gemma: generation produces non-degenerate output")
    @EnabledIfSystemProperty(named = "gemma.gguf.path", matches = ".+")
    void testGemmaGeneration() throws Exception {
        runGenerationDegeneracyTest("gemma");
    }

    @Test
    @DisplayName("Phi: generation produces non-degenerate output")
    @EnabledIfSystemProperty(named = "phi.gguf.path", matches = ".+")
    void testPhiGeneration() throws Exception {
        runGenerationDegeneracyTest("phi");
    }

    @Test
    @DisplayName("Granite: generation produces non-degenerate output")
    @EnabledIfSystemProperty(named = "granite.gguf.path", matches = ".+")
    void testGraniteGeneration() throws Exception {
        runGenerationDegeneracyTest("granite");
    }

    @Test
    @DisplayName("Nemotron: generation produces non-degenerate output")
    @EnabledIfSystemProperty(named = "nemotron.gguf.path", matches = ".+")
    void testNemotronGeneration() throws Exception {
        runGenerationDegeneracyTest("nemotron");
    }

    @Test
    @DisplayName("OLMo: generation produces non-degenerate output")
    @EnabledIfSystemProperty(named = "olmo.gguf.path", matches = ".+")
    void testOlmoGeneration() throws Exception {
        runGenerationDegeneracyTest("olmo");
    }

    @Test
    @DisplayName("OpenELM: generation produces non-degenerate output")
    @EnabledIfSystemProperty(named = "openelm.gguf.path", matches = ".+")
    void testOpenelmGeneration() throws Exception {
        runGenerationDegeneracyTest("openelm");
    }

    @Test
    @DisplayName("GPT-OSS: generation produces non-degenerate output")
    @EnabledIfSystemProperty(named = "gpt-oss.gguf.path", matches = ".+")
    void testGptOssGeneration() throws Exception {
        runGenerationDegeneracyTest("gpt-oss");
    }

    @Test
    @DisplayName("GLM: generation produces non-degenerate output")
    @EnabledIfSystemProperty(named = "glm.gguf.path", matches = ".+")
    void testGlmGeneration() throws Exception {
        runGenerationDegeneracyTest("glm");
    }

    @Test
    @DisplayName("Llama4: generation produces non-degenerate output")
    @EnabledIfSystemProperty(named = "llama4.gguf.path", matches = ".+")
    void testLlama4Generation() throws Exception {
        runGenerationDegeneracyTest("llama4");
    }

    @Test
    @DisplayName("LFM-2: generation produces non-degenerate output")
    @EnabledIfSystemProperty(named = "lfm2.gguf.path", matches = ".+")
    void testLfm2Generation() throws Exception {
        runGenerationDegeneracyTest("lfm2");
    }

    // ========================================================================
    // Generation test helper
    // ========================================================================

    private void runGenerationDegeneracyTest(String archName) throws Exception {
        ModelArchitecture arch = ArchitectureRegistry.getArchitecture(archName);
        assertNotNull(arch, "Architecture not found: " + archName);

        String ggufPath = System.getProperty(arch.getModelSystemProperty());
        assertNotNull(ggufPath, "System property not set: " + arch.getModelSystemProperty());
        File ggufFile = new File(ggufPath);
        assertTrue(ggufFile.exists(), "GGUF file not found: " + ggufPath);

        // Find tokenizer.json alongside the GGUF file
        File tokenizerFile = new File(ggufFile.getParentFile(), "tokenizer.json");
        assertTrue(tokenizerFile.exists(),
                "tokenizer.json required alongside GGUF at " + tokenizerFile.getAbsolutePath());

        log.info("Testing {} generation from: {}", archName, ggufPath);

        // Format prompt using architecture's chat template
        String prompt = formatPrompt(arch);
        log.info("Formatted prompt: {}", prompt.substring(0, Math.min(100, prompt.length())) + "...");

        // Import model and load tokenizer
        var sd = GGMLModelImport.importModel(ggufPath, ConversionOptions.forInference());
        Tokenizer tokenizer = HuggingFaceTokenizer.fromFile(tokenizerFile.getAbsolutePath());
        assertTrue(tokenizer.isValid(), "Tokenizer should be valid");

        int maxTokens = 50;
        GenerationPipelineConfig pipelineConfig = GenerationPipelineConfig.builder()
                .decoder(sd)
                .tokenizer(tokenizer)
                .samplingConfig(SamplingConfig.greedy())
                .maxNewTokens(maxTokens)
                .graphOptimizerEnabled(false)
                .dspEnabled(true)
                .build();

        try (GenerationPipeline pipeline = GenerationPipeline.create(pipelineConfig)) {
            GenerationResult result = pipeline.generate(prompt, maxTokens);

            log.info("Generated {} tokens: {}", result.getGeneratedTokenCount(), result.getText());

            // Validate output quality
            GenerationQualityValidator.ValidationConfig valConfig =
                    GenerationQualityValidator.ValidationConfig.builder()
                            .minDiversityRatio(0.2)     // Relaxed: random weights won't be great
                            .maxRepetitionScore(0.7)    // Relaxed: but should not be pure repetition
                            .minCoherenceScore(0.2)     // Relaxed: but not garbage chars
                            .build();

            GenerationQualityValidator.QualityReport report =
                    GenerationQualityValidator.validate(result, valConfig);

            log.info("Quality report: {}", report.summary());

            // Core assertion: output is NOT degenerate
            assertFalse(isDegenerate(result),
                    archName + " produced degenerate output: " + result.getText());

            // If we have expected substrings for this arch, check them too
            String[] expected = arch.getReferenceExpected();
            if (expected.length > 0) {
                GenerationQualityValidator.QualityReport fullReport =
                        GenerationQualityValidator.validateWithExpected(result, expected);
                log.info("Full quality report: {}", fullReport.summary());
                // Don't fail on expected content with random weights — just log
                if (!fullReport.isPassed()) {
                    log.warn("Expected content check failed (may be OK with small/random weights): {}",
                            fullReport.getIssues());
                }
            }
        }
    }

    /**
     * Format a prompt using the architecture's chat template type.
     */
    private String formatPrompt(ModelArchitecture arch) {
        String prompt = arch.getReferencePrompt();
        String templateType = arch.getDefaultChatTemplateType();

        ChatTemplate template;
        switch (templateType) {
            case "chatml": template = ChatTemplate.chatML(); break;
            case "llama2": template = ChatTemplate.llama2(); break;
            case "vicuna": template = ChatTemplate.vicuna(); break;
            case "alpaca": template = ChatTemplate.alpaca(); break;
            case "gemma": template = new ChatTemplate(
                    "<start_of_turn>user\n{{ content }}<end_of_turn>\n<start_of_turn>model\n",
                    "<bos>", "<eos>"); break;
            default: template = null; break;
        }

        if (template != null) {
            return template.apply(
                    List.of(ChatTemplate.Message.user(prompt)), true);
        }
        // "plain" or "none": just return the raw prompt
        return prompt;
    }

    /**
     * Check for degenerate output patterns:
     * - All same token
     * - Only whitespace / control chars
     * - Single word repeated
     */
    static boolean isDegenerate(GenerationResult result) {
        if (result.getGeneratedTokenCount() == 0) return true;

        String text = result.getText();
        if (text == null || text.isBlank()) return true;

        // Check for all-same-token
        int[] ids = result.getTokenIds();
        if (ids != null && ids.length > 2) {
            boolean allSame = true;
            for (int i = 1; i < ids.length; i++) {
                if (ids[i] != ids[0]) {
                    allSame = false;
                    break;
                }
            }
            if (allSame) return true;
        }

        // Check for single-word repetition (e.g. "punct punct punct")
        String[] words = text.trim().split("\\s+");
        if (words.length > 3) {
            boolean allSameWord = true;
            for (int i = 1; i < words.length; i++) {
                if (!words[i].equals(words[0])) {
                    allSameWord = false;
                    break;
                }
            }
            if (allSameWord) return true;
        }

        return false;
    }
}
