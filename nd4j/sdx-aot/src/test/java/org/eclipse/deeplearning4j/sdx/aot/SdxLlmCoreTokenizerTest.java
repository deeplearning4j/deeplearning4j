/* ******************************************************************************
 * Apache License 2.0
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
package org.eclipse.deeplearning4j.sdx.aot;

import org.eclipse.deeplearning4j.llm.tokenizer.Tokenizer;
import org.eclipse.deeplearning4j.llm.tokenizer.ChatTemplate;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.nd4j.dsp.model.SdxTargetProfile;
import org.nd4j.dsp.runtime.SdxRuntime;
import org.nd4j.shade.jackson.databind.JsonNode;
import org.nd4j.shade.jackson.databind.ObjectMapper;

import java.io.File;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for SdxLlmCore tokenizer utilities.
 *
 * These tests exercise the GGUF-embedded tokenizer JSON builder (R8-2 fix)
 * without requiring a native image or a real GGUF file on disk — they operate
 * purely in Java using the shaded Jackson API that sdx-aot already depends on.
 */
class SdxLlmCoreTokenizerTest {

    @TempDir
    Path temporaryDirectory;

    @Test
    void rawSourceIdentityAcceptsArbitraryContainerExtensions() throws Exception {
        Path source = temporaryDirectory.resolve("model.gguf");
        Files.write(source, new byte[] {1, 2, 3, 4, 5});

        SdxGgufModelPreparer.RawSourceIdentity identity =
                SdxGgufModelPreparer.RawSourceIdentity.identify(source);

        assertEquals(5L, identity.bytes());
        assertEquals(64, identity.sha256().length());
    }

    @Test
    void rejectsHuggingFaceGenerationDefaultsAsNativeSdxContract() throws Exception {
        Path config = temporaryDirectory.resolve("generation_config.json");
        Files.writeString(config,
                "{\"bos_token_id\":1,\"eos_token_id\":[2,3],\"temperature\":0.7}",
                StandardCharsets.UTF_8);

        assertFalse(SdxGgufModelPreparer.isNativeTextGenerationContract(config));
    }

    @Test
    void acceptsCompleteRecurrentNativeSdxContract() throws Exception {
        Path config = temporaryDirectory.resolve("text-generation.json");
        Files.writeString(config,
                "{\"formatVersion\":2,"
                        + "\"profile\":\"causal-lm-in-graph-state-v2\","
                        + "\"io\":{"
                        + "\"inputIds\":\"input_ids\",\"causalMask\":\"attention_mask\","
                        + "\"positionOffset\":\"position_offset\","
                        + "\"cachePosition\":\"cache_position\","
                        + "\"actualSequenceLength\":\"actual_sequence_length\","
                        + "\"logits\":\"logits\","
                        + "\"kvKeyInputs\":[\"past_key_values.0.key\"],"
                        + "\"kvValueInputs\":[\"past_key_values.0.value\"],"
                        + "\"prefillKeyOutputs\":[\"k_rope_0\"],"
                        + "\"prefillValueOutputs\":[\"v_heads_0\"],"
                        + "\"recurrentStates\":[{\"input\":\"state.0\","
                        + "\"output\":\"state_out.0\",\"kind\":\"GDN\","
                        + "\"dataType\":\"FLOAT32\",\"shape\":[1]}]},"
                        + "\"execution\":{\"kvLayout\":\"BSHD\","
                        + "\"kvDtype\":\"FLOAT32\",\"maskDtype\":\"FLOAT32\","
                        + "\"planOwnsKvScatter\":true},"
                        + "\"tokens\":{\"padId\":0,\"eosIds\":[2,3]},"
                        + "\"limits\":{\"contextLength\":128,\"maxPrefillLength\":127}}",
                StandardCharsets.UTF_8);

        assertTrue(SdxGgufModelPreparer.isNativeTextGenerationContract(config));
    }

    @Test
    void buildBpeTokenizerJson_wellFormed() throws Exception {
        // Minimal GPT-2 BPE vocabulary: letters a-z + space + BOS/EOS
        List<String> tokens = new ArrayList<>();
        tokens.add("<|endoftext|>"); // id=0 (BOS/EOS for GPT-2)
        for (char c = 'a'; c <= 'z'; c++) tokens.add(String.valueOf(c));
        tokens.add("Ġ"); // space prefix (byte-level BPE convention)

        List<String> merges = List.of("a b", "b c", "c d");

        String json = SdxLlmCore.buildBpeTokenizerJson(tokens, merges, 0, 0, "gpt2");
        assertNotNull(json);
        assertFalse(json.isBlank());

        // Must parse as valid JSON
        ObjectMapper m = new ObjectMapper();
        JsonNode root = m.readTree(json);
        assertEquals("1.0", root.path("version").asText());

        JsonNode model = root.path("model");
        assertEquals("BPE", model.path("type").asText());

        // Vocab must contain all tokens
        JsonNode vocab = model.path("vocab");
        assertTrue(vocab.has("<|endoftext|>"), "Expected BOS token in vocab");
        assertEquals(0, vocab.get("<|endoftext|>").asInt());
        assertEquals(1, vocab.get("a").asInt());

        // Merges
        JsonNode mergesNode = model.path("merges");
        assertTrue(mergesNode.isArray());
        assertEquals(3, mergesNode.size());
        assertEquals("a b", mergesNode.get(0).asText());

        // added_tokens: BOS == EOS (same id 0), so only one added_token entry expected
        JsonNode addedTokens = root.path("added_tokens");
        assertTrue(addedTokens.isArray());
        assertEquals(1, addedTokens.size());
        assertEquals(0, addedTokens.get(0).path("id").asInt());
        assertTrue(addedTokens.get(0).path("special").asBoolean());
    }

    @Test
    void buildBpeTokenizerJson_separateBosEos() throws Exception {
        List<String> tokens = new ArrayList<>();
        tokens.add("<|im_start|>"); // id=0  BOS
        tokens.add("<|im_end|>");   // id=1  EOS
        for (char c = 'a'; c <= 'z'; c++) tokens.add(String.valueOf(c));

        String json = SdxLlmCore.buildBpeTokenizerJson(tokens, List.of(), 0, 1, "qwen2");
        JsonNode root = new ObjectMapper().readTree(json);

        JsonNode addedTokens = root.path("added_tokens");
        assertEquals(2, addedTokens.size(), "Separate BOS and EOS should both appear");
        assertEquals(0, addedTokens.get(0).path("id").asInt());
        assertEquals(1, addedTokens.get(1).path("id").asInt());
    }

    @Test
    void buildBpeTokenizerJson_emptyMerges() throws Exception {
        List<String> tokens = List.of("a", "b", "c");
        // SentencePiece-based tokenizers may have no merges
        String json = SdxLlmCore.buildBpeTokenizerJson(tokens, List.of(), -1, -1, "llama");
        JsonNode root = new ObjectMapper().readTree(json);
        JsonNode mergesNode = root.path("model").path("merges");
        assertTrue(mergesNode.isArray());
        assertEquals(0, mergesNode.size());

        // Out-of-range BOS/EOS (-1) → added_tokens array must be empty (no AIOOBE)
        assertEquals(0, root.path("added_tokens").size());
    }

    @Test
    void buildBpeTokenizerJson_nullabilityGuards() throws Exception {
        List<String> tokens = List.of("a", "b");
        // Null merges list should not throw
        String json = SdxLlmCore.buildBpeTokenizerJson(tokens, null, 0, 1, "gpt2");
        JsonNode root = new ObjectMapper().readTree(json);
        JsonNode mergesNode = root.path("model").path("merges");
        assertTrue(mergesNode.isArray());
        assertEquals(0, mergesNode.size(), "null merges should produce empty merges array");
    }

    // -----------------------------------------------------------------------
    // R8 item 4: token_type array — CONTROL tokens must appear in added_tokens
    // -----------------------------------------------------------------------

    /**
     * Synthetic Qwen2.5-style vocabulary: 4 NORMAL tokens + 3 CONTROL tokens.
     * With tokenTypes provided, all non-NORMAL (type != 1) tokens must land in
     * added_tokens with special=true, deduplicated against BOS/EOS.
     */
    @Test
    void buildBpeTokenizerJson_controlTokensPromotedToAddedTokens() throws Exception {
        // Indices: 0=<|endoftext|> NORMAL, 1=hello NORMAL, 2=world NORMAL,
        //          3=<|im_start|> CONTROL(3), 4=<|im_end|> CONTROL(3), 5=<|tool_call|> CONTROL(3)
        List<String> tokens = new ArrayList<>();
        tokens.add("<|endoftext|>");  // 0 NORMAL (BOS)
        tokens.add("hello");          // 1 NORMAL
        tokens.add("world");          // 2 NORMAL
        tokens.add("<|im_start|>");   // 3 CONTROL
        tokens.add("<|im_end|>");     // 4 CONTROL (EOS)
        tokens.add("<|tool_call|>");  // 5 CONTROL

        // GGUF token_type: 1=NORMAL, 3=CONTROL
        int[] tokenTypes = {1, 1, 1, 3, 3, 3};

        String json = SdxLlmCore.buildBpeTokenizerJson(tokens, List.of(), 0, 4, "qwen2", tokenTypes);
        JsonNode root = new ObjectMapper().readTree(json);
        JsonNode addedTokens = root.path("added_tokens");
        assertTrue(addedTokens.isArray());

        // Collect all added token ids
        Set<Integer> addedIds = new HashSet<>();
        for (JsonNode t : addedTokens) {
            assertTrue(t.path("special").asBoolean(), "All added tokens must have special=true");
            addedIds.add(t.path("id").asInt());
        }

        // BOS(0) + EOS(4) + <|im_start|>(3) + <|tool_call|>(5) — NORMAL tokens (1,2) excluded
        assertTrue(addedIds.contains(0), "BOS id=0 must be present");
        assertTrue(addedIds.contains(3), "<|im_start|> id=3 must be present (CONTROL type)");
        assertTrue(addedIds.contains(4), "<|im_end|>/EOS id=4 must be present");
        assertTrue(addedIds.contains(5), "<|tool_call|> id=5 must be present (CONTROL type)");
        assertFalse(addedIds.contains(1), "NORMAL token id=1 must NOT be in added_tokens");
        assertFalse(addedIds.contains(2), "NORMAL token id=2 must NOT be in added_tokens");
        assertEquals(4, addedIds.size(), "Exactly 4 special tokens (BOS/EOS + 2 extra CONTROL)");
    }

    /**
     * Null tokenTypes: behaviour must be identical to the legacy 5-arg overload
     * (only BOS and EOS in added_tokens).
     */
    @Test
    void buildBpeTokenizerJson_nullTokenTypesBackwardCompat() throws Exception {
        List<String> tokens = new ArrayList<>();
        tokens.add("<|bos|>"); // 0 BOS
        tokens.add("<|eos|>"); // 1 EOS
        tokens.add("hello");   // 2 NORMAL

        // Null tokenTypes → same as old 5-arg path
        String json6 = SdxLlmCore.buildBpeTokenizerJson(tokens, List.of(), 0, 1, "gpt2", null);
        String json5 = SdxLlmCore.buildBpeTokenizerJson(tokens, List.of(), 0, 1, "gpt2");
        assertEquals(json5, json6, "null tokenTypes should produce identical output to 5-arg overload");

        JsonNode root = new ObjectMapper().readTree(json6);
        assertEquals(2, root.path("added_tokens").size(), "Only BOS+EOS when tokenTypes=null");
    }

    /**
     * R8 item 4 integration — real GGUF file, embedded tokenizer path.
     * Asserts that <|im_start|> and <|im_end|> each encode to EXACTLY ONE token id
     * (151644 and 151645 respectively) via the embedded (no-sidecar) path.
     *
     * Skipped automatically when the model is not present (CI without model cache).
     */
    @Test
    void embeddedTokenizer_qwen25_chatmlMarkersAreSingleIds() throws Exception {
        // Check both 0.5b q4 and fp16; use whichever is present.
        String[] candidates = {
            System.getProperty("user.home") + "/.kompile/models/chat/qwen2.5-0.5b-instruct-q4_k_m.gguf",
            System.getProperty("user.home") + "/.kompile/models/chat/qwen2.5-0.5b-instruct-fp16.gguf",
            System.getProperty("user.home") + "/.kompile/models/chat/qwen2.5-1.5b-instruct-fp16.gguf",
        };
        String modelPath = null;
        for (String c : candidates) {
            if (new File(c).exists()) { modelPath = c; break; }
        }
        Assumptions.assumeTrue(modelPath != null,
                "No Qwen2.5 GGUF found under ~/.kompile/models/chat/ — skipping live tokenizer test");

        // resolveTokenizer with no sidecar → must use the GGUF-embedded path.
        // Temporarily rename any sidecar to ensure embedded path is exercised.
        File sidecar = new File(new File(modelPath).getParentFile(), "tokenizer.json");
        Assumptions.assumeFalse(sidecar.exists(),
                "Sidecar tokenizer.json present — embedded path not tested; remove it to run this test");

        Tokenizer tok = SdxLlmCore.resolveTokenizer(modelPath, null);
        assertNotNull(tok, "Tokenizer must load from embedded GGUF metadata");

        // <|im_start|> must be a single token with id 151644
        int[] imStartIds = tok.encode("<|im_start|>", false).getIds();
        assertEquals(1, imStartIds.length,
                "<|im_start|> must tokenize to exactly 1 id (was " + imStartIds.length
                + " — ChatML delimiters are split, R8 item 4 NOT fixed)");
        assertEquals(151644, imStartIds[0],
                "<|im_start|> must have id=151644");

        // <|im_end|> must be a single token with id 151645
        int[] imEndIds = tok.encode("<|im_end|>", false).getIds();
        assertEquals(1, imEndIds.length,
                "<|im_end|> must tokenize to exactly 1 id (was " + imEndIds.length + ")");
        assertEquals(151645, imEndIds[0],
                "<|im_end|> must have id=151645");
    }

    @Test
    void structuredChatRequestPreservesToolsCallsAndResults() throws Exception {
        String requestJson = "{"
                + "\"messages\":["
                + "{\"role\":\"user\",\"content\":\"Find Alice\"},"
                + "{\"role\":\"assistant\",\"content\":\"raw\",\"tool_calls\":[{"
                + "\"id\":\"c1\",\"type\":\"function\",\"function\":{"
                + "\"name\":\"graph_search\",\"arguments\":{\"query\":\"Alice\"}}}]},"
                + "{\"role\":\"tool\",\"content\":\"{}\","
                + "\"tool_call_id\":\"c1\",\"name\":\"graph_search\"}],"
                + "\"tools\":[{\"name\":\"graph_search\",\"description\":\"Search\","
                + "\"parameters\":{\"type\":\"object\"}}],"
                + "\"tool_choice\":\"required\"}";

        ChatTemplate.Request request = SdxLlmCore.parseChatRequest(requestJson);

        assertEquals(ChatTemplate.ToolChoice.REQUIRED, request.getToolChoice());
        assertEquals("graph_search", request.getTools().get(0).getName());
        assertEquals("graph_search",
                request.getMessages().get(1).getToolCalls().get(0).getName());
        assertEquals("c1", request.getMessages().get(2).getToolCallId());
        assertEquals("tool", request.getMessages().get(2).getRole());
    }

    @Test
    void structuredChatRequestRejectsMissingMessages() {
        assertThrows(IllegalArgumentException.class,
                () -> SdxLlmCore.parseChatRequest("{\"tools\":[]}"));
    }

    @Test
    void abiV2ExportsCanonicalPreparationCompiledLoadAndStreaming() {
        Set<String> exports = java.util.Arrays.stream(SdxLlmCApi.class.getDeclaredMethods())
                .map(java.lang.reflect.Method::getName)
                .collect(Collectors.toSet());
        assertEquals(2, SdxLlmCApi.ABI_VERSION);
        assertTrue(exports.contains("sdxLlmPrepareGguf"));
        assertTrue(exports.contains("sdxLlmResolveModelBundle"));
        assertTrue(exports.contains("sdxLlmLoadCompiledModel"));
        assertTrue(exports.contains("sdxLlmGenerateStreaming"));
        assertTrue(exports.contains("sdxLlmRenderChatPrompt"));
        assertTrue(exports.contains("sdxLlmParseChatResult"));
        assertEquals("sdx-prepared-text-model-v4", SdxGgufModelPreparer.PREPARED_SCHEMA);
    }

    @Test
    void rawGgufAttestationUsesPhysicalBytesBeforeCanonicalImport() throws Exception {
        Path source = temporaryDirectory.resolve("source.gguf");
        Path cache = temporaryDirectory.resolve("cache");
        Files.write(source, new byte[] {
                'G', 'G', 'U', 'F', 3, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0
        });
        IllegalArgumentException failure = assertThrows(
                IllegalArgumentException.class,
                () -> SdxGgufModelPreparer.prepare(
                        source.toString(),
                        null,
                        "android-arm64-nnapi-accelerator",
                        cache.toString(),
                        "{\"verifiedSourceSha256\":\"" + "0".repeat(64) +
                                "\",\"verifiedSourceBytes\":16}"));
        assertEquals(
                "Verified source SHA-256 did not match the GGUF bytes",
                failure.getMessage());
    }

    @Test
    void embeddedNativeImageFreezesProcessOwnedSymbolResolutionAtBuildTime() throws Exception {
        String resource = "/META-INF/native-image/org.eclipse.deeplearning4j/sdx-aot/native-image.properties";
        try (InputStream input = SdxNativeLibs.class.getResourceAsStream(resource)) {
            assertNotNull(input, "Missing SDX native-image build configuration");
            String configuration = new String(input.readAllBytes(), StandardCharsets.UTF_8);
            assertTrue(configuration.contains("-Dorg.nd4j.native.symbolResolution=process"));
            assertTrue(configuration.contains(
                    "--initialize-at-build-time=org.slf4j,ch.qos.logback," +
                            "org.eclipse.deeplearning4j.sdx.aot.SdxNativeLibs," +
                            "org.nd4j.nativeblas.NativeSymbolResolution"));
            assertTrue(configuration.contains(
                    "--initialize-at-run-time=org.nd4j.linalg.api.ops," +
                            "org.nd4j.autodiff.samediff," +
                            "org.nd4j.linalg.learning.config," +
                            "org.nd4j.linalg.api.memory.deallocation.DeallocatorService"),
                    "Stateful ND4J operation classes must initialize at runtime");
            assertTrue(configuration.contains(
                    "org.bytedeco.javacpp.Loader$Helper"),
                    "JavaCPP helper classes must initialize after the embedded native libraries are loaded");
            assertTrue(configuration.contains(
                    "--initialize-at-run-time=org.nd4j.linalg.cpu.nativecpu.NDArray," +
                            "org.nd4j.linalg.cpu.nativecpu.CpuNDArrayFactory," +
                            "org.nd4j.linalg.cpu.nativecpu.CpuBackend," +
                            "org.nd4j.linalg.cpu.nativecpu.CpuEnvironment," +
                            "org.nd4j.linalg.cpu.nativecpu.buffer.CpuDeallocator," +
                            "org.nd4j.linalg.cpu.nativecpu.bindings.Nd4jCpu$Environment"),
                    "Stateful CPU backend classes must initialize only after Android loads the process-owned native libraries");
            assertFalse(configuration.contains("org.nd4j.cpu.blas.processSymbols"));
        }

    }

    @Test
    void embeddedJavaCppSymbolLookupGuardFailsClosedForEveryEntryPoint() {
        IllegalStateException loaderFailure = assertThrows(
                IllegalStateException.class,
                () -> Target_org_bytedeco_javacpp_Loader.addressof("cblas_sgemm"));
        IllegalStateException helperFailure = assertThrows(
                IllegalStateException.class,
                () -> Target_org_bytedeco_javacpp_Loader_Helper.addressof("cublasSgemm_v2"));

        assertEquals(
                SdxNativeImageJavaCppSafety.FAILURE_PREFIX + "cblas_sgemm",
                loaderFailure.getMessage());
        assertEquals(
                SdxNativeImageJavaCppSafety.FAILURE_PREFIX + "cublasSgemm_v2",
                helperFailure.getMessage());
    }

    @Test
    void tensorG3AliasSelectsGenericArmPlannerWithoutRuntimeJit() {
        SdxTargetProfile target = SdxTargetProfile.fromId("tensor-g3");
        SdxRuntime.ModelOptions options = SdxCompiledLlmCore.runtimeOptions(target);
        assertEquals(SdxTargetProfile.ANDROID_ARM64_NNAPI_ACCELERATOR, target);
        assertEquals(SdxRuntime.SDX_BACKEND_ARM_HYBRID, options.backend);
        assertEquals(0, options.strict_backend);
        assertEquals(0, options.allow_runtime_jit);
        assertEquals(SdxRuntime.SDX_GPU_TARGET_AUTO, options.gpu_target);
    }
}
