/* ******************************************************************************
 * Apache License 2.0
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
package org.eclipse.deeplearning4j.sdx.aot;

import org.eclipse.deeplearning4j.llm.tokenizer.ChatTemplate;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.api.parallel.ResourceLock;
import org.junit.jupiter.api.parallel.Resources;
import org.nd4j.common.config.ND4JSystemProperties;
import org.nd4j.dsp.model.SdxTargetProfile;
import org.nd4j.dsp.runtime.SdxRuntime;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.shade.jackson.databind.JsonNode;
import org.nd4j.shade.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for SdxLlmCore tokenizer utilities.
 *
 * These tests verify that tokenizer resolution requires the canonical Hugging Face
 * tokenizer assets; generation must never synthesize a tokenizer from GGUF metadata.
 */
@ResourceLock(Resources.SYSTEM_PROPERTIES)
class SdxLlmCoreTokenizerTest {

    @TempDir
    Path temporaryDirectory;

    private String originalDiagnostics;
    private String originalDiagnosticsLevel;
    private String originalNativeDump;

    @BeforeEach
    void captureDiagnosticProperties() {
        originalDiagnostics = System.getProperty(ND4JSystemProperties.DSP_DIAGNOSTICS);
        originalDiagnosticsLevel = System.getProperty(ND4JSystemProperties.DSP_DIAGNOSTICS_LEVEL);
        originalNativeDump = System.getProperty(ND4JSystemProperties.DSP_NATIVE_DUMP_OUTPUTS);
    }

    @AfterEach
    void restoreDiagnosticProperties() {
        restoreProperty(ND4JSystemProperties.DSP_DIAGNOSTICS, originalDiagnostics);
        restoreProperty(ND4JSystemProperties.DSP_DIAGNOSTICS_LEVEL, originalDiagnosticsLevel);
        restoreProperty(ND4JSystemProperties.DSP_NATIVE_DUMP_OUTPUTS, originalNativeDump);
    }

    @Test
    void opSanityDiagnosticModeEnablesComparableNativeValueRecords() {
        JsonNode options = new ObjectMapper().createObjectNode()
                .put("diagnosticMode", "op_sanity");

        assertEquals("op_sanity", SdxGgufModelPreparer.configureDiagnostics(options));
        assertEquals("VERIFY", System.getProperty(ND4JSystemProperties.DSP_DIAGNOSTICS));
        assertEquals("full", System.getProperty(ND4JSystemProperties.DSP_DIAGNOSTICS_LEVEL));
        assertEquals("true", System.getProperty(ND4JSystemProperties.DSP_NATIVE_DUMP_OUTPUTS));
    }

    @Test
    void offDiagnosticModeIsDefaultAndClearsPriorImporterDiagnostics() {
        System.setProperty(ND4JSystemProperties.DSP_DIAGNOSTICS, "ALL");
        System.setProperty(ND4JSystemProperties.DSP_DIAGNOSTICS_LEVEL, "full");
        System.setProperty(ND4JSystemProperties.DSP_NATIVE_DUMP_OUTPUTS, "true");

        assertEquals(
                "off",
                SdxGgufModelPreparer.configureDiagnostics(new ObjectMapper().createObjectNode()));
        assertNull(System.getProperty(ND4JSystemProperties.DSP_DIAGNOSTICS));
        assertNull(System.getProperty(ND4JSystemProperties.DSP_DIAGNOSTICS_LEVEL));
        assertNull(System.getProperty(ND4JSystemProperties.DSP_NATIVE_DUMP_OUTPUTS));
    }

    @Test
    void backendAuditModeCapturesPlacementAndReplayWithoutValueDumps() {
        JsonNode options = new ObjectMapper().createObjectNode()
                .put("diagnosticMode", "backend_audit");

        assertEquals("backend_audit", SdxGgufModelPreparer.configureDiagnostics(options));
        assertEquals("BACKEND,COMPILE,EXECUTE,SEGMENT,EMULATED_REPLAY,GRAPH_REPLAY",
                System.getProperty(ND4JSystemProperties.DSP_DIAGNOSTICS));
        assertEquals("detailed", System.getProperty(ND4JSystemProperties.DSP_DIAGNOSTICS_LEVEL));
        assertNull(System.getProperty(ND4JSystemProperties.DSP_NATIVE_DUMP_OUTPUTS));
    }

    @Test
    void unknownDiagnosticModeDoesNotMutateImporterDiagnostics() {
        System.setProperty(ND4JSystemProperties.DSP_DIAGNOSTICS, "VERIFY");
        System.setProperty(ND4JSystemProperties.DSP_DIAGNOSTICS_LEVEL, "detailed");
        System.setProperty(ND4JSystemProperties.DSP_NATIVE_DUMP_OUTPUTS, "false");
        JsonNode options = new ObjectMapper().createObjectNode()
                .put("diagnosticMode", "unknown");

        assertThrows(IllegalArgumentException.class,
                () -> SdxGgufModelPreparer.configureDiagnostics(options));
        assertEquals("VERIFY", System.getProperty(ND4JSystemProperties.DSP_DIAGNOSTICS));
        assertEquals("detailed", System.getProperty(ND4JSystemProperties.DSP_DIAGNOSTICS_LEVEL));
        assertEquals("false", System.getProperty(ND4JSystemProperties.DSP_NATIVE_DUMP_OUTPUTS));
    }

    private static void restoreProperty(String name, String value) {
        if (value == null) System.clearProperty(name);
        else System.setProperty(name, value);
    }

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
    void tokenizerCacheIdentityTracksExactHuggingFaceAssetBytes() throws Exception {
        Path tokenizer = temporaryDirectory.resolve("tokenizer.json");
        Path tokenizerConfig = temporaryDirectory.resolve("tokenizer_config.json");
        Path generationConfig = temporaryDirectory.resolve("generation_config.json");
        Files.writeString(tokenizer, "{\"decoder\":{\"type\":\"ByteLevel\"}}",
                StandardCharsets.UTF_8);
        Files.writeString(tokenizerConfig, "{\"chat_template\":\"{{ messages }}\"}",
                StandardCharsets.UTF_8);
        Files.writeString(generationConfig, "{\"eos_token_id\":1}",
                StandardCharsets.UTF_8);

        String original = SdxGgufModelPreparer.tokenizerAssetIdentity(
                tokenizer, tokenizerConfig, generationConfig);
        String repeated = SdxGgufModelPreparer.tokenizerAssetIdentity(
                tokenizer, tokenizerConfig, generationConfig);
        Files.writeString(tokenizer, "{\"decoder\":null}", StandardCharsets.UTF_8);
        String changedTokenizer = SdxGgufModelPreparer.tokenizerAssetIdentity(
                tokenizer, tokenizerConfig, generationConfig);
        Files.writeString(tokenizerConfig, "{\"chat_template\":\"changed\"}",
                StandardCharsets.UTF_8);
        String changedConfig = SdxGgufModelPreparer.tokenizerAssetIdentity(
                tokenizer, tokenizerConfig, generationConfig);

        assertEquals(original, repeated);
        assertEquals(64, original.length());
        assertNotEquals(original, changedTokenizer);
        assertNotEquals(changedTokenizer, changedConfig);
        assertNotEquals(changedConfig, SdxGgufModelPreparer.tokenizerAssetIdentity(
                tokenizer, tokenizerConfig, null));
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
    void rejectsMissingCanonicalTokenizerInsteadOfSynthesizingFromGguf() throws Exception {
        Path model = temporaryDirectory.resolve("model.gguf");
        Files.write(model, new byte[] {'G', 'G', 'U', 'F'});

        IOException failure = assertThrows(IOException.class,
                () -> SdxLlmCore.resolveTokenizer(model.toString(), null));
        assertTrue(failure.getMessage().contains("Canonical Hugging Face tokenizer.json"));
        assertTrue(failure.getMessage().contains("reconstruction is disabled"));
    }

    @Test
    void structuredChatRequestPreservesToolsCallsAndResults() throws Exception {
        String requestJson = "{"
                + "\"messages\":["
                + "{\"role\":\"user\",\"content\":\"Find Alice\"},"
                + "{\"role\":\"assistant\",\"content\":\"raw\",\"tool_calls\":[{"
                + "\"id\":\"c1\",\"type\":\"function\",\"function\":{"
                + "\"name\":\"lookup_record\",\"arguments\":{\"query\":\"Alice\"}}}]},"
                + "{\"role\":\"tool\",\"content\":\"{}\","
                + "\"tool_call_id\":\"c1\",\"name\":\"lookup_record\"}],"
                + "\"tools\":[{\"name\":\"lookup_record\",\"description\":\"Search\","
                + "\"parameters\":{\"type\":\"object\"}}],"
                + "\"tool_choice\":\"required\"}";

        ChatTemplate.Request request = SdxLlmCore.parseChatRequest(requestJson);

        assertEquals(ChatTemplate.ToolChoice.REQUIRED, request.getToolChoice());
        assertEquals("lookup_record", request.getTools().get(0).getName());
        assertEquals("lookup_record",
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
        assertEquals("sdx-prepared-text-model-v5", SdxGgufModelPreparer.PREPARED_SCHEMA);
        assertEquals("ggml-runtime-packed-gdn-v7", SdxGgufModelPreparer.GRAPH_IMPORT_ABI);
    }

    @Test
    void modelUnloadPurgesDspCacheAfterPipelineStateAndBeforeDecoderClose() throws IOException {
        String source = Files.readString(Path.of(
                "src/main/java/org/eclipse/deeplearning4j/sdx/aot/SdxLlmCore.java"));
        int closeMethod = source.indexOf("public void close() {");
        int pipelineClose = source.indexOf("pipeline.close();", closeMethod);
        int cacheClear = source.indexOf("decoder.clearDynamicShapePlanCache();", closeMethod);
        int decoderClose = source.indexOf("decoder.close();", closeMethod);

        assertTrue(closeMethod >= 0);
        assertTrue(pipelineClose > closeMethod);
        assertTrue(cacheClear > pipelineClose);
        assertTrue(decoderClose > cacheClear);
    }

    @Test
    void runtimeQuantizedProfilesKeepPackedActivationsAndKvStateFloat32() {
        for (String mode : List.of(
                "RUNTIME_QUANTIZED_MATMUL",
                "RUNTIME_QUANTIZED_INT8",
                "RUNTIME_QUANTIZED_INT4")) {
            JsonNode options = new ObjectMapper().createObjectNode()
                    .put("graphImportAbi", SdxGgufModelPreparer.GRAPH_IMPORT_ABI)
                    .put("conversionMode", mode);

            SdxGgufModelPreparer.PreparationProfile profile =
                    SdxGgufModelPreparer.PreparationProfile.from(options);

            assertEquals(DataType.FLOAT, profile.conversionOptions().getTargetDataType(), mode);
        }
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
                        "{\"graphImportAbi\":\"" + SdxGgufModelPreparer.GRAPH_IMPORT_ABI +
                                "\",\"verifiedSourceSha256\":\"" + "0".repeat(64) +
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
