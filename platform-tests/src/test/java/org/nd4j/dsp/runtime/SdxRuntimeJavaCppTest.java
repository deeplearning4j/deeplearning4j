/*
 *  ******************************************************************************
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.dsp.runtime;

import org.bytedeco.javacpp.BytePointer;
import org.eclipse.deeplearning4j.llm.generation.GenerationPipeline;
import org.eclipse.deeplearning4j.llm.tokenizer.ChatTemplate;
import org.eclipse.deeplearning4j.llm.tokenizer.Tokenizer;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.diagnostics.DspDiagnostics;
import org.nd4j.dsp.runtime.bindings.SdxNative;

import java.nio.file.Files;
import java.nio.file.Path;
import java.lang.reflect.Proxy;
import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class SdxRuntimeJavaCppTest {

    @Test
    void loadsGeneratedTransportAndCreatesRuntime() {
        try (SdxRuntime runtime = SdxRuntime.create()) {
            assertEquals(1, runtime.abiVersion());
            assertFalse(runtime.lastError() == null);
        }
    }

    @Test
    void selectedBackendTransportOwnsDiagnosticsLifecycle() throws Exception {
        Path report = Files.createTempFile("sdx-selected-backend-diagnostics", ".json");
        try (SdxRuntime runtime = SdxRuntime.create()) {
            runtime.configureDiagnostics(
                    DspDiagnostics.VERIFY,
                    DspDiagnostics.LEVEL_FULL,
                    report.toString());
            try {
                runtime.clearDiagnostics();
                runtime.recordDiagnosticEvent(
                        DspDiagnostics.VERIFY, "SDX_SELECTED_BACKEND_MARKER");

                runtime.flushDiagnostics();

                assertTrue(Files.readString(report).contains("SDX_SELECTED_BACKEND_MARKER"));
            } finally {
                runtime.configureDiagnostics(
                        DspDiagnostics.NONE,
                        DspDiagnostics.LEVEL_SUMMARY,
                        "");
                runtime.clearDiagnostics();
            }
        } finally {
            Files.deleteIfExists(report);
        }
    }

    @Test
    void compiledAotDiagnosticsUseTheSelectedSdxBackendTransport() throws Exception {
        Path sourceRoot = Path.of("../nd4j/sdx-aot/src/main/java/org/eclipse/deeplearning4j/sdx/aot");
        String cApi = Files.readString(sourceRoot.resolve("SdxLlmCApi.java"));
        String compiledCore = Files.readString(sourceRoot.resolve("SdxCompiledLlmCore.java"));

        assertFalse(cApi.contains("DspDiagnostics"),
                "The outer AOT image must not flush its embedded CPU NativeOps singleton");
        assertTrue(cApi.contains("core.clearDiagnostics()"));
        assertTrue(cApi.contains("core.flushDiagnostics()"));
        assertTrue(compiledCore.contains("runtime.clearDiagnostics()"));
        assertTrue(compiledCore.contains("runtime.flushDiagnostics()"));
    }

    @Test
    void compiledChatParsingUsesTheImportedModelToolProtocolByDefault() throws Exception {
        Path source = Path.of("../nd4j/sdx-aot/src/main/java/org/eclipse/deeplearning4j/sdx/aot",
                "SdxCompiledLlmCore.java");
        String compiledCore = Files.readString(source);

        assertTrue(compiledCore.contains(
                "ChatTemplate.ToolCallFormat toolCallFormat = request.getToolCallFormat()"));
        assertTrue(compiledCore.contains(
                        "GenerationPipeline.selectModelToolCallFormat(\n                    toolCallFormat, template, tokenizer)"),
                "A provider-neutral request must inherit template and tokenizer protocol metadata");
        assertTrue(compiledCore.contains(
                "request.getTools(), toolCallFormat, request.getToolChoice()"));
        assertTrue(compiledCore.contains(
                "int maxNewTokens = positive(options.path(\"maxNewTokens\")"));
        assertFalse(compiledCore.contains("int maxNewTokens = fixedContextCapacity"),
                "Per-request generation length must not be replaced by plan capacity");
    }

    @Test
    void modelToolProtocolUsesDecodedTokenizerSentinelsWithoutReverseLookup() {
        Tokenizer tokenizer = nativeSentinelTokenizer(Map.of(), Set.of(10, 11));
        ChatTemplate template = new ChatTemplate("plain JSON chat template", "", "");

        assertEquals(ChatTemplate.ToolCallFormat.NATIVE,
                GenerationPipeline.selectModelToolCallFormat(template, tokenizer));
    }

    @Test
    void modelToolProtocolUsesAddedTokenizerSentinelsAndPreservesExplicitOverride() {
        Tokenizer tokenizer = nativeSentinelTokenizer(Map.of(
                ChatTemplate.NATIVE_TOOL_CALL_START, 10,
                ChatTemplate.NATIVE_TOOL_CALL_END, 11), Set.of());
        ChatTemplate template = new ChatTemplate("plain JSON chat template", "", "");

        assertEquals(ChatTemplate.ToolCallFormat.NATIVE,
                GenerationPipeline.selectModelToolCallFormat(template, tokenizer));
        assertEquals(ChatTemplate.ToolCallFormat.JSON,
                GenerationPipeline.selectModelToolCallFormat(
                        ChatTemplate.ToolCallFormat.JSON, template, tokenizer));
    }

    @Test
    void mobileLoaderReleasesPartialOwnershipAndRequiresMappedArmHybridWeights()
            throws Exception {
        Path legacy = Path.of("../libnd4j/include/legacy/impl");
        String loader = Files.readString(legacy.resolve("NativeOps_dsp_shared.cpp"));
        String runtime = Files.readString(legacy.resolve("DspRuntimeC.cpp"));

        int allocation = loader.indexOf("handle = new LoadedModelHandle()");
        int catchBlock = loader.indexOf("catch (const std::exception& e)", allocation);
        int catchDelete = loader.indexOf("delete handle;", catchBlock);
        assertTrue(allocation >= 0);
        assertTrue(catchBlock > allocation);
        assertTrue(catchDelete > catchBlock,
                "A failed mapped-model load must release its partial owner");
        assertTrue(runtime.contains(
                "backend == static_cast<int>(SDX_BACKEND_ARM_HYBRID)"));
    }

    @Test
    void fixedGenerationPreservesTheConcreteNativeContextFailure() throws Exception {
        Path detail = Path.of("../libnd4j/include/dsp/runtime/detail/DspRuntimeInternal.h");
        Path legacy = Path.of("../libnd4j/include/legacy/impl");
        String bridge = Files.readString(detail);
        String runtime = Files.readString(legacy.resolve("DspRuntimeC.cpp"));
        String generation = Files.readString(legacy.resolve("SdxGenerationSession.cpp"));

        assertTrue(bridge.contains("std::string contextError(const sdx_context_t* context);"));
        assertTrue(runtime.contains("return context == nullptr ? std::string() : context->last_error;"));
        assertTrue(generation.contains(
                "*error = sd::dsp::runtime::detail::contextError(session->decodeContext);"));
        assertTrue(generation.contains("fixed generation step failed with status"));
    }

    @Test
    void everyNativeOpsBindingExposesDiagnosticFlush() throws Exception {
        Path backends = Path.of("../nd4j/nd4j-backends/nd4j-backend-impls");
        String[] bindings = {
                "nd4j-native/src/main/java/org/nd4j/linalg/cpu/nativecpu/bindings/Nd4jCpu.java",
                "nd4j-cuda-backend-common/src/main/java/org/nd4j/linalg/jcublas/bindings/Nd4jCuda.java",
                "nd4j-vulkan/src/main/java/org/nd4j/linalg/vulkan/bindings/Nd4jVulkan.java",
                "nd4j-tpu/src/main/java/org/nd4j/linalg/jtpu/bindings/Nd4jTpu.java",
                "nd4j-minimizer/src/main/java/org/nd4j/linalg/minimal/bindings/Nd4jMinimal.java"
        };
        for (String binding : bindings) {
            assertTrue(Files.readString(backends.resolve(binding)).contains("dspDiagFlushJson()"),
                    binding + " must expose the shared NativeOps diagnostic flush");
        }
    }

    @Test
    void diagnosticClearStartsABoundedEpochWithoutResettingCachedPlans() throws Exception {
        Path graphRoot = Path.of("../libnd4j/include/graph");
        String diagnostics = Files.readString(graphRoot.resolve("impl/DspDiagnostics.cpp"));
        String planHeader = Files.readString(graphRoot.resolve("NativeDynamicShapePlan.h"));
        String planExecution = Files.readString(graphRoot.resolve("impl/NativeDynamicShapePlan.cpp"));
        String segments = Files.readString(graphRoot.resolve("impl/NativeDynamicShapePlan_segments.cpp"));

        assertTrue(diagnostics.contains("epoch_.fetch_add(1"));
        assertTrue(planHeader.contains("int diagnosticExecuteCount()"));
        assertTrue(planHeader.contains("diagnosticEpochBaseExecuteCount_ = executeCount_"));
        assertTrue(planExecution.contains("anySegmentNeedsWarmup(), diagnosticExecCount"));
        assertTrue(planExecution.contains("backendExecutionPolicy.verifyCompiledExecution || opSanityActive"));
        assertTrue(segments.contains("plan->diagnosticExecuteCount()"));
    }

    @Test
    void mobileOptionsAreStrictAndAotOnly() {
        SdxRuntime.ModelOptions options =
                SdxRuntime.ModelOptions.mobileVulkan();

        assertEquals(SdxRuntime.SDX_BACKEND_VULKAN, options.backend);
        assertEquals(1, options.strict_backend);
        assertEquals(0, options.allow_runtime_jit);
        assertEquals(SdxRuntime.SDX_GPU_TARGET_VULKAN, options.gpu_target);

        SdxRuntime.ModelOptions hexagon =
                SdxRuntime.ModelOptions.mobileHexagon();
        assertEquals(SdxRuntime.SDX_BACKEND_HEXAGON, hexagon.backend);
        assertEquals(1, hexagon.strict_backend);
        assertEquals(0, hexagon.allow_runtime_jit);
        assertEquals(SdxRuntime.SDX_GPU_TARGET_AUTO, hexagon.gpu_target);

        String cacheDirectory = "/tmp/sdx-device-cache";
        SdxRuntime.ModelOptions nnapi = SdxRuntime.ModelOptions.mobileNnapiAccelerator()
                .deviceCompilationCacheDirectory(cacheDirectory);
        assertEquals(SdxRuntime.SDX_BACKEND_NNAPI, nnapi.backend);
        assertEquals(1, nnapi.strict_backend);
        assertEquals(0, nnapi.allow_runtime_jit);
        assertEquals(SdxRuntime.SDX_GPU_TARGET_AUTO, nnapi.gpu_target);
        assertEquals(cacheDirectory, nnapi.device_compilation_cache_directory);
        assertDoesNotThrow(() -> SdxNative.sdx_model_options_t.class.getDeclaredMethod(
                "device_compilation_cache_directory", BytePointer.class));

        SdxRuntime.RunOptions hexagonRun =
                SdxRuntime.RunOptions.mobileHexagon();
        assertEquals(SdxRuntime.SDX_BACKEND_HEXAGON, hexagonRun.backend);
        assertEquals(SdxRuntime.SDX_GPU_TARGET_AUTO, hexagonRun.gpu_target);

        SdxRuntime.RunOptions nnapiRun =
                SdxRuntime.RunOptions.mobileNnapiAccelerator();
        assertEquals(SdxRuntime.SDX_BACKEND_NNAPI, nnapiRun.backend);
        assertEquals(SdxRuntime.SDX_GPU_TARGET_AUTO, nnapiRun.gpu_target);

        SdxRuntime.ModelOptions openVino = SdxRuntime.ModelOptions.desktopOpenVino();
        assertEquals(SdxRuntime.SDX_BACKEND_OPENVINO, openVino.backend);
        assertEquals(1, openVino.strict_backend);
        assertEquals(1, openVino.allow_runtime_jit);
        assertEquals(SdxRuntime.SDX_GPU_TARGET_AUTO, openVino.gpu_target);

        SdxRuntime.RunOptions openVinoRun = SdxRuntime.RunOptions.desktopOpenVino();
        assertEquals(SdxRuntime.SDX_BACKEND_OPENVINO, openVinoRun.backend);
        assertEquals(SdxRuntime.SDX_GPU_TARGET_AUTO, openVinoRun.gpu_target);
    }

    private static Tokenizer nativeSentinelTokenizer(
            Map<String, Integer> addedTokens, Set<Integer> specialTokenIds) {
        return (Tokenizer) Proxy.newProxyInstance(
                Tokenizer.class.getClassLoader(),
                new Class<?>[]{Tokenizer.class},
                (proxy, method, args) -> {
                    if ("getTokenId".equals(method.getName()) || "getToken".equals(method.getName())) {
                        return null;
                    }
                    if ("getAddedTokens".equals(method.getName())) {
                        return addedTokens;
                    }
                    if ("getSpecialTokenIds".equals(method.getName())) {
                        return specialTokenIds;
                    }
                    if ("decode".equals(method.getName())) {
                        int tokenId = ((int[]) args[0])[0];
                        return tokenId == 10
                                ? ChatTemplate.NATIVE_TOOL_CALL_START
                                : ChatTemplate.NATIVE_TOOL_CALL_END;
                    }
                    throw new UnsupportedOperationException(method.getName());
                });
    }
}
