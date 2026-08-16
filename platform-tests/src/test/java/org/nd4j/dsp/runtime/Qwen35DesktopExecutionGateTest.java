/*
 * ******************************************************************************
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * SPDX-License-Identifier: Apache-2.0
 * *****************************************************************************
 */
package org.nd4j.dsp.runtime;

import org.eclipse.deeplearning4j.llm.generation.SdxTextGenerationConfig;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.serde.SDZSerializer;
import org.nd4j.ggml.GGMLModelImport;

import java.io.File;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Collections;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Explicit desktop admission gate for the exact Qwen3.5 model used by the
 * Android smoke test. Generic graph/runtime failures must be fixed here before
 * an APK is built or a device accelerator is investigated.
 *
 * <p>The generated canonical bundle is cached under {@code target}. Set
 * {@code -Dsdx.qwen.gguf=/absolute/path/Qwen3.5-0.8B-BF16.gguf} when the
 * bundle must be created, and pass {@code -Dsdx.qwen.rebuildBundle=true} to
 * force re-import and re-export.</p>
 */
class Qwen35DesktopExecutionGateTest {

    private static final String MODEL_PROPERTY = "sdx.qwen.gguf";
    private static final String BUNDLE_PROPERTY = "sdx.qwen.bundle";
    private static final String PROMPT_PROPERTY = "sdx.qwen.promptTokenIds";

    @Test
    void exactQwenBundleGeneratesPrefillAndWarmDecodeTokensWithoutFallback()
            throws Exception {
        String ggufValue = System.getProperty(MODEL_PROPERTY);
        Path gguf = ggufValue == null || ggufValue.trim().isEmpty()
                ? null
                : Paths.get(ggufValue).toAbsolutePath().normalize();

        Path bundle = Paths.get(System.getProperty(
                BUNDLE_PROPERTY, "target/sdx-desktop-gate/qwen35-0.8b"))
                .toAbsolutePath().normalize();
        Path model = bundle.resolve("model.sdz");
        Path generationConfig = bundle.resolve("text-generation.json");
        Path manifest = bundle.resolve("manifest.json");

        boolean rebuild = Boolean.getBoolean("sdx.qwen.rebuildBundle")
                || !Files.isRegularFile(model)
                || !Files.isRegularFile(generationConfig)
                || !Files.isRegularFile(manifest);
        if (rebuild) {
            assertTrue(gguf != null,
                    "Set -D" + MODEL_PROPERTY
                            + " when the canonical desktop-gate bundle must be built");
            assertTrue(Files.isRegularFile(gguf), "GGUF does not exist: " + gguf);
            exportCanonicalBundle(gguf, bundle, model, generationConfig, manifest);
        }

        String modelSource = gguf != null
                ? gguf.getFileName().toString()
                : bundle.getFileName().toString() + "/model.sdz";
        long[] promptTokenIds = promptTokenIds();
        try (SdxRuntime runtime = SdxRuntime.create();
             SdxRuntime.SdxModel loaded = runtime.loadModel(
                     bundle.toString(), SdxRuntime.ModelOptions.desktopOpenVino());
             SdxTextSession session = loaded.createTextSession()) {
            SdxTextSession.GenerationResult result = session.generate(
                    promptTokenIds,
                    new SdxTextSession.GenerationOptions(2)
                            .minNewTokens(2)
                            .temperature(0.0)
                            .topK(0)
                            .topP(1.0),
                    null,
                    null);
            SdxTextSession.GenerationReport report = result.report();

            System.out.printf(
                    "SDX_DESKTOP_GATE model=%s model_bytes=%d prompt_tokens=%d "
                            + "generated_tokens=%d backend_report=%s requested_backend=%d "
                            + "applied_backend=%d backend_status=%d used_fallback=%d "
                            + "plan_phase=%d execution_count=%d token_ids=%s%n",
                    modelSource,
                    Files.size(model),
                    promptTokenIds.length,
                    result.tokenCount(),
                    report.backendReportAvailable(),
                    report.requestedBackend(),
                    report.appliedBackend(),
                    report.backendStatusCode(),
                    report.usedFallback(),
                    report.planPhase(),
                    report.executionCount(),
                    Arrays.toString(result.tokenIds()));

            assertEquals(2, result.tokenCount(),
                    "The desktop gate must cross prefill into the first decode warmup step");
            assertTrue(report.backendReportAvailable(),
                    "A requested route is not execution proof; the native report is required");
            assertEquals(SdxRuntime.SDX_BACKEND_OPENVINO, report.requestedBackend());
            assertEquals(SdxRuntime.SDX_BACKEND_OPENVINO, report.appliedBackend());
            assertEquals(SdxRuntime.SDX_STATUS_OK, report.backendStatusCode());
            assertEquals(0, report.usedFallback(),
                    "Slot-by-slot/host fallback is forbidden by this gate");
            assertTrue(report.executionCount() >= 1,
                    "The applied-backend report must come from an executed context");
        }
    }

    private static void exportCanonicalBundle(
            Path gguf,
            Path bundle,
            Path model,
            Path generationConfig,
            Path manifest) throws Exception {
        Files.createDirectories(bundle);
        SameDiff graph = GGMLModelImport.importModel(gguf.toFile());
        assertTrue(graph.hasVariable("model.embed_tokens.weight"),
                "GGUF import omitted model.embed_tokens.weight");
        assertTrue(graph.getVariable("model.embed_tokens.weight").getArr() != null,
                "GGUF import left model.embed_tokens.weight unbound before SDZ export");
        assertFalse(graph.hasVariable("lm_head.weight"),
                "Qwen3.5 ties its output head to the embedding; duplicating it adds about 970 MiB");
        System.out.printf(
                "SDX_DESKTOP_EXPORT graph_ops=%d graph_variables=%d embedding_elements=%d%n",
                graph.getOps().size(),
                graph.getVariables().size(),
                graph.getVariable("model.embed_tokens.weight").getArr().length());

        SdxTextGenerationConfig.Options options =
                SdxTextGenerationConfig.Options.builder()
                        .contextLength(512)
                        .maxPrefillLength(128)
                        .padId(248044)
                        .eosIds(Collections.singletonList(248046))
                        .maxNewTokens(2)
                        .minNewTokens(2)
                        .temperature(0.0)
                        .topK(0)
                        .topP(1.0)
                        .build();

        SdxTextGenerationConfig.write(graph, options, generationConfig);
        SDZSerializer.save(
                graph,
                model.toFile(),
                false,
                Collections.singletonMap("sdx.desktopGate", "qwen35-0.8b-bf16"));

        SameDiff roundTrip = SameDiff.load(model.toFile(), false);
        assertTrue(roundTrip.hasVariable("model.embed_tokens.weight"),
                "SDZ round trip omitted model.embed_tokens.weight");
        assertTrue(roundTrip.getVariable("model.embed_tokens.weight").getArr() != null,
                "SDZ round trip lost model.embed_tokens.weight data");
        assertFalse(roundTrip.hasVariable("lm_head.weight"),
                "SDZ round trip must retain the tied output-head alias without a duplicate table");
        System.out.printf(
                "SDX_DESKTOP_EXPORT_ROUND_TRIP graph_ops=%d graph_variables=%d "
                        + "embedding_elements=%d%n",
                roundTrip.getOps().size(),
                roundTrip.getVariables().size(),
                roundTrip.getVariable("model.embed_tokens.weight").getArr().length());

        String manifestJson = "{\n"
                + "  \"formatVersion\": 1,\n"
                + "  \"modelId\": \"Qwen3.5-0.8B-BF16\",\n"
                + "  \"producer\": {\"tool\": \"Qwen35DesktopExecutionGateTest\", "
                + "\"version\": \"1\"},\n"
                + "  \"modelPath\": \"model.sdz\",\n"
                + "  \"targets\": [\"desktop-x86_64-openvino\"],\n"
                + "  \"preferredBackends\": [\"OPENVINO\"],\n"
                + "  \"gpuTarget\": \"AUTO\",\n"
                + "  \"configPath\": \"text-generation.json\",\n"
                + "  \"textGeneration\": {\"configPath\": \"text-generation.json\"},\n"
                + "  \"compatibility\": {\"minRuntimeAbi\": 1, \"maxRuntimeAbi\": 1}\n"
                + "}\n";
        Files.write(manifest, manifestJson.getBytes(StandardCharsets.UTF_8));
    }

    private static long[] promptTokenIds() {
        String configured = System.getProperty(PROMPT_PROPERTY, "40,374,264,1273,13");
        String[] values = configured.split(",");
        long[] result = new long[values.length];
        for (int i = 0; i < values.length; i++) {
            result[i] = Long.parseLong(values[i].trim());
            if (result[i] < 0) {
                throw new IllegalArgumentException(
                        PROMPT_PROPERTY + " contains a negative token ID");
            }
        }
        return result;
    }
}
