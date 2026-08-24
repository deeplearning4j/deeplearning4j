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
import org.eclipse.deeplearning4j.llm.tokenizer.ChatTemplate;
import org.eclipse.deeplearning4j.llm.tokenizer.HuggingFaceTokenizer;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.autodiff.samediff.serde.SDZSerializer;
import org.nd4j.ggml.GGMLModelExport;
import org.nd4j.ggml.GGMLModelImport;
import org.nd4j.ggml.convert.ConversionOptions;
import org.nd4j.ggml.export.ExportOptions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.custom.GatedDeltaRule;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
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
    private static final String TOKENIZER_PROPERTY = "sdx.qwen.tokenizer";
    private static final String REBUILD_BUNDLE_PROPERTY = "sdx.qwen.rebuildBundle";
    private static final String ANDROID_SMOKE_PROMPT_PROPERTY = "sdx.qwen.androidSmokePrompt";
    private static final String ANDROID_Q4_PROFILE_PROPERTY = "sdx.qwen.androidQ4Profile";
    private static final String ANDROID_SMOKE_PROMPT = "Reply with the single word: ready";
    private static final long[] ANDROID_SMOKE_PROMPT_IDS = {
            248045, 846, 198, 20206, 440, 279, 3074, 3299, 25, 5354,
            248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271
    };
    private static final long[] GRAPH_ASSISTANT_PROMPT_IDS = {
            248045, 8678, 198, 2523, 513, 264, 4618, 17313, 13, 5272, 279, 2420,
            4618, 7141, 948, 279, 4087, 13402, 383, 1208, 11, 10781, 11, 14318,
            11, 466, 11532, 303, 279, 4618, 13, 9437, 955, 9468, 14318, 1518,
            12945, 29256, 11, 635, 524, 16525, 7167, 12688, 11, 321, 2873, 264,
            61446, 4087, 42873, 303, 279, 5743, 4618, 795, 13, 248046, 198,
            248045, 846, 198, 9419, 248046, 198, 248045, 74455, 198, 248068, 271,
            248069, 271
    };

    @Test
    void exactQwenBundleGeneratesPrefillAndWarmDecodeTokensWithoutFallback()
            throws Exception {
        if (Boolean.getBoolean("nd4j.dsp.native.dumpOutputs")) {
            String nativeDumpEnvironment = System.getenv("ND4J_DSP_NATIVE_DUMP_OUTPUTS");
            assertTrue("1".equals(nativeDumpEnvironment)
                            || Boolean.parseBoolean(nativeDumpEnvironment),
                    "The native op-sanity system property must reach the native runtime environment");
            // Load the selected ND4J backend before the backend-neutral SDX JNI
            // transport. With pathsFirst this makes incremental native tests use
            // the freshly rebuilt library rather than a same-version JavaCPP cache.
            Nd4j.getEnvironment();
        }
        String ggufValue = System.getProperty(MODEL_PROPERTY);
        Path gguf = ggufValue == null || ggufValue.trim().isEmpty()
                ? null
                : Paths.get(ggufValue).toAbsolutePath().normalize();

        boolean androidQ4Profile = Boolean.getBoolean(ANDROID_Q4_PROFILE_PROPERTY);
        String defaultBundle = androidQ4Profile
                ? "target/sdx-desktop-gate/qwen35-0.8b-android-q4"
                : "target/sdx-desktop-gate/qwen35-0.8b";
        Path bundle = Paths.get(System.getProperty(BUNDLE_PROPERTY, defaultBundle))
                .toAbsolutePath().normalize();
        Path model = bundle.resolve("model.sdz");
        Path generationConfig = bundle.resolve("text-generation.json");
        Path manifest = bundle.resolve("manifest.json");
        Path tokenizerPath = Paths.get(System.getProperty(
                TOKENIZER_PROPERTY,
                Paths.get(System.getProperty("user.home"), ".cache", "dl4j-llm-models",
                        "Qwen3.5-0.8B-serving", "tokenizer.json").toString()))
                .toAbsolutePath().normalize();
        Path tokenizerConfig = tokenizerPath.resolveSibling("tokenizer_config.json");
        assertTrue(Files.isRegularFile(tokenizerPath),
                "Canonical Hugging Face tokenizer does not exist: " + tokenizerPath);
        assertTrue(Files.isRegularFile(tokenizerConfig),
                "Canonical Hugging Face tokenizer_config.json does not exist: " + tokenizerConfig);

        boolean rebuild = Boolean.getBoolean(REBUILD_BUNDLE_PROPERTY)
                || !Files.isRegularFile(model)
                || !Files.isRegularFile(generationConfig)
                || !Files.isRegularFile(manifest);
        if (rebuild) {
            assertTrue(gguf != null,
                    "Set -D" + MODEL_PROPERTY
                            + " when the canonical desktop-gate bundle must be built");
            assertTrue(Files.isRegularFile(gguf), "GGUF does not exist: " + gguf);
            exportCanonicalBundle(
                    gguf, bundle, model, generationConfig, manifest, androidQ4Profile, rebuild);
            if (Boolean.getBoolean("sdx.qwen.exportOnly")) {
                System.out.printf("SDX_DESKTOP_EXPORT_ONLY bundle=%s model_bytes=%d%n",
                        bundle, Files.size(model));
                return;
            }
        }

        String modelSource = gguf != null
                ? gguf.getFileName().toString()
                : bundle.getFileName().toString() + "/model.sdz";
        try (SdxRuntime runtime = SdxRuntime.create();
             SdxRuntime.SdxModel loaded = runtime.loadModel(
                     bundle.toString(), desktopStrictCpuOptions());
             SdxTextSession session = loaded.createTextSession();
             HuggingFaceTokenizer tokenizer =
                     HuggingFaceTokenizer.fromFile(tokenizerPath.toFile())) {
            boolean androidSmokePrompt = Boolean.getBoolean(ANDROID_SMOKE_PROMPT_PROPERTY);
            List<ChatTemplate.Message> messages = androidSmokePrompt
                    ? Collections.singletonList(ChatTemplate.Message.user(ANDROID_SMOKE_PROMPT))
                    : Arrays.asList(
                            ChatTemplate.Message.system(
                                    "You are a graph assistant. Use the available graph tools when the answer "
                                            + "depends on people, organizations, entities, or relationships in the graph. "
                                            + "Inspect relevant entities before drawing conclusions, do not invent missing "
                                            + "facts, and give a concise answer grounded in the returned graph data."),
                            ChatTemplate.Message.user("Hello"));
            String prompt = tokenizer.applyChatTemplate(messages, true);
            long[] promptTokenIds = tokenizer.encodeLong(prompt, false);
            SdxTextSession.GenerationResult result = session.generate(
                    promptTokenIds,
                    new SdxTextSession.GenerationOptions(16)
                            .minNewTokens(1)
                            .temperature(0.0)
                            .topK(0)
                            .topP(1.0),
                    null,
                    null);
            SdxTextSession.GenerationReport report = result.report();
            String decoded = tokenizer.decode(result.tokenIds(), true);
            StringBuilder streamed = new StringBuilder();
            try (HuggingFaceTokenizer.DecodeStream decoder =
                         tokenizer.newDecodeStream(true)) {
                for (long tokenId : result.tokenIds()) streamed.append(decoder.step(tokenId));
            }

            System.out.printf(
                    "SDX_DESKTOP_GATE model=%s model_bytes=%d prompt_tokens=%d "
                            + "generated_tokens=%d backend_report=%s requested_backend=%d "
                            + "applied_backend=%d backend_status=%d used_fallback=%d "
                            + "plan_phase=%d execution_count=%d token_ids=%s decoded=%s%n",
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
                    Arrays.toString(result.tokenIds()),
                    decoded.replace("\n", "\\n"));

            assertTrue(result.tokenCount() > 0 && result.tokenCount() <= 16,
                    "The desktop gate must generate at least one bounded decode token");
            if (androidSmokePrompt) {
                assertArrayEquals(ANDROID_SMOKE_PROMPT_IDS, promptTokenIds,
                        "The desktop gate must render the exact Android smoke-test prompt tokens");
                assertArrayEquals(new long[] {2232, 248046}, result.tokenIds(),
                        "The Android smoke prompt must retain the known-good greedy token sequence");
                assertEquals("ready", decoded,
                        "The Android smoke prompt must retain its exact coherent desktop result");
            } else if (androidQ4Profile) {
                assertArrayEquals(
                        new long[] {9419, 0, 2500, 628, 353, 7543, 488, 440, 678, 4618, 3134, 30, 248046},
                        result.tokenIds(),
                        "The Android Q4 profile must retain its known-good greedy token sequence");
                assertEquals("Hello! How can I assist you with your graph query?", decoded,
                        "The Android Q4 profile must retain its coherent desktop result");
            } else {
                assertArrayEquals(
                        new long[] {9419, 0, 2500, 628, 353, 7543, 488, 440, 678, 3134, 3242, 30, 248046},
                        result.tokenIds(),
                        "The graph-assistant prompt must retain the known-good Qwen logits and greedy token sequence");
                assertEquals("Hello! How can I assist you with your query today?", decoded,
                        "The desktop acceptance gate must reject semantically invalid but decodable tokens");
            }
            assertFalse(decoded.isBlank(), "The exact Qwen tokenizer must decode generated text");
            assertEquals(decoded, streamed.toString(),
                    "Streaming and final JavaCPP BytePointer decode must agree byte-for-byte");
            assertFalse(decoded.contains("Ġ") || decoded.contains("Ċ")
                            || decoded.contains("Ã©") || decoded.contains("å»"),
                    "Raw ByteLevel vocabulary symbols escaped the Hugging Face decoder: " + decoded);
            assertTrue(report.backendReportAvailable(),
                    "A requested route is not execution proof; the native report is required");
            assertEquals(SdxRuntime.SDX_BACKEND_SLOT_BY_SLOT, report.requestedBackend());
            assertEquals(SdxRuntime.SDX_BACKEND_SLOT_BY_SLOT, report.appliedBackend());
            assertEquals(SdxRuntime.SDX_STATUS_OK, report.backendStatusCode());
            assertEquals(0, report.usedFallback(),
                    "Slot-by-slot/host fallback is forbidden by this gate");
            assertTrue(report.executionCount() >= 1,
                    "The applied-backend report must come from an executed context");
        }
    }

    @Test
    void firstQwenGdrActualLengthSelectsTheSequentialContract() throws Exception {
        Path bundle = Paths.get(System.getProperty(
                BUNDLE_PROPERTY,
                "target/sdx-desktop-gate/qwen35-0.8b-android-q4-fixed"))
                .toAbsolutePath().normalize();
        assertTrue(Files.isRegularFile(bundle.resolve("model.sdz")),
                "Corrected Android Q4 bundle is required: " + bundle);

        String[] intermediateNames;
        try (SameDiff graph = SDZSerializer.load(bundle.resolve("model.sdz").toFile(), false)) {
            SameDiffOp firstGdr = graph.getOps().values().stream()
                    .filter(candidate -> candidate.getOp() != null &&
                            "gated_delta_rule".equals(candidate.getOp().opName()) &&
                            candidate.getOutputsOfOp().contains("gdn_state_out_0"))
                    .findFirst().orElse(null);
            assertNotNull(firstGdr, "Serialized graph must contain the first GDR op");
            assertEquals(7, firstGdr.getInputsToOp().size(),
                    "First GDR must retain Q/K/V/beta/gate/state/actualLen inputs");
            intermediateNames = firstGdr.getInputsToOp().subList(0, 5).toArray(new String[0]);
            System.out.println("QWEN_GDR_INPUT_NAMES " + Arrays.toString(intermediateNames));
        }
        try (SdxRuntime runtime = SdxRuntime.create();
             SdxRuntime.SdxModel loaded = runtime.loadModel(
                     bundle.toString(), desktopStrictCpuOptions());
             SdxRuntime.SdxContext context = loaded.createInferenceContext(intermediateNames)) {
            Map<String, INDArray> ownedInputs = new HashMap<>();
            ownedInputs.put("input_ids", Nd4j.createFromArray(GRAPH_ASSISTANT_PROMPT_IDS)
                    .reshape(1, GRAPH_ASSISTANT_PROMPT_IDS.length));
            ownedInputs.put("actual_sequence_length",
                    Nd4j.scalar(DataType.INT64, GRAPH_ASSISTANT_PROMPT_IDS.length));
            ownedInputs.put("position_offset", Nd4j.scalar(DataType.INT64, 0));
            ownedInputs.put("cache_position", Nd4j.scalar(DataType.INT64, 0));
            ownedInputs.put("past_conv_state.0", Nd4j.zeros(DataType.FLOAT, 1, 6144, 3));
            ownedInputs.put("past_gdn_state.0", Nd4j.zeros(DataType.FLOAT, 1, 16, 128, 128));

            Object[] inputs = new Object[context.numInputs()];
            for (int index = 0; index < inputs.length; index++) {
                String name = context.inputName(index);
                INDArray array = ownedInputs.get(name);
                assertTrue(array != null, "Unexpected first-GDR context input: " + name);
                inputs[index] = array;
            }

            INDArray q = Nd4j.create(DataType.FLOAT, 1, 72, 16, 128);
            INDArray k = Nd4j.create(DataType.FLOAT, 1, 72, 16, 128);
            INDArray v = Nd4j.create(DataType.FLOAT, 1, 72, 16, 128);
            INDArray beta = Nd4j.create(DataType.FLOAT, 1, 72, 16);
            INDArray gate = Nd4j.create(DataType.FLOAT, 1, 72, 16);
            context.runNd4j(inputs, new Object[]{q, k, v, beta, gate});

            String vectorExport = System.getProperty("sdx.qwen.gdrVectorExport");
            if (vectorExport != null && !vectorExport.isBlank()) {
                int dKey = (int) q.size(3);
                int dValue = (int) v.size(3);
                ByteBuffer vector = ByteBuffer
                        .allocate((dKey * 2 + dValue + 2) * Float.BYTES)
                        .order(ByteOrder.LITTLE_ENDIAN);
                for (int d = 0; d < dKey; d++) vector.putFloat(q.getFloat(0, 0, 0, d));
                for (int d = 0; d < dKey; d++) vector.putFloat(k.getFloat(0, 0, 0, d));
                for (int d = 0; d < dValue; d++) vector.putFloat(v.getFloat(0, 0, 0, d));
                vector.putFloat(beta.getFloat(0, 0, 0));
                vector.putFloat(gate.getFloat(0, 0, 0));
                Files.write(Paths.get(vectorExport), vector.array());
                System.out.println("QWEN_GDR_VECTOR_EXPORT " + vectorExport);
            }

            INDArray zeroState = ownedInputs.get("past_gdn_state.0");
            INDArray actualLength = ownedInputs.get("actual_sequence_length");
            INDArray[] sequential = Nd4j.exec(new GatedDeltaRule(
                    q, k, v, beta, gate, zeroState, actualLength));
            INDArray oneLength = Nd4j.scalar(DataType.INT64, 1L);
            INDArray[] oneToken = Nd4j.exec(new GatedDeltaRule(
                    q, k, v, beta, gate, zeroState, oneLength));
            INDArray[] chunked = Nd4j.exec(new GatedDeltaRule(
                    q, k, v, beta, gate, zeroState));

            long sequentialOutputHash = opSanityHash(sequential[0]);
            long sequentialStateHash = opSanityHash(sequential[1]);
            long chunkedOutputHash = opSanityHash(chunked[0]);
            long chunkedStateHash = opSanityHash(chunked[1]);
            System.out.printf(
                    "QWEN_GDR_PATH sequential_out=0x%016x sequential_state=0x%016x " +
                            "one_token_out=0x%016x one_token_state=0x%016x " +
                            "chunked_out=0x%016x chunked_state=0x%016x%n",
                    sequentialOutputHash, sequentialStateHash,
                    opSanityHash(oneToken[0]), opSanityHash(oneToken[1]),
                    chunkedOutputHash, chunkedStateHash);
            System.out.printf(
                    "QWEN_GDR_ONE_TOKEN output_first=%.9g output_min=%.9g output_max=%.9g " +
                            "state_first=%.9g state_min=%.9g state_max=%.9g%n",
                    oneToken[0].getDouble(0),
                    oneToken[0].minNumber().doubleValue(), oneToken[0].maxNumber().doubleValue(),
                    oneToken[1].getDouble(0),
                    oneToken[1].minNumber().doubleValue(), oneToken[1].maxNumber().doubleValue());
            System.out.printf(
                    "QWEN_GDR_SEQUENTIAL output_first=%.9g state_first=%.9g " +
                            "state_min=%.9g state_max=%.9g%n",
                    sequential[0].getDouble(0), sequential[1].getDouble(0),
                    sequential[1].minNumber().doubleValue(), sequential[1].maxNumber().doubleValue());
            System.out.printf(
                    "QWEN_GDR_CHUNKED output_first=%.9g output_min=%.9g output_max=%.9g " +
                            "state_first=%.9g state_min=%.9g state_max=%.9g%n",
                    chunked[0].getDouble(0),
                    chunked[0].minNumber().doubleValue(), chunked[0].maxNumber().doubleValue(),
                    chunked[1].getDouble(0),
                    chunked[1].minNumber().doubleValue(), chunked[1].maxNumber().doubleValue());

            assertEquals(0x2114163817fe43eaL, sequentialOutputHash,
                    "actualLen=72 must retain the known-good sequential output");
            assertEquals(0x9fbf870820c29e4eL, sequentialStateHash,
                    "actualLen=72 must retain the known-good sequential state");

            for (INDArray array : ownedInputs.values()) {
                if (!array.wasClosed()) array.close();
            }
            for (INDArray array : new INDArray[]{q, k, v, beta, gate, oneLength,
                    sequential[0], sequential[1], oneToken[0], oneToken[1],
                    chunked[0], chunked[1]}) {
                if (!array.wasClosed()) array.close();
            }
        }
    }

    private static long opSanityHash(INDArray array) {
        long hash = 0xcbf29ce484222325L;
        for (long index = 0; index < array.length(); index++) {
            long bits = Double.doubleToRawLongBits(array.getDouble(index));
            for (int shift = 0; shift < Long.SIZE; shift += Byte.SIZE) {
                hash ^= (bits >>> shift) & 0xffL;
                hash *= 0x100000001b3L;
            }
        }
        return hash;
    }

    private static SdxRuntime.ModelOptions desktopStrictCpuOptions() {
        return new SdxRuntime.ModelOptions()
                .backend(SdxRuntime.SDX_BACKEND_SLOT_BY_SLOT)
                .strictBackend(true)
                .allowRuntimeJit(true);
    }

    private static void exportCanonicalBundle(
            Path gguf,
            Path bundle,
            Path model,
            Path generationConfig,
            Path manifest,
            boolean androidQ4Profile,
            boolean rebuild) throws Exception {
        Files.createDirectories(bundle);
        SameDiff graph;
        if (androidQ4Profile) {
            Path optimizedQ4 = bundle.resolve("optimized-q4_k.gguf");
            if (rebuild) Files.deleteIfExists(optimizedQ4);
            if (!Files.isRegularFile(optimizedQ4)) {
                GGMLModelExport.requantize(
                        gguf.toFile(), optimizedQ4.toFile(), ExportOptions.QuantizationType.Q4_K);
            }
            graph = GGMLModelImport.importModel(optimizedQ4.toFile(), ConversionOptions.builder()
                    .quantizationMode(ConversionOptions.QuantizationMode.RUNTIME_QUANTIZED_MATMUL)
                    .targetDataType(DataType.FLOAT)
                    .embeddingDataType(DataType.HALF)
                    .lastPositionLogitsOnly(true)
                    .forTraining(false)
                    .preserveTokenizerInfo(true)
                    .kvQuantFormat(1)
                    .tensorBatchSize(4)
                    .useMemoryMapping(true)
                    .build());
        } else if (Boolean.getBoolean("sdx.qwen.mobileProfile")) {
            graph = GGMLModelImport.importModel(gguf.toFile(), ConversionOptions.builder()
                    .quantizationMode(ConversionOptions.QuantizationMode.DEQUANTIZE_TO_FLOAT16)
                    .targetDataType(DataType.FLOAT16)
                    .embeddingDataType(DataType.HALF)
                    .lastPositionLogitsOnly(true)
                    .forTraining(false)
                    .preserveTokenizerInfo(true)
                    .kvQuantFormat(0)
                    .tensorBatchSize(10)
                    .useMemoryMapping(true)
                    .build());
        } else {
            graph = GGMLModelImport.importModel(gguf.toFile());
        }
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

}
