/*
 * SPDX-License-Identifier: Apache-2.0
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0: https://www.apache.org/licenses/LICENSE-2.0.
 * See the NOTICE file distributed with this work for copyright ownership.
 */
package org.eclipse.deeplearning4j.llm;

import com.google.gson.Gson;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import lombok.extern.slf4j.Slf4j;
import org.bytedeco.javacpp.BytePointer;
import org.eclipse.deeplearning4j.llm.data.LLMModelDownloader;
import org.eclipse.deeplearning4j.llm.generation.ChatGenerationResult;
import org.eclipse.deeplearning4j.llm.generation.GenerationPipeline;
import org.eclipse.deeplearning4j.llm.generation.GenerationPipeline.GenerationSession;
import org.eclipse.deeplearning4j.llm.generation.GenerationPipeline.ModelMetadata;
import org.eclipse.deeplearning4j.llm.generation.kvcache.KvCacheStrategy;
import org.eclipse.deeplearning4j.safetensors.ModelOptQwenConfig;
import org.eclipse.deeplearning4j.llm.generation.GenerationPipelineConfig;
import org.eclipse.deeplearning4j.llm.generation.GenerationResult;
import org.eclipse.deeplearning4j.llm.generation.sampling.SamplingConfig;
import org.eclipse.deeplearning4j.llm.tokenizer.ChatTemplate;
import org.eclipse.deeplearning4j.llm.tokenizer.HuggingFaceTokenizer;
import org.eclipse.deeplearning4j.safetensors.ModelOptQwenImporter;
import org.eclipse.deeplearning4j.safetensors.ModelOptQwenImporter.ImportedModel;
import org.eclipse.deeplearning4j.safetensors.SafeTensorsHeader;
import org.eclipse.deeplearning4j.safetensors.SafeTensorsHeader.TensorInfo;
import org.eclipse.deeplearning4j.safetensors.SafeTensorsReader;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.io.Reader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Pinned real-checkpoint tests, independently enabled with -Dqwen.nvfp4.import=true
 * (sequential bit-exact storage import) or -Dqwen.nvfp4.generate=true (whole text
 * model execution), or -Dqwen.nvfp4.mtp=true (native bundled predictor versus greedy).
 * Downloads approximately 22 GB, including a 2.54 GB embedding.
 * Generation retains packed weights and BF16 activations/KV plus FP32 recurrent
 * state; allow sufficient disk and host/device memory. Short generation checks
 * are correctness tests, not throughput benchmarks.
 */
@Slf4j
public class TestQwenNvfp4Import {
    private static final String REPO = "nvidia/Qwen3.6-27B-NVFP4";
    private static final String REVISION = "0893e1606ff3d5f97a441f405d5fc541a6bdf404";
    private static final String BASE = "https://huggingface.co/" + REPO + "/resolve/" + REVISION + "/";
    // Names verified in the pinned remote weight_map, not inferred from GGUF layouts.
    private static final String EMBEDDING = "model.language_model.embed_tokens.weight";
    private static final String FP8 = "model.language_model.layers.0.linear_attn.in_proj_z";
    private static final String NVFP4 = "model.language_model.layers.0.mlp.down_proj";
    private static final Gson GSON = new Gson();

    @Test
    @EnabledIfSystemProperty(named = "qwen.nvfp4.import", matches = "true")
    void importActualCheckpointStorage() throws Exception {
        JsonObject config = json(download("config.json"));
        JsonObject quant = json(download("hf_quant_config.json")).getAsJsonObject("quantization");
        JsonObject index = json(download("model.safetensors.index.json"));
        JsonObject weights = index.getAsJsonObject("weight_map");
        assertEquals("qwen3_5", config.get("model_type").getAsString());
        assertEquals("MIXED_PRECISION", quant.get("quant_algo").getAsString());
        JsonObject layers = quant.getAsJsonObject("quantized_layers");
        JsonObject configLayers = config.getAsJsonObject("quantization_config")
                .getAsJsonObject("quantized_layers");
        assertEquals(layers, configLayers, "The two quantization manifests must agree");
        assertEquals("FP8", layers.getAsJsonObject(FP8).get("quant_algo").getAsString());
        assertEquals("W4A16_NVFP4", layers.getAsJsonObject(NVFP4).get("quant_algo").getAsString());
        int groupSize = layers.getAsJsonObject(NVFP4).get("group_size").getAsInt();
        assertEquals(16, groupSize);
        for (Map.Entry<String, JsonElement> layer : layers.entrySet()) {
            JsonObject description = layer.getValue().getAsJsonObject();
            String algorithm = description.get("quant_algo").getAsString();
            assertTrue(algorithm.equals("FP8") || algorithm.equals("W4A16_NVFP4"), layer.getKey());
            if (algorithm.equals("W4A16_NVFP4")) {
                assertEquals(16, description.get("group_size").getAsInt(), layer.getKey());
            }
            assertTrue(weights.has(layer.getKey() + ".weight"), layer.getKey());
        }

        Set<String> shardNames = new TreeSet<>();
        for (Map.Entry<String, JsonElement> entry : weights.entrySet()) {
            shardNames.add(entry.getValue().getAsString());
        }
        assertEquals(3, shardNames.size(), "Pinned checkpoint shard count");
        Map<String, File> shards = new LinkedHashMap<>();
        Map<String, SafeTensorsHeader> headers = new LinkedHashMap<>();
        long totalBytes = 0;
        for (String shard : shardNames) {
            assertTrue(shard.matches("model-[0-9]{5}-of-00003\\.safetensors"), shard);
            File file = download(shard);
            SafeTensorsHeader header = SafeTensorsHeader.fromFile(file);
            totalBytes = Math.addExact(totalBytes, validateHeader(file, shard, header, weights));
            shards.put(shard, file);
            headers.put(shard, header);
        }
        assertEquals(index.getAsJsonObject("metadata").get("total_size").getAsLong(), totalBytes);

        JsonObject text = config.getAsJsonObject("text_config");
        long hidden = text.get("hidden_size").getAsLong();
        long intermediate = text.get("intermediate_size").getAsLong();
        TensorInfo embedding = info(EMBEDDING, weights, headers);
        assertTrue(embedding.getDataLength() > Integer.MAX_VALUE, "Exercise the >2 GB tensor path");
        importTensor(EMBEDDING, "BF16", DataType.BFLOAT16,
                new long[]{text.get("vocab_size").getAsLong(), hidden}, false, weights, shards, headers);
        long valueWidth = Math.multiplyExact(text.get("linear_num_value_heads").getAsLong(),
                text.get("linear_value_head_dim").getAsLong());
        importTensor(FP8 + ".weight", "F8_E4M3", DataType.FLOAT8,
                new long[]{valueWidth, hidden}, false, weights, shards, headers);
        assertEquals(0, intermediate % groupSize);
        long[] logical = {hidden, intermediate};
        long[] packed = {hidden, intermediate / 2}; // two FP4 nibbles in each U8, not an ND4J FP4 array
        log.info("STORAGE NVFP4 logical={} packed={} groupSize={} (no unpack/compute)",
                Arrays.toString(logical), Arrays.toString(packed), groupSize);
        importTensor(NVFP4 + ".weight", "U8", DataType.UINT8, packed, false, weights, shards, headers);
        importScales(FP8, false, valueWidth, hidden, groupSize, weights, shards, headers);
        importScales(NVFP4, true, hidden, intermediate, groupSize, weights, shards, headers);
        log.info("STORAGE import complete: {}@{}, {} shards, {} payload bytes; no generation/compute tested",
                REPO, REVISION, shards.size(), totalBytes);
    }

    @Test
    @EnabledIfSystemProperty(named = "qwen.nvfp4.generate", matches = "true")
    void generateWithActualPackedCheckpoint() throws Exception {
        generateWithActualPackedCheckpoint(false);
    }

    @Test
    @EnabledIfSystemProperty(named = "qwen.nvfp4.mtp", matches = "true")
    void bundledMtpMatchesGreedyWithActualPackedCheckpoint() throws Exception {
        generateWithActualPackedCheckpoint(true);
    }

    /**
     * Steady-state decode benchmark: builds the plan once per pass via
     * {@link GenerationPipeline#startSession(String, int)} (MTP session, then greedy
     * session) and times only the token loop, so plan-build/Triton-warmup cost is
     * excluded. Reports decode tok/s and MTP acceptance per pass.
     * Enabled with -Dqwen.nvfp4.mtpBench=true.
     */
    @Test
    @EnabledIfSystemProperty(named = "qwen.nvfp4.mtpBench", matches = "true")
    void steadyStateMtpDecodeThroughput() throws Exception {
        int maxTokens = Integer.getInteger("qwen.nvfp4.benchTokens", 250);
        int maxPrefill = Integer.getInteger("qwen.nvfp4.maxPrefillLength", 128);
        int contextCap = Integer.getInteger("qwen.nvfp4.maxKvCacheLength", 448);
        String prompt = "Write a short story about a robot who learns to paint.";

        File configFile = download("config.json");
        File quantFile = download("hf_quant_config.json");
        File generationFile = download("generation_config.json");
        JsonObject index = json(download("model.safetensors.index.json"));
        Set<String> shardNames = new TreeSet<>();
        for (Map.Entry<String, JsonElement> entry : index.getAsJsonObject("weight_map").entrySet()) {
            shardNames.add(entry.getValue().getAsString());
        }
        List<File> shards = new ArrayList<>();
        for (String shard : shardNames) shards.add(download(shard));
        String tokenizerJson = Files.readString(download("tokenizer.json").toPath(), StandardCharsets.UTF_8);
        JsonObject tokenizerConfig = json(download("tokenizer_config.json"));
        String template = Files.readString(download("chat_template.jinja").toPath(), StandardCharsets.UTF_8);
        tokenizerConfig.addProperty("chat_template", template);

        try (HuggingFaceTokenizer tokenizer = HuggingFaceTokenizer.fromJson(tokenizerJson, GSON.toJson(tokenizerConfig));
             ImportedModel model = ModelOptQwenImporter.importTextOnly(
                     configFile, quantFile, generationFile, shards, DataType.BFLOAT16, true)) {
            ModelOptQwenConfig importedConfig = model.getConfig();
            GenerationPipelineConfig pipelineConfig = GenerationPipelineConfig.builder()
                    .decoder(model.getGraph()).tokenizer(tokenizer)
                    .modelMetadata(ModelMetadata.of(importedConfig.getBosTokenId(), importedConfig.getEosTokenId(),
                            importedConfig.getPadTokenId(), template, importedConfig.getStopTokenIds(),
                            importedConfig.getStopTokenIds()))
                    .kvCacheStrategy(KvCacheStrategy.STATIC)
                    .dspEnabled(true)
                    .maxSpeculativeTokens(Integer.getInteger("qwen.nvfp4.mtpK", 4))
                    .samplingConfig(SamplingConfig.speculative())
                    .maxNewTokens(maxTokens).maxPrefillLength(maxPrefill).maxKvCacheLength(contextCap)
                    .build();
            try (GenerationPipeline pipeline = GenerationPipeline.create(pipelineConfig)) {
                if (pipeline.getDecoder() != model.getGraph()) model.close();

                // ── Pass 1: MTP steady-state decode ──
                pipeline.setSamplingConfig(SamplingConfig.speculative());
                long mtpDecodeNs = 0;
                int mtpTokens = 0;
                int proposed = 0, accepted = 0, steps = 0;
                try (GenerationSession session = pipeline.startSession(prompt, maxTokens)) {
                    long t0 = System.nanoTime();
                    // First session call includes warmup decode; measure it separately
                    // so the reported rate is the steady-state replay loop.
                    GenerationResult first = session.generate(1);
                    long t1 = System.nanoTime();
                    GenerationResult rest = session.generate(maxTokens - 1);
                    long t2 = System.nanoTime();
                    mtpDecodeNs = t2 - t1;
                    mtpTokens = rest.getTokenIds().length;
                    proposed = rest.getTotalSpeculativeTokens();
                    accepted = rest.getTotalAcceptedTokens();
                    steps = rest.getSpeculativeSteps();
                    log.info("NVFP4-BENCH MTP warmupMs={} steadyTokens={} steadyMs={} tok/s={}",
                            (t1 - t0) / 1_000_000, mtpTokens, mtpDecodeNs / 1_000_000,
                            mtpTokens * 1e9 / Math.max(1, mtpDecodeNs));
                    log.info("NVFP4-BENCH MTP proposed={} accepted={} steps={} acceptance={} text={}",
                            proposed, accepted, steps,
                            proposed > 0 ? (double) accepted / proposed : 0.0,
                            rest.getText());
                }

                // ── Pass 2: greedy steady-state decode (fresh session; rebuild is setup, not measured) ──
                pipeline.setSamplingConfig(SamplingConfig.greedy());
                long greedyDecodeNs = 0;
                int greedyTokens = 0;
                try (GenerationSession session = pipeline.startSession(prompt, maxTokens)) {
                    session.generate(1);
                    long t1 = System.nanoTime();
                    GenerationResult rest = session.generate(maxTokens - 1);
                    greedyDecodeNs = System.nanoTime() - t1;
                    greedyTokens = rest.getTokenIds().length;
                    log.info("NVFP4-BENCH greedy steadyTokens={} steadyMs={} tok/s={}",
                            greedyTokens, greedyDecodeNs / 1_000_000,
                            greedyTokens * 1e9 / Math.max(1, greedyDecodeNs));
                }
                log.info("NVFP4-BENCH SUMMARY mtpTok/s={} greedyTok/s={} speedup={} acceptance={}/{}",
                        mtpTokens * 1e9 / Math.max(1, mtpDecodeNs),
                        greedyTokens * 1e9 / Math.max(1, greedyDecodeNs),
                        (mtpTokens * (double) greedyDecodeNs) / Math.max(1, mtpDecodeNs * (double) greedyTokens),
                        accepted, proposed);
            }
        }
    }

    @Test
    @EnabledIfSystemProperty(named = "qwen.nvfp4.mtpPrefix", matches = "true")
    void bundledMtpPrefixMatchesGreedy() throws Exception {
        generateWithActualPackedCheckpoint(true, true);
    }

    private void generateWithActualPackedCheckpoint(boolean mtp) throws Exception {
        generateWithActualPackedCheckpoint(mtp, false);
    }

    private void generateWithActualPackedCheckpoint(boolean mtp, boolean prefixOnly) throws Exception {
        // A diagnostic prefix is not a replacement for the complete-answer MTP test.
        int maxTokens = prefixOnly ? Integer.getInteger("qwen.nvfp4.mtpPrefixTokens", 160)
                : Integer.getInteger("qwen.nvfp4.maxTokens", 64);
        int maxPrefill = Integer.getInteger("qwen.nvfp4.maxPrefillLength", 128);
        int contextCap = Integer.getInteger("qwen.nvfp4.maxKvCacheLength", prefixOnly ? 320 : 256);
        assertTrue(maxTokens >= 3 && maxPrefill > 0, "Require a real decode budget and positive prefill size");
        assertTrue((long) maxPrefill + maxTokens <= contextCap,
                "KV capacity must hold the padded prompt and the entire requested decode budget");

        File configFile = download("config.json");
        File quantFile = download("hf_quant_config.json");
        File generationFile = download("generation_config.json");
        JsonObject config = json(configFile);
        JsonObject index = json(download("model.safetensors.index.json"));
        JsonObject weights = index.getAsJsonObject("weight_map");
        JsonObject quantized = json(quantFile).getAsJsonObject("quantization")
                .getAsJsonObject("quantized_layers");
        assertTrue(contextCap <= config.getAsJsonObject("text_config")
                .get("max_position_embeddings").getAsInt(), "Context exceeds checkpoint limit");
        Set<String> shardNames = new TreeSet<>();
        for (Map.Entry<String, JsonElement> entry : weights.entrySet()) {
            shardNames.add(entry.getValue().getAsString());
        }
        assertEquals(3, shardNames.size());
        List<File> shards = new ArrayList<>();
        Map<String, SafeTensorsHeader> headers = new LinkedHashMap<>();
        long payloadBytes = 0;
        for (String shard : shardNames) {
            assertTrue(shard.matches("model-[0-9]{5}-of-00003\\.safetensors"), shard);
            File file = download(shard);
            SafeTensorsHeader header = SafeTensorsHeader.fromFile(file);
            payloadBytes = Math.addExact(payloadBytes, validateHeader(file, shard, header, weights));
            shards.add(file);
            headers.put(shard, header);
        }
        assertEquals(index.getAsJsonObject("metadata").get("total_size").getAsLong(), payloadBytes);

        String tokenizerJson = Files.readString(download("tokenizer.json").toPath(), StandardCharsets.UTF_8);
        JsonObject tokenizerConfig = json(download("tokenizer_config.json"));
        String template = Files.readString(download("chat_template.jinja").toPath(), StandardCharsets.UTF_8);
        assertFalse(template.isBlank(), "Pinned checkpoint must provide its chat template");
        if (tokenizerConfig.has("chat_template") && !tokenizerConfig.get("chat_template").isJsonNull()) {
            assertEquals(template.trim(), tokenizerConfig.get("chat_template").getAsString().trim());
        }
        // HF stores this template as a separate resource. Preserve every other config field;
        // use the exact pinned Jinja source, never a hand-written ChatML substitute.
        tokenizerConfig.addProperty("chat_template", template);
        try (HuggingFaceTokenizer tokenizer = HuggingFaceTokenizer.fromJson(tokenizerJson, GSON.toJson(tokenizerConfig));
             ImportedModel model = ModelOptQwenImporter.importTextOnly(
                     configFile, quantFile, generationFile, shards, DataType.BFLOAT16, true)) {
            assertEquals(DataType.BFLOAT16, model.getComputeType());
            ModelOptQwenConfig importedConfig = model.getConfig();
            assertEquals(248046, tokenizer.getEosTokenId(), "Pinned tokenizer EOS changed");
            assertEquals(tokenizer.getEosTokenId(), importedConfig.getEosTokenId());
            assertEquals(Set.of(248046, 248044), importedConfig.getStopTokenIds());
            assertEquals(248044, importedConfig.getBosTokenId());
            assertEquals(248044, importedConfig.getPadTokenId());
            assertEquals(1, importedConfig.getArchitecture().getNumMtpLayers());
            for (String name : List.of("mtp_input_ids", "mtp_target_hidden_states", "mtp_position_offset",
                    "mtp_cache_position", "mtp_causal_mask", "mtp_past_key_values.0.key",
                    "mtp_past_key_values.0.value", "mtp_key_states", "mtp_value_states",
                    "mtp_hidden_states", "mtp_logits")) {
                assertNotNull(model.getGraph().getVariable(name), "Missing native predictor ABI: " + name);
            }
            assertMtpCheckpointPayload(model.getGraph(), importedConfig, weights, headers);
            assertNotNull(model.getGraph().getVariable("target_hidden_states"));
            assertNotNull(model.getGraph().getVariable("lm_logits"), "MTP needs all target verification rows");
            assertPackedExecutionGraph(model.getGraph(), quantized, weights, headers);
            assertCacheInputTypes(model.getGraph());
            assertEquals(DataType.BFLOAT16,
                    model.getGraph().getArrForVarName("model.embed_tokens.weight").dataType());

            GenerationPipelineConfig pipelineConfig = GenerationPipelineConfig.builder()
                    .decoder(model.getGraph()).tokenizer(tokenizer)
                    .modelMetadata(ModelMetadata.of(importedConfig.getBosTokenId(), importedConfig.getEosTokenId(),
                            importedConfig.getPadTokenId(), template, importedConfig.getStopTokenIds(),
                            importedConfig.getStopTokenIds()))
                    .kvCacheStrategy(KvCacheStrategy.STATIC)
                    .dspEnabled(true)
                    // K=1 isolates post-commit slot-0 state (trunk carry, single KV row)
                    // from self-carry chain depth: qwen.nvfp4.mtpK overrides for the
                    // acceptance-collapse diagnosis (2/333 at K=4 on 160 tokens).
                    .maxSpeculativeTokens(mtp ? Integer.getInteger("qwen.nvfp4.mtpK", 4) : 0)
                    .samplingConfig(mtp ? SamplingConfig.speculative() : SamplingConfig.greedy())
                    .maxNewTokens(maxTokens).maxPrefillLength(maxPrefill).maxKvCacheLength(contextCap)
                    .build();
            // Pipeline owns generation buffers/plans, not this caller-supplied graph. Close it
            // first so no native borrower remains when ImportedModel releases packed arrays.
            try (GenerationPipeline pipeline = GenerationPipeline.create(pipelineConfig)) {
                // GraphOptimizer deep-copies the decoder. Once that owned copy exists,
                // this test has no further use for the original model's allocations.
                // ImportedModel.close() is idempotent; the outer resource scope remains
                // responsible when optimization returns the original graph instead.
                if (pipeline.getDecoder() != model.getGraph()) model.close();
                // Audit the actual post-optimizer decoder too: dense substitutes must not pass.
                assertPackedExecutionGraph(pipeline.getDecoder(), quantized, weights, headers);
                assertCacheInputTypes(pipeline.getDecoder());
                String[] prompts = {
                        "What is 2 + 2? Reply briefly with only a JSON object whose key is answer and value is the integer result.",
                        "What is the capital of France? Reply briefly with only a JSON object whose key is answer and value is the city name.",
                        "Repeat the phrase blue sky. Reply briefly with only a JSON object whose key is answer and value is that exact phrase."
                };
                String[] expected = {"{\"answer\":4}", "{\"answer\":\"Paris\"}", "{\"answer\":\"blue sky\"}"};
                List<ChatGenerationResult> responses = new ArrayList<>();
                for (int promptIndex = 0; promptIndex < (prefixOnly ? 1 : prompts.length); promptIndex++) {
                    String prompt = prompts[promptIndex];
                    ChatTemplate.Request request = ChatTemplate.Request.builder()
                            .messages(List.of(ChatTemplate.Message.user(prompt))).addGenerationPrompt(true).build();
                    // generate(String) applies this same tokenizer-owned template exactly once.
                    // Retain the default model thinking protocol, including its prefilling of <think>.
                    int promptTokens = tokenizer.encodePrompt(prompt).getIds().length;
                    assertTrue(promptTokens <= maxPrefill, "Prompt would be truncated: " + prompt);
                    log.info("NVFP4 generation prompt={} promptTokens={} maxNewTokens={} contextCap={}",
                            prompt, promptTokens, maxTokens, contextCap);
                    if (mtp) pipeline.setSamplingConfig(SamplingConfig.speculative());
                    long start = System.nanoTime();
                    GenerationResult generated = pipeline.generate(prompt, maxTokens);
                    long elapsedMs = (System.nanoTime() - start) / 1_000_000;
                    if (mtp) {
                        log.info("NVFP4 native predictor prompt={} proposed={} accepted={} steps={}", prompt,
                                generated.getTotalSpeculativeTokens(), generated.getTotalAcceptedTokens(),
                                generated.getSpeculativeSteps());
                        assertTrue(generated.getTotalSpeculativeTokens() > 0, "Predictor proposed no tokens: " + prompt);
                        assertTrue(generated.getTotalAcceptedTokens() > 0, "Predictor accepted no tokens: " + prompt);
                        assertTrue(generated.getSpeculativeSteps() > 0, "Predictor never ran: " + prompt);
                        pipeline.setSamplingConfig(SamplingConfig.greedy());
                        GenerationResult greedy = pipeline.generate(prompt, maxTokens);
                        int divergence = Arrays.mismatch(greedy.getTokenIds(), generated.getTokenIds());
                        log.info("NVFP4 MTP parity firstDivergence={} greedyTokens={} mtpTokens={} greedyText={} mtpText={}",
                                divergence, Arrays.toString(greedy.getTokenIds()), Arrays.toString(generated.getTokenIds()),
                                greedy.getText(), generated.getText());
                        assertArrayEquals(greedy.getTokenIds(), generated.getTokenIds(),
                                "Native checkpoint MTP must match greedy on the same prompt: " + prompt
                                        + "; first divergence=" + divergence);
                    }
                    log.info("NVFP4 generation response={} generatedTokens={} tokenIds={} elapsedMs={} finish={}",
                            generated.getText(), generated.getGeneratedTokenCount(),
                            Arrays.toString(generated.getTokenIds()), elapsedMs, generated.getFinishReason());
                    assertTrue(generated.getGeneratedTokenCount() > 0 && generated.getGeneratedTokenCount() <= maxTokens);
                    assertEquals(generated.getGeneratedTokenCount(), generated.getTokenIds().length);
                    assertEquals(promptTokens, generated.getPromptTokenCount());
                    responses.add(pipeline.parseChatOutput(request, generated.getText()));
                }
                // Check all three answers, not just nonempty output or the presence of an echoed
                // prompt keyword. Parsing uses the model protocol to separate reasoning from answer.
                if (prefixOnly) {
                    log.info("NVFP4 MTP prefix parity passed; complete reasoning/final answers remain a separate required test");
                } else {
                    assertAll("Pinned NVFP4 deterministic prompt correctness",
                            () -> assertAnswer(expected[0], responses.get(0), prompts[0]),
                            () -> assertAnswer(expected[1], responses.get(1), prompts[1]),
                            () -> assertAnswer(expected[2], responses.get(2), prompts[2]));
                }
            }
        }
    }

    private static void assertAnswer(String expected, ChatGenerationResult response, String prompt) {
        assertTrue(response.getParseErrors().isEmpty(), prompt + ": " + response.getParseErrors());
        assertEquals(GSON.fromJson(expected, JsonObject.class),
                GSON.fromJson(response.getContent().trim(), JsonObject.class),
                prompt + " raw response: " + response.getRawText());
    }

    private static void assertCacheInputTypes(SameDiff graph) {
        int kvInputs = 0;
        for (String input : graph.inputs()) {
            if (input.startsWith("past_key_values.") || input.startsWith("past_conv_state.")) {
                assertEquals(DataType.BFLOAT16, graph.getVariable(input).dataType(), input);
                if (input.startsWith("past_key_values.")) kvInputs++;
            } else if (input.startsWith("past_gdn_state.")) {
                assertEquals(DataType.FLOAT, graph.getVariable(input).dataType(), input);
            }
        }
        assertTrue(kvInputs > 0, "Must execute the actual hybrid attention graph with BF16 KV caches");
    }

    private static void assertPackedExecutionGraph(SameDiff graph, JsonObject quantized,
            JsonObject weights, Map<String, SafeTensorsHeader> headers) {
        Map<String, Integer> expected = new LinkedHashMap<>();
        expected.put("modelopt_nvfp4_linear", 0);
        expected.put("modelopt_fp8_linear", 0);
        long expectedWeightBytes = 0;
        for (Map.Entry<String, JsonElement> entry : quantized.entrySet()) {
            String name = entry.getKey();
            if (!name.startsWith("model.language_model.") && !name.equals("lm_head")) continue;
            String algorithm = entry.getValue().getAsJsonObject().get("quant_algo").getAsString();
            assertTrue(algorithm.equals("W4A16_NVFP4") || algorithm.equals("FP8"), name);
            String op = algorithm.equals("FP8") ? "modelopt_fp8_linear" : "modelopt_nvfp4_linear";
            expected.put(op, expected.get(op) + 1);
            expectedWeightBytes = Math.addExact(expectedWeightBytes, info(name + ".weight", weights, headers).getDataLength());
        }
        // The shared head has three legitimate consumers: full target verification,
        // last-position prefill, and predictor logits. Storage is counted exactly once.
        String headOp = "FP8".equals(quantized.getAsJsonObject("lm_head").get("quant_algo").getAsString())
                ? "modelopt_fp8_linear" : "modelopt_nvfp4_linear";
        expected.put(headOp, expected.get(headOp) + 2);
        Map<String, List<String>> headConsumers = new LinkedHashMap<>();
        Map<String, Integer> actual = new LinkedHashMap<>();
        actual.put("modelopt_nvfp4_linear", 0);
        actual.put("modelopt_fp8_linear", 0);
        Set<String> projectionWeights = new HashSet<>();
        long actualWeightBytes = 0;
        for (SameDiffOp op : graph.getOps().values()) {
            String name = op.getOp().opName();
            if (!actual.containsKey(name)) continue;
            actual.put(name, actual.get(name) + 1);
            List<String> inputs = op.getInputsToOp();
            assertEquals(4, inputs.size(), name);
            String weightName = inputs.get(1);
            boolean firstUse = projectionWeights.add(weightName);
            if ("lm_head.weight".equals(weightName)) {
                assertEquals(1, op.getOutputsOfOp().size());
                String output = op.getOutputsOfOp().get(0);
                assertTrue(Set.of("lm_logits", "lm_logits_last", "mtp_logits").contains(output), output);
                assertNull(headConsumers.put(output, new ArrayList<>(inputs.subList(1, 4))),
                        "Duplicate shared-head consumer: " + output);
            } else {
                assertTrue(firstUse, "Duplicate projection instead of complete model: " + weightName);
            }
            assertEquals(VariableType.CONSTANT, graph.getVariable(weightName).getVariableType(), weightName);
            INDArray weight = graph.getArrForVarName(weightName);
            INDArray scale = graph.getArrForVarName(inputs.get(2));
            INDArray secondScale = graph.getArrForVarName(inputs.get(3));
            assertNotNull(weight, weightName);
            assertNotNull(scale, inputs.get(2));
            assertNotNull(secondScale, inputs.get(3));
            boolean packed = name.equals("modelopt_nvfp4_linear");
            assertEquals(packed ? DataType.UINT8 : DataType.FLOAT8, weight.dataType(), weightName);
            assertEquals(2, weight.rank(), weightName);
            assertEquals(packed ? DataType.FLOAT8 : DataType.FLOAT, scale.dataType(), inputs.get(2));
            if (packed) {
                assertEquals(0L, weight.size(1) % 8);
                assertArrayEquals(new long[]{weight.size(0), weight.size(1) / 8}, scale.shape());
            } else {
                assertEquals(1L, scale.length());
            }
            assertEquals(DataType.FLOAT, secondScale.dataType());
            assertEquals(1L, secondScale.length());
            if (firstUse) {
                actualWeightBytes = Math.addExact(actualWeightBytes, Math.multiplyExact(weight.length(), weight.dataType().width()));
            }
        }
        assertEquals(Set.of("lm_logits", "lm_logits_last", "mtp_logits"), headConsumers.keySet());
        assertEquals(headConsumers.get("lm_logits"), headConsumers.get("mtp_logits"),
                "Predictor must reuse exactly the target packed head and scales");
        assertEquals(headConsumers.get("lm_logits"), headConsumers.get("lm_logits_last"));
        assertTrue(expected.values().stream().allMatch(count -> count > 0), "Checkpoint must exercise both formats");
        assertEquals(expected, actual, "Every quantized text projection must execute a ModelOpt op");
        assertEquals(expectedWeightBytes, actualWeightBytes, "Packed text projection storage must remain unchanged");
        log.info("NVFP4 execution graph ops={} packedProjectionBytes={}", actual, actualWeightBytes);
    }

    private static void assertMtpCheckpointPayload(SameDiff graph, ModelOptQwenConfig config,
            JsonObject weights, Map<String, SafeTensorsHeader> headers) throws IOException {
        Set<String> expected = new HashSet<>(List.of("mtp.fc.weight", "mtp.norm.weight",
                "mtp.pre_fc_norm_embedding.weight", "mtp.pre_fc_norm_hidden.weight"));
        for (String suffix : List.of("input_layernorm", "post_attention_layernorm", "self_attn.q_proj",
                "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "self_attn.q_norm",
                "self_attn.k_norm", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")) {
            expected.add("mtp.layers.0." + suffix + ".weight");
        }
        Set<String> actual = new HashSet<>();
        for (Map.Entry<String, JsonElement> entry : weights.entrySet()) {
            if (entry.getKey().startsWith("mtp.")) actual.add(entry.getKey());
        }
        assertEquals(expected, actual, "Every pinned MTP tensor must be mapped; no silent omission");
        for (String name : expected) assertEquals("BF16", info(name, weights, headers).getDtype(), name);
        int layer = config.getArchitecture().getNumLayers();
        assertArrayEquals(new long[]{config.getArchitecture().getHiddenSize(),
                        2L * config.getArchitecture().getHiddenSize()},
                graph.getArrForVarName("mtp.eh_proj.weight").shape());
        assertEquals(DataType.BFLOAT16, graph.getArrForVarName("mtp.eh_proj.weight").dataType());
        assertNotNull(graph.getVariable("gate_sigmoid_" + layer), "Predictor uses Qwen sigmoid attention gating");
        Map<String, String> norms = Map.of(
                "mtp.pre_fc_norm_embedding.weight", "mtp.enorm.weight",
                "mtp.pre_fc_norm_hidden.weight", "mtp.hnorm.weight",
                "mtp.norm.weight", "mtp.shared_head.norm.weight");
        for (Map.Entry<String, String> norm : norms.entrySet()) {
            File shard = download(weights.get(norm.getKey()).getAsString());
            try (SafeTensorsReader reader = SafeTensorsReader.open(shard);
                 INDArray source = reader.readTensor(norm.getKey())) {
                INDArray mapped = graph.getArrForVarName(norm.getValue());
                assertEquals(DataType.FLOAT, mapped.dataType(), "Norm offset must be added in FP32");
                assertArrayEquals(source.shape(), mapped.shape());
                for (int i = 0; i < source.length(); i++) {
                    assertEquals(1.0f + source.getFloat(i), mapped.getFloat(i), 0.0f, norm.getKey());
                }
            }
        }
        for (String projection : List.of("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")) {
            String name = "model.layers." + layer + "." + projection + ".weight";
            assertEquals(VariableType.CONSTANT, graph.getVariable(name).getVariableType(), name);
            assertEquals(DataType.BFLOAT16, graph.getArrForVarName(name).dataType(), name);
            assertArrayEquals(info("mtp.layers.0." + projection + ".weight", weights, headers).getShape(),
                    graph.getArrForVarName(name).shape(), name);
        }
    }

    private static File download(String name) throws IOException {
        // downloadCustom caches by filename only. Its downloader moves a temporary file,
        // but does not verify cached contents or Content-Length/checksums. Revision-keyed
        // names prevent cross-revision reuse; validate every range and exact file size below.
        // This is structural integrity, not a cryptographic authenticity assertion.
        File file = LLMModelDownloader.downloadCustom(BASE + name,
                "nvidia-Qwen3.6-27B-NVFP4-" + REVISION + "-" + name);
        assertTrue(file.isFile() && file.length() > 0, "Empty/missing cache entry: " + file);
        log.info("STORAGE artifact {} bytes={} source={}{}", file, file.length(), BASE, name);
        return file;
    }

    private static JsonObject json(File file) throws IOException {
        try (Reader reader = Files.newBufferedReader(file.toPath(), StandardCharsets.UTF_8)) {
            return GSON.fromJson(reader, JsonObject.class);
        }
    }

    private static long validateHeader(File file, String shard, SafeTensorsHeader header, JsonObject weights) {
        Set<String> expectedNames = new HashSet<>();
        for (Map.Entry<String, JsonElement> entry : weights.entrySet()) {
            if (shard.equals(entry.getValue().getAsString())) {
                expectedNames.add(entry.getKey());
            }
        }
        assertEquals(expectedNames, header.getTensorNames(), "Index/header mismatch: " + shard);
        List<TensorInfo> ordered = new ArrayList<>(header.getTensors().values());
        ordered.sort(Comparator.comparingLong(TensorInfo::getDataStart));
        long end = 0;
        for (TensorInfo tensor : ordered) {
            assertNotNull(tensor.getShape(), tensor.getName());
            assertNotNull(tensor.getDataOffsets(), tensor.getName());
            assertEquals(2, tensor.getDataOffsets().length, tensor.getName());
            assertEquals(end, tensor.getDataStart(), "Gap/overlap: " + tensor.getName());
            long elements = 1;
            for (long dim : tensor.getShape()) {
                assertTrue(dim > 0, tensor.getName());
                elements = Math.multiplyExact(elements, dim);
            }
            long bytes = Math.multiplyExact(elements, tensor.getSafeTensorsDtype().getBytesPerElement());
            end = Math.addExact(end, bytes);
            assertEquals(end, tensor.getDataEnd(), "Invalid tensor range: " + tensor.getName());
            assertTrue(Math.addExact(header.getDataOffset(), end) <= file.length(),
                    "Truncated cache entry: " + file + " tensor=" + tensor.getName());
        }
        assertEquals(file.length(), Math.addExact(header.getDataOffset(), end), "Trailing/truncated data: " + file);
        log.info("STORAGE header {} tensors={} headerBytes={} payloadBytes={}",
                shard, ordered.size(), header.getHeaderSize(), end);
        return end;
    }

    private static TensorInfo info(String name, JsonObject weights, Map<String, SafeTensorsHeader> headers) {
        assertTrue(weights.has(name), "Missing index tensor: " + name);
        TensorInfo result = headers.get(weights.get(name).getAsString()).getTensorInfo(name);
        assertNotNull(result, name);
        assertEquals(name, result.getName());
        return result;
    }

    private static void importScales(String prefix, boolean packed, long rows, long columns, int groupSize,
                                     JsonObject weights, Map<String, File> shards,
                                     Map<String, SafeTensorsHeader> headers) throws IOException {
        Set<String> names = new TreeSet<>();
        for (Map.Entry<String, JsonElement> entry : weights.entrySet()) {
            if (entry.getKey().startsWith(prefix + ".") && entry.getKey().contains("scale")) {
                names.add(entry.getKey());
            }
        }
        Set<String> expected = new TreeSet<>(Arrays.asList(prefix + ".input_scale", prefix + ".weight_scale"));
        if (packed) {
            expected.add(prefix + ".weight_scale_2");
        }
        assertEquals(expected, names, "Scale names discovered from weight_map");
        for (String name : names) {
            TensorInfo tensor = info(name, weights, headers);
            boolean blockScale = packed && name.equals(prefix + ".weight_scale");
            long[] shape = tensor.getShape();
            if (blockScale) {
                assertArrayEquals(new long[]{rows, columns / groupSize}, shape, name);
            } else {
                long elements = 1;
                for (long dim : shape) {
                    elements = Math.multiplyExact(elements, dim);
                }
                assertEquals(1L, elements, "Per-tensor scale: " + name);
            }
            importTensor(name, blockScale ? "F8_E4M3" : "F32",
                    blockScale ? DataType.FLOAT8 : DataType.FLOAT, shape, true, weights, shards, headers);
        }
    }

    private static void importTensor(String name, String storageDtype, DataType dtype, long[] shape,
                                     boolean scale, JsonObject weights, Map<String, File> shards,
                                     Map<String, SafeTensorsHeader> headers) throws IOException {
        String shard = weights.get(name).getAsString();
        TensorInfo tensor = info(name, weights, headers);
        assertEquals(storageDtype, tensor.getDtype(), name);
        assertArrayEquals(shape, tensor.getShape(), name);
        File file = shards.get(shard);
        try (SafeTensorsReader reader = SafeTensorsReader.open(file);
             INDArray array = reader.readTensor(name)) {
            assertEquals(dtype, array.dataType(), name);
            assertArrayEquals(shape, array.shape(), name);
            assertEquals(tensor.getDataLength(), Math.multiplyExact(array.length(), dtype.width()), name);
            // Read back ALL imported host bytes, including bytes past the 2 GB boundary.
            // No toFloatVector/cast/dup/device op and no whole-tensor Java byte[] allocation.
            BytePointer pointer = new BytePointer(array.data().pointer()).capacity(tensor.getDataLength());
            byte[] expected = new byte[1024 * 1024];
            byte[] actual = new byte[expected.length];
            try (RandomAccessFile source = new RandomAccessFile(file, "r")) {
                source.seek(Math.addExact(headers.get(shard).getDataOffset(), tensor.getDataStart()));
                for (long offset = 0; offset < tensor.getDataLength(); ) {
                    int count = (int) Math.min(expected.length, tensor.getDataLength() - offset);
                    source.readFully(expected, 0, count);
                    pointer.position(offset).get(actual, 0, count);
                    int width = dtype.width();
                    for (int i = 0; i < count; i++) {
                        int nativeIndex = ByteOrder.nativeOrder() == ByteOrder.LITTLE_ENDIAN ? i
                                : i - i % width + width - 1 - i % width;
                        if (expected[i] != actual[nativeIndex]) {
                            fail(name + " storage mismatch at byte " + (offset + i));
                        }
                    }
                    if (scale) {
                        assertFiniteScales(expected, count, storageDtype, name, offset);
                    }
                    offset += count;
                }
            }
            // pointer borrows the array's allocation; only array owns/closes it.
            log.info("STORAGE imported {} shape={} dtype={} nd4j={} bytes={} shard={} (bit-exact)",
                    name, Arrays.toString(shape), storageDtype, dtype, tensor.getDataLength(), file);
        }
    }

    private static void assertFiniteScales(byte[] bytes, int count, String dtype, String name, long offset) {
        if (dtype.equals("F8_E4M3")) {
            // E4M3FN reserves both signs of 0x7f for NaN; 0x7e is finite 448.
            for (int i = 0; i < count; i++) {
                if ((bytes[i] & 0x7f) == 0x7f) {
                    fail(name + " non-finite scale at byte " + (offset + i));
                }
            }
        } else {
            assertEquals("F32", dtype, name);
            ByteBuffer buffer = ByteBuffer.wrap(bytes, 0, count).order(ByteOrder.LITTLE_ENDIAN);
            while (buffer.hasRemaining()) {
                if (!Float.isFinite(buffer.getFloat())) {
                    fail(name + " non-finite scale near byte " + (offset + buffer.position()));
                }
            }
        }
    }
}
