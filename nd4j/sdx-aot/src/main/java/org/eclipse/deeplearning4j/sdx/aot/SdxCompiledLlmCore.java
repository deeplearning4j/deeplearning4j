/*
 * Copyright (c) Eclipse Deeplearning4j
 * SPDX-License-Identifier: Apache-2.0
 */
package org.eclipse.deeplearning4j.sdx.aot;

import org.eclipse.deeplearning4j.llm.generation.ChatGenerationResult;
import org.eclipse.deeplearning4j.llm.tokenizer.ChatTemplate;
import org.eclipse.deeplearning4j.llm.tokenizer.HuggingFaceTokenizer;
import org.nd4j.dsp.model.SdxTargetProfile;
import org.nd4j.dsp.runtime.SdxRuntime;
import org.nd4j.dsp.runtime.SdxTextSession;
import org.nd4j.shade.jackson.databind.JsonNode;
import org.nd4j.shade.jackson.databind.ObjectMapper;
import org.nd4j.shade.jackson.databind.node.ObjectNode;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Objects;
import java.util.function.BooleanSupplier;
import java.util.function.Consumer;

/**
 * Shared compiled-model execution path. Target profiles select strict runtime options;
 * tokenization, chat protocol, generation and execution reporting remain backend-neutral.
 */
final class SdxCompiledLlmCore implements SdxLlmModel {
    private static final ObjectMapper MAPPER = new ObjectMapper();

    private final SdxRuntime runtime;
    private final SdxRuntime.SdxModel model;
    private final SdxTextSession session;
    private final HuggingFaceTokenizer tokenizer;
    private final SdxTargetProfile target;
    private final int defaultMaxNewTokens;
    private volatile SdxTextSession.GenerationReport lastReport;
    private volatile long[] lastPromptTokenIds = new long[0];
    private volatile long[] lastGeneratedTokenIds = new long[0];

    private SdxCompiledLlmCore(SdxRuntime runtime, SdxRuntime.SdxModel model,
                               SdxTextSession session, HuggingFaceTokenizer tokenizer,
                               SdxTargetProfile target, int defaultMaxNewTokens) {
        this.runtime = runtime;
        this.model = model;
        this.session = session;
        this.tokenizer = tokenizer;
        this.target = target;
        this.defaultMaxNewTokens = defaultMaxNewTokens;
    }

    static SdxCompiledLlmCore load(String bundlePath, String tokenizerPath,
                                   String targetProfile, String optionsJson) throws IOException {
        Objects.requireNonNull(bundlePath, "bundlePath");
        SdxTargetProfile target = SdxTargetProfile.fromId(requireText(targetProfile, "target profile"));
        JsonNode options = parseObject(optionsJson);
        SdxRuntime runtime = null;
        SdxRuntime.SdxModel model = null;
        SdxTextSession session = null;
        HuggingFaceTokenizer tokenizer = null;
        try {
            runtime = SdxRuntime.create();
            SdxRuntime.ModelOptions modelOptions = runtimeOptions(target);
            String deviceCache = textOrNull(options, "deviceCompilationCacheDirectory");
            if (deviceCache != null) modelOptions.deviceCompilationCacheDirectory(deviceCache);
            model = runtime.loadModel(bundlePath, modelOptions);

            String resolvedTokenizerPath = tokenizerPath;
            if (resolvedTokenizerPath == null || resolvedTokenizerPath.isBlank()) {
                resolvedTokenizerPath = model.tokenizerPath();
            }
            Path tokenizerFile = Path.of(requireText(resolvedTokenizerPath, "tokenizer path"))
                    .toAbsolutePath().normalize();
            if (Files.isDirectory(tokenizerFile)) tokenizerFile = tokenizerFile.resolve("tokenizer.json");
            if (!Files.isRegularFile(tokenizerFile)) {
                throw new IOException("tokenizer.json is not a regular file: " + tokenizerFile);
            }
            Path tokenizerConfig = tokenizerFile.resolveSibling("tokenizer_config.json");
            if (!Files.isRegularFile(tokenizerConfig)) {
                throw new IOException("tokenizer_config.json is required beside " + tokenizerFile);
            }
            tokenizer = HuggingFaceTokenizer.fromFile(tokenizerFile.toFile());
            session = model.createTextSession();
            int defaultMaxNewTokens = positive(options.path("maxNewTokens").asInt(128), "maxNewTokens");
            return new SdxCompiledLlmCore(runtime, model, session, tokenizer,
                    target, defaultMaxNewTokens);
        } catch (Throwable failure) {
            closeQuietly(session);
            closeQuietly(tokenizer);
            closeQuietly(model);
            closeQuietly(runtime);
            if (failure instanceof IOException) throw (IOException) failure;
            throw new IOException("Could not open compiled SDX model: " + failure.getMessage(), failure);
        }
    }

    @Override
    public synchronized String generateText(String prompt, String optionsJson) throws IOException {
        return generateStreaming(prompt, optionsJson, null, null);
    }

    @Override
    public synchronized String generateStreaming(String prompt, String optionsJson,
                                                 Consumer<String> onChunk,
                                                 BooleanSupplier shouldCancel) throws IOException {
        if (prompt == null || prompt.isBlank()) throw new IllegalArgumentException("prompt must not be blank");
        JsonNode options = parseObject(optionsJson);
        int maxNewTokens = positive(options.path("maxNewTokens").asInt(defaultMaxNewTokens),
                "maxNewTokens");
        SdxTextSession.GenerationOptions generation = generationOptions(maxNewTokens,
                options.path("sampling"));
        long[] promptIds = tokenizer.encodeLong(prompt, false);
        if (promptIds.length == 0) throw new IllegalArgumentException("prompt encoded to zero tokens");
        lastPromptTokenIds = promptIds.clone();
        lastGeneratedTokenIds = new long[0];

        session.reset();
        HuggingFaceTokenizer.DecodeStream decoder =
                onChunk == null ? null : tokenizer.newDecodeStream(true);
        try {
            SdxTextSession.GenerationResult result = session.generate(promptIds, generation, tokenId -> {
                if (decoder == null) return;
                String chunk = decoder.step(tokenId);
                if (!chunk.isEmpty()) onChunk.accept(chunk);
            }, shouldCancel);
            lastReport = result.report();
            lastGeneratedTokenIds = result.tokenIds().clone();
            return tokenizer.decode(result.tokenIds(), true);
        } catch (RuntimeException failure) {
            throw new IOException("Compiled SDX generation failed: " + failure.getMessage(), failure);
        } finally {
            closeQuietly(decoder);
        }
    }

    @Override
    public String generateChat(String requestJson, String optionsJson) throws IOException {
        ChatTemplate.Request request = SdxLlmCore.parseChatRequest(requestJson);
        String prompt = renderChatPrompt(requestJson, request.isAddGenerationPrompt());
        return parseChatResult(requestJson, generateText(prompt, optionsJson));
    }

    @Override
    public String parseChatResult(String requestJson, String rawText) throws IOException {
        ChatTemplate.Request request = SdxLlmCore.parseChatRequest(requestJson);
        String templateSource = tokenizer.getChatTemplate();
        ChatGenerationResult result;
        if (templateSource == null || templateSource.isBlank()) {
            result = new ChatGenerationResult(rawText, request.getTools(), request.getToolCallFormat());
        } else {
            ChatTemplate template = new ChatTemplate(templateSource,
                    tokenizer.getBosToken(), tokenizer.getEosToken());
            result = new ChatGenerationResult(rawText, template.parseAssistantOutput(rawText),
                    request.getTools(), request.getToolCallFormat(), request.getToolChoice());
        }
        return SdxLlmCore.chatResultJson(result);
    }

    @Override
    public String renderChatPrompt(String messagesOrContextJson,
                                   boolean addGenerationPrompt) throws IOException {
        if (messagesOrContextJson == null || messagesOrContextJson.isBlank()) {
            throw new IllegalArgumentException("chat messages/context JSON must not be blank");
        }
        JsonNode input = MAPPER.readTree(messagesOrContextJson);
        ObjectNode context;
        if (input != null && input.isArray()) {
            context = MAPPER.createObjectNode();
            context.set("messages", input);
        } else if (input != null && input.isObject()) {
            context = ((ObjectNode) input).deepCopy();
        } else {
            throw new IllegalArgumentException("chat input must be a JSON message array or context object");
        }
        JsonNode messages = context.get("messages");
        if (messages == null || !messages.isArray() || messages.isEmpty()) {
            throw new IllegalArgumentException("messages_json must contain a non-empty messages array");
        }
        context.put("add_generation_prompt", addGenerationPrompt);
        return tokenizer.applyChatTemplateContext(context.toString());
    }

    @Override
    public int[] tokenize(String text, boolean addSpecialTokens) {
        return tokenizer.encode(text, addSpecialTokens).getIds();
    }

    @Override
    public String detokenize(int[] ids, boolean skipSpecialTokens) {
        return tokenizer.decode(ids, skipSpecialTokens);
    }

    @Override
    public String lastResultJson() {
        ObjectNode result = MAPPER.createObjectNode();
        SdxTextSession.GenerationReport report = lastReport;
        if (report == null) {
            result.put("hasResult", false);
            return result.toString();
        }
        result.put("hasResult", true);
        result.put("finishReason", report.finishReason().name());
        result.put("nativeFinishReason", report.nativeFinishReason());
        result.put("promptTokens", report.promptTokenCount());
        result.put("generatedTokens", report.generatedTokenCount());
        result.put("totalGeneratedTokens", report.totalGeneratedTokenCount());
        result.put("contextPosition", report.contextPosition());
        result.put("elapsedTimeNanos", report.elapsedTimeNanos());
        result.put("prefillTimeNanos", report.prefillTimeNanos());
        result.put("decodeTimeNanos", report.decodeTimeNanos());
        result.put("decodeTokensPerSecond", report.decodeTokensPerSecond());
        result.put("backendReportAvailable", report.backendReportAvailable());
        result.put("requestedBackend", report.requestedBackend());
        result.put("appliedBackend", report.appliedBackend());
        result.put("backendStatusCode", report.backendStatusCode());
        result.put("usedFallback", report.usedFallback());
        result.put("requestedGpuTarget", report.requestedGpuTarget());
        result.put("appliedGpuTarget", report.appliedGpuTarget());
        result.put("planPhase", report.planPhase());
        result.put("executionCount", report.executionCount());
        for (long tokenId : lastPromptTokenIds) result.withArray("promptTokenIds").add(tokenId);
        for (long tokenId : lastGeneratedTokenIds) result.withArray("generatedTokenIds").add(tokenId);
        return result.toString();
    }

    @Override
    public String infoJson() {
        ObjectNode result = MAPPER.createObjectNode();
        result.put("executionMode", "compiled-sdx");
        result.put("targetProfile", target.id());
        result.put("runtimeAbiVersion", runtime.abiVersion());
        result.put("vocabSize", tokenizer.getVocabSize());
        result.put("bosTokenId", tokenizer.getBosTokenId());
        result.put("eosTokenId", tokenizer.getEosTokenId());
        result.put("hasChatTemplate", tokenizer.getChatTemplate() != null);
        return result.toString();
    }

    @Override
    public synchronized void close() {
        RuntimeException failure = null;
        failure = close(failure, session);
        failure = close(failure, tokenizer);
        failure = close(failure, model);
        failure = close(failure, runtime);
        if (failure != null) throw failure;
    }

    static SdxRuntime.ModelOptions runtimeOptions(SdxTargetProfile target) {
        switch (target) {
            case ANDROID_ARM64_VULKAN:
                return SdxRuntime.ModelOptions.mobileVulkan();
            case ANDROID_ARM64_HEXAGON_HTP:
                return SdxRuntime.ModelOptions.mobileHexagon();
            case ANDROID_ARM64_NNAPI_ACCELERATOR:
                return new SdxRuntime.ModelOptions()
                        .backend(SdxRuntime.SDX_BACKEND_ARM_HYBRID)
                        .strictBackend(false)
                        .allowRuntimeJit(false)
                        .gpuTarget(SdxRuntime.SDX_GPU_TARGET_AUTO);
            case IOS_ARM64_METAL:
                return new SdxRuntime.ModelOptions()
                        .backend(SdxRuntime.SDX_BACKEND_MLX)
                        .strictBackend(true)
                        .allowRuntimeJit(false)
                        .gpuTarget(SdxRuntime.SDX_GPU_TARGET_METAL);
            default:
                throw new IllegalArgumentException(
                        "Target profile does not use the shared SDX runtime: " + target.id());
        }
    }

    private static SdxTextSession.GenerationOptions generationOptions(int maxNewTokens,
                                                                     JsonNode sampling) {
        JsonNode value = sampling == null || !sampling.isObject()
                ? MAPPER.createObjectNode() : sampling;
        SdxTextSession.GenerationOptions options = new SdxTextSession.GenerationOptions(maxNewTokens)
                .temperature(value.path("temperature").asDouble(0.0))
                .topK(value.path("topK").asInt(0))
                .topP(value.path("topP").asDouble(1.0))
                .repetitionPenalty(value.path("repetitionPenalty").asDouble(1.0))
                .frequencyPenalty(value.path("frequencyPenalty").asDouble(0.0))
                .presencePenalty(value.path("presencePenalty").asDouble(0.0))
                .typicalP(value.path("typicalP").asDouble(1.0))
                .seed(value.path("seed").asLong(0L));
        if (value.has("minP")) options.minP(value.get("minP").asDouble());
        if (value.has("minNewTokens")) options.minNewTokens(value.get("minNewTokens").asInt());
        if (value.has("xtcProbability") || value.has("xtcThreshold")) {
            options.xtc(value.path("xtcProbability").asDouble(0.0),
                    value.path("xtcThreshold").asDouble(0.1));
        }
        return options;
    }

    private static JsonNode parseObject(String json) throws IOException {
        if (json == null || json.isBlank()) return MAPPER.createObjectNode();
        JsonNode value = MAPPER.readTree(json);
        if (value == null || !value.isObject()) throw new IllegalArgumentException("options JSON must be an object");
        return value;
    }

    private static String textOrNull(JsonNode node, String field) {
        JsonNode value = node.get(field);
        return value == null || value.isNull() || value.asText().isBlank() ? null : value.asText();
    }

    private static String requireText(String value, String label) {
        if (value == null || value.isBlank()) throw new IllegalArgumentException(label + " must not be blank");
        return value;
    }

    private static int positive(int value, String label) {
        if (value <= 0) throw new IllegalArgumentException(label + " must be positive");
        return value;
    }

    private static RuntimeException close(RuntimeException failure, AutoCloseable resource) {
        if (resource == null) return failure;
        try {
            resource.close();
        } catch (Exception closeFailure) {
            RuntimeException wrapped = closeFailure instanceof RuntimeException
                    ? (RuntimeException) closeFailure
                    : new IllegalStateException("Could not close compiled SDX resource", closeFailure);
            if (failure == null) return wrapped;
            failure.addSuppressed(wrapped);
        }
        return failure;
    }

    private static void closeQuietly(AutoCloseable resource) {
        if (resource == null) return;
        try {
            resource.close();
        } catch (Exception ignored) {
            // Preserve the original load/generation failure.
        }
    }
}
