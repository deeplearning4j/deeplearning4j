/*
 * ******************************************************************************
 * *
 * * This program and the accompanying materials are made available under the
 * * terms of the Apache License, Version 2.0 which is available at
 * * https://www.apache.org/licenses/LICENSE-2.0.
 * *
 * * SPDX-License-Identifier: Apache-2.0
 * *****************************************************************************
 */

package org.nd4j.dsp.runtime.litertlm;

import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.Pointer;
import org.nd4j.dsp.runtime.litertlm.bindings.LiteRtLmNative;
import org.nd4j.dsp.runtime.litertlm.bindings.LiteRtLmNative.LiteRtLmConversation;
import org.nd4j.dsp.runtime.litertlm.bindings.LiteRtLmNative.LiteRtLmConversationConfig;
import org.nd4j.dsp.runtime.litertlm.bindings.LiteRtLmNative.LiteRtLmConversationOptionalArgs;
import org.nd4j.dsp.runtime.litertlm.bindings.LiteRtLmNative.LiteRtLmEngine;
import org.nd4j.dsp.runtime.litertlm.bindings.LiteRtLmNative.LiteRtLmEngineSettings;
import org.nd4j.dsp.runtime.litertlm.bindings.LiteRtLmNative.LiteRtLmJsonResponse;
import org.nd4j.dsp.runtime.litertlm.bindings.LiteRtLmNative.LiteRtLmSamplerParams;
import org.nd4j.dsp.runtime.litertlm.bindings.LiteRtLmNative.LiteRtLmSessionConfig;
import org.nd4j.dsp.runtime.litertlm.bindings.LiteRtLmNative.LiteRtLmStreamCallback;

import java.lang.reflect.Field;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Consumer;

/**
 * Device-only local chat session backed by LiteRT-LM on the Google Tensor G5
 * TPU.
 *
 * <p>The backend is intentionally fixed to {@code npu}. This class never offers
 * CPU, GPU, NNAPI, or BLAS fallback. The model must be an AOT-compiled
 * {@code .litertlm} package for {@code Tensor_G5}; unsupported devices and
 * incomplete native packages fail before model loading.</p>
 */
public final class SdxLiteRtLmChatSession implements AutoCloseable {

    public static final String BACKEND = "npu";
    public static final String REQUIRED_SOC = "Tensor_G5";
    public static final int DEFAULT_CONTEXT_TOKENS = 1280;
    public static final int DEFAULT_MAX_OUTPUT_TOKENS = 256;

    private final Object lifecycleLock = new Object();
    private final LiteRtLmEngine engine;
    private final LiteRtLmConversation conversation;
    private final int maxOutputTokens;
    private StreamState activeStream;
    private boolean closed;

    private SdxLiteRtLmChatSession(Builder builder) {
        requireGoogleTensorG5();
        validateModel(builder.modelPath);
        validateDispatchDirectory(builder.dispatchLibraryDirectory);

        LiteRtLmEngineSettings settings =
                LiteRtLmNative.litert_lm_engine_settings_create(
                        builder.modelPath.toAbsolutePath().toString(),
                        BACKEND,
                        null,
                        null);
        requirePointer(settings, "create LiteRT-LM engine settings");

        LiteRtLmEngine createdEngine = null;
        LiteRtLmConversation createdConversation = null;
        LiteRtLmSamplerParams sampler = null;
        LiteRtLmSessionConfig sessionConfig = null;
        LiteRtLmConversationConfig conversationConfig = null;
        try {
            LiteRtLmNative.litert_lm_engine_settings_set_litert_dispatch_lib_dir(
                    settings,
                    builder.dispatchLibraryDirectory.toAbsolutePath().toString());
            LiteRtLmNative.litert_lm_engine_settings_set_max_num_tokens(
                    settings, builder.contextTokens);
            if (builder.cacheDirectory != null) {
                LiteRtLmNative.litert_lm_engine_settings_set_cache_dir(
                        settings, builder.cacheDirectory.toAbsolutePath().toString());
            }
            if (builder.enableBenchmark) {
                LiteRtLmNative.litert_lm_engine_settings_enable_benchmark(settings);
            }

            createdEngine = LiteRtLmNative.litert_lm_engine_create(settings);
            requirePointer(createdEngine, "create LiteRT-LM NPU engine");

            sampler = LiteRtLmNative.litert_lm_sampler_params_create(
                    LiteRtLmNative.kLiteRtLmSamplerTypeTopP);
            requirePointer(sampler, "create LiteRT-LM sampler");
            LiteRtLmNative.litert_lm_sampler_params_set_top_k(
                    sampler, builder.topK);
            LiteRtLmNative.litert_lm_sampler_params_set_top_p(
                    sampler, builder.topP);
            LiteRtLmNative.litert_lm_sampler_params_set_temperature(
                    sampler, builder.temperature);
            LiteRtLmNative.litert_lm_sampler_params_set_seed(
                    sampler, builder.seed);

            sessionConfig = LiteRtLmNative.litert_lm_session_config_create();
            requirePointer(sessionConfig, "create LiteRT-LM session config");
            LiteRtLmNative.litert_lm_session_config_set_max_output_tokens(
                    sessionConfig, builder.maxOutputTokens);
            LiteRtLmNative.litert_lm_session_config_set_apply_prompt_template(
                    sessionConfig, true);
            LiteRtLmNative.litert_lm_session_config_set_sampler_params(
                    sessionConfig, sampler);

            conversationConfig =
                    LiteRtLmNative.litert_lm_conversation_config_create();
            requirePointer(conversationConfig,
                    "create LiteRT-LM conversation config");
            LiteRtLmNative.litert_lm_conversation_config_set_session_config(
                    conversationConfig, sessionConfig);
            if (builder.systemMessage != null) {
                LiteRtLmNative.litert_lm_conversation_config_set_system_message(
                        conversationConfig,
                        messageJson("system", builder.systemMessage));
            }

            createdConversation = LiteRtLmNative.litert_lm_conversation_create(
                    createdEngine, conversationConfig);
            requirePointer(createdConversation,
                    "create LiteRT-LM conversation");
        } catch (RuntimeException | Error failure) {
            if (!isNull(createdConversation)) {
                LiteRtLmNative.litert_lm_conversation_delete(
                        createdConversation);
            }
            if (!isNull(createdEngine)) {
                LiteRtLmNative.litert_lm_engine_delete(createdEngine);
            }
            throw failure;
        } finally {
            if (!isNull(conversationConfig)) {
                LiteRtLmNative.litert_lm_conversation_config_delete(
                        conversationConfig);
            }
            if (!isNull(sessionConfig)) {
                LiteRtLmNative.litert_lm_session_config_delete(sessionConfig);
            }
            if (!isNull(sampler)) {
                LiteRtLmNative.litert_lm_sampler_params_delete(sampler);
            }
            LiteRtLmNative.litert_lm_engine_settings_delete(settings);
        }

        this.engine = createdEngine;
        this.conversation = createdConversation;
        this.maxOutputTokens = builder.maxOutputTokens;
    }

    public static Builder builder(Path modelPath, Path dispatchLibraryDirectory) {
        return new Builder(modelPath, dispatchLibraryDirectory);
    }

    /**
     * Sends one user turn and returns LiteRT-LM's complete JSON response.
     */
    public String sendMessage(String text) {
        return sendMessageJson(messageJson("user", requireText(text)), null,
                maxOutputTokens);
    }

    /**
     * Sends a preformatted LiteRT-LM message and returns the raw response JSON.
     */
    public String sendMessageJson(String messageJson, String extraContextJson,
                                  int turnMaxOutputTokens) {
        Objects.requireNonNull(messageJson, "messageJson");
        synchronized (lifecycleLock) {
            ensureOpenAndIdle();
            LiteRtLmConversationOptionalArgs optionalArgs =
                    createOptionalArgs(turnMaxOutputTokens);
            LiteRtLmJsonResponse response = null;
            try {
                response = LiteRtLmNative.litert_lm_conversation_send_message(
                        conversation, messageJson, extraContextJson, optionalArgs);
                requirePointer(response, "run LiteRT-LM conversation turn");
                BytePointer json =
                        LiteRtLmNative.litert_lm_json_response_get_string(response);
                requirePointer(json, "read LiteRT-LM conversation response");
                return new String(json.getStringBytes(), StandardCharsets.UTF_8);
            } finally {
                if (!isNull(response)) {
                    LiteRtLmNative.litert_lm_json_response_delete(response);
                }
                deleteOptionalArgs(optionalArgs);
            }
        }
    }

    /**
     * Streams response text chunks. The returned future completes with their
     * concatenation. Only one turn may be active on a conversation.
     */
    public CompletableFuture<String> sendMessageStreaming(
            String text, Consumer<String> chunkConsumer) {
        Objects.requireNonNull(chunkConsumer, "chunkConsumer");
        String json = messageJson("user", requireText(text));

        synchronized (lifecycleLock) {
            ensureOpenAndIdle();
            StreamState state = new StreamState(
                    createOptionalArgs(maxOutputTokens), chunkConsumer);
            activeStream = state;
            int status =
                    LiteRtLmNative.litert_lm_conversation_send_message_stream(
                            conversation, json, null, state.optionalArgs,
                            state.callback, null);
            if (status != 0) {
                activeStream = null;
                state.release();
                throw new NativeCallException(
                        "LiteRT-LM failed to start NPU streaming (status="
                                + status + ")");
            }
            return state.future;
        }
    }

    public int tokenCount() {
        synchronized (lifecycleLock) {
            ensureOpen();
            int count =
                    LiteRtLmNative.litert_lm_conversation_get_token_count(
                            conversation);
            if (count < 0) {
                throw new NativeCallException(
                        "LiteRT-LM failed to read the conversation token count");
            }
            return count;
        }
    }

    public void cancel() {
        synchronized (lifecycleLock) {
            ensureOpen();
            if (activeStream != null) {
                LiteRtLmNative.litert_lm_conversation_cancel_process(
                        conversation);
            }
        }
    }

    @Override
    public void close() {
        StreamState stream;
        synchronized (lifecycleLock) {
            if (closed) {
                return;
            }
            stream = activeStream;
            if (stream != null) {
                LiteRtLmNative.litert_lm_conversation_cancel_process(
                        conversation);
            }
        }

        if (stream != null) {
            try {
                stream.future.get(5, TimeUnit.SECONDS);
            } catch (Exception ignored) {
                stream.fail(new NativeCallException(
                        "Timed out cancelling LiteRT-LM streaming"));
            }
        }

        synchronized (lifecycleLock) {
            if (closed) {
                return;
            }
            closed = true;
            if (activeStream != null) {
                activeStream.release();
                activeStream = null;
            }
            LiteRtLmNative.litert_lm_conversation_delete(conversation);
            LiteRtLmNative.litert_lm_engine_delete(engine);
        }
    }

    static String messageJson(String role, String text) {
        Objects.requireNonNull(role, "role");
        Objects.requireNonNull(text, "text");
        return "{\"role\":\"" + jsonEscape(role)
                + "\",\"content\":[{\"type\":\"text\",\"text\":\""
                + jsonEscape(text) + "\"}]}";
    }

    static String normalizedSoc(String soc) {
        if (soc == null) {
            return "";
        }
        return soc.toLowerCase(Locale.ROOT)
                .replace("_", "")
                .replace("-", "")
                .replace(" ", "");
    }

    private LiteRtLmConversationOptionalArgs createOptionalArgs(
            int turnMaxOutputTokens) {
        if (turnMaxOutputTokens <= 0) {
            throw new IllegalArgumentException(
                    "max output tokens must be positive");
        }
        LiteRtLmConversationOptionalArgs args =
                LiteRtLmNative.litert_lm_conversation_optional_args_create();
        requirePointer(args, "create LiteRT-LM turn options");
        LiteRtLmNative
                .litert_lm_conversation_optional_args_set_max_output_tokens(
                        args, turnMaxOutputTokens);
        return args;
    }

    private static void deleteOptionalArgs(
            LiteRtLmConversationOptionalArgs optionalArgs) {
        if (!isNull(optionalArgs)) {
            LiteRtLmNative.litert_lm_conversation_optional_args_delete(
                    optionalArgs);
        }
    }

    private void ensureOpenAndIdle() {
        ensureOpen();
        if (activeStream != null) {
            throw new IllegalStateException(
                    "a LiteRT-LM conversation turn is already streaming");
        }
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException(
                    "LiteRT-LM chat session is closed");
        }
    }

    private static void validateModel(Path modelPath) {
        if (!Files.isRegularFile(modelPath)) {
            throw new IllegalArgumentException(
                    "LiteRT-LM model does not exist: " + modelPath);
        }
        if (!modelPath.getFileName().toString()
                .toLowerCase(Locale.ROOT).endsWith(".litertlm")) {
            throw new IllegalArgumentException(
                    "Google Tensor G5 requires an AOT .litertlm model: "
                            + modelPath);
        }
    }

    private static void validateDispatchDirectory(Path dispatchDirectory) {
        if (!Files.isDirectory(dispatchDirectory)) {
            throw new IllegalArgumentException(
                    "LiteRT dispatch directory does not exist: "
                            + dispatchDirectory);
        }
    }

    private static void requireGoogleTensorG5() {
        String socModel;
        try {
            Class<?> buildClass = Class.forName("android.os.Build");
            Field socField = buildClass.getField("SOC_MODEL");
            socModel = String.valueOf(socField.get(null));
        } catch (ReflectiveOperationException failure) {
            throw new UnsupportedOperationException(
                    "Google Tensor G5 provider requires Android Build.SOC_MODEL",
                    failure);
        }

        if (!"tensorg5".equals(normalizedSoc(socModel))) {
            throw new UnsupportedOperationException(
                    "Google Tensor G5 provider refuses unsupported SoC: "
                            + socModel);
        }
    }

    private static String requireText(String text) {
        Objects.requireNonNull(text, "text");
        if (text.isEmpty()) {
            throw new IllegalArgumentException("message text must not be empty");
        }
        return text;
    }

    private static String jsonEscape(String value) {
        StringBuilder escaped = new StringBuilder(value.length() + 16);
        for (int index = 0; index < value.length(); index++) {
            char character = value.charAt(index);
            switch (character) {
                case '"':
                    escaped.append("\\\"");
                    break;
                case '\\':
                    escaped.append("\\\\");
                    break;
                case '\b':
                    escaped.append("\\b");
                    break;
                case '\f':
                    escaped.append("\\f");
                    break;
                case '\n':
                    escaped.append("\\n");
                    break;
                case '\r':
                    escaped.append("\\r");
                    break;
                case '\t':
                    escaped.append("\\t");
                    break;
                default:
                    if (character < 0x20) {
                        escaped.append(String.format(Locale.ROOT,
                                "\\u%04x", (int) character));
                    } else {
                        escaped.append(character);
                    }
            }
        }
        return escaped.toString();
    }

    private static void requirePointer(Pointer pointer, String operation) {
        if (isNull(pointer)) {
            throw new NativeCallException(
                    "Failed to " + operation
                            + "; the direct NPU provider does not fall back");
        }
    }

    private static boolean isNull(Pointer pointer) {
        return pointer == null || pointer.isNull();
    }

    private final class StreamState {
        private final LiteRtLmConversationOptionalArgs optionalArgs;
        private final Consumer<String> chunkConsumer;
        private final CompletableFuture<String> future =
                new CompletableFuture<>();
        private final StringBuilder text = new StringBuilder();
        private final AtomicBoolean released = new AtomicBoolean();
        private final LiteRtLmStreamCallback callback =
                new LiteRtLmStreamCallback() {
                    @Override
                    public void call(Pointer callbackData, BytePointer chunk,
                                     boolean isFinal, BytePointer errorMessage) {
                        String error = value(errorMessage);
                        if (error != null && !error.isEmpty()) {
                            fail(new NativeCallException(error));
                            return;
                        }
                        String next = value(chunk);
                        if (next != null && !next.isEmpty()) {
                            text.append(next);
                            try {
                                chunkConsumer.accept(next);
                            } catch (RuntimeException consumerFailure) {
                                LiteRtLmNative
                                        .litert_lm_conversation_cancel_process(
                                                conversation);
                                fail(consumerFailure);
                                return;
                            }
                        }
                        if (isFinal) {
                            future.complete(text.toString());
                            finish();
                        }
                    }
                };

        private StreamState(
                LiteRtLmConversationOptionalArgs optionalArgs,
                Consumer<String> chunkConsumer) {
            this.optionalArgs = optionalArgs;
            this.chunkConsumer = chunkConsumer;
        }

        private void fail(Throwable failure) {
            if (future.completeExceptionally(failure)) {
                finish();
            }
        }

        private void finish() {
            synchronized (lifecycleLock) {
                if (activeStream == this) {
                    activeStream = null;
                }
            }
            release();
        }

        private void release() {
            if (released.compareAndSet(false, true)) {
                deleteOptionalArgs(optionalArgs);
                callback.close();
            }
        }
    }

    private static String value(BytePointer pointer) {
        return isNull(pointer)
                ? null
                : new String(pointer.getStringBytes(), StandardCharsets.UTF_8);
    }

    public static final class Builder {
        private final Path modelPath;
        private final Path dispatchLibraryDirectory;
        private Path cacheDirectory;
        private String systemMessage;
        private int contextTokens = DEFAULT_CONTEXT_TOKENS;
        private int maxOutputTokens = DEFAULT_MAX_OUTPUT_TOKENS;
        private int topK = 40;
        private float topP = 0.95f;
        private float temperature = 0.8f;
        private int seed;
        private boolean enableBenchmark = true;

        private Builder(Path modelPath, Path dispatchLibraryDirectory) {
            this.modelPath = Objects.requireNonNull(modelPath, "modelPath");
            this.dispatchLibraryDirectory = Objects.requireNonNull(
                    dispatchLibraryDirectory, "dispatchLibraryDirectory");
        }

        public Builder cacheDirectory(Path cacheDirectory) {
            this.cacheDirectory = cacheDirectory;
            return this;
        }

        public Builder systemMessage(String systemMessage) {
            this.systemMessage = systemMessage;
            return this;
        }

        public Builder contextTokens(int contextTokens) {
            if (contextTokens <= 0) {
                throw new IllegalArgumentException(
                        "context tokens must be positive");
            }
            this.contextTokens = contextTokens;
            return this;
        }

        public Builder maxOutputTokens(int maxOutputTokens) {
            if (maxOutputTokens <= 0) {
                throw new IllegalArgumentException(
                        "max output tokens must be positive");
            }
            this.maxOutputTokens = maxOutputTokens;
            return this;
        }

        public Builder sampler(int topK, float topP, float temperature,
                               int seed) {
            if (topK <= 0) {
                throw new IllegalArgumentException("topK must be positive");
            }
            if (!(topP > 0.0f && topP <= 1.0f)) {
                throw new IllegalArgumentException(
                        "topP must be in (0, 1]");
            }
            if (temperature < 0.0f) {
                throw new IllegalArgumentException(
                        "temperature must not be negative");
            }
            this.topK = topK;
            this.topP = topP;
            this.temperature = temperature;
            this.seed = seed;
            return this;
        }

        public Builder enableBenchmark(boolean enableBenchmark) {
            this.enableBenchmark = enableBenchmark;
            return this;
        }

        public SdxLiteRtLmChatSession build() {
            return new SdxLiteRtLmChatSession(this);
        }
    }

    public static final class NativeCallException extends RuntimeException {
        public NativeCallException(String message) {
            super(message);
        }
    }
}
