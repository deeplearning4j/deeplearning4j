/*
 *  ******************************************************************************
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  ******************************************************************************
 */

package org.eclipse.deeplearning4j.tokenizers;

import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.IntPointer;
import org.eclipse.deeplearning4j.tokenizers.bindings.TokenizersNative;
import org.eclipse.deeplearning4j.tokenizers.bindings.TokenizersNative.OpaqueDecodeStream;
import org.eclipse.deeplearning4j.tokenizers.bindings.TokenizersNative.OpaqueEncoding;
import org.eclipse.deeplearning4j.tokenizers.bindings.TokenizersNative.OpaqueTokenizer;
import org.eclipse.deeplearning4j.tokenizers.bindings.TokenizersNative.TokenizerResult;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.IntBuffer;
import java.time.LocalDate;
import java.time.ZoneOffset;
import java.time.format.DateTimeFormatter;
import java.util.List;
import java.util.Locale;
import java.util.Objects;

/**
 * Lifecycle-safe Java facade over the generated JavaCPP tokenizer binding.
 *
 * <p>This class is intentionally small. The Rust tokenizer remains the single
 * implementation of tokenization behavior; this facade only owns native handles,
 * copies results into Java values, and guarantees matching native free calls.</p>
 */
public final class NativeTokenizer implements AutoCloseable {

    private static final DateTimeFormatter CHAT_DATE_FORMAT =
            DateTimeFormatter.ofPattern("dd MMM yyyy", Locale.ENGLISH);

    private final TokenizersNative api;
    private OpaqueTokenizer handle;

    private NativeTokenizer(TokenizersNative api, OpaqueTokenizer handle) {
        this.api = api;
        this.handle = handle;
    }

    /**
     * Loads a tokenizer from a local tokenizer JSON file.
     */
    public static NativeTokenizer fromFile(String tokenizerPath) {
        Objects.requireNonNull(tokenizerPath, "tokenizerPath");
        if (tokenizerPath.isEmpty()) {
            throw new IllegalArgumentException("tokenizerPath must not be empty");
        }
        TokenizersNative api = new TokenizersNative();
        return own(api, api.createTokenizerFromFile(tokenizerPath),
                "load tokenizer from " + tokenizerPath);
    }

    /**
     * Loads a tokenizer from a tokenizer JSON document already in memory.
     */
    public static NativeTokenizer fromJson(String tokenizerJson) {
        Objects.requireNonNull(tokenizerJson, "tokenizerJson");
        if (tokenizerJson.isEmpty()) {
            throw new IllegalArgumentException("tokenizerJson must not be empty");
        }
        TokenizersNative api = new TokenizersNative();
        return own(api, api.createTokenizerFromJson(tokenizerJson),
                "load tokenizer JSON");
    }

    private static NativeTokenizer own(TokenizersNative api,
                                       OpaqueTokenizer candidate,
                                       String operation) {
        if (isNull(candidate)) {
            throw failure(api, operation);
        }
        if (!api.tokenizerIsValid(candidate)) {
            api.freeTokenizer(candidate);
            throw failure(api, operation);
        }
        return new NativeTokenizer(api, candidate);
    }

    /**
     * Encodes text and returns a Java-owned copy of the token IDs.
     */
    public synchronized int[] encode(String text, boolean addSpecialTokens) {
        Objects.requireNonNull(text, "text");
        OpaqueTokenizer tokenizer = requireOpen();
        OpaqueEncoding encoding =
                api.encodeText(tokenizer, text, addSpecialTokens);
        if (isNull(encoding)) {
            throw failure(api, "encode text");
        }

        try {
            long length = api.encodingGetLength(encoding);
            if (length > Integer.MAX_VALUE) {
                throw new IllegalStateException(
                        "Tokenizer returned too many token IDs: " + length);
            }
            if (length == 0) {
                return new int[0];
            }

            IntPointer nativeIds = api.encodingGetIds(encoding);
            if (isNull(nativeIds)) {
                throw failure(api, "read encoded token IDs");
            }
            int[] ids = new int[(int) length];
            nativeIds.position(0).get(ids);
            return ids;
        } finally {
            api.freeEncoding(encoding);
        }
    }

    /**
     * Encodes directly to the int64 token representation consumed by SDX
     * generation graphs. Rust token IDs are unsigned 32-bit values, so this
     * adapter preserves their full range without app-owned conversion loops.
     */
    public long[] encodeLong(String text, boolean addSpecialTokens) {
        int[] tokenIds = encode(text, addSpecialTokens);
        long[] result = new long[tokenIds.length];
        for (int i = 0; i < tokenIds.length; i++) {
            result[i] = Integer.toUnsignedLong(tokenIds[i]);
        }
        return result;
    }

    /**
     * Renders the chat template carried by a complete tokenizer_config.json.
     *
     * <p>The native tokenizer stack owns Jinja compatibility as well as token
     * encoding. Keeping both behind this facade prevents Android and desktop
     * consumers from drifting into model-specific prompt implementations.</p>
     *
     * @param tokenizerConfigJson full tokenizer_config.json document
     * @param messages ordered chat messages
     * @param addGenerationPrompt whether to append the assistant generation turn
     * @return the exact prompt text to encode with special-token insertion disabled
     */
    public synchronized String applyChatTemplate(
            String tokenizerConfigJson,
            List<ChatMessage> messages,
            boolean addGenerationPrompt) {
        requireOpen();
        return renderChatTemplate(api, tokenizerConfigJson, messages, addGenerationPrompt);
    }

    /**
     * Renders the model-owned chat template from the complete Hugging Face
     * runtime context. The context is an object containing {@code messages} and
     * may also contain {@code tools}, {@code documents}, generation flags, and
     * model-specific template arguments.
     */
    public synchronized String applyChatTemplateContext(
            String tokenizerConfigJson,
            String contextJson) {
        requireOpen();
        return renderChatTemplateContext(api, tokenizerConfigJson, contextJson);
    }

    /**
     * Renders a Hugging Face chat template without creating an otherwise-unused
     * tokenizer handle. Template evaluation only consumes tokenizer_config.json;
     * keeping that fact explicit lets SDX share the native MiniJinja implementation
     * while its generation tokenizer remains owned by the LLM pipeline.
     */
    public static String renderChatTemplate(
            String tokenizerConfigJson,
            List<ChatMessage> messages,
            boolean addGenerationPrompt) {
        return renderChatTemplate(
                new TokenizersNative(), tokenizerConfigJson, messages, addGenerationPrompt);
    }

    /**
     * Renders a complete Hugging Face chat-template runtime context without
     * creating a tokenizer handle.
     */
    public static String renderChatTemplateContext(
            String tokenizerConfigJson,
            String contextJson) {
        return renderChatTemplateContext(
                new TokenizersNative(), tokenizerConfigJson, contextJson);
    }

    private static String renderChatTemplate(
            TokenizersNative api,
            String tokenizerConfigJson,
            List<ChatMessage> messages,
            boolean addGenerationPrompt) {
        Objects.requireNonNull(tokenizerConfigJson, "tokenizerConfigJson");
        Objects.requireNonNull(messages, "messages");
        if (tokenizerConfigJson.isBlank()) {
            throw new IllegalArgumentException(
                    "tokenizerConfigJson must not be blank");
        }
        if (messages.isEmpty()) {
            throw new IllegalArgumentException(
                    "messages must contain at least one chat turn");
        }

        String messagesJson = messagesJson(messages);
        String currentDate = LocalDate.now(ZoneOffset.UTC)
                .format(CHAT_DATE_FORMAT);
        try (BytePointer nativeMessages = new BytePointer(messagesJson);
             BytePointer nativeConfig = new BytePointer(tokenizerConfigJson);
             BytePointer nativeDate = new BytePointer(currentDate)) {
            BytePointer rendered = api.applyChatTemplate(
                    nativeMessages,
                    nativeConfig,
                    nativeDate,
                    addGenerationPrompt);
            if (isNull(rendered)) {
                throw failure(api, "render tokenizer chat template");
            }
            try {
                return rendered.getString();
            } finally {
                api.freeString(rendered);
            }
        }
    }

    private static String renderChatTemplateContext(
            TokenizersNative api,
            String tokenizerConfigJson,
            String contextJson) {
        Objects.requireNonNull(tokenizerConfigJson, "tokenizerConfigJson");
        Objects.requireNonNull(contextJson, "contextJson");
        if (tokenizerConfigJson.isBlank()) {
            throw new IllegalArgumentException(
                    "tokenizerConfigJson must not be blank");
        }
        if (contextJson.isBlank()) {
            throw new IllegalArgumentException(
                    "contextJson must not be blank");
        }

        String currentDate = LocalDate.now(ZoneOffset.UTC)
                .format(CHAT_DATE_FORMAT);
        try (BytePointer nativeContext = new BytePointer(contextJson);
             BytePointer nativeConfig = new BytePointer(tokenizerConfigJson);
             BytePointer nativeDate = new BytePointer(currentDate)) {
            BytePointer rendered = api.applyChatTemplateContext(
                    nativeContext,
                    nativeConfig,
                    nativeDate);
            if (isNull(rendered)) {
                throw failure(api, "render tokenizer chat template context");
            }
            try {
                return rendered.getString();
            } finally {
                api.freeString(rendered);
            }
        }
    }

    /**
     * Decodes token IDs and returns a Java-owned string.
     */
    public synchronized String decode(int[] tokenIds,
                                      boolean skipSpecialTokens) {
        Objects.requireNonNull(tokenIds, "tokenIds");
        OpaqueTokenizer tokenizer = requireOpen();
        if (tokenIds.length == 0) {
            return "";
        }

        IntBuffer nativeIds = ByteBuffer
                .allocateDirect(Math.multiplyExact(tokenIds.length, Integer.BYTES))
                .order(ByteOrder.nativeOrder())
                .asIntBuffer();
        nativeIds.put(tokenIds);
        nativeIds.flip();

        BytePointer decoded = api.decodeIds(
                tokenizer, nativeIds, tokenIds.length, skipSpecialTokens);
        if (isNull(decoded)) {
            throw failure(api, "decode token IDs");
        }
        try {
            return decoded.getString();
        } finally {
            api.freeString(decoded);
        }
    }

    /**
     * Decodes SDX int64 token IDs through the same native tokenizer.
     */
    public String decode(long[] tokenIds, boolean skipSpecialTokens) {
        Objects.requireNonNull(tokenIds, "tokenIds");
        int[] nativeIds = new int[tokenIds.length];
        for (int i = 0; i < tokenIds.length; i++) {
            long tokenId = tokenIds[i];
            if (tokenId < 0 || tokenId > 0xffff_ffffL) {
                throw new IllegalArgumentException(
                        "tokenIds[" + i + "] is outside the unsigned 32-bit range: "
                                + tokenId);
            }
            nativeIds[i] = (int) tokenId;
        }
        return decode(nativeIds, skipSpecialTokens);
    }

    /**
     * Creates a stateful incremental decoder backed by the tokenizer library's
     * native DecodeStream implementation.
     *
     * <p>The returned decoder owns a native reference to the tokenizer and may
     * therefore outlive this facade. Close it independently when generation
     * finishes.</p>
     */
    public synchronized DecodeStream newDecodeStream(
            boolean skipSpecialTokens) {
        OpaqueDecodeStream stream =
                api.createDecodeStream(requireOpen(), skipSpecialTokens);
        if (isNull(stream)) {
            throw failure(api, "create decode stream");
        }
        return new DecodeStream(api, stream);
    }

    /**
     * Lifecycle-safe facade over the native incremental decoder.
     */
    public static final class DecodeStream implements AutoCloseable {

        private final TokenizersNative api;
        private OpaqueDecodeStream handle;

        private DecodeStream(TokenizersNative api, OpaqueDecodeStream handle) {
            this.api = api;
            this.handle = handle;
        }

        /**
         * Decodes one SDX token. An empty string means the tokenizer needs more
         * tokens before it can emit a complete text chunk.
         */
        public synchronized String step(long tokenId) {
            if (tokenId < 0 || tokenId > 0xffff_ffffL) {
                throw new IllegalArgumentException(
                        "tokenId is outside the unsigned 32-bit range: "
                                + tokenId);
            }
            if (isNull(handle)) {
                throw new IllegalStateException("Decode stream is closed");
            }
            String chunk = api.decodeStreamStep(handle, (int) tokenId);
            if (chunk == null) {
                throw failure(api, "decode stream token");
            }
            return chunk;
        }

        @Override
        public synchronized void close() {
            if (!isNull(handle)) {
                api.freeDecodeStream(handle);
                handle = null;
            }
        }
    }

    /**
     * Dependency-free message value accepted by the native chat renderer.
     */
    public static final class ChatMessage {
        private final String role;
        private final String content;

        public ChatMessage(String role, String content) {
            this.role = Objects.requireNonNull(role, "role");
            this.content = Objects.requireNonNull(content, "content");
            if (role.isBlank()) {
                throw new IllegalArgumentException("role must not be blank");
            }
        }

        public String role() {
            return role;
        }

        public String content() {
            return content;
        }

        public static ChatMessage system(String content) {
            return new ChatMessage("system", content);
        }

        public static ChatMessage user(String content) {
            return new ChatMessage("user", content);
        }

        public static ChatMessage assistant(String content) {
            return new ChatMessage("assistant", content);
        }

        public static ChatMessage tool(String content) {
            return new ChatMessage("tool", content);
        }
    }

    /**
     * Returns the tokenizer vocabulary size.
     */
    public synchronized long vocabSize() {
        return api.getVocabSize(requireOpen());
    }

    /**
     * Resolves an exact tokenizer vocabulary entry without applying normalization,
     * pre-tokenization, or unknown-token substitution.
     *
     * @return the token ID, or {@code -1} when the exact token is absent
     */
    public synchronized int tokenToId(String token) {
        Objects.requireNonNull(token, "token");
        int[] tokenId = new int[1];
        return api.getTokenId(requireOpen(), token, tokenId) ? tokenId[0] : -1;
    }

    /**
     * Returns whether this wrapper still owns a valid native tokenizer.
     */
    public synchronized boolean isValid() {
        return !isNull(handle) && api.tokenizerIsValid(handle);
    }

    public String version() {
        return api.getTokenizerVersion();
    }

    public String buildInfo() {
        return api.getBuildInfo();
    }

    /**
     * Frees the native tokenizer exactly once.
     */
    @Override
    public synchronized void close() {
        if (!isNull(handle)) {
            api.freeTokenizer(handle);
            handle = null;
        }
    }

    private OpaqueTokenizer requireOpen() {
        if (isNull(handle)) {
            throw new IllegalStateException("Tokenizer is closed");
        }
        return handle;
    }

    private static String messagesJson(List<ChatMessage> messages) {
        StringBuilder json = new StringBuilder();
        json.append('[');
        for (int i = 0; i < messages.size(); i++) {
            ChatMessage message = Objects.requireNonNull(
                    messages.get(i), "messages[" + i + "]");
            if (i > 0) {
                json.append(',');
            }
            json.append("{\"role\":");
            appendJsonString(json, message.role());
            json.append(",\"content\":");
            appendJsonString(json, message.content());
            json.append('}');
        }
        json.append(']');
        return json.toString();
    }

    private static void appendJsonString(StringBuilder output, String value) {
        output.append('"');
        for (int i = 0; i < value.length(); i++) {
            char character = value.charAt(i);
            switch (character) {
                case '"':
                    output.append("\\\"");
                    break;
                case '\\':
                    output.append("\\\\");
                    break;
                case '\b':
                    output.append("\\b");
                    break;
                case '\f':
                    output.append("\\f");
                    break;
                case '\n':
                    output.append("\\n");
                    break;
                case '\r':
                    output.append("\\r");
                    break;
                case '\t':
                    output.append("\\t");
                    break;
                default:
                    if (character < 0x20) {
                        output.append("\\u");
                        String hex = Integer.toHexString(character);
                        for (int padding = hex.length(); padding < 4; padding++) {
                            output.append('0');
                        }
                        output.append(hex);
                    } else {
                        output.append(character);
                    }
            }
        }
        output.append('"');
    }

    private static boolean isNull(org.bytedeco.javacpp.Pointer pointer) {
        return pointer == null || pointer.isNull();
    }

    private static IllegalStateException failure(TokenizersNative api,
                                                 String operation) {
        TokenizerResult result = api.getLastError();
        if (isNull(result)) {
            return new IllegalStateException(operation + " failed");
        }
        try {
            String message = result.error_message();
            String suffix = message == null || message.isEmpty()
                    ? ""
                    : ": " + message;
            return new IllegalStateException(
                    operation + " failed (tokenizer error "
                            + result.error_code() + ")" + suffix);
        } finally {
            result.close();
        }
    }
}
