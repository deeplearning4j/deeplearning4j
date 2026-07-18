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
import java.util.Objects;

/**
 * Lifecycle-safe Java facade over the generated JavaCPP tokenizer binding.
 *
 * <p>This class is intentionally small. The Rust tokenizer remains the single
 * implementation of tokenization behavior; this facade only owns native handles,
 * copies results into Java values, and guarantees matching native free calls.</p>
 */
public final class NativeTokenizer implements AutoCloseable {

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
     * Returns the tokenizer vocabulary size.
     */
    public synchronized long vocabSize() {
        return api.getVocabSize(requireOpen());
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
