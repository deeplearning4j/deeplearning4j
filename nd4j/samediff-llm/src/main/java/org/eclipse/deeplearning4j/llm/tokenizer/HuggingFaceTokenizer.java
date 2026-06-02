/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.eclipse.deeplearning4j.llm.tokenizer;

import lombok.extern.slf4j.Slf4j;
import org.eclipse.deeplearning4j.llm.config.TokenizerConfig;
import org.nd4j.ggml.GGMLModelImport;
import org.nd4j.ggml.format.GGMLMetadata;

import java.io.File;
import java.util.*;

/**
 * HuggingFace tokenizer implementation using native Rust JNI bindings.
 *
 * <p>This tokenizer uses the HuggingFace tokenizers Rust library (v0.21) via JNI,
 * providing high-performance tokenization that is fully compatible with
 * the original Python implementation. This includes proper support for:</p>
 * <ul>
 *   <li>Byte-level BPE (used by GPT-2, Qwen, SmolDocling, etc.)</li>
 *   <li>SentencePiece/Unigram models (used by LLaMA, T5, etc.)</li>
 *   <li>WordPiece (used by BERT)</li>
 *   <li>All special token handling and chat templates</li>
 * </ul>
 *
 * <p><strong>IMPORTANT:</strong> This tokenizer requires the native tokenizers library
 * (nd4j-tokenizers module). Add the appropriate platform-specific dependency:</p>
 * <pre>{@code
 * <dependency>
 *     <groupId>org.eclipse.deeplearning4j</groupId>
 *     <artifactId>tokenizers-native</artifactId>
 *     <classifier>${javacpp.platform}</classifier>
 * </dependency>
 * }</pre>
 *
 * <p>Example usage:</p>
 * <pre>{@code
 * // Load from tokenizer.json file
 * HuggingFaceTokenizer tokenizer = HuggingFaceTokenizer.fromFile("path/to/tokenizer.json");
 *
 * // Encode text
 * Encoding encoding = tokenizer.encode("Hello, world!", true);
 * System.out.println("Token IDs: " + Arrays.toString(encoding.getIds()));
 *
 * // Decode back to text
 * String decoded = tokenizer.decode(encoding.getIds(), true);
 * System.out.println("Decoded: " + decoded);
 *
 * // Clean up
 * tokenizer.close();
 * }</pre>
 *
 * @author Eclipse Deeplearning4j Contributors
 */
@Slf4j
public class HuggingFaceTokenizer implements Tokenizer {

    // Flag indicating if native bindings are available
    private static final boolean NATIVE_AVAILABLE;
    private static final String NATIVE_VERSION;
    private static final String NATIVE_LOAD_ERROR;

    static {
        boolean nativeAvailable = false;
        String nativeVersion = "not available";
        String loadError = null;
        try {
            Class.forName("org.eclipse.deeplearning4j.tokenizers.bindings.TokenizersNative");
            nativeAvailable = true;
            nativeVersion = getNativeVersionInternal();
            log.info("Tokenizers native library loaded, version: {}", nativeVersion);
        } catch (ClassNotFoundException e) {
            loadError = "Native tokenizers library not found. Add nd4j-tokenizers dependency with platform classifier.";
            log.error(loadError);
        } catch (NoClassDefFoundError e) {
            loadError = "Native tokenizers library class definition error: " + e.getMessage();
            log.error(loadError, e);
        } catch (UnsatisfiedLinkError e) {
            loadError = "Native tokenizers library failed to load: " + e.getMessage() +
                       ". Ensure the native library is available for your platform.";
            log.error(loadError, e);
        }
        NATIVE_AVAILABLE = nativeAvailable;
        NATIVE_VERSION = nativeVersion;
        NATIVE_LOAD_ERROR = loadError;
    }

    // Delegate to native implementation
    private final NativeTokenizerImpl impl;
    private TokenizerConfig config;
    private volatile boolean closed = false;

    /**
     * Private constructor - use factory methods.
     */
    private HuggingFaceTokenizer(NativeTokenizerImpl impl, TokenizerConfig config) {
        this.impl = impl;
        this.config = config;
    }

    /**
     * Check if native tokenizer bindings are available.
     *
     * @return true if native bindings are loaded
     */
    public static boolean isNativeAvailable() {
        return NATIVE_AVAILABLE;
    }

    /**
     * Get the version of the native tokenizers library.
     *
     * @return version string, or "not available" if native not loaded
     */
    public static String getNativeVersion() {
        return NATIVE_VERSION;
    }

    /**
     * Ensure native library is available, throwing if not.
     */
    private static void requireNative() {
        if (!NATIVE_AVAILABLE) {
            throw new TokenizerException(
                "Native tokenizers library is required but not available. " +
                (NATIVE_LOAD_ERROR != null ? NATIVE_LOAD_ERROR :
                 "Add nd4j-tokenizers dependency with platform classifier to your project.")
            );
        }
    }

    private static String getNativeVersionInternal() {
        try {
            return NativeTokenizerImpl.getVersion();
        } catch (Exception e) {
            return "unknown";
        }
    }

    /**
     * Create a tokenizer from a tokenizer.json file.
     *
     * @param path path to the tokenizer.json file
     * @return the tokenizer instance
     * @throws TokenizerException if loading fails or native library not available
     */
    public static HuggingFaceTokenizer fromFile(String path) {
        return fromFile(new File(path));
    }

    /**
     * Create a tokenizer from a tokenizer.json file.
     *
     * <p>This method requires the native tokenizers library. The native library
     * provides full compatibility with HuggingFace tokenizers including byte-level
     * BPE support for models like Qwen, GPT-2, and SmolDocling.</p>
     *
     * @param file the tokenizer.json file
     * @return the tokenizer instance
     * @throws TokenizerException if loading fails or native library not available
     */
    public static HuggingFaceTokenizer fromFile(File file) {
        requireNative();

        if (!file.exists()) {
            throw new TokenizerException("Tokenizer file not found: " + file.getAbsolutePath());
        }

        // Try to load config from same directory. If tokenizer_config.json is
        // absent or incomplete, use GGUF sidecar tokenizer metadata as the
        // model-owned fallback for chat templates and special token strings.
        TokenizerConfig config = null;
        File parentDir = file.getParentFile();
        if (parentDir != null) {
            config = loadTokenizerConfig(parentDir);
            config = mergeGgufTokenizerMetadata(parentDir, config);
        }

        NativeTokenizerImpl impl = NativeTokenizerImpl.fromFile(file.getAbsolutePath());
        log.debug("Created native tokenizer from: {}", file.getAbsolutePath());

        return new HuggingFaceTokenizer(impl, config);
    }

    private static TokenizerConfig loadTokenizerConfig(File parentDir) {
        File configFile = new File(parentDir, "tokenizer_config.json");
        if (!configFile.exists()) {
            return null;
        }
        try {
            return TokenizerConfig.fromFile(configFile);
        } catch (Exception e) {
            log.warn("Could not load tokenizer_config.json: {}", e.getMessage());
            return null;
        }
    }

    private static TokenizerConfig mergeGgufTokenizerMetadata(File parentDir, TokenizerConfig config) {
        if (hasCompleteChatTemplateConfig(config)) {
            return config;
        }

        File ggufSidecar = findGgufSidecar(parentDir);
        if (ggufSidecar == null) {
            return config;
        }

        try {
            GGMLMetadata metadata = GGMLModelImport.inspectModel(ggufSidecar);
            TokenizerConfig ggufConfig = TokenizerConfig.fromGgufMetadata(metadata.getTokenizerInfo());
            if (ggufConfig == null) {
                return config;
            }
            if (config == null) {
                log.debug("Loaded tokenizer metadata from GGUF sidecar: {}", ggufSidecar.getAbsolutePath());
                return ggufConfig;
            }
            config.fillMissingFrom(ggufConfig);
            log.debug("Filled missing tokenizer metadata from GGUF sidecar: {}", ggufSidecar.getAbsolutePath());
            return config;
        } catch (Exception e) {
            log.warn("Could not load tokenizer metadata from GGUF sidecar {}: {}",
                    ggufSidecar.getAbsolutePath(), e.getMessage());
            return config;
        }
    }

    private static boolean hasCompleteChatTemplateConfig(TokenizerConfig config) {
        return config != null
                && config.hasChatTemplate()
                && !isBlank(config.getBosToken())
                && !isBlank(config.getEosToken());
    }

    private static boolean isBlank(String value) {
        return value == null || value.isBlank();
    }

    private static File findGgufSidecar(File parentDir) {
        File[] ggufs = parentDir.listFiles((dir, name) -> name.toLowerCase(Locale.ROOT).endsWith(".gguf"));
        if (ggufs == null || ggufs.length == 0) {
            return null;
        }
        Arrays.sort(ggufs, Comparator.comparing(File::getName));
        return ggufs[0];
    }

    /**
     * Create a tokenizer from a JSON string.
     *
     * <p>This method requires the native tokenizers library.</p>
     *
     * @param json the tokenizer JSON configuration
     * @return the tokenizer instance
     * @throws TokenizerException if parsing fails or native library not available
     */
    public static HuggingFaceTokenizer fromJson(String json) {
        requireNative();

        NativeTokenizerImpl impl = NativeTokenizerImpl.fromJson(json);
        log.debug("Created native tokenizer from JSON");

        return new HuggingFaceTokenizer(impl, null);
    }

    /**
     * Create a tokenizer from a model directory containing tokenizer.json.
     *
     * <p>This method requires the native tokenizers library.</p>
     *
     * @param modelDir the model directory
     * @return the tokenizer instance
     * @throws TokenizerException if loading fails or native library not available
     */
    public static HuggingFaceTokenizer fromDirectory(File modelDir) {
        File tokenizerFile = new File(modelDir, "tokenizer.json");
        if (!tokenizerFile.exists()) {
            throw new TokenizerException("tokenizer.json not found in: " + modelDir.getAbsolutePath());
        }
        return fromFile(tokenizerFile);
    }

    /**
     * Create a tokenizer from a model directory path.
     *
     * @param modelDirPath the model directory path
     * @return the tokenizer instance
     * @throws TokenizerException if loading fails or native library not available
     */
    public static HuggingFaceTokenizer fromDirectory(String modelDirPath) {
        return fromDirectory(new File(modelDirPath));
    }

    @Override
    public Encoding encode(String text, boolean addSpecialTokens) {
        checkNotClosed();
        return impl.encode(text, addSpecialTokens);
    }

    @Override
    public List<Encoding> encodeBatch(List<String> texts, boolean addSpecialTokens) {
        checkNotClosed();
        List<Encoding> results = new ArrayList<>(texts.size());
        for (String text : texts) {
            results.add(impl.encode(text, addSpecialTokens));
        }
        return results;
    }

    @Override
    public String decode(int[] ids, boolean skipSpecialTokens) {
        checkNotClosed();
        return impl.decode(ids, skipSpecialTokens);
    }

    @Override
    public List<String> decodeBatch(List<int[]> idsBatch, boolean skipSpecialTokens) {
        checkNotClosed();
        List<String> results = new ArrayList<>(idsBatch.size());
        for (int[] ids : idsBatch) {
            results.add(impl.decode(ids, skipSpecialTokens));
        }
        return results;
    }

    @Override
    public int getVocabSize() {
        checkNotClosed();
        return impl.getVocabSize();
    }

    @Override
    public Integer getTokenId(String token) {
        checkNotClosed();
        return impl.getTokenId(token);
    }

    @Override
    public String getToken(int id) {
        checkNotClosed();
        return impl.getToken(id);
    }

    @Override
    public Map<String, Integer> getVocab() {
        checkNotClosed();
        return impl.getVocab();
    }

    @Override
    public int getPadTokenId() {
        return impl.getPadTokenId();
    }

    @Override
    public int getBosTokenId() {
        return impl.getBosTokenId();
    }

    @Override
    public int getEosTokenId() {
        return impl.getEosTokenId();
    }

    @Override
    public int getUnkTokenId() {
        return impl.getUnkTokenId();
    }

    @Override
    public boolean isValid() {
        return !closed && impl.isValid();
    }

    /**
     * Check if this tokenizer is using native bindings.
     *
     * @return always true, as this implementation requires native bindings
     */
    public boolean isUsingNative() {
        return true;
    }

    /**
     * Get the tokenizer configuration.
     *
     * @return the tokenizer config, or null if not loaded
     */
    public TokenizerConfig getConfig() {
        return config;
    }

    @Override
    public String getChatTemplate() {
        return config != null ? config.getChatTemplate() : null;
    }

    @Override
    public String getBosToken() {
        String token = config != null ? config.getBosToken() : null;
        return token != null ? token : Tokenizer.super.getBosToken();
    }

    @Override
    public String getEosToken() {
        String token = config != null ? config.getEosToken() : null;
        return token != null ? token : Tokenizer.super.getEosToken();
    }

    @Override
    public void close() {
        if (!closed) {
            impl.close();
            closed = true;
        }
    }

    private void checkNotClosed() {
        if (closed) {
            throw new IllegalStateException("Tokenizer has been closed");
        }
    }

    // ========================================================================
    // Native implementation using Rust JNI bindings (HuggingFace tokenizers v0.21)
    // Uses direct JavaCPP calls - same pattern as kompile for reliability
    // ========================================================================

    private static class NativeTokenizerImpl {
        // Use direct imports via reflection to avoid compile-time dependency
        // but call methods directly once loaded (no reflection on each call)
        private static volatile boolean initialized = false;

        private final Object nativeLib;  // TokenizersNative instance
        private final Object nativeTokenizer;  // OpaqueTokenizer instance

        // Method handles for direct calls (cached on init)
        private static java.lang.reflect.Method encodeTextMethod;
        private static java.lang.reflect.Method encodingGetLengthMethod;
        private static java.lang.reflect.Method encodingGetIdsMethod;
        private static java.lang.reflect.Method freeEncodingMethod;
        private static java.lang.reflect.Method decodeIdsMethod;
        private static java.lang.reflect.Method getVocabSizeMethod;
        private static java.lang.reflect.Method tokenizerIsValidMethod;
        private static java.lang.reflect.Method freeTokenizerMethod;
        private static java.lang.reflect.Method getVersionMethod;
        private static java.lang.reflect.Method createFromFileMethod;
        private static java.lang.reflect.Method createFromJsonMethod;
        private static Class<?> opaqueTokenizerClass;
        private static Class<?> opaqueEncodingClass;

        private int padTokenId = -1;
        private int bosTokenId = -1;
        private int eosTokenId = -1;
        private int unkTokenId = -1;

        private static synchronized void initializeMethodHandles() throws Exception {
            if (initialized) return;

            Class<?> nativeClass = Class.forName("org.eclipse.deeplearning4j.tokenizers.bindings.TokenizersNative");
            opaqueTokenizerClass = Class.forName("org.eclipse.deeplearning4j.tokenizers.bindings.TokenizersNative$OpaqueTokenizer");
            opaqueEncodingClass = Class.forName("org.eclipse.deeplearning4j.tokenizers.bindings.TokenizersNative$OpaqueEncoding");
            Class<?> intPointerClass = Class.forName("org.bytedeco.javacpp.IntPointer");

            // Cache all method handles
            createFromFileMethod = nativeClass.getMethod("createTokenizerFromFile", String.class);
            createFromJsonMethod = nativeClass.getMethod("createTokenizerFromJson", String.class);
            encodeTextMethod = nativeClass.getMethod("encodeText", opaqueTokenizerClass, String.class, boolean.class);
            encodingGetLengthMethod = nativeClass.getMethod("encodingGetLength", opaqueEncodingClass);
            encodingGetIdsMethod = nativeClass.getMethod("encodingGetIds", opaqueEncodingClass);
            freeEncodingMethod = nativeClass.getMethod("freeEncoding", opaqueEncodingClass);
            decodeIdsMethod = nativeClass.getMethod("decodeIds", opaqueTokenizerClass, int[].class, long.class, boolean.class);
            getVocabSizeMethod = nativeClass.getMethod("getVocabSize", opaqueTokenizerClass);
            tokenizerIsValidMethod = nativeClass.getMethod("tokenizerIsValid", opaqueTokenizerClass);
            freeTokenizerMethod = nativeClass.getMethod("freeTokenizer", opaqueTokenizerClass);
            getVersionMethod = nativeClass.getMethod("getTokenizerVersion");

            initialized = true;
        }

        private NativeTokenizerImpl(Object nativeLib, Object nativeTokenizer) {
            this.nativeLib = nativeLib;
            this.nativeTokenizer = nativeTokenizer;
        }

        static NativeTokenizerImpl fromFile(String path) {
            try {
                initializeMethodHandles();

                // Create new TokenizersNative instance (like kompile does)
                Class<?> nativeClass = Class.forName("org.eclipse.deeplearning4j.tokenizers.bindings.TokenizersNative");
                Object nativeLib = nativeClass.getDeclaredConstructor().newInstance();

                // Call createTokenizerFromFile with String directly (not BytePointer)
                Object handle = createFromFileMethod.invoke(nativeLib, path);

                if (handle == null) {
                    throw new TokenizerException("Failed to create native tokenizer from file: " + path);
                }

                // Check if handle is null pointer
                java.lang.reflect.Method isNullMethod = handle.getClass().getMethod("isNull");
                if ((Boolean) isNullMethod.invoke(handle)) {
                    throw new TokenizerException("Failed to create native tokenizer from file (null handle): " + path);
                }

                NativeTokenizerImpl impl = new NativeTokenizerImpl(nativeLib, handle);
                impl.initializeSpecialTokens();
                return impl;
            } catch (TokenizerException e) {
                throw e;
            } catch (Exception e) {
                throw new TokenizerException("Failed to create native tokenizer: " + e.getMessage(), e);
            }
        }

        static NativeTokenizerImpl fromJson(String json) {
            try {
                initializeMethodHandles();

                // Create new TokenizersNative instance (like kompile does)
                Class<?> nativeClass = Class.forName("org.eclipse.deeplearning4j.tokenizers.bindings.TokenizersNative");
                Object nativeLib = nativeClass.getDeclaredConstructor().newInstance();

                // Call createTokenizerFromJson with String directly
                Object handle = createFromJsonMethod.invoke(nativeLib, json);

                if (handle == null) {
                    throw new TokenizerException("Failed to create native tokenizer from JSON");
                }

                // Check if handle is null pointer
                java.lang.reflect.Method isNullMethod = handle.getClass().getMethod("isNull");
                if ((Boolean) isNullMethod.invoke(handle)) {
                    throw new TokenizerException("Failed to create native tokenizer from JSON (null handle)");
                }

                NativeTokenizerImpl impl = new NativeTokenizerImpl(nativeLib, handle);
                impl.initializeSpecialTokens();
                return impl;
            } catch (TokenizerException e) {
                throw e;
            } catch (Exception e) {
                throw new TokenizerException("Failed to create native tokenizer: " + e.getMessage(), e);
            }
        }

        static String getVersion() {
            try {
                initializeMethodHandles();
                Class<?> nativeClass = Class.forName("org.eclipse.deeplearning4j.tokenizers.bindings.TokenizersNative");
                Object nativeLib = nativeClass.getDeclaredConstructor().newInstance();
                Object result = getVersionMethod.invoke(nativeLib);
                if (result != null) {
                    return result.toString();
                }
            } catch (Exception e) {
                // Ignore
            }
            return "unknown";
        }

        private void initializeSpecialTokens() {
            // Resolve special token IDs by encoding known special token strings.
            // The native tokenizer knows about added_tokens from tokenizer.json,
            // so encoding them (without adding extra special tokens) returns their IDs.
            bosTokenId = resolveSpecialToken("<|im_start|>");
            eosTokenId = resolveSpecialToken("<|im_end|>");
            padTokenId = resolveSpecialToken("<|im_end|>"); // Often same as EOS
            // Try common fallbacks if the above didn't resolve
            if (eosTokenId < 0) {
                eosTokenId = resolveSpecialToken("<|endoftext|>");
            }
            if (bosTokenId < 0) {
                bosTokenId = resolveSpecialToken("<s>");
            }
            if (eosTokenId < 0) {
                eosTokenId = resolveSpecialToken("</s>");
            }
        }

        private int resolveSpecialToken(String tokenStr) {
            // Strategy 1: encode without special token processing — native tokenizer
            // resolves added_tokens (like <|im_end|>) to their single IDs directly.
            try {
                Encoding enc = encode(tokenStr, false);
                int[] ids = enc.getIds();
                if (ids != null && ids.length == 1) {
                    return ids[0];
                }
            } catch (Exception e) {
                // Token not in vocabulary - try fallback
            }
            // Strategy 2: encode WITH special tokens enabled — some tokenizer configs
            // only resolve added_tokens when addSpecialTokens=true.
            try {
                Encoding enc = encode(tokenStr, true);
                int[] ids = enc.getIds();
                if (ids != null && ids.length == 1) {
                    return ids[0];
                }
            } catch (Exception e) {
                // Token not in vocabulary
            }
            return -1;
        }

        public Encoding encode(String text, boolean addSpecialTokens) {
            try {
                // Call encodeText directly with String (not BytePointer)
                Object encoding = encodeTextMethod.invoke(nativeLib, nativeTokenizer, text, addSpecialTokens);

                if (encoding == null) {
                    throw new TokenizerException("Failed to encode text");
                }

                // Check if encoding is null pointer
                java.lang.reflect.Method isNullMethod = encoding.getClass().getMethod("isNull");
                if ((Boolean) isNullMethod.invoke(encoding)) {
                    throw new TokenizerException("Failed to encode text (null encoding)");
                }

                try {
                    // Get length
                    long length = (Long) encodingGetLengthMethod.invoke(nativeLib, encoding);

                    // Get IDs via IntPointer
                    Object idsPtr = encodingGetIdsMethod.invoke(nativeLib, encoding);
                    int[] ids = new int[(int) length];
                    if (idsPtr != null && length > 0) {
                        java.lang.reflect.Method getMethod = idsPtr.getClass().getMethod("get", int[].class);
                        getMethod.invoke(idsPtr, ids);
                    }

                    // Attention mask - default to all 1s
                    int[] attentionMask = new int[(int) length];
                    Arrays.fill(attentionMask, 1);

                    return Encoding.builder()
                            .ids(ids)
                            .tokens(new String[(int) length])
                            .attentionMask(attentionMask)
                            .build();
                } finally {
                    // Free encoding - this is the only free we need to do
                    freeEncodingMethod.invoke(nativeLib, encoding);
                }
            } catch (TokenizerException e) {
                throw e;
            } catch (Exception e) {
                throw new TokenizerException("Failed to encode: " + e.getMessage(), e);
            }
        }

        public String decode(int[] ids, boolean skipSpecialTokens) {
            try {
                // Call decodeIds with int[] directly (like kompile does)
                // No need to create IntPointer - JavaCPP handles it
                Object result = decodeIdsMethod.invoke(nativeLib, nativeTokenizer, ids, (long) ids.length, skipSpecialTokens);

                if (result != null) {
                    return result.toString();
                }
                return "";
            } catch (Exception e) {
                throw new TokenizerException("Failed to decode: " + e.getMessage(), e);
            }
        }

        public int getVocabSize() {
            try {
                return ((Number) getVocabSizeMethod.invoke(nativeLib, nativeTokenizer)).intValue();
            } catch (Exception e) {
                return 0;
            }
        }

        public Integer getTokenId(String token) {
            // Resolve via encoding: if the token encodes to exactly one ID, return it
            try {
                Encoding enc = encode(token, false);
                int[] ids = enc.getIds();
                if (ids != null && ids.length == 1) {
                    return ids[0];
                }
            } catch (Exception e) {
                // Token not found
            }
            return null;
        }

        public String getToken(int id) {
            // Not in current native bindings
            return null;
        }

        public Map<String, Integer> getVocab() {
            return Collections.emptyMap();
        }

        public int getPadTokenId() { return padTokenId; }
        public int getBosTokenId() { return bosTokenId; }
        public int getEosTokenId() { return eosTokenId; }
        public int getUnkTokenId() { return unkTokenId; }

        public boolean isValid() {
            try {
                return (Boolean) tokenizerIsValidMethod.invoke(nativeLib, nativeTokenizer);
            } catch (Exception e) {
                return false;
            }
        }

        public void close() {
            if (nativeTokenizer != null) {
                try {
                    freeTokenizerMethod.invoke(nativeLib, nativeTokenizer);
                } catch (Exception e) {
                    // Ignore
                }
            }
        }
    }
}
