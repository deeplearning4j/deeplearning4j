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

import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.eclipse.deeplearning4j.llm.config.TokenizerConfig;
import org.eclipse.deeplearning4j.model.download.ModelDownloader;
import org.eclipse.deeplearning4j.tokenizers.NativeTokenizer;
import org.nd4j.shade.jackson.databind.JsonNode;
import org.nd4j.shade.jackson.databind.ObjectMapper;

import java.io.File;
import java.nio.file.Files;
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
    private static final ObjectMapper JSON_MAPPER = new ObjectMapper();

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
    @Getter private TokenizerConfig config;
    private final String tokenizerConfigJson;
    private final Map<String, Integer> addedTokenIdsByContent;
    private final Map<Integer, String> addedTokensById;
    private final Set<Integer> addedSpecialTokenIds;
    private volatile boolean closed = false;

    /**
     * Private constructor - use factory methods.
     */
    private HuggingFaceTokenizer(NativeTokenizerImpl impl, TokenizerConfig config,
                                 String tokenizerConfigJson, String tokenizerJson) {
        this.impl = impl;
        this.config = config;
        this.tokenizerConfigJson = tokenizerConfigJson;
        this.addedTokenIdsByContent = parseAddedTokenIds(tokenizerJson);
        Map<Integer, String> byId = new LinkedHashMap<>();
        this.addedTokenIdsByContent.forEach((content, id) -> byId.put(id, content));
        this.addedTokensById = Collections.unmodifiableMap(byId);
        this.addedSpecialTokenIds = parseSpecialTokenIds(tokenizerJson);
        this.impl.initializeSpecialTokens(config);
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

        TokenizerConfig config = null;
        String tokenizerConfigJson = null;
        File parentDir = file.getParentFile();
        if (parentDir != null) {
            tokenizerConfigJson = loadTokenizerConfigJson(parentDir);
            if (tokenizerConfigJson != null) {
                try {
                    config = TokenizerConfig.fromJson(tokenizerConfigJson);
                } catch (Exception e) {
                    throw new TokenizerException("Invalid tokenizer_config.json: " + e.getMessage(), e);
                }
            }
        }

        String tokenizerJson;
        try {
            tokenizerJson = Files.readString(file.toPath());
        } catch (Exception e) {
            throw new TokenizerException("Could not load tokenizer.json: " + e.getMessage(), e);
        }
        NativeTokenizerImpl impl = NativeTokenizerImpl.fromFile(file.getAbsolutePath());
        log.debug("Created native tokenizer from: {}", file.getAbsolutePath());

        return new HuggingFaceTokenizer(impl, config, tokenizerConfigJson, tokenizerJson);
    }

    private static String loadTokenizerConfigJson(File parentDir) {
        File configFile = new File(parentDir, "tokenizer_config.json");
        if (!configFile.exists()) {
            return null;
        }
        try {
            return Files.readString(configFile.toPath());
        } catch (Exception e) {
            throw new TokenizerException("Could not load tokenizer_config.json: " + e.getMessage(), e);
        }
    }

    /**
     * Parse all added tokens that tokenizer.json marks as special. This is the
     * tokenizer-owned protocol vocabulary; it must not be inferred from model
     * names or hard-coded token strings in generation code.
     */
    static Set<Integer> parseSpecialTokenIds(String tokenizerJson) {
        if (tokenizerJson == null || tokenizerJson.isBlank()) {
            return Collections.emptySet();
        }
        try {
            JsonNode addedTokens = JSON_MAPPER.readTree(tokenizerJson).path("added_tokens");
            if (!addedTokens.isArray()) {
                return Collections.emptySet();
            }
            Set<Integer> ids = new LinkedHashSet<>();
            for (JsonNode token : addedTokens) {
                JsonNode id = token.get("id");
                if (token.path("special").asBoolean(false)
                        && id != null && id.canConvertToInt() && id.asInt() >= 0) {
                    ids.add(id.asInt());
                }
            }
            return Collections.unmodifiableSet(ids);
        } catch (Exception e) {
            throw new TokenizerException("Invalid tokenizer.json special-token metadata: "
                    + e.getMessage(), e);
        }
    }

    /** Retain every tokenizer-declared added token, including decodable protocol delimiters. */
    static Map<String, Integer> parseAddedTokenIds(String tokenizerJson) {
        if (tokenizerJson == null || tokenizerJson.isBlank()) {
            return Collections.emptyMap();
        }
        try {
            JsonNode addedTokens = JSON_MAPPER.readTree(tokenizerJson).path("added_tokens");
            if (!addedTokens.isArray()) {
                return Collections.emptyMap();
            }
            Map<String, Integer> byContent = new LinkedHashMap<>();
            Map<Integer, String> byId = new LinkedHashMap<>();
            for (JsonNode token : addedTokens) {
                JsonNode idNode = token.get("id");
                JsonNode contentNode = token.get("content");
                if (idNode == null || !idNode.canConvertToInt() || idNode.asInt() < 0
                        || contentNode == null || !contentNode.isTextual()) {
                    continue;
                }
                int id = idNode.asInt();
                String content = contentNode.asText();
                Integer priorId = byContent.putIfAbsent(content, id);
                String priorContent = byId.putIfAbsent(id, content);
                if ((priorId != null && priorId != id)
                        || (priorContent != null && !priorContent.equals(content))) {
                    throw new TokenizerException(
                            "Conflicting tokenizer.json added-token metadata for id=" + id
                                    + " content=" + content);
                }
            }
            return Collections.unmodifiableMap(byContent);
        } catch (TokenizerException e) {
            throw e;
        } catch (Exception e) {
            throw new TokenizerException("Invalid tokenizer.json added-token metadata: "
                    + e.getMessage(), e);
        }
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

        return new HuggingFaceTokenizer(impl, null, null, json);
    }

    /**
     * Create a tokenizer from tokenizer.json plus its complete model-owned
     * tokenizer_config.json. This is used by importers that reconstruct tokenizer
     * assets from a container such as GGUF.
     */
    public static HuggingFaceTokenizer fromJson(String tokenizerJson,
                                                String tokenizerConfigJson) {
        requireNative();
        TokenizerConfig config = null;
        if (tokenizerConfigJson != null && !tokenizerConfigJson.isBlank()) {
            try {
                config = TokenizerConfig.fromJson(tokenizerConfigJson);
            } catch (Exception e) {
                throw new TokenizerException("Invalid tokenizer_config.json: " + e.getMessage(), e);
            }
        }
        NativeTokenizerImpl impl = NativeTokenizerImpl.fromJson(tokenizerJson);
        log.debug("Created native tokenizer from JSON with model chat configuration");
        return new HuggingFaceTokenizer(impl, config, tokenizerConfigJson, tokenizerJson);
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

    /**
     * Download and load a tokenizer from a Hugging Face model repository.
     * Files are cached under {@code cacheRoot/repositoryId} and reused on later calls.
     */
    public static HuggingFaceTokenizer fromPretrained(String repositoryId, File cacheRoot) {
        if (repositoryId == null || !repositoryId.matches("[A-Za-z0-9._-]+/[A-Za-z0-9._-]+")) {
            throw new TokenizerException("Invalid Hugging Face repository id: " + repositoryId);
        }
        if (cacheRoot == null) {
            throw new TokenizerException("Tokenizer cache root is required");
        }
        File repositoryCache = new File(cacheRoot, repositoryId);
        String baseUrl = "https://huggingface.co/" + repositoryId + "/resolve/main/";
        try {
            ModelDownloader.download(baseUrl + "tokenizer.json", "tokenizer.json", repositoryCache);
            try {
                ModelDownloader.download(baseUrl + "tokenizer_config.json", "tokenizer_config.json", repositoryCache);
            } catch (java.io.IOException e) {
                log.warn("No tokenizer_config.json available for {}: {}", repositoryId, e.getMessage());
            }
            return fromDirectory(repositoryCache);
        } catch (java.io.IOException e) {
            throw new TokenizerException("Failed to download tokenizer from " + repositoryId, e);
        }
    }

    /** Load a repository tokenizer from the standard DL4J user cache. */
    public static HuggingFaceTokenizer fromPretrained(String repositoryId) {
        return fromPretrained(repositoryId,
                new File(System.getProperty("user.home"), ".cache/dl4j/tokenizers"));
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
        Integer addedId = addedTokenIdsByContent.get(token);
        if (addedId != null) {
            return addedId;
        }
        return impl.getTokenId(token);
    }

    @Override
    public String getToken(int id) {
        checkNotClosed();
        String addedToken = addedTokensById.get(id);
        if (addedToken != null) {
            return addedToken;
        }
        return impl.getToken(id);
    }

    @Override
    public Map<String, Integer> getVocab() {
        checkNotClosed();
        Map<String, Integer> nativeVocab = impl.getVocab();
        if (nativeVocab == null || nativeVocab.isEmpty()) {
            return addedTokenIdsByContent;
        }
        Map<String, Integer> merged = new LinkedHashMap<>(nativeVocab);
        merged.putAll(addedTokenIdsByContent);
        return Collections.unmodifiableMap(merged);
    }

    @Override
    public Map<String, Integer> getAddedTokens() {
        return addedTokenIdsByContent;
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
    public Set<Integer> getSpecialTokenIds() {
        Set<Integer> ids = new LinkedHashSet<>(addedSpecialTokenIds);
        ids.addAll(Tokenizer.super.getSpecialTokenIds());
        return Collections.unmodifiableSet(ids);
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

    @Override
    public String getChatTemplate() {
        return config != null ? config.getChatTemplate() : null;
    }

    @Override
    public String applyChatTemplate(List<ChatTemplate.Message> messages,
                                    boolean addGenerationPrompt) {
        return applyChatTemplate(ChatTemplate.Request.builder()
                .messages(messages)
                .addGenerationPrompt(addGenerationPrompt)
                .build(), null);
    }

    @Override
    public String applyChatTemplate(ChatTemplate.Request request,
                                    String chatTemplateOverride) {
        checkNotClosed();
        if (chatTemplateOverride != null && !chatTemplateOverride.isBlank()) {
            return Tokenizer.super.applyChatTemplate(request, chatTemplateOverride);
        }
        if (tokenizerConfigJson == null || tokenizerConfigJson.isBlank()
                || config == null || !config.hasChatTemplate()) {
            throw new IllegalStateException(
                    "Tokenizer import does not provide tokenizer_config.json with a chat_template");
        }
        return NativeTokenizer.renderChatTemplateContext(
                tokenizerConfigJson, ChatTemplate.requestContextJson(request));
    }

    @Override
    public String applyChatTemplateContext(String contextJson) {
        checkNotClosed();
        if (tokenizerConfigJson == null || tokenizerConfigJson.isBlank()
                || config == null || !config.hasChatTemplate()) {
            throw new IllegalStateException(
                    "Tokenizer import does not provide tokenizer_config.json with a chat_template");
        }
        return NativeTokenizer.renderChatTemplateContext(tokenizerConfigJson, contextJson);
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
        private static java.lang.reflect.Method encodingGetTokensMethod;
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

        private @Getter int padTokenId = -1;
        private @Getter int bosTokenId = -1;
        private @Getter int eosTokenId = -1;
        private @Getter int unkTokenId = -1;

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
            encodingGetTokensMethod = nativeClass.getMethod("encodingGetTokens", opaqueEncodingClass);
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

                return new NativeTokenizerImpl(nativeLib, handle);
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

                return new NativeTokenizerImpl(nativeLib, handle);
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

        private void initializeSpecialTokens(TokenizerConfig config) {
            if (config == null) return;
            bosTokenId = resolveSpecialToken(config.getBosToken());
            eosTokenId = resolveSpecialToken(config.getEosToken());
            padTokenId = resolveSpecialToken(config.getPadToken());
            unkTokenId = resolveSpecialToken(config.getUnkToken());
        }

        private int resolveSpecialToken(String tokenStr) {
            if (tokenStr == null || tokenStr.isEmpty()) return -1;
            try {
                Encoding enc = encode(tokenStr, false);
                int[] ids = enc.getIds();
                if (ids != null && ids.length == 1) {
                    return ids[0];
                }
            } catch (Exception e) {
                log.debug("Configured special token is not resolvable: {}", tokenStr);
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

                    // Token STRINGS via encoding_get_tokens (const char**). Previously
                    // this array was left as nulls even though the native binding
                    // exposes the strings — Encoding.getTokens() returned [null, ...].
                    // Token strings are informational; never fail an encode over them.
                    String[] tokens = new String[(int) length];
                    try {
                        Object tokensPtr = encodingGetTokensMethod.invoke(nativeLib, encoding);
                        if (tokensPtr != null && length > 0) {
                            java.lang.reflect.Method isNullPtr = tokensPtr.getClass().getMethod("isNull");
                            if (!(Boolean) isNullPtr.invoke(tokensPtr)) {
                                java.lang.reflect.Method getStringMethod =
                                        tokensPtr.getClass().getMethod("getString", long.class);
                                for (int i = 0; i < (int) length; i++) {
                                    tokens[i] = (String) getStringMethod.invoke(tokensPtr, (long) i);
                                }
                            }
                        }
                    } catch (Exception tokensError) {
                        log.debug("encoding_get_tokens unavailable, token strings left null: {}",
                                tokensError.getMessage());
                    }

                    // Attention mask - default to all 1s
                    int[] attentionMask = new int[(int) length];
                    Arrays.fill(attentionMask, 1);

                    return Encoding.builder()
                            .ids(ids)
                            .tokens(tokens)
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
