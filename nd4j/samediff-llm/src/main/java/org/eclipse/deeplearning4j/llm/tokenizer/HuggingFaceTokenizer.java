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
import org.nd4j.shade.jackson.databind.node.ObjectNode;

import java.io.File;
import java.nio.charset.StandardCharsets;
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
            // Load through the one JavaCPP facade used by every runtime.
            // This keeps UTF-8 BytePointer overload selection compile-time checked.
            nativeVersion = getNativeVersionInternal();
            nativeAvailable = true;
            log.info("Tokenizers native library loaded, version: {}", nativeVersion);
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

    // The single model-owned JavaCPP facade used by desktop and mobile.
    private final NativeTokenizer impl;
    @Getter private TokenizerConfig config;
    private final String tokenizerConfigJson;
    private final Map<String, Integer> addedTokenIdsByContent;
    private final Map<Integer, String> addedTokensById;
    private final Set<Integer> addedSpecialTokenIds;
    private int padTokenId = -1;
    private int bosTokenId = -1;
    private int eosTokenId = -1;
    private int unkTokenId = -1;
    private volatile boolean closed = false;

    /**
     * Private constructor - use factory methods.
     */
    private HuggingFaceTokenizer(NativeTokenizer impl, TokenizerConfig config,
                                 String tokenizerConfigJson, String tokenizerJson) {
        this.impl = impl;
        this.config = config;
        this.tokenizerConfigJson = tokenizerConfigJson;
        this.addedTokenIdsByContent = parseAddedTokenIds(tokenizerJson);
        Map<Integer, String> byId = new LinkedHashMap<>();
        this.addedTokenIdsByContent.forEach((content, id) -> byId.put(id, content));
        this.addedTokensById = Collections.unmodifiableMap(byId);
        this.addedSpecialTokenIds = parseSpecialTokenIds(tokenizerJson);
        initializeSpecialTokens(config);
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
        return NativeTokenizer.nativeVersion();
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
        NativeTokenizer impl = NativeTokenizer.fromFile(file.getAbsolutePath());
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

        NativeTokenizer impl = NativeTokenizer.fromJson(json);
        log.debug("Created native tokenizer from JSON");

        return new HuggingFaceTokenizer(impl, null, null, json);
    }

    /**
     * Create a tokenizer from the canonical Hugging Face tokenizer.json and its
     * complete model-owned tokenizer_config.json. Callers that need chat behavior
     * must supply both files from the same model revision; this overload does not
     * synthesize tokenizer metadata.
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
        NativeTokenizer impl = NativeTokenizer.fromJson(tokenizerJson);
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
            // Hugging Face's tokenizer_config.json is part of the tokenizer contract for
            // chat models. Do not silently continue without it: the permissive path can
            // instantiate a tokenizer that encodes bytes but cannot honor the model's
            // special-token/chat-template configuration.
            ModelDownloader.download(baseUrl + "tokenizer_config.json", "tokenizer_config.json", repositoryCache);
            return fromDirectory(repositoryCache);
        } catch (java.io.IOException e) {
            throw new TokenizerException("Failed to download the complete Hugging Face tokenizer from "
                    + repositoryId + " (tokenizer.json and tokenizer_config.json are required)", e);
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
        NativeTokenizer.EncodedText encoded = impl.encodeWithTokens(text, addSpecialTokens);
        int[] attentionMask = new int[encoded.ids().length];
        Arrays.fill(attentionMask, 1);
        return Encoding.builder()
                .ids(encoded.ids())
                .tokens(encoded.tokens())
                .attentionMask(attentionMask)
                .build();
    }

    /**
     * Encodes directly to the unsigned int64 token representation consumed by
     * SDX, through the same model-owned tokenizer used by desktop callers.
     */
    public long[] encodeLong(String text, boolean addSpecialTokens) {
        checkNotClosed();
        return impl.encodeLong(text, addSpecialTokens);
    }

    @Override
    public List<Encoding> encodeBatch(List<String> texts, boolean addSpecialTokens) {
        checkNotClosed();
        List<Encoding> results = new ArrayList<>(texts.size());
        for (String text : texts) {
            results.add(encode(text, addSpecialTokens));
        }
        return results;
    }

    @Override
    public String decode(int[] ids, boolean skipSpecialTokens) {
        checkNotClosed();
        return impl.decode(ids, skipSpecialTokens);
    }

    /** Decode SDX int64 token IDs through this same tokenizer handle. */
    public String decode(long[] ids, boolean skipSpecialTokens) {
        checkNotClosed();
        return impl.decode(ids, skipSpecialTokens);
    }

    /** Creates a stateful decoder owned by this tokenizer implementation. */
    public DecodeStream newDecodeStream(boolean skipSpecialTokens) {
        checkNotClosed();
        return new DecodeStream(impl.newDecodeStream(skipSpecialTokens));
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
        long size = impl.vocabSize();
        if (size < 0 || size > Integer.MAX_VALUE) {
            throw new TokenizerException("Native tokenizer returned an invalid vocabulary size: " + size);
        }
        return (int) size;
    }

    @Override
    public Integer getTokenId(String token) {
        checkNotClosed();
        Integer addedId = addedTokenIdsByContent.get(token);
        if (addedId != null) {
            return addedId;
        }
        int nativeId = impl.tokenToId(token);
        return nativeId < 0 ? null : nativeId;
    }

    @Override
    public String getToken(int id) {
        checkNotClosed();
        String addedToken = addedTokensById.get(id);
        if (addedToken != null) {
            return addedToken;
        }
        return impl.idToToken(id);
    }

    @Override
    public Map<String, Integer> getVocab() {
        checkNotClosed();
        if (addedTokenIdsByContent.isEmpty()) {
            throw new UnsupportedOperationException(
                    "The generated tokenizer binding does not expose vocabulary enumeration");
        }
        return addedTokenIdsByContent;
    }

    @Override
    public Map<String, Integer> getAddedTokens() {
        return addedTokenIdsByContent;
    }

    @Override
    public int getPadTokenId() {
        return padTokenId;
    }

    @Override
    public int getBosTokenId() {
        return bosTokenId;
    }

    @Override
    public int getEosTokenId() {
        return eosTokenId;
    }

    @Override
    public int getUnkTokenId() {
        return unkTokenId;
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
        String renderConfig = tokenizerConfigJson;
        if (chatTemplateOverride != null && !chatTemplateOverride.isBlank()) {
            renderConfig = tokenizerConfigWithChatTemplate(chatTemplateOverride);
        } else if (renderConfig == null || renderConfig.isBlank()
                || config == null || !config.hasChatTemplate()) {
            throw new IllegalStateException(
                    "Tokenizer import does not provide tokenizer_config.json with a chat_template");
        }
        return impl.applyChatTemplateContext(
                renderConfig, ChatTemplate.requestContextJson(request));
    }

    private String tokenizerConfigWithChatTemplate(String chatTemplate) {
        try {
            JsonNode parsed = tokenizerConfigJson == null || tokenizerConfigJson.isBlank()
                    ? JSON_MAPPER.createObjectNode()
                    : JSON_MAPPER.readTree(tokenizerConfigJson);
            if (!(parsed instanceof ObjectNode)) {
                throw new TokenizerException("tokenizer_config.json must contain a JSON object");
            }
            ((ObjectNode) parsed).put("chat_template", chatTemplate);
            return JSON_MAPPER.writeValueAsString(parsed);
        } catch (TokenizerException e) {
            throw e;
        } catch (Exception e) {
            throw new TokenizerException(
                    "Could not apply imported model chat template: " + e.getMessage(), e);
        }
    }

    @Override
    public String applyChatTemplateContext(String contextJson) {
        checkNotClosed();
        if (tokenizerConfigJson == null || tokenizerConfigJson.isBlank()
                || config == null || !config.hasChatTemplate()) {
            throw new IllegalStateException(
                    "Tokenizer import does not provide tokenizer_config.json with a chat_template");
        }
        return impl.applyChatTemplateContext(tokenizerConfigJson, contextJson);
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

    private void initializeSpecialTokens(TokenizerConfig tokenizerConfig) {
        if (tokenizerConfig == null) {
            return;
        }
        bosTokenId = resolveSpecialToken(tokenizerConfig.getBosToken());
        eosTokenId = resolveSpecialToken(tokenizerConfig.getEosToken());
        padTokenId = resolveSpecialToken(tokenizerConfig.getPadToken());
        unkTokenId = resolveSpecialToken(tokenizerConfig.getUnkToken());
    }

    private int resolveSpecialToken(String token) {
        if (token == null || token.isEmpty()) {
            return -1;
        }
        Integer tokenId = getTokenId(token);
        return tokenId == null ? -1 : tokenId;
    }

    /** Public streaming decoder that retains the same native tokenizer state. */
    public static final class DecodeStream implements AutoCloseable {
        private final NativeTokenizer.DecodeStream delegate;

        private DecodeStream(NativeTokenizer.DecodeStream delegate) {
            this.delegate = delegate;
        }

        public String step(long tokenId) {
            return delegate.step(tokenId);
        }

        @Override
        public void close() {
            delegate.close();
        }
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

}
