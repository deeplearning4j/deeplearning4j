/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
package org.eclipse.deeplearning4j.sdx.aot;

import lombok.extern.slf4j.Slf4j;
import org.eclipse.deeplearning4j.llm.generation.ChatGenerationResult;
import org.eclipse.deeplearning4j.llm.generation.GenerationPipeline;
import org.eclipse.deeplearning4j.llm.generation.GenerationPipelineConfig;
import org.eclipse.deeplearning4j.llm.generation.GenerationResult;
import org.eclipse.deeplearning4j.llm.generation.sampling.SamplingConfig;
import org.eclipse.deeplearning4j.llm.tokenizer.ChatTemplate;
import org.eclipse.deeplearning4j.llm.tokenizer.Encoding;
import org.eclipse.deeplearning4j.llm.tokenizer.HuggingFaceTokenizer;
import org.eclipse.deeplearning4j.llm.tokenizer.Tokenizer;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.ggml.GGMLModelImport;
import org.nd4j.ggml.format.GGMLMetadata;
import org.nd4j.shade.jackson.databind.JsonNode;
import org.nd4j.shade.jackson.databind.ObjectMapper;
import org.nd4j.shade.jackson.databind.node.ArrayNode;
import org.nd4j.shade.jackson.databind.node.ObjectNode;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

/**
 * JVM-independent engine behind the SDX AOT exports: one place that loads a model
 * (GGUF or native SameDiff formats), resolves a tokenizer, owns a long-lived
 * {@link GenerationPipeline}, and runs generation. Used by both the {@code sdx-llm}
 * CLI ({@link SdxLlmCli}) and the {@code libsdx_llm} C ABI ({@link SdxLlmCApi}).
 *
 * <p>All option plumbing is JSON (shaded Jackson tree API only — no databind
 * reflection) so the C ABI stays stable while options evolve. See ADR 0109.</p>
 */
@Slf4j
public final class SdxLlmCore implements SdxLlmModel {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    private final SameDiff decoder;
    private final Tokenizer tokenizer;
    private final GenerationPipeline pipeline;
    private final int defaultMaxNewTokens;
    private final SamplingConfig defaultSampling;
    private volatile GenerationResult lastResult;

    private SdxLlmCore(SameDiff decoder, Tokenizer tokenizer, GenerationPipeline pipeline,
                       int defaultMaxNewTokens, SamplingConfig defaultSampling) {
        this.decoder = decoder;
        this.tokenizer = tokenizer;
        this.pipeline = pipeline;
        this.defaultMaxNewTokens = defaultMaxNewTokens;
        this.defaultSampling = defaultSampling;
    }

    /**
     * Load a model + tokenizer and build the generation pipeline.
     *
     * @param modelPath     .gguf/.ggml (imported via nd4j-ggml) or .sdz/.sdnb/.fb (SameDiff.load)
     * @param tokenizerPath optional tokenizer.json file or directory containing one; when null,
     *                      the model's parent directory is tried
     * @param optionsJson   optional JSON: {"maxNewTokens":128,"graphOptimizer":true,
     *                      "sampling":{"preset":"greedy"}} — see {@link #parseSampling(JsonNode)}
     */
    public static SdxLlmCore load(String modelPath, String tokenizerPath, String optionsJson) throws IOException {
        JsonNode opts = optionsJson == null || optionsJson.isEmpty()
                ? MAPPER.createObjectNode() : MAPPER.readTree(optionsJson);

        SameDiff decoder = loadDecoder(modelPath);
        Tokenizer tokenizer = resolveTokenizer(modelPath, tokenizerPath);

        int maxNewTokens = opts.path("maxNewTokens").asInt(128);
        SamplingConfig sampling = parseSampling(opts.path("sampling"));
        boolean graphOptimizer = opts.path("graphOptimizer").asBoolean(true);

        GenerationPipelineConfig config = GenerationPipelineConfig.builder()
                .decoder(decoder)
                .tokenizer(tokenizer)
                .samplingConfig(sampling)
                .maxNewTokens(maxNewTokens)
                .graphOptimizerEnabled(graphOptimizer)
                .build();

        GenerationPipeline pipeline = GenerationPipeline.create(config);
        log.info("sdx-llm loaded model={} tokenizer={} maxNewTokens={}", modelPath,
                tokenizerPath == null ? "(model dir)" : tokenizerPath, maxNewTokens);
        return new SdxLlmCore(decoder, tokenizer, pipeline, maxNewTokens, sampling);
    }

    /** Load a decoder graph from GGUF (via nd4j-ggml) or a native SameDiff format. */
    public static SameDiff loadDecoder(String modelPath) throws IOException {
        File modelFile = new File(modelPath);
        if (!modelFile.exists()) {
            throw new IOException("Model file not found: " + modelPath);
        }
        String name = modelFile.getName().toLowerCase(Locale.ROOT);
        if (name.endsWith(".gguf") || name.endsWith(".ggml")) {
            try {
                return GGMLModelImport.importModel(modelFile);
            } catch (Exception e) {
                throw new IOException("GGUF import failed for " + modelPath, e);
            }
        }
        if (name.endsWith(".sdz") || name.endsWith(".sdnb") || name.endsWith(".fb")) {
            return SameDiff.load(modelFile, false);
        }
        throw new IOException("Unsupported model format: " + modelPath
                + " (expected .gguf, .ggml, .sdz, .sdnb or .fb)");
    }

    /**
     * Resolve a tokenizer: explicit file/directory first, then sidecar {@code tokenizer.json}
     * next to the model, then GGUF-embedded tokenizer data (R1 / R8-2: no sidecar required
     * when the GGUF carries {@code tokenizer.ggml.*} arrays).
     *
     * <p>GGUF-embedded path: reads {@code tokenizer.ggml.model} (BPE type), {@code .tokens}
     * (vocab array), {@code .merges} (merge rules), and BOS/EOS IDs from the GGUF metadata via
     * {@link GGMLModelImport#inspectModel}, then constructs a HuggingFace tokenizer.json in
     * memory and calls {@link HuggingFaceTokenizer#fromJson}. Qwen2/GPT-2 BPE GGUFs work with
     * this path.</p>
     *
     * @param modelPath     path to the model file (checked for .gguf/.ggml extension)
     * @param tokenizerPath optional explicit tokenizer path; {@code null} triggers auto-resolve
     * @return a ready {@link Tokenizer}
     * @throws IOException if no tokenizer source is found or construction fails
     */
    public static Tokenizer resolveTokenizer(String modelPath, String tokenizerPath) throws IOException {
        // 1) Explicit tokenizer path: file or directory.
        if (tokenizerPath != null && !tokenizerPath.isEmpty()) {
            File f = new File(tokenizerPath);
            if (!f.exists()) {
                throw new IOException("Tokenizer path not found: " + tokenizerPath);
            }
            return f.isDirectory() ? HuggingFaceTokenizer.fromDirectory(f) : HuggingFaceTokenizer.fromFile(f);
        }

        // 2) Sidecar tokenizer.json in the model's parent directory.
        File parent = new File(modelPath).getAbsoluteFile().getParentFile();
        File tokenizerJson = new File(parent, "tokenizer.json");
        if (tokenizerJson.exists()) {
            log.debug("sdx-llm: loading sidecar tokenizer from {}", tokenizerJson);
            return HuggingFaceTokenizer.fromDirectory(parent);
        }

        // 3) GGUF-embedded tokenizer: read tokenizer.ggml.* metadata and build on the fly.
        String name = new File(modelPath).getName().toLowerCase(Locale.ROOT);
        if (name.endsWith(".gguf") || name.endsWith(".ggml")) {
            Tokenizer embedded = tryLoadEmbeddedGgufTokenizer(modelPath);
            if (embedded != null) {
                return embedded;
            }
        }

        throw new IOException("No tokenizer found for " + modelPath
                + " — pass --tokenizer <path> or place tokenizer.json next to the model."
                + " (GGUF-embedded path tried but tokenizer.ggml.tokens was missing or empty.)");
    }

    /**
     * Attempt to build a {@link HuggingFaceTokenizer} directly from the GGUF metadata arrays.
     * Returns {@code null} (with a warning) when the required keys are absent.
     */
    @SuppressWarnings("unchecked")
    private static Tokenizer tryLoadEmbeddedGgufTokenizer(String modelPath) throws IOException {
        GGMLMetadata meta;
        try {
            meta = GGMLModelImport.inspectModel(new File(modelPath));
        } catch (Exception e) {
            log.warn("sdx-llm: GGUF inspect failed for tokenizer probe ({}): {}", modelPath, e.getMessage());
            return null;
        }

        Map<String, Object> raw = meta.getRawMetadata();
        if (raw == null || raw.isEmpty()) {
            log.warn("sdx-llm: GGUF metadata empty for {}", modelPath);
            return null;
        }

        Object tokensObj = raw.get("tokenizer.ggml.tokens");
        if (!(tokensObj instanceof List) || ((List<?>) tokensObj).isEmpty()) {
            log.warn("sdx-llm: GGUF has no tokenizer.ggml.tokens in {}" +
                    " — found keys: {}", modelPath, raw.keySet());
            return null;
        }
        List<String> tokens = (List<String>) tokensObj;

        // Merges are required for BPE (gpt2 / qwen2 model type); SentencePiece (llama) does not use them.
        Object mergesObj = raw.get("tokenizer.ggml.merges");
        List<String> merges = (mergesObj instanceof List) ? (List<String>) mergesObj : new ArrayList<>();

        // BOS/EOS/UNK from metadata or TokenizerInfo fallback.
        GGMLMetadata.TokenizerInfo ti = meta.getTokenizerInfo();
        int bosId = (ti != null) ? ti.getBosTokenId() : 1;
        int eosId = (ti != null) ? ti.getEosTokenId() : 2;
        String tokenizerModel = (ti != null && ti.getModel() != null) ? ti.getModel() : "gpt2";

        // R8 item 4 fix: read token_type array so CONTROL/USER_DEFINED tokens get added as special.
        // GGUFReader stores INT32 arrays as int[] primitives.
        Object tokenTypeObj = raw.get("tokenizer.ggml.token_type");
        int[] tokenTypes = null;
        if (tokenTypeObj instanceof int[]) {
            tokenTypes = (int[]) tokenTypeObj;
        } else if (tokenTypeObj instanceof List) {
            // Defensive: handle boxed List<Integer> if a future reader change boxes the array.
            List<?> list = (List<?>) tokenTypeObj;
            tokenTypes = new int[list.size()];
            for (int i = 0; i < list.size(); i++) {
                tokenTypes[i] = ((Number) list.get(i)).intValue();
            }
        }

        int controlCount = 0;
        if (tokenTypes != null) {
            for (int t : tokenTypes) {
                if (t != 1) controlCount++; // 1 = NORMAL in GGUF token_type
            }
        }
        log.info("sdx-llm: building embedded tokenizer from GGUF ({} vocab={} merges={} model={} bos={} eos={} specialTokens={})",
                modelPath, tokens.size(), merges.size(), tokenizerModel, bosId, eosId, controlCount);

        String json = buildBpeTokenizerJson(tokens, merges, bosId, eosId, tokenizerModel, tokenTypes);
        String tokenizerConfigJson = embeddedTokenizerConfigJson(raw, tokens, bosId, eosId);
        try {
            return HuggingFaceTokenizer.fromJson(json, tokenizerConfigJson);
        } catch (Exception e) {
            throw new IOException("GGUF-embedded tokenizer construction failed for " + modelPath
                    + " — vocab=" + tokens.size() + " merges=" + merges.size(), e);
        }
    }

    static String embeddedTokenizerConfigJson(Map<String, Object> metadata,
                                                      List<String> tokens,
                                                      int bosId, int eosId) {
        Object template = metadata.get("tokenizer.chat_template");
        if (!(template instanceof String) || ((String) template).isBlank()) {
            return null;
        }
        ObjectNode config = MAPPER.createObjectNode();
        config.put("chat_template", (String) template);
        if (bosId >= 0 && bosId < tokens.size()) config.put("bos_token", tokens.get(bosId));
        if (eosId >= 0 && eosId < tokens.size()) config.put("eos_token", tokens.get(eosId));
        Object addBos = metadata.get("tokenizer.ggml.add_bos_token");
        Object addEos = metadata.get("tokenizer.ggml.add_eos_token");
        if (addBos instanceof Boolean) config.put("add_bos_token", (Boolean) addBos);
        if (addEos instanceof Boolean) config.put("add_eos_token", (Boolean) addEos);
        return config.toString();
    }

    /**
     * Backward-compatible overload: no token_type array (e.g. older GGUF files or tests).
     * Delegates to the full 6-arg form with {@code tokenTypes = null}.
     *
     * <p>The Rust HF tokenizers library (used by {@link HuggingFaceTokenizer#fromJson}) accepts
     * the standard tokenizer.json schema with a {@code "model"} section. For BPE models the
     * required structure is:
     * <pre>
     * {
     *   "version": "1.0",
     *   "model": {
     *     "type": "BPE",
     *     "vocab": { "<token>": id, ... },
     *     "merges": [ "a b", ... ],
     *     "byte_fallback": false,
     *     "fuse_unk": false
     *   },
     *   "added_tokens": [],
     *   "normalizer": null,
     *   "pre_tokenizer": null,
     *   "post_processor": null,
     *   "decoder": null
     * }
     * </pre>
     * GGUF {@code tokenizer.ggml.tokens} is a flat array indexed by token ID. The GGUF
     * {@code tokenizer.ggml.merges} array contains merge rules as "a b" strings.</p>
     */
    static String buildBpeTokenizerJson(List<String> tokens, List<String> merges,
                                        int bosId, int eosId, String ggmlModel) throws IOException {
        return buildBpeTokenizerJson(tokens, merges, bosId, eosId, ggmlModel, null);
    }

    /**
     * Build a minimal HuggingFace tokenizer.json from GGUF {@code tokenizer.ggml.*} arrays,
     * including all special (non-NORMAL) tokens in {@code added_tokens}.
     *
     * <p><b>R8 item 4 fix:</b> GGUF stores a parallel {@code tokenizer.ggml.token_type} int
     * array where {@code 1 = NORMAL} (regular BPE pieces) and any other value (
     * {@code 2 = UNKNOWN}, {@code 3 = CONTROL}, {@code 4 = USER_DEFINED}, {@code 5 = UNUSED},
     * {@code 6 = BYTE}) marks tokens that must appear in {@code added_tokens} with
     * {@code "special": true} so the Rust HF tokenizer treats them as atomic units rather
     * than splitting them character-by-character. Without this, {@code <|im_start|>} (type=3,
     * id=151644 in Qwen2.5) tokenizes as 6 characters instead of one token, producing garbled
     * output in sidecar-free (embedded tokenizer) mode.</p>
     *
     * @param tokenTypes parallel int array from {@code tokenizer.ggml.token_type}; may be
     *                   {@code null} (treated as all-NORMAL).
     */
    static String buildBpeTokenizerJson(List<String> tokens, List<String> merges,
                                        int bosId, int eosId, String ggmlModel,
                                        int[] tokenTypes) throws IOException {
        ObjectMapper m = new ObjectMapper();
        ObjectNode root = m.createObjectNode();
        root.put("version", "1.0");

        // vocab map: token → id
        ObjectNode vocab = m.createObjectNode();
        for (int i = 0; i < tokens.size(); i++) {
            vocab.put(tokens.get(i), i);
        }

        ObjectNode model = m.createObjectNode();
        // "gpt2" and "qwen2" in GGUF both map to BPE in the HF tokenizers schema.
        model.put("type", "BPE");
        model.set("vocab", vocab);

        ArrayNode mergesArr = m.createArrayNode();
        if (merges != null) {
            for (String merge : merges) {
                mergesArr.add(merge);
            }
        }
        model.set("merges", mergesArr);
        model.put("byte_fallback", false);
        model.put("fuse_unk", false);
        root.set("model", model);

        // added_tokens: BOS + EOS always; then every token whose type != 1 (NORMAL).
        // GGUF token_type values: 1=NORMAL, 2=UNKNOWN, 3=CONTROL, 4=USER_DEFINED,
        // 5=UNUSED, 6=BYTE. All non-NORMAL tokens must be in added_tokens so the Rust
        // HF tokenizer treats them as atomic. This fixes ChatML delimiters like
        // <|im_start|> (id=151644, type=3) being split character-by-character.
        ArrayNode addedTokens = m.createArrayNode();
        Set<Integer> alreadyAdded = new HashSet<>();

        if (bosId >= 0 && bosId < tokens.size()) {
            addedTokens.add(makeAddedToken(m, bosId, tokens.get(bosId)));
            alreadyAdded.add(bosId);
        }
        if (eosId >= 0 && eosId < tokens.size() && eosId != bosId) {
            addedTokens.add(makeAddedToken(m, eosId, tokens.get(eosId)));
            alreadyAdded.add(eosId);
        }

        // Sweep all tokens: add any with non-NORMAL type that aren't already added.
        if (tokenTypes != null) {
            int limit = Math.min(tokenTypes.length, tokens.size());
            for (int i = 0; i < limit; i++) {
                if (tokenTypes[i] != 1 && !alreadyAdded.contains(i)) {
                    addedTokens.add(makeAddedToken(m, i, tokens.get(i)));
                    alreadyAdded.add(i);
                }
            }
        }
        root.set("added_tokens", addedTokens);

        root.putNull("normalizer");
        root.putNull("pre_tokenizer");
        root.putNull("post_processor");
        root.putNull("decoder");

        return m.writeValueAsString(root);
    }

    /** Build a single added_tokens entry with standard fields. */
    private static ObjectNode makeAddedToken(ObjectMapper m, int id, String content) {
        ObjectNode node = m.createObjectNode();
        node.put("id", id);
        node.put("content", content);
        node.put("single_word", false);
        node.put("lstrip", false);
        node.put("rstrip", false);
        node.put("normalized", false);
        node.put("special", true);
        return node;
    }

    /**
     * Sampling options: {"preset":"greedy|precise|default|creative"} or explicit fields
     * {"temperature":0.8,"topK":40,"topP":0.9,"repetitionPenalty":1.1,"seed":42}.
     * Explicit fields imply doSample=true unless "doSample" says otherwise.
     */
    public static SamplingConfig parseSampling(JsonNode node) {
        if (node == null || node.isMissingNode() || node.isNull()) {
            return SamplingConfig.greedy();
        }
        String preset = node.path("preset").asText("");
        switch (preset.toLowerCase(Locale.ROOT)) {
            case "greedy":
                return SamplingConfig.greedy();
            case "precise":
                return SamplingConfig.precise();
            case "default":
                return SamplingConfig.defaultConfig();
            case "creative":
                return SamplingConfig.creative();
            case "":
                break;
            default:
                throw new IllegalArgumentException("Unknown sampling preset: " + preset);
        }
        if (!node.fieldNames().hasNext()) {
            return SamplingConfig.greedy();
        }
        SamplingConfig.SamplingConfigBuilder b = SamplingConfig.builder();
        if (node.has("temperature")) b.temperature(node.get("temperature").asDouble());
        if (node.has("topK")) b.topK(node.get("topK").asInt());
        if (node.has("topP")) b.topP(node.get("topP").asDouble());
        if (node.has("repetitionPenalty")) b.repetitionPenalty(node.get("repetitionPenalty").asDouble());
        if (node.has("seed")) b.seed(node.get("seed").asLong());
        b.doSample(node.path("doSample").asBoolean(true));
        return b.build();
    }

    /**
     * Run generation. Per-call option JSON may override {"maxNewTokens":N,"sampling":{...}}.
     * The pipeline (and its compiled DSP plan / KV state) is reused across calls.
     */
    public GenerationResult generate(String prompt, String optionsJson) throws IOException {
        int maxNewTokens = defaultMaxNewTokens;
        SamplingConfig sampling = defaultSampling;
        if (optionsJson != null && !optionsJson.isEmpty()) {
            JsonNode opts = MAPPER.readTree(optionsJson);
            maxNewTokens = opts.path("maxNewTokens").asInt(maxNewTokens);
            if (opts.has("sampling")) {
                sampling = parseSampling(opts.get("sampling"));
            }
        }
        GenerationResult result = pipeline.generate(prompt, maxNewTokens, sampling);
        lastResult = result;
        return result;
    }

    @Override
    public String generateText(String prompt, String optionsJson) throws IOException {
        return generate(prompt, optionsJson).getText();
    }

    /**
     * Run one complete structured chat turn. The imported tokenizer owns prompt
     * rendering, output normalization, reasoning blocks, and tool-call syntax.
     */
    public String generateChat(String requestJson, String optionsJson) throws IOException {
        ChatTemplate.Request request = parseChatRequest(requestJson);
        int maxNewTokens = defaultMaxNewTokens;
        SamplingConfig sampling = defaultSampling;
        if (optionsJson != null && !optionsJson.isEmpty()) {
            JsonNode opts = MAPPER.readTree(optionsJson);
            maxNewTokens = opts.path("maxNewTokens").asInt(maxNewTokens);
            if (opts.has("sampling")) {
                sampling = parseSampling(opts.get("sampling"));
            }
        }
        ChatGenerationResult result = pipeline.generateChat(request, maxNewTokens, sampling);
        return chatResultJson(result);
    }

    /** Decode streamed/raw assistant output through the imported model protocol. */
    public String parseChatResult(String requestJson, String rawText) throws IOException {
        ChatGenerationResult result = pipeline.parseChatOutput(
                parseChatRequest(requestJson), rawText);
        return chatResultJson(result);
    }

    static ChatTemplate.Request parseChatRequest(String requestJson) throws IOException {
        if (requestJson == null || requestJson.isBlank()) {
            throw new IllegalArgumentException("chat request JSON must not be blank");
        }
        JsonNode root = MAPPER.readTree(requestJson);
        if (root == null || !root.isObject()) {
            throw new IllegalArgumentException("chat request must be a JSON object");
        }
        JsonNode messagesNode = root.get("messages");
        if (messagesNode == null || !messagesNode.isArray()) {
            throw new IllegalArgumentException("chat request messages must be an array");
        }

        List<ChatTemplate.Message> messages = new ArrayList<>();
        for (JsonNode messageNode : messagesNode) {
            if (!messageNode.isObject()) {
                throw new IllegalArgumentException("chat message must be an object");
            }
            String role = requiredText(messageNode, "role");
            String content = messageNode.hasNonNull("content")
                    ? messageNode.get("content").asText() : null;
            List<ChatTemplate.ToolCall> calls = parseToolCalls(messageNode.get("tool_calls"));
            String toolCallId = optionalText(messageNode, "tool_call_id");
            String toolName = optionalText(messageNode, "name");
            messages.add(new ChatTemplate.Message(
                    role, content, null, calls, toolCallId, toolName));
        }

        List<ChatTemplate.Tool> tools = new ArrayList<>();
        JsonNode toolsNode = root.get("tools");
        if (toolsNode != null && !toolsNode.isNull()) {
            if (!toolsNode.isArray()) {
                throw new IllegalArgumentException("chat request tools must be an array");
            }
            for (JsonNode toolNode : toolsNode) {
                JsonNode function = toolNode.has("function") ? toolNode.get("function") : toolNode;
                if (function == null || !function.isObject()) {
                    throw new IllegalArgumentException("chat tool must be an object");
                }
                String name = requiredText(function, "name");
                String description = optionalText(function, "description");
                Map<String, Object> parameters = function.has("parameters")
                        ? MAPPER.convertValue(function.get("parameters"), Map.class)
                        : Map.of();
                tools.add(ChatTemplate.Tool.function(name, description, parameters));
            }
        }

        ChatTemplate.Request.Builder builder = ChatTemplate.Request.builder()
                .messages(messages)
                .tools(tools)
                .addGenerationPrompt(root.path("add_generation_prompt").asBoolean(true))
                .toolChoice(parseEnum(root, "tool_choice", ChatTemplate.ToolChoice.class,
                        ChatTemplate.ToolChoice.AUTO));
        if (root.hasNonNull("tool_definition_format")) {
            builder.toolDefinitionFormat(parseEnum(
                    root, "tool_definition_format", ChatTemplate.ToolDefinitionFormat.class, null));
        }
        if (root.hasNonNull("tool_call_format")) {
            builder.toolCallFormat(parseEnum(
                    root, "tool_call_format", ChatTemplate.ToolCallFormat.class, null));
        }
        if (root.has("template_arguments") && root.get("template_arguments").isObject()) {
            builder.templateArguments(MAPPER.convertValue(
                    root.get("template_arguments"), Map.class));
        }
        return builder.build();
    }

    private static List<ChatTemplate.ToolCall> parseToolCalls(JsonNode node) {
        if (node == null || node.isNull()) {
            return List.of();
        }
        if (!node.isArray()) {
            throw new IllegalArgumentException("message tool_calls must be an array");
        }
        List<ChatTemplate.ToolCall> calls = new ArrayList<>();
        for (JsonNode callNode : node) {
            JsonNode function = callNode.has("function") ? callNode.get("function") : callNode;
            if (function == null || !function.isObject()) {
                throw new IllegalArgumentException("message tool call must be an object");
            }
            String name = requiredText(function, "name");
            JsonNode args = function.has("arguments")
                    ? function.get("arguments") : function.get("args");
            Map<String, Object> arguments;
            if (args == null || args.isNull()) {
                arguments = Map.of();
            } else if (args.isTextual()) {
                try {
                    JsonNode parsed = MAPPER.readTree(args.asText());
                    if (parsed == null || !parsed.isObject()) {
                        throw new IllegalArgumentException("tool call arguments must be a JSON object");
                    }
                    arguments = MAPPER.convertValue(parsed, Map.class);
                } catch (IOException e) {
                    throw new IllegalArgumentException("tool call arguments were not valid JSON", e);
                }
            } else if (args.isObject()) {
                arguments = MAPPER.convertValue(args, Map.class);
            } else {
                throw new IllegalArgumentException("tool call arguments must be an object");
            }
            calls.add(ChatTemplate.ToolCall.function(
                    optionalText(callNode, "id"), name, arguments));
        }
        return List.copyOf(calls);
    }

    static String chatResultJson(ChatGenerationResult result) {
        ObjectNode root = MAPPER.createObjectNode();
        root.put("rawText", result.getRawText());
        root.put("content", result.getContent());
        root.put("reasoningContent", result.getReasoningContent());
        ArrayNode calls = root.putArray("toolCalls");
        for (ChatTemplate.ToolCall call : result.getToolCalls()) {
            ObjectNode value = calls.addObject();
            if (call.getId() != null) value.put("id", call.getId());
            value.put("name", call.getName());
            value.set("arguments", MAPPER.valueToTree(call.getArguments()));
        }
        ArrayNode errors = root.putArray("protocolErrors");
        for (String error : result.getParseErrors()) {
            errors.add(error);
        }
        return root.toString();
    }

    private static String requiredText(JsonNode node, String field) {
        String value = optionalText(node, field);
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException(field + " must not be blank");
        }
        return value;
    }

    private static String optionalText(JsonNode node, String field) {
        JsonNode value = node == null ? null : node.get(field);
        return value == null || value.isNull() ? null : value.asText();
    }

    private static <E extends Enum<E>> E parseEnum(
            JsonNode root, String field, Class<E> type, E defaultValue) {
        JsonNode value = root.get(field);
        if (value == null || value.isNull() || value.asText().isBlank()) {
            return defaultValue;
        }
        return Enum.valueOf(type, value.asText().trim().toUpperCase(Locale.ROOT));
    }

    /**
     * Render either an ordered message array or a complete structured chat context
     * through the tokenizer/model-owned MiniJinja template engine.
     */
    public String renderChatPrompt(String messagesOrContextJson,
                                   boolean addGenerationPrompt) throws IOException {
        if (messagesOrContextJson == null || messagesOrContextJson.isBlank()) {
            throw new IllegalArgumentException("chat messages/context JSON must not be blank");
        }
        JsonNode input = MAPPER.readTree(messagesOrContextJson);
        ObjectNode context;
        if (input.isArray()) {
            context = MAPPER.createObjectNode();
            context.set("messages", input);
        } else if (input.isObject()) {
            context = ((ObjectNode) input).deepCopy();
        } else {
            throw new IllegalArgumentException("chat input must be a JSON message array or context object");
        }
        context.put("add_generation_prompt", addGenerationPrompt);
        return tokenizer.applyChatTemplateContext(context.toString());
    }

    public int[] tokenize(String text, boolean addSpecialTokens) {
        Encoding encoding = tokenizer.encode(text, addSpecialTokens);
        return encoding.getIds();
    }

    public String detokenize(int[] ids, boolean skipSpecialTokens) {
        return tokenizer.decode(ids, skipSpecialTokens);
    }

    public Tokenizer tokenizer() {
        return tokenizer;
    }

    public SameDiff decoder() {
        return decoder;
    }

    /** Stats of the most recent {@link #generate}, as a stable JSON document. */
    public String lastResultJson() {
        GenerationResult r = lastResult;
        ObjectNode node = MAPPER.createObjectNode();
        if (r == null) {
            node.put("hasResult", false);
            return node.toString();
        }
        node.put("hasResult", true);
        node.put("generatedTokens", r.getGeneratedTokenCount());
        node.put("promptTokens", r.getPromptTokenCount());
        node.put("totalTokens", r.getTotalTokenCount());
        node.put("generationTimeMs", r.getGenerationTimeMs());
        node.put("tokensPerSecond", r.getTokensPerSecond());
        node.put("firstTokenLatencyMs", r.getFirstTokenLatencyMs());
        node.put("finishReason", String.valueOf(r.getFinishReason()));
        return node.toString();
    }

    /** Model/tokenizer summary for the CLI `info` command and C ABI introspection. */
    public String infoJson() {
        ObjectNode node = MAPPER.createObjectNode();
        node.put("variables", decoder.variables().size());
        node.put("inputs", String.join(",", decoder.inputs()));
        node.put("outputs", String.join(",", decoder.outputs()));
        node.put("vocabSize", tokenizer.getVocabSize());
        node.put("bosTokenId", tokenizer.getBosTokenId());
        node.put("eosTokenId", tokenizer.getEosTokenId());
        node.put("hasChatTemplate", tokenizer.getChatTemplate() != null);
        return node.toString();
    }

    @Override
    public void close() {
        try {
            pipeline.close();
        } catch (Exception e) {
            log.warn("Error closing generation pipeline", e);
        }
    }
}
