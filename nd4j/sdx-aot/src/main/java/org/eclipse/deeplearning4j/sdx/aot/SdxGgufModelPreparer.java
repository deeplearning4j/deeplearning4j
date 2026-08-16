/*
 * Copyright (c) Eclipse Deeplearning4j
 * SPDX-License-Identifier: Apache-2.0
 */
package org.eclipse.deeplearning4j.sdx.aot;

import org.eclipse.deeplearning4j.llm.generation.SdxTextGenerationConfig;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.serde.SDZSerializer;
import org.nd4j.common.config.ND4JSystemProperties;
import org.nd4j.dsp.model.SdxCompiledModel;
import org.nd4j.dsp.model.SdxModelCache;
import org.nd4j.dsp.model.SdxModelCompiler;
import org.nd4j.dsp.model.SdxPlatformProviderDescriptor;
import org.nd4j.dsp.model.SdxSourceIdentity;
import org.nd4j.dsp.model.SdxTargetProfile;
import org.nd4j.dsp.model.SdxTextModelAssets;
import org.nd4j.ggml.GGMLImportException;
import org.nd4j.ggml.GGMLExportException;
import org.nd4j.ggml.GGMLModelImport;
import org.nd4j.ggml.GGMLModelExport;
import org.nd4j.ggml.convert.ConversionOptions;
import org.nd4j.ggml.export.ExportOptions;
import org.nd4j.ggml.format.GGMLMetadata;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.shade.jackson.databind.JsonNode;
import org.nd4j.shade.jackson.databind.ObjectMapper;
import org.nd4j.shade.jackson.databind.node.ObjectNode;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Backend-neutral raw-model preparation used by every {@code libsdx_llm} consumer.
 * Import, source attestation, immutable cache admission and text-asset validation are
 * shared; only the {@link SdxTargetProfile} compiler/provider policy is backend specific.
 */
final class SdxGgufModelPreparer {
    static final String PREPARED_SCHEMA = "sdx-prepared-text-model-v5";
    static final String GRAPH_IMPORT_ABI = "ggml-runtime-packed-gdn-v3";
    static final String RESOLVED_SCHEMA = "sdx-resolved-text-model-v1";

    private static final ObjectMapper MAPPER = new ObjectMapper();

    private SdxGgufModelPreparer() {
    }

    static synchronized String prepare(String sourceGguf, String tokenizerPath,
                                       String targetProfile, String cacheDirectory,
                                       String optionsJson) throws IOException {
        Path source = requireRegularFile(sourceGguf, "source GGUF");
        SdxTargetProfile target = SdxTargetProfile.fromId(requireText(targetProfile, "target profile"));
        SdxModelCache cache = new SdxModelCache(Path.of(requireText(cacheDirectory, "cache directory")));
        Files.createDirectories(cache.root());
        Path temporaryRoot = configureTemporaryDirectory(cache);

        JsonNode options = parseObject(optionsJson);
        PreparationProfile profile = PreparationProfile.from(options);
        String diagnosticMode = configureDiagnostics(options);
        RawSourceIdentity sourceIdentity = RawSourceIdentity.identify(source);
        verifyAttestation(sourceIdentity, options);

        Path preparedRoot = cache.root().resolve("prepared").resolve(sourceIdentity.sha256())
                .resolve(profile.sha256());
        Path canonicalPointer = preparedRoot.resolve("canonical.path");
        Path optimizedSource = profile.existingOptimizedSource(source, preparedRoot);
        Path canonical = readCanonicalPointer(cache, canonicalPointer);
        SdxSourceIdentity canonicalIdentity =
                canonical == null ? null : SdxSourceIdentity.identify(canonical);
        if (canonical != null) {
            try {
                SdxCompiledModel cached = cache.resolve(canonical, target);
                if (isNativeTextGenerationContract(
                        cached.requireTextModelAssets().textGenerationConfig())) {
                    int cachedContextLength = contextLength(source);
                    requireUnchangedRawSource(source, sourceIdentity);
                    return preparedJson(sourceIdentity, canonicalIdentity, canonical, cached, true,
                            cachedContextLength, target, profile, diagnosticMode, optimizedSource);
                }
            } catch (SdxModelCache.MissingCompiledModelException missingTarget) {
                // The canonical import is reusable; compile only the missing target below.
            }
        }

        Files.createDirectories(preparedRoot);
        optimizedSource = profile.materializeOptimizedSource(source, preparedRoot, temporaryRoot);
        TokenizerAssets tokenizerAssets = materializeTokenizerAssets(
                source, tokenizerPath, preparedRoot.resolve("text-assets"));
        boolean publishCanonicalPointer = false;
        if (canonical == null) {
            Path generated = Files.createTempFile(temporaryRoot, "gguf-import-", ".sdz");
            boolean admitted = false;
            try {
                convertToSdz(optimizedSource, generated, tokenizerAssets, profile);
                canonical = cache.admitGeneratedSource(generated);
                admitted = true;
            } finally {
                if (!admitted) Files.deleteIfExists(generated);
            }
            canonicalIdentity = SdxSourceIdentity.identify(canonical);
            publishCanonicalPointer = true;
        } else {
            writeTextGenerationConfig(canonical, tokenizerAssets);
        }
        if (!isNativeTextGenerationContract(tokenizerAssets.textGenerationConfig)) {
            throw new IOException("Derived text-generation metadata does not satisfy the native SDX contract: "
                    + tokenizerAssets.textGenerationConfig);
        }

        requireUnchangedRawSource(source, sourceIdentity);
        if (publishCanonicalPointer) {
            writeCanonicalPointer(canonicalPointer, canonical);
        }
        SdxPlatformProviderDescriptor provider = target.platformProvider();
        String targetSoc = provider.defaultTargetSoc();
        SdxModelCompiler compiler = new SdxModelCompiler(cache);
        SdxModelCompiler.CompileOptions compileOptions = SdxModelCompiler.CompileOptions.builder()
                .tokenizer(tokenizerAssets.tokenizer)
                .tokenizerConfig(tokenizerAssets.tokenizerConfig)
                .textGenerationConfig(tokenizerAssets.textGenerationConfig)
                .targetSoc(targetSoc)
                .build();
        SdxCompiledModel compiled = compiler.compile(
                canonical,
                target,
                SdxModelCompiler.requireBuiltInTargetCompiler(target, targetSoc, false),
                compileOptions);
        return preparedJson(sourceIdentity, canonicalIdentity, canonical, compiled, false,
                tokenizerAssets.contextLength, target, profile, diagnosticMode, optimizedSource);
    }

    static String resolve(String sourceSdz, String targetProfile, String cacheDirectory)
            throws IOException {
        Path source = requireRegularFile(sourceSdz, "source SDZ");
        SdxTargetProfile target = SdxTargetProfile.fromId(requireText(targetProfile, "target profile"));
        SdxModelCache cache = new SdxModelCache(Path.of(requireText(cacheDirectory, "cache directory")));
        configureTemporaryDirectory(cache);
        SdxCompiledModel compiled = cache.resolve(source, target);
        SdxTextModelAssets assets = compiled.requireTextModelAssets();
        ObjectNode result = MAPPER.createObjectNode();
        result.put("schema", RESOLVED_SCHEMA);
        result.put("targetProfile", target.id());
        result.put("modelPath", compiled.runtimeModelPath().toString());
        result.put("tokenizerPath", assets.tokenizer().toString());
        result.put("tokenizerConfigPath", assets.tokenizerConfig().toString());
        result.put("textGenerationConfigPath", assets.textGenerationConfig().toString());
        result.put("compileKey", compiled.compileKey());
        result.put("compilerId", compiled.compilerId());
        result.put("executionProvider", target.platformProvider().providerId());
        return result.toString();
    }

    private static Path configureTemporaryDirectory(SdxModelCache cache) throws IOException {
        Path temporaryRoot = cache.root().resolve("tmp").toAbsolutePath().normalize();
        Files.createDirectories(temporaryRoot);
        System.setProperty(ND4JSystemProperties.ND4J_TEMP_DIR_PROPERTY, temporaryRoot.toString());
        return temporaryRoot;
    }

    private static String preparedJson(RawSourceIdentity sourceIdentity,
                                       SdxSourceIdentity canonicalIdentity, Path canonical,
                                       SdxCompiledModel compiled, boolean cacheHit,
                                       int contextLength, SdxTargetProfile target,
                                       PreparationProfile profile, String diagnosticMode,
                                       Path optimizedSource)
            throws IOException {
        SdxTextModelAssets assets = compiled.requireTextModelAssets();
        SdxPlatformProviderDescriptor provider = target.platformProvider();
        ObjectNode result = MAPPER.createObjectNode();
        result.put("schema", PREPARED_SCHEMA);
        result.put("cacheHit", cacheHit);
        result.put("sourceSha256", sourceIdentity.sha256());
        result.put("sourceBytes", sourceIdentity.bytes());
        result.put("canonicalSdzLogicalSha256", canonicalIdentity.sha256());
        result.put("canonicalSdzLogicalBytes", canonicalIdentity.logicalBytes());
        result.put("canonicalSdzPath", canonical.toString());
        result.put("canonicalSdzBytes", Files.size(canonical));
        result.put("targetProfile", target.id());
        result.put("modelPath", compiled.runtimeModelPath().toString());
        result.put("tokenizerPath", assets.tokenizer().toString());
        result.put("compileKey", compiled.compileKey());
        result.put("targetSoc", provider.defaultTargetSoc());
        result.put("contextLength", contextLength);
        result.put("maxPrefillLength", Math.max(1, contextLength - 1));
        result.put("executionProvider", provider.providerId());
        result.put("conversionProfileSha256", profile.sha256());
        result.set("conversionProfile", profile.json().deepCopy());
        result.put("diagnosticMode", diagnosticMode);
        result.put("optimizedSourcePath", optimizedSource.toString());
        result.put("optimizedSourceBytes", Files.size(optimizedSource));
        result.put("importResourcesReleased", true);
        return result.toString();
    }

    private static Path readCanonicalPointer(SdxModelCache cache, Path pointer) throws IOException {
        if (!Files.isRegularFile(pointer)) return null;
        String value = Files.readString(pointer, StandardCharsets.UTF_8).trim();
        if (value.isEmpty()) return null;
        Path canonical = Path.of(value).toAbsolutePath().normalize();
        if (!canonical.startsWith(cache.root()) || !Files.isRegularFile(canonical)) return null;
        return canonical;
    }

    private static void writeCanonicalPointer(Path pointer, Path canonical) throws IOException {
        Files.createDirectories(pointer.getParent());
        Path temporary = Files.createTempFile(pointer.getParent(), "canonical-", ".path");
        Files.writeString(temporary, canonical.toAbsolutePath().normalize().toString(),
                StandardCharsets.UTF_8);
        try {
            Files.move(temporary, pointer, StandardCopyOption.ATOMIC_MOVE,
                    StandardCopyOption.REPLACE_EXISTING);
        } catch (java.nio.file.AtomicMoveNotSupportedException unsupported) {
            Files.move(temporary, pointer, StandardCopyOption.REPLACE_EXISTING);
        }
    }

    private static void verifyAttestation(RawSourceIdentity identity, JsonNode options) {
        boolean hasSha256 = options.hasNonNull("verifiedSourceSha256");
        boolean hasBytes = options.hasNonNull("verifiedSourceBytes");
        if (hasSha256 != hasBytes) {
            throw new IllegalArgumentException(
                    "Verified source SHA-256 and byte count must be supplied together");
        }
        if (hasSha256) {
            String expected = options.get("verifiedSourceSha256").asText().toLowerCase(Locale.ROOT);
            if (!expected.matches("[0-9a-f]{64}")) {
                throw new IllegalArgumentException(
                        "Verified source SHA-256 must be 64 hexadecimal characters");
            }
            if (!identity.sha256().equals(expected)) {
                throw new IllegalArgumentException("Verified source SHA-256 did not match the GGUF bytes");
            }
            long expectedBytes = options.get("verifiedSourceBytes").asLong();
            if (expectedBytes <= 0) {
                throw new IllegalArgumentException("Verified source byte count must be positive");
            }
            if (identity.bytes() != expectedBytes) {
                throw new IllegalArgumentException(
                        "Verified source byte count did not match the GGUF bytes");
            }
        }
    }

    private static void requireUnchangedRawSource(Path source, RawSourceIdentity expected)
            throws IOException {
        RawSourceIdentity actual = RawSourceIdentity.identify(source);
        if (!expected.sha256().equals(actual.sha256()) || expected.bytes() != actual.bytes()) {
            throw new IOException("Raw GGUF changed while SDX preparation was in progress: " + source);
        }
    }

    @SuppressWarnings("unchecked")
    private static TokenizerAssets materializeTokenizerAssets(
            Path source, String explicitTokenizerPath, Path destination) throws IOException {
        Files.createDirectories(destination);
        GGMLMetadata metadata = inspect(source);
        Path tokenizerSource = null;
        Path configSource = null;
        Path generationSource = null;
        if (explicitTokenizerPath != null && !explicitTokenizerPath.isBlank()) {
            Path explicit = Path.of(explicitTokenizerPath).toAbsolutePath().normalize();
            tokenizerSource = Files.isDirectory(explicit) ? explicit.resolve("tokenizer.json") : explicit;
            Path parent = Files.isDirectory(explicit) ? explicit : explicit.getParent();
            configSource = parent.resolve("tokenizer_config.json");
            generationSource = firstRegular(parent.resolve("generation_config.json"),
                    parent.resolve("text_generation_config.json"));
        } else {
            Path parent = source.toAbsolutePath().getParent();
            tokenizerSource = parent.resolve("tokenizer.json");
            configSource = parent.resolve("tokenizer_config.json");
            generationSource = firstRegular(parent.resolve("generation_config.json"),
                    parent.resolve("text_generation_config.json"));
        }

        Path tokenizer = destination.resolve("tokenizer.json");
        Path tokenizerConfig = destination.resolve("tokenizer_config.json");
        Path textGeneration = destination.resolve("text_generation.json");
        int contextLength = contextLength(metadata);

        if (Files.isRegularFile(tokenizerSource)) {
            Files.copy(tokenizerSource, tokenizer, StandardCopyOption.REPLACE_EXISTING);
            if (!Files.isRegularFile(configSource)) {
                throw new IOException("tokenizer_config.json is required beside " + tokenizerSource);
            }
            Files.copy(configSource, tokenizerConfig, StandardCopyOption.REPLACE_EXISTING);
        } else {
            Map<String, Object> raw = metadata.getRawMetadata();
            Object tokensValue = raw.get("tokenizer.ggml.tokens");
            if (!(tokensValue instanceof List) || ((List<?>) tokensValue).isEmpty()) {
                throw new IOException("GGUF does not contain tokenizer.ggml.tokens: " + source);
            }
            List<String> tokens = (List<String>) tokensValue;
            Object mergesValue = raw.get("tokenizer.ggml.merges");
            List<String> merges = mergesValue instanceof List
                    ? (List<String>) mergesValue : new ArrayList<>();
            GGMLMetadata.TokenizerInfo info = metadata.getTokenizerInfo();
            int bosId = info == null ? 1 : info.getBosTokenId();
            int eosId = info == null ? 2 : info.getEosTokenId();
            String model = info == null || info.getModel() == null ? "gpt2" : info.getModel();
            int[] tokenTypes = tokenTypes(raw.get("tokenizer.ggml.token_type"));
            Files.writeString(tokenizer,
                    SdxLlmCore.buildBpeTokenizerJson(tokens, merges, bosId, eosId, model, tokenTypes),
                    StandardCharsets.UTF_8);
            String config = SdxLlmCore.embeddedTokenizerConfigJson(raw, tokens, bosId, eosId);
            if (config == null || config.isBlank()) {
                throw new IOException("GGUF does not contain tokenizer.chat_template: " + source);
            }
            Files.writeString(tokenizerConfig, config, StandardCharsets.UTF_8);
        }

        SdxTextGenerationConfig.Options generationOptions = generationOptions(
                tokenizerConfig, generationSource, metadata, contextLength);
        return new TokenizerAssets(tokenizer, tokenizerConfig, textGeneration,
                contextLength, generationOptions);
    }

    private static int[] tokenTypes(Object value) {
        if (value instanceof int[]) return (int[]) value;
        if (!(value instanceof List)) return null;
        List<?> values = (List<?>) value;
        int[] result = new int[values.size()];
        for (int i = 0; i < result.length; i++) result[i] = ((Number) values.get(i)).intValue();
        return result;
    }

    private static int contextLength(Path source) throws IOException {
        return contextLength(inspect(source));
    }

    private static int contextLength(GGMLMetadata metadata) {
        for (Map.Entry<String, Object> entry : metadata.getRawMetadata().entrySet()) {
            if (entry.getKey().endsWith(".context_length") && entry.getValue() instanceof Number) {
                int value = ((Number) entry.getValue()).intValue();
                if (value > 1) return value;
            }
        }
        return 4096;
    }

    private static void convertToSdz(
            Path source, Path destination, TokenizerAssets tokenizerAssets,
            PreparationProfile profile) throws IOException {
        try (SameDiff graph = GGMLModelImport.importModel(
                source.toFile(), profile.conversionOptions())) {
            SdxTextGenerationConfig.write(
                    graph, tokenizerAssets.generationOptions, tokenizerAssets.textGenerationConfig);
            Map<String, String> metadata = new HashMap<>();
            metadata.put("source_format", "ggml");
            metadata.put("source_file", source.getFileName().toString());
            metadata.put("conversion_profile_sha256", profile.sha256());
            metadata.put("conversion_profile", profile.json().toString());
            metadata.put("conversion_timestamp", String.valueOf(System.currentTimeMillis()));
            SDZSerializer.save(graph, destination.toFile(), false, metadata);
        } catch (GGMLImportException failure) {
            throw new IOException("Could not import optimized GGUF into canonical SDZ: " + source,
                    failure);
        } catch (RuntimeException failure) {
            throw new IOException("Runtime failure while importing optimized GGUF into canonical SDZ: "
                    + source + " (" + failure.getClass().getSimpleName() + ": "
                    + failure.getMessage() + ")", failure);
        }
    }

    private static void writeTextGenerationConfig(
            Path canonical, TokenizerAssets tokenizerAssets) throws IOException {
        try (SameDiff graph = SDZSerializer.load(canonical.toFile(), false)) {
            SdxTextGenerationConfig.write(
                    graph, tokenizerAssets.generationOptions, tokenizerAssets.textGenerationConfig);
        }
    }

    static boolean isNativeTextGenerationContract(Path config) {
        if (config == null || !Files.isRegularFile(config)) return false;
        try {
            JsonNode root = MAPPER.readTree(config.toFile());
            if (root == null || !root.isObject() || !root.path("formatVersion").canConvertToInt()) {
                return false;
            }
            int version = root.path("formatVersion").intValue();
            String profile = root.path("profile").asText("");
            boolean versionAndProfile = version == SdxTextGenerationConfig.KV_ONLY_FORMAT_VERSION
                    ? SdxTextGenerationConfig.KV_ONLY_PROFILE.equals(profile)
                    : version == SdxTextGenerationConfig.RECURRENT_STATE_FORMAT_VERSION
                    && SdxTextGenerationConfig.RECURRENT_STATE_PROFILE.equals(profile);
            if (!versionAndProfile || !root.path("io").isObject()
                    || !root.path("execution").isObject()
                    || !root.path("tokens").isObject()
                    || !root.path("limits").isObject()) {
                return false;
            }
            JsonNode io = root.path("io");
            if (!hasNonEmptyText(io, "inputIds", "causalMask", "positionOffset",
                    "cachePosition", "actualSequenceLength", "logits")
                    || !hasNonEmptyTextArray(io, "kvKeyInputs", "kvValueInputs",
                    "prefillKeyOutputs", "prefillValueOutputs")) {
                return false;
            }
            JsonNode execution = root.path("execution");
            if (!"BSHD".equals(execution.path("kvLayout").asText())
                    || !execution.path("kvDtype").isTextual()
                    || !execution.path("maskDtype").isTextual()
                    || !execution.path("planOwnsKvScatter").asBoolean(false)) {
                return false;
            }
            JsonNode tokens = root.path("tokens");
            JsonNode eos = tokens.path("eosIds");
            if (!tokens.path("padId").canConvertToInt() || tokens.path("padId").intValue() < 0
                    || !eos.isArray() || eos.isEmpty()) {
                return false;
            }
            for (JsonNode token : eos) {
                if (!token.canConvertToInt() || token.intValue() < 0) return false;
            }
            JsonNode limits = root.path("limits");
            int contextLength = limits.path("contextLength").asInt(-1);
            int maxPrefillLength = limits.path("maxPrefillLength").asInt(-1);
            if (contextLength < 2 || maxPrefillLength < 1 || maxPrefillLength >= contextLength) {
                return false;
            }
            JsonNode recurrent = io.path("recurrentStates");
            return version != SdxTextGenerationConfig.RECURRENT_STATE_FORMAT_VERSION
                    || recurrent.isArray() && !recurrent.isEmpty();
        } catch (IOException | RuntimeException invalid) {
            return false;
        }
    }

    private static boolean hasNonEmptyText(JsonNode object, String... fields) {
        for (String field : fields) {
            if (!object.path(field).isTextual() || object.path(field).asText().isBlank()) {
                return false;
            }
        }
        return true;
    }

    private static boolean hasNonEmptyTextArray(JsonNode object, String... fields) {
        for (String field : fields) {
            JsonNode values = object.path(field);
            if (!values.isArray() || values.isEmpty()) return false;
            for (JsonNode value : values) {
                if (!value.isTextual() || value.asText().isBlank()) return false;
            }
        }
        return true;
    }

    private static SdxTextGenerationConfig.Options generationOptions(
            Path tokenizerConfig, Path generationSource, GGMLMetadata metadata,
            int contextLength) throws IOException {
        JsonNode tokenizer = readOptionalObject(tokenizerConfig);
        JsonNode generation = readOptionalObject(generationSource);
        GGMLMetadata.TokenizerInfo tokenizerInfo = metadata.getTokenizerInfo();

        Integer bosId = firstNonNegativeInteger(
                generation.get("bos_token_id"), tokenizer.get("bos_token_id"));
        if (bosId == null && tokenizerInfo != null && tokenizerInfo.getBosTokenId() >= 0) {
            bosId = tokenizerInfo.getBosTokenId();
        }

        List<Integer> eosIds = firstTokenIds(
                generation.get("eos_token_id"), tokenizer.get("eos_token_id"));
        if (eosIds.isEmpty() && tokenizerInfo != null && tokenizerInfo.getEosTokenId() >= 0) {
            eosIds.add(tokenizerInfo.getEosTokenId());
        }
        if (eosIds.isEmpty()) {
            throw new IOException("Tokenizer metadata does not define a non-negative EOS token ID");
        }

        Integer padId = firstNonNegativeInteger(
                generation.get("pad_token_id"), tokenizer.get("pad_token_id"));
        if (padId == null) padId = eosIds.get(0);

        int maxNewTokens = boundedInteger(
                generation.get("max_new_tokens"), 128, 1, contextLength - 1);
        int minNewTokens = boundedInteger(
                generation.get("min_new_tokens"), 0, 0, maxNewTokens);

        return SdxTextGenerationConfig.Options.builder()
                .contextLength(contextLength)
                .maxPrefillLength(contextLength - 1)
                .bosId(bosId)
                .padId(padId)
                .eosIds(eosIds)
                .maxNewTokens(maxNewTokens)
                .minNewTokens(minNewTokens)
                .temperature(nonNegativeDouble(generation.get("temperature"), 0.0))
                .topK(boundedInteger(generation.get("top_k"), 0, 0, Integer.MAX_VALUE))
                .topP(boundedDouble(generation.get("top_p"), 1.0, 0.0, 1.0))
                .repetitionPenalty(positiveDouble(
                        generation.get("repetition_penalty"), 1.0))
                .seed(integerValue(generation.get("seed"), 0))
                .build();
    }

    private static JsonNode readOptionalObject(Path path) throws IOException {
        if (path == null || !Files.isRegularFile(path)) return MAPPER.createObjectNode();
        JsonNode value = MAPPER.readTree(path.toFile());
        if (value == null || !value.isObject()) {
            throw new IOException("Expected a JSON object in " + path);
        }
        return value;
    }

    private static Integer firstNonNegativeInteger(JsonNode... values) {
        for (JsonNode value : values) {
            if (value != null && value.isIntegralNumber() && value.canConvertToInt()
                    && value.intValue() >= 0) {
                return value.intValue();
            }
        }
        return null;
    }

    private static List<Integer> firstTokenIds(JsonNode... values) {
        for (JsonNode value : values) {
            LinkedHashSet<Integer> result = new LinkedHashSet<>();
            if (value != null && value.isIntegralNumber() && value.canConvertToInt()
                    && value.intValue() >= 0) {
                result.add(value.intValue());
            } else if (value != null && value.isArray()) {
                for (JsonNode element : value) {
                    if (element.isIntegralNumber() && element.canConvertToInt()
                            && element.intValue() >= 0) {
                        result.add(element.intValue());
                    }
                }
            }
            if (!result.isEmpty()) return new ArrayList<>(result);
        }
        return new ArrayList<>();
    }

    private static int boundedInteger(JsonNode value, int fallback, int minimum, int maximum) {
        long parsed = value != null && value.isIntegralNumber() ? value.longValue() : fallback;
        return (int) Math.max(minimum, Math.min(maximum, parsed));
    }

    private static long integerValue(JsonNode value, long fallback) {
        return value != null && value.isIntegralNumber() ? value.longValue() : fallback;
    }

    private static double nonNegativeDouble(JsonNode value, double fallback) {
        return boundedDouble(value, fallback, 0.0, Double.MAX_VALUE);
    }

    private static double positiveDouble(JsonNode value, double fallback) {
        double parsed = value != null && value.isNumber() ? value.doubleValue() : fallback;
        return Double.isFinite(parsed) && parsed > 0.0 ? parsed : fallback;
    }

    private static double boundedDouble(
            JsonNode value, double fallback, double minimum, double maximum) {
        double parsed = value != null && value.isNumber() ? value.doubleValue() : fallback;
        if (!Double.isFinite(parsed)) return fallback;
        return Math.max(minimum, Math.min(maximum, parsed));
    }

    private static GGMLMetadata inspect(Path source) throws IOException {
        try {
            return GGMLModelImport.inspectModel(source.toFile());
        } catch (GGMLImportException failure) {
            throw new IOException("Could not inspect GGUF metadata: " + source, failure);
        }
    }

    private static JsonNode parseObject(String json) throws IOException {
        if (json == null || json.isBlank()) return MAPPER.createObjectNode();
        JsonNode value = MAPPER.readTree(json);
        if (value == null || !value.isObject()) {
            throw new IllegalArgumentException("options JSON must be an object");
        }
        return value;
    }

    private static String configureDiagnostics(JsonNode options) {
        String mode = options.path("diagnosticMode").asText("standard").trim()
                .toLowerCase(Locale.ROOT);
        switch (mode) {
            case "standard":
                return mode;
            case "verbose":
                System.setProperty(ND4JSystemProperties.DSP_DIAGNOSTICS,
                        "COMPILE,EXECUTE,TIMING,MEMORY");
                System.setProperty(ND4JSystemProperties.DSP_DIAGNOSTICS_LEVEL, "detailed");
                return mode;
            case "dsp":
                System.setProperty(ND4JSystemProperties.DSP_DIAGNOSTICS, "ALL");
                System.setProperty(ND4JSystemProperties.DSP_DIAGNOSTICS_LEVEL, "full");
                return mode;
            default:
                throw new IllegalArgumentException("Unknown diagnosticMode: " + mode);
        }
    }

    private static String sha256(String value) {
        MessageDigest digest;
        try {
            digest = MessageDigest.getInstance("SHA-256");
        } catch (NoSuchAlgorithmException impossible) {
            throw new IllegalStateException("SHA-256 is unavailable", impossible);
        }
        StringBuilder result = new StringBuilder(64);
        for (byte item : digest.digest(value.getBytes(StandardCharsets.UTF_8))) {
            result.append(String.format(Locale.ROOT, "%02x", item & 0xff));
        }
        return result.toString();
    }

    private static Path firstRegular(Path first, Path second) {
        if (Files.isRegularFile(first)) return first;
        return Files.isRegularFile(second) ? second : null;
    }

    private static Path requireRegularFile(String value, String label) throws IOException {
        Path path = Path.of(requireText(value, label)).toAbsolutePath().normalize();
        if (!Files.isRegularFile(path)) throw new IOException(label + " is not a regular file: " + path);
        return path;
    }

    private static String requireText(String value, String label) {
        if (value == null || value.isBlank()) throw new IllegalArgumentException(label + " must not be blank");
        return value;
    }

    /**
     * Physical identity of the downloaded raw container. This is intentionally separate
     * from {@link SdxSourceIdentity}, whose digest is the logical identity of canonical
     * SameDiff .sdz/.sdnb sources.
     */
    static final class RawSourceIdentity {
        private final String sha256;
        private final long bytes;

        private RawSourceIdentity(String sha256, long bytes) {
            this.sha256 = sha256;
            this.bytes = bytes;
        }

        static RawSourceIdentity identify(Path source) throws IOException {
            MessageDigest digest;
            try {
                digest = MessageDigest.getInstance("SHA-256");
            } catch (NoSuchAlgorithmException impossible) {
                throw new IllegalStateException("SHA-256 is unavailable", impossible);
            }
            long bytes = 0;
            byte[] buffer = new byte[1024 * 1024];
            try (InputStream input = Files.newInputStream(source)) {
                int read;
                while ((read = input.read(buffer)) >= 0) {
                    if (read == 0) continue;
                    digest.update(buffer, 0, read);
                    bytes += read;
                }
            }
            StringBuilder sha256 = new StringBuilder(64);
            for (byte value : digest.digest()) {
                sha256.append(String.format(Locale.ROOT, "%02x", value & 0xff));
            }
            return new RawSourceIdentity(sha256.toString(), bytes);
        }

        String sha256() {
            return sha256;
        }

        long bytes() {
            return bytes;
        }
    }

    /** Validated, canonicalized conversion settings. Diagnostics are intentionally not cached. */
    static final class PreparationProfile {
        private final ConversionOptions.QuantizationMode conversionMode;
        private final ExportOptions.QuantizationType requantizeType;
        private final int kvQuantFormat;
        private final int tensorBatchSize;
        private final boolean useMemoryMapping;
        private final ObjectNode json;
        private final String sha256;

        private PreparationProfile(ConversionOptions.QuantizationMode conversionMode,
                                   ExportOptions.QuantizationType requantizeType,
                                   int kvQuantFormat, int tensorBatchSize,
                                   boolean useMemoryMapping) {
            this.conversionMode = conversionMode;
            this.requantizeType = requantizeType;
            this.kvQuantFormat = kvQuantFormat;
            this.tensorBatchSize = tensorBatchSize;
            this.useMemoryMapping = useMemoryMapping;
            ObjectNode canonical = MAPPER.createObjectNode();
            canonical.put("graphImportAbi", GRAPH_IMPORT_ABI);
            canonical.put("conversionMode", conversionMode.name());
            canonical.put("requantizeType", requantizeType == null ? "NONE" : requantizeType.name());
            // These graph-shaping fields participate in the cache identity so an older
            // FP32-embedding/full-logits bundle cannot be reopened after this profile changes.
            canonical.put("embeddingDataType", DataType.HALF.name());
            canonical.put("logitsMode", "LAST_POSITION_ONLY");
            canonical.put("kvQuantFormat", kvQuantFormat);
            canonical.put("tensorBatchSize", tensorBatchSize);
            canonical.put("useMemoryMapping", useMemoryMapping);
            this.json = canonical;
            this.sha256 = SdxGgufModelPreparer.sha256(canonical.toString());
        }

        static PreparationProfile from(JsonNode options) {
            String graphImportAbi = options.path("graphImportAbi").asText("");
            if (!GRAPH_IMPORT_ABI.equals(graphImportAbi)) {
                throw new IllegalArgumentException("graphImportAbi must be " + GRAPH_IMPORT_ABI
                        + " but was '" + graphImportAbi + "'");
            }
            String modeText = options.path("conversionMode")
                    .asText(ConversionOptions.QuantizationMode.DEQUANTIZE_TO_FLOAT16.name())
                    .trim().toUpperCase(Locale.ROOT);
            ConversionOptions.QuantizationMode mode;
            try {
                mode = ConversionOptions.QuantizationMode.valueOf(modeText);
            } catch (IllegalArgumentException invalid) {
                throw new IllegalArgumentException("Unknown conversionMode: " + modeText, invalid);
            }
            String requantizeText = options.path("requantizeType").asText("NONE")
                    .trim().toUpperCase(Locale.ROOT);
            ExportOptions.QuantizationType requantize = null;
            if (!requantizeText.equals("NONE")) {
                try {
                    requantize = ExportOptions.QuantizationType.valueOf(requantizeText);
                } catch (IllegalArgumentException invalid) {
                    throw new IllegalArgumentException("Unknown requantizeType: " + requantizeText,
                            invalid);
                }
                if (requantize != ExportOptions.QuantizationType.Q4_K
                        && requantize != ExportOptions.QuantizationType.Q6_K
                        && requantize != ExportOptions.QuantizationType.Q8_0) {
                    throw new IllegalArgumentException(
                            "Mobile packed execution supports requantizeType Q4_K, Q6_K, or Q8_0");
                }
                mode = requantize == ExportOptions.QuantizationType.Q8_0
                        ? ConversionOptions.QuantizationMode.RUNTIME_QUANTIZED_INT8
                        : ConversionOptions.QuantizationMode.RUNTIME_QUANTIZED_MATMUL;
            }
            int kv = options.path("kvQuantFormat").asInt(0);
            if (kv < 0 || kv > 4) {
                throw new IllegalArgumentException("kvQuantFormat must be between 0 and 4");
            }
            int batch = options.path("tensorBatchSize").asInt(10);
            if (batch < 1 || batch > 256) {
                throw new IllegalArgumentException("tensorBatchSize must be between 1 and 256");
            }
            boolean mmap = options.path("useMemoryMapping").asBoolean(true);
            return new PreparationProfile(mode, requantize, kv, batch, mmap);
        }

        ConversionOptions conversionOptions() {
            return ConversionOptions.builder()
                    .quantizationMode(conversionMode)
                    .targetDataType(targetDataType(conversionMode))
                    .embeddingDataType(DataType.HALF)
                    .lastPositionLogitsOnly(true)
                    .forTraining(false)
                    .preserveTokenizerInfo(true)
                    .kvQuantFormat(kvQuantFormat)
                    .tensorBatchSize(tensorBatchSize)
                    .useMemoryMapping(useMemoryMapping)
                    .build();
        }

        private static DataType targetDataType(ConversionOptions.QuantizationMode mode) {
            switch (mode) {
                case DEQUANTIZE_TO_FLOAT32:
                case RUNTIME_QUANTIZED_MATMUL:
                case RUNTIME_QUANTIZED_INT8:
                case RUNTIME_QUANTIZED_INT4:
                    // Packed QMatMul produces FP32 activations. Keep graph state, including
                    // prefill/decode KV tensors, in the same dtype for the native SDX contract.
                    return DataType.FLOAT;
                case DEQUANTIZE_TO_BFLOAT16:
                    return DataType.BFLOAT16;
                case DEQUANTIZE_TO_FLOAT8_E4M3:
                    return DataType.FLOAT8;
                case DEQUANTIZE_TO_FLOAT8_E5M2:
                    return DataType.FLOAT8_E5M2;
                default:
                    return DataType.HALF;
            }
        }

        Path existingOptimizedSource(Path source, Path preparedRoot) {
            if (requantizeType == null) return source;
            Path optimized = optimizedPath(preparedRoot);
            return Files.isRegularFile(optimized) ? optimized : source;
        }

        Path materializeOptimizedSource(Path source, Path preparedRoot, Path temporaryRoot)
                throws IOException {
            if (requantizeType == null) return source;
            Path optimized = optimizedPath(preparedRoot);
            if (Files.isRegularFile(optimized)) return optimized;
            Path temporary = Files.createTempFile(temporaryRoot, "requantize-", ".gguf");
            boolean published = false;
            try {
                GGMLModelExport.requantize(source.toFile(), temporary.toFile(), requantizeType);
                try {
                    Files.move(temporary, optimized, StandardCopyOption.ATOMIC_MOVE);
                } catch (java.nio.file.AtomicMoveNotSupportedException unsupported) {
                    Files.move(temporary, optimized);
                }
                published = true;
                return optimized;
            } catch (GGMLExportException failure) {
                throw new IOException("Could not create " + requantizeType
                        + " optimized GGUF derivative from " + source, failure);
            } finally {
                if (!published) Files.deleteIfExists(temporary);
            }
        }

        private Path optimizedPath(Path preparedRoot) {
            return preparedRoot.resolve("optimized-"
                    + requantizeType.name().toLowerCase(Locale.ROOT) + ".gguf");
        }

        ObjectNode json() {
            return json;
        }

        String sha256() {
            return sha256;
        }
    }

    private static final class TokenizerAssets {
        private final Path tokenizer;
        private final Path tokenizerConfig;
        private final Path textGenerationConfig;
        private final int contextLength;
        private final SdxTextGenerationConfig.Options generationOptions;

        private TokenizerAssets(Path tokenizer, Path tokenizerConfig,
                                Path textGenerationConfig, int contextLength,
                                SdxTextGenerationConfig.Options generationOptions) {
            this.tokenizer = tokenizer;
            this.tokenizerConfig = tokenizerConfig;
            this.textGenerationConfig = textGenerationConfig;
            this.contextLength = contextLength;
            this.generationOptions = generationOptions;
        }
    }
}
