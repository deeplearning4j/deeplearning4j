/*
 * Copyright (c) Eclipse Deeplearning4j
 * SPDX-License-Identifier: Apache-2.0
 */
package org.eclipse.deeplearning4j.sdx.aot;

import org.nd4j.dsp.model.SdxCompiledModel;
import org.nd4j.dsp.model.SdxModelCache;
import org.nd4j.dsp.model.SdxModelCompiler;
import org.nd4j.dsp.model.SdxPlatformProviderDescriptor;
import org.nd4j.dsp.model.SdxSourceIdentity;
import org.nd4j.dsp.model.SdxTargetProfile;
import org.nd4j.dsp.model.SdxTextModelAssets;
import org.nd4j.ggml.GGMLImportException;
import org.nd4j.ggml.GGMLModelImport;
import org.nd4j.ggml.format.GGMLMetadata;
import org.nd4j.shade.jackson.databind.JsonNode;
import org.nd4j.shade.jackson.databind.ObjectMapper;
import org.nd4j.shade.jackson.databind.node.ObjectNode;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Backend-neutral raw-model preparation used by every {@code libsdx_llm} consumer.
 * Import, source attestation, immutable cache admission and text-asset validation are
 * shared; only the {@link SdxTargetProfile} compiler/provider policy is backend specific.
 */
final class SdxGgufModelPreparer {
    static final String PREPARED_SCHEMA = "sdx-prepared-text-model-v2";
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

        JsonNode options = parseObject(optionsJson);
        SdxSourceIdentity sourceIdentity = SdxSourceIdentity.identify(source);
        verifyAttestation(sourceIdentity, options);

        Path preparedRoot = cache.root().resolve("prepared").resolve(sourceIdentity.sha256());
        Path canonicalPointer = preparedRoot.resolve("canonical.path");
        Path canonical = readCanonicalPointer(cache, canonicalPointer);
        if (canonical != null) {
            try {
                SdxCompiledModel cached = cache.resolve(canonical, target);
                return preparedJson(sourceIdentity, canonical, cached, true,
                        contextLength(source), target);
            } catch (SdxModelCache.MissingCompiledModelException missingTarget) {
                // The canonical import is reusable; compile only the missing target below.
            }
        }

        Files.createDirectories(preparedRoot);
        if (canonical == null) {
            Path temporaryRoot = cache.root().resolve("tmp");
            Files.createDirectories(temporaryRoot);
            Path generated = Files.createTempFile(temporaryRoot, "gguf-import-", ".sdz");
            boolean admitted = false;
            try {
                convertToSdz(source, generated);
                canonical = cache.admitGeneratedSource(generated);
                admitted = true;
            } finally {
                if (!admitted) Files.deleteIfExists(generated);
            }
            writeCanonicalPointer(canonicalPointer, canonical);
        }

        TokenizerAssets tokenizerAssets = materializeTokenizerAssets(
                source, tokenizerPath, preparedRoot.resolve("text-assets"));
        SdxPlatformProviderDescriptor provider = target.platformProvider();
        String targetSoc = provider.defaultTargetSoc();
        SdxModelCompiler compiler = new SdxModelCompiler(cache);
        SdxModelCompiler.CompileOptions compileOptions = SdxModelCompiler.CompileOptions.builder()
                .tokenizer(tokenizerAssets.tokenizer)
                .tokenizerConfig(tokenizerAssets.tokenizerConfig)
                .textGenerationConfig(tokenizerAssets.textGenerationConfig)
                .targetSoc(targetSoc)
                .modelId(sourceIdentity.sha256())
                .cacheKeyProperty("sourceFormat", "gguf")
                .build();
        SdxCompiledModel compiled = compiler.compile(
                canonical,
                target,
                SdxModelCompiler.requireBuiltInTargetCompiler(target, targetSoc, false),
                compileOptions);
        return preparedJson(sourceIdentity, canonical, compiled, false,
                tokenizerAssets.contextLength, target);
    }

    static String resolve(String sourceSdz, String targetProfile, String cacheDirectory)
            throws IOException {
        Path source = requireRegularFile(sourceSdz, "source SDZ");
        SdxTargetProfile target = SdxTargetProfile.fromId(requireText(targetProfile, "target profile"));
        SdxModelCache cache = new SdxModelCache(Path.of(requireText(cacheDirectory, "cache directory")));
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

    private static String preparedJson(SdxSourceIdentity sourceIdentity, Path canonical,
                                       SdxCompiledModel compiled, boolean cacheHit,
                                       int contextLength, SdxTargetProfile target)
            throws IOException {
        SdxTextModelAssets assets = compiled.requireTextModelAssets();
        SdxPlatformProviderDescriptor provider = target.platformProvider();
        ObjectNode result = MAPPER.createObjectNode();
        result.put("schema", PREPARED_SCHEMA);
        result.put("cacheHit", cacheHit);
        result.put("sourceSha256", sourceIdentity.sha256());
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

    private static void verifyAttestation(SdxSourceIdentity identity, JsonNode options) {
        if (options.hasNonNull("verifiedSourceSha256")) {
            String expected = options.get("verifiedSourceSha256").asText().toLowerCase(Locale.ROOT);
            if (!identity.sha256().equals(expected)) {
                throw new IllegalArgumentException("Verified source SHA-256 did not match the GGUF bytes");
            }
        }
        if (options.hasNonNull("verifiedSourceBytes")
                && identity.logicalBytes() != options.get("verifiedSourceBytes").asLong()) {
            throw new IllegalArgumentException("Verified source byte count did not match the GGUF bytes");
        }
    }

    @SuppressWarnings("unchecked")
    private static TokenizerAssets materializeTokenizerAssets(
            Path source, String explicitTokenizerPath, Path destination) throws IOException {
        Files.createDirectories(destination);
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
        int contextLength = contextLength(source);
        GGMLMetadata metadata = null;

        if (Files.isRegularFile(tokenizerSource)) {
            Files.copy(tokenizerSource, tokenizer, StandardCopyOption.REPLACE_EXISTING);
            if (!Files.isRegularFile(configSource)) {
                throw new IOException("tokenizer_config.json is required beside " + tokenizerSource);
            }
            Files.copy(configSource, tokenizerConfig, StandardCopyOption.REPLACE_EXISTING);
        } else {
            metadata = inspect(source);
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
            ObjectNode generated = MAPPER.createObjectNode();
            generated.put("bos_token_id", bosId);
            generated.put("eos_token_id", eosId);
            Files.writeString(textGeneration, generated.toString(), StandardCharsets.UTF_8);
        }

        if (!Files.isRegularFile(textGeneration)) {
            if (generationSource != null) {
                Files.copy(generationSource, textGeneration, StandardCopyOption.REPLACE_EXISTING);
            } else {
                Files.writeString(textGeneration, "{}", StandardCharsets.UTF_8);
            }
        }
        return new TokenizerAssets(tokenizer, tokenizerConfig, textGeneration, contextLength);
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
        GGMLMetadata metadata = inspect(source);
        for (Map.Entry<String, Object> entry : metadata.getRawMetadata().entrySet()) {
            if (entry.getKey().endsWith(".context_length") && entry.getValue() instanceof Number) {
                int value = ((Number) entry.getValue()).intValue();
                if (value > 1) return value;
            }
        }
        return 4096;
    }

    private static void convertToSdz(Path source, Path destination) throws IOException {
        try {
            GGMLModelImport.convertToSDZ(source.toFile(), destination.toFile());
        } catch (GGMLImportException failure) {
            throw new IOException("Could not import GGUF into canonical SDZ: " + source, failure);
        }
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

    private static final class TokenizerAssets {
        private final Path tokenizer;
        private final Path tokenizerConfig;
        private final Path textGenerationConfig;
        private final int contextLength;

        private TokenizerAssets(Path tokenizer, Path tokenizerConfig,
                                Path textGenerationConfig, int contextLength) {
            this.tokenizer = tokenizer;
            this.tokenizerConfig = tokenizerConfig;
            this.textGenerationConfig = textGenerationConfig;
            this.contextLength = contextLength;
        }
    }
}
