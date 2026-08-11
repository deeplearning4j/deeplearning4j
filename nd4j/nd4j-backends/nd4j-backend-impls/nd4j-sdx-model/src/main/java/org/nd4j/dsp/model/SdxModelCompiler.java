/*
 * Copyright (c) Eclipse Deeplearning4j
 * SPDX-License-Identifier: Apache-2.0
 */
package org.nd4j.dsp.model;

import java.io.BufferedInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.TreeMap;
import java.util.UUID;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * Mainline SDX compile-through-cache API.
 *
 * <p>A caller provides one canonical SDZ and a target profile. The target
 * compiler is invoked only on a content-addressed miss; its provider artifact is
 * copied into an immutable cache object and may be embedded back into an SDZ.
 * Applications never select or parse provider file formats.</p>
 */
public final class SdxModelCompiler {
    public static final String COMPILER_ABI = "sdx-model-compiler-v5";

    private final SdxModelCache cache;

    public SdxModelCompiler(SdxModelCache cache) {
        this.cache = Objects.requireNonNull(cache, "cache");
    }

    public SdxCompiledModel compile(
            Path sourceModel,
            SdxTargetProfile target,
            TargetCompiler compiler) throws IOException {
        return compile(sourceModel, target, compiler, CompileOptions.builder().build());
    }

    public SdxCompiledModel compile(
            Path sourceModel,
            SdxTargetProfile target,
            TargetCompiler compiler,
            CompileOptions options) throws IOException {
        Objects.requireNonNull(sourceModel, "sourceModel");
        Objects.requireNonNull(target, "target");
        Objects.requireNonNull(compiler, "compiler");
        Objects.requireNonNull(options, "options");

        Path source = sourceModel.toAbsolutePath().normalize();
        SdxSourceIdentity identity = cache.identify(source);
        Path cachedSource = cache.ensureSourceCached(source, identity);
        CompileInputSnapshot snapshot = createInputSnapshot(
                cachedSource,
                identity,
                options,
                compiler.requiresIsolatedSourceSnapshot());
        try {
            CompileOptions compileOptions = snapshot.options();
            SdxQuantizationContract quantization =
                    compileOptions.quantizationConfig == null
                            ? null
                            : SdxQuantizationContract.load(
                                    compileOptions.quantizationConfig);
            if (quantization != null) {
                quantization.validateForCompilation(
                        identity, target, compileOptions.targetSoc());
            }
            String compileKey = compileKey(
                    snapshot.sourceModel(),
                    identity,
                    target,
                    compiler,
                    compileOptions);

            return cache.withCompileLock(compileKey, () -> {
            java.util.Optional<SdxCompiledModel> existing =
                    cache.resolveByCompileKey(source, identity, target, compileKey);
            if (existing.isPresent()) {
                return existing.get();
            }

            Path staging = cache.newStagingDirectory(compileKey);
            Path work = staging.resolveSibling(
                    staging.getFileName() + "." + UUID.randomUUID() + ".work");
            Files.createDirectory(work);
            try {
                Path suggestedOutput = suggestedCompilerOutput(work, target);
                Path suggestedModelOutput = suggestedCompilerModelOutput(work);
                CompilationContext context = new CompilationContext(
                        snapshot.sourceModel(),
                        identity,
                        target,
                        work,
                        suggestedOutput,
                        suggestedModelOutput,
                        compileOptions);
                SdxPlatformProviderDescriptor provider = target.platformProvider();
                Path compiledArtifact = compiler.compile(context);
                Path payloadRoot = staging.resolve(
                        target.runtimeKind() == SdxTargetProfile.RuntimeKind.SDX_BUNDLE
                                ? "bundle"
                                : "payload");
                Path artifactDestination = null;
                if (provider.requiresAotArtifact()) {
                    if (compiledArtifact == null) {
                        throw new IOException(
                                "SDX target compiler returned no artifact for " + target.id());
                    }
                    Path artifact = compiledArtifact.toAbsolutePath().normalize();
                    artifactDestination = payloadRoot.resolve(target.artifactRelativePath());
                    copyArtifact(artifact, artifactDestination, target.artifactKind());
                } else {
                    if (target.artifactKind() != SdxTargetProfile.ArtifactKind.NONE) {
                        throw new IOException(
                                "Runtime-specialized target declares a bundle-owned artifact: "
                                        + target.id());
                    }
                    if (compiledArtifact != null) {
                        throw new IOException(
                                "Runtime-specialized target " + target.id()
                                        + " must not emit a model-owned accelerator artifact");
                    }
                }
                snapshot.verifyUnchanged(identity);

                SdxSourceIdentity compiledModelIdentity = copyOptionalCompiledModel(
                        context.suggestedModelOutput(), payloadRoot);
                if (quantization != null
                        && target == SdxTargetProfile.ANDROID_ARM64_NNAPI_ACCELERATOR) {
                    if (compiledModelIdentity == null) {
                        throw new IOException(
                                "NNAPI quantization must emit the derived SDZ at "
                                        + context.suggestedModelOutput());
                    }
                    if (identity.sha256().equals(compiledModelIdentity.sha256())) {
                        throw new IOException(
                                "NNAPI quantization emitted an unchanged derived SDZ; "
                                        + "a real graph rewrite is required");
                    }
                }

                // Validate the private staged copy. A compiler output can be
                // concurrently replaced; only bytes admitted to the immutable
                // cache are authoritative.
                if (artifactDestination != null) {
                    validateArtifact(
                            target,
                            artifactDestination,
                            identity,
                            compiledModelIdentity,
                            compileOptions);
                    if (quantization != null) {
                        quantization.validateArtifact(artifactDestination);
                    }
                }

                String runtimeModelRelative = compiledModelIdentity == null
                        ? "../../../sources/" + identity.sha256()
                                + "/" + identity.sourceFileName()
                        : "graph/model.sdz";

                String tokenizerRelative = copyOptionalAsset(
                        compileOptions.tokenizer,
                        payloadRoot,
                        "assets/tokenizer/tokenizer.json",
                        staging);
                String tokenizerConfigRelative = copyOptionalAsset(
                        compileOptions.tokenizerConfig,
                        payloadRoot,
                        "assets/tokenizer/tokenizer_config.json",
                        staging);
                String textConfigRelative = copyOptionalAsset(
                        compileOptions.textGenerationConfig,
                        payloadRoot,
                        "metadata/text-generation.json",
                        staging);
                String quantizationRelative = copyOptionalAsset(
                        compileOptions.quantizationConfig,
                        payloadRoot,
                        "metadata/quantization.json",
                        staging);

                String runtimeRelative;
                if (target.runtimeKind() == SdxTargetProfile.RuntimeKind.SDX_BUNDLE) {
                    writeBundleManifest(
                            payloadRoot,
                            identity,
                            target,
                            compileKey,
                            compiler,
                            runtimeModelRelative,
                            compiledModelIdentity,
                            tokenizerRelative,
                            textConfigRelative,
                            quantizationRelative,
                            compileOptions);
                    runtimeRelative = "bundle";
                } else {
                    runtimeRelative =
                            "payload/" + target.artifactRelativePath();
                }

                return cache.publish(
                        source,
                        identity,
                        target,
                        compileKey,
                        compiler.id(),
                        compiler.version(),
                        staging,
                        runtimeRelative,
                        tokenizerRelative,
                        tokenizerConfigRelative,
                        textConfigRelative,
                        quantizationRelative);
            } catch (Throwable failure) {
                SdxModelCache.deleteTree(staging);
                if (failure instanceof IOException) {
                    throw (IOException) failure;
                }
                throw new IOException(
                        "SDX compilation failed for target " + target.id(), failure);
            } finally {
                SdxModelCache.deleteTree(work);
            }
            });
        } finally {
            snapshot.close();
        }
    }

    private static CompileInputSnapshot createInputSnapshot(
            Path cachedSource,
            SdxSourceIdentity identity,
            CompileOptions options,
            boolean isolateSource) throws IOException {
        Path root = cachedSource.resolveSibling(
                "." + UUID.randomUUID() + ".compiler-inputs");
        Files.createDirectory(root);
        try {
            Path source = cachedSource;
            if (isolateSource) {
                source = root.resolve(identity.sourceFileName());
                Files.copy(cachedSource, source);
                SdxSourceIdentity copiedSource = SdxSourceIdentity.identify(source);
                if (!identity.sha256().equals(copiedSource.sha256())
                        || identity.logicalBytes() != copiedSource.logicalBytes()) {
                    throw new IOException(
                            "Canonical SDZ cache does not match the requested source identity");
                }
            }

            Path tokenizer = copySnapshotInput(root, "tokenizer", options.tokenizer);
            Path tokenizerConfig = copySnapshotInput(
                    root, "tokenizer-config", options.tokenizerConfig);
            Path textConfig = copySnapshotInput(
                    root, "text-generation", options.textGenerationConfig);
            Path quantization = copySnapshotInput(
                    root, "quantization", options.quantizationConfig);
            CompileOptions.Builder builder = CompileOptions.builder()
                    .modelId(options.modelId)
                    .targetSoc(options.targetSoc);
            if (tokenizer != null) {
                builder.tokenizer(tokenizer);
            }
            if (tokenizerConfig != null) {
                builder.tokenizerConfig(tokenizerConfig);
            }
            if (textConfig != null) {
                builder.textGenerationConfig(textConfig);
            }
            if (quantization != null) {
                builder.quantizationConfig(quantization);
            }
            for (Map.Entry<String, String> entry
                    : options.cacheKeyProperties.entrySet()) {
                builder.cacheKeyProperty(entry.getKey(), entry.getValue());
            }

            CompileOptions snapshottedOptions = builder.build();
            Map<Path, String> fingerprints = new LinkedHashMap<>();
            addSnapshotFingerprint(fingerprints, tokenizer);
            addSnapshotFingerprint(fingerprints, tokenizerConfig);
            addSnapshotFingerprint(fingerprints, textConfig);
            addSnapshotFingerprint(fingerprints, quantization);
            return new CompileInputSnapshot(
                    root, source, snapshottedOptions, fingerprints, isolateSource);
        } catch (IOException | RuntimeException | Error failure) {
            try {
                SdxModelCache.deleteTree(root);
            } catch (IOException cleanupFailure) {
                failure.addSuppressed(cleanupFailure);
            }
            throw failure;
        }
    }

    private static Path copySnapshotInput(Path root, String label, Path input)
            throws IOException {
        if (input == null) {
            return null;
        }
        Path source = input.toAbsolutePath().normalize();
        if (Files.isSymbolicLink(source)
                || !Files.isRegularFile(source)
                || Files.size(source) <= 0L) {
            throw new IOException("Compiler input is missing, empty, or a symlink: " + source);
        }
        Path directory = root.resolve(label);
        Files.createDirectory(directory);
        Path destination = directory.resolve(source.getFileName().toString());
        Files.copy(source, destination);
        return destination;
    }

    private static void addSnapshotFingerprint(
            Map<Path, String> fingerprints, Path input) throws IOException {
        if (input != null) {
            fingerprints.put(input, inputFingerprint(input));
        }
    }

    private static String inputFingerprint(Path input) throws IOException {
        if (!Files.isRegularFile(input) || Files.isSymbolicLink(input)) {
            throw new IOException("Snapshotted compiler input is unavailable: " + input);
        }
        return Files.size(input) + ":" + SdxSourceIdentity.sha256(input);
    }

    private static final class CompileInputSnapshot implements AutoCloseable {
        private final Path root;
        private final Path sourceModel;
        private final CompileOptions options;
        private final Map<Path, String> fingerprints;
        private final boolean verifySourceIdentity;

        private CompileInputSnapshot(
                Path root,
                Path sourceModel,
                CompileOptions options,
                Map<Path, String> fingerprints,
                boolean verifySourceIdentity) {
            this.root = root;
            this.sourceModel = sourceModel;
            this.options = options;
            this.fingerprints = fingerprints;
            this.verifySourceIdentity = verifySourceIdentity;
        }

        private Path sourceModel() {
            return sourceModel;
        }

        private CompileOptions options() {
            return options;
        }

        private void verifyUnchanged(SdxSourceIdentity expectedSource)
                throws IOException {
            if (verifySourceIdentity) {
                SdxSourceIdentity current = SdxSourceIdentity.identify(sourceModel);
                if (!expectedSource.sha256().equals(current.sha256())
                        || expectedSource.logicalBytes() != current.logicalBytes()) {
                    throw new IOException("Target compiler modified its canonical SDZ input");
                }
            }
            for (Map.Entry<Path, String> entry : fingerprints.entrySet()) {
                if (!entry.getValue().equals(inputFingerprint(entry.getKey()))) {
                    throw new IOException(
                            "Target compiler modified a snapshotted input: "
                                    + entry.getKey());
                }
            }
        }

        @Override
        public void close() throws IOException {
            SdxModelCache.deleteTree(root);
        }
    }

    public String compileKey(
            Path sourceModel,
            SdxSourceIdentity identity,
            SdxTargetProfile target,
            TargetCompiler compiler,
            CompileOptions options) throws IOException {
        TreeMap<String, String> material = new TreeMap<>();
        material.put("compilerAbi", COMPILER_ABI);
        material.put("cacheAbi", SdxModelCache.CACHE_ABI);
        material.put("sourceSha256", identity.sha256());
        material.put("sourceLogicalBytes", Long.toString(identity.logicalBytes()));
        material.put("target", target.id());
        SdxPlatformProviderDescriptor provider = target.platformProvider();
        material.put("providerId", provider.providerId());
        material.put("providerAbiVersion", Integer.toString(provider.providerAbiVersion()));
        material.put("artifactFormat", provider.artifactFormat());
        material.put(
                "artifactFormatVersion",
                Integer.toString(provider.artifactFormatVersion()));
        material.put("compilerId", requireSingleLine(compiler.id(), "compiler id"));
        material.put("compilerVersion", requireSingleLine(
                compiler.version(), "compiler version"));
        material.put("compilerMaterial", requireSingleLine(
                compiler.cacheKeyMaterial(sourceModel, target, options),
                "compiler cache-key material"));
        addOptionalDigest(material, "tokenizer", options.tokenizer);
        addOptionalDigest(material, "tokenizerConfig", options.tokenizerConfig);
        addOptionalDigest(
                material, "textGenerationConfig", options.textGenerationConfig);
        addOptionalDigest(
                material, "quantizationConfig", options.quantizationConfig);
        if (options.modelId != null) {
            material.put("modelId", requireSingleLine(options.modelId, "model id"));
        }
        if (options.targetSoc != null) {
            material.put(
                    "targetSoc",
                    requireSingleLine(options.targetSoc, "target SoC"));
        }
        for (Map.Entry<String, String> entry : options.cacheKeyProperties.entrySet()) {
            material.put(
                    "option." + requireKey(entry.getKey()),
                    requireSingleLine(entry.getValue(), "compiler option value"));
        }

        MessageDigest digest = sha256Digest();
        for (Map.Entry<String, String> entry : material.entrySet()) {
            digest.update(entry.getKey().getBytes(StandardCharsets.UTF_8));
            digest.update((byte) '=');
            digest.update(entry.getValue().getBytes(StandardCharsets.UTF_8));
            digest.update((byte) '\n');
        }
        return SdxSourceIdentity.hex(digest.digest());
    }

    private static void addOptionalDigest(
            Map<String, String> material, String name, Path path) throws IOException {
        if (path == null) {
            material.put(name, "none");
            return;
        }
        Path input = path.toAbsolutePath().normalize();
        if (!Files.isRegularFile(input) || Files.size(input) <= 0L) {
            throw new IOException("SDX compiler input is missing or empty: " + input);
        }
        material.put(name + ".sha256", SdxSourceIdentity.sha256(input));
        material.put(name + ".bytes", Long.toString(Files.size(input)));
    }

    private static Path suggestedCompilerOutput(
            Path work, SdxTargetProfile target) throws IOException {
        Path output = target.hasPackagedArtifact()
                ? work.resolve("compiled").resolve(target.artifactRelativePath())
                : work.resolve("compiled").resolve("runtime-specialization.no-artifact");
        Files.createDirectories(output.getParent());
        return output;
    }

    private static Path suggestedCompilerModelOutput(Path work) throws IOException {
        Path output = work.resolve("compiled").resolve("graph/model.sdz");
        Files.createDirectories(output.getParent());
        return output;
    }

    private static SdxSourceIdentity copyOptionalCompiledModel(
            Path input, Path payloadRoot) throws IOException {
        if (!Files.exists(input)) {
            return null;
        }
        if (!Files.isRegularFile(input) || Files.size(input) <= 0L) {
            throw new IOException("Target compiler emitted an invalid derived SDZ: " + input);
        }
        SdxSourceIdentity identity = SdxSourceIdentity.identify(input);
        Path destination = payloadRoot.resolve("graph/model.sdz");
        Files.createDirectories(destination.getParent());
        Files.copy(input, destination, StandardCopyOption.REPLACE_EXISTING);
        return identity;
    }

    private static String copyOptionalAsset(
            Path input,
            Path payloadRoot,
            String payloadRelative,
            Path stagingRoot) throws IOException {
        if (input == null) {
            return null;
        }
        Path source = input.toAbsolutePath().normalize();
        if (!Files.isRegularFile(source) || Files.size(source) <= 0L) {
            throw new IOException("SDX metadata asset is missing or empty: " + source);
        }
        Path destination = payloadRoot.resolve(payloadRelative);
        Files.createDirectories(destination.getParent());
        Files.copy(source, destination, StandardCopyOption.REPLACE_EXISTING);
        return portable(stagingRoot.relativize(destination));
    }

    private static void copyArtifact(
            Path source,
            Path destination,
            SdxTargetProfile.ArtifactKind kind) throws IOException {
        if (kind == SdxTargetProfile.ArtifactKind.FILE) {
            Files.createDirectories(destination.getParent());
            Files.copy(source, destination, StandardCopyOption.REPLACE_EXISTING);
            return;
        }

        Path destinationRoot = destination.toAbsolutePath().normalize();
        Files.createDirectories(destinationRoot);
        try (Stream<Path> stream = Files.walk(source)) {
            List<Path> paths = stream.sorted().collect(Collectors.toList());
            for (Path path : paths) {
                if (Files.isSymbolicLink(path)) {
                    throw new IOException(
                            "SDX compiler output contains a symlink: " + path);
                }
                Path relative = source.relativize(path);
                Path target = destinationRoot.resolve(relative.toString()).normalize();
                if (!target.startsWith(destinationRoot)) {
                    throw new IOException("SDX compiler output escapes its cache object");
                }
                if (Files.isDirectory(path)) {
                    Files.createDirectories(target);
                } else if (Files.isRegularFile(path)) {
                    Files.createDirectories(target.getParent());
                    Files.copy(path, target, StandardCopyOption.REPLACE_EXISTING);
                }
            }
        }
    }

    private static void validateArtifact(
            SdxTargetProfile target,
            Path artifact,
            SdxSourceIdentity sourceIdentity,
            SdxSourceIdentity derivedModelIdentity,
            CompileOptions options) throws IOException {
        if (target.artifactKind() == SdxTargetProfile.ArtifactKind.FILE) {
            if (!Files.isRegularFile(artifact) || Files.size(artifact) <= 0L) {
                throw new IOException(
                        "Target compiler produced no file for " + target.id());
            }
            if (target == SdxTargetProfile.ANDROID_ARM64_GOOGLE_TENSOR_G5) {
                if (!artifact.getFileName().toString()
                        .toLowerCase(java.util.Locale.ROOT)
                        .endsWith(".litertlm")) {
                    throw new IOException(
                            "Tensor G5 compiler artifact must use the .litertlm extension: "
                                    + artifact);
                }
                LiteRtLmPackageValidator.validate(artifact);
            } else if (target
                    == SdxTargetProfile.ANDROID_ARM64_NNAPI_ACCELERATOR) {
                SdxNnapiDevicePolicy.load(artifact).validateFor(
                        target,
                        options.targetSoc(),
                        sourceIdentity,
                        derivedModelIdentity);
            }
            return;
        }

        if (!Files.isDirectory(artifact)) {
            throw new IOException(
                    "Target compiler produced no artifact directory for " + target.id());
        }

        if (target == SdxTargetProfile.ANDROID_ARM64_VULKAN) {
            validateVulkanDirectory(artifact);
        } else if (target == SdxTargetProfile.ANDROID_ARM64_HEXAGON_HTP) {
            validateHexagonDirectory(artifact);
        } else if (isDirectoryEmpty(artifact)) {
            throw new IOException(
                    "Target compiler produced an empty directory for " + target.id());
        }
    }

    private static void validateVulkanDirectory(Path directory) throws IOException {
        List<Path> spirv = filesWithSuffix(directory, ".spv");
        if (spirv.isEmpty()) {
            throw new IOException("Vulkan compiler produced no SPIR-V modules");
        }
        for (Path module : spirv) {
            if (Files.size(module) < 20L || Files.size(module) % 4L != 0L) {
                throw new IOException("Invalid SPIR-V byte length: " + module);
            }
            byte[] magic = new byte[4];
            try (InputStream input = Files.newInputStream(module)) {
                if (input.read(magic) != magic.length
                        || magic[0] != 0x03
                        || magic[1] != 0x02
                        || magic[2] != 0x23
                        || magic[3] != 0x07) {
                    throw new IOException("Invalid SPIR-V magic: " + module);
                }
            }
            Path metadata = replaceSuffix(module, ".spv", ".meta");
            String text = readSmallText(metadata);
            if (!text.contains("cacheAbi=vulkan-spirv-disk-cache-v2")) {
                throw new IOException(
                        "Vulkan artifact has no supported cache ABI: " + metadata);
            }
        }
    }

    private static void validateHexagonDirectory(Path directory) throws IOException {
        List<Path> kernels = filesWithSuffix(directory, ".bin");
        if (kernels.isEmpty()) {
            throw new IOException("Hexagon compiler produced no AOT kernels");
        }
        for (Path kernel : kernels) {
            if (Files.size(kernel) <= 0L) {
                throw new IOException("Hexagon compiler produced an empty kernel: " + kernel);
            }
            Path metadata = replaceSuffix(kernel, ".bin", ".meta");
            String text = readSmallText(metadata);
            if (!text.contains("cacheAbi=sdx-hexagon-aot-v1")) {
                throw new IOException(
                        "Hexagon artifact has no supported cache ABI: " + metadata);
            }
        }
    }

    private static List<Path> filesWithSuffix(Path directory, String suffix)
            throws IOException {
        try (Stream<Path> stream = Files.walk(directory)) {
            return stream.filter(Files::isRegularFile)
                    .filter(path -> path.getFileName().toString().endsWith(suffix))
                    .sorted()
                    .collect(Collectors.toList());
        }
    }

    private static boolean isDirectoryEmpty(Path directory) throws IOException {
        try (Stream<Path> stream = Files.walk(directory)) {
            return stream.noneMatch(Files::isRegularFile);
        }
    }

    private static Path replaceSuffix(Path input, String oldSuffix, String newSuffix)
            throws IOException {
        String name = input.getFileName().toString();
        if (!name.endsWith(oldSuffix)) {
            throw new IOException("Unexpected artifact suffix: " + input);
        }
        Path result = input.resolveSibling(
                name.substring(0, name.length() - oldSuffix.length()) + newSuffix);
        if (!Files.isRegularFile(result)) {
            throw new IOException("Artifact metadata sidecar is missing: " + result);
        }
        return result;
    }

    private static String readSmallText(Path path) throws IOException {
        if (!Files.isRegularFile(path) || Files.size(path) > 1024 * 1024L) {
            throw new IOException("Artifact metadata is missing or too large: " + path);
        }
        return Files.readString(path, StandardCharsets.UTF_8);
    }

    private static void writeBundleManifest(
            Path bundleRoot,
            SdxSourceIdentity identity,
            SdxTargetProfile target,
            String compileKey,
            TargetCompiler compiler,
            String runtimeModelRelative,
            SdxSourceIdentity compiledModelIdentity,
            String tokenizerRelative,
            String textConfigRelative,
            String quantizationRelative,
            CompileOptions options) throws IOException {
        Files.createDirectories(bundleRoot);
        SdxPlatformProviderDescriptor provider = target.platformProvider();

        String compiledArtifactsJson;
        if (!provider.requiresAotArtifact()) {
            compiledArtifactsJson =
                    "{\"canonicalSdzSha256\":\"" + identity.sha256() + "\"}";
        } else {
            String artifactWithinBundle = target.artifactRelativePath();
            String compiledArtifact;
            if (target.artifactKind() == SdxTargetProfile.ArtifactKind.FILE
                    && target == SdxTargetProfile.ANDROID_ARM64_GOOGLE_TENSOR_G5) {
                compiledArtifact = "{\"path\":\"" + json(artifactWithinBundle) + "\"}";
            } else {
                compiledArtifact = "\"" + json(artifactWithinBundle) + "\"";
            }
            compiledArtifactsJson =
                    "{\"" + json(target.manifestArtifactKey()) + "\":" + compiledArtifact + "}";
        }

        StringBuilder manifest = new StringBuilder();
        manifest.append("{\n")
                .append("  \"formatVersion\": 1,\n")
                .append("  \"modelId\": \"")
                .append(json(options.modelId != null
                        ? options.modelId : identity.sha256()))
                .append("\",\n")
                .append("  \"producer\": {\"tool\": \"SdxModelCompiler\", ")
                .append("\"version\": \"").append(COMPILER_ABI).append("\"},\n")
                .append("  \"modelPath\": \"")
                .append(json(runtimeModelRelative)).append("\",\n")
                .append("  \"targets\": [\"")
                .append(json(target.id())).append("\"],\n")
                .append("  \"preferredBackends\": [\"")
                .append(json(target.backend())).append("\"],\n")
                .append("  \"gpuTarget\": \"")
                .append(json(target.gpuTarget())).append("\",\n")
                .append("  \"platformProvider\": {\"id\":\"")
                .append(json(provider.providerId()))
                .append("\",\"providerAbiVersion\":")
                .append(provider.providerAbiVersion())
                .append(",\"artifactFormat\":\"")
                .append(json(provider.artifactFormat()))
                .append("\",\"artifactFormatVersion\":")
                .append(provider.artifactFormatVersion())
                .append(",\"platform\":\"")
                .append(provider.platform().name().toLowerCase(Locale.ROOT))
                .append("\",\"architecture\":\"")
                .append(json(provider.architecture()))
                .append("\",\"accelerator\":\"")
                .append(provider.accelerator().name().toLowerCase(Locale.ROOT))
                .append("\",\"requiresAotArtifact\":")
                .append(provider.requiresAotArtifact())
                .append(",\"allowRuntimeJit\":")
                .append(provider.allowsRuntimeJit())
                .append(",\"allowCpuFallback\":")
                .append(provider.allowsCpuFallback())
                .append("},\n")
                .append("  \"compiledArtifacts\": ")
                .append(compiledArtifactsJson).append(",\n")
                .append("  \"textGeneration\": ");

        if (tokenizerRelative == null && textConfigRelative == null) {
            manifest.append("{}");
        } else {
            manifest.append("{");
            boolean separator = false;
            if (tokenizerRelative != null) {
                manifest.append("\"tokenizerPath\":\"")
                        .append(json(stripBundlePrefix(tokenizerRelative)))
                        .append("\"");
                separator = true;
            }
            if (textConfigRelative != null) {
                if (separator) {
                    manifest.append(",");
                }
                manifest.append("\"configPath\":\"")
                        .append(json(stripBundlePrefix(textConfigRelative)))
                        .append("\"");
            }
            manifest.append("}");
        }
        manifest.append(",\n  \"quantization\": ");
        if (quantizationRelative == null) {
            manifest.append("{}");
        } else {
            manifest.append("{\"configPath\":\"")
                    .append(json(stripBundlePrefix(quantizationRelative)))
                    .append("\"}");
        }
        manifest.append(",\n")
                .append("  \"source\": {\"sha256\": \"")
                .append(identity.sha256())
                .append("\", \"logicalBytes\": ")
                .append(identity.logicalBytes()).append("},\n");
        if (compiledModelIdentity != null) {
            manifest.append("  \"compiledModel\": {\"sha256\": \"")
                    .append(compiledModelIdentity.sha256())
                    .append("\", \"logicalBytes\": ")
                    .append(compiledModelIdentity.logicalBytes())
                    .append("},\n");
        }
        manifest.append("  \"compileCache\": {\"abi\": \"")
                .append(SdxModelCache.CACHE_ABI)
                .append("\", \"compileKey\": \"")
                .append(compileKey)
                .append("\", \"compilerId\": \"")
                .append(json(compiler.id()))
                .append("\", \"compilerVersion\": \"")
                .append(json(compiler.version()))
                .append("\"},\n")
                .append("  \"compatibility\": {\"minRuntimeAbi\": 1, ")
                .append("\"maxRuntimeAbi\": 1}\n")
                .append("}\n");

        Files.writeString(
                bundleRoot.resolve("manifest.json"),
                manifest.toString(),
                StandardCharsets.UTF_8,
                StandardOpenOption.CREATE_NEW,
                StandardOpenOption.WRITE);
    }

    private static String stripBundlePrefix(String stagingRelative) throws IOException {
        if (!stagingRelative.startsWith("bundle/")) {
            throw new IOException(
                    "SDX bundle metadata path is outside the bundle: " + stagingRelative);
        }
        return stagingRelative.substring("bundle/".length());
    }

    private static String json(String value) {
        StringBuilder escaped = new StringBuilder(value.length() + 16);
        for (int i = 0; i < value.length(); i++) {
            char c = value.charAt(i);
            switch (c) {
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
                    if (c < 0x20) {
                        escaped.append(String.format(Locale.ROOT, "\\u%04x", (int) c));
                    } else {
                        escaped.append(c);
                    }
            }
        }
        return escaped.toString();
    }

    private static String portable(Path path) {
        return path.toString().replace(path.getFileSystem().getSeparator(), "/");
    }

    private static String requireSingleLine(String value, String label) {
        if (value == null || value.trim().isEmpty()
                || value.indexOf('\n') >= 0 || value.indexOf('\r') >= 0) {
            throw new IllegalArgumentException("Invalid " + label);
        }
        return value;
    }

    private static String requireKey(String key) {
        if (key == null || !key.matches("[A-Za-z0-9_.-]+")) {
            throw new IllegalArgumentException("Invalid SDX compiler option key: " + key);
        }
        return key;
    }

    private static MessageDigest sha256Digest() {
        try {
            return MessageDigest.getInstance("SHA-256");
        } catch (NoSuchAlgorithmException impossible) {
            throw new IllegalStateException("SHA-256 is required by the JDK", impossible);
        }
    }

    static String treeDigest(Path input) throws IOException {
        Path root = input.toAbsolutePath().normalize();
        MessageDigest digest = sha256Digest();
        if (Files.isRegularFile(root)) {
            digest.update("file\n".getBytes(StandardCharsets.UTF_8));
            digest.update(SdxSourceIdentity.sha256(root).getBytes(StandardCharsets.US_ASCII));
            digest.update(Long.toString(Files.size(root)).getBytes(StandardCharsets.US_ASCII));
            return SdxSourceIdentity.hex(digest.digest());
        }
        if (!Files.isDirectory(root)) {
            throw new IOException("Compiler cache-key input is missing: " + root);
        }
        try (Stream<Path> stream = Files.walk(root)) {
            List<Path> files = stream.filter(Files::isRegularFile)
                    .sorted()
                    .collect(Collectors.toList());
            for (Path file : files) {
                if (Files.isSymbolicLink(file)) {
                    throw new IOException("Compiler input contains a symlink: " + file);
                }
                String relative = portable(root.relativize(file));
                digest.update(relative.getBytes(StandardCharsets.UTF_8));
                digest.update((byte) '\n');
                digest.update(SdxSourceIdentity.sha256(file)
                        .getBytes(StandardCharsets.US_ASCII));
                digest.update((byte) '\n');
            }
        }
        return SdxSourceIdentity.hex(digest.digest());
    }

    public interface TargetCompiler {
        String id();

        String version();

        /**
         * Whether the compiler needs a private full-model source snapshot.
         * External and graph-rewriting compilers are isolated by default. A
         * trusted policy-only compiler may return false when it never reads or
         * mutates model bytes, avoiding a second multi-gigabyte SDZ copy.
         */
        default boolean requiresIsolatedSourceSnapshot() {
            return true;
        }

        /**
         * Deterministic material that changes whenever compiler configuration,
         * calibration, SoC constraints, or other code-generation inputs change.
         */
        String cacheKeyMaterial(
                Path sourceModel,
                SdxTargetProfile target,
                CompileOptions options) throws IOException;

        /**
         * Compile the canonical source and return the generated target artifact.
         * The implementation should write to {@link CompilationContext#suggestedOutput()}.
         * A runtime-specialized provider returns {@code null}; that is accepted only when its
         * target descriptor explicitly forbids a bundle-owned AOT artifact.
         */
        Path compile(CompilationContext context) throws Exception;
    }

    public static final class CompilationContext {
        private final Path sourceModel;
        private final SdxSourceIdentity sourceIdentity;
        private final SdxTargetProfile target;
        private final Path workDirectory;
        private final Path suggestedOutput;
        private final Path suggestedModelOutput;
        private final CompileOptions options;

        private CompilationContext(
                Path sourceModel,
                SdxSourceIdentity sourceIdentity,
                SdxTargetProfile target,
                Path workDirectory,
                Path suggestedOutput,
                Path suggestedModelOutput,
                CompileOptions options) {
            this.sourceModel = sourceModel;
            this.sourceIdentity = sourceIdentity;
            this.target = target;
            this.workDirectory = workDirectory;
            this.suggestedOutput = suggestedOutput;
            this.suggestedModelOutput = suggestedModelOutput;
            this.options = options;
        }

        public Path sourceModel() {
            return sourceModel;
        }

        public SdxSourceIdentity sourceIdentity() {
            return sourceIdentity;
        }

        public SdxTargetProfile target() {
            return target;
        }

        public Path workDirectory() {
            return workDirectory;
        }

        public Path suggestedOutput() {
            return suggestedOutput;
        }

        /**
         * Optional target-derived canonical SDZ. Compilers that rewrite or
         * quantize the SameDiff graph write this path; otherwise it stays absent.
         */
        public Path suggestedModelOutput() {
            return suggestedModelOutput;
        }

        public CompileOptions options() {
            return options;
        }
    }

    public static final class CompileOptions {
        private final Path tokenizer;
        private final Path tokenizerConfig;
        private final Path textGenerationConfig;
        private final Path quantizationConfig;
        private final String modelId;
        private final String targetSoc;
        private final Map<String, String> cacheKeyProperties;

        private CompileOptions(Builder builder) {
            this.tokenizer = normalize(builder.tokenizer);
            this.tokenizerConfig = normalize(builder.tokenizerConfig);
            this.textGenerationConfig = normalize(builder.textGenerationConfig);
            this.quantizationConfig = normalize(builder.quantizationConfig);
            this.modelId = builder.modelId;
            this.targetSoc = builder.targetSoc == null
                    ? null
                    : requireSingleLine(builder.targetSoc, "target SoC").trim();
            this.cacheKeyProperties = Collections.unmodifiableMap(
                    new TreeMap<>(builder.cacheKeyProperties));
        }

        public static Builder builder() {
            return new Builder();
        }

        public Path tokenizer() {
            return tokenizer;
        }

        public Path tokenizerConfig() {
            return tokenizerConfig;
        }

        public Path textGenerationConfig() {
            return textGenerationConfig;
        }

        public Path quantizationConfig() {
            return quantizationConfig;
        }

        public String modelId() {
            return modelId;
        }

        public String targetSoc() {
            return targetSoc;
        }

        public Map<String, String> cacheKeyProperties() {
            return cacheKeyProperties;
        }

        private static Path normalize(Path path) {
            return path == null ? null : path.toAbsolutePath().normalize();
        }

        public static final class Builder {
            private Path tokenizer;
            private Path tokenizerConfig;
            private Path textGenerationConfig;
            private Path quantizationConfig;
            private String modelId;
            private String targetSoc;
            private final Map<String, String> cacheKeyProperties =
                    new LinkedHashMap<>();

            public Builder tokenizer(Path value) {
                tokenizer = value;
                return this;
            }

            public Builder tokenizerConfig(Path value) {
                tokenizerConfig = value;
                return this;
            }

            public Builder textGenerationConfig(Path value) {
                textGenerationConfig = value;
                return this;
            }

            public Builder quantizationConfig(Path value) {
                quantizationConfig = value;
                return this;
            }

            public Builder modelId(String value) {
                modelId = value;
                return this;
            }

            public Builder targetSoc(String value) {
                targetSoc = value;
                return this;
            }

            public Builder cacheKeyProperty(String key, String value) {
                cacheKeyProperties.put(
                        requireKey(key),
                        requireSingleLine(value, "compiler option value"));
                return this;
            }

            public CompileOptions build() {
                return new CompileOptions(this);
            }
        }
    }

    /**
     * Selects the in-process compiler used by every Java, Graal AOT, desktop, and mobile
     * staging entry point for a canonical SDZ. Container ingestion is deliberately absent
     * from this decision: GGUF, Hugging Face, and project flows all hand the same SDZ to this
     * method after import.
     *
     * @param target target runtime profile
     * @param targetSoc explicit SoC, or {@code null}/{@code blank} for the provider default
     * @param quantizationRequested whether compilation must rewrite the canonical SDZ
     * @return a built-in compiler when this distribution owns the target implementation
     */
    public static Optional<TargetCompiler> builtInTargetCompiler(
            SdxTargetProfile target,
            String targetSoc,
            boolean quantizationRequested) {
        Objects.requireNonNull(target, "target");
        String soc = targetSoc == null || targetSoc.isBlank()
                ? target.platformProvider().defaultTargetSoc()
                : requireSingleLine(targetSoc, "target SoC").trim();
        if (!quantizationRequested && target == SdxTargetProfile.IOS_ARM64_METAL) {
            return Optional.of(metalDeviceCompilationPolicy(soc));
        }
        if (target == SdxTargetProfile.ANDROID_ARM64_NNAPI_ACCELERATOR) {
            if (!quantizationRequested) {
                return Optional.of(nnapiDeviceCompilationPolicy(soc));
            }
            if (SdxTensorG3NnapiCompiler.TARGET_SOC.equals(soc)) {
                return Optional.of(new SdxTensorG3NnapiCompiler());
            }
        }
        return Optional.empty();
    }

    /**
     * Fail-closed form of {@link #builtInTargetCompiler(SdxTargetProfile, String, boolean)}
     * for self-contained/AOT entry points that cannot delegate to an external toolchain.
     */
    public static TargetCompiler requireBuiltInTargetCompiler(
            SdxTargetProfile target,
            String targetSoc,
            boolean quantizationRequested) {
        return builtInTargetCompiler(target, targetSoc, quantizationRequested)
                .orElseThrow(() -> new IllegalArgumentException(
                        "No built-in SDX target compiler is available for " + target.id()
                                + " (targetSoc="
                                + (targetSoc == null || targetSoc.isBlank()
                                        ? target.platformProvider().defaultTargetSoc()
                                        : targetSoc.trim())
                                + ", quantizationRequested=" + quantizationRequested + ")"));
    }

    /**
     * Staging policy for the production Apple MLX/Metal provider.
     *
     * <p>No raw metallib is produced or accepted here. The canonical SDZ and text assets are
     * content-addressed by the host cache, then the shared libnd4j MLX/DSP replay runtime
     * specializes shapes into the app-owned device compilation cache. The separately gated
     * {@code sdx.metal-aot.v1} provider owns raw metallibs and fixed Metal plans.</p>
     */
    public static TargetCompiler metalDeviceCompilationPolicy(String targetSoc) {
        String soc = requireSingleLine(targetSoc, "Metal target SoC").trim();
        return new TargetCompiler() {
            @Override
            public String id() {
                return "sdx-mlx-device-compilation";
            }

            @Override
            public String version() {
                return "1";
            }

            @Override
            public String cacheKeyMaterial(
                    Path sourceModel,
                    SdxTargetProfile target,
                    CompileOptions options) throws IOException {
                requireMetalRuntimeTarget(target, options, soc);
                return "policyAbi:sdx-mlx-device-compilation-v1;targetSoc:" + soc;
            }

            @Override
            public Path compile(CompilationContext context) throws IOException {
                requireMetalRuntimeTarget(context.target(), context.options(), soc);
                return null;
            }
        };
    }

    private static void requireMetalRuntimeTarget(
            SdxTargetProfile target,
            CompileOptions options,
            String expectedSoc) throws IOException {
        if (target != SdxTargetProfile.IOS_ARM64_METAL) {
            throw new IOException(
                    "MLX device compilation policy cannot stage target " + target.id());
        }
        if (options.quantizationConfig() != null) {
            throw new IOException(
                    "MLX device compilation policy cannot quantize the canonical SDZ. "
                            + "Stage an already-quantized SDZ or configure a graph-rewrite compiler.");
        }
        if (options.targetSoc() != null && !expectedSoc.equals(options.targetSoc())) {
            throw new IOException(
                    "MLX policy SoC " + expectedSoc
                            + " does not match requested SoC " + options.targetSoc());
        }
    }

    /**
     * Built-in policy compiler for unquantized SameDiff models that are compiled
     * by Android's NNAPI driver on the target device.
     *
     * <p>The generated artifact is an auditable, content-addressed policy rather
     * than a fabricated host AOT blob. The accelerator-only native runtime pins
     * compilation to one {@code DEVICE_ACCELERATOR}, requires complete graph
     * support, and uses NNAPI's persistent device compilation cache. Quantized
     * graphs require a real target compiler that rewrites the canonical SDZ with
     * explicit quantization metadata and are therefore rejected here.</p>
     */
    public static TargetCompiler nnapiDeviceCompilationPolicy(String targetSoc) {
        String soc = requireSingleLine(targetSoc, "NNAPI target SoC").trim();
        return new TargetCompiler() {
            @Override
            public String id() {
                return "sdx-nnapi-device-policy";
            }

            @Override
            public String version() {
                return "1";
            }

            @Override
            public boolean requiresIsolatedSourceSnapshot() {
                return false;
            }

            @Override
            public String cacheKeyMaterial(
                    Path sourceModel,
                    SdxTargetProfile target,
                    CompileOptions options) throws IOException {
                requireUnquantizedNnapiTarget(target, options);
                requireMatchingTargetSoc(options, soc);
                return "policyAbi:" + SdxNnapiDevicePolicy.POLICY_ABI
                        + ";targetSoc:" + soc;
            }

            @Override
            public Path compile(CompilationContext context) throws IOException {
                requireUnquantizedNnapiTarget(context.target(), context.options());
                requireMatchingTargetSoc(context.options(), soc);
                Path output = context.suggestedOutput();
                SdxNnapiDevicePolicy.create(
                        context.target(), soc, context.sourceIdentity()).write(output);
                return output;
            }
        };
    }

    private static void requireUnquantizedNnapiTarget(
            SdxTargetProfile target,
            CompileOptions options) throws IOException {
        if (target != SdxTargetProfile.ANDROID_ARM64_NNAPI_ACCELERATOR) {
            throw new IOException(
                    "NNAPI device compilation policy cannot compile target " + target.id());
        }
        if (options.quantizationConfig() != null) {
            throw new IOException(
                    "NNAPI device compilation policy cannot quantize a model. "
                            + "Configure an SDX target compiler that rewrites the canonical SDZ "
                            + "with explicit per-channel scale and zero-point metadata.");
        }
    }

    private static void requireMatchingTargetSoc(
            CompileOptions options, String expectedSoc) throws IOException {
        if (options.targetSoc() != null
                && !expectedSoc.equals(options.targetSoc())) {
            throw new IOException(
                    "NNAPI policy SoC " + expectedSoc
                            + " does not match requested SoC " + options.targetSoc());
        }
    }

    /**
     * Migration adapter for already-produced AOT artifacts. New integrations
     * should expose their vendor/native compiler as a {@link TargetCompiler}.
     */
    public static TargetCompiler preparedArtifact(
            Path artifact,
            String compilerId,
            String compilerVersion) {
        Path prepared = artifact.toAbsolutePath().normalize();
        return new TargetCompiler() {
            @Override
            public String id() {
                return compilerId;
            }

            @Override
            public String version() {
                return compilerVersion;
            }

            @Override
            public String cacheKeyMaterial(
                    Path sourceModel,
                    SdxTargetProfile target,
                    CompileOptions options) throws IOException {
                return "prepared:" + treeDigest(prepared);
            }

            @Override
            public Path compile(CompilationContext context) {
                return prepared;
            }
        };
    }

    /**
     * Host-only toolchain adapter. No shell is involved: arguments are passed
     * directly to the executable. The canonical SDZ, target, output, optional
     * quantization/tokenizer/tokenizer-configuration/generation inputs, and
     * cache-key options are all
     * forwarded to the target compiler in the same invocation.
     */
    public static TargetCompiler externalCommand(
            List<String> command,
            String compilerId,
            String compilerVersion,
            String configurationFingerprint) {
        List<String> baseCommand =
                Collections.unmodifiableList(new ArrayList<>(command));
        if (baseCommand.isEmpty()) {
            throw new IllegalArgumentException("Compiler command is empty");
        }
        String fingerprint = requireSingleLine(
                configurationFingerprint, "compiler configuration fingerprint");
        return new TargetCompiler() {
            @Override
            public String id() {
                return compilerId;
            }

            @Override
            public String version() {
                return compilerVersion;
            }

            @Override
            public String cacheKeyMaterial(
                    Path sourceModel,
                    SdxTargetProfile target,
                    CompileOptions options) {
                return "command:" + String.join("\u0000", baseCommand)
                        + "\u0000config:" + fingerprint;
            }

            @Override
            public Path compile(CompilationContext context) throws Exception {
                List<String> invocation = new ArrayList<>(baseCommand);
                invocation.add("--input");
                invocation.add(context.sourceModel().toString());
                invocation.add("--target");
                invocation.add(context.target().id());
                invocation.add("--output");
                invocation.add(context.suggestedOutput().toString());
                invocation.add("--source-sha256");
                invocation.add(context.sourceIdentity().sha256());
                invocation.add("--model-output");
                invocation.add(context.suggestedModelOutput().toString());
                CompileOptions options = context.options();
                addOptionalCompilerInput(
                        invocation, "--quantization-config", options.quantizationConfig());
                addOptionalCompilerInput(
                        invocation, "--tokenizer", options.tokenizer());
                addOptionalCompilerInput(
                        invocation, "--tokenizer-config", options.tokenizerConfig());
                addOptionalCompilerInput(
                        invocation, "--text-generation-config",
                        options.textGenerationConfig());
                if (options.targetSoc() != null) {
                    invocation.add("--target-soc");
                    invocation.add(options.targetSoc());
                }
                if (options.modelId() != null && !options.modelId().trim().isEmpty()) {
                    invocation.add("--model-id");
                    invocation.add(options.modelId());
                }
                for (Map.Entry<String, String> entry
                        : options.cacheKeyProperties().entrySet()) {
                    invocation.add("--option");
                    invocation.add(entry.getKey() + "=" + entry.getValue());
                }

                ProcessBuilder builder = new ProcessBuilder(invocation);
                builder.directory(context.workDirectory().toFile());
                builder.redirectErrorStream(true);
                Process process = builder.start();
                ByteArrayOutputStream output = new ByteArrayOutputStream();
                try (InputStream input = new BufferedInputStream(process.getInputStream())) {
                    byte[] buffer = new byte[8192];
                    int total = 0;
                    while (true) {
                        int read = input.read(buffer);
                        if (read < 0) {
                            break;
                        }
                        if (total < 256 * 1024) {
                            int retained = Math.min(read, 256 * 1024 - total);
                            output.write(buffer, 0, retained);
                            total += retained;
                        }
                    }
                }
                int status;
                try {
                    status = process.waitFor();
                } catch (InterruptedException interrupted) {
                    process.destroyForcibly();
                    Thread.currentThread().interrupt();
                    throw new IOException("SDX target compiler was interrupted", interrupted);
                }
                if (status != 0) {
                    throw new IOException(
                            "SDX target compiler exited with " + status + ": "
                                    + output.toString(StandardCharsets.UTF_8));
                }
                return context.suggestedOutput();
            }
        };
    }

    private static void addOptionalCompilerInput(
            List<String> invocation, String flag, Path value) {
        if (value != null) {
            invocation.add(flag);
            invocation.add(value.toString());
        }
    }
}
