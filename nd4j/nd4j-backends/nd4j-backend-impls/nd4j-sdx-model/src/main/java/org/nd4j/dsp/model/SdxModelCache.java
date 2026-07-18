/*
 * Copyright (c) Eclipse Deeplearning4j
 * SPDX-License-Identifier: Apache-2.0
 */
package org.nd4j.dsp.model;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.channels.FileChannel;
import java.nio.channels.FileLock;
import java.nio.charset.StandardCharsets;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Base64;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.TreeMap;
import java.util.UUID;
import java.util.function.Supplier;
import java.util.regex.Pattern;
import java.util.stream.Stream;
import java.util.zip.CRC32;
import java.util.zip.Deflater;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;
import java.util.zip.ZipOutputStream;

/**
 * Immutable content-addressed cache for target artifacts derived from one SDZ.
 *
 * <p>The cache is the deployment boundary. Applications name the canonical SDZ
 * and a target profile; provider files are resolved internally. Host compilation
 * may embed cache objects back into an SDZ under {@code META-INF/sdx-cache/}.
 * Mobile then installs only its selected target object into app-owned storage.</p>
 */
public final class SdxModelCache {
    public static final String CACHE_ABI = "sdx-model-cache-v1";
    public static final String EMBEDDED_ROOT = "META-INF/sdx-cache/v1/";
    public static final String DEFAULT_CACHE_PROPERTY = "org.nd4j.sdx.modelCacheDir";
    public static final String DEFAULT_CACHE_ENV = "ND4J_SDX_MODEL_CACHE_DIR";

    private static final String MANIFEST_FILE = "cache.properties";
    private static final String VALIDATED_FILE = ".validated";
    private static final String INDEX_FILE = "source.properties";
    private static final int MAX_MANIFEST_BYTES = 1024 * 1024;
    private static final int MAX_CACHE_FILES = 100_000;
    private static final long ZIP_TIME = 315532800000L;
    private static final Pattern SHA256 = Pattern.compile("^[0-9a-f]{64}$");
    private static final Pattern SAFE_KEY = Pattern.compile("^[A-Za-z0-9_.-]+$");
    private static final Base64.Encoder PATH_ENCODER =
            Base64.getUrlEncoder().withoutPadding();
    private static final Base64.Decoder PATH_DECODER = Base64.getUrlDecoder();

    private final Path root;

    public SdxModelCache(Path cacheDirectory) {
        Objects.requireNonNull(cacheDirectory, "cacheDirectory");
        this.root = cacheDirectory.toAbsolutePath().normalize().resolve("v1");
    }

    public static SdxModelCache defaultCache() {
        String configured = System.getProperty(DEFAULT_CACHE_PROPERTY);
        if (configured == null || configured.trim().isEmpty()) {
            configured = System.getenv(DEFAULT_CACHE_ENV);
        }
        if (configured == null || configured.trim().isEmpty()) {
            String home = System.getProperty("user.home", ".");
            configured = Path.of(home, ".kompile", "cache", "sdx", "models").toString();
        }
        return new SdxModelCache(Path.of(configured));
    }

    public Path root() {
        return root;
    }

    public SdxSourceIdentity identify(Path sourceModel) throws IOException {
        return SdxSourceIdentity.identify(sourceModel);
    }

    /**
     * Resolve a target from an installed cache object or an embedded SDZ cache.
     * No runtime/JIT compilation is attempted here.
     */
    public SdxCompiledModel resolve(Path sourceModel, SdxTargetProfile target) throws IOException {
        Objects.requireNonNull(target, "target");
        Path source = requireSource(sourceModel);
        SdxSourceIdentity identity = identify(source);
        ensureSourceCached(source, identity);

        Optional<SdxCompiledModel> installed = resolveReference(source, identity, target);
        if (installed.isPresent()) {
            return installed.get();
        }

        if (tryInstallEmbedded(source, identity, target)) {
            installed = resolveReference(source, identity, target);
            if (installed.isPresent()) {
                return installed.get();
            }
        }

        throw new MissingCompiledModelException(
                "No compiled SDX cache entry for source=" + identity.sha256()
                        + ", target=" + target.id()
                        + ". Compile/package this SDZ on a host with the target toolchain; "
                        + "mobile execution never falls back to JIT or another backend.");
    }

    /**
     * Produce a single deployable SDZ containing the canonical model plus the
     * selected immutable target cache objects.
     */
    public void packageCompiledSdz(
            Path sourceModel,
            Collection<SdxTargetProfile> targets,
            Path outputSdz) throws IOException {
        Objects.requireNonNull(targets, "targets");
        if (targets.isEmpty()) {
            throw new IllegalArgumentException("At least one SDX target is required");
        }

        Path source = requireSource(sourceModel);
        if (!source.getFileName().toString().toLowerCase().endsWith(".sdz")) {
            throw new IOException("Single-file packaging requires an .sdz source: " + source);
        }
        Path output = outputSdz.toAbsolutePath().normalize();
        if (!output.getFileName().toString().toLowerCase().endsWith(".sdz")) {
            throw new IOException("Compiled model output must retain the .sdz extension: " + output);
        }
        if (source.equals(output)) {
            throw new IOException("Compiled SDZ output must differ from the source path");
        }

        SdxSourceIdentity identity = identify(source);
        List<SdxCompiledModel> compiled = new ArrayList<>();
        Set<SdxTargetProfile> uniqueTargets = new HashSet<>(targets);
        for (SdxTargetProfile target : uniqueTargets) {
            compiled.add(resolve(source, target));
        }
        compiled.sort(Comparator.comparing(model -> model.target().id()));

        Path parent = output.getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }
        Path temporary = output.resolveSibling(
                "." + output.getFileName() + "." + UUID.randomUUID() + ".pending");
        Files.deleteIfExists(temporary);

        try (ZipFile input = new ZipFile(source.toFile());
             ZipOutputStream archive = new ZipOutputStream(new BufferedOutputStream(
                     Files.newOutputStream(
                             temporary,
                             StandardOpenOption.CREATE_NEW,
                             StandardOpenOption.WRITE)))) {
            archive.setLevel(Deflater.DEFAULT_COMPRESSION);
            for (ZipEntry sourceEntry : SdxSourceIdentity.sortedSourceEntries(input)) {
                copySourceEntry(input, sourceEntry, archive);
            }

            Map<String, String> index = new TreeMap<>();
            index.put("cacheAbi", CACHE_ABI);
            index.put("sourceSha256", identity.sha256());
            index.put("sourceLogicalBytes", Long.toString(identity.logicalBytes()));
            index.put("targetCount", Integer.toString(compiled.size()));
            for (int i = 0; i < compiled.size(); i++) {
                index.put("target." + i, compiled.get(i).target().id());
            }
            putStoredBytes(
                    archive,
                    EMBEDDED_ROOT + INDEX_FILE,
                    canonicalMapBytes(index));

            for (SdxCompiledModel model : compiled) {
                Path entry = model.cacheEntry();
                Path manifestPath = entry.resolve(MANIFEST_FILE);
                Map<String, String> manifest = readCanonicalMap(manifestPath);
                String prefix = EMBEDDED_ROOT + model.target().id() + "/";
                putStoredFile(archive, prefix + MANIFEST_FILE, manifestPath);
                for (FileRecord file : fileRecords(manifest)) {
                    putStoredFile(
                            archive,
                            prefix + file.path,
                            resolveContained(entry, file.path));
                }
            }
        } catch (Throwable failure) {
            Files.deleteIfExists(temporary);
            if (failure instanceof IOException) {
                throw (IOException) failure;
            }
            throw new IOException("Could not package compiled SDZ", failure);
        }

        moveReplacing(temporary, output);
    }

    Optional<SdxCompiledModel> resolveByCompileKey(
            Path source,
            SdxSourceIdentity identity,
            SdxTargetProfile target,
            String compileKey) throws IOException {
        requireSha256(compileKey, "compileKey");
        Path object = objectDirectory(compileKey);
        if (!Files.isDirectory(object)) {
            return Optional.empty();
        }
        try {
            SdxCompiledModel resolved = validateObject(
                    source, identity, target, compileKey, object, false);
            writeReference(identity, target, compileKey);
            return Optional.of(resolved);
        } catch (IOException invalid) {
            return Optional.empty();
        }
    }

    Path newStagingDirectory(String compileKey) throws IOException {
        requireSha256(compileKey, "compileKey");
        Path temporaryRoot = root.resolve("tmp");
        Files.createDirectories(temporaryRoot);
        Path staging = temporaryRoot.resolve(
                compileKey + "." + UUID.randomUUID() + ".pending");
        Files.createDirectory(staging);
        return staging;
    }

    SdxCompiledModel publish(
            Path source,
            SdxSourceIdentity identity,
            SdxTargetProfile target,
            String compileKey,
            String compilerId,
            String compilerVersion,
            Path staging,
            String runtimeRelativePath,
            String tokenizerRelativePath,
            String textConfigRelativePath,
            String quantizationRelativePath) throws IOException {
        requireSha256(compileKey, "compileKey");
        requireToken(compilerId, "compilerId");
        requireToken(compilerVersion, "compilerVersion");
        requireStaging(staging);

        List<FileRecord> files = collectFiles(staging);
        if (files.isEmpty()) {
            throw new IOException("SDX compiler produced an empty cache object");
        }

        Map<String, String> manifest = new TreeMap<>();
        manifest.put("cacheAbi", CACHE_ABI);
        manifest.put("compileKey", compileKey);
        manifest.put("sourceSha256", identity.sha256());
        manifest.put("sourceLogicalBytes", Long.toString(identity.logicalBytes()));
        manifest.put("sourceFileName", identity.sourceFileName());
        manifest.put("target", target.id());
        manifest.put("runtimeKind", target.runtimeKind().name());
        manifest.put("runtimePath64", encodePath(runtimeRelativePath));
        manifest.put("compilerId64", encodeText(compilerId));
        manifest.put("compilerVersion64", encodeText(compilerVersion));
        if (tokenizerRelativePath != null) {
            manifest.put("tokenizerPath64", encodePath(tokenizerRelativePath));
        }
        if (textConfigRelativePath != null) {
            manifest.put("textConfigPath64", encodePath(textConfigRelativePath));
        }
        if (quantizationRelativePath != null) {
            manifest.put("quantizationPath64", encodePath(quantizationRelativePath));
        }
        manifest.put("fileCount", Integer.toString(files.size()));
        for (int i = 0; i < files.size(); i++) {
            FileRecord file = files.get(i);
            manifest.put("file." + i + ".path64", encodePath(file.path));
            manifest.put("file." + i + ".sha256", file.sha256);
            manifest.put("file." + i + ".bytes", Long.toString(file.bytes));
        }
        Files.write(
                staging.resolve(MANIFEST_FILE),
                canonicalMapBytes(manifest),
                StandardOpenOption.CREATE_NEW,
                StandardOpenOption.WRITE);

        Path object = objectDirectory(compileKey);
        Files.createDirectories(object.getParent());
        if (Files.exists(object)) {
            deleteTree(staging);
        } else {
            moveWithoutReplace(staging, object);
        }

        SdxCompiledModel resolved = validateObject(
                source, identity, target, compileKey, object, true);
        writeReference(identity, target, compileKey);
        return resolved;
    }

    Path ensureSourceCached(Path source, SdxSourceIdentity identity) throws IOException {
        Path destination = sourceDirectory(identity).resolve(identity.sourceFileName());
        if (Files.isRegularFile(destination) && Files.size(destination) > 0L) {
            return destination;
        }

        Files.createDirectories(destination.getParent());
        String sourceSuffix = identity.sourceFileName().endsWith(".sdnb")
                ? ".sdnb"
                : ".sdz";
        Path pending = destination.resolveSibling(
                "." + destination.getFileName() + "." + UUID.randomUUID()
                        + ".pending" + sourceSuffix);
        try {
            Files.copy(source, pending, StandardCopyOption.REPLACE_EXISTING);
            SdxSourceIdentity copied = identify(pending);
            if (!identity.sha256().equals(copied.sha256())) {
                throw new IOException("Canonical SDZ changed while being cached");
            }
        } catch (IOException failure) {
            Files.deleteIfExists(pending);
            throw failure;
        }

        try {
            moveWithoutReplace(pending, destination);
        } catch (java.nio.file.FileAlreadyExistsException raced) {
            Files.deleteIfExists(pending);
        }
        return destination;
    }

    <T> T withCompileLock(String compileKey, IoSupplier<T> action) throws IOException {
        requireSha256(compileKey, "compileKey");
        Path lockDirectory = root.resolve("locks");
        Files.createDirectories(lockDirectory);
        Path lockPath = lockDirectory.resolve(compileKey + ".lock");
        try (FileChannel channel = FileChannel.open(
                     lockPath,
                     StandardOpenOption.CREATE,
                     StandardOpenOption.WRITE);
             FileLock ignored = channel.lock()) {
            try {
                return action.get();
            } catch (IOException failure) {
                throw failure;
            } catch (RuntimeException failure) {
                throw failure;
            } catch (Exception failure) {
                throw new IOException("SDX target compiler failed", failure);
            }
        }
    }

    private Optional<SdxCompiledModel> resolveReference(
            Path source,
            SdxSourceIdentity identity,
            SdxTargetProfile target) throws IOException {
        Path reference = referencePath(identity, target);
        if (!Files.isRegularFile(reference)) {
            return Optional.empty();
        }
        String compileKey = Files.readString(reference, StandardCharsets.US_ASCII).trim();
        if (!SHA256.matcher(compileKey).matches()) {
            return Optional.empty();
        }
        return resolveByCompileKey(source, identity, target, compileKey);
    }

    private boolean tryInstallEmbedded(
            Path source,
            SdxSourceIdentity identity,
            SdxTargetProfile target) throws IOException {
        if (!source.getFileName().toString().toLowerCase().endsWith(".sdz")) {
            return false;
        }

        String prefix = EMBEDDED_ROOT + target.id() + "/";
        try (ZipFile zip = new ZipFile(source.toFile())) {
            ZipEntry manifestEntry = zip.getEntry(prefix + MANIFEST_FILE);
            if (manifestEntry == null || manifestEntry.isDirectory()) {
                return false;
            }
            byte[] manifestBytes = readLimited(zip.getInputStream(manifestEntry));
            Map<String, String> manifest = readCanonicalMap(manifestBytes);
            validateManifestIdentity(manifest, identity, target, null);
            String compileKey = required(manifest, "compileKey");

            return withCompileLock(compileKey, () -> {
                Optional<SdxCompiledModel> existing =
                        resolveByCompileKey(source, identity, target, compileKey);
                if (existing.isPresent()) {
                    return true;
                }

                Path staging = newStagingDirectory(compileKey);
                try {
                    int count = parseCount(manifest, "fileCount", MAX_CACHE_FILES);
                    for (int i = 0; i < count; i++) {
                        String relative = decodePath(required(
                                manifest, "file." + i + ".path64"));
                        Path destination = resolveContained(staging, relative);
                        ZipEntry embedded = zip.getEntry(prefix + relative);
                        if (embedded == null || embedded.isDirectory()) {
                            throw new IOException(
                                    "Embedded SDX cache is missing " + prefix + relative);
                        }
                        Files.createDirectories(destination.getParent());
                        try (InputStream input = new BufferedInputStream(
                                     zip.getInputStream(embedded));
                             BufferedOutputStream output = new BufferedOutputStream(
                                     Files.newOutputStream(
                                             destination,
                                             StandardOpenOption.CREATE_NEW,
                                             StandardOpenOption.WRITE))) {
                            copy(input, output);
                        }
                    }
                    Files.write(
                            staging.resolve(MANIFEST_FILE),
                            manifestBytes,
                            StandardOpenOption.CREATE_NEW,
                            StandardOpenOption.WRITE);

                    validateObject(
                            source, identity, target, compileKey, staging, true);
                    Path object = objectDirectory(compileKey);
                    Files.createDirectories(object.getParent());
                    if (Files.exists(object)) {
                        deleteTree(staging);
                    } else {
                        moveWithoutReplace(staging, object);
                    }
                    validateObject(
                            source, identity, target, compileKey, object, true);
                    writeReference(identity, target, compileKey);
                    return true;
                } catch (Throwable failure) {
                    deleteTree(staging);
                    if (failure instanceof IOException) {
                        throw (IOException) failure;
                    }
                    throw new IOException("Could not install embedded SDX cache", failure);
                }
            });
        }
    }

    private SdxCompiledModel validateObject(
            Path source,
            SdxSourceIdentity identity,
            SdxTargetProfile target,
            String expectedCompileKey,
            Path object,
            boolean forceHashes) throws IOException {
        Path manifestPath = object.resolve(MANIFEST_FILE);
        Map<String, String> manifest = readCanonicalMap(manifestPath);
        validateManifestIdentity(manifest, identity, target, expectedCompileKey);

        String manifestSha = SdxSourceIdentity.sha256(manifestPath);
        Path marker = object.resolve(VALIDATED_FILE);
        boolean fast = !forceHashes
                && Files.isRegularFile(marker)
                && manifestSha.equals(Files.readString(marker, StandardCharsets.US_ASCII).trim());

        for (FileRecord file : fileRecords(manifest)) {
            Path path = resolveContained(object, file.path);
            if (!Files.isRegularFile(path) || Files.size(path) != file.bytes) {
                throw new IOException("SDX cache file is missing or changed: " + file.path);
            }
            if (!fast && !file.sha256.equals(SdxSourceIdentity.sha256(path))) {
                throw new IOException("SDX cache checksum mismatch: " + file.path);
            }
        }

        String runtimeRelative = decodePath(required(manifest, "runtimePath64"));
        Path runtimePath = resolveContained(object, runtimeRelative);
        if (!Files.exists(runtimePath)) {
            throw new IOException("SDX runtime path is missing: " + runtimeRelative);
        }

        Path tokenizer = optionalPath(object, manifest.get("tokenizerPath64"));
        Path textConfig = optionalPath(object, manifest.get("textConfigPath64"));
        Path quantization = optionalPath(object, manifest.get("quantizationPath64"));
        String compilerId = decodeText(required(manifest, "compilerId64"));
        String compilerVersion = decodeText(required(manifest, "compilerVersion64"));

        Path cachedSource = sourceDirectory(identity).resolve(identity.sourceFileName());
        if (!Files.isRegularFile(cachedSource) || Files.size(cachedSource) <= 0L) {
            throw new IOException("Canonical SDZ source is missing from cache: " + cachedSource);
        }

        if (!fast) {
            writeAtomically(marker, (manifestSha + "\n").getBytes(StandardCharsets.US_ASCII));
        }

        return new SdxCompiledModel(
                source,
                object,
                runtimePath,
                tokenizer,
                textConfig,
                quantization,
                identity,
                target,
                required(manifest, "compileKey"),
                compilerId,
                compilerVersion);
    }

    private void validateManifestIdentity(
            Map<String, String> manifest,
            SdxSourceIdentity identity,
            SdxTargetProfile target,
            String expectedCompileKey) throws IOException {
        if (!CACHE_ABI.equals(required(manifest, "cacheAbi"))) {
            throw new IOException("Unsupported SDX model cache ABI");
        }
        String key = required(manifest, "compileKey");
        requireSha256(key, "compileKey");
        if (expectedCompileKey != null && !expectedCompileKey.equals(key)) {
            throw new IOException("SDX cache compile key mismatch");
        }
        if (!identity.sha256().equals(required(manifest, "sourceSha256"))) {
            throw new IOException("SDX cache belongs to a different canonical source model");
        }
        long logicalBytes = parseLong(manifest, "sourceLogicalBytes");
        if (logicalBytes != identity.logicalBytes()) {
            throw new IOException("SDX cache source logical size mismatch");
        }
        if (!identity.sourceFileName().equals(required(manifest, "sourceFileName"))) {
            throw new IOException("SDX cache source format mismatch");
        }
        if (!target.id().equals(required(manifest, "target"))) {
            throw new IOException("SDX cache target mismatch");
        }
        if (!target.runtimeKind().name().equals(required(manifest, "runtimeKind"))) {
            throw new IOException("SDX cache runtime kind mismatch");
        }
        parseCount(manifest, "fileCount", MAX_CACHE_FILES);
    }

    private List<FileRecord> collectFiles(Path rootDirectory) throws IOException {
        List<FileRecord> files = new ArrayList<>();
        try (Stream<Path> stream = Files.walk(rootDirectory)) {
            List<Path> paths = stream
                    .sorted()
                    .collect(java.util.stream.Collectors.toList());
            for (Path path : paths) {
                if (Files.isSymbolicLink(path)) {
                    throw new IOException("SDX cache object contains a symlink: " + path);
                }
                if (!Files.isRegularFile(path)) {
                    continue;
                }
                Path relative = rootDirectory.relativize(path);
                String portable = portablePath(relative);
                if (MANIFEST_FILE.equals(portable) || VALIDATED_FILE.equals(portable)) {
                    continue;
                }
                files.add(new FileRecord(
                        portable,
                        SdxSourceIdentity.sha256(path),
                        Files.size(path)));
            }
        }
        files.sort(Comparator.comparing(file -> file.path));
        return files;
    }

    private List<FileRecord> fileRecords(Map<String, String> manifest) throws IOException {
        int count = parseCount(manifest, "fileCount", MAX_CACHE_FILES);
        List<FileRecord> files = new ArrayList<>(count);
        Set<String> paths = new HashSet<>();
        for (int i = 0; i < count; i++) {
            String path = decodePath(required(manifest, "file." + i + ".path64"));
            if (!paths.add(path)) {
                throw new IOException("Duplicate SDX cache file path: " + path);
            }
            String sha = required(manifest, "file." + i + ".sha256");
            requireSha256(sha, "file checksum");
            long bytes = parseLong(manifest, "file." + i + ".bytes");
            if (bytes < 0L) {
                throw new IOException("Negative SDX cache file size: " + path);
            }
            files.add(new FileRecord(path, sha, bytes));
        }
        return files;
    }

    private void writeReference(
            SdxSourceIdentity identity,
            SdxTargetProfile target,
            String compileKey) throws IOException {
        Path reference = referencePath(identity, target);
        Files.createDirectories(reference.getParent());
        writeAtomically(
                reference,
                (compileKey + "\n").getBytes(StandardCharsets.US_ASCII));
    }

    private Path referencePath(SdxSourceIdentity identity, SdxTargetProfile target) {
        return root.resolve("index")
                .resolve(identity.sha256())
                .resolve(target.id() + ".ref");
    }

    private Path sourceDirectory(SdxSourceIdentity identity) {
        return root.resolve("sources").resolve(identity.sha256());
    }

    private Path objectDirectory(String compileKey) {
        return root.resolve("objects").resolve(compileKey);
    }

    private static Path requireSource(Path sourceModel) throws IOException {
        Objects.requireNonNull(sourceModel, "sourceModel");
        Path source = sourceModel.toAbsolutePath().normalize();
        if (!Files.isRegularFile(source) || Files.size(source) <= 0L) {
            throw new IOException("SDX source model is missing or empty: " + source);
        }
        return source;
    }

    private static void requireStaging(Path staging) throws IOException {
        if (staging == null || !Files.isDirectory(staging)) {
            throw new IOException("SDX compiler staging directory is missing");
        }
        if (Files.exists(staging.resolve(MANIFEST_FILE))) {
            throw new IOException("SDX compiler staging directory already has a cache manifest");
        }
    }

    private static Path optionalPath(Path object, String encoded) throws IOException {
        if (encoded == null || encoded.isEmpty()) {
            return null;
        }
        Path path = resolveContained(object, decodePath(encoded));
        if (!Files.isRegularFile(path)) {
            throw new IOException("SDX cache metadata asset is missing: " + path);
        }
        return path;
    }

    private static Path resolveContained(Path root, String relative) throws IOException {
        requireRelativePath(relative);
        Path normalizedRoot = root.toAbsolutePath().normalize();
        Path path = normalizedRoot.resolve(relative).normalize();
        if (!path.startsWith(normalizedRoot)) {
            throw new IOException("SDX cache path escapes its object: " + relative);
        }
        return path;
    }

    private static void requireRelativePath(String value) throws IOException {
        if (value == null || value.isEmpty() || value.startsWith("/")
                || value.indexOf('\\') >= 0) {
            throw new IOException("Unsafe SDX cache relative path: " + value);
        }
        for (String part : value.split("/")) {
            if (part.isEmpty() || ".".equals(part) || "..".equals(part)) {
                throw new IOException("Unsafe SDX cache relative path: " + value);
            }
        }
    }

    private static String portablePath(Path path) throws IOException {
        String portable = path.toString().replace(path.getFileSystem().getSeparator(), "/");
        requireRelativePath(portable);
        return portable;
    }

    static String encodePath(String value) throws IOException {
        requireRelativePath(value);
        return encodeText(value);
    }

    static String decodePath(String value) throws IOException {
        String decoded = decodeText(value);
        requireRelativePath(decoded);
        return decoded;
    }

    private static String encodeText(String value) {
        return PATH_ENCODER.encodeToString(value.getBytes(StandardCharsets.UTF_8));
    }

    private static String decodeText(String value) throws IOException {
        try {
            return new String(PATH_DECODER.decode(value), StandardCharsets.UTF_8);
        } catch (IllegalArgumentException invalid) {
            throw new IOException("Invalid base64 value in SDX cache manifest", invalid);
        }
    }

    private static Map<String, String> readCanonicalMap(Path path) throws IOException {
        if (!Files.isRegularFile(path) || Files.size(path) > MAX_MANIFEST_BYTES) {
            throw new IOException("SDX cache manifest is missing or too large: " + path);
        }
        try (InputStream input = Files.newInputStream(path)) {
            return readCanonicalMap(readLimited(input));
        }
    }

    private static Map<String, String> readCanonicalMap(byte[] bytes) throws IOException {
        Map<String, String> values = new HashMap<>();
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(
                new ByteArrayInputStream(bytes), StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null) {
                if (line.isEmpty()) {
                    continue;
                }
                int separator = line.indexOf('=');
                if (separator <= 0) {
                    throw new IOException("Invalid SDX cache manifest line");
                }
                String key = line.substring(0, separator);
                String value = line.substring(separator + 1);
                if (!SAFE_KEY.matcher(key).matches() || values.put(key, value) != null) {
                    throw new IOException("Invalid or duplicate SDX cache manifest key: " + key);
                }
            }
        }
        return values;
    }

    private static byte[] canonicalMapBytes(Map<String, String> values) throws IOException {
        TreeMap<String, String> sorted = new TreeMap<>(values);
        StringBuilder content = new StringBuilder();
        for (Map.Entry<String, String> entry : sorted.entrySet()) {
            if (!SAFE_KEY.matcher(entry.getKey()).matches()
                    || entry.getValue().indexOf('\n') >= 0
                    || entry.getValue().indexOf('\r') >= 0) {
                throw new IOException("Invalid SDX cache manifest value: " + entry.getKey());
            }
            content.append(entry.getKey())
                    .append('=')
                    .append(entry.getValue())
                    .append('\n');
        }
        return content.toString().getBytes(StandardCharsets.UTF_8);
    }

    private static byte[] readLimited(InputStream input) throws IOException {
        try (InputStream stream = input;
             ByteArrayOutputStream output = new ByteArrayOutputStream()) {
            byte[] buffer = new byte[8192];
            int total = 0;
            while (true) {
                int read = stream.read(buffer);
                if (read < 0) {
                    break;
                }
                total += read;
                if (total > MAX_MANIFEST_BYTES) {
                    throw new IOException("SDX cache manifest exceeds "
                            + MAX_MANIFEST_BYTES + " bytes");
                }
                output.write(buffer, 0, read);
            }
            return output.toByteArray();
        }
    }

    private static String required(Map<String, String> values, String key) throws IOException {
        String value = values.get(key);
        if (value == null || value.isEmpty()) {
            throw new IOException("SDX cache manifest is missing " + key);
        }
        return value;
    }

    private static int parseCount(
            Map<String, String> values, String key, int maximum) throws IOException {
        long parsed = parseLong(values, key);
        if (parsed < 0L || parsed > maximum) {
            throw new IOException("SDX cache manifest " + key + " is out of range");
        }
        return (int) parsed;
    }

    private static long parseLong(Map<String, String> values, String key) throws IOException {
        try {
            return Long.parseLong(required(values, key));
        } catch (NumberFormatException invalid) {
            throw new IOException("SDX cache manifest has invalid " + key, invalid);
        }
    }

    private static void requireSha256(String value, String label) throws IOException {
        if (value == null || !SHA256.matcher(value).matches()) {
            throw new IOException("Invalid " + label + " in SDX cache");
        }
    }

    private static void requireToken(String value, String label) throws IOException {
        if (value == null || value.trim().isEmpty()
                || value.indexOf('\n') >= 0 || value.indexOf('\r') >= 0) {
            throw new IOException("Invalid SDX " + label);
        }
    }

    private static void writeAtomically(Path destination, byte[] bytes) throws IOException {
        Files.createDirectories(destination.getParent());
        Path pending = destination.resolveSibling(
                "." + destination.getFileName() + "." + UUID.randomUUID() + ".pending");
        try {
            Files.write(
                    pending,
                    bytes,
                    StandardOpenOption.CREATE_NEW,
                    StandardOpenOption.WRITE);
            moveReplacing(pending, destination);
        } catch (Throwable failure) {
            Files.deleteIfExists(pending);
            if (failure instanceof IOException) {
                throw (IOException) failure;
            }
            throw new IOException("Could not write SDX cache metadata", failure);
        }
    }

    private static void moveReplacing(Path source, Path destination) throws IOException {
        try {
            Files.move(
                    source,
                    destination,
                    StandardCopyOption.ATOMIC_MOVE,
                    StandardCopyOption.REPLACE_EXISTING);
        } catch (AtomicMoveNotSupportedException unsupported) {
            Files.move(source, destination, StandardCopyOption.REPLACE_EXISTING);
        }
    }

    private static void moveWithoutReplace(Path source, Path destination) throws IOException {
        try {
            Files.move(source, destination, StandardCopyOption.ATOMIC_MOVE);
        } catch (AtomicMoveNotSupportedException unsupported) {
            Files.move(source, destination);
        }
    }

    static void deleteTree(Path root) throws IOException {
        if (root == null || !Files.exists(root)) {
            return;
        }
        IOException[] failure = new IOException[1];
        try (Stream<Path> stream = Files.walk(root)) {
            stream.sorted(Comparator.reverseOrder()).forEach(path -> {
                try {
                    Files.deleteIfExists(path);
                } catch (IOException error) {
                    if (failure[0] == null) {
                        failure[0] = error;
                    }
                }
            });
        }
        if (failure[0] != null) {
            throw failure[0];
        }
    }

    private static void copySourceEntry(
            ZipFile input, ZipEntry source, ZipOutputStream output) throws IOException {
        ZipEntry destination = new ZipEntry(source.getName());
        destination.setTime(ZIP_TIME);
        if (source.getMethod() == ZipEntry.STORED) {
            destination.setMethod(ZipEntry.STORED);
            destination.setSize(source.getSize());
            destination.setCompressedSize(source.getSize());
            destination.setCrc(source.getCrc());
        } else {
            destination.setMethod(ZipEntry.DEFLATED);
        }
        output.putNextEntry(destination);
        try (InputStream stream = new BufferedInputStream(input.getInputStream(source))) {
            copy(stream, output);
        }
        output.closeEntry();
    }

    private static void putStoredBytes(
            ZipOutputStream output, String name, byte[] bytes) throws IOException {
        CRC32 crc = new CRC32();
        crc.update(bytes);
        ZipEntry entry = storedEntry(name, bytes.length, crc.getValue());
        output.putNextEntry(entry);
        output.write(bytes);
        output.closeEntry();
    }

    private static void putStoredFile(
            ZipOutputStream output, String name, Path file) throws IOException {
        CRC32 crc = new CRC32();
        long size = 0L;
        byte[] buffer = new byte[1024 * 1024];
        try (InputStream input = new BufferedInputStream(Files.newInputStream(file))) {
            while (true) {
                int read = input.read(buffer);
                if (read < 0) {
                    break;
                }
                crc.update(buffer, 0, read);
                size += read;
            }
        }

        output.putNextEntry(storedEntry(name, size, crc.getValue()));
        try (InputStream input = new BufferedInputStream(Files.newInputStream(file))) {
            copy(input, output);
        }
        output.closeEntry();
    }

    private static ZipEntry storedEntry(String name, long size, long crc) throws IOException {
        SdxSourceIdentity.requireSafeEntryName(name);
        ZipEntry entry = new ZipEntry(name);
        entry.setTime(ZIP_TIME);
        entry.setMethod(ZipEntry.STORED);
        entry.setSize(size);
        entry.setCompressedSize(size);
        entry.setCrc(crc);
        return entry;
    }

    private static long copy(InputStream input, java.io.OutputStream output) throws IOException {
        byte[] buffer = new byte[1024 * 1024];
        long total = 0L;
        while (true) {
            int read = input.read(buffer);
            if (read < 0) {
                break;
            }
            output.write(buffer, 0, read);
            total += read;
        }
        return total;
    }

    private static final class FileRecord {
        private final String path;
        private final String sha256;
        private final long bytes;

        private FileRecord(String path, String sha256, long bytes) {
            this.path = path;
            this.sha256 = sha256;
            this.bytes = bytes;
        }
    }

    @FunctionalInterface
    interface IoSupplier<T> {
        T get() throws Exception;
    }

    public static final class MissingCompiledModelException extends IOException {
        public MissingCompiledModelException(String message) {
            super(message);
        }
    }
}
