/*
 * Copyright (c) Eclipse Deeplearning4j
 * SPDX-License-Identifier: Apache-2.0
 */
package org.nd4j.dsp.model;

import java.net.URI;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Provider-neutral Hugging Face GGUF/GGML repository resolution.
 *
 * <p>The resolver deliberately contains no HTTP or JSON implementation. Callers fetch
 * {@link #apiUri(Reference)} from Hugging Face, translate the returned sibling list to
 * {@link RepositoryFile} values, and pass it to {@link #resolve(Reference, String, List)}.
 * This keeps repository parsing, tree scoping, immutable download URLs, and ambiguous
 * quantization handling identical across desktop services and Android clients.</p>
 */
public final class HuggingFaceGgmlResolver {

    private static final Pattern REPOSITORY_ID = Pattern.compile(
            "[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}");
    private static final Pattern REVISION = Pattern.compile("[A-Za-z0-9][A-Za-z0-9._/-]{0,255}");
    private static final Pattern IMMUTABLE_REVISION = Pattern.compile("(?i)[0-9a-f]{40,64}");
    private static final Pattern CONTENT_SHA256 = Pattern.compile("(?i)[0-9a-f]{64}");
    private static final Pattern SPLIT_GGUF_SHARD = Pattern.compile(
            "(?i).+-[0-9]{5}-of-[0-9]{5}\\.gguf");
    private static final Pattern QUANTIZATION_HINT = Pattern.compile(
            "(?i)(IQ[1-4](?:_[A-Z0-9]+)+|Q[2-8](?:_[A-Z0-9]+)*|BF16|FP16|F16|INT8)");
    private static final List<String> TOKENIZER_ASSET_NAMES = List.of(
            "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
            "generation_config.json", "config.json", "chat_template.jinja");
    private static final int MAX_REPOSITORY_FILES = 100_000;

    private HuggingFaceGgmlResolver() {
    }

    public enum Kind {
        REPOSITORY,
        TREE,
        BLOB,
        RESOLVE
    }

    public static final class Reference {
        private final String repository;
        private final String requestedRevision;
        private final String requestedPath;
        private final Kind kind;
        private final String canonicalReference;

        private Reference(
                String repository,
                String requestedRevision,
                String requestedPath,
                Kind kind,
                String canonicalReference) {
            this.repository = repository;
            this.requestedRevision = requestedRevision;
            this.requestedPath = requestedPath;
            this.kind = kind;
            this.canonicalReference = canonicalReference;
        }

        public String getRepository() {
            return repository;
        }

        public String getRequestedRevision() {
            return requestedRevision;
        }

        public String getRequestedPath() {
            return requestedPath;
        }

        public Kind getKind() {
            return kind;
        }

        public String getCanonicalReference() {
            return canonicalReference;
        }

        public boolean isExactModel() {
            return requestedPath != null && isGgufOrGgml(requestedPath)
                    && (kind == Kind.BLOB || kind == Kind.RESOLVE);
        }
    }

    public static final class RepositoryFile {
        private final String path;
        private final long size;
        private final String sha256;

        public RepositoryFile(String path, long size) {
            this(path, size, null);
        }

        public RepositoryFile(String path, long size, String sha256) {
            this.path = requireRepositoryPath(path, "repository file");
            this.size = size;
            this.sha256 = canonicalSha256(sha256);
        }

        public String getPath() {
            return path;
        }

        public long getSize() {
            return size;
        }

        /** Optional immutable LFS content digest supplied by the repository API. */
        public String getSha256() {
            return sha256;
        }
    }

    public static final class Candidate {
        private final String path;
        private final long size;
        private final String format;
        private final String quantizationHint;
        private final URI downloadUri;
        private final boolean commitPinned;
        private final String sha256;
        private final List<TokenizerAsset> tokenizerAssets;

        private Candidate(
                String path,
                long size,
                String format,
                String quantizationHint,
                URI downloadUri,
                boolean commitPinned,
                String sha256,
                List<TokenizerAsset> tokenizerAssets) {
            this.path = path;
            this.size = size;
            this.format = format;
            this.quantizationHint = quantizationHint;
            this.downloadUri = downloadUri;
            this.commitPinned = commitPinned;
            this.sha256 = sha256;
            this.tokenizerAssets = List.copyOf(tokenizerAssets);
        }

        public String getPath() {
            return path;
        }

        public long getSize() {
            return size;
        }

        public String getFormat() {
            return format;
        }

        public String getQuantizationHint() {
            return quantizationHint;
        }

        public URI getDownloadUri() {
            return downloadUri;
        }

        /** True when the download URL names an immutable repository commit SHA. */
        public boolean isCommitPinned() {
            return commitPinned;
        }

        /** Optional immutable LFS content digest supplied by repository discovery. */
        public String getSha256() {
            return sha256;
        }

        /** Model-specific tokenizer/config files resolved from the same immutable commit. */
        public List<TokenizerAsset> getTokenizerAssets() {
            return tokenizerAssets;
        }
    }

    public static final class TokenizerAsset {
        private final String name;
        private final String path;
        private final long size;
        private final String sha256;
        private final URI downloadUri;

        private TokenizerAsset(
                String name, String path, long size, String sha256, URI downloadUri) {
            this.name = name;
            this.path = path;
            this.size = size;
            this.sha256 = sha256;
            this.downloadUri = downloadUri;
        }

        public String getName() {
            return name;
        }

        public String getPath() {
            return path;
        }

        public long getSize() {
            return size;
        }

        public String getSha256() {
            return sha256;
        }

        public URI getDownloadUri() {
            return downloadUri;
        }
    }

    public static final class Discovery {
        private final Reference reference;
        private final String resolvedRevision;
        private final List<Candidate> candidates;

        private Discovery(Reference reference, String resolvedRevision, List<Candidate> candidates) {
            this.reference = reference;
            this.resolvedRevision = resolvedRevision;
            this.candidates = List.copyOf(candidates);
        }

        public Reference getReference() {
            return reference;
        }

        public String getResolvedRevision() {
            return resolvedRevision;
        }

        public List<Candidate> getCandidates() {
            return candidates;
        }

        public boolean requiresSelection() {
            return candidates.size() > 1;
        }

        public Optional<Candidate> selectedCandidate() {
            return candidates.size() == 1 ? Optional.of(candidates.get(0)) : Optional.empty();
        }
    }

    /** Parse owner/repository or a canonical public repository/tree/blob/resolve URL. */
    public static Reference parse(String value) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException("Hugging Face repository or URL is required");
        }
        String input = value.trim();
        if (!input.contains("://")) {
            String repository = requireRepositoryId(input);
            return new Reference(
                    repository,
                    "main",
                    null,
                    Kind.REPOSITORY,
                    "https://huggingface.co/" + repository);
        }

        final URI uri;
        try {
            uri = URI.create(input);
        } catch (IllegalArgumentException invalid) {
            throw new IllegalArgumentException("Invalid Hugging Face URL", invalid);
        }
        String host = uri.getHost() == null ? "" : uri.getHost().toLowerCase(Locale.ROOT);
        if (!"https".equalsIgnoreCase(uri.getScheme())
                || (!"huggingface.co".equals(host) && !"www.huggingface.co".equals(host))
                || uri.getPort() != -1
                || uri.getUserInfo() != null
                || uri.getFragment() != null
                || !isCanonicalPath(uri.getRawPath())) {
            throw new IllegalArgumentException(
                    "Hugging Face URLs must be public canonical HTTPS URLs without credentials, "
                            + "ports, fragments, or path traversal");
        }

        String[] segments = uri.getRawPath().replaceFirst("^/+", "").split("/", -1);
        if (segments.length < 2) {
            throw new IllegalArgumentException("Hugging Face URL must identify an owner and repository");
        }
        String repository = requireRepositoryId(segments[0] + "/" + segments[1]);
        if (segments.length == 2) {
            if (uri.getRawQuery() != null) {
                throw unsupportedQuery();
            }
            return new Reference(
                    repository,
                    "main",
                    null,
                    Kind.REPOSITORY,
                    "https://huggingface.co/" + repository);
        }
        if (segments.length < 4) {
            throw new IllegalArgumentException("Hugging Face tree/blob/resolve URL is incomplete");
        }

        final Kind kind;
        switch (segments[2].toLowerCase(Locale.ROOT)) {
            case "tree":
                kind = Kind.TREE;
                break;
            case "blob":
                kind = Kind.BLOB;
                break;
            case "resolve":
                kind = Kind.RESOLVE;
                break;
            default:
                throw new IllegalArgumentException(
                        "Supported Hugging Face URLs are repository, tree, blob, or resolve URLs");
        }
        String rawQuery = uri.getRawQuery();
        if (rawQuery != null && !(kind == Kind.RESOLVE && "download=true".equals(rawQuery))) {
            throw unsupportedQuery();
        }
        String revision = requireRevision(decodePathSegment(segments[3], true, "revision"));
        String path = joinPath(segments, 4);
        if ((kind == Kind.BLOB || kind == Kind.RESOLVE) && path == null) {
            throw new IllegalArgumentException("Hugging Face blob/resolve URL must identify a file");
        }
        if ((kind == Kind.BLOB || kind == Kind.RESOLVE) && !isGgufOrGgml(path)) {
            throw new IllegalArgumentException("Hugging Face model URL must end in .gguf or .ggml");
        }
        if ((kind == Kind.BLOB || kind == Kind.RESOLVE) && isSplitGgufShard(path)) {
            throw splitGgufShardFailure();
        }
        String canonical = "https://huggingface.co/" + repository + "/"
                + kind.name().toLowerCase(Locale.ROOT) + "/" + encodePathSegment(revision)
                + (path == null ? "" : "/" + encodeRepositoryPath(path))
                + (rawQuery == null ? "" : "?download=true");
        return new Reference(repository, revision, path, kind, canonical);
    }

    /** Hugging Face model API endpoint used to obtain the immutable commit and sibling list. */
    public static URI apiUri(Reference reference) {
        requireReference(reference);
        return URI.create("https://huggingface.co/api/models/" + reference.getRepository()
                + "/revision/" + encodePathSegment(reference.getRequestedRevision())
                + "?blobs=true");
    }

    /** Resolve an already-exact blob/resolve URL without a repository API request. */
    public static Discovery exact(Reference reference) {
        requireReference(reference);
        if (!reference.isExactModel()) {
            throw new IllegalArgumentException("An exact Hugging Face GGUF/GGML file URL is required");
        }
        Candidate candidate = candidate(
                reference.getRepository(),
                reference.getRequestedRevision(),
                reference.getRequestedPath(),
                -1L,
                null,
                List.of());
        return new Discovery(reference, reference.getRequestedRevision(), List.of(candidate));
    }

    /**
     * Resolve repository files at the immutable commit returned by Hugging Face.
     * Multiple GGUF/GGML files remain explicit candidates; no first-file heuristic is used.
     */
    public static Discovery resolve(
            Reference reference,
            String resolvedRevision,
            List<RepositoryFile> repositoryFiles) {
        requireReference(reference);
        if (reference.isExactModel()) {
            return exact(reference);
        }
        if (resolvedRevision == null || !IMMUTABLE_REVISION.matcher(resolvedRevision.trim()).matches()) {
            throw new IllegalArgumentException(
                    "Hugging Face API did not return an immutable commit SHA");
        }
        if (repositoryFiles == null) {
            throw new IllegalArgumentException("Hugging Face API response omitted repository files");
        }
        if (repositoryFiles.size() > MAX_REPOSITORY_FILES) {
            throw new IllegalArgumentException("Hugging Face repository file list is unreasonably large");
        }

        String revision = resolvedRevision.trim().toLowerCase(Locale.ROOT);
        String scope = reference.getKind() == Kind.TREE ? reference.getRequestedPath() : null;
        Map<String, RepositoryFile> repositoryByPath = new LinkedHashMap<>();
        for (RepositoryFile file : repositoryFiles) {
            if (file == null) {
                continue;
            }
            RepositoryFile previous = repositoryByPath.putIfAbsent(file.getPath(), file);
            if (previous != null) {
                throw new IllegalArgumentException(
                        "Hugging Face API returned a duplicate repository path: " + file.getPath());
            }
        }
        Map<String, Candidate> byPath = new LinkedHashMap<>();
        boolean splitGgufFound = false;
        for (RepositoryFile file : repositoryByPath.values()) {
            if (!withinScope(file.getPath(), scope) || !isGgufOrGgml(file.getPath())) {
                continue;
            }
            if (isSplitGgufShard(file.getPath())) {
                splitGgufFound = true;
                continue;
            }
            Candidate previous = byPath.putIfAbsent(
                    file.getPath(),
                    candidate(
                            reference.getRepository(), revision, file.getPath(), file.getSize(),
                            file.getSha256(), tokenizerAssets(
                                    reference.getRepository(), revision, file.getPath(), repositoryByPath)));
            if (previous != null) {
                throw new IllegalArgumentException("Duplicate model candidate: " + file.getPath());
            }
        }
        List<Candidate> candidates = new ArrayList<>(byPath.values());
        candidates.sort(Comparator.comparing(Candidate::getPath));
        if (candidates.isEmpty()) {
            if (splitGgufFound) {
                throw splitGgufShardFailure();
            }
            throw new IllegalArgumentException(
                    "Hugging Face repository contains no GGUF/GGML files"
                            + (scope == null ? "" : " under " + scope));
        }
        return new Discovery(reference, revision, candidates);
    }

    public static String requireRepositoryId(String repository) {
        if (repository == null || !REPOSITORY_ID.matcher(repository).matches()) {
            throw new IllegalArgumentException(
                    "Hugging Face repository must be an owner/repository identifier");
        }
        return repository;
    }

    private static Candidate candidate(
            String repository,
            String revision,
            String path,
            long size,
            String sha256,
            List<TokenizerAsset> tokenizerAssets) {
        String safePath = requireRepositoryPath(path, "model file");
        if (isSplitGgufShard(safePath)) {
            throw splitGgufShardFailure();
        }
        String format = safePath.toLowerCase(Locale.ROOT).endsWith(".ggml") ? "ggml" : "gguf";
        Matcher matcher = QUANTIZATION_HINT.matcher(fileName(safePath));
        String quantization = matcher.find() ? matcher.group(1).toUpperCase(Locale.ROOT) : null;
        return new Candidate(
                safePath,
                size,
                format,
                quantization,
                downloadUri(repository, revision, safePath),
                IMMUTABLE_REVISION.matcher(revision).matches(),
                canonicalSha256(sha256),
                tokenizerAssets);
    }

    private static List<TokenizerAsset> tokenizerAssets(
            String repository,
            String revision,
            String modelPath,
            Map<String, RepositoryFile> repositoryByPath) {
        int separator = modelPath.lastIndexOf('/');
        String directory = separator < 0 ? null : modelPath.substring(0, separator);
        List<TokenizerAsset> assets = new ArrayList<>();
        for (String name : TOKENIZER_ASSET_NAMES) {
            String localPath = directory == null ? name : directory + "/" + name;
            RepositoryFile file = repositoryByPath.get(localPath);
            if (file == null && directory != null) {
                file = repositoryByPath.get(name);
            }
            if (file != null) {
                assets.add(new TokenizerAsset(
                        name,
                        file.getPath(),
                        file.getSize(),
                        file.getSha256(),
                        downloadUri(repository, revision, file.getPath())));
            }
        }
        return assets;
    }

    private static String canonicalSha256(String sha256) {
        if (sha256 == null || sha256.isBlank()) {
            return null;
        }
        String normalized = sha256.trim().toLowerCase(Locale.ROOT);
        if (!CONTENT_SHA256.matcher(normalized).matches()) {
            throw new IllegalArgumentException("Invalid Hugging Face LFS SHA-256 digest");
        }
        return normalized;
    }

    private static URI downloadUri(String repository, String revision, String path) {
        return URI.create("https://huggingface.co/" + repository + "/resolve/"
                + encodePathSegment(revision) + "/" + encodeRepositoryPath(path)
                + "?download=true");
    }

    private static void requireReference(Reference reference) {
        if (reference == null) {
            throw new IllegalArgumentException("Hugging Face reference is required");
        }
    }

    private static String requireRevision(String revision) {
        if (revision == null || !REVISION.matcher(revision).matches()) {
            throw new IllegalArgumentException("Invalid Hugging Face revision");
        }
        for (String segment : revision.split("/", -1)) {
            if (segment.isBlank() || ".".equals(segment) || "..".equals(segment)) {
                throw new IllegalArgumentException("Invalid Hugging Face revision");
            }
        }
        return revision;
    }

    private static String requireRepositoryPath(String path, String label) {
        if (path == null || path.isBlank() || path.indexOf('\\') >= 0 || path.startsWith("/")) {
            throw new IllegalArgumentException("Invalid Hugging Face " + label + " path");
        }
        for (String segment : path.split("/", -1)) {
            if (segment.isBlank() || ".".equals(segment) || "..".equals(segment)) {
                throw new IllegalArgumentException("Invalid Hugging Face " + label + " path");
            }
            for (int index = 0; index < segment.length(); index++) {
                if (Character.isISOControl(segment.charAt(index))) {
                    throw new IllegalArgumentException("Invalid Hugging Face " + label + " path");
                }
            }
        }
        return path;
    }

    private static String joinPath(String[] segments, int start) {
        if (segments.length <= start) {
            return null;
        }
        StringBuilder path = new StringBuilder();
        for (int index = start; index < segments.length; index++) {
            String segment = decodePathSegment(segments[index], false, "repository path");
            if (path.length() > 0) {
                path.append('/');
            }
            path.append(segment);
        }
        return path.toString();
    }

    private static boolean isCanonicalPath(String rawPath) {
        if (rawPath == null || rawPath.isBlank()
                || rawPath.indexOf('\\') >= 0 || rawPath.contains("//")) {
            return false;
        }
        for (String segment : rawPath.split("/", -1)) {
            if (".".equals(segment) || "..".equals(segment)) {
                return false;
            }
        }
        return true;
    }

    private static String decodePathSegment(String rawSegment, boolean allowSlash, String label) {
        final String decoded;
        try {
            String decodedPath = URI.create("https://huggingface.co/" + rawSegment).getPath();
            decoded = decodedPath.substring(1);
        } catch (RuntimeException invalid) {
            throw new IllegalArgumentException("Invalid Hugging Face " + label, invalid);
        }
        if (decoded.isBlank() || decoded.indexOf('\\') >= 0
                || (!allowSlash && decoded.indexOf('/') >= 0)
                || ".".equals(decoded) || "..".equals(decoded)) {
            throw new IllegalArgumentException("Invalid Hugging Face " + label);
        }
        for (int index = 0; index < decoded.length(); index++) {
            if (Character.isISOControl(decoded.charAt(index))) {
                throw new IllegalArgumentException("Invalid Hugging Face " + label);
            }
        }
        return decoded;
    }

    private static String encodePathSegment(String segment) {
        try {
            String encoded = new URI(null, null, "/" + segment, null).getRawPath().substring(1);
            return encoded.replace("/", "%2F");
        } catch (Exception invalid) {
            throw new IllegalArgumentException("Invalid Hugging Face path segment", invalid);
        }
    }

    private static String encodeRepositoryPath(String path) {
        StringBuilder encoded = new StringBuilder();
        for (String segment : requireRepositoryPath(path, "repository").split("/", -1)) {
            if (encoded.length() > 0) {
                encoded.append('/');
            }
            encoded.append(encodePathSegment(segment));
        }
        return encoded.toString();
    }

    private static IllegalArgumentException unsupportedQuery() {
        return new IllegalArgumentException(
                "Hugging Face URLs cannot contain query parameters except download=true "
                        + "on a resolve URL");
    }

    private static IllegalArgumentException splitGgufShardFailure() {
        return new IllegalArgumentException(
                "Split GGUF shard files cannot be acquired individually. Use a monolithic "
                        + "GGUF/GGML file or prepare the complete shard set on a host.");
    }

    private static boolean withinScope(String path, String scope) {
        return scope == null || scope.isBlank() || path.equals(scope) || path.startsWith(scope + "/");
    }

    private static boolean isGgufOrGgml(String path) {
        if (path == null) {
            return false;
        }
        String lower = path.toLowerCase(Locale.ROOT);
        return lower.endsWith(".gguf") || lower.endsWith(".ggml");
    }

    private static boolean isSplitGgufShard(String path) {
        return path != null && SPLIT_GGUF_SHARD.matcher(fileName(path)).matches();
    }

    private static String fileName(String path) {
        int slash = path.lastIndexOf('/');
        return slash < 0 ? path : path.substring(slash + 1);
    }
}
