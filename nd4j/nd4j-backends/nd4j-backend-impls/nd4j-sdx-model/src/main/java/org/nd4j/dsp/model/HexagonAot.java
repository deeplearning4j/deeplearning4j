/*
 * Copyright (c) Eclipse Deeplearning4j
 * SPDX-License-Identifier: Apache-2.0
 */
package org.nd4j.dsp.model;

import java.io.IOException;
import java.math.BigDecimal;
import java.math.BigInteger;
import java.nio.charset.StandardCharsets;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.LinkOption;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.regex.Pattern;

/**
 * Deterministic Qualcomm Hexagon/HTP AOT planning and artifact validation for SDX.
 *
 * <p>This is compiler-side metadata for the existing SDZ compile/cache flow, not
 * an application model format. Its JSON and sidecar contract remains stable for
 * existing Qualcomm vendor compilation workflows.</p>
 */
final class HexagonAot {
    static final int FORMAT_VERSION = 1;
    static final String CACHE_ABI = "sdx-hexagon-aot-v1";
    static final int ADAPTER_ABI = 1;
    static final String RANGE_SEMANTICS = "inclusive";
    static final String MANIFEST_NAME = "hexagon-aot-manifest.json";

    private static final Pattern SOC = Pattern.compile("^[A-Za-z0-9._-]+$");
    private static final Pattern ARTIFACT =
            Pattern.compile("^hexagon_([0-9]+)_([0-9]+)_([0-9a-f]{16})[.](bin|meta)$");
    private static final Pattern SHA256 = Pattern.compile("^[0-9a-f]{64}$");
    private static final BigInteger UINT64_MODULUS = BigInteger.ONE.shiftLeft(64);
    private static final List<String> METADATA_ORDER = List.of(
            "cacheAbi", "adapterAbi", "soc", "rangeSemantics", "startSlot",
            "endSlot", "shapeKey", "byteSize", "sha256");

    private HexagonAot() {
    }

    static int plan(
            Path segmentsJson,
            String soc,
            String modelId,
            Path output,
            boolean includeNoncapturable,
            boolean allowUnstable,
            boolean allowEmpty) throws IOException {
        Object parsed = Json.parse(Files.readString(segmentsJson, StandardCharsets.UTF_8));
        if (!(parsed instanceof List)) {
            throw invalid("segments JSON must be an array");
        }
        if (!SOC.matcher(soc).matches()) {
            throw invalid(
                    "SoC must contain only ASCII letters, digits, dot, underscore, or dash");
        }

        List<Map<String, Object>> segments = new ArrayList<>();
        List<Map<String, Object>> skipped = new ArrayList<>();
        for (Object item : (List<?>) parsed) {
            if (!(item instanceof Map)) {
                throw invalid("segment entry is not an object: " + item);
            }
            @SuppressWarnings("unchecked")
            Map<String, Object> source = (Map<String, Object>) item;
            Normalized normalized =
                    normalize(source, includeNoncapturable, allowUnstable);
            if (normalized.segment != null) {
                segments.add(normalized.segment);
            }
            if (normalized.skipped != null) {
                skipped.add(normalized.skipped);
            }
        }
        segments.sort(Comparator
                .comparingLong(
                        (Map<String, Object> value) ->
                                integerUnchecked(value, "startSlot"))
                .thenComparingLong(value -> integerUnchecked(value, "endSlot"))
                .thenComparingLong(value -> integerUnchecked(value, "index")));

        Set<String> artifacts = new TreeSet<>();
        for (Map<String, Object> segment : segments) {
            if (!artifacts.add(string(segment, "artifact"))) {
                throw invalid("duplicate range/shape artifact names in replay plan");
            }
        }
        if (segments.isEmpty() && !allowEmpty) {
            throw invalid("replay plan produced no eligible Hexagon AOT segments");
        }

        Map<String, Object> request = new LinkedHashMap<>();
        request.put("formatVersion", BigDecimal.valueOf(FORMAT_VERSION));
        request.put("cacheAbi", CACHE_ABI);
        request.put("adapterAbi", BigDecimal.valueOf(ADAPTER_ABI));
        request.put("soc", soc);
        request.put("modelId", modelId == null ? fileStem(segmentsJson) : modelId);
        request.put("rangeSemantics", RANGE_SEMANTICS);
        request.put("sourceSegmentsSha256", sha256(Json.canonical(parsed)));
        request.put("segments", segments);
        request.put("skippedSegments", skipped);
        writeJson(output, request);
        return segments.size();
    }

    static int finalizeArtifacts(Path requestPath, Path kernelDirectory) throws IOException {
        Map<String, Object> request = loadRequest(requestPath);
        Path kernelRoot = kernelDirectory.toAbsolutePath().normalize();
        Files.createDirectories(kernelRoot);
        requireDirectory(kernelRoot);
        for (Map<String, Object> segment : segments(request)) {
            Path payloadPath = directChild(
                    kernelRoot, string(segment, "artifact"));
            if (!Files.isRegularFile(payloadPath, LinkOption.NOFOLLOW_LINKS)) {
                throw invalid("compiled kernel is missing: " + payloadPath);
            }
            byte[] payload = Files.readAllBytes(payloadPath);
            if (payload.length == 0) {
                throw invalid("compiled kernel is empty: " + payloadPath);
            }
            Path metadataPath = directChild(
                    kernelRoot, string(segment, "metadata"));
            writeAtomic(metadataPath, metadataText(expectedMetadata(request, segment, payload)));
        }

        List<Map<String, Object>> entries =
                verifyArtifacts(request, kernelRoot, false);
        Map<String, Object> manifest =
                expectedManifest(requestPath, request, entries);
        writeJson(directChild(kernelRoot, MANIFEST_NAME), manifest);
        verifyManifest(requestPath, request, kernelRoot, entries);
        return entries.size();
    }

    static int verify(Path requestPath, Path kernelDirectory) throws IOException {
        Map<String, Object> request = loadRequest(requestPath);
        Path kernelRoot = kernelDirectory.toAbsolutePath().normalize();
        List<Map<String, Object>> entries =
                verifyArtifacts(request, kernelRoot, true);
        verifyManifest(requestPath, request, kernelRoot, entries);
        return entries.size();
    }

    private static Normalized normalize(
            Map<String, Object> source,
            boolean includeNoncapturable,
            boolean allowUnstable) throws IOException {
        long index = integer(source, "index");
        long start = integer(source, "startSlot");
        long end = integer(source, "endSlot");
        long numOps = integer(source, "numOps");
        if (start < 0 || end < start) {
            throw invalid("invalid inclusive segment range: " + start + ".." + end);
        }
        long expectedOps;
        try {
            expectedOps = Math.addExact(Math.subtractExact(end, start), 1);
        } catch (ArithmeticException overflow) {
            throw invalid("invalid inclusive segment range: " + start + ".." + end);
        }
        if (numOps != expectedOps) {
            throw invalid("segment " + index + " numOps=" + numOps + ", expected "
                    + expectedOps + " for inclusive bounds");
        }

        BigInteger shapeKey = shapeKey(source.getOrDefault("shapeKey", BigDecimal.ZERO));
        String shapeStatus = String.valueOf(
                source.getOrDefault("shapeKeyStatus", "UNSET"));
        List<Object> reasons = new ArrayList<>();
        if (!booleanValue(source.getOrDefault("isCapturable", Boolean.FALSE))
                && !includeNoncapturable) {
            reasons.add("not-capturable");
        }
        if (booleanValue(source.getOrDefault("compilationFailed", Boolean.FALSE))) {
            reasons.add("prior-compilation-failed");
        }
        if (shapeKey.signum() == 0) {
            reasons.add("shape-key-unset");
        }
        if (!"STABLE".equals(shapeStatus) && !allowUnstable) {
            reasons.add("shape-key-" + shapeStatus.toLowerCase(java.util.Locale.ROOT));
        }

        String shapeHex = String.format("%016x", shapeKey);
        String base = "hexagon_" + start + "_" + end + "_" + shapeHex;
        Map<String, Object> normalized = new LinkedHashMap<>();
        normalized.put("index", BigDecimal.valueOf(index));
        normalized.put("startSlot", BigDecimal.valueOf(start));
        normalized.put("endSlot", BigDecimal.valueOf(end));
        normalized.put("numOps", BigDecimal.valueOf(numOps));
        normalized.put("rangeSemantics", RANGE_SEMANTICS);
        normalized.put("shapeKey", new BigDecimal(shapeKey));
        normalized.put("shapeKeyHex", shapeHex);
        normalized.put("shapeKeyStatus", shapeStatus);
        normalized.put("artifact", base + ".bin");
        normalized.put("metadata", base + ".meta");

        Object operations = source.getOrDefault("ops", new LinkedHashMap<>());
        if (!(operations instanceof Map)) {
            throw invalid("segment ops must be an object");
        }
        @SuppressWarnings("unchecked")
        Map<String, Object> ops = (Map<String, Object>) operations;
        normalized.put("ops", new TreeMap<>(ops));

        if (reasons.isEmpty()) {
            return new Normalized(normalized, null);
        }
        Map<String, Object> skip = new LinkedHashMap<>();
        skip.put("index", BigDecimal.valueOf(index));
        skip.put("reasons", reasons);
        return new Normalized(null, skip);
    }

    private static Map<String, Object> loadRequest(Path path) throws IOException {
        Object value = Json.parse(Files.readString(path, StandardCharsets.UTF_8));
        if (!(value instanceof Map)) {
            throw invalid("AOT request must be an object");
        }
        @SuppressWarnings("unchecked")
        Map<String, Object> request = (Map<String, Object>) value;
        if (integer(request, "formatVersion") != FORMAT_VERSION) {
            throw invalid("unsupported AOT request formatVersion");
        }
        if (!CACHE_ABI.equals(request.get("cacheAbi"))) {
            throw invalid("unsupported Hexagon cache ABI");
        }
        if (integer(request, "adapterAbi") != ADAPTER_ABI) {
            throw invalid("unsupported Hexagon adapter ABI");
        }
        if (!RANGE_SEMANTICS.equals(request.get("rangeSemantics"))) {
            throw invalid("Hexagon segment ranges must be inclusive");
        }
        Object soc = request.get("soc");
        if (!(soc instanceof String) || !SOC.matcher((String) soc).matches()) {
            throw invalid("invalid or missing request SoC");
        }
        String modelId = string(request, "modelId");
        if (modelId.trim().isEmpty()) {
            throw invalid("request modelId must not be empty");
        }
        String sourceSegmentsSha256 = string(request, "sourceSegmentsSha256");
        if (!SHA256.matcher(sourceSegmentsSha256).matches()) {
            throw invalid("invalid sourceSegmentsSha256");
        }

        Set<String> artifacts = new TreeSet<>();
        Set<String> metadata = new TreeSet<>();
        for (Map<String, Object> segment : segments(request)) {
            validateRequestSegment(segment);
            if (!artifacts.add(string(segment, "artifact"))) {
                throw invalid("duplicate request artifact name");
            }
            if (!metadata.add(string(segment, "metadata"))) {
                throw invalid("duplicate request metadata name");
            }
        }
        return request;
    }

    private static List<Map<String, Object>> verifyArtifacts(
            Map<String, Object> request,
            Path kernelDirectory,
            boolean requireManifest) throws IOException {
        requireDirectory(kernelDirectory);
        Set<String> requiredNames = new TreeSet<>();
        for (Map<String, Object> segment : segments(request)) {
            requiredNames.add(string(segment, "artifact"));
            requiredNames.add(string(segment, "metadata"));
        }
        if (requireManifest) {
            requiredNames.add(MANIFEST_NAME);
        }
        Set<String> allowedNames = new TreeSet<>(requiredNames);
        allowedNames.add(MANIFEST_NAME);
        Set<String> actualNames = new TreeSet<>();
        try (DirectoryStream<Path> files = Files.newDirectoryStream(kernelDirectory)) {
            for (Path file : files) {
                String name = file.getFileName().toString();
                if (!Files.isRegularFile(file, LinkOption.NOFOLLOW_LINKS)) {
                    throw invalid("kernel directory entry is not a regular file: " + name);
                }
                actualNames.add(name);
            }
        }
        Set<String> unexpected = new TreeSet<>(actualNames);
        unexpected.removeAll(allowedNames);
        if (!unexpected.isEmpty()) {
            throw invalid("unexpected Hexagon AOT files: " + String.join(", ", unexpected));
        }
        Set<String> missing = new TreeSet<>(requiredNames);
        missing.removeAll(actualNames);
        if (!missing.isEmpty()) {
            throw invalid("missing Hexagon AOT files: " + String.join(", ", missing));
        }

        List<Map<String, Object>> entries = new ArrayList<>();
        for (Map<String, Object> segment : segments(request)) {
            String binName = string(segment, "artifact");
            String metadataName = string(segment, "metadata");
            Path payloadPath = directChild(kernelDirectory, binName);
            byte[] payload = Files.readAllBytes(payloadPath);
            if (payload.length == 0) {
                throw invalid("empty Hexagon kernel: " + payloadPath);
            }
            Map<String, String> expected = expectedMetadata(request, segment, payload);
            Map<String, String> actual =
                    readMetadata(directChild(kernelDirectory, metadataName));
            if (!actual.equals(expected)) {
                throw invalid("metadata mismatch for " + binName
                        + ": expected " + expected + ", got " + actual);
            }

            Map<String, Object> entry = new LinkedHashMap<>();
            entry.put("artifact", binName);
            entry.put("metadata", metadataName);
            entry.put("startSlot", required(segment, "startSlot"));
            entry.put("endSlot", required(segment, "endSlot"));
            entry.put("shapeKeyHex", required(segment, "shapeKeyHex"));
            entry.put("byteSize", BigDecimal.valueOf(payload.length));
            entry.put("sha256", expected.get("sha256"));
            entries.add(entry);
        }
        return entries;
    }

    private static void validateRequestSegment(Map<String, Object> segment)
            throws IOException {
        long index = integer(segment, "index");
        long start = integer(segment, "startSlot");
        long end = integer(segment, "endSlot");
        long numOps = integer(segment, "numOps");
        if (index < 0 || start < 0 || end < start) {
            throw invalid("invalid request segment range or index");
        }
        long expectedOps;
        try {
            expectedOps = Math.addExact(Math.subtractExact(end, start), 1);
        } catch (ArithmeticException overflow) {
            throw invalid("invalid request segment range");
        }
        if (numOps != expectedOps) {
            throw invalid("request segment numOps does not match inclusive range");
        }
        if (!RANGE_SEMANTICS.equals(segment.get("rangeSemantics"))) {
            throw invalid("request segment ranges must be inclusive");
        }

        Object shapeValue = required(segment, "shapeKey");
        if (!(shapeValue instanceof BigDecimal)) {
            throw invalid("request shapeKey must be an unsigned 64-bit JSON integer");
        }
        final BigInteger shapeKey;
        try {
            shapeKey = ((BigDecimal) shapeValue).toBigIntegerExact();
        } catch (ArithmeticException malformed) {
            throw invalid("request shapeKey must be an unsigned 64-bit JSON integer");
        }
        if (shapeKey.signum() < 0 || shapeKey.compareTo(UINT64_MODULUS) >= 0) {
            throw invalid("request shapeKey is outside the unsigned 64-bit range");
        }
        String shapeHex = string(segment, "shapeKeyHex");
        String expectedShapeHex = String.format("%016x", shapeKey);
        if (!expectedShapeHex.equals(shapeHex)) {
            throw invalid("request shapeKeyHex does not match shapeKey");
        }

        String base = "hexagon_" + start + "_" + end + "_" + shapeHex;
        String artifact = string(segment, "artifact");
        String metadata = string(segment, "metadata");
        if (!(base + ".bin").equals(artifact)
                || !(base + ".meta").equals(metadata)
                || !ARTIFACT.matcher(artifact).matches()
                || !ARTIFACT.matcher(metadata).matches()) {
            throw invalid("request artifact and metadata names must match the segment range and shape");
        }
        Object operations = required(segment, "ops");
        if (!(operations instanceof Map)) {
            throw invalid("request segment ops must be an object");
        }
    }

    private static Map<String, Object> expectedManifest(
            Path requestPath,
            Map<String, Object> request,
            List<Map<String, Object>> entries) throws IOException {
        Map<String, Object> manifest = new LinkedHashMap<>();
        manifest.put("formatVersion", BigDecimal.valueOf(FORMAT_VERSION));
        manifest.put("cacheAbi", CACHE_ABI);
        manifest.put("adapterAbi", BigDecimal.valueOf(ADAPTER_ABI));
        manifest.put("soc", request.get("soc"));
        manifest.put("modelId", required(request, "modelId"));
        manifest.put("rangeSemantics", RANGE_SEMANTICS);
        manifest.put("requestSha256", sha256(Files.readAllBytes(requestPath)));
        manifest.put("artifacts", entries);
        return manifest;
    }

    private static void verifyManifest(
            Path requestPath,
            Map<String, Object> request,
            Path kernelDirectory,
            List<Map<String, Object>> entries) throws IOException {
        Path manifestPath = directChild(kernelDirectory, MANIFEST_NAME);
        Object parsed = Json.parse(Files.readString(manifestPath, StandardCharsets.UTF_8));
        if (!(parsed instanceof Map)
                || !parsed.equals(expectedManifest(requestPath, request, entries))) {
            throw invalid("Hexagon AOT manifest does not match the request and artifacts");
        }
    }

    private static void requireDirectory(Path directory) throws IOException {
        if (!Files.isDirectory(directory, LinkOption.NOFOLLOW_LINKS)) {
            throw invalid("kernel directory not found or is a symbolic link: " + directory);
        }
    }

    private static Path directChild(Path directory, String name) throws IOException {
        Path normalizedDirectory = directory.toAbsolutePath().normalize();
        Path resolved = normalizedDirectory.resolve(name).normalize();
        if (resolved.getParent() == null
                || !resolved.getParent().equals(normalizedDirectory)) {
            throw invalid("artifact path escapes the kernel directory: " + name);
        }
        return resolved;
    }

    private static Map<String, String> expectedMetadata(
            Map<String, Object> request,
            Map<String, Object> segment,
            byte[] payload) {
        Map<String, String> result = new LinkedHashMap<>();
        result.put("cacheAbi", CACHE_ABI);
        result.put("adapterAbi", Integer.toString(ADAPTER_ABI));
        result.put("soc", String.valueOf(request.get("soc")));
        result.put("rangeSemantics", RANGE_SEMANTICS);
        result.put("startSlot", Long.toString(integerUnchecked(segment, "startSlot")));
        result.put("endSlot", Long.toString(integerUnchecked(segment, "endSlot")));
        result.put("shapeKey", String.valueOf(segment.get("shapeKeyHex")));
        result.put("byteSize", Integer.toString(payload.length));
        result.put("sha256", sha256(payload));
        return result;
    }

    private static Map<String, String> readMetadata(Path path) throws IOException {
        Map<String, String> result = new LinkedHashMap<>();
        List<String> lines;
        try {
            lines = Files.readAllLines(path, StandardCharsets.UTF_8);
        } catch (IOException failure) {
            throw invalid("cannot read metadata " + path + ": " + failure.getMessage());
        }
        for (int i = 0; i < lines.size(); i++) {
            String line = lines.get(i).trim();
            if (line.isEmpty() || line.startsWith("#")) {
                continue;
            }
            int separator = line.indexOf('=');
            if (separator < 0) {
                throw invalid(path + ":" + (i + 1) + ": expected key=value");
            }
            String key = line.substring(0, separator);
            if (result.containsKey(key)) {
                throw invalid(path + ":" + (i + 1) + ": duplicate key " + key);
            }
            result.put(key, line.substring(separator + 1));
        }
        return result;
    }

    private static byte[] metadataText(Map<String, String> values) {
        StringBuilder result = new StringBuilder();
        for (String key : METADATA_ORDER) {
            result.append(key).append('=').append(values.get(key)).append('\n');
        }
        return result.toString().getBytes(StandardCharsets.UTF_8);
    }

    private static void writeJson(Path path, Object value) throws IOException {
        writeAtomic(path, (Json.pretty(value) + "\n").getBytes(StandardCharsets.UTF_8));
    }

    private static void writeAtomic(Path path, byte[] payload) throws IOException {
        Path parent = path.toAbsolutePath().normalize().getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }
        Path temporary = path.resolveSibling(path.getFileName() + ".tmp");
        Files.write(temporary, payload);
        try {
            Files.move(
                    temporary,
                    path,
                    StandardCopyOption.ATOMIC_MOVE,
                    StandardCopyOption.REPLACE_EXISTING);
        } catch (AtomicMoveNotSupportedException ignored) {
            Files.move(temporary, path, StandardCopyOption.REPLACE_EXISTING);
        }
    }

    private static BigInteger shapeKey(Object value) throws IOException {
        final BigInteger parsed;
        try {
            String text;
            if (value instanceof BigDecimal) {
                text = ((BigDecimal) value).toBigIntegerExact().toString();
            } else {
                text = String.valueOf(value);
            }
            int sign = 1;
            if (text.startsWith("-")) {
                sign = -1;
                text = text.substring(1);
            } else if (text.startsWith("+")) {
                text = text.substring(1);
            }
            int radix = 10;
            if (text.startsWith("0x") || text.startsWith("0X")) {
                radix = 16;
                text = text.substring(2);
            } else if (text.startsWith("0o") || text.startsWith("0O")) {
                radix = 8;
                text = text.substring(2);
            } else if (text.startsWith("0b") || text.startsWith("0B")) {
                radix = 2;
                text = text.substring(2);
            }
            parsed = new BigInteger(text, radix).multiply(BigInteger.valueOf(sign));
        } catch (ArithmeticException | NumberFormatException failure) {
            throw invalid("invalid shape key: " + value);
        }
        return parsed.mod(UINT64_MODULUS);
    }

    private static boolean booleanValue(Object value) {
        if (value == null || Boolean.FALSE.equals(value)) {
            return false;
        }
        if (value instanceof BigDecimal) {
            return ((BigDecimal) value).signum() != 0;
        }
        if (value instanceof String) {
            return !((String) value).isEmpty();
        }
        return true;
    }

    private static long integer(Map<String, Object> value, String key) throws IOException {
        Object found = required(value, key);
        try {
            if (found instanceof BigDecimal) {
                return ((BigDecimal) found).longValueExact();
            }
            if (found instanceof String) {
                return Long.parseLong((String) found);
            }
        } catch (ArithmeticException | NumberFormatException failure) {
            throw invalid("malformed integer field " + key + ": " + found);
        }
        throw invalid("malformed integer field " + key + ": " + found);
    }

    private static long integerUnchecked(Map<String, Object> value, String key) {
        try {
            return integer(value, key);
        } catch (IOException impossible) {
            throw new IllegalStateException(impossible);
        }
    }

    private static Object required(Map<String, Object> value, String key) throws IOException {
        if (!value.containsKey(key)) {
            throw invalid("missing request field: " + key);
        }
        return value.get(key);
    }

    private static String string(Map<String, Object> value, String key) throws IOException {
        Object found = required(value, key);
        if (!(found instanceof String)) {
            throw invalid("request field " + key + " must be a string");
        }
        return (String) found;
    }

    @SuppressWarnings("unchecked")
    private static List<Map<String, Object>> segments(Map<String, Object> request)
            throws IOException {
        Object found = request.get("segments");
        if (!(found instanceof List)) {
            throw invalid("request segments must be an array");
        }
        for (Object segment : (List<?>) found) {
            if (!(segment instanceof Map)) {
                throw invalid("request segment must be an object");
            }
        }
        return (List<Map<String, Object>>) found;
    }

    private static String fileStem(Path path) {
        String name = path.getFileName().toString();
        int dot = name.lastIndexOf('.');
        return dot <= 0 ? name : name.substring(0, dot);
    }

    private static String sha256(byte[] payload) {
        try {
            byte[] digest = MessageDigest.getInstance("SHA-256").digest(payload);
            StringBuilder result = new StringBuilder(64);
            for (byte value : digest) {
                result.append(String.format("%02x", value & 0xff));
            }
            return result.toString();
        } catch (NoSuchAlgorithmException impossible) {
            throw new IllegalStateException("JVM does not provide SHA-256", impossible);
        }
    }

    private static IOException invalid(String message) {
        return new IOException("Invalid Hexagon AOT contract: " + message);
    }

    private static final class Normalized {
        private final Map<String, Object> segment;
        private final Map<String, Object> skipped;

        private Normalized(
                Map<String, Object> segment, Map<String, Object> skipped) {
            this.segment = segment;
            this.skipped = skipped;
        }
    }

    /** Strict, dependency-free JSON codec with lexicographically sorted object keys. */
    private static final class Json {
        private Json() {
        }

        private static Object parse(String input) throws IOException {
            return new Parser(input).parse();
        }

        private static byte[] canonical(Object value) {
            return render(value, false).getBytes(StandardCharsets.UTF_8);
        }

        private static String pretty(Object value) {
            return render(value, true);
        }

        private static String render(Object value, boolean pretty) {
            StringBuilder result = new StringBuilder();
            append(result, value, pretty, 0);
            return result.toString();
        }

        private static void append(
                StringBuilder out, Object value, boolean pretty, int depth) {
            if (value == null) {
                out.append("null");
            } else if (value instanceof String) {
                quote(out, (String) value);
            } else if (value instanceof Boolean) {
                out.append(value);
            } else if (value instanceof BigDecimal) {
                BigDecimal number = (BigDecimal) value;
                out.append(number.scale() <= 0
                        ? number.toBigIntegerExact()
                        : number.toString());
            } else if (value instanceof Map) {
                @SuppressWarnings("unchecked")
                Map<String, Object> object = (Map<String, Object>) value;
                out.append('{');
                if (!object.isEmpty()) {
                    int position = 0;
                    for (Map.Entry<String, Object> entry :
                            new TreeMap<>(object).entrySet()) {
                        if (position++ > 0) {
                            out.append(',');
                        }
                        newlineIndent(out, pretty, depth + 1);
                        quote(out, entry.getKey());
                        out.append(pretty ? ": " : ":");
                        append(out, entry.getValue(), pretty, depth + 1);
                    }
                    newlineIndent(out, pretty, depth);
                }
                out.append('}');
            } else if (value instanceof List) {
                List<?> array = (List<?>) value;
                out.append('[');
                for (int i = 0; i < array.size(); i++) {
                    if (i > 0) {
                        out.append(',');
                    }
                    newlineIndent(out, pretty, depth + 1);
                    append(out, array.get(i), pretty, depth + 1);
                }
                if (!array.isEmpty()) {
                    newlineIndent(out, pretty, depth);
                }
                out.append(']');
            } else {
                throw new IllegalArgumentException("Unsupported JSON value: " + value);
            }
        }

        private static void newlineIndent(
                StringBuilder out, boolean pretty, int depth) {
            if (pretty) {
                out.append('\n');
                for (int i = 0; i < depth * 2; i++) {
                    out.append(' ');
                }
            }
        }

        private static void quote(StringBuilder out, String value) {
            out.append('"');
            for (int i = 0; i < value.length(); i++) {
                char c = value.charAt(i);
                switch (c) {
                    case '"':
                        out.append("\\\"");
                        break;
                    case '\\':
                        out.append("\\\\");
                        break;
                    case '\b':
                        out.append("\\b");
                        break;
                    case '\f':
                        out.append("\\f");
                        break;
                    case '\n':
                        out.append("\\n");
                        break;
                    case '\r':
                        out.append("\\r");
                        break;
                    case '\t':
                        out.append("\\t");
                        break;
                    default:
                        if (c < 0x20) {
                            out.append(String.format("\\u%04x", (int) c));
                        } else {
                            out.append(c);
                        }
                }
            }
            out.append('"');
        }

        private static final class Parser {
            private final String input;
            private int position;
            private int depth;

            private Parser(String input) throws IOException {
                if (input == null || input.isEmpty()) {
                    throw invalid("JSON document is empty");
                }
                this.input = input;
            }

            private Object parse() throws IOException {
                Object result = value();
                whitespace();
                if (position != input.length()) {
                    throw syntax("unexpected trailing content");
                }
                return result;
            }

            private Object value() throws IOException {
                whitespace();
                if (position >= input.length()) {
                    throw syntax("expected a value");
                }
                if (++depth > 64) {
                    throw syntax("maximum nesting depth exceeded");
                }
                try {
                    char c = input.charAt(position);
                    switch (c) {
                        case '{':
                            return object();
                        case '[':
                            return array();
                        case '"':
                            return string();
                        case 't':
                            literal("true");
                            return Boolean.TRUE;
                        case 'f':
                            literal("false");
                            return Boolean.FALSE;
                        case 'n':
                            literal("null");
                            return null;
                        default:
                            if (c == '-' || Character.isDigit(c)) {
                                return number();
                            }
                            throw syntax("unexpected character '" + c + "'");
                    }
                } finally {
                    depth--;
                }
            }

            private Map<String, Object> object() throws IOException {
                expect('{');
                whitespace();
                Map<String, Object> result = new LinkedHashMap<>();
                if (take('}')) {
                    return result;
                }
                while (true) {
                    whitespace();
                    if (position >= input.length() || input.charAt(position) != '"') {
                        throw syntax("object key must be a string");
                    }
                    String key = string();
                    whitespace();
                    expect(':');
                    Object value = value();
                    if (result.containsKey(key)) {
                        throw syntax("duplicate object key: " + key);
                    }
                    result.put(key, value);
                    whitespace();
                    if (take('}')) {
                        return result;
                    }
                    expect(',');
                }
            }

            private List<Object> array() throws IOException {
                expect('[');
                whitespace();
                List<Object> result = new ArrayList<>();
                if (take(']')) {
                    return result;
                }
                while (true) {
                    result.add(value());
                    whitespace();
                    if (take(']')) {
                        return result;
                    }
                    expect(',');
                }
            }

            private String string() throws IOException {
                expect('"');
                StringBuilder result = new StringBuilder();
                while (position < input.length()) {
                    char c = input.charAt(position++);
                    if (c == '"') {
                        return result.toString();
                    }
                    if (c == '\\') {
                        if (position >= input.length()) {
                            throw syntax("unterminated string escape");
                        }
                        char escaped = input.charAt(position++);
                        switch (escaped) {
                            case '"':
                            case '\\':
                            case '/':
                                result.append(escaped);
                                break;
                            case 'b':
                                result.append('\b');
                                break;
                            case 'f':
                                result.append('\f');
                                break;
                            case 'n':
                                result.append('\n');
                                break;
                            case 'r':
                                result.append('\r');
                                break;
                            case 't':
                                result.append('\t');
                                break;
                            case 'u':
                                result.append(unicode());
                                break;
                            default:
                                throw syntax("invalid string escape");
                        }
                    } else {
                        if (c < 0x20) {
                            throw syntax("unescaped control character in string");
                        }
                        result.append(c);
                    }
                }
                throw syntax("unterminated string");
            }

            private char unicode() throws IOException {
                if (position + 4 > input.length()) {
                    throw syntax("incomplete unicode escape");
                }
                int value = 0;
                for (int i = 0; i < 4; i++) {
                    int digit = Character.digit(input.charAt(position++), 16);
                    if (digit < 0) {
                        throw syntax("invalid unicode escape");
                    }
                    value = (value << 4) | digit;
                }
                return (char) value;
            }

            private BigDecimal number() throws IOException {
                int start = position;
                take('-');
                if (take('0')) {
                    if (position < input.length()
                            && Character.isDigit(input.charAt(position))) {
                        throw syntax("leading zero in number");
                    }
                } else {
                    digits();
                }
                if (take('.')) {
                    digits();
                }
                if (take('e') || take('E')) {
                    if (!take('+')) {
                        take('-');
                    }
                    digits();
                }
                try {
                    return new BigDecimal(input.substring(start, position));
                } catch (NumberFormatException failure) {
                    throw syntax("invalid number");
                }
            }

            private void digits() throws IOException {
                int start = position;
                while (position < input.length()
                        && Character.isDigit(input.charAt(position))) {
                    position++;
                }
                if (position == start) {
                    throw syntax("expected decimal digit");
                }
            }

            private void literal(String literal) throws IOException {
                if (!input.startsWith(literal, position)) {
                    throw syntax("invalid literal");
                }
                position += literal.length();
            }

            private void whitespace() {
                while (position < input.length()) {
                    char c = input.charAt(position);
                    if (c == ' ' || c == '\n' || c == '\r' || c == '\t') {
                        position++;
                    } else {
                        return;
                    }
                }
            }

            private boolean take(char expected) {
                if (position < input.length() && input.charAt(position) == expected) {
                    position++;
                    return true;
                }
                return false;
            }

            private void expect(char expected) throws IOException {
                if (!take(expected)) {
                    throw syntax("expected '" + expected + "'");
                }
            }

            private IOException syntax(String message) {
                return invalid("JSON syntax at offset " + position + ": " + message);
            }
        }
    }
}
