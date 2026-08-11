/*
 * Copyright (c) Eclipse Deeplearning4j
 * SPDX-License-Identifier: Apache-2.0
 */
package org.nd4j.dsp.model;

import java.io.IOException;
import java.math.BigDecimal;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.UUID;
import java.util.regex.Pattern;

/**
 * Validated, fail-closed quantization input for an SDX target compilation.
 *
 * <p>The parser intentionally has no third-party dependency so the same compile
 * API can be embedded in the Android runtime. A contract is validated before a
 * cache key is accepted or a target compiler is invoked.</p>
 */
public final class SdxQuantizationContract {
    private static final int MAX_JSON_BYTES = 1024 * 1024;
    private static final Pattern SHA256 = Pattern.compile("[0-9a-f]{64}");
    private static final Set<String> PROVIDERS =
            Set.of("sdx-graph", "litert-lm");
    private static final Set<String> ACTIVATION_DTYPES =
            Set.of("FLOAT16", "FLOAT32", "INT8");
    private static final Set<String> CALIBRATION_METHODS =
            Set.of("minmax", "percentile", "entropy");
    private static final Set<String> SCHEMES =
            Set.of("int8-per-channel", "int8-per-tensor");

    private final String scheme;
    private final String provider;
    private final List<String> targetSocs;
    private final String weightGranularity;
    private final String activationDtype;
    private final BigDecimal weightScale;
    private final BigDecimal activationScale;
    private final BigDecimal outputScale;
    private final String sourceModelSha256;
    private final String aotArtifactSha256;

    private SdxQuantizationContract(
            String scheme,
            String provider,
            List<String> targetSocs,
            String weightGranularity,
            String activationDtype,
            BigDecimal weightScale,
            BigDecimal activationScale,
            BigDecimal outputScale,
            String sourceModelSha256,
            String aotArtifactSha256) {
        this.scheme = scheme;
        this.provider = provider;
        this.targetSocs = Collections.unmodifiableList(new ArrayList<>(targetSocs));
        this.weightGranularity = weightGranularity;
        this.activationDtype = activationDtype;
        this.weightScale = weightScale;
        this.activationScale = activationScale;
        this.outputScale = outputScale;
        this.sourceModelSha256 = sourceModelSha256;
        this.aotArtifactSha256 = aotArtifactSha256;
    }

    public static SdxQuantizationContract load(Path input) throws IOException {
        Path path = input.toAbsolutePath().normalize();
        if (!Files.isRegularFile(path)) {
            throw new IOException("SDX quantization contract is missing: " + path);
        }
        long size = Files.size(path);
        if (size <= 0L || size > MAX_JSON_BYTES) {
            throw new IOException(
                    "SDX quantization contract must contain 1.." + MAX_JSON_BYTES
                            + " bytes: " + path);
        }
        return parse(Files.readString(path, StandardCharsets.UTF_8));
    }

    public static SdxQuantizationContract parse(String json) throws IOException {
        Map<String, Object> root = parseJsonObject(json);

        requireInteger(root.get("formatVersion"), "formatVersion", 1);
        String scheme = string(root.get("scheme"), "scheme");
        if (!SCHEMES.contains(scheme)) {
            throw invalid("scheme must be int8-per-channel or int8-per-tensor");
        }

        String provider = string(root.get("provider"), "provider");
        if (!PROVIDERS.contains(provider)) {
            throw invalid("provider must be sdx-graph or litert-lm");
        }
        requireBoolean(root.get("deviceOnly"), "deviceOnly", true);
        requireBoolean(root.get("allowFloatFallback"), "allowFloatFallback", false);
        requireBoolean(root.get("requireVendorAot"), "requireVendorAot", true);

        List<String> targetSocs = nonEmptyStrings(root.get("targetSocs"), "targetSocs");
        String sourceSha = optionalSha256(root.get("sourceModelSha256"), "sourceModelSha256");
        String artifactSha =
                optionalSha256(root.get("aotArtifactSha256"), "aotArtifactSha256");

        Map<String, Object> weights = object(root.get("weights"), "weights");
        requireString(weights.get("dtype"), "weights.dtype", "INT8");
        requireString(weights.get("scaleDtype"), "weights.scaleDtype", "FLOAT32");
        String weightGranularity = string(weights.get("granularity"), "weights.granularity");
        String expectedGranularity = "int8-per-channel".equals(scheme)
                ? "per-channel"
                : "per-tensor";
        if (!expectedGranularity.equals(weightGranularity)) {
            throw invalid("weights.granularity must match scheme " + scheme);
        }
        Object channelAxis = weights.get("channelAxis");
        if ("per-channel".equals(weightGranularity)) {
            integer(channelAxis, "weights.channelAxis");
        } else if (channelAxis != null) {
            throw invalid("weights.channelAxis is not valid for per-tensor weights");
        }
        requireBoolean(weights.get("symmetric"), "weights.symmetric", true);
        requireInteger(weights.get("zeroPoint"), "weights.zeroPoint", 0);
        BigDecimal weightScale = "per-tensor".equals(weightGranularity)
                ? validatePerTensorScaleMetadata(weights, "weights", false)
                : null;

        Map<String, Object> activations =
                object(root.get("activations"), "activations");
        String activationDtype =
                string(activations.get("dtype"), "activations.dtype");
        if (!ACTIVATION_DTYPES.contains(activationDtype)) {
            throw invalid("activations.dtype must be FLOAT16, FLOAT32, or INT8");
        }

        Object calibrationValue = activations.get("calibration");
        BigDecimal activationScale = null;
        if ("INT8".equals(activationDtype)) {
            Map<String, Object> calibration =
                    object(calibrationValue, "activations.calibration");
            String method =
                    string(calibration.get("method"), "activations.calibration.method");
            if (!CALIBRATION_METHODS.contains(method)) {
                throw invalid(
                        "activations.calibration.method must be minmax, percentile, or entropy");
            }
            int samples = integer(
                    calibration.get("sampleCount"),
                    "activations.calibration.sampleCount");
            if (samples < 32) {
                throw invalid(
                        "activations.calibration.sampleCount must be at least 32");
            }
            String datasetSha = optionalSha256(
                    calibration.get("datasetSha256"),
                    "activations.calibration.datasetSha256");
            if (datasetSha == null) {
                throw invalid(
                        "INT8 activation calibration requires datasetSha256");
            }
            if ("percentile".equals(method)) {
                BigDecimal percentile = number(
                        calibration.get("percentile"),
                        "activations.calibration.percentile");
                if (percentile.compareTo(BigDecimal.valueOf(90L)) < 0
                        || percentile.compareTo(BigDecimal.valueOf(100L)) >= 0) {
                    throw invalid(
                            "percentile calibration requires percentile in [90, 100)");
                }
            }
            activationScale = validatePerTensorScaleMetadata(
                    activations, "activations", false);
        } else if (calibrationValue != null) {
            throw invalid("calibration is only valid for INT8 activations");
        }

        BigDecimal outputScale = null;
        Object outputsValue = root.get("outputs");
        if (outputsValue != null) {
            Map<String, Object> outputs = object(outputsValue, "outputs");
            requireString(outputs.get("dtype"), "outputs.dtype", "INT8");
            outputScale = validatePerTensorScaleMetadata(
                    outputs, "outputs", true);
        }

        Object excluded = root.get("excludedOps");
        if (excluded != null) {
            strings(excluded, "excludedOps", false);
        }

        return new SdxQuantizationContract(
                scheme,
                provider,
                targetSocs,
                weightGranularity,
                activationDtype,
                weightScale,
                activationScale,
                outputScale,
                sourceSha,
                artifactSha);
    }

    /** Shared strict parser for package-local SDX compiler contracts. */
    static Map<String, Object> parseJsonObject(String json) throws IOException {
        return object(new JsonParser(json).parse(), "root");
    }

    /**
     * Materialize the canonical device-only weight-INT8/F16-activation research
     * contract for one target. This keeps profile generation beside strict parsing
     * and target validation instead of duplicating provider rules in staging tools.
     *
     * @return the parsed contract that was atomically written
     */
    public static SdxQuantizationContract writeWeightInt8Profile(
            Path output,
            SdxTargetProfile target,
            String targetSoc) throws IOException {
        if (output == null) {
            throw new NullPointerException("output");
        }
        if (target == null) {
            throw new NullPointerException("target");
        }
        if (targetSoc == null || targetSoc.trim().isEmpty()
                || targetSoc.indexOf('\n') >= 0 || targetSoc.indexOf('\r') >= 0) {
            throw invalid("targetSoc must be a non-empty single-line value");
        }

        if (target == SdxTargetProfile.ANDROID_ARM64_NNAPI_ACCELERATOR) {
            throw invalid(
                    "NNAPI profile generation requires explicit calibrated activation/output "
                            + "per-tensor scales");
        }
        String provider = target == SdxTargetProfile.ANDROID_ARM64_GOOGLE_TENSOR_G5
                ? "litert-lm"
                : "sdx-graph";
        String json = "{"
                + "\"formatVersion\":1,"
                + "\"scheme\":\"int8-per-channel\","
                + "\"provider\":\"" + json(provider) + "\","
                + "\"targetSocs\":[\"" + json(targetSoc.trim()) + "\"],"
                + "\"deviceOnly\":true,"
                + "\"allowFloatFallback\":false,"
                + "\"requireVendorAot\":true,"
                + "\"weights\":{"
                + "\"dtype\":\"INT8\","
                + "\"scaleDtype\":\"FLOAT32\","
                + "\"granularity\":\"per-channel\","
                + "\"channelAxis\":0,"
                + "\"symmetric\":true,"
                + "\"zeroPoint\":0"
                + "},"
                + "\"activations\":{\"dtype\":\"FLOAT16\"},"
                + "\"excludedOps\":[]"
                + "}";
        SdxQuantizationContract contract = parse(json);
        contract.validateTarget(target);

        Path destination = output.toAbsolutePath().normalize();
        if (Files.isSymbolicLink(destination)) {
            throw new IOException(
                    "Refusing to replace symbolic-link quantization contract: " + destination);
        }
        Path parent = destination.getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }
        Path temporary = destination.resolveSibling(
                "." + destination.getFileName() + "." + UUID.randomUUID() + ".pending");
        try {
            Files.writeString(temporary, json + "\n", StandardCharsets.UTF_8);
            try {
                Files.move(
                        temporary,
                        destination,
                        StandardCopyOption.ATOMIC_MOVE,
                        StandardCopyOption.REPLACE_EXISTING);
            } catch (AtomicMoveNotSupportedException unsupported) {
                Files.move(temporary, destination, StandardCopyOption.REPLACE_EXISTING);
            }
        } finally {
            Files.deleteIfExists(temporary);
        }
        return contract;
    }

    public void validateForCompilation(
            SdxSourceIdentity sourceIdentity,
            SdxTargetProfile target) throws IOException {
        validateForCompilation(sourceIdentity, target, null);
    }

    public void validateForCompilation(
            SdxSourceIdentity sourceIdentity,
            SdxTargetProfile target,
            String requestedTargetSoc) throws IOException {
        if (sourceModelSha256 != null
                && !sourceModelSha256.equals(sourceIdentity.sha256())) {
            throw invalid(
                    "sourceModelSha256 does not match the canonical SDZ identity");
        }

        validateTarget(target);
        if (requestedTargetSoc != null) {
            String exactSoc = requestedTargetSoc.trim();
            if (exactSoc.isEmpty()
                    || exactSoc.indexOf('\n') >= 0
                    || exactSoc.indexOf('\r') >= 0) {
                throw invalid("requested target SoC must be a non-empty single-line value");
            }
            if (!targetSocs.contains(exactSoc)) {
                throw invalid(
                        "targetSocs must include the requested target SoC " + exactSoc
                                + " for " + target.id());
            }
        }
    }

    private void validateTarget(SdxTargetProfile target) throws IOException {
        switch (target) {
            case ANDROID_ARM64_GOOGLE_TENSOR_G5:
                requireProviderAndSoc("litert-lm", "Tensor_G5", target);
                break;
            case ANDROID_ARM64_NNAPI_ACCELERATOR:
                requireProvider("sdx-graph", target);
                if (!targetSocs.contains("Tensor_G3")
                        && !targetSocs.contains("Android_NNAPI")) {
                    throw invalid(
                            "targetSocs must include Tensor_G3 or Android_NNAPI for "
                                    + target.id());
                }
                if (!"int8-per-tensor".equals(scheme)
                        || !"per-tensor".equals(weightGranularity)) {
                    throw invalid(
                            "NNAPI quantized_matmul requires signed symmetric per-tensor "
                                    + "INT8 weights");
                }
                if (!"INT8".equals(activationDtype)
                        || weightScale == null
                        || activationScale == null
                        || outputScale == null) {
                    throw invalid(
                            "NNAPI quantized_matmul requires calibrated weight and activation/output "
                                    + "per-tensor scale metadata");
                }
                break;
            case ANDROID_ARM64_HEXAGON_HTP:
                requireProvider("sdx-graph", target);
                if (targetSocs.stream().noneMatch(soc -> soc.matches("SM[0-9]+"))) {
                    throw invalid(
                            "targetSocs must include a Qualcomm SM SoC for " + target.id());
                }
                break;
            case ANDROID_ARM64_VULKAN:
                requireProviderAndSoc(
                        "sdx-graph", "Android_Vulkan_1_1", target);
                break;
            default:
                requireProvider("sdx-graph", target);
                break;
        }
    }

    public void validateArtifact(Path artifact) throws IOException {
        if (aotArtifactSha256 == null) {
            return;
        }
        Path path = artifact.toAbsolutePath().normalize();
        String actual = Files.isRegularFile(path)
                ? SdxSourceIdentity.sha256(path)
                : SdxModelCompiler.treeDigest(path);
        if (!aotArtifactSha256.equals(actual)) {
            throw invalid(
                    "aotArtifactSha256 does not match target compiler output");
        }
    }

    public String provider() {
        return provider;
    }

    public List<String> targetSocs() {
        return targetSocs;
    }

    public String activationDtype() {
        return activationDtype;
    }

    public float weightScale() {
        return requiredScale(weightScale, "weights.scale");
    }

    public float activationScale() {
        return requiredScale(activationScale, "activations.scale");
    }

    public float outputScale() {
        return requiredScale(outputScale, "outputs.scale");
    }

    private static float requiredScale(BigDecimal value, String field) {
        if (value == null) {
            throw new IllegalStateException(field + " is not present in this quantization contract");
        }
        return value.floatValue();
    }

    private static BigDecimal validatePerTensorScaleMetadata(
            Map<String, Object> values,
            String field,
            boolean required) throws IOException {
        Object scale = values.get("scale");
        Object scaleDtype = values.get("scaleDtype");
        Object granularity = values.get("granularity");
        Object zeroPoint = values.get("zeroPoint");
        boolean present = scale != null || scaleDtype != null || granularity != null
                || zeroPoint != null;
        if (!present) {
            if (required) {
                throw invalid(field + " requires calibrated per-tensor scale metadata");
            }
            return null;
        }
        requireString(scaleDtype, field + ".scaleDtype", "FLOAT32");
        requireString(granularity, field + ".granularity", "per-tensor");
        BigDecimal parsedScale = number(scale, field + ".scale");
        if (parsedScale.compareTo(BigDecimal.ZERO) <= 0
                || !Float.isFinite(parsedScale.floatValue())) {
            throw invalid(field + ".scale must be finite and positive");
        }
        requireInteger(zeroPoint, field + ".zeroPoint", 0);
        return parsedScale;
    }

    public String summaryJson() {
        StringBuilder out = new StringBuilder(192);
        out.append("{\"activationDtype\":\"")
                .append(json(activationDtype))
                .append("\",\"allowFloatFallback\":false,\"deviceOnly\":true")
                .append(",\"formatVersion\":1,\"provider\":\"")
                .append(json(provider))
                .append("\",\"requireVendorAot\":true")
                .append(",\"scheme\":\"")
                .append(json(scheme))
                .append("\",\"targetSocs\":[");
        for (int i = 0; i < targetSocs.size(); i++) {
            if (i > 0) {
                out.append(',');
            }
            out.append('\"').append(json(targetSocs.get(i))).append('\"');
        }
        return out.append("],\"weightDtype\":\"INT8\"}").toString();
    }

    private void requireProviderAndSoc(
            String expectedProvider,
            String expectedSoc,
            SdxTargetProfile target) throws IOException {
        requireProvider(expectedProvider, target);
        if (!targetSocs.contains(expectedSoc)) {
            throw invalid(
                    "targetSocs must include " + expectedSoc + " for " + target.id());
        }
    }

    private void requireProvider(
            String expectedProvider,
            SdxTargetProfile target) throws IOException {
        if (!expectedProvider.equals(provider)) {
            throw invalid(
                    "provider must be " + expectedProvider + " for " + target.id());
        }
    }

    private static Map<String, Object> object(Object value, String field)
            throws IOException {
        if (!(value instanceof Map)) {
            throw invalid(field + " must be an object");
        }
        @SuppressWarnings("unchecked")
        Map<String, Object> result = (Map<String, Object>) value;
        return result;
    }

    private static List<String> nonEmptyStrings(Object value, String field)
            throws IOException {
        return strings(value, field, true);
    }

    private static List<String> strings(
            Object value, String field, boolean requireNonEmpty) throws IOException {
        if (!(value instanceof List)) {
            throw invalid(field + " must be a string array");
        }
        List<?> values = (List<?>) value;
        if (requireNonEmpty && values.isEmpty()) {
            throw invalid(field + " must be a non-empty string array");
        }
        List<String> result = new ArrayList<>(values.size());
        for (Object item : values) {
            if (!(item instanceof String) || ((String) item).trim().isEmpty()) {
                throw invalid(field + " must contain only non-empty strings");
            }
            result.add((String) item);
        }
        return result;
    }

    private static String string(Object value, String field) throws IOException {
        if (!(value instanceof String) || ((String) value).trim().isEmpty()) {
            throw invalid(field + " must be a non-empty string");
        }
        return (String) value;
    }

    private static void requireString(
            Object value, String field, String expected) throws IOException {
        if (!expected.equals(value)) {
            throw invalid(field + " must be " + expected);
        }
    }

    private static void requireBoolean(
            Object value, String field, boolean expected) throws IOException {
        if (!(value instanceof Boolean) || ((Boolean) value) != expected) {
            throw invalid(field + " must be " + expected);
        }
    }

    private static int integer(Object value, String field) throws IOException {
        try {
            return number(value, field).intValueExact();
        } catch (ArithmeticException invalid) {
            throw invalid(field + " must be an integer");
        }
    }

    private static void requireInteger(
            Object value, String field, int expected) throws IOException {
        if (integer(value, field) != expected) {
            throw invalid(field + " must be " + expected);
        }
    }

    private static BigDecimal number(Object value, String field)
            throws IOException {
        if (!(value instanceof BigDecimal)) {
            throw invalid(field + " must be a number");
        }
        return (BigDecimal) value;
    }

    private static String optionalSha256(Object value, String field)
            throws IOException {
        if (value == null) {
            return null;
        }
        if (!(value instanceof String) || !SHA256.matcher((String) value).matches()) {
            throw invalid(field + " must be a lowercase SHA-256 digest");
        }
        return (String) value;
    }

    private static IOException invalid(String message) {
        return new IOException("Invalid SDX quantization contract: " + message);
    }

    private static String json(String value) {
        return value.replace("\\", "\\\\").replace("\"", "\\\"");
    }

    /** Strict JSON parser for small compiler metadata documents. */
    private static final class JsonParser {
        private final String input;
        private int position;
        private int depth;

        private JsonParser(String input) throws IOException {
            if (input == null || input.isEmpty()) {
                throw invalid("JSON document is empty");
            }
            if (input.getBytes(StandardCharsets.UTF_8).length > MAX_JSON_BYTES) {
                throw invalid("JSON document exceeds " + MAX_JSON_BYTES + " bytes");
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
                    case '\"':
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
                        if (c == '-' || (c >= '0' && c <= '9')) {
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
                if (position >= input.length() || input.charAt(position) != '\"') {
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
            expect('\"');
            StringBuilder result = new StringBuilder();
            while (position < input.length()) {
                char c = input.charAt(position++);
                if (c == '\"') {
                    return result.toString();
                }
                if (c == '\\') {
                    if (position >= input.length()) {
                        throw syntax("unterminated string escape");
                    }
                    char escaped = input.charAt(position++);
                    switch (escaped) {
                        case '\"':
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
            } catch (NumberFormatException malformed) {
                throw syntax("invalid number");
            }
        }

        private void digits() throws IOException {
            int start = position;
            while (position < input.length()
                    && Character.isDigit(input.charAt(position))) {
                position++;
            }
            if (start == position) {
                throw syntax("expected decimal digit");
            }
        }

        private void literal(String expected) throws IOException {
            if (!input.startsWith(expected, position)) {
                throw syntax("invalid literal");
            }
            position += expected.length();
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
