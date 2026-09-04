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
import java.util.Objects;
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
            Set.of("int8-per-channel", "int8-per-tensor", "q4-k-per-op-int8-boundaries");

    private final String scheme;
    private final String provider;
    private final List<String> targetSocs;
    private final String weightGranularity;
    private final String activationDtype;
    private final BigDecimal weightScale;
    private final BigDecimal activationScale;
    private final BigDecimal outputScale;
    private final String calibrationMethod;
    private final int calibrationSampleCount;
    private final String calibrationDatasetSha256;
    private final Map<String, OperatorCalibration> operatorCalibrations;
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
            String calibrationMethod,
            int calibrationSampleCount,
            String calibrationDatasetSha256,
            Map<String, OperatorCalibration> operatorCalibrations,
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
        this.calibrationMethod = calibrationMethod;
        this.calibrationSampleCount = calibrationSampleCount;
        this.calibrationDatasetSha256 = calibrationDatasetSha256;
        this.operatorCalibrations = Collections.unmodifiableMap(
                new LinkedHashMap<>(operatorCalibrations));
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
            throw invalid("scheme must be int8-per-channel, int8-per-tensor, "
                    + "or q4-k-per-op-int8-boundaries");
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
                        && !"q4-k-per-op-int8-boundaries".equals(scheme)
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
        String calibrationMethod = null;
        int calibrationSampleCount = 0;
        String calibrationDatasetSha256 = null;
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
            calibrationMethod = method;
            calibrationSampleCount = samples;
            calibrationDatasetSha256 = datasetSha;
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

        Map<String, OperatorCalibration> operatorCalibrations =
                parseOperatorCalibrations(root.get("operatorCalibrations"));
        if (!operatorCalibrations.isEmpty() && sourceSha == null) {
            throw invalid("operatorCalibrations require sourceModelSha256");
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
                calibrationMethod,
                calibrationSampleCount,
                calibrationDatasetSha256,
                operatorCalibrations,
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
        writeContractAtomically(output, json);
        return contract;
    }

    /**
     * Atomically materialize compiler-owned Tensor G3 Q4_K calibration for the
     * exact canonical SDZ. Applications never construct or interpret this file.
     */
    public static SdxQuantizationContract writeTensorG3Q4Profile(
            Path output,
            SdxSourceIdentity sourceIdentity,
            SdxTensorG3Q4Calibration.Result calibration) throws IOException {
        Objects.requireNonNull(output, "output");
        Objects.requireNonNull(sourceIdentity, "sourceIdentity");
        Objects.requireNonNull(calibration, "calibration");
        if (calibration.sampleCount() < SdxTensorG3Q4Calibration.REQUIRED_SAMPLE_COUNT) {
            throw invalid("Tensor G3 Q4 calibration requires at least "
                    + SdxTensorG3Q4Calibration.REQUIRED_SAMPLE_COUNT + " samples");
        }
        if (!SHA256.matcher(calibration.datasetSha256()).matches()) {
            throw invalid("Tensor G3 Q4 calibration dataset digest must be lowercase SHA-256");
        }
        if (!calibration.hasQ4Operations()) {
            throw invalid("Tensor G3 Q4 calibration contains no Q4 operations");
        }

        StringBuilder json = new StringBuilder(1024);
        json.append('{')
                .append("\"formatVersion\":1")
                .append(",\"scheme\":\"q4-k-per-op-int8-boundaries\"")
                .append(",\"provider\":\"sdx-graph\"")
                .append(",\"targetSocs\":[\"Tensor_G3\"]")
                .append(",\"deviceOnly\":true")
                .append(",\"allowFloatFallback\":false")
                .append(",\"requireVendorAot\":true")
                .append(",\"sourceModelSha256\":\"")
                .append(sourceIdentity.sha256()).append('"')
                .append(",\"weights\":{\"dtype\":\"INT8\"")
                .append(",\"scaleDtype\":\"FLOAT32\"")
                .append(",\"granularity\":\"per-tensor\"")
                .append(",\"symmetric\":true,\"zeroPoint\":0}")
                .append(",\"activations\":{\"dtype\":\"INT8\",\"calibration\":{")
                .append("\"method\":\"minmax\",\"sampleCount\":")
                .append(calibration.sampleCount())
                .append(",\"datasetSha256\":\"")
                .append(calibration.datasetSha256()).append("\"}}")
                .append(",\"operatorCalibrations\":{");
        boolean first = true;
        for (Map.Entry<String, SdxTensorG3Q4Calibration.OperatorCalibration> entry
                : new java.util.TreeMap<>(calibration.operatorCalibrations()).entrySet()) {
            if (!first) {
                json.append(',');
            }
            first = false;
            SdxTensorG3Q4Calibration.OperatorCalibration value = entry.getValue();
            json.append('"').append(json(entry.getKey())).append("\":{")
                    .append("\"opType\":\"ggml_qmatmul\"")
                    .append(",\"activations\":{\"scaleDtype\":\"FLOAT32\"")
                    .append(",\"granularity\":\"per-tensor\",\"scale\":")
                    .append(Float.toString(value.activationScale()))
                    .append(",\"zeroPoint\":0}")
                    .append(",\"outputs\":{\"scaleDtype\":\"FLOAT32\"")
                    .append(",\"granularity\":\"per-tensor\",\"scale\":")
                    .append(Float.toString(value.outputScale()))
                    .append(",\"zeroPoint\":0,\"interiorQuantizationMax\":126}}");
        }
        json.append("},\"excludedOps\":[]}");

        String encoded = json.toString();
        SdxQuantizationContract contract = parse(encoded);
        contract.validateForCompilation(
                sourceIdentity,
                SdxTargetProfile.ANDROID_ARM64_NNAPI_ACCELERATOR,
                SdxTensorG3NnapiCompiler.TARGET_SOC);
        writeContractAtomically(output, encoded);
        return contract;
    }

    private static void writeContractAtomically(Path output, String json) throws IOException {
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
                boolean q4OperatorCalibration = !operatorCalibrations.isEmpty();
                if (q4OperatorCalibration) {
                    if (!"q4-k-per-op-int8-boundaries".equals(scheme)
                            || !"per-tensor".equals(weightGranularity)
                            || !"INT8".equals(activationDtype)
                            || sourceModelSha256 == null
                            || calibrationSampleCount < 32
                            || calibrationDatasetSha256 == null) {
                        throw invalid(
                                "Tensor G3 Q4_K requires source-bound per-op INT8 boundary "
                                        + "calibration from at least 32 samples");
                    }
                    break;
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

    public Map<String, OperatorCalibration> operatorCalibrations() {
        return operatorCalibrations;
    }

    public boolean isTensorG3Q4PerOperator() {
        return "q4-k-per-op-int8-boundaries".equals(scheme)
                && !operatorCalibrations.isEmpty();
    }

    public OperatorCalibration operatorCalibration(String opName) {
        return operatorCalibrations.get(opName);
    }

    public int calibrationSampleCount() {
        return calibrationSampleCount;
    }

    public String calibrationDatasetSha256() {
        return calibrationDatasetSha256;
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
        float convertedScale = parsedScale.floatValue();
        if (parsedScale.compareTo(BigDecimal.ZERO) <= 0
                || !Float.isFinite(convertedScale) || convertedScale <= 0.0f) {
            throw invalid(field + ".scale must be finite and positive");
        }
        requireInteger(zeroPoint, field + ".zeroPoint", 0);
        return parsedScale;
    }

    private static Map<String, OperatorCalibration> parseOperatorCalibrations(
            Object value) throws IOException {
        if (value == null) {
            return Collections.emptyMap();
        }
        Map<String, Object> entries = object(value, "operatorCalibrations");
        if (entries.isEmpty()) {
            throw invalid("operatorCalibrations must not be empty when present");
        }
        Map<String, OperatorCalibration> result = new java.util.TreeMap<>();
        for (Map.Entry<String, Object> entry : entries.entrySet()) {
            String opName = entry.getKey();
            if (opName == null || opName.trim().isEmpty()
                    || opName.indexOf('\n') >= 0 || opName.indexOf('\r') >= 0) {
                throw invalid("operatorCalibrations keys must be non-empty single-line op names");
            }
            Map<String, Object> calibration = object(
                    entry.getValue(), "operatorCalibrations." + opName);
            String opType = string(
                    calibration.get("opType"),
                    "operatorCalibrations." + opName + ".opType");
            if (!"ggml_qmatmul".equals(opType)) {
                throw invalid("operatorCalibrations." + opName
                        + ".opType must be ggml_qmatmul");
            }
            BigDecimal activation = validatePerTensorScaleMetadata(
                    object(calibration.get("activations"),
                            "operatorCalibrations." + opName + ".activations"),
                    "operatorCalibrations." + opName + ".activations", true);
            BigDecimal output = validatePerTensorScaleMetadata(
                    object(calibration.get("outputs"),
                            "operatorCalibrations." + opName + ".outputs"),
                    "operatorCalibrations." + opName + ".outputs", true);
            validateQ4EnvelopeScale(
                    activation, 127,
                    "operatorCalibrations." + opName + ".activations.scale");
            validateQ4EnvelopeScale(
                    output, 126,
                    "operatorCalibrations." + opName + ".outputs.scale");
            Map<String, Object> outputMetadata = object(
                    calibration.get("outputs"),
                    "operatorCalibrations." + opName + ".outputs");
            requireInteger(
                    outputMetadata.get("interiorQuantizationMax"),
                    "operatorCalibrations." + opName
                            + ".outputs.interiorQuantizationMax",
                    126);
            result.put(opName, new OperatorCalibration(opType, activation, output));
        }
        return result;
    }

    private static void validateQ4EnvelopeScale(
            BigDecimal scale, int quantizationMaximum, String field) throws IOException {
        float envelope = scale.floatValue() * quantizationMaximum;
        if (!Float.isFinite(envelope) || envelope <= 0.0f) {
            throw invalid(field + " overflows its INT8 calibration envelope");
        }
    }

    public String summaryJson() {
        StringBuilder out = new StringBuilder(512);
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
        out.append("],\"weightDtype\":\"INT8\"");
        if (sourceModelSha256 != null) {
            out.append(",\"sourceModelSha256\":\"")
                    .append(sourceModelSha256).append('"');
        }
        if (calibrationMethod != null) {
            out.append(",\"calibration\":{\"method\":\"")
                    .append(json(calibrationMethod))
                    .append("\",\"sampleCount\":")
                    .append(calibrationSampleCount)
                    .append(",\"datasetSha256\":\"")
                    .append(calibrationDatasetSha256).append("\"}");
        }
        if (weightScale != null) {
            out.append(",\"weightScale\":").append(weightScale.toPlainString());
        }
        if (activationScale != null) {
            out.append(",\"activationScale\":").append(activationScale.toPlainString());
        }
        if (outputScale != null) {
            out.append(",\"outputScale\":").append(outputScale.toPlainString());
        }
        if (!operatorCalibrations.isEmpty()) {
            out.append(",\"operatorCalibrations\":{");
            boolean first = true;
            for (Map.Entry<String, OperatorCalibration> entry
                    : operatorCalibrations.entrySet()) {
                if (!first) out.append(',');
                first = false;
                out.append('"').append(json(entry.getKey())).append("\":")
                        .append(entry.getValue().summaryJson());
            }
            out.append('}');
        }
        return out.append('}').toString();
    }

    public static final class OperatorCalibration {
        private final String opType;
        private final BigDecimal activationScale;
        private final BigDecimal outputScale;

        private OperatorCalibration(
                String opType, BigDecimal activationScale, BigDecimal outputScale) {
            this.opType = opType;
            this.activationScale = activationScale;
            this.outputScale = outputScale;
        }

        public String opType() {
            return opType;
        }

        public float activationScale() {
            return activationScale.floatValue();
        }

        public float outputScale() {
            return outputScale.floatValue();
        }

        public String[] nnapiQ4SArguments(
                int sampleCount, String datasetSha256) {
            return new String[] {
                    "sdx.nnapi.q4.calibration.v1",
                    Integer.toString(sampleCount),
                    datasetSha256,
                    Float.toString(activationScale()),
                    Float.toString(outputScale())
            };
        }

        private String summaryJson() {
            return "{\"opType\":\"" + json(opType)
                    + "\",\"activationScale\":" + activationScale.toPlainString()
                    + ",\"outputScale\":" + outputScale.toPlainString()
                    + ",\"interiorQuantizationMax\":126}";
        }
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
        StringBuilder escaped = new StringBuilder(value.length());
        for (int index = 0; index < value.length(); index++) {
            char character = value.charAt(index);
            switch (character) {
                case '\\': escaped.append("\\\\"); break;
                case '"': escaped.append("\\\""); break;
                case '\b': escaped.append("\\b"); break;
                case '\f': escaped.append("\\f"); break;
                case '\n': escaped.append("\\n"); break;
                case '\r': escaped.append("\\r"); break;
                case '\t': escaped.append("\\t"); break;
                default:
                    if (character < 0x20) {
                        escaped.append(String.format(
                                java.util.Locale.ROOT, "\\u%04x", (int) character));
                    } else {
                        escaped.append(character);
                    }
            }
        }
        return escaped.toString();
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
