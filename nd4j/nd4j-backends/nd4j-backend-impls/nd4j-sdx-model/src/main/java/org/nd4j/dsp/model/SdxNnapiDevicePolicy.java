/*
 * Copyright (c) Eclipse Deeplearning4j
 * SPDX-License-Identifier: Apache-2.0
 */
package org.nd4j.dsp.model;

import java.io.IOException;
import java.math.BigDecimal;
import java.nio.charset.StandardCharsets;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.LinkOption;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.UUID;
import java.util.regex.Pattern;

/**
 * Strict SDX policy contract for Android NNAPI device compilation.
 *
 * <p>The policy is not a portable host-compiled model. It binds one canonical
 * SDZ (and, when present, one derived quantized SDZ) to an exact target SoC and
 * records the accelerator-only runtime requirements that Android enforces while
 * compiling through the NNAPI driver.</p>
 */
public final class SdxNnapiDevicePolicy {
    public static final int FORMAT_VERSION = 1;
    public static final String POLICY_ABI =
            "sdx-nnapi-device-compilation-policy-v1";
    public static final String DEVICE_TYPE = "DEVICE_ACCELERATOR";
    public static final String COMPILATION_LOCATION = "android-nnapi-device";

    private static final int MAX_POLICY_BYTES = 1024 * 1024;
    private static final Pattern SOC = Pattern.compile("[A-Za-z0-9._-]{1,128}");
    private static final Pattern SHA256 = Pattern.compile("[0-9a-f]{64}");
    private static final Set<String> BASE_FIELDS = Set.of(
            "formatVersion",
            "policyAbi",
            "target",
            "targetSoc",
            "sourceSha256",
            "deviceType",
            "requireWholeGraph",
            "allowCpu",
            "allowGpu",
            "allowFallback",
            "compilationLocation",
            "persistentCache");

    private final String target;
    private final String targetSoc;
    private final String sourceSha256;
    private final String derivedModelSha256;

    private SdxNnapiDevicePolicy(
            String target,
            String targetSoc,
            String sourceSha256,
            String derivedModelSha256) {
        this.target = target;
        this.targetSoc = targetSoc;
        this.sourceSha256 = sourceSha256;
        this.derivedModelSha256 = derivedModelSha256;
    }

    public static SdxNnapiDevicePolicy create(
            SdxTargetProfile target,
            String targetSoc,
            SdxSourceIdentity sourceIdentity) throws IOException {
        return create(target, targetSoc, sourceIdentity, null);
    }

    public static SdxNnapiDevicePolicy create(
            SdxTargetProfile target,
            String targetSoc,
            SdxSourceIdentity sourceIdentity,
            SdxSourceIdentity derivedModelIdentity) throws IOException {
        requireNnapiTarget(target);
        Objects.requireNonNull(sourceIdentity, "sourceIdentity");
        String soc = requireSoc(targetSoc);
        String sourceSha = requireSha256(
                sourceIdentity.sha256(), "source model SHA-256");
        String derivedSha = derivedModelIdentity == null
                ? null
                : requireSha256(
                        derivedModelIdentity.sha256(), "derived model SHA-256");
        return new SdxNnapiDevicePolicy(target.id(), soc, sourceSha, derivedSha);
    }

    public static SdxNnapiDevicePolicy load(Path input) throws IOException {
        Path path = Objects.requireNonNull(input, "input").toAbsolutePath().normalize();
        if (!Files.isRegularFile(path, LinkOption.NOFOLLOW_LINKS)
                || Files.isSymbolicLink(path)) {
            throw invalid("NNAPI device policy is missing: " + path);
        }
        long bytes = Files.size(path);
        if (bytes <= 0L || bytes > MAX_POLICY_BYTES) {
            throw invalid(
                    "NNAPI device policy must contain 1.." + MAX_POLICY_BYTES
                            + " bytes: " + path);
        }

        Map<String, Object> root = SdxQuantizationContract.parseJsonObject(
                Files.readString(path, StandardCharsets.UTF_8));
        Set<String> expected = new LinkedHashSet<>(BASE_FIELDS);
        if (root.containsKey("derivedModelSha256")) {
            expected.add("derivedModelSha256");
        }
        if (!root.keySet().equals(expected)) {
            throw invalid(
                    "NNAPI device policy fields must be exactly " + expected
                            + ", found " + root.keySet());
        }
        requireInteger(root.get("formatVersion"), "formatVersion", FORMAT_VERSION);
        requireString(root.get("policyAbi"), "policyAbi", POLICY_ABI);
        String target = requireString(root.get("target"), "target");
        String soc = requireSoc(requireString(root.get("targetSoc"), "targetSoc"));
        String sourceSha = requireSha256(
                requireString(root.get("sourceSha256"), "sourceSha256"),
                "sourceSha256");
        String derivedSha = root.containsKey("derivedModelSha256")
                ? requireSha256(
                        requireString(
                                root.get("derivedModelSha256"),
                                "derivedModelSha256"),
                        "derivedModelSha256")
                : null;
        requireString(root.get("deviceType"), "deviceType", DEVICE_TYPE);
        requireBoolean(root.get("requireWholeGraph"), "requireWholeGraph", true);
        requireBoolean(root.get("allowCpu"), "allowCpu", false);
        requireBoolean(root.get("allowGpu"), "allowGpu", false);
        requireBoolean(root.get("allowFallback"), "allowFallback", false);
        requireString(
                root.get("compilationLocation"),
                "compilationLocation",
                COMPILATION_LOCATION);
        requireBoolean(root.get("persistentCache"), "persistentCache", true);
        return new SdxNnapiDevicePolicy(target, soc, sourceSha, derivedSha);
    }

    public void validateFor(
            SdxTargetProfile expectedTarget,
            String expectedTargetSoc,
            SdxSourceIdentity expectedSource,
            SdxSourceIdentity expectedDerivedModel) throws IOException {
        requireNnapiTarget(expectedTarget);
        Objects.requireNonNull(expectedSource, "expectedSource");
        if (!expectedTarget.id().equals(target)) {
            throw invalid("target does not match " + expectedTarget.id());
        }
        if (expectedTargetSoc != null
                && !requireSoc(expectedTargetSoc).equals(targetSoc)) {
            throw invalid("targetSoc does not match requested SoC " + expectedTargetSoc);
        }
        if (!expectedSource.sha256().equals(sourceSha256)) {
            throw invalid("sourceSha256 does not match the canonical SDZ");
        }
        if (expectedDerivedModel == null) {
            if (derivedModelSha256 != null) {
                throw invalid("unquantized policy unexpectedly binds a derived SDZ");
            }
        } else if (!expectedDerivedModel.sha256().equals(derivedModelSha256)) {
            throw invalid("derivedModelSha256 does not match the emitted SDZ");
        }
    }

    public void write(Path output) throws IOException {
        Path destination = Objects.requireNonNull(output, "output")
                .toAbsolutePath().normalize();
        if (Files.isSymbolicLink(destination)) {
            throw invalid("Refusing to replace symbolic-link NNAPI policy: " + destination);
        }
        Path parent = destination.getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }
        Path temporary = destination.resolveSibling(
                "." + destination.getFileName() + "." + UUID.randomUUID() + ".pending");
        try {
            Files.writeString(
                    temporary,
                    canonicalJson(),
                    StandardCharsets.UTF_8,
                    StandardOpenOption.CREATE_NEW,
                    StandardOpenOption.WRITE);
            try {
                Files.move(
                        temporary,
                        destination,
                        StandardCopyOption.ATOMIC_MOVE,
                        StandardCopyOption.REPLACE_EXISTING);
            } catch (AtomicMoveNotSupportedException unsupported) {
                Files.move(
                        temporary,
                        destination,
                        StandardCopyOption.REPLACE_EXISTING);
            }
        } finally {
            Files.deleteIfExists(temporary);
        }
    }

    public String target() {
        return target;
    }

    public String targetSoc() {
        return targetSoc;
    }

    public String sourceSha256() {
        return sourceSha256;
    }

    public String derivedModelSha256() {
        return derivedModelSha256;
    }

    public String policyAbi() {
        return POLICY_ABI;
    }

    public String deviceType() {
        return DEVICE_TYPE;
    }

    public boolean requireWholeGraph() {
        return true;
    }

    public boolean allowCpu() {
        return false;
    }

    public boolean allowGpu() {
        return false;
    }

    public boolean allowFallback() {
        return false;
    }

    public String compilationLocation() {
        return COMPILATION_LOCATION;
    }

    public boolean persistentCache() {
        return true;
    }

    private String canonicalJson() {
        StringBuilder out = new StringBuilder(640)
                .append("{\n")
                .append("  \"formatVersion\": ").append(FORMAT_VERSION).append(",\n")
                .append("  \"policyAbi\": \"").append(json(POLICY_ABI)).append("\",\n")
                .append("  \"target\": \"").append(json(target)).append("\",\n")
                .append("  \"targetSoc\": \"").append(json(targetSoc)).append("\",\n")
                .append("  \"sourceSha256\": \"").append(sourceSha256).append("\",\n");
        if (derivedModelSha256 != null) {
            out.append("  \"derivedModelSha256\": \"")
                    .append(derivedModelSha256).append("\",\n");
        }
        return out.append("  \"deviceType\": \"").append(DEVICE_TYPE).append("\",\n")
                .append("  \"requireWholeGraph\": true,\n")
                .append("  \"allowCpu\": false,\n")
                .append("  \"allowGpu\": false,\n")
                .append("  \"allowFallback\": false,\n")
                .append("  \"compilationLocation\": \"")
                .append(COMPILATION_LOCATION).append("\",\n")
                .append("  \"persistentCache\": true\n")
                .append("}\n")
                .toString();
    }

    private static void requireNnapiTarget(SdxTargetProfile target) throws IOException {
        if (target != SdxTargetProfile.ANDROID_ARM64_NNAPI_ACCELERATOR) {
            throw invalid("NNAPI policy cannot describe target " + target);
        }
    }

    private static String requireSoc(String value) throws IOException {
        if (value == null || !SOC.matcher(value.trim()).matches()) {
            throw invalid(
                    "targetSoc must contain only ASCII letters, digits, dot, underscore, or dash");
        }
        return value.trim();
    }

    private static String requireSha256(String value, String label) throws IOException {
        if (value == null || !SHA256.matcher(value).matches()) {
            throw invalid(label + " must be a lowercase SHA-256 digest");
        }
        return value;
    }

    private static String requireString(Object value, String label) throws IOException {
        if (!(value instanceof String)) {
            throw invalid(label + " must be a string");
        }
        return (String) value;
    }

    private static void requireString(Object value, String label, String expected)
            throws IOException {
        String actual = requireString(value, label);
        if (!expected.equals(actual)) {
            throw invalid(label + " must be " + expected);
        }
    }

    private static void requireBoolean(
            Object value, String label, boolean expected) throws IOException {
        if (!(value instanceof Boolean) || ((Boolean) value) != expected) {
            throw invalid(label + " must be " + expected);
        }
    }

    private static void requireInteger(Object value, String label, int expected)
            throws IOException {
        if (!(value instanceof BigDecimal)
                || ((BigDecimal) value).compareTo(BigDecimal.valueOf(expected)) != 0) {
            throw invalid(label + " must be " + expected);
        }
    }

    private static String json(String value) {
        return value.replace("\\", "\\\\").replace("\"", "\\\"");
    }

    private static IOException invalid(String message) {
        return new IOException("Invalid SDX NNAPI device policy: " + message);
    }
}
