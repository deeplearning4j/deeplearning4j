/*
 * Copyright (c) Eclipse Deeplearning4j
 * SPDX-License-Identifier: Apache-2.0
 */
package org.nd4j.dsp.model;

import java.util.Locale;
import java.util.Objects;

/**
 * Stable target identities used by the SDX compile cache.
 *
 * <p>The target is part of the cache key. Provider file extensions are deliberately
 * absent from the public application contract; they are internal properties of a
 * target profile and may change when its compiler ABI changes.</p>
 */
public enum SdxTargetProfile {
    ANDROID_ARM64_VULKAN(
            "android-arm64-vulkan",
            RuntimeKind.SDX_BUNDLE,
            ArtifactKind.DIRECTORY,
            "VULKAN",
            "VULKAN",
            "vulkanSpirv",
            "artifacts/vulkan/spirv"),
    ANDROID_ARM64_HEXAGON_HTP(
            "android-arm64-hexagon-htp",
            RuntimeKind.SDX_BUNDLE,
            ArtifactKind.DIRECTORY,
            "HEXAGON",
            "AUTO",
            "hexagonKernels",
            "artifacts/hexagon/kernels"),
    ANDROID_ARM64_GOOGLE_TENSOR_G5(
            "android-arm64-google-tensor-g5",
            RuntimeKind.DIRECT_ARTIFACT,
            ArtifactKind.FILE,
            "NPU",
            "AUTO",
            "tensorG5LiteRtLm",
            "artifacts/tensor-g5/model.litertlm"),
    IOS_ARM64_METAL(
            "ios-arm64-metal",
            RuntimeKind.SDX_BUNDLE,
            ArtifactKind.FILE,
            "METAL",
            "METAL",
            "metalLibrary",
            "artifacts/metal/model.metallib");

    public enum RuntimeKind {
        SDX_BUNDLE,
        DIRECT_ARTIFACT
    }

    public enum ArtifactKind {
        FILE,
        DIRECTORY
    }

    private final String id;
    private final RuntimeKind runtimeKind;
    private final ArtifactKind artifactKind;
    private final String backend;
    private final String gpuTarget;
    private final String manifestArtifactKey;
    private final String artifactRelativePath;

    SdxTargetProfile(
            String id,
            RuntimeKind runtimeKind,
            ArtifactKind artifactKind,
            String backend,
            String gpuTarget,
            String manifestArtifactKey,
            String artifactRelativePath) {
        this.id = id;
        this.runtimeKind = runtimeKind;
        this.artifactKind = artifactKind;
        this.backend = backend;
        this.gpuTarget = gpuTarget;
        this.manifestArtifactKey = manifestArtifactKey;
        this.artifactRelativePath = artifactRelativePath;
    }

    public String id() {
        return id;
    }

    public RuntimeKind runtimeKind() {
        return runtimeKind;
    }

    public ArtifactKind artifactKind() {
        return artifactKind;
    }

    public String backend() {
        return backend;
    }

    public String gpuTarget() {
        return gpuTarget;
    }

    public String manifestArtifactKey() {
        return manifestArtifactKey;
    }

    public String artifactRelativePath() {
        return artifactRelativePath;
    }

    public static SdxTargetProfile fromId(String value) {
        Objects.requireNonNull(value, "value");
        String normalized = value.trim().toLowerCase(Locale.ROOT);
        for (SdxTargetProfile profile : values()) {
            if (profile.id.equals(normalized)) {
                return profile;
            }
        }
        if ("vulkan-gpu".equals(normalized) || "vulkan".equals(normalized)) {
            return ANDROID_ARM64_VULKAN;
        }
        if ("hexagon-htp".equals(normalized) || "hexagon".equals(normalized)) {
            return ANDROID_ARM64_HEXAGON_HTP;
        }
        if ("google-tensor-g5".equals(normalized) || "tensor-g5".equals(normalized)) {
            return ANDROID_ARM64_GOOGLE_TENSOR_G5;
        }
        if ("metal".equals(normalized)) {
            return IOS_ARM64_METAL;
        }
        throw new IllegalArgumentException("Unsupported SDX target profile: " + value);
    }
}
