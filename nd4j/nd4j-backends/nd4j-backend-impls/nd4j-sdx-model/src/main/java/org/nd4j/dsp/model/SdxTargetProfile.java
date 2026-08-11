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
    ANDROID_ARM64_NNAPI_ACCELERATOR(
            "android-arm64-nnapi-accelerator",
            RuntimeKind.SDX_BUNDLE,
            ArtifactKind.FILE,
            "NNAPI",
            "AUTO",
            "nnapiAcceleratorPolicy",
            "artifacts/nnapi/accelerator-only.json"),
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
            ArtifactKind.NONE,
            "MLX",
            "METAL",
            null,
            null),
    /**
     * Separate Core ML compilation/runtime contract for Apple Neural Engine.
     *
     * <p>The public Core ML compute-unit API also permits CPU execution, so the
     * physical-device provider must prove the accelerator-only policy or fail
     * session creation. This profile must never be implemented as a Metal alias.</p>
     */
    IOS_ARM64_COREML_ANE(
            "ios-arm64-coreml-ane",
            RuntimeKind.SDX_BUNDLE,
            ArtifactKind.DIRECTORY,
            "COREML",
            "AUTO",
            "coreMlModel",
            "artifacts/coreml/model.mlmodelc");

    public enum RuntimeKind {
        SDX_BUNDLE,
        DIRECT_ARTIFACT
    }

    public enum ArtifactKind {
        NONE,
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
        this.id = Objects.requireNonNull(id, "id");
        this.runtimeKind = Objects.requireNonNull(runtimeKind, "runtimeKind");
        this.artifactKind = Objects.requireNonNull(artifactKind, "artifactKind");
        this.backend = Objects.requireNonNull(backend, "backend");
        this.gpuTarget = Objects.requireNonNull(gpuTarget, "gpuTarget");
        if (artifactKind == ArtifactKind.NONE) {
            if (manifestArtifactKey != null || artifactRelativePath != null) {
                throw new IllegalArgumentException(
                        "Runtime-specialized targets cannot declare a bundle-owned artifact");
            }
            this.manifestArtifactKey = null;
            this.artifactRelativePath = null;
        } else {
            this.manifestArtifactKey =
                    Objects.requireNonNull(manifestArtifactKey, "manifestArtifactKey");
            this.artifactRelativePath =
                    Objects.requireNonNull(artifactRelativePath, "artifactRelativePath");
        }
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

    public boolean hasPackagedArtifact() {
        return artifactKind != ArtifactKind.NONE;
    }

    public String manifestArtifactKey() {
        if (!hasPackagedArtifact()) {
            throw new IllegalStateException(id + " has no bundle-owned accelerator artifact");
        }
        return manifestArtifactKey;
    }

    public String artifactRelativePath() {
        if (!hasPackagedArtifact()) {
            throw new IllegalStateException(id + " has no bundle-owned accelerator artifact");
        }
        return artifactRelativePath;
    }

    /**
     * Returns the portable per-chip SDK descriptor for this target.
     */
    public SdxPlatformProviderDescriptor platformProvider() {
        return SdxPlatformSdk.requireProvider(this);
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
        if ("nnapi-accelerator".equals(normalized)
                || "google-tensor-g3".equals(normalized)
                || "tensor-g3".equals(normalized)
                || "pixel-8a".equals(normalized)) {
            return ANDROID_ARM64_NNAPI_ACCELERATOR;
        }
        if ("google-tensor-g5".equals(normalized) || "tensor-g5".equals(normalized)) {
            return ANDROID_ARM64_GOOGLE_TENSOR_G5;
        }
        if ("metal".equals(normalized) || "ios-metal".equals(normalized)) {
            return IOS_ARM64_METAL;
        }
        if ("coreml".equals(normalized)
                || "coreml-ane".equals(normalized)
                || "ane".equals(normalized)
                || "ios-coreml-ane".equals(normalized)) {
            return IOS_ARM64_COREML_ANE;
        }
        throw new IllegalArgumentException("Unsupported SDX target profile: " + value);
    }
}
