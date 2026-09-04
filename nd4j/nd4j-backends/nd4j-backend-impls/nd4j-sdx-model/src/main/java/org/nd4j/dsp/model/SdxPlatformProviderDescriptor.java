/*
 * Copyright (c) Eclipse Deeplearning4j
 * SPDX-License-Identifier: Apache-2.0
 */
package org.nd4j.dsp.model;

import java.util.Objects;

/**
 * Immutable public contract for one accelerator implementation in the SDX mobile SDK.
 *
 * <p>The descriptor exposes stable identities and policy, never a provider-specific model
 * filename to an application. {@link SdxModelCompiler} owns derivation and caching from the
 * canonical SDZ; runtimes use the descriptor to select exactly one linked provider and fail
 * when that provider, its required AOT artifact, or its device-specialization policy is
 * unavailable.</p>
 */
public final class SdxPlatformProviderDescriptor {
    public enum Platform {
        ANDROID,
        IOS
    }

    public enum Accelerator {
        VULKAN,
        HEXAGON_HTP,
        NNAPI_TENSOR_G3,
        LITERT_TENSOR_G5,
        METAL,
        COREML_ANE
    }

    private final SdxTargetProfile targetProfile;
    private final String providerId;
    private final int providerAbiVersion;
    private final String artifactFormat;
    private final int artifactFormatVersion;
    private final Platform platform;
    private final String architecture;
    private final Accelerator accelerator;
    private final String defaultTargetSoc;
    private final boolean requiresAotArtifact;
    private final boolean allowsRuntimeJit;
    private final boolean allowsCpuFallback;
    private final boolean supportedOnSimulator;
    private final int fixedTextContextCapacity;

    SdxPlatformProviderDescriptor(
            SdxTargetProfile targetProfile,
            String providerId,
            int providerAbiVersion,
            String artifactFormat,
            int artifactFormatVersion,
            Platform platform,
            String architecture,
            Accelerator accelerator,
            String defaultTargetSoc,
            boolean requiresAotArtifact,
            boolean allowsRuntimeJit,
            boolean allowsCpuFallback,
            boolean supportedOnSimulator,
            int fixedTextContextCapacity) {
        this.targetProfile = Objects.requireNonNull(targetProfile, "targetProfile");
        this.providerId = requireIdentity(providerId, "providerId");
        if (providerAbiVersion <= 0) {
            throw new IllegalArgumentException("providerAbiVersion must be positive");
        }
        this.providerAbiVersion = providerAbiVersion;
        this.artifactFormat = requireIdentity(artifactFormat, "artifactFormat");
        if (artifactFormatVersion <= 0) {
            throw new IllegalArgumentException("artifactFormatVersion must be positive");
        }
        this.artifactFormatVersion = artifactFormatVersion;
        this.platform = Objects.requireNonNull(platform, "platform");
        this.architecture = requireIdentity(architecture, "architecture");
        this.accelerator = Objects.requireNonNull(accelerator, "accelerator");
        this.defaultTargetSoc = requireIdentity(defaultTargetSoc, "defaultTargetSoc");
        this.requiresAotArtifact = requiresAotArtifact;
        this.allowsRuntimeJit = allowsRuntimeJit;
        this.allowsCpuFallback = allowsCpuFallback;
        this.supportedOnSimulator = supportedOnSimulator;
        if (fixedTextContextCapacity < 0) {
            throw new IllegalArgumentException("fixedTextContextCapacity must not be negative");
        }
        this.fixedTextContextCapacity = fixedTextContextCapacity;
        if (requiresAotArtifact == allowsRuntimeJit) {
            throw new IllegalArgumentException(
                    "A mobile provider must select exactly one compilation owner: "
                            + "bundle-owned AOT or runtime device specialization");
        }
        if (allowsCpuFallback && accelerator != Accelerator.COREML_ANE) {
            throw new IllegalArgumentException(
                    "Only Core ML/ANE may acknowledge vendor-managed CPU scheduling");
        }
        if (accelerator == Accelerator.COREML_ANE && !allowsCpuFallback) {
            throw new IllegalArgumentException(
                    "Core ML/ANE must acknowledge CPU_AND_NEURAL_ENGINE scheduling");
        }
    }

    public SdxTargetProfile targetProfile() {
        return targetProfile;
    }

    public String providerId() {
        return providerId;
    }

    public int providerAbiVersion() {
        return providerAbiVersion;
    }

    public String artifactFormat() {
        return artifactFormat;
    }

    public int artifactFormatVersion() {
        return artifactFormatVersion;
    }

    public Platform platform() {
        return platform;
    }

    public String architecture() {
        return architecture;
    }

    public Accelerator accelerator() {
        return accelerator;
    }

    public String defaultTargetSoc() {
        return defaultTargetSoc;
    }

    public boolean requiresAotArtifact() {
        return requiresAotArtifact;
    }

    public boolean allowsRuntimeJit() {
        return allowsRuntimeJit;
    }

    public boolean allowsCpuFallback() {
        return allowsCpuFallback;
    }

    /**
     * Whether the provider contract may be probed on a simulator.
     *
     * <p>This is not a runtime availability claim. In particular, Core ML/ANE remains
     * physical-device-only.</p>
     */
    public boolean supportedOnSimulator() {
        return supportedOnSimulator;
    }

    /**
     * Fixed physical token capacity for one reusable compiled text plan.
     * Zero delegates capacity selection to bundle metadata.
     */
    public int fixedTextContextCapacity() {
        return fixedTextContextCapacity;
    }

    private static String requireIdentity(String value, String name) {
        Objects.requireNonNull(value, name);
        String normalized = value.trim();
        if (normalized.isEmpty()
                || normalized.indexOf('\n') >= 0
                || normalized.indexOf('\r') >= 0) {
            throw new IllegalArgumentException(name + " must be a non-empty single-line value");
        }
        return normalized;
    }
}
