/*
 * Copyright (c) Eclipse Deeplearning4j
 * SPDX-License-Identifier: Apache-2.0
 */
package org.nd4j.dsp.model;

import static org.nd4j.dsp.model.SdxPlatformProviderDescriptor.Accelerator;
import static org.nd4j.dsp.model.SdxPlatformProviderDescriptor.Platform;

import java.util.ArrayList;
import java.util.Collections;
import java.util.EnumMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

/**
 * Stable registry for the built-in SDX mobile platform providers.
 *
 * <p>This is a compile/package contract, not a runtime availability registry. Native packages
 * still have to link and explicitly register the requested provider. There is deliberately no
 * "best available" resolution because that would make accelerator-only packages silently cross
 * provider or CPU boundaries.</p>
 */
public final class SdxPlatformSdk {
    public static final int PROVIDER_ABI_VERSION = 1;
    public static final int ARTIFACT_FORMAT_VERSION = 1;

    private static final Map<SdxTargetProfile, SdxPlatformProviderDescriptor> BY_TARGET;
    private static final Map<String, SdxPlatformProviderDescriptor> BY_PROVIDER_ID;
    private static final List<SdxPlatformProviderDescriptor> PROVIDERS;

    static {
        EnumMap<SdxTargetProfile, SdxPlatformProviderDescriptor> targets =
                new EnumMap<>(SdxTargetProfile.class);
        LinkedHashMap<String, SdxPlatformProviderDescriptor> providerIds =
                new LinkedHashMap<>();

        register(targets, providerIds, descriptor(
                SdxTargetProfile.ANDROID_ARM64_VULKAN,
                "sdx.vulkan.v1",
                "sdx-vulkan-plan",
                Platform.ANDROID,
                Accelerator.VULKAN,
                "Android_Vulkan_1_1",
                true,
                false,
                false,
                false));
        register(targets, providerIds, descriptor(
                SdxTargetProfile.ANDROID_ARM64_HEXAGON_HTP,
                "sdx.hexagon-htp.v1",
                "sdx-htp-context",
                Platform.ANDROID,
                Accelerator.HEXAGON_HTP,
                "SM8650",
                true,
                false,
                false,
                false));
        register(targets, providerIds, descriptor(
                SdxTargetProfile.ANDROID_ARM64_NNAPI_ACCELERATOR,
                "sdx.nnapi-tensor-g3.v1",
                "sdx-nnapi-plan",
                Platform.ANDROID,
                Accelerator.NNAPI_TENSOR_G3,
                "Tensor_G3",
                true,
                false,
                false,
                false));
        register(targets, providerIds, descriptor(
                SdxTargetProfile.ANDROID_ARM64_GOOGLE_TENSOR_G5,
                "sdx.litert-tensor-g5.v1",
                "sdx-litert-model",
                Platform.ANDROID,
                Accelerator.LITERT_TENSOR_G5,
                "Tensor_G5",
                true,
                false,
                false,
                false));
        register(targets, providerIds, descriptor(
                SdxTargetProfile.IOS_ARM64_METAL,
                "sdx.metal.v1",
                "sdz-dsp-mlx",
                Platform.IOS,
                Accelerator.METAL,
                "Apple_ARM64_MLX_Metal",
                false,
                true,
                false,
                true));
        register(targets, providerIds, descriptor(
                SdxTargetProfile.IOS_ARM64_COREML_ANE,
                "sdx.coreml-ane.v1",
                "sdx-coreml-package",
                Platform.IOS,
                Accelerator.COREML_ANE,
                "Apple_ARM64_ANE",
                true,
                false,
                true,
                false));

        if (targets.size() != SdxTargetProfile.values().length) {
            throw new ExceptionInInitializerError(
                    "Every SDX target profile must have exactly one platform provider");
        }
        BY_TARGET = Collections.unmodifiableMap(targets);
        BY_PROVIDER_ID = Collections.unmodifiableMap(providerIds);
        PROVIDERS = Collections.unmodifiableList(new ArrayList<>(providerIds.values()));
    }

    private SdxPlatformSdk() {
    }

    public static SdxPlatformProviderDescriptor requireProvider(
            SdxTargetProfile targetProfile) {
        Objects.requireNonNull(targetProfile, "targetProfile");
        SdxPlatformProviderDescriptor descriptor = BY_TARGET.get(targetProfile);
        if (descriptor == null) {
            throw new IllegalArgumentException(
                    "No SDX platform provider for target " + targetProfile.id());
        }
        return descriptor;
    }

    public static SdxPlatformProviderDescriptor requireProvider(String targetProfile) {
        return requireProvider(SdxTargetProfile.fromId(targetProfile));
    }

    public static Optional<SdxPlatformProviderDescriptor> findProviderById(
            String providerId) {
        if (providerId == null) {
            return Optional.empty();
        }
        return Optional.ofNullable(BY_PROVIDER_ID.get(providerId.trim()));
    }

    public static SdxPlatformProviderDescriptor requireProviderById(String providerId) {
        return findProviderById(providerId).orElseThrow(() ->
                new IllegalArgumentException("SDX provider is not registered: " + providerId));
    }

    public static List<SdxPlatformProviderDescriptor> providers() {
        return PROVIDERS;
    }

    private static SdxPlatformProviderDescriptor descriptor(
            SdxTargetProfile target,
            String providerId,
            String artifactFormat,
            Platform platform,
            Accelerator accelerator,
            String defaultTargetSoc,
            boolean requiresAotArtifact,
            boolean allowsRuntimeJit,
            boolean allowsCpuFallback,
            boolean supportedOnSimulator) {
        return new SdxPlatformProviderDescriptor(
                target,
                providerId,
                PROVIDER_ABI_VERSION,
                artifactFormat,
                ARTIFACT_FORMAT_VERSION,
                platform,
                "arm64",
                accelerator,
                defaultTargetSoc,
                requiresAotArtifact,
                allowsRuntimeJit,
                allowsCpuFallback,
                supportedOnSimulator);
    }

    private static void register(
            Map<SdxTargetProfile, SdxPlatformProviderDescriptor> targets,
            Map<String, SdxPlatformProviderDescriptor> providerIds,
            SdxPlatformProviderDescriptor descriptor) {
        if (targets.put(descriptor.targetProfile(), descriptor) != null) {
            throw new ExceptionInInitializerError(
                    "Duplicate SDX target profile: " + descriptor.targetProfile().id());
        }
        if (providerIds.put(descriptor.providerId(), descriptor) != null) {
            throw new ExceptionInInitializerError(
                    "Duplicate SDX provider id: " + descriptor.providerId());
        }
    }
}
