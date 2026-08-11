/*
 * Copyright (c) Eclipse Deeplearning4j
 * SPDX-License-Identifier: Apache-2.0
 */
package org.nd4j.dsp.model;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashSet;
import java.util.Set;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class SdxPlatformSdkTest {
    @TempDir
    Path temporary;

    @Test
    void everyTargetHasOneExactFailClosedProvider() {
        assertEquals(SdxTargetProfile.values().length, SdxPlatformSdk.providers().size());
        Set<String> providerIds = new HashSet<>();

        for (SdxTargetProfile target : SdxTargetProfile.values()) {
            SdxPlatformProviderDescriptor provider = target.platformProvider();
            assertSame(target, provider.targetProfile());
            assertTrue(providerIds.add(provider.providerId()));
            assertEquals("arm64", provider.architecture());
            assertEquals(SdxPlatformSdk.PROVIDER_ABI_VERSION, provider.providerAbiVersion());
            assertEquals(
                    SdxPlatformSdk.ARTIFACT_FORMAT_VERSION,
                    provider.artifactFormatVersion());
            assertNotEquals(
                    provider.requiresAotArtifact(),
                    provider.allowsRuntimeJit());
            if (provider.accelerator()
                    == SdxPlatformProviderDescriptor.Accelerator.COREML_ANE) {
                assertTrue(provider.allowsCpuFallback());
            } else {
                assertFalse(provider.allowsCpuFallback());
            }
            assertSame(provider, SdxPlatformSdk.requireProvider(target.id()));
            assertSame(provider, SdxPlatformSdk.requireProviderById(provider.providerId()));
        }

        assertFalse(SdxPlatformSdk.findProviderById("sdx.metal.v2").isPresent());
        assertThrows(
                IllegalArgumentException.class,
                () -> SdxPlatformSdk.requireProviderById("sdx.metal.v2"));
    }

    @Test
    void tensorG3ProfileSelectsStrictNnapiRuntimeBackend() {
        SdxTargetProfile tensorG3 =
                SdxTargetProfile.ANDROID_ARM64_NNAPI_ACCELERATOR;

        assertEquals("NNAPI", tensorG3.backend());
        assertEquals("AUTO", tensorG3.gpuTarget());
        assertSame(tensorG3, SdxTargetProfile.fromId("tensor-g3"));
        assertSame(tensorG3, SdxTargetProfile.fromId("pixel-8a"));
        assertFalse(tensorG3.platformProvider().allowsCpuFallback());
    }

    @Test
    void metalAndCoreMlAneAreSeparateAppleProviders() {
        SdxPlatformProviderDescriptor metal =
                SdxTargetProfile.IOS_ARM64_METAL.platformProvider();
        SdxPlatformProviderDescriptor coreMl =
                SdxTargetProfile.IOS_ARM64_COREML_ANE.platformProvider();

        assertEquals(SdxPlatformProviderDescriptor.Platform.IOS, metal.platform());
        assertEquals(SdxPlatformProviderDescriptor.Platform.IOS, coreMl.platform());
        assertEquals(SdxPlatformProviderDescriptor.Accelerator.METAL, metal.accelerator());
        assertEquals(
                SdxPlatformProviderDescriptor.Accelerator.COREML_ANE,
                coreMl.accelerator());
        assertNotEquals(metal.providerId(), coreMl.providerId());
        assertNotEquals(metal.artifactFormat(), coreMl.artifactFormat());
        assertEquals("sdz-dsp-mlx", metal.artifactFormat());
        assertEquals("Apple_ARM64_MLX_Metal", metal.defaultTargetSoc());
        assertEquals("MLX", metal.targetProfile().backend());
        assertEquals(SdxTargetProfile.ArtifactKind.NONE, metal.targetProfile().artifactKind());
        assertFalse(metal.targetProfile().hasPackagedArtifact());
        assertFalse(metal.requiresAotArtifact());
        assertTrue(metal.allowsRuntimeJit());
        assertFalse(metal.allowsCpuFallback());
        assertEquals(
                SdxTargetProfile.ArtifactKind.DIRECTORY,
                coreMl.targetProfile().artifactKind());
        assertTrue(coreMl.requiresAotArtifact());
        assertFalse(coreMl.allowsRuntimeJit());
        assertTrue(coreMl.allowsCpuFallback());
        assertTrue(metal.supportedOnSimulator());
        assertFalse(coreMl.supportedOnSimulator());
        assertSame(SdxTargetProfile.IOS_ARM64_METAL, SdxTargetProfile.fromId("ios-metal"));
        assertSame(
                SdxTargetProfile.IOS_ARM64_COREML_ANE,
                SdxTargetProfile.fromId("coreml-ane"));
        assertSame(
                SdxTargetProfile.IOS_ARM64_COREML_ANE,
                SdxTargetProfile.fromId("ane"));
    }

    @Test
    void compilerSeparatesMetalDeviceSpecializationFromCoreMlArtifact() throws Exception {
        Path source = createSourceSdz(temporary.resolve("source.sdz"));
        Path coreMlModel = Files.createDirectories(
                temporary.resolve("model.mlmodelc"));
        Files.write(
                coreMlModel.resolve("model.mil"),
                new byte[] {'M', 'I', 'L', 1});

        SdxModelCompiler compiler =
                new SdxModelCompiler(new SdxModelCache(temporary.resolve("cache")));
        SdxCompiledModel metal = compiler.compile(
                source,
                SdxTargetProfile.IOS_ARM64_METAL,
                SdxModelCompiler.metalDeviceCompilationPolicy(
                        "Apple_ARM64_MLX_Metal"),
                SdxModelCompiler.CompileOptions.builder()
                        .targetSoc("Apple_ARM64_MLX_Metal")
                        .build());
        SdxCompiledModel coreMl = compiler.compile(
                source,
                SdxTargetProfile.IOS_ARM64_COREML_ANE,
                SdxModelCompiler.preparedArtifact(
                        coreMlModel, "test-coreml-compiler", "1"));

        String metalManifest = Files.readString(
                metal.runtimeModelPath().resolve("manifest.json"),
                StandardCharsets.UTF_8);
        String coreMlManifest = Files.readString(
                coreMl.runtimeModelPath().resolve("manifest.json"),
                StandardCharsets.UTF_8);

        assertTrue(metalManifest.contains("\"id\":\"sdx.metal.v1\""));
        assertTrue(metalManifest.contains("\"artifactFormat\":\"sdz-dsp-mlx\""));
        assertTrue(metalManifest.contains(
                "\"canonicalSdzSha256\":\"" + metal.sourceIdentity().sha256() + "\""));
        assertFalse(metalManifest.contains("metalLibrary"));
        assertTrue(metalManifest.contains("\"allowCpuFallback\":false"));
        assertTrue(metalManifest.contains("\"allowRuntimeJit\":true"));
        assertFalse(Files.exists(metal.runtimeModelPath().resolve("artifacts/metal")));

        assertTrue(coreMlManifest.contains("\"id\":\"sdx.coreml-ane.v1\""));
        assertTrue(coreMlManifest.contains(
                "\"artifactFormat\":\"sdx-coreml-package\""));
        assertTrue(coreMlManifest.contains(
                "\"coreMlModel\":\"artifacts/coreml/model.mlmodelc\""));
        assertTrue(coreMlManifest.contains("\"allowCpuFallback\":true"));
        assertTrue(coreMlManifest.contains("\"allowRuntimeJit\":false"));
        assertTrue(Files.isDirectory(
                coreMl.runtimeModelPath().resolve("artifacts/coreml/model.mlmodelc")));
        assertNotEquals(metal.compileKey(), coreMl.compileKey());
    }

    @Test
    void productionMetalRejectsModelOwnedRawMetallib() throws Exception {
        Path source = createSourceSdz(temporary.resolve("raw-metal-source.sdz"));
        Path metallib = Files.write(
                temporary.resolve("raw-model.metallib"),
                new byte[] {'M', 'T', 'L', 'B', 1});
        SdxModelCompiler compiler =
                new SdxModelCompiler(new SdxModelCache(temporary.resolve("raw-metal-cache")));

        java.io.IOException failure = assertThrows(
                java.io.IOException.class,
                () -> compiler.compile(
                        source,
                        SdxTargetProfile.IOS_ARM64_METAL,
                        SdxModelCompiler.preparedArtifact(
                                metallib, "legacy-raw-metal", "1")));
        assertTrue(failure.getMessage().contains("must not emit"));
    }

    private static Path createSourceSdz(Path output) throws Exception {
        Files.createDirectories(output.toAbsolutePath().normalize().getParent());
        try (ZipOutputStream zip = new ZipOutputStream(Files.newOutputStream(output))) {
            put(zip, "model.sdnb", new byte[] {'S', 'D', 'N', 'B', 1, 2, 3, 4});
            put(
                    zip,
                    "metadata.properties",
                    "model=platform-sdk-test\n".getBytes(StandardCharsets.UTF_8));
        }
        return output;
    }

    private static void put(ZipOutputStream zip, String name, byte[] value)
            throws Exception {
        ZipEntry entry = new ZipEntry(name);
        entry.setTime(0L);
        zip.putNextEntry(entry);
        zip.write(value);
        zip.closeEntry();
    }
}
