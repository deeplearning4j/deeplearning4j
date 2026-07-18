/*
 * Copyright (c) Eclipse Deeplearning4j
 * SPDX-License-Identifier: Apache-2.0
 */
package org.nd4j.dsp.model;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class SdxModelCompilerTest {
    @TempDir
    Path temporary;

    @Test
    void compilesOnceAndResolvesEmbeddedTargetsFromOneSdz() throws Exception {
        Path source = createSourceSdz(temporary.resolve("source.sdz"));
        Path tokenizer = Files.writeString(
                temporary.resolve("tokenizer.json"), "{\"version\":1}\n");
        Path llmConfig = Files.writeString(
                temporary.resolve("llm.json"), "{\"prefill\":\"input_ids\"}\n");

        SdxModelCache hostCache = new SdxModelCache(temporary.resolve("host-cache"));
        SdxModelCompiler compiler = new SdxModelCompiler(hostCache);
        AtomicInteger vulkanCompiles = new AtomicInteger();
        AtomicInteger tensorCompiles = new AtomicInteger();

        SdxModelCompiler.CompileOptions options =
                SdxModelCompiler.CompileOptions.builder()
                        .tokenizer(tokenizer)
                        .textGenerationConfig(llmConfig)
                        .cacheKeyProperty("shapeEnvelope", "batch1-context2048")
                        .build();

        SdxCompiledModel vulkan = compiler.compile(
                source,
                SdxTargetProfile.ANDROID_ARM64_VULKAN,
                fakeCompiler("test-vulkan", "1", vulkanCompiles),
                options);
        SdxCompiledModel vulkanAgain = compiler.compile(
                source,
                SdxTargetProfile.ANDROID_ARM64_VULKAN,
                fakeCompiler("test-vulkan", "1", vulkanCompiles),
                options);
        SdxCompiledModel tensor = compiler.compile(
                source,
                SdxTargetProfile.ANDROID_ARM64_GOOGLE_TENSOR_G5,
                fakeCompiler("test-tensor-g5", "7", tensorCompiles),
                SdxModelCompiler.CompileOptions.builder()
                        .cacheKeyProperty("quantization", "int8-per-channel")
                        .build());

        assertEquals(1, vulkanCompiles.get());
        assertEquals(1, tensorCompiles.get());
        assertEquals(vulkan.compileKey(), vulkanAgain.compileKey());
        assertTrue(Files.isDirectory(vulkan.runtimeModelPath()));
        assertTrue(Files.isRegularFile(tensor.runtimeModelPath()));
        assertTrue(vulkan.tokenizerPath().isPresent());

        String originalIdentity = hostCache.identify(source).sha256();
        Path packaged = temporary.resolve("compiled-model.sdz");
        hostCache.packageCompiledSdz(
                source,
                Arrays.asList(
                        SdxTargetProfile.ANDROID_ARM64_VULKAN,
                        SdxTargetProfile.ANDROID_ARM64_GOOGLE_TENSOR_G5),
                packaged);
        assertEquals(originalIdentity, hostCache.identify(packaged).sha256());

        SdxModelCache mobileCache =
                new SdxModelCache(temporary.resolve("mobile-cache"));
        SdxCompiledModel mobileVulkan = mobileCache.resolve(
                packaged, SdxTargetProfile.ANDROID_ARM64_VULKAN);
        SdxCompiledModel mobileTensor = mobileCache.resolve(
                packaged,
                SdxTargetProfile.ANDROID_ARM64_GOOGLE_TENSOR_G5);

        assertEquals(originalIdentity, mobileVulkan.sourceIdentity().sha256());
        assertEquals(originalIdentity, mobileTensor.sourceIdentity().sha256());
        assertTrue(Files.isDirectory(mobileVulkan.runtimeModelPath()));
        assertTrue(Files.isRegularFile(
                mobileVulkan.runtimeModelPath().resolve("manifest.json")));
        assertTrue(Files.isRegularFile(mobileTensor.runtimeModelPath()));
        assertArrayEquals(
                "tensor-g5-aot".getBytes(StandardCharsets.UTF_8),
                Files.readAllBytes(mobileTensor.runtimeModelPath()));
    }

    @Test
    void compilerVersionAndOptionsInvalidateTheTargetKey() throws Exception {
        Path source = createSourceSdz(temporary.resolve("identity.sdz"));
        SdxModelCache cache = new SdxModelCache(temporary.resolve("cache"));
        SdxModelCompiler compiler = new SdxModelCompiler(cache);

        AtomicInteger compiles = new AtomicInteger();
        SdxCompiledModel first = compiler.compile(
                source,
                SdxTargetProfile.ANDROID_ARM64_GOOGLE_TENSOR_G5,
                fakeCompiler("tensor", "1", compiles),
                SdxModelCompiler.CompileOptions.builder()
                        .cacheKeyProperty("calibration", "set-a")
                        .build());
        SdxCompiledModel second = compiler.compile(
                source,
                SdxTargetProfile.ANDROID_ARM64_GOOGLE_TENSOR_G5,
                fakeCompiler("tensor", "2", compiles),
                SdxModelCompiler.CompileOptions.builder()
                        .cacheKeyProperty("calibration", "set-a")
                        .build());
        SdxCompiledModel third = compiler.compile(
                source,
                SdxTargetProfile.ANDROID_ARM64_GOOGLE_TENSOR_G5,
                fakeCompiler("tensor", "2", compiles),
                SdxModelCompiler.CompileOptions.builder()
                        .cacheKeyProperty("calibration", "set-b")
                        .build());

        assertEquals(3, compiles.get());
        assertTrue(!first.compileKey().equals(second.compileKey()));
        assertTrue(!second.compileKey().equals(third.compileKey()));
    }

    @Test
    void cacheMissDoesNotFallBackToRuntimeCompilation() throws Exception {
        Path source = createSourceSdz(temporary.resolve("plain.sdz"));
        SdxModelCache cache = new SdxModelCache(temporary.resolve("empty-cache"));
        SdxModelCache.MissingCompiledModelException failure = assertThrows(
                SdxModelCache.MissingCompiledModelException.class,
                () -> cache.resolve(
                        source, SdxTargetProfile.ANDROID_ARM64_HEXAGON_HTP));
        assertTrue(failure.getMessage().contains("never falls back"));
    }

    private static SdxModelCompiler.TargetCompiler fakeCompiler(
            String id, String version, AtomicInteger invocations) {
        return new SdxModelCompiler.TargetCompiler() {
            @Override
            public String id() {
                return id;
            }

            @Override
            public String version() {
                return version;
            }

            @Override
            public String cacheKeyMaterial(
                    Path sourceModel,
                    SdxTargetProfile target,
                    SdxModelCompiler.CompileOptions options) {
                return "fixed-test-toolchain";
            }

            @Override
            public Path compile(SdxModelCompiler.CompilationContext context)
                    throws IOException {
                invocations.incrementAndGet();
                Path output = context.suggestedOutput();
                if (context.target().artifactKind()
                        == SdxTargetProfile.ArtifactKind.DIRECTORY) {
                    Files.createDirectories(output);
                    Path module = output.resolve("spv_0123456789abcdef.spv");
                    byte[] spirv = new byte[20];
                    spirv[0] = 0x03;
                    spirv[1] = 0x02;
                    spirv[2] = 0x23;
                    spirv[3] = 0x07;
                    Files.write(module, spirv);
                    Files.writeString(
                            output.resolve("spv_0123456789abcdef.meta"),
                            "cacheAbi=vulkan-spirv-disk-cache-v2\n"
                                    + "descriptorBindings=0\n"
                                    + "spirvWords=5\n");
                } else {
                    Files.createDirectories(output.getParent());
                    Files.write(
                            output,
                            "tensor-g5-aot".getBytes(StandardCharsets.UTF_8));
                }
                return output;
            }
        };
    }

    private static Path createSourceSdz(Path output) throws IOException {
        try (ZipOutputStream zip = new ZipOutputStream(Files.newOutputStream(output))) {
            put(zip, "model.sdnb", new byte[] {'S', 'D', 'N', 'B', 1, 2, 3, 4});
            put(zip, "metadata.properties",
                    "model=test\n".getBytes(StandardCharsets.UTF_8));
        }
        return output;
    }

    private static void put(ZipOutputStream zip, String name, byte[] bytes)
            throws IOException {
        ZipEntry entry = new ZipEntry(name);
        entry.setTime(0L);
        zip.putNextEntry(entry);
        zip.write(bytes);
        zip.closeEntry();
    }
}
