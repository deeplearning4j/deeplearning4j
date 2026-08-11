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
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Enumeration;
import java.util.List;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;
import java.util.zip.ZipOutputStream;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class SdxMultiTargetBundleTest {
    private static final long CANONICAL_ZIP_TIME = 315532800000L;
    private static final byte[] TENSOR_PAYLOAD =
            LiteRtLmTestPackage.create(1, 5, 0);

    @TempDir
    Path temporary;

    @Test
    void packagesCanonicalGraphVulkanAndTensorDerivative() throws Exception {
        Path source = createSourceSdz(temporary.resolve("model.sdz"));
        Path spirv = createVulkanArtifact(temporary.resolve("spirv"));
        Path tensorModel =
                Files.write(temporary.resolve("model.litertlm"), TENSOR_PAYLOAD);

        SdxModelCache cache = new SdxModelCache(temporary.resolve("cache"));
        SdxModelCompiler compiler = new SdxModelCompiler(cache);
        SdxCompiledModel vulkan = compiler.compile(
                source,
                SdxTargetProfile.ANDROID_ARM64_VULKAN,
                SdxModelCompiler.preparedArtifact(spirv, "test-vulkan", "1"));
        SdxCompiledModel tensor = compiler.compile(
                source,
                SdxTargetProfile.ANDROID_ARM64_GOOGLE_TENSOR_G5,
                SdxModelCompiler.preparedArtifact(tensorModel, "test-tensor-g5", "1"));

        assertEquals(
                "artifacts/vulkan/spirv",
                SdxTargetProfile.ANDROID_ARM64_VULKAN.artifactRelativePath());
        assertEquals(
                "artifacts/tensor-g5/model.litertlm",
                SdxTargetProfile.ANDROID_ARM64_GOOGLE_TENSOR_G5.artifactRelativePath());
        assertTrue(Files.readString(
                        vulkan.runtimeModelPath().resolve("manifest.json"),
                        StandardCharsets.UTF_8)
                .contains("\"vulkanSpirv\":\"artifacts/vulkan/spirv\""));
        assertArrayEquals(TENSOR_PAYLOAD, Files.readAllBytes(tensor.runtimeModelPath()));
        assertEquals(SdxSourceIdentity.sha256(tensorModel),
                SdxSourceIdentity.sha256(tensor.runtimeModelPath()));

        List<SdxTargetProfile> targets = Arrays.asList(
                SdxTargetProfile.ANDROID_ARM64_VULKAN,
                SdxTargetProfile.ANDROID_ARM64_GOOGLE_TENSOR_G5);
        Path packed = temporary.resolve("model-with-targets.sdz");
        Path secondPacked = temporary.resolve("second-with-targets.sdz");
        cache.packageCompiledSdz(source, targets, packed);
        cache.packageCompiledSdz(source, targets, secondPacked);

        assertArrayEquals(Files.readAllBytes(packed), Files.readAllBytes(secondPacked));
        try (ZipFile archive = new ZipFile(packed.toFile())) {
            assertTrue(archive.getEntry(
                    SdxModelCache.EMBEDDED_ROOT + "source.properties") != null);
            ZipEntry tensorEntry = findEntry(
                    archive, "artifacts/tensor-g5/model.litertlm");
            assertArrayEquals(
                    TENSOR_PAYLOAD,
                    archive.getInputStream(tensorEntry).readAllBytes());

            Enumeration<? extends ZipEntry> entries = archive.entries();
            while (entries.hasMoreElements()) {
                assertEquals(CANONICAL_ZIP_TIME, entries.nextElement().getTime());
            }
        }

        SdxModelCache mobileCache = new SdxModelCache(temporary.resolve("mobile-cache"));
        assertTrue(Files.isDirectory(mobileCache.resolve(
                        packed, SdxTargetProfile.ANDROID_ARM64_VULKAN)
                .runtimeModelPath()));
        assertArrayEquals(
                TENSOR_PAYLOAD,
                Files.readAllBytes(mobileCache.resolve(
                                packed,
                                SdxTargetProfile.ANDROID_ARM64_GOOGLE_TENSOR_G5)
                        .runtimeModelPath()));
    }

    @Test
    void rejectsNonLiteRtLmTensorDerivative() throws Exception {
        Path source = createSourceSdz(temporary.resolve("invalid-source.sdz"));
        Path invalid = Files.write(
                temporary.resolve("model.bin"),
                "not-litertlm".getBytes(StandardCharsets.UTF_8));
        SdxModelCompiler compiler =
                new SdxModelCompiler(new SdxModelCache(temporary.resolve("invalid-cache")));

        IOException failure = assertThrows(
                IOException.class,
                () -> compiler.compile(
                        source,
                        SdxTargetProfile.ANDROID_ARM64_GOOGLE_TENSOR_G5,
                        SdxModelCompiler.preparedArtifact(
                                invalid, "test-tensor-g5", "1")));
        assertTrue(failure.getMessage().contains(".litertlm"));
    }

    private static Path createVulkanArtifact(Path directory) throws IOException {
        Files.createDirectories(directory);
        byte[] spirv = ByteBuffer.allocate(20)
                .order(ByteOrder.LITTLE_ENDIAN)
                .putInt(0x07230203)
                .putInt(0x00010000)
                .putInt(0)
                .putInt(1)
                .putInt(0)
                .array();
        Files.write(directory.resolve("spv_0123456789abcdef.spv"), spirv);
        Files.writeString(
                directory.resolve("spv_0123456789abcdef.meta"),
                "cacheAbi=vulkan-spirv-disk-cache-v2\n"
                        + "descriptorBindings=0;1\n"
                        + "spirvWords=5\n",
                StandardCharsets.UTF_8);
        return directory;
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
        entry.setTime(CANONICAL_ZIP_TIME);
        zip.putNextEntry(entry);
        zip.write(bytes);
        zip.closeEntry();
    }

    private static ZipEntry findEntry(ZipFile archive, String suffix) throws IOException {
        return archive.stream()
                .filter(entry -> entry.getName().endsWith(suffix))
                .findFirst()
                .orElseThrow(() -> new IOException("Missing ZIP entry ending in " + suffix));
    }
}
