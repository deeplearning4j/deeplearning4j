/*
 * Copyright (c) Eclipse Deeplearning4j
 * SPDX-License-Identifier: Apache-2.0
 */
package org.nd4j.dsp.model;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
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
    void generatedSdzAdmissionMovesOnceAndReusesContentAddressedSource() throws Exception {
        SdxModelCache cache = new SdxModelCache(temporary.resolve("generated-cache"));
        Path generated = createSourceSdz(temporary.resolve("generated.sdz"));
        SdxSourceIdentity identity = cache.identify(generated);

        Path admitted = cache.admitGeneratedSource(generated);
        assertFalse(Files.exists(generated));
        assertTrue(Files.isRegularFile(admitted));
        assertEquals(identity.sha256(), cache.identify(admitted).sha256());

        Path duplicate = Files.copy(admitted, temporary.resolve("duplicate.sdz"));
        Path reused = cache.admitGeneratedSource(duplicate);
        assertEquals(admitted, reused);
        assertFalse(Files.exists(duplicate));
    }

    @Test
    void inventoriesExistingCompiledModelsAndPhysicalStorageWithoutRecompiling() throws Exception {
        Path source = createSourceSdz(temporary.resolve("inventory-source.sdz"));
        SdxModelCache cache = new SdxModelCache(temporary.resolve("inventory-cache"));
        AtomicInteger compiles = new AtomicInteger();
        SdxCompiledModel compiled = new SdxModelCompiler(cache).compile(
                source,
                SdxTargetProfile.ANDROID_ARM64_GOOGLE_TENSOR_G5,
                fakeCompiler("inventory-compiler", "1", compiles));

        SdxModelCacheInventory inventory = cache.inventory(
                SdxTargetProfile.ANDROID_ARM64_GOOGLE_TENSOR_G5);

        assertEquals(1, compiles.get());
        assertEquals(1, inventory.entries().size());
        SdxCachedModel entry = inventory.entries().get(0);
        assertEquals(compiled.compileKey(), entry.compileKey());
        assertEquals(compiled.sourceIdentity().sha256(), entry.sourceSha256());
        assertEquals(compiled.sourceIdentity().logicalBytes(), entry.sourceLogicalBytes());
        assertEquals(compiled.target(), entry.target());
        assertEquals(compiled.compilerId(), entry.compilerId());
        assertEquals(compiled.compilerVersion(), entry.compilerVersion());
        assertEquals(compiled.runtimeModelPath(), entry.runtimeModelPath());
        assertTrue(Files.isRegularFile(entry.sourceModel()));
        assertTrue(entry.sourcePhysicalBytes() > 0L);
        assertTrue(entry.objectPhysicalBytes() > 0L);
        assertTrue(inventory.totalPhysicalBytes() >= entry.referencedPhysicalBytes());
        assertEquals(entry.sourcePhysicalBytes(), inventory.referencedSourceBytes());
        assertEquals(entry.objectPhysicalBytes(), inventory.referencedObjectBytes());
        assertEquals(0, inventory.invalidReferenceCount());

        assertTrue(cache.inventory(SdxTargetProfile.ANDROID_ARM64_VULKAN)
                .entries().isEmpty());
        assertEquals(1, compiles.get(), "inventory must never invoke a target compiler");

        byte[] runtimeBytes = Files.readAllBytes(compiled.runtimeModelPath());
        runtimeBytes[0] ^= 0x01;
        Files.write(compiled.runtimeModelPath(), runtimeBytes);
        assertEquals(1, cache.inventory(
                SdxTargetProfile.ANDROID_ARM64_GOOGLE_TENSOR_G5).entries().size(),
                "metadata inventory intentionally avoids hashing large payloads");
        IOException corruption = assertThrows(
                IOException.class,
                () -> cache.resolveVerified(
                        source, SdxTargetProfile.ANDROID_ARM64_GOOGLE_TENSOR_G5));
        assertTrue(corruption.getMessage().contains("checksum mismatch"));
    }

    @Test
    void inventoryBoundsMalformedReferenceMetadata() throws Exception {
        SdxModelCache cache = new SdxModelCache(temporary.resolve("malformed-inventory"));
        Path reference = cache.root().resolve("index")
                .resolve("a".repeat(64))
                .resolve(SdxTargetProfile.ANDROID_ARM64_NNAPI_ACCELERATOR.id() + ".ref");
        Files.createDirectories(reference.getParent());
        Files.write(reference, new byte[4_096]);

        SdxModelCacheInventory inventory = cache.inventory(
                SdxTargetProfile.ANDROID_ARM64_NNAPI_ACCELERATOR);

        assertTrue(inventory.entries().isEmpty());
        assertEquals(1, inventory.invalidReferenceCount());
        assertTrue(inventory.totalPhysicalBytes() >= 4_096L);
    }

    @Test
    void compilesOnceAndResolvesEmbeddedTargetsFromOneSdz() throws Exception {
        Path source = createSourceSdz(temporary.resolve("source.sdz"));
        Path tokenizer = Files.writeString(
                temporary.resolve("tokenizer.json"), "{\"version\":1}\n");
        Path tokenizerConfig = Files.writeString(
                temporary.resolve("tokenizer_config.json"),
                "{\"chat_template\":\"{{ messages }}\",\"eos_token\":\"</s>\"}\n");
        Path llmConfig = Files.writeString(
                temporary.resolve("llm.json"), "{\"prefill\":\"input_ids\"}\n");

        SdxModelCache hostCache = new SdxModelCache(temporary.resolve("host-cache"));
        SdxModelCompiler compiler = new SdxModelCompiler(hostCache);
        AtomicInteger vulkanCompiles = new AtomicInteger();
        AtomicInteger tensorCompiles = new AtomicInteger();

        SdxModelCompiler.CompileOptions options =
                SdxModelCompiler.CompileOptions.builder()
                        .tokenizer(tokenizer)
                        .tokenizerConfig(tokenizerConfig)
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
        assertTrue(vulkan.tokenizerConfigPath().isPresent());
        assertEquals(
                "{\"chat_template\":\"{{ messages }}\",\"eos_token\":\"</s>\"}\n",
                Files.readString(vulkan.tokenizerConfigPath().orElseThrow()));
        SdxTextModelAssets hostTextAssets = vulkan.requireTextModelAssets();
        assertEquals(vulkan.tokenizerPath().orElseThrow(), hostTextAssets.tokenizer());
        assertEquals(
                vulkan.tokenizerConfigPath().orElseThrow(),
                hostTextAssets.tokenizerConfig());

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
        SdxTextModelAssets mobileTextAssets = mobileVulkan.requireTextModelAssets();
        assertEquals(
                "{\"chat_template\":\"{{ messages }}\",\"eos_token\":\"</s>\"}\n",
                Files.readString(mobileTextAssets.tokenizerConfig()));
        assertTrue(Files.isRegularFile(mobileTensor.runtimeModelPath()));
        assertArrayEquals(
                liteRtLmPackage(),
                Files.readAllBytes(mobileTensor.runtimeModelPath()));
    }

    @Test
    void textConsumersRejectIncompleteCompiledArtifacts() throws Exception {
        Path source = createSourceSdz(temporary.resolve("incomplete-text.sdz"));
        Path tokenizer = Files.writeString(
                temporary.resolve("incomplete-tokenizer.json"), "{\"version\":1}\n");
        SdxModelCompiler compiler = new SdxModelCompiler(
                new SdxModelCache(temporary.resolve("incomplete-cache")));
        SdxCompiledModel model = compiler.compile(
                source,
                SdxTargetProfile.ANDROID_ARM64_VULKAN,
                fakeCompiler("test-vulkan", "1", new AtomicInteger()),
                SdxModelCompiler.CompileOptions.builder()
                        .tokenizer(tokenizer)
                        .build());

        IOException failure = assertThrows(
                IOException.class,
                model::requireTextModelAssets);
        assertTrue(failure.getMessage().contains("tokenizer configuration"));
        assertTrue(failure.getMessage().toLowerCase().contains("rebuild the sdz"));
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
    void validatesQuantizationBeforeInvokingTargetCompiler() throws Exception {
        Path source = createSourceSdz(temporary.resolve("quantized.sdz"));
        Path invalid = Files.writeString(
                temporary.resolve("invalid-quantization.json"),
                quantizationJson().replace(
                        "\"allowFloatFallback\":false",
                        "\"allowFloatFallback\":true"));
        AtomicInteger compiles = new AtomicInteger();
        SdxModelCompiler compiler =
                new SdxModelCompiler(new SdxModelCache(temporary.resolve("quant-cache")));

        assertThrows(
                IOException.class,
                () -> compiler.compile(
                        source,
                        SdxTargetProfile.ANDROID_ARM64_NNAPI_ACCELERATOR,
                        fakeCompiler("test-nnapi", "1", compiles),
                        SdxModelCompiler.CompileOptions.builder()
                                .quantizationConfig(invalid)
                                .build()));
        assertEquals(0, compiles.get());

        Path valid = Files.writeString(
                temporary.resolve("valid-quantization.json"),
                quantizationJson());
        IOException wrongRequestedSoc = assertThrows(
                IOException.class,
                () -> compiler.compile(
                        source,
                        SdxTargetProfile.ANDROID_ARM64_NNAPI_ACCELERATOR,
                        fakeCompiler("test-nnapi", "1", compiles),
                        SdxModelCompiler.CompileOptions.builder()
                                .quantizationConfig(valid)
                                .targetSoc("Tensor_G4")
                                .build()));
        assertTrue(wrongRequestedSoc.getMessage().contains("Tensor_G4"));
        assertEquals(0, compiles.get());

        SdxCompiledModel compiled = compiler.compile(
                source,
                SdxTargetProfile.ANDROID_ARM64_NNAPI_ACCELERATOR,
                fakeCompiler("test-nnapi", "1", compiles),
                SdxModelCompiler.CompileOptions.builder()
                        .quantizationConfig(valid)
                        .targetSoc("Tensor_G3")
                        .build());
        assertEquals(1, compiles.get());
        assertTrue(compiled.quantizationConfigPath().isPresent());
    }

    @Test
    void sharedBuiltInCompilerSelectionCoversDesktopAndAotEntryPoints() {
        assertEquals(
                "sdx-mlx-device-compilation",
                SdxModelCompiler.requireBuiltInTargetCompiler(
                                SdxTargetProfile.IOS_ARM64_METAL, null, false)
                        .id());
        assertEquals(
                "sdx-nnapi-device-policy",
                SdxModelCompiler.requireBuiltInTargetCompiler(
                                SdxTargetProfile.ANDROID_ARM64_NNAPI_ACCELERATOR,
                                null,
                                false)
                        .id());
        assertTrue(
                SdxModelCompiler.requireBuiltInTargetCompiler(
                                SdxTargetProfile.ANDROID_ARM64_NNAPI_ACCELERATOR,
                                SdxTensorG3NnapiCompiler.TARGET_SOC,
                                true)
                        instanceof SdxTensorG3NnapiCompiler);
        assertTrue(
                SdxModelCompiler.builtInTargetCompiler(
                                SdxTargetProfile.ANDROID_ARM64_VULKAN, null, false)
                        .isEmpty());
        assertThrows(
                IllegalArgumentException.class,
                () -> SdxModelCompiler.requireBuiltInTargetCompiler(
                        SdxTargetProfile.ANDROID_ARM64_VULKAN, null, false));
    }

    @Test
    void writesAuditableNnapiDeviceCompilationPolicyWithoutDerivedModel()
            throws Exception {
        Path source = createSourceSdz(temporary.resolve("pixel-source.sdz"));
        SdxModelCache cache = new SdxModelCache(temporary.resolve("pixel-cache"));
        SdxModelCompiler.TargetCompiler policyCompiler =
                SdxModelCompiler.nnapiDeviceCompilationPolicy("Tensor_G3");
        assertFalse(policyCompiler.requiresIsolatedSourceSnapshot());
        SdxCompiledModel compiled = new SdxModelCompiler(cache).compile(
                source,
                SdxTargetProfile.ANDROID_ARM64_NNAPI_ACCELERATOR,
                policyCompiler,
                SdxModelCompiler.CompileOptions.builder()
                        .targetSoc("Tensor_G3")
                        .build());

        Path policyPath = compiled.runtimeModelPath()
                .resolve("artifacts/nnapi/accelerator-only.json");
        SdxNnapiDevicePolicy policy = SdxNnapiDevicePolicy.load(policyPath);
        assertEquals(SdxNnapiDevicePolicy.POLICY_ABI, policy.policyAbi());
        assertEquals("android-arm64-nnapi-accelerator", policy.target());
        assertEquals("Tensor_G3", policy.targetSoc());
        assertEquals(cache.identify(source).sha256(), policy.sourceSha256());
        assertEquals(null, policy.derivedModelSha256());
        assertEquals("DEVICE_ACCELERATOR", policy.deviceType());
        assertTrue(policy.requireWholeGraph());
        assertTrue(!policy.allowCpu());
        assertTrue(!policy.allowGpu());
        assertTrue(!policy.allowFallback());
        assertEquals("android-nnapi-device", policy.compilationLocation());
        assertTrue(policy.persistentCache());
        String manifest = Files.readString(
                compiled.runtimeModelPath().resolve("manifest.json"));
        assertTrue(!manifest.contains("\"compiledModel\""));
        assertTrue(manifest.contains("../../../sources/"));
    }

    @Test
    void nnapiDeviceCompilationPolicyRejectsQuantizationAndOtherTargets()
            throws Exception {
        Path source = createSourceSdz(temporary.resolve("policy-reject.sdz"));
        Path quantization = Files.writeString(
                temporary.resolve("policy-quantization.json"), quantizationJson());
        SdxModelCompiler compiler = new SdxModelCompiler(
                new SdxModelCache(temporary.resolve("policy-reject-cache")));

        IOException quantized = assertThrows(
                IOException.class,
                () -> compiler.compile(
                        source,
                        SdxTargetProfile.ANDROID_ARM64_NNAPI_ACCELERATOR,
                        SdxModelCompiler.nnapiDeviceCompilationPolicy("Tensor_G3"),
                        SdxModelCompiler.CompileOptions.builder()
                                .quantizationConfig(quantization)
                                .build()));
        assertTrue(quantized.getMessage().contains("cannot quantize"));

        IOException wrongTarget = assertThrows(
                IOException.class,
                () -> compiler.compile(
                        source,
                        SdxTargetProfile.ANDROID_ARM64_VULKAN,
                        SdxModelCompiler.nnapiDeviceCompilationPolicy("Tensor_G3")));
        assertTrue(wrongTarget.getMessage().contains("cannot compile target"));
    }

    @Test
    void rejectsUnchangedDerivedSdzForQuantizedNnapi() throws Exception {
        Path source = createSourceSdz(temporary.resolve("unchanged-source.sdz"));
        Path quantization = Files.writeString(
                temporary.resolve("unchanged-quantization.json"), quantizationJson());
        IOException failure = assertThrows(
                IOException.class,
                () -> new SdxModelCompiler(new SdxModelCache(
                                temporary.resolve("unchanged-cache")))
                        .compile(
                                source,
                                SdxTargetProfile.ANDROID_ARM64_NNAPI_ACCELERATOR,
                                nnapiCompiler(true, false),
                                SdxModelCompiler.CompileOptions.builder()
                                        .quantizationConfig(quantization)
                                        .targetSoc("Tensor_G3")
                                        .build()));
        assertTrue(failure.getMessage().contains("unchanged derived SDZ"));
    }

    @Test
    void rejectsNnapiPolicyThatPermitsFallback() throws Exception {
        Path source = createSourceSdz(temporary.resolve("fallback-policy.sdz"));
        IOException failure = assertThrows(
                IOException.class,
                () -> new SdxModelCompiler(new SdxModelCache(
                                temporary.resolve("fallback-policy-cache")))
                        .compile(
                                source,
                                SdxTargetProfile.ANDROID_ARM64_NNAPI_ACCELERATOR,
                                nnapiCompiler(false, true),
                                SdxModelCompiler.CompileOptions.builder()
                                        .targetSoc("Tensor_G3")
                                        .build()));
        assertTrue(failure.getMessage().contains("allowFallback must be false"));
    }

    @Test
    void compilesFromImmutableSnapshotsWhenCallerInputsChange() throws Exception {
        Path source = createSourceSdz(temporary.resolve("mutable-source.sdz"));
        Path tokenizer = Files.writeString(
                temporary.resolve("mutable-tokenizer.json"), "original-tokenizer\n");
        SdxSourceIdentity original = SdxSourceIdentity.identify(source);
        SdxModelCompiler.TargetCompiler mutatingCallerInputs =
                new SdxModelCompiler.TargetCompiler() {
                    @Override
                    public String id() {
                        return "snapshot-test";
                    }

                    @Override
                    public String version() {
                        return "1";
                    }

                    @Override
                    public String cacheKeyMaterial(
                            Path sourceModel,
                            SdxTargetProfile target,
                            SdxModelCompiler.CompileOptions options) {
                        return "snapshot-test";
                    }

                    @Override
                    public Path compile(SdxModelCompiler.CompilationContext context)
                            throws IOException {
                        createSourceSdz(source, (byte) 9);
                        Files.writeString(tokenizer, "changed-tokenizer\n");
                        assertTrue(!context.sourceModel().equals(source));
                        assertEquals(
                                original.sha256(),
                                SdxSourceIdentity.identify(context.sourceModel()).sha256());
                        assertEquals(
                                "original-tokenizer\n",
                                Files.readString(context.options().tokenizer()));
                        writeVulkanArtifact(context.suggestedOutput());
                        return context.suggestedOutput();
                    }
                };

        SdxCompiledModel compiled = new SdxModelCompiler(
                new SdxModelCache(temporary.resolve("snapshot-cache"))).compile(
                        source,
                        SdxTargetProfile.ANDROID_ARM64_VULKAN,
                        mutatingCallerInputs,
                        SdxModelCompiler.CompileOptions.builder()
                                .tokenizer(tokenizer)
                                .build());

        assertEquals(original.sha256(), compiled.sourceIdentity().sha256());
        assertTrue(!original.sha256().equals(
                SdxSourceIdentity.identify(source).sha256()));
        assertEquals("original-tokenizer\n", Files.readString(
                compiled.tokenizerPath().orElseThrow()));
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
                if (context.target()
                        == SdxTargetProfile.ANDROID_ARM64_NNAPI_ACCELERATOR) {
                    SdxSourceIdentity derived = null;
                    if (context.options().quantizationConfig() != null) {
                        createSourceSdz(context.suggestedModelOutput(), (byte) 8);
                        derived = SdxSourceIdentity.identify(
                                context.suggestedModelOutput());
                    }
                    SdxNnapiDevicePolicy.create(
                            context.target(),
                            context.options().targetSoc() == null
                                    ? "Tensor_G3"
                                    : context.options().targetSoc(),
                            context.sourceIdentity(),
                            derived).write(output);
                } else if (context.target().artifactKind()
                        == SdxTargetProfile.ArtifactKind.DIRECTORY) {
                    writeVulkanArtifact(output);
                } else {
                    Files.createDirectories(output.getParent());
                    Files.write(output, liteRtLmPackage());
                }
                return output;
            }
        };
    }

    private static SdxModelCompiler.TargetCompiler nnapiCompiler(
            boolean unchangedDerivedModel, boolean allowFallback) {
        return new SdxModelCompiler.TargetCompiler() {
            @Override
            public String id() {
                return "test-nnapi-policy";
            }

            @Override
            public String version() {
                return "1";
            }

            @Override
            public String cacheKeyMaterial(
                    Path sourceModel,
                    SdxTargetProfile target,
                    SdxModelCompiler.CompileOptions options) {
                return "unchanged:" + unchangedDerivedModel
                        + ";fallback:" + allowFallback;
            }

            @Override
            public Path compile(SdxModelCompiler.CompilationContext context)
                    throws IOException {
                SdxSourceIdentity derived = null;
                if (context.options().quantizationConfig() != null) {
                    if (unchangedDerivedModel) {
                        Files.copy(
                                context.sourceModel(),
                                context.suggestedModelOutput());
                    } else {
                        createSourceSdz(context.suggestedModelOutput(), (byte) 7);
                    }
                    derived = SdxSourceIdentity.identify(
                            context.suggestedModelOutput());
                }
                SdxNnapiDevicePolicy.create(
                        context.target(),
                        context.options().targetSoc(),
                        context.sourceIdentity(),
                        derived).write(context.suggestedOutput());
                if (allowFallback) {
                    Files.writeString(
                            context.suggestedOutput(),
                            Files.readString(context.suggestedOutput()).replace(
                                    "\"allowFallback\": false",
                                    "\"allowFallback\": true"));
                }
                return context.suggestedOutput();
            }
        };
    }

    private static void writeVulkanArtifact(Path output) throws IOException {
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
    }

    private static String quantizationJson() {
        return "{"
                + "\"formatVersion\":1,"
                + "\"scheme\":\"int8-per-tensor\","
                + "\"provider\":\"sdx-graph\","
                + "\"targetSocs\":[\"Tensor_G3\"],"
                + "\"deviceOnly\":true,"
                + "\"allowFloatFallback\":false,"
                + "\"requireVendorAot\":true,"
                + "\"weights\":{"
                + "\"dtype\":\"INT8\","
                + "\"scaleDtype\":\"FLOAT32\","
                + "\"granularity\":\"per-tensor\","
                + "\"scale\":0.015625,"
                + "\"symmetric\":true,"
                + "\"zeroPoint\":0},"
                + "\"activations\":{\"dtype\":\"INT8\","
                + "\"scaleDtype\":\"FLOAT32\","
                + "\"granularity\":\"per-tensor\","
                + "\"scale\":0.03125,\"zeroPoint\":0,"
                + "\"calibration\":{\"method\":\"minmax\","
                + "\"sampleCount\":128,"
                + "\"datasetSha256\":\""
                + "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\"}},"
                + "\"outputs\":{\"dtype\":\"INT8\","
                + "\"scaleDtype\":\"FLOAT32\","
                + "\"granularity\":\"per-tensor\","
                + "\"scale\":0.0625,\"zeroPoint\":0},"
                + "\"excludedOps\":[]"
                + "}";
    }

    private static byte[] liteRtLmPackage() {
        return LiteRtLmTestPackage.create(1, 5, 0);
    }

    private static Path createSourceSdz(Path output) throws IOException {
        return createSourceSdz(output, (byte) 4);
    }

    private static Path createSourceSdz(Path output, byte marker) throws IOException {
        Files.createDirectories(output.toAbsolutePath().normalize().getParent());
        try (ZipOutputStream zip = new ZipOutputStream(Files.newOutputStream(output))) {
            put(zip, "model.sdnb", new byte[] {'S', 'D', 'N', 'B', 1, 2, 3, marker});
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
