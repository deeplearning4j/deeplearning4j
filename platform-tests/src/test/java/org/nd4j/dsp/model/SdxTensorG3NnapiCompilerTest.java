/*
 * Copyright (c) Eclipse Deeplearning4j
 * SPDX-License-Identifier: Apache-2.0
 */
package org.nd4j.dsp.model;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;
import java.util.zip.ZipOutputStream;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.autodiff.samediff.serde.SDZSerializer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.factory.Nd4j;

class SdxTensorG3NnapiCompilerTest {
    private static final Map<String, byte[]> BUNDLE_ENTRIES = bundleEntries();

    @TempDir
    Path temporary;

    @Test
    void rewritesNativeLayoutQuantizedMatmulPrunesFloatWeightsAndReusesCache()
            throws Exception {
        Path source = completeSdz(temporary.resolve("canonical.sdz"), GraphKind.CONSTANT_WEIGHTS);
        Path quantization = writeContract(temporary.resolve("quantization.json"), validContract());
        SdxModelCompiler compiler =
                new SdxModelCompiler(new SdxModelCache(temporary.resolve("cache")));
        SdxModelCompiler.CompileOptions options = SdxModelCompiler.CompileOptions.builder()
                .quantizationConfig(quantization)
                .targetSoc("Tensor_G3")
                .build();

        SdxCompiledModel first = compiler.compile(
                source,
                SdxTargetProfile.ANDROID_ARM64_NNAPI_ACCELERATOR,
                new SdxTensorG3NnapiCompiler(),
                options);
        Path derived = first.cacheEntry().resolve("bundle/graph/model.sdz");
        long firstModified = Files.getLastModifiedTime(derived).toMillis();
        SdxCompiledModel second = compiler.compile(
                source,
                SdxTargetProfile.ANDROID_ARM64_NNAPI_ACCELERATOR,
                new SdxTensorG3NnapiCompiler(),
                options);

        assertEquals(first.compileKey(), second.compileKey());
        assertEquals(first.cacheEntry(), second.cacheEntry());
        assertEquals(firstModified, Files.getLastModifiedTime(derived).toMillis());
        assertNotEquals(
                SdxSourceIdentity.identify(source).sha256(),
                SdxSourceIdentity.identify(derived).sha256());
        for (Map.Entry<String, byte[]> expected : BUNDLE_ENTRIES.entrySet()) {
            assertArrayEquals(expected.getValue(), zipEntry(derived, expected.getKey()));
        }
        assertModelEntriesStored(derived);

        try (SameDiff rewritten = SDZSerializer.load(derived.toFile(), false)) {
            SameDiffOp quantized = rewritten.getOps().values().stream()
                    .filter(op -> "quantized_matmul".equals(op.getOp().opName()))
                    .findFirst()
                    .orElseThrow();
            assertEquals(5, quantized.getInputsToOp().size());
            assertArrayEquals(
                    new long[] {1L},
                    ((DynamicCustomOp) quantized.getOp()).iArgs());
            INDArray packedWeights =
                    rewritten.getVariable(quantized.getInputsToOp().get(1)).getArr();
            assertEquals(
                    DataType.INT8,
                    packedWeights.dataType());
            assertArrayEquals(new long[] {3L, 2L}, packedWeights.shape());
            assertEquals(16, packedWeights.getInt(0, 0));
            assertEquals(64, packedWeights.getInt(0, 1));
            assertEquals(-32, packedWeights.getInt(1, 0));
            assertEquals(8, packedWeights.getInt(1, 1));
            assertEquals(48, packedWeights.getInt(2, 0));
            assertEquals(-16, packedWeights.getInt(2, 1));
            assertFalse(rewritten.hasVariable("weights"));
            for (int i = 2; i < 5; i++) {
                SDVariable scale = rewritten.getVariable(quantized.getInputsToOp().get(i));
                assertEquals(DataType.FLOAT, scale.dataType());
                assertEquals(1L, scale.getArr().length());
            }
            assertTrue(rewritten.getOps().values().stream()
                    .noneMatch(op -> "mmul".equals(op.getOp().opName())
                            || "matmul".equals(op.getOp().opName())));
        }
    }

    @Test
    void rejectsDynamicWeights() throws Exception {
        IOException failure = compileFailure(GraphKind.DYNAMIC_WEIGHTS, validContract());
        assertTrue(failure.getMessage().contains("dynamic/non-constant weights"));
    }

    @Test
    void rejectsUnsupportedBatchedWeightLayout() throws Exception {
        IOException failure = compileFailure(GraphKind.RANK3_WEIGHTS, validContract());
        assertTrue(failure.getMessage().contains("dense rank-2"));
    }

    @Test
    void rejectsGraphWithoutEligibleRewrite() throws Exception {
        IOException failure = compileFailure(GraphKind.NO_MATMUL, validContract());
        assertTrue(failure.getMessage().contains("no eligible"));
    }

    @Test
    void rejectsMissingCalibrationAndPerChannelMetadataBeforeGraphRewrite() throws Exception {
        String missingCalibration = validContract().replace(
                ",\"calibration\":{\"method\":\"minmax\",\"sampleCount\":32,"
                        + "\"datasetSha256\":\""
                        + "a".repeat(64) + "\"}",
                "");
        IOException calibrationFailure =
                compileFailure(GraphKind.CONSTANT_WEIGHTS, missingCalibration);
        assertTrue(calibrationFailure.getMessage().contains("calibration"));

        String perChannel = validContract()
                .replace("\"scheme\":\"int8-per-tensor\"",
                        "\"scheme\":\"int8-per-channel\"")
                .replace("\"granularity\":\"per-tensor\",\"scale\":0.015625",
                        "\"granularity\":\"per-channel\",\"channelAxis\":1");
        IOException perChannelFailure =
                compileFailure(GraphKind.CONSTANT_WEIGHTS, perChannel);
        assertTrue(perChannelFailure.getMessage().contains("per-tensor INT8 weights"));
    }

    private IOException compileFailure(GraphKind kind, String contract) throws Exception {
        Path source = completeSdz(
                temporary.resolve(kind.name().toLowerCase() + "-" + System.nanoTime() + ".sdz"),
                kind);
        Path quantization = writeContract(
                temporary.resolve("contract-" + System.nanoTime() + ".json"), contract);
        SdxModelCompiler compiler =
                new SdxModelCompiler(new SdxModelCache(
                        temporary.resolve("cache-" + System.nanoTime())));
        return assertThrows(IOException.class, () -> compiler.compile(
                source,
                SdxTargetProfile.ANDROID_ARM64_NNAPI_ACCELERATOR,
                new SdxTensorG3NnapiCompiler(),
                SdxModelCompiler.CompileOptions.builder()
                        .quantizationConfig(quantization)
                        .targetSoc("Tensor_G3")
                        .build()));
    }

    private Path completeSdz(Path output, GraphKind kind) throws Exception {
        Path graphOnly = temporary.resolve("graph-" + System.nanoTime() + ".sdz");
        try (SameDiff graph = SameDiff.create()) {
            switch (kind) {
                case CONSTANT_WEIGHTS: {
                    SDVariable activation =
                            graph.placeHolder("activation", DataType.FLOAT, 1, 2);
                    SDVariable weights = graph.constant(
                            "weights", Nd4j.createFromArray(new float[][] {
                                    {0.25f, -0.5f, 0.75f},
                                    {1.0f, 0.125f, -0.25f}
                            }));
                    graph.mmul("projection", activation, weights);
                    break;
                }
                case DYNAMIC_WEIGHTS: {
                    SDVariable activation =
                            graph.placeHolder("activation", DataType.FLOAT, 1, 2);
                    SDVariable weights =
                            graph.placeHolder("weights", DataType.FLOAT, 2, 2);
                    graph.mmul("projection", activation, weights);
                    break;
                }
                case RANK3_WEIGHTS: {
                    SDVariable activation =
                            graph.placeHolder("activation", DataType.FLOAT, 1, 1, 2);
                    SDVariable weights = graph.constant(
                            "weights", Nd4j.createFromArray(new float[][][] {{
                                    {0.25f, -0.5f}, {1.0f, 0.125f}
                            }}));
                    graph.mmul("projection", activation, weights);
                    break;
                }
                case NO_MATMUL:
                    graph.constant("constant", Nd4j.scalar(1.0f)).add(1.0);
                    break;
                default:
                    throw new AssertionError(kind);
            }
            SDZSerializer.save(graph, graphOnly.toFile(), false, null);
        }
        mergeBundle(graphOnly, output);
        return output;
    }

    private static void mergeBundle(Path graphOnly, Path output) throws IOException {
        try (ZipFile source = new ZipFile(graphOnly.toFile());
                OutputStream stream = Files.newOutputStream(output);
                ZipOutputStream zip = new ZipOutputStream(stream)) {
            java.util.Enumeration<? extends ZipEntry> entries = source.entries();
            while (entries.hasMoreElements()) {
                ZipEntry input = entries.nextElement();
                ZipEntry copied = new ZipEntry(input.getName());
                copied.setTime(0L);
                zip.putNextEntry(copied);
                try (InputStream bytes = source.getInputStream(input)) {
                    bytes.transferTo(zip);
                }
                zip.closeEntry();
            }
            for (Map.Entry<String, byte[]> extra : BUNDLE_ENTRIES.entrySet()) {
                ZipEntry entry = new ZipEntry(extra.getKey());
                entry.setTime(0L);
                zip.putNextEntry(entry);
                zip.write(extra.getValue());
                zip.closeEntry();
            }
        }
    }

    private static void assertModelEntriesStored(Path archive) throws IOException {
        boolean foundModelEntry = false;
        try (ZipFile zip = new ZipFile(archive.toFile())) {
            java.util.Enumeration<? extends ZipEntry> entries = zip.entries();
            while (entries.hasMoreElements()) {
                ZipEntry entry = entries.nextElement();
                if (entry.isDirectory()) {
                    continue;
                }
                try (InputStream input = zip.getInputStream(entry)) {
                    byte[] magic = input.readNBytes(4);
                    if (magic.length == 4
                            && magic[0] == 'S'
                            && magic[1] == 'D'
                            && magic[2] == 'N'
                            && magic[3] == 'B') {
                        foundModelEntry = true;
                        assertEquals(ZipEntry.STORED, entry.getMethod(), entry.getName());
                    }
                }
            }
        }
        assertTrue(foundModelEntry, "Expected at least one SDNB model entry");
    }

    private static byte[] zipEntry(Path archive, String name) throws IOException {
        try (ZipFile zip = new ZipFile(archive.toFile())) {
            ZipEntry entry = zip.getEntry(name);
            if (entry == null) {
                throw new IOException("Missing ZIP entry " + name);
            }
            try (InputStream input = zip.getInputStream(entry)) {
                return input.readAllBytes();
            }
        }
    }

    private static Path writeContract(Path output, String contract) throws IOException {
        return Files.writeString(output, contract, StandardCharsets.UTF_8);
    }

    private static String validContract() {
        return "{"
                + "\"formatVersion\":1,"
                + "\"scheme\":\"int8-per-tensor\","
                + "\"provider\":\"sdx-graph\","
                + "\"targetSocs\":[\"Tensor_G3\"],"
                + "\"deviceOnly\":true,"
                + "\"allowFloatFallback\":false,"
                + "\"requireVendorAot\":true,"
                + "\"weights\":{\"dtype\":\"INT8\","
                + "\"scaleDtype\":\"FLOAT32\","
                + "\"granularity\":\"per-tensor\",\"scale\":0.015625,"
                + "\"symmetric\":true,\"zeroPoint\":0},"
                + "\"activations\":{\"dtype\":\"INT8\","
                + "\"scaleDtype\":\"FLOAT32\","
                + "\"granularity\":\"per-tensor\",\"scale\":0.03125,"
                + "\"zeroPoint\":0,\"calibration\":{\"method\":\"minmax\","
                + "\"sampleCount\":32,\"datasetSha256\":\"" + "a".repeat(64) + "\"}},"
                + "\"outputs\":{\"dtype\":\"INT8\","
                + "\"scaleDtype\":\"FLOAT32\","
                + "\"granularity\":\"per-tensor\",\"scale\":0.0625,"
                + "\"zeroPoint\":0},"
                + "\"excludedOps\":[]"
                + "}";
    }

    private static Map<String, byte[]> bundleEntries() {
        Map<String, byte[]> entries = new LinkedHashMap<>();
        entries.put("tokenizer.json", "{\"version\":1}\n".getBytes(StandardCharsets.UTF_8));
        entries.put("config.json", "{\"model_type\":\"test\"}\n".getBytes(StandardCharsets.UTF_8));
        entries.put("chat_template.json", "{\"template\":\"{{ messages }}\"}\n"
                .getBytes(StandardCharsets.UTF_8));
        entries.put("generation_config.json", "{\"max_new_tokens\":16}\n"
                .getBytes(StandardCharsets.UTF_8));
        return entries;
    }

    private enum GraphKind {
        CONSTANT_WEIGHTS,
        DYNAMIC_WEIGHTS,
        RANK3_WEIGHTS,
        NO_MATMUL
    }
}
