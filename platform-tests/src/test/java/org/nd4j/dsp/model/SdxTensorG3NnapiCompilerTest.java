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
    void annotatesQ4MatmulWithFinalizedPerOperatorCalibration() throws Exception {
        Path source = completeSdz(temporary.resolve("canonical-q4.sdz"), GraphKind.Q4_WEIGHTS);
        String sourceSha = SdxSourceIdentity.identify(source).sha256();
        Path quantization = writeContract(
                temporary.resolve("quantization-q4.json"),
                validQ4Contract(sourceSha, "projection_q4"));
        SdxModelCompiler compiler =
                new SdxModelCompiler(new SdxModelCache(temporary.resolve("cache-q4")));

        SdxCompiledModel compiled = compiler.compile(
                source,
                SdxTargetProfile.ANDROID_ARM64_NNAPI_ACCELERATOR,
                new SdxTensorG3NnapiCompiler(),
                SdxModelCompiler.CompileOptions.builder()
                        .quantizationConfig(quantization)
                        .targetSoc("Tensor_G3")
                        .build());

        Path derived = compiled.cacheEntry().resolve("bundle/graph/model.sdz");
        try (SameDiff rewritten = SDZSerializer.load(derived.toFile(), false)) {
            SameDiffOp q4 = rewritten.getOps().get("projection_q4");
            assertTrue(q4 != null);
            assertEquals("ggml_qmatmul", q4.getOp().opName());
            assertEquals(2, q4.getInputsToOp().size());
            assertArrayEquals(new long[] {8L, 2L, 256L, 0L},
                    ((DynamicCustomOp) q4.getOp()).iArgs());
            assertArrayEquals(new String[] {
                            "sdx.nnapi.q4.calibration.v1", "64", "a".repeat(64),
                            "0.03125", "0.0625"
                    }, ((DynamicCustomOp) q4.getOp()).sArgs());
            SDVariable packedWeight = rewritten.getVariable(q4.getInputsToOp().get(1));
            assertEquals(org.nd4j.autodiff.samediff.VariableType.VARIABLE,
                    packedWeight.getVariableType());
            assertEquals(DataType.INT8, packedWeight.dataType());
            assertArrayEquals(new long[] {288L}, packedWeight.getArr().shape());
            assertEquals(0L, packedWeight.getArr().sumNumber().longValue());
        }
    }

    @Test
    void rejectsMissingAndStaleQ4CalibrationEntries() throws Exception {
        Path source = completeSdz(temporary.resolve("missing-q4.sdz"), GraphKind.Q4_WEIGHTS);
        String sourceSha = SdxSourceIdentity.identify(source).sha256();

        IOException missing = compileFailure(
                source, validQ4Contract(sourceSha, "another_q4"));
        assertTrue(missing.getMessage().contains("no source-bound per-op calibration"));

        IOException stale = compileFailure(
                source, validQ4Contract(sourceSha, "projection_q4")
                        .replace("\"operatorCalibrations\":{",
                                "\"operatorCalibrations\":{\"stale_q4\":"
                                        + q4CalibrationEntry() + ","));
        assertTrue(stale.getMessage().contains("stale or non-Q4 op"));
    }

    @Test
    void annotatesOnlyQ4AndPreservesDenseQ6AndQ8Operations() throws Exception {
        Path source = completeSdz(temporary.resolve("mixed.sdz"), GraphKind.MIXED_WEIGHTS);
        String sourceSha = SdxSourceIdentity.identify(source).sha256();
        Path quantization = writeContract(
                temporary.resolve("mixed-quantization.json"),
                validQ4Contract(sourceSha, "projection_q4"));
        SdxModelCompiler compiler = new SdxModelCompiler(
                new SdxModelCache(temporary.resolve("mixed-cache")));

        SdxCompiledModel compiled = compiler.compile(
                source,
                SdxTargetProfile.ANDROID_ARM64_NNAPI_ACCELERATOR,
                new SdxTensorG3NnapiCompiler(),
                SdxModelCompiler.CompileOptions.builder()
                        .quantizationConfig(quantization)
                        .targetSoc("Tensor_G3")
                        .build());

        try (SameDiff rewritten = SDZSerializer.load(
                compiled.cacheEntry().resolve("bundle/graph/model.sdz").toFile(), false)) {
            SameDiffOp q4 = rewritten.getOps().get("projection_q4");
            SameDiffOp q6 = rewritten.getOps().get("projection_q6");
            SameDiffOp q8 = rewritten.getOps().get("projection_q8");
            SameDiffOp dense = rewritten.getOps().values().stream()
                    .filter(candidate -> candidate.getOutputsOfOp() != null
                            && candidate.getOutputsOfOp().contains("dense_projection"))
                    .findFirst()
                    .orElseThrow();
            assertArrayEquals(new String[] {
                            "sdx.nnapi.q4.calibration.v1", "64", "a".repeat(64),
                            "0.03125", "0.0625"
                    }, ((DynamicCustomOp) q4.getOp()).sArgs());
            assertArrayEquals(new long[] {10L, 2L, 256L, 0L},
                    ((DynamicCustomOp) q6.getOp()).iArgs());
            assertArrayEquals(new long[] {4L, 2L, 256L, 0L},
                    ((DynamicCustomOp) q8.getOp()).iArgs());
            String[] q6Strings = ((DynamicCustomOp) q6.getOp()).sArgs();
            String[] q8Strings = ((DynamicCustomOp) q8.getOp()).sArgs();
            assertTrue(q6Strings == null || q6Strings.length == 0);
            assertTrue(q8Strings == null || q8Strings.length == 0);
            assertTrue("mmul".equals(dense.getOp().opName())
                    || "matmul".equals(dense.getOp().opName()));
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

    private IOException compileFailure(Path source, String contract) throws Exception {
        Path quantization = writeContract(
                temporary.resolve("contract-" + System.nanoTime() + ".json"), contract);
        SdxModelCompiler compiler = new SdxModelCompiler(new SdxModelCache(
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
                case Q4_WEIGHTS: {
                    addQ4Projection(graph);
                    break;
                }
                case MIXED_WEIGHTS: {
                    SDVariable denseActivation =
                            graph.placeHolder("dense_activation", DataType.FLOAT, 1, 2);
                    SDVariable denseWeights = graph.constant(
                            "dense_weights", Nd4j.createFromArray(new float[][] {
                                    {0.25f, -0.5f}, {1.0f, 0.125f}
                            }));
                    graph.mmul("dense_projection", denseActivation, denseWeights);
                    addQ4Projection(graph);
                    addPackedProjection(graph, "projection_q6", "activation_q6",
                            "weights_q6", 10L);
                    addPackedProjection(graph, "projection_q8", "activation_q8",
                            "weights_q8", 4L);
                    break;
                }
                default:
                    throw new AssertionError(kind);
            }
            SDZSerializer.save(graph, graphOnly.toFile(), false, null);
        }
        mergeBundle(graphOnly, output);
        return output;
    }

    private static void addQ4Projection(SameDiff graph) {
        addPackedProjection(graph, "projection_q4", "activation", "weights_q4", 8L);
    }

    private static void addPackedProjection(
            SameDiff graph,
            String opName,
            String activationName,
            String weightName,
            long quantizationType) {
        SDVariable activation =
                graph.placeHolder(activationName, DataType.FLOAT, 1, 256);
        long packedBytes;
        if (quantizationType == 8L) {
            packedBytes = 288L; // Q4_K: 144 bytes per 256-element row
        } else if (quantizationType == 10L) {
            packedBytes = 420L; // Q6_K: 210 bytes per 256-element row
        } else if (quantizationType == 4L) {
            packedBytes = 544L; // Q8_0: 8 * 34 bytes per row
        } else {
            throw new IllegalArgumentException("unsupported packed test type "
                    + quantizationType);
        }
        SDVariable weights = graph.var(
                weightName, Nd4j.zeros(DataType.INT8, packedBytes));
        DynamicCustomOp q4 = new DynamicCustomOp() {
            @Override
            public String opName() {
                return "ggml_qmatmul";
            }

            @Override
            public java.util.List<DataType> calculateOutputDataTypes(
                    java.util.List<DataType> inputDataTypes) {
                return java.util.Collections.singletonList(DataType.FLOAT);
            }
        };
        q4.setSameDiff(graph);
        q4.setOwnName(opName);
        q4.addIArgument(quantizationType, 2L, 256L, 0L);
        graph.addArgsFor(new String[] {activation.name(), weights.name()}, q4);
        q4.outputVariables(opName);
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

    private static String validQ4Contract(String sourceSha, String opName) {
        return "{"
                + "\"formatVersion\":1,"
                + "\"scheme\":\"q4-k-per-op-int8-boundaries\","
                + "\"provider\":\"sdx-graph\","
                + "\"targetSocs\":[\"Tensor_G3\"],"
                + "\"deviceOnly\":true,\"allowFloatFallback\":false,"
                + "\"requireVendorAot\":true,"
                + "\"sourceModelSha256\":\"" + sourceSha + "\","
                + "\"weights\":{\"dtype\":\"INT8\",\"scaleDtype\":\"FLOAT32\","
                + "\"granularity\":\"per-tensor\",\"symmetric\":true,\"zeroPoint\":0},"
                + "\"activations\":{\"dtype\":\"INT8\",\"calibration\":{"
                + "\"method\":\"minmax\",\"sampleCount\":64,"
                + "\"datasetSha256\":\"" + "a".repeat(64) + "\"}},"
                + "\"operatorCalibrations\":{\"" + opName + "\":"
                + q4CalibrationEntry() + "},\"excludedOps\":[]}";
    }

    private static String q4CalibrationEntry() {
        return "{\"opType\":\"ggml_qmatmul\","
                + "\"activations\":{\"scaleDtype\":\"FLOAT32\","
                + "\"granularity\":\"per-tensor\",\"scale\":0.03125,\"zeroPoint\":0},"
                + "\"outputs\":{\"scaleDtype\":\"FLOAT32\","
                + "\"granularity\":\"per-tensor\",\"scale\":0.0625,\"zeroPoint\":0,"
                + "\"interiorQuantizationMax\":126}}";
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
        NO_MATMUL,
        Q4_WEIGHTS,
        MIXED_WEIGHTS
    }
}
