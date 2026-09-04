/*
 * Copyright (c) Eclipse Deeplearning4j
 * SPDX-License-Identifier: Apache-2.0
 */
package org.nd4j.dsp.model;

import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.autodiff.samediff.serde.SDZSerializer;
import org.nd4j.ggml.GGMLImportException;
import org.nd4j.ggml.format.GGMLDataType;
import org.nd4j.ggml.format.GGMLTensorInfo;
import org.nd4j.ggml.format.GGUFReader;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.shade.jackson.databind.JsonNode;
import org.nd4j.shade.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Real raw-GGUF integration gate for the compiler-owned Tensor G3 calibration lifecycle. */
class SdxTensorG3AutomaticCalibrationIntegrationTest {
    private static final String MODEL_PROPERTY = "sdx.tensorG3.calibration.gguf";
    private static final String TOKENIZER_PROPERTY = "sdx.tensorG3.calibration.tokenizer";
    private static final String CACHE_PROPERTY = "sdx.tensorG3.calibration.cache";

    @Test
    void rawPreparationPacksQ4CalibratesCompilesAndReusesExactTarget() throws Exception {
        Path model = configuredPath(MODEL_PROPERTY);
        Path tokenizer = configuredPath(TOKENIZER_PROPERTY);
        String cacheValue = System.getProperty(CACHE_PROPERTY);
        Assumptions.assumeTrue(model != null && Files.isRegularFile(model),
                "Set -D" + MODEL_PROPERTY + " to a real BF16 or quantized GGUF");
        Assumptions.assumeTrue(tokenizer != null && Files.isRegularFile(tokenizer),
                "Set -D" + TOKENIZER_PROPERTY + " to its tokenizer.json");
        Assumptions.assumeTrue(cacheValue != null && !cacheValue.isBlank(),
                "Set -D" + CACHE_PROPERTY + " to an isolated cache directory");
        Path cacheRoot = Path.of(cacheValue).toAbsolutePath().normalize();
        Files.createDirectories(cacheRoot);

        String options = "{"
                + "\"graphImportAbi\":\"ggml-fixed-plan-rolling-context-q4-linears-v9\","
                + "\"conversionMode\":\"RUNTIME_QUANTIZED_MATMUL\","
                + "\"requantizeType\":\"Q4_K\","
                + "\"embeddingDataType\":\"HALF\","
                + "\"logitsMode\":\"LAST_POSITION_ONLY\","
                + "\"kvQuantFormat\":1,"
                + "\"tensorBatchSize\":4,"
                + "\"useMemoryMapping\":true,"
                + "\"diagnosticMode\":\"off\"} ";

        JsonNode cold = new ObjectMapper().readTree(prepare(
                model, tokenizer, cacheRoot, options));
        assertFalse(cold.path("cacheHit").asBoolean(true));
        Path canonical = Path.of(cold.path("canonicalSdzPath").asText())
                .toAbsolutePath().normalize();
        assertTrue(Files.isRegularFile(canonical));
        Path optimizedSource = Path.of(cold.path("optimizedSourcePath").asText())
                .toAbsolutePath().normalize();
        assertTrue(Files.isRegularFile(optimizedSource));
        try (GGUFReader sourceReader = new GGUFReader(model.toFile());
             GGUFReader optimizedReader = new GGUFReader(optimizedSource.toFile())) {
            GGMLTensorInfo sourceEmbedding = tokenEmbedding(sourceReader);
            GGMLTensorInfo optimizedEmbedding = tokenEmbedding(optimizedReader);
            assertEquals(sourceEmbedding.getDataType(), optimizedEmbedding.getDataType(),
                    "Tensor G3 requantization must preserve the authored token embedding dtype");
            assertTrue(optimizedReader.getTensorInfos().stream()
                            .anyMatch(tensor -> tensor.getNumDimensions() == 2
                                    && tensor.getDataType() == GGMLDataType.GGML_TYPE_Q4_K),
                    "Tensor G3 optimized source must contain packed Q4_K linear weights");
        }

        SdxModelCache cache = new SdxModelCache(cacheRoot);
        SdxCompiledModel compiled = cache.resolveVerified(
                canonical, SdxTargetProfile.ANDROID_ARM64_NNAPI_ACCELERATOR);
        assertEquals(SdxTensorG3NnapiCompiler.COMPILER_ID, compiled.compilerId());
        assertEquals(SdxTensorG3NnapiCompiler.COMPILER_VERSION,
                compiled.compilerVersion());
        Path quantization = compiled.quantizationConfigPath().orElseThrow();
        SdxQuantizationContract contract = SdxQuantizationContract.load(quantization);
        contract.validateForCompilation(
                compiled.sourceIdentity(),
                SdxTargetProfile.ANDROID_ARM64_NNAPI_ACCELERATOR,
                SdxTensorG3NnapiCompiler.TARGET_SOC);
        assertTrue(contract.isTensorG3Q4PerOperator());
        assertEquals(SdxTensorG3Q4Calibration.REQUIRED_SAMPLE_COUNT,
                contract.calibrationSampleCount());
        assertFalse(contract.operatorCalibrations().isEmpty());

        Path derived = compiled.cacheEntry().resolve("bundle/graph/model.sdz");
        assertTrue(Files.isRegularFile(derived));
        int q4Count = 0;
        try (SameDiff graph = SDZSerializer.load(derived.toFile(), false)) {
            for (SameDiffOp op : graph.getOps().values()) {
                if (op.getOp() == null
                        || !"ggml_qmatmul".equals(op.getOp().opName())) {
                    continue;
                }
                DynamicCustomOp qmatmul = (DynamicCustomOp) op.getOp();
                long[] integerArgs = qmatmul.iArgs();
                assertTrue(integerArgs != null && integerArgs.length > 0);
                String[] strings = qmatmul.sArgs();
                if (integerArgs[0] == 8L) {
                    q4Count++;
                    assertTrue(strings != null && strings.length == 5,
                            "Q4 operation lacks finalized calibration: " + op.getName());
                    assertTrue(contract.operatorCalibrations().containsKey(op.getName()));
                } else {
                    assertTrue(strings == null || strings.length == 0,
                            "Non-Q4 packed operation was annotated: " + op.getName());
                }
            }
        }
        assertTrue(q4Count > 0,
                "Tensor G3 canonical SDZ must contain Q4_K ggml_qmatmul operations");
        assertEquals(contract.operatorCalibrations().size(), q4Count);

        JsonNode warm = new ObjectMapper().readTree(prepare(
                model, tokenizer, cacheRoot, options));
        assertTrue(warm.path("cacheHit").asBoolean(false));
        assertEquals(cold.path("compileKey").asText(), warm.path("compileKey").asText());
        assertEquals(cold.path("modelPath").asText(), warm.path("modelPath").asText());
    }

    private static String prepare(
            Path model, Path tokenizer, Path cache, String options) throws Exception {
        Class<?> type = Class.forName(
                "org.eclipse.deeplearning4j.sdx.aot.SdxGgufModelPreparer");
        Method method = type.getDeclaredMethod(
                "prepare", String.class, String.class, String.class,
                String.class, String.class);
        method.setAccessible(true);
        try {
            return (String) method.invoke(
                    null,
                    model.toString(),
                    tokenizer.toString(),
                    SdxTargetProfile.ANDROID_ARM64_NNAPI_ACCELERATOR.id(),
                    cache.toString(),
                    options);
        } catch (InvocationTargetException wrapped) {
            Throwable cause = wrapped.getCause();
            if (cause instanceof Exception) {
                throw (Exception) cause;
            }
            throw wrapped;
        }
    }

    private static Path configuredPath(String property) {
        String value = System.getProperty(property);
        return value == null || value.isBlank()
                ? null : Path.of(value).toAbsolutePath().normalize();
    }

    private static GGMLTensorInfo tokenEmbedding(GGUFReader reader)
            throws IOException, GGMLImportException {
        return reader.getTensorInfos().stream()
                .filter(tensor -> tensor.getName().toLowerCase(Locale.ROOT)
                        .contains("token_embd"))
                .findFirst()
                .orElseThrow(() -> new IllegalStateException(
                        "GGUF does not contain a token embedding tensor"));
    }
}
