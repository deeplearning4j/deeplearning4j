/*
 * Copyright (c) Eclipse Deeplearning4j
 * SPDX-License-Identifier: Apache-2.0
 */
package org.nd4j.dsp.model;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class SdxQuantizationContractTest {
    @TempDir
    Path temporary;

    @Test
    void acceptsFailClosedPerChannelWeightOnlyContractForHexagon() throws Exception {
        SdxQuantizationContract contract =
                SdxQuantizationContract.parse(perChannelContract("FLOAT16", ""));

        assertEquals("sdx-graph", contract.provider());
        assertEquals("FLOAT16", contract.activationDtype());
        assertEquals(java.util.List.of("SM8650"), contract.targetSocs());
        assertTrue(contract.summaryJson().contains("\"allowFloatFallback\":false"));
        assertTrue(contract.summaryJson().contains("\"scheme\":\"int8-per-channel\""));
    }

    @Test
    void materializesCanonicalPerChannelProfilesForSupportedAndroidTargets()
            throws Exception {
        Map<SdxTargetProfile, String> targets = new LinkedHashMap<>();
        targets.put(SdxTargetProfile.ANDROID_ARM64_VULKAN, "Android_Vulkan_1_1");
        targets.put(SdxTargetProfile.ANDROID_ARM64_HEXAGON_HTP, "SM8650");
        targets.put(SdxTargetProfile.ANDROID_ARM64_GOOGLE_TENSOR_G5, "Tensor_G5");

        for (Map.Entry<SdxTargetProfile, String> entry : targets.entrySet()) {
            Path output = temporary.resolve(entry.getKey().id() + ".json");
            SdxQuantizationContract written =
                    SdxQuantizationContract.writeWeightInt8Profile(
                            output, entry.getKey(), entry.getValue());
            SdxQuantizationContract loaded = SdxQuantizationContract.load(output);

            assertEquals(written.summaryJson(), loaded.summaryJson());
            assertEquals(java.util.List.of(entry.getValue()), loaded.targetSocs());
            assertEquals(
                    entry.getKey() == SdxTargetProfile.ANDROID_ARM64_GOOGLE_TENSOR_G5
                            ? "litert-lm"
                            : "sdx-graph",
                    loaded.provider());
            assertTrue(Files.readString(output).contains(
                    "\"granularity\":\"per-channel\""));
        }
    }

    @Test
    void nnapiProfileMaterializationFailsWithoutCalibratedScales() {
        Path output = temporary.resolve("tensor-g3.json");

        IOException failure = assertThrows(
                IOException.class,
                () -> SdxQuantizationContract.writeWeightInt8Profile(
                        output,
                        SdxTargetProfile.ANDROID_ARM64_NNAPI_ACCELERATOR,
                        "Tensor_G3"));

        assertTrue(failure.getMessage().contains("calibrated activation/output"));
        assertTrue(Files.notExists(output));
    }

    @Test
    void acceptsCalibratedPerTensorNnapiContract() throws Exception {
        SdxQuantizationContract contract =
                SdxQuantizationContract.parse(validNnapiContract());

        assertDoesNotThrow(() -> contract.validateForCompilation(
                sourceIdentity(), SdxTargetProfile.ANDROID_ARM64_NNAPI_ACCELERATOR));
        assertEquals("INT8", contract.activationDtype());
        assertTrue(contract.summaryJson().contains("\"scheme\":\"int8-per-tensor\""));
    }

    @Test
    void acceptsSourceBoundPerOperatorQ4Calibration() throws Exception {
        String sourceSha = repeat('b', 64);
        SdxQuantizationContract contract = SdxQuantizationContract.parse(
                validQ4Contract(sourceSha, "decoder.q_proj", 0.03125, 0.0625));

        SdxQuantizationContract.OperatorCalibration calibration =
                contract.operatorCalibration("decoder.q_proj");
        assertEquals("ggml_qmatmul", calibration.opType());
        assertEquals(0.03125f, calibration.activationScale());
        assertEquals(0.0625f, calibration.outputScale());
        assertEquals(64, contract.calibrationSampleCount());
        assertEquals(repeat('a', 64), contract.calibrationDatasetSha256());
        assertEquals(java.util.List.of(
                        "sdx.nnapi.q4.calibration.v1", "64", repeat('a', 64),
                        "0.03125", "0.0625"),
                java.util.Arrays.asList(calibration.nnapiQ4SArguments(
                        contract.calibrationSampleCount(),
                        contract.calibrationDatasetSha256())));
        assertTrue(contract.summaryJson().contains("decoder.q_proj"));
        assertTrue(contract.summaryJson().contains("\"activationScale\":0.03125"));
        assertTrue(contract.summaryJson().contains("\"interiorQuantizationMax\":126"));
    }

    @Test
    void rejectsUnboundOrUnderSampledPerOperatorQ4Calibration() {
        String sourceSha = repeat('b', 64);
        String valid = validQ4Contract(sourceSha, "decoder.q_proj", 0.03125, 0.0625);

        IOException unbound = assertThrows(IOException.class,
                () -> SdxQuantizationContract.parse(valid.replace(
                        "\"sourceModelSha256\":\"" + sourceSha + "\",", "")));
        assertTrue(unbound.getMessage().contains("sourceModelSha256"));

        IOException underSampled = assertThrows(IOException.class,
                () -> SdxQuantizationContract.parse(
                        valid.replace("\"sampleCount\":64", "\"sampleCount\":31")));
        assertTrue(underSampled.getMessage().contains("at least 32"));

        IOException wrongType = assertThrows(IOException.class,
                () -> SdxQuantizationContract.parse(
                        valid.replace("\"opType\":\"ggml_qmatmul\"",
                                "\"opType\":\"matmul\"")));
        assertTrue(wrongType.getMessage().contains("opType"));

        IOException underflow = assertThrows(IOException.class,
                () -> SdxQuantizationContract.parse(
                        valid.replace("\"scale\":0.03125", "\"scale\":1e-1000")));
        assertTrue(underflow.getMessage().contains("finite and positive"));

        IOException overflow = assertThrows(IOException.class,
                () -> SdxQuantizationContract.parse(
                        valid.replace("\"scale\":0.03125", "\"scale\":3e38")));
        assertTrue(overflow.getMessage().contains("overflows its INT8 calibration envelope"));

        IOException missingInteriorMaximum = assertThrows(IOException.class,
                () -> SdxQuantizationContract.parse(
                        valid.replace(",\"interiorQuantizationMax\":126", "")));
        assertTrue(missingInteriorMaximum.getMessage().contains(
                "interiorQuantizationMax"));
    }

    @Test
    void rejectsPerChannelContractForNnapiButPreservesHexagonAndTensorG5Support()
            throws Exception {
        SdxQuantizationContract hexagon =
                SdxQuantizationContract.parse(perChannelContract("FLOAT16", ""));
        assertDoesNotThrow(() -> hexagon.validateForCompilation(
                sourceIdentity(), SdxTargetProfile.ANDROID_ARM64_HEXAGON_HTP));

        SdxQuantizationContract tensorG5 = SdxQuantizationContract.parse(
                perChannelContract("FLOAT16", "")
                        .replace("\"provider\":\"sdx-graph\"",
                                "\"provider\":\"litert-lm\"")
                        .replace("SM8650", "Tensor_G5"));
        assertDoesNotThrow(() -> tensorG5.validateForCompilation(
                sourceIdentity(), SdxTargetProfile.ANDROID_ARM64_GOOGLE_TENSOR_G5));

        SdxQuantizationContract perChannelTensorG3 = SdxQuantizationContract.parse(
                perChannelContract("FLOAT16", "").replace("SM8650", "Tensor_G3"));
        IOException failure = assertThrows(
                IOException.class,
                () -> perChannelTensorG3.validateForCompilation(
                        sourceIdentity(),
                        SdxTargetProfile.ANDROID_ARM64_NNAPI_ACCELERATOR));
        assertTrue(failure.getMessage().contains("per-tensor INT8 weights"));
    }

    @Test
    void nnapiFailsClosedWhenRequiredCalibratedScaleMetadataIsMissing() throws Exception {
        String valid = validNnapiContract();

        String missingActivationScale = valid.replace(
                "\"scaleDtype\":\"FLOAT32\",\"granularity\":\"per-tensor\","
                        + "\"scale\":0.03125,\"zeroPoint\":0,",
                "");
        assertNnapiRejected(missingActivationScale, "activation/output");

        String missingOutputScale = valid.replace(
                ",\"outputs\":{\"dtype\":\"INT8\","
                        + "\"scaleDtype\":\"FLOAT32\","
                        + "\"granularity\":\"per-tensor\","
                        + "\"scale\":0.0625,\"zeroPoint\":0}",
                "");
        assertNnapiRejected(missingOutputScale, "activation/output");
    }

    @Test
    void rejectsFloatFallbackAndDuplicateKeys() {
        String fallback = perChannelContract("FLOAT16", "")
                .replace("\"allowFloatFallback\":false",
                        "\"allowFloatFallback\":true");
        assertThrows(
                IOException.class,
                () -> SdxQuantizationContract.parse(fallback));

        String duplicate = perChannelContract("FLOAT16", "")
                .replace("\"formatVersion\":1",
                        "\"formatVersion\":1,\"formatVersion\":1");
        IOException failure = assertThrows(
                IOException.class,
                () -> SdxQuantizationContract.parse(duplicate));
        assertTrue(failure.getMessage().contains("duplicate object key"));
    }

    @Test
    void requiresBoundCalibrationForInt8Activations() {
        assertThrows(
                IOException.class,
                () -> SdxQuantizationContract.parse(
                        perChannelContract("INT8", "")));

        SdxQuantizationContract contract = assertDoesNotThrow(
                () -> SdxQuantizationContract.parse(
                        perChannelContract("INT8", calibration())));
        assertEquals("INT8", contract.activationDtype());
    }

    @Test
    void bindsNnapiContractToTargetAndSourceIdentity() throws Exception {
        String sourceSha = repeat('b', 64);
        SdxQuantizationContract contract = SdxQuantizationContract.parse(
                validNnapiContract().replace(
                        "\"weights\":",
                        "\"sourceModelSha256\":\"" + sourceSha + "\",\"weights\":"));
        SdxSourceIdentity wrong = sourceIdentity();

        IOException sourceFailure = assertThrows(
                IOException.class,
                () -> contract.validateForCompilation(
                        wrong, SdxTargetProfile.ANDROID_ARM64_NNAPI_ACCELERATOR));
        assertTrue(sourceFailure.getMessage().contains("sourceModelSha256"));

        SdxQuantizationContract wrongSoc = SdxQuantizationContract.parse(
                validNnapiContract().replace("Tensor_G3", "Tensor_G5"));
        IOException targetFailure = assertThrows(
                IOException.class,
                () -> wrongSoc.validateForCompilation(
                        wrong, SdxTargetProfile.ANDROID_ARM64_NNAPI_ACCELERATOR));
        assertTrue(targetFailure.getMessage().contains("Tensor_G3"));
    }

    private void assertNnapiRejected(String json, String message) throws Exception {
        SdxQuantizationContract contract = SdxQuantizationContract.parse(json);
        IOException failure = assertThrows(
                IOException.class,
                () -> contract.validateForCompilation(
                        sourceIdentity(),
                        SdxTargetProfile.ANDROID_ARM64_NNAPI_ACCELERATOR));
        assertTrue(failure.getMessage().contains(message));
    }

    private SdxSourceIdentity sourceIdentity() throws IOException {
        return SdxSourceIdentity.identify(Files.write(
                temporary.resolve("model-" + System.nanoTime() + ".sdnb"),
                new byte[] {'S', 'D', 'N', 'B', 1, 2, 3, 4}));
    }

    private static String validNnapiContract() {
        return "{"
                + "\"formatVersion\":1,"
                + "\"scheme\":\"int8-per-tensor\","
                + "\"provider\":\"sdx-graph\","
                + "\"targetSocs\":[\"Tensor_G3\"],"
                + "\"deviceOnly\":true,"
                + "\"allowFloatFallback\":false,"
                + "\"requireVendorAot\":true,"
                + "\"weights\":{"
                + "\"dtype\":\"INT8\",\"scaleDtype\":\"FLOAT32\","
                + "\"granularity\":\"per-tensor\",\"scale\":0.015625,"
                + "\"symmetric\":true,\"zeroPoint\":0},"
                + "\"activations\":{"
                + "\"dtype\":\"INT8\","
                + "\"scaleDtype\":\"FLOAT32\",\"granularity\":\"per-tensor\","
                + "\"scale\":0.03125,\"zeroPoint\":0,"
                + calibration().substring(1)
                + "},"
                + "\"outputs\":{\"dtype\":\"INT8\","
                + "\"scaleDtype\":\"FLOAT32\",\"granularity\":\"per-tensor\","
                + "\"scale\":0.0625,\"zeroPoint\":0},"
                + "\"excludedOps\":[]"
                + "}";
    }

    private static String validQ4Contract(
            String sourceSha, String opName, double activationScale, double outputScale) {
        return "{"
                + "\"formatVersion\":1,"
                + "\"scheme\":\"q4-k-per-op-int8-boundaries\","
                + "\"provider\":\"sdx-graph\","
                + "\"targetSocs\":[\"Tensor_G3\"],"
                + "\"deviceOnly\":true,"
                + "\"allowFloatFallback\":false,"
                + "\"requireVendorAot\":true,"
                + "\"sourceModelSha256\":\"" + sourceSha + "\","
                + "\"weights\":{\"dtype\":\"INT8\",\"scaleDtype\":\"FLOAT32\","
                + "\"granularity\":\"per-tensor\",\"symmetric\":true,\"zeroPoint\":0},"
                + "\"activations\":{\"dtype\":\"INT8\",\"calibration\":{"
                + "\"method\":\"minmax\",\"sampleCount\":64,"
                + "\"datasetSha256\":\"" + repeat('a', 64) + "\"}},"
                + "\"operatorCalibrations\":{\"" + opName + "\":{"
                + "\"opType\":\"ggml_qmatmul\","
                + "\"activations\":{\"scaleDtype\":\"FLOAT32\","
                + "\"granularity\":\"per-tensor\",\"scale\":" + activationScale
                + ",\"zeroPoint\":0},"
                + "\"outputs\":{\"scaleDtype\":\"FLOAT32\","
                + "\"granularity\":\"per-tensor\",\"scale\":" + outputScale
                + ",\"zeroPoint\":0,\"interiorQuantizationMax\":126}}},"
                + "\"excludedOps\":[]"
                + "}";
    }

    private static String perChannelContract(String activationDtype, String calibration) {
        return "{"
                + "\"formatVersion\":1,"
                + "\"scheme\":\"int8-per-channel\","
                + "\"provider\":\"sdx-graph\","
                + "\"targetSocs\":[\"SM8650\"],"
                + "\"deviceOnly\":true,"
                + "\"allowFloatFallback\":false,"
                + "\"requireVendorAot\":true,"
                + "\"weights\":{"
                + "\"dtype\":\"INT8\","
                + "\"scaleDtype\":\"FLOAT32\","
                + "\"granularity\":\"per-channel\","
                + "\"channelAxis\":0,"
                + "\"symmetric\":true,"
                + "\"zeroPoint\":0"
                + "},"
                + "\"activations\":{"
                + "\"dtype\":\"" + activationDtype + "\""
                + calibration
                + "},"
                + "\"excludedOps\":[]"
                + "}";
    }

    private static String calibration() {
        return ",\"calibration\":{"
                + "\"method\":\"percentile\","
                + "\"percentile\":99.9,"
                + "\"sampleCount\":128,"
                + "\"datasetSha256\":\"" + repeat('a', 64) + "\"}";
    }

    private static String repeat(char value, int count) {
        char[] result = new char[count];
        java.util.Arrays.fill(result, value);
        return new String(result);
    }
}
