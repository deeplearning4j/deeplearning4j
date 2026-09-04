/*
 * Copyright (c) Eclipse Deeplearning4j
 * SPDX-License-Identifier: Apache-2.0
 */
package org.nd4j.dsp.model;

import org.nd4j.autodiff.listeners.Listener;
import org.nd4j.autodiff.listeners.Operation;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.factory.Nd4j;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class SdxTensorG3Q4CalibrationTest {

    @TempDir
    Path temporary;

    @Test
    void collectsThirtyTwoCompleteSamplesAndBuildsGuardedPowerOfTwoScales()
            throws Exception {
        try (SameDiff graph = q4Graph(8L)) {
            SdxTensorG3Q4Calibration.CalibrationCollector collector =
                    SdxTensorG3Q4Calibration.CalibrationCollector.forGraph(graph);
            Set<String> required = collector.requiredVariables(graph).inferenceVariables();
            assertEquals(2, required.size());
            assertFalse(required.contains("activation"));
            assertFalse(required.contains("projection_q4"));
            long absoluteMaximumOps = graph.getOps().values().stream()
                    .map(SameDiffOp::getOp)
                    .filter(op -> op != null && "amax".equals(op.opName()))
                    .count();
            assertEquals(2L, absoluteMaximumOps);
            assertFalse(graph.getOps().values().stream()
                    .map(SameDiffOp::getOp)
                    .anyMatch(op -> op != null && "abs".equals(op.opName())));
            for (int sample = 0;
                    sample < SdxTensorG3Q4Calibration.REQUIRED_SAMPLE_COUNT;
                    sample++) {
                collector.beginSample(sample);
                try (INDArray activation =
                             Nd4j.createFromArray(1.0f + sample / 16.0f, -3.0f);
                     INDArray output =
                             Nd4j.createFromArray(-5.0f, 2.0f + sample / 32.0f)) {
                    collector.activationAvailable(
                            graph, null, null, null, "activation", activation);
                    collector.activationAvailable(
                            graph, null, null, null, "projection_q4", output);
                }
                collector.endSample();
            }

            String digest = "b".repeat(64);
            SdxTensorG3Q4Calibration.Result result = collector.result(digest);
            SdxTensorG3Q4Calibration.OperatorCalibration calibration =
                    result.operatorCalibrations().get("projection_q4");

            assertEquals(SdxTensorG3Q4Calibration.REQUIRED_SAMPLE_COUNT,
                    result.sampleCount());
            assertEquals(digest, result.datasetSha256());
            assertEquals(3.0, calibration.observedActivationAbsoluteMaximum(), 0.0);
            assertEquals(5.0, calibration.observedOutputAbsoluteMaximum(), 0.0);
            assertEquals(8.0, calibration.activationScale() * 127.0, 1e-5);
            assertEquals(16.0, calibration.outputScale() * 126.0, 1e-5);

            Path source = Files.write(
                    temporary.resolve("canonical.sdnb"),
                    new byte[] {'S', 'D', 'N', 'B', 1, 2, 3, 4});
            SdxSourceIdentity identity = SdxSourceIdentity.identify(source);
            Path contractPath = temporary.resolve("quantization.json");
            SdxQuantizationContract contract =
                    SdxQuantizationContract.writeTensorG3Q4Profile(
                            contractPath, identity, result);
            assertTrue(Files.isRegularFile(contractPath));
            assertTrue(contract.isTensorG3Q4PerOperator());
            assertEquals(digest, contract.calibrationDatasetSha256());
            assertEquals(SdxTensorG3Q4Calibration.REQUIRED_SAMPLE_COUNT,
                    contract.calibrationSampleCount());
            assertEquals(calibration.activationScale(),
                    contract.operatorCalibration("projection_q4").activationScale());
        }
    }

    @Test
    void rejectsIncompleteSamplesAndZeroOnlyCalibration() throws Exception {
        try (SameDiff graph = q4Graph(8L)) {
            SdxTensorG3Q4Calibration.CalibrationCollector incomplete =
                    SdxTensorG3Q4Calibration.CalibrationCollector.forGraph(graph);
            incomplete.beginSample(0);
            try (INDArray activation = Nd4j.ones(1)) {
                incomplete.activationAvailable(
                        graph, null, null, null, "activation", activation);
            }
            IllegalStateException missing = assertThrows(
                    IllegalStateException.class, incomplete::endSample);
            assertTrue(missing.getMessage().contains("both boundaries"));

            SdxTensorG3Q4Calibration.CalibrationCollector zeroOnly =
                    SdxTensorG3Q4Calibration.CalibrationCollector.forGraph(graph);
            for (int sample = 0;
                    sample < SdxTensorG3Q4Calibration.REQUIRED_SAMPLE_COUNT;
                    sample++) {
                zeroOnly.beginSample(sample);
                try (INDArray activation = Nd4j.zeros(1);
                     INDArray output = Nd4j.zeros(1)) {
                    zeroOnly.activationAvailable(
                            graph, null, null, null, "activation", activation);
                    zeroOnly.activationAvailable(
                            graph, null, null, null, "projection_q4", output);
                }
                zeroOnly.endSample();
            }
            IllegalStateException zero = assertThrows(
                    IllegalStateException.class,
                    () -> zeroOnly.result("c".repeat(64)));
            assertTrue(zero.getMessage().contains("finite and positive"));
        }
    }

    @Test
    void datasetIdentityIsOrderedAndTokenizerBound() throws Exception {
        List<String> prompts = new ArrayList<>(
                SdxTensorG3Q4Calibration.calibrationPrompts());
        String first = SdxTensorG3Q4Calibration.datasetSha256(
                "a".repeat(64), prompts);
        String repeated = SdxTensorG3Q4Calibration.datasetSha256(
                "a".repeat(64), prompts);
        prompts.set(0, prompts.get(0) + " changed");
        String changedPrompt = SdxTensorG3Q4Calibration.datasetSha256(
                "a".repeat(64), prompts);
        String changedTokenizer = SdxTensorG3Q4Calibration.datasetSha256(
                "b".repeat(64), SdxTensorG3Q4Calibration.calibrationPrompts());

        assertEquals(first, repeated);
        assertNotEquals(first, changedPrompt);
        assertNotEquals(first, changedTokenizer);
        assertEquals(64, first.length());
        assertThrows(IOException.class, () ->
                SdxTensorG3Q4Calibration.datasetSha256(
                        "not-a-digest", SdxTensorG3Q4Calibration.calibrationPrompts()));
    }

    @Test
    void publicProducerRunsEverySampleAndRestoresGraphListeners() throws Exception {
        try (SameDiff graph = q4Graph(8L)) {
            AtomicInteger executions = new AtomicInteger();
            SdxTensorG3Q4Calibration.Result result =
                    SdxTensorG3Q4Calibration.calibrate(
                            graph,
                            "d".repeat(64),
                            SdxTensorG3Q4Calibration.calibrationPrompts(),
                            prompt -> {
                                int sample = executions.incrementAndGet();
                                try (INDArray activation =
                                             Nd4j.createFromArray((float) sample, -2.0f);
                                     INDArray output =
                                             Nd4j.createFromArray(-3.0f, sample / 2.0f)) {
                                    for (Listener listener
                                            : new ArrayList<>(graph.getListeners())) {
                                        if (!listener.isActive(Operation.INFERENCE)) {
                                            continue;
                                        }
                                        listener.activationAvailable(
                                                graph, null, null, null,
                                                "activation", activation);
                                        listener.activationAvailable(
                                                graph, null, null, null,
                                                "projection_q4", output);
                                    }
                                }
                            });

            assertEquals(SdxTensorG3Q4Calibration.REQUIRED_SAMPLE_COUNT,
                    executions.get());
            assertEquals(SdxTensorG3Q4Calibration.REQUIRED_SAMPLE_COUNT,
                    result.sampleCount());
            assertTrue(result.hasQ4Operations());
            assertTrue(graph.getListeners().isEmpty(),
                    "Calibration must restore the graph listener set");
        }
    }

    @Test
    void dspInferencePublishesDirectPlaceholderAndQ4OutputBoundaries() throws Exception {
        try (SameDiff graph = q4Graph(8L);
             INDArray activation =
                     Nd4j.ones(DataType.FLOAT, 1, 256)) {
            SdxTensorG3Q4Calibration.CalibrationCollector collector =
                    SdxTensorG3Q4Calibration.CalibrationCollector.forGraph(graph);
            assertFalse(collector.requiresAllActivations());
            graph.setListeners(collector);
            collector.beginSample(0);
            graph.output(Map.of("activation", activation), "projection_q4");
            collector.endSample();
        }
    }

    @Test
    void ignoresQ6AndQ8PackedOperations() throws Exception {
        try (SameDiff q6 = q4Graph(10L); SameDiff q8 = q4Graph(4L)) {
            assertFalse(SdxTensorG3Q4Calibration.requiresCalibration(q6));
            assertFalse(SdxTensorG3Q4Calibration.requiresCalibration(q8));
        }
    }

    private static SameDiff q4Graph(long quantizationType) {
        SameDiff graph = SameDiff.create();
        SDVariable activation =
                graph.placeHolder("activation", DataType.FLOAT, 1, 256);
        SDVariable weights = graph.constant(
                "weights", Nd4j.zeros(DataType.INT8, 288));
        DynamicCustomOp qmatmul = new DynamicCustomOp() {
            @Override
            public String opName() {
                return "ggml_qmatmul";
            }

            @Override
            public List<DataType> calculateOutputDataTypes(
                    List<DataType> inputDataTypes) {
                return List.of(DataType.FLOAT);
            }
        };
        qmatmul.setSameDiff(graph);
        qmatmul.setOwnName("projection_q4");
        qmatmul.addIArgument(quantizationType, 2L, 256L, 0L);
        graph.addArgsFor(new String[] {activation.name(), weights.name()}, qmatmul);
        qmatmul.outputVariables("projection_q4");
        return graph;
    }
}
