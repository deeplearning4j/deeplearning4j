/*
 *  ******************************************************************************
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  ******************************************************************************
 */
package org.eclipse.deeplearning4j.nd4j.autodiff.samediff;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.DspHandle;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.LinkedHashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

@Tag(TagNames.FULL_CI)
@DisplayName("Strict oneDNN Graph backend")
class OneDnnGraphBackendStrictReplayTest {

    private SameDiff reference;
    private SameDiff oneDnn;

    @BeforeEach
    void setUp() {
        String backendName = Nd4j.getBackend().getClass().getName().toLowerCase();
        assumeTrue(backendName.contains("cpu") || backendName.contains("native"),
                "strict oneDNN tests require the CPU/native backend");
        assumeTrue(Nd4j.getBackend().buildInfo().contains("HAVE_ONEDNN"),
                "strict oneDNN tests require a build with the oneDNN Graph API");
        Nd4j.getRandom().setSeed(12345);
    }

    @AfterEach
    void tearDown() {
        if (oneDnn != null) oneDnn.close();
        if (reference != null) reference.close();
    }

    private static SameDiff graph(GraphExecutionMode mode, INDArray weights, INDArray bias) {
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 2, 4);
        SDVariable weight = sd.var("weight", weights);
        SDVariable biasVariable = sd.var("bias", bias);
        SDVariable product = sd.mmul("product", input, weight);
        SDVariable biased = product.add("biased", biasVariable);
        sd.nn().relu("out", biased, 0.0);
        sd.setOutputs("out");
        sd.setGraphExecutionMode(mode);
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);
        return sd;
    }

    private static Map<String, INDArray> inputs(INDArray input) {
        Map<String, INDArray> values = new LinkedHashMap<>();
        values.put("input", input);
        return values;
    }

    @Test
    @DisplayName("canonical descriptor emitters preserve slot parity and own the segment")
    void strictOneDnnMatchesSlotBySlotAcrossChangingInputs() {
        INDArray weights = Nd4j.create(new float[]{
                0.10f, -0.20f, 0.30f,
                0.40f, 0.50f, -0.60f,
                -0.70f, 0.80f, 0.90f,
                1.00f, -1.10f, 1.20f
        }, 4, 3);
        INDArray bias = Nd4j.create(new float[]{0.25f, -0.50f, 0.75f}, 1, 3);
        reference = graph(GraphExecutionMode.SLOT_BY_SLOT, weights.dup(), bias.dup());
        oneDnn = graph(GraphExecutionMode.ONEDNN, weights.dup(), bias.dup());

        INDArray input = Nd4j.create(DataType.FLOAT, 2, 4);
        for (int step = 0; step < 8; step++) {
            input.assign((step + 1) * 0.125);
            INDArray expected = reference.output(inputs(input), "out").get("out");
            INDArray actual = oneDnn.output(inputs(input), "out").get("out");
            assertTrue(expected.equalsWithEps(actual, 1e-5),
                    "oneDNN mismatch at step " + step + " expected=" + expected + " actual=" + actual);
        }

        DspHandle handle = oneDnn.dsp();
        assertTrue(handle.isCompiled(), "strict oneDNN plan was not compiled");
        assertTrue(handle.graphExecutionMode() == GraphExecutionMode.ONEDNN,
                "native mode 20 did not round-trip: " + handle.graphExecutionMode());
        assertTrue(handle.numSegments() > 0, "strict oneDNN plan has no segments");
        assertTrue(handle.segmentCompiledBackend(0).contains("OneDNN"),
                "segment was not owned by OneDNN: " + handle.segmentCompiledBackend(0));
        String audit = handle.segmentCompilationAudit(0);
        assertTrue(audit.contains("matmul") || audit.contains("product"), audit);
        assertFalse(audit.contains("\"wasCompiled\":false"), audit);
    }

    @Test
    @DisplayName("strict mode rejects operations without an exact emitter")
    void strictOneDnnDoesNotFallBackForUnsupportedReduction() {
        oneDnn = SameDiff.create();
        SDVariable input = oneDnn.placeHolder("input", DataType.FLOAT, 2, 4);
        input.mean("out", true, 1);
        oneDnn.setOutputs("out");
        oneDnn.setGraphExecutionMode(GraphExecutionMode.ONEDNN);
        oneDnn.setDspAutoCompileEnabled(true);
        oneDnn.setDspNativeAutoCompileEnabled(true);
        INDArray value = Nd4j.ones(DataType.FLOAT, 2, 4);

        assertThrows(RuntimeException.class, () -> {
            for (int execution = 0; execution < 4; execution++) {
                oneDnn.output(inputs(value), "out");
            }
        });
    }

    @Test
    @DisplayName("segment invalidation clears and rebuilds the owned oneDNN artifact")
    void strictOneDnnRebuildsAfterPublicInvalidation() {
        INDArray weights = Nd4j.ones(DataType.FLOAT, 4, 3).muli(0.25);
        INDArray bias = Nd4j.ones(DataType.FLOAT, 1, 3).muli(0.5);
        oneDnn = graph(GraphExecutionMode.ONEDNN, weights, bias);
        INDArray input = Nd4j.ones(DataType.FLOAT, 2, 4);
        INDArray before = null;
        for (int execution = 0; execution < 6; execution++) {
            before = oneDnn.output(inputs(input), "out").get("out").dup();
        }

        DspHandle handle = oneDnn.dsp();
        handle.invalidateSegmentCache(0);
        INDArray after = oneDnn.output(inputs(input), "out").get("out");
        assertTrue(before.equalsWithEps(after, 1e-5),
                "rebuild changed result: before=" + before + " after=" + after);
        assertTrue(handle.segmentCompiledBackend(0).contains("OneDNN"),
                "segment was not rebuilt by OneDNN");
    }

    @Test
    @DisplayName("same-shape plans do not share mutable oneDNN artifacts")
    void strictOneDnnArtifactsArePlanOwned() {
        INDArray firstWeights = Nd4j.ones(DataType.FLOAT, 4, 3).muli(0.1);
        INDArray secondWeights = Nd4j.ones(DataType.FLOAT, 4, 3).muli(0.9);
        INDArray bias = Nd4j.zeros(DataType.FLOAT, 1, 3);
        oneDnn = graph(GraphExecutionMode.ONEDNN, firstWeights, bias.dup());
        SameDiff second = graph(GraphExecutionMode.ONEDNN, secondWeights, bias.dup());
        try {
            INDArray input = Nd4j.ones(DataType.FLOAT, 2, 4);
            double firstSum = 0.0;
            double secondSum = 0.0;
            for (int execution = 0; execution < 8; execution++) {
                firstSum = oneDnn.output(inputs(input), "out").get("out")
                        .sumNumber().doubleValue();
                secondSum = second.output(inputs(input), "out").get("out")
                        .sumNumber().doubleValue();
            }
            assertTrue(Math.abs(firstSum - secondSum) > 1.0,
                    "same-shape plans cross-hit mutable state: " + firstSum + " vs " + secondSum);
            assertTrue(oneDnn.dsp().segmentCompiledBackend(0).contains("OneDNN"));
            assertTrue(second.dsp().segmentCompiledBackend(0).contains("OneDNN"));
        } finally {
            second.close();
        }
    }
}
