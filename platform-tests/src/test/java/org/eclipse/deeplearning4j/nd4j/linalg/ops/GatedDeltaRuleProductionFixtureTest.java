/*
 * ******************************************************************************
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * SPDX-License-Identifier: Apache-2.0
 * *****************************************************************************
 */
package org.eclipse.deeplearning4j.nd4j.linalg.ops;

import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.custom.GatedDeltaRule;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.util.LinkedHashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Replays an opt-in production failure fixture captured by GenerationPipeline.
 *
 * <p>Run with {@code -Dgdr.production.fixture.dir=/path/to/fixture}. Without that property the
 * test is skipped, so large model-derived tensors never become ordinary source fixtures.</p>
 */
public class GatedDeltaRuleProductionFixtureTest {

    @Test
    public void exactProductionFixtureHasStableReferenceAndFiniteNativeResult() throws Exception {
        String configured = System.getProperty("gdr.production.fixture.dir");
        assumeTrue(configured != null && !configured.isBlank(),
                "Set -Dgdr.production.fixture.dir to a captured non-finite GDR fixture");
        File directory = new File(configured).getAbsoluteFile();
        assumeTrue(directory.isDirectory(), "Fixture directory does not exist: " + directory);

        try (INDArray q = load(directory, "input-00-gdn_q_scaled_0.bin");
             INDArray k = load(directory, "input-01-gdn_k_compute_0.bin");
             INDArray v = load(directory, "input-02-gdn_v_compute_0.bin");
             INDArray beta = load(directory, "input-03-gdn_beta_compute_0.bin");
             INDArray gate = load(directory, "input-04-gdn_gate_decay_0.bin");
             INDArray state = load(directory, "input-05-past_gdn_state.0.bin");
             INDArray actualLength = load(directory, "input-06-actual_sequence_length.bin");
             INDArray capturedOutput = load(directory, "output-00-gdn_out_0.bin");
             INDArray capturedState = load(directory, "output-01-gdn_state_out_0.bin")) {

            assertArrayEquals(q.shape(), k.shape(), "Q and K geometry changed");
            assertArrayEquals(q.shape(), v.shape(), "Q and V geometry changed");
            assertTrue(q.rank() == 4 && beta.rank() == 3 && gate.rank() == 3,
                    "Unexpected captured GDR ranks");
            assertTrue(q.dataType() == DataType.FLOAT && k.dataType() == DataType.FLOAT
                            && v.dataType() == DataType.FLOAT && beta.dataType() == DataType.FLOAT
                            && gate.dataType() == DataType.FLOAT && state.dataType() == DataType.FLOAT,
                    "Production GDR inputs must use FLOAT storage");

            int sequence = Math.toIntExact(q.size(1));
            int heads = Math.toIntExact(q.size(2));
            int keyDim = Math.toIntExact(q.size(3));
            int valueDim = Math.toIntExact(v.size(3));
            long activeLength = actualLength.getLong(0);
            assertTrue(activeLength == sequence,
                    "Captured active length must cover the full production prompt");
            assertArrayEquals(new long[]{1, heads, keyDim, valueDim}, state.shape(),
                    "Captured recurrent-state geometry changed");

            float[] qValues = q.data().asFloat();
            float[] kValues = k.data().asFloat();
            float[] vValues = v.data().asFloat();
            float[] betaValues = beta.data().asFloat();
            float[] gateValues = gate.data().asFloat();
            float[] stateValues = state.data().asFloat();
            float[] capturedValues = capturedOutput.data().asFloat();
            float[] capturedStateValues = capturedState.data().asFloat();

            assertAllFinite("q", qValues);
            assertAllFinite("k", kValues);
            assertAllFinite("v", vValues);
            assertAllFinite("beta", betaValues);
            assertAllFinite("gate", gateValues);
            assertAllFinite("state", stateValues);

            double stateMax = maxAbs(stateValues);
            // Decode-step fixtures carry a NON-zero (but finite) recurrent state — the
            // state written by the previous decode step of the same request. Only the
            // original prefill fixtures have all-zero state. Require finite, not zero.
            assertTrue(java.lang.Double.isFinite(stateMax),
                    "Captured initial recurrent state is not finite: " + stateMax);

            Range betaRange = range(betaValues);
            Range gateRange = range(gateValues);
            assertTrue(betaRange.min >= 0.0 && betaRange.max <= 1.0,
                    "Captured beta escaped [0,1]: " + betaRange);
            assertTrue(gateRange.max <= 0.0,
                    "Captured log-domain gate contains growth instead of decay: " + gateRange);

            Range keyNormRange = keyNormRange(kValues, sequence, heads, keyDim);
            assertTrue(keyNormRange.min > 0.98 && keyNormRange.max < 1.02,
                    "Captured K vectors are not normalized: " + keyNormRange);

            ArrayStats capturedStats = stats(capturedValues);
            assertTrue(capturedStats.firstNonFinite >= 0,
                    "The selected fixture no longer contains the production non-finite output");
            int capturedDv = capturedStats.firstNonFinite % valueDim;
            int capturedRow = capturedStats.firstNonFinite / valueDim;
            int capturedHead = capturedRow % heads;
            int capturedToken = capturedRow / heads;

            ReferenceStats reference = runDoubleReferenceHead(
                    qValues, kValues, vValues, betaValues, gateValues, stateValues,
                    sequence, heads, keyDim, valueDim, capturedHead);
            System.out.printf(
                    "GDR_FIXTURE input sequence=%d heads=%d keyDim=%d valueDim=%d "
                            + "beta=%s gate=%s kNorm=%s captured=%s firstCaptured=[t=%d,h=%d,dv=%d] "
                            + "capturedState=%s doubleReference=%s%n",
                    sequence, heads, keyDim, valueDim, betaRange, gateRange, keyNormRange,
                    capturedStats, capturedToken, capturedHead, capturedDv,
                    stats(capturedStateValues), reference);
            assertTrue(reference.firstNonFiniteToken < 0,
                    "Independent double recurrence became non-finite at token "
                            + reference.firstNonFiniteToken);

            float[] scratchOutput;
            float[] scratchState;
            INDArray[] scratchReplay = Nd4j.exec(new GatedDeltaRule(
                    q, k, v, beta, gate, state));
            try {
                scratchOutput = scratchReplay[0].data().asFloat();
                scratchState = scratchReplay[1].data().asFloat();
                ArrayStats scratchOutputStats = stats(scratchOutput);
                ArrayStats scratchStateStats = stats(scratchState);
                System.out.printf("GDR_FIXTURE_SCRATCH output=%s state=%s%n",
                        scratchOutputStats, scratchStateStats);
                assertTrue(scratchOutputStats.firstNonFinite < 0
                                && scratchStateStats.firstNonFinite < 0,
                        "Established scratch execution became non-finite: output="
                                + scratchOutputStats + " state=" + scratchStateStats);
            } finally {
                scratchReplay[0].close();
                scratchReplay[1].close();
            }

            INDArray[] replay = Nd4j.exec(new GatedDeltaRule(
                    q, k, v, beta, gate, state, actualLength));
            try {
                float[] replayValues = replay[0].data().asFloat();
                float[] replayStateValues = replay[1].data().asFloat();
                ArrayStats replayStats = stats(replayValues);
                ArrayStats replayStateStats = stats(replayStateValues);
                DifferenceStats outputDifference = difference(scratchOutput, replayValues);
                DifferenceStats stateDifference = difference(scratchState, replayStateValues);
                System.out.printf("GDR_FIXTURE_NATIVE backend=%s output=%s state=%s "
                                + "scratchDifference={output=%s,state=%s}%n",
                        Nd4j.getBackend().getClass().getSimpleName(), replayStats,
                        replayStateStats, outputDifference, stateDifference);
                assertFalse(replayStats.firstNonFinite >= 0 || replayStateStats.firstNonFinite >= 0,
                        "Native GDR became non-finite while the independent recurrence stayed finite: output="
                                + replayStats + " state=" + replayStateStats);
                assertEquals(0.0, outputDifference.maxAbs, 0.0,
                        "Direct-state GDR output differs from the scratch reference at index "
                                + outputDifference.index);
                assertEquals(0.0, stateDifference.maxAbs, 0.0,
                        "Direct-state GDR state differs from the scratch reference at index "
                                + stateDifference.index);
            } finally {
                replay[0].close();
                replay[1].close();
            }

            try (SameDiff graph = SameDiff.create()) {
                SDVariable qVar = graph.placeHolder("q", DataType.FLOAT, q.shape());
                SDVariable kVar = graph.placeHolder("k", DataType.FLOAT, k.shape());
                SDVariable vVar = graph.placeHolder("v", DataType.FLOAT, v.shape());
                SDVariable betaVar = graph.placeHolder("beta", DataType.FLOAT, beta.shape());
                SDVariable gateVar = graph.placeHolder("gate", DataType.FLOAT, gate.shape());
                SDVariable stateVar = graph.placeHolder("state", DataType.FLOAT, state.shape());
                SDVariable actualLengthVar = graph.placeHolder("actual_length", DataType.INT64);
                SDVariable[] graphOutputs = new GatedDeltaRule(
                        graph, qVar, kVar, vVar, betaVar, gateVar, stateVar, actualLengthVar)
                        .outputVariables();
                graph.updateVariableNameAndReference(graphOutputs[0], "gdr_output");
                graph.updateVariableNameAndReference(graphOutputs[1], "gdr_state");
                graph.setGraphExecutionMode(GraphExecutionMode.TRITON);

                Map<String, INDArray> inputs = new LinkedHashMap<>();
                inputs.put("q", q);
                inputs.put("k", k);
                inputs.put("v", v);
                inputs.put("beta", beta);
                inputs.put("gate", gate);
                inputs.put("state", state);
                inputs.put("actual_length", actualLength);
                Map<String, INDArray> oneSlot = graph.output(inputs, "gdr_output", "gdr_state");
                ArrayStats oneSlotOutput = stats(oneSlot.get("gdr_output").data().asFloat());
                ArrayStats oneSlotState = stats(oneSlot.get("gdr_state").data().asFloat());
                DifferenceStats oneSlotOutputDifference = difference(
                        scratchOutput, oneSlot.get("gdr_output").data().asFloat());
                DifferenceStats oneSlotStateDifference = difference(
                        scratchState, oneSlot.get("gdr_state").data().asFloat());
                System.out.printf("GDR_FIXTURE_ONE_SLOT_DSP output=%s state=%s "
                                + "scratchDifference={output=%s,state=%s}%n",
                        oneSlotOutput, oneSlotState, oneSlotOutputDifference,
                        oneSlotStateDifference);
                assertTrue(oneSlotOutput.firstNonFinite < 0 && oneSlotState.firstNonFinite < 0,
                        "One-slot DSP execution became non-finite: output=" + oneSlotOutput
                                + " state=" + oneSlotState);
                assertEquals(0.0, oneSlotOutputDifference.maxAbs, 0.0,
                        "One-slot DSP output differs from the scratch reference at index "
                                + oneSlotOutputDifference.index);
                assertEquals(0.0, oneSlotStateDifference.maxAbs, 0.0,
                        "One-slot DSP state differs from the scratch reference at index "
                                + oneSlotStateDifference.index);
            }

            try (SameDiff graph = SameDiff.create()) {
                SDVariable qVar = graph.placeHolder("q", DataType.FLOAT, q.shape());
                SDVariable kVar = graph.placeHolder("k", DataType.FLOAT, k.shape());
                SDVariable vVar = graph.placeHolder("v", DataType.FLOAT, v.shape());
                SDVariable betaVar = graph.placeHolder("beta", DataType.FLOAT, beta.shape());
                SDVariable gateVar = graph.placeHolder("gate", DataType.FLOAT, gate.shape());
                SDVariable stateVar = graph.placeHolder("state", DataType.FLOAT, state.shape());
                SDVariable actualLengthVar = graph.placeHolder("actual_length", DataType.INT64);

                SDVariable qPrepared = qVar.mul("q_prepared", 1.0);
                SDVariable kPrepared = kVar.mul("k_prepared", 1.0);
                SDVariable vPrepared = vVar.mul("v_prepared", 1.0);
                SDVariable betaPrepared = betaVar.mul("beta_prepared", 1.0);
                SDVariable gatePrepared = gateVar.mul("gate_prepared", 1.0);
                SDVariable[] graphOutputs = new GatedDeltaRule(
                        graph, qPrepared, kPrepared, vPrepared, betaPrepared, gatePrepared,
                        stateVar, actualLengthVar).outputVariables();
                graph.updateVariableNameAndReference(graphOutputs[0], "gdr_output");
                graph.updateVariableNameAndReference(graphOutputs[1], "gdr_state");
                graphOutputs[0].mul("post_gdr", 1.0);
                graph.setGraphExecutionMode(GraphExecutionMode.TRITON);

                Map<String, INDArray> inputs = new LinkedHashMap<>();
                inputs.put("q", q);
                inputs.put("k", k);
                inputs.put("v", v);
                inputs.put("beta", beta);
                inputs.put("gate", gate);
                inputs.put("state", state);
                inputs.put("actual_length", actualLength);
                Map<String, INDArray> mixed = graph.output(inputs, "post_gdr", "gdr_state");
                ArrayStats mixedOutput = stats(mixed.get("post_gdr").data().asFloat());
                ArrayStats mixedState = stats(mixed.get("gdr_state").data().asFloat());
                DifferenceStats mixedOutputDifference = difference(
                        scratchOutput, mixed.get("post_gdr").data().asFloat());
                DifferenceStats mixedStateDifference = difference(
                        scratchState, mixed.get("gdr_state").data().asFloat());
                System.out.printf("GDR_FIXTURE_MIXED_DSP output=%s state=%s "
                                + "scratchDifference={output=%s,state=%s}%n",
                        mixedOutput, mixedState, mixedOutputDifference, mixedStateDifference);
                assertTrue(mixedOutput.firstNonFinite < 0 && mixedState.firstNonFinite < 0,
                        "Minimal mixed Triton/native DSP execution became non-finite: output="
                                + mixedOutput + " state=" + mixedState);
                assertEquals(0.0, mixedOutputDifference.maxAbs, 0.0,
                        "Mixed DSP output differs from the scratch reference at index "
                                + mixedOutputDifference.index);
                assertEquals(0.0, mixedStateDifference.maxAbs, 0.0,
                        "Mixed DSP state differs from the scratch reference at index "
                                + mixedStateDifference.index);
            }
        }
    }

    private static INDArray load(File directory, String name) throws Exception {
        File file = new File(directory, name);
        assertTrue(file.isFile(), "Missing fixture array: " + file);
        return Nd4j.readBinary(file);
    }

    private static void assertAllFinite(String name, float[] values) {
        ArrayStats arrayStats = stats(values);
        assertTrue(arrayStats.firstNonFinite < 0, name + " contains non-finite values: " + arrayStats);
    }

    private static Range keyNormRange(float[] values, int sequence, int heads, int keyDim) {
        double min = Double.POSITIVE_INFINITY;
        double max = Double.NEGATIVE_INFINITY;
        for (int token = 0; token < sequence; token++) {
            for (int head = 0; head < heads; head++) {
                int base = (token * heads + head) * keyDim;
                double sum = 0.0;
                for (int dimension = 0; dimension < keyDim; dimension++) {
                    double value = values[base + dimension];
                    sum += value * value;
                }
                double norm = Math.sqrt(sum);
                min = Math.min(min, norm);
                max = Math.max(max, norm);
            }
        }
        return new Range(min, max);
    }

    private static ReferenceStats runDoubleReferenceHead(
            float[] q, float[] k, float[] v, float[] beta, float[] gate, float[] initialState,
            int sequence, int heads, int keyDim, int valueDim, int head) {
        double[] state = new double[keyDim * valueDim];
        int initialBase = head * keyDim * valueDim;
        for (int i = 0; i < state.length; i++) {
            state[i] = initialState[initialBase + i];
        }

        int firstNonFiniteToken = -1;
        double maxAbsOutput = 0.0;
        double maxAbsState = maxAbs(state);
        for (int token = 0; token < sequence; token++) {
            int vectorBase = (token * heads + head) * keyDim;
            int valueBase = (token * heads + head) * valueDim;
            int scalarIndex = token * heads + head;
            double decay = Math.exp(gate[scalarIndex]);
            double betaValue = beta[scalarIndex];

            for (int dv = 0; dv < valueDim; dv++) {
                double prediction = 0.0;
                for (int dk = 0; dk < keyDim; dk++) {
                    prediction += state[dk * valueDim + dv] * k[vectorBase + dk];
                }
                double delta = v[valueBase + dv] - decay * prediction;
                double betaDelta = betaValue * delta;
                for (int dk = 0; dk < keyDim; dk++) {
                    int stateIndex = dk * valueDim + dv;
                    state[stateIndex] = decay * state[stateIndex]
                            + betaDelta * k[vectorBase + dk];
                    maxAbsState = Math.max(maxAbsState, Math.abs(state[stateIndex]));
                }

                double output = 0.0;
                for (int dk = 0; dk < keyDim; dk++) {
                    output += state[dk * valueDim + dv] * q[vectorBase + dk];
                }
                maxAbsOutput = Math.max(maxAbsOutput, Math.abs(output));
                if (!Double.isFinite(output) && firstNonFiniteToken < 0) {
                    firstNonFiniteToken = token;
                }
            }
        }
        return new ReferenceStats(firstNonFiniteToken, maxAbsOutput, maxAbsState);
    }

    private static ArrayStats stats(float[] values) {
        int firstNonFinite = -1;
        int nonFiniteCount = 0;
        double maxAbsFinite = 0.0;
        for (int i = 0; i < values.length; i++) {
            float value = values[i];
            if (!Float.isFinite(value)) {
                if (firstNonFinite < 0) {
                    firstNonFinite = i;
                }
                nonFiniteCount++;
            } else {
                maxAbsFinite = Math.max(maxAbsFinite, Math.abs((double) value));
            }
        }
        return new ArrayStats(firstNonFinite, nonFiniteCount, maxAbsFinite);
    }

    private static DifferenceStats difference(float[] expected, float[] actual) {
        assertEquals(expected.length, actual.length, "Array lengths differ");
        int index = -1;
        double maxAbs = 0.0;
        for (int i = 0; i < expected.length; i++) {
            double delta = Math.abs((double) expected[i] - actual[i]);
            if (delta > maxAbs) {
                maxAbs = delta;
                index = i;
            }
        }
        return new DifferenceStats(index, maxAbs);
    }

    private static Range range(float[] values) {
        double min = Double.POSITIVE_INFINITY;
        double max = Double.NEGATIVE_INFINITY;
        for (float value : values) {
            min = Math.min(min, value);
            max = Math.max(max, value);
        }
        return new Range(min, max);
    }

    private static double maxAbs(float[] values) {
        double max = 0.0;
        for (float value : values) {
            max = Math.max(max, Math.abs((double) value));
        }
        return max;
    }

    private static double maxAbs(double[] values) {
        double max = 0.0;
        for (double value : values) {
            max = Math.max(max, Math.abs(value));
        }
        return max;
    }

    private static final class Range {
        final double min;
        final double max;

        Range(double min, double max) {
            this.min = min;
            this.max = max;
        }

        @Override
        public String toString() {
            return "[" + min + "," + max + "]";
        }
    }

    private static final class ArrayStats {
        final int firstNonFinite;
        final int nonFiniteCount;
        final double maxAbsFinite;

        ArrayStats(int firstNonFinite, int nonFiniteCount, double maxAbsFinite) {
            this.firstNonFinite = firstNonFinite;
            this.nonFiniteCount = nonFiniteCount;
            this.maxAbsFinite = maxAbsFinite;
        }

        @Override
        public String toString() {
            return "{firstNonFinite=" + firstNonFinite + ", nonFiniteCount=" + nonFiniteCount
                    + ", maxAbsFinite=" + maxAbsFinite + "}";
        }
    }

    private static final class ReferenceStats {
        final int firstNonFiniteToken;
        final double maxAbsOutput;
        final double maxAbsState;

        ReferenceStats(int firstNonFiniteToken, double maxAbsOutput, double maxAbsState) {
            this.firstNonFiniteToken = firstNonFiniteToken;
            this.maxAbsOutput = maxAbsOutput;
            this.maxAbsState = maxAbsState;
        }

        @Override
        public String toString() {
            return "{firstNonFiniteToken=" + firstNonFiniteToken + ", maxAbsOutput="
                    + maxAbsOutput + ", maxAbsState=" + maxAbsState + "}";
        }
    }

    private static final class DifferenceStats {
        final int index;
        final double maxAbs;

        DifferenceStats(int index, double maxAbs) {
            this.index = index;
            this.maxAbs = maxAbs;
        }

        @Override
        public String toString() {
            return "{index=" + index + ",maxAbs=" + maxAbs + "}";
        }
    }
}
