/*
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 * See the NOTICE file distributed with this work for additional information
 * regarding copyright ownership.
 * SPDX-License-Identifier: Apache-2.0
 */
package org.eclipse.deeplearning4j.llm.generation;

import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlanExecutor;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.custom.AutoregressiveDecode;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Review round 6, patches A+B: token-selection parity between the CPU
 * argmax, the corrected CUDA single-row kernel, and the (previously
 * uncorrected) CUDA multi-row VERIFIER kernel.
 *
 * <p>Adversarial logits from the review (all reachable in a NATIVE decode
 * step, not synthetic kernel calls):</p>
 * <ol>
 *   <li>VERY-NEGATIVE FINITE row [-3e30, -2e30, -4e30]: correct answer is
 *       token 1. The old absent-thread init (-1e30 + idx 0) beat every real
 *       logit; the single-row fix returned vocabSize=3 as a token. The shared
 *       valid-candidate rule must return 1.</li>
 *   <li>TIED maxima across strided chunks (vocab 512: logits[1]=logits[256]=10):
 *       correct answer is the LOWEST index 1 (CPU contract). The old multi-row
 *       reduction retained the thread-0 chunk's token 256.</li>
 *   <li>FULL ACCEPTANCE with a tied BONUS row: the verifier emits
 *       [accepted draft, bonus] in one commit — the bonus token must equal the
 *       CPU selector's choice, proving the verifier kernel got the same fix.</li>
 *   <li>NaN in an ACTIVE verification row: the round-6 validity gate must
 *       fail the step before any token/state publication.</li>
 * </ol>
 *
 * <p>The plan harness mirrors TestMtpRerunScratchIsolation's TinyPlan pattern:
 * a greedy (no-proposal) scalar step runs the REAL native controller and its
 * argmax selector over a [1,1,V] logits row; a K=1 window step runs the
 * multi-row verifier. Encoding the logits as graph constants keeps the
 * oracle deterministic.</p>
 */
public class TestNativeArgmaxParity {
    private static final int BASE_TARGET_POSITION = 1;
    private static final int CACHE = 16;

    /** Very-negative finite row: expected argmax = 1 (index 1 holds -2e30). */
    @Test
    public void testVeryNegativeFiniteRowSelectsTrueMaximum() {
        runScalarRow(new float[]{-3e30f, -2e30f, -4e30f}, 1,
                "very-negative finite row must select token 1, not an absent-thread token");
    }

    /** Tied maxima across strided chunks, vocab 512: expected argmax = 1 (lowest index). */
    @Test
    public void testTiedMaximaSelectLowestIndex() {
        float[] logits = new float[512];
        logits[1] = 10.0f;
        logits[256] = 10.0f;
        runScalarRow(logits, 1,
                "tied maxima must select the lowest vocabulary index (CPU contract)");
    }

    /**
     * FULL ACCEPTANCE with a tied bonus row (round 6, finding 1's decisive
     * case): K=1, the verification row 0 uniquely matches the draft, and row 1
     * (the bonus) has tied maxima at 1 and 256. The emitted sequence must be
     * [draft, 1] — proving the MULTI-ROW verifier kernel shares the tie rule.
     */
    @Test
    public void testFullyAcceptedTiedBonusRowEmitsLowestIndexBonus() {
        float[] bonusRow = new float[512];
        bonusRow[1] = 10.0f;
        bonusRow[256] = 10.0f;
        runWindowStep(1, bonusRow, false,
                "fully accepted step must emit the CPU-parity bonus token (lowest index), "
                        + "not the multi-row kernel's chunk token");
    }

    /**
     * Round 6, finding 3: NaN in an ACTIVE verification row must fail the
     * step loudly (SPEC VERIFY VALIDITY GUARD) before any token/state commit.
     */
    @Test
    public void testNanVerificationRowFailsBeforeCommit() {
        float[] bonusRow = new float[512];
        java.util.Arrays.fill(bonusRow, Float.NaN);
        runWindowStep(1, bonusRow, true,
                "NaN verification row must trip the validity guard before emission");
    }

    /**
     * Greedy scalar step: the native controller's SINGLE-ROW selector runs over
     * one [1,1,V] logits row; assert the emitted token equals expectedToken.
     */
    private void runScalarRow(float[] logits, long expectedToken, String message) {
        try (TinyScalarPlan plan = new TinyScalarPlan(logits)) {
            try (INDArray embeddings = Nd4j.zeros(DataType.FLOAT, 1, 1, 1);
                 INDArray table = Nd4j.ones(DataType.FLOAT, 2, 1);
                 INDArray positions = Nd4j.valueArrayOf(new long[]{1, 1}, BASE_TARGET_POSITION, DataType.INT64)) {
                AutoregressiveDecode op = new AutoregressiveDecode(
                        embeddings, table, plan.input("ids"), plan.input("mask"), positions,
                        null, plan.executor.getNativePlanHandle(), plan.executor.getCachedOpContext(),
                        plan.executor.getCurrentPlan().getExternalInputKeys().length, plan.outputs.size(),
                        -1, -1, plan.ext("mask"), -1, plan.ext("ids"), plan.out("logits"),
                        -1, plan.ext("position"), plan.ext("cache_position"),
                        new int[0], new int[0], new int[0], new int[0], new int[0], new int[0],
                        1, -1, 0, BASE_TARGET_POSITION, 0.0, 0, 0.0, 1.0, Set.of());
                INDArray[] result = Nd4j.getExecutioner().exec(op);
                try {
                    long tokenCount = result[1].getLong(0);
                    assertTrue(tokenCount >= 1, "scalar step must emit at least one token");
                    assertEquals(expectedToken, result[0].getLong(0), message);
                } finally {
                    for (INDArray array : result) array.close();
                }
            }
        }
    }

    /**
     * One K=1 window step with the given bonus-row logits (row 1 of the
     * verification output). The window graph's row-0 logits uniquely favor
     * the predictor's draft so the step fully accepts; nanRow=true fills
     * row 1 with NaN to trip the validity gate.
     */
    private void runWindowStep(long draftToken, float[] bonusRowLogits, boolean nanRow,
                               String message) {
        try (TinyWindowPlan window = new TinyWindowPlan(draftToken, bonusRowLogits, nanRow);
             TinyPredictorPlan predictor = new TinyPredictorPlan(draftToken)) {
            try (INDArray embeddings = Nd4j.zeros(DataType.FLOAT, 1, 1, 1);
                 INDArray table = Nd4j.ones(DataType.FLOAT, 2, 1);
                 INDArray positions = Nd4j.valueArrayOf(new long[]{1, 1}, BASE_TARGET_POSITION, DataType.INT64)) {
                DynamicShapePlanExecutor t = window.executor;
                DynamicShapePlanExecutor p = predictor.executor;
                AutoregressiveDecode op = new AutoregressiveDecode(
                        embeddings, table, window.input("ids"), window.input("mask"), positions,
                        null, t.getNativePlanHandle(), t.getCachedOpContext(),
                        t.getCurrentPlan().getExternalInputKeys().length, window.outputs.size(),
                        -1, -1, window.ext("mask"), -1, window.ext("ids"), window.out("logits"),
                        -1, window.ext("position"), window.ext("cache_position"),
                        new int[0], new int[0], new int[0], new int[0], new int[0], new int[0],
                        2, -1, 0, BASE_TARGET_POSITION, 0.0, 0, 0.0, 1.0, Set.of());
                op.withDecodePolicy(AutoregressiveDecode.DECODE_STRATEGY_SPECULATIVE,
                                1, window.width, 1, 1, -1, 1, 1.0, 0.0, 0)
                        .withActualSequenceLengthExtIdx(window.ext("actual_length"))
                        .withSpeculativeDecoding(1, AutoregressiveDecode.SPECULATOR_TYPE_MTP)
                        .withMtpPlan(predictor.input("ids"), predictor.input("carry"),
                                predictor.input("mask"), predictor.input("position"),
                                predictor.input("cache_position"),
                                new INDArray[]{predictor.input("key"), predictor.input("value")},
                                p.getNativePlanHandle(), p.getCachedOpContext(),
                                p.getCurrentPlan().getExternalInputKeys().length, predictor.outputs.size(),
                                predictor.ext("ids"), predictor.ext("carry"), predictor.ext("mask"),
                                predictor.ext("position"), predictor.ext("cache_position"),
                                new int[]{predictor.ext("key"), predictor.ext("value")},
                                predictor.out("logits"), predictor.out("hidden"), window.out("hidden"));
                // Round 6 finding 3: an invalid ACTIVE verification row must fail
                // the step BEFORE any token/state publication — the op throws, so
                // this arm asserts the throw itself (the guard message is the
                // assertion target).
                if (nanRow) {
                    RuntimeException thrown = assertThrows(RuntimeException.class,
                            () -> Nd4j.getExecutioner().exec(op), message);
                    // The native REQUIRE_TRUE message lives somewhere in the cause
                    // chain (wrapped by the op-execution RuntimeException).
                    StringBuilder text = new StringBuilder();
                    Throwable cause = thrown;
                    while (cause != null) {
                        text.append(cause.getMessage()).append(" | ");
                        cause = cause.getCause();
                    }
                    assertTrue(text.toString().contains("SPEC VERIFY VALIDITY GUARD"),
                            message + " - wrong failure: " + text);
                    return;
                }
                INDArray[] result = Nd4j.getExecutioner().exec(op);
                try {
                    long tokenCount = result[1].getLong(0);
                    // Full acceptance: [draft, bonus]. Bonus = lowest tied index (1).
                    assertEquals(2, tokenCount, message);
                    assertEquals(draftToken, result[0].getLong(0), message);
                    assertEquals(1, result[0].getLong(1),
                            "bonus token must follow the shared lowest-index tie rule");
                } finally {
                    for (INDArray array : result) array.close();
                }
            }
        }
    }

    /** W=1 scalar target whose logits row is a fixed constant vector. */
    private static final class TinyScalarPlan implements AutoCloseable {
        private final SameDiff graph = SameDiff.create();
        private final Map<String, INDArray> inputs = new LinkedHashMap<>();
        private final List<String> outputs = new ArrayList<>();
        private final DynamicShapePlanExecutor executor;

        private TinyScalarPlan(float[] logitsRow) {
            // ids needs an output consumer: a placeholder without a downstream
            // op is dropped from the compiled plan's external-input list.
            SDVariable ids = placeholder("ids", Nd4j.ones(DataType.INT64, 1, 1));
            output(ids.add("ids_echo", 1));
            INDArray mask = Nd4j.valueArrayOf(new long[]{1, 1, 1, CACHE}, -Float.MAX_VALUE, DataType.FLOAT);
            for (int k = 0; k < BASE_TARGET_POSITION; k++) mask.putScalar(new long[]{0, 0, 0, k}, 0);
            output(placeholder("mask", mask).add("mask_echo", 1));
            output(placeholder("position", Nd4j.valueArrayOf(new long[]{1}, BASE_TARGET_POSITION, DataType.INT64)).add("position_echo", 1));
            output(placeholder("cache_position", Nd4j.valueArrayOf(new long[]{1}, BASE_TARGET_POSITION, DataType.INT64)).add("cache_echo", 1));
            output(placeholder("actual_length", Nd4j.ones(DataType.INT64, 1)).add("len_echo", 1));
            SDVariable logits = graph.reshape("logits", graph.constant(
                    Nd4j.createFromArray(logitsRow).reshape(1, 1, logitsRow.length)), 1, 1, logitsRow.length);
            output(logits);
            graph.setDspAutoCompileEnabled(true);
            graph.setDspNativeAutoCompileEnabled(true);
            for (int i = 0; i < 8; i++) graph.output(inputs, outputs.toArray(new String[0]));
            executor = graph.getOrCreateSession().getDynamicShapePlanExecutor();
            assertNotNull(executor);
            assertNotNull(executor.getNativePlanHandle());
            assertFalse(executor.getNativePlanHandle().isNull());
        }

        private SDVariable placeholder(String name, INDArray value) {
            inputs.put(name, value);
            return graph.placeHolder(name, value.dataType(), value.shape());
        }
        private void output(SDVariable value) { outputs.add(value.name()); }
        private INDArray input(String name) { return inputs.get(name); }
        private int ext(String name) {
            int index = executor.findExternalInputIndex(name);
            assertTrue(index >= 0, "missing input " + name);
            return index;
        }
        private int out(String name) {
            int index = executor.findOutputIndex(name);
            assertTrue(index >= 0, "missing output " + name);
            return index;
        }
        @Override public void close() { graph.close(); }
    }

    /**
     * W=2 window target: verification row 0 favors draftToken (full
     * acceptance), row 1 (bonus) uses the supplied logits vector. Token-
     * dependent recurrent state is not needed here — the step fully accepts,
     * so the state commit follows the verify pass (round 6 patch C's
     * multi-row RECOVERY arm lives in TestMtpRerunDisagreementMatrix).
     */
    private static final class TinyWindowPlan implements AutoCloseable {
        private final SameDiff graph = SameDiff.create();
        private final Map<String, INDArray> inputs = new LinkedHashMap<>();
        private final List<String> outputs = new ArrayList<>();
        private final DynamicShapePlanExecutor executor;
        private final int width = 2;

        private TinyWindowPlan(long draftToken, float[] bonusRowLogits, boolean nanRow) {
            SDVariable ids = placeholder("ids", Nd4j.ones(DataType.INT64, 1, width));
            INDArray mask = Nd4j.valueArrayOf(new long[]{1, 1, width, CACHE}, -Float.MAX_VALUE, DataType.FLOAT);
            for (int row = 0; row < width; row++)
                for (int k = 0; k < BASE_TARGET_POSITION; k++) mask.putScalar(new long[]{0, 0, row, k}, 0);
            output(placeholder("mask", mask).add("mask_echo", 1));
            output(placeholder("position", Nd4j.valueArrayOf(new long[]{1}, BASE_TARGET_POSITION, DataType.INT64)).add("position_echo", 1));
            output(placeholder("cache_position", Nd4j.valueArrayOf(new long[]{1}, BASE_TARGET_POSITION, DataType.INT64)).add("cache_echo", 1));
            output(placeholder("actual_length", Nd4j.ones(DataType.INT64, 1)).add("len_echo", 1));
            // Verification rows: row 0 = one-hot at draftToken (accepts); row 1 = bonus vector.
            float[] row0 = new float[bonusRowLogits.length];
            row0[(int) draftToken] = 100.0f;
            INDArray logitsArr = Nd4j.zeros(DataType.FLOAT, 1, width, bonusRowLogits.length);
            for (int v = 0; v < bonusRowLogits.length; v++) {
                logitsArr.putScalar(new long[]{0, 0, v}, row0[v]);
                logitsArr.putScalar(new long[]{0, 1, v}, bonusRowLogits[v]);
            }
            SDVariable logits = graph.reshape("logits", graph.constant(logitsArr), 1, width, bonusRowLogits.length);
            output(logits);
            // Token-dependent hidden for the predictor carry publication.
            SDVariable hidden = graph.reshape("hidden",
                    ids.castTo(DataType.FLOAT).mul(11).reshape(1, width, 1), 1, width, 1);
            output(hidden);
            graph.setDspAutoCompileEnabled(true);
            graph.setDspNativeAutoCompileEnabled(true);
            for (int i = 0; i < 8; i++) graph.output(inputs, outputs.toArray(new String[0]));
            executor = graph.getOrCreateSession().getDynamicShapePlanExecutor();
            assertNotNull(executor);
            assertNotNull(executor.getNativePlanHandle());
            assertFalse(executor.getNativePlanHandle().isNull());
        }

        private SDVariable placeholder(String name, INDArray value) {
            inputs.put(name, value);
            return graph.placeHolder(name, value.dataType(), value.shape());
        }
        private void output(SDVariable value) { outputs.add(value.name()); }
        private INDArray input(String name) { return inputs.get(name); }
        private int ext(String name) {
            int index = executor.findExternalInputIndex(name);
            assertTrue(index >= 0, "missing input " + name);
            return index;
        }
        private int out(String name) {
            int index = executor.findOutputIndex(name);
            assertTrue(index >= 0, "missing output " + name);
            return index;
        }
        @Override public void close() { graph.close(); }
    }

    /** Predictor proposing draftToken (the accepted draft), BSHD KV layout. */
    private static final class TinyPredictorPlan implements AutoCloseable {
        private final SameDiff graph = SameDiff.create();
        private final Map<String, INDArray> inputs = new LinkedHashMap<>();
        private final List<String> outputs = new ArrayList<>();
        private final DynamicShapePlanExecutor executor;

        private TinyPredictorPlan(long draftToken) {
            SDVariable ids = placeholder("ids", Nd4j.ones(DataType.INT64, 1, 1));
            SDVariable carry = placeholder("carry", Nd4j.valueArrayOf(new long[]{1, 1, 1}, 5, DataType.FLOAT));
            SDVariable mask = placeholder("mask",
                    Nd4j.valueArrayOf(new long[]{1, 1, 1, CACHE}, -Float.MAX_VALUE, DataType.FLOAT));
            output(placeholder("position", Nd4j.zeros(DataType.INT64, 1)).add("position_echo", 1));
            SDVariable position = placeholder("cache_position", Nd4j.zeros(DataType.INT64, 1));
            SDVariable key = placeholder("key", Nd4j.zeros(DataType.FLOAT, 1, CACHE, 1, 1));
            SDVariable value = placeholder("value", Nd4j.zeros(DataType.FLOAT, 1, CACHE, 1, 1));
            SDVariable tokenF = ids.reshape(1, 1, 1, 1).castTo(DataType.FLOAT);
            SDVariable k = carry.reshape(1, 1, 1, 1).mul(100).add(tokenF);
            SDVariable v = k.add(0.5);
            output(graph.nn().dotProductAttentionV2("attention", k, v, k, null, null,
                    key, value, position, mask, 0.0, 0.0, false, false));
            output(carry.add("hidden", 1));
            float[] bias = new float[2];
            bias[(int) draftToken] = 40.0f;
            SDVariable zero = ids.castTo(DataType.FLOAT).mul(0).reshape(1, 1, 1);
            SDVariable rawLogits = zero.add(graph.constant(
                    Nd4j.createFromArray(bias).reshape(1, 1, 2)));
            output(graph.castTo("logits", rawLogits, DataType.FLOAT));
            graph.setDspAutoCompileEnabled(true);
            graph.setDspNativeAutoCompileEnabled(true);
            for (int i = 0; i < 8; i++) graph.output(inputs, outputs.toArray(new String[0]));
            executor = graph.getOrCreateSession().getDynamicShapePlanExecutor();
            assertNotNull(executor);
            assertNotNull(executor.getNativePlanHandle());
            assertFalse(executor.getNativePlanHandle().isNull());
        }

        private SDVariable placeholder(String name, INDArray value) {
            inputs.put(name, value);
            return graph.placeHolder(name, value.dataType(), value.shape());
        }
        private void output(SDVariable value) { outputs.add(value.name()); }
        private INDArray input(String name) { return inputs.get(name); }
        private int ext(String name) {
            int index = executor.findExternalInputIndex(name);
            assertTrue(index >= 0, "missing input " + name);
            return index;
        }
        private int out(String name) {
            int index = executor.findOutputIndex(name);
            assertTrue(index >= 0, "missing output " + name);
            return index;
        }
        @Override public void close() { graph.close(); }
    }
}
