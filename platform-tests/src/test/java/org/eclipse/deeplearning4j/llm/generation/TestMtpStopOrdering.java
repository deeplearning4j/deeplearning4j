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
import org.nd4j.autodiff.samediff.SDIndex;
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

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * T1 of ADRs/mtp-reference-contract-repair-plan.md (audit finding F3): the CUDA
 * speculative commit must run stop matching on the AUTHORITATIVE emitted token
 * only, and exactly once.
 *
 * <p>The verification pass executes at actual_sequence_length = W while the
 * asl=1 re-execution that produces the committed state runs at
 * actual_sequence_length = 1. These fixtures make the target's row-0 argmax a
 * deterministic function of that scalar, so the verification argmax (what the
 * buggy ordering feeds the stop matcher) and the re-run argmax (what is
 * actually emitted) are Pinned to different vocabulary slots per mode:
 * <ul>
 *   <li>VERIFY_EOS: verification predicts EOS, re-run predicts token 0. The
 *       stale ordering stops generation after emitting a non-EOS token.</li>
 *   <li>RERUN_EOS: verification predicts 0, re-run emits EOS. The stale
 *       ordering emits EOS but never stops.</li>
 *   <li>RERUN_STOP2: verification predicts 0, re-run emits stop id 2. A scalar
 *       stop id that never reaches the matcher under the stale ordering.</li>
 *   <li>MULTI_STOP: verification (wide pass) predicts the phantom stop token 2
 *       from decode position 2 on while the re-run keeps emitting 0. The stale
 *       matcher assembles the stop suffix [0, 2] from tokens that were never
 *       emitted.</li>
 *   <li>NONE: no disagreement (control - must pass before and after the fix).</li>
 * </ul>
 *
 * <p>All cases use the placeholder/warmup native-plan pattern from
 * {@link TestNativeSpeculativeTerminalState}: small synthetic DSP plans, no
 * model downloads, and a proposing MTP predictor so every step takes the
 * speculative single-token-commit path (verification + asl=1 re-run).
 */
public class TestMtpStopOrdering {
    private static final int WIDTH = 5;
    private static final int CACHE = 64;

    private enum Disagreement { NONE, VERIFY_EOS, RERUN_EOS, RERUN_STOP2, MULTI_STOP }

    /** (a) verify predicts EOS, rerun predicts normal: no premature stop allowed. */
    @Test
    public void testVerifyEosRerunNormalDoesNotStopOnSupersededToken() {
        try (SpecPlan plan = new SpecPlan(Disagreement.VERIFY_EOS)) {
            DecodeResult result = runDecode(plan, 4, 1, List.of(), new int[0]);
            assertEmitted(result, 4, new long[]{0, 0, 0, 0});
        }
    }

    /** (b) verify predicts normal, rerun emits EOS: termination is mandatory. */
    @Test
    public void testVerifyNormalRerunEosStopsOnEmittedEos() {
        try (SpecPlan plan = new SpecPlan(Disagreement.RERUN_EOS)) {
            DecodeResult result = runDecode(plan, 2, 1, List.of(), new int[0]);
            assertEmitted(result, 1, new long[]{1});
        }
    }

    /** Scalar stop id emitted by the rerun must terminate even though verify disagrees. */
    @Test
    public void testRerunStopTokenTerminatesEvenWhenVerifyDisagrees() {
        try (SpecPlan plan = new SpecPlan(Disagreement.RERUN_STOP2)) {
            DecodeResult result = runDecode(plan, 2, -1, List.of(new int[]{2}), new int[0]);
            assertEmitted(result, 1, new long[]{2});
        }
    }

    /** (c) multi-token stop suffix must never be assembled from a token the verify
     *  pass imagined but the rerun never emitted. Budget 4 exceeds the premature
     *  stop point so the stale ordering is observable as a short emission. */
    @Test
    public void testMultiTokenStopSequenceAcrossRerunReplacement() {
        try (SpecPlan plan = new SpecPlan(Disagreement.MULTI_STOP)) {
            DecodeResult result = runDecode(plan, 4, -1, List.of(new int[]{0, 2}), new int[0]);
            assertEmitted(result, 4, new long[]{0, 0, 0, 0});
        }
    }

    /** Control: without forced disagreement the loop must be unchanged. */
    @Test
    public void testControlNoDisagreementEmitsStableSequence() {
        try (SpecPlan plan = new SpecPlan(Disagreement.NONE)) {
            DecodeResult result = runDecode(plan, 3, -1, List.of(), new int[0]);
            assertEmitted(result, 3, new long[]{0, 0, 0});
        }
    }

    /** Continuation across session calls after a stop event: call 1 stops on the
     *  rerun-emitted EOS; a second call primed with that emitted suffix must
     *  refuse to generate (the stop matched at the session boundary). The stop
     *  sequence is 2 tokens wide so the trailer's history window
     *  (maxSeqLen - 1 = 1) retains the primed token [1]. */
    @Test
    public void testContinuationAcrossCallsAfterStopEvent() {
        try (SpecPlan plan = new SpecPlan(Disagreement.RERUN_EOS)) {
            DecodeResult first = runDecode(plan, 4, 1, List.of(new int[]{2, 0}), new int[0]);
            assertEmitted(first, 1, new long[]{1});

            DecodeResult second = runDecode(plan, 4, 1, List.of(new int[]{2, 0}), new int[]{1});
            assertEquals(0, second.tokenCount,
                    "stop primed from the preceding emitted suffix must terminate immediately");
            assertTrue(second.proposed == 0.0f, "no proposal work may run past a primed stop");
        }
    }

    private DecodeResult runDecode(SpecPlan plan, int budget, int eos, List<int[]> stops,
                                   int[] precedingTokens) {
        AutoregressiveDecode op = new AutoregressiveDecode(
                plan.embeddings, plan.table, plan.target.input("ids"), plan.target.input("mask"),
                plan.positions, null,
                plan.targetExecutor.getNativePlanHandle(), plan.targetExecutor.getCachedOpContext(),
                plan.targetExecutor.getCurrentPlan().getExternalInputKeys().length,
                plan.targetOutputs.size(),
                -1, -1, plan.target.ext("mask"), -1, plan.target.ext("ids"), plan.target.out("logits"),
                -1, plan.target.ext("position"), plan.target.ext("cache_position"),
                new int[0], new int[0],
                new int[]{plan.target.ext("gdn")}, new int[]{plan.target.out("gdn_next")},
                new int[]{plan.target.ext("conv")}, new int[]{plan.target.out("conv_next")},
                budget, eos, 0, 0, 0.0, 0, 0.0, 1.0, Set.of());
        op.withDecodePolicy(AutoregressiveDecode.DECODE_STRATEGY_SPECULATIVE,
                1, WIDTH, 1, 1, -1, 1, 1.0, 0.0, 0)
                .withSpeculativeDecoding(WIDTH - 1, AutoregressiveDecode.SPECULATOR_TYPE_MTP)
                .withActualSequenceLengthExtIdx(plan.target.ext("actual_length"));
        op.withStopSequences(stops, precedingTokens)
                .withMtpPlan(plan.predictor.input("ids"), plan.predictor.input("carry"),
                        plan.predictor.input("mask"), plan.predictor.input("position"),
                        plan.predictor.input("cache_position"),
                        new INDArray[]{plan.predictor.input("key"), plan.predictor.input("value")},
                        plan.predictorExecutor.getNativePlanHandle(),
                        plan.predictorExecutor.getCachedOpContext(),
                        plan.predictorExecutor.getCurrentPlan().getExternalInputKeys().length,
                        plan.predictorOutputs.size(),
                        plan.predictor.ext("ids"), plan.predictor.ext("carry"), plan.predictor.ext("mask"),
                        plan.predictor.ext("position"), plan.predictor.ext("cache_position"),
                        new int[]{plan.predictor.ext("key"), plan.predictor.ext("value")},
                        plan.predictor.out("logits"), plan.predictor.out("hidden"),
                        plan.target.out("hidden"));
        INDArray[] result = Nd4j.getExecutioner().exec(op);
        long[] tokens = new long[(int) result[1].getLong(0)];
        for (int i = 0; i < tokens.length; i++) {
            tokens[i] = result[0].getLong(i);
        }
        return new DecodeResult(tokens, (int) result[1].getLong(0), result[2],
                result[2].getFloat(7), result[2].getFloat(8), result[2].getFloat(9));
    }

    private void assertEmitted(DecodeResult result, int expectedCount, long[] expectedTokens) {
        assertEquals(expectedCount, result.tokenCount, "emitted token count");
        for (int i = 0; i < expectedTokens.length; i++) {
            assertEquals(expectedTokens[i], result.tokens[i], "emitted token at " + i);
        }
        assertTrue(result.proposed > 0.0f, "must exercise a proposing speculative step");
        assertTrue(result.specSteps >= 1.0f, "must exercise the speculative commit path");
        assertTrue(result.timing.getFloat(6) >= 0.0f, "repetition finish marker must not be set");
    }

    private static final class DecodeResult {
        private final long[] tokens;
        private final int tokenCount;
        private final INDArray timing;
        private final float proposed;
        private final float accepted;
        private final float specSteps;

        private DecodeResult(long[] tokens, int tokenCount, INDArray timing,
                             float proposed, float accepted, float specSteps) {
            this.tokens = tokens;
            this.tokenCount = tokenCount;
            this.timing = timing;
            this.proposed = proposed;
            this.accepted = accepted;
            this.specSteps = specSteps;
        }
    }

    /**
     * Target graph whose row-0 argmax is a deterministic function of
     * actual_sequence_length. Verify executes at asl=W, the authoritative rerun
     * at asl=1, which forces the chosen verify/rerun disagreement.
     */
    private static final class SpecPlan implements AutoCloseable {
        private final SameDiff targetGraph = SameDiff.create();
        private final SameDiff predictorGraph = SameDiff.create();
        private final Map<String, INDArray> targetInputs = new LinkedHashMap<>();
        private final Map<String, INDArray> predictorInputs = new LinkedHashMap<>();
        private final List<String> targetOutputs = new ArrayList<>();
        private final List<String> predictorOutputs = new ArrayList<>();
        private final DynamicShapePlanExecutor targetExecutor;
        private final DynamicShapePlanExecutor predictorExecutor;
        private final INDArray embeddings = Nd4j.zeros(DataType.FLOAT, 1, 1, 1);
        private final INDArray table = Nd4j.ones(DataType.FLOAT, 2, 1);
        private final INDArray positions = Nd4j.zeros(DataType.INT64, 1, 1);
        private final TinyTarget target;
        private final TinyPredictor predictor;

        private SpecPlan(Disagreement mode) {
            buildTarget(mode);
            buildPredictor();
            targetGraph.setDspAutoCompileEnabled(true);
            targetGraph.setDspNativeAutoCompileEnabled(true);
            predictorGraph.setDspAutoCompileEnabled(true);
            predictorGraph.setDspNativeAutoCompileEnabled(true);
            for (int i = 0; i < 8; i++) {
                targetGraph.output(targetInputs, targetOutputs.toArray(new String[0]));
                predictorGraph.output(predictorInputs, predictorOutputs.toArray(new String[0]));
            }
            targetExecutor = targetGraph.getOrCreateSession().getDynamicShapePlanExecutor();
            predictorExecutor = predictorGraph.getOrCreateSession().getDynamicShapePlanExecutor();
            assertNotNull(targetExecutor);
            assertNotNull(targetExecutor.getNativePlanHandle());
            assertFalse(targetExecutor.getNativePlanHandle().isNull());
            assertNotNull(targetExecutor.getCachedOpContext());
            assertFalse(targetExecutor.getCachedOpContext().isNull());
            assertNotNull(predictorExecutor);
            assertNotNull(predictorExecutor.getNativePlanHandle());
            assertFalse(predictorExecutor.getNativePlanHandle().isNull());
            target = new TinyTarget();
            predictor = new TinyPredictor();
        }

        private void buildTarget(Disagreement mode) {
            SDVariable ids = placeholder(targetGraph, targetInputs,
                    "ids", Nd4j.ones(DataType.INT64, 1, WIDTH));
            SDVariable mask = placeholder(targetGraph, targetInputs, "mask",
                    Nd4j.valueArrayOf(new long[]{1, 1, WIDTH, CACHE}, -Float.MAX_VALUE, DataType.FLOAT));
            SDVariable position = placeholder(targetGraph, targetInputs,
                    "position", Nd4j.zeros(DataType.INT64, 1));
            SDVariable cachePosition = placeholder(targetGraph, targetInputs,
                    "cache_position", Nd4j.zeros(DataType.INT64, 1));
            addOutput(targetGraph, targetOutputs, mask.add("mask_echo", 1));
            addOutput(targetGraph, targetOutputs, position.add("position_echo", 1));
            addOutput(targetGraph, targetOutputs, cachePosition.add("cache_echo", 1));

            // Row t of baseAll is t * [6, 1, 0]: argmax 0 for every token t >= 1.
            SDVariable tokenRows = ids.castTo(DataType.FLOAT).reshape(WIDTH, 1);
            SDVariable onesW = targetGraph.constant("onesW",
                    Nd4j.ones(DataType.FLOAT, 1, 3));
            SDVariable w0 = targetGraph.constant("w0",
                    Nd4j.createFromArray(6.0f, 1.0f, 0.0f).reshape(1, 3));
            SDVariable baseAll = tokenRows.mmul(onesW).mul(w0);
            SDVariable row0Base = baseAll.get(SDIndex.point(0, true), SDIndex.all());

            // The rerun discriminator: verification executes at asl=W (>= 1.5),
            // the authoritative single-token rerun at asl=1. The echo output
            // keeps actual_length a plan input even when the mode's delta is a
            // constant (NONE).
            SDVariable asl = placeholder(targetGraph, targetInputs,
                    "actual_length", Nd4j.ones(DataType.INT64, 1));
            addOutput(targetGraph, targetOutputs, asl.add("asl_echo", 1));
            SDVariable aslF = asl.castTo(DataType.FLOAT);
            SDVariable isScalar = aslF.lt(1.5).castTo(DataType.FLOAT);
            SDVariable isWide = aslF.gt(1.5).castTo(DataType.FLOAT);
            // Decode-step discriminator: the decode loop advances the position
            // scalar every step (prefill placeholder value is 0).
            SDVariable posScalar = position.castTo(DataType.FLOAT);

            SDVariable delta;
            switch (mode) {
                case NONE:
                    delta = targetGraph.constant("delta_none", Nd4j.zeros(DataType.FLOAT, 1, 3));
                    break;
                case VERIFY_EOS:
                    // Verify: t*W0 + [0,7,0] -> argmax 1 (EOS). Rerun: t*W0 -> argmax 0.
                    delta = isWide.mul(targetGraph.constant("delta_verify_eos",
                            Nd4j.createFromArray(0.0f, 7.0f, 0.0f).reshape(1, 3)));
                    break;
                case RERUN_EOS: {
                    // Verify: t*W0 -> argmax 0. Rerun: (t+1)*[1,6,0] -> argmax 1 (EOS).
                    SDVariable ones3 = targetGraph.constant("ones3",
                            Nd4j.ones(DataType.FLOAT, 1, 3));
                    SDVariable w1 = targetGraph.constant("w1",
                            Nd4j.createFromArray(1.0f, 6.0f, 0.0f).reshape(1, 3));
                    SDVariable flipped = tokenRows.add(1.0).mmul(ones3).mul(w1)
                            .get(SDIndex.point(0, true), SDIndex.all());
                    delta = isScalar.mul(flipped.sub(row0Base));
                    break;
                }
                case RERUN_STOP2: {
                    // Verify: t*W0 -> argmax 0. Rerun: (t+1)*[1,1,6] -> argmax 2.
                    SDVariable ones3b = targetGraph.constant("ones3b",
                            Nd4j.ones(DataType.FLOAT, 1, 3));
                    SDVariable w2 = targetGraph.constant("w2",
                            Nd4j.createFromArray(1.0f, 1.0f, 6.0f).reshape(1, 3));
                    SDVariable flipped = tokenRows.add(1.0).mmul(ones3b).mul(w2)
                            .get(SDIndex.point(0, true), SDIndex.all());
                    delta = isScalar.mul(flipped.sub(row0Base));
                    break;
                }
                case MULTI_STOP: {
                    // Verify (wide pass) predicts the phantom stop token 2 from
                    // position >= 2 on; the rerun (asl=1, authoritative
                    // emission) always predicts 0. The stale matcher sees
                    // [.., 0, 2, 2] and matches stop [0, 2] after emitting only
                    // zeros; the correct matcher must never fire.
                    SDVariable late = posScalar.gt(1.5).castTo(DataType.FLOAT);
                    delta = isWide.mul(late.mul(targetGraph.constant("delta_multi_stop",
                            Nd4j.createFromArray(0.0f, 0.0f, 70.0f).reshape(1, 3))));
                    break;
                }
                default:
                    throw new IllegalStateException(mode.name());
            }
            SDVariable row0Logits = row0Base.add(delta);
            SDVariable row0Out = targetGraph.reshape("logits_row0", row0Logits, 1, 1, 3);
            SDVariable stubRows = baseAll.get(SDIndex.interval(1, WIDTH), SDIndex.all());
            SDVariable stubOut = targetGraph.reshape("logits_stub", stubRows, 1, WIDTH - 1, 3);
            SDVariable logits = targetGraph.concat("logits", 1, row0Out, stubOut);
            addOutput(targetGraph, targetOutputs, logits);

            float[] rows = new float[WIDTH];
            for (int i = 0; i < WIDTH; i++) rows[i] = i;
            SDVariable offsets = targetGraph.constant("rows",
                    Nd4j.createFromArray(rows).reshape(1, WIDTH, 1));
            addOutput(targetGraph, targetOutputs,
                    tokenRows.reshape(1, WIDTH, 1).add("hidden", offsets));
            SDVariable length = placeholder(targetGraph, targetInputs,
                    "actual_length_echo", Nd4j.ones(DataType.INT64, 1)).castTo(DataType.FLOAT);
            addOutput(targetGraph, targetOutputs,
                    placeholder(targetGraph, targetInputs, "gdn", Nd4j.zeros(DataType.FLOAT, 3))
                            .add("gdn_next", length));
            addOutput(targetGraph, targetOutputs,
                    placeholder(targetGraph, targetInputs, "conv", Nd4j.zeros(DataType.FLOAT, 3))
                            .add("conv_next", length.mul(10)));
        }

        private void buildPredictor() {
            SDVariable ids = placeholder(predictorGraph, predictorInputs,
                    "ids", Nd4j.ones(DataType.INT64, 1, 1));
            SDVariable mask = placeholder(predictorGraph, predictorInputs, "mask",
                    Nd4j.valueArrayOf(new long[]{1, 1, 1, CACHE}, -Float.MAX_VALUE, DataType.FLOAT));
            SDVariable position = placeholder(predictorGraph, predictorInputs,
                    "position", Nd4j.zeros(DataType.INT64, 1));
            SDVariable cachePosition = placeholder(predictorGraph, predictorInputs,
                    "cache_position", Nd4j.zeros(DataType.INT64, 1));
            addOutput(predictorGraph, predictorOutputs, mask.add("mask_echo", 1));
            addOutput(predictorGraph, predictorOutputs, position.add("position_echo", 1));
            addOutput(predictorGraph, predictorOutputs, cachePosition.add("cache_echo", 1));
            SDVariable tokenRows = ids.castTo(DataType.FLOAT).reshape(1, 1);
            SDVariable weight = predictorGraph.constant("weight",
                    Nd4j.createFromArray(0.0f, 10.0f).reshape(1, 2));
            addOutput(predictorGraph, predictorOutputs,
                    predictorGraph.reshape("logits", predictorGraph.mmul(tokenRows, weight), 1, 1, 2));
            SDVariable carry = placeholder(predictorGraph, predictorInputs,
                    "carry", Nd4j.zeros(DataType.FLOAT, 1, 1, 1));
            addOutput(predictorGraph, predictorOutputs, carry.add("hidden", 1));
            addOutput(predictorGraph, predictorOutputs,
                    placeholder(predictorGraph, predictorInputs, "key",
                            Nd4j.zeros(DataType.FLOAT, 1, 1, CACHE, 1)).add("key_echo", 1));
            addOutput(predictorGraph, predictorOutputs,
                    placeholder(predictorGraph, predictorInputs, "value",
                            Nd4j.zeros(DataType.FLOAT, 1, 1, CACHE, 1)).add("value_echo", 1));
        }

        private SDVariable placeholder(SameDiff graph, Map<String, INDArray> inputs,
                                       String name, INDArray value) {
            inputs.put(name, value);
            return graph.placeHolder(name, value.dataType(), value.shape());
        }

        private void addOutput(SameDiff graph, List<String> outputs, SDVariable variable) {
            outputs.add(variable.name());
        }

        private class TinyTarget {
            private INDArray input(String name) { return targetInputs.get(name); }
            private int ext(String name) {
                int index = targetExecutor.findExternalInputIndex(name);
                assertTrue(index >= 0, "missing target input " + name);
                return index;
            }
            private int out(String name) {
                int index = targetExecutor.findOutputIndex(name);
                assertTrue(index >= 0, "missing target output " + name);
                return index;
            }
        }

        private class TinyPredictor {
            private INDArray input(String name) { return predictorInputs.get(name); }
            private int ext(String name) {
                int index = predictorExecutor.findExternalInputIndex(name);
                assertTrue(index >= 0, "missing predictor input " + name);
                return index;
            }
            private int out(String name) {
                int index = predictorExecutor.findOutputIndex(name);
                assertTrue(index >= 0, "missing predictor output " + name);
                return index;
            }
        }

        @Override public void close() {
            targetGraph.close();
            predictorGraph.close();
        }
    }
}
