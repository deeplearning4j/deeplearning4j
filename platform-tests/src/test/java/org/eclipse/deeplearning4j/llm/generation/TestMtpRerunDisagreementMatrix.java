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
import org.nd4j.autodiff.samediff.execution.DynamicShapePlanExecutor.NativeExecutionBinding;
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
 * Review round 5, patch-1/2 adversarial fixtures: the rerun disagreement
 * (A verify winner -> B rerun winner) exercised against the SCALAR-BINDING
 * path that the scratch fixture cannot reach (no scalar binding, empty
 * recurrent mappings there), for ONE-row and MULTI-row commits.
 *
 * <p>Contract under test (commit 9ab218cedb): the rerun disagreement gate
 * compares against the IMMUTABLE verification winner; a truncated multi-row
 * commit publishes the width-1 token into every consumer array (emission,
 * matcher, predictor pending input); and the scalar-boundary publication
 * replaces the recursive predictor carry with the target hidden row 0.</p>
 *
 * <p>Oracle (target base P=3, K=2, out-of-place recurrence
 * F(state, token) = state + consumed): verify row 0 winner = draft 1;
 * scalar rerun (asl=1) row 0 winner = 0. Emitted token 0 MUST be the rerun
 * winner 0, the predictor pending input MUST be 0 (not the superseded verify
 * draft 1), and the retained GDN/conv state MUST equal the independent
 * width-1 advance F(3,1)=4 / 10*F... i.e. the state that the width-1 pass
 * committed, not the multi-row value.</p>
 */
public class TestMtpRerunDisagreementMatrix {
    private static final int START = 3;
    private static final int CACHE = 16;
    /** Verify-pass row-0 argmax (token 0) - the superseded winner. */
    private static final long VERIFY_WINNER = 0;
    /** Rerun (asl=1) row-0 argmax (token 1) - the authoritative winner. */
    private static final long RERUN_WINNER = 1;

    /**
     * One-row disagreement with a scalar binding: the emitted token and the
     * predictor pending input must BOTH be the rerun winner, and the retained
     * recurrent state must equal the independent width-1 advance.
     */
    @Test
    public void testOneRowDisagreementWithScalarBinding() {
        runDisagreement(1, true);
    }

    /**
     * Same contract WITHOUT a scalar binding: the no-binding one-row flip
     * previously emitted the rerun winner but published the OLD verify token
     * to the predictor pending input (round-5 finding 1B).
     */
    @Test
    public void testOneRowDisagreementWithoutScalarBinding() {
        runDisagreement(1, false);
    }

    /**
     * ROUND 6 PATCH C - the REAL multi-row recovery transaction:
     * verification accepts draft A at row 0 and rejects row 1 (provisional
     * consumed m=2), the accepted-prefix RERUN runs at asl=2 and FLIPS row 0
     * from A to C, and the shortened recovery at asl=1 selects token AND
     * state together. The recurrence is TOKEN-DEPENDENT and out-of-place:
     * F(s; token) = 10*s + token, so the state after [base, A] is observably
     * different from the state after [base] alone - a stale multi-row commit
     * (state 10*(10*3+1)+1 = 311) cannot masquerade as the width-1 recovery
     * state (10*3+2 = 32).
     */
    @Test
    public void testMultiRowShortenRecoversWidthOneState() {
        try (TinyTokenWindow window = new TinyTokenWindow();
             TinyTokenPredictor predictor = new TinyTokenPredictor()) {
            runTokenSpeculative(window, predictor);
        }
    }

    private void runTokenSpeculative(TinyTokenWindow window, TinyTokenPredictor predictor) {
        try (INDArray embeddings = Nd4j.zeros(DataType.FLOAT, 1, 1, 1);
             INDArray table = Nd4j.ones(DataType.FLOAT, 2, 1);
             INDArray positions = Nd4j.valueArrayOf(new long[]{1, 1}, START, DataType.INT64)) {
            DynamicShapePlanExecutor t = window.executor;
            DynamicShapePlanExecutor p = predictor.executor;
            AutoregressiveDecode op = new AutoregressiveDecode(
                    embeddings, table, window.input("ids"), window.input("mask"), positions,
                    null, t.getNativePlanHandle(), t.getCachedOpContext(),
                    t.getCurrentPlan().getExternalInputKeys().length, window.outputs.size(),
                    -1, -1, window.ext("mask"), -1, window.ext("ids"), window.out("logits"),
                    -1, window.ext("position"), window.ext("cache_position"),
                    new int[0], new int[0],
                    new int[]{window.ext("gdn")}, new int[]{window.out("gdn_next")},
                    new int[]{window.ext("conv")}, new int[]{window.out("conv_next")},
                    2, -1, 0, START, 0.0, 0, 0.0, 1.0, Set.of());
            op.withDecodePolicy(AutoregressiveDecode.DECODE_STRATEGY_SPECULATIVE,
                            1, window.width, 1, 1, -1, 1, 1.0, 0.0, 0)
                    .withActualSequenceLengthExtIdx(window.ext("actual_length"))
                    .withSpeculativeDecoding(window.width - 1, AutoregressiveDecode.SPECULATOR_TYPE_MTP)
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
            INDArray[] result = Nd4j.getExecutioner().exec(op);
            try {
                // WAVE-2 TRUNCATE CONTRACT (review finding 3, commit e11e809280):
                // the verify pass accepts draft 1 (provisional m=2), the
                // accepted-prefix rerun at asl=2 flips row 0 to 0, and the
                // authoritative commit is the WIDTH-1 readout - the accepted
                // draft is never emitted (accepted-EMITTED=0), and each of the
                // two steps emits the asl=1 row-0 winner (token 2).
                assertEquals(0.0f, result[2].getFloat(8), 0.0f,
                        "accepted-EMITTED = 0: the truncate contract replaces the "
                                + "provisional commit with the width-1 readout; no draft "
                                + "is emitted from the flipped step");
                assertEquals(2, result[1].getLong(0),
                        "two steps, one authoritative token each (budget 2)");
                assertEquals(2, result[0].getLong(0),
                        "emitted token 0 = width-1 (asl=1) row-0 winner, never the "
                                + "superseded verify draft (1) or the asl=2 rerun winner (0)");
                assertEquals(2, result[0].getLong(1),
                        "emitted token 1 = width-1 row-0 winner again");
                // TOKEN-DEPENDENT state proof: the recurrent state advances by
                // the CONSUMED INPUT token (the width-1 pass's input row 0),
                // while the emitted token is that pass's OUTPUT and becomes the
                // next step's consumed input:
                //   step 0: consumes base 1 -> F(3;1)  = 10*3+1  = 31
                //   step 1: consumes emitted 2 -> F(31;2) = 10*31+2 = 312
                // A stale multi-row commit would show 10*31+1=311 (consume the
                // draft) or 10*31+0=310 (consume the asl=2 winner).
                assertEquals(312.0f, window.input("gdn").getFloat(0), 0.0f,
                        "retained GDN must equal the two consumed-input advances "
                                + "(3 -> 31 -> 312), not a draft-consuming 311 or an "
                                + "asl=2-winner 310");
                assertEquals(100f * 30201 + 2, window.input("conv").getFloat(0), 0.0f,
                        "retained conv follows its own recurrence F(s;tok)=100s+tok from "
                                + "302: 302 -> 30201 -> 3020102 - the consumed-input chain, "
                                + "not any multi-row value");
                // Predictor pending input = the finalized width-1 token.
                assertEquals(2, predictor.input("ids").getLong(0),
                        "predictor pending input must be the finalized width-1 token");
                // Pending geometry: target P = START + 2 committed = 5;
                // predictor pending row = P - 1 = 4.
                assertEquals(START + 2, window.input("position").getLong(0));
                assertEquals(START + 2, window.input("cache_position").getLong(0));
                assertEquals(START + 1, predictor.input("cache_position").getLong(0));
                assertEquals(START + 1, predictor.input("position").getLong(0));
            } finally {
                for (INDArray array : result) array.close();
            }
        }
    }

    private void runDisagreement(int acceptedPrefix, boolean useScalarBinding) {
        // Out-of-place recurrence: F(state; consumed) = (gdn+consumed, conv+10*consumed).
        // Each step the verify pass REJECTS the draft (accepted=0 -> consumedCount=1),
        // so the authoritative rerun executes width-1 at asl=1: consumed = ids+1 = 2.
        // Two steps: gdn 3->5->7, conv 7->27->47.
        float gdnAfter = 3 + 2 + 2;
        float convAfter = 7 + 20 + 20;
        try (TinyWindow window = new TinyWindow(3, acceptedPrefix);
             TinyWidthOne scalar = new TinyWidthOne();
             TinyPredictor predictor = new TinyPredictor()) {
            if (useScalarBinding) {
                // Capture the REAL width-one binding exactly like the pipeline warmup.
                DynamicShapePlanExecutor s = scalar.executor;
                scalar.warmDecode();
                s.setShapesFrozen(true);
                try (NativeExecutionBinding binding = s.captureNativeExecutionBinding()) {
                    assertEquals(1, binding.getExternalInputsSnapshot()[
                                    binding.findExternalInputIndex("ids")].size(1),
                            "captured scalar binding must be width-one geometry");
                    runSpeculative(window, predictor, binding, gdnAfter, convAfter);
                }
            } else {
                runSpeculative(window, predictor, null, gdnAfter, convAfter);
            }
        }
    }

    private void runSpeculative(TinyWindow window, TinyPredictor predictor,
                                NativeExecutionBinding binding,
                                float gdnAfter, float convAfter) {
        try (INDArray embeddings = Nd4j.zeros(DataType.FLOAT, 1, 1, 1);
             INDArray table = Nd4j.ones(DataType.FLOAT, 2, 1);
             INDArray positions = Nd4j.valueArrayOf(new long[]{1, 1}, START, DataType.INT64)) {
            DynamicShapePlanExecutor t = window.executor;
            DynamicShapePlanExecutor p = predictor.executor;
            AutoregressiveDecode op = new AutoregressiveDecode(
                    embeddings, table, window.input("ids"), window.input("mask"), positions,
                    null, t.getNativePlanHandle(), t.getCachedOpContext(),
                    t.getCurrentPlan().getExternalInputKeys().length, window.outputs.size(),
                    -1, -1, window.ext("mask"), -1, window.ext("ids"), window.out("logits"),
                    -1, window.ext("position"), window.ext("cache_position"),
                    new int[0], new int[0],
                    new int[]{window.ext("gdn")}, new int[]{window.out("gdn_next")},
                    new int[]{window.ext("conv")}, new int[]{window.out("conv_next")},
                    2, -1, 0, START, 0.0, 0, 0.0, 1.0, Set.of());
            op.withDecodePolicy(AutoregressiveDecode.DECODE_STRATEGY_SPECULATIVE,
                            1, window.width, 1, 1, -1, 1, 1.0, 0.0, 0)
                    .withActualSequenceLengthExtIdx(window.ext("actual_length"))
                    .withSpeculativeDecoding(window.width - 1, AutoregressiveDecode.SPECULATOR_TYPE_MTP)
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
            if (binding != null) {
                op.withScalarTargetPlan(binding, t.getCurrentPlan().getExternalInputKeys(),
                        window.outputs, "ids", "mask", "position", "cache_position",
                        "actual_length", "logits", "hidden");
                binding.beginNativeUse();
            }
            INDArray[] result;
            try {
                result = Nd4j.getExecutioner().exec(op);
            } finally {
                if (binding != null) {
                    Nd4j.getExecutioner().commit();
                    binding.completeNativeUse();
                }
            }
            try {
                // Budget 2: step 0 disagrees (rerun winner), step 1 scalar-continues.
                // Emitted token 0 must be the RERUN winner in BOTH arms - the
                // superseded verify draft (1) must never appear.
                assertEquals(2, result[1].getLong(0), "refreshed emission plus continuation");
                assertEquals(RERUN_WINNER, result[0].getLong(0),
                        "emitted token 0 must be the rerun winner, never the verify draft");
                // Predictor pending input must ALSO be the finalized rerun winner:
                // the no-binding arm previously published the superseded verify token.
                assertEquals(RERUN_WINNER, predictor.input("ids").getLong(0),
                        "predictor pending input must be the finalized rerun winner, "
                                + "not the superseded verification token");
                // Retained state must equal the width-1 rerun advances across
                // both steps - the SHORTEN/refresh replacement outputs, not
                // the multi-row verify values.
                assertEquals(gdnAfter, window.input("gdn").getFloat(0), 0.0f,
                        "retained GDN must equal the width-1 advance after recovery");
                assertEquals(convAfter, window.input("conv").getFloat(0), 0.0f,
                        "retained conv must equal the width-1 advance after recovery");
                // The carry comes from the width-1 pass's hidden output row 0 of
                // the LAST step: 11*(ids+1) + gdnMid + convMid where gdnMid=5,
                // convMid=27 are the mid-session states committed by step 0.
                float gdnMid = 3 + 2;
                float convMid = 7 + 20;
                assertEquals(11 * 2 + gdnMid + convMid,
                        predictor.input("carry").getFloat(0), 0.0f,
                        "predictor carry must be the width-1 target hidden, never the recursive self-hidden");
                // Pending positions: target 3+2=5; predictor row = target-1 = 4.
                assertEquals(START + 2, window.input("position").getLong(0));
                assertEquals(START + 2, window.input("cache_position").getLong(0));
                assertEquals(START + 1, predictor.input("cache_position").getLong(0));
                assertEquals(START + 1, predictor.input("position").getLong(0));
            } finally {
                for (INDArray array : result) array.close();
            }
        }
    }

    /**
     * Window target (width 3, K=2): verify row 0 winner = VERIFY_WINNER;
     * at asl=1 (scalar rerun geometry) row 0 flips to RERUN_WINNER.
     * Out-of-place recurrence F(state; consumed) via gdn/conv outputs.
     */
    private static final class TinyWindow implements AutoCloseable {
        private final SameDiff graph = SameDiff.create();
        private final Map<String, INDArray> inputs = new LinkedHashMap<>();
        private final List<String> outputs = new ArrayList<>();
        private final DynamicShapePlanExecutor executor;
        private final int width;

        private TinyWindow(int width, long acceptedPrefix) {
            this.width = width;
            SDVariable ids = placeholder("ids", Nd4j.ones(DataType.INT64, 1, width));
            INDArray mask = Nd4j.valueArrayOf(new long[]{1, 1, width, CACHE}, -Float.MAX_VALUE, DataType.FLOAT);
            for (int row = 0; row < width; row++)
                for (int k = 0; k < START; k++) mask.putScalar(new long[]{0, 0, row, k}, 0);
            output(placeholder("mask", mask).add("mask_echo", 1));
            output(placeholder("position", Nd4j.valueArrayOf(new long[]{1}, START, DataType.INT64)).add("position_echo", 1));
            output(placeholder("cache_position", Nd4j.valueArrayOf(new long[]{1}, START, DataType.INT64)).add("cache_echo", 1));
            SDVariable gdn = placeholder("gdn", Nd4j.createFromArray(3.0f));
            SDVariable conv = placeholder("conv", Nd4j.createFromArray(7.0f));
            SDVariable length = placeholder("actual_length", Nd4j.ones(DataType.INT64, 1)).castTo(DataType.FLOAT);
            SDVariable delta = ids.castTo(DataType.FLOAT).add(1);
            float[] rows = new float[width];
            for (int i = 0; i < width; i++) rows[i] = i;
            SDVariable active = graph.constant(Nd4j.createFromArray(rows).reshape(1, width)).lt(length).castTo(DataType.FLOAT);
            SDVariable consumed = delta.mul(active).sum();
            output(gdn.add("gdn_next", consumed));
            output(conv.add("conv_next", consumed.mul(10)));
            SDVariable hidden = graph.reshape("hidden",
                    graph.cumsum(delta, false, false, 1).mul(11).add(gdn).add(conv), 1, width, 1);
            output(hidden);
            // Logits: row 0 winner = VERIFY_WINNER (token 0) at asl>1 (the wide
            // verify pass); at asl==1 (the scalar/width-1 rerun geometry) row 0
            // flips to RERUN_WINNER. Rows >= 1 keep class 0.
            SDVariable hid = hidden.reshape(width, 1);
            SDVariable weights = graph.constant(Nd4j.createFromArray(0.0f, 0.0f).reshape(1, 2));
            // Base bias favors VERIFY_WINNER class always; the RERUN bonus fires
            // only when length == 1 (isOne), flipping row 0 to RERUN_WINNER.
            float[] baseBias = new float[2];
            baseBias[(int) VERIFY_WINNER] = 10.0f;
            SDVariable baseBiasVar = graph.constant(Nd4j.createFromArray(baseBias).reshape(1, 2));
            float[] rerunBias = new float[2];
            rerunBias[(int) RERUN_WINNER] = 100.0f;
            SDVariable rerunBiasVar = graph.constant(Nd4j.createFromArray(rerunBias).reshape(1, 2));
            SDVariable isOne = length.eq(1).castTo(DataType.FLOAT).reshape(1, 1);
            SDVariable logits = graph.reshape("logits",
                    graph.mmul(hid, weights).add(baseBiasVar).add(rerunBiasVar.mul(isOne)), 1, width, 2);
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
     * ROUND 6 PATCH C window target (width 3, K=2). TOKEN-DEPENDENT
     * out-of-place recurrence: gdn_next = 10*gdn + sum(consumed tokens);
     * conv_next = 100*gdn + sum(consumed tokens). Logits discriminate by
     * actual_length so each geometry has a distinct, deterministic winner:
     * asl=3 (verify): rows [1, 0, 0] -> draft 1 accepted, draft 2 rejected
     *                 (draft 3 would also lose to 0).
     * asl=2 (accepted-prefix rerun): rows [0, x] -> row-0 winner flips to 0
     *                 -> RERUN_TRUNCATE_COMMIT fires.
     * asl=1 (shortened width-1): rows [2] -> authoritative token 2.
     */
    private static final class TinyTokenWindow implements AutoCloseable {
        private final SameDiff graph = SameDiff.create();
        private final Map<String, INDArray> inputs = new LinkedHashMap<>();
        private final List<String> outputs = new ArrayList<>();
        private final DynamicShapePlanExecutor executor;
        private final int width = 3;

        private TinyTokenWindow() {
            SDVariable ids = placeholder("ids", Nd4j.ones(DataType.INT64, 1, width));
            INDArray mask = Nd4j.valueArrayOf(new long[]{1, 1, width, CACHE}, -Float.MAX_VALUE, DataType.FLOAT);
            for (int row = 0; row < width; row++)
                for (int k = 0; k < START; k++) mask.putScalar(new long[]{0, 0, row, k}, 0);
            output(placeholder("mask", mask).add("mask_echo", 1));
            output(placeholder("position", Nd4j.valueArrayOf(new long[]{1}, START, DataType.INT64)).add("position_echo", 1));
            output(placeholder("cache_position", Nd4j.valueArrayOf(new long[]{1}, START, DataType.INT64)).add("cache_echo", 1));
            SDVariable gdn = placeholder("gdn", Nd4j.createFromArray(3.0f));
            SDVariable conv = placeholder("conv", Nd4j.createFromArray(302.0f));
            SDVariable length = placeholder("actual_length", Nd4j.ones(DataType.INT64, 1)).castTo(DataType.FLOAT);
            SDVariable delta = ids.castTo(DataType.FLOAT).add(1);
            float[] rows = new float[width];
            for (int i = 0; i < width; i++) rows[i] = i;
            SDVariable active = graph.constant(Nd4j.createFromArray(rows).reshape(1, width)).lt(length).castTo(DataType.FLOAT);
            SDVariable consumedTokens = ids.castTo(DataType.FLOAT).mul(active).sum();
            SDVariable consumedCount = active.sum();
            // TOKEN-DEPENDENT state: F(s; tokens) = 10*s + sum(tokens).
            output(gdn.mul(10).add("gdn_next", consumedTokens));
            output(conv.mul(100).add("conv_next", consumedTokens));
            SDVariable hidden = graph.reshape("hidden",
                    graph.cumsum(delta, false, false, 1).mul(11).add(gdn), 1, width, 1);
            output(hidden);
            // ROUND 6 PATCH C logits - ROW-DEPENDENT winners + length gates:
            //   asl=3 (verify):    row0 winner=1 (accept draft 1),
            //                      row1 winner=0 (reject draft 2)
            //   asl=2 (rerun):     row0 flips to 0 -> RERUN_TRUNCATE_COMMIT
            //   asl=1 (width-1):   row0 winner=2 -> authoritative correction
            // Constants are per-row/per-class; the length gates only touch row 0.
            SDVariable isLE2 = length.lt(3).castTo(DataType.FLOAT).reshape(1, 1, 1);
            SDVariable isOne = length.eq(1).castTo(DataType.FLOAT).reshape(1, 1, 1);
            float[] logitsBase = new float[width * 3];
            logitsBase[0 * 3 + 1] = 10.0f;  // row 0 -> token 1 (accept draft 1)
            logitsBase[1 * 3 + 0] = 10.0f;  // row 1 -> token 0 (reject draft 2)
            logitsBase[2 * 3 + 1] = 10.0f;  // row 2 (padding row, unused)
            float[] gate0 = new float[width * 3];
            gate0[0 * 3 + 0] = 1.0f;        // row 0, class 0: +100*isLE2 -> flip to 0
            float[] gate1 = new float[width * 3];
            gate1[0 * 3 + 2] = 1.0f;        // row 0, class 2: +200*isOne -> width-1 token 2
            SDVariable rawLogits = graph.constant(Nd4j.createFromArray(logitsBase).reshape(1, width, 3))
                    .add(graph.constant(Nd4j.createFromArray(gate0).reshape(1, width, 3)).mul(isLE2.mul(100)))
                    .add(graph.constant(Nd4j.createFromArray(gate1).reshape(1, width, 3)).mul(isOne.mul(200)));
            SDVariable logitsVar = graph.identity("logits", rawLogits);
            output(logitsVar);
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
     * ROUND 6 PATCH C predictor: proposes draft 1 (accepted at verify row 0)
     * then draft 3 (rejected at verify row 1). BSHD KV layout.
     */
    private static final class TinyTokenPredictor implements AutoCloseable {
        private final SameDiff graph = SameDiff.create();
        private final Map<String, INDArray> inputs = new LinkedHashMap<>();
        private final List<String> outputs = new ArrayList<>();
        private final DynamicShapePlanExecutor executor;

        private TinyTokenPredictor() {
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
            SDVariable hidden = carry.add("hidden", 1);
            output(hidden);
            // CHAINED DRAFTS: each proposal call consumes the previous draft and
            // self-propagates its own hidden as the next carry. Encoding the
            // class bonus as a function of the carry makes the draft sequence
            // deterministic within the 3-token vocabulary:
            //   call 0 (carry 5): bias 60 at class 1 -> draft 1
            //   call 1 (carry 6): bias 60 at class 2 -> draft 2
            // The verify pass (asl=3) then accepts draft 1 (row-0 winner 1) and
            // rejects draft 2 (row-1 winner 0). The window's token-2 bonus fires
            // only at asl=1, so draft 2 never matches a verify row.
            SDVariable isCarry5 = carry.reshape(1).eq(5).castTo(DataType.FLOAT).reshape(1, 1, 1);
            SDVariable isCarry6 = carry.reshape(1).eq(6).castTo(DataType.FLOAT).reshape(1, 1, 1);
            float[] biasAlways = new float[3];
            biasAlways[1] = 10.0f;
            float[] biasC5 = new float[3];
            biasC5[1] = 60.0f;
            float[] biasC6 = new float[3];
            biasC6[2] = 60.0f;
            SDVariable zero = ids.castTo(DataType.FLOAT).mul(0).reshape(1, 1, 1);
            SDVariable rawLogits = zero
                    .add(graph.constant(Nd4j.createFromArray(biasAlways).reshape(1, 1, 3)))
                    .add(graph.constant(Nd4j.createFromArray(biasC5).reshape(1, 1, 3)).mul(isCarry5))
                    .add(graph.constant(Nd4j.createFromArray(biasC6).reshape(1, 1, 3)).mul(isCarry6));
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

    /** Real width-one target for the scalar binding (same recurrence family). */
    private static final class TinyWidthOne implements AutoCloseable {
        private final SameDiff graph = SameDiff.create();
        private final Map<String, INDArray> inputs = new LinkedHashMap<>();
        private final List<String> outputs = new ArrayList<>();
        private final DynamicShapePlanExecutor executor;

        private TinyWidthOne() {
            SDVariable ids = placeholder("ids", Nd4j.ones(DataType.INT64, 1, 1));
            INDArray mask = Nd4j.valueArrayOf(new long[]{1, 1, 1, CACHE}, -Float.MAX_VALUE, DataType.FLOAT);
            for (int k = 0; k < START; k++) mask.putScalar(new long[]{0, 0, 0, k}, 0);
            output(placeholder("mask", mask).add("mask_echo", 1));
            output(placeholder("position", Nd4j.valueArrayOf(new long[]{1}, START, DataType.INT64)).add("position_echo", 1));
            output(placeholder("cache_position", Nd4j.valueArrayOf(new long[]{1}, START, DataType.INT64)).add("cache_echo", 1));
            SDVariable gdn = placeholder("gdn", Nd4j.createFromArray(3.0f));
            SDVariable conv = placeholder("conv", Nd4j.createFromArray(7.0f));
            SDVariable length = placeholder("actual_length", Nd4j.ones(DataType.INT64, 1)).castTo(DataType.FLOAT);
            SDVariable delta = ids.castTo(DataType.FLOAT).add(1);
            SDVariable active = graph.constant(Nd4j.createFromArray(0.0f).reshape(1, 1)).lt(length).castTo(DataType.FLOAT);
            SDVariable consumed = delta.mul(active).sum();
            output(gdn.add("gdn_next", consumed));
            output(conv.add("conv_next", consumed.mul(10)));
            SDVariable hidden = graph.reshape("hidden",
                    graph.cumsum(delta, false, false, 1).mul(11).add(gdn).add(conv), 1, 1, 1);
            output(hidden);
            SDVariable hid = hidden.reshape(1, 1);
            SDVariable weights = graph.constant(Nd4j.createFromArray(0.0f, 0.0f).reshape(1, 2));
            // Same discriminator as the window graph: at asl==1 the row-0 winner
            // flips to RERUN_WINNER (matching the width-1 scalar rerun geometry).
            SDVariable baseBiasVar = graph.constant(Nd4j.createFromArray(
                    new float[]{10.0f, 0.0f}).reshape(1, 2));
            SDVariable rerunBiasVar = graph.constant(Nd4j.createFromArray(
                    new float[]{0.0f, 100.0f}).reshape(1, 2));
            SDVariable isOneS = length.eq(1).castTo(DataType.FLOAT).reshape(1, 1);
            SDVariable logits = graph.reshape("logits",
                    graph.mmul(hid, weights).add(baseBiasVar).add(rerunBiasVar.mul(isOneS)), 1, 1, 2);
            output(logits);
            graph.setDspAutoCompileEnabled(true);
            graph.setDspNativeAutoCompileEnabled(true);
            for (int i = 0; i < 8; i++) graph.output(inputs, outputs.toArray(new String[0]));
            executor = graph.getOrCreateSession().getDynamicShapePlanExecutor();
            assertNotNull(executor);
            assertNotNull(executor.getNativePlanHandle());
            assertFalse(executor.getNativePlanHandle().isNull());
        }

        /** Complete one decode so the binding captures a valid post-execution state. */
        private void warmDecode() {
            try (INDArray embeddings = Nd4j.zeros(DataType.FLOAT, 1, 1, 1);
                 INDArray table = Nd4j.ones(DataType.FLOAT, 2, 1);
                 INDArray positions = Nd4j.valueArrayOf(new long[]{1, 1}, START, DataType.INT64)) {
                AutoregressiveDecode op = new AutoregressiveDecode(
                        embeddings, table, input("ids"), input("mask"), positions,
                        null, executor.getNativePlanHandle(), executor.getCachedOpContext(),
                        executor.getCurrentPlan().getExternalInputKeys().length, outputs.size(),
                        -1, -1, ext("mask"), -1, ext("ids"), out("logits"),
                        -1, ext("position"), ext("cache_position"),
                        new int[0], new int[0],
                        new int[]{ext("gdn")}, new int[]{out("gdn_next")},
                        new int[]{ext("conv")}, new int[]{out("conv_next")},
                        1, -1, 0, START, 0.0, 0, 0.0, 1.0, Set.of());
                op.withDecodePolicy(AutoregressiveDecode.DECODE_STRATEGY_GREEDY,
                        1, 1, 1, 1, -1, 1, 1.0, 0.0, 0)
                        .withActualSequenceLengthExtIdx(ext("actual_length"));
                INDArray[] result = Nd4j.getExecutioner().exec(op);
                for (INDArray array : result) array.close();
            }
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

    /** Predictor proposing draft 1 (the superseded verify winner), BSHD KV layout. */
    private static final class TinyPredictor implements AutoCloseable {
        private final SameDiff graph = SameDiff.create();
        private final Map<String, INDArray> inputs = new LinkedHashMap<>();
        private final List<String> outputs = new ArrayList<>();
        private final DynamicShapePlanExecutor executor;

        private TinyPredictor() {
            SDVariable ids = placeholder("ids", Nd4j.ones(DataType.INT64, 1, 1));
            SDVariable carry = placeholder("carry", Nd4j.valueArrayOf(new long[]{1, 1, 1}, 5, DataType.FLOAT));
            SDVariable mask = placeholder("mask",
                    Nd4j.valueArrayOf(new long[]{1, 1, 1, CACHE}, -Float.MAX_VALUE, DataType.FLOAT));
            // position/cache_position must be plan EXTERNAL inputs: give both an
            // echo output consumer (placeholders without a downstream op are
            // dropped from the compiled external-input list).
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
            // Draft = RERUN_WINNER (1). The verify pass (asl>1) argmaxes to
            // VERIFY_WINNER (0), so the draft is REJECTED - accepted=0 keeps
            // consumedCount at 1 and forces the authoritative rerun to execute
            // width-1, where the flip to RERUN_WINNER is observable.
            float[] bias = new float[2];
            bias[(int) RERUN_WINNER] = 40.0f;
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
