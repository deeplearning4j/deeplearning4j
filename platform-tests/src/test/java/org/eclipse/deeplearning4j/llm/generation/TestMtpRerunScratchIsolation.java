/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * REVIEW ROUND 2 REGRESSIONS (two distinct contracts, per review):
 *
 * Token conventions in this fixture (vocab = 5):
 *   base pending input  = 1
 *   draft token         = 4 (the predictor's argmax)
 *   correction token    = 0
 *   rerun winner token  = 2 (only in the length-dependent variant)
 * All four IDs are DISTINCT so repeated tokens cannot hide position errors.
 *
 * TEST 1 - REPAIR-POSITION CORRECTNESS (consumed-input oracle):
 *   Draft p is written into target input row p+1; verification row r
 *   processes input r. An accepted prefix of m emitted tokens consumes
 *   target inputs [base, d0..d(m-2)]; the correction token is PENDING - it
 *   is never consumed and must NEVER be repaired into predictor KV.
 *   Predictor KV key = carry*100 + consumedInputToken.
 *   accepted=2, K=3: consumed = [1(base), 4(d0), 4(d1)], emitted = [4,4,0],
 *   carries = [7, 100, 101] -> keys [701, 10004, 10104]. key[2]=10104 (NOT
 *   10100 - that would mean the pending correction leaked into KV).
 *
 * TEST 2 - RERUN-PUBLICATION CORRECTNESS (verify-vs-rerun disagreement):
 *   Target logits DEPEND on actual_length: the verification pass runs at
 *   asl=WIDTH(4) and row 0's winner is draft 4; the authoritative rerun
 *   advances asl to consumedCount (2 < WIDTH) and row 0's winner becomes 2.
 *   The scalar refresh must flip emission to 2, and the predictor's pending
 *   input - a device copy taken inside setMtpNextInputCuda - must be 2 as
 *   well. The old stale-verification handoff published 4.
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

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class TestMtpRerunScratchIsolation {
    private static final int K = 3;
    private static final int WIDTH = K + 1;
    private static final int CACHE = 8;
    private static final int VOCAB = 5;
    private static final long BASE_TOKEN = 1;
    private static final long DRAFT_TOKEN = 4;
    private static final long CORRECTION_TOKEN = 0;
    private static final long RERUN_TOKEN = 2;

    @Test
    public void testRepairConsumesOnlyConsumedInputTokens() {
        // accepted=2: drafts 4,4 accepted; correction 0 pending.
        // Predictor consumed inputs: [1(base), 4, 4]; 0 must never enter KV.
        try (Plan target = target(2, 40.0f, 10.0f, false);
             Plan predictor = predictor()) {
            runCommit(target, predictor, result -> {
                assertEquals(3, result[1].getLong(0), "two accepted drafts plus correction");
                assertEquals(2, result[2].getFloat(8), 0.0f, "forced acceptance length");
                assertEquals(DRAFT_TOKEN, result[0].getLong(0), "emitted 0 = accepted draft");
                assertEquals(DRAFT_TOKEN, result[0].getLong(1), "emitted 1 = accepted draft");
                assertEquals(CORRECTION_TOKEN, result[0].getLong(2),
                        "emitted 2 = correction (pending, NOT consumed)");
                float[] carries = {7f, 100f, 101f};
                long[] consumedTokens = {BASE_TOKEN, DRAFT_TOKEN, DRAFT_TOKEN};
                for (int q = 0; q < 3; q++) {
                    float expectedKey = carries[q] * 100 + consumedTokens[q];
                    assertEquals(expectedKey,
                            predictor.input("key").getFloat(0, q, 0, 0), 0.0f,
                            "predictor KV row " + q + " = carry*100 + CONSUMED token "
                                    + consumedTokens[q] + " (pending correction must never enter KV)");
                    assertEquals(expectedKey + 0.5f,
                            predictor.input("value").getFloat(0, q, 0, 0), 0.0f,
                            "predictor KV value row " + q);
                }
                assertEquals(100 + 2, predictor.input("carry").getFloat(0), 0.0f,
                        "next-call carry = target hidden row consumedCount-1");
                assertEquals(3, predictor.input("position").getLong(0));
                assertEquals(3, predictor.input("cache_position").getLong(0));
                float rejectedBias = predictor.input("mask").getFloat(3);
                assertTrue(rejectedBias <= -1e9f,
                        "rejected proposal row must stay masked, got " + rejectedBias);
            });
        }
    }

    @Test
    public void testRerunWinnerPublicationReachesPredictor() {
        // accepted=1: verify (asl=2) row 0 winner = draft 4; scalar rerun
        // (asl=1) row 0 winner = 2. Emission, target pending input, and
        // predictor pending input must ALL be the finalized 2 - never the
        // superseded verification 4.
        try (Plan target = target(1, 40.0f, 10.0f, true);
             Plan predictor = predictor()) {
            runCommit(target, predictor, result -> {
                assertEquals(2, result[1].getLong(0), "refreshed emission plus correction");
                // Finalized-emission acceptance: the flipped draft was NOT
                // emitted, so the emitted-acceptance count is 0 even though
                // the verification pass agreed on it (the verification
                // agreement alone is not an emission fact).
                assertEquals(0.0f, result[2].getFloat(8), 0.0f,
                        "accepted-EMITTED must count only drafts actually emitted");
                assertEquals(RERUN_TOKEN, result[0].getLong(0),
                        "emitted token 0 must be the RERUN winner, not the verify winner");
                // Pending next input after consuming 2 rows = emitted[1] = the
                // correction token (per the reviewer's consumed/pending table:
                // the final emitted token stays pending, never consumed). The
                // stale-handoff defect was emitted[0] - that is now the
                // finalized rerun winner, not the superseded verify token.
                assertEquals(CORRECTION_TOKEN, predictor.input("ids").getLong(0),
                        "predictor pending input must be the pending correction, "
                                + "with emitted[0] taken from the finalized rerun winner");
                assertEquals(2, predictor.input("position").getLong(0));
                assertEquals(2, predictor.input("cache_position").getLong(0));
            });
        }
    }

    private interface CommitAssert {
        void check(INDArray[] result);
    }

    private void runCommit(Plan target, Plan predictor, CommitAssert asserter) {
        target.compile();
        predictor.compile();
        predictor.input("key").assign(-1);
        predictor.input("value").assign(-1);
        predictor.input("carry").assign(7);
        try (INDArray embeddings = Nd4j.zeros(DataType.FLOAT, 1, 1, 1);
             INDArray table = Nd4j.ones(DataType.FLOAT, 2, 1);
             INDArray positions = Nd4j.zeros(DataType.INT64, 1, 1)) {
            AutoregressiveDecode op = new AutoregressiveDecode(
                    embeddings, table, target.input("ids"), target.input("mask"), positions, null,
                    target.executor.getNativePlanHandle(), target.executor.getCachedOpContext(),
                    target.executor.getCurrentPlan().getExternalInputKeys().length, target.outputs.size(),
                    -1, -1, target.ext("mask"), -1, target.ext("ids"), target.out("logits"),
                    -1, target.ext("position"), target.ext("cache_position"),
                    new int[0], new int[0], new int[0], new int[0], new int[0], new int[0],
                    WIDTH, 0, 0, 0, 0.0, 0, 0.0, 1.0, Set.of());
            op.withDecodePolicy(AutoregressiveDecode.DECODE_STRATEGY_SPECULATIVE,
                            1, WIDTH, 1, 1, -1, 1, 1.0, 0.0, 0)
                    .withSpeculativeDecoding(K, AutoregressiveDecode.SPECULATOR_TYPE_MTP)
                    .withActualSequenceLengthExtIdx(target.ext("actual_length"))
                    .withMtpPlan(predictor.input("ids"), predictor.input("carry"),
                            predictor.input("mask"), predictor.input("position"),
                            predictor.input("cache_position"),
                            new INDArray[]{predictor.input("key"), predictor.input("value")},
                            predictor.executor.getNativePlanHandle(), predictor.executor.getCachedOpContext(),
                            predictor.executor.getCurrentPlan().getExternalInputKeys().length,
                            predictor.outputs.size(), predictor.ext("ids"), predictor.ext("carry"),
                            predictor.ext("mask"), predictor.ext("position"), predictor.ext("cache_position"),
                            new int[]{predictor.ext("key"), predictor.ext("value")},
                            predictor.out("logits"), predictor.out("hidden"), target.out("hidden"));
            INDArray[] result = Nd4j.getExecutioner().exec(op);
            try {
                asserter.check(result);
            } finally {
                for (INDArray array : result) array.close();
            }
        }
    }

    /**
     * Window target with VOCAB-class logits. Row r winner = DRAFT_TOKEN when
     * r &lt; accepted, CORRECTION_TOKEN otherwise (4x logit gap keeps the
     * argmax robust). When {@code lengthDependent}, row 0's winner flips to
     * RERUN_TOKEN when actual_length == 1 (the scalar rerun's width-one
     * geometry), while the asl&gt;=2 verification winners stay 4/0.
     */
    private static Plan target(int accepted, float matchLogit, float otherLogit,
                               boolean lengthDependent) {
        Plan p = new Plan();
        SDVariable ids = p.placeholder("ids", Nd4j.ones(DataType.INT64, 1, WIDTH));
        p.echo("mask", Nd4j.valueArrayOf(new long[]{1, 1, WIDTH, CACHE}, -Float.MAX_VALUE, DataType.FLOAT));
        p.echo("position", Nd4j.zeros(DataType.INT64, 1));
        p.echo("cache_position", Nd4j.zeros(DataType.INT64, 1));
        SDVariable length = p.placeholder("actual_length", Nd4j.ones(DataType.INT64, 1));
        p.output(length.add("actual_length_echo", 1));
        float[] logits = new float[WIDTH * VOCAB];
        float[] hidden = new float[WIDTH];
        for (int row = 0; row < WIDTH; row++) {
            logits[row * VOCAB + (int) DRAFT_TOKEN] = row < accepted ? matchLogit : otherLogit;
            logits[row * VOCAB + (int) CORRECTION_TOKEN] =
                    row < accepted ? otherLogit : matchLogit;
            hidden[row] = 100 + row;
        }
        SDVariable zero = ids.castTo(DataType.FLOAT).mul(0).reshape(1, WIDTH, 1);
        SDVariable raw = zero.add(p.graph.constant(
                Nd4j.createFromArray(logits).reshape(1, WIDTH, VOCAB)));
        SDVariable finalLogits = raw;
        if (lengthDependent) {
            // Bonus to RERUN_TOKEN class of row 0 when the recurrence length is
            // SHORTENED below the full verification window (asl < WIDTH): the
            // rerun advances asl to consumedCount (2 here) while the verify
            // pass ran at asl=WIDTH(4). Base row-0 winner is DRAFT_TOKEN at
            // matchLogit (40); bonus 100 flips to RERUN_TOKEN (10 -> 110).
            SDVariable shortened = length.castTo(DataType.FLOAT)
                    .lt(WIDTH).castTo(DataType.FLOAT).reshape(1, 1, 1);
            SDVariable bonus = shortened.mul(100);
            float[] selector = new float[WIDTH * VOCAB];
            selector[(int) RERUN_TOKEN] = 1.0f; // row 0, RERUN_TOKEN class
            SDVariable selectorVar = p.graph.constant(
                    Nd4j.createFromArray(selector).reshape(1, WIDTH, VOCAB));
            finalLogits = raw.add(bonus.mul(selectorVar));
        }
        p.output(p.graph.castTo("logits", finalLogits, DataType.FLOAT));
        p.output(zero.add("hidden",
                p.graph.constant(Nd4j.createFromArray(hidden).reshape(1, WIDTH, 1))));
        return p;
    }

    /**
     * Predictor proposing DRAFT_TOKEN with VOCAB-class logits, whose KV write
     * encodes BOTH the carry and the INPUT TOKEN: key = carry*100 + ids[0],
     * value = key + 0.5, written through the production
     * dotProductAttentionV2 cache path.
     */
    private static Plan predictor() {
        Plan p = new Plan();
        SDVariable ids = p.placeholder("ids", Nd4j.ones(DataType.INT64, 1, 1));
        SDVariable carry = p.placeholder("carry", Nd4j.valueArrayOf(new long[]{1, 1, 1}, 7, DataType.FLOAT));
        SDVariable mask = p.placeholder("mask",
                Nd4j.valueArrayOf(new long[]{1, 1, 1, CACHE}, -Float.MAX_VALUE, DataType.FLOAT));
        p.echo("position", Nd4j.zeros(DataType.INT64, 1));
        SDVariable position = p.placeholder("cache_position", Nd4j.zeros(DataType.INT64, 1));
        SDVariable key = p.placeholder("key", Nd4j.zeros(DataType.FLOAT, 1, CACHE, 1, 1));
        SDVariable value = p.placeholder("value", Nd4j.zeros(DataType.FLOAT, 1, CACHE, 1, 1));
        SDVariable tokenF = ids.reshape(1, 1, 1, 1).castTo(DataType.FLOAT);
        SDVariable k = carry.reshape(1, 1, 1, 1).mul(100).add(tokenF);
        SDVariable v = k.add(0.5);
        p.output(p.graph.nn().dotProductAttentionV2("attention", k, v, k, null, null,
                key, value, position, mask, 0.0, 0.0, false, false));
        p.output(carry.add("hidden", 1));
        // Draft proposal: DRAFT_TOKEN must win the argmax. One-hot logits on
        // class DRAFT_TOKEN (40 vs 0 elsewhere) keep the winner unambiguous.
        float[] bias = new float[VOCAB];
        bias[(int) DRAFT_TOKEN] = 40.0f;
        SDVariable zero = ids.castTo(DataType.FLOAT).mul(0).reshape(1, 1, 1);
        SDVariable rawLogits = zero.add(p.graph.constant(
                Nd4j.createFromArray(bias).reshape(1, 1, VOCAB)));
        p.output(p.graph.castTo("logits", rawLogits, DataType.FLOAT));
        return p;
    }

    private static final class Plan implements AutoCloseable {
        private final SameDiff graph = SameDiff.create();
        private final Map<String, INDArray> inputs = new LinkedHashMap<>();
        private final List<String> outputs = new ArrayList<>();
        private DynamicShapePlanExecutor executor;

        private SDVariable placeholder(String name, INDArray value) {
            inputs.put(name, value);
            return graph.placeHolder(name, value.dataType(), value.shape());
        }

        private void echo(String name, INDArray value) {
            output(placeholder(name, value).add(name + "_echo", 1));
        }

        private void output(SDVariable value) { outputs.add(value.name()); }
        private INDArray input(String name) { return inputs.get(name); }

        private void compile() {
            graph.setDspAutoCompileEnabled(true);
            graph.setDspNativeAutoCompileEnabled(true);
            for (int i = 0; i < 8; i++) graph.output(inputs, outputs.toArray(new String[0]));
            executor = graph.getOrCreateSession().getDynamicShapePlanExecutor();
            assertNotNull(executor);
            assertNotNull(executor.getNativePlanHandle());
            assertTrue(!executor.getNativePlanHandle().isNull(), "native plan required");
            assertNotNull(executor.getCachedOpContext());
        }

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
