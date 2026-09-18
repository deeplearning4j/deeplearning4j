/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * STAGE 2 REGRESSION (rerun scratch isolation, token-sensitive repair):
 *
 * The rerun-refresh argmax was written into specArgmaxDevice[0] BEFORE the
 * predictor retained-row repair loop consumed specArgmaxDevice[j] as the
 * repair input token. When consumedCount > 1 the repair therefore re-wrote
 * predictor KV rows with a token absent from the authoritative emitted
 * prefix (the rerun's diagnostic row-0 argmax rather than the committed
 * sequence).
 *
 * This fixture makes the defect OBSERVABLE: the predictor's KV row encodes
 * BOTH its input carry AND its input token (key = carry*100 + token), with
 * DISTINCT token IDs across the committed prefix (draft 2 accepted, draft 2
 * accepted, correction 0; K=3, acceptedDrafts=2, consumedCount=3). If repair
 * consumes any token other than the committed sequence's token at that row,
 * the stored KV value diverges from the expected encoding.
 *
 * The KV therefore proves the full dataflow: rerun argmax must go to a
 * scratch buffer, and repair must read exactly the committed token sequence.
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

    @Test
    public void testRepairConsumesOnlyCommittedTokensWithDistinctIds() {
        // Drafts are all token 1 (bias [0,10]). Window target row winners:
        // rows 0,1 validate drafts 0,1 as token 1 (match, 4x logit gap),
        // row 2's argmax is token 0 (mismatch) -> acceptedDrafts=2,
        // emitted = [1, 1, 0]. DISTINCT token values across the prefix.
        try (Plan target = target(2); Plan predictor = predictor()) {
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
                    assertEquals(3, result[1].getLong(0), "two accepted drafts plus correction");
                    assertEquals(2, result[2].getFloat(8), 0.0f, "forced acceptance length");
                    assertEquals(1, result[0].getLong(0), "emitted token 0 = accepted draft 1");
                    assertEquals(1, result[0].getLong(1), "emitted token 1 = accepted draft 1");
                    assertEquals(0, result[0].getLong(2), "emitted token 2 = correction 0");

                    // Predictor KV row q must encode its committed input token:
                    // key(q) = carry*100 + committedToken(q), value = key + 0.5.
                    // carry row 0 is the initial carry 7; rows q>0 use target
                    // hidden row q-1 (100 + q - 1 in this fixture).
                    // committedTokens = [1, 1, 0] (positions base+0..base+2).
                    long[] committedTokens = {1, 1, 0};
                    float[] carries = {7f, 100f, 101f};
                    for (int q = 0; q < 3; q++) {
                        float expectedKey = carries[q] * 100 + committedTokens[q];
                        assertEquals(expectedKey,
                                predictor.input("key").getFloat(0, q, 0, 0), 0.0f,
                                "predictor KV key row " + q + " must encode carry*100 + COMMITTED token "
                                        + committedTokens[q] + " (repair must never consume a rerun-scratch token)");
                        assertEquals(expectedKey + 0.5f,
                                predictor.input("value").getFloat(0, q, 0, 0), 0.0f,
                                "predictor KV value row " + q);
                    }
                    assertEquals(100 + 2, predictor.input("carry").getFloat(0), 0.0f,
                            "next-call carry must be target hidden row consumedCount-1 = 2");
                    assertEquals(3, predictor.input("position").getLong(0));
                    assertEquals(3, predictor.input("cache_position").getLong(0));
                    // Rejected proposal rows beyond the proposal write horizon
                    // must be INVISIBLE to the next step: not unmasked. The
                    // exact fill value is implementation detail (constructor
                    // rows keep their -MAX_VALUE fill; masked rows the op's
                    // FLOAT fill).
                    float rejectedBias = predictor.input("mask").getFloat(3);
                    assertTrue(rejectedBias <= -1e9f,
                            "rejected proposal row must stay masked, got " + rejectedBias);
                } finally {
                    for (INDArray array : result) array.close();
                }
            }
        }
    }

    /**
     * Window target with per-row token winners: rows [0, accepted) win with
     * token 2 (matching the drafts), row `accepted` wins with token 0
     * (mismatch -> correction). Hidden row r = 100 + r.
     */
    private static Plan target(int accepted) {
        Plan p = new Plan();
        SDVariable ids = p.placeholder("ids", Nd4j.ones(DataType.INT64, 1, WIDTH));
        p.echo("mask", Nd4j.valueArrayOf(new long[]{1, 1, WIDTH, CACHE}, -Float.MAX_VALUE, DataType.FLOAT));
        p.echo("position", Nd4j.zeros(DataType.INT64, 1));
        p.echo("cache_position", Nd4j.zeros(DataType.INT64, 1));
        p.echo("actual_length", Nd4j.ones(DataType.INT64, 1));
        float[] logits = new float[WIDTH * 2];
        float[] hidden = new float[WIDTH];
        for (int row = 0; row < WIDTH; row++) {
            // Winner token 1 on matching rows (index 1 of the 2-class logits),
            // winner token 0 on the first mismatching row. 4x logit gap keeps
            // the argmax robust to any BF16/fp reordering in the plan.
            logits[2 * row + (row < accepted ? 1 : 0)] = 40;
            logits[2 * row + (row < accepted ? 0 : 1)] = 10;
            hidden[row] = 100 + row;
        }
        SDVariable zeros = ids.castTo(DataType.FLOAT).mul(0).reshape(1, WIDTH, 1);
        SDVariable rawLogits = zeros.add(p.graph.constant(Nd4j.createFromArray(logits).reshape(1, WIDTH, 2)));
        p.output(p.graph.castTo("logits", rawLogits, DataType.FLOAT));
        p.output(zeros.add("hidden", p.graph.constant(Nd4j.createFromArray(hidden).reshape(1, WIDTH, 1))));
        return p;
    }

    /**
     * Predictor whose KV write encodes BOTH the carry and the INPUT TOKEN:
     * key = carry*100 + ids[0], value = key + 0.5. Written through the op's
     * production dotProductAttentionV2 cache write path (ids-dependent, so
     * the native plan retains the input; carry-dependent so input+carry are
     * both observable).
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
        SDVariable zero = ids.castTo(DataType.FLOAT).mul(0).reshape(1, 1, 1);
        SDVariable rawLogits = zero.add(p.graph.constant(Nd4j.createFromArray(0.0f, 10.0f).reshape(1, 1, 2)));
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
