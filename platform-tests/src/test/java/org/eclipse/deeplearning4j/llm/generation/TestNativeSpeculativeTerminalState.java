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

/** Small native-plan controller fixtures; no model downloads or Java decode loop.
 * Uses the placeholder/warmup pattern from DspExtInputTestSupport. Synthetic
 * recurrent outputs count consumed rows, so committing an overlong verification
 * pass is observable without relying on numerically sensitive model logits.
 *
 * <p>Multi-token terminal truncation (budget/stop inside one accepted batch) is
 * a CPU-contract assertion: ADR 0106 Phase 2b keeps CUDA on the single-token
 * commit until the CUDA full-state parity gate passes, so the budget/stop
 * multi-row cases here assert CPU semantics only (see 017d7ff7).</p>
 */
public class TestNativeSpeculativeTerminalState {
    private static final int WIDTH = 5;
    private static final int CACHE = 16;
    /**
     * Synthetic native target start position. The predictor row mapping is
     * r = target position - 1, so target base 1 maps to predictor base row 0.
     * Target-side committed positions are [BASE, BASE+emitted); the predictor's
     * pending rope/slot is P+emitted-1 == emitted (same value as before).
     */
    private static final int BASE_TARGET_POSITION = 1;

    @Test
    public void testAcceptedEosCommitsOnlyConsumedInputs() {
        checkTerminalPrefix(8, 1, List.of(), 1, 4, 1);
    }

    @Test
    public void testTokenBudgetCommitsOnlyConsumedInputs() {
        // ADR 0106 Phase 2b exit: parity proven (bc3f5c2a), CUDA multi-token restored.
        // Proposal capacity already reserves the bonus token: two drafts + bonus.
        checkTerminalPrefix(3, -1, List.of(), 3, 2, 2);
    }

    @Test
    public void testStopSequenceInsideAcceptedBatchCommitsOnlyConsumedInputs() {
        checkTerminalPrefix(8, -1, List.of(new int[]{1, 1}), 2, 4, 2);
    }

    static boolean isCudaBackend() {
        String exec = Nd4j.getExecutioner().getClass().getName().toLowerCase();
        String backend = Nd4j.getBackend().getClass().getName().toLowerCase();
        return exec.contains("cuda") || backend.contains("jcublas") || backend.contains("cuda");
    }

    private void checkTerminalPrefix(int budget, int eos, List<int[]> stops,
                                     int emitted, int proposed, int accepted) {
        try (TinyPlan target = new TinyPlan(WIDTH, false);
             TinyPlan predictor = new TinyPlan(1, true);
             INDArray embeddings = Nd4j.zeros(DataType.FLOAT, 1, 1, 1);
             INDArray table = Nd4j.ones(DataType.FLOAT, 2, 1);
             INDArray positions = Nd4j.valueArrayOf(new long[]{1, 1}, BASE_TARGET_POSITION)) {
            DynamicShapePlanExecutor t = target.executor;
            DynamicShapePlanExecutor p = predictor.executor;
            AutoregressiveDecode op = new AutoregressiveDecode(
                    embeddings, table, target.input("ids"), target.input("mask"), positions,
                    null, t.getNativePlanHandle(), t.getCachedOpContext(),
                    t.getCurrentPlan().getExternalInputKeys().length, target.outputs.size(),
                    -1, -1, target.ext("mask"), -1, target.ext("ids"), target.out("logits"),
                    -1, target.ext("position"), target.ext("cache_position"),
                    new int[0], new int[0],
                    new int[]{target.ext("gdn")}, new int[]{target.out("gdn_next")},
                    new int[]{target.ext("conv")}, new int[]{target.out("conv_next")},
                    budget, eos, 0, BASE_TARGET_POSITION, 0.0, 0, 0.0, 1.0, Set.of());
            op.withDecodePolicy(AutoregressiveDecode.DECODE_STRATEGY_SPECULATIVE,
                    1, WIDTH, 1, 1, -1, 1, 1.0, 0.0, 0)
                    .withSpeculativeDecoding(WIDTH - 1, AutoregressiveDecode.SPECULATOR_TYPE_MTP)
                    .withActualSequenceLengthExtIdx(target.ext("actual_length"))
                    .withStopSequences(stops)
                    .withMtpPlan(predictor.input("ids"), predictor.input("carry"),
                            predictor.input("mask"), predictor.input("position"),
                            predictor.input("cache_position"),
                            new INDArray[]{predictor.input("key"), predictor.input("value")},
                            p.getNativePlanHandle(), p.getCachedOpContext(),
                            p.getCurrentPlan().getExternalInputKeys().length, predictor.outputs.size(),
                            predictor.ext("ids"), predictor.ext("carry"), predictor.ext("mask"),
                            predictor.ext("position"), predictor.ext("cache_position"),
                            new int[]{predictor.ext("key"), predictor.ext("value")},
                            predictor.out("logits"), predictor.out("hidden"), target.out("hidden"));
            INDArray[] result = Nd4j.getExecutioner().exec(op);
            assertEquals(emitted, result[1].getLong(0));
            for (int i = 0; i < emitted; i++) assertEquals(1, result[0].getLong(i));
            assertEquals(proposed, result[2].getFloat(7), 0.0);
            assertEquals(accepted, result[2].getFloat(8), 0.0);
            assertEquals(1, result[2].getFloat(9), 0.0, "must exercise a proposing native step");
            assertEquals(emitted, target.input("actual_length").getLong(0));
            for (int i = 0; i < 3; i++) {
                assertEquals(emitted, target.input("gdn").getFloat(i), 0.0);
                assertEquals(10 * emitted, target.input("conv").getFloat(i), 0.0);
            }
            // Target-side positions advance from the synthetic base: committed
            // target positions are [BASE, BASE+emitted), so the next target
            // position (position ids / position / cache_position) is BASE+emitted.
            assertEquals(BASE_TARGET_POSITION + emitted, positions.getLong(0));
            assertEquals(BASE_TARGET_POSITION + emitted, target.input("position").getLong(0));
            assertEquals(BASE_TARGET_POSITION + emitted, target.input("cache_position").getLong(0));
            // Predictor pending rope/slot = P + emitted - 1 == emitted (packet 2).
            assertEquals(emitted, predictor.input("position").getLong(0));
            assertEquals(emitted, predictor.input("cache_position").getLong(0));
            assertEquals(1, target.input("ids").getLong(0));
            assertEquals(1, predictor.input("ids").getLong(0));
            // Target hidden row r is r+1. EOS is pending, so carry is row emitted-1.
            assertEquals(emitted, predictor.input("carry").getFloat(0), 0.0);
            // Causal visibility prefixes: the target's visible prefix runs from
            // slot 0 (causal history) through the last committed position
            // BASE+emitted-1; the predictor's retained rows are [0, emitted).
            assertCommittedMask(target.input("mask"), BASE_TARGET_POSITION + emitted);
            assertCommittedMask(predictor.input("mask"), emitted);
        }
    }

    private static void assertCommittedMask(INDArray mask, int committed) {
        float[] values;
        try (INDArray copy = mask.dup()) {
            values = copy.data().asFloat();
        }
        for (int i = 0; i < values.length; i++) {
            int slot = i % CACHE;
            if (slot < committed) assertEquals(0.0f, values[i], 0.0f);
            else assertTrue(values[i] <= -1e9f, "unconsumed KV visible at mask index " + i);
        }
    }

    private static final class TinyPlan implements AutoCloseable {
        private final SameDiff graph = SameDiff.create();
        private final Map<String, INDArray> inputs = new LinkedHashMap<>();
        private final List<String> outputs = new ArrayList<>();
        private final DynamicShapePlanExecutor executor;

        private TinyPlan(int width, boolean predictor) {
            SDVariable ids = placeholder("ids", Nd4j.ones(DataType.INT64, 1, width));
            SDVariable mask = placeholder("mask", Nd4j.valueArrayOf(
                    new long[]{1, 1, width, CACHE}, -Float.MAX_VALUE, DataType.FLOAT));
            SDVariable position = placeholder("position", Nd4j.zeros(DataType.INT64, 1));
            SDVariable cachePosition = placeholder("cache_position", Nd4j.zeros(DataType.INT64, 1));
            // Keep every mutable binding in the tiny native plan, without a model import.
            output(mask.add("mask_echo", 1));
            output(position.add("position_echo", 1));
            output(cachePosition.add("cache_echo", 1));
            SDVariable tokenRows = ids.castTo(DataType.FLOAT).reshape(width, 1);
            SDVariable weight = graph.constant("weight", Nd4j.createFromArray(0.0f, 10.0f).reshape(1, 2));
            output(graph.reshape("logits", graph.mmul(tokenRows, weight), 1, width, 2));
            if (predictor) {
                SDVariable carry = placeholder("carry", Nd4j.zeros(DataType.FLOAT, 1, 1, 1));
                output(carry.add("hidden", 1));
                output(placeholder("key", Nd4j.zeros(DataType.FLOAT, 1, 1, CACHE, 1)).add("key_echo", 1));
                output(placeholder("value", Nd4j.zeros(DataType.FLOAT, 1, 1, CACHE, 1)).add("value_echo", 1));
            } else {
                float[] rows = new float[width];
                for (int i = 0; i < width; i++) rows[i] = i;
                SDVariable offsets = graph.constant("rows", Nd4j.createFromArray(rows).reshape(1, width, 1));
                output(tokenRows.reshape(1, width, 1).add("hidden", offsets));
                SDVariable length = placeholder("actual_length", Nd4j.ones(DataType.INT64, 1)).castTo(DataType.FLOAT);
                output(placeholder("gdn", Nd4j.zeros(DataType.FLOAT, 3)).add("gdn_next", length));
                output(placeholder("conv", Nd4j.zeros(DataType.FLOAT, 3)).add("conv_next", length.mul(10)));
            }
            graph.setDspAutoCompileEnabled(true);
            graph.setDspNativeAutoCompileEnabled(true);
            for (int i = 0; i < 8; i++) graph.output(inputs, outputs.toArray(new String[0]));
            executor = graph.getOrCreateSession().getDynamicShapePlanExecutor();
            assertNotNull(executor);
            assertNotNull(executor.getNativePlanHandle());
            assertFalse(executor.getNativePlanHandle().isNull());
            assertNotNull(executor.getCachedOpContext());
            assertFalse(executor.getCachedOpContext().isNull());
        }

        private SDVariable placeholder(String name, INDArray value) {
            inputs.put(name, value);
            return graph.placeHolder(name, value.dataType(), value.shape());
        }

        private void output(SDVariable variable) { outputs.add(variable.name()); }
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
