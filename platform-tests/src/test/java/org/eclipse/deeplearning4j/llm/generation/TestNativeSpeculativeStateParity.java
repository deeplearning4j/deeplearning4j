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

/** Minimal T3b recurrent-state oracle. No actual attention/KV mutation is claimed here. */
public class TestNativeSpeculativeStateParity {
    private static final int START = 3;
    private static final int CACHE = 16;

    @Test
    public void testRejectThenContinueMatchesSameTarget() {
        // Teacher-forced reference prefix: pending input 1, then correction 1.
        // All target arms have the SAME function; predictor argmax 0 forces rejection.
        float gdn = 3, conv = 7;
        for (int token : new int[]{1, 1}) {
            gdn += token + 1;
            conv += 10 * (token + 1);
        }
        float nextHidden = gdn + conv + 11 * (1 + 1);
        float[] expectedLogits = {nextHidden, nextHidden + 100};
        try (TinyPlan scalar = new TinyPlan(1, false);
             TinyPlan window = new TinyPlan(3, false);
             TinyPlan rejected = new TinyPlan(3, false);
             TinyPlan predictor = new TinyPlan(1, true)) {
            float[] scalarLogits = decodeAndInspect(scalar, null, gdn, conv);
            assertArrayEquals(expectedLogits, scalarLogits, 0.0f, "independent scalar oracle");
            assertArrayEquals(scalarLogits, decodeAndInspect(window, null, gdn, conv), 0.0f,
                    "active-length-one window must implement the scalar target");
            assertArrayEquals(scalarLogits, decodeAndInspect(rejected, predictor, gdn, conv), 0.0f,
                    "rejected verification must not contaminate the subsequent target step");
        }
    }

    @Test
    public void testScalarTargetRerunMatchesGreedyGeometry() {
        // Same teacher-forced prefix as the reject oracle. The scalar binding must be a REAL
        // width-one plan (own context, width-1 ids) and the speculative op must execute the
        // rerun through it: remapped logits and committed state equal the independent scalar
        // target's outputs, and the predictor carry comes from the remapped hidden output.
        float gdn = 3, conv = 7;
        for (int token : new int[]{1, 1}) {
            gdn += token + 1;
            conv += 10 * (token + 1);
        }
        try (TinyPlan scalar = new TinyPlan(1, false);
             TinyPlan window = new TinyPlan(3, false);
             TinyPlan predictor = new TinyPlan(1, true)) {
            // The scalar binding must come from the REAL width-one plan: capture it from the
            // width-1 target after a completed execution, exactly like the pipeline does at warmup.
            // The window plan (width 3) then plays the verification substrate.
            DynamicShapePlanExecutor s = scalar.executor;
            decodeAndInspect(scalar, null, gdn, conv);
            s.setShapesFrozen(true);
            try (NativeExecutionBinding binding = s.captureNativeExecutionBinding()) {
                assertEquals(1, binding.getExternalInputsSnapshot()[binding.findExternalInputIndex("ids")].size(1),
                        "captured scalar binding must be width-one geometry");
                DynamicShapePlanExecutor t = window.executor;
                DynamicShapePlanExecutor p = predictor.executor;
                AutoregressiveDecode op = new AutoregressiveDecode(
                        Nd4j.zeros(DataType.FLOAT, 1, 1, 1), Nd4j.ones(DataType.FLOAT, 2, 1),
                        window.input("ids"), window.input("mask"),
                        Nd4j.valueArrayOf(new long[]{1, 1}, START, DataType.INT64),
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
                                predictor.out("logits"), predictor.out("hidden"), window.out("hidden"))
                        .withScalarTargetPlan(binding, window.executor.getCurrentPlan().getExternalInputKeys(),
                                window.outputs, "ids", "mask", "position", "cache_position",
                                "actual_length", "logits", "hidden");
                binding.beginNativeUse();
                try {
                    INDArray[] result = Nd4j.getExecutioner().exec(op);
                    assertEquals(2, result[1].getLong(0));
                    assertEquals(1, result[0].getLong(0), "emitted token must come from the scalar rerun");
                    assertEquals(0, result[2].getFloat(8), 0.0f, "forced zero acceptance");
                    assertEquals(gdn, window.input("gdn").getFloat(0), 0.0f,
                            "committed GDN must match the independent scalar advance");
                    assertEquals(conv, window.input("conv").getFloat(0), 0.0f,
                            "committed conv must match the independent scalar advance");
                    assertEquals(gdn + conv, predictor.input("carry").getFloat(0), 0.0f,
                            "predictor carry must come from the remapped scalar hidden");
                    assertPendingState(window);
                } finally {
                    Nd4j.getExecutioner().commit();
                    binding.completeNativeUse();
                }
            }
        }
    }

    private float[] decodeAndInspect(TinyPlan target, TinyPlan predictor, float gdn, float conv) {
        try (INDArray embeddings = Nd4j.zeros(DataType.FLOAT, 1, 1, 1);
             INDArray table = Nd4j.ones(DataType.FLOAT, 2, 1);
             INDArray positions = Nd4j.valueArrayOf(new long[]{1, 1}, START, DataType.INT64)) {
            DynamicShapePlanExecutor t = target.executor;
            AutoregressiveDecode op = new AutoregressiveDecode(
                    embeddings, table, target.input("ids"), target.input("mask"), positions,
                    null, t.getNativePlanHandle(), t.getCachedOpContext(),
                    t.getCurrentPlan().getExternalInputKeys().length, target.outputs.size(),
                    -1, -1, target.ext("mask"), -1, target.ext("ids"), target.out("logits"),
                    -1, target.ext("position"), target.ext("cache_position"),
                    new int[0], new int[0],
                    new int[]{target.ext("gdn")}, new int[]{target.out("gdn_next")},
                    new int[]{target.ext("conv")}, new int[]{target.out("conv_next")},
                    2, -1, 0, START, 0.0, 0, 0.0, 1.0, Set.of());
            op.withDecodePolicy(predictor == null ? AutoregressiveDecode.DECODE_STRATEGY_GREEDY
                            : AutoregressiveDecode.DECODE_STRATEGY_SPECULATIVE,
                    1, target.width, 1, 1, -1, 1, 1.0, 0.0, 0)
                    .withActualSequenceLengthExtIdx(target.ext("actual_length"));
            if (predictor != null) {
                DynamicShapePlanExecutor p = predictor.executor;
                op.withSpeculativeDecoding(target.width - 1, AutoregressiveDecode.SPECULATOR_TYPE_MTP)
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
            }
            INDArray[] result = Nd4j.getExecutioner().exec(op);
            assertEquals(2, result[1].getLong(0));
            assertEquals(1, result[0].getLong(0), "correction is the teacher-forced next input");
            assertEquals(1, result[0].getLong(1), "one subsequent token must be committed");
            // Budget two permits one proposal, rejection, then one non-proposing continuation.
            assertEquals(predictor == null ? 0 : 1, result[2].getFloat(7), 0.0f);
            assertEquals(0, result[2].getFloat(8), 0.0f, "must force zero acceptance");
            assertEquals(predictor == null ? 0 : 1, result[2].getFloat(9), 0.0f);
            assertEquals(gdn, target.input("gdn").getFloat(0), 0.0f);
            assertEquals(conv, target.input("conv").getFloat(0), 0.0f);
            assertEquals(1, target.input("actual_length").getLong(0));
            assertEquals(START + 2, positions.getLong(0));
            assertPendingState(target);
            if (predictor != null) {
                assertPendingState(predictor);
                assertEquals(gdn + conv, predictor.input("carry").getFloat(0), 0.0f);
            }
            // Probe the next logits with the still-pending token, without committing feedback.
            // Request the identical output set to retain the existing plan/cache lifecycle.
            INDArray logits = target.graph.output(target.inputs,
                    target.outputs.toArray(new String[0])).get("logits");
            return new float[]{logits.getFloat(0, 0, 0), logits.getFloat(0, 0, 1)};
        }
    }

    private static void assertPendingState(TinyPlan plan) {
        assertEquals(START + 2, plan.input("position").getLong(0));
        assertEquals(START + 2, plan.input("cache_position").getLong(0));
        assertEquals(1, plan.input("ids").getLong(0), "final emitted token remains pending");
        // Only row zero is the next active query; padded window rows are not committed state.
        for (int k = 0; k < CACHE; k++) {
            float bias = plan.input("mask").getFloat(0, 0, 0, k);
            if (k < START + 2) assertEquals(0.0f, bias, 0.0f);
            else assertTrue(bias <= -1e9f, "uncommitted position visible: " + k);
        }
    }

    private static final class TinyPlan implements AutoCloseable {
        private final SameDiff graph = SameDiff.create();
        private final Map<String, INDArray> inputs = new LinkedHashMap<>();
        private final List<String> outputs = new ArrayList<>();
        private final DynamicShapePlanExecutor executor;
        private final int width;

        private TinyPlan(int width, boolean predictor) {
            this.width = width;
            SDVariable ids = placeholder("ids", Nd4j.ones(DataType.INT64, 1, width));
            INDArray mask = Nd4j.valueArrayOf(new long[]{1, 1, width, CACHE}, -Float.MAX_VALUE, DataType.FLOAT);
            for (int row = 0; row < width; row++)
                for (int k = 0; k < START; k++) mask.putScalar(new long[]{0, 0, row, k}, 0);
            output(placeholder("mask", mask).add("mask_echo", 1));
            output(placeholder("position", Nd4j.valueArrayOf(new long[]{1}, START, DataType.INT64)).add("position_echo", 1));
            output(placeholder("cache_position", Nd4j.valueArrayOf(new long[]{1}, START, DataType.INT64)).add("cache_echo", 1));
            SDVariable hidden;
            if (predictor) {
                hidden = placeholder("carry", Nd4j.valueArrayOf(new long[]{1, 1, 1}, 5, DataType.FLOAT)).add("hidden", 1);
                output(placeholder("key", Nd4j.zeros(DataType.FLOAT, 1, 1, CACHE, 1)).add("key_echo", 1));
                output(placeholder("value", Nd4j.zeros(DataType.FLOAT, 1, 1, CACHE, 1)).add("value_echo", 1));
            } else {
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
                hidden = graph.reshape("hidden", graph.cumsum(delta, false, false, 1).mul(11).add(gdn).add(conv), 1, width, 1);
            }
            output(hidden);
            SDVariable rows = predictor ? ids.castTo(DataType.FLOAT).mul(0).reshape(width, 1) : hidden.reshape(width, 1);
            SDVariable weights = graph.constant(Nd4j.ones(DataType.FLOAT, 1, 2));
            SDVariable bias = graph.constant(Nd4j.createFromArray(predictor ? 100.0f : 0.0f, predictor ? 0.0f : 100.0f));
            output(graph.reshape("logits", graph.mmul(rows, weights).add(bias), 1, width, 2));
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
