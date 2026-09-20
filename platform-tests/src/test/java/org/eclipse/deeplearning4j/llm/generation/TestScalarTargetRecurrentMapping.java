/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * STAGE 1 REGRESSION (recurrent-state index domain, non-identity mapping):
 *
 * scalarInputToTarget[i] maps SCALAR external-input index i to TARGET
 * external-input index ti. gdnStateExtIndices/convStateExtIndices entries
 * are ALREADY target-domain indices. The old double-indexing
 *
 *     scalarInputToTarget[gdnStateExtIndices[s]] == scalarInputToTarget[i]
 *
 * re-mapped a target index through the scalar-indexed vector. With an
 * identity input ordering it accidentally agreed with the correct
 * comparison; with a NON-IDENTITY mapping it classified the wrong inputs as
 * recurrent, so the scalar plan's private GDN/conv snapshots were never
 * refreshed (frozen at capture-time values) - or the vector access ran out
 * of bounds when the target recurrent index exceeded the scalar count.
 *
 * The fixture FORCES a non-identity ordering: the window graph carries one
 * extra constant (wired value-preservingly into an echo output), shifting
 * the window plan's external-input indices relative to the width-1 scalar
 * binding captured from a separate graph. The test asserts the shift really
 * happened, then drives TWO proposing speculative steps and requires:
 *
 *   step 1: scalar snapshot refreshed from the window's pre-step state;
 *   step 2: scalar snapshot tracks the window's (further advanced)
 *           pre-step state - NOT the capture-time value (broken
 *           classification would freeze it) and NOT the post-commit window
 *           value (that would mean the snapshot ALIASES the window buffer,
 *           defeating the private-snapshot isolation).
 *
 * Private snapshot storage is enforced by the op's own validation
 * (scalar recurrent input must NOT share the target's DataBuffer).
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

public class TestScalarTargetRecurrentMapping {
    private static final int START = 3;
    private static final int CACHE = 16;

    @Test
    public void testNonIdentityMappingRefreshesPrivateSnapshotsAcrossSteps() {
        // Committed state bookkeeping (tiny target: gdn += consumed*2, conv += consumed*20;
        // every consumed row is pending token 1 / emitted token 1, delta = 2).
        //  warmup (greedy, budget 1): consumes the prefill correction -> gdn 5, conv 27.
        //  capture the scalar binding here (its private arrays hold 5 / 27).
        //  spec step 1: consumes 1 row -> window gdn 7, conv 47.
        //    snapshot refresh must read the PRE-step window state (5 / 27).
        //  spec step 2: consumes 1 row -> window gdn 9, conv 67.
        //    snapshot refresh must now read (7 / 47): advanced beyond the
        //    capture values (proves per-step refresh) but strictly behind the
        //    window's committed values (proves private storage, no aliasing).
        final float gdnAfterWarmup = 5, convAfterWarmup = 27;
        final float gdnAfterStep1 = 7, convAfterStep1 = 47;
        final float gdnAfterStep2 = 9, convAfterStep2 = 67;

        try (TinyPlan scalar = new TinyPlan(1, false, false);
             TinyPlan window = new TinyPlan(3, false, true);
             TinyPlan predictor = new TinyPlan(1, true, false)) {
            DynamicShapePlanExecutor s = scalar.executor;
            decodeOnce(scalar);
            s.setShapesFrozen(true);
            try (NativeExecutionBinding binding = s.captureNativeExecutionBinding()) {
                // The two graphs must NOT be input-order identical: the window
                // graph's extra constant shifts its external-input indices.
                int scalarGdn = binding.findExternalInputIndex("gdn");
                int windowGdn = window.ext("gdn");
                assertNotEquals(scalarGdn, windowGdn,
                        "fixture requires a non-identity scalar->target input ordering");
                assertNotEquals(binding.findExternalInputIndex("conv"), window.ext("conv"),
                        "fixture requires a non-identity scalar->target input ordering (conv)");

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
                        new int[]{windowGdn}, new int[]{window.out("gdn_next")},
                        new int[]{window.ext("conv")}, new int[]{window.out("conv_next")},
                        3, -1, 0, START, 0.0, 0, 0.0, 1.0, Set.of());
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
                    try {
                        // Correction + two speculative emissions (both steps
                        // force zero acceptance and consume exactly one row).
                        assertEquals(3, result[1].getLong(0), "correction plus two speculative emissions");
                        assertEquals(2, result[2].getFloat(9), 0.0f, "two proposing speculative steps");
                        assertEquals(0, result[2].getFloat(8), 0.0f, "forced zero acceptance every step");
                        // Budget 3 reserves the 3rd unit for the last emission,
                        // so step 2 proposes with K'=1 capacity: 2+1 proposals.
                        assertEquals(3, result[2].getFloat(7), 0.0f,
                                "step 1 proposes 2 drafts; step 2 proposes 1 (budget-capped)");

                        // Window committed state advanced through BOTH consumed rows.
                        assertEquals(gdnAfterStep2, window.input("gdn").getFloat(0), 0.0f,
                                "committed window GDN after two consumed rows");
                        assertEquals(convAfterStep2, window.input("conv").getFloat(0), 0.0f,
                                "committed window conv after two consumed rows");

                        // Scalar private snapshots: refreshed to the LAST rerun's
                        // pre-step window state - neither the capture-time value
                        // (frozen = broken index-domain classification) nor the
                        // post-commit window value (aliasing).
                        INDArray[] scalarInputs = binding.getExternalInputsSnapshot();
                        assertEquals(gdnAfterStep1, scalarInputs[scalarGdn].getFloat(0), 0.0f,
                                "scalar private GDN snapshot must be the step-2 PRE-step window state "
                                        + "(capture value would mean never refreshed; window value would mean aliasing)");
                        assertEquals(convAfterStep1, scalarInputs[binding.findExternalInputIndex("conv")].getFloat(0), 0.0f,
                                "scalar private conv snapshot must be the step-2 PRE-step window state");
                    } finally {
                        for (INDArray array : result) array.close();
                    }
                } finally {
                    Nd4j.getExecutioner().commit();
                    binding.completeNativeUse();
                }
            }
        }
    }

    /** One greedy warmup step consuming the prefill correction. */
    private void decodeOnce(TinyPlan target) {
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
                    1, -1, 0, START, 0.0, 0, 0.0, 1.0, Set.of());
            op.withDecodePolicy(AutoregressiveDecode.DECODE_STRATEGY_GREEDY,
                    1, target.width, 1, 1, -1, 1, 1.0, 0.0, 0)
                    .withActualSequenceLengthExtIdx(target.ext("actual_length"));
            INDArray[] result = Nd4j.getExecutioner().exec(op);
            try {
                assertEquals(1, result[1].getLong(0), "warmup emits the prefill correction");
            } finally {
                for (INDArray array : result) array.close();
            }
        }
    }

    /**
     * Tiny recurrent target/predictor. {@code extraConstant} inserts a value-
     * preserving extra constant op into the graph (window only), shifting the
     * plan's external-input ordering relative to graphs built without it.
     */
    private static final class TinyPlan implements AutoCloseable {
        private final SameDiff graph = SameDiff.create();
        private final Map<String, INDArray> inputs = new LinkedHashMap<>();
        private final List<String> outputs = new ArrayList<>();
        private final DynamicShapePlanExecutor executor;
        private final int width;

        private TinyPlan(int width, boolean predictor, boolean extraConstant) {
            this.width = width;
            SDVariable ids = placeholder("ids", Nd4j.ones(DataType.INT64, 1, width));
            INDArray mask = Nd4j.valueArrayOf(new long[]{1, 1, width, CACHE}, -Float.MAX_VALUE, DataType.FLOAT);
            for (int row = 0; row < width; row++)
                for (int k = 0; k < START; k++) mask.putScalar(new long[]{0, 0, row, k}, 0);
            // Position echo; the window graph folds an EXTRA CONSTANT into the
            // same output value (+0) so the output NAME set stays identical to
            // the scalar graph while its external-input ORDERING shifts by the
            // extra constant.
            SDVariable positionPh = placeholder("position",
                    Nd4j.valueArrayOf(new long[]{1}, START, DataType.INT64));
            SDVariable positionEcho = positionPh.add("position_pre_echo", 1);
            if (extraConstant) {
                SDVariable extra = graph.constant("extra_shift_constant", Nd4j.scalar(0.0f));
                positionEcho = positionEcho.add("position_echo", extra);
            } else {
                positionEcho = positionEcho.rename("position_echo");
            }
            output(positionEcho);
            output(placeholder("mask", mask).add("mask_echo", 1));
            output(placeholder("cache_position", Nd4j.valueArrayOf(new long[]{1}, START, DataType.INT64)).add("cache_echo", 1));
            SDVariable hidden;
            if (predictor) {
                hidden = placeholder("carry", Nd4j.valueArrayOf(new long[]{1, 1, 1}, 5, DataType.FLOAT)).add("hidden", 1);
                // BSHD cache layout contract (round-4 finding E): [batch, maxSeqLen, heads, dim].
                output(placeholder("key", Nd4j.zeros(DataType.FLOAT, 1, CACHE, 1, 1)).add("key_echo", 1));
                output(placeholder("value", Nd4j.zeros(DataType.FLOAT, 1, CACHE, 1, 1)).add("value_echo", 1));
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
            SDVariable rowsVar = predictor ? ids.castTo(DataType.FLOAT).mul(0).reshape(width, 1) : hidden.reshape(width, 1);
            SDVariable weights = graph.constant(Nd4j.ones(DataType.FLOAT, 1, 2));
            SDVariable bias = graph.constant(Nd4j.createFromArray(predictor ? 100.0f : 0.0f, predictor ? 0.0f : 100.0f));
            output(graph.reshape("logits", graph.mmul(rowsVar, weights).add(bias), 1, width, 2));
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
