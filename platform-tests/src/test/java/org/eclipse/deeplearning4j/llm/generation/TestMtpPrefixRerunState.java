/*
 * SPDX-License-Identifier: Apache-2.0
 */
package org.eclipse.deeplearning4j.llm.generation;

import org.bytedeco.javacpp.Pointer;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlanExecutor;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlanExecutor.NativeExecutionBinding;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.custom.AutoregressiveDecode;
import org.nd4j.linalg.api.ops.impl.transforms.custom.CausalConv1d;
import org.nd4j.linalg.api.ops.impl.transforms.custom.GatedDeltaRule;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.api.shape.options.ArrayOptionsHelper;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.OpaqueContext;
import org.nd4j.nativeblas.OpaqueDataBuffer;
import org.nd4j.nativeblas.OpaqueNDArray;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Model-free native MTP transaction: physical W5 verify -> restored accepted-prefix
 * rerun, with real causal_conv1d and gated_delta_rule state feedback, target KV,
 * target hidden carry and predictor KV repair. EOS terminates the FIRST commit;
 * a budget of five permits all four proposals even when only one is accepted.
 *
 * <p>All arithmetic is FLOAT32. One channel/head is intentional: convolution
 * weights [1/4, 1/2, 1], q=k=1, beta=1/2, gate=0 give the sequential recurrence
 * s[t] = (s[t-1] + conv[t])/2, NOT a summed-token proxy. A Java scalar loop is
 * the independent numerical oracle. Class-dependent hidden contributions make
 * every active-row logit observable, while large fixed biases force acceptance.
 * A zero attention query gives an independently calculable mean of visible KV
 * values; hidden = recurrent output + attention, so KV visibility affects logits.
 *
 * <p>This is not a Qwen reproduction or proof of CUDA graph capture. It does not
 * exercise scalar-plan handoff, quantized weights, or disagreement-triggered
 * double rollback within ONE transaction. W4/W2/W1 are separate transactions
 * on the same plans/buffers, each restored from the same saved input snapshot.
 * Rejected KV rows are scratch (attention physically writes W5); assert every
 * retained row and the terminal mask, not a promise that scratch is erased.
 */
public class TestMtpPrefixRerunState {
    private static final int WIDTH = 5;
    private static final int CACHE = 8;
    private static final int VOCAB = 5;
    private static final int START = 1;
    private static final int BASE = 1;
    private static final int DRAFT = 4;
    private static final int EOS = 0;
    private static final float SENTINEL = -7f;
    // FP32 attention divides by up to five, then predictor repair scales carry by eight.
    private static final double EPS = 5e-5;

    @Test
    public void testAcceptedPrefixesRepeatFromSameSnapshot() {
        try (Plan target = target(); Plan predictor = predictor()) {
            target.compile();
            predictor.compile();
            reset(target, predictor);
            try (Snapshot targetBefore = new Snapshot(target);
                 Snapshot predictorBefore = new Snapshot(predictor)) {
                // Same handles, same allocation addresses, including across shrinking prefixes.
                long handle = target.executor.getNativePlanHandle().address();
                for (int consumed : new int[]{4, 4, 2, 2, 1, 1}) {
                    targetBefore.restore(target);
                    predictorBefore.restore(predictor);
                    target.input("accepted").assign(consumed - 1);
                    target.binding.beginNativeUse();
                    try {
                        predictor.binding.beginNativeUse();
                        try {
                            runTransaction(target, predictor, consumed);
                        } finally {
                            Nd4j.getExecutioner().commit();
                            predictor.binding.completeNativeUse();
                        }
                    } finally {
                        target.binding.completeNativeUse();
                    }
                    assertEquals(handle, target.executor.getNativePlanHandle().address(),
                            "must retain the physical-W5 plan across transactions");
                }
            }
        }
    }

    private static void runTransaction(Plan target, Plan predictor, int consumed) {
        target.assertInputBinding("conv");
        target.assertInputBinding("gdn");
        try (INDArray embeddings = Nd4j.zeros(DataType.FLOAT, 1, 1, 1);
             INDArray table = Nd4j.ones(DataType.FLOAT, VOCAB, 1);
             INDArray positions = Nd4j.valueArrayOf(new long[]{1, 1}, START, DataType.INT64)) {
            AutoregressiveDecode op = new AutoregressiveDecode(
                    embeddings, table, target.input("ids"), target.input("mask"), positions,
                    new INDArray[]{target.input("key"), target.input("value")},
                    target.binding.getPlanHandle(), target.binding.getContextHandle(),
                    target.binding.getInputCount(), target.binding.getOutputCount(),
                    -1, -1, target.ext("mask"), -1, target.ext("ids"), target.out("logits"),
                    -1, target.ext("position"), target.ext("cache_position"),
                    new int[]{target.ext("key"), target.ext("value")}, new int[0],
                    new int[]{target.ext("gdn")}, new int[]{target.out("gdn_next")},
                    new int[]{target.ext("conv")}, new int[]{target.out("conv_next")},
                    WIDTH, EOS, 1, START, 0.0, 0, 0.0, 1.0, Set.of());
            op.withDecodePolicy(AutoregressiveDecode.DECODE_STRATEGY_SPECULATIVE,
                            1, WIDTH, 1, 1, -1, 1, 1.0, 0.0, 0)
                    .withActualSequenceLengthExtIdx(target.ext("actual_length"))
                    .withSpeculativeDecoding(WIDTH - 1, AutoregressiveDecode.SPECULATOR_TYPE_MTP)
                    .withMtpPlan(predictor.input("ids"), predictor.input("carry"),
                            predictor.input("mask"), predictor.input("position"),
                            predictor.input("cache_position"),
                            new INDArray[]{predictor.input("key"), predictor.input("value")},
                            predictor.binding.getPlanHandle(), predictor.binding.getContextHandle(),
                            predictor.binding.getInputCount(),
                            predictor.binding.getOutputCount(), predictor.ext("ids"), predictor.ext("carry"),
                            predictor.ext("mask"), predictor.ext("position"), predictor.ext("cache_position"),
                            new int[]{predictor.ext("key"), predictor.ext("value")},
                            predictor.out("logits"), predictor.out("hidden"), target.out("hidden"));
            INDArray[] result = Nd4j.getExecutioner().exec(op);
            try {
                String label = "consumed=" + consumed;
                assertEquals(consumed, result[1].getLong(0), label + ": one EOS-terminated transaction");
                assertEquals(WIDTH - 1, result[2].getFloat(7), 0f, label + ": four proposals require W5 verification");
                assertEquals(consumed - 1, result[2].getFloat(8), 0f, label + ": accepted drafts");
                assertEquals(1, result[2].getFloat(9), 0f, label + ": exactly one speculative transaction");
                for (int row = 0; row < consumed; row++) {
                    assertEquals(row == consumed - 1 ? EOS : DRAFT, result[0].getLong(row),
                            label + ": emitted token " + row);
                }
                assertEquals(WIDTH, target.input("ids").size(1), "physical width must remain five");
                assertEquals(consumed, target.input("actual_length").getLong(0),
                        label + ": authoritative rerun length, not full verification length");
                checkNumerics(target, predictor, consumed, label);
                assertEquals(START + consumed, positions.getLong(0), label + ": published position");
            } finally {
                for (INDArray array : result) array.close();
            }
        }
    }

    private static void checkNumerics(Plan target, Plan predictor, int consumed, String label) {
        // Independent row-by-row reference; no ND4J op or window graph is used here.
        double state = 2.0;
        double older = 0.25;
        double previous = -0.5;
        double carry = 7.0;
        double visibleValueSum = SENTINEL; // one pre-existing target KV row
        try (INDArray logits = target.copyOutput("logits");
             INDArray hidden = target.copyOutput("hidden")) {
            assertArrayEquals(new long[]{1, WIDTH, VOCAB}, logits.shape());
            assertArrayEquals(new long[]{1, WIDTH, 1}, hidden.shape());
            for (int row = 0; row < consumed; row++) {
                int token = row == 0 ? BASE : DRAFT; // EOS is pending, never consumed.
                double convolution = 0.25 * older + 0.5 * previous + token;
                state = 0.5 * state + 0.5 * convolution;
                older = previous;
                previous = token;
                visibleValueSum += convolution;
                double expectedHidden = state + visibleValueSum / (START + row + 1);
                assertEquals(expectedHidden, hidden.getDouble(0, row, 0), EPS, label + ": hidden " + row);
                for (int cls = 0; cls < VOCAB; cls++) {
                    int winner = row < consumed - 1 ? DRAFT : EOS;
                    double expected = (cls == winner ? 40.0 : 0.0) + expectedHidden * (cls + 1) / 16.0;
                    assertEquals(expected, logits.getDouble(0, row, cls), EPS,
                            label + ": logit " + row + "/" + cls);
                }
                assertEquals(state, target.input("key").getDouble(0, START + row, 0, 0), EPS,
                        label + ": retained target key " + row);
                assertEquals(convolution, target.input("value").getDouble(0, START + row, 0, 0), EPS,
                        label + ": retained target value " + row);
                int predictorRow = START - 1 + row;
                double predictorKey = 64.0 * predictorRow + 8.0 * carry + token;
                assertEquals(predictorKey, predictor.input("key").getDouble(0, predictorRow, 0, 0), EPS,
                        label + ": repaired predictor key " + row);
                assertEquals(predictorKey + 0.5, predictor.input("value").getDouble(0, predictorRow, 0, 0), EPS,
                        label + ": repaired predictor value " + row);
                carry = expectedHidden;
            }
        }
        target.assertInputBinding("conv");
        target.assertInputBinding("gdn");
        assertEquals(state, target.input("gdn").getDouble(0), EPS, label + ": full GDN state");
        assertEquals(older, target.input("conv").getDouble(0), EPS, label + ": conv older");
        assertEquals(previous, target.input("conv").getDouble(1), EPS, label + ": conv newest");
        assertEquals(carry, predictor.input("carry").getDouble(0), EPS, label + ": pending carry");
        assertEquals(EOS, target.input("ids").getLong(0), label + ": target pending token");
        assertEquals(EOS, predictor.input("ids").getLong(0), label + ": predictor pending token");
        for (String position : new String[]{"position", "cache_position"}) {
            assertEquals(START + consumed, target.input(position).getLong(0), label + ": target " + position);
            assertEquals(START + consumed - 1, predictor.input(position).getLong(0),
                    label + ": predictor " + position);
        }
        for (int row = 0; row < WIDTH; row++) {
            for (int col = 0; col < CACHE; col++) {
                double bias = target.input("mask").getDouble(0, 0, row, col);
                if (col < START + consumed) assertEquals(0, bias, EPS, label + ": visible target KV");
                else assertTrue(bias <= -1e9, label + ": rejected target KV must be masked");
            }
        }
        for (int col = 0; col < CACHE; col++) {
            double bias = predictor.input("mask").getDouble(col);
            if (col < START + consumed - 1) assertEquals(0, bias, EPS, label + ": visible predictor KV");
            else assertTrue(bias <= -1e9, label + ": pending/rejected predictor KV must be masked at " + col);
        }
        // Existing prefix and rows outside the physical write window must not change.
        for (String kv : new String[]{"key", "value"}) {
            assertEquals(SENTINEL, target.input(kv).getFloat(0), 0f, label + ": pre-existing " + kv);
            for (int row = START + WIDTH - 2; row < CACHE; row++) {
                assertEquals(SENTINEL, predictor.input(kv).getFloat(0, row, 0, 0), 0f,
                        label + ": untouched predictor " + kv + " " + row);
            }
            for (int row = START + WIDTH; row < CACHE; row++) {
                assertEquals(SENTINEL, target.input(kv).getFloat(0, row, 0, 0), 0f,
                        label + ": untouched target " + kv + " " + row);
            }
        }
    }

    private static Plan target() {
        Plan p = new Plan();
        SDVariable ids = p.placeholder("ids", Nd4j.ones(DataType.INT64, 1, WIDTH));
        SDVariable mask = p.placeholder("mask", mask(WIDTH));
        p.echo("position", Nd4j.valueArrayOf(new long[]{1}, START, DataType.INT64));
        SDVariable position = p.placeholder("cache_position", Nd4j.valueArrayOf(new long[]{1}, START, DataType.INT64));
        SDVariable length = p.placeholder("actual_length", Nd4j.scalar(DataType.INT64, WIDTH));
        SDVariable accepted = p.placeholder("accepted", Nd4j.valueArrayOf(new long[]{1}, 3, DataType.INT64));
        SDVariable convState = p.placeholder("conv", Nd4j.createFromArray(0.25f, -0.5f).reshape(1, 1, 2));
        SDVariable gdnState = p.placeholder("gdn", Nd4j.valueArrayOf(new long[]{1, 1, 1, 1}, 2, DataType.FLOAT));
        SDVariable key = p.placeholder("key", cache());
        SDVariable value = p.placeholder("value", cache());
        SDVariable x = ids.castTo(DataType.FLOAT).reshape(1, WIDTH, 1);
        SDVariable[] conv = new CausalConv1d(p.graph, x,
                p.graph.constant(Nd4j.createFromArray(0.25f, 0.5f, 1f).reshape(1, 3)),
                null, convState, length, 0, 0).outputVariables();
        p.namedOutput(conv[1], "conv_next");
        SDVariable qk = p.graph.constant(Nd4j.ones(DataType.FLOAT, 1, WIDTH, 1, 1));
        SDVariable[] gdn = new GatedDeltaRule(p.graph, qk, qk, conv[0].reshape(1, WIDTH, 1, 1),
                p.graph.constant(Nd4j.valueArrayOf(new long[]{1, WIDTH, 1}, 0.5, DataType.FLOAT)),
                p.graph.constant(Nd4j.zeros(DataType.FLOAT, 1, WIDTH, 1)), gdnState, length).outputVariables();
        p.namedOutput(gdn[1], "gdn_next");
        SDVariable attention = p.graph.nn().dotProductAttentionV2("attention", qk.mul(0),
                conv[0].reshape(1, WIDTH, 1, 1), gdn[0], null, null,
                key, value, position, mask, 0.0, 0.0, false, false);
        SDVariable hidden = gdn[0].add(attention).reshape(1, WIDTH, 1);
        p.namedOutput(hidden, "hidden");
        SDVariable rows = p.graph.constant(Nd4j.createFromArray(0L, 1L, 2L, 3L, 4L).reshape(1, WIDTH, 1));
        SDVariable match = rows.lt(accepted.reshape(1, 1, 1)).castTo(DataType.FLOAT);
        SDVariable draftBias = p.graph.constant(Nd4j.createFromArray(0f, 0f, 0f, 0f, 40f).reshape(1, 1, VOCAB));
        SDVariable eosBias = p.graph.constant(Nd4j.createFromArray(40f, 0f, 0f, 0f, 0f).reshape(1, 1, VOCAB));
        SDVariable slope = p.graph.constant(Nd4j.createFromArray(1f, 2f, 3f, 4f, 5f).reshape(1, 1, VOCAB)).div(16);
        p.namedOutput(match.mul(draftBias).add(match.rsub(1).mul(eosBias)).add(hidden.mul(slope)), "logits");
        return p;
    }

    private static Plan predictor() {
        Plan p = new Plan();
        SDVariable ids = p.placeholder("ids", Nd4j.ones(DataType.INT64, 1, 1));
        SDVariable carry = p.placeholder("carry", Nd4j.valueArrayOf(new long[]{1, 1, 1}, 7, DataType.FLOAT));
        SDVariable mask = p.placeholder("mask", mask(1));
        SDVariable rope = p.placeholder("position", Nd4j.zeros(DataType.INT64, 1));
        SDVariable position = p.placeholder("cache_position", Nd4j.zeros(DataType.INT64, 1));
        SDVariable key = p.placeholder("key", cache());
        SDVariable value = p.placeholder("value", cache());
        SDVariable k = carry.reshape(1, 1, 1, 1).mul(8)
                .add(ids.castTo(DataType.FLOAT).reshape(1, 1, 1, 1))
                .add(rope.castTo(DataType.FLOAT).reshape(1, 1, 1, 1).mul(64));
        p.output(p.graph.nn().dotProductAttentionV2("attention", k, k.add(0.5), k, null, null,
                key, value, position, mask, 0.0, 0.0, false, false));
        p.output(carry.add("hidden", 1));
        p.namedOutput(ids.castTo(DataType.FLOAT).mul(0).reshape(1, 1, 1).add(p.graph.constant(
                Nd4j.createFromArray(0f, 0f, 0f, 0f, 40f).reshape(1, 1, VOCAB))), "logits");
        return p;
    }

    private static INDArray cache() {
        return Nd4j.valueArrayOf(new long[]{1, CACHE, 1, 1}, SENTINEL, DataType.FLOAT);
    }

    private static INDArray mask(int rows) {
        return Nd4j.valueArrayOf(new long[]{1, 1, rows, CACHE}, -Float.MAX_VALUE, DataType.FLOAT);
    }

    private static void reset(Plan target, Plan predictor) {
        target.input("ids").assign(BASE);
        target.input("actual_length").assign(WIDTH);
        target.input("gdn").assign(2);
        target.input("conv").putScalar(0, 0.25f);
        target.input("conv").putScalar(1, -0.5f);
        target.input("position").assign(START);
        target.input("cache_position").assign(START);
        predictor.input("ids").assign(BASE);
        predictor.input("carry").assign(7);
        predictor.input("position").assign(START - 1);
        predictor.input("cache_position").assign(START - 1);
        for (Plan p : new Plan[]{target, predictor}) {
            p.input("key").assign(SENTINEL);
            p.input("value").assign(SENTINEL);
            p.input("mask").assign(-Float.MAX_VALUE);
        }
        for (int row = 0; row < WIDTH; row++) target.input("mask").putScalar(new long[]{0, 0, row, 0}, 0);
    }

    private static final class Snapshot implements AutoCloseable {
        private final Map<String, INDArray> saved = new LinkedHashMap<>();
        Snapshot(Plan p) { p.inputs.forEach((name, array) -> saved.put(name, array.dup())); }
        void restore(Plan p) { saved.forEach((name, array) -> p.input(name).assign(array)); }
        @Override public void close() { saved.values().forEach(INDArray::close); }
    }

    private static final class Plan implements AutoCloseable {
        private final SameDiff graph = SameDiff.create();
        private final Map<String, INDArray> inputs = new LinkedHashMap<>();
        private final List<String> outputs = new ArrayList<>();
        private DynamicShapePlanExecutor executor;
        private NativeExecutionBinding binding;
        SDVariable placeholder(String name, INDArray value) {
            inputs.put(name, value);
            return graph.placeHolder(name, value.dataType(), value.shape());
        }
        void output(SDVariable v) { outputs.add(v.name()); }
        void namedOutput(SDVariable v, String name) {
            graph.updateVariableNameAndReference(v, name);
            output(v);
        }
        void echo(String name, INDArray value) { output(placeholder(name, value).add(name + "_echo", 1)); }
        INDArray input(String name) { return inputs.get(name); }
        void compile() {
            graph.setDspAutoCompileEnabled(true);
            graph.setDspNativeAutoCompileEnabled(true);
            for (int i = 0; i < 8; i++) graph.output(inputs, outputs.toArray(new String[0]));
            executor = graph.getOrCreateSession().getDynamicShapePlanExecutor();
            assertNotNull(executor);
            assertNotNull(executor.getNativePlanHandle());
            assertFalse(executor.getNativePlanHandle().isNull());
            assertNotNull(executor.getCachedOpContext());
            binding = executor.captureNativeExecutionBinding();
            String[] keys = binding.getExternalInputKeysSnapshot();
            INDArray[] boundInputs = binding.getExternalInputsSnapshot();
            for (int i = 0; i < keys.length; i++) {
                if (inputs.containsKey(keys[i])) inputs.put(keys[i], boundInputs[i]);
            }
        }
        int ext(String name) {
            int index = binding.findExternalInputIndex(name);
            assertTrue(index >= 0, "missing input " + name);
            return index;
        }
        int out(String name) {
            int index = binding.findOutputIndex(name);
            assertTrue(index >= 0, "missing output " + name);
            return index;
        }
        void assertInputBinding(String name) {
            OpaqueContext context = binding.getContextHandle();
            NativeOps ops = context.backendOwner().nativeOps();
            OpaqueNDArray bound = ops.getInputArrayNative(context, ext(name));
            assertNotNull(bound, "missing bound input " + name);
            Pointer nativeBuffer = ops.getOpaqueNDArraySpecialBuffer(bound);
            Pointer javaBuffer = ops.dbSpecialBuffer(input(name).data().opaqueBuffer());
            assertNotNull(nativeBuffer);
            assertNotNull(javaBuffer);
            assertEquals(javaBuffer.address(), nativeBuffer.address(),
                    "native input binding must reference fixture storage: " + name);
        }

        /** Copy native outputs WITHOUT another graph execution or taking ownership of plan buffers. */
        INDArray copyOutput(String name) {
            OpaqueContext context = binding.getContextHandle();
            NativeOps ops = context.backendOwner().nativeOps();
            Integer slot = executor.getCurrentPlan().getOutputNameToSlotIndex().get(name);
            assertNotNull(slot, "missing native output slot " + name);
            OpaqueNDArray output = ops.getPlanSlotOutputArray(executor.getNativePlanHandle(), slot);
            assertNotNull(output);
            assertFalse(output.isNull());
            output.attachOwner(context.backendOwner());
            long[] info = OpaqueNDArray.getOpaqueNDArrayShapeInfo(output);
            assertEquals(DataType.FLOAT, ArrayOptionsHelper.dataType(info), "native output dtype " + name);
            long length = OpaqueNDArray.getOpaqueNDArrayLength(output);
            INDArray copy = Nd4j.createUninitialized(DataType.FLOAT, Shape.shape(info), Shape.stride(info), Shape.order(info));
            Pointer special = ops.getOpaqueNDArraySpecialBuffer(output);
            Pointer primary = special == null || special.isNull() ? ops.getOpaqueNDArrayBuffer(output) : null;
            OpaqueDataBuffer source = ops.dbCreateExternalDataBuffer(length, DataType.FLOAT.toInt(), primary, special);
            try {
                assertNotNull(source);
                assertFalse(source.isNull());
                ops.copyBuffer(copy.data().opaqueBuffer(), length, source, 0, 0);
                Nd4j.getExecutioner().commit();
            } catch (RuntimeException | Error e) {
                copy.close();
                throw e;
            } finally {
                if (source != null && !source.isNull()) ops.deleteDataBuffer(source);
            }
            return copy;
        }
        @Override public void close() {
            if (binding != null) binding.close();
            graph.close();
        }
    }
}
