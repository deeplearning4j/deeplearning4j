/*
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

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

/**
 * Packet 02 binding contracts for the native output manifest and the scalar-target
 * mapping. The native decode must declare the FULL requested-output list of the
 * prepared target plan (including outputs the Java loop never consumes), and the
 * scalar mapping must be name-based on the scalar binding side.
 */
public class TestMtpPrefixBindingContracts {
    private static final int K = 1;
    private static final int WIDTH = K + 1;
    private static final int CACHE = 8;
    private static final int BASE_TARGET_POSITION = 1;

    /** Target window plan with an EXTRA requested output ("extra") the consumer loop ignores. */
    private static Plan target() {
        Plan p = new Plan();
        p.placeholder("ids", Nd4j.ones(DataType.INT64, 1, WIDTH));
        p.echo("mask", Nd4j.valueArrayOf(new long[]{1, 1, WIDTH, CACHE}, -Float.MAX_VALUE, DataType.FLOAT));
        p.echo("position", Nd4j.zeros(DataType.INT64, 1));
        p.echo("cache_position", Nd4j.zeros(DataType.INT64, 1));
        p.echo("actual_length", Nd4j.ones(DataType.INT64, 1));
        // Verification row 0 disagrees with the predictor draft (token 0 vs 1), so
        // every step forces the scalar-width-1 rerun path.
        float[] logits = new float[WIDTH * 2];
        float[] hidden = new float[WIDTH];
        for (int row = 0; row < WIDTH; row++) {
            logits[2 * row] = 10;
            hidden[row] = 100 + row;
        }
        SDVariable ids = p.graph.getVariable("ids");
        SDVariable zeros = ids.castTo(DataType.FLOAT).mul(0).reshape(1, WIDTH, 1);
        SDVariable rawLogits = zeros.add(p.graph.constant("shared_logits",
                Nd4j.createFromArray(logits).reshape(1, WIDTH, 2)));
        p.output(p.graph.castTo("logits", rawLogits, DataType.FLOAT));
        p.output(zeros.add("hidden", p.graph.constant("shared_hidden",
                Nd4j.createFromArray(hidden).reshape(1, WIDTH, 1))));
        p.output(zeros.add("extra", p.graph.constant("shared_extra",
                Nd4j.valueArrayOf(new long[]{1, WIDTH, 1}, 42, DataType.FLOAT))));
        return p;
    }

    /** Width-1 scalar target sharing every input NAME with the target plan. The
     *  scalar outputs must cover every output the scalar ABI maps from the target
     *  (logits/hidden/extra), but the scalar plan has its own CONSTANT arrays for
     *  them - constants are not external inputs, so the scalar external-input key
     *  set stays clean (no 'scalar_extra' key). */
    private static Plan scalarTarget() {
        Plan p = new Plan();
        p.placeholder("ids", Nd4j.ones(DataType.INT64, 1, 1));
        p.echo("mask", Nd4j.valueArrayOf(new long[]{1, 1, 1, CACHE}, -Float.MAX_VALUE, DataType.FLOAT));
        p.echo("position", Nd4j.zeros(DataType.INT64, 1));
        p.echo("cache_position", Nd4j.zeros(DataType.INT64, 1));
        p.echo("actual_length", Nd4j.ones(DataType.INT64, 1));
        SDVariable ids = p.graph.getVariable("ids");
        SDVariable zeros = ids.castTo(DataType.FLOAT).mul(0).reshape(1, 1, 1);
        // Greedy-identical row-0 logits: token 0, matching the verification row 0.
        // The constants carry the SAME names as the target plan's constants - the
        // scalar binding's external-input keys must map into the target key set by
        // name (production derives both plans from one graph).
        SDVariable rawLogits = zeros.add(p.graph.constant("shared_logits",
                Nd4j.createFromArray(10.0f, 0.0f).reshape(1, 1, 2)));
        p.output(p.graph.castTo("logits", rawLogits, DataType.FLOAT));
        p.output(zeros.add("hidden", p.graph.constant("shared_hidden",
                Nd4j.valueArrayOf(new long[]{1, 1, 1}, 100, DataType.FLOAT))));
        p.output(zeros.add("extra", p.graph.constant("shared_extra",
                Nd4j.valueArrayOf(new long[]{1, 1, 1}, 42, DataType.FLOAT))));
        return p;
    }

    /** Minimal MTP predictor: always proposes token 1. The cache_position
     *  placeholder is the attention position input (consumed), mirroring the
     *  production predictor geometry; unconsumed placeholders are dropped from
     *  the compiled plan's external inputs. */
    private static Plan predictor() {
        Plan p = new Plan();
        p.placeholder("ids", Nd4j.ones(DataType.INT64, 1, 1));
        p.placeholder("carry", Nd4j.valueArrayOf(new long[]{1, 1, 1}, 7, DataType.FLOAT));
        p.placeholder("mask", Nd4j.valueArrayOf(new long[]{1, 1, 1, CACHE}, -Float.MAX_VALUE, DataType.FLOAT));
        p.echo("position", Nd4j.zeros(DataType.INT64, 1));
        p.placeholder("cache_position", Nd4j.zeros(DataType.INT64, 1));
        p.placeholder("key", Nd4j.zeros(DataType.FLOAT, 1, CACHE, 1, 1));
        p.placeholder("value", Nd4j.zeros(DataType.FLOAT, 1, CACHE, 1, 1));
        SDVariable carry = p.graph.getVariable("carry");
        SDVariable k = carry.reshape(1, 1, 1, 1);
        SDVariable v = carry.add(10).reshape(1, 1, 1, 1);
        SDVariable position = p.graph.getVariable("cache_position");
        SDVariable mask = p.graph.getVariable("mask");
        SDVariable key = p.graph.getVariable("key");
        SDVariable value = p.graph.getVariable("value");
        p.output(p.graph.nn().dotProductAttentionV2("attention", k, v, k, null, null,
                key, value, position, mask, 0.0, 0.0, false, false));
        p.output(carry.add("hidden", 1));
        SDVariable ids = p.graph.getVariable("ids");
        SDVariable zero = ids.castTo(DataType.FLOAT).mul(0).reshape(1, 1, 1);
        SDVariable rawLogits = zero.add(p.graph.constant(Nd4j.createFromArray(0.0f, 10.0f).reshape(1, 1, 2)));
        p.output(p.graph.castTo("logits", rawLogits, DataType.FLOAT));
        return p;
    }

    @Test
    void testNativeOutputManifestIncludesUnconsumedExtraOutput() {
        try (Plan target = target()) {
            target.compile();
            List<String> manifest =
                    new ArrayList<>(target.executor.getCurrentPlan().getRequestedOutputs());
            assertEquals(target.outputs.size(), manifest.size(),
                    "the native manifest must carry every requested output of the frozen plan");
            assertTrue(manifest.contains("extra"),
                    "the extra output must be part of the native manifest even though no Java loop consumes it");
            assertTrue(target.executor.findOutputIndex("extra") >= 0,
                    "findOutputIndex must resolve the extra output from the plan");
        }
    }

    @Test
    void testScalarTargetMappingIsNameBasedAcrossPermutedScalarOrder() {
        try (Plan target = target(); Plan scalar = scalarTarget(); Plan predictor = predictor();
             INDArray embeddings = Nd4j.zeros(DataType.FLOAT, 1, 1, 1);
             INDArray table = Nd4j.ones(DataType.FLOAT, 2, 1);
             INDArray positions = Nd4j.valueArrayOf(new long[]{1, 1}, BASE_TARGET_POSITION)) {
            target.compile();
            scalar.compile();
            predictor.compile();
            List<String> manifest =
                    new ArrayList<>(target.executor.getCurrentPlan().getRequestedOutputs());
            String[] targetKeys = target.executor.getCurrentPlan().getExternalInputKeys();
            try (DynamicShapePlanExecutor.NativeExecutionBinding binding =
                         scalar.executor.captureNativeExecutionBinding()) {
                AutoregressiveDecode op = baseOp(target, predictor, embeddings, table, positions);
                op.withScalarTargetPlan(binding, targetKeys, manifest,
                        "ids", "mask", "position", "cache_position", "actual_length",
                        "logits", "hidden");
                int ni = op.getTArgument(58).intValue();
                int no = op.getTArgument(59).intValue();
                assertEquals(manifest.size(), no,
                        "the scalar trailer must map EVERY target requested output");
                for (int i = 0; i < no; i++) {
                    String name = manifest.get(i);
                    int expected = binding.findOutputIndex(name);
                    assertTrue(expected >= 0, "scalar binding must contain output " + name);
                    assertEquals(expected, op.getTArgument(60 + ni + i).intValue(),
                            "mapping for target output '" + name + "' must be its name-based scalar index");
                }
            }
            // A nonexistent target output must fail before any execution.
            try (DynamicShapePlanExecutor.NativeExecutionBinding binding =
                         scalar.executor.captureNativeExecutionBinding()) {
                AutoregressiveDecode op = baseOp(target, predictor, embeddings, table, positions);
                List<String> bogus = new ArrayList<>(manifest);
                bogus.add("nonexistent_output");
                IllegalArgumentException ex = assertThrows(IllegalArgumentException.class,
                        () -> op.withScalarTargetPlan(binding, targetKeys, bogus,
                                "ids", "mask", "position", "cache_position", "actual_length",
                                "logits", "hidden"));
                assertTrue(ex.getMessage().contains("nonexistent_output"),
                        "the unmapped output name must appear in the failure");
            }
        }
    }

    @Test
    void testControlOffNativeExecutionRequiresTheFullManifest() {
        try (Plan target = target(); Plan scalar = scalarTarget(); Plan predictor = predictor();
             INDArray embeddings = Nd4j.zeros(DataType.FLOAT, 1, 1, 1);
             INDArray table = Nd4j.ones(DataType.FLOAT, 2, 1);
             INDArray positions = Nd4j.valueArrayOf(new long[]{1, 1}, BASE_TARGET_POSITION)) {
            target.compile();
            scalar.compile();
            predictor.compile();
            List<String> manifest =
                    new ArrayList<>(target.executor.getCurrentPlan().getRequestedOutputs());
            String[] targetKeys = target.executor.getCurrentPlan().getExternalInputKeys();
            List<String> reduced = List.of("logits", "hidden");

            try (DynamicShapePlanExecutor.NativeExecutionBinding binding =
                         scalar.executor.captureNativeExecutionBinding()) {
                // Reduced manifest: the native split-check must fail loudly with the
                // map-length message BEFORE any decode execution.
                AutoregressiveDecode reducedOp = baseOp(target, predictor, embeddings, table, positions);
                reducedOp.withScalarTargetPlan(binding, targetKeys, reduced,
                        "ids", "mask", "position", "cache_position", "actual_length",
                        "logits", "hidden");
                try {
                    Nd4j.getExecutioner().exec(reducedOp);
                    fail("the reduced manifest must fail the native map-length invariant");
                } catch (RuntimeException e) {
                    // The native REQUIRE_TRUE text lives in the cause chain under the
                    // execution wrapper; walk it instead of reading only the wrapper.
                    String chain = causeChainText(e);
                    assertTrue(chain.contains("target-to-scalar map length"),
                            "expected the split map-length message in the cause chain, got: " + chain);
                }

                // Full manifest: the same binding must execute through the native
                // scalar ABI without count mismatches.
                AutoregressiveDecode fullOp = baseOp(target, predictor, embeddings, table, positions);
                fullOp.withScalarTargetPlan(binding, targetKeys, manifest,
                        "ids", "mask", "position", "cache_position", "actual_length",
                        "logits", "hidden");
                INDArray[] result = Nd4j.getExecutioner().exec(fullOp);
                try {
                    assertEquals(1, result[1].getLong(0),
                            "single-row commit policy emits exactly one token");
                } finally {
                    for (INDArray array : result) array.close();
                }
            }
        }
    }

    private static AutoregressiveDecode baseOp(Plan target, Plan predictor,
                                               INDArray embeddings, INDArray table, INDArray positions) {
        List<String> manifest =
                new ArrayList<>(target.executor.getCurrentPlan().getRequestedOutputs());
        AutoregressiveDecode op = new AutoregressiveDecode(
                embeddings, table, target.input("ids"), target.input("mask"), positions, null,
                target.executor.getNativePlanHandle(), target.executor.getCachedOpContext(),
                target.executor.getCurrentPlan().getExternalInputKeys().length, manifest.size(),
                -1, -1, target.ext("mask"), -1, target.ext("ids"), target.out("logits"),
                -1, target.ext("position"), target.ext("cache_position"),
                new int[0], new int[0], new int[0], new int[0], new int[0], new int[0],
                WIDTH, 0, 0, BASE_TARGET_POSITION, 0.0, 0, 0.0, 1.0, Set.of());
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
        return op;
    }

    private static String causeChainText(Throwable t) {
        StringBuilder sb = new StringBuilder();
        Throwable cursor = t;
        while (cursor != null) {
            if (sb.length() > 0) sb.append(" << ");
            sb.append(String.valueOf(cursor.getMessage()));
            cursor = cursor.getCause();
        }
        return sb.toString();
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
