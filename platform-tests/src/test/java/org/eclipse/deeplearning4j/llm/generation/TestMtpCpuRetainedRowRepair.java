/*
 * SPDX-License-Identifier: Apache-2.0
 */
package org.eclipse.deeplearning4j.llm.generation;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
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
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Native CPU multi-token commit contract; no checkpoint or model-sized graph required. */
public class TestMtpCpuRetainedRowRepair {
    private static final int K = 3;
    private static final int WIDTH = K + 1;
    private static final int CACHE = 8;

    @BeforeAll
    static void requireMultiRowCommitMode() {
        // The surefire mapping in platform-tests/pom.xml only forwards
        // SD_MTP_MULTI_ROW_COMMIT into the forked JVM when the Maven property is
        // provided at launch. Without it the native op runs the single-row commit
        // policy and every multi-row expectation below fails for policy reasons,
        // not correctness. Fail fast with the launch instruction instead.
        assertEquals("1", System.getenv("SD_MTP_MULTI_ROW_COMMIT"),
                "This class validates multi-row commits. "
                        + "Run with -Dnd4j.mtp.multiRowCommit=1.");
    }
    /**
     * Synthetic native target start position (packet 5). The predictor row
     * mapping is r = target position - 1, so a zero origin would map the first
     * target position to a NEGATIVE predictor row. With base target position 1
     * the predictor base row is 0 and a final emitted count m leaves the
     * pending predictor row at m - the original retained-key/value and mask
     * oracles are unchanged.
     */
    private static final int BASE_TARGET_POSITION = 1;

    @ParameterizedTest(name = "accepted prefix length {0}")
    @ValueSource(ints = {0, 1, 2, 3})
    void testRetainedRowsUseTargetCarry(int accepted) {
        checkRetainedRows(accepted, DataType.FLOAT, DataType.FLOAT);
    }

    @Test
    void testBfloatPredictorLogits() {
        checkRetainedRows(1, DataType.FLOAT, DataType.BFLOAT16);
    }

    @Test
    void testBfloatVerificationLogits() {
        // Two accepted rows give the verification rows distinct winners, so the
        // BF16 byte-stride addressing is exercised beyond the shared base row.
        checkRetainedRows(2, DataType.BFLOAT16, DataType.FLOAT);
    }

    private void checkRetainedRows(int accepted, DataType targetType, DataType predictorType) {
        // ADR 0106 Phase 2b exit: token-exact parity proven (milestone bc3f5c2a,
        // emissionDeltas 0/251); CUDA multi-token emission restored, so the
        // retained-row repair contract is asserted on BOTH backends.
        try (Plan target = target(accepted, targetType); Plan predictor = predictor(predictorType)) {
            target.compile();
            predictor.compile();
            // Warmup executes the in-place scatter. Restore the initial decode state.
            predictor.input("key").assign(-1);
            predictor.input("value").assign(-1);
            predictor.input("carry").assign(7);
            // Stage the target controls at the synthetic base position so the
            // first verification execution and the native loop's per-step writes
            // agree on the origin (packet 5).
            target.input("position").assign(BASE_TARGET_POSITION);
            target.input("cache_position").assign(BASE_TARGET_POSITION);
            try (INDArray embeddings = Nd4j.zeros(DataType.FLOAT, 1, 1, 1);
                 INDArray table = Nd4j.ones(DataType.FLOAT, 2, 1);
                 INDArray positions = Nd4j.valueArrayOf(new long[]{1, 1}, BASE_TARGET_POSITION)) {
                AutoregressiveDecode op = new AutoregressiveDecode(
                        embeddings, table, target.input("ids"), target.input("mask"), positions, null,
                        target.executor.getNativePlanHandle(), target.executor.getCachedOpContext(),
                        target.executor.getCurrentPlan().getExternalInputKeys().length, target.outputs.size(),
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
                INDArray[] result = Nd4j.getExecutioner().exec(op);
                try {
                    assertEquals(accepted + 1, result[1].getLong(0), "accepted tokens plus EOS correction/bonus");
                    assertEquals(K, result[2].getFloat(7), 0.0f, "all three drafts must execute");
                    assertEquals(accepted, result[2].getFloat(8), 0.0f, "forced acceptance length");
                    assertEquals(1, result[2].getFloat(9), 0.0f, "exactly one speculative step");
                    for (int q = 0; q <= accepted; q++) {
                        assertEquals(q < accepted ? 1 : 0, result[0].getLong(q), "emitted token " + q);
                        // Scalar predictor reference: KV(q) stores its INPUT carry.
                        // q=0 uses the initial carry; every retained q>0 must use target h(q-1).
                        float expected = q == 0 ? 7 : 100 + q - 1;
                        assertEquals(expected, predictor.input("key").getFloat(0, q, 0, 0), 0.0f,
                                "retained key row " + q + " must not keep recursive predictor carry");
                        assertEquals(expected + 10, predictor.input("value").getFloat(0, q, 0, 0), 0.0f,
                                "retained value row " + q);
                    }
                    for (int q = accepted + 1; q < K; q++) {
                        assertEquals(-1e9f, predictor.input("mask").getFloat(q), 0.0f,
                                "rejected proposal row must use the FLOAT causal-mask fill: " + q);
                    }
                    assertEquals(100 + accepted, predictor.input("carry").getFloat(0), 0.0f,
                            "repair must not overwrite the committed target carry for the next call");
                    assertEquals(accepted + 1, predictor.input("position").getLong(0));
                    assertEquals(accepted + 1, predictor.input("cache_position").getLong(0));
                } finally {
                    for (INDArray array : result) array.close();
                }
            }
        }
    }

    private static Plan target(int accepted, DataType logitsType) {
        Plan p = new Plan();
        SDVariable ids = p.placeholder("ids", Nd4j.ones(DataType.INT64, 1, WIDTH));
        p.echo("mask", Nd4j.valueArrayOf(new long[]{1, 1, WIDTH, CACHE}, -Float.MAX_VALUE, DataType.FLOAT));
        p.echo("position", Nd4j.zeros(DataType.INT64, 1));
        p.echo("cache_position", Nd4j.zeros(DataType.INT64, 1));
        p.echo("actual_length", Nd4j.ones(DataType.INT64, 1));
        float[] logits = new float[WIDTH * 2];
        float[] hidden = new float[WIDTH];
        for (int row = 0; row < WIDTH; row++) {
            logits[2 * row + (row < accepted ? 1 : 0)] = 10;
            hidden[row] = 100 + row;
        }
        // Depend on ids so the native plan retains the input; logits do not depend on window length.
        SDVariable zeros = ids.castTo(DataType.FLOAT).mul(0).reshape(1, WIDTH, 1);
        SDVariable rawLogits = zeros.add(p.graph.constant(Nd4j.createFromArray(logits).reshape(1, WIDTH, 2)));
        p.output(p.graph.castTo("logits", rawLogits, logitsType));
        p.output(zeros.add("hidden", p.graph.constant(Nd4j.createFromArray(hidden).reshape(1, WIDTH, 1))));
        return p;
    }

    private static Plan predictor(DataType logitsType) {
        Plan p = new Plan();
        SDVariable ids = p.placeholder("ids", Nd4j.ones(DataType.INT64, 1, 1));
        SDVariable carry = p.placeholder("carry", Nd4j.valueArrayOf(new long[]{1, 1, 1}, 7, DataType.FLOAT));
        SDVariable mask = p.placeholder("mask",
                Nd4j.valueArrayOf(new long[]{1, 1, 1, CACHE}, -Float.MAX_VALUE, DataType.FLOAT));
        p.echo("position", Nd4j.zeros(DataType.INT64, 1));
        SDVariable position = p.placeholder("cache_position", Nd4j.zeros(DataType.INT64, 1));
        SDVariable key = p.placeholder("key", Nd4j.zeros(DataType.FLOAT, 1, CACHE, 1, 1));
        SDVariable value = p.placeholder("value", Nd4j.zeros(DataType.FLOAT, 1, CACHE, 1, 1));
        SDVariable k = carry.reshape(1, 1, 1, 1);
        SDVariable v = carry.add(10).reshape(1, 1, 1, 1);
        // Use the production op's built-in cache writes, not a simulated cache update.
        p.output(p.graph.nn().dotProductAttentionV2("attention", k, v, k, null, null,
                key, value, position, mask, 0.0, 0.0, false, false));
        p.output(carry.add("hidden", 1));
        SDVariable zero = ids.castTo(DataType.FLOAT).mul(0).reshape(1, 1, 1);
        SDVariable rawLogits = zero.add(p.graph.constant(Nd4j.createFromArray(0.0f, 10.0f).reshape(1, 1, 2)));
        p.output(p.graph.castTo("logits", rawLogits, logitsType));
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
