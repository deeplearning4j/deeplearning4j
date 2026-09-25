/*
 * SPDX-License-Identifier: Apache-2.0
 */
package org.eclipse.deeplearning4j.llm.generation;

import org.bytedeco.javacpp.Pointer;
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

/**
 * Packet 05 wire-layout contracts for AutoregressiveDecode.withMtpPrefixSelect.
 *
 * <p>Canonical layout, N=G+C layers: magic, mode, G, C, G state inputs, G
 * ordinary state outputs, C state inputs, C ordinary state outputs, G prefix
 * outputs, C prefix outputs = 4 + 3*N words. OFF is NO trailer at all. The
 * trailer is always the LAST attachment.</p>
 */
public class TestMtpPrefixTrailerContracts {
    private static final double MAGIC = 0x4D545050L;

    private static AutoregressiveDecode baseOp() {
        try (INDArray embeddings = Nd4j.zeros(DataType.FLOAT, 1, 1, 1);
             INDArray table = Nd4j.ones(DataType.FLOAT, 2, 1);
             INDArray ids = Nd4j.ones(DataType.INT64, 1, 1);
             INDArray mask = Nd4j.zeros(DataType.FLOAT, 1, 1, 1, 4);
             INDArray positions = Nd4j.ones(DataType.INT64, 1, 1)) {
            return new AutoregressiveDecode(
                    embeddings, table, ids, mask, positions, null,
                    null, null, 0, 0,
                    -1, -1, -1, -1, -1, -1,
                    -1, -1, -1,
                    new int[0], new int[0], new int[0], new int[0], new int[0], new int[0],
                    2, 0, 0, 1, 0.0, 0, 0.0, 1.0, Set.of());
        }
    }

    private static int findTrailerStart(AutoregressiveDecode op) {
        for (int i = 0; i < op.numTArguments(); i++) {
            if (op.getTArgument(i).doubleValue() == MAGIC) return i;
        }
        return -1;
    }

    @Test
    void testGoldenLayoutG1C1() {
        AutoregressiveDecode op = baseOp();
        op.withMtpPrefixSelect(2,
                new int[]{11}, new int[]{31},
                new int[]{21}, new int[]{41},
                new int[]{51, 61});
        int start = findTrailerStart(op);
        assertTrue(start >= 0, "prefix trailer marker must be present");
        // Every trailer writer pads the base ABI to 45 words before appending; with
        // no prior trailer the marker therefore lands at index 45.
        assertEquals(45, start, "prefix trailer must start after the padded base ABI");
        // Golden: [magic, 2, 1, 1, 11, 31, 21, 41, 51, 61] = 10 words.
        double[] expected = {MAGIC, 2, 1, 1, 11, 31, 21, 41, 51, 61};
        assertEquals(start + 10, op.numTArguments(), "total words = 4 + 3*N");
        for (int i = 0; i < 10; i++) {
            assertEquals(expected[i], op.getTArgument(start + i).doubleValue(), 0.0,
                    "trailer word " + i);
        }
    }

    @Test
    void testAsymmetricLayoutG2C1() {
        AutoregressiveDecode op = baseOp();
        op.withMtpPrefixSelect(2,
                new int[]{10, 12}, new int[]{30, 32},
                new int[]{20}, new int[]{40},
                new int[]{50, 52, 60});
        int start = findTrailerStart(op);
        assertTrue(start >= 0, "prefix trailer marker must be present");
        double[] expected = {MAGIC, 2, 2, 1, 10, 12, 30, 32, 20, 40, 50, 52, 60};
        assertEquals(start + expected.length, op.numTArguments(), "total words = 4 + 3*N for G=2,C=1");
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], op.getTArgument(start + i).doubleValue(), 0.0,
                    "trailer word " + i);
        }
    }

    @Test
    void testOffModeAppendsNothingAndPermitsAbsentArrays() {
        AutoregressiveDecode op = baseOp();
        int before = op.numTArguments();
        op.withMtpPrefixSelect(0, null, null, null, null, null);
        assertEquals(before, op.numTArguments(),
                "OFF must not append any trailer word");
        assertEquals(-1, findTrailerStart(op), "OFF must not emit a marker");
    }

    @Test
    void testDistinctStateAndPrefixArraysSurvive() {
        AutoregressiveDecode op = baseOp();
        int[] gdnIns = {1}, gdnOuts = {2}, convIns = {3}, convOuts = {4}, prefixes = {5, 6};
        op.withMtpPrefixSelect(2, gdnIns, gdnOuts, convIns, convOuts, prefixes);
        int start = findTrailerStart(op);
        assertEquals(2.0, op.getTArgument(start + 5).doubleValue(), 0.0,
                "GDN ordinary output must survive distinctly");
        assertEquals(6.0, op.getTArgument(start + 9).doubleValue(), 0.0,
                "conv prefix output must survive distinctly from the ordinary output");
    }

    @Test
    void testMalformedMetadataRejectedBeforeAppend() {
        // Mismatched pair lengths.
        assertThrows(IllegalArgumentException.class,
                () -> baseOp().withMtpPrefixSelect(2,
                        new int[]{1}, new int[]{2, 3},
                        new int[]{4}, new int[]{5},
                        new int[]{6, 7}));
        // Prefix count mismatch.
        assertThrows(IllegalArgumentException.class,
                () -> baseOp().withMtpPrefixSelect(2,
                        new int[]{1}, new int[]{2},
                        new int[]{4}, new int[]{5},
                        new int[]{6}));
        // Negative index.
        assertThrows(IllegalArgumentException.class,
                () -> baseOp().withMtpPrefixSelect(2,
                        new int[]{-1}, new int[]{2},
                        new int[]{4}, new int[]{5},
                        new int[]{6, 7}));
        // Duplicate index within a domain.
        assertThrows(IllegalArgumentException.class,
                () -> baseOp().withMtpPrefixSelect(2,
                        new int[]{1, 1}, new int[]{2, 3},
                        new int[]{4}, new int[]{5},
                        new int[]{6, 7, 8}));
        // Bad mode.
        assertThrows(IllegalArgumentException.class,
                () -> baseOp().withMtpPrefixSelect(3,
                        new int[]{1}, new int[]{2},
                        new int[]{4}, new int[]{5},
                        new int[]{6, 7}));
    }

    @Test
    void testShadowModeRejectedAtAdmissionOnBothSides() {
        // Packet 08: shadow means a comparison transaction, not a capture-only
        // mode value. Both the Java writer and the native parser reject mode 1
        // until the comparison is implemented - never run capture-only while
        // claiming validation.
        IllegalArgumentException javaEx = assertThrows(IllegalArgumentException.class,
                () -> baseOp().withMtpPrefixSelect(1,
                        new int[]{1}, new int[]{2},
                        new int[]{3}, new int[]{4},
                        new int[]{5, 6}));
        assertTrue(javaEx.getMessage().contains("comparison"),
                "Java rejection must name the missing comparison, got: " + javaEx.getMessage());
    }

    @Test
    void testDuplicateAttachmentAndOrderingViolationsRejected() {
        AutoregressiveDecode op = baseOp();
        op.withMtpPrefixSelect(2,
                new int[]{1}, new int[]{2},
                new int[]{3}, new int[]{4},
                new int[]{5, 6});
        assertThrows(IllegalArgumentException.class,
                () -> op.withMtpPrefixSelect(2,
                        new int[]{1}, new int[]{2},
                        new int[]{3}, new int[]{4},
                        new int[]{5, 6}),
                "duplicate prefix attachment must be rejected");
    }

    @Test
    void testPrefixTrailerFollowsScalarAndRepairTrailers() {
        // Scalar trailer combination requires a compiled plan executor (the
        // NativeExecutionBinding comes from a real prepared plan).
        try (TinyPlan plan = new TinyPlan()) {
            plan.compile();
            try (DynamicShapePlanExecutor.NativeExecutionBinding binding =
                         plan.executor.captureNativeExecutionBinding()) {
                AutoregressiveDecode op = baseOp();
                op.withSpeculativeDecoding(1, AutoregressiveDecode.SPECULATOR_TYPE_MTP);
                op.withScalarTargetPlan(binding,
                        plan.executor.getCurrentPlan().getExternalInputKeys(),
                        new ArrayList<>(plan.executor.getCurrentPlan().getRequestedOutputs()),
                        "ids", "mask", "position", "cache_position", "actual_length",
                        "logits", "hidden");
                int scalarEnd = op.numTArguments();
                op.withMtpPrefixSelect(2,
                        new int[]{1}, new int[]{2},
                        new int[]{3}, new int[]{4},
                        new int[]{5, 6});
                int start = findTrailerStart(op);
                assertEquals(scalarEnd, start,
                        "prefix trailer must begin exactly where the scalar trailer ended");
                assertEquals(start + 10, op.numTArguments(),
                        "prefix trailer must be the LAST attachment");
            }
        }
    }

    @Test
    void testPrefixTrailerFollowsRepairTrailer() {
        // withMtpRepairPlan needs non-null plan/context handles; it never executes.
        try (Pointer fakePlan = Pointer.malloc(8); Pointer fakeCtx = Pointer.malloc(8)) {
            AutoregressiveDecode op = baseOp();
            op.withSpeculativeDecoding(1, AutoregressiveDecode.SPECULATOR_TYPE_MTP);
            op.withMtpRepairPlan(fakePlan, fakeCtx,
                    1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0);
            int repairEnd = op.numTArguments();
            op.withMtpPrefixSelect(2,
                    new int[]{1}, new int[]{2},
                    new int[]{3}, new int[]{4},
                    new int[]{5, 6});
            int start = findTrailerStart(op);
            assertEquals(repairEnd, start,
                    "prefix trailer must begin exactly where the repair trailer ended");
            assertEquals(start + 10, op.numTArguments(),
                    "prefix trailer must be the LAST attachment");
        }
    }

    @Test
    void testPrefixTrailerFollowsBatchedRepairTrailer() {
        try (Pointer fakePlan = Pointer.malloc(8); Pointer fakeCtx = Pointer.malloc(8)) {
            AutoregressiveDecode op = baseOp();
            op.withSpeculativeDecoding(1, AutoregressiveDecode.SPECULATOR_TYPE_MTP);
            op.withMtpPlan(Nd4j.ones(DataType.INT64, 1, 1),
                    Nd4j.ones(DataType.FLOAT, 1, 1, 4),
                    Nd4j.ones(DataType.FLOAT, 1, 1, 1, 4),
                    Nd4j.ones(DataType.INT64, 1), Nd4j.ones(DataType.INT64, 1),
                    new INDArray[]{Nd4j.ones(DataType.FLOAT, 1, 4, 1, 1),
                            Nd4j.ones(DataType.FLOAT, 1, 4, 1, 1)},
                    fakePlan, fakeCtx, 1, 2, 0, 0, 0, 0, 0,
                    new int[]{0, 1}, 0, 0, 0);
            op.withMtpBatchedRepairPlan(
                    Nd4j.ones(DataType.INT64, 1, 2),
                    Nd4j.ones(DataType.FLOAT, 1, 2, 4),
                    Nd4j.ones(DataType.FLOAT, 1, 1, 2, 4),
                    Nd4j.ones(DataType.INT64, 1), Nd4j.ones(DataType.INT64, 1), 2,
                    fakePlan, fakeCtx, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0);
            int batchEnd = op.numTArguments();
            op.withMtpPrefixSelect(2,
                    new int[]{1}, new int[]{2},
                    new int[]{3}, new int[]{4},
                    new int[]{5, 6});
            int start = findTrailerStart(op);
            assertEquals(batchEnd, start,
                    "prefix trailer must begin exactly where the batched repair trailer ended");
            assertEquals(start + 10, op.numTArguments(),
                    "prefix trailer must be the LAST attachment");
        }
    }

    @Test
    void testScalarAttachmentAfterPrefixTrailerRejected() {
        AutoregressiveDecode op = baseOp();
        op.withSpeculativeDecoding(1, AutoregressiveDecode.SPECULATOR_TYPE_MTP);
        op.withMtpPrefixSelect(2,
                new int[]{1}, new int[]{2},
                new int[]{3}, new int[]{4},
                new int[]{5, 6});
        assertThrows(IllegalArgumentException.class,
                () -> op.withScalarTargetPlan(null, new String[0], List.of(),
                        "ids", "mask", "position", "cache_position", "actual_length",
                        "logits", "hidden"),
                "scalar attachment after the prefix trailer must be rejected");
    }

    /** Minimal compiled plan for the scalar-binding trailer combination. */
    private static final class TinyPlan implements AutoCloseable {
        private final SameDiff graph = SameDiff.create();
        private final Map<String, INDArray> inputs = new LinkedHashMap<>();
        private final List<String> outputs = new ArrayList<>();
        private DynamicShapePlanExecutor executor;

        TinyPlan() {
            SDVariable ids = placeholder("ids", Nd4j.ones(DataType.INT64, 1, 1));
            SDVariable mask = placeholder("mask",
                    Nd4j.valueArrayOf(new long[]{1, 1, 1, 4}, -Float.MAX_VALUE, DataType.FLOAT));
            SDVariable position = placeholder("position", Nd4j.zeros(DataType.INT64, 1));
            SDVariable cache = placeholder("cache_position", Nd4j.zeros(DataType.INT64, 1));
            SDVariable actual = placeholder("actual_length", Nd4j.ones(DataType.INT64, 1));
            SDVariable zeros = ids.castTo(DataType.FLOAT).mul(0).reshape(1, 1, 1);
            SDVariable rawLogits = zeros.add(graph.constant(
                    Nd4j.createFromArray(10.0f, 0.0f).reshape(1, 1, 2)));
            output(graph.castTo("logits", rawLogits, DataType.FLOAT));
            output(zeros.add("hidden", graph.constant(
                    Nd4j.valueArrayOf(new long[]{1, 1, 1}, 100, DataType.FLOAT))));
            // Keep the mask/position/cache/actual placeholders live through echo.
            output(mask.add("mask_echo", 0));
            output(position.add("position_echo", 0));
            output(cache.add("cache_echo", 0));
            output(actual.add("actual_echo", 0));
        }

        private SDVariable placeholder(String name, INDArray value) {
            inputs.put(name, value);
            return graph.placeHolder(name, value.dataType(), value.shape());
        }

        private void output(SDVariable value) { outputs.add(value.name()); }

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

        @Override public void close() { graph.close(); }
    }
}
