/*
 * Copyright (c) Eclipse Deeplearning4j Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.nd4j.ggml.architecture;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.serde.SDZSerializer;
import org.nd4j.autodiff.samediff.optimize.GraphOptimizer;
import org.nd4j.ggml.convert.GGMLToSameDiffConverter;
import org.nd4j.ggml.convert.ConversionOptions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Focused numerical comparison for the optimizer-vs-packed-graph question.
 *
 * <p>Converts the real Gemma GGUF once, compares its raw graph (runtime-quantized
 * weights, HALF activations) with a fresh load passed through {@link GraphOptimizer}, with the packed-graph
 * guard bypassed via -Dnd4j.optimizer.allowPackedGraphs=true (set by this test)
 * — then runs both graphs on identical inputs and asserts elementwise parity
 * within HALF tolerance.</p>
 *
 * <p>Property-gated like Gemma4LastPositionConversionIT so it never runs in
 * ordinary CI:</p>
 * <ul>
 *   <li>-Dgemma4.parity.gguf=... (required; real GGUF path)</li>
 *   <li>-Dgemma4.parity.sdz.dir=... (optional; temp workspace)</li>
 * </ul>
 *
 * <p>Verdict semantics:
 * <ul>
 *   <li>Matching outputs support parity only for this model, input and execution.</li>
 *   <li>A difference needs localization; it does not by itself identify a faulty pass.</li>
 *   <li>Max-abs-delta reported either way so magnitude is on the record.</li>
 * </ul></p>
 */
@Tag("integration")
class Gemma4OptimizerParityIT {

    private static final String GGUF_PROPERTY = "gemma4.parity.gguf";
    private static final String DIR_PROPERTY = "gemma4.parity.sdz.dir";

    private static final int TEST_SEQ = 8;      // tiny prefill — parity, not perf
    private static final int TEST_BATCH = 1;
    private static final int CACHE_CAPACITY = TEST_SEQ + 3; // exercise masked unused slots

    @Test
    void optimizedPackedGraphMatchesRawNumerically(@TempDir Path tempDir) throws Exception {
        String ggufProp = System.getProperty(GGUF_PROPERTY);
        assumeTrueGguf(ggufProp);
        Path gguf = Path.of(ggufProp);
        assertTrue(Files.isRegularFile(gguf), "GGUF not found: " + gguf);

        Path workspace = System.getProperty(DIR_PROPERTY) != null
                ? Path.of(System.getProperty(DIR_PROPERTY)) : tempDir;
        Files.createDirectories(workspace);
        Path rawPath = workspace.resolve("gemma4-parity-raw.sdz");

        // ── PHASE 1: convert once, load, run RAW inference immediately, verify
        // finiteness, snapshot outputs to host, then drop the graph. Holding
        // both execution graphs simultaneously would needlessly retain model memory.
        GGMLToSameDiffConverter converter = new GGMLToSameDiffConverter(
                ConversionOptions.builder()
                        .quantizationMode(ConversionOptions.QuantizationMode.RUNTIME_QUANTIZED_MATMUL)
                        .targetDataType(DataType.HALF)
                        .build());
        converter.convertToSDZ(gguf.toFile(), rawPath.toFile());
        assertTrue(Files.size(rawPath) > 0, "raw conversion produced no output");

        Map<String, Snapshot> rawHost;
        try (SameDiff rawSd = SDZSerializer.load(rawPath.toFile(), false)) {
            Map<String, INDArray> rawInputs = buildInputs(rawSd);
            try {
                Map<String, INDArray> rawOut = rawSd.output(rawInputs, rawSd.outputs());
                assertEquals(new java.util.HashSet<>(rawSd.outputs()), rawOut.keySet());
                rawHost = snapshots(rawOut);
            } finally {
                closeInputs(rawInputs);
            }
        }

        // Establish repeatability independently of graph transformations. Fresh inputs
        // are essential: execution updates the supplied KV cache arrays in place.
        try (SameDiff repeatSd = SDZSerializer.load(rawPath.toFile(), false)) {
            Map<String, INDArray> repeatInputs = buildInputs(repeatSd);
            try {
                Map<String, Snapshot> repeatHost = snapshots(
                        repeatSd.output(repeatInputs, repeatSd.outputs()));
                assertEquals(rawHost.keySet(), repeatHost.keySet());
                double worstRepeatDelta = 0.0;
                for (Map.Entry<String, Snapshot> e : rawHost.entrySet()) {
                    Snapshot repeat = repeatHost.get(e.getKey());
                    assertArrayEquals(e.getValue().shape, repeat.shape, e.getKey());
                    assertEquals(e.getValue().dtype, repeat.dtype, e.getKey());
                    double delta = maxAbsDelta(e.getValue().values, repeat.values);
                    System.out.println("RAW REPEAT OUTPUT " + e.getKey() + " maxAbsDelta=" + delta);
                    worstRepeatDelta = Math.max(worstRepeatDelta, delta);
                }
                assertEquals(0.0, worstRepeatDelta,
                        "raw/raw is not exact; establish numerical repeatability before attributing optimizer drift");
            } finally {
                closeInputs(repeatInputs);
            }
        }

        // ── PHASE 2: fresh load and optimize. Retain the source until inference
        // finishes because optimizer output may share its constants.
        try (SameDiff optSd = SDZSerializer.load(rawPath.toFile(), false)) {
        String previous = System.getProperty("nd4j.optimizer.allowPackedGraphs");
        System.setProperty("nd4j.optimizer.allowPackedGraphs", "true");
        SameDiff optimized;
        try {
            Map<String, Integer> appliedPasses = new java.util.TreeMap<>();
            optimized = GraphOptimizer.optimize(optSd, optSd.outputs(), GraphOptimizer.defaultOptimizations(),
                    new org.nd4j.autodiff.samediff.optimize.debug.OptimizationDebugger() {
                        public void beforeOptimizationCheck(SameDiff sd,
                                org.nd4j.autodiff.samediff.internal.SameDiffOp op,
                                org.nd4j.autodiff.samediff.optimize.Optimizer optimizer) { }
                        public void afterOptimizationsCheck(SameDiff sd,
                                org.nd4j.autodiff.samediff.internal.SameDiffOp op,
                                org.nd4j.autodiff.samediff.optimize.Optimizer optimizer, boolean applied) {
                            if (applied) appliedPasses.merge(optimizer.getClass().getSimpleName(), 1, Integer::sum);
                        }
                    });
            System.out.println("PARITY APPLIED PASSES " + appliedPasses);
        } finally {
            if (previous == null) System.clearProperty("nd4j.optimizer.allowPackedGraphs");
            else System.setProperty("nd4j.optimizer.allowPackedGraphs", previous);
        }
        assertNotNull(optimized, "optimizer returned null");
        try {
        Map<String, INDArray> optInputs = buildInputs(optimized);
        try {
        Map<String, INDArray> optOut = optimized.output(optInputs, optimized.outputs());
        assertEquals(rawHost.keySet(), new java.util.HashSet<>(optimized.outputs()),
                "optimizer changed the declared output set");
        assertEquals(rawHost.keySet(), optOut.keySet(), "optimizer dropped or added outputs");
        Map<String, Snapshot> optHost = snapshots(optOut);

        // ── Compare raw host snapshots against optimized outputs.
        int compared = 0;
        double worstDelta = 0.0;
        String worstName = null;
        for (Map.Entry<String, Snapshot> e : rawHost.entrySet()) {
            Snapshot opt = optHost.get(e.getKey());
            assertArrayEquals(e.getValue().shape, opt.shape, "shape: " + e.getKey());
            assertEquals(e.getValue().dtype, opt.dtype, "dtype: " + e.getKey());
            double delta = maxAbsDelta(e.getValue().values, opt.values);
            System.out.println("PARITY OUTPUT " + e.getKey() + " elements="
                    + opt.values.length + " maxAbsDelta=" + delta);
            if (delta > worstDelta) {
                worstDelta = delta;
                worstName = e.getKey();
            }
            compared++;
        }
        assertTrue(compared > 0, "no shared outputs to compare");

        // Preserve the existing 0.05 acceptance gate. This is an empirical gate,
        // not a universal HALF error bound and not proof of optimizer correctness.
        assertTrue(worstDelta <= 0.05,
                "PARITY FAILURE: max abs delta " + worstDelta + " on " + worstName
                        + " exceeds the existing 0.05 comparison gate");
        System.out.println("PARITY OK: compared " + compared
                + " outputs, max abs delta " + worstDelta + " (" + worstName + ")");
        } finally {
            closeInputs(optInputs);
        }
        } finally {
            if (optimized != optSd) optimized.close();
        }
        }
    }

    private static void assumeTrueGguf(String prop) {
        if (prop == null || prop.isBlank()) {
            throw new org.opentest4j.TestAbortedException(
                    "gemma4.parity.gguf not set — skipping optimizer parity IT");
        }
    }

    private static Map<String, INDArray> buildInputs(SameDiff sd) {
        Map<String, INDArray> inputs = new LinkedHashMap<>();
        INDArray ids = Nd4j.createFromArray(new long[][]{
                {2L, 101L, 9394L, 20495L, 108L, 235248L, 6853L, 108L}});
        inputs.put("input_ids", ids.reshape(TEST_BATCH, TEST_SEQ));
        inputs.put("position_offset", Nd4j.scalar(DataType.INT64, 0L));
        inputs.put("cache_position", Nd4j.scalar(DataType.INT64, 0L));
        inputs.put("actual_sequence_length", Nd4j.scalar(DataType.INT64, TEST_SEQ));
        // Additive attention bias: zero permits attention; negative masks it.
        INDArray mask = Nd4j.valueArrayOf(new long[]{TEST_BATCH, 1, TEST_SEQ, CACHE_CAPACITY},
                -Float.MAX_VALUE, DataType.FLOAT);
        for (int q = 0; q < TEST_SEQ; q++) {
            for (int k = 0; k <= q; k++) mask.putScalar(new int[]{0, 0, q, k}, 0.0f);
        }
        inputs.put("_causal_mask", mask);

        // Gemma4Architecture declares [batch, capacity, kvHeads, headDim].
        for (SDVariable ph : sd.placeHolders()) {
            String name = ph.name();
            if (inputs.containsKey(name)) continue;
            if (!name.startsWith("past_key_values.")) {
                throw new IllegalStateException("Unresolved non-KV placeholder: " + name);
            }
            long[] shape = ph.getShape();
            assertNotNull(shape, name);
            assertEquals(4, shape.length, name);
            assertTrue(shape[2] > 0 && shape[3] > 0, "unresolved KV head dimensions: " + name);
            long[] concrete = {TEST_BATCH, CACHE_CAPACITY, shape[2], shape[3]};
            for (int d = 0; d < shape.length; d++)
                assertTrue(shape[d] < 0 || shape[d] == concrete[d], "KV dimension mismatch: " + name);
            inputs.put(name, Nd4j.zeros(ph.dataType(), concrete));
        }
        return inputs;
    }

    private static long countNonFinite(float[] a) {
        long n = 0;
        for (float v : a) if (!Float.isFinite(v)) n++;
        return n;
    }

    private static double maxAbsDelta(float[] a, float[] b) {
        assertEquals(a.length, b.length, "element counts differ");
        assertEquals(0, countNonFinite(a), "non-finite reference");
        assertEquals(0, countNonFinite(b), "non-finite candidate");
        double worst = 0.0;
        for (int i = 0; i < a.length; i++) {
            double d = Math.abs((double)a[i] - b[i]);
            if (d > worst) worst = d;
        }
        return worst;
    }

    private static Map<String, Snapshot> snapshots(Map<String, INDArray> outputs) {
        assertFalse(outputs.isEmpty(), "graph produced no outputs");
        Map<String, Snapshot> result = new LinkedHashMap<>();
        for (Map.Entry<String, INDArray> e : outputs.entrySet()) {
            INDArray a = e.getValue();
            assertNotNull(a, e.getKey());
            // Own a C-order logical copy so views/offsets do not expose parent-buffer data.
            // Only the Java float[] survives: an INDArray.dup() is not a host snapshot.
            try (INDArray copy = a.dup('c')) {
                float[] values = copy.data().asFloat();
                assertEquals(a.length(), values.length, e.getKey());
                assertEquals(0, countNonFinite(values), "non-finite output: " + e.getKey());
                result.put(e.getKey(), new Snapshot(a.shape().clone(), a.dataType(), values));
            }
        }
        return result;
    }

    private static void closeInputs(Map<String, INDArray> inputs) {
        for (INDArray a : inputs.values())
            if (!a.wasClosed() && a.closeable()) a.close();
    }

    private static final class Snapshot {
        final long[] shape;
        final DataType dtype;
        final float[] values;
        Snapshot(long[] shape, DataType dtype, float[] values) {
            this.shape = shape;
            this.dtype = dtype;
            this.values = values;
        }
    }

    @Test
    void rawDspNarrowingCastRoundTripRetainsRounding() {
        try (SameDiff sd = SameDiff.create();
             INDArray input = Nd4j.createFromArray(1.0003f, -1.0003f, 0.3333f);
             INDArray half = input.castTo(DataType.HALF);
             INDArray widened = half.castTo(DataType.FLOAT)) {
            SDVariable x = sd.placeHolder("input", DataType.FLOAT, 3);
            SDVariable output = x.castTo(DataType.HALF).castTo(DataType.FLOAT).rename("output");
            sd.setOutputs(output.name());
            Snapshot expected = snapshots(Map.of("output", widened)).get("output");
            Snapshot actual = snapshots(sd.output(Map.of("input", input), sd.outputs())).get("output");
            assertEquals(DataType.FLOAT, actual.dtype);
            assertEquals(0.0, maxAbsDelta(expected.values, actual.values),
                    "raw DSP must retain the explicit HALF rounding boundary");
        }
    }

    @Test
    void narrowingCastRoundTripRetainsRounding() {
        try (SameDiff sd = SameDiff.create();
             INDArray input = Nd4j.createFromArray(1.0003f, -1.0003f, 0.3333f)) {
            SDVariable x = sd.placeHolder("input", DataType.FLOAT, 3);
            SDVariable rounded = x.castTo(DataType.HALF);
            SDVariable output = rounded.castTo(DataType.FLOAT).rename("output");
            sd.setOutputs(output.name());
            Snapshot reference;
            try (INDArray half = input.castTo(DataType.HALF);
                 INDArray widened = half.castTo(DataType.FLOAT)) {
                reference = snapshots(Map.of("output", widened)).get("output");
            }
            assertEquals(1.0f, reference.values[0], "HALF round trip must round this input");
            try (SameDiff optimized = GraphOptimizer.optimize(sd, sd.outputs(), java.util.List.of(
                    () -> java.util.List.of(new org.nd4j.autodiff.samediff.optimize.optimizations
                            .QuantizationOptimizations.RemoveRedundantCasts())))) {
                assertEquals(2L, optimized.getOps().values().stream()
                                .filter(op -> "cast".equals(op.getOp().opName())).count(),
                        "Java optimizer must retain both casts independently of native execution");
                Snapshot actual = snapshots(optimized.output(Map.of("input", input), optimized.outputs())).get("output");
                assertEquals(0.0, maxAbsDelta(reference.values, actual.values),
                        "cast elimination must retain the explicit HALF rounding boundary");
            }
        }
    }

    @Test
    void castCompositionRetainsLossyBoundariesAndOptimizesWidening() {
        DataType[][] cases = {
                {DataType.FLOAT, DataType.HALF, DataType.FLOAT},
                {DataType.FLOAT, DataType.BFLOAT16, DataType.FLOAT},
                {DataType.HALF, DataType.BFLOAT16, DataType.HALF},
                {DataType.BFLOAT16, DataType.HALF, DataType.BFLOAT16},
                {DataType.DOUBLE, DataType.FLOAT, DataType.DOUBLE},
                {DataType.LONG, DataType.FLOAT, DataType.LONG},
                {DataType.INT, DataType.BYTE, DataType.INT},
                {DataType.BYTE, DataType.UBYTE, DataType.BYTE},
                {DataType.FLOAT, DataType.HALF, DataType.DOUBLE}
        };
        for (DataType[] types : cases) {
            try (SameDiff sd = SameDiff.create()) {
                SDVariable x = sd.placeHolder("input", types[0], 3);
                sd.setOutputs(x.castTo(types[1]).castTo(types[2]).rename("output").name());
                try (SameDiff optimized = GraphOptimizer.optimize(sd, sd.outputs(), java.util.List.of(
                        () -> java.util.List.of(new org.nd4j.autodiff.samediff.optimize.optimizations
                                .QuantizationOptimizations.RemoveRedundantCasts())))) {
                    assertEquals(2L, optimized.getOps().values().stream()
                            .filter(op -> "cast".equals(op.getOp().opName())).count(),
                            java.util.Arrays.toString(types));
                }
            }
        }
        for (DataType[] types : new DataType[][]{
                {DataType.HALF, DataType.FLOAT, DataType.HALF},
                {DataType.BFLOAT16, DataType.FLOAT, DataType.BFLOAT16},
                {DataType.FLOAT, DataType.DOUBLE, DataType.FLOAT},
                {DataType.HALF, DataType.FLOAT, DataType.DOUBLE},
                {DataType.INT, DataType.LONG, DataType.INT}}) {
            try (SameDiff sd = SameDiff.create()) {
                SDVariable x = sd.placeHolder("input", types[0], 3);
                sd.setOutputs(x.castTo(types[1]).castTo(types[2]).rename("output").name());
                try (SameDiff optimized = GraphOptimizer.optimize(sd, sd.outputs(), java.util.List.of(
                        () -> java.util.List.of(new org.nd4j.autodiff.samediff.optimize.optimizations
                                .QuantizationOptimizations.RemoveRedundantCasts())))) {
                    assertEquals(1L, optimized.getOps().values().stream()
                            .filter(op -> "cast".equals(op.getOp().opName())).count(),
                            java.util.Arrays.toString(types));
                }
            }
        }
    }

    @Test
    void rawDspCastCompositionMatchesEagerIncludingKnownSourceWidening() {
        // The leading cast gives native fusion an authoritative source dtype.
        // Include non-round-trip targets: these must never become identities.
        for (DataType[] types : new DataType[][]{
                {DataType.FLOAT, DataType.HALF, DataType.FLOAT},
                {DataType.FLOAT, DataType.BFLOAT16, DataType.FLOAT},
                {DataType.HALF, DataType.BFLOAT16, DataType.HALF},
                {DataType.BFLOAT16, DataType.HALF, DataType.BFLOAT16},
                {DataType.LONG, DataType.FLOAT, DataType.LONG},
                {DataType.INT, DataType.BYTE, DataType.INT},
                {DataType.INT, DataType.UBYTE, DataType.INT},
                {DataType.INT, DataType.UINT16, DataType.INT},
                {DataType.LONG, DataType.UINT32, DataType.LONG},
                {DataType.INT, DataType.UBYTE, DataType.FLOAT},
                {DataType.LONG, DataType.UINT32, DataType.DOUBLE},
                {DataType.BYTE, DataType.UBYTE, DataType.INT},
                {DataType.UBYTE, DataType.BYTE, DataType.INT},
                {DataType.FLOAT, DataType.HALF, DataType.DOUBLE},
                {DataType.HALF, DataType.FLOAT, DataType.HALF},
                {DataType.BFLOAT16, DataType.FLOAT, DataType.BFLOAT16},
                {DataType.FLOAT, DataType.DOUBLE, DataType.FLOAT}}) {
            try (SameDiff sd = SameDiff.create();
                 INDArray input = Nd4j.createFromArray(1.0003, -1.003, 257.0, 16777217.0);
                 INDArray source = input.castTo(types[0]);
                 INDArray intermediate = source.castTo(types[1]);
                 INDArray expected = intermediate.castTo(types[2])) {
                SDVariable x = sd.placeHolder("input", DataType.DOUBLE, 4);
                sd.setOutputs(x.castTo(types[0]).castTo(types[1]).castTo(types[2]).rename("output").name());
                for (int execution = 0; execution < 4; execution++) {
                    INDArray actual = sd.outputSingle(Map.of("input", input), "output");
                    assertEquals(expected.dataType(), actual.dataType());
                    assertArrayEquals(expected.data().asDouble(), actual.data().asDouble(),
                            java.util.Arrays.toString(types) + " execution=" + execution);
                }
            }
        }
    }

    @Test
    void rawDspUnsignedCastsPreserveHighBitsAndTruncation() {
        // Typed external inputs exercise signedness even without a producing cast.
        // Same-width reinterpretation, narrowing, widening, and float conversions
        // must agree before and after the plan freezes and enters replay.
        for (DataType sourceType : new DataType[]{DataType.BYTE, DataType.UBYTE,
                DataType.SHORT, DataType.UINT16, DataType.INT, DataType.UINT32,
                DataType.LONG, DataType.UINT64}) {
            for (DataType targetType : new DataType[]{DataType.INT, DataType.LONG,
                    DataType.FLOAT, DataType.DOUBLE}) {
                if (sourceType == targetType) continue;
                try (SameDiff sd = SameDiff.create();
                     INDArray bits = Nd4j.createFromArray(0L, 1L, 127L, 128L, 255L,
                             32768L, 65535L, 2147483648L, 4294967295L, -1L, Long.MIN_VALUE);
                     INDArray input = bits.castTo(sourceType);
                     INDArray expected = input.castTo(targetType)) {
                    SDVariable x = sd.placeHolder("input", sourceType, input.length());
                    sd.setOutputs(x.castTo(targetType).rename("output").name());
                    for (int execution = 0; execution < 6; execution++) {
                        INDArray actual = sd.outputSingle(Map.of("input", input), "output");
                        assertEquals(targetType, actual.dataType());
                        assertArrayEquals(expected.data().asDouble(), actual.data().asDouble(),
                                sourceType + " -> " + targetType + " execution=" + execution);
                    }
                }
            }
        }
        for (DataType targetType : new DataType[]{DataType.UBYTE, DataType.UINT16,
                DataType.UINT32, DataType.UINT64}) {
            double upper = targetType == DataType.UBYTE ? 255.75
                    : targetType == DataType.UINT16 ? 65535.75 : 4294967295.75;
            try (SameDiff sd = SameDiff.create();
                 INDArray input = Nd4j.createFromArray(0.75, 1.75, 127.75, 128.75, upper);
                 INDArray expected = input.castTo(targetType)) {
                SDVariable x = sd.placeHolder("input", DataType.DOUBLE, input.length());
                sd.setOutputs(x.castTo(targetType).rename("output").name());
                for (int execution = 0; execution < 6; execution++) {
                    INDArray actual = sd.outputSingle(Map.of("input", input), "output");
                    assertEquals(targetType, actual.dataType());
                    assertArrayEquals(expected.data().asDouble(), actual.data().asDouble(),
                            "DOUBLE -> " + targetType + " execution=" + execution);
                }
            }
        }
    }

    @Test
    void comparisonInspectsTailAndRejectsNonFiniteOrSizeMismatch() {
        float[] a = new float[1_000_003];
        float[] b = a.clone();
        b[b.length - 1] = 0.125f;
        assertEquals(0.125, maxAbsDelta(a, b));
        b[b.length - 1] = Float.NaN;
        assertEquals(1, countNonFinite(b));
        assertThrows(AssertionError.class, () -> maxAbsDelta(a, b));
        assertThrows(AssertionError.class, () -> maxAbsDelta(new float[1], new float[2]));
    }

    @Test
    void inputContractUsesBatchOneAndAdditivePaddedMask() {
        try (SameDiff sd = SameDiff.create()) {
            sd.placeHolder("past_key_values.0.key", DataType.HALF, -1, -1, 2, 4);
            Map<String, INDArray> inputs = buildInputs(sd);
            try {
                assertArrayEquals(new long[]{1, CACHE_CAPACITY, 2, 4},
                        inputs.get("past_key_values.0.key").shape());
                assertEquals(0, inputs.get("cache_position").rank());
                float[] mask = inputs.get("_causal_mask").data().asFloat();
                for (int q = 0; q < TEST_SEQ; q++)
                    for (int k = 0; k < CACHE_CAPACITY; k++)
                        assertEquals(k <= q ? 0.0f : -Float.MAX_VALUE, mask[q * CACHE_CAPACITY + k]);
            } finally {
                closeInputs(inputs);
            }
        }
    }

    @Test
    void snapshotOwnsLogicalViewValuesAfterSourceMutation() {
        try (INDArray source = Nd4j.createFromArray(new float[][]{{1, 2, 3}, {4, 5, 6}})) {
            INDArray view = source.transpose();
            Snapshot snapshot = snapshots(Map.of("view", view)).get("view");
            assertArrayEquals(new long[]{3, 2}, snapshot.shape);
            source.assign(99);
            assertArrayEquals(new float[]{1, 4, 2, 5, 3, 6}, snapshot.values);
        }
    }
}
