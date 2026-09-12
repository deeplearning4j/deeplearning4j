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

        // ── PHASE 2: fresh load and optimize. Retain the source until inference
        // finishes because optimizer output may share its constants.
        try (SameDiff optSd = SDZSerializer.load(rawPath.toFile(), false)) {
        String previous = System.getProperty("nd4j.optimizer.allowPackedGraphs");
        System.setProperty("nd4j.optimizer.allowPackedGraphs", "true");
        SameDiff optimized;
        try {
            optimized = GraphOptimizer.optimize(optSd, optSd.outputs());
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
