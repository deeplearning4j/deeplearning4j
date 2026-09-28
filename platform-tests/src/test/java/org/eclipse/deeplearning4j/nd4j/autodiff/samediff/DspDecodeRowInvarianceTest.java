/*
 * SPDX-License-Identifier: Apache-2.0
 */
package org.eclipse.deeplearning4j.nd4j.autodiff.samediff;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import org.junit.jupiter.params.provider.ValueSource;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.custom.RmsNorm;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Row invariance of the decode ops that multi-row speculative commit relies on.
 *
 * <p>Speculative verification computes K + 1 token positions in one window launch;
 * greedy decoding computes each in its own width-1 launch. A multi-row commit emits
 * the window rows directly, so every op must give each row the same bits whatever
 * the launch width. Each case runs one graph through DSP (compiled, captured and
 * replayed; width 1 and width W are two shape-keyed plans of the same graph, as in
 * the model) and compares the W-row launch against W chained width-1 launches:
 * every row, and for recurrent ops the final state. Every launch must also give
 * the same bits in native warmup and compiled replay, since the two modes of a
 * decode step can run in different lifecycle phases. Shapes are the
 * Qwen3.6-27B decode shapes.</p>
 */
class DspDecodeRowInvarianceTest {
    private static final int WINDOW = 5;
    private static final int WARMUP = 10;
    private static final float MASKED = -3.4028235e+38f;

    private boolean dspEnabled;
    private boolean graphCapture;
    private boolean compileAll;

    @BeforeEach
    void enableCompiledReplay() {
        assumeTrue(Nd4j.backends().isCudaAvailable(), "Requires CUDA with Triton");
        var environment = Nd4j.getEnvironment();
        dspEnabled = InferenceSession.isDynamicShapePlanEnabled();
        graphCapture = environment.tritonGraphCapture();
        compileAll = environment.tritonCompileAll();
        InferenceSession.setDynamicShapePlanEnabled(true);
        environment.setTritonGraphCapture(true);
        environment.setTritonCompileAll(true);
        Nd4j.getRandom().setSeed(12345);
    }

    @AfterEach
    void restore() {
        var environment = Nd4j.getEnvironment();
        environment.setTritonCompileAll(compileAll);
        environment.setTritonGraphCapture(graphCapture);
        InferenceSession.setDynamicShapePlanEnabled(dspEnabled);
    }

    /** Full-attention layer: GQA over the static BSHD KV cache written in place at cache_position. */
    @ParameterizedTest(name = "attention {0}")
    @ValueSource(strings = {"BFLOAT16", "FLOAT"})
    void cachedAttention(String type) {
        DataType dtype = DataType.valueOf(type);
        final int heads = 24, kvHeads = 4, dim = 256, cache = 320, start = 37;
        try (SameDiff sd = graph()) {
            SDVariable q = sd.placeHolder("q", dtype, 1, -1, heads, dim);
            SDVariable k = sd.placeHolder("k", dtype, 1, -1, kvHeads, dim);
            SDVariable v = sd.placeHolder("v", dtype, 1, -1, kvHeads, dim);
            SDVariable keyCache = sd.placeHolder("key_cache", dtype, 1, cache, kvHeads, dim);
            SDVariable valueCache = sd.placeHolder("value_cache", dtype, 1, cache, kvHeads, dim);
            SDVariable position = sd.placeHolder("cache_position", DataType.INT64);
            SDVariable mask = sd.placeHolder("mask", DataType.FLOAT, 1, 1, -1, cache);
            sd.nn().dotProductAttentionV2("out", q, v, k, null, null, keyCache, valueCache, position, mask,
                    0.0, 0.0, false, false);
            sd.setOutputs("out");

            INDArray qs = random(dtype, 1, WINDOW, heads, dim);
            INDArray ks = random(dtype, 1, WINDOW, kvHeads, dim);
            INDArray vs = random(dtype, 1, WINDOW, kvHeads, dim);
            INDArray keys = Nd4j.zeros(dtype, 1, cache, kvHeads, dim);
            INDArray values = Nd4j.zeros(dtype, 1, cache, kvHeads, dim);
            keys.get(all(), NDArrayIndex.interval(0, start), all(), all()).assign(random(dtype, 1, start, kvHeads, dim));
            values.get(all(), NDArrayIndex.interval(0, start), all(), all()).assign(random(dtype, 1, start, kvHeads, dim));
            INDArray keyState = keys.dup(), valueState = values.dup();

            Step window = () -> {
                keyState.assign(keys);
                valueState.assign(values);
                return sd.output(Map.of("q", qs, "k", ks, "v", vs, "key_cache", keyState, "value_cache", valueState,
                        "cache_position", Nd4j.scalar((long) start), "mask", causalMask(start, 0, WINDOW, cache)),
                        "out").get("out").dup();
            };
            INDArray windowOut = settled(window, "attention window " + dtype);

            List<INDArray> rows = new ArrayList<>();
            for (int r = 0; r < WINDOW; r++) {
                final int row = r;
                // Each width-1 step sees the cache the previous steps wrote.
                if (row == 0) {
                    keyState.assign(keys);
                    valueState.assign(values);
                }
                INDArray keyBefore = keyState.dup(), valueBefore = valueState.dup();
                Step scalar = () -> {
                    keyState.assign(keyBefore);
                    valueState.assign(valueBefore);
                    return sd.output(Map.of("q", row(qs, row), "k", row(ks, row), "v", row(vs, row),
                            "key_cache", keyState, "value_cache", valueState,
                            "cache_position", Nd4j.scalar((long) (start + row)),
                            "mask", causalMask(start, row, 1, cache)), "out").get("out").dup();
                };
                rows.add(settled(scalar, "attention width-1 row " + row + " " + dtype));
            }
            assertRows("attention " + dtype, windowOut, rows);
        }
    }

    /** Linear-attention layer: gated delta rule recurrence with FP32 state. */
    @ParameterizedTest(name = "gated delta rule")
    @ValueSource(ints = {0})
    void gatedDeltaRule(int unused) {
        final int heads = 48, dk = 128, dv = 128;
        try (SameDiff sd = graph()) {
            SDVariable q = sd.placeHolder("q", DataType.FLOAT, 1, -1, heads, dk);
            SDVariable k = sd.placeHolder("k", DataType.FLOAT, 1, -1, heads, dk);
            SDVariable v = sd.placeHolder("v", DataType.FLOAT, 1, -1, heads, dv);
            SDVariable beta = sd.placeHolder("beta", DataType.FLOAT, 1, -1, heads);
            SDVariable gate = sd.placeHolder("gate", DataType.FLOAT, 1, -1, heads);
            SDVariable state = sd.placeHolder("state", DataType.FLOAT, 1, heads, dk, dv);
            SDVariable length = sd.placeHolder("length", DataType.INT64);
            sd.nn().gatedDeltaRule(new String[]{"out", "state_out"}, q, k, v, beta, gate, state, length);
            sd.setOutputs("out", "state_out");

            INDArray qs = unitRows(random(DataType.FLOAT, 1, WINDOW, heads, dk)).muli(1.0 / Math.sqrt(dk));
            INDArray ks = unitRows(random(DataType.FLOAT, 1, WINDOW, heads, dk));
            INDArray vs = random(DataType.FLOAT, 1, WINDOW, heads, dv);
            INDArray betas = Nd4j.rand(DataType.FLOAT, 1, WINDOW, heads);
            INDArray gates = Nd4j.rand(DataType.FLOAT, 1, WINDOW, heads).muli(-0.5);
            INDArray initial = random(DataType.FLOAT, 1, heads, dk, dv).muli(0.1);

            Map<String, INDArray> windowOutputs = settledOutputs(() -> sd.output(Map.of("q", qs, "k", ks, "v", vs,
                    "beta", betas, "gate", gates, "state", initial.dup(), "length", Nd4j.scalar((long) WINDOW)),
                    "out", "state_out"), "gated delta rule window");

            List<INDArray> rows = new ArrayList<>();
            INDArray carried = initial.dup();
            for (int r = 0; r < WINDOW; r++) {
                final int row = r;
                final INDArray stateIn = carried;
                Map<String, INDArray> step = settledOutputs(() -> sd.output(Map.of("q", row(qs, row),
                        "k", row(ks, row), "v", row(vs, row), "beta", row(betas, row), "gate", row(gates, row),
                        "state", stateIn.dup(), "length", Nd4j.scalar(1L)), "out", "state_out"),
                        "gated delta rule width-1 row " + row);
                rows.add(step.get("out"));
                carried = step.get("state_out");
            }
            assertRows("gated delta rule", windowOutputs.get("out"), rows);
            assertBits("gated delta rule final state", windowOutputs.get("state_out"), carried);
        }
    }

    /** Linear-attention layer: depthwise causal conv1d (SiLU) with rolling state. */
    @ParameterizedTest(name = "causal conv1d")
    @ValueSource(ints = {0})
    void causalConv1d(int unused) {
        final int channels = 10240, kernel = 4;
        DataType dtype = DataType.BFLOAT16;
        try (SameDiff sd = graph()) {
            SDVariable x = sd.placeHolder("x", dtype, 1, -1, channels);
            SDVariable weight = sd.constant("weight", random(dtype, channels, kernel));
            SDVariable state = sd.placeHolder("state", dtype, 1, channels, kernel - 1);
            SDVariable length = sd.placeHolder("length", DataType.INT64);
            sd.nn().causalConv1d(new String[]{"out", "state_out"}, x, weight, null, state, length, 1, 0);
            sd.setOutputs("out", "state_out");

            INDArray xs = random(dtype, 1, WINDOW, channels);
            INDArray initial = random(dtype, 1, channels, kernel - 1);
            Map<String, INDArray> windowOutputs = settledOutputs(() -> sd.output(Map.of("x", xs,
                    "state", initial.dup(), "length", Nd4j.scalar((long) WINDOW)), "out", "state_out"),
                    "causal conv1d window");
            List<INDArray> rows = new ArrayList<>();
            INDArray carried = initial.dup();
            for (int r = 0; r < WINDOW; r++) {
                final int row = r;
                final INDArray stateIn = carried;
                Map<String, INDArray> step = settledOutputs(() -> sd.output(Map.of("x", row(xs, row),
                        "state", stateIn.dup(), "length", Nd4j.scalar(1L)), "out", "state_out"),
                        "causal conv1d width-1 row " + row);
                rows.add(step.get("out"));
                carried = step.get("state_out");
            }
            assertRows("causal conv1d", windowOutputs.get("out"), rows);
            assertBits("causal conv1d final state", windowOutputs.get("state_out"), carried);
        }
    }

    /** Row-wise reductions: the fused RMSNorm and the q/k l2 normalization of the linear-attention layer. */
    @ParameterizedTest(name = "{0}")
    @EnumSource(RowReduction.class)
    void rowReductions(RowReduction reduction) {
        try (SameDiff sd = graph()) {
            SDVariable x = sd.placeHolder("x", reduction.dtype, reduction.shape(-1));
            switch (reduction) {
                case RMS_NORM: {
                    SDVariable gamma = sd.constant("gamma", random(reduction.dtype, reduction.width));
                    new RmsNorm(sd, x, gamma, 1e-6).outputVariable().rename("out");
                    break;
                }
                case HEAD_RMS_NORM: {
                    SDVariable gamma = sd.constant("gamma", random(reduction.dtype, reduction.width));
                    new RmsNorm(sd, x, gamma, 1e-6).outputVariable().rename("out");
                    break;
                }
                case L2_NORM: {
                    SDVariable norm = sd.math.sqrt(x.mul(x).sum(true, -1).add(1e-6));
                    x.div("out", norm);
                    break;
                }
                default:
                    throw new IllegalStateException(reduction.name());
            }
            sd.setOutputs("out");
            INDArray xs = random(reduction.dtype, reduction.shape(WINDOW));
            INDArray windowOut = settled(() -> sd.output(Map.of("x", xs), "out").get("out").dup(),
                    reduction + " window");
            List<INDArray> rows = new ArrayList<>();
            for (int r = 0; r < WINDOW; r++) {
                final int row = r;
                rows.add(settled(() -> sd.output(Map.of("x", row(xs, row)), "out").get("out").dup(),
                        reduction + " width-1 row " + row));
            }
            assertRows(reduction.toString(), windowOut, rows);
        }
    }

    /** Packed projections: FP8 (cuBLASLt, decode-class algorithm) and NVFP4 (weight-only tensor-core GEMM). */
    @ParameterizedTest(name = "{0}")
    @EnumSource(QuantizedProjection.class)
    void quantizedLinear(QuantizedProjection projection) {
        java.util.Random random = new java.util.Random(20260928L);
        final int n = projection.columns, k = projection.depth;
        final boolean nvfp4 = projection.nvfp4;
        byte[] weightBytes = new byte[nvfp4 ? n * k / 2 : n * k];
        for (int i = 0; i < weightBytes.length; i++) {
            // NVFP4 nibbles are all finite; FP8 E4M3 avoids the NaN encodings (exponent+mantissa all ones).
            weightBytes[i] = nvfp4 ? (byte) random.nextInt(256)
                    : (byte) ((random.nextBoolean() ? 0x80 : 0) | random.nextInt(0x70));
        }
        INDArray weights = raw(nvfp4 ? DataType.UBYTE : DataType.FLOAT8, weightBytes, n, nvfp4 ? k / 2 : k);
        INDArray scale;
        if (nvfp4) {
            byte[] blocks = new byte[n * (k / 16)];
            for (int i = 0; i < blocks.length; i++) blocks[i] = (byte) (0x30 + random.nextInt(0x11));
            scale = raw(DataType.FLOAT8, blocks, n, k / 16);
        } else {
            scale = Nd4j.scalar(DataType.FLOAT, 0.0123f);
        }
        INDArray second = Nd4j.scalar(DataType.FLOAT, nvfp4 ? 0.1003f : 0.0457f);
        try (SameDiff sd = graph()) {
            SDVariable x = sd.placeHolder("x", DataType.BFLOAT16, 1, -1, k);
            SDVariable w = sd.constant("w", weights);
            SDVariable s = sd.constant("scale", scale);
            SDVariable t = sd.constant("second", second);
            (nvfp4 ? new org.nd4j.linalg.api.ops.impl.transforms.custom.ModelOptNvfp4Linear(sd, x, w, s, t, false)
                    : new org.nd4j.linalg.api.ops.impl.transforms.custom.ModelOptFp8Linear(sd, x, w, s, t, false))
                    .outputVariable().rename("out");
            sd.setOutputs("out");
            INDArray xs = random(DataType.BFLOAT16, 1, WINDOW, k);
            INDArray windowOut = settled(() -> sd.output(Map.of("x", xs), "out").get("out").dup(),
                    projection + " window");
            List<INDArray> rows = new ArrayList<>();
            for (int r = 0; r < WINDOW; r++) {
                final int row = r;
                rows.add(settled(() -> sd.output(Map.of("x", row(xs, row)), "out").get("out").dup(),
                        projection + " width-1 row " + row));
            }
            assertRows(projection.toString(), windowOut, rows);
        }
    }

    enum QuantizedProjection {
        FP8_QKV(false, 5120, 10240),
        NVFP4_GATE(true, 5120, 17408),
        NVFP4_DOWN(true, 17408, 5120);

        final boolean nvfp4;
        final int depth;
        final int columns;

        QuantizedProjection(boolean nvfp4, int depth, int columns) {
            this.nvfp4 = nvfp4;
            this.depth = depth;
            this.columns = columns;
        }
    }

    /** Typed one-byte storage filled with raw encodings (no numeric conversion). */
    private static INDArray raw(DataType dtype, byte[] bytes, long... shape) {
        INDArray array = Nd4j.createUninitialized(dtype, shape, 'c');
        new org.bytedeco.javacpp.BytePointer(array.data().pointer()).capacity(bytes.length).put(bytes);
        Nd4j.getAffinityManager().tagLocation(array, org.nd4j.linalg.api.concurrency.AffinityManager.Location.HOST);
        return array;
    }

    enum RowReduction {
        RMS_NORM(DataType.BFLOAT16, 5120, new long[0]),
        HEAD_RMS_NORM(DataType.FLOAT, 128, new long[]{48}),
        L2_NORM(DataType.FLOAT, 128, new long[]{48});

        final DataType dtype;
        final int width;
        final long[] heads;

        RowReduction(DataType dtype, int width, long[] heads) {
            this.dtype = dtype;
            this.width = width;
            this.heads = heads;
        }

        long[] shape(long rows) {
            long[] shape = new long[3 + heads.length];
            shape[0] = 1;
            shape[1] = rows;
            System.arraycopy(heads, 0, shape, 2, heads.length);
            shape[shape.length - 1] = width;
            return shape;
        }
    }

    // ── harness ────────────────────────────────────────────────────────────

    private interface Step {
        INDArray run();
    }

    private interface MultiStep {
        Map<String, INDArray> run();
    }

    private static SameDiff graph() {
        SameDiff sd = SameDiff.create();
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);
        return sd;
    }

    /**
     * Runs one launch through the plan's whole lifecycle (native slot-by-slot
     * warmup, freeze, compiled capture and replay) with identical inputs. Every
     * repetition must give the first one's bits: compiled execution is exact
     * with native execution. Returns the replayed result.
     */
    private static INDArray settled(Step step, String context) {
        INDArray first = step.run();
        INDArray last = first;
        for (int i = 1; i < WARMUP; i++) {
            last = step.run();
            assertBits(context + " repetition " + i + " (native warmup vs compiled)", first, last);
        }
        return last;
    }

    private static Map<String, INDArray> settledOutputs(MultiStep step, String context) {
        Map<String, INDArray> first = copy(step.run());
        Map<String, INDArray> last = first;
        for (int i = 1; i < WARMUP; i++) {
            last = copy(step.run());
            for (String name : first.keySet()) assertBits(context + " " + name + " repetition " + i
                    + " (native warmup vs compiled)", first.get(name), last.get(name));
        }
        return last;
    }

    private static Map<String, INDArray> copy(Map<String, INDArray> outputs) {
        Map<String, INDArray> copies = new HashMap<>();
        outputs.forEach((name, array) -> copies.put(name, array.dup()));
        return copies;
    }

    private static void assertRows(String context, INDArray window, List<INDArray> rows) {
        assertEquals(WINDOW, window.size(1), context);
        for (int r = 0; r < rows.size(); r++) assertBits(context + " row " + r, row(window, r), rows.get(r));
    }

    /** Bitwise equality (every storage type here converts to DOUBLE exactly). */
    private static void assertBits(String context, INDArray expected, INDArray actual) {
        assertEquals(expected.dataType(), actual.dataType(), context);
        assertTrue(java.util.Arrays.equals(expected.shape(), actual.shape()),
                context + ": shapes " + java.util.Arrays.toString(expected.shape()) + " vs "
                        + java.util.Arrays.toString(actual.shape()));
        double[] e = expected.castTo(DataType.DOUBLE).dup('c').data().asDouble();
        double[] a = actual.castTo(DataType.DOUBLE).dup('c').data().asDouble();
        long mismatches = 0;
        int first = -1;
        double maxAbs = 0;
        for (int i = 0; i < e.length; i++) {
            if (Double.doubleToRawLongBits(e[i]) == Double.doubleToRawLongBits(a[i])) continue;
            if (mismatches++ == 0) first = i;
            maxAbs = Math.max(maxAbs, Math.abs(e[i] - a[i]));
        }
        assertEquals(0, mismatches, context + ": " + mismatches + " of " + e.length + " elements differ, first at "
                + first + " (" + (first >= 0 ? e[first] + " vs " + a[first] : "") + "), max |diff| " + maxAbs);
    }

    private static INDArrayIndex all() {
        return NDArrayIndex.all();
    }

    /** Row r of a [1, rows, ...] tensor as a [1, 1, ...] copy. */
    private static INDArray row(INDArray array, int r) {
        INDArrayIndex[] index = new INDArrayIndex[array.rank()];
        for (int d = 0; d < index.length; d++) index[d] = NDArrayIndex.all();
        index[1] = NDArrayIndex.interval(r, r + 1);
        return array.get(index).dup('c');
    }

    private static INDArray random(DataType dtype, long... shape) {
        return Nd4j.randn(DataType.FLOAT, shape).castTo(dtype);
    }

    private static INDArray unitRows(INDArray x) {
        INDArray norm = x.mul(x).sum(true, -1);
        return x.div(org.nd4j.linalg.ops.transforms.Transforms.sqrt(norm, false));
    }

    /** Additive causal bias for rows [firstRow, firstRow + rows) of a window starting at cache position start. */
    private static INDArray causalMask(int start, int firstRow, int rows, int cache) {
        INDArray mask = Nd4j.valueArrayOf(new long[]{1, 1, rows, cache}, MASKED, DataType.FLOAT);
        for (int r = 0; r < rows; r++) {
            mask.get(NDArrayIndex.point(0), NDArrayIndex.point(0), NDArrayIndex.point(r),
                    NDArrayIndex.interval(0, start + firstRow + r + 1)).assign(0.0f);
        }
        return mask;
    }
}
