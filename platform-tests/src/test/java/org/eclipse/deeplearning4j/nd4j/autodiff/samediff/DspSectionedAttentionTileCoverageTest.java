/* SPDX-License-Identifier: Apache-2.0 */
package org.eclipse.deeplearning4j.nd4j.autodiff.samediff;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.diagnostics.DspDiagnostics;
import org.nd4j.autodiff.samediff.execution.DspPlanAssertions;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlan;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.NativeOpsHolder;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.regex.Pattern;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Triton attention must cover every (batch * head, q tile) block and apply the causal
 * flag, scale and bias it was given, whether it compiles as its own kernel or behind
 * another section of one.
 *
 * <p>Crawl r12 (2026-09-28) compiled a Qwen3.5 prefill range into a two-phase Triton
 * kernel: an elementwise norm/rope section, then dot_product_attention_v2 over a
 * [1,1,3072,4096] bias. The section planner had read the op's [batch, seq, heads,
 * headDim] q as [batch, heads, seq, headDim], so it kept a prefill of many q tiles
 * behind the norm section. There the attention launched on the kernel's 1D grid: every
 * block read program_id(y) == 0 as its q tile and decoded batch = program_id(x) / heads
 * out of range, and the first composite replay faulted with CUDA 700.</p>
 *
 * <p>The planner and the emitters now read one attention contract. Prefill, with more
 * than one q tile per head, compiles as its own kernel on the 2D grid. A decode step,
 * one q tile per head, stays behind the section before it and must find its
 * (batch * head) block in the 1D program id.</p>
 *
 * <p>The Triton emitters also ignored dot_product_attention_v2's causal flag and scale:
 * a causal decoder without a bias ran unmasked, and a scale of 1.0 ran as
 * 1/sqrt(headDim).</p>
 */
class DspSectionedAttentionTileCoverageTest {
    private static final int Q_HEADS = 8;
    private static final int KV_HEADS = 2;
    private static final int HEAD_DIM = 256;
    /** Six q tiles at the BM=16 tile chosen for headDim 256. */
    private static final int SEQ = 96;
    private static final int CACHE = 128;
    /** The cache row of the first decode step; each later step writes the next row. */
    private static final int DECODE_POSITION = 40;
    /** Covers compile, slot-by-slot, capture and several replays. */
    private static final int STEPS = 6;
    private static final double TOLERANCE = 1e-2;
    private static final String LEADING_ATTENTION_SECTION = "SECTION[0] type=FUSED_ATTENTION";
    private static final Pattern TRAILING_ATTENTION_SECTION =
            Pattern.compile("SECTION\\[[1-9][0-9]*\\] type=FUSED_ATTENTION");

    /** The Qwen3.5 serving form: causal=false, the mask is a bias over the KV cache. */
    @Test
    void cachedPrefillMatchesReference() {
        assumeTriton();
        long[] cacheShape = {1, CACHE, KV_HEADS, HEAD_DIM};
        try (INDArray keyCache = Nd4j.create(DataType.FLOAT, cacheShape);
             INDArray valueCache = Nd4j.create(DataType.FLOAT, cacheShape);
             INDArray position = Nd4j.scalar(DataType.INT64, 0L);
             INDArray bias = Nd4j.createFromArray(maskAboveDiagonal(new float[SEQ * CACHE], CACHE))
                     .reshape(1, 1, SEQ, CACHE)) {
            Map<String, INDArray> feeds = Map.of("key_cache", keyCache, "value_cache", valueCache,
                    "cache_position", position, "bias", bias);
            // The in-place write puts K/V in cache rows [0, SEQ); the bias masks the rest,
            // which leaves causal attention over the SEQ keys.
            check(true, SEQ, Q_HEADS, KV_HEADS, feeds,
                    (sd, q, k, v) -> sd.nn().dotProductAttentionV2("attn", q, v, k, null, null,
                            sd.placeHolder("key_cache", DataType.FLOAT, cacheShape),
                            sd.placeHolder("value_cache", DataType.FLOAT, cacheShape),
                            sd.placeHolder("cache_position", DataType.INT64),
                            sd.placeHolder("bias", DataType.FLOAT, 1, 1, SEQ, CACHE),
                            0.0, 0.0, false, false),
                    false,
                    (step, q, w, k, v) -> reference(q, w, k, v, SEQ, SEQ, Q_HEADS, KV_HEADS, 0.0, true, null));
        }
    }

    /**
     * The Qwen3.5 serving decode step: one query row over the cache, its K/V written at
     * cache_position and the bias masking every later row.
     */
    @Test
    void cachedDecodeMatchesReference() {
        assumeTriton();
        long[] cacheShape = {1, CACHE, KV_HEADS, HEAD_DIM};
        int rowValues = KV_HEADS * HEAD_DIM;
        try (INDArray keyCache = Nd4j.create(DataType.FLOAT, cacheShape);
             INDArray valueCache = Nd4j.create(DataType.FLOAT, cacheShape);
             INDArray position = Nd4j.scalar(DataType.INT64, 0L);
             INDArray bias = Nd4j.create(DataType.FLOAT, 1, 1, 1, CACHE)) {
            Map<String, INDArray> feeds = Map.of("key_cache", keyCache, "value_cache", valueCache,
                    "cache_position", position, "bias", bias);
            check(true, 1, Q_HEADS, KV_HEADS, feeds,
                    (sd, q, k, v) -> sd.nn().dotProductAttentionV2("attn", q, v, k, null, null,
                            sd.placeHolder("key_cache", DataType.FLOAT, cacheShape),
                            sd.placeHolder("value_cache", DataType.FLOAT, cacheShape),
                            sd.placeHolder("cache_position", DataType.INT64),
                            sd.placeHolder("bias", DataType.FLOAT, 1, 1, 1, CACHE),
                            0.0, 0.0, false, false),
                    true,
                    (step, q, w, k, v) -> {
                        int row = DECODE_POSITION + step;
                        // Fresh cache contents every step, so the step reads rows the
                        // reference knows rather than what an earlier step wrote.
                        try (INDArray keyRows = Nd4j.randn(DataType.FLOAT, cacheShape);
                             INDArray valueRows = Nd4j.randn(DataType.FLOAT, cacheShape)) {
                            keyCache.assign(keyRows);
                            valueCache.assign(valueRows);
                        }
                        position.assign(row);
                        float[] mask = new float[CACHE];
                        Arrays.fill(mask, row + 1, CACHE, -1.0e9f);
                        try (INDArray maskValues = Nd4j.createFromArray(mask).reshape(1, 1, 1, CACHE)) {
                            bias.assign(maskValues);
                        }
                        // Keys 0..row: the cache rows before the step's own, then its K/V.
                        float[] keys = Arrays.copyOf(keyCache.data().asFloat(), (row + 1) * rowValues);
                        float[] values = Arrays.copyOf(valueCache.data().asFloat(), (row + 1) * rowValues);
                        System.arraycopy(k, 0, keys, row * rowValues, rowValues);
                        System.arraycopy(v, 0, values, row * rowValues, rowValues);
                        return reference(q, w, keys, values, 1, row + 1, Q_HEADS, KV_HEADS, 0.0, false, null);
                    });
        }
    }

    /** The plain decoder form: the op's causal flag is the only mask. 0 means 1/sqrt(headDim). */
    @ParameterizedTest(name = "scale={0}")
    @ValueSource(doubles = {0.0, 1.0})
    void causalPrefillMatchesReference(double scale) {
        assumeTriton();
        check(true, SEQ, Q_HEADS, KV_HEADS, Map.of(),
                (sd, q, k, v) -> sd.nn().dotProductAttentionV2("attn", q, v, k, null, null,
                        null, null, null, null, scale, 0.0, true, false),
                false,
                (step, q, w, k, v) -> reference(q, w, k, v, SEQ, SEQ, Q_HEADS, KV_HEADS, scale, true, null));
    }

    /**
     * Rank-3 single-head attention with a rank-2 [seqQ, seqK] bias, which the native
     * CUDA helper used to apply as its first element on every score. DSP off runs the
     * native op; DSP on runs the Triton emitter.
     */
    @ParameterizedTest(name = "dsp={0}")
    @ValueSource(booleans = {false, true})
    void singleHeadPrefillWithSquareBiasMatchesReference(boolean dsp) {
        assumeTriton();
        float[] biasValues = new float[SEQ * SEQ];
        for (int row = 0; row < SEQ; row++) {
            for (int key = 0; key <= row; key++) {
                biasValues[row * SEQ + key] = 0.05f * ((row * 7 + key * 3) % 11 - 5);
            }
        }
        maskAboveDiagonal(biasValues, SEQ);
        try (INDArray bias = Nd4j.createFromArray(biasValues).reshape(SEQ, SEQ)) {
            check(dsp, SEQ, 1, 1, Map.of("bias", bias),
                    (sd, q, k, v) -> sd.nn().dotProductAttentionV2("attn", q, v, k, null, null,
                            sd.placeHolder("bias", DataType.FLOAT, SEQ, SEQ), 0.0, 0.0, false, false),
                    false,
                    (step, q, w, k, v) -> reference(q, w, k, v, SEQ, SEQ, 1, 1, 0.0, false, biasValues));
        }
    }

    /** Declares the op named "attn" over the weighted q and the k/v placeholders. */
    @FunctionalInterface
    private interface AttentionDeclaration {
        void declare(SameDiff sd, SDVariable q, SDVariable k, SDVariable v);
    }

    /** Sets a step's remaining feeds and returns the output expected for its q, weight, k and v. */
    @FunctionalInterface
    private interface StepExpectation {
        float[] prepare(int step, float[] q, float[] weight, float[] k, float[] v);
    }

    /**
     * Runs {@link #STEPS} steps with fresh q/k/v values, each checked against what
     * {@code expectation} returns. q is multiplied by a weight constant first, the
     * elementwise section ahead of attention that stands in for the crawl kernel's
     * norm/rope section. Multi-head q/k/v are [1, seq, heads, headDim]; single-head ones
     * are [1, seq, headDim]. With DSP on, the attention builds without falling back to the
     * native op and replays. It compiles behind the q section when {@code trailing} and
     * automatic placement put both ops on one device, and as its own kernel otherwise; a
     * trailing case that placement split is reported as skipped once the rest has passed.
     */
    private static void check(boolean dsp, int seq, int qHeads, int kvHeads, Map<String, INDArray> extraFeeds,
                              AttentionDeclaration declaration, boolean trailing, StepExpectation expectation) {
        long[] qShape = qHeads == 1 ? new long[] {1, seq, HEAD_DIM} : new long[] {1, seq, qHeads, HEAD_DIM};
        long[] kvShape = kvHeads == 1 ? new long[] {1, seq, HEAD_DIM} : new long[] {1, seq, kvHeads, HEAD_DIM};
        boolean enabled = InferenceSession.isDynamicShapePlanEnabled();
        int mask = Nd4j.getNativeOps().dspDiagGetEnabledMask();
        int level = Nd4j.getNativeOps().dspDiagGetLevel();
        try {
            InferenceSession.setDynamicShapePlanEnabled(dsp);
            // The serving profile: section fusion, attention in Triton, graph capture.
            Nd4j.getEnvironment().applyOptimalLLMConfig();
            DspDiagnostics.setCategories(DspDiagnostics.COMPILE);
            DspDiagnostics.setLevel(DspDiagnostics.LEVEL_FULL);
            DspDiagnostics.clear();
            Nd4j.getRandom().setSeed(20260928L);
            // The graph is declared last so it closes before the arrays it was fed.
            try (INDArray weight = Nd4j.rand(DataType.FLOAT, HEAD_DIM).addi(0.5);
                 INDArray q = Nd4j.create(DataType.FLOAT, qShape);
                 INDArray k = Nd4j.create(DataType.FLOAT, kvShape);
                 INDArray v = Nd4j.create(DataType.FLOAT, kvShape);
                 SameDiff sd = SameDiff.create()) {
                SDVariable weighted = sd.placeHolder("q", DataType.FLOAT, qShape)
                        .mul(sd.constant("q_weight", weight));
                declaration.declare(sd, weighted, sd.placeHolder("k", DataType.FLOAT, kvShape),
                        sd.placeHolder("v", DataType.FLOAT, kvShape));
                Map<String, INDArray> feeds = new HashMap<>(extraFeeds);
                feeds.put("q", q);
                feeds.put("k", k);
                feeds.put("v", v);

                float[] w = weight.data().asFloat();
                for (int step = 0; step < STEPS; step++) {
                    // New values every step, so a block left unwritten cannot pass on
                    // an earlier step's output.
                    try (INDArray qValues = Nd4j.randn(DataType.FLOAT, qShape);
                         INDArray kValues = Nd4j.randn(DataType.FLOAT, kvShape);
                         INDArray vValues = Nd4j.randn(DataType.FLOAT, kvShape)) {
                        q.assign(qValues);
                        k.assign(kValues);
                        v.assign(vValues);
                    }
                    float[] expected = expectation.prepare(step, q.data().asFloat(), w, k.data().asFloat(),
                            v.data().asFloat());
                    INDArray out = sd.outputSingle(feeds, "attn");
                    try (INDArray actual = out.dup('c')) {
                        assertMatches(step, qHeads, expected, actual.data().asFloat());
                    }
                }

                if (dsp) {
                    String report = DspDiagnostics.getJsonReport();
                    DynamicShapePlan plan = sd.getOrCreateSession().getDynamicShapePlanExecutor().getCurrentPlan();
                    int qDevice = deviceOf(plan, weighted.name());
                    int attentionDevice = deviceOf(plan, "attn");
                    boolean together = qDevice == attentionDevice;
                    String placement = "; " + plan.getDeviceAssignmentSummary() + ", q section on device "
                            + qDevice + ", attention on device " + attentionDevice;
                    if (trailing && together) {
                        assertTrue(TRAILING_ATTENTION_SECTION.matcher(report).find(),
                                "one q tile per head keeps the attention behind the q section, on that "
                                        + "kernel's 1D grid" + placement);
                    } else {
                        assertTrue(report.contains(LEADING_ATTENTION_SECTION),
                                "an attention of several q tiles per head, or one placed on another device "
                                        + "than the q section, must compile as its own kernel" + placement);
                        assertFalse(TRAILING_ATTENTION_SECTION.matcher(report).find(),
                                "an attention of several q tiles per head must not trail another section, "
                                        + "the kernel the crawl faulted in" + placement);
                    }
                    assertFalse(report.contains("outside the emitter contract"),
                            "the section planner must read this attention's dimensions from the emitter contract");
                    assertFalse(report.contains("buildModule EXCEPTION"),
                            "every Triton range must build, including its attention section");
                    assertFalse(report.contains("runs natively"),
                            "the Triton emitters must take this attention instead of leaving it to the native op");
                    DspPlanAssertions.assertTotalGraphReplaysAtLeast(sd, 1,
                            "the composite replay is where the crawl faulted");
                    // Automatic placement decides whether the two ops share a device, and tests
                    // do not pin it. A split decode passed everything above but never trailed.
                    assumeTrue(!trailing || together, "placement split the q section from the attention, "
                            + "so the trailing attention section was not exercised" + placement);
                }
            }
        } finally {
            DspDiagnostics.clear();
            DspDiagnostics.setCategories(mask);
            DspDiagnostics.setLevel(level);
            InferenceSession.setDynamicShapePlanEnabled(enabled);
        }
    }

    /** The device automatic placement gave the op that produces {@code output}. */
    private static int deviceOf(DynamicShapePlan plan, String output) {
        for (var slot : plan.getSlots()) {
            if (Arrays.asList(slot.getOutputVarNames()).contains(output)) return slot.getTargetDeviceId();
        }
        throw new AssertionError("no DSP slot produces " + output + ": " + plan.getDeviceAssignmentSummary());
    }

    private static void assumeTriton() {
        assumeTrue(Nd4j.backends().isCudaAvailable(), "Requires CUDA with Triton");
        assumeTrue(NativeOpsHolder.getInstance().getDeviceNativeOps().isTritonAvailable(),
                "Requires Triton");
    }

    /** Sets -1e9 on every key after the row's own in a [SEQ, keys] bias, so row i sees keys 0..i. */
    private static float[] maskAboveDiagonal(float[] values, int keys) {
        for (int row = 0; row < SEQ; row++) {
            for (int key = row + 1; key < keys; key++) values[row * keys + key] = -1.0e9f;
        }
        return values;
    }

    /**
     * GQA attention of {@code rows} query rows over {@code keyCount} keys: (q * weight) is
     * [rows, qHeads, headDim], k and v are [keyCount, kvHeads, headDim]. A zero scale means
     * 1/sqrt(headDim); causal row i sees keys 0..i; a non-null bias is [rows, keyCount],
     * added to the scaled scores.
     */
    private static float[] reference(float[] q, float[] weight, float[] k, float[] v, int rows, int keyCount,
                                     int qHeads, int kvHeads, double scale, boolean causal, float[] bias) {
        int group = qHeads / kvHeads;
        double factor = scale == 0 ? 1.0 / Math.sqrt(HEAD_DIM) : scale;
        float[] out = new float[rows * qHeads * HEAD_DIM];
        double[] scores = new double[keyCount];
        for (int row = 0; row < rows; row++) {
            int keys = causal ? row + 1 : keyCount;
            for (int head = 0; head < qHeads; head++) {
                int kvHead = head / group;
                int qBase = (row * qHeads + head) * HEAD_DIM;
                double max = Double.NEGATIVE_INFINITY;
                for (int key = 0; key < keys; key++) {
                    int kBase = (key * kvHeads + kvHead) * HEAD_DIM;
                    double dot = 0;
                    for (int d = 0; d < HEAD_DIM; d++) dot += q[qBase + d] * weight[d] * k[kBase + d];
                    scores[key] = dot * factor + (bias == null ? 0 : bias[row * keyCount + key]);
                    max = Math.max(max, scores[key]);
                }
                double sum = 0;
                for (int key = 0; key < keys; key++) {
                    scores[key] = Math.exp(scores[key] - max);
                    sum += scores[key];
                }
                for (int d = 0; d < HEAD_DIM; d++) {
                    double acc = 0;
                    for (int key = 0; key < keys; key++) {
                        acc += scores[key] * v[(key * kvHeads + kvHead) * HEAD_DIM + d];
                    }
                    out[qBase + d] = (float) (acc / sum);
                }
            }
        }
        return out;
    }

    /** Names the first diverging q row, so a missing tile reads as row 16, 32, .... */
    private static void assertMatches(int step, int qHeads, float[] expected, float[] actual) {
        assertEquals(expected.length, actual.length, "attention output length at step " + step);
        int rowStride = qHeads * HEAD_DIM;
        for (int index = 0; index < expected.length; index++) {
            double diff = Math.abs(expected[index] - actual[index]);
            if (!(diff <= TOLERANCE)) {
                fail("step " + step + ": first mismatch at q row " + index / rowStride + ", head "
                        + (index % rowStride) / HEAD_DIM + ": expected " + expected[index]
                        + ", actual " + actual[index]);
            }
        }
    }
}
