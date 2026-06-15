/*
 *  ******************************************************************************
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */
package org.eclipse.deeplearning4j.frameworks.samediff.dsp.frozen;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.*;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.DspHandle;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Environment;
import org.nd4j.linalg.factory.Nd4j;

import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests that DSP plan outputs maintain correct rank across lifecycle transitions.
 *
 * Reproduces the staging server crash where autoregressive_decode received a rank-0
 * logits output (expected rank-2 or rank-3) after:
 *   1. markExternalInputVariable invalidation resets execCount to 0
 *   2. SHAPE_KEY_MATCHED bumps execCount to 2 ("already compiled")
 *   3. Plan output slot populated from stale/null shape cache → rank-0 scalar
 *
 * These tests verify that output arrays always have the expected rank regardless
 * of DSP execution mode, lifecycle phase, or invalidation events.
 */
@Slf4j
@NativeTag
@Tag(TagNames.SAMEDIFF)
@DisplayName("DSP Output Rank Stability")
public class TestDspOutputRankStability {

    private static final int BATCH = 1;
    private static final int SEQ_LEN = 4;
    private static final int VOCAB_SIZE = 32;
    private static final int HIDDEN = 16;
    private static final int WARMUP_RUNS = 6;
    private static final long GRAPH_SEED = 777L;

    enum ExecMode {
        NO_DSP,
        DSP_SLOT_ONLY,
        DSP_FULL
    }

    private static void applyMode(SameDiff sd, ExecMode mode) {
        Environment env = Nd4j.getEnvironment();
        switch (mode) {
            case NO_DSP:
                sd.setDspAutoCompileEnabled(false);
                sd.setDspNativeAutoCompileEnabled(false);
                sd.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);
                env.setTritonGraphCapture(false);
                break;
            case DSP_SLOT_ONLY:
                sd.setDspAutoCompileEnabled(true);
                sd.setDspNativeAutoCompileEnabled(true);
                sd.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);
                env.setTritonGraphCapture(false);
                env.setDspFreezeMergeSegments(false);
                break;
            case DSP_FULL:
                sd.setDspAutoCompileEnabled(true);
                sd.setDspNativeAutoCompileEnabled(true);
                sd.setGraphExecutionMode(GraphExecutionMode.AUTO);
                env.setTritonGraphCapture(true);
                env.setDspFreezeMergeSegments(true);
                env.setTritonConsolidatedArgTable(true);
                env.setTritonArgDirtyTracking(true);
                env.setTritonCompileAll(true);
                break;
        }
    }

    @AfterEach
    void teardown() {
        Environment env = Nd4j.getEnvironment();
        env.setTritonGraphCapture(true);
        env.setDspFreezeMergeSegments(true);
        env.setTritonConsolidatedArgTable(true);
        env.setTritonArgDirtyTracking(true);
        env.setTritonCompileAll(true);
    }

    private static INDArray detArray(java.util.Random rng, int rows, int cols, float scale, float offset) {
        float[] data = new float[rows * cols];
        for (int i = 0; i < data.length; i++) {
            data[i] = (float) rng.nextGaussian() * scale + offset;
        }
        return Nd4j.createFromArray(data).reshape(rows, cols);
    }

    private static INDArray createDeterministicInput2D(int step) {
        java.util.Random jRng = new java.util.Random(42L + step);
        float[] data = new float[BATCH * HIDDEN];
        for (int i = 0; i < data.length; i++) {
            data[i] = (float) jRng.nextGaussian();
        }
        return Nd4j.createFromArray(data).reshape(BATCH, HIDDEN);
    }

    private static INDArray createDeterministicInput3D(int step) {
        java.util.Random jRng = new java.util.Random(42L + step * 1000);
        float[] data = new float[BATCH * SEQ_LEN * HIDDEN];
        for (int i = 0; i < data.length; i++) {
            data[i] = (float) jRng.nextGaussian();
        }
        return Nd4j.createFromArray(data).reshape(BATCH, SEQ_LEN, HIDDEN);
    }

    // ═══════════════════════════════════════════════════════════════════════════════
    // Graph builders — simulate LLM logits-producing patterns
    // ═══════════════════════════════════════════════════════════════════════════════

    /**
     * Builds a graph that produces rank-3 output [batch, seqLen, vocabSize]
     * simulating an LLM decoder that produces logits over vocabulary.
     * input [batch, seqLen, hidden] → matmul → relu → projection → [batch, seqLen, vocabSize]
     */
    private static SameDiff buildRank3LogitsGraph(java.util.Random rng) {
        SameDiff sd = SameDiff.create();

        SDVariable x = sd.placeHolder("x", DataType.FLOAT, BATCH, SEQ_LEN, HIDDEN);

        // Reshape to 2D for matmul: [batch*seqLen, hidden]
        SDVariable flat = sd.reshape("flat", x, BATCH * SEQ_LEN, HIDDEN);

        // Hidden layer
        SDVariable w1 = sd.var("w1", detArray(rng, HIDDEN, HIDDEN, 0.1f, 0.0f));
        SDVariable h = sd.mmul("mm1", flat, w1);
        h = sd.nn().relu("relu1", h, 0);

        // Projection to vocab: [batch*seqLen, vocabSize]
        SDVariable wProj = sd.var("w_proj", detArray(rng, HIDDEN, VOCAB_SIZE, 0.1f, 0.0f));
        SDVariable logitsFlat = sd.mmul("mm_proj", h, wProj);

        // Reshape back to 3D: [batch, seqLen, vocabSize]
        SDVariable logits = sd.reshape("out", logitsFlat, BATCH, SEQ_LEN, VOCAB_SIZE);

        sd.setOutputs("out");
        return sd;
    }

    /**
     * Builds a graph that produces rank-2 output [batch, vocabSize]
     * simulating an LLM decoder producing logits for a single decode step.
     */
    private static SameDiff buildRank2LogitsGraph(java.util.Random rng) {
        SameDiff sd = SameDiff.create();

        SDVariable x = sd.placeHolder("x", DataType.FLOAT, BATCH, HIDDEN);

        SDVariable w1 = sd.var("w1", detArray(rng, HIDDEN, HIDDEN, 0.1f, 0.0f));
        SDVariable h = sd.mmul("mm1", x, w1);
        h = sd.nn().relu("relu1", h, 0);

        SDVariable wProj = sd.var("w_proj", detArray(rng, HIDDEN, VOCAB_SIZE, 0.1f, 0.0f));
        SDVariable out = sd.mmul("out", h, wProj);

        sd.setOutputs("out");
        return sd;
    }

    /**
     * Builds a graph with multiple outputs — one rank-3 "logits" and one rank-2 "hidden_state".
     * Simulates the case where logitsOutputIdx must correctly identify the logits output.
     */
    private static SameDiff buildMultiOutputGraph(java.util.Random rng) {
        SameDiff sd = SameDiff.create();

        SDVariable x = sd.placeHolder("x", DataType.FLOAT, BATCH, SEQ_LEN, HIDDEN);
        SDVariable flat = sd.reshape("flat", x, BATCH * SEQ_LEN, HIDDEN);

        // Hidden layer — output as rank-2
        SDVariable w1 = sd.var("w1", detArray(rng, HIDDEN, HIDDEN, 0.1f, 0.0f));
        SDVariable h = sd.mmul("mm1", flat, w1);
        h = sd.nn().relu("hidden_state", h, 0);

        // Projection to vocab — output as rank-3
        SDVariable wProj = sd.var("w_proj", detArray(rng, HIDDEN, VOCAB_SIZE, 0.1f, 0.0f));
        SDVariable logitsFlat = sd.mmul("mm_proj", h, wProj);
        SDVariable logits = sd.reshape("logits", logitsFlat, BATCH, SEQ_LEN, VOCAB_SIZE);

        sd.setOutputs("hidden_state", "logits");
        return sd;
    }

    // ═══════════════════════════════════════════════════════════════════════════════
    // Test 1: Output rank preserved across repeated sd.output() calls
    // ═══════════════════════════════════════════════════════════════════════════════

    static Stream<Arguments> modeProvider() {
        List<Arguments> args = new ArrayList<>();
        for (ExecMode mode : ExecMode.values()) {
            args.add(Arguments.of(mode));
        }
        return args.stream();
    }

    @ParameterizedTest(name = "rank3_logits_stable | {0}")
    @MethodSource("modeProvider")
    @DisplayName("Rank-3 logits output rank preserved across repeated executions")
    void testRank3LogitsRankStable(ExecMode mode) {
        java.util.Random rng = new java.util.Random(GRAPH_SEED);
        SameDiff sd = buildRank3LogitsGraph(rng);
        applyMode(sd, mode);

        try {
            for (int step = 0; step < WARMUP_RUNS + 4; step++) {
                INDArray input = createDeterministicInput3D(step);
                Map<String, INDArray> ph = Collections.singletonMap("x", input);

                final int s = step;
                Map<String, INDArray> result = assertDoesNotThrow(
                        () -> sd.output(ph, "out"),
                        "Step " + s + " (" + mode + "): sd.output() must not throw");

                INDArray out = result.get("out");
                assertNotNull(out, "Step " + step + " (" + mode + "): output should not be null");

                assertEquals(3, out.rank(),
                        "Step " + step + " (" + mode + "): output rank must be 3 (got " + out.rank() +
                                ", shape=" + Arrays.toString(out.shape()) + ")");

                assertEquals(BATCH, out.size(0),
                        "Step " + step + ": batch dim mismatch");
                assertEquals(SEQ_LEN, out.size(1),
                        "Step " + step + ": seqLen dim mismatch");
                assertEquals(VOCAB_SIZE, out.size(2),
                        "Step " + step + ": vocabSize dim mismatch");

                assertFalse(out.isNaN().any(),
                        "Step " + step + " (" + mode + "): output contains NaN");
                assertFalse(out.isInfinite().any(),
                        "Step " + step + " (" + mode + "): output contains Inf");

                log.info("[RANK3-STABLE] mode={} step={} rank={} shape={}",
                        mode, step, out.rank(), Arrays.toString(out.shape()));
            }
        } finally {
            try { sd.close(); } catch (Exception ignored) {}
        }
    }

    @ParameterizedTest(name = "rank2_logits_stable | {0}")
    @MethodSource("modeProvider")
    @DisplayName("Rank-2 logits output rank preserved across repeated executions")
    void testRank2LogitsRankStable(ExecMode mode) {
        java.util.Random rng = new java.util.Random(GRAPH_SEED);
        SameDiff sd = buildRank2LogitsGraph(rng);
        applyMode(sd, mode);

        try {
            for (int step = 0; step < WARMUP_RUNS + 4; step++) {
                INDArray input = createDeterministicInput2D(step);
                Map<String, INDArray> ph = Collections.singletonMap("x", input);

                final int s = step;
                Map<String, INDArray> result = assertDoesNotThrow(
                        () -> sd.output(ph, "out"),
                        "Step " + s + " (" + mode + "): sd.output() must not throw");

                INDArray out = result.get("out");
                assertNotNull(out, "Step " + step + " (" + mode + "): output should not be null");

                assertEquals(2, out.rank(),
                        "Step " + step + " (" + mode + "): output rank must be 2 (got " + out.rank() +
                                ", shape=" + Arrays.toString(out.shape()) + ")");

                assertEquals(BATCH, out.size(0),
                        "Step " + step + ": batch dim mismatch");
                assertEquals(VOCAB_SIZE, out.size(1),
                        "Step " + step + ": vocabSize dim mismatch");

                log.info("[RANK2-STABLE] mode={} step={} rank={} shape={}",
                        mode, step, out.rank(), Arrays.toString(out.shape()));
            }
        } finally {
            try { sd.close(); } catch (Exception ignored) {}
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════════
    // Test 2: Output rank after markExternalInputVariable invalidation
    //         This reproduces the staging server pattern:
    //           warmup → markVariable → "already compiled" fast-path → execution
    // ═══════════════════════════════════════════════════════════════════════════════

    @ParameterizedTest(name = "rank3_after_markVariable | {0}")
    @MethodSource("modeProvider")
    @DisplayName("Rank-3 output rank preserved after markExternalInputVariable")
    void testRank3AfterMarkVariable(ExecMode mode) {
        java.util.Random rng = new java.util.Random(GRAPH_SEED);
        SameDiff sd = buildRank3LogitsGraph(rng);
        applyMode(sd, mode);

        try {
            // Phase 1: Initial warmup — compile the plan
            for (int step = 0; step < 3; step++) {
                INDArray input = createDeterministicInput3D(step);
                Map<String, INDArray> ph = Collections.singletonMap("x", input);
                Map<String, INDArray> result = sd.output(ph, "out");
                INDArray out = result.get("out");
                assertNotNull(out, "Warmup step " + step + " output null");
                assertEquals(3, out.rank(),
                        "Warmup step " + step + ": rank degraded to " + out.rank());
            }

            // Phase 2: markVariable on the placeholder — triggers invalidation + re-warmup
            DspHandle h = sd.dsp();
            if (h.isCompiled()) {
                int extIdx = h.extInputIndex("x");
                if (extIdx >= 0) {
                    log.info("[MARK-VAR] Marking placeholder 'x' at ext[{}] as variable", extIdx);
                    h.markVariable(extIdx);

                    DspHandle.StepSnapshot snap = h.captureStepSnapshot();
                    log.info("[MARK-VAR] Post-markVariable: execCount={} phase={}",
                            snap.executeCount, snap.planPhase);
                } else {
                    log.info("[MARK-VAR] 'x' not found in ext inputs, skipping markVariable");
                }
            } else {
                log.info("[MARK-VAR] Plan not compiled for mode={}, skipping markVariable", mode);
            }

            // Phase 3: Post-invalidation execution — this is where rank-0 would appear
            for (int step = 0; step < WARMUP_RUNS + 2; step++) {
                INDArray input = createDeterministicInput3D(step + 100);
                Map<String, INDArray> ph = Collections.singletonMap("x", input);

                final int s = step;
                Map<String, INDArray> result = assertDoesNotThrow(
                        () -> sd.output(ph, "out"),
                        "Post-mark step " + s + " (" + mode + "): sd.output() must not throw");

                INDArray out = result.get("out");
                assertNotNull(out, "Post-mark step " + step + " (" + mode + "): output is null");

                assertTrue(out.rank() >= 2,
                        "Post-mark step " + step + " (" + mode + "): output rank is " +
                                out.rank() + " (expected >= 2, shape=" +
                                Arrays.toString(out.shape()) + "). " +
                                "This is the staging server rank-0 bug!");

                assertEquals(3, out.rank(),
                        "Post-mark step " + step + " (" + mode + "): output rank must be 3 (got " +
                                out.rank() + ", shape=" + Arrays.toString(out.shape()) + ")");

                assertEquals(BATCH, out.size(0));
                assertEquals(SEQ_LEN, out.size(1));
                assertEquals(VOCAB_SIZE, out.size(2));

                assertFalse(out.isNaN().any(),
                        "Post-mark step " + step + " (" + mode + "): NaN in output");

                log.info("[MARK-VAR] mode={} post-mark step={} rank={} shape={}",
                        mode, step, out.rank(), Arrays.toString(out.shape()));
            }
        } finally {
            try { sd.close(); } catch (Exception ignored) {}
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════════
    // Test 3: Multi-output graph — rank preserved for BOTH outputs
    //         The staging bug was specifically that logitsOutputIdx pointed to
    //         a slot whose output had degraded rank.
    // ═══════════════════════════════════════════════════════════════════════════════

    @ParameterizedTest(name = "multi_output_rank_stable | {0}")
    @MethodSource("modeProvider")
    @DisplayName("Multi-output graph preserves rank for all outputs across lifecycle")
    void testMultiOutputRankStable(ExecMode mode) {
        java.util.Random rng = new java.util.Random(GRAPH_SEED);
        SameDiff sd = buildMultiOutputGraph(rng);
        applyMode(sd, mode);

        try {
            for (int step = 0; step < WARMUP_RUNS + 4; step++) {
                INDArray input = createDeterministicInput3D(step);
                Map<String, INDArray> ph = Collections.singletonMap("x", input);

                final int s = step;
                Map<String, INDArray> result = assertDoesNotThrow(
                        () -> sd.output(ph, "hidden_state", "logits"),
                        "Step " + s + " (" + mode + "): sd.output() must not throw");

                // Check hidden_state output — should be rank-2 [batch*seqLen, hidden]
                INDArray hiddenState = result.get("hidden_state");
                assertNotNull(hiddenState,
                        "Step " + step + ": hidden_state output is null");
                assertEquals(2, hiddenState.rank(),
                        "Step " + step + ": hidden_state rank must be 2 (got " +
                                hiddenState.rank() + ", shape=" +
                                Arrays.toString(hiddenState.shape()) + ")");

                // Check logits output — should be rank-3 [batch, seqLen, vocabSize]
                INDArray logits = result.get("logits");
                assertNotNull(logits,
                        "Step " + step + ": logits output is null");
                assertEquals(3, logits.rank(),
                        "Step " + step + ": logits rank must be 3 (got " +
                                logits.rank() + ", shape=" +
                                Arrays.toString(logits.shape()) + ")");
                assertEquals(BATCH, logits.size(0));
                assertEquals(SEQ_LEN, logits.size(1));
                assertEquals(VOCAB_SIZE, logits.size(2));

                assertFalse(logits.isNaN().any(),
                        "Step " + step + ": logits contains NaN");
                assertFalse(hiddenState.isNaN().any(),
                        "Step " + step + ": hidden_state contains NaN");

                log.info("[MULTI-OUT] mode={} step={} hidden_state.rank={} logits.rank={} logits.shape={}",
                        mode, step, hiddenState.rank(), logits.rank(),
                        Arrays.toString(logits.shape()));
            }
        } finally {
            try { sd.close(); } catch (Exception ignored) {}
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════════
    // Test 4: Multi-output after markVariable — the exact staging server crash
    //         scenario where logitsOutputIdx pointed to rank-0 after invalidation
    // ═══════════════════════════════════════════════════════════════════════════════

    @ParameterizedTest(name = "multi_output_after_markVariable | {0}")
    @MethodSource("modeProvider")
    @DisplayName("Multi-output ranks preserved after markVariable invalidation")
    void testMultiOutputAfterMarkVariable(ExecMode mode) {
        java.util.Random rng = new java.util.Random(GRAPH_SEED);
        SameDiff sd = buildMultiOutputGraph(rng);
        applyMode(sd, mode);

        try {
            // Phase 1: Warmup
            for (int step = 0; step < 3; step++) {
                INDArray input = createDeterministicInput3D(step);
                Map<String, INDArray> ph = Collections.singletonMap("x", input);
                Map<String, INDArray> result = sd.output(ph, "hidden_state", "logits");
                assertNotNull(result.get("logits"),
                        "Warmup step " + step + " logits null");
                assertNotNull(result.get("hidden_state"),
                        "Warmup step " + step + " hidden_state null");
            }

            // Phase 2: markVariable
            DspHandle h = sd.dsp();
            if (h.isCompiled()) {
                int extIdx = h.extInputIndex("x");
                if (extIdx >= 0) {
                    h.markVariable(extIdx);
                    log.info("[MULTI-MARK] Marked 'x' at ext[{}] as variable", extIdx);
                }
            }

            // Phase 3: Post-invalidation — BOTH outputs must maintain rank
            for (int step = 0; step < WARMUP_RUNS + 2; step++) {
                INDArray input = createDeterministicInput3D(step + 200);
                Map<String, INDArray> ph = Collections.singletonMap("x", input);

                final int s = step;
                Map<String, INDArray> result = assertDoesNotThrow(
                        () -> sd.output(ph, "hidden_state", "logits"),
                        "Post-mark step " + s + " must not throw");

                INDArray logits = result.get("logits");
                assertNotNull(logits, "Post-mark step " + step + ": logits is null");

                assertTrue(logits.rank() >= 2,
                        "Post-mark step " + step + " (" + mode + "): logits rank is " +
                                logits.rank() + " (expected >= 2). " +
                                "This reproduces the rank-0 staging server crash!");

                assertEquals(3, logits.rank(),
                        "Post-mark step " + step + ": logits rank must be 3 (got " +
                                logits.rank() + ", shape=" +
                                Arrays.toString(logits.shape()) + ")");

                INDArray hiddenState = result.get("hidden_state");
                assertNotNull(hiddenState, "Post-mark step " + step + ": hidden_state is null");
                assertEquals(2, hiddenState.rank(),
                        "Post-mark step " + step + ": hidden_state rank must be 2 (got " +
                                hiddenState.rank() + ")");

                log.info("[MULTI-MARK] mode={} post-mark step={} logits.rank={} hidden.rank={}",
                        mode, step, logits.rank(), hiddenState.rank());
            }
        } finally {
            try { sd.close(); } catch (Exception ignored) {}
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════════
    // Test 5: Simulate decode loop — alternating prefill (rank-3) and decode (rank-2)
    //         output shapes through the same plan, checking rank at every step
    // ═══════════════════════════════════════════════════════════════════════════════

    @Test
    @DisplayName("Decode-loop shape transition: rank-3 prefill then rank-2 decode steps")
    void testDecodeLoopShapeTransition() {
        java.util.Random rng = new java.util.Random(GRAPH_SEED);
        SameDiff sd = SameDiff.create();

        // Graph with dynamic seqLen: input [batch, -1, hidden] → output [batch, -1, vocab]
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, BATCH, -1, HIDDEN);
        SDVariable flat = sd.reshape("flat", x, -1, HIDDEN);

        SDVariable w1 = sd.var("w1", detArray(rng, HIDDEN, HIDDEN, 0.1f, 0.0f));
        SDVariable h = sd.mmul("mm1", flat, w1);
        h = sd.nn().relu("relu1", h, 0);

        SDVariable wProj = sd.var("w_proj", detArray(rng, HIDDEN, VOCAB_SIZE, 0.1f, 0.0f));
        SDVariable logitsFlat = sd.mmul("mm_proj", h, wProj);

        // Dynamic reshape back — use shape of input to derive seqLen
        SDVariable xShape = sd.shape("x_shape", x);
        SDVariable batchDim = sd.gather("batch_dim", xShape, sd.constant("idx0", Nd4j.scalar(DataType.INT32, 0)), 0);
        SDVariable seqDim = sd.gather("seq_dim", xShape, sd.constant("idx1", Nd4j.scalar(DataType.INT32, 1)), 0);
        SDVariable vocabConst = sd.constant("vocab_const", Nd4j.scalar(DataType.INT64, VOCAB_SIZE));
        SDVariable outShape = sd.stack("out_shape", 0, batchDim, seqDim, vocabConst);
        SDVariable logits = sd.reshape("out", logitsFlat, outShape);

        sd.setOutputs("out");

        // Enable DSP
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);
        sd.setGraphExecutionMode(GraphExecutionMode.AUTO);

        try {
            // Prefill: seqLen = 4
            log.info("[DECODE-LOOP] Phase 1: Prefill (seqLen={})", SEQ_LEN);
            for (int step = 0; step < 3; step++) {
                INDArray input = createDeterministicInput3D(step);
                Map<String, INDArray> ph = Collections.singletonMap("x", input);
                Map<String, INDArray> result = sd.output(ph, "out");
                INDArray out = result.get("out");
                assertNotNull(out, "Prefill step " + step + ": output null");
                assertEquals(3, out.rank(),
                        "Prefill step " + step + ": output rank must be 3 (got " + out.rank() + ")");
                assertEquals(SEQ_LEN, out.size(1),
                        "Prefill step " + step + ": seqLen dim mismatch");
                assertEquals(VOCAB_SIZE, out.size(2),
                        "Prefill step " + step + ": vocab dim mismatch");
                log.info("[DECODE-LOOP] prefill step={} rank={} shape={}",
                        step, out.rank(), Arrays.toString(out.shape()));
            }

            // Decode: seqLen = 1 (single token at a time)
            log.info("[DECODE-LOOP] Phase 2: Decode (seqLen=1)");
            for (int step = 0; step < WARMUP_RUNS + 2; step++) {
                java.util.Random stepRng = new java.util.Random(42L + (step + 50) * 1000);
                float[] data = new float[BATCH * 1 * HIDDEN];
                for (int i = 0; i < data.length; i++) {
                    data[i] = (float) stepRng.nextGaussian();
                }
                INDArray input = Nd4j.createFromArray(data).reshape(BATCH, 1, HIDDEN);
                Map<String, INDArray> ph = Collections.singletonMap("x", input);

                final int s = step;
                Map<String, INDArray> result = assertDoesNotThrow(
                        () -> sd.output(ph, "out"),
                        "Decode step " + s + ": sd.output() must not throw");

                INDArray out = result.get("out");
                assertNotNull(out, "Decode step " + step + ": output is null");

                assertTrue(out.rank() >= 2,
                        "Decode step " + step + ": output rank is " + out.rank() +
                                " (expected >= 2, shape=" + Arrays.toString(out.shape()) +
                                "). This is the rank-0 bug!");

                assertEquals(3, out.rank(),
                        "Decode step " + step + ": output rank must be 3 (got " + out.rank() +
                                ", shape=" + Arrays.toString(out.shape()) + ")");
                assertEquals(1, out.size(1),
                        "Decode step " + step + ": seqLen should be 1");
                assertEquals(VOCAB_SIZE, out.size(2),
                        "Decode step " + step + ": vocab dim mismatch");

                log.info("[DECODE-LOOP] decode step={} rank={} shape={}",
                        step, out.rank(), Arrays.toString(out.shape()));
            }
        } finally {
            try { sd.close(); } catch (Exception ignored) {}
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════════
    // Test 6: Output freshness with rank-3 — outputs must change when inputs change
    //         Even if rank is correct, stale data from prior execution is a bug
    // ═══════════════════════════════════════════════════════════════════════════════

    @ParameterizedTest(name = "rank3_output_freshness | {0}")
    @MethodSource("modeProvider")
    @DisplayName("Rank-3 logits output values change with different inputs (not stale)")
    void testRank3OutputFreshness(ExecMode mode) {
        java.util.Random rng = new java.util.Random(GRAPH_SEED);
        SameDiff sd = buildRank3LogitsGraph(rng);
        applyMode(sd, mode);

        int steps = WARMUP_RUNS + 4;
        List<INDArray> outputs = new ArrayList<>(steps);

        try {
            for (int step = 0; step < steps; step++) {
                INDArray input = createDeterministicInput3D(step);
                Map<String, INDArray> ph = Collections.singletonMap("x", input);
                Map<String, INDArray> result = sd.output(ph, "out");

                INDArray out = result.get("out");
                assertNotNull(out, "Step " + step + ": output null");
                assertEquals(3, out.rank(), "Step " + step + ": rank should be 3");

                outputs.add(out.dup());
            }

            // Verify outputs differ when inputs differ
            int staleCount = 0;
            for (int i = 1; i < outputs.size(); i++) {
                double maxDiff = outputs.get(i).sub(outputs.get(i - 1))
                        .amaxNumber().doubleValue();
                if (maxDiff < 1e-5) {
                    staleCount++;
                    log.error("[FRESHNESS] mode={} step {} same as step {}: maxDiff={}",
                            mode, i, i - 1, maxDiff);
                }
            }

            assertEquals(0, staleCount,
                    mode + ": " + staleCount + "/" + (outputs.size() - 1) +
                            " consecutive output pairs are identical — stale data!");

            log.info("[FRESHNESS] mode={} all {} outputs distinct", mode, steps);
        } finally {
            try { sd.close(); } catch (Exception ignored) {}
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════════
    // Test 7: DspHandle.replay() with rank-3 output — exercise the replay path
    //         that the native decode loop uses
    // ═══════════════════════════════════════════════════════════════════════════════

    @Test
    @DisplayName("DspHandle.replay() preserves rank-3 logits output")
    void testDspHandleReplayRank3() {
        java.util.Random rng = new java.util.Random(GRAPH_SEED);
        SameDiff sd = buildRank3LogitsGraph(rng);
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        try {
            // Initial compilation via sd.output()
            INDArray input = createDeterministicInput3D(0);
            Map<String, INDArray> ph = new LinkedHashMap<>();
            ph.put("x", input);
            Map<String, INDArray> result = sd.output(ph, "out");
            assertNotNull(result.get("out"), "Initial output null");
            assertEquals(3, result.get("out").rank(), "Initial output rank must be 3");

            DspHandle h = sd.dsp();
            if (!h.isCompiled()) {
                log.warn("Plan not compiled — skipping replay test");
                return;
            }

            // Mark placeholder as variable (simulates what native decode does)
            int extIdx = h.extInputIndex("x");
            if (extIdx >= 0) {
                h.markVariable(extIdx);
            }

            // Replay with different inputs
            for (int step = 0; step < 8; step++) {
                INDArray stepInput = createDeterministicInput3D(step + 50);
                ph.put("x", stepInput);

                Map<String, INDArray> replayResult = h.replay(ph);
                INDArray out = replayResult.get("out");
                assertNotNull(out, "Replay step " + step + ": output null");

                assertTrue(out.rank() >= 2,
                        "Replay step " + step + ": output rank is " + out.rank() +
                                " (expected >= 2). Rank-0 bug in replay path!");

                assertEquals(3, out.rank(),
                        "Replay step " + step + ": output rank must be 3 (got " +
                                out.rank() + ", shape=" + Arrays.toString(out.shape()) + ")");

                log.info("[REPLAY] step={} rank={} shape={}", step, out.rank(),
                        Arrays.toString(out.shape()));
            }
        } finally {
            try { sd.close(); } catch (Exception ignored) {}
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════════
    // Test 8: Rapid re-compilation — multiple markVariable cycles
    //         Simulates the lfm2.5-1.2b staging scenario where each layer
    //         triggers markExternalInputVariable invalidation
    // ═══════════════════════════════════════════════════════════════════════════════

    @Test
    @DisplayName("Rank-3 output survives multiple markVariable invalidation cycles")
    void testMultipleMarkVariableCycles() {
        java.util.Random rng = new java.util.Random(GRAPH_SEED);
        SameDiff sd = buildRank3LogitsGraph(rng);
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);
        sd.setGraphExecutionMode(GraphExecutionMode.AUTO);

        try {
            int numCycles = 3;
            for (int cycle = 0; cycle < numCycles; cycle++) {
                log.info("[MULTI-CYCLE] === Cycle {} ===", cycle);

                // Warmup
                for (int step = 0; step < 3; step++) {
                    INDArray input = createDeterministicInput3D(cycle * 100 + step);
                    Map<String, INDArray> ph = Collections.singletonMap("x", input);
                    Map<String, INDArray> result = sd.output(ph, "out");
                    INDArray out = result.get("out");
                    assertNotNull(out, "Cycle " + cycle + " warmup " + step + ": null");
                    assertEquals(3, out.rank(),
                            "Cycle " + cycle + " warmup " + step + ": rank " + out.rank());
                }

                // Mark variable — triggers invalidation
                DspHandle h = sd.dsp();
                if (h.isCompiled()) {
                    int extIdx = h.extInputIndex("x");
                    if (extIdx >= 0) {
                        h.markVariable(extIdx);
                        DspHandle.StepSnapshot snap = h.captureStepSnapshot();
                        log.info("[MULTI-CYCLE] cycle={} post-mark execCount={} phase={}",
                                cycle, snap.executeCount, snap.planPhase);
                    }
                }

                // Post-invalidation execution
                for (int step = 0; step < WARMUP_RUNS; step++) {
                    INDArray input = createDeterministicInput3D(cycle * 100 + step + 50);
                    Map<String, INDArray> ph = Collections.singletonMap("x", input);

                    final int c = cycle, s = step;
                    Map<String, INDArray> result = assertDoesNotThrow(
                            () -> sd.output(ph, "out"),
                            "Cycle " + c + " post-mark step " + s + ": must not throw");

                    INDArray out = result.get("out");
                    assertNotNull(out,
                            "Cycle " + cycle + " post-mark step " + step + ": null");

                    assertTrue(out.rank() >= 2,
                            "Cycle " + cycle + " post-mark step " + step +
                                    ": rank-0 bug! rank=" + out.rank());

                    assertEquals(3, out.rank(),
                            "Cycle " + cycle + " post-mark step " + step +
                                    ": rank must be 3 (got " + out.rank() + ")");

                    log.info("[MULTI-CYCLE] cycle={} post-mark step={} rank={} shape={}",
                            cycle, step, out.rank(), Arrays.toString(out.shape()));
                }
            }
        } finally {
            try { sd.close(); } catch (Exception ignored) {}
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════════
    // Test 10: Sequential markVariable cycles — markVariable invalidation followed
    //          by re-execution must not degrade output rank.
    // ═══════════════════════════════════════════════════════════════════════════════

    @Test
    @DisplayName("markVariable + re-execution — rank must not degrade")
    void testMarkVariableCyclesSequential() throws Exception {
        java.util.Random rng = new java.util.Random(GRAPH_SEED);
        SameDiff sd = buildRank3LogitsGraph(rng);
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);
        sd.setGraphExecutionMode(GraphExecutionMode.AUTO);

        try {
            // Warmup
            for (int i = 0; i < 3; i++) {
                INDArray input = createDeterministicInput3D(i);
                Map<String, INDArray> ph = Collections.singletonMap("x", input);
                Map<String, INDArray> result = sd.output(ph, "out");
                assertNotNull(result.get("out"), "Warmup step " + i + " null");
                assertEquals(3, result.get("out").rank(), "Warmup step " + i + " rank");
            }

            // markVariable cycles: invalidate then re-execute
            int numCycles = 5;
            int stepsPerCycle = 5;
            int rankErrors = 0;

            for (int cycle = 0; cycle < numCycles; cycle++) {
                // Mark variable to invalidate the plan
                DspHandle h = sd.dsp();
                if (h.isCompiled()) {
                    int extIdx = h.extInputIndex("x");
                    if (extIdx >= 0) {
                        h.markVariable(extIdx);
                        log.info("[MARK-CYCLE] cycle={}: markVariable fired", cycle);
                    }
                }

                // Re-execute after invalidation
                for (int step = 0; step < stepsPerCycle; step++) {
                    INDArray input = createDeterministicInput3D(cycle * 1000 + step);
                    Map<String, INDArray> ph = Collections.singletonMap("x", input);
                    Map<String, INDArray> result = sd.output(ph, "out");
                    INDArray out = result.get("out");

                    assertNotNull(out, "cycle=" + cycle + " step=" + step + ": output null");
                    if (out.rank() != 3) {
                        rankErrors++;
                        log.error("[MARK-CYCLE] cycle={} step={}: wrong rank={} shape={}",
                                cycle, step, out.rank(), Arrays.toString(out.shape()));
                    }
                }
            }

            assertEquals(0, rankErrors,
                    "Output rank degraded after markVariable invalidation! " +
                            rankErrors + " rank errors across " + (numCycles * stepsPerCycle) + " steps.");
        } finally {
            try { sd.close(); } catch (Exception ignored) {}
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════════
    // Transformer-like graph builders for realistic concurrent tests
    // ═══════════════════════════════════════════════════════════════════════════════

    private static final int NUM_HEADS = 4;
    private static final int HEAD_DIM = HIDDEN / NUM_HEADS;  // 4

    /**
     * Builds a transformer-attention-like graph:
     *   x [batch, seqLen, hidden]
     *   → RMSNorm → Q/K/V projection (3 matmuls)
     *   → reshape to [batch, seqLen, numHeads, headDim]
     *   → transpose to [batch, numHeads, seqLen, headDim]
     *   → Q*K^T / sqrt(headDim) → softmax → * V → concat heads
     *   → output projection → add residual → [batch, seqLen, hidden]
     *
     * This exercises the op patterns an LFM model uses:
     *   matmul, reshape, permute, softmax, elementwise add/mul
     */
    private static SameDiff buildAttentionGraph(java.util.Random rng) {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, BATCH, SEQ_LEN, HIDDEN);

        // Flatten for matmul: [batch*seqLen, hidden]
        SDVariable flat = sd.reshape("flat", x, BATCH * SEQ_LEN, HIDDEN);

        // RMSNorm approximation: x * rsqrt(mean(x^2) + eps)
        SDVariable xSq = sd.math().square("x_sq", flat);
        SDVariable meanSq = sd.mean("mean_sq", xSq, true, 1);
        SDVariable eps = sd.constant("eps", Nd4j.scalar(DataType.FLOAT, 1e-6f));
        SDVariable rmsInv = sd.math().rsqrt("rms_inv", meanSq.add("mean_sq_eps", eps));
        SDVariable normed = flat.mul("normed", rmsInv);

        // Q, K, V projections: [batch*seqLen, hidden] → [batch*seqLen, hidden]
        SDVariable wQ = sd.var("wQ", detArray(rng, HIDDEN, HIDDEN, 0.1f, 0.0f));
        SDVariable wK = sd.var("wK", detArray(rng, HIDDEN, HIDDEN, 0.1f, 0.0f));
        SDVariable wV = sd.var("wV", detArray(rng, HIDDEN, HIDDEN, 0.1f, 0.0f));
        SDVariable q = sd.mmul("q_proj", normed, wQ);
        SDVariable k = sd.mmul("k_proj", normed, wK);
        SDVariable v = sd.mmul("v_proj", normed, wV);

        // Reshape to multi-head: [batch, seqLen, numHeads, headDim]
        SDVariable qHeads = sd.reshape("q_heads", q, BATCH, SEQ_LEN, NUM_HEADS, HEAD_DIM);
        SDVariable kHeads = sd.reshape("k_heads", k, BATCH, SEQ_LEN, NUM_HEADS, HEAD_DIM);
        SDVariable vHeads = sd.reshape("v_heads", v, BATCH, SEQ_LEN, NUM_HEADS, HEAD_DIM);

        // Transpose to [batch, numHeads, seqLen, headDim]
        SDVariable qT = sd.permute("q_t", qHeads, 0, 2, 1, 3);
        SDVariable kT = sd.permute("k_t", kHeads, 0, 2, 1, 3);
        SDVariable vT = sd.permute("v_t", vHeads, 0, 2, 1, 3);

        // Attention scores: Q * K^T → [batch, numHeads, seqLen, seqLen]
        SDVariable kTrans = sd.permute("k_trans", kT, 0, 1, 3, 2);
        SDVariable scores = sd.mmul("attn_scores", qT, kTrans);
        SDVariable scale = sd.constant("scale", Nd4j.scalar(DataType.FLOAT, 1.0f / (float) Math.sqrt(HEAD_DIM)));
        SDVariable scaledScores = scores.mul("scaled_scores", scale);

        // Softmax on last axis
        SDVariable attnWeights = sd.nn().softmax("attn_weights", scaledScores, -1);

        // Weighted values: [batch, numHeads, seqLen, headDim]
        SDVariable attnOut = sd.mmul("attn_out", attnWeights, vT);

        // Concat heads: transpose back and reshape → [batch*seqLen, hidden]
        SDVariable attnBack = sd.permute("attn_back", attnOut, 0, 2, 1, 3);
        SDVariable attnFlat = sd.reshape("attn_flat", attnBack, BATCH * SEQ_LEN, HIDDEN);

        // Output projection
        SDVariable wO = sd.var("wO", detArray(rng, HIDDEN, HIDDEN, 0.1f, 0.0f));
        SDVariable projected = sd.mmul("out_proj", attnFlat, wO);

        // Residual connection + reshape to 3D
        SDVariable residual = projected.add("residual", flat);
        SDVariable out3d = sd.reshape("out", residual, BATCH, SEQ_LEN, HIDDEN);

        sd.setOutputs("out");
        return sd;
    }

    /**
     * Builds a two-layer transformer-like graph (attention + FFN) producing
     * rank-3 logits [batch, seqLen, vocabSize].
     *
     * Layer 1: attention block (matmul Q/K/V, softmax, concat heads)
     * Layer 2: FFN (up-project, SiLU gate, down-project)
     * Final:   logits projection to vocab
     *
     * This is closer to a real LFM decode step than the simple matmul+relu graph.
     */
    private static SameDiff buildTransformerLogitsGraph(java.util.Random rng) {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, BATCH, SEQ_LEN, HIDDEN);

        // ── Layer 1: simplified attention ──
        SDVariable flat = sd.reshape("flat", x, BATCH * SEQ_LEN, HIDDEN);

        SDVariable wQ = sd.var("wQ", detArray(rng, HIDDEN, HIDDEN, 0.1f, 0.0f));
        SDVariable wK = sd.var("wK", detArray(rng, HIDDEN, HIDDEN, 0.1f, 0.0f));
        SDVariable wV = sd.var("wV", detArray(rng, HIDDEN, HIDDEN, 0.1f, 0.0f));
        SDVariable q = sd.mmul("q", flat, wQ);
        SDVariable k = sd.mmul("k", flat, wK);
        SDVariable v = sd.mmul("v", flat, wV);

        // Simplified attention: softmax(Q*K^T / sqrt(d)) * V
        SDVariable k3 = sd.reshape("k3", k, BATCH, SEQ_LEN, HIDDEN);
        SDVariable kTrans = sd.permute("k_trans", k3,
                0, 2, 1);  // [batch, hidden, seqLen]
        SDVariable q3 = sd.reshape("q3", q, BATCH, SEQ_LEN, HIDDEN);
        SDVariable qk = sd.mmul("qk", q3, kTrans);  // [batch, seqLen, seqLen]
        SDVariable scale = sd.constant("scale", Nd4j.scalar(DataType.FLOAT, 1.0f / (float) Math.sqrt(HIDDEN)));
        SDVariable scores = qk.mul("scaled", scale);
        SDVariable attn = sd.nn().softmax("attn", scores, -1);
        SDVariable v3 = sd.reshape("v3", v, BATCH, SEQ_LEN, HIDDEN);
        SDVariable context = sd.mmul("context", attn, v3);  // [batch, seqLen, hidden]
        SDVariable contextFlat = sd.reshape("ctx_flat", context, BATCH * SEQ_LEN, HIDDEN);

        // Output projection + residual
        SDVariable wO = sd.var("wO", detArray(rng, HIDDEN, HIDDEN, 0.1f, 0.0f));
        SDVariable attnOut = sd.mmul("attn_out", contextFlat, wO);
        SDVariable h = attnOut.add("res1", flat);

        // ── Layer 2: FFN with SiLU gate ──
        int ffnHidden = HIDDEN * 2;
        SDVariable wUp = sd.var("wUp", detArray(rng, HIDDEN, ffnHidden, 0.1f, 0.0f));
        SDVariable wGate = sd.var("wGate", detArray(rng, HIDDEN, ffnHidden, 0.1f, 0.0f));
        SDVariable wDown = sd.var("wDown", detArray(rng, ffnHidden, HIDDEN, 0.1f, 0.0f));

        SDVariable up = sd.mmul("ffn_up", h, wUp);
        SDVariable gate = sd.mmul("ffn_gate", h, wGate);
        SDVariable gateAct = sd.nn().sigmoid("silu_sig", gate).mul("silu", gate);
        SDVariable ffn = sd.mmul("ffn_down", up.mul("gated", gateAct), wDown);
        SDVariable h2 = ffn.add("res2", h);

        // ── Final: logits projection ──
        SDVariable wLogits = sd.var("wLogits", detArray(rng, HIDDEN, VOCAB_SIZE, 0.1f, 0.0f));
        SDVariable logitsFlat = sd.mmul("logits_flat", h2, wLogits);
        SDVariable logits = sd.reshape("out", logitsFlat, BATCH, SEQ_LEN, VOCAB_SIZE);

        sd.setOutputs("out");
        return sd;
    }

    // ═══════════════════════════════════════════════════════════════════════════════
    // Test 11: Concurrent sd.output() — multiple threads calling sd.output()
    //          on the same SameDiff instance simultaneously.
    //          SameDiff has per-thread sessions and plans, so this should work.
    //          This reproduces the LFM staging crash where concurrent inference
    //          caused double-free / corruption in the native plan execution.
    // ═══════════════════════════════════════════════════════════════════════════════

    @ParameterizedTest(name = "concurrent_output_rank3 | {0}")
    @MethodSource("modeProvider")
    @DisplayName("Concurrent sd.output() on same SameDiff — rank-3 preserved across threads")
    void testConcurrentOutputRank3(ExecMode mode) throws Exception {
        java.util.Random rng = new java.util.Random(GRAPH_SEED);
        SameDiff sd = buildRank3LogitsGraph(rng);
        applyMode(sd, mode);

        // Pre-warm on main thread (see testConcurrentAttentionGraph for rationale)
        for (int i = 0; i < WARMUP_RUNS; i++) {
            INDArray input = createDeterministicInput3D(i);
            Map<String, INDArray> ph = Collections.singletonMap("x", input);
            Map<String, INDArray> result = sd.output(ph, "out");
            assertNotNull(result.get("out"), "Pre-warm step " + i + " null");
            assertEquals(3, result.get("out").rank(), "Pre-warm step " + i + " rank");
        }

        int numThreads = 2;
        int stepsPerThread = WARMUP_RUNS + 4;
        ExecutorService executor = Executors.newFixedThreadPool(numThreads);
        AtomicInteger rankErrors = new AtomicInteger(0);
        AtomicInteger nullErrors = new AtomicInteger(0);
        AtomicInteger throwErrors = new AtomicInteger(0);
        CountDownLatch startLatch = new CountDownLatch(1);
        CountDownLatch doneLatch = new CountDownLatch(numThreads);

        try {
            for (int t = 0; t < numThreads; t++) {
                final int threadId = t;
                executor.submit(() -> {
                    try {
                        startLatch.await();
                        for (int step = 0; step < stepsPerThread; step++) {
                            INDArray input = createDeterministicInput3D(threadId * 10000 + step);
                            Map<String, INDArray> ph = Collections.singletonMap("x", input);
                            try {
                                Map<String, INDArray> result = sd.output(ph, "out");
                                INDArray out = result.get("out");
                                if (out == null) {
                                    nullErrors.incrementAndGet();
                                    log.error("[CONCURRENT] thread={} step={}: output is NULL", threadId, step);
                                } else if (out.rank() != 3) {
                                    rankErrors.incrementAndGet();
                                    log.error("[CONCURRENT] thread={} step={}: wrong rank={} shape={}",
                                            threadId, step, out.rank(), Arrays.toString(out.shape()));
                                } else {
                                    log.info("[CONCURRENT] thread={} step={} mode={} rank={} shape={}",
                                            threadId, step, mode, out.rank(), Arrays.toString(out.shape()));
                                }
                            } catch (Exception e) {
                                throwErrors.incrementAndGet();
                                log.error("[CONCURRENT] thread={} step={}: EXCEPTION: {}",
                                        threadId, step, e.getMessage(), e);
                            }
                        }
                    } catch (InterruptedException ie) {
                        Thread.currentThread().interrupt();
                    } finally {
                        doneLatch.countDown();
                    }
                });
            }

            // Release all threads at once
            startLatch.countDown();
            assertTrue(doneLatch.await(120, TimeUnit.SECONDS),
                    "Concurrent test timed out after 120s");

            assertEquals(0, throwErrors.get(),
                    mode + ": " + throwErrors.get() + " threads threw exceptions (crash/double-free)");
            assertEquals(0, nullErrors.get(),
                    mode + ": " + nullErrors.get() + " null outputs across threads");
            assertEquals(0, rankErrors.get(),
                    mode + ": " + rankErrors.get() + " rank errors across " +
                            (numThreads * stepsPerThread) + " total steps");

            log.info("[CONCURRENT] mode={} PASSED: {} threads x {} steps = {} total, 0 errors",
                    mode, numThreads, stepsPerThread, numThreads * stepsPerThread);
        } finally {
            executor.shutdownNow();
            try { sd.close(); } catch (Exception ignored) {}
        }
    }

    @Test
    @DisplayName("Concurrent sd.output() on same multi-output graph")
    void testConcurrentMultiOutput() throws Exception {
        java.util.Random rng = new java.util.Random(GRAPH_SEED);
        SameDiff sd = buildMultiOutputGraph(rng);
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);
        sd.setGraphExecutionMode(GraphExecutionMode.AUTO);

        // Pre-warm on main thread (see testConcurrentAttentionGraph for rationale)
        for (int i = 0; i < WARMUP_RUNS; i++) {
            INDArray input = createDeterministicInput3D(i);
            Map<String, INDArray> ph = Collections.singletonMap("x", input);
            Map<String, INDArray> result = sd.output(ph, "hidden_state", "logits");
            assertNotNull(result.get("logits"), "Pre-warm step " + i + " logits null");
            assertNotNull(result.get("hidden_state"), "Pre-warm step " + i + " hidden_state null");
        }

        int numThreads = 4;
        int stepsPerThread = WARMUP_RUNS + 4;
        ExecutorService executor = Executors.newFixedThreadPool(numThreads);
        AtomicInteger errors = new AtomicInteger(0);
        CountDownLatch startLatch = new CountDownLatch(1);
        CountDownLatch doneLatch = new CountDownLatch(numThreads);

        try {
            for (int t = 0; t < numThreads; t++) {
                final int threadId = t;
                executor.submit(() -> {
                    try {
                        startLatch.await();
                        for (int step = 0; step < stepsPerThread; step++) {
                            INDArray input = createDeterministicInput3D(threadId * 10000 + step);
                            Map<String, INDArray> ph = Collections.singletonMap("x", input);
                            try {
                                Map<String, INDArray> result = sd.output(ph, "hidden_state", "logits");
                                INDArray logits = result.get("logits");
                                INDArray hiddenState = result.get("hidden_state");

                                if (logits == null || hiddenState == null) {
                                    errors.incrementAndGet();
                                    log.error("[CONCURRENT-MULTI] thread={} step={}: null output " +
                                            "(logits={} hidden={})", threadId, step,
                                            logits != null, hiddenState != null);
                                } else if (logits.rank() != 3 || hiddenState.rank() != 2) {
                                    errors.incrementAndGet();
                                    log.error("[CONCURRENT-MULTI] thread={} step={}: wrong ranks " +
                                            "logits.rank={} hidden.rank={}", threadId, step,
                                            logits.rank(), hiddenState.rank());
                                }
                            } catch (Exception e) {
                                errors.incrementAndGet();
                                log.error("[CONCURRENT-MULTI] thread={} step={}: EXCEPTION: {}",
                                        threadId, step, e.getMessage(), e);
                            }
                        }
                    } catch (InterruptedException ie) {
                        Thread.currentThread().interrupt();
                    } finally {
                        doneLatch.countDown();
                    }
                });
            }

            startLatch.countDown();
            assertTrue(doneLatch.await(120, TimeUnit.SECONDS),
                    "Concurrent multi-output test timed out after 120s");

            assertEquals(0, errors.get(),
                    errors.get() + " errors across " + (numThreads * stepsPerThread) +
                            " concurrent multi-output steps");
        } finally {
            executor.shutdownNow();
            try { sd.close(); } catch (Exception ignored) {}
        }
    }

    @Test
    @DisplayName("Concurrent sd.output() with rank-2 graph — stress per-thread plan isolation")
    void testConcurrentOutputRank2() throws Exception {
        java.util.Random rng = new java.util.Random(GRAPH_SEED);
        SameDiff sd = buildRank2LogitsGraph(rng);
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);
        sd.setGraphExecutionMode(GraphExecutionMode.AUTO);

        // Pre-warm on main thread (see testConcurrentAttentionGraph for rationale)
        for (int i = 0; i < WARMUP_RUNS; i++) {
            INDArray input = createDeterministicInput2D(i);
            Map<String, INDArray> ph = Collections.singletonMap("x", input);
            Map<String, INDArray> result = sd.output(ph, "out");
            assertNotNull(result.get("out"), "Pre-warm step " + i + " null");
            assertEquals(2, result.get("out").rank(), "Pre-warm step " + i + " rank");
        }

        int numThreads = 4;
        int stepsPerThread = WARMUP_RUNS + 4;
        ExecutorService executor = Executors.newFixedThreadPool(numThreads);
        AtomicInteger errors = new AtomicInteger(0);
        CountDownLatch startLatch = new CountDownLatch(1);
        CountDownLatch doneLatch = new CountDownLatch(numThreads);

        try {
            for (int t = 0; t < numThreads; t++) {
                final int threadId = t;
                executor.submit(() -> {
                    try {
                        startLatch.await();
                        for (int step = 0; step < stepsPerThread; step++) {
                            INDArray input = createDeterministicInput2D(threadId * 10000 + step);
                            Map<String, INDArray> ph = Collections.singletonMap("x", input);
                            try {
                                Map<String, INDArray> result = sd.output(ph, "out");
                                INDArray out = result.get("out");
                                if (out == null) {
                                    errors.incrementAndGet();
                                    log.error("[CONCURRENT-R2] thread={} step={}: null output",
                                            threadId, step);
                                } else if (out.rank() != 2) {
                                    errors.incrementAndGet();
                                    log.error("[CONCURRENT-R2] thread={} step={}: wrong rank={} shape={}",
                                            threadId, step, out.rank(), Arrays.toString(out.shape()));
                                }
                            } catch (Exception e) {
                                errors.incrementAndGet();
                                log.error("[CONCURRENT-R2] thread={} step={}: EXCEPTION: {}",
                                        threadId, step, e.getMessage(), e);
                            }
                        }
                    } catch (InterruptedException ie) {
                        Thread.currentThread().interrupt();
                    } finally {
                        doneLatch.countDown();
                    }
                });
            }

            startLatch.countDown();
            assertTrue(doneLatch.await(120, TimeUnit.SECONDS),
                    "Concurrent rank-2 test timed out after 120s");

            assertEquals(0, errors.get(),
                    errors.get() + " errors across " + (numThreads * stepsPerThread) +
                            " concurrent rank-2 steps");
        } finally {
            executor.shutdownNow();
            try { sd.close(); } catch (Exception ignored) {}
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════════
    // Test 12: Concurrent sd.output() with attention graph — 8 threads
    //          Uses buildAttentionGraph which exercises:
    //          RMSNorm, Q/K/V matmul, reshape 2D→4D, permute, softmax,
    //          attention matmul, concat heads, output projection, residual add.
    //          This is the same op pattern an LFM attention layer uses.
    // ═══════════════════════════════════════════════════════════════════════════════

    @ParameterizedTest(name = "concurrent_attention | {0}")
    @MethodSource("modeProvider")
    @DisplayName("Concurrent sd.output() with attention graph — 8 threads, rank-3 preserved")
    void testConcurrentAttentionGraph(ExecMode mode) throws Exception {
        java.util.Random rng = new java.util.Random(GRAPH_SEED);
        SameDiff sd = buildAttentionGraph(rng);
        applyMode(sd, mode);

        // Pre-warm on main thread: complete the full DSP lifecycle (compile → warmup →
        // Triton compile → CUDA graph capture → replay) BEFORE spawning worker threads.
        // This ensures releaseGpuIntermediates() (which calls cudaDeviceSynchronize) and
        // all other synchronous CUDA operations during compilation complete before any
        // worker thread starts its own capture. Without this, compilation on one thread
        // can poison another thread's CUDA graph capture via synchronous CUDA APIs.
        for (int i = 0; i < WARMUP_RUNS; i++) {
            INDArray input = createDeterministicInput3D(i);
            Map<String, INDArray> ph = Collections.singletonMap("x", input);
            Map<String, INDArray> result = sd.output(ph, "out");
            assertNotNull(result.get("out"), "Pre-warm step " + i + " null");
            assertEquals(3, result.get("out").rank(), "Pre-warm step " + i + " rank");
        }

        int numThreads = 8;
        int stepsPerThread = WARMUP_RUNS + 6;
        ExecutorService executor = Executors.newFixedThreadPool(numThreads);
        AtomicInteger rankErrors = new AtomicInteger(0);
        AtomicInteger nullErrors = new AtomicInteger(0);
        AtomicInteger throwErrors = new AtomicInteger(0);
        CountDownLatch startLatch = new CountDownLatch(1);
        CountDownLatch doneLatch = new CountDownLatch(numThreads);

        try {
            for (int t = 0; t < numThreads; t++) {
                final int threadId = t;
                executor.submit(() -> {
                    try {
                        startLatch.await();
                        for (int step = 0; step < stepsPerThread; step++) {
                            INDArray input = createDeterministicInput3D(threadId * 10000 + step);
                            Map<String, INDArray> ph = Collections.singletonMap("x", input);
                            try {
                                Map<String, INDArray> result = sd.output(ph, "out");
                                INDArray out = result.get("out");
                                if (out == null) {
                                    nullErrors.incrementAndGet();
                                    log.error("[ATTN] thread={} step={} mode={}: output is NULL",
                                            threadId, step, mode);
                                } else if (out.rank() != 3) {
                                    rankErrors.incrementAndGet();
                                    log.error("[ATTN] thread={} step={} mode={}: wrong rank={} shape={}",
                                            threadId, step, mode, out.rank(), Arrays.toString(out.shape()));
                                } else {
                                    assertArrayEquals(new long[]{BATCH, SEQ_LEN, HIDDEN}, out.shape(),
                                            "thread=" + threadId + " step=" + step + " mode=" + mode +
                                                    ": shape mismatch");
                                }
                            } catch (Exception e) {
                                throwErrors.incrementAndGet();
                                log.error("[ATTN] thread={} step={} mode={}: EXCEPTION: {}",
                                        threadId, step, mode, e.getMessage(), e);
                            }
                        }
                    } catch (InterruptedException ie) {
                        Thread.currentThread().interrupt();
                    } finally {
                        doneLatch.countDown();
                    }
                });
            }

            startLatch.countDown();
            assertTrue(doneLatch.await(180, TimeUnit.SECONDS),
                    "Concurrent attention test timed out after 180s");

            assertEquals(0, throwErrors.get(),
                    mode + ": " + throwErrors.get() + " threads threw exceptions");
            assertEquals(0, nullErrors.get(),
                    mode + ": " + nullErrors.get() + " null outputs");
            assertEquals(0, rankErrors.get(),
                    mode + ": " + rankErrors.get() + " rank errors across " +
                            (numThreads * stepsPerThread) + " total steps");

            log.info("[ATTN] mode={} PASSED: {} threads x {} steps = {} total, 0 errors",
                    mode, numThreads, stepsPerThread, numThreads * stepsPerThread);
        } finally {
            executor.shutdownNow();
            try { sd.close(); } catch (Exception ignored) {}
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════════
    // Test 13: Concurrent sd.output() with transformer+logits graph — 8 threads
    //          Uses buildTransformerLogitsGraph which exercises:
    //          attention block (Q/K/V matmul, softmax, context aggregation),
    //          FFN block (up-project, SiLU gate, down-project),
    //          logits projection, residual connections, reshape 2D→3D.
    //          This is the closest pattern to a real LFM decode step.
    // ═══════════════════════════════════════════════════════════════════════════════

    @ParameterizedTest(name = "concurrent_transformer_logits | {0}")
    @MethodSource("modeProvider")
    @DisplayName("Concurrent sd.output() with transformer logits graph — 8 threads, rank-3 preserved")
    void testConcurrentTransformerLogits(ExecMode mode) throws Exception {
        java.util.Random rng = new java.util.Random(GRAPH_SEED);
        SameDiff sd = buildTransformerLogitsGraph(rng);
        applyMode(sd, mode);

        // Pre-warm on main thread (see testConcurrentAttentionGraph for rationale)
        for (int i = 0; i < WARMUP_RUNS; i++) {
            INDArray input = createDeterministicInput3D(i);
            Map<String, INDArray> ph = Collections.singletonMap("x", input);
            Map<String, INDArray> result = sd.output(ph, "out");
            assertNotNull(result.get("out"), "Pre-warm step " + i + " null");
            assertEquals(3, result.get("out").rank(), "Pre-warm step " + i + " rank");
        }

        int numThreads = 8;
        int stepsPerThread = WARMUP_RUNS + 6;
        ExecutorService executor = Executors.newFixedThreadPool(numThreads);
        AtomicInteger rankErrors = new AtomicInteger(0);
        AtomicInteger nullErrors = new AtomicInteger(0);
        AtomicInteger throwErrors = new AtomicInteger(0);
        CountDownLatch startLatch = new CountDownLatch(1);
        CountDownLatch doneLatch = new CountDownLatch(numThreads);

        try {
            for (int t = 0; t < numThreads; t++) {
                final int threadId = t;
                executor.submit(() -> {
                    try {
                        startLatch.await();
                        for (int step = 0; step < stepsPerThread; step++) {
                            INDArray input = createDeterministicInput3D(threadId * 10000 + step);
                            Map<String, INDArray> ph = Collections.singletonMap("x", input);
                            try {
                                Map<String, INDArray> result = sd.output(ph, "out");
                                INDArray out = result.get("out");
                                if (out == null) {
                                    nullErrors.incrementAndGet();
                                    log.error("[XFMR] thread={} step={} mode={}: output is NULL",
                                            threadId, step, mode);
                                } else if (out.rank() != 3) {
                                    rankErrors.incrementAndGet();
                                    log.error("[XFMR] thread={} step={} mode={}: wrong rank={} shape={}",
                                            threadId, step, mode, out.rank(), Arrays.toString(out.shape()));
                                } else {
                                    assertArrayEquals(new long[]{BATCH, SEQ_LEN, VOCAB_SIZE}, out.shape(),
                                            "thread=" + threadId + " step=" + step + " mode=" + mode +
                                                    ": shape mismatch");
                                }
                            } catch (Exception e) {
                                throwErrors.incrementAndGet();
                                log.error("[XFMR] thread={} step={} mode={}: EXCEPTION: {}",
                                        threadId, step, mode, e.getMessage(), e);
                            }
                        }
                    } catch (InterruptedException ie) {
                        Thread.currentThread().interrupt();
                    } finally {
                        doneLatch.countDown();
                    }
                });
            }

            startLatch.countDown();
            assertTrue(doneLatch.await(180, TimeUnit.SECONDS),
                    "Concurrent transformer logits test timed out after 180s");

            assertEquals(0, throwErrors.get(),
                    mode + ": " + throwErrors.get() + " threads threw exceptions");
            assertEquals(0, nullErrors.get(),
                    mode + ": " + nullErrors.get() + " null outputs");
            assertEquals(0, rankErrors.get(),
                    mode + ": " + rankErrors.get() + " rank errors across " +
                            (numThreads * stepsPerThread) + " total steps");

            log.info("[XFMR] mode={} PASSED: {} threads x {} steps = {} total, 0 errors",
                    mode, numThreads, stepsPerThread, numThreads * stepsPerThread);
        } finally {
            executor.shutdownNow();
            try { sd.close(); } catch (Exception ignored) {}
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════════
    // Test 14: Concurrent sd.output() with markVariable invalidation stress
    //          8 threads execute while one thread periodically calls markVariable
    //          to invalidate the plan. This reproduces the scenario where
    //          markExternalInputVariable resets execCount → SHAPE_KEY_MATCHED
    //          bumps execCount to 2 → stale plan output with rank-0.
    // ═══════════════════════════════════════════════════════════════════════════════

    @ParameterizedTest(name = "concurrent_mark_invalidation | {0}")
    @MethodSource("modeProvider")
    @DisplayName("Concurrent sd.output() with periodic markVariable invalidation — rank stability")
    void testConcurrentMarkInvalidation(ExecMode mode) throws Exception {
        java.util.Random rng = new java.util.Random(GRAPH_SEED);
        SameDiff sd = buildTransformerLogitsGraph(rng);
        applyMode(sd, mode);

        // Warmup sequentially first to compile plans
        for (int i = 0; i < WARMUP_RUNS; i++) {
            INDArray input = createDeterministicInput3D(i);
            Map<String, INDArray> ph = Collections.singletonMap("x", input);
            Map<String, INDArray> result = sd.output(ph, "out");
            assertNotNull(result.get("out"), "Warmup step " + i + " null");
            assertEquals(3, result.get("out").rank(), "Warmup step " + i + " rank");
        }

        int numWorkers = 7;
        int stepsPerWorker = 10;
        ExecutorService executor = Executors.newFixedThreadPool(numWorkers + 1);
        AtomicInteger rankErrors = new AtomicInteger(0);
        AtomicInteger nullErrors = new AtomicInteger(0);
        AtomicInteger throwErrors = new AtomicInteger(0);
        AtomicInteger markCount = new AtomicInteger(0);
        CountDownLatch startLatch = new CountDownLatch(1);
        CountDownLatch doneLatch = new CountDownLatch(numWorkers + 1);

        try {
            // Worker threads: continuously call sd.output()
            for (int t = 0; t < numWorkers; t++) {
                final int threadId = t;
                executor.submit(() -> {
                    try {
                        startLatch.await();
                        for (int step = 0; step < stepsPerWorker; step++) {
                            INDArray input = createDeterministicInput3D(threadId * 10000 + step);
                            Map<String, INDArray> ph = Collections.singletonMap("x", input);
                            try {
                                Map<String, INDArray> result = sd.output(ph, "out");
                                INDArray out = result.get("out");
                                if (out == null) {
                                    nullErrors.incrementAndGet();
                                    log.error("[MARK-STRESS] thread={} step={}: output NULL", threadId, step);
                                } else if (out.rank() != 3) {
                                    rankErrors.incrementAndGet();
                                    log.error("[MARK-STRESS] thread={} step={}: rank={} shape={}",
                                            threadId, step, out.rank(), Arrays.toString(out.shape()));
                                }
                            } catch (Exception e) {
                                throwErrors.incrementAndGet();
                                log.error("[MARK-STRESS] thread={} step={}: EXCEPTION: {}",
                                        threadId, step, e.getMessage(), e);
                            }
                        }
                    } catch (InterruptedException ie) {
                        Thread.currentThread().interrupt();
                    } finally {
                        doneLatch.countDown();
                    }
                });
            }

            // Invalidator thread: calls markVariable periodically
            executor.submit(() -> {
                try {
                    startLatch.await();
                    for (int i = 0; i < 5; i++) {
                        Thread.sleep(10);  // Let workers run a few steps
                        DspHandle h = sd.dsp();
                        if (h.isCompiled()) {
                            int extIdx = h.extInputIndex("x");
                            if (extIdx >= 0) {
                                h.markVariable(extIdx);
                                markCount.incrementAndGet();
                                log.info("[MARK-STRESS] invalidation #{}", markCount.get());
                            }
                        }
                    }
                } catch (InterruptedException ie) {
                    Thread.currentThread().interrupt();
                } catch (Exception e) {
                    log.error("[MARK-STRESS] invalidator exception: {}", e.getMessage(), e);
                } finally {
                    doneLatch.countDown();
                }
            });

            startLatch.countDown();
            assertTrue(doneLatch.await(180, TimeUnit.SECONDS),
                    "Concurrent mark-invalidation test timed out after 180s");

            log.info("[MARK-STRESS] mode={}: {} invalidations fired during {} worker steps",
                    mode, markCount.get(), numWorkers * stepsPerWorker);

            assertEquals(0, throwErrors.get(),
                    mode + ": " + throwErrors.get() + " exceptions during concurrent mark stress");
            assertEquals(0, nullErrors.get(),
                    mode + ": " + nullErrors.get() + " null outputs during concurrent mark stress");
            assertEquals(0, rankErrors.get(),
                    mode + ": " + rankErrors.get() + " rank errors during concurrent mark stress " +
                            "(markVariable invalidation caused rank degradation)");
        } finally {
            executor.shutdownNow();
            try { sd.close(); } catch (Exception ignored) {}
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════════
    // Test 15: Concurrent sd.output() with independent SameDiff instances per thread
    //          Each thread creates its own SameDiff with the transformer graph,
    //          but they share the same native static initialization code paths.
    //          This stresses the static initializer races even more than shared-instance
    //          tests, because each thread hits initialize() / getTraitTable() / etc.
    // ═══════════════════════════════════════════════════════════════════════════════

    @ParameterizedTest(name = "concurrent_independent_instances | {0}")
    @MethodSource("modeProvider")
    @DisplayName("Concurrent sd.output() with independent SameDiff instances — static init stress")
    void testConcurrentIndependentInstances(ExecMode mode) throws Exception {
        // Pre-warm one instance on the main thread to populate the Triton singleton
        // cache and complete all synchronous CUDA operations (releaseGpuIntermediates,
        // module loading, etc.) before worker threads start their own captures.
        {
            java.util.Random warmRng = new java.util.Random(GRAPH_SEED);
            SameDiff warmSd = buildTransformerLogitsGraph(warmRng);
            applyMode(warmSd, mode);
            try {
                for (int i = 0; i < WARMUP_RUNS; i++) {
                    INDArray input = createDeterministicInput3D(i);
                    Map<String, INDArray> ph = Collections.singletonMap("x", input);
                    Map<String, INDArray> result = warmSd.output(ph, "out");
                    assertNotNull(result.get("out"), "Pre-warm step " + i + " null");
                }
            } finally {
                try { warmSd.close(); } catch (Exception ignored) {}
            }
        }

        int numThreads = 8;
        int stepsPerThread = WARMUP_RUNS + 4;
        ExecutorService executor = Executors.newFixedThreadPool(numThreads);
        AtomicInteger rankErrors = new AtomicInteger(0);
        AtomicInteger throwErrors = new AtomicInteger(0);
        CountDownLatch startLatch = new CountDownLatch(1);
        CountDownLatch doneLatch = new CountDownLatch(numThreads);
        List<SameDiff> instances = new CopyOnWriteArrayList<>();

        try {
            for (int t = 0; t < numThreads; t++) {
                final int threadId = t;
                executor.submit(() -> {
                    SameDiff sd = null;
                    try {
                        // Each thread creates its own graph — different seed per thread
                        java.util.Random rng = new java.util.Random(GRAPH_SEED + threadId);
                        sd = buildTransformerLogitsGraph(rng);
                        applyMode(sd, mode);
                        instances.add(sd);

                        startLatch.await();
                        for (int step = 0; step < stepsPerThread; step++) {
                            INDArray input = createDeterministicInput3D(threadId * 10000 + step);
                            Map<String, INDArray> ph = Collections.singletonMap("x", input);
                            try {
                                Map<String, INDArray> result = sd.output(ph, "out");
                                INDArray out = result.get("out");
                                if (out == null || out.rank() != 3) {
                                    rankErrors.incrementAndGet();
                                    log.error("[INDEP] thread={} step={} mode={}: " +
                                                    "out={} rank={} shape={}", threadId, step, mode,
                                            out != null, out != null ? out.rank() : -1,
                                            out != null ? Arrays.toString(out.shape()) : "null");
                                } else {
                                    assertArrayEquals(new long[]{BATCH, SEQ_LEN, VOCAB_SIZE}, out.shape(),
                                            "thread=" + threadId + " step=" + step);
                                }
                            } catch (Exception e) {
                                throwErrors.incrementAndGet();
                                log.error("[INDEP] thread={} step={}: EXCEPTION: {}",
                                        threadId, step, e.getMessage(), e);
                            }
                        }
                    } catch (InterruptedException ie) {
                        Thread.currentThread().interrupt();
                    } finally {
                        doneLatch.countDown();
                    }
                });
            }

            startLatch.countDown();
            assertTrue(doneLatch.await(180, TimeUnit.SECONDS),
                    "Concurrent independent instances test timed out");

            assertEquals(0, throwErrors.get(),
                    mode + ": " + throwErrors.get() + " exceptions with independent instances");
            assertEquals(0, rankErrors.get(),
                    mode + ": " + rankErrors.get() + " rank errors with independent instances");

            log.info("[INDEP] mode={} PASSED: {} threads x {} steps, 0 errors",
                    mode, numThreads, stepsPerThread);
        } finally {
            executor.shutdownNow();
            for (SameDiff sd : instances) {
                try { sd.close(); } catch (Exception ignored) {}
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════════
    // Isolated test: mean(keepDims=true) shape through DSP
    // ═══════════════════════════════════════════════════════════════════════════════

    @ParameterizedTest(name = "mean_keepDims_shape | {0}")
    @MethodSource("modeProvider")
    @DisplayName("mean(keepDims=true) produces rank-2 output [4,1] not rank-1 [4]")
    void testMeanKeepDimsShape(ExecMode mode) {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 4, 16);
        SDVariable xSq = sd.math().square("x_sq", x);
        SDVariable meanSq = sd.mean("mean_sq", xSq, true, 1);
        SDVariable eps = sd.constant("eps", Nd4j.scalar(DataType.FLOAT, 1e-6f));
        SDVariable rmsInv = sd.math().rsqrt("rms_inv", meanSq.add("mean_sq_eps", eps));
        SDVariable normed = x.mul("normed", rmsInv);
        sd.setOutputs("mean_sq", "normed");

        applyMode(sd, mode);

        // Enable debug/verbose to see all shapes from existing infra
        Nd4j.getEnvironment().setDebug(true);
        Nd4j.getEnvironment().setVerbose(true);

        try {
            INDArray input = Nd4j.randn(DataType.FLOAT, 4, 16);
            Map<String, INDArray> result = sd.output(Collections.singletonMap("x", input), "mean_sq", "normed");

            INDArray meanResult = result.get("mean_sq");
            INDArray normedResult = result.get("normed");

            log.info("[MEAN_KEEPDIMS] mode={} mean_sq: rank={} shape={}", mode,
                    meanResult.rank(), Arrays.toString(meanResult.shape()));
            log.info("[MEAN_KEEPDIMS] mode={} normed: rank={} shape={}", mode,
                    normedResult.rank(), Arrays.toString(normedResult.shape()));

            assertEquals(2, meanResult.rank(),
                    mode + ": mean(keepDims=true) should produce rank-2, got rank-" + meanResult.rank() +
                            " shape=" + Arrays.toString(meanResult.shape()));
            assertArrayEquals(new long[]{4, 1}, meanResult.shape(),
                    mode + ": mean(keepDims=true, axis=1) on [4,16] should produce [4,1], got " +
                            Arrays.toString(meanResult.shape()));
            assertArrayEquals(new long[]{4, 16}, normedResult.shape(),
                    mode + ": normed=x*rmsInv should broadcast [4,16]*[4,1]->[4,16], got " +
                            Arrays.toString(normedResult.shape()));
        } finally {
            Nd4j.getEnvironment().setDebug(false);
            Nd4j.getEnvironment().setVerbose(false);
            try { sd.close(); } catch (Exception ignored) {}
        }
    }
}
