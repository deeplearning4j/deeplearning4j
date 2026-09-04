/*
 * Copyright (c) 2026 Kompile
 *
 * Per-decode-step allocation churn regression test.
 *
 * Campaign finding (2026-09-02): the serving child's reserved memory grew
 * ~64 MB per frozen decode step (~1500 live allocations per step) and the
 * growth survived every boundary clear. No existing platform test asserted
 * per-step allocation counts — testRepeatedExecutionsReusePoolBuffers
 * explicitly skips exact counts, and testArrayLifecycleNoLeaks measures
 * across whole sd.output cycles — so the churn never surfaced in the harness.
 *
 * This test reproduces that pattern in-harness: a small static-KV decoder is
 * executed through N frozen decode steps and the allocated-memory total is
 * sampled across the steady-state window (after graph warm). The live-memory
 * delta per step must stay bounded; unbounded growth is exactly the
 * serving-child ratchet.
 */
package org.eclipse.deeplearning4j.nd4j.autodiff.samediff;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.SDVariable;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

@Slf4j
public class DspDecodeStepChurnTest {

    // Same small decoder geometry as DspLifecycleValidationTest's static-KV fixture.
    private static final int KV_HEADS = 2;
    private static final int KV_HEAD_DIM = 4;
    private static final int KV_HIDDEN = KV_HEADS * KV_HEAD_DIM; // 8
    private static final int KV_VOCAB = 16;
    private static final int KV_MAX_LEN = 8;

    /** Total decode steps; the first WARMUP_STEPS are excluded from the leak window. */
    private static final int STEPS = 30;
    private static final int WARMUP_STEPS = 5;

    /**
     * Bounded per-step live-memory growth. Each step's transient intermediates
     * must be reclaimed (freed or pooled-reused) before the next step samples.
     * 2 MB per step tolerates metric noise and logits carryover; the serving
     * churn measured ~64 MB/step, an order of magnitude above this bound.
     */
    private static final long MAX_BYTES_PER_STEP = 2L * 1024 * 1024;

    @Test
    public void frozenDecodeStepsDoNotAccumulateLiveMemory() {
        assumeTrue(Nd4j.backends().isCudaAvailable(), "CUDA required");
        try {
            assumeTrue(Nd4j.getNativeOps().isTritonAvailable(), "Triton required");
        } catch (Throwable t) {
            assumeTrue(false, "Triton required");
        }

        Nd4j.getRandom().setSeed(42L);

        SameDiff sd = buildStaticKvDecoder(KV_MAX_LEN);
        try {
            sd.setGraphExecutionMode(GraphExecutionMode.AUTO);
            sd.setDspAutoCompileEnabled(true);
            sd.setDspNativeAutoCompileEnabled(true);

            List<Map<String, INDArray>> live = new ArrayList<>();
            long bytesPerStepSum = 0;
            int samples = 0;
            long worstStepDelta = 0;

            for (int step = 0; step < STEPS; step++) {
                Map<String, INDArray> inputs = staticKvInputs(KV_MAX_LEN, step);
                Map<String, INDArray> out = sd.output(inputs,
                        "logits", "present_key", "present_value");
                Nd4j.getExecutioner().commit();

                // Close outputs the way GenerationPipeline does per decode step.
                closeAll(out);
                closeAll(inputs);

                if (step >= WARMUP_STEPS) {
                    long before = Nd4j.getMemoryManager().allocatedMemory(0);
                    Map<String, INDArray> inputs2 = staticKvInputs(KV_MAX_LEN, step + 1);
                    Map<String, INDArray> out2 = sd.output(inputs2,
                            "logits", "present_key", "present_value");
                    Nd4j.getExecutioner().commit();
                    long after = Nd4j.getMemoryManager().allocatedMemory(0);

                    closeAll(out2);
                    closeAll(inputs2);

                    long delta = after - before;
                    bytesPerStepSum += delta;
                    samples++;
                    worstStepDelta = Math.max(worstStepDelta, delta);
                    log.info("step={} liveMemoryDelta={} B (avg {} B/step over {} samples)",
                            step + 1, delta, bytesPerStepSum / samples, samples);
                }
            }

            long avgPerStep = samples == 0 ? 0 : bytesPerStepSum / samples;
            log.info("frozenDecodeStepsDoNotAccumulateLiveMemory: samples={} avgPerStep={}B "
                            + "worstStep={}B bound={}B",
                    samples, avgPerStep, worstStepDelta, MAX_BYTES_PER_STEP);

            assertTrue(avgPerStep <= MAX_BYTES_PER_STEP,
                    "Per-step live-memory churn " + avgPerStep + " B/step exceeds bound "
                            + MAX_BYTES_PER_STEP + " B/step (worst single step "
                            + worstStepDelta + " B) — serving-child ratchet reproduced");
        } finally {
            sd.close();
        }
    }

    // ─── Fixture (same geometry as DspLifecycleValidationTest) ─────────────

    private static SameDiff buildStaticKvDecoder(int maxKvLen) {
        SameDiff sd = SameDiff.create();

        SDVariable inputsEmbeds = sd.placeHolder("input_embeds",
                DataType.FLOAT, 1, 1, KV_HIDDEN);
        SDVariable attentionMask = sd.placeHolder("attention_mask",
                DataType.FLOAT, 1, 1, 1, maxKvLen + 1);
        SDVariable pastKey = sd.placeHolder("past_key",
                DataType.FLOAT, 1, KV_HEADS, maxKvLen, KV_HEAD_DIM);
        SDVariable pastValue = sd.placeHolder("past_value",
                DataType.FLOAT, 1, KV_HEADS, maxKvLen, KV_HEAD_DIM);

        SDVariable wq = sd.var("wq", Nd4j.randn(DataType.FLOAT, KV_HIDDEN, KV_HIDDEN).muli(0.05));
        SDVariable wk = sd.var("wk", Nd4j.randn(DataType.FLOAT, KV_HIDDEN, KV_HIDDEN).muli(0.05));
        SDVariable wv = sd.var("wv", Nd4j.randn(DataType.FLOAT, KV_HIDDEN, KV_HIDDEN).muli(0.05));
        SDVariable wo = sd.var("wo", Nd4j.randn(DataType.FLOAT, KV_HIDDEN, KV_HIDDEN).muli(0.05));
        SDVariable wlogits = sd.var("wlogits",
                Nd4j.randn(DataType.FLOAT, KV_HIDDEN, KV_VOCAB).muli(0.05));

        SDVariable squeezed = sd.squeeze("squeezed", inputsEmbeds, 1);
        SDVariable qFlat = sd.mmul("q_flat", squeezed, wq);
        SDVariable kFlat = sd.mmul("k_flat", squeezed, wk);
        SDVariable vFlat = sd.mmul("v_flat", squeezed, wv);

        SDVariable qNew = sd.reshape("q_new", qFlat, 1, KV_HEADS, 1, KV_HEAD_DIM);
        SDVariable kNew = sd.reshape("k_new", kFlat, 1, KV_HEADS, 1, KV_HEAD_DIM);
        SDVariable vNew = sd.reshape("v_new", vFlat, 1, KV_HEADS, 1, KV_HEAD_DIM);

        SDVariable presentKey = sd.concat("present_key", 2, pastKey, kNew);
        SDVariable presentValue = sd.concat("present_value", 2, pastValue, vNew);

        // Scores: q · k^T => [1, heads, 1, maxKvLen+1] (4-D batched mmul, like the real fixture)
        SDVariable kT = sd.permute("kT", presentKey, 0, 1, 3, 2);
        SDVariable scores = sd.mmul("scores", qNew, kT);
        SDVariable scaled = scores.mul("scaled", (float) (1.0 / Math.sqrt(KV_HEAD_DIM)));
        SDVariable masked = scaled.add("masked", attentionMask);
        SDVariable probs = sd.nn.softmax("probs", masked, -1);

        // Attention output: probs · v => [1, heads, 1, head_dim]
        SDVariable attnOut = sd.mmul("attn_out", probs, presentValue);

        SDVariable attnFlat = sd.reshape("attn_flat", attnOut, 1, KV_HIDDEN);
        SDVariable projected = sd.mmul("projected", attnFlat, wo);
        SDVariable logitsFlat = sd.mmul("logits_flat", projected, wlogits);
        SDVariable logits = sd.reshape("logits", logitsFlat, 1, 1, KV_VOCAB);

        sd.setOutputs("logits", "present_key", "present_value");
        return sd;
    }

    private static Map<String, INDArray> staticKvInputs(int maxKvLen, long position) {
        Map<String, INDArray> in = new LinkedHashMap<>();
        in.put("input_embeds", Nd4j.randn(DataType.FLOAT, 1, 1, KV_HIDDEN).muli(0.1));
        INDArray mask = Nd4j.valueArrayOf(new long[]{1, 1, 1, maxKvLen + 1}, -1e9f, DataType.FLOAT);
        for (long i = 0; i <= position && i < maxKvLen; i++) {
            mask.putScalar(new long[]{0, 0, 0, i}, 0.0);
        }
        mask.putScalar(new long[]{0, 0, 0, maxKvLen}, 0.0);
        in.put("attention_mask", mask);
        in.put("past_key", Nd4j.randn(DataType.FLOAT, 1, KV_HEADS, maxKvLen, KV_HEAD_DIM).muli(0.05));
        in.put("past_value", Nd4j.randn(DataType.FLOAT, 1, KV_HEADS, maxKvLen, KV_HEAD_DIM).muli(0.05));
        return in;
    }

    private static void closeAll(Map<String, INDArray> arrays) {
        if (arrays == null) return;
        for (INDArray arr : arrays.values()) {
            if (arr != null && arr.closeable() && !arr.wasClosed()) {
                arr.close();
            }
        }
    }
}
