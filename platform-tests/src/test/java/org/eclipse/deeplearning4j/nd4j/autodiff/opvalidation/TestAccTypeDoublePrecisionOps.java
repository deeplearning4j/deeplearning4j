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

package org.eclipse.deeplearning4j.nd4j.autodiff.opvalidation;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.Test;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.transforms.custom.MoeWeightedSum;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Double-precision correctness gate for the CUDA op helpers converted to the AccType
 * accumulation pattern (accumulate in double when the element type is double, float
 * otherwise) during the 2026-07-14 reduce-consolidation / AccType campaign.
 *
 * Two flavours:
 *  - PURE-ACCUMULATION ops (moe_weighted_sum, segment_gemm) have no internal
 *    float-only state, so a double run is checked against an exact double reference.
 *    Without AccType these accumulate long inner products in float and drift from the
 *    reference; with AccType a double input accumulates in double and matches tightly.
 *  - MIXED-PRECISION ops (causal_conv1d, linear_attention_decode, lightning_attention)
 *    keep a float32 recurrent/conv state BY DESIGN, so the meaningful gate is that the
 *    double execution path runs, produces finite output of the right shape, and stays
 *    close to the float path. This catches double-path buffer-overruns/crashes — the
 *    class of bug that a float-only pass silently hides (cf. center_and_sharpen_bp,
 *    where a FLOAT32 scratch buffer reinterpreted as double overran and only failed
 *    once a double test exercised it).
 */
@Slf4j
@NativeTag
public class TestAccTypeDoublePrecisionOps extends BaseOpValidation {

    /** Assert every element is finite. Uses min/max rather than sum(x^2): filter ops mask
     *  tokens to -DataTypeUtils::max (a huge but FINITE sentinel, e.g. -DBL_MAX) whose square
     *  overflows to +Inf — so a sum-of-squares check would spuriously fail. min/max finite
     *  correctly treats the sentinel as finite while still catching a real Inf/NaN. */
    private static void assertAllFinite(INDArray x, String msg) {
        double mn = x.minNumber().doubleValue();
        double mx = x.maxNumber().doubleValue();
        assertTrue(Double.isFinite(mn) && Double.isFinite(mx), msg + " (min=" + mn + ", max=" + mx + ")");
    }

    // ── moe_weighted_sum (dense): out[t][d] = Σ_k weights[t][k] * expertOutputs[t][k][d] ──
    @Test
    public void testMoeWeightedSumDoubleReference() {
        int T = 6, topK = 8, D = 512;   // large topK*... to make float accumulation drift visible
        Nd4j.getRandom().setSeed(4242L);

        INDArray eo = Nd4j.rand(DataType.DOUBLE, T, topK, D);
        INDArray w  = Nd4j.rand(DataType.DOUBLE, T, topK);
        w = w.div(w.sum(true, 1));       // normalise per-token weights to sum 1

        INDArray[] outs = Nd4j.exec(new MoeWeightedSum(eo, w));
        INDArray result = outs[0];
        assertArrayEquals(new long[]{T, D}, result.shape(), "moe_weighted_sum output shape");
        assertAllFinite(result, "moe_weighted_sum double output must be finite");

        // Exact double reference: einsum over topK.
        INDArray ref = Nd4j.create(DataType.DOUBLE, T, D);
        for (int t = 0; t < T; t++) {
            for (int k = 0; k < topK; k++) {
                double wk = w.getDouble(t, k);
                for (int d = 0; d < D; d++) {
                    ref.putScalar(new int[]{t, d}, ref.getDouble(t, d) + wk * eo.getDouble(t, k, d));
                }
            }
        }
        double maxDiff = Transforms.abs(result.sub(ref)).maxNumber().doubleValue();
        log.info("moe_weighted_sum double maxDiff vs reference = {}", maxDiff);
        assertTrue(maxDiff < 1e-12, "double moe_weighted_sum must match double reference (maxDiff=" + maxDiff + ")");
    }

    // ── segment_gemm: per-expert matmul, out[t] = input[t] @ weights[expert(t)] ──
    @Test
    public void testSegmentGemmDoubleReference() {
        int inDim = 4, outDim = 3;
        // 2 experts: tokens 0..2 -> expert 0, tokens 3..4 -> expert 1
        long[] offsetsArr = {0, 3};
        long[] sizesArr   = {3, 2};
        int totalTokens = 5, numExperts = 2;
        Nd4j.getRandom().setSeed(909L);

        INDArray input   = Nd4j.rand(DataType.DOUBLE, totalTokens, inDim);
        INDArray weights = Nd4j.rand(DataType.DOUBLE, numExperts, inDim, outDim);
        INDArray offsets = Nd4j.createFromArray(offsetsArr);   // INT64
        INDArray sizes   = Nd4j.createFromArray(sizesArr);     // INT64
        INDArray output  = Nd4j.create(DataType.DOUBLE, totalTokens, outDim);

        Nd4j.exec(DynamicCustomOp.builder("segment_gemm")
                .addInputs(input, weights, offsets, sizes)
                .addOutputs(output)
                .build());
        assertAllFinite(output, "segment_gemm double output must be finite");

        // Exact double reference: per-segment matmul.
        INDArray ref = Nd4j.create(DataType.DOUBLE, totalTokens, outDim);
        for (int e = 0; e < numExperts; e++) {
            long off = offsetsArr[e], sz = sizesArr[e];
            INDArray rows = input.get(org.nd4j.linalg.indexing.NDArrayIndex.interval(off, off + sz),
                                      org.nd4j.linalg.indexing.NDArrayIndex.all());
            INDArray we = weights.get(org.nd4j.linalg.indexing.NDArrayIndex.point(e),
                                      org.nd4j.linalg.indexing.NDArrayIndex.all(),
                                      org.nd4j.linalg.indexing.NDArrayIndex.all());
            INDArray seg = rows.mmul(we);   // [sz, outDim]
            ref.put(new org.nd4j.linalg.indexing.INDArrayIndex[]{
                    org.nd4j.linalg.indexing.NDArrayIndex.interval(off, off + sz),
                    org.nd4j.linalg.indexing.NDArrayIndex.all()}, seg);
        }
        double maxDiff = Transforms.abs(output.sub(ref)).maxNumber().doubleValue();
        log.info("segment_gemm double maxDiff vs reference = {}", maxDiff);
        assertTrue(maxDiff < 1e-10, "double segment_gemm must match per-segment matmul (maxDiff=" + maxDiff + ")");
    }

    // ── causal_conv1d: double path runs, finite, correct shape, close to float path ──
    @Test
    public void testCausalConv1dDoublePathRunsAndConsistent() {
        int B = 2, L = 7, D = 8, K = 4;
        Nd4j.getRandom().setSeed(77L);
        INDArray xD = Nd4j.rand(DataType.DOUBLE, B, L, D);
        INDArray wD = Nd4j.rand(DataType.DOUBLE, D, K);

        INDArray outD   = Nd4j.create(DataType.DOUBLE, B, L, D);
        INDArray stateD = Nd4j.create(DataType.DOUBLE, B, D, K - 1);
        Nd4j.exec(DynamicCustomOp.builder("causal_conv1d")
                .addInputs(xD, wD).addOutputs(outD, stateD).build());
        assertAllFinite(outD, "causal_conv1d double output must be finite");
        assertArrayEquals(new long[]{B, L, D}, outD.shape(), "causal_conv1d output shape");

        INDArray outF   = Nd4j.create(DataType.FLOAT, B, L, D);
        INDArray stateF = Nd4j.create(DataType.FLOAT, B, D, K - 1);
        Nd4j.exec(DynamicCustomOp.builder("causal_conv1d")
                .addInputs(xD.castTo(DataType.FLOAT), wD.castTo(DataType.FLOAT))
                .addOutputs(outF, stateF).build());

        double maxDiff = Transforms.abs(outD.sub(outF.castTo(DataType.DOUBLE))).maxNumber().doubleValue();
        log.info("causal_conv1d double-vs-float maxDiff = {}", maxDiff);
        assertTrue(maxDiff < 1e-3, "causal_conv1d double path must track float within tol (maxDiff=" + maxDiff + ")");
    }

    // ── linear_attention_decode: decode-step; state/decay float32 by design ──
    // Double path (query/key/value double -> q·state dot accumulates in double) must run
    // finite with the right output shape. This exercises the AccType dot accumulation.
    @Test
    public void testLinearAttentionDecodeDoublePathRuns() {
        int B = 2, H = 2, dk = 4, dv = 4;
        Nd4j.getRandom().setSeed(31L);
        INDArray q = Nd4j.rand(DataType.DOUBLE, B, 1, H, dk);
        INDArray k = Nd4j.rand(DataType.DOUBLE, B, 1, H, dk);
        INDArray v = Nd4j.rand(DataType.DOUBLE, B, 1, H, dv);
        INDArray decay = Nd4j.rand(DataType.FLOAT, H).addi(0.1f);      // float32 by contract
        INDArray state = Nd4j.rand(DataType.FLOAT, B, H, dv, dk);      // float32 by contract
        INDArray out = Nd4j.create(DataType.DOUBLE, B, 1, H, dv);

        Nd4j.exec(DynamicCustomOp.builder("linear_attention_decode")
                .addInputs(q, k, v, decay, state).addOutputs(out).build());
        assertAllFinite(out, "linear_attention_decode double output must be finite");
        assertArrayEquals(new long[]{B, 1, H, dv}, out.shape(), "linear_attention_decode output shape");
    }

    // ── lightning_attention: prefill linear attention; state float32 by design ──
    @Test
    public void testLightningAttentionDoublePathRuns() {
        int B = 2, seq = 5, H = 2, d = 4;
        Nd4j.getRandom().setSeed(52L);
        INDArray q = Nd4j.rand(DataType.DOUBLE, B, seq, H, d);
        INDArray k = Nd4j.rand(DataType.DOUBLE, B, seq, H, d);
        INDArray v = Nd4j.rand(DataType.DOUBLE, B, seq, H, d);
        INDArray decay = Nd4j.rand(DataType.FLOAT, H).addi(0.1f);
        INDArray state = Nd4j.zeros(DataType.FLOAT, B, H, d, d);
        INDArray out = Nd4j.create(DataType.DOUBLE, B, seq, H, d);

        Nd4j.exec(DynamicCustomOp.builder("lightning_attention")
                .addInputs(q, k, v, decay, state).addOutputs(out).build());
        assertAllFinite(out, "lightning_attention double output must be finite");
        assertArrayEquals(new long[]{B, seq, H, d}, out.shape(), "lightning_attention output shape");
    }

    // ── top_k_renorm: softmax -> keep top-k -> renormalize; deterministic ──
    // Double path (AccType softmax/renorm accumulation) must run finite, renormalize each row
    // to 1, and track the float path. Gates the sampling-op AccType conversion.
    @Test
    public void testTopKRenormDoublePathRunsAndConsistent() {
        int B = 3, V = 128, k = 10;
        Nd4j.getRandom().setSeed(123L);
        INDArray logitsD = Nd4j.randn(DataType.DOUBLE, B, V);

        INDArray outD = Nd4j.create(DataType.DOUBLE, B, V);
        Nd4j.exec(DynamicCustomOp.builder("top_k_renorm")
                .addInputs(logitsD).addOutputs(outD).addIntegerArguments(k).build());
        assertAllFinite(outD, "top_k_renorm double output must be finite");
        for (int b = 0; b < B; b++)
            assertEquals(1.0, outD.getRow(b).sumNumber().doubleValue(), 1e-9,
                    "top_k_renorm row " + b + " must renormalize to 1");

        INDArray outF = Nd4j.create(DataType.FLOAT, B, V);
        Nd4j.exec(DynamicCustomOp.builder("top_k_renorm")
                .addInputs(logitsD.castTo(DataType.FLOAT)).addOutputs(outF).addIntegerArguments(k).build());
        double maxDiff = Transforms.abs(outD.sub(outF.castTo(DataType.DOUBLE))).maxNumber().doubleValue();
        log.info("top_k_renorm double-vs-float maxDiff = {}", maxDiff);
        assertTrue(maxDiff < 1e-3, "top_k_renorm double path must track float within tol (maxDiff=" + maxDiff + ")");
    }

    // ── top_p_renorm: softmax -> nucleus (cumprob >= p) -> renormalize; deterministic ──
    @Test
    public void testTopPRenormDoublePathRunsAndConsistent() {
        int B = 3, V = 128;
        double p = 0.9;
        Nd4j.getRandom().setSeed(321L);
        INDArray logitsD = Nd4j.randn(DataType.DOUBLE, B, V);

        INDArray outD = Nd4j.create(DataType.DOUBLE, B, V);
        Nd4j.exec(DynamicCustomOp.builder("top_p_renorm")
                .addInputs(logitsD).addOutputs(outD).addFloatingPointArguments(p).build());
        assertAllFinite(outD, "top_p_renorm double output must be finite");
        for (int b = 0; b < B; b++)
            assertEquals(1.0, outD.getRow(b).sumNumber().doubleValue(), 1e-9,
                    "top_p_renorm row " + b + " must renormalize to 1");

        INDArray outF = Nd4j.create(DataType.FLOAT, B, V);
        Nd4j.exec(DynamicCustomOp.builder("top_p_renorm")
                .addInputs(logitsD.castTo(DataType.FLOAT)).addOutputs(outF).addFloatingPointArguments(p).build());
        double maxDiff = Transforms.abs(outD.sub(outF.castTo(DataType.DOUBLE))).maxNumber().doubleValue();
        log.info("top_p_renorm double-vs-float maxDiff = {}", maxDiff);
        assertTrue(maxDiff < 1e-3, "top_p_renorm double path must track float within tol (maxDiff=" + maxDiff + ")");
    }

    // ── sampling_penalties (applyPenaltiesKernel): repetition/freq/presence penalties ──
    // minP=0 so only the deterministic penalty kernel runs; double path must track float.
    @Test
    public void testSamplingPenaltiesDoublePathConsistent() {
        int V = 96;
        Nd4j.getRandom().setSeed(11L);
        INDArray logitsD = Nd4j.randn(DataType.DOUBLE, 1, V);
        INDArray inputIds = Nd4j.createFromArray(new long[]{3L, 3L, 7L, 15L}).reshape(1, 4);  // token 3 repeated

        INDArray outD = Nd4j.create(DataType.DOUBLE, 1, V);
        Nd4j.exec(DynamicCustomOp.builder("sampling_penalties")
                .addInputs(logitsD, inputIds).addOutputs(outD)
                .addFloatingPointArguments(1.5, 0.1, 0.05, 0.0).build());   // rep, freq, pres, minP=0
        assertAllFinite(outD, "sampling_penalties double output must be finite");

        INDArray outF = Nd4j.create(DataType.FLOAT, 1, V);
        Nd4j.exec(DynamicCustomOp.builder("sampling_penalties")
                .addInputs(logitsD.castTo(DataType.FLOAT), inputIds).addOutputs(outF)
                .addFloatingPointArguments(1.5, 0.1, 0.05, 0.0).build());
        double maxDiff = Transforms.abs(outD.sub(outF.castTo(DataType.DOUBLE))).maxNumber().doubleValue();
        log.info("sampling_penalties double-vs-float maxDiff = {}", maxDiff);
        assertTrue(maxDiff < 1e-3, "sampling_penalties double path must track float within tol (maxDiff=" + maxDiff + ")");
    }

    // ── token_sample greedy (greedyArgmaxKernel): CANARY for the float-narrowing bug ──
    // idx 7 is the strict max only in DOUBLE (5.0 + 1e-9 rounds to 5.0 in float, tying idx 3).
    // An AccType (double) argmax returns 7; a float argmax would not resolve the tie to 7.
    @Test
    public void testTokenSampleGreedyDoubleArgmaxCanary() {
        int V = 64;
        INDArray logits = Nd4j.valueArrayOf(new long[]{1, V}, -10.0, DataType.DOUBLE);
        logits.putScalar(0, 3, 5.0);           // float-indistinguishable runner-up (earlier index)
        logits.putScalar(0, 7, 5.0 + 1e-9);    // true double max
        INDArray out = Nd4j.create(DataType.INT64, 1);

        Nd4j.exec(DynamicCustomOp.builder("token_sample")
                .addInputs(logits).addOutputs(out)
                .addFloatingPointArguments(0.0, 0.0)     // temperature=0 -> greedy
                .addIntegerArguments(0, 0).build());      // topK=0, seed=0
        long picked = out.getLong(0);
        log.info("token_sample greedy argmax picked idx {}", picked);
        assertEquals(7L, picked, "greedy argmax must resolve the double-only max at idx 7");
    }

    // ── token_sample sampling (tempTopKTopPSampleKernel): output stays in the top-k set ──
    @Test
    public void testTokenSampleTopKStaysInKeptSet() {
        int V = 64;
        INDArray logits = Nd4j.valueArrayOf(new long[]{1, V}, -20.0, DataType.DOUBLE);
        logits.putScalar(0, 3, 8.0);   // only idx 3 and 7 have meaningful mass
        logits.putScalar(0, 7, 7.5);
        for (int seed = 1; seed <= 40; seed++) {
            INDArray out = Nd4j.create(DataType.INT64, 1);
            Nd4j.exec(DynamicCustomOp.builder("token_sample")
                    .addInputs(logits).addOutputs(out)
                    .addFloatingPointArguments(1.0, 1.0)      // temperature=1, topP=1
                    .addIntegerArguments(2, seed).build());    // topK=2, seed
            long picked = out.getLong(0);
            assertTrue(picked == 3 || picked == 7,
                    "token_sample topK=2 must pick from {3,7}, got " + picked + " (seed " + seed + ")");
        }
    }

    // ── typical_p_filter (new op): deterministic entropy-deviation filter ──
    // Runs the double AccType path: no-op at 1.0, masks a proper subset at 0.5, output finite.
    @Test
    public void testTypicalPFilterDoublePath() {
        int V = 128;
        Nd4j.getRandom().setSeed(7L);
        INDArray logitsD = Nd4j.randn(DataType.DOUBLE, 1, V);

        INDArray noop = Nd4j.create(DataType.DOUBLE, 1, V);
        Nd4j.exec(DynamicCustomOp.builder("typical_p_filter")
                .addInputs(logitsD).addOutputs(noop).addFloatingPointArguments(1.0).build());
        assertTrue(Transforms.abs(noop.sub(logitsD)).maxNumber().doubleValue() < 1e-12,
                "typical_p_filter typicalP=1.0 must be an exact no-op");

        INDArray outD = Nd4j.create(DataType.DOUBLE, 1, V);
        Nd4j.exec(DynamicCustomOp.builder("typical_p_filter")
                .addInputs(logitsD).addOutputs(outD).addFloatingPointArguments(0.5).build());
        // masked tokens are -inf (generation masking convention); kept tokens are finite.
        assertTrue(outD.minNumber().doubleValue() < -1e30, "typical_p_filter must mask some tokens (-inf)");
        assertTrue(outD.maxNumber().doubleValue() > -1e30, "typical_p_filter must keep some tokens");
    }

    // ── xtc_filter (new op): Exclude Top Choices; xtcProbability=1.0 forces deterministic apply ──
    // Among above-threshold tokens {3,7,11}, XTC masks all but the lowest-probability one (11).
    @Test
    public void testXtcFilterDoublePathForceApply() {
        int V = 64;
        INDArray logits = Nd4j.valueArrayOf(new long[]{1, V}, -20.0, DataType.DOUBLE);
        logits.putScalar(0, 3, 5.0);    // p ~ 0.66
        logits.putScalar(0, 7, 4.0);    // p ~ 0.24
        logits.putScalar(0, 11, 3.0);   // p ~ 0.09  (lowest above-threshold -> kept)
        INDArray outD = Nd4j.create(DataType.DOUBLE, 1, V);

        Nd4j.exec(DynamicCustomOp.builder("xtc_filter")
                .addInputs(logits).addOutputs(outD)
                .addFloatingPointArguments(1.0, 0.05)   // xtcProbability=1 (force), xtcThreshold=0.05
                .addIntegerArguments(42).build());        // seed
        // excluded tokens are masked to -inf; the kept token stays finite.
        assertTrue(outD.getDouble(0, 3) < -1e30, "xtc_filter must exclude the top choice (idx 3)");
        assertTrue(outD.getDouble(0, 7) < -1e30, "xtc_filter must exclude idx 7");
        assertEquals(3.0, outD.getDouble(0, 11), 1e-12, "xtc_filter must keep the lowest above-threshold token (idx 11)");
    }
}
