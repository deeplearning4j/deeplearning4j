package org.eclipse.deeplearning4j.nd4j.linalg.graph;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.validation.GradCheckUtil;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.jupiter.api.Assertions.assertTrue;

/** Minimal repros to isolate the set2Set CUDA recurrent-backward bug. */
public class Set2SetReproTest extends BaseNd4jTestWithBackends {

    /** (1) A variable fans out into TWO concats, then summed -> tests concat-with-reused-input grad accumulation. */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void reproConcatFanout(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(1L);
        SameDiff sd = SameDiff.create();
        SDVariable m = sd.var("m", Nd4j.randn(DataType.DOUBLE, 1, 3));
        SDVariable a = sd.var("a", Nd4j.randn(DataType.DOUBLE, 1, 3));
        SDVariable b = sd.var("b", Nd4j.randn(DataType.DOUBLE, 1, 3));
        SDVariable c1 = sd.concat(1, m, a);
        SDVariable c2 = sd.concat(1, m, b);   // m reused
        SDVariable out = c1.add(c2);
        sd.sum("loss", out);
        sd.setLossVariables("loss");
        assertTrue(GradCheckUtil.checkGradients(sd, null), "concat-fanout grad check failed");
    }

    /** (2) A weight reused in TWO mmuls (across "steps") -> tests weight-reuse grad accumulation. */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void reproWeightReuse(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(2L);
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.var("x", Nd4j.randn(DataType.DOUBLE, 2, 3));
        SDVariable W = sd.var("W", Nd4j.randn(DataType.DOUBLE, 3, 3).muli(0.5));
        SDVariable y1 = sd.math().tanh(sd.mmul(x, W));
        SDVariable y2 = sd.math().tanh(sd.mmul(y1, W));   // W reused
        sd.sum("loss", y2);
        sd.setLossVariables("loss");
        assertTrue(GradCheckUtil.checkGradients(sd, null), "weight-reuse grad check failed");
    }

    /** (3) set2Set GRU-loop essence (no attention): m fans out + W reused + concat, in a 2-step loop. */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void reproMiniLoop(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(3L);
        SameDiff sd = SameDiff.create();
        long d = 3;
        SDVariable m = sd.var("m", Nd4j.randn(DataType.DOUBLE, 1, d));
        SDVariable h0 = sd.var("h", Nd4j.randn(DataType.DOUBLE, 1, d));
        SDVariable W = sd.var("W", Nd4j.randn(DataType.DOUBLE, 2 * d, d).muli(0.5));
        SDVariable h = h0;
        for (int t = 0; t < 2; t++) {
            SDVariable xh = sd.concat(1, m, h);   // m + h fan out each step
            h = sd.math().tanh(sd.mmul(xh, W));   // W reused each step
        }
        SDVariable out = sd.concat(1, m, h);      // m reused again
        sd.sum("loss", out);
        sd.setLossVariables("loss");
        assertTrue(GradCheckUtil.checkGradients(sd, null), "mini-loop grad check failed");
    }

    /** Full set2Set GRU-gate cell without attention: isolates sigmoid/multiply gate backward. */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void reproGatedMiniLoop(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(50L);
        SameDiff sd = SameDiff.create();
        long d = 3;
        SDVariable m = sd.var("m", Nd4j.randn(DataType.DOUBLE, 1, d));
        SDVariable h = sd.var("h", Nd4j.randn(DataType.DOUBLE, 1, d));
        SDVariable wZr = sd.var("wZr", Nd4j.randn(DataType.DOUBLE, 2 * d, d).muli(0.5));
        SDVariable bZr = sd.var("bZr", Nd4j.randn(DataType.DOUBLE, 1, d).muli(0.1));
        SDVariable wZu = sd.var("wZu", Nd4j.randn(DataType.DOUBLE, 2 * d, d).muli(0.5));
        SDVariable bZu = sd.var("bZu", Nd4j.randn(DataType.DOUBLE, 1, d).muli(0.1));
        SDVariable wC = sd.var("wC", Nd4j.randn(DataType.DOUBLE, 2 * d, d).muli(0.5));
        SDVariable bC = sd.var("bC", Nd4j.randn(DataType.DOUBLE, 1, d).muli(0.1));
        for (int t = 0; t < 2; t++) {
            SDVariable xh = sd.concat(1, m, h);
            SDVariable zr = sd.nn().sigmoid(sd.mmul(xh, wZr).add(bZr));
            SDVariable zu = sd.nn().sigmoid(sd.mmul(xh, wZu).add(bZu));
            SDVariable rh = sd.concat(1, m, zr.mul(h));
            SDVariable hh = sd.math().tanh(sd.mmul(rh, wC).add(bC));
            h = h.mul(zu.mul(-1.0).add(1.0)).add(zu.mul(hh));
        }
        SDVariable out = sd.concat(1, m, h);
        sd.sum("loss", out);
        sd.setLossVariables("loss");
        assertTrue(GradCheckUtil.checkGradients(sd, null), "gated-mini-loop grad check failed");
    }

    /** One set2Set step with full GRU gates: attention plus all gate math, without multi-step recurrence. */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void reproAttnGatedT1(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(50L);
        SameDiff sd = SameDiff.create();
        long d = 3, n = 4;
        SDVariable node = sd.var("node", Nd4j.randn(DataType.DOUBLE, n, d));
        SDVariable h = sd.var("h", Nd4j.randn(DataType.DOUBLE, 1, d));
        SDVariable wZr = sd.var("wZr", Nd4j.randn(DataType.DOUBLE, 2 * d, d).muli(0.5));
        SDVariable bZr = sd.var("bZr", Nd4j.randn(DataType.DOUBLE, 1, d).muli(0.1));
        SDVariable wZu = sd.var("wZu", Nd4j.randn(DataType.DOUBLE, 2 * d, d).muli(0.5));
        SDVariable bZu = sd.var("bZu", Nd4j.randn(DataType.DOUBLE, 1, d).muli(0.1));
        SDVariable wC = sd.var("wC", Nd4j.randn(DataType.DOUBLE, 2 * d, d).muli(0.5));
        SDVariable bC = sd.var("bC", Nd4j.randn(DataType.DOUBLE, 1, d).muli(0.1));
        SDVariable xKV = sd.reshape(node, 1L, -1L, d);
        SDVariable qQ = sd.reshape(h, 1L, 1L, d);
        SDVariable attn = sd.nn().dotProductAttentionV2(qQ, xKV, xKV, null, null, 0.0, 0.0, false, false);
        SDVariable m = sd.reshape(attn, 1L, d);
        SDVariable xh = sd.concat(1, m, h);
        SDVariable zr = sd.nn().sigmoid(sd.mmul(xh, wZr).add(bZr));
        SDVariable zu = sd.nn().sigmoid(sd.mmul(xh, wZu).add(bZu));
        SDVariable rh = sd.concat(1, m, zr.mul(h));
        SDVariable hh = sd.math().tanh(sd.mmul(rh, wC).add(bC));
        h = h.mul(zu.mul(-1.0).add(1.0)).add(zu.mul(hh));
        SDVariable out = sd.concat(1, m, h);
        sd.sum("loss", out);
        sd.setLossVariables("loss");
        assertTrue(GradCheckUtil.checkGradients(sd, null), "attn-gated-T1 grad check failed");
    }

    /** (4) set2Set structure: attention RECOMPUTED inside the loop (m = attention(reshape(h)) each step). */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void reproAttnInLoop(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(4L);
        SameDiff sd = SameDiff.create();
        long d = 3, n = 4;
        SDVariable h0 = sd.var("h", Nd4j.randn(DataType.DOUBLE, 1, d));
        SDVariable node = sd.var("node", Nd4j.randn(DataType.DOUBLE, n, d));
        SDVariable W = sd.var("W", Nd4j.randn(DataType.DOUBLE, 2 * d, d).muli(0.5));
        SDVariable xKV = sd.reshape(node, 1L, -1L, d);
        SDVariable h = h0;
        SDVariable m = sd.zerosLike(h0);
        for (int t = 0; t < 2; t++) {
            SDVariable qQ = sd.reshape(h, 1L, 1L, d);
            SDVariable attn = sd.nn().dotProductAttentionV2(qQ, xKV, xKV, null, null, 0.0, 0.0, false, false);
            m = sd.reshape(attn, 1L, d);
            SDVariable xh = sd.concat(1, m, h);
            h = sd.math().tanh(sd.mmul(xh, W));
        }
        SDVariable out = sd.concat(1, m, h);
        sd.sum("loss", out);
        sd.setLossVariables("loss");
        assertTrue(GradCheckUtil.checkGradients(sd, null), "attn-in-loop grad check failed");
    }

    /** (5) Same as (4) but ONE step -> isolates single-attention vs multi-attention-instance. */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void reproAttnInLoopT1(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(5L);
        SameDiff sd = SameDiff.create();
        long d = 3, n = 4;
        SDVariable h0 = sd.var("h", Nd4j.randn(DataType.DOUBLE, 1, d));
        SDVariable node = sd.var("node", Nd4j.randn(DataType.DOUBLE, n, d));
        SDVariable W = sd.var("W", Nd4j.randn(DataType.DOUBLE, 2 * d, d).muli(0.5));
        SDVariable xKV = sd.reshape(node, 1L, -1L, d);
        SDVariable qQ = sd.reshape(h0, 1L, 1L, d);
        SDVariable attn = sd.nn().dotProductAttentionV2(qQ, xKV, xKV, null, null, 0.0, 0.0, false, false);
        SDVariable m = sd.reshape(attn, 1L, d);
        SDVariable xh = sd.concat(1, m, h0);
        SDVariable h = sd.math().tanh(sd.mmul(xh, W));
        SDVariable out = sd.concat(1, m, h);
        sd.sum("loss", out);
        sd.setLossVariables("loss");
        assertTrue(GradCheckUtil.checkGradients(sd, null), "attn-in-loop-T1 grad check failed");
    }

    /** (6) Attention -> mmul -> sum: NON-UNIFORM incoming gradient into the attention _bp (no loop/concat/reuse). */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void reproAttnNonUniformEps(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(6L);
        SameDiff sd = SameDiff.create();
        long d = 4, n = 5;
        SDVariable q = sd.var("q", Nd4j.randn(DataType.DOUBLE, 1, 1, d));
        SDVariable v = sd.var("v", Nd4j.randn(DataType.DOUBLE, 1, n, d));
        SDVariable k = sd.var("k", Nd4j.randn(DataType.DOUBLE, 1, n, d));
        SDVariable W2 = sd.var("W2", Nd4j.randn(DataType.DOUBLE, d, d));
        SDVariable attn = sd.nn().dotProductAttentionV2(q, v, k, null, null, 0.0, 0.0, false, false);
        SDVariable m = sd.reshape(attn, 1L, d);
        SDVariable y = sd.mmul(m, W2);     // non-uniform transform => non-uniform eps into attention bp
        sd.sum("loss", y);
        sd.setLossVariables("loss");
        assertTrue(GradCheckUtil.checkGradients(sd, null), "attn non-uniform-eps grad check failed");
    }

    /** (8) Attention output reused by TWO consumers -> grad into the attention output is accumulated, then bp'd. */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void reproAttnOutputReused(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(8L);
        SameDiff sd = SameDiff.create();
        long d = 4, n = 5;
        SDVariable q = sd.var("q", Nd4j.randn(DataType.DOUBLE, 1, 1, d));
        SDVariable v = sd.var("v", Nd4j.randn(DataType.DOUBLE, 1, n, d));
        SDVariable k = sd.var("k", Nd4j.randn(DataType.DOUBLE, 1, n, d));
        SDVariable W1 = sd.var("W1", Nd4j.randn(DataType.DOUBLE, d, d));
        SDVariable W2 = sd.var("W2", Nd4j.randn(DataType.DOUBLE, d, d));
        SDVariable attn = sd.nn().dotProductAttentionV2(q, v, k, null, null, 0.0, 0.0, false, false);
        SDVariable m = sd.reshape(attn, 1L, d);
        SDVariable o1 = sd.mmul(m, W1);   // consumer 1 of m
        SDVariable o2 = sd.mmul(m, W2);   // consumer 2 of m
        SDVariable out = o1.add(o2);
        sd.sum("loss", out);
        sd.setLossVariables("loss");
        assertTrue(GradCheckUtil.checkGradients(sd, null), "attn-output-reused grad check failed");
    }

    /** (11) Attention output reused via TWO CONCATs (concat-specific fan-out of an attention output). */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void reproAttnOutConcatReuse(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(11L);
        SameDiff sd = SameDiff.create();
        long d = 4, n = 5;
        SDVariable q = sd.var("q", Nd4j.randn(DataType.DOUBLE, 1, 1, d));
        SDVariable v = sd.var("v", Nd4j.randn(DataType.DOUBLE, 1, n, d));
        SDVariable k = sd.var("k", Nd4j.randn(DataType.DOUBLE, 1, n, d));
        SDVariable a = sd.var("a", Nd4j.randn(DataType.DOUBLE, 1, d));
        SDVariable b = sd.var("b", Nd4j.randn(DataType.DOUBLE, 1, d));
        SDVariable attn = sd.nn().dotProductAttentionV2(q, v, k, null, null, 0.0, 0.0, false, false);
        SDVariable m = sd.reshape(attn, 1L, d);
        SDVariable c1 = sd.concat(1, m, a);
        SDVariable c2 = sd.concat(1, m, b);
        SDVariable out = c1.add(c2);
        sd.sum("loss", out);
        sd.setLossVariables("loss");
        assertTrue(GradCheckUtil.checkGradients(sd, null), "attn-out-concat-reuse grad check failed");
    }

    /** (12) The attention QUERY variable is ALSO used downstream (h0 = query AND concat(m,h0)). */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void reproQueryVarReused(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(12L);
        SameDiff sd = SameDiff.create();
        long d = 4, n = 5;
        SDVariable h0 = sd.var("h", Nd4j.randn(DataType.DOUBLE, 1, d));
        SDVariable node = sd.var("node", Nd4j.randn(DataType.DOUBLE, n, d));
        SDVariable xKV = sd.reshape(node, 1L, -1L, d);
        SDVariable qQ = sd.reshape(h0, 1L, 1L, d);                       // h0 -> attention query
        SDVariable attn = sd.nn().dotProductAttentionV2(qQ, xKV, xKV, null, null, 0.0, 0.0, false, false);
        SDVariable m = sd.reshape(attn, 1L, d);
        SDVariable out = sd.concat(1, m, h0);                            // h0 reused here
        sd.sum("loss", out);
        sd.setLossVariables("loss");
        assertTrue(GradCheckUtil.checkGradients(sd, null), "query-var-reused grad check failed");
    }
}
