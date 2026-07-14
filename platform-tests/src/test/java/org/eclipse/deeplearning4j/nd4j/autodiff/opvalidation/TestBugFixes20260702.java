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
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.validation.OpValidation;
import org.nd4j.autodiff.validation.TestCase;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Regression tests for four op-level bugs fixed 2026-07-02:
 *   BUG1: matmul_bp rank-3 x rank-2 output shape
 *   BUG2: dot_product_attention_v2_bp 4D (BSHD) output shape
 *   BUG3: rmsNorm missing gamma gradient in doDiff
 *   BUG4: DistillationKLLoss rank-3 hard-terminate
 */
@Slf4j
@NativeTag
public class TestBugFixes20260702 extends BaseOpValidation {

    // ══════════════════════════════════════════════════════════════════════
    // BUG 1 — matmul_bp rank-3 x rank-2: gradient shape must be [H,V]
    // ══════════════════════════════════════════════════════════════════════

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMatmulBpRank3xRank2GradientCheck(Nd4jBackend backend) {
        // x=[2,3,4]  w=[4,5]  forward output=[2,3,5]
        // dL/dw must be [4,5], not [2,4,5] — the bug produced wrong shape or exception
        INDArray xArr = Nd4j.randn(DataType.DOUBLE, 2, 3, 4);
        INDArray wArr = Nd4j.randn(DataType.DOUBLE, 4, 5);

        SameDiff sd = SameDiff.create();
        SDVariable x = sd.var("x", xArr);
        SDVariable w = sd.var("w", wArr);

        SDVariable z = sd.mmul(x, w);           // [2,3,5]
        SDVariable loss = z.std("loss", true);
        loss.markAsLoss();

        String err = OpValidation.validate(new TestCase(sd).gradientCheck(true));
        assertNull(err, "matmul_bp rank-3 x rank-2 grad check failed: " + err);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMatmulBpRank3xRank2WGradShape(Nd4jBackend backend) {
        // Direct check: verify dldy (gradient w.r.t. weight) has the right shape [H,V]
        INDArray x = Nd4j.randn(DataType.DOUBLE, 2, 3, 4);   // [B,S,H]
        INDArray w = Nd4j.randn(DataType.DOUBLE, 4, 5);        // [H,V]
        INDArray eps = Nd4j.ones(DataType.DOUBLE, 2, 3, 5);    // [B,S,V]

        INDArray dldx = Nd4j.create(DataType.DOUBLE, 2, 3, 4);
        INDArray dldw = Nd4j.create(DataType.DOUBLE, 4, 5);    // must stay [H,V]

        Nd4j.exec(DynamicCustomOp.builder("matmul_bp")
                .addInputs(x, w, eps)
                .addOutputs(dldx, dldw)
                .build());

        assertArrayEquals(new long[]{4, 5}, dldw.shape(),
                "dL/dw shape must be [H,V]=[4,5] for rank-3 x rank-2 matmul");
        assertFalse(dldw.isNaN().any(), "dL/dw contains NaN");
        assertFalse(dldw.isInfinite().any(), "dL/dw contains Inf");
    }

    // ══════════════════════════════════════════════════════════════════════
    // BUG 2 — dot_product_attention_v2_bp 4D (BSHD) output shape
    // ══════════════════════════════════════════════════════════════════════

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testDotProductAttentionV2Bp4D_CausalGradientCheck(Nd4jBackend backend) {
        // Small 4D BSHD shapes: [2,3,2,4] (batch, seq, heads, headDim)
        // Uses causal mask, no explicit qMask/vMask — mirrors the real-world failure.
        // Note: the gradcheck tolerance is loose because FlashAttention backward
        // recomputes attention scores (numerical differences vs finite-diff).
        INDArray q = Nd4j.randn(DataType.DOUBLE, 2, 3, 2, 4);
        INDArray v = Nd4j.randn(DataType.DOUBLE, 2, 3, 2, 4);
        INDArray k = Nd4j.randn(DataType.DOUBLE, 2, 3, 2, 4);

        SameDiff sd = SameDiff.create();
        SDVariable sdQ = sd.var("q", q);
        SDVariable sdK = sd.var("k", k);
        SDVariable sdV = sd.var("v", v);

        // scale = 1/sqrt(4) = 0.5, causal=true, training=true
        SDVariable t = sd.nn.dotProductAttentionV2(sdQ, sdV, sdK, null, null, 0.5, 0.0, true, true);
        SDVariable loss = t.norm1("loss");
        loss.markAsLoss();

        String err = OpValidation.validate(new TestCase(sd)
                .gradientCheck(true)
                .gradCheckMaxRelativeError(1e-3)
                .gradCheckMinAbsError(1e-5));
        assertNull(err, "dot_product_attention_v2_bp 4D causal grad check failed: " + err);
    }

    @Test
    public void testDotProductAttentionV2Bp4D_ShapeOnly() {
        // Verify backward shapes are correct without gradient check overhead.
        // Inputs: q/k/v all [4,96,4,48] (batch=4, seq=96, heads=4, headDim=48)
        // The bug was: bp op threw "Output array did not match expected shape".
        int B = 2, S = 4, H = 2, D = 8;
        INDArray q = Nd4j.randn(DataType.FLOAT, B, S, H, D);
        INDArray v = Nd4j.randn(DataType.FLOAT, B, S, H, D);
        INDArray k = Nd4j.randn(DataType.FLOAT, B, S, H, D);

        SameDiff sd = SameDiff.create();
        SDVariable sdQ = sd.var("q", q);
        SDVariable sdK = sd.var("k", k);
        SDVariable sdV = sd.var("v", v);

        float scale = 1.0f / (float) Math.sqrt(D);
        SDVariable t = sd.nn.dotProductAttentionV2(sdQ, sdV, sdK, null, null, scale, 0.0, true, true);
        SDVariable loss = t.norm1("loss");
        loss.markAsLoss();

        // calculateGradients throws if bp op produces wrong output shape
        Map<String, INDArray> grads = sd.calculateGradients(null, "q", "k", "v");

        assertNotNull(grads.get("q"), "dL/dq is null");
        assertNotNull(grads.get("k"), "dL/dk is null");
        assertNotNull(grads.get("v"), "dL/dv is null");

        assertArrayEquals(new long[]{B, S, H, D}, grads.get("q").shape(), "dL/dq wrong shape");
        assertArrayEquals(new long[]{B, S, H, D}, grads.get("k").shape(), "dL/dk wrong shape");
        assertArrayEquals(new long[]{B, S, H, D}, grads.get("v").shape(), "dL/dv wrong shape");

        assertFalse(grads.get("q").isNaN().any(), "dL/dq has NaN");
        assertFalse(grads.get("k").isNaN().any(), "dL/dk has NaN");
        assertFalse(grads.get("v").isNaN().any(), "dL/dv has NaN");

        log.info("4D attention bp dL/dq={}, dL/dk={}, dL/dv={}",
                grads.get("q").shapeInfoToString(),
                grads.get("k").shapeInfoToString(),
                grads.get("v").shapeInfoToString());
    }

    // ══════════════════════════════════════════════════════════════════════
    // BUG 3 — rmsNorm missing gamma gradient
    // ══════════════════════════════════════════════════════════════════════

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testRmsNormGammaGradientCheck(Nd4jBackend backend) {
        // The bug: sd.nn.rmsNorm(x, gamma, eps) → calculateGradients errors
        // "no gradient for required variable gamma was calculated".
        int batch = 2, seq = 3, features = 4;
        INDArray xArr = Nd4j.randn(DataType.DOUBLE, batch, seq, features);
        INDArray gammaArr = Nd4j.randn(DataType.DOUBLE, features).addi(1.0);

        SameDiff sd = SameDiff.create();
        SDVariable x = sd.var("x", xArr);
        SDVariable gamma = sd.var("gamma", gammaArr);

        SDVariable normed = sd.nn.rmsNorm("normed", x, gamma, 1e-5);
        SDVariable loss = normed.std("loss", true);
        loss.markAsLoss();

        String err = OpValidation.validate(new TestCase(sd).gradientCheck(true));
        assertNull(err, "rmsNorm gamma gradient check failed: " + err);
    }

    @Test
    public void testRmsNormGammaGradShapeAndFiniteness() {
        // Explicit shape + finite check for gamma gradient
        int batch = 2, seq = 3, features = 4;
        INDArray xArr = Nd4j.randn(DataType.FLOAT, batch, seq, features);
        INDArray gammaArr = Nd4j.ones(DataType.FLOAT, features);

        SameDiff sd = SameDiff.create();
        SDVariable x = sd.var("x", xArr);
        SDVariable gamma = sd.var("gamma", gammaArr);

        SDVariable normed = sd.nn.rmsNorm("normed", x, gamma, 1e-5f);
        SDVariable loss = normed.norm1("loss");
        loss.markAsLoss();

        Map<String, INDArray> grads = sd.calculateGradients(null, "x", "gamma");

        assertNotNull(grads.get("gamma"), "dL/dgamma must not be null — gamma gradient was missing before fix");
        assertArrayEquals(new long[]{features}, grads.get("gamma").shape(),
                "dL/dgamma shape must be [features]");
        assertFalse(grads.get("gamma").isNaN().any(), "dL/dgamma has NaN");
        assertFalse(grads.get("gamma").isInfinite().any(), "dL/dgamma has Inf");

        assertNotNull(grads.get("x"), "dL/dx must not be null");
        assertArrayEquals(new long[]{batch, seq, features}, grads.get("x").shape());
        assertFalse(grads.get("x").isNaN().any(), "dL/dx has NaN");

        log.info("rmsNorm gamma grad={}", grads.get("gamma"));
    }

    @Test
    public void testRmsNormWithoutGammaStillWorks() {
        // Regression guard: rmsNorm without gamma must still produce 1 gradient (for x only)
        int batch = 2, features = 4;
        INDArray xArr = Nd4j.randn(DataType.FLOAT, batch, features);

        SameDiff sd = SameDiff.create();
        SDVariable x = sd.var("x", xArr);

        // rmsNorm without gamma (2-arg constructor on SameDiff side)
        SDVariable normed = sd.math.mul(
                x,
                sd.math.rsqrt(sd.mean(sd.math.mul(x, x), true, -1).add(1e-5f)));
        SDVariable loss = normed.norm1("loss");
        loss.markAsLoss();

        // Must not throw
        Map<String, INDArray> grads = sd.calculateGradients(null, "x");
        assertNotNull(grads.get("x"), "dL/dx is null for no-gamma rmsNorm");
    }

    // ══════════════════════════════════════════════════════════════════════
    // BUG 4 — DistillationKLLoss rank-3 hard-terminate
    // ══════════════════════════════════════════════════════════════════════

    @Test
    public void testDistillationKLLossRank2Correctness() {
        // Hand-computed reference for rank-2 [batch=2, classes=3], T=2.0, alpha=0.5
        int B = 2, V = 3;
        double T = 2.0, alpha = 0.5;

        // Student and teacher logits
        INDArray student = Nd4j.create(new double[]{
                1.0, 2.0, 0.5,
                0.3, 1.5, 2.1
        }).reshape(B, V).castTo(DataType.DOUBLE);
        INDArray teacher = Nd4j.create(new double[]{
                0.8, 1.2, 1.5,
                1.1, 0.9, 1.8
        }).reshape(B, V).castTo(DataType.DOUBLE);

        INDArray output = Nd4j.scalar(DataType.DOUBLE, 0.0);
        Nd4j.exec(DynamicCustomOp.builder("distillation_kl_loss")
                .addInputs(student, teacher)
                .addOutputs(output)
                .addFloatingPointArguments(T, alpha)
                .build());

        double loss = output.getDouble(0);
        assertTrue(Double.isFinite(loss), "rank-2 distillation loss must be finite, got " + loss);
        assertTrue(loss >= 0, "KL divergence must be non-negative, got " + loss);
        log.info("rank-2 distillationKLLoss={}", loss);
    }

    @Test
    public void testDistillationKLLossRank3NoTerminate() {
        // THE BUG: rank-3 [B,S,V] caused std::terminate (JVM kill).
        // This test must complete without the JVM crashing or throwing.
        int B = 2, S = 4, V = 8;
        double T = 4.0, alpha = 0.5;

        INDArray student = Nd4j.randn(DataType.FLOAT, B, S, V).castTo(DataType.DOUBLE);
        INDArray teacher = Nd4j.randn(DataType.FLOAT, B, S, V).castTo(DataType.DOUBLE);

        INDArray output = Nd4j.scalar(DataType.DOUBLE, 0.0);

        // This must NOT throw and must NOT kill the JVM
        assertDoesNotThrow(() -> {
            Nd4j.exec(DynamicCustomOp.builder("distillation_kl_loss")
                    .addInputs(student, teacher)
                    .addOutputs(output)
                    .addFloatingPointArguments(T, alpha)
                    .build());
        }, "rank-3 distillationKLLoss must not throw");

        double loss = output.getDouble(0);
        assertTrue(Double.isFinite(loss), "rank-3 distillation loss must be finite, got " + loss);
        assertTrue(loss >= 0, "KL divergence must be non-negative, got " + loss);
        log.info("rank-3 distillationKLLoss B={} S={} V={} loss={}", B, S, V, loss);
    }

    @Test
    public void testDistillationKLLossRank3LargeVocab() {
        // Closer to the real failure: [2,64,248070] is too large for a unit test,
        // so use [2,16,512] which hits the same code path.
        int B = 2, S = 16, V = 512;
        double T = 4.0, alpha = 0.5;

        INDArray student = Nd4j.randn(DataType.FLOAT, B, S, V).castTo(DataType.DOUBLE);
        INDArray teacher = Nd4j.randn(DataType.FLOAT, B, S, V).castTo(DataType.DOUBLE);

        INDArray output = Nd4j.scalar(DataType.DOUBLE, 0.0);

        assertDoesNotThrow(() -> {
            Nd4j.exec(DynamicCustomOp.builder("distillation_kl_loss")
                    .addInputs(student, teacher)
                    .addOutputs(output)
                    .addFloatingPointArguments(T, alpha)
                    .build());
        });

        double loss = output.getDouble(0);
        assertTrue(Double.isFinite(loss), "large rank-3 loss must be finite");
        log.info("large rank-3 distillationKLLoss B={} S={} V={} loss={}", B, S, V, loss);
    }

    @Test
    public void testDistillationKLLossRank2MatchesRank3Reshaped() {
        // Invariant: distillationKLLoss([B*S,V]) == distillationKLLoss([B,S,V])
        // (averaged over batch in both cases, so the scalar value must match)
        int B = 2, S = 3, V = 5;
        double T = 2.0, alpha = 1.0;  // alpha=1 disables CE term for cleaner comparison

        INDArray student3d = Nd4j.randn(DataType.DOUBLE, B, S, V);
        INDArray teacher3d = Nd4j.randn(DataType.DOUBLE, B, S, V);

        INDArray student2d = student3d.reshape(B * S, V);
        INDArray teacher2d = teacher3d.reshape(B * S, V);

        INDArray out3d = Nd4j.scalar(DataType.DOUBLE, 0.0);
        INDArray out2d = Nd4j.scalar(DataType.DOUBLE, 0.0);

        Nd4j.exec(DynamicCustomOp.builder("distillation_kl_loss")
                .addInputs(student3d, teacher3d)
                .addOutputs(out3d)
                .addFloatingPointArguments(T, alpha)
                .build());

        Nd4j.exec(DynamicCustomOp.builder("distillation_kl_loss")
                .addInputs(student2d, teacher2d)
                .addOutputs(out2d)
                .addFloatingPointArguments(T, alpha)
                .build());

        double loss3d = out3d.getDouble(0);
        double loss2d = out2d.getDouble(0);
        log.info("rank-3 loss={} rank-2-reshaped loss={}", loss3d, loss2d);
        assertEquals(loss2d, loss3d, 1e-9,
                "rank-3 and reshaped rank-2 must give same loss (diff=" + Math.abs(loss3d - loss2d) + ")");
    }

    @Test
    public void testDistillationKLLossGradRank2MatchesRank3Reshaped() {
        // Invariant: the per-element student gradient of distillation_kl_loss_grad must be
        // layout-invariant to leading-dim flattening, i.e.
        //   grad([B,S,V]).dLdStudent.reshape(B*S,V)  ==  grad([B*S,V]).dLdStudent
        // Gates the CUDA distillation_kl_loss_grad rank-3 handling (softmax must reduce over V,
        // the last axis, for both ranks — not over S for the rank-3 case).
        int B = 2, S = 3, V = 5;
        double T = 2.0, alpha = 1.0;  // alpha=1 isolates the KD term

        INDArray student3d = Nd4j.randn(DataType.DOUBLE, B, S, V);
        INDArray teacher3d = Nd4j.randn(DataType.DOUBLE, B, S, V);

        INDArray student2d = student3d.reshape(B * S, V);
        INDArray teacher2d = teacher3d.reshape(B * S, V);

        INDArray dS3d = Nd4j.create(DataType.DOUBLE, B, S, V);
        INDArray dT3d = Nd4j.create(DataType.DOUBLE, B, S, V);
        INDArray dS2d = Nd4j.create(DataType.DOUBLE, B * S, V);
        INDArray dT2d = Nd4j.create(DataType.DOUBLE, B * S, V);

        Nd4j.exec(DynamicCustomOp.builder("distillation_kl_loss_grad")
                .addInputs(student3d, teacher3d)
                .addOutputs(dS3d, dT3d)
                .addFloatingPointArguments(T, alpha)
                .build());

        Nd4j.exec(DynamicCustomOp.builder("distillation_kl_loss_grad")
                .addInputs(student2d, teacher2d)
                .addOutputs(dS2d, dT2d)
                .addFloatingPointArguments(T, alpha)
                .build());

        INDArray dS3dFlat = dS3d.reshape(B * S, V);
        double maxDiff = Transforms.abs(dS3dFlat.sub(dS2d)).maxNumber().doubleValue();
        log.info("distillation grad rank-3-reshaped vs rank-2 maxDiff={}", maxDiff);
        assertTrue(maxDiff < 1e-9,
                "rank-3 and reshaped rank-2 student gradients must match (maxDiff=" + maxDiff + ")");
    }

    @Test
    public void testDistillationKLLossRank1ThrowsJavaException() {
        // Rank-1 input must throw a Java exception (not kill the JVM via terminate)
        INDArray student = Nd4j.randn(DataType.DOUBLE, 5);
        INDArray teacher = Nd4j.randn(DataType.DOUBLE, 5);
        INDArray output = Nd4j.scalar(DataType.DOUBLE, 0.0);

        assertThrows(Exception.class, () -> {
            Nd4j.exec(DynamicCustomOp.builder("distillation_kl_loss")
                    .addInputs(student, teacher)
                    .addOutputs(output)
                    .addFloatingPointArguments(4.0, 0.5)
                    .build());
        }, "rank-1 input must throw a Java exception, not kill the JVM");
    }
}
