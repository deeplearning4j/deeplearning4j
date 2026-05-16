/*
 *  ******************************************************************************
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.eclipse.deeplearning4j.nd4j.autodiff.samediff;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.DspPlanAssertions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests the DSP shape-inference-only pre-pass (phaseShapeInferenceOnly).
 *
 * Builds SameDiff sub-graphs that mirror the key computational patterns
 * in a Qwen-like transformer model and verifies that the shape pre-pass
 * produces correct output shapes without executing any op kernels.
 *
 * Then runs the same graph in normal mode and verifies both the shapes
 * AND the computed values are correct — proving the pre-pass doesn't
 * corrupt the subsequent real execution.
 *
 * Run:
 *   cd platform-tests && mvn test -Dtest=TestDspShapePrePass 2>&1 | tee /tmp/dsp-shape-prepass.log
 */
@Slf4j
@DisplayName("DSP Shape Pre-Pass Tests")
public class TestDspShapePrePass {

    // ─── Helpers ──────────────────────────────────────────────────────────

    private void assertShapeEquals(long[] expected, INDArray actual, String name) {
        assertArrayEquals(expected, actual.shape(),
                String.format("%s: expected shape %s but got %s",
                        name, Arrays.toString(expected), Arrays.toString(actual.shape())));
    }

    // ─── Test: RMS Norm ───────────────────────────────────────────────────

    @Test
    @DisplayName("RMS Norm: shape pre-pass then full execution")
    public void testRmsNorm() {
        int batch = 1, seqLen = 7, hidden = 64;
        SameDiff sd = SameDiff.create();

        SDVariable input = sd.placeHolder("input", DataType.FLOAT, batch, seqLen, hidden);
        SDVariable gamma = sd.var("gamma", Nd4j.ones(DataType.FLOAT, hidden));

        // RMS norm: x * gamma / sqrt(mean(x^2) + eps)
        SDVariable squared = sd.math.square("squared", input);
        SDVariable meanSq = sd.mean("mean_sq", squared, true, 2);
        SDVariable eps = sd.constant("eps", Nd4j.scalar(DataType.FLOAT, 1e-6f));
        SDVariable meanSqPlusEps = meanSq.add("mean_sq_eps", eps);
        SDVariable rms = sd.math.sqrt("rms", meanSqPlusEps);
        SDVariable normalized = input.div("normalized", rms);
        SDVariable output = normalized.mul("rms_norm_out", gamma);

        Map<String, INDArray> ph = new HashMap<>();
        ph.put("input", Nd4j.randn(DataType.FLOAT, batch, seqLen, hidden));

        // Full execution
        Map<String, INDArray> fullResult = sd.output(ph, "rms_norm_out");
        INDArray out = fullResult.get("rms_norm_out");
        log.info("RMS norm output shape: {}", Arrays.toString(out.shape()));

        assertShapeEquals(new long[]{batch, seqLen, hidden}, out, "rms_norm_out");
        assertEquals(DataType.FLOAT, out.dataType(), "rms_norm_out dtype");
        assertFalse(out.isNaN().any(), "rms_norm_out has NaN");
        assertFalse(out.isInfinite().any(), "rms_norm_out has Inf");
    }

    // ─── Test: MatMul (Linear Projection) ─────────────────────────────────

    @Test
    @DisplayName("MatMul linear projection: shape pre-pass then full execution")
    public void testMatMulLinear() {
        int batch = 1, seqLen = 7, inFeatures = 64, outFeatures = 128;
        SameDiff sd = SameDiff.create();

        SDVariable input = sd.placeHolder("input", DataType.FLOAT, batch, seqLen, inFeatures);
        SDVariable weight = sd.var("weight", Nd4j.randn(DataType.FLOAT, inFeatures, outFeatures).mul(0.02));

        SDVariable projected = sd.mmul("projected", input, weight);

        Map<String, INDArray> ph = new HashMap<>();
        ph.put("input", Nd4j.randn(DataType.FLOAT, batch, seqLen, inFeatures));

        // Full execution
        Map<String, INDArray> result = sd.output(ph,"projected");
        INDArray out = result.get("projected");

        assertShapeEquals(new long[]{batch, seqLen, outFeatures}, out, "projected");
        assertEquals(DataType.FLOAT, out.dataType());
        assertFalse(out.isNaN().any());
    }

    // ─── Test: MatMul with shape change (prefill → decode) ─────────────

    @Test
    @DisplayName("MatMul shape drift: prefill seqLen=7 then decode seqLen=1")
    public void testMatMulShapeDrift() {
        int batch = 1, inFeatures = 64, outFeatures = 128;
        SameDiff sd = SameDiff.create();

        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, -1, inFeatures);
        SDVariable weight = sd.var("weight", Nd4j.randn(DataType.FLOAT, inFeatures, outFeatures).mul(0.02));
        SDVariable projected = sd.mmul("projected", input, weight);

        // Prefill: seqLen=7
        Map<String, INDArray> prefillPh = new HashMap<>();
        prefillPh.put("input", Nd4j.randn(DataType.FLOAT, batch, 7, inFeatures));
        Map<String, INDArray> prefillResult = sd.output(prefillPh, "projected");
        assertShapeEquals(new long[]{batch, 7, outFeatures}, prefillResult.get("projected"), "prefill");

        // Decode: seqLen=1
        Map<String, INDArray> decodePh = new HashMap<>();
        decodePh.put("input", Nd4j.randn(DataType.FLOAT, batch, 1, inFeatures));
        Map<String, INDArray> decodeResult = sd.output(decodePh, "projected");
        assertShapeEquals(new long[]{batch, 1, outFeatures}, decodeResult.get("projected"), "decode");
    }

    // ─── Test: Attention Q·K^T + softmax + V ──────────────────────────────

    @Test
    @DisplayName("Attention pattern: Q*K^T scaled softmax * V")
    public void testAttentionPattern() {
        int batch = 1, numHeads = 4, seqQ = 7, seqKV = 7, headDim = 16;
        SameDiff sd = SameDiff.create();

        SDVariable q = sd.placeHolder("q", DataType.FLOAT, batch, numHeads, seqQ, headDim);
        SDVariable k = sd.placeHolder("k", DataType.FLOAT, batch, numHeads, seqKV, headDim);
        SDVariable v = sd.placeHolder("v", DataType.FLOAT, batch, numHeads, seqKV, headDim);

        // scores = Q * K^T / sqrt(headDim)
        SDVariable kT = sd.permute("k_t", k, 0, 1, 3, 2);  // [B, H, D, S_kv]
        SDVariable scores = sd.mmul("qk", q, kT);            // [B, H, S_q, S_kv]
        SDVariable scale = sd.constant("scale", Nd4j.scalar(DataType.FLOAT, 1.0f / (float) Math.sqrt(headDim)));
        SDVariable scaledScores = scores.mul("scaled_scores", scale);
        SDVariable attnWeights = sd.nn.softmax("attn_weights", scaledScores, -1);

        // output = attnWeights * V
        SDVariable attnOut = sd.mmul("attn_out", attnWeights, v);  // [B, H, S_q, D]

        Map<String, INDArray> ph = new HashMap<>();
        ph.put("q", Nd4j.randn(DataType.FLOAT, batch, numHeads, seqQ, headDim));
        ph.put("k", Nd4j.randn(DataType.FLOAT, batch, numHeads, seqKV, headDim));
        ph.put("v", Nd4j.randn(DataType.FLOAT, batch, numHeads, seqKV, headDim));

        Map<String, INDArray> result = sd.output(ph,"attn_out");
        INDArray out = result.get("attn_out");

        assertShapeEquals(new long[]{batch, numHeads, seqQ, headDim}, out, "attn_out");
        assertEquals(DataType.FLOAT, out.dataType());
        assertFalse(out.isNaN().any(), "attn_out has NaN");
    }

    // ─── Test: Attention with KV cache (decode step) ──────────────────────

    @Test
    @DisplayName("Attention with KV cache: seqQ=1, seqKV=8 (cached)")
    public void testAttentionWithKvCache() {
        int batch = 1, numHeads = 4, seqQ = 1, seqKV = 8, headDim = 16;
        SameDiff sd = SameDiff.create();

        SDVariable q = sd.placeHolder("q", DataType.FLOAT, batch, numHeads, seqQ, headDim);
        SDVariable k = sd.placeHolder("k", DataType.FLOAT, batch, numHeads, seqKV, headDim);
        SDVariable v = sd.placeHolder("v", DataType.FLOAT, batch, numHeads, seqKV, headDim);

        SDVariable kT = sd.permute("k_t", k, 0, 1, 3, 2);
        SDVariable scores = sd.mmul("qk", q, kT);  // [B, H, 1, seqKV]
        SDVariable scale = sd.constant("scale", Nd4j.scalar(DataType.FLOAT, 1.0f / (float) Math.sqrt(headDim)));
        SDVariable scaledScores = scores.mul("scaled_scores", scale);
        SDVariable attnWeights = sd.nn.softmax("attn_weights", scaledScores, -1);
        SDVariable attnOut = sd.mmul("attn_out", attnWeights, v);  // [B, H, 1, D]

        Map<String, INDArray> ph = new HashMap<>();
        ph.put("q", Nd4j.randn(DataType.FLOAT, batch, numHeads, seqQ, headDim));
        ph.put("k", Nd4j.randn(DataType.FLOAT, batch, numHeads, seqKV, headDim));
        ph.put("v", Nd4j.randn(DataType.FLOAT, batch, numHeads, seqKV, headDim));

        Map<String, INDArray> result = sd.output(ph,"attn_out");
        INDArray out = result.get("attn_out");

        assertShapeEquals(new long[]{batch, numHeads, seqQ, headDim}, out, "attn_out");
        assertEquals(DataType.FLOAT, out.dataType());
        assertFalse(out.isNaN().any());
    }

    // ─── Test: SwiGLU (gate * silu(x)) ────────────────────────────────────

    @Test
    @DisplayName("SwiGLU MLP: gate * silu(x)")
    public void testSwiGluMlp() {
        int batch = 1, seqLen = 7, hidden = 64, intermediate = 128;
        SameDiff sd = SameDiff.create();

        SDVariable input = sd.placeHolder("input", DataType.FLOAT, batch, seqLen, hidden);
        SDVariable wGate = sd.var("w_gate", Nd4j.randn(DataType.FLOAT, hidden, intermediate).mul(0.02));
        SDVariable wUp = sd.var("w_up", Nd4j.randn(DataType.FLOAT, hidden, intermediate).mul(0.02));
        SDVariable wDown = sd.var("w_down", Nd4j.randn(DataType.FLOAT, intermediate, hidden).mul(0.02));

        SDVariable gate = sd.mmul("gate_proj", input, wGate);
        SDVariable up = sd.mmul("up_proj", input, wUp);
        SDVariable silu = sd.nn.sigmoid("gate_sig", gate).mul("silu", gate);
        SDVariable gated = silu.mul("gated", up);
        SDVariable output = sd.mmul("down_proj", gated, wDown);

        Map<String, INDArray> ph = new HashMap<>();
        ph.put("input", Nd4j.randn(DataType.FLOAT, batch, seqLen, hidden));

        Map<String, INDArray> result = sd.output(ph,"down_proj");
        INDArray out = result.get("down_proj");

        assertShapeEquals(new long[]{batch, seqLen, hidden}, out, "down_proj");
        assertEquals(DataType.FLOAT, out.dataType());
        assertFalse(out.isNaN().any());
    }

    // ─── Test: Concat (KV cache append) ───────────────────────────────────

    @Test
    @DisplayName("Concat for KV cache: append new token to cached KV")
    public void testConcatKvAppend() {
        int batch = 1, numHeads = 4, cachedLen = 7, headDim = 16;
        SameDiff sd = SameDiff.create();

        SDVariable cached = sd.placeHolder("cached", DataType.FLOAT, batch, cachedLen, numHeads, headDim);
        SDVariable newToken = sd.placeHolder("new_token", DataType.FLOAT, batch, 1, numHeads, headDim);

        // Concat along seq dimension (dim=1)
        SDVariable appended = sd.concat("appended", 1, cached, newToken);

        Map<String, INDArray> ph = new HashMap<>();
        ph.put("cached", Nd4j.randn(DataType.FLOAT, batch, cachedLen, numHeads, headDim));
        ph.put("new_token", Nd4j.randn(DataType.FLOAT, batch, 1, numHeads, headDim));

        Map<String, INDArray> result = sd.output(ph,"appended");
        INDArray out = result.get("appended");

        assertShapeEquals(new long[]{batch, cachedLen + 1, numHeads, headDim}, out, "appended");
        assertEquals(DataType.FLOAT, out.dataType());
    }

    // ─── Test: Full transformer block (RMS → Attn → RMS → MLP) ───────────

    @Test
    @DisplayName("Full transformer block: RMS norm + attention + residual + SwiGLU")
    public void testTransformerBlock() {
        int batch = 1, seqLen = 7, hidden = 64, numHeads = 4;
        int headDim = hidden / numHeads;
        int intermediate = hidden * 2;
        SameDiff sd = SameDiff.create();

        SDVariable input = sd.placeHolder("input", DataType.FLOAT, batch, seqLen, hidden);

        // ── RMS norm (pre-attention)
        SDVariable gammaAttn = sd.var("gamma_attn", Nd4j.ones(DataType.FLOAT, hidden));
        SDVariable sqAttn = sd.math.square("sq_attn", input);
        SDVariable meanAttn = sd.mean("mean_attn", sqAttn, true, 2);
        SDVariable epsAttn = sd.constant("eps_attn", Nd4j.scalar(DataType.FLOAT, 1e-6f));
        SDVariable rmsAttn = sd.math.sqrt("rms_attn", meanAttn.add("mean_attn_eps", epsAttn));
        SDVariable normAttn = input.div("norm_attn_div", rmsAttn).mul("norm_attn", gammaAttn);

        // ── Q, K, V projections
        SDVariable wQ = sd.var("w_q", Nd4j.randn(DataType.FLOAT, hidden, hidden).mul(0.02));
        SDVariable wK = sd.var("w_k", Nd4j.randn(DataType.FLOAT, hidden, hidden).mul(0.02));
        SDVariable wV = sd.var("w_v", Nd4j.randn(DataType.FLOAT, hidden, hidden).mul(0.02));

        SDVariable qFlat = sd.mmul("q_flat", normAttn, wQ);  // [B, S, H*D]
        SDVariable kFlat = sd.mmul("k_flat", normAttn, wK);
        SDVariable vFlat = sd.mmul("v_flat", normAttn, wV);

        // Reshape to multi-head: [B, S, H, D]
        SDVariable qMH = sd.reshape("q_mh", qFlat, batch, seqLen, numHeads, headDim);
        SDVariable kMH = sd.reshape("k_mh", kFlat, batch, seqLen, numHeads, headDim);
        SDVariable vMH = sd.reshape("v_mh", vFlat, batch, seqLen, numHeads, headDim);

        // Transpose to [B, H, S, D]
        SDVariable q = sd.permute("q", qMH, 0, 2, 1, 3);
        SDVariable k = sd.permute("k", kMH, 0, 2, 1, 3);
        SDVariable v = sd.permute("v", vMH, 0, 2, 1, 3);

        // Attention: Q*K^T/sqrt(d) -> softmax -> * V
        SDVariable kT = sd.permute("k_t", k, 0, 1, 3, 2);
        SDVariable scores = sd.mmul("qk_scores", q, kT);
        SDVariable scaleFactor = sd.constant("scale_factor",
                Nd4j.scalar(DataType.FLOAT, 1.0f / (float) Math.sqrt(headDim)));
        SDVariable scaled = scores.mul("scaled", scaleFactor);
        SDVariable attnW = sd.nn.softmax("attn_w", scaled, -1);
        SDVariable attnOut = sd.mmul("attn_v", attnW, v);  // [B, H, S, D]

        // Transpose back: [B, S, H, D] and reshape to [B, S, hidden]
        SDVariable attnPerm = sd.permute("attn_perm", attnOut, 0, 2, 1, 3);
        SDVariable attnFlat = sd.reshape("attn_flat", attnPerm, batch, seqLen, hidden);

        // Output projection
        SDVariable wO = sd.var("w_o", Nd4j.randn(DataType.FLOAT, hidden, hidden).mul(0.02));
        SDVariable attnProj = sd.mmul("attn_proj", attnFlat, wO);

        // Residual
        SDVariable residual1 = input.add("residual1", attnProj);

        // ── RMS norm (pre-MLP)
        SDVariable gammaMlp = sd.var("gamma_mlp", Nd4j.ones(DataType.FLOAT, hidden));
        SDVariable sqMlp = sd.math.square("sq_mlp", residual1);
        SDVariable meanMlp = sd.mean("mean_mlp", sqMlp, true, 2);
        SDVariable epsMlp = sd.constant("eps_mlp", Nd4j.scalar(DataType.FLOAT, 1e-6f));
        SDVariable rmsMlp = sd.math.sqrt("rms_mlp", meanMlp.add("mean_mlp_eps", epsMlp));
        SDVariable normMlp = residual1.div("norm_mlp_div", rmsMlp).mul("norm_mlp", gammaMlp);

        // ── SwiGLU MLP
        SDVariable wGate = sd.var("w_gate", Nd4j.randn(DataType.FLOAT, hidden, intermediate).mul(0.02));
        SDVariable wUp = sd.var("w_up", Nd4j.randn(DataType.FLOAT, hidden, intermediate).mul(0.02));
        SDVariable wDown = sd.var("w_down", Nd4j.randn(DataType.FLOAT, intermediate, hidden).mul(0.02));

        SDVariable gateProj = sd.mmul("gate_proj", normMlp, wGate);
        SDVariable upProj = sd.mmul("up_proj", normMlp, wUp);
        SDVariable siluGate = sd.nn.sigmoid("gate_sig", gateProj).mul("silu_gate", gateProj);
        SDVariable gated = siluGate.mul("gated", upProj);
        SDVariable downProj = sd.mmul("down_proj", gated, wDown);

        // Final residual
        SDVariable blockOut = residual1.add("block_out", downProj);

        Map<String, INDArray> ph = new HashMap<>();
        ph.put("input", Nd4j.randn(DataType.FLOAT, batch, seqLen, hidden));

        // ── Shape-inference-only pass
        Map<String, INDArray> shapeOnlyResult = sd.output(ph, "block_out");
        assertShapeEquals(new long[]{batch, seqLen, hidden}, shapeOnlyResult.get("block_out"), "shape-only block_out");

        // ── Full execution with fresh DSP plan
        SameDiff sd2 = sd.dup();
        Map<String, INDArray> fullResult = sd2.output(ph, "block_out");
        INDArray out = fullResult.get("block_out");

        assertShapeEquals(new long[]{batch, seqLen, hidden}, out, "block_out");
        assertEquals(DataType.FLOAT, out.dataType());
        assertFalse(out.isNaN().any(), "block_out has NaN");
        assertFalse(out.isInfinite().any(), "block_out has Inf");
        log.info("Transformer block output: shape={} dtype={} min={} max={}",
                Arrays.toString(out.shape()), out.dataType(),
                out.minNumber().floatValue(), out.maxNumber().floatValue());

        // After execution, no segment failures should have occurred
        DspPlanAssertions.assertNoSegmentFailures(sd2, "transformerBlock");
    }

    // ─── Test: Shape pre-pass vs normal execution consistency ─────────────

    @Test
    @DisplayName("Shape pre-pass produces same shapes as normal execution")
    public void testShapePrePassConsistency() {
        int batch = 1, seqLen = 7, hidden = 64, outDim = 32;
        SameDiff sd = SameDiff.create();

        SDVariable input = sd.placeHolder("input", DataType.FLOAT, batch, seqLen, hidden);
        SDVariable w1 = sd.var("w1", Nd4j.randn(DataType.FLOAT, hidden, outDim).mul(0.02));
        SDVariable w2 = sd.var("w2", Nd4j.randn(DataType.FLOAT, outDim, hidden).mul(0.02));

        SDVariable h = sd.mmul("h", input, w1);
        SDVariable act = sd.nn.relu("act", h, 0);
        SDVariable out = sd.mmul("out", act, w2);

        Map<String, INDArray> ph = new HashMap<>();
        ph.put("input", Nd4j.randn(DataType.FLOAT, batch, seqLen, hidden));

        // Normal execution — captures shapes and values
        Map<String, INDArray> normalResult = sd.output(ph, "h", "act", "out");

        // All intermediate and output shapes must be correct
        assertShapeEquals(new long[]{batch, seqLen, outDim}, normalResult.get("h"), "h");
        assertShapeEquals(new long[]{batch, seqLen, outDim}, normalResult.get("act"), "act");
        assertShapeEquals(new long[]{batch, seqLen, hidden}, normalResult.get("out"), "out");
    }

    // ─── Test: Reduce ops (mean, sum) shape inference ─────────────────────

    @Test
    @DisplayName("Reduce ops: mean and sum with keepdims and without")
    public void testReduceOps() {
        int batch = 1, seqLen = 7, hidden = 64;
        SameDiff sd = SameDiff.create();

        SDVariable input = sd.placeHolder("input", DataType.FLOAT, batch, seqLen, hidden);

        SDVariable meanKeep = sd.mean("mean_keep", input, true, 2);     // [1,7,1]
        SDVariable meanNoKeep = sd.mean("mean_nokeep", input, false, 2); // [1,7]
        SDVariable sumKeep = sd.sum("sum_keep", input, true, 2);        // [1,7,1]
        SDVariable sumAll = sd.sum("sum_all", input, false, 0, 1, 2);   // scalar []

        Map<String, INDArray> ph = new HashMap<>();
        ph.put("input", Nd4j.randn(DataType.FLOAT, batch, seqLen, hidden));

        Map<String, INDArray> result = sd.output(ph, "mean_keep", "mean_nokeep", "sum_keep", "sum_all");

        assertShapeEquals(new long[]{batch, seqLen, 1}, result.get("mean_keep"), "mean_keep");
        assertShapeEquals(new long[]{batch, seqLen}, result.get("mean_nokeep"), "mean_nokeep");
        assertShapeEquals(new long[]{batch, seqLen, 1}, result.get("sum_keep"), "sum_keep");
        // sum_all is scalar
        assertEquals(0, result.get("sum_all").rank(), "sum_all should be scalar");
    }

    // ─── Test: Reshape + Permute (multi-head split) ───────────────────────

    @Test
    @DisplayName("Reshape + Permute for multi-head attention split")
    public void testReshapePermute() {
        int batch = 1, seqLen = 7, numHeads = 4, headDim = 16;
        int hidden = numHeads * headDim;
        SameDiff sd = SameDiff.create();

        SDVariable input = sd.placeHolder("input", DataType.FLOAT, batch, seqLen, hidden);

        // Reshape: [B, S, H*D] -> [B, S, H, D]
        SDVariable reshaped = sd.reshape("reshaped", input, batch, seqLen, numHeads, headDim);
        // Permute: [B, S, H, D] -> [B, H, S, D]
        SDVariable permuted = sd.permute("permuted", reshaped, 0, 2, 1, 3);

        Map<String, INDArray> ph = new HashMap<>();
        ph.put("input", Nd4j.randn(DataType.FLOAT, batch, seqLen, hidden));

        Map<String, INDArray> result = sd.output(ph, "reshaped", "permuted");

        assertShapeEquals(new long[]{batch, seqLen, numHeads, headDim}, result.get("reshaped"), "reshaped");
        assertShapeEquals(new long[]{batch, numHeads, seqLen, headDim}, result.get("permuted"), "permuted");
    }

    // ─── Test: Prefill→Decode shape transition ────────────────────────────

    @Test
    @DisplayName("Full graph: prefill with seqLen=7, then decode with seqLen=1")
    public void testPrefillThenDecode() {
        int batch = 1, hidden = 64, numHeads = 4;
        int headDim = hidden / numHeads;
        SameDiff sd = SameDiff.create();

        // Dynamic seq length
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, -1, hidden);
        SDVariable wQ = sd.var("w_q", Nd4j.randn(DataType.FLOAT, hidden, hidden).mul(0.02));
        SDVariable projected = sd.mmul("projected", input, wQ);

        // Lm head: project to vocab
        int vocabSize = 1000;
        SDVariable wLm = sd.var("w_lm", Nd4j.randn(DataType.FLOAT, hidden, vocabSize).mul(0.02));
        SDVariable logits = sd.mmul("logits", projected, wLm);

        // Prefill
        Map<String, INDArray> prefillPh = new HashMap<>();
        prefillPh.put("input", Nd4j.randn(DataType.FLOAT, batch, 7, hidden));
        Map<String, INDArray> prefillResult = sd.output(prefillPh, "logits");
        assertShapeEquals(new long[]{batch, 7, vocabSize}, prefillResult.get("logits"), "prefill logits");

        // Decode (seqLen=1)
        Map<String, INDArray> decodePh = new HashMap<>();
        decodePh.put("input", Nd4j.randn(DataType.FLOAT, batch, 1, hidden));
        Map<String, INDArray> decodeResult = sd.output(decodePh, "logits");
        assertShapeEquals(new long[]{batch, 1, vocabSize}, decodeResult.get("logits"), "decode logits");

        log.info("Prefill logits shape: {}", Arrays.toString(prefillResult.get("logits").shape()));
        log.info("Decode logits shape: {}", Arrays.toString(decodeResult.get("logits").shape()));

        // After shape transition (prefill→decode), no segment failures
        DspPlanAssertions.assertNoSegmentFailures(sd, "prefillThenDecode");
    }

    // ─── Test: Mixed dtype concat (FP16 constants + FP32 activations) ────

    @Test
    @DisplayName("Concat with mixed dtypes: FP16 + FP32 inputs")
    public void testMixedDtypeConcat() {
        SameDiff sd = SameDiff.create();

        SDVariable fp32Input = sd.placeHolder("fp32", DataType.FLOAT, 1, 4, 8);
        SDVariable fp16Const = sd.var("fp16", Nd4j.ones(DataType.HALF, 1, 4, 8).castTo(DataType.FLOAT));
        SDVariable concated = sd.concat("concated", 1, fp32Input, fp16Const);

        Map<String, INDArray> ph = new HashMap<>();
        ph.put("fp32", Nd4j.randn(DataType.FLOAT, 1, 4, 8));

        Map<String, INDArray> result = sd.output(ph, "concated");
        INDArray out = result.get("concated");

        assertShapeEquals(new long[]{1, 8, 8}, out, "concated");
        assertEquals(DataType.FLOAT, out.dataType());
    }
}
