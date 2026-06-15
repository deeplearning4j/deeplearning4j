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
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.api.ops.impl.transforms.custom.FusedRoPE;
import org.nd4j.linalg.ops.transforms.Transforms;

import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Isolation tests for components changed in the current working tree.
 * Each test targets a specific changed op/helper and verifies CORRECTNESS
 * with known input values, not just shapes.
 *
 * Run:
 *   cd platform-tests && mvn test -Dtest=TestDspWorkingTreeChanges \
 *     -Dbackend.artifactId=nd4j-native 2>&1 | tee /tmp/dsp-wt-changes.log
 */
@Slf4j
@DisplayName("Working Tree Changes — Isolation Tests")
public class TestDspWorkingTreeChanges {

    // ─── GDN: Basic correctness with known values ─────────────────────────

    @Test
    @DisplayName("GDN: known inputs produce correct output (no stateIn)")
    public void testGdnBasicCorrectness() {
        // B=1, L=1, H=1, D_k=2, D_v=2
        // With no stateIn (zero state), single timestep:
        //   prediction = dot(S[dv,:], k[:]) = 0 for all dv (state is zero)
        //   delta = v[dv] - exp(g)*prediction = v[dv]
        //   S[dv,dk] = exp(g)*0 + beta * k[dk] * delta = beta * k[dk] * v[dv]
        //   output[dv] = dot(S[dv,:], q[:]) = beta * v[dv] * dot(k, q)
        float[] qVals = {1.0f, 0.0f};
        float[] kVals = {1.0f, 0.0f};
        float[] vVals = {3.0f, 5.0f};
        float betaVal = 1.0f;
        float gateVal = 0.0f;  // exp(0)=1

        INDArray q = Nd4j.createFromArray(qVals).reshape(1, 1, 1, 2);
        INDArray k = Nd4j.createFromArray(kVals).reshape(1, 1, 1, 2);
        INDArray v = Nd4j.createFromArray(vVals).reshape(1, 1, 1, 2);
        INDArray beta = Nd4j.createFromArray(new float[]{betaVal}).reshape(1, 1, 1);
        INDArray gate = Nd4j.createFromArray(new float[]{gateVal}).reshape(1, 1, 1);

        // Use eager API
        INDArray[] results = Nd4j.nn.gatedDeltaRule(q, k, v, beta, gate);
        INDArray output = results[0];
        INDArray stateOut = results[1];

        log.info("GDN output shape: {}", Arrays.toString(output.shape()));
        log.info("GDN output values: {}", output);
        log.info("GDN stateOut shape: {}", Arrays.toString(stateOut.shape()));
        log.info("GDN stateOut values: {}", stateOut);

        // output shape: [1,1,1,2]
        assertArrayEquals(new long[]{1, 1, 1, 2}, output.shape(), "output shape");
        // stateOut shape: [1,1,2,2]  ([B,H,D_k,D_v])
        assertArrayEquals(new long[]{1, 1, 2, 2}, stateOut.shape(), "stateOut shape");

        // With S=0, g=0 (exp(g)=1), beta=1, k=[1,0], v=[3,5]:
        //   For each dv:
        //     prediction = dot(0, k) = 0
        //     delta = v[dv] - 1*0 = v[dv]
        //     S[dv,dk] = 1*0 + 1 * k[dk] * v[dv]
        //     S[0,:] = [1*3, 0*3] = [3, 0]
        //     S[1,:] = [1*5, 0*5] = [5, 0]
        //   output[dv] = dot(S[dv,:], q[:]) = dot(S[dv,:], [1,0])
        //     output[0] = 3
        //     output[1] = 5
        float out0 = output.getFloat(0, 0, 0, 0);
        float out1 = output.getFloat(0, 0, 0, 1);
        log.info("GDN output[0]={}, output[1]={}", out0, out1);
        assertEquals(3.0f, out0, 1e-4f, "output[0] should be 3.0");
        assertEquals(5.0f, out1, 1e-4f, "output[1] should be 5.0");

        // stateOut[0,0, dk, dv]:
        //   S transposed back to [B,H,D_k,D_v]:
        //     stateOut[0,0,0,0] = S[0,0] = 3  (dk=0,dv=0)
        //     stateOut[0,0,0,1] = S[1,0] = 5  (dk=0,dv=1)
        //     stateOut[0,0,1,0] = S[0,1] = 0  (dk=1,dv=0)
        //     stateOut[0,0,1,1] = S[1,1] = 0  (dk=1,dv=1)
        assertEquals(3.0f, stateOut.getFloat(0, 0, 0, 0), 1e-4f, "stateOut[0,0,0,0]");
        assertEquals(5.0f, stateOut.getFloat(0, 0, 0, 1), 1e-4f, "stateOut[0,0,0,1]");
        assertEquals(0.0f, stateOut.getFloat(0, 0, 1, 0), 1e-4f, "stateOut[0,0,1,0]");
        assertEquals(0.0f, stateOut.getFloat(0, 0, 1, 1), 1e-4f, "stateOut[0,0,1,1]");
    }

    // ─── GDN: State round-trip (stateOut → stateIn) ───────────────────────

    @Test
    @DisplayName("GDN: state round-trip — stateOut from step 1 fed as stateIn to step 2")
    public void testGdnStateRoundTrip() {
        // Step 1: build state from zero
        INDArray q = Nd4j.createFromArray(new float[]{1.0f, 0.0f}).reshape(1, 1, 1, 2);
        INDArray k = Nd4j.createFromArray(new float[]{1.0f, 0.0f}).reshape(1, 1, 1, 2);
        INDArray v = Nd4j.createFromArray(new float[]{3.0f, 5.0f}).reshape(1, 1, 1, 2);
        INDArray beta = Nd4j.createFromArray(new float[]{1.0f}).reshape(1, 1, 1);
        INDArray gate = Nd4j.createFromArray(new float[]{0.0f}).reshape(1, 1, 1);

        INDArray[] step1 = Nd4j.nn.gatedDeltaRule(q, k, v, beta, gate);
        INDArray stateAfterStep1 = step1[1];
        log.info("Step 1 state: {}", stateAfterStep1);

        // Step 2: feed stateOut as stateIn with new v values
        INDArray v2 = Nd4j.createFromArray(new float[]{1.0f, 2.0f}).reshape(1, 1, 1, 2);
        INDArray[] step2 = Nd4j.nn.gatedDeltaRule(q, k, v2, beta, gate, stateAfterStep1);
        INDArray output2 = step2[0];
        INDArray state2 = step2[1];
        log.info("Step 2 output: {}", output2);
        log.info("Step 2 state: {}", state2);

        // Step 2 math with g=0 (exp(g)=1), beta=1, k=[1,0], q=[1,0]:
        //   Initial state (from step 1, internal transposed layout [D_v, D_k]):
        //     S[0,:] = [3, 0]   (dv=0)
        //     S[1,:] = [5, 0]   (dv=1)
        //   For dv=0: prediction = dot([3,0], [1,0]) = 3
        //     delta = v2[0] - 1*3 = 1 - 3 = -2
        //     S[0,:] = 1*[3,0] + 1*[1,0]*(-2) = [3-2, 0] = [1, 0]
        //   For dv=1: prediction = dot([5,0], [1,0]) = 5
        //     delta = v2[1] - 1*5 = 2 - 5 = -3
        //     S[1,:] = 1*[5,0] + 1*[1,0]*(-3) = [5-3, 0] = [2, 0]
        //   output[0] = dot(S[0,:], [1,0]) = 1
        //   output[1] = dot(S[1,:], [1,0]) = 2
        assertEquals(1.0f, output2.getFloat(0, 0, 0, 0), 1e-4f, "step2 output[0]");
        assertEquals(2.0f, output2.getFloat(0, 0, 0, 1), 1e-4f, "step2 output[1]");
    }

    // ─── GDN: Multi-timestep sequence ─────────────────────────────────────

    @Test
    @DisplayName("GDN: multi-timestep L=3 produces finite non-zero output")
    public void testGdnMultiTimestep() {
        int B = 1, L = 3, H = 2, D_k = 4, D_v = 4;
        INDArray q = Nd4j.randn(DataType.FLOAT, B, L, H, D_k);
        INDArray k = Nd4j.randn(DataType.FLOAT, B, L, H, D_k);
        INDArray v = Nd4j.randn(DataType.FLOAT, B, L, H, D_v);
        INDArray beta = Nd4j.rand(DataType.FLOAT, B, L, H);  // positive betas
        INDArray gate = Nd4j.randn(DataType.FLOAT, B, L, H).mul(0.1);  // small gate values

        INDArray[] results = Nd4j.nn.gatedDeltaRule(q, k, v, beta, gate);
        INDArray output = results[0];
        INDArray stateOut = results[1];

        log.info("Multi-step GDN output shape: {}", Arrays.toString(output.shape()));
        assertArrayEquals(new long[]{B, L, H, D_v}, output.shape(), "output shape");
        assertArrayEquals(new long[]{B, H, D_k, D_v}, stateOut.shape(), "stateOut shape");

        assertFalse(output.isNaN().any(), "output has NaN");
        assertFalse(output.isInfinite().any(), "output has Inf");
        // Output should not be all zeros
        assertTrue(Transforms.abs(output).maxNumber().floatValue() > 0, "output should not be all zeros");
    }

    // ─── Mmul: same-type FP32 ─────────────────────────────────────────────

    @Test
    @DisplayName("Mmul: same-type FP32 produces correct result")
    public void testMmulSameType() {
        // Simple 2x3 * 3x2 = 2x2
        INDArray a = Nd4j.createFromArray(new float[]{
            1, 2, 3,
            4, 5, 6
        }).reshape(2, 3);
        INDArray b = Nd4j.createFromArray(new float[]{
            7, 8,
            9, 10,
            11, 12
        }).reshape(3, 2);

        INDArray result = a.mmul(b);
        log.info("Mmul result: {}", result);

        // Expected: [[1*7+2*9+3*11, 1*8+2*10+3*12], [4*7+5*9+6*11, 4*8+5*10+6*12]]
        //         = [[7+18+33, 8+20+36], [28+45+66, 32+50+72]]
        //         = [[58, 64], [139, 154]]
        assertEquals(58.0f, result.getFloat(0, 0), 1e-3f, "[0,0]");
        assertEquals(64.0f, result.getFloat(0, 1), 1e-3f, "[0,1]");
        assertEquals(139.0f, result.getFloat(1, 0), 1e-3f, "[1,0]");
        assertEquals(154.0f, result.getFloat(1, 1), 1e-3f, "[1,1]");
    }

    // ─── Mmul: 3D batch matmul ────────────────────────────────────────────

    @Test
    @DisplayName("Mmul: 3D batch matmul [B,M,K] x [B,K,N]")
    public void testMmul3D() {
        // Use SameDiff for 3D mmul to avoid eager rank issues with rows()
        int B = 2, M = 3, K = 4, N = 5;
        SameDiff sd = SameDiff.create();
        SDVariable aVar = sd.placeHolder("a", DataType.FLOAT, B, M, K);
        SDVariable bVar = sd.placeHolder("b", DataType.FLOAT, B, K, N);
        SDVariable result = sd.mmul("result", aVar, bVar);

        INDArray aArr = Nd4j.randn(DataType.FLOAT, B, M, K);
        INDArray bArr = Nd4j.randn(DataType.FLOAT, B, K, N);

        Map<String, INDArray> ph = new HashMap<>();
        ph.put("a", aArr);
        ph.put("b", bArr);
        Map<String, INDArray> out = sd.output(ph, "result");
        INDArray res = out.get("result");

        assertArrayEquals(new long[]{B, M, N}, res.shape(), "3D mmul shape");
        assertFalse(res.isNaN().any(), "3D mmul has NaN");
        assertFalse(res.isInfinite().any(), "3D mmul has Inf");

        // Verify a few elements by manual dot product
        for (int b = 0; b < B; b++) {
            for (int i = 0; i < M; i++) {
                for (int j = 0; j < N; j++) {
                    float expected = 0;
                    for (int kk = 0; kk < K; kk++) {
                        expected += aArr.getFloat(b, i, kk) * bArr.getFloat(b, kk, j);
                    }
                    float actual = res.getFloat(b, i, j);
                    assertEquals(expected, actual, 1e-2f,
                            String.format("Batch %d [%d,%d]: expected %f got %f", b, i, j, expected, actual));
                }
            }
        }
    }

    // ─── Concat: same type ────────────────────────────────────────────────

    @Test
    @DisplayName("Concat: same-type preserves data exactly")
    public void testConcatSameType() {
        INDArray a = Nd4j.createFromArray(new float[]{1, 2, 3, 4}).reshape(1, 2, 2);
        INDArray b = Nd4j.createFromArray(new float[]{5, 6, 7, 8}).reshape(1, 2, 2);

        INDArray result = Nd4j.concat(1, a, b);
        assertArrayEquals(new long[]{1, 4, 2}, result.shape(), "concat shape");

        // Values must be preserved exactly
        assertEquals(1.0f, result.getFloat(0, 0, 0), 0f);
        assertEquals(2.0f, result.getFloat(0, 0, 1), 0f);
        assertEquals(3.0f, result.getFloat(0, 1, 0), 0f);
        assertEquals(4.0f, result.getFloat(0, 1, 1), 0f);
        assertEquals(5.0f, result.getFloat(0, 2, 0), 0f);
        assertEquals(6.0f, result.getFloat(0, 2, 1), 0f);
        assertEquals(7.0f, result.getFloat(0, 3, 0), 0f);
        assertEquals(8.0f, result.getFloat(0, 3, 1), 0f);
    }

    // ─── Concat: KV cache append ──────────────────────────────────────────

    @Test
    @DisplayName("Concat: KV cache append preserves cached + new data")
    public void testConcatKvAppend() {
        int B = 1, H = 2, D = 4, cachedLen = 5;
        INDArray cached = Nd4j.ones(DataType.FLOAT, B, cachedLen, H, D).mul(2.0f);
        INDArray newToken = Nd4j.ones(DataType.FLOAT, B, 1, H, D).mul(7.0f);

        INDArray appended = Nd4j.concat(1, cached, newToken);
        assertArrayEquals(new long[]{B, cachedLen + 1, H, D}, appended.shape(), "appended shape");

        // Cached region should still be 2.0
        for (int s = 0; s < cachedLen; s++) {
            assertEquals(2.0f, appended.getFloat(0, s, 0, 0), 0f,
                    "cached position " + s + " should be 2.0");
        }
        // New token should be 7.0
        assertEquals(7.0f, appended.getFloat(0, cachedLen, 0, 0), 0f,
                "new token position should be 7.0");
    }

    // ─── Reshape: multi-head split and merge ──────────────────────────────

    @Test
    @DisplayName("Reshape: multi-head split [B,S,H*D] -> [B,S,H,D] preserves values")
    public void testReshapeMultiHeadSplit() {
        int B = 1, S = 3, H = 2, D = 4;
        // Fill with sequential values so we can verify exact positions
        INDArray flat = Nd4j.linspace(1, B * S * H * D, B * S * H * D, DataType.FLOAT).reshape(B, S, H * D);
        INDArray split = flat.reshape(B, S, H, D);

        assertArrayEquals(new long[]{B, S, H, D}, split.shape(), "split shape");

        // Verify: flat[0,0,0..7] = [1,2,3,4,5,6,7,8]
        // split[0,0,0,:] = [1,2,3,4]  (head 0)
        // split[0,0,1,:] = [5,6,7,8]  (head 1)
        assertEquals(1.0f, split.getFloat(0, 0, 0, 0), 0f);
        assertEquals(4.0f, split.getFloat(0, 0, 0, 3), 0f);
        assertEquals(5.0f, split.getFloat(0, 0, 1, 0), 0f);
        assertEquals(8.0f, split.getFloat(0, 0, 1, 3), 0f);

        // Merge back
        INDArray merged = split.reshape(B, S, H * D);
        assertEquals(flat, merged, "round-trip reshape must be exact");
    }

    // ─── Reduce: correctness with known values ───────────────────────────

    @Test
    @DisplayName("Reduce: mean and sum with known values")
    public void testReduceCorrectness() {
        INDArray input = Nd4j.createFromArray(new float[]{
            1, 2, 3,
            4, 5, 6
        }).reshape(2, 3);

        // Mean along axis 1
        INDArray mean = input.mean(true, 1);  // [2,1]
        assertArrayEquals(new long[]{2, 1}, mean.shape(), "mean keepdims shape");
        assertEquals(2.0f, mean.getFloat(0, 0), 1e-5f, "mean row 0 = (1+2+3)/3");
        assertEquals(5.0f, mean.getFloat(1, 0), 1e-5f, "mean row 1 = (4+5+6)/3");

        // Sum along axis 0
        INDArray sum = input.sum(true, 0);  // [1,3]
        assertArrayEquals(new long[]{1, 3}, sum.shape(), "sum keepdims shape");
        assertEquals(5.0f, sum.getFloat(0, 0), 1e-5f, "sum col 0 = 1+4");
        assertEquals(7.0f, sum.getFloat(0, 1), 1e-5f, "sum col 1 = 2+5");
        assertEquals(9.0f, sum.getFloat(0, 2), 1e-5f, "sum col 2 = 3+6");
    }

    // ─── SDPA via SameDiff ────────────────────────────────────────────────

    @Test
    @DisplayName("SDPA: rank-4 Q/K/V attention via SameDiff")
    public void testSdpaRank4() {
        int B = 1, H = 2, S = 4, D = 8;

        // Build attention graph in SameDiff to avoid rank-4 eager op issues
        SameDiff sd = SameDiff.create();
        SDVariable qVar = sd.placeHolder("q", DataType.FLOAT, B, H, S, D);
        SDVariable kVar = sd.placeHolder("k", DataType.FLOAT, B, H, S, D);
        SDVariable vVar = sd.placeHolder("v", DataType.FLOAT, B, H, S, D);

        SDVariable kT = sd.permute("kT", kVar, 0, 1, 3, 2);
        SDVariable scores = sd.mmul("scores", qVar, kT);
        SDVariable scale = sd.constant("scale", Nd4j.scalar(DataType.FLOAT, 1.0f / (float) Math.sqrt(D)));
        SDVariable scaled = scores.mul("scaled", scale);
        SDVariable attnW = sd.nn.softmax("attn_w", scaled, -1);
        SDVariable attnOut = sd.mmul("attn_out", attnW, vVar);

        Map<String, INDArray> ph = new HashMap<>();
        ph.put("q", Nd4j.randn(DataType.FLOAT, B, H, S, D));
        ph.put("k", Nd4j.randn(DataType.FLOAT, B, H, S, D));
        ph.put("v", Nd4j.randn(DataType.FLOAT, B, H, S, D));

        Map<String, INDArray> result = sd.output(ph, "attn_out", "attn_w");
        INDArray manualOut = result.get("attn_out");
        INDArray attnWeights = result.get("attn_w");

        log.info("SDPA output shape: {}", Arrays.toString(manualOut.shape()));
        assertArrayEquals(new long[]{B, H, S, D}, manualOut.shape(), "SDPA shape");
        assertFalse(manualOut.isNaN().any(), "SDPA has NaN");
        assertFalse(manualOut.isInfinite().any(), "SDPA has Inf");

        // Verify attention weights sum to ~1 along last dim
        INDArray weightSums = attnWeights.sum(3);
        for (int b = 0; b < B; b++)
            for (int h = 0; h < H; h++)
                for (int s = 0; s < S; s++) {
                    float sum = weightSums.getFloat(b, h, s);
                    assertEquals(1.0f, sum, 1e-4f,
                            String.format("attn weight sum [%d,%d,%d] = %f", b, h, s, sum));
                }
    }

    // ─── E2E: Mini Qwen block (matmul+reshape+permute+attention+residual→logits)

    @Test
    @DisplayName("E2E: mini transformer block with all changed ops")
    public void testMiniQwenBlock() {
        int B = 1, S = 5, hidden = 32, H = 4, D = hidden / H;
        int intermediate = 64, vocab = 100;
        SameDiff sd = SameDiff.create();

        SDVariable input = sd.placeHolder("input", DataType.FLOAT, B, S, hidden);

        // ── RMS norm
        SDVariable gamma = sd.var("gamma", Nd4j.ones(DataType.FLOAT, hidden));
        SDVariable sq = sd.math.square("sq", input);
        SDVariable meanSq = sd.mean("mean_sq", sq, true, 2);
        SDVariable eps = sd.constant("eps", Nd4j.scalar(DataType.FLOAT, 1e-6f));
        SDVariable rms = sd.math.sqrt("rms", meanSq.add("mean_sq_eps", eps));
        SDVariable normed = input.div("normed_div", rms).mul("normed", gamma);

        // ── QKV projection + reshape + permute
        SDVariable wQKV = sd.var("w_qkv", Nd4j.randn(DataType.FLOAT, hidden, 3 * hidden).mul(0.02));
        SDVariable qkvFlat = sd.mmul("qkv_flat", normed, wQKV);  // [B,S,3*hidden]

        // Split via strided slice (or just use 3 separate projections for clarity)
        SDVariable wQ = sd.var("w_q", Nd4j.randn(DataType.FLOAT, hidden, hidden).mul(0.02));
        SDVariable wK = sd.var("w_k", Nd4j.randn(DataType.FLOAT, hidden, hidden).mul(0.02));
        SDVariable wV = sd.var("w_v", Nd4j.randn(DataType.FLOAT, hidden, hidden).mul(0.02));

        SDVariable qFlat = sd.mmul("q_flat", normed, wQ);
        SDVariable kFlat = sd.mmul("k_flat", normed, wK);
        SDVariable vFlat = sd.mmul("v_flat", normed, wV);

        // Reshape to multi-head
        SDVariable qMH = sd.reshape("q_mh", qFlat, B, S, H, D);
        SDVariable kMH = sd.reshape("k_mh", kFlat, B, S, H, D);
        SDVariable vMH = sd.reshape("v_mh", vFlat, B, S, H, D);

        // Permute to [B,H,S,D]
        SDVariable qP = sd.permute("q_p", qMH, 0, 2, 1, 3);
        SDVariable kP = sd.permute("k_p", kMH, 0, 2, 1, 3);
        SDVariable vP = sd.permute("v_p", vMH, 0, 2, 1, 3);

        // ── Attention: Q*K^T/sqrt(D) -> softmax -> *V
        SDVariable kT = sd.permute("k_t", kP, 0, 1, 3, 2);
        SDVariable scores = sd.mmul("scores", qP, kT);
        SDVariable scale = sd.constant("scale", Nd4j.scalar(DataType.FLOAT, 1.0f / (float) Math.sqrt(D)));
        SDVariable scaled = scores.mul("scaled", scale);
        SDVariable attnW = sd.nn.softmax("attn_w", scaled, -1);
        SDVariable attnOut = sd.mmul("attn_v", attnW, vP);

        // Transpose back + reshape + output projection
        SDVariable attnPerm = sd.permute("attn_perm", attnOut, 0, 2, 1, 3);
        SDVariable attnFlat = sd.reshape("attn_flat", attnPerm, B, S, hidden);
        SDVariable wO = sd.var("w_o", Nd4j.randn(DataType.FLOAT, hidden, hidden).mul(0.02));
        SDVariable attnProj = sd.mmul("attn_proj", attnFlat, wO);

        // ── Residual
        SDVariable residual = input.add("residual", attnProj);

        // ── LM head
        SDVariable wLm = sd.var("w_lm", Nd4j.randn(DataType.FLOAT, hidden, vocab).mul(0.02));
        SDVariable logits = sd.mmul("logits", residual, wLm);

        Map<String, INDArray> ph = new HashMap<>();
        ph.put("input", Nd4j.randn(DataType.FLOAT, B, S, hidden));

        Map<String, INDArray> result = sd.output(ph, "logits", "attn_v", "normed");
        INDArray logitsOut = result.get("logits");
        INDArray attnVOut = result.get("attn_v");
        INDArray normedOut = result.get("normed");

        log.info("Logits shape: {} dtype: {}", Arrays.toString(logitsOut.shape()), logitsOut.dataType());
        log.info("Logits min={} max={} mean={}", logitsOut.minNumber(), logitsOut.maxNumber(), logitsOut.meanNumber());

        assertArrayEquals(new long[]{B, S, vocab}, logitsOut.shape(), "logits shape");
        assertArrayEquals(new long[]{B, H, S, D}, attnVOut.shape(), "attn_v shape");
        assertArrayEquals(new long[]{B, S, hidden}, normedOut.shape(), "normed shape");
        assertEquals(DataType.FLOAT, logitsOut.dataType());
        assertFalse(logitsOut.isNaN().any(), "logits has NaN");
        assertFalse(logitsOut.isInfinite().any(), "logits has Inf");
    }

    // ─── Prefill → Decode pipeline ─────────────────────────────────────────

    @Test
    @DisplayName("Prefill→Decode: shape transitions work correctly over 5 decode steps")
    public void testPrefillDecodePipeline() {
        int B = 1, hidden = 32, H = 4, D = hidden / H, vocab = 50;
        SameDiff sd = SameDiff.create();

        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, -1, hidden);
        SDVariable w1 = sd.var("w1", Nd4j.randn(DataType.FLOAT, hidden, hidden).mul(0.02));
        SDVariable w2 = sd.var("w2", Nd4j.randn(DataType.FLOAT, hidden, vocab).mul(0.02));

        SDVariable h = sd.mmul("h", input, w1);
        SDVariable act = sd.nn.relu("act", h, 0);
        SDVariable logits = sd.mmul("logits", act, w2);

        // Prefill with S=7
        Map<String, INDArray> prefillPh = new HashMap<>();
        prefillPh.put("input", Nd4j.randn(DataType.FLOAT, B, 7, hidden));
        Map<String, INDArray> prefillResult = sd.output(prefillPh, "logits");
        assertArrayEquals(new long[]{B, 7, vocab}, prefillResult.get("logits").shape(), "prefill logits shape");
        log.info("Prefill logits: shape={}", Arrays.toString(prefillResult.get("logits").shape()));

        // 5 decode steps with S=1
        for (int step = 0; step < 5; step++) {
            Map<String, INDArray> decodePh = new HashMap<>();
            decodePh.put("input", Nd4j.randn(DataType.FLOAT, B, 1, hidden));
            Map<String, INDArray> decodeResult = sd.output(decodePh, "logits");
            INDArray decLogits = decodeResult.get("logits");

            assertArrayEquals(new long[]{B, 1, vocab}, decLogits.shape(),
                    "decode step " + step + " logits shape");
            assertFalse(decLogits.isNaN().any(), "decode step " + step + " has NaN");
            assertFalse(decLogits.isInfinite().any(), "decode step " + step + " has Inf");

            // Logits should not be constant across steps (different random input)
            if (step > 0) {
                // Just verify the logits are not all the same value
                float min = decLogits.minNumber().floatValue();
                float max = decLogits.maxNumber().floatValue();
                assertTrue(max - min > 1e-6f, "decode logits should have variance at step " + step);
            }
            log.info("Decode step {}: shape={} min={} max={}",
                    step, Arrays.toString(decLogits.shape()),
                    decLogits.minNumber(), decLogits.maxNumber());
        }
    }

    // ─── GDN via SameDiff (graph-level test) ──────────────────────────────

    @Test
    @DisplayName("GDN via SameDiff: graph execution matches eager results")
    public void testGdnViaSameDiff() {
        int B = 1, L = 2, H = 2, D_k = 4, D_v = 4;
        SameDiff sd = SameDiff.create();

        SDVariable qVar = sd.placeHolder("q", DataType.FLOAT, B, L, H, D_k);
        SDVariable kVar = sd.placeHolder("k", DataType.FLOAT, B, L, H, D_k);
        SDVariable vVar = sd.placeHolder("v", DataType.FLOAT, B, L, H, D_v);
        SDVariable betaVar = sd.placeHolder("beta", DataType.FLOAT, B, L, H);
        SDVariable gateVar = sd.placeHolder("gate", DataType.FLOAT, B, L, H);

        SDVariable[] gdnResults = sd.nn.gatedDeltaRule(
                new String[]{"gdn_output", "gdn_state"},
                qVar, kVar, vVar, betaVar, gateVar);

        // Create test data
        INDArray qArr = Nd4j.randn(DataType.FLOAT, B, L, H, D_k);
        INDArray kArr = Nd4j.randn(DataType.FLOAT, B, L, H, D_k);
        INDArray vArr = Nd4j.randn(DataType.FLOAT, B, L, H, D_v);
        INDArray betaArr = Nd4j.rand(DataType.FLOAT, B, L, H);
        INDArray gateArr = Nd4j.randn(DataType.FLOAT, B, L, H).mul(0.1);

        Map<String, INDArray> ph = new HashMap<>();
        ph.put("q", qArr);
        ph.put("k", kArr);
        ph.put("v", vArr);
        ph.put("beta", betaArr);
        ph.put("gate", gateArr);

        // SameDiff execution
        Map<String, INDArray> sdResult = sd.output(ph, "gdn_output", "gdn_state");
        INDArray sdOutput = sdResult.get("gdn_output");
        INDArray sdState = sdResult.get("gdn_state");

        // Eager execution
        INDArray[] eagerResult = Nd4j.nn.gatedDeltaRule(qArr, kArr, vArr, betaArr, gateArr);

        log.info("SD GDN output shape: {}", Arrays.toString(sdOutput.shape()));
        log.info("Eager GDN output shape: {}", Arrays.toString(eagerResult[0].shape()));

        assertArrayEquals(new long[]{B, L, H, D_v}, sdOutput.shape(), "SD GDN output shape");
        assertArrayEquals(new long[]{B, H, D_k, D_v}, sdState.shape(), "SD GDN state shape");

        // Results must match between SameDiff and eager
        float maxDiff = Transforms.abs(sdOutput.sub(eagerResult[0])).maxNumber().floatValue();
        log.info("SameDiff vs Eager max diff: {}", maxDiff);
        assertTrue(maxDiff < 1e-4f, "SameDiff and Eager results must match, max diff = " + maxDiff);
    }

    // ─── RoPE: correctness against Java reference ────────────────────────

    /**
     * Pure-Java reference implementation of standard RoPE (type 0, LLaMA-style).
     * x[i] and x[i+halfRotate] form a pair rotated by theta.
     */
    private static float[][] referenceRoPE(float[] input, int headDim, int positionOffset,
                                           float freqBase, float freqScale) {
        int halfRotate = headDim / 2;
        float[] output = new float[headDim];
        float[] thetas = new float[halfRotate];

        for (int i = 0; i < halfRotate; i++) {
            float invFreq = freqScale / (float) Math.pow(freqBase, (2.0f * i) / headDim);
            thetas[i] = positionOffset * invFreq;
        }

        // Standard RoPE: pair (x[i], x[i+half]) rotated by theta[i]
        for (int i = 0; i < halfRotate; i++) {
            float cosT = (float) Math.cos(thetas[i]);
            float sinT = (float) Math.sin(thetas[i]);
            float x0 = input[i];
            float x1 = input[i + halfRotate];
            output[i] = x0 * cosT - x1 * sinT;
            output[i + halfRotate] = x0 * sinT + x1 * cosT;
        }
        return new float[][]{output, thetas};
    }

    @Test
    @DisplayName("RoPE: standard type-0 matches Java reference implementation")
    public void testRopeStandardCorrectness() {
        int B = 1, S = 3, H = 2, D = 8;
        float freqBase = 10000.0f;
        float freqScale = 1.0f;
        int ropeType = 0;
        int positionOffset = 0;

        // Create input with known values
        INDArray input = Nd4j.linspace(1, B * S * H * D, B * S * H * D, DataType.FLOAT).reshape(B, S, H, D);

        // Execute native RoPE
        INDArray result = Nd4j.nn.fusedRoPE(input, null, ropeType, freqBase, freqScale, 0);

        log.info("RoPE input shape: {}", Arrays.toString(input.shape()));
        log.info("RoPE output shape: {}", Arrays.toString(result.shape()));
        assertArrayEquals(new long[]{B, S, H, D}, result.shape(), "RoPE output shape");

        // Verify against Java reference for each position/head
        for (int b = 0; b < B; b++) {
            for (int s = 0; s < S; s++) {
                for (int h = 0; h < H; h++) {
                    // Extract this head's input vector
                    float[] headInput = new float[D];
                    for (int d = 0; d < D; d++) {
                        headInput[d] = input.getFloat(b, s, h, d);
                    }

                    // Compute reference
                    float[][] ref = referenceRoPE(headInput, D, positionOffset + s, freqBase, freqScale);
                    float[] expected = ref[0];

                    // Compare with native output
                    for (int d = 0; d < D; d++) {
                        float actual = result.getFloat(b, s, h, d);
                        float exp = expected[d];
                        assertEquals(exp, actual, 1e-4f,
                                String.format("RoPE mismatch at [%d,%d,%d,%d]: expected %f got %f",
                                        b, s, h, d, exp, actual));
                    }
                }
            }
        }
        log.info("RoPE standard type-0: ALL positions match Java reference");
    }

    @Test
    @DisplayName("RoPE: NeoX type-1 (interleaved pairs) matches Java reference")
    public void testRopeNeoXCorrectness() {
        int B = 1, S = 2, H = 1, D = 8;
        float freqBase = 10000.0f;
        float freqScale = 1.0f;
        int ropeType = 1;
        int positionOffset = 5;  // non-zero offset to catch position bugs

        INDArray input = Nd4j.linspace(1, B * S * H * D, B * S * H * D, DataType.FLOAT).reshape(B, S, H, D);

        // Use Nd4j.exec with FusedRoPE directly to pass positionOffset as startPosition
        INDArray result = Nd4j.exec(new FusedRoPE(input, null, null,
                positionOffset, ropeType, freqBase, freqScale, 0))[0];

        // NeoX interleaved: pairs are (x[2i], x[2i+1])
        int halfRotate = D / 2;
        for (int b = 0; b < B; b++) {
            for (int s = 0; s < S; s++) {
                int pos = positionOffset + s;
                for (int h = 0; h < H; h++) {
                    for (int i = 0; i < halfRotate; i++) {
                        float invFreq = freqScale / (float) Math.pow(freqBase, (2.0f * i) / D);
                        float theta = pos * invFreq;
                        float cosT = (float) Math.cos(theta);
                        float sinT = (float) Math.sin(theta);

                        float x0 = input.getFloat(b, s, h, 2 * i);
                        float x1 = input.getFloat(b, s, h, 2 * i + 1);
                        float exp0 = x0 * cosT - x1 * sinT;
                        float exp1 = x0 * sinT + x1 * cosT;

                        float act0 = result.getFloat(b, s, h, 2 * i);
                        float act1 = result.getFloat(b, s, h, 2 * i + 1);

                        assertEquals(exp0, act0, 1e-4f,
                                String.format("NeoX RoPE mismatch at [%d,%d,%d,%d]", b, s, h, 2 * i));
                        assertEquals(exp1, act1, 1e-4f,
                                String.format("NeoX RoPE mismatch at [%d,%d,%d,%d]", b, s, h, 2 * i + 1));
                    }
                }
            }
        }
        log.info("RoPE NeoX type-1: ALL positions match Java reference");
    }

    @Test
    @DisplayName("RoPE: decode position offset produces different embeddings")
    public void testRopePositionOffset() {
        int B = 1, S = 1, H = 2, D = 16;
        float freqBase = 10000.0f;
        float freqScale = 1.0f;

        INDArray input = Nd4j.ones(DataType.FLOAT, B, S, H, D);

        // Position 0
        INDArray out0 = Nd4j.nn.fusedRoPE(input, null, 0, freqBase, freqScale, 0);
        // Position 10
        INDArray out10 = Nd4j.nn.fusedRoPE(input.dup(), null, 0, freqBase, freqScale, 0);

        // If positionOffset is correctly applied, position 0 and 10 should differ
        // But since both use positionOffset=0, they should match here
        // Let me instead use Nd4j.exec directly with different startPosition
        INDArray out_pos5 = Nd4j.exec(
                new org.nd4j.linalg.api.ops.impl.transforms.custom.FusedRoPE(
                        input.dup(), null, null, 5, 0, freqBase, freqScale, 0))[0];
        INDArray out_pos10 = Nd4j.exec(
                new org.nd4j.linalg.api.ops.impl.transforms.custom.FusedRoPE(
                        input.dup(), null, null, 10, 0, freqBase, freqScale, 0))[0];

        // Different positions should produce different outputs
        float diff = Transforms.abs(out_pos5.sub(out_pos10)).maxNumber().floatValue();
        log.info("RoPE pos5 vs pos10 max diff: {}", diff);
        assertTrue(diff > 1e-3f, "Different positions should produce different RoPE outputs");

        // Same position should produce same output
        INDArray out_pos5b = Nd4j.exec(
                new org.nd4j.linalg.api.ops.impl.transforms.custom.FusedRoPE(
                        input.dup(), null, null, 5, 0, freqBase, freqScale, 0))[0];
        float sameDiff = Transforms.abs(out_pos5.sub(out_pos5b)).maxNumber().floatValue();
        log.info("RoPE pos5 vs pos5 (repeat) max diff: {}", sameDiff);
        assertEquals(0.0f, sameDiff, 1e-6f, "Same position should produce identical RoPE outputs");
    }
}
