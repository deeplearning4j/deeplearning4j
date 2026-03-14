/*
 *  ******************************************************************************
 *  *
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
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.custom.OnnxMultiHeadAttention;
import org.nd4j.linalg.factory.Nd4j;

import java.util.HashMap;
import java.util.Map;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Isolation tests for the Triton fused attention kernel handling
 * onnx_multi_head_attention compound ops with 3D Q and external past_key/value.
 *
 * These tests compare Triton-compiled attention output against C++ reference
 * (SLOT_BY_SLOT baseline) to verify kernel correctness independently from the
 * full VLM pipeline.
 *
 * The compound MHA op:
 *   - Takes 3D Q/K/V [B, seq, H*D]
 *   - Takes 4D past_key/value [B, H, pastSeq, D] (BHSD format)
 *   - Takes 4D attention bias [B, H, seqQ, seqKV]
 *   - Internally reshapes, permutes, concatenates, runs FlashAttention
 *   - Outputs 3D attention [B, seqQ, H*D] and 4D present_key/value [B, H, totalSeq, D]
 *
 * Run:
 *   cd platform-tests && mvn test \
 *     -Dtest=TritonMhaKernelIsolationTest \
 *     -Dbackend.artifactId=nd4j-cuda-12.9
 */
@Slf4j
public class TritonMhaKernelIsolationTest {

    private static boolean tritonAvailable;

    @BeforeAll
    public static void setup() {
        tritonAvailable = Nd4j.getNativeOps().isTritonAvailable();
        log.info("Triton available: {}", tritonAvailable);
    }

    // ─── Parameter sets ─────────────────────────────────────────────────────

    static Stream<Arguments> mhaConfigs() {
        return Stream.of(
                // name, batch, numHeads, headDim, pastSeq, seqKVCur, withBias
                Arguments.of("SmolDocling_decode", 1, 9, 64, 100, 1, true),
                Arguments.of("SmolDocling_short", 1, 9, 64, 10, 1, true),
                Arguments.of("SmolDocling_long", 1, 9, 64, 500, 1, true),
                Arguments.of("2head_small", 1, 2, 32, 20, 1, true),
                Arguments.of("4head_noBias", 1, 4, 64, 50, 1, false),
                Arguments.of("8head_pow2", 1, 8, 64, 128, 1, true),
                Arguments.of("1head_minimal", 1, 1, 64, 5, 1, true)
        );
    }

    // ─── Core: Compare C++ reference vs Triton ──────────────────────────────

    /**
     * For each configuration, builds a SameDiff graph with onnx_multi_head_attention,
     * runs it with SLOT_BY_SLOT (C++ reference) and then with Triton ATTENTION compilation,
     * and compares the outputs.
     */
    @ParameterizedTest(name = "{0}")
    @MethodSource("mhaConfigs")
    @DisplayName("Triton MHA kernel matches C++ reference")
    public void testTritonMhaMatchesCppReference(String name, int batch, int numHeads,
                                                   int headDim, int pastSeq, int seqKVCur,
                                                   boolean withBias) {
        if (!tritonAvailable) {
            log.info("Skipping Triton test — not available");
            return;
        }

        int hidden = numHeads * headDim;
        int totalSeqKV = pastSeq + seqKVCur;

        // Create deterministic inputs
        Nd4j.getRandom().setSeed(42);
        INDArray query = Nd4j.randn(DataType.FLOAT, batch, 1, hidden);
        INDArray key = Nd4j.randn(DataType.FLOAT, batch, seqKVCur, hidden);
        INDArray value = Nd4j.randn(DataType.FLOAT, batch, seqKVCur, hidden);
        INDArray pastKey = Nd4j.randn(DataType.FLOAT, batch, numHeads, pastSeq, headDim);
        INDArray pastValue = Nd4j.randn(DataType.FLOAT, batch, numHeads, pastSeq, headDim);

        // Attention bias: [B, H, seqQ, totalSeqKV] — 0 for valid, -inf for masked
        // For decode: all positions valid (no future tokens to mask)
        INDArray attnBias = withBias
                ? Nd4j.zeros(DataType.FLOAT, batch, numHeads, 1, totalSeqKV)
                : null;

        log.info("[{}] Q=[{},{},{}] pastKey=[{},{},{},{}] bias={} totalSeqKV={}",
                name, batch, 1, hidden, batch, numHeads, pastSeq, headDim,
                withBias ? "[" + batch + "," + numHeads + ",1," + totalSeqKV + "]" : "none",
                totalSeqKV);

        // ── C++ reference: direct op execution ──
        INDArray refOutput = executeMhaOp(query, key, value, attnBias, pastKey, pastValue, numHeads);
        assertNotNull(refOutput, name + ": C++ reference output is null");
        log.info("[{}] C++ ref output: shape={}, mean={}, absMax={}",
                name, java.util.Arrays.toString(refOutput.shape()),
                refOutput.meanNumber().doubleValue(),
                refOutput.amaxNumber().doubleValue());

        // Verify C++ output is not garbage
        assertFalse(Double.isNaN(refOutput.meanNumber().doubleValue()),
                name + ": C++ reference output contains NaN");
        assertFalse(Double.isInfinite(refOutput.amaxNumber().doubleValue()),
                name + ": C++ reference output contains Inf");

        // ── Triton: SameDiff graph with DSP compilation ──
        INDArray tritonOutput = executeMhaViaSameDiff(
                query, key, value, attnBias, pastKey, pastValue, numHeads, true);
        assertNotNull(tritonOutput, name + ": Triton output is null");
        log.info("[{}] Triton output: shape={}, mean={}, absMax={}",
                name, java.util.Arrays.toString(tritonOutput.shape()),
                tritonOutput.meanNumber().doubleValue(),
                tritonOutput.amaxNumber().doubleValue());

        // Verify Triton output is not garbage
        assertFalse(Double.isNaN(tritonOutput.meanNumber().doubleValue()),
                name + ": Triton output contains NaN");
        assertFalse(Double.isInfinite(tritonOutput.amaxNumber().doubleValue()),
                name + ": Triton output contains Inf");

        // ── Compare ──
        assertArrayEquals(refOutput.shape(), tritonOutput.shape(),
                name + ": shape mismatch between C++ and Triton");

        double maxDiff = refOutput.sub(tritonOutput).amaxNumber().doubleValue();
        double meanDiff = refOutput.sub(tritonOutput).ameanNumber().doubleValue();
        double refNorm = refOutput.norm2Number().doubleValue();
        double relError = refNorm > 0 ? maxDiff / refNorm : maxDiff;

        log.info("[{}] maxDiff={}, meanDiff={}, relError={}", name, maxDiff, meanDiff, relError);

        assertTrue(maxDiff < 1e-2,
                name + ": Triton MHA output differs from C++ reference. maxDiff=" + maxDiff +
                        " meanDiff=" + meanDiff + " relError=" + relError);
    }

    // ─── Test: bias OOB when biasSeqK < seqK ────────────────────────────────

    /**
     * Tests the case where the attention bias covers fewer columns than the total
     * KV sequence length. In dual-buffer mode:
     *   pastSeq positions from past_key + seqKVCur positions from current K
     *   = totalSeqKV = pastSeq + seqKVCur
     *
     * But the bias might only cover pastSeq positions (from the static KV cache size).
     * The kernel must handle bias positions >= biasSeqK as 0 (no masking), not OOB read.
     */
    @Test
    @DisplayName("Bias shape shorter than total seqKV does not cause OOB")
    public void testBiasShorterThanTotalSeqKV() {
        if (!tritonAvailable) {
            log.info("Skipping — Triton not available");
            return;
        }

        int batch = 1, numHeads = 9, headDim = 64;
        int pastSeq = 100, seqKVCur = 1;
        int hidden = numHeads * headDim;

        Nd4j.getRandom().setSeed(123);
        INDArray query = Nd4j.randn(DataType.FLOAT, batch, 1, hidden);
        INDArray key = Nd4j.randn(DataType.FLOAT, batch, seqKVCur, hidden);
        INDArray value = Nd4j.randn(DataType.FLOAT, batch, seqKVCur, hidden);
        INDArray pastKey = Nd4j.randn(DataType.FLOAT, batch, numHeads, pastSeq, headDim);
        INDArray pastValue = Nd4j.randn(DataType.FLOAT, batch, numHeads, pastSeq, headDim);

        // Bias covers only pastSeq positions, NOT pastSeq + seqKVCur
        // The current token (position pastSeq) has no bias entry → should be treated as 0
        INDArray shortBias = Nd4j.zeros(DataType.FLOAT, batch, numHeads, 1, pastSeq);

        // C++ reference
        INDArray refOutput = executeMhaOp(query, key, value, shortBias, pastKey, pastValue, numHeads);

        // Triton
        INDArray tritonOutput = executeMhaViaSameDiff(
                query, key, value, shortBias, pastKey, pastValue, numHeads, true);

        assertNotNull(refOutput, "C++ reference is null");
        assertNotNull(tritonOutput, "Triton output is null");
        assertFalse(Double.isNaN(tritonOutput.meanNumber().doubleValue()),
                "Triton output contains NaN — possible OOB bias read");

        double maxDiff = refOutput.sub(tritonOutput).amaxNumber().doubleValue();
        log.info("shortBias test: maxDiff={}", maxDiff);
        assertTrue(maxDiff < 1e-2, "shortBias: maxDiff=" + maxDiff);
    }

    // ─── Test: static KV cache with masked positions ────────────────────────

    /**
     * Simulates the static KV cache scenario: past_key buffer is larger than
     * the number of valid positions. Invalid positions contain garbage data.
     * The attention bias masks them to -inf.
     */
    @Test
    @DisplayName("Static KV cache with masked garbage positions")
    public void testStaticKvCacheWithGarbagePositions() {
        if (!tritonAvailable) {
            log.info("Skipping — Triton not available");
            return;
        }

        int batch = 1, numHeads = 4, headDim = 64;
        int maxKvLen = 128;
        int validPositions = 50;
        int hidden = numHeads * headDim;

        Nd4j.getRandom().setSeed(77);
        INDArray query = Nd4j.randn(DataType.FLOAT, batch, 1, hidden);
        INDArray key = Nd4j.randn(DataType.FLOAT, batch, 1, hidden);
        INDArray value = Nd4j.randn(DataType.FLOAT, batch, 1, hidden);

        // Static KV cache: full buffer, positions validPositions..maxKvLen-1 are garbage
        INDArray pastKey = Nd4j.randn(DataType.FLOAT, batch, numHeads, maxKvLen, headDim);
        INDArray pastValue = Nd4j.randn(DataType.FLOAT, batch, numHeads, maxKvLen, headDim);

        // Attention bias: valid positions get 0, garbage positions get -inf
        int totalSeqKV = maxKvLen + 1; // pastKey + current
        INDArray bias = Nd4j.zeros(DataType.FLOAT, batch, numHeads, 1, totalSeqKV);
        // Mask out garbage positions (validPositions..maxKvLen-1) with -inf
        for (int pos = validPositions; pos < maxKvLen; pos++) {
            bias.putScalar(new int[]{0, 0, 0, pos}, -3.4028235e+38f);
        }
        // Broadcast to all heads
        for (int h = 1; h < numHeads; h++) {
            bias.put(new org.nd4j.linalg.indexing.INDArrayIndex[]{
                    org.nd4j.linalg.indexing.NDArrayIndex.point(0),
                    org.nd4j.linalg.indexing.NDArrayIndex.point(h),
                    org.nd4j.linalg.indexing.NDArrayIndex.all(),
                    org.nd4j.linalg.indexing.NDArrayIndex.all()
            }, bias.get(
                    org.nd4j.linalg.indexing.NDArrayIndex.point(0),
                    org.nd4j.linalg.indexing.NDArrayIndex.point(0),
                    org.nd4j.linalg.indexing.NDArrayIndex.all(),
                    org.nd4j.linalg.indexing.NDArrayIndex.all()
            ));
        }

        // C++ reference
        INDArray refOutput = executeMhaOp(query, key, value, bias, pastKey, pastValue, numHeads);
        assertNotNull(refOutput);

        // Triton
        INDArray tritonOutput = executeMhaViaSameDiff(
                query, key, value, bias, pastKey, pastValue, numHeads, true);
        assertNotNull(tritonOutput);

        assertFalse(Double.isNaN(refOutput.meanNumber().doubleValue()), "C++ ref has NaN");
        assertFalse(Double.isNaN(tritonOutput.meanNumber().doubleValue()), "Triton has NaN");

        double maxDiff = refOutput.sub(tritonOutput).amaxNumber().doubleValue();
        log.info("staticKvCache test: maxDiff={}", maxDiff);
        assertTrue(maxDiff < 1e-2, "staticKvCache: maxDiff=" + maxDiff);
    }

    // ─── Test: output numOutputs=1 (no present_key/value) ───────────────────

    /**
     * When the MHA op has numOutputs=1, only attention output is produced.
     * present_key/value are NOT emitted. The kernel must still compute
     * correct attention output.
     */
    @Test
    @DisplayName("MHA with numOutputs=1 still produces correct attention")
    public void testNumOutputs1() {
        if (!tritonAvailable) {
            log.info("Skipping — Triton not available");
            return;
        }

        int batch = 1, numHeads = 4, headDim = 64, pastSeq = 30;
        int hidden = numHeads * headDim;

        Nd4j.getRandom().setSeed(99);
        INDArray query = Nd4j.randn(DataType.FLOAT, batch, 1, hidden);
        INDArray key = Nd4j.randn(DataType.FLOAT, batch, 1, hidden);
        INDArray value = Nd4j.randn(DataType.FLOAT, batch, 1, hidden);
        INDArray pastKey = Nd4j.randn(DataType.FLOAT, batch, numHeads, pastSeq, headDim);
        INDArray pastValue = Nd4j.randn(DataType.FLOAT, batch, numHeads, pastSeq, headDim);

        // Reference with 3 outputs
        INDArray ref3 = executeMhaOp(query, key, value, null, pastKey, pastValue, numHeads);

        // With 1 output
        OnnxMultiHeadAttention op1 = new OnnxMultiHeadAttention(
                query.dup(), key.dup(), value.dup(),
                null, pastKey.dup(), pastValue.dup(),
                numHeads, 0.0, false, 1);
        INDArray[] out1 = Nd4j.exec(op1);

        assertNotNull(out1);
        assertTrue(out1.length >= 1, "Should have at least 1 output");
        assertArrayEquals(ref3.shape(), out1[0].shape(), "Output shape mismatch");

        double maxDiff = ref3.sub(out1[0]).amaxNumber().doubleValue();
        log.info("numOutputs=1 test: maxDiff={}", maxDiff);
        assertTrue(maxDiff < 1e-4, "numOutputs=1 attention should match 3-output. maxDiff=" + maxDiff);
    }

    // ─── Test: Triton vs C++ in SameDiff graph with multiple decode steps ───

    /**
     * Simulates multiple sequential decode steps within a SameDiff graph,
     * updating the KV cache each step. Verifies that the Triton kernel
     * produces correct output across multiple steps, not just the first.
     */
    @Test
    @DisplayName("Multi-step decode: Triton matches C++ across steps")
    public void testMultiStepDecode() {
        if (!tritonAvailable) {
            log.info("Skipping — Triton not available");
            return;
        }

        int batch = 1, numHeads = 4, headDim = 64;
        int hidden = numHeads * headDim;
        int numSteps = 5;

        Nd4j.getRandom().setSeed(55);

        // Start with some initial past context
        INDArray pastKey = Nd4j.randn(DataType.FLOAT, batch, numHeads, 10, headDim);
        INDArray pastValue = Nd4j.randn(DataType.FLOAT, batch, numHeads, 10, headDim);

        for (int step = 0; step < numSteps; step++) {
            int pastSeq = (int) pastKey.size(2);
            int totalSeq = pastSeq + 1;

            INDArray query = Nd4j.randn(DataType.FLOAT, batch, 1, hidden);
            INDArray key = Nd4j.randn(DataType.FLOAT, batch, 1, hidden);
            INDArray value = Nd4j.randn(DataType.FLOAT, batch, 1, hidden);
            INDArray bias = Nd4j.zeros(DataType.FLOAT, batch, numHeads, 1, totalSeq);

            // C++ reference
            INDArray refOutput = executeMhaOp(query, key, value, bias, pastKey, pastValue, numHeads);

            // Triton via SameDiff
            INDArray tritonOutput = executeMhaViaSameDiff(
                    query, key, value, bias, pastKey, pastValue, numHeads, true);

            double maxDiff = refOutput.sub(tritonOutput).amaxNumber().doubleValue();
            log.info("Step {}: pastSeq={}, maxDiff={}", step, pastSeq, maxDiff);
            assertTrue(maxDiff < 1e-2,
                    "Step " + step + ": maxDiff=" + maxDiff);

            // Update KV cache for next step (using C++ present_key/value as ground truth)
            OnnxMultiHeadAttention refOp = new OnnxMultiHeadAttention(
                    query.dup(), key.dup(), value.dup(),
                    bias.dup(), pastKey.dup(), pastValue.dup(),
                    numHeads, 0.0, false, 3);
            INDArray[] refOutputs = Nd4j.exec(refOp);
            pastKey = refOutputs[1].dup();
            pastValue = refOutputs[2].dup();
        }
    }

    // ─── Test: Direct op execution (no SameDiff) baseline ───────────────────

    /**
     * Verify that direct op execution produces non-garbage output for
     * compound MHA with past_key/value. This is the C++ baseline.
     */
    @Test
    @DisplayName("Direct MHA op with past KV produces valid output")
    public void testDirectMhaOpWithPastKV() {
        int batch = 1, numHeads = 9, headDim = 64, pastSeq = 50;
        int hidden = numHeads * headDim;

        Nd4j.getRandom().setSeed(42);
        INDArray query = Nd4j.randn(DataType.FLOAT, batch, 1, hidden);
        INDArray key = Nd4j.randn(DataType.FLOAT, batch, 1, hidden);
        INDArray value = Nd4j.randn(DataType.FLOAT, batch, 1, hidden);
        INDArray pastKey = Nd4j.randn(DataType.FLOAT, batch, numHeads, pastSeq, headDim);
        INDArray pastValue = Nd4j.randn(DataType.FLOAT, batch, numHeads, pastSeq, headDim);
        INDArray bias = Nd4j.zeros(DataType.FLOAT, batch, numHeads, 1, pastSeq + 1);

        OnnxMultiHeadAttention op = new OnnxMultiHeadAttention(
                query, key, value, bias, pastKey, pastValue,
                numHeads, 0.0, false, 3);
        INDArray[] outputs = Nd4j.exec(op);

        assertNotNull(outputs);
        assertTrue(outputs.length >= 1);

        INDArray attnOut = outputs[0];
        assertArrayEquals(new long[]{batch, 1, hidden}, attnOut.shape(),
                "Attention output shape mismatch");
        assertFalse(Double.isNaN(attnOut.meanNumber().doubleValue()),
                "Attention output contains NaN");
        assertFalse(Double.isInfinite(attnOut.amaxNumber().doubleValue()),
                "Attention output contains Inf");

        double absMax = attnOut.amaxNumber().doubleValue();
        assertTrue(absMax > 1e-6 && absMax < 1e6,
                "Attention output seems invalid: absMax=" + absMax);

        if (outputs.length >= 3) {
            INDArray presentKey = outputs[1];
            INDArray presentValue = outputs[2];
            assertArrayEquals(new long[]{batch, numHeads, pastSeq + 1, headDim}, presentKey.shape());
            assertArrayEquals(new long[]{batch, numHeads, pastSeq + 1, headDim}, presentValue.shape());
        }

        log.info("Direct MHA: output mean={}, absMax={}",
                attnOut.meanNumber().doubleValue(), absMax);
    }

    // ─── Test: Multi-op graph with MHA in the middle ──────────────────────
    //
    // This tests the scenario closer to the VLM decoder where MHA is NOT the
    // only op. The graph has: matmul(Q projection) → MHA → add(residual).
    // This exercises multi-section compilation where MHA receives SSA inputs
    // from preceding ops within the same segment.

    @Test
    @DisplayName("Multi-op graph: matmul → MHA → add matches C++ reference")
    public void testMultiOpGraphWithMha() {
        if (!tritonAvailable) {
            log.info("Skipping — Triton not available");
            return;
        }

        int batch = 1, numHeads = 4, headDim = 64, pastSeq = 30;
        int hidden = numHeads * headDim;
        int totalSeqKV = pastSeq + 1;

        Nd4j.getRandom().setSeed(42);

        // Inputs
        INDArray hiddenState = Nd4j.randn(DataType.FLOAT, batch, 1, hidden);
        INDArray qWeight = Nd4j.randn(DataType.FLOAT, hidden, hidden).divi(Math.sqrt(hidden));
        INDArray kWeight = Nd4j.randn(DataType.FLOAT, hidden, hidden).divi(Math.sqrt(hidden));
        INDArray vWeight = Nd4j.randn(DataType.FLOAT, hidden, hidden).divi(Math.sqrt(hidden));
        INDArray pastKey = Nd4j.randn(DataType.FLOAT, batch, numHeads, pastSeq, headDim);
        INDArray pastValue = Nd4j.randn(DataType.FLOAT, batch, numHeads, pastSeq, headDim);
        INDArray bias = Nd4j.zeros(DataType.FLOAT, batch, numHeads, 1, totalSeqKV);

        // ── C++ reference: compute step by step ──
        // Q = hiddenState @ qWeight  [1,1,256] — use reshape to 2D for mmul, then back to 3D
        INDArray h2d = hiddenState.reshape(batch, hidden);
        INDArray qProj = h2d.mmul(qWeight).reshape(batch, 1, hidden);
        INDArray kProj = h2d.mmul(kWeight).reshape(batch, 1, hidden);
        INDArray vProj = h2d.mmul(vWeight).reshape(batch, 1, hidden);
        INDArray cppAttn = executeMhaOp(qProj, kProj, vProj, bias, pastKey, pastValue, numHeads);
        // residual = hiddenState + attnOutput
        INDArray cppResidual = hiddenState.add(cppAttn);
        log.info("C++ multi-op: residual mean={}, absMax={}",
                cppResidual.meanNumber(), cppResidual.amaxNumber());

        // ── Triton: SameDiff graph ──
        SameDiff sd = SameDiff.create();
        SDVariable hVar = sd.placeHolder("hidden", DataType.FLOAT, batch, 1, hidden);
        SDVariable qwVar = sd.constant("qWeight", qWeight);
        SDVariable kwVar = sd.constant("kWeight", kWeight);
        SDVariable vwVar = sd.constant("vWeight", vWeight);
        SDVariable pkVar = sd.placeHolder("past_key", DataType.FLOAT, pastKey.shape());
        SDVariable pvVar = sd.placeHolder("past_value", DataType.FLOAT, pastValue.shape());
        SDVariable bVar = sd.placeHolder("bias", DataType.FLOAT, bias.shape());

        // Q/K/V projections via matmul
        SDVariable q = sd.mmul("q_proj", hVar, qwVar);
        SDVariable k = sd.mmul("k_proj", hVar, kwVar);
        SDVariable v = sd.mmul("v_proj", hVar, vwVar);

        // MHA
        OnnxMultiHeadAttention mha = new OnnxMultiHeadAttention(
                sd, q, k, v, bVar, pkVar, pvVar,
                numHeads, 0.0, false, 1);
        SDVariable[] mhaOutputs = mha.outputVariables();

        // Residual add
        SDVariable residual = hVar.add("residual", mhaOutputs[0]);

        // Configure Triton
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);
        Nd4j.getEnvironment().setTritonIncludeTypes("ATTENTION");

        Map<String, INDArray> feedDict = new HashMap<>();
        feedDict.put("hidden", hiddenState.dup());
        feedDict.put("past_key", pastKey.dup());
        feedDict.put("past_value", pastValue.dup());
        feedDict.put("bias", bias.dup());

        Map<String, INDArray> result = sd.output(feedDict, "residual");
        INDArray tritonResidual = result.get("residual");
        assertNotNull(tritonResidual, "Triton residual is null");

        log.info("Triton multi-op: residual mean={}, absMax={}",
                tritonResidual.meanNumber(), tritonResidual.amaxNumber());

        assertFalse(Double.isNaN(tritonResidual.meanNumber().doubleValue()),
                "Triton residual contains NaN");

        double maxDiff = cppResidual.sub(tritonResidual).amaxNumber().doubleValue();
        double meanDiff = cppResidual.sub(tritonResidual).ameanNumber().doubleValue();
        log.info("Multi-op maxDiff={}, meanDiff={}", maxDiff, meanDiff);

        assertTrue(maxDiff < 1e-2,
                "Multi-op graph: Triton output differs from C++. maxDiff=" + maxDiff);

        sd.close();
    }

    // ─── Test: MHA with ATTENTION include type explicitly set ────────────────

    @Test
    @DisplayName("Triton compileAll with ATTENTION type matches C++ reference")
    public void testTritonCompileAllWithAttention() {
        if (!tritonAvailable) {
            log.info("Skipping — Triton not available");
            return;
        }

        int batch = 1, numHeads = 9, headDim = 64, pastSeq = 100;
        int hidden = numHeads * headDim;
        int totalSeqKV = pastSeq + 1;

        Nd4j.getRandom().setSeed(42);
        INDArray query = Nd4j.randn(DataType.FLOAT, batch, 1, hidden);
        INDArray key = Nd4j.randn(DataType.FLOAT, batch, 1, hidden);
        INDArray value = Nd4j.randn(DataType.FLOAT, batch, 1, hidden);
        INDArray pastKey = Nd4j.randn(DataType.FLOAT, batch, numHeads, pastSeq, headDim);
        INDArray pastValue = Nd4j.randn(DataType.FLOAT, batch, numHeads, pastSeq, headDim);
        INDArray bias = Nd4j.zeros(DataType.FLOAT, batch, numHeads, 1, totalSeqKV);

        // C++ reference
        INDArray refOutput = executeMhaOp(query, key, value, bias, pastKey, pastValue, numHeads);

        // Triton with compileAll + ATTENTION
        SameDiff sd = SameDiff.create();
        SDVariable qVar = sd.placeHolder("query", DataType.FLOAT, query.shape());
        SDVariable kVar = sd.placeHolder("key", DataType.FLOAT, key.shape());
        SDVariable vVar = sd.placeHolder("value", DataType.FLOAT, value.shape());
        SDVariable bVar = sd.placeHolder("attn_bias", DataType.FLOAT, bias.shape());
        SDVariable pkVar = sd.placeHolder("past_key", DataType.FLOAT, pastKey.shape());
        SDVariable pvVar = sd.placeHolder("past_value", DataType.FLOAT, pastValue.shape());

        OnnxMultiHeadAttention mha = new OnnxMultiHeadAttention(
                sd, qVar, kVar, vVar, bVar, pkVar, pvVar,
                numHeads, 0.0, false, 1);
        SDVariable[] outputs = mha.outputVariables();

        // Enable Triton compileAll with ATTENTION
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);
        Nd4j.getEnvironment().setTritonIncludeTypes("CONST_GEN,GATHER,CONCAT,SPLIT,STACK,ATTENTION");
        Nd4j.getEnvironment().setTritonCompileAll(true);
        Nd4j.getEnvironment().setTritonSectionFusion(true);

        Map<String, INDArray> feedDict = new HashMap<>();
        feedDict.put("query", query.dup());
        feedDict.put("key", key.dup());
        feedDict.put("value", value.dup());
        feedDict.put("attn_bias", bias.dup());
        feedDict.put("past_key", pastKey.dup());
        feedDict.put("past_value", pastValue.dup());

        Map<String, INDArray> result = sd.output(feedDict, outputs[0].name());
        INDArray tritonOutput = result.get(outputs[0].name());
        assertNotNull(tritonOutput, "Triton output is null");

        assertFalse(Double.isNaN(tritonOutput.meanNumber().doubleValue()),
                "Triton output contains NaN");

        double maxDiff = refOutput.sub(tritonOutput).amaxNumber().doubleValue();
        log.info("compileAll+ATTENTION: maxDiff={}", maxDiff);
        assertTrue(maxDiff < 1e-2, "compileAll+ATTENTION: maxDiff=" + maxDiff);

        sd.close();
    }

    // ─── Test: Validate present_key content matches C++ reference ──────────

    /**
     * Validates that the Triton kernel's present_key output (output[1]) matches the
     * C++ reference. The C++ op produces present_key = concat(past_key, current_K)
     * in BHSD format [B, H, pastSeq+seqKVCur, D]. The Triton kernel must produce
     * the same content — both past positions AND the new current position.
     *
     * This test catches the bug where emitPresentKvWrite only writes the current K
     * at position pastSeq, leaving positions 0..pastSeq-1 uninitialized.
     */
    @Test
    @DisplayName("Present key/value content matches C++ reference (all positions)")
    public void testPresentKeyContentMatchesCppReference() {
        if (!tritonAvailable) {
            log.info("Skipping — Triton not available");
            return;
        }

        int batch = 1, numHeads = 4, headDim = 64, pastSeq = 20;
        int hidden = numHeads * headDim;

        Nd4j.getRandom().setSeed(42);
        INDArray query = Nd4j.randn(DataType.FLOAT, batch, 1, hidden);
        INDArray key = Nd4j.randn(DataType.FLOAT, batch, 1, hidden);
        INDArray value = Nd4j.randn(DataType.FLOAT, batch, 1, hidden);
        INDArray pastKey = Nd4j.randn(DataType.FLOAT, batch, numHeads, pastSeq, headDim);
        INDArray pastValue = Nd4j.randn(DataType.FLOAT, batch, numHeads, pastSeq, headDim);
        INDArray bias = Nd4j.zeros(DataType.FLOAT, batch, numHeads, 1, pastSeq + 1);

        // C++ reference: get ALL 3 outputs
        OnnxMultiHeadAttention refOp = new OnnxMultiHeadAttention(
                query.dup(), key.dup(), value.dup(),
                bias.dup(), pastKey.dup(), pastValue.dup(),
                numHeads, 0.0, false, 3);
        INDArray[] refOutputs = Nd4j.exec(refOp);
        INDArray refPresentKey = refOutputs[1].dup();
        INDArray refPresentValue = refOutputs[2].dup();

        log.info("C++ present_key shape={}, mean={}, absMax={}",
                java.util.Arrays.toString(refPresentKey.shape()),
                refPresentKey.meanNumber().doubleValue(),
                refPresentKey.amaxNumber().doubleValue());

        // Triton: get ALL 3 outputs
        INDArray[] tritonOutputs = executeMhaViaSameDiffAll(
                query, key, value, bias, pastKey, pastValue, numHeads, true);

        assertNotNull(tritonOutputs, "Triton outputs are null");
        assertTrue(tritonOutputs.length >= 3, "Expected 3 outputs, got " + tritonOutputs.length);
        assertNotNull(tritonOutputs[1], "Triton present_key is null");
        assertNotNull(tritonOutputs[2], "Triton present_value is null");

        INDArray tritonPresentKey = tritonOutputs[1];
        INDArray tritonPresentValue = tritonOutputs[2];

        log.info("Triton present_key shape={}, mean={}, absMax={}",
                java.util.Arrays.toString(tritonPresentKey.shape()),
                tritonPresentKey.meanNumber().doubleValue(),
                tritonPresentKey.amaxNumber().doubleValue());

        // Validate shapes match
        assertArrayEquals(refPresentKey.shape(), tritonPresentKey.shape(),
                "present_key shape mismatch: ref=" + java.util.Arrays.toString(refPresentKey.shape()) +
                " triton=" + java.util.Arrays.toString(tritonPresentKey.shape()));

        // Validate ALL positions in present_key (past + current)
        double pkMaxDiff = refPresentKey.sub(tritonPresentKey).amaxNumber().doubleValue();
        double pkMeanDiff = refPresentKey.sub(tritonPresentKey).ameanNumber().doubleValue();
        log.info("present_key: maxDiff={}, meanDiff={}", pkMaxDiff, pkMeanDiff);

        // Check past positions specifically (positions 0..pastSeq-1)
        if (refPresentKey.rank() == 4 && refPresentKey.size(2) > pastSeq) {
            INDArray refPast = refPresentKey.get(
                    org.nd4j.linalg.indexing.NDArrayIndex.all(),
                    org.nd4j.linalg.indexing.NDArrayIndex.all(),
                    org.nd4j.linalg.indexing.NDArrayIndex.interval(0, pastSeq),
                    org.nd4j.linalg.indexing.NDArrayIndex.all());
            INDArray tritonPast = tritonPresentKey.get(
                    org.nd4j.linalg.indexing.NDArrayIndex.all(),
                    org.nd4j.linalg.indexing.NDArrayIndex.all(),
                    org.nd4j.linalg.indexing.NDArrayIndex.interval(0, pastSeq),
                    org.nd4j.linalg.indexing.NDArrayIndex.all());
            double pastMaxDiff = refPast.sub(tritonPast).amaxNumber().doubleValue();
            log.info("present_key PAST positions [0..{}]: maxDiff={}", pastSeq - 1, pastMaxDiff);
            assertTrue(pastMaxDiff < 1e-4,
                    "present_key past positions differ! maxDiff=" + pastMaxDiff +
                    " — Triton kernel did not copy past keys into present_key output");
        }

        // Check current position (position pastSeq)
        if (refPresentKey.rank() == 4 && refPresentKey.size(2) > pastSeq) {
            INDArray refCur = refPresentKey.get(
                    org.nd4j.linalg.indexing.NDArrayIndex.all(),
                    org.nd4j.linalg.indexing.NDArrayIndex.all(),
                    org.nd4j.linalg.indexing.NDArrayIndex.point(pastSeq),
                    org.nd4j.linalg.indexing.NDArrayIndex.all());
            INDArray tritonCur = tritonPresentKey.get(
                    org.nd4j.linalg.indexing.NDArrayIndex.all(),
                    org.nd4j.linalg.indexing.NDArrayIndex.all(),
                    org.nd4j.linalg.indexing.NDArrayIndex.point(pastSeq),
                    org.nd4j.linalg.indexing.NDArrayIndex.all());
            double curMaxDiff = refCur.sub(tritonCur).amaxNumber().doubleValue();
            log.info("present_key CURRENT position [{}]: maxDiff={}", pastSeq, curMaxDiff);
            assertTrue(curMaxDiff < 1e-4,
                    "present_key current position differs! maxDiff=" + curMaxDiff);
        }

        assertTrue(pkMaxDiff < 1e-2,
                "present_key overall maxDiff=" + pkMaxDiff + " meanDiff=" + pkMeanDiff);

        // Also validate present_value
        double pvMaxDiff = refPresentValue.sub(tritonPresentValue).amaxNumber().doubleValue();
        log.info("present_value: maxDiff={}", pvMaxDiff);
        assertTrue(pvMaxDiff < 1e-2,
                "present_value maxDiff=" + pvMaxDiff);
    }

    // ─── Test: Multi-step decode using TRITON present_key for cache updates ─

    /**
     * Unlike testMultiStepDecode (which uses C++ present_key for cache updates),
     * this test uses the TRITON kernel's present_key output to update the KV cache.
     * This is what happens in the real VLM pipeline.
     *
     * If emitPresentKvWrite doesn't copy past keys, the cache accumulates garbage
     * and attention degrades rapidly across decode steps.
     */
    @Test
    @DisplayName("Multi-step decode using Triton present_key for cache updates")
    public void testMultiStepDecodeWithTritonPresentKey() {
        if (!tritonAvailable) {
            log.info("Skipping — Triton not available");
            return;
        }

        int batch = 1, numHeads = 4, headDim = 64;
        int hidden = numHeads * headDim;
        int numSteps = 5;

        Nd4j.getRandom().setSeed(55);

        // Start with some initial past context
        INDArray cppPastKey = Nd4j.randn(DataType.FLOAT, batch, numHeads, 10, headDim);
        INDArray cppPastValue = Nd4j.randn(DataType.FLOAT, batch, numHeads, 10, headDim);
        INDArray tritonPastKey = cppPastKey.dup();
        INDArray tritonPastValue = cppPastValue.dup();

        for (int step = 0; step < numSteps; step++) {
            int pastSeq = (int) cppPastKey.size(2);
            int totalSeq = pastSeq + 1;

            INDArray query = Nd4j.randn(DataType.FLOAT, batch, 1, hidden);
            INDArray key = Nd4j.randn(DataType.FLOAT, batch, 1, hidden);
            INDArray value = Nd4j.randn(DataType.FLOAT, batch, 1, hidden);
            INDArray bias = Nd4j.zeros(DataType.FLOAT, batch, numHeads, 1, totalSeq);

            // C++ reference: all 3 outputs
            OnnxMultiHeadAttention refOp = new OnnxMultiHeadAttention(
                    query.dup(), key.dup(), value.dup(),
                    bias.dup(), cppPastKey.dup(), cppPastValue.dup(),
                    numHeads, 0.0, false, 3);
            INDArray[] refOutputs = Nd4j.exec(refOp);
            INDArray refAttn = refOutputs[0];

            // Triton: all 3 outputs, using TRITON's own present_key for cache
            INDArray[] tritonOutputs = executeMhaViaSameDiffAll(
                    query, key, value, bias, tritonPastKey, tritonPastValue, numHeads, true);

            assertNotNull(tritonOutputs, "Step " + step + ": Triton outputs null");
            assertTrue(tritonOutputs.length >= 3,
                    "Step " + step + ": expected 3 outputs, got " + tritonOutputs.length);

            INDArray tritonAttn = tritonOutputs[0];

            double attnMaxDiff = refAttn.sub(tritonAttn).amaxNumber().doubleValue();
            log.info("Step {}: pastSeq={}, attn maxDiff={}", step, pastSeq, attnMaxDiff);
            assertTrue(attnMaxDiff < 1e-1,
                    "Step " + step + ": attention output diverged. maxDiff=" + attnMaxDiff +
                    " — KV cache likely corrupted by missing past key copy");

            // Update BOTH caches
            cppPastKey = refOutputs[1].dup();
            cppPastValue = refOutputs[2].dup();
            tritonPastKey = tritonOutputs[1].dup();
            tritonPastValue = tritonOutputs[2].dup();

            // Check that present_key matches between C++ and Triton
            double pkDiff = cppPastKey.sub(tritonPastKey).amaxNumber().doubleValue();
            log.info("Step {}: present_key maxDiff={}", step, pkDiff);
            assertTrue(pkDiff < 1e-2,
                    "Step " + step + ": present_key diverged. maxDiff=" + pkDiff);
        }
    }

    // ─── Test: Multi-section compileAll with matmul + MHA + add ────────────

    /**
     * Tests the FUSED_ATTENTION section inside a multi-section compiled kernel.
     * This mimics the VLM pipeline: matmul projections → MHA → residual add,
     * all compiled together with compileAll + section fusion.
     *
     * In single-section isolation, attention works. In multi-section, the phase
     * launch grid and section boundaries may cause issues.
     */
    @Test
    @DisplayName("Multi-section compileAll: matmul + MHA + add in one Triton kernel")
    public void testMultiSectionCompileAllWithMha() {
        if (!tritonAvailable) {
            log.info("Skipping — Triton not available");
            return;
        }

        int batch = 1, numHeads = 4, headDim = 64, pastSeq = 30;
        int hidden = numHeads * headDim;
        int totalSeqKV = pastSeq + 1;

        Nd4j.getRandom().setSeed(42);

        INDArray hiddenState = Nd4j.randn(DataType.FLOAT, batch, 1, hidden);
        INDArray qWeight = Nd4j.randn(DataType.FLOAT, hidden, hidden).divi(Math.sqrt(hidden));
        INDArray kWeight = Nd4j.randn(DataType.FLOAT, hidden, hidden).divi(Math.sqrt(hidden));
        INDArray vWeight = Nd4j.randn(DataType.FLOAT, hidden, hidden).divi(Math.sqrt(hidden));
        INDArray pastKey = Nd4j.randn(DataType.FLOAT, batch, numHeads, pastSeq, headDim);
        INDArray pastValue = Nd4j.randn(DataType.FLOAT, batch, numHeads, pastSeq, headDim);
        INDArray bias = Nd4j.zeros(DataType.FLOAT, batch, numHeads, 1, totalSeqKV);

        // ── C++ reference ──
        INDArray h2d = hiddenState.reshape(batch, hidden);
        INDArray qProj = h2d.mmul(qWeight).reshape(batch, 1, hidden);
        INDArray kProj = h2d.mmul(kWeight).reshape(batch, 1, hidden);
        INDArray vProj = h2d.mmul(vWeight).reshape(batch, 1, hidden);
        INDArray cppAttn = executeMhaOp(qProj, kProj, vProj, bias, pastKey, pastValue, numHeads);
        INDArray cppResidual = hiddenState.add(cppAttn);
        log.info("C++ multi-section ref: residual mean={}, absMax={}",
                cppResidual.meanNumber(), cppResidual.amaxNumber());

        // ── Triton: compileAll with section fusion ──
        SameDiff sd = SameDiff.create();
        SDVariable hVar = sd.placeHolder("hidden", DataType.FLOAT, batch, 1, hidden);
        SDVariable qwVar = sd.constant("qWeight", qWeight);
        SDVariable kwVar = sd.constant("kWeight", kWeight);
        SDVariable vwVar = sd.constant("vWeight", vWeight);
        SDVariable pkVar = sd.placeHolder("past_key", DataType.FLOAT, pastKey.shape());
        SDVariable pvVar = sd.placeHolder("past_value", DataType.FLOAT, pastValue.shape());
        SDVariable bVar = sd.placeHolder("bias", DataType.FLOAT, bias.shape());

        SDVariable q = sd.mmul("q_proj", hVar, qwVar);
        SDVariable k = sd.mmul("k_proj", hVar, kwVar);
        SDVariable v = sd.mmul("v_proj", hVar, vwVar);

        OnnxMultiHeadAttention mha = new OnnxMultiHeadAttention(
                sd, q, k, v, bVar, pkVar, pvVar,
                numHeads, 0.0, false, 1);
        SDVariable[] mhaOutputs = mha.outputVariables();
        SDVariable residual = hVar.add("residual", mhaOutputs[0]);

        // Enable compileAll with ALL types including MATMUL and ATTENTION
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);
        Nd4j.getEnvironment().setTritonIncludeTypes(
                "FUSED_ELEMENTWISE,IDENTITY,SCALAR,BROADCAST,CONSTANT_GENERATION,MATMUL,ATTENTION");
        Nd4j.getEnvironment().setTritonCompileAll(true);
        Nd4j.getEnvironment().setTritonSectionFusion(true);

        Map<String, INDArray> feedDict = new HashMap<>();
        feedDict.put("hidden", hiddenState.dup());
        feedDict.put("past_key", pastKey.dup());
        feedDict.put("past_value", pastValue.dup());
        feedDict.put("bias", bias.dup());

        Map<String, INDArray> result = sd.output(feedDict, "residual");
        INDArray tritonResidual = result.get("residual");
        assertNotNull(tritonResidual, "Triton multi-section residual is null");

        log.info("Triton multi-section: residual mean={}, absMax={}",
                tritonResidual.meanNumber(), tritonResidual.amaxNumber());

        assertFalse(Double.isNaN(tritonResidual.meanNumber().doubleValue()),
                "Triton multi-section output contains NaN");
        assertFalse(Double.isInfinite(tritonResidual.amaxNumber().doubleValue()),
                "Triton multi-section output contains Inf");

        double maxDiff = cppResidual.sub(tritonResidual).amaxNumber().doubleValue();
        double meanDiff = cppResidual.sub(tritonResidual).ameanNumber().doubleValue();
        log.info("Multi-section compileAll: maxDiff={}, meanDiff={}", maxDiff, meanDiff);

        assertTrue(maxDiff < 1e-1,
                "Multi-section compileAll: attention diverged from C++. maxDiff=" + maxDiff +
                " — this suggests a section integration issue (grid, phase, or cross-section data flow)");

        sd.close();
    }

    // ─── Test: 2-layer decoder-like graph with compileAll ──────────────────

    /**
     * Tests a mini decoder with 2 attention layers, each with QKV projection,
     * MHA, residual add, and a simple normalization-like reduce. This more
     * closely mimics the VLM structure to catch issues that only appear with
     * multiple attention sections in one compiled kernel.
     */
    @Test
    @DisplayName("2-layer decoder with compileAll: 2× (matmul + MHA + add + reduce)")
    public void testTwoLayerDecoderCompileAll() {
        if (!tritonAvailable) {
            log.info("Skipping — Triton not available");
            return;
        }

        int batch = 1, numHeads = 4, headDim = 64, pastSeq = 20;
        int hidden = numHeads * headDim;
        int totalSeqKV = pastSeq + 1;

        Nd4j.getRandom().setSeed(42);

        INDArray hiddenState = Nd4j.randn(DataType.FLOAT, batch, 1, hidden);
        // Layer 0 weights
        INDArray qWeight0 = Nd4j.randn(DataType.FLOAT, hidden, hidden).divi(Math.sqrt(hidden));
        INDArray kWeight0 = Nd4j.randn(DataType.FLOAT, hidden, hidden).divi(Math.sqrt(hidden));
        INDArray vWeight0 = Nd4j.randn(DataType.FLOAT, hidden, hidden).divi(Math.sqrt(hidden));
        INDArray pastKey0 = Nd4j.randn(DataType.FLOAT, batch, numHeads, pastSeq, headDim);
        INDArray pastValue0 = Nd4j.randn(DataType.FLOAT, batch, numHeads, pastSeq, headDim);
        // Layer 1 weights
        INDArray qWeight1 = Nd4j.randn(DataType.FLOAT, hidden, hidden).divi(Math.sqrt(hidden));
        INDArray kWeight1 = Nd4j.randn(DataType.FLOAT, hidden, hidden).divi(Math.sqrt(hidden));
        INDArray vWeight1 = Nd4j.randn(DataType.FLOAT, hidden, hidden).divi(Math.sqrt(hidden));
        INDArray pastKey1 = Nd4j.randn(DataType.FLOAT, batch, numHeads, pastSeq, headDim);
        INDArray pastValue1 = Nd4j.randn(DataType.FLOAT, batch, numHeads, pastSeq, headDim);
        INDArray bias = Nd4j.zeros(DataType.FLOAT, batch, numHeads, 1, totalSeqKV);

        // ── C++ reference: 2-layer decode ──
        INDArray h = hiddenState;
        // Layer 0
        INDArray h2d0 = h.reshape(batch, hidden);
        INDArray qP0 = h2d0.mmul(qWeight0).reshape(batch, 1, hidden);
        INDArray kP0 = h2d0.mmul(kWeight0).reshape(batch, 1, hidden);
        INDArray vP0 = h2d0.mmul(vWeight0).reshape(batch, 1, hidden);
        INDArray attn0 = executeMhaOp(qP0, kP0, vP0, bias, pastKey0, pastValue0, numHeads);
        h = h.add(attn0); // residual
        // Layer 1
        INDArray h2d1 = h.reshape(batch, hidden);
        INDArray qP1 = h2d1.mmul(qWeight1).reshape(batch, 1, hidden);
        INDArray kP1 = h2d1.mmul(kWeight1).reshape(batch, 1, hidden);
        INDArray vP1 = h2d1.mmul(vWeight1).reshape(batch, 1, hidden);
        INDArray attn1 = executeMhaOp(qP1, kP1, vP1, bias, pastKey1, pastValue1, numHeads);
        INDArray cppResult = h.add(attn1);

        log.info("C++ 2-layer ref: mean={}, absMax={}",
                cppResult.meanNumber(), cppResult.amaxNumber());

        // ── Triton: SameDiff with compileAll ──
        SameDiff sd = SameDiff.create();
        SDVariable hVar = sd.placeHolder("hidden", DataType.FLOAT, batch, 1, hidden);

        // Layer 0
        SDVariable qw0 = sd.constant("qw0", qWeight0);
        SDVariable kw0 = sd.constant("kw0", kWeight0);
        SDVariable vw0 = sd.constant("vw0", vWeight0);
        SDVariable pk0 = sd.placeHolder("pk0", DataType.FLOAT, pastKey0.shape());
        SDVariable pv0 = sd.placeHolder("pv0", DataType.FLOAT, pastValue0.shape());
        SDVariable bVar = sd.placeHolder("bias", DataType.FLOAT, bias.shape());

        SDVariable q0 = sd.mmul("q0", hVar, qw0);
        SDVariable k0 = sd.mmul("k0", hVar, kw0);
        SDVariable v0 = sd.mmul("v0", hVar, vw0);
        OnnxMultiHeadAttention mha0 = new OnnxMultiHeadAttention(
                sd, q0, k0, v0, bVar, pk0, pv0,
                numHeads, 0.0, false, 1);
        SDVariable residual0 = hVar.add("residual0", mha0.outputVariables()[0]);

        // Layer 1
        SDVariable qw1 = sd.constant("qw1", qWeight1);
        SDVariable kw1 = sd.constant("kw1", kWeight1);
        SDVariable vw1 = sd.constant("vw1", vWeight1);
        SDVariable pk1 = sd.placeHolder("pk1", DataType.FLOAT, pastKey1.shape());
        SDVariable pv1 = sd.placeHolder("pv1", DataType.FLOAT, pastValue1.shape());

        SDVariable q1 = sd.mmul("q1", residual0, qw1);
        SDVariable k1 = sd.mmul("k1", residual0, kw1);
        SDVariable v1 = sd.mmul("v1", residual0, vw1);
        OnnxMultiHeadAttention mha1 = new OnnxMultiHeadAttention(
                sd, q1, k1, v1, bVar, pk1, pv1,
                numHeads, 0.0, false, 1);
        SDVariable result2 = residual0.add("result", mha1.outputVariables()[0]);

        // CompileAll with all types
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);
        Nd4j.getEnvironment().setTritonIncludeTypes(
                "FUSED_ELEMENTWISE,IDENTITY,SCALAR,BROADCAST,CONSTANT_GENERATION,MATMUL,ATTENTION");
        Nd4j.getEnvironment().setTritonCompileAll(true);
        Nd4j.getEnvironment().setTritonSectionFusion(true);

        Map<String, INDArray> feedDict = new HashMap<>();
        feedDict.put("hidden", hiddenState.dup());
        feedDict.put("pk0", pastKey0.dup());
        feedDict.put("pv0", pastValue0.dup());
        feedDict.put("pk1", pastKey1.dup());
        feedDict.put("pv1", pastValue1.dup());
        feedDict.put("bias", bias.dup());

        Map<String, INDArray> out = sd.output(feedDict, "result");
        INDArray tritonResult = out.get("result");
        assertNotNull(tritonResult, "Triton 2-layer result is null");

        log.info("Triton 2-layer: mean={}, absMax={}",
                tritonResult.meanNumber(), tritonResult.amaxNumber());

        assertFalse(Double.isNaN(tritonResult.meanNumber().doubleValue()),
                "2-layer Triton output contains NaN");

        double maxDiff = cppResult.sub(tritonResult).amaxNumber().doubleValue();
        log.info("2-layer decoder compileAll: maxDiff={}", maxDiff);

        assertTrue(maxDiff < 1e-1,
                "2-layer decoder: attention diverged. maxDiff=" + maxDiff +
                " — multi-layer attention integration issue");

        sd.close();
    }

    // ─── Test: compileAll with frozen shapes (simulating graph capture path) ─

    /**
     * Tests attention compilation with frozen shapes, which is used in CUDA graph
     * capture. When shapes are frozen, the DSP pre-allocates output arrays and
     * reuses them across iterations. This tests whether the attention section
     * works correctly with shape freezing and repeated execution.
     */
    @Test
    @DisplayName("CompileAll with frozen shapes and repeated execution (2 iterations)")
    public void testCompileAllFrozenShapesRepeatedExec() {
        if (!tritonAvailable) {
            log.info("Skipping — Triton not available");
            return;
        }

        int batch = 1, numHeads = 9, headDim = 64, pastSeq = 100;
        int hidden = numHeads * headDim;  // 576 — SmolDocling hidden size
        int totalSeqKV = pastSeq + 1;

        Nd4j.getRandom().setSeed(42);

        INDArray hiddenState = Nd4j.randn(DataType.FLOAT, batch, 1, hidden);
        INDArray qWeight = Nd4j.randn(DataType.FLOAT, hidden, hidden).divi(Math.sqrt(hidden));
        INDArray kWeight = Nd4j.randn(DataType.FLOAT, hidden, hidden).divi(Math.sqrt(hidden));
        INDArray vWeight = Nd4j.randn(DataType.FLOAT, hidden, hidden).divi(Math.sqrt(hidden));
        INDArray pastKey = Nd4j.randn(DataType.FLOAT, batch, numHeads, pastSeq, headDim);
        INDArray pastValue = Nd4j.randn(DataType.FLOAT, batch, numHeads, pastSeq, headDim);
        INDArray bias = Nd4j.zeros(DataType.FLOAT, batch, numHeads, 1, totalSeqKV);

        // ── C++ reference ──
        INDArray h2d = hiddenState.reshape(batch, hidden);
        INDArray qProj = h2d.mmul(qWeight).reshape(batch, 1, hidden);
        INDArray kProj = h2d.mmul(kWeight).reshape(batch, 1, hidden);
        INDArray vProj = h2d.mmul(vWeight).reshape(batch, 1, hidden);
        INDArray cppAttn = executeMhaOp(qProj, kProj, vProj, bias, pastKey, pastValue, numHeads);
        INDArray cppResidual = hiddenState.add(cppAttn);

        // ── Triton: SameDiff with compileAll ──
        SameDiff sd = SameDiff.create();
        SDVariable hVar = sd.placeHolder("hidden", DataType.FLOAT, batch, 1, hidden);
        SDVariable qwVar = sd.constant("qWeight", qWeight);
        SDVariable kwVar = sd.constant("kWeight", kWeight);
        SDVariable vwVar = sd.constant("vWeight", vWeight);
        SDVariable pkVar = sd.placeHolder("past_key", DataType.FLOAT, pastKey.shape());
        SDVariable pvVar = sd.placeHolder("past_value", DataType.FLOAT, pastValue.shape());
        SDVariable bVar = sd.placeHolder("bias", DataType.FLOAT, bias.shape());

        SDVariable q = sd.mmul("q_proj", hVar, qwVar);
        SDVariable k = sd.mmul("k_proj", hVar, kwVar);
        SDVariable v = sd.mmul("v_proj", hVar, vwVar);
        OnnxMultiHeadAttention mha = new OnnxMultiHeadAttention(
                sd, q, k, v, bVar, pkVar, pvVar,
                numHeads, 0.0, false, 1);
        SDVariable residual = hVar.add("residual", mha.outputVariables()[0]);

        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);
        Nd4j.getEnvironment().setTritonIncludeTypes(
                "FUSED_ELEMENTWISE,IDENTITY,SCALAR,BROADCAST,CONSTANT_GENERATION,MATMUL,ATTENTION");
        Nd4j.getEnvironment().setTritonCompileAll(true);
        Nd4j.getEnvironment().setTritonSectionFusion(true);

        Map<String, INDArray> feedDict = new HashMap<>();
        feedDict.put("hidden", hiddenState.dup());
        feedDict.put("past_key", pastKey.dup());
        feedDict.put("past_value", pastValue.dup());
        feedDict.put("bias", bias.dup());

        // Run TWICE to exercise recompilation / cache reuse path
        for (int iter = 0; iter < 2; iter++) {
            Map<String, INDArray> result = sd.output(feedDict, "residual");
            INDArray tritonResidual = result.get("residual");
            assertNotNull(tritonResidual, "Iter " + iter + ": null");

            double maxDiff = cppResidual.sub(tritonResidual).amaxNumber().doubleValue();
            log.info("Iter {}: maxDiff={}, mean={}", iter, maxDiff,
                    tritonResidual.meanNumber().doubleValue());

            assertTrue(maxDiff < 1e-1,
                    "Iter " + iter + ": maxDiff=" + maxDiff);
        }

        sd.close();
    }

    // ─── Helper: execute MHA op directly (C++ reference) ────────────────────

    private INDArray executeMhaOp(INDArray query, INDArray key, INDArray value,
                                   INDArray attnBias, INDArray pastKey, INDArray pastValue,
                                   int numHeads) {
        OnnxMultiHeadAttention op = new OnnxMultiHeadAttention(
                query.dup(), key.dup(), value.dup(),
                attnBias != null ? attnBias.dup() : null,
                pastKey.dup(), pastValue.dup(),
                numHeads, 0.0, false, 3);
        INDArray[] outputs = Nd4j.exec(op);
        return outputs[0].dup();
    }

    // ─── Helper: execute MHA via SameDiff with Triton compilation ───────────

    private INDArray executeMhaViaSameDiff(INDArray query, INDArray key, INDArray value,
                                            INDArray attnBias, INDArray pastKey, INDArray pastValue,
                                            int numHeads, boolean enableTriton) {
        INDArray[] allOutputs = executeMhaViaSameDiffAll(
                query, key, value, attnBias, pastKey, pastValue, numHeads, enableTriton);
        return allOutputs != null && allOutputs.length > 0 ? allOutputs[0] : null;
    }

    // ─── Helper: execute MHA via SameDiff returning ALL outputs ──────────────

    private INDArray[] executeMhaViaSameDiffAll(INDArray query, INDArray key, INDArray value,
                                                INDArray attnBias, INDArray pastKey, INDArray pastValue,
                                                int numHeads, boolean enableTriton) {
        SameDiff sd = SameDiff.create();

        SDVariable qVar = sd.placeHolder("query", DataType.FLOAT, query.shape());
        SDVariable kVar = sd.placeHolder("key", DataType.FLOAT, key.shape());
        SDVariable vVar = sd.placeHolder("value", DataType.FLOAT, value.shape());

        SDVariable biasVar = attnBias != null
                ? sd.placeHolder("attn_bias", DataType.FLOAT, attnBias.shape())
                : null;

        SDVariable pastKeyVar = sd.placeHolder("past_key", DataType.FLOAT, pastKey.shape());
        SDVariable pastValueVar = sd.placeHolder("past_value", DataType.FLOAT, pastValue.shape());

        OnnxMultiHeadAttention mha = new OnnxMultiHeadAttention(
                sd, qVar, kVar, vVar,
                biasVar, pastKeyVar, pastValueVar,
                numHeads, 0.0, false, 3);
        SDVariable[] outputs = mha.outputVariables();

        // Configure execution mode
        if (enableTriton && tritonAvailable) {
            sd.setDspAutoCompileEnabled(true);
            sd.setDspNativeAutoCompileEnabled(true);
            Nd4j.getEnvironment().setTritonIncludeTypes("ATTENTION");
        }

        Map<String, INDArray> feedDict = new HashMap<>();
        feedDict.put("query", query.dup());
        feedDict.put("key", key.dup());
        feedDict.put("value", value.dup());
        if (attnBias != null) {
            feedDict.put("attn_bias", attnBias.dup());
        }
        feedDict.put("past_key", pastKey.dup());
        feedDict.put("past_value", pastValue.dup());

        // Request all outputs
        String[] outputNames = new String[outputs.length];
        for (int i = 0; i < outputs.length; i++) {
            outputNames[i] = outputs[i].name();
        }

        Map<String, INDArray> result = sd.output(feedDict, outputNames);
        INDArray[] ret = new INDArray[outputs.length];
        for (int i = 0; i < outputs.length; i++) {
            INDArray out = result.get(outputs[i].name());
            ret[i] = out != null ? out.dup() : null;
        }

        sd.close();
        return ret;
    }
}
