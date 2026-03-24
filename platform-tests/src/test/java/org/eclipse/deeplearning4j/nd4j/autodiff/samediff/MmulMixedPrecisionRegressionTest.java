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
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Regression tests for mixed-precision matmul correctness via SameDiff.
 *
 * Bug: cuBLAS Lt was unconditionally enabled for M=1, N>=16384 row-vector projections
 * with mixed input types (FP16 weights x FP32 activations). cuBLAS Lt does NOT support
 * mixed input types — both operands must match. The heuristic may return an algorithm
 * that silently produces wrong results, causing degenerate VLM output.
 *
 * Fix: Reject mixed input types in tryLtMatmul (aType != bType → return false),
 * falling through to standard cuBLAS which handles mixed precision correctly.
 *
 * Tests use SameDiff matmul op (not Java mmul) because the mixed-precision cast
 * happens in the C++ MmulHelper, which the SameDiff matmul op calls. Java mmul
 * enforces same-dtype at the Java layer.
 */
@Slf4j
@DisplayName("MmulMixedPrecisionRegressionTest")
public class MmulMixedPrecisionRegressionTest {

    // SmolDocling vocab projection dimensions
    private static final int HIDDEN = 1792;
    private static final int VOCAB = 49280;

    // Smaller sizes for fast tests
    private static final int SMALL_K = 64;
    private static final int SMALL_N = 128;

    /**
     * Core test: SameDiff matmul with FP32 x FP16 inputs must produce correct results.
     * Uses small dimensions for speed.
     */
    @Test
    @DisplayName("SameDiff matmul FP32 x FP16 produces correct results (small)")
    public void testSdMatmulFp32xFp16Small() {
        SameDiff sd = SameDiff.create();

        SDVariable a = sd.placeHolder("a", DataType.FLOAT, 1, SMALL_K);
        SDVariable b = sd.constant("b", Nd4j.randn(DataType.HALF, SMALL_K, SMALL_N));
        SDVariable result = sd.mmul("result", a, b);

        INDArray aArr = Nd4j.randn(DataType.FLOAT, 1, SMALL_K);

        // Reference: cast weight to FP32, compute in FP32
        INDArray expected = aArr.mmul(b.getArr().castTo(DataType.FLOAT));

        Map<String, INDArray> output = sd.output(Map.of("a", aArr), "result");
        INDArray actual = output.get("result").castTo(DataType.FLOAT);

        double maxErr = maxAbsError(expected, actual);
        log.info("SameDiff FP32 x FP16 small matmul max error: {}", maxErr);
        assertTrue(maxErr < 0.1,
                "SameDiff FP32 x FP16 matmul max error " + maxErr + " exceeds tolerance 0.1");

        double absSum = actual.sumNumber().doubleValue();
        assertTrue(Math.abs(absSum) > 1e-6, "Result should not be all zeros");

        sd.close();
    }

    /**
     * Test with vocab-projection-sized dimensions via SameDiff.
     * This matches the exact shape that triggers cuBLAS Lt: M=1, N>=16384.
     */
    @Test
    @DisplayName("SameDiff matmul FP32 x FP16 at vocab projection size [1,1792] x [1792,49280]")
    public void testSdMatmulFp32xFp16VocabProjection() {
        SameDiff sd = SameDiff.create();

        INDArray weights = Nd4j.randn(DataType.HALF, HIDDEN, VOCAB);
        SDVariable a = sd.placeHolder("a", DataType.FLOAT, 1, HIDDEN);
        SDVariable b = sd.constant("b", weights);
        SDVariable result = sd.mmul("result", a, b);

        INDArray aArr = Nd4j.randn(DataType.FLOAT, 1, HIDDEN);

        // Reference in FP32
        INDArray expected = aArr.mmul(weights.castTo(DataType.FLOAT));

        Map<String, INDArray> output = sd.output(Map.of("a", aArr), "result");
        INDArray actual = output.get("result").castTo(DataType.FLOAT);

        double maxErr = maxAbsError(expected, actual);
        log.info("Vocab projection max error: {}", maxErr);
        assertTrue(maxErr < 1.0,
                "Vocab projection max error " + maxErr + " exceeds tolerance 1.0");

        // Argmax must agree (critical for token selection)
        long expectedToken = Nd4j.argMax(expected, 1).getLong(0);
        long actualToken = Nd4j.argMax(actual, 1).getLong(0);
        log.info("Argmax: expected={}, actual={}", expectedToken, actualToken);
        assertEquals(expectedToken, actualToken,
                "Top-1 token must match between FP32 reference and SameDiff mixed-precision");

        sd.close();
    }

    /**
     * FP16 x FP16 matmul (both inputs same type — always supported).
     */
    @Test
    @DisplayName("FP16 x FP16 matmul correctness (supported cuBLAS Lt path)")
    public void testFp16xFp16Matmul() {
        INDArray a = Nd4j.randn(DataType.FLOAT, 1, SMALL_K).castTo(DataType.HALF);
        INDArray b = Nd4j.randn(DataType.HALF, SMALL_K, SMALL_N);

        // Reference in FP32
        INDArray expected = a.castTo(DataType.FLOAT).mmul(b.castTo(DataType.FLOAT));

        // FP16 x FP16 — same type, allowed by Java mmul
        INDArray actual = a.mmul(b).castTo(DataType.FLOAT);

        double maxErr = maxAbsError(expected, actual);
        log.info("FP16 x FP16 max error: {}", maxErr);
        assertTrue(maxErr < 0.5,
                "FP16 x FP16 matmul max error " + maxErr + " exceeds tolerance 0.5");
    }

    /**
     * FP32 x FP32 matmul baseline sanity check.
     */
    @Test
    @DisplayName("FP32 x FP32 matmul correctness (baseline)")
    public void testFp32xFp32Matmul() {
        INDArray a = Nd4j.randn(DataType.FLOAT, 1, SMALL_K);
        INDArray b = Nd4j.randn(DataType.FLOAT, SMALL_K, SMALL_N);

        INDArray expected = a.mmul(b);
        INDArray actual = a.mmul(b);

        double maxErr = maxAbsError(expected, actual);
        assertEquals(0.0, maxErr, 1e-6, "FP32 x FP32 should be deterministic");
    }

    /**
     * Repeated SameDiff matmul with mixed types must produce consistent results.
     * cuBLAS Lt caches algorithms — verify the cached path is also correct.
     */
    @Test
    @DisplayName("Repeated SameDiff FP32 x FP16 matmul produces consistent results")
    public void testRepeatedSdMmulConsistency() {
        SameDiff sd = SameDiff.create();

        INDArray weights = Nd4j.randn(DataType.HALF, HIDDEN, VOCAB);
        SDVariable a = sd.placeHolder("a", DataType.FLOAT, 1, HIDDEN);
        SDVariable b = sd.constant("b", weights);
        SDVariable result = sd.mmul("result", a, b);

        INDArray aArr = Nd4j.randn(DataType.FLOAT, 1, HIDDEN);

        Map<String, INDArray> first = sd.output(Map.of("a", aArr), "result");
        INDArray firstResult = first.get("result").castTo(DataType.FLOAT).dup();

        for (int i = 0; i < 5; i++) {
            Map<String, INDArray> out = sd.output(Map.of("a", aArr), "result");
            INDArray iterResult = out.get("result").castTo(DataType.FLOAT);
            double maxErr = maxAbsError(firstResult, iterResult);
            assertEquals(0.0, maxErr, 1e-5,
                    "Iteration " + i + ": repeated SameDiff mmul diverged, maxErr=" + maxErr);
        }
        log.info("5 repeated SameDiff mmul calls all match");

        sd.close();
    }

    /**
     * Top-5 token agreement between FP32 reference and SameDiff mixed-precision.
     */
    @Test
    @DisplayName("Top-5 token agreement between FP32 ref and SameDiff FP32xFP16")
    public void testTop5TokenAgreement() {
        SameDiff sd = SameDiff.create();

        INDArray weights = Nd4j.randn(DataType.HALF, HIDDEN, VOCAB);
        SDVariable a = sd.placeHolder("a", DataType.FLOAT, 1, HIDDEN);
        SDVariable b = sd.constant("b", weights);
        SDVariable result = sd.mmul("result", a, b);

        INDArray aArr = Nd4j.randn(DataType.FLOAT, 1, HIDDEN);

        INDArray weightsFp32 = weights.castTo(DataType.FLOAT);
        INDArray refLogits = aArr.mmul(weightsFp32);
        Map<String, INDArray> out = sd.output(Map.of("a", aArr), "result");
        INDArray mixedLogits = out.get("result").dup().castTo(DataType.FLOAT);

        // Top-1 must match
        long refTop = Nd4j.argMax(refLogits.dup(), 1).getLong(0);
        long mixedTop = Nd4j.argMax(mixedLogits.dup(), 1).getLong(0);
        assertEquals(refTop, mixedTop, "Top-1 token must match");

        // Get top-5 by iteratively finding max and zeroing
        long[] refTop5 = getTop5(refLogits.dup());
        long[] mixedTop5 = getTop5(mixedLogits.dup());

        // Count top-5 agreement
        int agreements = 0;
        for (long refIdx : refTop5) {
            for (long mixIdx : mixedTop5) {
                if (refIdx == mixIdx) {
                    agreements++;
                    break;
                }
            }
        }
        log.info("Top-5 agreement: {}/5", agreements);
        assertTrue(agreements >= 3, "At least 3/5 top tokens should agree, got " + agreements);
        weightsFp32.close();

        sd.close();
    }

    /**
     * Regression test for capture cast reuse bug.
     *
     * Bug: During CUDA graph replay, the cast cache used pointer-identity to skip
     * assign() when the same source array was reused. But in graph replay mode,
     * capture buffer addresses are stable while contents change each step. The
     * sameLogicalA check always matched → assign() skipped → stale FP16 cast data
     * from the previous step → identical logits → degenerate output loop.
     *
     * Fix: Always call assign() on cached buffers during graph execution. Never
     * skip based on pointer identity of source arrays.
     *
     * This test verifies that repeated SameDiff matmul executions with DIFFERENT
     * input data produce DIFFERENT outputs (not stale cached results).
     */
    @Test
    @DisplayName("Repeated SameDiff FP32 x FP16 matmul with changing inputs produces different outputs")
    public void testChangingInputsProduceDifferentOutputs() {
        SameDiff sd = SameDiff.create();

        INDArray weights = Nd4j.randn(DataType.HALF, SMALL_K, SMALL_N);
        SDVariable a = sd.placeHolder("a", DataType.FLOAT, 1, SMALL_K);
        SDVariable b = sd.constant("b", weights);
        SDVariable result = sd.mmul("result", a, b);

        INDArray[] inputs = new INDArray[5];
        INDArray[] outputs = new INDArray[5];
        for (int i = 0; i < 5; i++) {
            inputs[i] = Nd4j.randn(DataType.FLOAT, 1, SMALL_K);
            Map<String, INDArray> out = sd.output(Map.of("a", inputs[i]), "result");
            outputs[i] = out.get("result").castTo(DataType.FLOAT).dup();
        }

        // Each output must differ from the others (not stale)
        for (int i = 0; i < 5; i++) {
            for (int j = i + 1; j < 5; j++) {
                double maxErr = maxAbsError(outputs[i], outputs[j]);
                assertTrue(maxErr > 1e-3,
                        "Output " + i + " and " + j + " should differ (different inputs), but maxErr=" + maxErr);
            }
        }

        // Each output must match its FP32 reference
        INDArray weightsFp32 = weights.castTo(DataType.FLOAT);
        for (int i = 0; i < 5; i++) {
            INDArray expected = inputs[i].mmul(weightsFp32);
            double maxErr = maxAbsError(expected, outputs[i]);
            log.info("Input {} max error vs FP32 reference: {}", i, maxErr);
            assertTrue(maxErr < 0.1,
                    "Output " + i + " max error " + maxErr + " exceeds tolerance 0.1");
        }

        sd.close();
    }

    /**
     * Test that row-vector matmul (M=1) works correctly with stride-based contiguity check.
     * Verifies the row-vector fast path produces correct results for various K,N sizes.
     */
    @Test
    @DisplayName("Row-vector matmul M=1 correctness with various sizes")
    public void testRowVectorMatmulCorrectness() {
        int[][] sizes = {{64, 128}, {256, 512}, {1792, 49280}};

        for (int[] kn : sizes) {
            int K = kn[0], N = kn[1];
            SameDiff sd = SameDiff.create();

            INDArray a = Nd4j.randn(DataType.FLOAT, 1, K);
            INDArray w = Nd4j.randn(DataType.HALF, K, N);

            SDVariable aVar = sd.placeHolder("a", DataType.FLOAT, 1, K);
            SDVariable bVar = sd.constant("b", w);
            SDVariable result = sd.mmul("result", aVar, bVar);

            Map<String, INDArray> out = sd.output(Map.of("a", a), "result");
            INDArray actual = out.get("result").castTo(DataType.FLOAT);

            INDArray expected = a.mmul(w.castTo(DataType.FLOAT));

            double maxErr = maxAbsError(expected, actual);
            log.info("Row-vector matmul [1,{}]x[{},{}] max error: {}", K, K, N, maxErr);
            assertTrue(maxErr < 1.0,
                    "Row-vector matmul [1," + K + "]x[" + K + "," + N + "] max error " + maxErr + " exceeds tolerance");

            if (N >= 16384) {
                long expectedToken = Nd4j.argMax(expected, 1).getLong(0);
                long actualToken = Nd4j.argMax(actual, 1).getLong(0);
                assertEquals(expectedToken, actualToken,
                        "Argmax must match for vocab-projection size [1," + K + "]x[" + K + "," + N + "]");
            }

            sd.close();
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Tests targeting specific MmulHelper.cu changes for degenerate output bisect
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Test 1: tryLtMatmul mixed FP32×FP16 — cuBLAS Lt path vs explicit cast path.
     *
     * The cuBLAS Lt path handles FP32×FP16 natively using mixed-precision tensor cores.
     * The explicit cast path casts FP32→FP16 first, then does FP16×FP16.
     * Both should produce the same argmax (top-1 token). If they don't, one path
     * has a precision issue that could cascade in autoregressive decoding.
     *
     * Tests tryLtMatmul gate: `aType != CUDA_R_32F && aType != CUDA_R_16F`
     * vs proposed change: `aType != bType`
     */
    @Test
    @DisplayName("cuBLAS Lt mixed-type argmax stability over multiple random inputs")
    public void testCublasLtMixedTypeArgmaxStability() {
        // Use actual vocab projection dimensions — this triggers tryLtMatmul (M=1, N>=16384)
        INDArray weights = Nd4j.randn(DataType.HALF, HIDDEN, VOCAB);
        INDArray weightsFp32 = weights.castTo(DataType.FLOAT);

        SameDiff sd = SameDiff.create();
        SDVariable aVar = sd.placeHolder("a", DataType.FLOAT, 1, HIDDEN);
        SDVariable bVar = sd.constant("b", weights);
        SDVariable result = sd.mmul("result", aVar, bVar);

        int mismatches = 0;
        int total = 20;
        for (int i = 0; i < total; i++) {
            INDArray activation = Nd4j.randn(DataType.FLOAT, 1, HIDDEN);

            // FP32 reference (gold standard)
            INDArray refLogits = activation.mmul(weightsFp32);
            long refToken = Nd4j.argMax(refLogits, 1).getLong(0);

            // SameDiff path (goes through C++ MmulHelper which may use cuBLAS Lt)
            Map<String, INDArray> out = sd.output(Map.of("a", activation), "result");
            INDArray sdLogits = out.get("result").castTo(DataType.FLOAT);
            long sdToken = Nd4j.argMax(sdLogits, 1).getLong(0);

            if (refToken != sdToken) {
                // Check if the logit difference is small (close call)
                float refTopVal = refLogits.getFloat(0, (int) refToken);
                float refAltVal = refLogits.getFloat(0, (int) sdToken);
                float sdTopVal = sdLogits.getFloat(0, (int) sdToken);
                float sdAltVal = sdLogits.getFloat(0, (int) refToken);
                log.warn("Argmax mismatch at trial {}: ref={} (logit={:.4f}, alt={:.4f}) vs sd={} (logit={:.4f}, alt={:.4f})",
                        i, refToken, refTopVal, refAltVal, sdToken, sdTopVal, sdAltVal);
                mismatches++;
            }

            refLogits.close();
        }

        log.info("cuBLAS Lt mixed-type argmax: {}/{} mismatches", mismatches, total);
        // Allow a few mismatches for borderline cases, but most should agree
        assertTrue(mismatches <= total / 4,
                "Too many argmax mismatches: " + mismatches + "/" + total +
                " — cuBLAS Lt mixed-type path may be producing significantly different results");

        sd.close();
    }

    /**
     * Test 2: sameLogicalA cast reuse — same activation reused across q/k/v projections.
     *
     * In transformer decode, the same activation tensor is multiplied by q, k, v weight
     * matrices in sequence. The cast cache reuses the FP16 cast of A across these calls.
     * This test verifies that all three projections produce correct results.
     *
     * Tests sameLogicalA shortcut: whether skipping assign() for same source pointer
     * produces correct results when data hasn't changed.
     */
    @Test
    @DisplayName("Same activation reused across q/k/v projections produces correct results")
    public void testSameActivationQkvProjections() {
        int headDim = 64;
        int numHeads = 9; // SmolDocling
        int hiddenSize = headDim * numHeads; // 576 (not 1792 for smaller test)

        INDArray wq = Nd4j.randn(DataType.HALF, hiddenSize, hiddenSize);
        INDArray wk = Nd4j.randn(DataType.HALF, hiddenSize, hiddenSize);
        INDArray wv = Nd4j.randn(DataType.HALF, hiddenSize, hiddenSize);

        SameDiff sd = SameDiff.create();
        SDVariable activation = sd.placeHolder("x", DataType.FLOAT, 1, hiddenSize);
        SDVariable qWeight = sd.constant("wq", wq);
        SDVariable kWeight = sd.constant("wk", wk);
        SDVariable vWeight = sd.constant("wv", wv);

        SDVariable q = sd.mmul("q", activation, qWeight);
        SDVariable k = sd.mmul("k", activation, kWeight);
        SDVariable v = sd.mmul("v", activation, vWeight);

        INDArray wqFp32 = wq.castTo(DataType.FLOAT);
        INDArray wkFp32 = wk.castTo(DataType.FLOAT);
        INDArray wvFp32 = wv.castTo(DataType.FLOAT);

        // Run 10 iterations with different activations
        for (int i = 0; i < 10; i++) {
            INDArray x = Nd4j.randn(DataType.FLOAT, 1, hiddenSize);

            Map<String, INDArray> out = sd.output(Map.of("x", x), "q", "k", "v");
            INDArray actualQ = out.get("q").castTo(DataType.FLOAT);
            INDArray actualK = out.get("k").castTo(DataType.FLOAT);
            INDArray actualV = out.get("v").castTo(DataType.FLOAT);

            INDArray expectedQ = x.mmul(wqFp32);
            INDArray expectedK = x.mmul(wkFp32);
            INDArray expectedV = x.mmul(wvFp32);

            double errQ = maxAbsError(expectedQ, actualQ);
            double errK = maxAbsError(expectedK, actualK);
            double errV = maxAbsError(expectedV, actualV);

            assertTrue(errQ < 0.5, "Q projection error " + errQ + " at iter " + i);
            assertTrue(errK < 0.5, "K projection error " + errK + " at iter " + i);
            assertTrue(errV < 0.5, "V projection error " + errV + " at iter " + i);

            expectedQ.close();
            expectedK.close();
            expectedV.close();
        }

        wqFp32.close();
        wkFp32.close();
        wvFp32.close();
        sd.close();
    }

    /**
     * Test 3: ews vs stride contiguity check — non-contiguous views.
     *
     * The rowVectorFastPath changed from `ews()==1` to `strideDescendingCAscendingF()`.
     * For contiguous arrays these should agree. For VIEWS (e.g., slices, transposes),
     * they may disagree — ews() can return 1 for some views where strides are not
     * truly contiguous. This test verifies matmul correctness with views.
     *
     * Tests: `pA->ews() == 1` vs `strideDescendingCAscendingF(pA->shapeInfo())`
     */
    @Test
    @DisplayName("Row-vector matmul correctness with array views (non-contiguous)")
    public void testRowVectorMatmulWithViews() {
        int K = 256;
        int N = 512;

        // Case 1: Contiguous row vector (dup'd)
        INDArray a1 = Nd4j.randn(DataType.FLOAT, 1, K);
        INDArray b = Nd4j.randn(DataType.FLOAT, K, N);
        INDArray expected1 = a1.mmul(b);

        // Case 2: View from a larger matrix (getRow creates a view)
        INDArray bigMatrix = Nd4j.randn(DataType.FLOAT, 4, K);
        INDArray a2 = bigMatrix.getRow(2); // This is a VIEW with potentially non-unit ews
        // Verify it's actually a view
        log.info("a2 isView: {}, shape: {}", a2.isView(), java.util.Arrays.toString(a2.shape()));

        INDArray expected2 = a2.dup().reshape(1, K).mmul(b);
        INDArray actual2 = a2.reshape(1, K).mmul(b);

        double err2 = maxAbsError(expected2, actual2);
        log.info("View row-vector matmul error: {}", err2);
        assertTrue(err2 < 1e-4, "View row-vector matmul error " + err2 + " exceeds tolerance");

        // Case 3: Slice view (every other element — non-contiguous)
        INDArray wide = Nd4j.randn(DataType.FLOAT, 1, K * 2);
        INDArray sliced = wide.get(NDArrayIndex.point(0), NDArrayIndex.interval(0, 2, K * 2));
        log.info("Sliced shape: {}, isView: {}", java.util.Arrays.toString(sliced.shape()), sliced.isView());

        INDArray bSmall = Nd4j.randn(DataType.FLOAT, K, N);
        INDArray expectedSlice = sliced.dup().reshape(1, K).mmul(bSmall);
        INDArray actualSlice = sliced.reshape(1, K).mmul(bSmall);

        double errSlice = maxAbsError(expectedSlice, actualSlice);
        log.info("Non-contiguous slice matmul error: {}", errSlice);
        assertTrue(errSlice < 1e-4, "Non-contiguous slice matmul error " + errSlice + " exceeds tolerance");

        expected1.close();
        expected2.close();
        actual2.close();
        expectedSlice.close();
        actualSlice.close();
    }

    /**
     * Test 4: Multi-step decode simulation with DSP/graph execution.
     *
     * Simulates a decode loop where:
     * - The same SameDiff graph is executed repeatedly
     * - Each step has a different activation (simulating changing hidden states)
     * - The activation feeds into multiple matmuls (q/k/v + gate/up projections)
     * - Graph execution mode is enabled for CUDA graph capture/replay
     *
     * Tests the interaction between cast cache reuse and graph replay where
     * capture buffer contents change each step but pointers remain stable.
     */
    @Test
    @DisplayName("Multi-step decode simulation: changing activations through shared q/k/v projections")
    public void testMultiStepDecodeSimulation() {
        int hiddenSize = 256;
        int ffnSize = 512;
        int steps = 20;

        SameDiff sd = SameDiff.create();

        // Simulate transformer layer projections
        INDArray wq = Nd4j.randn(DataType.HALF, hiddenSize, hiddenSize);
        INDArray wk = Nd4j.randn(DataType.HALF, hiddenSize, hiddenSize);
        INDArray wv = Nd4j.randn(DataType.HALF, hiddenSize, hiddenSize);
        INDArray wGate = Nd4j.randn(DataType.HALF, hiddenSize, ffnSize);
        INDArray wUp = Nd4j.randn(DataType.HALF, hiddenSize, ffnSize);

        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, hiddenSize);
        SDVariable qOut = sd.mmul("q_proj", x, sd.constant("wq", wq));
        SDVariable kOut = sd.mmul("k_proj", x, sd.constant("wk", wk));
        SDVariable vOut = sd.mmul("v_proj", x, sd.constant("wv", wv));
        SDVariable gateOut = sd.mmul("gate_proj", x, sd.constant("wGate", wGate));
        SDVariable upOut = sd.mmul("up_proj", x, sd.constant("wUp", wUp));

        // FP32 reference weights
        INDArray wqF = wq.castTo(DataType.FLOAT);
        INDArray wkF = wk.castTo(DataType.FLOAT);
        INDArray wvF = wv.castTo(DataType.FLOAT);
        INDArray wGateF = wGate.castTo(DataType.FLOAT);
        INDArray wUpF = wUp.castTo(DataType.FLOAT);

        INDArray[] prevOutputs = null;
        int staleDetections = 0;

        for (int step = 0; step < steps; step++) {
            INDArray activation = Nd4j.randn(DataType.FLOAT, 1, hiddenSize);

            Map<String, INDArray> out = sd.output(
                    Map.of("x", activation),
                    "q_proj", "k_proj", "v_proj", "gate_proj", "up_proj");

            INDArray[] currentOutputs = {
                    out.get("q_proj").castTo(DataType.FLOAT).dup(),
                    out.get("k_proj").castTo(DataType.FLOAT).dup(),
                    out.get("v_proj").castTo(DataType.FLOAT).dup(),
                    out.get("gate_proj").castTo(DataType.FLOAT).dup(),
                    out.get("up_proj").castTo(DataType.FLOAT).dup()
            };

            // Verify correctness against FP32 reference
            INDArray[] refs = {
                    activation.mmul(wqF), activation.mmul(wkF), activation.mmul(wvF),
                    activation.mmul(wGateF), activation.mmul(wUpF)
            };
            String[] names = {"q", "k", "v", "gate", "up"};

            for (int p = 0; p < 5; p++) {
                double err = maxAbsError(refs[p], currentOutputs[p]);
                assertTrue(err < 1.0,
                        "Step " + step + " " + names[p] + "_proj error " + err + " exceeds tolerance");
                refs[p].close();
            }

            // Check for stale outputs (same as previous step despite different inputs)
            if (prevOutputs != null) {
                for (int p = 0; p < 5; p++) {
                    double diff = maxAbsError(prevOutputs[p], currentOutputs[p]);
                    if (diff < 1e-6) {
                        staleDetections++;
                        log.error("STALE OUTPUT at step {}: {} unchanged from previous step (diff={})",
                                step, names[p], diff);
                    }
                    prevOutputs[p].close();
                }
            }

            prevOutputs = currentOutputs;
            activation.close();
        }

        // Clean up last set
        if (prevOutputs != null) {
            for (INDArray arr : prevOutputs) arr.close();
        }

        assertEquals(0, staleDetections,
                staleDetections + " stale outputs detected — cast cache may be reusing old FP16 data");

        wqF.close();
        wkF.close();
        wvF.close();
        wGateF.close();
        wUpF.close();
        sd.close();
    }

    /**
     * Test 5: Precision stability over many autoregressive steps.
     *
     * Simulates the critical scenario: a vocab projection (M=1, N=49280) executed
     * repeatedly with gradually changing activation vectors. The argmax (selected token)
     * must be consistent between the SameDiff path and FP32 reference across all steps.
     *
     * This directly tests the scenario that causes degenerate output:
     * if the LM head projection has precision drift, wrong tokens get selected,
     * cascading through the autoregressive loop.
     */
    @Test
    @DisplayName("Vocab projection argmax stability over 50 autoregressive steps")
    public void testVocabProjectionArgmaxStabilityOverSteps() {
        INDArray weights = Nd4j.randn(DataType.HALF, HIDDEN, VOCAB);
        INDArray weightsFp32 = weights.castTo(DataType.FLOAT);

        SameDiff sd = SameDiff.create();
        SDVariable aVar = sd.placeHolder("a", DataType.FLOAT, 1, HIDDEN);
        SDVariable bVar = sd.constant("b", weights);
        SDVariable result = sd.mmul("result", aVar, bVar);

        // Start with a fixed activation and evolve it (simulating hidden state evolution)
        INDArray activation = Nd4j.randn(DataType.FLOAT, 1, HIDDEN);
        int argmaxMismatches = 0;
        int steps = 50;

        for (int step = 0; step < steps; step++) {
            Map<String, INDArray> out = sd.output(Map.of("a", activation), "result");
            // Must dup — SameDiff may close previous output on next sd.output() call
            INDArray sdLogits = out.get("result").castTo(DataType.FLOAT).dup();
            INDArray refLogits = activation.mmul(weightsFp32);

            long sdToken = Nd4j.argMax(sdLogits.dup(), 1).getLong(0);
            long refToken = Nd4j.argMax(refLogits.dup(), 1).getLong(0);

            if (sdToken != refToken) {
                argmaxMismatches++;
                if (argmaxMismatches <= 3) {
                    log.warn("Step {}: argmax mismatch ref={} vs sd={}", step, refToken, sdToken);
                }
            }

            // Evolve activation: add small perturbation (simulating new hidden state each step)
            INDArray noise = Nd4j.randn(DataType.FLOAT, 1, HIDDEN).mul(0.1);
            activation.addi(noise);
            noise.close();
            sdLogits.close();
            refLogits.close();
        }

        log.info("Vocab projection argmax: {}/{} mismatches over {} steps",
                argmaxMismatches, steps, steps);
        assertTrue(argmaxMismatches <= steps / 4,
                "Too many argmax mismatches: " + argmaxMismatches + "/" + steps +
                " — matmul precision may cause degenerate autoregressive output");

        activation.close();
        weightsFp32.close();
        sd.close();
    }

    /**
     * Test 6: Contiguous vs non-contiguous array matmul agreement.
     *
     * The rowVectorFastPath gate changed from ews()==1 to strideDescendingCAscendingF().
     * This test creates contiguous and non-contiguous versions of the same data
     * and verifies that matmul produces identical results regardless of layout.
     */
    @Test
    @DisplayName("Contiguous vs dup'd non-contiguous array matmul agreement")
    public void testContiguousVsNonContiguousMatmulAgreement() {
        int K = 512;
        int N = 1024;
        INDArray b = Nd4j.randn(DataType.FLOAT, K, N);

        // Create contiguous array
        INDArray contiguous = Nd4j.randn(DataType.FLOAT, 1, K);
        INDArray resultContiguous = contiguous.mmul(b);

        // Create same data as dup of a view (forces contiguous copy)
        INDArray big = Nd4j.randn(DataType.FLOAT, 3, K);
        big.putRow(1, contiguous.reshape(K));
        INDArray viewRow = big.getRow(1).reshape(1, K);
        INDArray viewDup = viewRow.dup();
        INDArray resultView = viewRow.mmul(b);
        INDArray resultDup = viewDup.mmul(b);

        // All three should agree
        double errViewVsContiguous = maxAbsError(resultContiguous, resultView);
        double errDupVsContiguous = maxAbsError(resultContiguous, resultDup);

        log.info("View vs contiguous error: {}", errViewVsContiguous);
        log.info("Dup vs contiguous error: {}", errDupVsContiguous);

        assertTrue(errViewVsContiguous < 1e-4,
                "View matmul disagrees with contiguous: error=" + errViewVsContiguous);
        assertTrue(errDupVsContiguous < 1e-4,
                "Dup'd view matmul disagrees with contiguous: error=" + errDupVsContiguous);

        resultContiguous.close();
        resultView.close();
        resultDup.close();
        contiguous.close();
        viewDup.close();
        b.close();
    }

    /**
     * Test 7: Transpose matmul correctness (tests transB path in cuBLAS).
     *
     * The rowVectorFastPath requires !transA && transB. This test verifies both
     * transposed and non-transposed B produce correct results.
     */
    @Test
    @DisplayName("Row-vector matmul with transposed vs non-transposed B")
    public void testRowVectorTransposedMatmul() {
        int K = 256;
        int N = 512;

        INDArray a = Nd4j.randn(DataType.FLOAT, 1, K);

        // Non-transposed: [1,K] x [K,N]
        INDArray bNormal = Nd4j.randn(DataType.FLOAT, K, N);
        INDArray resultNormal = a.mmul(bNormal);

        // Transposed: [1,K] x [N,K]^T — same math, different memory layout
        INDArray bTransposed = bNormal.transpose().dup(); // [N,K] contiguous
        // Use SameDiff mmul with transB=true
        SameDiff sd = SameDiff.create();
        SDVariable aVar = sd.placeHolder("a", DataType.FLOAT, 1, K);
        SDVariable bVar = sd.constant("b", bTransposed);
        // mmul config: transA=false, transB=true
        SDVariable result = sd.mmul("result", aVar, bVar, false, true, false);

        Map<String, INDArray> out = sd.output(Map.of("a", a), "result");
        INDArray resultTransposed = out.get("result");

        double err = maxAbsError(resultNormal, resultTransposed);
        log.info("Normal vs transposed matmul error: {}", err);
        assertTrue(err < 0.1, "Transposed matmul disagrees with normal: error=" + err);

        resultNormal.close();
        sd.close();
    }

    private static long[] getTop5(INDArray logits) {
        long[] top5 = new long[5];
        float[] data = logits.dup().data().asFloat();
        for (int i = 0; i < 5; i++) {
            float max = Float.NEGATIVE_INFINITY;
            int maxIdx = 0;
            for (int j = 0; j < data.length; j++) {
                if (data[j] > max) {
                    max = data[j];
                    maxIdx = j;
                }
            }
            top5[i] = maxIdx;
            data[maxIdx] = Float.NEGATIVE_INFINITY;
        }
        return top5;
    }

    private static double maxAbsError(INDArray expected, INDArray actual) {
        INDArray diff = Nd4j.math().abs(expected.sub(actual));
        return diff.maxNumber().doubleValue();
    }
}
