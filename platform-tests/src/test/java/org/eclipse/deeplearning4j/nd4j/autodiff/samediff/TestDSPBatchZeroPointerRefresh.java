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
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.Arrays;
import java.util.Collections;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for stale batch-zero pointer detection in DSP CUDA graph replay.
 *
 * <p>Background: {@code segBatchZeroEntries} in NativeDynamicShapePlan stores raw GPU pointers
 * ({@code void*}) captured during warmup. During CUDA graph replay, {@code cudaMemsetAsync} writes
 * to these pointers. If the underlying buffer was freed/reallocated between capture and replay,
 * the pointer is stale, causing CUDA error 700 (illegal memory access).</p>
 *
 * <p>The fix adds {@code outputSlotIndex} to {@code BatchZeroEntry} so pointers can be refreshed
 * from {@code slotArrayCache_} before replay. These tests verify that batch-zero pointer refresh
 * works correctly across multiple frozen replay calls with various graph patterns.</p>
 *
 * <p>Key assertion strategy: results on call 4+ (first replay) must match results on call 2
 * (first frozen). NaN or divergence indicates stale/garbage data from batch-zero writing to
 * freed memory.</p>
 */
@Slf4j
@Tag("samediff")
public class TestDSPBatchZeroPointerRefresh extends BaseNd4jTestWithBackends {

    private static final double TOL = 1e-4;

    @Override
    public char ordering() {
        return 'c';
    }

    /**
     * Enable DSP auto-compilation on a SameDiff instance so that
     * outputDirect() goes through the DSP execution path.
     */
    private void enableDsp(SameDiff sd) {
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);
    }

    /**
     * Assert two arrays match within tolerance, with a descriptive message.
     */
    private void assertClose(INDArray expected, INDArray actual, String label) {
        assertArrayEquals(expected.shape(), actual.shape(),
                label + ": shape mismatch - expected " + Arrays.toString(expected.shape())
                        + " got " + Arrays.toString(actual.shape()));
        double maxDiff = expected.sub(actual).amaxNumber().doubleValue();
        assertTrue(maxDiff < TOL, label + ": max diff " + maxDiff + " exceeds tolerance " + TOL);
    }

    // ========================================================================
    // Test 1: Frozen Replay Multiple Calls (Baseline)
    // ========================================================================

    @Test
    @DisplayName("Batch-zero pointer refresh: frozen replay across 8 calls with matmul+add+relu")
    public void testFrozenReplayMultipleCalls() {
        // Basic frozen replay test. Create a simple graph (matmul + add + relu),
        // execute with frozen shapes 8 times. If batch-zero refresh works correctly,
        // all calls produce identical correct results. If stale pointers are used,
        // calls 3+ (replay) will produce NaN or garbage.
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, 64);
        SDVariable weights = sd.constant("weights", Nd4j.randn(DataType.FLOAT, 64, 64).muli(0.1));
        SDVariable bias = sd.constant("bias", Nd4j.zeros(DataType.FLOAT, 1, 64));

        SDVariable mm = sd.mmul("mm", input, weights);
        SDVariable added = mm.add("added", bias);
        SDVariable output = sd.nn.relu("output", added, 0);

        INDArray inputData = Nd4j.randn(DataType.FLOAT, 1, 64);

        // Get ground truth from standard execution
        Map<String, INDArray> standardResult = sd.output(
                Collections.singletonMap("input", inputData), "output");
        INDArray expected = standardResult.get("output").dup();

        // Enable DSP and execute multiple times with frozen shapes
        enableDsp(sd);

        INDArray firstDspResult = null;
        for (int call = 0; call < 8; call++) {
            Map<String, INDArray> result = sd.outputDirect(
                    Collections.singletonMap("input", inputData), "output");
            INDArray actual = result.get("output").dup();

            // Must not produce NaN (stale pointer detection)
            assertFalse(actual.isNaN().any(),
                    "Call " + call + ": frozen replay produced NaN — likely stale batch-zero pointer");

            // Must match standard execution
            assertClose(expected, actual, "Call " + call);

            // All DSP calls must produce identical results for identical inputs
            if (firstDspResult == null) {
                firstDspResult = actual;
            } else {
                double iterDiff = firstDspResult.sub(actual).amaxNumber().doubleValue();
                assertEquals(0.0, iterDiff, TOL,
                        "Call " + call + ": DSP result changed from call 0 — diff = " + iterDiff
                                + " — possible stale batch-zero data");
            }

            log.info("Call {}: max diff from reference = {}", call,
                    expected.sub(actual).amaxNumber().doubleValue());
        }

        sd.close();
    }

    // ========================================================================
    // Test 2: Frozen Replay With In-Place Input Update
    // ========================================================================

    @Test
    @DisplayName("Batch-zero pointer refresh: different input data between frozen calls")
    public void testFrozenReplayWithInPlaceInputUpdate() {
        // Create a graph, enable DSP, then provide DIFFERENT input data between
        // frozen calls (same shape, different values). Verifies that capture buffer
        // D2D copies work alongside batch-zero refresh — outputs must change correctly
        // with the new inputs. Stale batch-zero pointers would produce results that
        // don't correspond to the current input.
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 2, 32);
        SDVariable w = sd.constant("w", Nd4j.randn(DataType.FLOAT, 32, 16).muli(0.1));
        SDVariable b = sd.constant("b", Nd4j.zeros(DataType.FLOAT, 2, 16));

        SDVariable mm = sd.mmul("mm", input, w);
        SDVariable added = mm.add("added", b);
        SDVariable output = sd.nn.relu("output", added, 0);

        enableDsp(sd);

        INDArray[] results = new INDArray[6];
        INDArray[] inputs = new INDArray[6];

        for (int call = 0; call < 6; call++) {
            // Generate different input data each call (same shape)
            inputs[call] = Nd4j.randn(DataType.FLOAT, 2, 32).muli(call + 1);

            // Get standard reference for this specific input
            Map<String, INDArray> standardResult = sd.output(
                    Collections.singletonMap("input", inputs[call]), "output");
            INDArray expected = standardResult.get("output").dup();

            // Get DSP result
            Map<String, INDArray> dspResult = sd.outputDirect(
                    Collections.singletonMap("input", inputs[call]), "output");
            results[call] = dspResult.get("output").dup();

            // Must not produce NaN
            assertFalse(results[call].isNaN().any(),
                    "Call " + call + ": input update produced NaN — batch-zero pointer likely stale");

            // Must match standard execution for THIS input
            assertClose(expected, results[call], "Call " + call + " (input update)");

            log.info("Call {}: max diff from reference = {}", call,
                    expected.sub(results[call]).amaxNumber().doubleValue());
        }

        // Verify results are actually different from each other (not stale copies)
        for (int i = 1; i < 6; i++) {
            double diff = results[i].sub(results[0]).amaxNumber().doubleValue();
            assertTrue(diff > 1e-6,
                    "Results " + i + " and 0 are identical — output not responding to input change, "
                            + "possible stale batch-zero data");
        }

        sd.close();
    }

    // ========================================================================
    // Test 3: Frozen Replay Attention Pattern
    // ========================================================================

    @Test
    @DisplayName("Batch-zero pointer refresh: attention pattern (matmul->scale->softmax->matmul)")
    public void testFrozenReplayAttentionPattern() {
        // Mimics the attention pattern: Q*K^T -> scale -> softmax -> *V
        // This is the exact pattern of seg[1320-1323] where the original CUDA error
        // 700 occurred. The softmax creates a segment boundary (gap op), and the
        // batch-zero entries for the matmul outputs must survive across replay.
        SameDiff sd = SameDiff.create();

        int seqLen = 8;
        int headDim = 16;

        SDVariable query = sd.placeHolder("query", DataType.FLOAT, 1, seqLen, headDim);
        SDVariable key = sd.placeHolder("key", DataType.FLOAT, 1, seqLen, headDim);
        SDVariable value = sd.placeHolder("value", DataType.FLOAT, 1, seqLen, headDim);

        // Q * K^T -> [1, seqLen, seqLen]
        SDVariable keyT = sd.permute("keyT", key, 0, 2, 1);
        SDVariable scores = sd.mmul("scores", query, keyT);

        // Scale by 1/sqrt(headDim)
        float scale = (float) (1.0 / Math.sqrt(headDim));
        SDVariable scaled = scores.mul("scaled", sd.constant("scale", Nd4j.scalar(DataType.FLOAT, scale)));

        // Softmax over last dimension
        SDVariable attnWeights = sd.nn.softmax("attnWeights", scaled, -1);

        // Attention output: weights * V -> [1, seqLen, headDim]
        SDVariable output = sd.mmul("output", attnWeights, value);

        // Create inputs
        INDArray queryArr = Nd4j.randn(DataType.FLOAT, 1, seqLen, headDim).muli(0.1);
        INDArray keyArr = Nd4j.randn(DataType.FLOAT, 1, seqLen, headDim).muli(0.1);
        INDArray valueArr = Nd4j.randn(DataType.FLOAT, 1, seqLen, headDim).muli(0.1);

        Map<String, INDArray> feed = Map.of("query", queryArr, "key", keyArr, "value", valueArr);

        // Get standard reference
        Map<String, INDArray> standardResult = sd.output(feed, "output");
        INDArray expected = standardResult.get("output").dup();

        enableDsp(sd);

        INDArray firstDspResult = null;
        for (int call = 0; call < 6; call++) {
            Map<String, INDArray> result = sd.outputDirect(feed, "output");
            INDArray actual = result.get("output").dup();

            // Must not produce NaN — the attention softmax path is where
            // the original stale batch-zero pointer bug manifested
            assertFalse(actual.isNaN().any(),
                    "Call " + call + ": attention pattern produced NaN — "
                            + "stale batch-zero pointer in softmax/matmul segment");

            // Relaxed tolerance for attention pattern (softmax magnifies small differences)
            float maxDiff = Transforms.abs(actual.sub(expected)).maxNumber().floatValue();
            assertTrue(maxDiff < 0.01f,
                    "Call " + call + ": attention result too far from reference: maxDiff=" + maxDiff);

            if (firstDspResult == null) {
                firstDspResult = actual;
            } else {
                double iterDiff = firstDspResult.sub(actual).amaxNumber().doubleValue();
                assertEquals(0.0, iterDiff, 1e-5,
                        "Call " + call + ": attention output changed from call 0 — diff = " + iterDiff);
            }

            log.info("Call {}: attention max diff from reference = {}", call, maxDiff);
        }

        sd.close();
    }

    // ========================================================================
    // Test 4: Frozen Replay Large Graph (10+ ops, multiple segments)
    // ========================================================================

    @Test
    @DisplayName("Batch-zero pointer refresh: deep 5-layer network (12+ ops, multiple segments)")
    public void testFrozenReplayLargeGraph() {
        // Create a deeper graph with 10+ ops (chain of matmul + add + activation layers).
        // This creates multiple segments, each with its own batch-zero entries. All
        // segments' batch-zero pointers must remain valid across replays.
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 32);

        // Layer 1: matmul + bias + relu
        SDVariable w1 = sd.constant("w1", Nd4j.randn(DataType.FLOAT, 32, 64).muli(0.05));
        SDVariable b1 = sd.constant("b1", Nd4j.zeros(DataType.FLOAT, 2, 64));
        SDVariable h1 = sd.nn.relu("h1", sd.mmul("mm1", x, w1).add("add1", b1), 0);

        // Layer 2: matmul + bias + sigmoid (gap op — segment boundary)
        SDVariable w2 = sd.constant("w2", Nd4j.randn(DataType.FLOAT, 64, 64).muli(0.05));
        SDVariable b2 = sd.constant("b2", Nd4j.zeros(DataType.FLOAT, 2, 64));
        SDVariable h2 = sd.nn.sigmoid("h2", sd.mmul("mm2", h1, w2).add("add2", b2));

        // Layer 3: matmul + bias + relu
        SDVariable w3 = sd.constant("w3", Nd4j.randn(DataType.FLOAT, 64, 32).muli(0.05));
        SDVariable b3 = sd.constant("b3", Nd4j.zeros(DataType.FLOAT, 2, 32));
        SDVariable h3 = sd.nn.relu("h3", sd.mmul("mm3", h2, w3).add("add3", b3), 0);

        // Layer 4: matmul + bias + tanh
        SDVariable w4 = sd.constant("w4", Nd4j.randn(DataType.FLOAT, 32, 16).muli(0.05));
        SDVariable b4 = sd.constant("b4", Nd4j.zeros(DataType.FLOAT, 2, 16));
        SDVariable h4 = sd.nn.tanh("h4", sd.mmul("mm4", h3, w4).add("add4", b4));

        // Layer 5: matmul + bias (output)
        SDVariable w5 = sd.constant("w5", Nd4j.randn(DataType.FLOAT, 16, 8).muli(0.05));
        SDVariable b5 = sd.constant("b5", Nd4j.zeros(DataType.FLOAT, 2, 8));
        SDVariable output = sd.mmul("mm5", h4, w5).add("output", b5);

        INDArray inputArr = Nd4j.randn(DataType.FLOAT, 2, 32);

        // Get ground truth
        Map<String, INDArray> standardResult = sd.output(
                Collections.singletonMap("x", inputArr), "output");
        INDArray expected = standardResult.get("output").dup();

        enableDsp(sd);

        for (int call = 0; call < 6; call++) {
            Map<String, INDArray> result = sd.outputDirect(
                    Collections.singletonMap("x", inputArr), "output");
            INDArray actual = result.get("output").dup();

            assertFalse(actual.isNaN().any(),
                    "Call " + call + ": deep network produced NaN — "
                            + "stale batch-zero pointer in one of 5 layers/segments");

            // Use relaxed tolerance for deep network (numerical differences accumulate)
            float maxDiff = Transforms.abs(actual.sub(expected)).maxNumber().floatValue();
            assertTrue(maxDiff < 0.05f,
                    "Call " + call + ": deep network result too far from reference: maxDiff=" + maxDiff);

            log.info("Call {}: deep network max diff = {}", call, maxDiff);
        }

        sd.close();
    }

    // ========================================================================
    // Test 5: Frozen Replay With Intermediate Zeros
    // ========================================================================

    @Test
    @DisplayName("Batch-zero pointer refresh: intermediate accumulators must be zeroed each call")
    public void testFrozenReplayWithIntermediateZeros() {
        // Create a graph where intermediate results MUST be zeroed before each execution
        // (reduction/accumulation patterns). If batch-zero fails to zero correctly,
        // stale data from the previous call leaks through, and the output will differ
        // between calls even with the same input — or worse, differ between calls with
        // DIFFERENT inputs because old data contaminates the accumulator.
        //
        // We verify this by executing with DIFFERENT inputs each time and checking that
        // each result matches a fresh standard execution of the same graph.
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 4, 16);

        // Two parallel branches that both reduce, then combine
        // Branch A: matmul -> reduce_sum -> [4, 1]
        SDVariable wA = sd.constant("wA", Nd4j.randn(DataType.FLOAT, 16, 8).muli(0.1));
        SDVariable mmA = sd.mmul("mmA", x, wA);
        SDVariable sumA = sd.sum("sumA", mmA, true, 1); // [4, 1] — accumulator must be zeroed

        // Branch B: matmul -> reduce_mean -> [4, 1]
        SDVariable wB = sd.constant("wB", Nd4j.randn(DataType.FLOAT, 16, 8).muli(0.1));
        SDVariable mmB = sd.mmul("mmB", x, wB);
        SDVariable meanB = sd.mean("meanB", mmB, true, 1); // [4, 1] — accumulator must be zeroed

        // Combine: broadcast multiply
        SDVariable combined = sumA.mul("combined", meanB); // [4, 1]

        // Final matmul to produce wider output
        SDVariable wOut = sd.constant("wOut", Nd4j.randn(DataType.FLOAT, 1, 4).muli(0.1));
        SDVariable output = sd.mmul("output", combined, wOut); // [4, 4]

        enableDsp(sd);

        INDArray[] results = new INDArray[6];
        for (int call = 0; call < 6; call++) {
            // Different input each call to detect stale accumulator data
            INDArray inputArr = Nd4j.randn(DataType.FLOAT, 4, 16).muli(0.5 * (call + 1));

            // Fresh standard execution for reference
            Map<String, INDArray> standardResult = sd.output(
                    Collections.singletonMap("x", inputArr), "output");
            INDArray expected = standardResult.get("output").dup();

            // DSP execution
            Map<String, INDArray> dspResult = sd.outputDirect(
                    Collections.singletonMap("x", inputArr), "output");
            results[call] = dspResult.get("output").dup();

            // Must not produce NaN
            assertFalse(results[call].isNaN().any(),
                    "Call " + call + ": accumulator pattern produced NaN — "
                            + "batch-zero likely failed to zero reduction accumulator");

            // Must match standard execution (stale data would cause mismatch)
            float maxDiff = Transforms.abs(results[call].sub(expected)).maxNumber().floatValue();
            assertTrue(maxDiff < 0.01f,
                    "Call " + call + ": accumulator result too far from reference: maxDiff=" + maxDiff
                            + " — stale data may be leaking from previous call's accumulator");

            log.info("Call {}: accumulator max diff from reference = {}", call, maxDiff);
        }

        // Verify results are actually different across calls (stale accumulator would
        // produce same or drifting results regardless of input)
        for (int i = 1; i < 6; i++) {
            double diff = results[i].sub(results[0]).amaxNumber().doubleValue();
            assertTrue(diff > 1e-6,
                    "Results " + i + " and 0 are identical despite different inputs — "
                            + "accumulator batch-zero is not working, stale data persists");
        }

        sd.close();
    }

    // ========================================================================
    // Test 6: Frozen Replay With Softmax + Residual (Complex Segment Pattern)
    // ========================================================================

    @Test
    @DisplayName("Batch-zero pointer refresh: softmax + residual creates complex segment boundaries")
    public void testFrozenReplaySoftmaxResidual() {
        // Softmax is a gap op that creates segment boundaries. A residual connection
        // that bypasses the softmax creates cross-segment data dependencies. Both the
        // softmax branch and the residual branch need valid batch-zero pointers.
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 2, 32);
        SDVariable w1 = sd.constant("w1", Nd4j.randn(DataType.FLOAT, 32, 32).muli(0.05));

        // Main path: matmul -> softmax (gap op, segment boundary)
        SDVariable mm = sd.mmul("mm", input, w1);
        SDVariable soft = sd.nn.softmax("soft", mm, -1);

        // Residual: add input to softmax output
        SDVariable residual = soft.add("residual", input);

        // Final projection
        SDVariable w2 = sd.constant("w2", Nd4j.randn(DataType.FLOAT, 32, 16).muli(0.05));
        SDVariable output = sd.mmul("output", residual, w2);

        INDArray inputArr = Nd4j.randn(DataType.FLOAT, 2, 32).muli(0.1);

        // Standard reference
        Map<String, INDArray> standardResult = sd.output(
                Collections.singletonMap("input", inputArr), "output");
        INDArray expected = standardResult.get("output").dup();

        enableDsp(sd);

        for (int call = 0; call < 6; call++) {
            Map<String, INDArray> result = sd.outputDirect(
                    Collections.singletonMap("input", inputArr), "output");
            INDArray actual = result.get("output").dup();

            assertFalse(actual.isNaN().any(),
                    "Call " + call + ": softmax+residual produced NaN — "
                            + "batch-zero pointer stale in cross-segment residual path");

            float maxDiff = Transforms.abs(actual.sub(expected)).maxNumber().floatValue();
            assertTrue(maxDiff < 0.01f,
                    "Call " + call + ": softmax+residual too far from reference: maxDiff=" + maxDiff);
        }

        sd.close();
    }

    // ========================================================================
    // Test 7: Multiple Plan Lifecycles (create, execute, close, repeat)
    // ========================================================================

    @Test
    @DisplayName("Batch-zero pointer refresh: survives plan lifecycle destroy/recreate")
    public void testBatchZeroAcrossPlanLifecycles() {
        // Create and destroy multiple DSP plans. Each lifecycle gets new GPU allocations,
        // so batch-zero pointers from a previous lifecycle are definitely invalid. This
        // tests that each new plan properly initializes its batch-zero pointers rather
        // than inheriting stale ones.
        for (int lifecycle = 0; lifecycle < 3; lifecycle++) {
            SameDiff sd = SameDiff.create();
            SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 16);
            SDVariable w1 = sd.constant("w1", Nd4j.randn(DataType.FLOAT, 16, 32).muli(0.1));
            SDVariable w2 = sd.constant("w2", Nd4j.randn(DataType.FLOAT, 32, 8).muli(0.1));

            SDVariable h = sd.nn.relu("h", sd.mmul("mm1", x, w1), 0);
            SDVariable output = sd.mmul("output", h, w2);

            INDArray inputArr = Nd4j.randn(DataType.FLOAT, 2, 16);

            Map<String, INDArray> standardResult = sd.output(
                    Collections.singletonMap("x", inputArr), "output");
            INDArray expected = standardResult.get("output").dup();

            enableDsp(sd);

            for (int call = 0; call < 5; call++) {
                Map<String, INDArray> result = sd.outputDirect(
                        Collections.singletonMap("x", inputArr), "output");
                INDArray actual = result.get("output").dup();

                assertFalse(actual.isNaN().any(),
                        "Lifecycle " + lifecycle + " call " + call + ": produced NaN");

                assertClose(expected, actual,
                        "Lifecycle " + lifecycle + " call " + call);
            }

            sd.close();
            log.info("Lifecycle {} completed successfully", lifecycle);
        }
    }
}
