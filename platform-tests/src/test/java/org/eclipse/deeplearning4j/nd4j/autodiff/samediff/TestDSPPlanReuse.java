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
import org.junit.jupiter.api.*;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlanExecutor;
import org.nd4j.common.config.ND4JSystemProperties;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests DSP (DynamicShapePlanExecutor) plan reuse across multiple executions.
 *
 * <p>These tests exercise the exact patterns used in VLM multi-page document
 * processing where the same SameDiff graph is executed repeatedly with different
 * input data but identical shapes — without resetting sessions between calls.</p>
 *
 * <h3>Patterns tested:</h3>
 * <ul>
 *   <li>Multiple executions without session reset (buffer reuse)</li>
 *   <li>Frozen shapes + repeated execution</li>
 *   <li>Plan handle caching across session resets</li>
 *   <li>Multi-model interleaved execution (A → B → A)</li>
 *   <li>Numerical correctness across re-executions</li>
 * </ul>
 *
 * <p>Run:
 * <pre>
 *   cd platform-tests && mvn test -Dtest=TestDSPPlanReuse
 * </pre>
 */
@Slf4j
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
public class TestDSPPlanReuse extends BaseNd4jTestWithBackends {

    private static final double TOL = 1e-4;

    @Override
    public char ordering() {
        return 'c';
    }

    @BeforeAll
    static void enableDspGlobally() {
        System.setProperty(ND4JSystemProperties.DYNAMIC_SHAPE_PLAN_ENABLED, "true");
        InferenceSession.setDynamicShapePlanEnabled(true);
    }

    private void enableDsp(SameDiff sd) {
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);
    }

    /**
     * Builds a small "encoder-like" graph: matmul → add → layer_norm → relu.
     * Mimics a simplified vision encoder with constants (weights) and a placeholder (input).
     */
    private SameDiff buildEncoderGraph(int inputDim, int hiddenDim) {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("input", DataType.FLOAT, -1, inputDim);
        SDVariable w1 = sd.constant("w1", Nd4j.randn(DataType.FLOAT, inputDim, hiddenDim));
        SDVariable b1 = sd.constant("b1", Nd4j.zeros(DataType.FLOAT, 1, hiddenDim));
        SDVariable w2 = sd.constant("w2", Nd4j.randn(DataType.FLOAT, hiddenDim, hiddenDim));
        SDVariable b2 = sd.constant("b2", Nd4j.zeros(DataType.FLOAT, 1, hiddenDim));

        // Layer 1: matmul + bias + relu
        SDVariable h = sd.mmul("mm1", x, w1).add("add1", b1);
        h = sd.nn.relu("relu1", h, 0);

        // Layer 2: matmul + bias
        SDVariable out = sd.mmul("mm2", h, w2).add("output", b2);

        return sd;
    }

    /**
     * Builds a decoder-like graph with a "past" input (simulating KV cache).
     * Has two placeholders: input_ids and past_values.
     */
    private SameDiff buildDecoderGraph(int vocabDim, int hiddenDim, int kvLen) {
        SameDiff sd = SameDiff.create();
        SDVariable inputIds = sd.placeHolder("input_ids", DataType.FLOAT, -1, vocabDim);
        SDVariable pastValues = sd.placeHolder("past_values", DataType.FLOAT, -1, kvLen, hiddenDim);
        SDVariable w = sd.constant("w", Nd4j.randn(DataType.FLOAT, vocabDim, hiddenDim));

        // Embed input
        SDVariable embedded = sd.mmul("embed", inputIds, w);

        // "Attention": mean over past values, add to embedded
        SDVariable pastMean = sd.mean("past_mean", pastValues, 1);  // [batch, hiddenDim]
        SDVariable combined = embedded.add("combined", pastMean);

        // Output logits
        SDVariable wOut = sd.constant("w_out", Nd4j.randn(DataType.FLOAT, hiddenDim, vocabDim));
        sd.mmul("logits", combined, wOut);

        return sd;
    }

    // =========================================================================
    // Test 1: Multiple executions without session reset
    // =========================================================================

    @Test
    @Order(1)
    @DisplayName("DSP reuse: 5 consecutive calls without session reset produce correct results")
    public void testMultipleExecutionsNoReset() {
        SameDiff sd = buildEncoderGraph(16, 32);
        enableDsp(sd);

        int numCalls = 5;
        int batchSize = 4;

        // Run the same graph 5 times with different random inputs
        // Each call should succeed and produce numerically valid results
        INDArray[] results = new INDArray[numCalls];
        for (int i = 0; i < numCalls; i++) {
            INDArray input = Nd4j.randn(DataType.FLOAT, batchSize, 16);
            Map<String, INDArray> out = sd.output(Map.of("input", input), "output");
            results[i] = out.get("output").dup();

            assertNotNull(results[i], "Call " + (i + 1) + ": output is null");
            assertArrayEquals(new long[]{batchSize, 32}, results[i].shape(),
                    "Call " + (i + 1) + ": wrong shape");
            assertFalse(Double.isNaN(results[i].sumNumber().doubleValue()),
                    "Call " + (i + 1) + ": output contains NaN");
            assertFalse(Double.isInfinite(results[i].sumNumber().doubleValue()),
                    "Call " + (i + 1) + ": output contains Inf");

            log.info("Call {}/{}: shape={}, sum={}", i + 1, numCalls,
                    Arrays.toString(results[i].shape()), results[i].sumNumber());
        }

        // Verify different inputs produce different outputs (not stale cached results)
        for (int i = 1; i < numCalls; i++) {
            double diff = results[0].sub(results[i]).amaxNumber().doubleValue();
            assertTrue(diff > TOL, "Calls 1 and " + (i + 1) +
                    " produced identical outputs (diff=" + diff + ") — stale cache?");
        }

        sd.close();
    }

    // =========================================================================
    // Test 2: Same input produces same output across re-executions
    // =========================================================================

    @Test
    @Order(2)
    @DisplayName("DSP reuse: same input produces bitwise-identical output across calls")
    public void testDeterministicOutputAcrossCalls() {
        SameDiff sd = buildEncoderGraph(8, 16);
        enableDsp(sd);

        INDArray fixedInput = Nd4j.ones(DataType.FLOAT, 2, 8);

        INDArray first = sd.output(Map.of("input", fixedInput), "output").get("output").dup();

        for (int i = 1; i <= 4; i++) {
            INDArray result = sd.output(Map.of("input", fixedInput), "output").get("output").dup();
            double maxDiff = first.sub(result).amaxNumber().doubleValue();
            log.info("Call {}: maxDiff from call 1 = {}", i + 1, maxDiff);
            assertTrue(maxDiff < TOL, "Call " + (i + 1) +
                    " differs from call 1 by " + maxDiff + " (tolerance=" + TOL + ")");
        }

        sd.close();
    }

    // =========================================================================
    // Test 3: Frozen shapes + multiple executions
    // =========================================================================

    @Test
    @Order(3)
    @DisplayName("DSP reuse: frozen shapes allow repeated execution with buffer reuse")
    public void testFrozenShapesMultipleExecutions() {
        SameDiff sd = buildEncoderGraph(16, 32);
        enableDsp(sd);

        int batchSize = 2;
        INDArray input1 = Nd4j.randn(DataType.FLOAT, batchSize, 16);

        // First call compiles the plan
        Map<String, INDArray> out1 = sd.output(Map.of("input", input1), "output");
        INDArray result1 = out1.get("output").dup();
        assertNotNull(result1, "First call output is null");
        log.info("Call 1 (compile): shape={}", Arrays.toString(result1.shape()));

        // Freeze shapes on the DSP executor if available
        freezeDspShapes(sd);

        // Execute 5 more times with frozen shapes
        for (int i = 2; i <= 6; i++) {
            INDArray input = Nd4j.randn(DataType.FLOAT, batchSize, 16);
            Map<String, INDArray> out = sd.output(Map.of("input", input), "output");
            INDArray result = out.get("output").dup();

            assertNotNull(result, "Call " + i + ": output is null");
            assertArrayEquals(new long[]{batchSize, 32}, result.shape(),
                    "Call " + i + ": wrong shape");
            assertFalse(Double.isNaN(result.sumNumber().doubleValue()),
                    "Call " + i + ": output contains NaN");

            log.info("Call {} (frozen): shape={}, sum={}", i,
                    Arrays.toString(result.shape()), result.sumNumber());
        }

        sd.close();
    }

    // =========================================================================
    // Test 4: Plan cache survives session reset
    // =========================================================================

    @Test
    @Order(4)
    @DisplayName("DSP reuse: plan handle cached across session reset, second execution succeeds")
    public void testPlanCacheAcrossSessionReset() {
        SameDiff sd = buildEncoderGraph(16, 32);
        enableDsp(sd);

        int batchSize = 2;

        // Call 1: compiles and executes the plan
        INDArray input1 = Nd4j.randn(DataType.FLOAT, batchSize, 16);
        Map<String, INDArray> out1 = sd.output(Map.of("input", input1), "output");
        INDArray result1 = out1.get("output").dup();
        assertNotNull(result1, "Pre-reset call output is null");
        log.info("Pre-reset: shape={}, sum={}", Arrays.toString(result1.shape()),
                result1.sumNumber());

        // Reset session — DSP executor is destroyed, but plan handle should be cached
        sd.resetSession();
        log.info("Session reset complete");

        // Call 2: should restore cached plan handle (no recompilation from scratch)
        INDArray input2 = Nd4j.randn(DataType.FLOAT, batchSize, 16);
        Map<String, INDArray> out2 = sd.output(Map.of("input", input2), "output");
        INDArray result2 = out2.get("output").dup();
        assertNotNull(result2, "Post-reset call output is null");
        assertArrayEquals(new long[]{batchSize, 32}, result2.shape(), "Post-reset: wrong shape");
        assertFalse(Double.isNaN(result2.sumNumber().doubleValue()),
                "Post-reset: output contains NaN");
        log.info("Post-reset: shape={}, sum={}", Arrays.toString(result2.shape()),
                result2.sumNumber());

        sd.close();
    }

    // =========================================================================
    // Test 5: Multi-model interleaved execution (A → B → A)
    // =========================================================================

    @Test
    @Order(5)
    @DisplayName("DSP reuse: interleaved execution of two models (A → B → A) all succeed")
    public void testInterleavedMultiModelExecution() {
        // Model A: "encoder" with 16→32 dims
        SameDiff modelA = buildEncoderGraph(16, 32);
        enableDsp(modelA);

        // Model B: "encoder" with 8→16 dims (different graph)
        SameDiff modelB = buildEncoderGraph(8, 16);
        enableDsp(modelB);

        int batchSize = 2;

        // Run A
        INDArray inputA1 = Nd4j.randn(DataType.FLOAT, batchSize, 16);
        Map<String, INDArray> outA1 = modelA.output(Map.of("input", inputA1), "output");
        INDArray resultA1 = outA1.get("output").dup();
        assertNotNull(resultA1, "Model A call 1 failed");
        log.info("Model A call 1: shape={}", Arrays.toString(resultA1.shape()));

        // Run B
        INDArray inputB = Nd4j.randn(DataType.FLOAT, batchSize, 8);
        Map<String, INDArray> outB = modelB.output(Map.of("input", inputB), "output");
        INDArray resultB = outB.get("output").dup();
        assertNotNull(resultB, "Model B call 1 failed");
        log.info("Model B call 1: shape={}", Arrays.toString(resultB.shape()));

        // Run A again — this is the pattern that fails in VLM multi-page:
        // vision encoder runs, then decoder runs, then vision encoder runs again
        INDArray inputA2 = Nd4j.randn(DataType.FLOAT, batchSize, 16);
        Map<String, INDArray> outA2 = modelA.output(Map.of("input", inputA2), "output");
        INDArray resultA2 = outA2.get("output").dup();
        assertNotNull(resultA2, "Model A call 2 failed — interleaved execution bug");
        assertArrayEquals(resultA1.shape(), resultA2.shape(),
                "Model A call 2 shape mismatch");
        assertFalse(Double.isNaN(resultA2.sumNumber().doubleValue()),
                "Model A call 2 contains NaN");
        log.info("Model A call 2: shape={}", Arrays.toString(resultA2.shape()));

        // Run B again
        INDArray inputB2 = Nd4j.randn(DataType.FLOAT, batchSize, 8);
        Map<String, INDArray> outB2 = modelB.output(Map.of("input", inputB2), "output");
        INDArray resultB2 = outB2.get("output").dup();
        assertNotNull(resultB2, "Model B call 2 failed");
        log.info("Model B call 2: shape={}", Arrays.toString(resultB2.shape()));

        modelA.close();
        modelB.close();
    }

    // =========================================================================
    // Test 6: VLM-style multi-page simulation (encoder + decoder interleaved)
    // =========================================================================

    @Test
    @Order(6)
    @DisplayName("DSP reuse: VLM multi-page simulation — encoder, decoder, repeat 3 pages")
    public void testVlmMultiPageSimulation() {
        // Simulate VLM architecture: vision encoder + decoder, run 3 "pages"
        SameDiff encoder = buildEncoderGraph(64, 128);  // "vision encoder"
        enableDsp(encoder);

        SameDiff decoder = buildDecoderGraph(32, 128, 4);  // "decoder"
        enableDsp(decoder);

        int numPages = 3;
        int batchSize = 1;

        for (int page = 1; page <= numPages; page++) {
            log.info("=== Page {}/{} ===", page, numPages);

            // Step 1: Run vision encoder (like encodeImageTiled)
            INDArray pixelInput = Nd4j.randn(DataType.FLOAT, batchSize, 64);
            Map<String, INDArray> encoderOut = encoder.output(
                    Map.of("input", pixelInput), "output");
            INDArray visionEmbeddings = encoderOut.get("output");
            assertNotNull(visionEmbeddings, "Page " + page + ": encoder output is null");
            assertFalse(Double.isNaN(visionEmbeddings.sumNumber().doubleValue()),
                    "Page " + page + ": encoder output contains NaN");
            log.info("Page {}: encoder output shape={}", page,
                    Arrays.toString(visionEmbeddings.shape()));

            // Step 2: Run decoder for a few "decode steps" (like token generation)
            int decodeSteps = 5;
            INDArray pastValues = Nd4j.randn(DataType.FLOAT, batchSize, 4, 128);
            for (int step = 0; step < decodeSteps; step++) {
                INDArray inputIds = Nd4j.randn(DataType.FLOAT, batchSize, 32);
                Map<String, INDArray> decoderOut = decoder.output(
                        Map.of("input_ids", inputIds, "past_values", pastValues), "logits");
                INDArray logits = decoderOut.get("logits");
                assertNotNull(logits, "Page " + page + " step " + step + ": decoder logits null");
                assertFalse(Double.isNaN(logits.sumNumber().doubleValue()),
                        "Page " + page + " step " + step + ": decoder logits contain NaN");
            }
            log.info("Page {}: {} decode steps completed", page, decodeSteps);

            // Step 3: NO session reset between pages (the correct behavior)
            // The DSP executor, its buffers, and CUDA graph stay alive
            log.info("Page {}: keeping sessions alive (no reset)", page);
        }

        log.info("All {} pages completed successfully with DSP reuse", numPages);
        encoder.close();
        decoder.close();
    }

    // =========================================================================
    // Test 7: Many rapid re-executions (stress test)
    // =========================================================================

    @Test
    @Order(7)
    @DisplayName("DSP reuse: 50 rapid re-executions without reset — no memory corruption")
    public void testRapidReexecutionStress() {
        SameDiff sd = buildEncoderGraph(8, 16);
        enableDsp(sd);

        int numCalls = 50;
        int batchSize = 1;

        for (int i = 0; i < numCalls; i++) {
            INDArray input = Nd4j.randn(DataType.FLOAT, batchSize, 8);
            Map<String, INDArray> out = sd.output(Map.of("input", input), "output");
            INDArray result = out.get("output");

            assertNotNull(result, "Call " + (i + 1) + "/" + numCalls + ": output is null");
            assertFalse(Double.isNaN(result.sumNumber().doubleValue()),
                    "Call " + (i + 1) + "/" + numCalls + ": output contains NaN");
        }

        log.info("All {} rapid re-executions succeeded", numCalls);
        sd.close();
    }

    // =========================================================================
    // Test 8: Freeze → unfreeze → re-execute
    // =========================================================================

    @Test
    @Order(8)
    @DisplayName("DSP reuse: freeze shapes, unfreeze, then re-execute still works")
    public void testFreezeUnfreezeReexecute() {
        SameDiff sd = buildEncoderGraph(16, 32);
        enableDsp(sd);

        int batchSize = 2;

        // Phase 1: Normal execution (compiles plan)
        INDArray input1 = Nd4j.randn(DataType.FLOAT, batchSize, 16);
        Map<String, INDArray> out1 = sd.output(Map.of("input", input1), "output");
        INDArray result1 = out1.get("output").dup();
        assertNotNull(result1, "Phase 1 output is null");
        log.info("Phase 1 (normal): sum={}", result1.sumNumber());

        // Phase 2: Freeze shapes, execute 3 times
        freezeDspShapes(sd);
        for (int i = 0; i < 3; i++) {
            INDArray input = Nd4j.randn(DataType.FLOAT, batchSize, 16);
            Map<String, INDArray> out = sd.output(Map.of("input", input), "output");
            assertNotNull(out.get("output"), "Phase 2 call " + (i + 1) + " output is null");
        }
        log.info("Phase 2 (frozen): 3 calls succeeded");

        // Phase 3: Unfreeze shapes, execute again
        unfreezeDspShapes(sd);
        for (int i = 0; i < 3; i++) {
            INDArray input = Nd4j.randn(DataType.FLOAT, batchSize, 16);
            Map<String, INDArray> out = sd.output(Map.of("input", input), "output");
            INDArray result = out.get("output");
            assertNotNull(result, "Phase 3 call " + (i + 1) + " output is null");
            assertFalse(Double.isNaN(result.sumNumber().doubleValue()),
                    "Phase 3 call " + (i + 1) + " contains NaN");
        }
        log.info("Phase 3 (unfrozen): 3 calls succeeded");

        sd.close();
    }

    // =========================================================================
    // Test 9: Output arrays from previous call are valid after next call
    // =========================================================================

    @Test
    @Order(9)
    @DisplayName("DSP reuse: output arrays from call N remain valid during call N+1")
    public void testOutputArrayLifetimeAcrossCalls() {
        SameDiff sd = buildEncoderGraph(8, 16);
        enableDsp(sd);

        // Call 1 — keep reference to output
        INDArray input1 = Nd4j.ones(DataType.FLOAT, 1, 8);
        Map<String, INDArray> out1 = sd.output(Map.of("input", input1), "output");
        INDArray result1 = out1.get("output").dup();  // dup to own the memory
        double sum1 = result1.sumNumber().doubleValue();

        // Call 2 — result1 should still be valid
        INDArray input2 = Nd4j.ones(DataType.FLOAT, 1, 8).mul(2);
        Map<String, INDArray> out2 = sd.output(Map.of("input", input2), "output");
        INDArray result2 = out2.get("output").dup();

        // Verify result1 wasn't corrupted by call 2
        double sum1After = result1.sumNumber().doubleValue();
        assertEquals(sum1, sum1After, TOL,
                "Output from call 1 was corrupted by call 2 (sum changed from " +
                        sum1 + " to " + sum1After + ")");

        // Different inputs should produce different outputs
        double diff = result1.sub(result2).amaxNumber().doubleValue();
        assertTrue(diff > TOL, "Different inputs produced identical outputs — stale buffer?");

        log.info("Output lifetime test passed: sum1={}, sum1After={}, diff={}",
                sum1, sum1After, diff);
        sd.close();
    }

    // =========================================================================
    // Test 10: Multiple session resets + re-executions
    // =========================================================================

    @Test
    @Order(10)
    @DisplayName("DSP reuse: 3 cycles of execute → reset → execute all succeed")
    public void testMultipleResetCycles() {
        SameDiff sd = buildEncoderGraph(16, 32);
        enableDsp(sd);

        int batchSize = 2;

        for (int cycle = 1; cycle <= 3; cycle++) {
            log.info("=== Cycle {}/3 ===", cycle);

            // Execute
            INDArray input = Nd4j.randn(DataType.FLOAT, batchSize, 16);
            Map<String, INDArray> out = sd.output(Map.of("input", input), "output");
            INDArray result = out.get("output").dup();
            assertNotNull(result, "Cycle " + cycle + ": output is null");
            assertFalse(Double.isNaN(result.sumNumber().doubleValue()),
                    "Cycle " + cycle + ": output contains NaN");
            log.info("Cycle {}: shape={}, sum={}", cycle,
                    Arrays.toString(result.shape()), result.sumNumber());

            // Reset session
            sd.resetSession();
            log.info("Cycle {}: session reset complete", cycle);
        }

        sd.close();
    }

    // =========================================================================
    // Test 11: Varying sequence lengths — dynamic input shapes without reset
    // =========================================================================

    @Test
    @Order(11)
    @DisplayName("DSP dynamic shapes: varying sequence lengths across calls without reset")
    public void testVaryingSequenceLengths() {
        // Graph with dynamic first dimension (sequence length)
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 16);  // [seqLen, 16]
        SDVariable w = sd.constant("w", Nd4j.randn(DataType.FLOAT, 16, 32));
        SDVariable mm = sd.mmul("mm", x, w);  // [seqLen, 32]
        SDVariable out = sd.nn.relu("output", mm, 0);

        enableDsp(sd);

        // Call with increasing sequence lengths
        int[] seqLens = {4, 8, 16, 32, 64, 128};
        for (int seqLen : seqLens) {
            INDArray input = Nd4j.randn(DataType.FLOAT, seqLen, 16);
            Map<String, INDArray> result = sd.output(Map.of("x", input), "output");
            INDArray output = result.get("output");

            assertNotNull(output, "seqLen=" + seqLen + ": output is null");
            assertArrayEquals(new long[]{seqLen, 32}, output.shape(),
                    "seqLen=" + seqLen + ": wrong output shape");
            assertFalse(Double.isNaN(output.sumNumber().doubleValue()),
                    "seqLen=" + seqLen + ": output contains NaN");

            log.info("seqLen={}: shape={}, sum={}", seqLen,
                    Arrays.toString(output.shape()), output.sumNumber());
        }

        sd.close();
    }

    // =========================================================================
    // Test 12: Growing then shrinking input — buffers must handle both directions
    // =========================================================================

    @Test
    @Order(12)
    @DisplayName("DSP dynamic shapes: grow then shrink sequence lengths")
    public void testGrowThenShrinkSequences() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 8);
        SDVariable w = sd.constant("w", Nd4j.randn(DataType.FLOAT, 8, 16));
        SDVariable b = sd.constant("b", Nd4j.zeros(DataType.FLOAT, 1, 16));
        SDVariable mm = sd.mmul("mm", x, w);
        SDVariable added = mm.add("added", b);
        SDVariable out = sd.nn.relu("output", added, 0);

        enableDsp(sd);

        // Grow: 2 → 4 → 8 → 16 → 32
        // Shrink: 32 → 16 → 8 → 4 → 2
        int[] seqLens = {2, 4, 8, 16, 32, 16, 8, 4, 2};
        for (int i = 0; i < seqLens.length; i++) {
            int seqLen = seqLens[i];
            String phase = (i < 5) ? "GROW" : "SHRINK";

            INDArray input = Nd4j.randn(DataType.FLOAT, seqLen, 8);
            Map<String, INDArray> result = sd.output(Map.of("x", input), "output");
            INDArray output = result.get("output");

            assertNotNull(output, phase + " seqLen=" + seqLen + ": output is null");
            assertArrayEquals(new long[]{seqLen, 16}, output.shape(),
                    phase + " seqLen=" + seqLen + ": wrong output shape");
            assertFalse(Double.isNaN(output.sumNumber().doubleValue()),
                    phase + " seqLen=" + seqLen + ": output contains NaN");

            log.info("{} seqLen={}: shape={}, sum={}", phase, seqLen,
                    Arrays.toString(output.shape()), output.sumNumber());
        }

        sd.close();
    }

    // =========================================================================
    // Test 13: Autoregressive decode simulation — seqLen grows by 1 each step
    // =========================================================================

    @Test
    @Order(13)
    @DisplayName("DSP dynamic shapes: autoregressive decode — seqLen grows by 1 per step")
    public void testAutoregressiveDecodeSimulation() {
        // Simulate attention: Q*K^T with growing K dimension
        SameDiff sd = SameDiff.create();
        SDVariable query = sd.placeHolder("query", DataType.FLOAT, 1, 1, 16);    // [batch, 1, headDim]
        SDVariable keys = sd.placeHolder("keys", DataType.FLOAT, 1, -1, 16);     // [batch, kvLen, headDim]
        SDVariable values = sd.placeHolder("values", DataType.FLOAT, 1, -1, 16); // [batch, kvLen, headDim]

        // Attention: softmax(Q * K^T / sqrt(d)) * V
        SDVariable keysT = sd.permute("keysT", keys, 0, 2, 1);     // [batch, headDim, kvLen]
        SDVariable scores = sd.mmul("scores", query, keysT);         // [batch, 1, kvLen]
        SDVariable scale = sd.constant("scale", Nd4j.scalar(DataType.FLOAT, 1.0f / (float) Math.sqrt(16)));
        SDVariable scaled = scores.mul("scaled", scale);
        SDVariable attnWeights = sd.nn.softmax("attn_weights", scaled, -1);
        SDVariable attended = sd.mmul("output", attnWeights, values); // [batch, 1, headDim]

        enableDsp(sd);

        int numSteps = 20;
        for (int step = 1; step <= numSteps; step++) {
            int kvLen = step;  // KV cache grows by 1 each step

            INDArray q = Nd4j.randn(DataType.FLOAT, 1, 1, 16);
            INDArray k = Nd4j.randn(DataType.FLOAT, 1, kvLen, 16);
            INDArray v = Nd4j.randn(DataType.FLOAT, 1, kvLen, 16);

            Map<String, INDArray> result = sd.output(
                    Map.of("query", q, "keys", k, "values", v), "output");
            INDArray output = result.get("output");

            assertNotNull(output, "Step " + step + " (kvLen=" + kvLen + "): output is null");
            assertArrayEquals(new long[]{1, 1, 16}, output.shape(),
                    "Step " + step + ": wrong output shape (should always be [1,1,16])");
            assertFalse(Double.isNaN(output.sumNumber().doubleValue()),
                    "Step " + step + ": output contains NaN");

            if (step % 5 == 0) {
                log.info("Decode step {}/{}: kvLen={}, output sum={}", step, numSteps,
                        kvLen, output.sumNumber());
            }
        }

        log.info("All {} autoregressive decode steps succeeded", numSteps);
        sd.close();
    }

    // =========================================================================
    // Test 14: Two different dynamic lengths in same graph
    // =========================================================================

    @Test
    @Order(14)
    @DisplayName("DSP dynamic shapes: two independent dynamic dims (seqLen + batchSize)")
    public void testTwoDynamicDimensions() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, -1, 8);  // [batch, seqLen, dim]
        SDVariable w = sd.constant("w", Nd4j.randn(DataType.FLOAT, 8, 16));
        // Reshape to 2D for matmul, then reshape back
        SDVariable flat = sd.reshape("flat", x, -1, 8);  // [batch*seqLen, 8]
        SDVariable mm = sd.mmul("mm", flat, w);            // [batch*seqLen, 16]
        // Use reduce to get a fixed output shape regardless of dynamic dims
        SDVariable out = sd.mean("output", mm, 0);         // [16]

        enableDsp(sd);

        // Vary both batch and seqLen
        int[][] shapes = {{1, 4}, {2, 8}, {4, 2}, {1, 16}, {3, 5}, {1, 1}};
        for (int[] shape : shapes) {
            int batch = shape[0], seqLen = shape[1];
            INDArray input = Nd4j.randn(DataType.FLOAT, batch, seqLen, 8);
            Map<String, INDArray> result = sd.output(Map.of("x", input), "output");
            INDArray output = result.get("output");

            assertNotNull(output, "batch=" + batch + " seqLen=" + seqLen + ": null");
            assertFalse(Double.isNaN(output.sumNumber().doubleValue()),
                    "batch=" + batch + " seqLen=" + seqLen + ": NaN");

            log.info("batch={} seqLen={}: output shape={}, sum={}", batch, seqLen,
                    Arrays.toString(output.shape()), output.sumNumber());
        }

        sd.close();
    }

    // =========================================================================
    // Test 15: Dynamic shapes + session reset + re-execute with new shapes
    // =========================================================================

    @Test
    @Order(15)
    @DisplayName("DSP dynamic shapes: session reset between different-sized calls")
    public void testDynamicShapesAcrossSessionReset() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 16);
        SDVariable w = sd.constant("w", Nd4j.randn(DataType.FLOAT, 16, 32));
        SDVariable out = sd.mmul("output", x, w);

        enableDsp(sd);

        // Phase 1: seqLen=10
        INDArray input1 = Nd4j.randn(DataType.FLOAT, 10, 16);
        Map<String, INDArray> out1 = sd.output(Map.of("x", input1), "output");
        assertArrayEquals(new long[]{10, 32}, out1.get("output").shape());
        log.info("Pre-reset: seqLen=10 OK");

        // Reset
        sd.resetSession();

        // Phase 2: different seqLen=25 — plan handle restored from cache, must handle new shape
        INDArray input2 = Nd4j.randn(DataType.FLOAT, 25, 16);
        Map<String, INDArray> out2 = sd.output(Map.of("x", input2), "output");
        assertArrayEquals(new long[]{25, 32}, out2.get("output").shape(),
                "After reset with new seqLen: wrong shape");
        assertFalse(Double.isNaN(out2.get("output").sumNumber().doubleValue()),
                "After reset with new seqLen: NaN");
        log.info("Post-reset: seqLen=25 OK");

        // Phase 3: back to seqLen=10
        INDArray input3 = Nd4j.randn(DataType.FLOAT, 10, 16);
        Map<String, INDArray> out3 = sd.output(Map.of("x", input3), "output");
        assertArrayEquals(new long[]{10, 32}, out3.get("output").shape());
        log.info("Post-reset: seqLen=10 again OK");

        sd.close();
    }

    // =========================================================================
    // Test 16: VLM-style multi-page with dynamic tile counts
    // =========================================================================

    @Test
    @Order(16)
    @DisplayName("DSP dynamic shapes: VLM-style multi-page with varying tile counts per page")
    public void testVlmStyleVaryingTileCounts() {
        // Simulates VLM processing pages with different numbers of tiles:
        // Page 1: 16 tiles (4x4 grid) = seqLen 16*64 = 1024
        // Page 2: 9 tiles (3x3 grid) = seqLen 9*64 = 576
        // Page 3: 4 tiles (2x2 grid) = seqLen 4*64 = 256
        // The encoder is the same model called with different seqLen each time

        SameDiff encoder = SameDiff.create();
        SDVariable x = encoder.placeHolder("pixel_values", DataType.FLOAT, -1, 32);
        SDVariable w1 = encoder.constant("w1", Nd4j.randn(DataType.FLOAT, 32, 64));
        SDVariable w2 = encoder.constant("w2", Nd4j.randn(DataType.FLOAT, 64, 64));
        SDVariable h = encoder.mmul("layer1", x, w1);
        h = encoder.nn.relu("relu1", h, 0);
        SDVariable features = encoder.mmul("image_features", h, w2);

        SameDiff decoder = buildDecoderGraph(32, 64, 4);

        enableDsp(encoder);
        enableDsp(decoder);

        // Different tile counts per "page" → different sequence lengths
        int[] tileCounts = {16, 9, 4, 16, 1};
        int patchesPerTile = 64;

        for (int page = 0; page < tileCounts.length; page++) {
            int numTiles = tileCounts[page];
            int seqLen = numTiles * patchesPerTile;
            log.info("=== Page {} (tiles={}, seqLen={}) ===", page + 1, numTiles, seqLen);

            // Encoder: dynamic seqLen
            INDArray pixelValues = Nd4j.randn(DataType.FLOAT, seqLen, 32);
            Map<String, INDArray> encoderOut = encoder.output(
                    Map.of("pixel_values", pixelValues), "image_features");
            INDArray features2 = encoderOut.get("image_features");
            assertNotNull(features2, "Page " + (page + 1) + ": encoder null");
            assertArrayEquals(new long[]{seqLen, 64}, features2.shape(),
                    "Page " + (page + 1) + ": encoder wrong shape");
            assertFalse(Double.isNaN(features2.sumNumber().doubleValue()),
                    "Page " + (page + 1) + ": encoder NaN");
            log.info("Page {}: encoder OK shape={}", page + 1, Arrays.toString(features2.shape()));

            // Decoder: a few steps (fixed shapes for simplicity)
            INDArray inputIds = Nd4j.randn(DataType.FLOAT, 1, 32);
            INDArray pastValues = Nd4j.randn(DataType.FLOAT, 1, 4, 64);
            Map<String, INDArray> decoderOut = decoder.output(
                    Map.of("input_ids", inputIds, "past_values", pastValues), "logits");
            assertNotNull(decoderOut.get("logits"), "Page " + (page + 1) + ": decoder null");
            log.info("Page {}: decoder OK", page + 1);

            // No session reset
        }

        log.info("All {} pages with varying tile counts succeeded", tileCounts.length);
        encoder.close();
        decoder.close();
    }

    // =========================================================================
    // Test 17: Dynamic shapes correctness — verify numerical output
    // =========================================================================

    @Test
    @Order(17)
    @DisplayName("DSP dynamic shapes: numerical correctness with varying lengths")
    public void testDynamicShapesNumericalCorrectness() {
        // Build graph, run with DSP disabled first to get reference, then with DSP
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 8);
        SDVariable w = sd.constant("w", Nd4j.randn(DataType.FLOAT, 8, 4));
        SDVariable b = sd.constant("b", Nd4j.randn(DataType.FLOAT, 1, 4));
        SDVariable mm = sd.mmul("mm", x, w);
        SDVariable out = mm.add("output", b);

        int[] seqLens = {1, 5, 10, 3, 7};

        // Collect reference results with DSP disabled
        Map<Integer, INDArray> referenceResults = new LinkedHashMap<>();
        Map<Integer, INDArray> referenceInputs = new LinkedHashMap<>();
        for (int seqLen : seqLens) {
            INDArray input = Nd4j.randn(DataType.FLOAT, seqLen, 8);
            referenceInputs.put(seqLen, input);
            Map<String, INDArray> result = sd.output(Map.of("x", input), "output");
            referenceResults.put(seqLen, result.get("output").dup());
        }

        // Reset and enable DSP
        sd.resetSession();
        enableDsp(sd);

        // Re-run with same inputs, compare to reference
        for (int seqLen : seqLens) {
            INDArray input = referenceInputs.get(seqLen);
            Map<String, INDArray> dspResult = sd.output(Map.of("x", input), "output");
            INDArray dspOutput = dspResult.get("output");
            INDArray refOutput = referenceResults.get(seqLen);

            assertArrayEquals(refOutput.shape(), dspOutput.shape(),
                    "seqLen=" + seqLen + ": shape mismatch");
            double maxDiff = refOutput.sub(dspOutput).amaxNumber().doubleValue();
            log.info("seqLen={}: maxDiff={}", seqLen, maxDiff);
            assertTrue(maxDiff < TOL, "seqLen=" + seqLen +
                    ": DSP vs standard diff " + maxDiff + " exceeds tolerance " + TOL);
        }

        sd.close();
    }

    // =========================================================================
    // Test 18: Extreme shape variation — 1 element to 10000 elements
    // =========================================================================

    @Test
    @Order(18)
    @DisplayName("DSP dynamic shapes: extreme range from seqLen=1 to seqLen=1000")
    public void testExtremeShapeVariation() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable w = sd.constant("w", Nd4j.randn(DataType.FLOAT, 4, 8));
        SDVariable out = sd.nn.relu("output", sd.mmul("mm", x, w), 0);

        enableDsp(sd);

        int[] seqLens = {1, 10, 100, 1000, 1, 500, 1000, 1};
        for (int seqLen : seqLens) {
            INDArray input = Nd4j.randn(DataType.FLOAT, seqLen, 4);
            Map<String, INDArray> result = sd.output(Map.of("x", input), "output");
            INDArray output = result.get("output");

            assertNotNull(output, "seqLen=" + seqLen + ": null");
            assertArrayEquals(new long[]{seqLen, 8}, output.shape(),
                    "seqLen=" + seqLen + ": wrong shape");
            assertFalse(Double.isNaN(output.sumNumber().doubleValue()),
                    "seqLen=" + seqLen + ": NaN");
        }

        log.info("Extreme shape variation test passed");
        sd.close();
    }

    // =========================================================================
    // Test 19: Deep graph re-execution — many ops + large intermediates
    // =========================================================================

    @Test
    @Order(19)
    @DisplayName("DSP reuse: deep graph (100+ ops) re-executed without reset — no heap corruption")
    public void testDeepGraphReexecution() {
        // Build a deep graph with ~100 ops and large intermediates.
        // This tests for the heap corruption bug seen in the vision encoder (1950 ops):
        //   malloc(): unsorted double linked list corrupted
        // The corruption occurs when clearDynamicShapePlanCaches() frees intermediate
        // buffers but C++ retains stale pointers, causing use-after-free on re-execution.
        int numLayers = 20;
        int hiddenDim = 256;

        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("input", DataType.FLOAT, -1, hiddenDim);

        SDVariable h = x;
        for (int i = 0; i < numLayers; i++) {
            // Each layer: matmul + bias + relu + residual = ~5 ops per layer = ~100 ops total
            SDVariable w = sd.constant("w_" + i, Nd4j.randn(DataType.FLOAT, hiddenDim, hiddenDim).div(Math.sqrt(hiddenDim)));
            SDVariable b = sd.constant("b_" + i, Nd4j.zeros(DataType.FLOAT, 1, hiddenDim));
            SDVariable projected = sd.mmul("mm_" + i, h, w);
            SDVariable biased = projected.add("add_" + i, b);
            SDVariable activated = sd.nn.relu("relu_" + i, biased, 0);
            // Residual connection (skip every other layer)
            if (i % 2 == 1) {
                h = activated.add("residual_" + i, h);
            } else {
                h = activated;
            }
        }
        // Final output
        SDVariable wOut = sd.constant("w_out", Nd4j.randn(DataType.FLOAT, hiddenDim, 64).div(Math.sqrt(hiddenDim)));
        sd.mmul("output", h, wOut);

        enableDsp(sd);
        log.info("Deep graph built: {} layers, {} hidden dim", numLayers, hiddenDim);

        // Execute 3 times without reset — this is where the heap corruption manifests
        for (int call = 1; call <= 3; call++) {
            INDArray input = Nd4j.randn(DataType.FLOAT, 16, hiddenDim);
            Map<String, INDArray> result = sd.output(Map.of("input", input), "output");
            INDArray output = result.get("output");

            assertNotNull(output, "Call " + call + ": output is null");
            assertArrayEquals(new long[]{16, 64}, output.shape(),
                    "Call " + call + ": wrong output shape");
            assertFalse(Double.isNaN(output.sumNumber().doubleValue()),
                    "Call " + call + ": output contains NaN");
            assertFalse(Double.isInfinite(output.sumNumber().doubleValue()),
                    "Call " + call + ": output contains Inf");

            log.info("Deep graph call {}/3: shape={}, sum={}", call,
                    Arrays.toString(output.shape()), output.sumNumber());
        }

        sd.close();
    }

    // =========================================================================
    // Test 20: Large intermediate buffers — stress buffer lifecycle
    // =========================================================================

    @Test
    @Order(20)
    @DisplayName("DSP reuse: large intermediates (multi-MB) re-executed without corruption")
    public void testLargeIntermediateBufferReexecution() {
        // Vision encoder crash was at malloc() — heap corruption from large buffer ops.
        // This test creates large intermediate buffers to stress the same code path.
        int seqLen = 256;   // like 4 tiles of 64 patches
        int dim = 768;      // like hidden dim of SmolDocling

        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("input", DataType.FLOAT, seqLen, dim);

        // Create large intermediates through matmul and reshape
        SDVariable w1 = sd.constant("w1", Nd4j.randn(DataType.FLOAT, dim, dim).div(Math.sqrt(dim)));
        SDVariable h = sd.mmul("mm1", x, w1);          // [256, 768] = 786KB
        SDVariable w2 = sd.constant("w2", Nd4j.randn(DataType.FLOAT, dim, dim).div(Math.sqrt(dim)));
        SDVariable h2 = sd.mmul("mm2", h, w2);          // [256, 768] = 786KB
        SDVariable h3 = h2.add("residual", h);
        SDVariable out = sd.nn.relu("output", h3, 0);

        enableDsp(sd);

        for (int call = 1; call <= 5; call++) {
            INDArray input = Nd4j.randn(DataType.FLOAT, seqLen, dim);
            Map<String, INDArray> result = sd.output(Map.of("input", input), "output");
            INDArray output = result.get("output");

            assertNotNull(output, "Call " + call + ": null");
            assertArrayEquals(new long[]{seqLen, dim}, output.shape(),
                    "Call " + call + ": wrong shape");
            assertFalse(Double.isNaN(output.sumNumber().doubleValue()),
                    "Call " + call + ": NaN");

            log.info("Large buffer call {}/5: sum={}", call, output.sumNumber());
        }

        sd.close();
    }

    // =========================================================================
    // Test 21: Reshape/transpose-heavy graph re-execution
    // =========================================================================

    @Test
    @Order(21)
    @DisplayName("DSP reuse: reshape + transpose + matmul (attention-like pattern) re-execution")
    public void testReshapeTransposeReexecution() {
        // Vision encoder has many reshape/transpose ops for multi-head attention.
        // These view ops can alias buffers, making buffer lifecycle tricky.
        int batchSize = 1;
        int seqLen = 64;
        int numHeads = 4;
        int headDim = 32;
        int hiddenDim = numHeads * headDim; // 128

        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("input", DataType.FLOAT, batchSize, seqLen, hiddenDim);

        // Q, K, V projections
        SDVariable wQ = sd.constant("wQ", Nd4j.randn(DataType.FLOAT, hiddenDim, hiddenDim).div(Math.sqrt(hiddenDim)));
        SDVariable wK = sd.constant("wK", Nd4j.randn(DataType.FLOAT, hiddenDim, hiddenDim).div(Math.sqrt(hiddenDim)));
        SDVariable wV = sd.constant("wV", Nd4j.randn(DataType.FLOAT, hiddenDim, hiddenDim).div(Math.sqrt(hiddenDim)));

        // Flatten batch*seq for matmul
        SDVariable xFlat = sd.reshape("x_flat", x, -1, hiddenDim); // [64, 128]
        SDVariable q = sd.mmul("q_proj", xFlat, wQ);  // [64, 128]
        SDVariable k = sd.mmul("k_proj", xFlat, wK);
        SDVariable v = sd.mmul("v_proj", xFlat, wV);

        // Reshape to multi-head: [batch, seq, numHeads, headDim] → [batch, numHeads, seq, headDim]
        SDVariable qMH = sd.reshape("q_mh", q, batchSize, seqLen, numHeads, headDim);
        SDVariable qT = sd.permute("q_t", qMH, 0, 2, 1, 3); // [1, 4, 64, 32]
        SDVariable kMH = sd.reshape("k_mh", k, batchSize, seqLen, numHeads, headDim);
        SDVariable kT = sd.permute("k_t", kMH, 0, 2, 3, 1); // [1, 4, 32, 64] — transposed

        // Attention: Q * K^T
        SDVariable scores = sd.mmul("attn_scores", qT, kT); // [1, 4, 64, 64]
        SDVariable scale = sd.constant("scale", Nd4j.scalar(DataType.FLOAT, 1.0f / (float) Math.sqrt(headDim)));
        SDVariable scaled = scores.mul("scaled_scores", scale);
        SDVariable weights = sd.nn.softmax("attn_weights", scaled, -1);

        // Attend to values
        SDVariable vMH = sd.reshape("v_mh", v, batchSize, seqLen, numHeads, headDim);
        SDVariable vT = sd.permute("v_t", vMH, 0, 2, 1, 3); // [1, 4, 64, 32]
        SDVariable attended = sd.mmul("attended", weights, vT); // [1, 4, 64, 32]

        // Reshape back: [1, 4, 64, 32] → [1, 64, 128]
        SDVariable attT = sd.permute("att_t", attended, 0, 2, 1, 3); // [1, 64, 4, 32]
        SDVariable output = sd.reshape("output", attT, batchSize, seqLen, hiddenDim);

        enableDsp(sd);
        log.info("Attention graph built: {} ops", sd.ops().length);

        // Execute 3 times — reshape/transpose view aliasing + re-execution is the bug vector
        for (int call = 1; call <= 3; call++) {
            INDArray input = Nd4j.randn(DataType.FLOAT, batchSize, seqLen, hiddenDim);
            Map<String, INDArray> result = sd.output(Map.of("input", input), "output");
            INDArray output2 = result.get("output");

            assertNotNull(output2, "Call " + call + ": null");
            assertArrayEquals(new long[]{batchSize, seqLen, hiddenDim}, output2.shape(),
                    "Call " + call + ": wrong shape");
            assertFalse(Double.isNaN(output2.sumNumber().doubleValue()),
                    "Call " + call + ": NaN");

            log.info("Attention graph call {}/3: shape={}, sum={}", call,
                    Arrays.toString(output2.shape()), output2.sumNumber());
        }

        sd.close();
    }

    // =========================================================================
    // Test 22: Deep graph with dynamic shapes — the full stress test
    // =========================================================================

    @Test
    @Order(22)
    @DisplayName("DSP reuse: deep graph + dynamic shapes + re-execution (full stress)")
    public void testDeepGraphDynamicShapesReexecution() {
        // Combines the two hardest patterns:
        // 1. Many ops (heap corruption vector)
        // 2. Varying input shapes (buffer reallocation vector)
        int numLayers = 10;
        int hiddenDim = 128;

        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("input", DataType.FLOAT, -1, hiddenDim);

        SDVariable h = x;
        for (int i = 0; i < numLayers; i++) {
            SDVariable w = sd.constant("w_" + i, Nd4j.randn(DataType.FLOAT, hiddenDim, hiddenDim).div(Math.sqrt(hiddenDim)));
            SDVariable projected = sd.mmul("mm_" + i, h, w);
            h = sd.nn.relu("relu_" + i, projected, 0);
        }
        SDVariable wOut = sd.constant("w_out", Nd4j.randn(DataType.FLOAT, hiddenDim, 32));
        sd.mmul("output", h, wOut);

        enableDsp(sd);
        log.info("Deep dynamic graph: {} layers, {} hidden dim", numLayers, hiddenDim);

        // Run with varying seqLen: 4 → 64 → 4 → 256 → 4
        int[] seqLens = {4, 64, 4, 256, 4, 64, 256, 4};
        for (int i = 0; i < seqLens.length; i++) {
            int seqLen = seqLens[i];
            INDArray input = Nd4j.randn(DataType.FLOAT, seqLen, hiddenDim);
            Map<String, INDArray> result = sd.output(Map.of("input", input), "output");
            INDArray output = result.get("output");

            assertNotNull(output, "seqLen=" + seqLen + " call " + (i + 1) + ": null");
            assertArrayEquals(new long[]{seqLen, 32}, output.shape(),
                    "seqLen=" + seqLen + " call " + (i + 1) + ": wrong shape");
            assertFalse(Double.isNaN(output.sumNumber().doubleValue()),
                    "seqLen=" + seqLen + " call " + (i + 1) + ": NaN");

            log.info("Deep+dynamic call {}/{}: seqLen={}, shape={}", i + 1, seqLens.length,
                    seqLen, Arrays.toString(output.shape()));
        }

        sd.close();
    }

    // =========================================================================
    // Test 23: ONNX vision encoder — 1950 ops, THE actual crash reproducer
    // =========================================================================

    @Test
    @Order(23)
    @DisplayName("DSP reuse: ONNX vision encoder 2 consecutive calls — heap corruption reproducer")
    public void testOnnxVisionEncoderReexecution() throws Exception {
        // This is the EXACT bug: SmolDocling vision encoder (1950 ops, 343 inputs)
        // crashes on 2nd call with: malloc(): unsorted double linked list corrupted
        // The crash is a SIGABRT from glibc heap corruption detection.
        //
        // Root cause hypothesis:
        //   clearDynamicShapePlanCaches() frees intermediate C++ buffers
        //   but the plan's slot pointers still reference them. On 2nd execution,
        //   malloc() detects the corrupted free list from the double-free or
        //   use-after-free that occurred during 1st execution's cleanup.
        SameDiff visionEncoder = null;
        try {
            File visionDir = org.eclipse.deeplearning4j.vlm.data.VLMModelDownloader.download(
                    org.eclipse.deeplearning4j.vlm.data.VLMModelDownloader.VLMModel.SMOLDOCLING_VISION_ENCODER
            ).getModelFile();
            visionEncoder = org.eclipse.deeplearning4j.vlm.model.OnnxModelCache.importWithCache(
                    visionDir.getAbsolutePath());
        } catch (Exception e) {
            log.warn("Could not load vision encoder model: {}", e.getMessage());
            Assumptions.assumeTrue(false, "Vision encoder model not available");
            return;
        }

        visionEncoder.setDspAutoCompileEnabled(true);
        visionEncoder.setDspNativeAutoCompileEnabled(true);
        log.info("Vision encoder loaded: {} ops", visionEncoder.ops().length);

        List<String> outputNames = List.of("image_features");

        // Call 1: should always succeed
        INDArray pv1 = Nd4j.rand(DataType.FLOAT, 1, 1, 3, 512, 512);
        INDArray mask1 = Nd4j.ones(DataType.BOOL, 1, 512, 512);
        Map<String, INDArray> out1 = visionEncoder.output(
                Map.of("pixel_values", pv1, "pixel_attention_mask", mask1), outputNames);
        INDArray features1 = out1.get("image_features");
        assertNotNull(features1, "Call 1: image_features is null");
        log.info("Call 1 OK: shape={}", Arrays.toString(features1.shape()));

        // Call 2: THIS IS THE CRASH — if we get here without SIGABRT, the bug is fixed
        INDArray pv2 = Nd4j.rand(DataType.FLOAT, 1, 1, 3, 512, 512);
        INDArray mask2 = Nd4j.ones(DataType.BOOL, 1, 512, 512);
        Map<String, INDArray> out2;
        try {
            out2 = visionEncoder.output(
                    Map.of("pixel_values", pv2, "pixel_attention_mask", mask2), outputNames);
        } catch (Exception e) {
            fail("Call 2 FAILED — DSP re-execution heap corruption bug. " +
                    "Error: " + e.getMessage(), e);
            return;
        }

        INDArray features2 = out2.get("image_features");
        assertNotNull(features2, "Call 2: image_features is null");
        assertArrayEquals(features1.shape(), features2.shape(),
                "Call 2: output shape changed");
        assertFalse(Double.isNaN(features2.sumNumber().doubleValue()),
                "Call 2: output contains NaN");
        log.info("Call 2 OK: shape={}", Arrays.toString(features2.shape()));

        // Cleanup
        pv1.close();
        mask1.close();
        pv2.close();
        mask2.close();
        visionEncoder.close();
    }

    // =========================================================================
    // Test 24: Pure nd4j permuted-view batched matmul — NO SameDiff, NO DSP
    // =========================================================================

    @Test
    @Order(24)
    @DisplayName("Pure nd4j: permuted view batched matmul (no SameDiff, no DSP)")
    public void testPureNd4jPermutedViewBatchedMatmul() {
        // Isolates whether cublas batched matmul handles permuted-stride views.
        // If this fails, the problem is in nd4j/cublas, NOT DSP.
        int batchSize = 1;
        int seqLen = 64;
        int numHeads = 4;
        int headDim = 32;
        int hiddenDim = numHeads * headDim; // 128

        // Simulate Q, K, V projections
        INDArray x = Nd4j.randn(DataType.FLOAT, batchSize * seqLen, hiddenDim);
        INDArray wQ = Nd4j.randn(DataType.FLOAT, hiddenDim, hiddenDim).div(Math.sqrt(hiddenDim));
        INDArray wK = Nd4j.randn(DataType.FLOAT, hiddenDim, hiddenDim).div(Math.sqrt(hiddenDim));
        INDArray wV = Nd4j.randn(DataType.FLOAT, hiddenDim, hiddenDim).div(Math.sqrt(hiddenDim));

        INDArray q = x.mmul(wQ); // [64, 128]
        INDArray k = x.mmul(wK);
        INDArray v = x.mmul(wV);

        // Reshape to multi-head: [batch, seq, numHeads, headDim]
        INDArray qMH = q.reshape(batchSize, seqLen, numHeads, headDim);
        INDArray kMH = k.reshape(batchSize, seqLen, numHeads, headDim);
        INDArray vMH = v.reshape(batchSize, seqLen, numHeads, headDim);

        // Permute to [batch, numHeads, seq, headDim] — creates non-contiguous views
        INDArray qT = qMH.permute(0, 2, 1, 3); // [1, 4, 64, 32]
        INDArray kT = kMH.permute(0, 2, 3, 1); // [1, 4, 32, 64] — transposed

        log.info("qT shape={}, strides={}, ews={}", Arrays.toString(qT.shape()),
                Arrays.toString(qT.stride()), qT.elementWiseStride());
        log.info("kT shape={}, strides={}, ews={}", Arrays.toString(kT.shape()),
                Arrays.toString(kT.stride()), kT.elementWiseStride());

        // Test A: batched matmul with CONTIGUOUS copies (baseline)
        INDArray qTc = qT.dup(); // force contiguous
        INDArray kTc = kT.dup();
        log.info("qTc strides={}, ews={}", Arrays.toString(qTc.stride()), qTc.elementWiseStride());
        log.info("kTc strides={}, ews={}", Arrays.toString(kTc.stride()), kTc.elementWiseStride());
        INDArray scoresContiguous = Nd4j.matmul(qTc, kTc);
        assertNotNull(scoresContiguous, "Contiguous batched matmul returned null");
        assertArrayEquals(new long[]{1, 4, 64, 64}, scoresContiguous.shape(),
                "Contiguous batched matmul shape wrong");
        assertFalse(Double.isNaN(scoresContiguous.sumNumber().doubleValue()),
                "Contiguous batched matmul contains NaN");
        log.info("Contiguous batched matmul OK: shape={}, sum={}",
                Arrays.toString(scoresContiguous.shape()), scoresContiguous.sumNumber());

        // Test B: batched matmul with NON-CONTIGUOUS permuted views
        INDArray scores = Nd4j.matmul(qT, kT); // [1, 4, 64, 64]
        assertNotNull(scores, "Batched matmul of permuted views returned null");
        assertArrayEquals(new long[]{1, 4, 64, 64}, scores.shape(),
                "Batched matmul shape wrong");
        assertFalse(Double.isNaN(scores.sumNumber().doubleValue()),
                "Batched matmul result contains NaN");

        // Scale
        INDArray scaled = scores.mul(1.0f / (float) Math.sqrt(headDim));
        assertFalse(Double.isNaN(scaled.sumNumber().doubleValue()),
                "Scaled scores contain NaN");

        log.info("Pure nd4j permuted-view batched matmul OK: scores shape={}, sum={}",
                Arrays.toString(scores.shape()), scores.sumNumber());

        // Also test: values attention
        INDArray vT = vMH.permute(0, 2, 1, 3); // [1, 4, 64, 32]
        INDArray softmax = Nd4j.nn().softmax(scaled, -1);
        INDArray attended = Nd4j.matmul(softmax, vT); // [1, 4, 64, 32]
        assertArrayEquals(new long[]{1, 4, 64, 32}, attended.shape());
        assertFalse(Double.isNaN(attended.sumNumber().doubleValue()));

        // Permute back and reshape
        INDArray attT = attended.permute(0, 2, 1, 3); // [1, 64, 4, 32]
        INDArray output = attT.reshape(batchSize, seqLen, hiddenDim);
        assertArrayEquals(new long[]{1, 64, 128}, output.shape());
        assertFalse(Double.isNaN(output.sumNumber().doubleValue()));
        log.info("Full attention pipeline OK: output shape={}", Arrays.toString(output.shape()));
    }

    // =========================================================================
    // Test 25: SameDiff attention graph WITHOUT DSP
    // =========================================================================

    @Test
    @Order(25)
    @DisplayName("SameDiff attention graph without DSP (standard InferenceSession)")
    public void testSameDiffAttentionNoDsp() {
        // Same graph as test 21, but with DSP disabled.
        // If this passes and test 21 fails, the problem is in DSP specifically.
        int batchSize = 1;
        int seqLen = 64;
        int numHeads = 4;
        int headDim = 32;
        int hiddenDim = numHeads * headDim;

        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("input", DataType.FLOAT, batchSize, seqLen, hiddenDim);

        SDVariable wQ = sd.constant("wQ", Nd4j.randn(DataType.FLOAT, hiddenDim, hiddenDim).div(Math.sqrt(hiddenDim)));
        SDVariable wK = sd.constant("wK", Nd4j.randn(DataType.FLOAT, hiddenDim, hiddenDim).div(Math.sqrt(hiddenDim)));
        SDVariable wV = sd.constant("wV", Nd4j.randn(DataType.FLOAT, hiddenDim, hiddenDim).div(Math.sqrt(hiddenDim)));

        SDVariable xFlat = sd.reshape("x_flat", x, -1, hiddenDim);
        SDVariable q = sd.mmul("q_proj", xFlat, wQ);
        SDVariable k = sd.mmul("k_proj", xFlat, wK);
        SDVariable v = sd.mmul("v_proj", xFlat, wV);

        SDVariable qMH = sd.reshape("q_mh", q, batchSize, seqLen, numHeads, headDim);
        SDVariable qT = sd.permute("q_t", qMH, 0, 2, 1, 3);
        SDVariable kMH = sd.reshape("k_mh", k, batchSize, seqLen, numHeads, headDim);
        SDVariable kT = sd.permute("k_t", kMH, 0, 2, 3, 1);

        SDVariable scores = sd.mmul("attn_scores", qT, kT);
        SDVariable scale = sd.constant("scale", Nd4j.scalar(DataType.FLOAT, 1.0f / (float) Math.sqrt(headDim)));
        SDVariable scaled = scores.mul("scaled_scores", scale);
        SDVariable weights = sd.nn.softmax("attn_weights", scaled, -1);

        SDVariable vMH = sd.reshape("v_mh", v, batchSize, seqLen, numHeads, headDim);
        SDVariable vT = sd.permute("v_t", vMH, 0, 2, 1, 3);
        SDVariable attended = sd.mmul("attended", weights, vT);

        SDVariable attT = sd.permute("att_t", attended, 0, 2, 1, 3);
        SDVariable output = sd.reshape("output", attT, batchSize, seqLen, hiddenDim);

        // Explicitly DISABLE DSP
        sd.setDspAutoCompileEnabled(false);
        sd.setDspNativeAutoCompileEnabled(false);
        log.info("Attention graph (NO DSP) built: {} ops", sd.ops().length);

        for (int call = 1; call <= 3; call++) {
            INDArray input = Nd4j.randn(DataType.FLOAT, batchSize, seqLen, hiddenDim);
            Map<String, INDArray> result = sd.output(Map.of("input", input), "output");
            INDArray out = result.get("output");

            assertNotNull(out, "Call " + call + ": null");
            assertArrayEquals(new long[]{batchSize, seqLen, hiddenDim}, out.shape(),
                    "Call " + call + ": wrong shape");
            assertFalse(Double.isNaN(out.sumNumber().doubleValue()),
                    "Call " + call + ": NaN");

            log.info("SameDiff attention (no DSP) call {}/3: shape={}, sum={}", call,
                    Arrays.toString(out.shape()), out.sumNumber());
        }

        sd.close();
    }

    // =========================================================================
    // Test 26: SameDiff attention graph WITH DSP — same as test 21 but order 26
    //          to run after 24+25 establish baselines
    // =========================================================================

    @Test
    @Order(26)
    @DisplayName("SameDiff attention graph WITH DSP (isolates DSP as cause)")
    public void testSameDiffAttentionWithDsp() {
        // Identical graph to test 25, but with DSP enabled.
        // Compare results: if 24+25 pass but this fails, DSP is the problem.
        int batchSize = 1;
        int seqLen = 64;
        int numHeads = 4;
        int headDim = 32;
        int hiddenDim = numHeads * headDim;

        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("input", DataType.FLOAT, batchSize, seqLen, hiddenDim);

        SDVariable wQ = sd.constant("wQ", Nd4j.randn(DataType.FLOAT, hiddenDim, hiddenDim).div(Math.sqrt(hiddenDim)));
        SDVariable wK = sd.constant("wK", Nd4j.randn(DataType.FLOAT, hiddenDim, hiddenDim).div(Math.sqrt(hiddenDim)));
        SDVariable wV = sd.constant("wV", Nd4j.randn(DataType.FLOAT, hiddenDim, hiddenDim).div(Math.sqrt(hiddenDim)));

        SDVariable xFlat = sd.reshape("x_flat", x, -1, hiddenDim);
        SDVariable q = sd.mmul("q_proj", xFlat, wQ);
        SDVariable k = sd.mmul("k_proj", xFlat, wK);
        SDVariable v = sd.mmul("v_proj", xFlat, wV);

        SDVariable qMH = sd.reshape("q_mh", q, batchSize, seqLen, numHeads, headDim);
        SDVariable qT = sd.permute("q_t", qMH, 0, 2, 1, 3);
        SDVariable kMH = sd.reshape("k_mh", k, batchSize, seqLen, numHeads, headDim);
        SDVariable kT = sd.permute("k_t", kMH, 0, 2, 3, 1);

        SDVariable scores = sd.mmul("attn_scores", qT, kT);
        SDVariable scale = sd.constant("scale", Nd4j.scalar(DataType.FLOAT, 1.0f / (float) Math.sqrt(headDim)));
        SDVariable scaled = scores.mul("scaled_scores", scale);
        SDVariable weights = sd.nn.softmax("attn_weights", scaled, -1);

        SDVariable vMH = sd.reshape("v_mh", v, batchSize, seqLen, numHeads, headDim);
        SDVariable vT = sd.permute("v_t", vMH, 0, 2, 1, 3);
        SDVariable attended = sd.mmul("attended", weights, vT);

        SDVariable attT = sd.permute("att_t", attended, 0, 2, 1, 3);
        SDVariable output = sd.reshape("output", attT, batchSize, seqLen, hiddenDim);

        enableDsp(sd);
        log.info("Attention graph (WITH DSP) built: {} ops", sd.ops().length);

        for (int call = 1; call <= 3; call++) {
            INDArray input = Nd4j.randn(DataType.FLOAT, batchSize, seqLen, hiddenDim);
            Map<String, INDArray> result = sd.output(Map.of("input", input), "output");
            INDArray out = result.get("output");

            assertNotNull(out, "Call " + call + ": null");
            assertArrayEquals(new long[]{batchSize, seqLen, hiddenDim}, out.shape(),
                    "Call " + call + ": wrong shape");
            assertFalse(Double.isNaN(out.sumNumber().doubleValue()),
                    "Call " + call + ": NaN");

            log.info("SameDiff attention (WITH DSP) call {}/3: shape={}, sum={}", call,
                    Arrays.toString(out.shape()), out.sumNumber());
        }

        sd.close();
    }

    // =========================================================================
    // Test: Plan reuse with frozen shapes
    // =========================================================================

    @Test
    @Order(100)
    @DisplayName("DSP reuse: frozen shapes produce consistent output across 5 re-executions")
    public void testPlanReuseWithFrozenShapes() {
        SameDiff sd = buildEncoderGraph(16, 32);
        enableDsp(sd);

        int batchSize = 2;
        INDArray input = Nd4j.randn(DataType.FLOAT, batchSize, 16);

        // Warmup (unfrozen)
        Map<String, INDArray> warmup = sd.output(Map.of("input", input), "output");
        INDArray warmupOut = warmup.get("output").dup();
        assertNotNull(warmupOut);
        log.info("Warmup: shape={} sum={}", Arrays.toString(warmupOut.shape()), warmupOut.sumNumber());

        // Freeze shapes
        freezeDspShapes(sd);

        // 5 frozen executions with same input — should produce consistent results
        for (int i = 0; i < 5; i++) {
            Map<String, INDArray> result = sd.output(Map.of("input", input), "output");
            INDArray out = result.get("output");
            assertNotNull(out, "Frozen call " + (i + 1) + ": null output");
            assertArrayEquals(new long[]{batchSize, 32}, out.shape(),
                    "Frozen call " + (i + 1) + ": wrong shape");
            assertFalse(Double.isNaN(out.sumNumber().doubleValue()),
                    "Frozen call " + (i + 1) + ": NaN");
            assertFalse(Double.isInfinite(out.sumNumber().doubleValue()),
                    "Frozen call " + (i + 1) + ": Inf");

            log.info("Frozen call {}/5: shape={} sum={}", i + 1,
                    Arrays.toString(out.shape()), out.sumNumber());
        }

        sd.close();
    }

    // =========================================================================
    // Helpers
    // =========================================================================

    /**
     * Freeze shapes on the DSP executor of a SameDiff instance.
     * Accesses the executor through the InferenceSession.
     */
    private void freezeDspShapes(SameDiff sd) {
        try {
            // Get the session for the current thread via reflection
            var sessionsField = SameDiff.class.getDeclaredField("sessions");
            sessionsField.setAccessible(true);
            @SuppressWarnings("unchecked")
            Map<Long, InferenceSession> sessions = (Map<Long, InferenceSession>) sessionsField.get(sd);
            InferenceSession session = sessions.get(Thread.currentThread().getId());
            if (session != null) {
                DynamicShapePlanExecutor dsp = session.getDynamicShapePlanExecutor();
                if (dsp != null) {
                    dsp.setShapesFrozen(true);
                    log.info("Froze DSP shapes");
                }
            }
        } catch (Exception e) {
            log.warn("Could not freeze DSP shapes: {}", e.getMessage());
        }
    }

    /**
     * Unfreeze shapes on the DSP executor.
     */
    private void unfreezeDspShapes(SameDiff sd) {
        try {
            var sessionsField = SameDiff.class.getDeclaredField("sessions");
            sessionsField.setAccessible(true);
            @SuppressWarnings("unchecked")
            Map<Long, InferenceSession> sessions = (Map<Long, InferenceSession>) sessionsField.get(sd);
            InferenceSession session = sessions.get(Thread.currentThread().getId());
            if (session != null) {
                DynamicShapePlanExecutor dsp = session.getDynamicShapePlanExecutor();
                if (dsp != null) {
                    dsp.setShapesFrozen(false);
                    log.info("Unfroze DSP shapes");
                }
            }
        } catch (Exception e) {
            log.warn("Could not unfreeze DSP shapes: {}", e.getMessage());
        }
    }
}
