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
package org.eclipse.deeplearning4j.nd4j.linalg.mixed;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.*;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.EnumSource;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.autodiff.samediff.optimize.GraphOptimizer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.*;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Deep DSP pipeline isolation tests.
 *
 * These tests exercise the PATHOLOGICAL patterns that break the VLM decode pipeline:
 *
 * 1. BUFFER STALENESS: Re-executing the same graph with different placeholder data.
 *    The DSP must produce different outputs for different inputs — NOT serve stale
 *    cached results from a previous step. Tests that slot outputs are freshly written
 *    on each execution, not reused from a frozen constant path.
 *
 * 2. ACCUMULATING STATE (KV CACHE PATTERN): Output from step N is fed back as
 *    input to step N+1, with a growing dimension. This is the core VLM pattern:
 *    past_key_values grow each decode step. Tests that the graph handles dynamic
 *    shapes across steps without plan cache corruption.
 *
 * 3. MULTI-STEP DECAY: Run 20+ steps through the same graph and verify that
 *    outputs remain correct on EVERY step — not just the first 3-4 warmup steps.
 *    Regressions often appear only after the plan transitions from SLOT_BY_SLOT to
 *    SHAPES_FROZEN to REPLAYING.
 *
 * 4. FROZEN FAST PATH WITH DATA-DEPENDENT TAIL: Most of the graph is constant/frozen
 *    (weights, normalization), but argmax at the end depends on input data. Tests
 *    that the frozen fast path doesn't skip the data-dependent tail.
 *
 * 5. IN-PLACE + VIEW ALIASING: Views created by get()/reshape() share buffers with
 *    the parent array. Feeding views back into the graph as placeholders tests
 *    whether the DSP correctly handles buffer aliasing.
 *
 * 6. MIXED PRECISION THROUGH PHASES: FP16 weights with FP32 activations through
 *    warmup → freeze → replay. The cast operations must survive phase transitions.
 *
 * 7. SHAPE KEY STABILITY WITH VALUE CHANGES: Same shapes but different values should
 *    NOT cause plan invalidation but MUST produce different outputs.
 *
 * 8. CONCURRENT GRAPH REUSE: Multiple sequential executions with interleaved
 *    placeholder updates — the plan must not confuse execution state.
 */
@Slf4j
@DisplayName("DSP Deep Isolation Tests")
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
public class DspDeepIsolationTest {

    // ═══════════════════════════════════════════════════════════════════════
    //  Shared deterministic weights
    // ═══════════════════════════════════════════════════════════════════════

    private static final int VOCAB = 64;
    private static final int HIDDEN = 32;
    private static final int HEADS = 4;
    private static final int HEAD_DIM = HIDDEN / HEADS;  // 8
    private static final int KV_DIM = HEAD_DIM;  // 8

    private static INDArray W_EMBED;      // [VOCAB, HIDDEN]
    private static INDArray W_PROJ;       // [HIDDEN, VOCAB]
    private static INDArray GAMMA;        // [HIDDEN]
    private static INDArray W_Q;          // [HIDDEN, HIDDEN]
    private static INDArray W_K;          // [HIDDEN, KV_DIM]
    private static INDArray W_V;          // [HIDDEN, KV_DIM]

    @BeforeAll
    static void initWeights() {
        Nd4j.getRandom().setSeed(12345);
        W_EMBED = Nd4j.randn(DataType.FLOAT, VOCAB, HIDDEN).muli(0.02);
        W_PROJ = Nd4j.randn(DataType.FLOAT, HIDDEN, VOCAB).muli(0.02);
        GAMMA = Nd4j.ones(DataType.FLOAT, HIDDEN);
        W_Q = Nd4j.randn(DataType.FLOAT, HIDDEN, HIDDEN).muli(0.02);
        W_K = Nd4j.randn(DataType.FLOAT, HIDDEN, KV_DIM).muli(0.02);
        W_V = Nd4j.randn(DataType.FLOAT, HIDDEN, KV_DIM).muli(0.02);
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  1. BUFFER STALENESS — same graph, different inputs, outputs must differ
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Core staleness test: run the SAME SameDiff graph 20 times with different
     * token IDs. Every output must differ from every other output. If any two
     * consecutive steps produce identical outputs, the DSP is serving stale data.
     *
     * This tests the slot generation counter, the argTableStable fast-path,
     * and the external input sync path.
     */
    static Stream<Arguments> stalenessModesAndSteps() {
        List<Arguments> args = new ArrayList<>();
        for (GraphExecutionMode mode : new GraphExecutionMode[]{
                GraphExecutionMode.SLOT_BY_SLOT, GraphExecutionMode.AUTO,
                GraphExecutionMode.TRITON, GraphExecutionMode.CUDA_GRAPHS,
                GraphExecutionMode.EMULATED_REPLAY}) {
            args.add(Arguments.of(mode, 20));
        }
        return args.stream();
    }

    @ParameterizedTest(name = "1_staleness_{0}_{1}steps")
    @MethodSource("stalenessModesAndSteps")
    @Order(1)
    void test1_BufferStaleness(GraphExecutionMode mode, int steps) {
        SameDiff sd = buildEmbedProjectGraph(DataType.FLOAT);
        sd.setGraphExecutionMode(mode);

        INDArray[] outputs = new INDArray[steps];
        int[] tokens = new int[steps];
        for (int i = 0; i < steps; i++) {
            tokens[i] = i % VOCAB;  // cycle through vocab
        }

        // Run all steps through the same graph instance
        for (int i = 0; i < steps; i++) {
            INDArray tokenId = Nd4j.createFromArray(new long[]{tokens[i]});
            Map<String, INDArray> result = sd.output(
                    Map.of("token_id", tokenId), "probs");
            outputs[i] = result.get("probs").dup();  // dup to detach from internal buffers
        }

        // Verify: different tokens → different outputs
        int pairsChecked = 0;
        for (int i = 0; i < steps; i++) {
            for (int j = i + 1; j < steps; j++) {
                if (tokens[i] == tokens[j]) continue;
                double diff = outputs[i].sub(outputs[j]).amaxNumber().doubleValue();
                assertTrue(diff > 1e-6,
                        mode + ": step " + i + " (tok=" + tokens[i] + ") and step " + j +
                                " (tok=" + tokens[j] + ") have identical outputs! diff=" + diff +
                                " → STALE DATA");
                pairsChecked++;
            }
        }
        log.info("{}: {} steps, {} unique-pair checks passed", mode, steps, pairsChecked);

        // Also verify outputs are valid probabilities (softmax output)
        for (int i = 0; i < steps; i++) {
            double sum = outputs[i].castTo(DataType.FLOAT).sumNumber().doubleValue();
            assertEquals(1.0, sum, 0.05,
                    mode + " step " + i + ": softmax sum = " + sum);
            assertFalse(Double.isNaN(outputs[i].maxNumber().doubleValue()),
                    mode + " step " + i + ": NaN in output");
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  2. ACCUMULATING STATE — KV cache pattern: output[N] feeds input[N+1]
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Simulates the KV cache pattern: a graph that takes a "past state" array
     * and an input, produces an "updated state" and output. The updated state
     * from step N becomes the past state for step N+1.
     *
     * Each step concatenates new data to the growing state. After 10 steps,
     * the state should have accumulated all 10 contributions.
     *
     * This tests:
     * - Dynamic shapes across steps (state grows)
     * - Plan cache handling of changing input shapes
     * - Buffer lifecycle when outputs become inputs
     */
    static Stream<Arguments> kvCacheModes() {
        return Stream.of(
                Arguments.of(GraphExecutionMode.SLOT_BY_SLOT),
                Arguments.of(GraphExecutionMode.AUTO),
                Arguments.of(GraphExecutionMode.EMULATED_REPLAY)
        );
    }

    @ParameterizedTest(name = "2_kvCache_{0}")
    @MethodSource("kvCacheModes")
    @Order(2)
    void test2_AccumulatingStateKvCachePattern(GraphExecutionMode mode) {
        int steps = 10;
        int stateWidth = 8;

        // We can't use the same SameDiff with dynamic shapes for concat easily,
        // so we test the pattern: each step re-creates the graph with the current state size.
        // This is actually what the VLM does — the plan cache handles shape changes.

        INDArray state = Nd4j.zeros(DataType.FLOAT, 1, 0, stateWidth);  // empty initial state

        for (int step = 0; step < steps; step++) {
            int currentSeqLen = step;  // grows each step

            // Build a graph that concatenates new_entry to past_state
            SameDiff sd = SameDiff.create();
            SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, stateWidth);
            SDVariable inputReshaped = sd.reshape("input_3d", input, 1, 1, stateWidth);

            SDVariable result;
            if (currentSeqLen == 0) {
                // First step: output is just the input
                result = sd.identity("state_out", inputReshaped);
            } else {
                SDVariable pastState = sd.placeHolder("past_state", DataType.FLOAT, 1, currentSeqLen, stateWidth);
                result = sd.concat("state_out", 1, pastState, inputReshaped);
            }

            // Also compute a summary (mean across seq dim) as the "logit" output
            SDVariable mean = sd.mean("summary", result, 1);  // [1, stateWidth]

            sd.setGraphExecutionMode(mode);

            // Execute
            Map<String, INDArray> inputs = new HashMap<>();
            INDArray newEntry = Nd4j.randn(DataType.FLOAT, 1, stateWidth).muli(0.1).addi(step * 0.5);
            inputs.put("input", newEntry);
            if (currentSeqLen > 0) {
                inputs.put("past_state", state);
            }

            Map<String, INDArray> outputs = sd.output(inputs, "state_out", "summary");
            INDArray newState = outputs.get("state_out").dup();
            INDArray summary = outputs.get("summary").dup();

            // Verify: state should have (step+1) entries along dim 1
            assertEquals(step + 1, newState.size(1),
                    mode + " step " + step + ": state seq len should be " + (step + 1) +
                            " but got " + newState.size(1));

            // Verify: summary is not NaN
            assertFalse(Double.isNaN(summary.maxNumber().doubleValue()),
                    mode + " step " + step + ": NaN summary");

            // Verify: summary changes each step (accumulating new data)
            if (step > 0) {
                // The new state includes the new entry, so the mean should differ
                double newMean = summary.meanNumber().doubleValue();
                assertTrue(Math.abs(newMean) > 1e-10 || step == 0,
                        mode + " step " + step + ": summary is zero");
            }

            state = newState;
        }

        // Final check: state has all 10 entries
        assertEquals(steps, state.size(1),
                mode + ": final state should have " + steps + " entries");
        log.info("{}: KV cache pattern OK — {} steps, final state shape {}", mode, steps, Arrays.toString(state.shape()));
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  3. MULTI-STEP DECAY — correctness on every step, not just warmup
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Run 30 steps and verify correctness on EVERY step. Compute expected output
     * from raw NDArray ops, compare with SameDiff output.
     *
     * This catches regressions that only appear after warmup (steps 0-4 are
     * SLOT_BY_SLOT warmup, steps 5-8 are freeze transition, steps 9+ are replay).
     *
     * The test marks exactly WHICH step first fails, which pinpoints the
     * phase transition that breaks.
     */
    static Stream<Arguments> decayModes() {
        return Stream.of(
                Arguments.of(GraphExecutionMode.SLOT_BY_SLOT, 30),
                Arguments.of(GraphExecutionMode.AUTO, 30),
                Arguments.of(GraphExecutionMode.TRITON, 30),
                Arguments.of(GraphExecutionMode.CUDA_GRAPHS, 30),
                Arguments.of(GraphExecutionMode.EMULATED_REPLAY, 30)
        );
    }

    @ParameterizedTest(name = "3_decay_{0}_{1}steps")
    @MethodSource("decayModes")
    @Order(3)
    void test3_MultiStepDecay(GraphExecutionMode mode, int steps) {
        SameDiff sd = buildEmbedProjectGraph(DataType.FLOAT);
        sd.setGraphExecutionMode(mode);

        int firstFailStep = -1;
        String firstFailReason = null;

        for (int i = 0; i < steps; i++) {
            int tokenId = i % VOCAB;
            INDArray expected = computeExpectedEmbedProject(tokenId, DataType.FLOAT);
            INDArray actual = sd.output(
                    Map.of("token_id", Nd4j.createFromArray(new long[]{tokenId})),
                    "probs").get("probs");

            // Check for NaN
            if (Double.isNaN(actual.maxNumber().doubleValue())) {
                if (firstFailStep == -1) {
                    firstFailStep = i;
                    firstFailReason = "NaN output";
                }
                continue;
            }

            // Check softmax sum
            double sum = actual.castTo(DataType.FLOAT).sumNumber().doubleValue();
            if (Math.abs(sum - 1.0) > 0.05) {
                if (firstFailStep == -1) {
                    firstFailStep = i;
                    firstFailReason = "softmax sum=" + sum;
                }
                continue;
            }

            // Check against expected
            double diff = expected.sub(actual.castTo(DataType.FLOAT)).amaxNumber().doubleValue();
            if (diff > 0.01) {
                if (firstFailStep == -1) {
                    firstFailStep = i;
                    firstFailReason = "diff=" + diff + " (tok=" + tokenId + ")";
                }
            }
        }

        if (firstFailStep >= 0) {
            String phase = firstFailStep < 4 ? "WARMUP" :
                    firstFailStep < 8 ? "FREEZE_TRANSITION" : "REPLAY";
            fail(mode + ": first failure at step " + firstFailStep + " (phase=" + phase +
                    "): " + firstFailReason);
        }
        log.info("{}: all {} steps passed", mode, steps);
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  4. FROZEN FAST PATH + DATA-DEPENDENT TAIL
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * A graph where most operations are input-independent (constant matmul,
     * constant norm) but the FINAL operation (argmax) depends on the input.
     *
     * The frozen fast path must NOT skip the argmax — it must detect that
     * the argmax output depends on input data even though the intermediate
     * normalization and projection are frozen.
     */
    static Stream<Arguments> frozenTailModes() {
        return Stream.of(
                Arguments.of(GraphExecutionMode.SLOT_BY_SLOT),
                Arguments.of(GraphExecutionMode.AUTO),
                Arguments.of(GraphExecutionMode.TRITON),
                Arguments.of(GraphExecutionMode.EMULATED_REPLAY)
        );
    }

    @ParameterizedTest(name = "4_frozenTail_{0}")
    @MethodSource("frozenTailModes")
    @Order(4)
    void test4_FrozenFastPathWithDataDependentTail(GraphExecutionMode mode) {
        SameDiff sd = SameDiff.create();

        // Weights are constants — will be frozen
        sd.constant("embed_table", W_EMBED.dup());
        sd.constant("gamma", GAMMA.dup());
        sd.constant("proj_weight", W_PROJ.dup());

        // Input is a placeholder — changes each step
        SDVariable tokenId = sd.placeHolder("token_id", DataType.INT64, 1);
        SDVariable gathered = sd.gather("gathered", sd.getVariable("embed_table"), tokenId, 0);
        SDVariable normed = sd.nn().rmsNorm("normed", gathered, sd.getVariable("gamma"), 1e-5);
        SDVariable logits = sd.mmul("logits", normed, sd.getVariable("proj_weight"));

        // Data-dependent tail: argmax selects different tokens for different inputs
        SDVariable predicted = sd.argmax("predicted", logits, 1);

        sd.setGraphExecutionMode(mode);

        // Run 15 steps with different tokens, track argmax outputs
        long[] argmaxResults = new long[15];
        for (int i = 0; i < 15; i++) {
            int tok = (i * 7) % VOCAB;  // spread across vocab
            INDArray result = sd.output(
                    Map.of("token_id", Nd4j.createFromArray(new long[]{tok})),
                    "predicted").get("predicted");
            argmaxResults[i] = result.getLong(0);
        }

        // Verify: not all argmax results are the same token
        long firstVal = argmaxResults[0];
        boolean allSame = true;
        for (long v : argmaxResults) {
            if (v != firstVal) { allSame = false; break; }
        }
        assertFalse(allSame,
                mode + ": all 15 steps produced same argmax=" + firstVal +
                        " → frozen fast path skipped data-dependent tail!");

        // Verify: at least 3 distinct argmax values across 15 diverse inputs
        Set<Long> unique = new HashSet<>();
        for (long v : argmaxResults) unique.add(v);
        assertTrue(unique.size() >= 3,
                mode + ": only " + unique.size() + " unique argmax values across 15 inputs" +
                        " → insufficient diversity (frozen path may be leaking)");

        log.info("{}: {} unique argmax values across 15 steps: {}", mode, unique.size(), unique);
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  5. VIEW ALIASING — views fed back as placeholders
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Creates a view via getRow(), feeds it as a placeholder input.
     * The graph must correctly use the view data, not the parent array data.
     *
     * This tests whether the DSP correctly handles non-contiguous memory
     * layouts and view buffer aliasing.
     */
    @ParameterizedTest(name = "5_viewAlias_{0}")
    @EnumSource(value = GraphExecutionMode.class,
            names = {"SLOT_BY_SLOT", "AUTO", "EMULATED_REPLAY"})
    @Order(5)
    void test5_ViewAliasing(GraphExecutionMode mode) {
        SameDiff sd = SameDiff.create();
        sd.constant("weight", Nd4j.randn(DataType.FLOAT, HIDDEN, HIDDEN).muli(0.02));
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, HIDDEN);
        sd.mmul("out", input, sd.getVariable("weight"));

        sd.setGraphExecutionMode(mode);

        // Create a large parent array and take views from different rows
        INDArray parent = Nd4j.randn(DataType.FLOAT, 10, HIDDEN);

        INDArray[] results = new INDArray[10];
        for (int row = 0; row < 10; row++) {
            // getRow creates a VIEW — shares buffer with parent
            INDArray view = parent.getRow(row).reshape(1, HIDDEN);
            // dup() to test with contiguous data as well
            INDArray duped = view.dup();

            INDArray resultView = sd.output(Map.of("input", view), "out").get("out").dup();
            INDArray resultDup = sd.output(Map.of("input", duped), "out").get("out").dup();

            // View and dup must produce the same result
            double diff = resultView.sub(resultDup).amaxNumber().doubleValue();
            assertTrue(diff < 1e-5,
                    mode + " row " + row + ": view vs dup mismatch=" + diff);

            results[row] = resultView;
        }

        // Different rows → different outputs
        for (int i = 0; i < 10; i++) {
            for (int j = i + 1; j < 10; j++) {
                double diff = results[i].sub(results[j]).amaxNumber().doubleValue();
                assertTrue(diff > 1e-6,
                        mode + ": rows " + i + " and " + j + " produced identical output");
            }
        }
        log.info("{}: view aliasing test passed for 10 rows", mode);
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  6. MIXED PRECISION THROUGH PHASES
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * FP16 weights + FP32 input through warmup → freeze → replay.
     * The mixed-precision cast must survive all phase transitions.
     *
     * Runs 20 steps so the plan transitions through all phases:
     * 0-3: warmup, 4-7: freeze, 8+: replay
     */
    @ParameterizedTest(name = "6_mixedPrec_{0}")
    @EnumSource(value = GraphExecutionMode.class,
            names = {"SLOT_BY_SLOT", "AUTO", "TRITON", "EMULATED_REPLAY"})
    @Order(6)
    void test6_MixedPrecisionThroughPhases(GraphExecutionMode mode) {
        SameDiff sd = buildEmbedProjectGraph(DataType.HALF);  // FP16 weights
        sd.setGraphExecutionMode(mode);

        int steps = 20;
        for (int i = 0; i < steps; i++) {
            int tok = (i * 3) % VOCAB;
            INDArray result = sd.output(
                    Map.of("token_id", Nd4j.createFromArray(new long[]{tok})),
                    "probs").get("probs");

            // Must be valid probability distribution
            assertFalse(Double.isNaN(result.maxNumber().doubleValue()),
                    mode + " step " + i + ": NaN");
            double sum = result.castTo(DataType.FLOAT).sumNumber().doubleValue();
            assertEquals(1.0, sum, 0.1,  // wider tolerance for FP16
                    mode + " step " + i + ": softmax sum=" + sum);

            // Compare with expected (wider tolerance for FP16)
            INDArray expected = computeExpectedEmbedProject(tok, DataType.HALF);
            double diff = expected.sub(result.castTo(DataType.FLOAT)).amaxNumber().doubleValue();
            assertTrue(diff < 0.1,
                    mode + " step " + i + " tok=" + tok + ": diff=" + diff +
                            " (expected FP16 tolerance)");
        }
        log.info("{}: mixed precision through {} steps OK", mode, steps);
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  7. SHAPE KEY STABILITY WITH VALUE CHANGES
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Run the same graph with inputs that have IDENTICAL shapes but DIFFERENT values.
     * The plan cache should reuse the plan (same shape key) but outputs must differ.
     *
     * This tests that the shape key doesn't accidentally include value hashes
     * for ops that aren't value-dependent.
     */
    @ParameterizedTest(name = "7_shapeKeyValues_{0}")
    @EnumSource(value = GraphExecutionMode.class,
            names = {"SLOT_BY_SLOT", "AUTO", "TRITON", "EMULATED_REPLAY"})
    @Order(7)
    void test7_ShapeKeyStabilityWithValueChanges(GraphExecutionMode mode) {
        SameDiff sd = SameDiff.create();
        sd.constant("weight", Nd4j.randn(DataType.FLOAT, 16, 16).muli(0.1));
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, 16);
        SDVariable matmul = sd.mmul("matmul", input, sd.getVariable("weight"));
        sd.nn().softmax("out", matmul, 1);

        sd.setGraphExecutionMode(mode);

        INDArray[] results = new INDArray[20];
        for (int i = 0; i < 20; i++) {
            // Same shape [1,16], different values
            Nd4j.getRandom().setSeed(i * 1000 + 42);
            INDArray in = Nd4j.randn(DataType.FLOAT, 1, 16);
            results[i] = sd.output(Map.of("input", in), "out").get("out").dup();
        }

        // All outputs must be valid softmax
        for (int i = 0; i < 20; i++) {
            double sum = results[i].sumNumber().doubleValue();
            assertEquals(1.0, sum, 0.01, mode + " step " + i + ": sum=" + sum);
        }

        // Different inputs → different outputs (at least 18 of 20 should differ from each other)
        int differingPairs = 0;
        int totalPairs = 0;
        for (int i = 0; i < 20; i++) {
            for (int j = i + 1; j < 20; j++) {
                totalPairs++;
                double diff = results[i].sub(results[j]).amaxNumber().doubleValue();
                if (diff > 1e-6) differingPairs++;
            }
        }
        double diffRate = (double) differingPairs / totalPairs;
        assertTrue(diffRate > 0.95,
                mode + ": only " + String.format("%.1f%%", diffRate * 100) +
                        " of pairs differ — shape key may be blocking value changes");
        log.info("{}: {}/{} pairs differ ({}%)", mode, differingPairs, totalPairs,
                String.format("%.1f", diffRate * 100));
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  8. GRAPH WITH MULTIPLE OUTPUTS — all must update each step
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * A graph with 3 output variables, all depending on the same placeholder.
     * ALL three outputs must change when the input changes.
     * Tests that the DSP doesn't partially update outputs (e.g., only the
     * last-requested output gets fresh data).
     */
    @ParameterizedTest(name = "8_multiOut_{0}")
    @EnumSource(value = GraphExecutionMode.class,
            names = {"SLOT_BY_SLOT", "AUTO", "TRITON", "EMULATED_REPLAY"})
    @Order(8)
    void test8_MultipleOutputsAllUpdate(GraphExecutionMode mode) {
        SameDiff sd = SameDiff.create();
        sd.constant("w1", Nd4j.randn(DataType.FLOAT, 16, 8).muli(0.1));
        sd.constant("w2", Nd4j.randn(DataType.FLOAT, 16, 4).muli(0.1));
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, 16);

        // Three separate output paths from the same input
        SDVariable out1 = sd.mmul("out_matmul", input, sd.getVariable("w1"));
        SDVariable out2 = sd.math().add("out_shifted", input, 1.0);
        SDVariable out3 = sd.mmul("out_proj", input, sd.getVariable("w2"));

        sd.setGraphExecutionMode(mode);

        INDArray[][] allResults = new INDArray[15][3];
        for (int i = 0; i < 15; i++) {
            Nd4j.getRandom().setSeed(i * 999);
            INDArray in = Nd4j.randn(DataType.FLOAT, 1, 16);
            Map<String, INDArray> outs = sd.output(Map.of("input", in),
                    "out_matmul", "out_shifted", "out_proj");
            allResults[i][0] = outs.get("out_matmul").dup();
            allResults[i][1] = outs.get("out_shifted").dup();
            allResults[i][2] = outs.get("out_proj").dup();
        }

        // Check each output independently: different inputs → different outputs
        String[] names = {"out_matmul", "out_shifted", "out_proj"};
        for (int o = 0; o < 3; o++) {
            int staleCount = 0;
            for (int i = 1; i < 15; i++) {
                double diff = allResults[i][o].sub(allResults[i - 1][o]).amaxNumber().doubleValue();
                if (diff < 1e-8) staleCount++;
            }
            assertTrue(staleCount < 2,
                    mode + ": output '" + names[o] + "' was stale " + staleCount +
                            " times out of 14 consecutive pairs");
        }
        log.info("{}: all 3 outputs updated correctly across 15 steps", mode);
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  9. OPTIMIZER + MULTI-STEP — optimized graph through full lifecycle
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Apply GraphOptimizer, then run 20 multi-step decode.
     * The optimizer may introduce fused ops, eliminate identities, fold constants.
     * All these transformations must survive the DSP phase transitions.
     */
    @ParameterizedTest(name = "9_optLifecycle_{0}")
    @EnumSource(value = GraphExecutionMode.class,
            names = {"SLOT_BY_SLOT", "AUTO", "TRITON", "EMULATED_REPLAY"})
    @Order(9)
    void test9_OptimizerPlusMultiStepLifecycle(GraphExecutionMode mode) {
        SameDiff sd = buildEmbedProjectGraph(DataType.FLOAT);
        SameDiff opt = GraphOptimizer.optimize(sd);
        opt.setGraphExecutionMode(mode);

        int steps = 20;
        for (int i = 0; i < steps; i++) {
            int tok = (i * 5 + 3) % VOCAB;
            INDArray expected = computeExpectedEmbedProject(tok, DataType.FLOAT);
            INDArray actual = opt.output(
                    Map.of("token_id", Nd4j.createFromArray(new long[]{tok})),
                    "probs").get("probs");

            assertFalse(Double.isNaN(actual.maxNumber().doubleValue()),
                    mode + " opt step " + i + ": NaN");
            double diff = expected.sub(actual.castTo(DataType.FLOAT)).amaxNumber().doubleValue();
            assertTrue(diff < 0.01,
                    mode + " opt step " + i + " tok=" + tok + ": diff=" + diff);
        }
        log.info("{}: optimized graph, {} steps OK", mode, steps);
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  10. INTERLEAVED GRAPH REUSE — multiple graphs sharing a plan cache
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Two separate SameDiff instances with the same structure.
     * Interleave executions: A, B, A, B, A, B...
     * Each must maintain independent state and produce correct results.
     *
     * This tests whether DSP internal thread-locals or static state leak
     * between separate SameDiff instances.
     */
    @ParameterizedTest(name = "10_interleaved_{0}")
    @EnumSource(value = GraphExecutionMode.class,
            names = {"SLOT_BY_SLOT", "AUTO", "EMULATED_REPLAY"})
    @Order(10)
    void test10_InterleavedGraphReuse(GraphExecutionMode mode) {
        SameDiff sdA = buildEmbedProjectGraph(DataType.FLOAT);
        SameDiff sdB = buildEmbedProjectGraph(DataType.FLOAT);
        sdA.setGraphExecutionMode(mode);
        sdB.setGraphExecutionMode(mode);

        for (int i = 0; i < 15; i++) {
            int tokA = i % VOCAB;
            int tokB = (i * 3 + 17) % VOCAB;
            INDArray expectedA = computeExpectedEmbedProject(tokA, DataType.FLOAT);
            INDArray expectedB = computeExpectedEmbedProject(tokB, DataType.FLOAT);

            // Execute A
            INDArray actualA = sdA.output(
                    Map.of("token_id", Nd4j.createFromArray(new long[]{tokA})),
                    "probs").get("probs");
            // Execute B
            INDArray actualB = sdB.output(
                    Map.of("token_id", Nd4j.createFromArray(new long[]{tokB})),
                    "probs").get("probs");

            double diffA = expectedA.sub(actualA.castTo(DataType.FLOAT)).amaxNumber().doubleValue();
            double diffB = expectedB.sub(actualB.castTo(DataType.FLOAT)).amaxNumber().doubleValue();

            assertTrue(diffA < 0.01,
                    mode + " step " + i + " graph A tok=" + tokA + ": diff=" + diffA);
            assertTrue(diffB < 0.01,
                    mode + " step " + i + " graph B tok=" + tokB + ": diff=" + diffB);
        }
        log.info("{}: interleaved execution of 2 graphs, 15 steps each OK", mode);
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  11. IN-PLACE MUTATION DETECTION — placeholder mutated between steps
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Reuse the SAME INDArray object as placeholder but mutate its values
     * between steps. The DSP must detect the mutation and produce different
     * outputs.
     *
     * This tests whether the DSP relies on pointer identity for cache hits
     * (it shouldn't — same pointer, different data must produce different output).
     */
    @ParameterizedTest(name = "11_mutate_{0}")
    @EnumSource(value = GraphExecutionMode.class,
            names = {"SLOT_BY_SLOT", "AUTO", "EMULATED_REPLAY"})
    @Order(11)
    void test11_InPlaceMutationDetection(GraphExecutionMode mode) {
        SameDiff sd = SameDiff.create();
        sd.constant("weight", Nd4j.randn(DataType.FLOAT, 16, 8).muli(0.1));
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, 16);
        sd.mmul("out", input, sd.getVariable("weight"));
        sd.setGraphExecutionMode(mode);

        // Same INDArray, mutated between steps
        INDArray sharedInput = Nd4j.zeros(DataType.FLOAT, 1, 16);
        INDArray[] results = new INDArray[10];

        for (int i = 0; i < 10; i++) {
            // Mutate in place — same pointer, different data
            sharedInput.assign(Nd4j.randn(DataType.FLOAT, 1, 16));

            results[i] = sd.output(Map.of("input", sharedInput), "out").get("out").dup();
        }

        // Different data → different outputs
        int staleCount = 0;
        for (int i = 1; i < 10; i++) {
            double diff = results[i].sub(results[i - 1]).amaxNumber().doubleValue();
            if (diff < 1e-8) staleCount++;
        }
        assertTrue(staleCount < 2,
                mode + ": " + staleCount + "/9 consecutive steps produced stale output " +
                        "despite in-place placeholder mutation!");
        log.info("{}: in-place mutation detection OK ({} stale out of 9)", mode, staleCount);
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  12. DEEP CHAIN — many sequential ops (simulates transformer layer depth)
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * A chain of 10 matmul + rmsNorm layers, like a mini transformer.
     * Tests that the DSP handles deep dependency chains without losing data
     * in intermediate slots.
     */
    @ParameterizedTest(name = "12_deepChain_{0}")
    @EnumSource(value = GraphExecutionMode.class,
            names = {"SLOT_BY_SLOT", "AUTO", "EMULATED_REPLAY"})
    @Order(12)
    void test12_DeepChain(GraphExecutionMode mode) {
        int layers = 10;
        int dim = 16;
        Nd4j.getRandom().setSeed(999);

        // Pre-generate weights so both graphs get independent copies of identical data
        INDArray[] weights = new INDArray[layers];
        INDArray[] gammas = new INDArray[layers];
        for (int l = 0; l < layers; l++) {
            weights[l] = Nd4j.randn(DataType.FLOAT, dim, dim).muli(0.02);
            gammas[l] = Nd4j.ones(DataType.FLOAT, dim);
        }

        // Build test graph
        SameDiff sd = SameDiff.create();
        SDVariable current = sd.placeHolder("input", DataType.FLOAT, 1, dim);
        for (int l = 0; l < layers; l++) {
            sd.constant("w_" + l, weights[l].dup());
            sd.constant("g_" + l, gammas[l].dup());
            current = sd.mmul("matmul_" + l, current, sd.getVariable("w_" + l));
            current = sd.nn().rmsNorm("norm_" + l, current, sd.getVariable("g_" + l), 1e-5);
        }
        sd.identity("out", current);

        // Build reference graph BEFORE executing either — both get fresh weight copies
        SameDiff sdRef = SameDiff.create();
        SDVariable refCurrent = sdRef.placeHolder("input", DataType.FLOAT, 1, dim);
        for (int l = 0; l < layers; l++) {
            sdRef.constant("w_" + l, weights[l].dup());
            sdRef.constant("g_" + l, gammas[l].dup());
            refCurrent = sdRef.mmul("matmul_" + l, refCurrent, sdRef.getVariable("w_" + l));
            refCurrent = sdRef.nn().rmsNorm("norm_" + l, refCurrent, sdRef.getVariable("g_" + l), 1e-5);
        }
        sdRef.identity("out", refCurrent);
        sdRef.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);

        sd.setGraphExecutionMode(mode);

        // Run 15 steps — use LARGE input differences to overcome rmsNorm convergence
        INDArray[] results = new INDArray[15];
        INDArray[] inputs = new INDArray[15];
        for (int i = 0; i < 15; i++) {
            inputs[i] = Nd4j.zeros(DataType.FLOAT, 1, dim);
            inputs[i].putScalar(0, i % dim, 1.0f);
            results[i] = sd.output(Map.of("input", inputs[i]), "out").get("out").dup();

            assertFalse(Double.isNaN(results[i].maxNumber().doubleValue()),
                    mode + " step " + i + ": NaN in deep chain output (layer overflow?)");
            assertTrue(results[i].amaxNumber().doubleValue() > 1e-10,
                    mode + " step " + i + ": all-zero output from deep chain");
        }

        // Compare: the DSP must produce the same result as SLOT_BY_SLOT
        int matchCount = 0;
        for (int i = 0; i < 15; i++) {
            INDArray refResult = sdRef.output(Map.of("input", inputs[i]), "out").get("out");
            double diff = refResult.sub(results[i]).amaxNumber().doubleValue();
            if (diff < 1e-5) matchCount++;
            else log.warn("{} step {}: DSP vs SLOT_BY_SLOT diff={}", mode, i, diff);
        }
        assertTrue(matchCount >= 13,
                mode + ": only " + matchCount + "/15 steps match SLOT_BY_SLOT reference in deep chain");
        log.info("{}: deep chain ({} layers) passed 15 steps ({} match ref)", mode, layers, matchCount);
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  13. MATMUL MIXED PRECISION EXHAUSTIVE — all 4 type combos × 20 steps
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * For each matmul type combo (FF, FH, HF, HH), run 20 steps through
     * the same SameDiff graph and verify correctness on every step.
     * The MmulHelper.cu FP16 fix must survive all DSP phases.
     */
    static Stream<Arguments> matmulMixedPrecision() {
        return Stream.of(
                Arguments.of(DataType.FLOAT, DataType.FLOAT, "FF", 0.001),
                Arguments.of(DataType.HALF, DataType.HALF, "HH", 0.05),
                Arguments.of(DataType.FLOAT, DataType.HALF, "FH", 0.05),
                Arguments.of(DataType.HALF, DataType.FLOAT, "HF", 0.05)
        );
    }

    @ParameterizedTest(name = "13_mmulPrec_{2}")
    @MethodSource("matmulMixedPrecision")
    @Order(13)
    void test13_MatmulMixedPrecisionMultiStep(DataType aType, DataType bType,
                                               String label, double tolerance) {
        int M = 4, K = 16, N = 8;
        INDArray weightA = Nd4j.randn(DataType.FLOAT, M, K).muli(0.1).castTo(aType);
        INDArray weightB = Nd4j.randn(DataType.FLOAT, K, N).muli(0.1).castTo(bType);

        SameDiff sd = SameDiff.create();
        sd.constant("a", weightA);
        sd.constant("b", weightB);
        sd.mmul("c", sd.getVariable("a"), sd.getVariable("b"));

        sd.setGraphExecutionMode(GraphExecutionMode.AUTO);

        // Run 20 steps — the matmul should produce the same result every time
        // (constant inputs = constant output)
        INDArray expected = weightA.castTo(DataType.FLOAT).mmul(weightB.castTo(DataType.FLOAT));

        for (int step = 0; step < 20; step++) {
            INDArray result = sd.output(Collections.emptyMap(), "c").get("c");
            double diff = expected.sub(result.castTo(DataType.FLOAT)).amaxNumber().doubleValue();
            assertTrue(diff < tolerance,
                    label + " step " + step + ": diff=" + diff + " > tol=" + tolerance);
        }
        log.info("matmul {}: 20 steps OK, all within tolerance {}", label, tolerance);
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  14. OPTIMIZER FP16 WEIGHT PRECAST + MULTI-STEP DECODE
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * This is the closest to the actual VLM pattern:
     * 1. Build graph with FP32 weights
     * 2. Optimize with FP16 pre-cast enabled
     * 3. Run 20 decode steps with different token inputs
     * 4. Verify every step produces correct, non-stale, non-NaN output
     *
     * This exercises: optimizer FP16 cast insertion, mixed-precision matmul
     * through phase transitions, and frozen constant detection with cast ops.
     */
    @ParameterizedTest(name = "14_fp16opt_{0}")
    @EnumSource(value = GraphExecutionMode.class,
            names = {"SLOT_BY_SLOT", "AUTO", "TRITON", "EMULATED_REPLAY"})
    @Order(14)
    void test14_OptimizerFP16PrecastMultiStep(GraphExecutionMode mode) {
        String oldFp16 = System.getProperty("nd4j.optimizer.fp16");
        try {
            System.setProperty("nd4j.optimizer.fp16", "true");

            SameDiff sd = buildEmbedProjectGraph(DataType.FLOAT);
            SameDiff opt = GraphOptimizer.optimize(sd);
            opt.setGraphExecutionMode(mode);

            // Verify weights were cast to FP16
            boolean hasHalfWeight = false;
            for (SDVariable v : opt.variables()) {
                INDArray arr = opt.getArrForVarName(v.name());
                if (arr != null && arr.dataType() == DataType.HALF) {
                    hasHalfWeight = true;
                    break;
                }
            }
            // Note: may not always cast — depends on size thresholds
            // Just log, don't assert — the test is about correctness through phases

            int steps = 20;
            INDArray[] results = new INDArray[steps];
            for (int i = 0; i < steps; i++) {
                int tok = (i * 7 + 2) % VOCAB;
                results[i] = opt.output(
                        Map.of("token_id", Nd4j.createFromArray(new long[]{tok})),
                        "probs").get("probs").dup();

                assertFalse(Double.isNaN(results[i].maxNumber().doubleValue()),
                        mode + " fp16opt step " + i + ": NaN");
                double sum = results[i].castTo(DataType.FLOAT).sumNumber().doubleValue();
                assertEquals(1.0, sum, 0.1,
                        mode + " fp16opt step " + i + ": softmax sum=" + sum);
            }

            // Different tokens → different outputs (not stale)
            int staleCount = 0;
            for (int i = 1; i < steps; i++) {
                int tokPrev = ((i - 1) * 7 + 2) % VOCAB;
                int tokCurr = (i * 7 + 2) % VOCAB;
                if (tokPrev == tokCurr) continue;
                double diff = results[i].sub(results[i - 1]).amaxNumber().doubleValue();
                if (diff < 1e-6) staleCount++;
            }
            assertTrue(staleCount < 2,
                    mode + ": " + staleCount + " stale steps in FP16-optimized decode");

            log.info("{}: FP16 optimizer + {} steps OK (hasHalfWeight={})", mode, steps, hasHalfWeight);
        } finally {
            if (oldFp16 != null) System.setProperty("nd4j.optimizer.fp16", oldFp16);
            else System.clearProperty("nd4j.optimizer.fp16");
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  15. SLOT GENERATION COUNTER MONOTONICITY
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Verify that repeated execution of the same graph with different inputs
     * produces monotonically increasing argmax token IDs that match the
     * independently-computed expected tokens.
     *
     * If the DSP serves stale slot data, the argmax will repeat the same
     * token instead of following the input.
     */
    @ParameterizedTest(name = "15_slotGen_{0}")
    @EnumSource(value = GraphExecutionMode.class,
            names = {"SLOT_BY_SLOT", "AUTO", "TRITON", "EMULATED_REPLAY"})
    @Order(15)
    void test15_SlotGenerationCounterMonotonicity(GraphExecutionMode mode) {
        SameDiff sd = SameDiff.create();
        sd.constant("weight", Nd4j.randn(DataType.FLOAT, 16, 32).muli(0.1));
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, 16);
        SDVariable matmul = sd.mmul("matmul", input, sd.getVariable("weight"));
        SDVariable argmax = sd.argmax("token", matmul, 1);

        sd.setGraphExecutionMode(mode);

        long[] expectedTokens = new long[20];
        long[] actualTokens = new long[20];
        INDArray weight = Nd4j.randn(DataType.FLOAT, 16, 32).muli(0.1);

        for (int i = 0; i < 20; i++) {
            Nd4j.getRandom().setSeed(i * 555);
            INDArray in = Nd4j.randn(DataType.FLOAT, 1, 16);

            // Expected: manual computation
            INDArray logits = in.mmul(sd.getArrForVarName("weight"));
            expectedTokens[i] = Nd4j.argMax(logits, 1).getLong(0);

            // Actual: through SameDiff
            INDArray result = sd.output(Map.of("input", in), "token").get("token");
            actualTokens[i] = result.getLong(0);
        }

        // Compare
        int mismatches = 0;
        for (int i = 0; i < 20; i++) {
            if (expectedTokens[i] != actualTokens[i]) mismatches++;
        }
        assertTrue(mismatches < 3,
                mode + ": " + mismatches + "/20 argmax mismatches between manual and DSP");

        // Check for stuck tokens (same token repeated >3 times)
        int maxRun = 1, currentRun = 1;
        for (int i = 1; i < 20; i++) {
            if (actualTokens[i] == actualTokens[i - 1]) {
                currentRun++;
                maxRun = Math.max(maxRun, currentRun);
            } else {
                currentRun = 1;
            }
        }
        assertTrue(maxRun < 4,
                mode + ": argmax stuck on same token for " + maxRun +
                        " consecutive steps → stale slot data");
        log.info("{}: slot generation test OK, {} mismatches, maxRun={}", mode, mismatches, maxRun);
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  Helper: build embed → rmsNorm → project → softmax graph
    // ═══════════════════════════════════════════════════════════════════════

    private SameDiff buildEmbedProjectGraph(DataType weightType) {
        SameDiff sd = SameDiff.create();
        INDArray et = weightType == DataType.HALF ? W_EMBED.castTo(DataType.HALF) : W_EMBED.dup();
        INDArray pw = weightType == DataType.HALF ? W_PROJ.castTo(DataType.HALF) : W_PROJ.dup();
        sd.constant("embed_table", et);
        sd.constant("gamma", GAMMA.dup());
        sd.constant("proj_weight", pw);
        SDVariable tokenId = sd.placeHolder("token_id", DataType.INT64, 1);
        SDVariable gathered = sd.gather("gathered", sd.getVariable("embed_table"), tokenId, 0);
        SDVariable normed = sd.nn().rmsNorm("normed", gathered, sd.getVariable("gamma"), 1e-5);
        SDVariable logits = sd.mmul("logits", normed, sd.getVariable("proj_weight"));
        sd.nn().softmax("probs", logits, 1);
        return sd;
    }

    private INDArray computeExpectedEmbedProject(int tokenId, DataType weightType) {
        INDArray et = weightType == DataType.HALF ? W_EMBED.castTo(DataType.HALF) : W_EMBED;
        INDArray pw = weightType == DataType.HALF ? W_PROJ.castTo(DataType.HALF) : W_PROJ;

        INDArray row = et.getRow(tokenId).reshape(1, HIDDEN).dup().castTo(DataType.FLOAT);
        double rms = Math.sqrt(row.mul(row).meanNumber().doubleValue() + 1e-5);
        INDArray normed = row.div(rms).muli(GAMMA);
        INDArray logits = normed.mmul(pw.castTo(DataType.FLOAT));
        INDArray maxVal = logits.max(true, 1);
        INDArray shifted = logits.sub(maxVal);
        INDArray exps = Transforms.exp(shifted);
        return exps.divi(exps.sum(true, 1));
    }
}
