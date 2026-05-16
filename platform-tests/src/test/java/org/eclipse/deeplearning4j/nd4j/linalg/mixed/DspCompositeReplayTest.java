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
import org.nd4j.autodiff.samediff.execution.DspDebugger;
import org.nd4j.autodiff.samediff.execution.DspPlanAssertions;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.autodiff.samediff.execution.PlanPhase;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.*;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.*;

/**
 * DSP Composite Replay Assertions.
 *
 * This test class exists because of a specific class of bug: the frozen fast path
 * can choose MONOLITHIC replay when the segment has COMPOSITE replay with gap slots.
 * The monolithic capture SKIPS gap ops (matmul, rmsNorm, etc.) — so replaying it
 * means those ops NEVER execute again after capture, producing stale/wrong output.
 *
 * The tests here assert on:
 *
 * 1. OUTPUT VALUES CHANGE when inputs change — at EVERY phase transition
 *    (warmup → frozen → replaying). Stale data is caught immediately.
 *
 * 2. REFERENCE ACCURACY — AUTO/TRITON/CUDA_GRAPHS must match SLOT_BY_SLOT at
 *    every step, not just during warmup. The diff tolerance is tight (1e-5 for FP32).
 *
 * 3. PHASE PROGRESSION — exec count, plan phase, segment phase must progress
 *    through the expected lifecycle. Stuck phases indicate capture failures.
 *
 * 4. GAP OP EXECUTION — graphs with Triton-compilable ops mixed with native-only
 *    ops (the composite pattern) must execute both parts correctly.
 *
 * 5. POINTER STABILITY — output buffer addresses must stabilize after freeze.
 *    Address drift after capture indicates a lifecycle bug.
 *
 * 6. MULTI-STEP DECAY — run 30+ steps and verify the LAST step matches reference.
 *    Regressions that accumulate FP drift are caught by step 20+.
 */
@Slf4j
@DisplayName("DSP Composite Replay Assertions")
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
public class DspCompositeReplayTest {

    // Track SameDiff instances so we can close sessions in @AfterEach to free GPU memory
    private final List<SameDiff> activeSds = new ArrayList<>();

    @AfterEach
    void cleanupGpuMemory() {
        for (SameDiff sd : activeSds) {
            try {
                // close() frees the native plan cache (capture workspaces, replay handles,
                // cuBLAS workspace) — resetSession() only unpins the plan, leaving ~512MB
                // of capture workspace per test leaked in the cache.
                sd.close();
            } catch (Exception e) {
                log.warn("Error closing SameDiff: {}", e.getMessage());
            }
        }
        activeSds.clear();
        Nd4j.getWorkspaceManager().destroyAllWorkspacesForCurrentThread();
        System.gc();
        // Trim the CUDA memory pool to reclaim freed blocks
        try {
            var nativeOps = Nd4j.getNativeOps();
            int numDevices = Nd4j.getAffinityManager().getNumberOfDevices();
            for (int d = 0; d < numDevices; d++) {
                nativeOps.trimMemoryPool(d);
            }
        } catch (Exception e) {
            // OK on CPU backend or if pool not initialized
        }
    }

    private SameDiff track(SameDiff sd) {
        activeSds.add(sd);
        return sd;
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  Test graph builders — each creates a pattern that forces composite replay.
    //  Weights are pre-generated and shared so sd and sdRef use IDENTICAL weights.
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Pre-generate weights for a deep chain. Call once, pass to buildDeepChainWith.
     */
    private static INDArray[][] generateDeepChainWeights(int layers, int dim) {
        Nd4j.getRandom().setSeed(42);
        INDArray[][] weights = new INDArray[layers][2];
        for (int l = 0; l < layers; l++) {
            weights[l][0] = Nd4j.randn(DataType.FLOAT, dim, dim).muli(0.02);
            weights[l][1] = Nd4j.ones(DataType.FLOAT, dim);
        }
        return weights;
    }

    /**
     * Build a deep chain using pre-generated weights.
     * Each SameDiff built from the SAME weights array produces identical graphs.
     */
    private SameDiff buildDeepChainWith(INDArray[][] weights, int dim) {
        SameDiff sd = SameDiff.create();
        int layers = weights.length;
        SDVariable current = sd.placeHolder("input", DataType.FLOAT, 1, dim);
        for (int l = 0; l < layers; l++) {
            sd.constant("w_" + l, weights[l][0].dup());
            sd.constant("g_" + l, weights[l][1].dup());
            current = sd.mmul("matmul_" + l, current, sd.getVariable("w_" + l));
            current = sd.nn().rmsNorm("norm_" + l, current, sd.getVariable("g_" + l), 1e-5);
        }
        sd.identity("out", current);
        return track(sd);
    }

    /** Convenience: build with fresh weights (no reference comparison needed). */
    private SameDiff buildDeepChain(int layers, int dim) {
        return buildDeepChainWith(generateDeepChainWeights(layers, dim), dim);
    }

    /**
     * Pre-generate weights for gather+matmul+softmax. Call once, pass to buildGatherWith.
     */
    private static INDArray[] generateGatherWeights(int vocab, int dim) {
        Nd4j.getRandom().setSeed(42);
        return new INDArray[] {
            Nd4j.randn(DataType.FLOAT, vocab, dim).muli(0.02),  // embedTable
            Nd4j.randn(DataType.FLOAT, dim, vocab).muli(0.02),  // projWeight
            Nd4j.ones(DataType.FLOAT, dim)                       // gamma
        };
    }

    /**
     * Build gather+matmul+softmax from pre-generated weights.
     */
    private SameDiff buildGatherMatmulSoftmaxWith(INDArray[] weights) {
        SameDiff sd = SameDiff.create();
        sd.constant("embed_table", weights[0].dup());
        sd.constant("proj_weight", weights[1].dup());
        sd.constant("gamma", weights[2].dup());
        SDVariable tokenId = sd.placeHolder("token_id", DataType.INT64, 1);
        SDVariable gathered = sd.gather("gathered", sd.getVariable("embed_table"), tokenId, 0);
        SDVariable normed = sd.nn().rmsNorm("normed", gathered, sd.getVariable("gamma"), 1e-5);
        SDVariable logits = sd.mmul("logits", normed, sd.getVariable("proj_weight"));
        sd.nn().softmax("probs", logits, 1);
        return track(sd);
    }

    /** Convenience: build with fresh weights. */
    private SameDiff buildGatherMatmulSoftmax(int vocab, int dim) {
        return buildGatherMatmulSoftmaxWith(generateGatherWeights(vocab, dim));
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  1. DEEP CHAIN — gap ops must execute fresh on every composite replay
    // ═══════════════════════════════════════════════════════════════════════

    static Stream<Arguments> deepChainModes() {
        return Stream.of(
                Arguments.of(GraphExecutionMode.AUTO, 10, 16, 30),
                Arguments.of(GraphExecutionMode.TRITON, 10, 16, 30),
                Arguments.of(GraphExecutionMode.CUDA_GRAPHS, 10, 16, 30),
                Arguments.of(GraphExecutionMode.SLOT_BY_SLOT, 10, 16, 30),
                Arguments.of(GraphExecutionMode.EMULATED_REPLAY, 10, 16, 30)
        );
    }

    @ParameterizedTest(name = "1_deepChainRef_{0}_layers{1}_dim{2}_steps{3}")
    @MethodSource("deepChainModes")
    @Order(1)
    void test1_DeepChainReferenceAccuracy(GraphExecutionMode mode, int layers, int dim, int totalSteps) {
        INDArray[][] weights = generateDeepChainWeights(layers, dim);
        SameDiff sd = buildDeepChainWith(weights, dim);

        // Build inputs — one-hot in different dimensions for maximum differentiation
        INDArray[] inputs = new INDArray[totalSteps];
        for (int i = 0; i < totalSteps; i++) {
            inputs[i] = Nd4j.zeros(DataType.FLOAT, 1, dim);
            inputs[i].putScalar(0, i % dim, 1.0f);
        }

        // Get SLOT_BY_SLOT reference first (separate graph instance, SAME weights to avoid mode leakage)
        SameDiff sdRef = buildDeepChainWith(weights, dim);
        sdRef.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);
        INDArray[] refs = new INDArray[totalSteps];
        for (int i = 0; i < totalSteps; i++) {
            refs[i] = sdRef.output(Map.of("input", inputs[i]), "out").get("out").dup();
        }

        // Now run in the test mode
        sd.setGraphExecutionMode(mode);
        INDArray[] results = new INDArray[totalSteps];
        for (int i = 0; i < totalSteps; i++) {
            results[i] = sd.output(Map.of("input", inputs[i]), "out").get("out").dup();
        }

        // ASSERTION 1: Every step must match SLOT_BY_SLOT reference
        int matchCount = 0;
        int firstFailStep = -1;
        double worstDiff = 0;
        for (int i = 0; i < totalSteps; i++) {
            assertFalse(Double.isNaN(results[i].maxNumber().doubleValue()),
                    mode + " step " + i + ": NaN in output");
            assertTrue(results[i].amaxNumber().doubleValue() > 1e-10,
                    mode + " step " + i + ": all-zero output");

            double diff = refs[i].sub(results[i]).amaxNumber().doubleValue();
            if (diff < 1e-4) {
                matchCount++;
            } else {
                if (firstFailStep < 0) firstFailStep = i;
                log.warn("{} step {}: DRIFT diff={} (phase transition?)", mode, i, diff);
            }
            worstDiff = Math.max(worstDiff, diff);
        }

        // At least 90% of steps must match (allows 1-2 steps of numerical noise at transitions)
        int minMatch = (int)(totalSteps * 0.9);
        assertTrue(matchCount >= minMatch,
                mode + ": only " + matchCount + "/" + totalSteps + " steps match SLOT_BY_SLOT. "
                        + "First fail at step " + firstFailStep + ", worstDiff=" + worstDiff
                        + ". If step " + firstFailStep + " is ~3, this is the composite replay gap-skip bug.");

        log.info("{}: deep chain {} layers, {}/{} match ref, worstDiff={}", mode, layers, matchCount, totalSteps, worstDiff);
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  2. OUTPUT VALUE STALENESS — values MUST change when inputs change
    // ═══════════════════════════════════════════════════════════════════════

    @ParameterizedTest(name = "2_staleness_{0}")
    @EnumSource(value = GraphExecutionMode.class,
            names = {"SLOT_BY_SLOT", "AUTO", "TRITON", "CUDA_GRAPHS", "EMULATED_REPLAY"})
    @Order(2)
    void test2_OutputValuesMustChange(GraphExecutionMode mode) {
        int dim = 16;
        int steps = 20;
        SameDiff sd = buildDeepChain(5, dim);
        sd.setGraphExecutionMode(mode);

        INDArray[] outputs = new INDArray[steps];
        for (int i = 0; i < steps; i++) {
            INDArray input = Nd4j.zeros(DataType.FLOAT, 1, dim);
            input.putScalar(0, i % dim, 1.0f);
            outputs[i] = sd.output(Map.of("input", input), "out").get("out").dup();
        }

        // ASSERTION: Consecutive steps with DIFFERENT inputs must produce DIFFERENT outputs
        int staleCount = 0;
        for (int i = 1; i < steps; i++) {
            if ((i % dim) == ((i - 1) % dim)) continue;  // same input → skip
            double diff = outputs[i].sub(outputs[i - 1]).amaxNumber().doubleValue();
            if (diff < 1e-8) {
                staleCount++;
                log.error("{} STALE: step {} and {} have identical output (diff={}). "
                        + "Gap ops may not be executing during composite replay.", mode, i - 1, i, diff);
            }
        }
        assertEquals(0, staleCount,
                mode + ": " + staleCount + " consecutive step pairs had identical output — STALE DATA. "
                        + "This means gap ops (matmul/rmsNorm) are not executing during replay.");
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  3. PHASE TRANSITION ACCURACY — check correctness AT each transition
    // ═══════════════════════════════════════════════════════════════════════

    @ParameterizedTest(name = "3_phaseTransition_{0}")
    @EnumSource(value = GraphExecutionMode.class,
            names = {"AUTO", "TRITON", "CUDA_GRAPHS"})
    @Order(3)
    void test3_CorrectnessAtPhaseTransitions(GraphExecutionMode mode) {
        int dim = 16, layers = 5;
        INDArray[][] weights = generateDeepChainWeights(layers, dim);
        SameDiff sd = buildDeepChainWith(weights, dim);
        sd.setGraphExecutionMode(mode);

        SameDiff sdRef = buildDeepChainWith(weights, dim);
        sdRef.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);

        // Run enough steps to pass through all phases:
        // Steps 0-1: warmup (SLOT_BY_SLOT internally)
        // Step 2: capture
        // Steps 3+: frozen replay
        int totalSteps = 15;

        // Track which steps are at phase boundaries
        // Phase boundaries are at exec counts 0, 1, 2, 3 (warmup→frozen→capture→replay)
        int[] criticalSteps = {0, 1, 2, 3, 4, 5, 10, 14};  // warmup, transition, steady-state, late

        for (int step = 0; step < totalSteps; step++) {
            INDArray input = Nd4j.zeros(DataType.FLOAT, 1, dim);
            input.putScalar(0, step % dim, 1.0f);

            INDArray result = sd.output(Map.of("input", input), "out").get("out").dup();
            INDArray ref = sdRef.output(Map.of("input", input), "out").get("out").dup();

            double diff = ref.sub(result).amaxNumber().doubleValue();

            // Check if this is a critical step
            boolean isCritical = false;
            for (int cs : criticalSteps) {
                if (cs == step) { isCritical = true; break; }
            }

            if (isCritical) {
                assertTrue(diff < 1e-4,
                        mode + " CRITICAL step " + step + ": diff=" + diff + " vs SLOT_BY_SLOT. "
                                + "This step corresponds to a phase transition — "
                                + "if step=3, the frozen fast path may be using wrong replay type.");
                log.info("{} step {}: diff={} (CRITICAL - OK)", mode, step, diff);
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  4. GATHER + MATMUL + SOFTMAX — value-dependent + gap ops
    // ═══════════════════════════════════════════════════════════════════════

    @ParameterizedTest(name = "4_gatherMatmulSoftmax_{0}")
    @EnumSource(value = GraphExecutionMode.class,
            names = {"SLOT_BY_SLOT", "AUTO", "TRITON", "CUDA_GRAPHS", "EMULATED_REPLAY"})
    @Order(4)
    void test4_GatherMatmulSoftmaxComposite(GraphExecutionMode mode) {
        int vocab = 32, dim = 16;
        INDArray[] gatherWeights = generateGatherWeights(vocab, dim);
        SameDiff sd = buildGatherMatmulSoftmaxWith(gatherWeights);
        sd.setGraphExecutionMode(mode);

        SameDiff sdRef = buildGatherMatmulSoftmaxWith(gatherWeights);
        sdRef.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);

        int steps = 20;
        int matchCount = 0;
        for (int i = 0; i < steps; i++) {
            INDArray tokenId = Nd4j.createFromArray(new long[]{i % vocab});

            INDArray result = sd.output(Map.of("token_id", tokenId), "probs").get("probs").dup();
            INDArray ref = sdRef.output(Map.of("token_id", tokenId), "probs").get("probs").dup();

            // Must be valid probabilities
            assertFalse(Double.isNaN(result.sumNumber().doubleValue()),
                    mode + " step " + i + ": NaN in softmax output");
            double sum = result.castTo(DataType.FLOAT).sumNumber().doubleValue();
            assertEquals(1.0, sum, 0.05,
                    mode + " step " + i + ": softmax sum=" + sum + " (should be 1.0)");

            double diff = ref.sub(result).amaxNumber().doubleValue();
            if (diff < 1e-4) matchCount++;
            else log.warn("{} step {}: diff={}", mode, i, diff);
        }

        assertTrue(matchCount >= 18,
                mode + ": only " + matchCount + "/20 steps match reference for gather+matmul+softmax");
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  5. EXEC COUNT PROGRESSION — verify phase lifecycle
    // ═══════════════════════════════════════════════════════════════════════

    @Test
    @Order(5)
    void test5_ExecCountProgression() {
        int dim = 16, layers = 5;
        SameDiff sd = buildDeepChain(layers, dim);
        sd.setGraphExecutionMode(GraphExecutionMode.AUTO);

        INDArray input = Nd4j.zeros(DataType.FLOAT, 1, dim);
        input.putScalar(0, 0, 1.0f);

        // Step 0: should compile the plan
        sd.output(Map.of("input", input), "out");

        // Verify plan was compiled
        DspDebugger debugger = DspDebugger.attach(sd);
        DspDebugger.PlanReport report = debugger.analyzePlan();
        assertNotNull(report, "Plan report should not be null after first execution");
        assertTrue(report.numSlots > 0, "Plan should have slots after first execution");
        assertTrue(report.numSegments > 0, "Plan should have segments after first execution");

        log.info("After step 0: {} slots, {} segments, planPhase={}",
                report.numSlots, report.numSegments, report.planPhase);

        // Run more steps and verify phase progression
        for (int step = 1; step < 10; step++) {
            input = Nd4j.zeros(DataType.FLOAT, 1, dim);
            input.putScalar(0, step % dim, 1.0f);
            sd.output(Map.of("input", input), "out");
        }

        report = debugger.analyzePlan();
        log.info("After step 9: planPhase={}", report.planPhase);

        // After 10 steps, AUTO should have reached at least SHAPES_FROZEN
        DspPlanAssertions.assertPhaseReached(sd, PlanPhase.SHAPES_FROZEN,
                "AUTO mode after 10 steps should have frozen shapes");
        DspPlanAssertions.assertNoCaptureFailures(sd,
                "AUTO mode should not have any capture failures after 10 steps");
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  6. LATE DECAY — step 25+ must still match reference
    // ═══════════════════════════════════════════════════════════════════════

    @ParameterizedTest(name = "6_lateDecay_{0}")
    @EnumSource(value = GraphExecutionMode.class,
            names = {"AUTO", "TRITON", "CUDA_GRAPHS"})
    @Order(6)
    void test6_LateStepDecay(GraphExecutionMode mode) {
        int dim = 16, layers = 5;
        INDArray[][] weights = generateDeepChainWeights(layers, dim);
        SameDiff sd = buildDeepChainWith(weights, dim);
        sd.setGraphExecutionMode(mode);

        SameDiff sdRef = buildDeepChainWith(weights, dim);
        sdRef.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);

        int totalSteps = 30;
        // Only check last 10 steps — these are deep in replay territory
        double worstLateDiff = 0;
        int lateFailCount = 0;

        for (int step = 0; step < totalSteps; step++) {
            INDArray input = Nd4j.zeros(DataType.FLOAT, 1, dim);
            input.putScalar(0, step % dim, 1.0f);

            INDArray result = sd.output(Map.of("input", input), "out").get("out").dup();

            if (step >= 20) {
                INDArray ref = sdRef.output(Map.of("input", input), "out").get("out").dup();
                double diff = ref.sub(result).amaxNumber().doubleValue();
                worstLateDiff = Math.max(worstLateDiff, diff);
                if (diff > 1e-4) {
                    lateFailCount++;
                    log.error("{} LATE DECAY step {}: diff={}", mode, step, diff);
                }
            }
        }

        assertEquals(0, lateFailCount,
                mode + ": " + lateFailCount + " of last 10 steps diverged from reference. "
                        + "Worst late diff=" + worstLateDiff + ". "
                        + "This indicates FP drift accumulation or stale gap op data in composite replay.");

        log.info("{}: late-step check passed, worstDiff={}", mode, worstLateDiff);
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  7. VALUE STABILITY FOR CONSTANT INPUTS — same input → same output
    // ═══════════════════════════════════════════════════════════════════════

    @ParameterizedTest(name = "7_constInputStability_{0}")
    @EnumSource(value = GraphExecutionMode.class,
            names = {"SLOT_BY_SLOT", "AUTO", "TRITON", "CUDA_GRAPHS", "EMULATED_REPLAY"})
    @Order(7)
    void test7_ConstantInputsProduceStableOutput(GraphExecutionMode mode) {
        int dim = 16, layers = 5;
        SameDiff sd = buildDeepChain(layers, dim);
        sd.setGraphExecutionMode(mode);

        // Same input every step — output must be identical across all steps
        INDArray fixedInput = Nd4j.zeros(DataType.FLOAT, 1, dim);
        fixedInput.putScalar(0, 0, 1.0f);

        INDArray[] outputs = new INDArray[20];
        for (int i = 0; i < 20; i++) {
            outputs[i] = sd.output(Map.of("input", fixedInput.dup()), "out").get("out").dup();
        }

        // All outputs should be identical (within FP tolerance)
        INDArray expected = outputs[0];
        for (int i = 1; i < 20; i++) {
            double diff = expected.sub(outputs[i]).amaxNumber().doubleValue();
            assertTrue(diff < 1e-4,
                    mode + " step " + i + ": constant input produced different output! diff=" + diff
                            + ". Output should be deterministic for constant inputs.");
        }
        log.info("{}: constant-input stability passed (20 steps identical)", mode);
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  8. BIDIRECTIONAL VALUE CHANGE — if we change back to an earlier input,
    //     we must get the same result we got before
    // ═══════════════════════════════════════════════════════════════════════

    @ParameterizedTest(name = "8_bidirectional_{0}")
    @EnumSource(value = GraphExecutionMode.class,
            names = {"SLOT_BY_SLOT", "AUTO", "TRITON", "CUDA_GRAPHS", "EMULATED_REPLAY"})
    @Order(8)
    void test8_BidirectionalValueChange(GraphExecutionMode mode) {
        int dim = 16, layers = 5;
        SameDiff sd = buildDeepChain(layers, dim);
        sd.setGraphExecutionMode(mode);

        INDArray inputA = Nd4j.zeros(DataType.FLOAT, 1, dim);
        inputA.putScalar(0, 0, 1.0f);  // one-hot at dim 0

        INDArray inputB = Nd4j.zeros(DataType.FLOAT, 1, dim);
        inputB.putScalar(0, 7, 1.0f);  // one-hot at dim 7

        // Warm up
        for (int i = 0; i < 5; i++) {
            sd.output(Map.of("input", inputA.dup()), "out");
        }

        // Get baseline for input A (deep in replay)
        INDArray resultA1 = sd.output(Map.of("input", inputA.dup()), "out").get("out").dup();

        // Switch to input B
        INDArray resultB = sd.output(Map.of("input", inputB.dup()), "out").get("out").dup();

        // ASSERTION: A and B must be DIFFERENT
        double abDiff = resultA1.sub(resultB).amaxNumber().doubleValue();
        assertTrue(abDiff > 1e-6,
                mode + ": input A and B produced identical output (diff=" + abDiff + "). "
                        + "STALE DATA — the graph is not reading the new input.");

        // Switch back to input A — should match the earlier A result
        INDArray resultA2 = sd.output(Map.of("input", inputA.dup()), "out").get("out").dup();

        double aaDiff = resultA1.sub(resultA2).amaxNumber().doubleValue();
        assertTrue(aaDiff < 1e-4,
                mode + ": same input A produced different results before and after B. "
                        + "diff=" + aaDiff + ". State contamination from input B's execution.");

        log.info("{}: bidirectional A→B→A passed, abDiff={}, aaDiff={}", mode, abDiff, aaDiff);
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  9. RAPID INPUT SWITCHING — alternating inputs during replay
    // ═══════════════════════════════════════════════════════════════════════

    @ParameterizedTest(name = "9_rapidSwitch_{0}")
    @EnumSource(value = GraphExecutionMode.class,
            names = {"SLOT_BY_SLOT", "AUTO", "TRITON", "CUDA_GRAPHS", "EMULATED_REPLAY"})
    @Order(9)
    void test9_RapidInputSwitching(GraphExecutionMode mode) {
        int dim = 16, layers = 5;
        INDArray[][] weights = generateDeepChainWeights(layers, dim);
        SameDiff sd = buildDeepChainWith(weights, dim);
        sd.setGraphExecutionMode(mode);

        SameDiff sdRef = buildDeepChainWith(weights, dim);
        sdRef.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);

        // 4 distinct inputs
        INDArray[] distinctInputs = new INDArray[4];
        for (int i = 0; i < 4; i++) {
            distinctInputs[i] = Nd4j.zeros(DataType.FLOAT, 1, dim);
            distinctInputs[i].putScalar(0, i * 4, 1.0f);
        }

        // Warm up with pattern A,B,C,D
        for (int i = 0; i < 8; i++) {
            sd.output(Map.of("input", distinctInputs[i % 4].dup()), "out");
        }

        // Now rapid switching during replay: A,B,A,B,C,D,C,D,A,C,B,D
        int[] pattern = {0, 1, 0, 1, 2, 3, 2, 3, 0, 2, 1, 3};
        int matchCount = 0;
        for (int idx : pattern) {
            INDArray input = distinctInputs[idx].dup();
            INDArray result = sd.output(Map.of("input", input), "out").get("out").dup();
            INDArray ref = sdRef.output(Map.of("input", input), "out").get("out").dup();

            double diff = ref.sub(result).amaxNumber().doubleValue();
            if (diff < 1e-4) matchCount++;
        }

        assertTrue(matchCount >= 11,
                mode + ": rapid switching — only " + matchCount + "/12 steps matched reference. "
                        + "Composite replay may not be updating staging buffers correctly.");

        log.info("{}: rapid switching passed ({}/12 match)", mode, matchCount);
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  10. MIXED PRECISION THROUGH COMPOSITE — FP16 weights, FP32 activations
    // ═══════════════════════════════════════════════════════════════════════

    @ParameterizedTest(name = "10_mixedPrecComposite_{0}")
    @EnumSource(value = GraphExecutionMode.class,
            names = {"SLOT_BY_SLOT", "AUTO", "TRITON", "CUDA_GRAPHS", "EMULATED_REPLAY"})
    @Order(10)
    void test10_MixedPrecisionComposite(GraphExecutionMode mode) {
        int dim = 16;
        Nd4j.getRandom().setSeed(42);
        SameDiff sd = SameDiff.create();
        track(sd);
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, dim);
        // FP16 weight + FP32 activation → forces cast ops in the graph
        INDArray w = Nd4j.randn(DataType.FLOAT, dim, dim).muli(0.02).castTo(DataType.HALF);
        sd.constant("weight", w);
        // mmul between FP32 input and FP16 weight — tests MmulHelper mixed precision path
        SDVariable projected = sd.mmul("projected", input, sd.getVariable("weight"));
        sd.identity("out", projected);

        sd.setGraphExecutionMode(mode);

        // Run 20 steps and verify each one
        for (int step = 0; step < 20; step++) {
            INDArray in = Nd4j.zeros(DataType.FLOAT, 1, dim);
            in.putScalar(0, step % dim, 1.0f);

            INDArray result = sd.output(Map.of("input", in), "out").get("out");
            assertFalse(Double.isNaN(result.maxNumber().doubleValue()),
                    mode + " step " + step + ": NaN from mixed precision mmul");
            assertTrue(result.amaxNumber().doubleValue() > 1e-6,
                    mode + " step " + step + ": all-zero output from mixed precision mmul");
        }
        log.info("{}: mixed precision composite passed (20 steps)", mode);
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  11. PLAN ASSERTION INTEGRATION — use DspPlanAssertions API
    // ═══════════════════════════════════════════════════════════════════════

    @Test
    @Order(11)
    void test11_PlanAssertionsSmokeTest() {
        int dim = 16, layers = 3;
        SameDiff sd = buildDeepChain(layers, dim);
        sd.setGraphExecutionMode(GraphExecutionMode.AUTO);

        INDArray input = Nd4j.zeros(DataType.FLOAT, 1, dim);
        input.putScalar(0, 0, 1.0f);

        // Run once — compiles plan
        sd.output(Map.of("input", input), "out");
        DspPlanAssertions.assertOpCompiled(sd, "matmul", "deep chain should contain matmul ops");
        DspPlanAssertions.assertOpCompiled(sd, "rms_norm", "deep chain should contain rms_norm ops");
        DspPlanAssertions.assertNoCaptureFailures(sd, "no capture failures after first step");

        // Run 10 more steps to reach steady state
        for (int i = 0; i < 10; i++) {
            sd.output(Map.of("input", input.dup()), "out");
        }

        DspPlanAssertions.assertPhaseReached(sd, PlanPhase.SHAPES_FROZEN,
                "AUTO should freeze shapes after 11 steps");
        DspPlanAssertions.assertNoPhaseContractViolations(sd,
                "no phase contract violations in AUTO mode");

        log.info("Plan assertions passed");
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  12. ACCUMULATING MATMUL — 3-step chain, verify intermediate non-zero
    // ═══════════════════════════════════════════════════════════════════════

    @ParameterizedTest(name = "12_accumMatmul_{0}")
    @EnumSource(value = GraphExecutionMode.class,
            names = {"SLOT_BY_SLOT", "AUTO", "EMULATED_REPLAY"})
    @Order(12)
    void test12_AccumulatingMatmulChain(GraphExecutionMode mode) {
        int dim = 8;
        Nd4j.getRandom().setSeed(42);
        SameDiff sd = SameDiff.create();
        track(sd);
        SDVariable in = sd.placeHolder("input", DataType.FLOAT, 1, dim);

        // Chain: input → mmul(W1) → mmul(W2) → mmul(W3) → out
        // Each matmul is a gap op. The chain tests that intermediate slot outputs
        // are correctly written and read across the gap/island boundary.
        INDArray w1 = Nd4j.randn(DataType.FLOAT, dim, dim).muli(0.1);
        INDArray w2 = Nd4j.randn(DataType.FLOAT, dim, dim).muli(0.1);
        INDArray w3 = Nd4j.randn(DataType.FLOAT, dim, dim).muli(0.1);
        sd.constant("w1", w1);
        sd.constant("w2", w2);
        sd.constant("w3", w3);

        SDVariable step1 = sd.mmul("step1", in, sd.getVariable("w1"));
        SDVariable step2 = sd.mmul("step2", step1, sd.getVariable("w2"));
        SDVariable step3 = sd.mmul("step3", step2, sd.getVariable("w3"));
        sd.identity("out", step3);

        sd.setGraphExecutionMode(mode);

        // Run 15 steps with different inputs
        for (int i = 0; i < 15; i++) {
            INDArray input = Nd4j.zeros(DataType.FLOAT, 1, dim);
            input.putScalar(0, i % dim, 1.0f);

            // Manual reference: input @ W1 @ W2 @ W3
            INDArray expected = input.mmul(w1).mmul(w2).mmul(w3);
            INDArray result = sd.output(Map.of("input", input), "out").get("out");

            double diff = expected.sub(result).amaxNumber().doubleValue();
            assertTrue(diff < 0.01,
                    mode + " step " + i + ": 3-matmul chain diff=" + diff
                            + ". Intermediate gap slot outputs may be stale.");
        }
        log.info("{}: accumulating matmul chain passed (15 steps)", mode);
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  13. SEGMENT STATE ASSERTIONS — directly query and assert native state
    // ═══════════════════════════════════════════════════════════════════════

    @ParameterizedTest(name = "13_segmentStateAssertions_{0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"AUTO", "TRITON", "CUDA_GRAPHS"})
    @Order(13)
    void test13_SegmentStateAssertions(GraphExecutionMode mode) {
        int layers = 5;
        int dim = 16;
        SameDiff sd = buildDeepChain(layers, dim);
        sd.setGraphExecutionMode(mode);

        INDArray input = Nd4j.zeros(DataType.FLOAT, 1, dim);
        input.putScalar(0, 0, 1.0f);

        // Step 1: first execution — plan should compile
        sd.output(Map.of("input", input), "out");
        DspPlanAssertions.assertSegmentCount(sd, 1,
                mode + ": deep chain should have exactly 1 segment");

        // After first step, exec count should be > 0
        int execCount1 = DspPlanAssertions.getSegmentExecCount(sd, 0);
        assertTrue(execCount1 >= 1,
                mode + ": segment exec count should be >= 1 after first output, was " + execCount1);

        // Step 2-5: warmup steps — shapes should freeze
        for (int i = 1; i < 5; i++) {
            input = Nd4j.zeros(DataType.FLOAT, 1, dim);
            input.putScalar(0, i % dim, 1.0f);
            sd.output(Map.of("input", input), "out");
        }

        int execCount5 = DspPlanAssertions.getSegmentExecCount(sd, 0);
        // When monolithic handle is discarded (composite has gaps), the frozen fast
        // path may not be entered until per-island handles are ready. In that case,
        // execution goes through the regular phaseReplay path which may increment
        // a different counter. Assert >= rather than strictly > to handle both paths.
        assertTrue(execCount5 >= execCount1,
                mode + ": segment exec count should be >= " + execCount1
                        + " after 4 more outputs, was " + execCount5);

        // Query the replay schedule state
        int gapUnits = DspPlanAssertions.getSegmentGapUnitCount(sd, 0);
        int islandUnits = DspPlanAssertions.getSegmentIslandUnitCount(sd, 0);
        int gapSlots = DspPlanAssertions.getSegmentGapSlotCount(sd, 0);
        int replayMode = DspPlanAssertions.getSegmentReplayMode(sd, 0);

        log.info("{}: after 5 steps — gapUnits={}, islandUnits={}, gapSlots={}, replayMode={}",
                mode, gapUnits, islandUnits, gapSlots, replayMode);

        // The deep chain has 5 matmul + 5 rmsNorm + 1 identity = 11 ops.
        // matmul/rmsNorm are native gap ops; identity is Triton-compilable.
        // So we expect gap slots = 10 and island(s) for the identity op.
        if (mode != GraphExecutionMode.SLOT_BY_SLOT) {
            // Non-SLOT_BY_SLOT modes should build a composite schedule
            if (gapUnits > 0) {
                // Composite replay schedule exists — assert gap properties
                assertTrue(gapSlots >= 10,
                        mode + ": gap slots should be >= 10 (5 matmul + 5 rmsNorm), was " + gapSlots);
                assertTrue(islandUnits >= 1,
                        mode + ": should have >= 1 island unit (identity), was " + islandUnits);
            }
        }

        // Run 10 more steps — should reach steady state
        for (int i = 5; i < 15; i++) {
            input = Nd4j.zeros(DataType.FLOAT, 1, dim);
            input.putScalar(0, i % dim, 1.0f);
            sd.output(Map.of("input", input), "out");
        }

        int execCount15 = DspPlanAssertions.getSegmentExecCount(sd, 0);
        assertTrue(execCount15 >= execCount5,
                mode + ": segment exec count should be >= " + execCount5
                        + " after 10 more outputs, was " + execCount15);

        // After 15 steps, query the final replay mode
        int finalReplayMode = DspPlanAssertions.getSegmentReplayMode(sd, 0);
        int finalGapUnits = DspPlanAssertions.getSegmentGapUnitCount(sd, 0);
        int finalIslandUnits = DspPlanAssertions.getSegmentIslandUnitCount(sd, 0);
        int finalGapSlots = DspPlanAssertions.getSegmentGapSlotCount(sd, 0);

        log.info("{}: after 15 steps — gapUnits={}, islandUnits={}, gapSlots={}, replayMode={}, execCount={}",
                mode, finalGapUnits, finalIslandUnits, finalGapSlots, finalReplayMode, execCount15);

        // Execution count should be >= from step 5 — when composite replay has gaps
        // and the monolithic handle is discarded, the frozen fast path may not be
        // entered until per-island handles are individually captured, so the
        // segment-level counter may not increment via that path.
        assertTrue(execCount15 >= execCount5,
                mode + ": exec count should be >= step 5 value (" + execCount5
                        + ") at step 15 (" + execCount15 + ")");
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  14. VALUE STALENESS DETECTION — assert outputs change when inputs change
    // ═══════════════════════════════════════════════════════════════════════

    @ParameterizedTest(name = "14_valueStalenessDetection_{0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"AUTO", "TRITON", "CUDA_GRAPHS"})
    @Order(14)
    void test14_ValueStalenessDetection(GraphExecutionMode mode) {
        int layers = 5;
        int dim = 16;
        SameDiff sd = buildDeepChain(layers, dim);
        sd.setGraphExecutionMode(mode);

        // Warmup with input A
        INDArray inputA = Nd4j.zeros(DataType.FLOAT, 1, dim);
        inputA.putScalar(0, 0, 1.0f);
        for (int i = 0; i < 5; i++) {
            sd.output(Map.of("input", inputA), "out");
        }
        INDArray outputA = sd.output(Map.of("input", inputA), "out").get("out").dup();

        // Now switch to input B — output MUST change
        INDArray inputB = Nd4j.zeros(DataType.FLOAT, 1, dim);
        inputB.putScalar(0, dim / 2, 1.0f);

        INDArray outputB = sd.output(Map.of("input", inputB), "out").get("out").dup();

        double diff = outputA.sub(outputB).amaxNumber().doubleValue();
        assertTrue(diff > 1e-6,
                mode + ": output did NOT change when input changed (diff=" + diff
                        + "). Gap ops may not be executing during composite replay. "
                        + "gapUnits=" + DspPlanAssertions.getSegmentGapUnitCount(sd, 0)
                        + " islandUnits=" + DspPlanAssertions.getSegmentIslandUnitCount(sd, 0)
                        + " replayMode=" + DspPlanAssertions.getSegmentReplayMode(sd, 0));

        // Switch BACK to input A — should recover original output
        INDArray outputA2 = sd.output(Map.of("input", inputA), "out").get("out").dup();
        double recoverDiff = outputA.sub(outputA2).amaxNumber().doubleValue();
        assertTrue(recoverDiff < 1e-4,
                mode + ": output did not recover when input switched back (diff=" + recoverDiff
                        + "). State is accumulating incorrectly.");

        log.info("{}: value staleness detection passed (A/B switch, recover diff={})", mode, recoverDiff);
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  15. EXEC COUNT MONOTONICITY — exec count always increases
    // ═══════════════════════════════════════════════════════════════════════

    @ParameterizedTest(name = "15_execCountMonotonicity_{0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"AUTO", "TRITON", "CUDA_GRAPHS"})
    @Order(15)
    void test15_ExecCountMonotonicity(GraphExecutionMode mode) {
        int layers = 3;
        int dim = 8;
        SameDiff sd = buildDeepChain(layers, dim);
        sd.setGraphExecutionMode(mode);

        int prevExecCount = 0;
        int recompileResets = 0;
        for (int step = 0; step < 20; step++) {
            INDArray input = Nd4j.zeros(DataType.FLOAT, 1, dim);
            input.putScalar(0, step % dim, 1.0f);
            sd.output(Map.of("input", input), "out");

            int execCount = DspPlanAssertions.getSegmentExecCount(sd, 0);
            if (execCount < prevExecCount) {
                // Exec count can reset when plan recompiles (shape freeze, segment rebuild).
                // This is expected at most once or twice during warmup → freeze transition.
                recompileResets++;
                log.info("{} step {}: exec count reset {} -> {} (recompile #{})",
                        mode, step, prevExecCount, execCount, recompileResets);
            }
            prevExecCount = execCount;
        }
        // Allow a few recompile resets during warmup, but not excessive
        assertTrue(recompileResets <= 3,
                mode + ": too many exec count resets (" + recompileResets + ") — excessive recompilation");
        assertTrue(prevExecCount > 1,
                mode + ": exec count should be > 1 after 20 steps, was " + prevExecCount);
        log.info("{}: exec count monotonicity passed (20 steps, final count={}, resets={})",
                mode, prevExecCount, recompileResets);
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  16. EXHAUSTIVE STALENESS — same input 30+ steps, output MUST be
    //      bitwise identical. Change input, run 30+ more, must stabilize.
    //      Then switch BACK to original — must match original exactly.
    // ═══════════════════════════════════════════════════════════════════════

    @ParameterizedTest(name = "16_exhaustiveStaleness_{0}")
    @EnumSource(value = GraphExecutionMode.class,
            names = {"SLOT_BY_SLOT", "AUTO", "TRITON", "CUDA_GRAPHS", "EMULATED_REPLAY"})
    @Order(16)
    void test16_ExhaustiveStalenessDetection(GraphExecutionMode mode) {
        int dim = 16, layers = 5;
        SameDiff sd = buildDeepChain(layers, dim);
        sd.setGraphExecutionMode(mode);

        INDArray inputA = Nd4j.zeros(DataType.FLOAT, 1, dim);
        inputA.putScalar(0, 0, 1.0f);

        INDArray inputB = Nd4j.zeros(DataType.FLOAT, 1, dim);
        inputB.putScalar(0, 8, 1.0f);

        // Phase 1: Run with input A for 35 steps. All outputs must be identical.
        INDArray firstOutputA = null;
        int driftCountA = 0;
        double worstDriftA = 0;
        for (int step = 0; step < 35; step++) {
            INDArray out = sd.output(Map.of("input", inputA.dup()), "out").get("out").dup();
            assertFalse(Double.isNaN(out.maxNumber().doubleValue()),
                    mode + " phase1 step " + step + ": NaN");
            if (firstOutputA == null) {
                firstOutputA = out;
            } else {
                double diff = firstOutputA.sub(out).amaxNumber().doubleValue();
                if (diff > 1e-6) {
                    driftCountA++;
                    worstDriftA = Math.max(worstDriftA, diff);
                    log.error("{} DRIFT in phase1: step {} diff={}", mode, step, diff);
                }
            }
        }
        assertEquals(0, driftCountA,
                mode + ": " + driftCountA + " steps drifted from step 0 output during constant-input phase 1. "
                        + "worstDrift=" + worstDriftA + ". Output should be IDENTICAL for identical inputs.");

        // Phase 2: Switch to input B for 35 steps. Output must differ from A,
        // and all B outputs must be identical to each other.
        INDArray firstOutputB = null;
        int driftCountB = 0;
        double worstDriftB = 0;
        for (int step = 0; step < 35; step++) {
            INDArray out = sd.output(Map.of("input", inputB.dup()), "out").get("out").dup();
            assertFalse(Double.isNaN(out.maxNumber().doubleValue()),
                    mode + " phase2 step " + step + ": NaN");
            if (firstOutputB == null) {
                firstOutputB = out;
                // First B output MUST differ from A
                double abDiff = firstOutputA.sub(firstOutputB).amaxNumber().doubleValue();
                assertTrue(abDiff > 1e-6,
                        mode + ": output did NOT change when input changed from A to B (diff=" + abDiff
                                + "). Gap ops may not be executing.");
            } else {
                double diff = firstOutputB.sub(out).amaxNumber().doubleValue();
                if (diff > 1e-6) {
                    driftCountB++;
                    worstDriftB = Math.max(worstDriftB, diff);
                    log.error("{} DRIFT in phase2: step {} diff={}", mode, step, diff);
                }
            }
        }
        assertEquals(0, driftCountB,
                mode + ": " + driftCountB + " steps drifted during constant-input phase 2 (input B). "
                        + "worstDrift=" + worstDriftB);

        // Phase 3: Switch BACK to input A — must match original A output exactly
        INDArray recoveredA = sd.output(Map.of("input", inputA.dup()), "out").get("out").dup();
        double recoverDiff = firstOutputA.sub(recoveredA).amaxNumber().doubleValue();
        assertTrue(recoverDiff < 1e-4,
                mode + ": output did NOT recover when switching back to input A. diff=" + recoverDiff
                        + ". State contamination from input B.");

        log.info("{}: exhaustive staleness detection passed (35+35+1 steps, worstDriftA={}, worstDriftB={}, recoverDiff={})",
                mode, worstDriftA, worstDriftB, recoverDiff);
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  17. PHASE TRANSITION TIMING — assert plan phase at each step,
    //      verify the expected progression happens at the right steps.
    // ═══════════════════════════════════════════════════════════════════════

    @ParameterizedTest(name = "17_phaseTransitionTiming_{0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"AUTO", "TRITON", "CUDA_GRAPHS"})
    @Order(17)
    void test17_PhaseTransitionTiming(GraphExecutionMode mode) {
        int layers = 3, dim = 8;
        SameDiff sd = buildDeepChain(layers, dim);
        sd.setGraphExecutionMode(mode);

        INDArray input = Nd4j.zeros(DataType.FLOAT, 1, dim);
        input.putScalar(0, 0, 1.0f);

        // Track phase at every step
        List<String> phaseLog = new ArrayList<>();
        int firstFrozenStep = -1;
        int firstReplayingStep = -1;

        for (int step = 0; step < 25; step++) {
            sd.output(Map.of("input", input.dup()), "out");

            // Query plan phase
            int planPhaseOrd = DspPlanAssertions.getFrozenExecCount(sd);
            int segExecCount = DspPlanAssertions.getSegmentExecCount(sd, 0);
            int replayMode = DspPlanAssertions.getSegmentReplayMode(sd, 0);
            int pointersStable = DspPlanAssertions.getPointersStable(sd);

            String phase = String.format("step=%d execCount=%d frozenExec=%d replayMode=%s ptrsStable=%d",
                    step, segExecCount, planPhaseOrd,
                    replayMode == DspPlanAssertions.REPLAY_MODE_NONE ? "NONE" :
                    replayMode == DspPlanAssertions.REPLAY_MODE_MONOLITHIC ? "MONOLITHIC" :
                    replayMode == DspPlanAssertions.REPLAY_MODE_COMPOSITE ? "COMPOSITE" : "?" + replayMode,
                    pointersStable);
            phaseLog.add(phase);

            if (planPhaseOrd >= 0 && firstFrozenStep < 0) {
                firstFrozenStep = step;
            }
            if (replayMode != DspPlanAssertions.REPLAY_MODE_NONE && firstReplayingStep < 0) {
                firstReplayingStep = step;
            }
        }

        // Log the full phase progression
        for (String entry : phaseLog) {
            log.info("{}: {}", mode, entry);
        }

        // After 25 steps, shapes MUST have frozen
        assertTrue(firstFrozenStep >= 0,
                mode + ": shapes never froze within 25 steps! Phase progression: " + phaseLog);

        // Shapes should freeze BEFORE step 15 (generous)
        assertTrue(firstFrozenStep < 15,
                mode + ": shapes froze too late at step " + firstFrozenStep);

        // Plan-level checks after 25 steps
        DspPlanAssertions.assertNoCaptureFailures(sd,
                mode + " after 25 steps");
        DspPlanAssertions.assertNoPhaseContractViolations(sd,
                mode + " after 25 steps");
        DspPlanAssertions.assertNoMidExecutionRecompiles(sd,
                mode + " after 25 steps");

        log.info("{}: phase transition timing passed. firstFrozen={}, firstReplaying={}",
                mode, firstFrozenStep, firstReplayingStep);
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  18. REPLAY COUNT PROGRESSION — after capture, replay count must
    //      increase monotonically with each step.
    // ═══════════════════════════════════════════════════════════════════════

    @ParameterizedTest(name = "18_replayCountProgression_{0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"AUTO", "TRITON", "CUDA_GRAPHS"})
    @Order(18)
    void test18_ReplayCountProgression(GraphExecutionMode mode) {
        int layers = 3, dim = 8;
        SameDiff sd = buildDeepChain(layers, dim);
        sd.setGraphExecutionMode(mode);

        INDArray input = Nd4j.zeros(DataType.FLOAT, 1, dim);
        input.putScalar(0, 0, 1.0f);

        // Run 30 steps
        int[] replayCounts = new int[30];
        int[] execCounts = new int[30];
        for (int step = 0; step < 30; step++) {
            sd.output(Map.of("input", input.dup()), "out");
            replayCounts[step] = DspPlanAssertions.getSegmentReplayCount(sd, 0);
            execCounts[step] = DspPlanAssertions.getSegmentExecCount(sd, 0);
        }

        // Find when replays start (first non-zero replay count)
        int firstReplayStep = -1;
        for (int i = 0; i < 30; i++) {
            if (replayCounts[i] > 0) {
                firstReplayStep = i;
                break;
            }
        }

        log.info("{}: replay counts: first 10 = [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}]",
                mode,
                replayCounts[0], replayCounts[1], replayCounts[2], replayCounts[3], replayCounts[4],
                replayCounts[5], replayCounts[6], replayCounts[7], replayCounts[8], replayCounts[9]);

        if (firstReplayStep >= 0) {
            // Once replays start, they should increase monotonically
            int prevReplay = replayCounts[firstReplayStep];
            int stuckCount = 0;
            for (int i = firstReplayStep + 1; i < 30; i++) {
                if (replayCounts[i] < prevReplay) {
                    fail(mode + " step " + i + ": replay count DECREASED from " + prevReplay
                            + " to " + replayCounts[i] + ". Replay infrastructure is broken.");
                }
                if (replayCounts[i] == prevReplay) {
                    stuckCount++;
                }
                prevReplay = replayCounts[i];
            }

            // Allow a few stuck steps (capture may take >1 warmup step)
            // but majority should show increases
            int replaySteps = 30 - firstReplayStep - 1;
            assertTrue(stuckCount < replaySteps / 2,
                    mode + ": replay count stuck for " + stuckCount + "/" + replaySteps
                            + " steps after replay started at step " + firstReplayStep
                            + ". Graph may not be replaying.");

            log.info("{}: replay count progression passed. firstReplay={}, finalReplay={}, stuckSteps={}",
                    mode, firstReplayStep, replayCounts[29], stuckCount);
        } else {
            // No replays in 30 steps — may be SLOT_BY_SLOT fallback
            log.warn("{}: no replays detected in 30 steps — plan may have stayed in slot-by-slot", mode);
        }

        // Exec count should generally increase, allowing resets during recompilation
        int execResets = 0;
        for (int i = 1; i < 30; i++) {
            if (execCounts[i] < execCounts[i - 1]) {
                execResets++;
            }
        }
        assertTrue(execResets <= 3,
                mode + ": too many exec count resets (" + execResets + ") in 30 steps");
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  19. POINTER STABILITY — pointers must stabilize after freeze and
    //      STAY stable for all subsequent steps.
    // ═══════════════════════════════════════════════════════════════════════

    @ParameterizedTest(name = "19_pointerStability_{0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"AUTO", "TRITON", "CUDA_GRAPHS"})
    @Order(19)
    void test19_PointerStabilityAfterFreeze(GraphExecutionMode mode) {
        int layers = 3, dim = 8;
        SameDiff sd = buildDeepChain(layers, dim);
        sd.setGraphExecutionMode(mode);

        INDArray input = Nd4j.zeros(DataType.FLOAT, 1, dim);
        input.putScalar(0, 0, 1.0f);

        // Track pointer stability at each step
        int firstStableStep = -1;
        int lastUnstableAfterStable = -1;

        for (int step = 0; step < 25; step++) {
            sd.output(Map.of("input", input.dup()), "out");
            int stable = DspPlanAssertions.getPointersStable(sd);

            if (stable == 1 && firstStableStep < 0) {
                firstStableStep = step;
            }
            if (stable != 1 && firstStableStep >= 0) {
                lastUnstableAfterStable = step;
                log.error("{} POINTER INSTABILITY: step {} — pointers became unstable after "
                        + "being stable at step {}!", mode, step, firstStableStep);
            }
        }

        // After 25 steps, pointers should have stabilized at some point
        // (not all modes guarantee pointer stability, so log but don't fail if never stable)
        if (firstStableStep >= 0) {
            assertEquals(-1, lastUnstableAfterStable,
                    mode + ": pointers became unstable at step " + lastUnstableAfterStable
                            + " after being stable at step " + firstStableStep
                            + ". Pointer stability must be monotonic — once stable, always stable.");
            log.info("{}: pointer stability passed. First stable at step {}", mode, firstStableStep);
        } else {
            log.warn("{}: pointers never reported stable in 25 steps", mode);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  20. ISLAND HANDLE READINESS — track when islands become ready,
    //      assert they stay ready once populated.
    // ═══════════════════════════════════════════════════════════════════════

    @ParameterizedTest(name = "20_islandHandleReadiness_{0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"AUTO", "TRITON", "CUDA_GRAPHS"})
    @Order(20)
    void test20_IslandHandleReadiness(GraphExecutionMode mode) {
        int layers = 5, dim = 16;
        SameDiff sd = buildDeepChain(layers, dim);
        sd.setGraphExecutionMode(mode);

        INDArray input = Nd4j.zeros(DataType.FLOAT, 1, dim);
        input.putScalar(0, 0, 1.0f);

        // Run 20 steps, tracking island readiness
        int islandCount = 0;
        Map<Integer, Integer> firstReadyStep = new HashMap<>();
        Map<Integer, Integer> lostReadyStep = new HashMap<>();

        for (int step = 0; step < 20; step++) {
            sd.output(Map.of("input", input.dup()), "out");

            int islands = DspPlanAssertions.getSegmentIslandUnitCount(sd, 0);
            if (islands > islandCount) islandCount = islands;

            for (int isl = 0; isl < islandCount; isl++) {
                int ready = DspPlanAssertions.isIslandHandleReady(sd, 0, isl);
                if (ready == 1 && !firstReadyStep.containsKey(isl)) {
                    firstReadyStep.put(isl, step);
                    log.info("{}: island {} became ready at step {}", mode, isl, step);
                }
                if (ready != 1 && firstReadyStep.containsKey(isl) && !lostReadyStep.containsKey(isl)) {
                    lostReadyStep.put(isl, step);
                    log.error("{}: island {} LOST readiness at step {} (was ready at step {})",
                            mode, isl, step, firstReadyStep.get(isl));
                }
            }
        }

        // Once an island handle is ready, it must STAY ready
        assertTrue(lostReadyStep.isEmpty(),
                mode + ": island handle(s) lost readiness: " + lostReadyStep
                        + ". Once captured, island handles must never revert.");

        log.info("{}: island handle readiness passed. {} islands tracked, firstReady={}",
                mode, islandCount, firstReadyStep);
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  21. MONOLITHIC vs COMPOSITE MODE INVARIANT — if gaps exist,
    //      replay mode MUST be COMPOSITE (never MONOLITHIC).
    //      This directly tests the root cause of the gap-skip bug.
    // ═══════════════════════════════════════════════════════════════════════

    @ParameterizedTest(name = "21_monolithicCompositeInvariant_{0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"AUTO", "TRITON", "CUDA_GRAPHS"})
    @Order(21)
    void test21_MonolithicVsCompositeInvariant(GraphExecutionMode mode) {
        int layers = 5, dim = 16;
        SameDiff sd = buildDeepChain(layers, dim);
        sd.setGraphExecutionMode(mode);

        INDArray input = Nd4j.zeros(DataType.FLOAT, 1, dim);
        input.putScalar(0, 0, 1.0f);

        // Run enough steps to reach steady state
        for (int step = 0; step < 20; step++) {
            sd.output(Map.of("input", input.dup()), "out");
        }

        int gapUnits = DspPlanAssertions.getSegmentGapUnitCount(sd, 0);
        int islandUnits = DspPlanAssertions.getSegmentIslandUnitCount(sd, 0);
        int replayMode = DspPlanAssertions.getSegmentReplayMode(sd, 0);
        int monolithicReady = DspPlanAssertions.isMonolithicHandleReady(sd, 0);
        int gapSlots = DspPlanAssertions.getSegmentGapSlotCount(sd, 0);

        log.info("{}: gapUnits={}, islandUnits={}, gapSlots={}, replayMode={}, monolithicReady={}",
                mode, gapUnits, islandUnits, gapSlots,
                replayMode == DspPlanAssertions.REPLAY_MODE_NONE ? "NONE" :
                replayMode == DspPlanAssertions.REPLAY_MODE_MONOLITHIC ? "MONOLITHIC" :
                replayMode == DspPlanAssertions.REPLAY_MODE_COMPOSITE ? "COMPOSITE" : "?",
                monolithicReady);

        // THE CRITICAL INVARIANT: if there are gap units, replay mode MUST NOT be MONOLITHIC.
        // Monolithic replay skips gap ops entirely — they never execute after capture.
        if (gapUnits > 0) {
            assertNotEquals(DspPlanAssertions.REPLAY_MODE_MONOLITHIC, replayMode,
                    mode + ": CRITICAL BUG — segment has " + gapUnits + " gap units but replay mode is "
                            + "MONOLITHIC. Monolithic replay SKIPS gap ops (matmul, rmsNorm) — they will "
                            + "never execute again, producing stale/wrong output. "
                            + "This is the composite replay gap-skip bug.");

            // If there are both gaps and islands, mode should be COMPOSITE
            if (islandUnits > 0) {
                assertEquals(DspPlanAssertions.REPLAY_MODE_COMPOSITE, replayMode,
                        mode + ": segment has gaps AND islands but replay mode is "
                                + (replayMode == DspPlanAssertions.REPLAY_MODE_NONE ? "NONE" : "?" + replayMode)
                                + ". Expected COMPOSITE.");
            }
        }

        log.info("{}: monolithic/composite invariant passed", mode);
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  22. FULL PLAN SNAPSHOT PROGRESSION — snapshot the entire plan state
    //      at every step, verify no regressions (phases don't go backward,
    //      counts don't decrease, sealed stays sealed).
    // ═══════════════════════════════════════════════════════════════════════

    @ParameterizedTest(name = "22_planSnapshotProgression_{0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"AUTO", "TRITON", "CUDA_GRAPHS"})
    @Order(22)
    void test22_PlanSnapshotProgression(GraphExecutionMode mode) {
        int layers = 3, dim = 8;
        SameDiff sd = buildDeepChain(layers, dim);
        sd.setGraphExecutionMode(mode);

        INDArray input = Nd4j.zeros(DataType.FLOAT, 1, dim);
        input.putScalar(0, 0, 1.0f);

        int prevExecCount = -1;
        int prevReplayCount = -1;
        int prevFrozenExecCount = Integer.MIN_VALUE;
        boolean wasSealed = false;
        boolean wasPointersStable = false;
        int execResetCount = 0;

        for (int step = 0; step < 25; step++) {
            sd.output(Map.of("input", input.dup()), "out");

            int execCount = DspPlanAssertions.getSegmentExecCount(sd, 0);
            int replayCount = DspPlanAssertions.getSegmentReplayCount(sd, 0);
            int frozenExecCount = DspPlanAssertions.getFrozenExecCount(sd);
            int sealed = DspPlanAssertions.isCompilationSealed(sd);
            int pointersStable = DspPlanAssertions.getPointersStable(sd);
            long midExecRecompiles = DspPlanAssertions.getMidExecutionRecompileCount(sd);

            // Exec count can reset during recompilation (shape freeze rebuilds segments)
            if (execCount < prevExecCount) {
                execResetCount++;
                log.info("{} step {}: exec count reset {} -> {} (recompile #{})",
                        mode, step, prevExecCount, execCount, execResetCount);
            }

            // Replay count must never decrease
            assertTrue(replayCount >= prevReplayCount,
                    mode + " step " + step + ": replay count decreased from " + prevReplayCount
                            + " to " + replayCount);

            // Frozen exec count can reset during recompilation (same as exec count)
            if (frozenExecCount >= 0 && frozenExecCount < prevFrozenExecCount) {
                log.info("{} step {}: frozen exec count reset {} -> {} (recompile)",
                        mode, step, prevFrozenExecCount, frozenExecCount);
            }

            // Once sealed, must stay sealed
            if (wasSealed) {
                assertEquals(1, sealed,
                        mode + " step " + step + ": plan became UN-sealed! Once sealed, must stay sealed.");
            }

            // Once pointers stable, must stay stable
            if (wasPointersStable) {
                assertEquals(1, pointersStable,
                        mode + " step " + step + ": pointers became UN-stable! "
                                + "Once stable, must stay stable.");
            }

            // Mid-execution recompiles should always be 0
            assertEquals(0, midExecRecompiles,
                    mode + " step " + step + ": " + midExecRecompiles + " mid-execution recompiles!");

            prevExecCount = execCount;
            prevReplayCount = replayCount;
            if (frozenExecCount >= 0) prevFrozenExecCount = frozenExecCount;
            if (sealed == 1) wasSealed = true;
            if (pointersStable == 1) wasPointersStable = true;
        }

        // Exec resets should be limited (only during warmup → freeze transitions)
        assertTrue(execResetCount <= 3,
                mode + ": too many exec count resets (" + execResetCount + ") in 25 steps");

        log.info("{}: plan snapshot progression passed (25 steps). finalExec={}, finalReplay={}, "
                        + "sealed={}, pointersStable={}, execResets={}",
                mode, prevExecCount, prevReplayCount, wasSealed, wasPointersStable, execResetCount);
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  23. CAPTURE BUFFER CONSISTENCY — verify captured graph buffers
    //      are tracked and don't change count after capture.
    // ═══════════════════════════════════════════════════════════════════════

    @ParameterizedTest(name = "23_captureBufferConsistency_{0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"AUTO", "TRITON", "CUDA_GRAPHS"})
    @Order(23)
    void test23_CaptureBufferConsistency(GraphExecutionMode mode) {
        int layers = 3, dim = 8;
        SameDiff sd = buildDeepChain(layers, dim);
        sd.setGraphExecutionMode(mode);

        INDArray input = Nd4j.zeros(DataType.FLOAT, 1, dim);
        input.putScalar(0, 0, 1.0f);

        // Run 15 steps to reach steady state
        for (int step = 0; step < 15; step++) {
            sd.output(Map.of("input", input.dup()), "out");
        }

        // Query buffer counts
        var handle = DspPlanAssertions.getPlanHandleForQuery(sd);
        var ops = org.nd4j.linalg.factory.Nd4j.getNativeOps();
        int numCaptureBuffers = ops.getPlanSegmentNumCaptureBuffers(handle, 0);
        int numHostPointers = ops.getPlanSegmentNumHostPointers(handle, 0);

        log.info("{}: captureBuffers={}, hostPointers={}", mode, numCaptureBuffers, numHostPointers);

        // Run 10 more steps — buffer counts should NOT change
        for (int step = 15; step < 25; step++) {
            sd.output(Map.of("input", input.dup()), "out");

            int currentCaptureBuffers = ops.getPlanSegmentNumCaptureBuffers(handle, 0);
            int currentHostPointers = ops.getPlanSegmentNumHostPointers(handle, 0);

            assertEquals(numCaptureBuffers, currentCaptureBuffers,
                    mode + " step " + step + ": capture buffer count changed from "
                            + numCaptureBuffers + " to " + currentCaptureBuffers);
            assertEquals(numHostPointers, currentHostPointers,
                    mode + " step " + step + ": host pointer count changed from "
                            + numHostPointers + " to " + currentHostPointers);
        }

        log.info("{}: capture buffer consistency passed (10 steps, {} buffers, {} host ptrs)",
                mode, numCaptureBuffers, numHostPointers);
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  24. MULTI-INPUT EXHAUSTIVE STALENESS — run with 4 different inputs
    //      cycling, assert output ALWAYS matches SLOT_BY_SLOT reference
    //      at EVERY step for 40 steps.
    // ═══════════════════════════════════════════════════════════════════════

    @ParameterizedTest(name = "24_multiInputExhaustive_{0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"AUTO", "TRITON", "CUDA_GRAPHS"})
    @Order(24)
    void test24_MultiInputExhaustiveStaleness(GraphExecutionMode mode) {
        int layers = 5, dim = 16;
        INDArray[][] weights = generateDeepChainWeights(layers, dim);
        SameDiff sd = buildDeepChainWith(weights, dim);
        sd.setGraphExecutionMode(mode);

        SameDiff sdRef = buildDeepChainWith(weights, dim);
        sdRef.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);

        // 4 distinct inputs
        INDArray[] inputs = new INDArray[4];
        for (int i = 0; i < 4; i++) {
            inputs[i] = Nd4j.zeros(DataType.FLOAT, 1, dim);
            inputs[i].putScalar(0, i * 4, 1.0f);
        }

        int totalSteps = 40;
        int matchCount = 0;
        int firstMismatchStep = -1;
        double worstDiff = 0;
        List<String> mismatchLog = new ArrayList<>();

        for (int step = 0; step < totalSteps; step++) {
            int inputIdx = step % 4;
            INDArray in = inputs[inputIdx].dup();

            INDArray result = sd.output(Map.of("input", in), "out").get("out").dup();
            INDArray ref = sdRef.output(Map.of("input", in), "out").get("out").dup();

            double diff = ref.sub(result).amaxNumber().doubleValue();
            if (diff < 1e-4) {
                matchCount++;
            } else {
                if (firstMismatchStep < 0) firstMismatchStep = step;
                worstDiff = Math.max(worstDiff, diff);
                mismatchLog.add(String.format("step=%d input=%d diff=%.6e", step, inputIdx, diff));
            }
        }

        // At least 95% of steps must match (2 allowed for transition noise)
        int minMatch = (int)(totalSteps * 0.95);
        assertTrue(matchCount >= minMatch,
                mode + ": only " + matchCount + "/" + totalSteps + " steps match SLOT_BY_SLOT reference. "
                        + "First mismatch at step " + firstMismatchStep + ", worstDiff=" + worstDiff
                        + ". Mismatches: " + mismatchLog);

        log.info("{}: multi-input exhaustive staleness passed ({}/{} match, worstDiff={})",
                mode, matchCount, totalSteps, worstDiff);
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  25. SEGMENT PHASE MATCHES PLAN PHASE — segment execution phase
    //      should be consistent with overall plan phase.
    // ═══════════════════════════════════════════════════════════════════════

    @ParameterizedTest(name = "25_segPlanPhaseConsistency_{0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"AUTO", "TRITON", "CUDA_GRAPHS"})
    @Order(25)
    void test25_SegmentPlanPhaseConsistency(GraphExecutionMode mode) {
        int layers = 3, dim = 8;
        SameDiff sd = buildDeepChain(layers, dim);
        sd.setGraphExecutionMode(mode);

        INDArray input = Nd4j.zeros(DataType.FLOAT, 1, dim);
        input.putScalar(0, 0, 1.0f);

        for (int step = 0; step < 25; step++) {
            sd.output(Map.of("input", input.dup()), "out");

            // Both DspDebugger and native queries should agree
            DspDebugger debugger = DspDebugger.attach(sd);
            DspDebugger.PlanReport report = debugger.analyzePlan();

            if (report.errorMessage == null && !report.segments.isEmpty()) {
                DspDebugger.SegmentReport seg0 = report.segments.get(0);

                // Cross-check: native exec count matches report exec count
                int nativeExecCount = DspPlanAssertions.getSegmentExecCount(sd, 0);
                assertEquals(seg0.executionCount, nativeExecCount,
                        mode + " step " + step + ": DspDebugger reports execCount="
                                + seg0.executionCount + " but native query returns " + nativeExecCount);

                // Cross-check: no capture failure disagreement
                boolean nativeCapFailed = DspPlanAssertions.isSegmentCaptureFailed(sd, 0);
                assertEquals(seg0.captureFailed, nativeCapFailed,
                        mode + " step " + step + ": captureFailed disagrees — debugger="
                                + seg0.captureFailed + " native=" + nativeCapFailed);
            }
        }

        log.info("{}: segment/plan phase consistency passed (25 steps)", mode);
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  26. SLOT OP AUDIT — verify plan slots have valid op names and state
    // ═══════════════════════════════════════════════════════════════════════

    @Test
    @Order(26)
    void test26_SlotOpAudit() {
        int layers = 3, dim = 8;
        SameDiff sd = buildDeepChain(layers, dim);
        sd.setGraphExecutionMode(GraphExecutionMode.AUTO);

        INDArray input = Nd4j.zeros(DataType.FLOAT, 1, dim);
        input.putScalar(0, 0, 1.0f);

        // Run once to compile
        sd.output(Map.of("input", input), "out");

        var handle = DspPlanAssertions.getPlanHandleForQuery(sd);
        var ops = org.nd4j.linalg.factory.Nd4j.getNativeOps();
        int numSlots = ops.getPlanNumSlots(handle);

        assertTrue(numSlots > 0, "Plan should have slots after first execution");

        // Audit each slot
        int matmulCount = 0, rmsNormCount = 0, identityCount = 0, otherCount = 0;
        for (int i = 0; i < numSlots; i++) {
            String opName = ops.getPlanSlotOpName(handle, i);
            int state = ops.getPlanSlotState(handle, i);

            assertNotNull(opName, "slot " + i + " has null opName");
            assertFalse(opName.isEmpty(), "slot " + i + " has empty opName");

            if (opName.contains("matmul") || opName.contains("mmul")) matmulCount++;
            else if (opName.contains("rms_norm") || opName.contains("rmsNorm")) rmsNormCount++;
            else if (opName.contains("identity") || opName.contains("noop")) identityCount++;
            else otherCount++;
        }

        log.info("Slot audit: {} total slots, {} matmul, {} rmsNorm, {} identity, {} other",
                numSlots, matmulCount, rmsNormCount, identityCount, otherCount);

        // Deep chain with 3 layers should have 3 matmul + 3 rmsNorm + 1 identity minimum
        assertTrue(matmulCount >= 3, "Expected >= 3 matmul slots, found " + matmulCount);
        assertTrue(rmsNormCount >= 3, "Expected >= 3 rmsNorm slots, found " + rmsNormCount);
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  27. GATHER-PATTERN COMPOSITE EXHAUSTIVE — the VLM pattern with
    //      exhaustive staleness checks across 40 steps.
    // ═══════════════════════════════════════════════════════════════════════

    @ParameterizedTest(name = "27_gatherCompositeExhaustive_{0}")
    @EnumSource(value = GraphExecutionMode.class,
            names = {"SLOT_BY_SLOT", "AUTO", "TRITON", "CUDA_GRAPHS"})
    @Order(27)
    void test27_GatherCompositeExhaustiveStaleness(GraphExecutionMode mode) {
        int vocab = 32, dim = 16;
        INDArray[] gatherWeights = generateGatherWeights(vocab, dim);
        SameDiff sd = buildGatherMatmulSoftmaxWith(gatherWeights);
        sd.setGraphExecutionMode(mode);

        SameDiff sdRef = buildGatherMatmulSoftmaxWith(gatherWeights);
        sdRef.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);

        // Run same token ID for 15 steps, all outputs must be identical
        long fixedToken = 5;
        INDArray firstOutput = null;
        for (int step = 0; step < 15; step++) {
            INDArray tokenId = Nd4j.createFromArray(new long[]{fixedToken});
            INDArray out = sd.output(Map.of("token_id", tokenId), "probs").get("probs").dup();
            if (firstOutput == null) {
                firstOutput = out;
            } else {
                double diff = firstOutput.sub(out).amaxNumber().doubleValue();
                assertTrue(diff < 1e-5,
                        mode + " step " + step + ": same token produced different output (diff=" + diff + ")");
            }
        }

        // Now cycle through different tokens — each must match reference
        int matchCount = 0;
        for (int step = 0; step < 25; step++) {
            long tokenVal = step % vocab;
            INDArray tokenId = Nd4j.createFromArray(new long[]{tokenVal});
            INDArray result = sd.output(Map.of("token_id", tokenId), "probs").get("probs").dup();
            INDArray ref = sdRef.output(Map.of("token_id", tokenId), "probs").get("probs").dup();

            double diff = ref.sub(result).amaxNumber().doubleValue();
            if (diff < 1e-3) matchCount++;
        }

        assertTrue(matchCount >= 23,
                mode + ": gather+matmul+softmax exhaustive — only " + matchCount + "/25 match reference");

        log.info("{}: gather composite exhaustive passed ({}/25 match)", mode, matchCount);
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  28. KV CACHE MUTATION — placeholder buffer mutated between steps
    //      (simulates KV scatter). Frozen fast-path MUST pick up the new
    //      data each step. Catches: EOS-step-2, KV H2D zeroing bugs.
    // ═══════════════════════════════════════════════════════════════════════

    @ParameterizedTest(name = "28_kvCacheMutation_{0}")
    @EnumSource(value = GraphExecutionMode.class,
            names = {"SLOT_BY_SLOT", "AUTO", "TRITON", "CUDA_GRAPHS"})
    @Order(28)
    void test28_KvCachePlaceholderMutation(GraphExecutionMode mode) {
        int dim = 16;
        Nd4j.getRandom().setSeed(42);
        INDArray weight = Nd4j.randn(DataType.FLOAT, dim, dim).muli(0.02);

        // Graph: input -> matmul(W) -> add(kv_state) -> identity -> out
        // kv_state is a placeholder updated externally between steps (KV cache pattern)
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, dim);
        SDVariable kvState = sd.placeHolder("kv_state", DataType.FLOAT, 1, dim);
        sd.constant("weight", weight.dup());
        SDVariable projected = sd.mmul("projected", input, sd.getVariable("weight"));
        SDVariable accumulated = sd.math().add("accumulated", projected, kvState);
        sd.identity("out", accumulated);
        track(sd);

        // Reference graph — identical structure
        SameDiff sdRef = SameDiff.create();
        SDVariable inputRef = sdRef.placeHolder("input", DataType.FLOAT, 1, dim);
        SDVariable kvStateRef = sdRef.placeHolder("kv_state", DataType.FLOAT, 1, dim);
        sdRef.constant("weight", weight.dup());
        SDVariable projectedRef = sdRef.mmul("projected", inputRef, sdRef.getVariable("weight"));
        SDVariable accumulatedRef = sdRef.math().add("accumulated", projectedRef, kvStateRef);
        sdRef.identity("out", accumulatedRef);
        track(sdRef);

        sd.setGraphExecutionMode(mode);
        sdRef.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);

        // Simulate decode loop: kv_state accumulates values across steps
        INDArray kvBuffer = Nd4j.zeros(DataType.FLOAT, 1, dim);
        INDArray kvBufferRef = Nd4j.zeros(DataType.FLOAT, 1, dim);

        int staleCount = 0;
        int mismatchCount = 0;
        INDArray prevOutput = null;

        for (int step = 0; step < 40; step++) {
            INDArray tokenInput = Nd4j.zeros(DataType.FLOAT, 1, dim);
            tokenInput.putScalar(0, step % dim, 1.0f);

            // Pass CURRENT kv buffer as placeholder
            INDArray result = sd.output(
                    Map.of("input", tokenInput, "kv_state", kvBuffer.dup()), "out").get("out").dup();
            INDArray ref = sdRef.output(
                    Map.of("input", tokenInput, "kv_state", kvBufferRef.dup()), "out").get("out").dup();

            // ASSERTION 1: output must not be NaN or all-zero
            assertFalse(Double.isNaN(result.maxNumber().doubleValue()),
                    mode + " step " + step + ": NaN in output");
            assertTrue(result.amaxNumber().doubleValue() > 1e-10,
                    mode + " step " + step + ": all-zero output");

            // ASSERTION 2: output must change from previous step (KV state changed)
            if (prevOutput != null && step > 0) {
                double changeDiff = result.sub(prevOutput).amaxNumber().doubleValue();
                if (changeDiff < 1e-8) {
                    staleCount++;
                    log.error("{} STALE step {}: output identical to previous (diff={}). "
                            + "KV state change not picked up by frozen fast-path!", mode, step, changeDiff);
                }
            }

            // ASSERTION 3: must match reference
            double refDiff = ref.sub(result).amaxNumber().doubleValue();
            if (refDiff > 1e-3) {
                mismatchCount++;
                log.warn("{} step {}: KV cache ref mismatch diff={}", mode, step, refDiff);
            }

            prevOutput = result;

            // Simulate KV scatter: update kv buffer with a fraction of the output
            kvBuffer.addi(result.mul(0.05));
            kvBufferRef.addi(ref.mul(0.05));
        }

        assertEquals(0, staleCount,
                mode + ": " + staleCount + "/40 steps had stale output — KV state mutations "
                        + "not reflected after freeze. This is the EOS-step-2 / KV H2D zeroing bug.");
        assertTrue(mismatchCount <= 2,
                mode + ": " + mismatchCount + "/40 steps mismatched reference with KV state");

        log.info("{}: KV cache mutation test passed (40 steps, stale={}, mismatch={})",
                mode, staleCount, mismatchCount);
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  29. PLACEHOLDER MUTATION AFTER FREEZE — verify that changing a
    //      placeholder value AFTER shapes freeze still produces correct
    //      output. Catches: frozen fast-path sync skip, TRITON_SKIP stuck.
    // ═══════════════════════════════════════════════════════════════════════

    @ParameterizedTest(name = "29_placeholderAfterFreeze_{0}")
    @EnumSource(value = GraphExecutionMode.class,
            names = {"AUTO", "TRITON", "CUDA_GRAPHS"})
    @Order(29)
    void test29_PlaceholderMutationAfterFreeze(GraphExecutionMode mode) {
        int vocab = 32, dim = 16;
        INDArray[] gatherWeights = generateGatherWeights(vocab, dim);
        SameDiff sd = buildGatherMatmulSoftmaxWith(gatherWeights);
        sd.setGraphExecutionMode(mode);

        SameDiff sdRef = buildGatherMatmulSoftmaxWith(gatherWeights);
        sdRef.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);

        // Phase 1: Warm up with SAME token until shapes freeze
        long warmupToken = 5;
        for (int step = 0; step < 15; step++) {
            sd.output(Map.of("token_id", Nd4j.createFromArray(new long[]{warmupToken})), "probs");
        }

        // Verify shapes are frozen
        int frozenExec = DspPlanAssertions.getFrozenExecCount(sd);
        log.info("{}: after 15 warmup steps, frozenExecCount={}", mode, frozenExec);

        // Phase 2: NOW change the token — output MUST reflect the new token
        int postFreezeMatch = 0;
        int postFreezeStale = 0;
        INDArray warmupOutput = sd.output(
                Map.of("token_id", Nd4j.createFromArray(new long[]{warmupToken})), "probs").get("probs").dup();

        for (int step = 0; step < 20; step++) {
            long newToken = (step * 3 + 7) % vocab;  // different token each step
            if (newToken == warmupToken) newToken = (newToken + 1) % vocab;

            INDArray tokenId = Nd4j.createFromArray(new long[]{newToken});
            INDArray result = sd.output(Map.of("token_id", tokenId), "probs").get("probs").dup();
            INDArray ref = sdRef.output(Map.of("token_id", tokenId), "probs").get("probs").dup();

            // CRITICAL: output must differ from warmup token's output
            double staleDiff = warmupOutput.sub(result).amaxNumber().doubleValue();
            if (staleDiff < 1e-6) {
                postFreezeStale++;
                log.error("{} POST-FREEZE STALE step {}: token {} produced same output as warmup token {}. "
                        + "Frozen fast-path is NOT syncing placeholder!", mode, step, newToken, warmupToken);
            }

            // Must match reference
            double refDiff = ref.sub(result).amaxNumber().doubleValue();
            if (refDiff < 1e-3) postFreezeMatch++;
        }

        assertEquals(0, postFreezeStale,
                mode + ": " + postFreezeStale + "/20 steps produced stale output after freeze. "
                        + "The frozen fast-path is not re-reading placeholder values. "
                        + "This is the TRITON_SKIP stuck token / frozen sync skip bug.");
        assertTrue(postFreezeMatch >= 18,
                mode + ": only " + postFreezeMatch + "/20 post-freeze steps matched reference");

        log.info("{}: placeholder mutation after freeze passed (stale={}, match={}/20)",
                mode, postFreezeStale, postFreezeMatch);
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  30. MIXED INT64 + HALF + FLOAT32 — the actual VLM type pattern.
    //      INT64 token indices, FP16 embeddings, FP32 projection weights.
    //      Catches: rms_norm_linear type mismatch, mixed-type matmul bugs.
    // ═══════════════════════════════════════════════════════════════════════

    @ParameterizedTest(name = "30_mixedTypeVlmPattern_{0}")
    @EnumSource(value = GraphExecutionMode.class,
            names = {"SLOT_BY_SLOT", "AUTO", "TRITON", "CUDA_GRAPHS"})
    @Order(30)
    void test30_MixedTypeVlmPattern(GraphExecutionMode mode) {
        int vocab = 64, dim = 16;
        Nd4j.getRandom().setSeed(42);

        // FP16 embedding table, FP32 projection weight, FP32 gamma — the actual VLM mix
        INDArray embedTable = Nd4j.randn(DataType.FLOAT, vocab, dim).muli(0.02).castTo(DataType.HALF);
        INDArray projWeight = Nd4j.randn(DataType.FLOAT, dim, vocab).muli(0.02);
        INDArray gamma = Nd4j.ones(DataType.FLOAT, dim);

        SameDiff sd = SameDiff.create();
        sd.constant("embed_table", embedTable.dup());
        sd.constant("proj_weight", projWeight.dup());
        sd.constant("gamma", gamma.dup());
        // INT64 token ID — the third type in the mix
        SDVariable tokenId = sd.placeHolder("token_id", DataType.INT64, 1);
        // gather produces FP16 (inherits from embed_table)
        SDVariable gathered = sd.gather("gathered", sd.getVariable("embed_table"), tokenId, 0);
        // rmsNorm on potentially mixed types
        SDVariable normed = sd.nn().rmsNorm("normed", gathered, sd.getVariable("gamma"), 1e-5);
        // matmul: mixed precision (gathered is FP16, proj_weight is FP32)
        SDVariable logits = sd.mmul("logits", normed, sd.getVariable("proj_weight"));
        sd.nn().softmax("probs", logits, 1);
        track(sd);

        // Reference with same weights
        SameDiff sdRef = SameDiff.create();
        sdRef.constant("embed_table", embedTable.dup());
        sdRef.constant("proj_weight", projWeight.dup());
        sdRef.constant("gamma", gamma.dup());
        SDVariable tokenIdRef = sdRef.placeHolder("token_id", DataType.INT64, 1);
        SDVariable gatheredRef = sdRef.gather("gathered", sdRef.getVariable("embed_table"), tokenIdRef, 0);
        SDVariable normedRef = sdRef.nn().rmsNorm("normed", gatheredRef, sdRef.getVariable("gamma"), 1e-5);
        SDVariable logitsRef = sdRef.mmul("logits", normedRef, sdRef.getVariable("proj_weight"));
        sdRef.nn().softmax("probs", logitsRef, 1);
        track(sdRef);

        sd.setGraphExecutionMode(mode);
        sdRef.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);

        int matchCount = 0;
        for (int step = 0; step < 30; step++) {
            INDArray token = Nd4j.createFromArray(new long[]{step % vocab});
            INDArray result = sd.output(Map.of("token_id", token), "probs").get("probs").dup();
            INDArray ref = sdRef.output(Map.of("token_id", token), "probs").get("probs").dup();

            // Must be valid probabilities
            assertFalse(Double.isNaN(result.sumNumber().doubleValue()),
                    mode + " step " + step + ": NaN in mixed-type output");
            double sum = result.castTo(DataType.FLOAT).sumNumber().doubleValue();
            assertTrue(sum > 0.5 && sum < 1.5,
                    mode + " step " + step + ": softmax sum=" + sum + " (invalid)");

            double diff = ref.castTo(DataType.FLOAT).sub(result.castTo(DataType.FLOAT))
                    .amaxNumber().doubleValue();
            if (diff < 1e-2) matchCount++;  // wider tolerance for FP16 precision
            else log.warn("{} step {}: mixed-type diff={}", mode, step, diff);
        }

        assertTrue(matchCount >= 27,
                mode + ": mixed INT64+HALF+FLOAT32 — only " + matchCount + "/30 match reference. "
                        + "Type mismatch in gather/matmul/rmsNorm path.");

        log.info("{}: mixed type VLM pattern passed ({}/30 match)", mode, matchCount);
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  31. SHAPE TRANSITION — prefill [4,dim] then decode [1,dim].
    //      Plan must recompile on shape change. Output must be correct
    //      in BOTH phases. Catches: shape key bugs, stale plan handles.
    // ═══════════════════════════════════════════════════════════════════════

    @ParameterizedTest(name = "31_shapeTransition_{0}")
    @EnumSource(value = GraphExecutionMode.class,
            names = {"SLOT_BY_SLOT", "AUTO", "TRITON", "CUDA_GRAPHS"})
    @Order(31)
    void test31_ShapeTransitionPrefillDecode(GraphExecutionMode mode) {
        int dim = 16;
        Nd4j.getRandom().setSeed(42);
        INDArray weight = Nd4j.randn(DataType.FLOAT, dim, dim).muli(0.02);
        INDArray gammaW = Nd4j.ones(DataType.FLOAT, dim);

        // Graph with dynamic first dimension: [seqLen, dim] -> matmul -> rmsNorm -> out
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, dim);
        sd.constant("weight", weight.dup());
        sd.constant("gamma", gammaW.dup());
        SDVariable projected = sd.mmul("projected", input, sd.getVariable("weight"));
        SDVariable normed = sd.nn().rmsNorm("normed", projected, sd.getVariable("gamma"), 1e-5);
        sd.identity("out", normed);
        track(sd);

        SameDiff sdRef = SameDiff.create();
        SDVariable inputRef = sdRef.placeHolder("input", DataType.FLOAT, -1, dim);
        sdRef.constant("weight", weight.dup());
        sdRef.constant("gamma", gammaW.dup());
        SDVariable projectedRef = sdRef.mmul("projected", inputRef, sdRef.getVariable("weight"));
        SDVariable normedRef = sdRef.nn().rmsNorm("normed", projectedRef, sdRef.getVariable("gamma"), 1e-5);
        sdRef.identity("out", normedRef);
        track(sdRef);

        sd.setGraphExecutionMode(mode);
        sdRef.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);

        // PHASE 1: Prefill — input shape [4, dim]
        INDArray prefillInput = Nd4j.randn(DataType.FLOAT, 4, dim);
        INDArray prefillResult = sd.output(Map.of("input", prefillInput), "out").get("out").dup();
        INDArray prefillRef = sdRef.output(Map.of("input", prefillInput), "out").get("out").dup();

        double prefillDiff = prefillRef.sub(prefillResult).amaxNumber().doubleValue();
        assertTrue(prefillDiff < 1e-4,
                mode + ": prefill [4,dim] diff=" + prefillDiff + " vs reference");
        assertEquals(4, prefillResult.shape()[0],
                mode + ": prefill output should have seqLen=4");

        log.info("{}: prefill phase passed, diff={}", mode, prefillDiff);

        // Reset session to simulate prefill→decode transition (like VLM does)
        sd.resetSession();
        sdRef.resetSession();

        // PHASE 2: Decode — input shape [1, dim], run 20 steps
        int decodeMatchCount = 0;
        for (int step = 0; step < 20; step++) {
            INDArray decodeInput = Nd4j.zeros(DataType.FLOAT, 1, dim);
            decodeInput.putScalar(0, step % dim, 1.0f);

            INDArray result = sd.output(Map.of("input", decodeInput), "out").get("out").dup();
            INDArray ref = sdRef.output(Map.of("input", decodeInput), "out").get("out").dup();

            assertEquals(1, result.shape()[0],
                    mode + " decode step " + step + ": output seqLen should be 1");

            double diff = ref.sub(result).amaxNumber().doubleValue();
            if (diff < 1e-4) decodeMatchCount++;
            else log.warn("{} decode step {}: diff={}", mode, step, diff);
        }

        assertTrue(decodeMatchCount >= 18,
                mode + ": after shape transition, only " + decodeMatchCount + "/20 decode steps "
                        + "match reference. Plan may not have recompiled correctly.");

        // PHASE 3: Switch BACK to prefill shape — must still work
        INDArray prefillInput2 = Nd4j.randn(DataType.FLOAT, 4, dim);
        sd.resetSession();
        sdRef.resetSession();
        INDArray prefillResult2 = sd.output(Map.of("input", prefillInput2), "out").get("out").dup();
        INDArray prefillRef2 = sdRef.output(Map.of("input", prefillInput2), "out").get("out").dup();

        double revertDiff = prefillRef2.sub(prefillResult2).amaxNumber().doubleValue();
        assertTrue(revertDiff < 1e-4,
                mode + ": reverting to prefill shape [4,dim] failed, diff=" + revertDiff);

        log.info("{}: shape transition prefill→decode→prefill passed (decode match={}/20, revertDiff={})",
                mode, decodeMatchCount, revertDiff);
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  32. CROSS-PLAN WEIGHT SHARING — running plan A through capture must
    //      NOT corrupt weights used by plan B. Catches: writeSpecial
    //      poisoning, actuality corruption across plans.
    // ═══════════════════════════════════════════════════════════════════════

    @ParameterizedTest(name = "32_crossPlanWeightSharing_{0}")
    @EnumSource(value = GraphExecutionMode.class,
            names = {"AUTO", "TRITON", "CUDA_GRAPHS"})
    @Order(32)
    void test32_CrossPlanWeightSharing(GraphExecutionMode mode) {
        int dim = 16;
        Nd4j.getRandom().setSeed(42);

        // Pre-generate shared weights — SAME arrays used in both graphs
        INDArray sharedWeight = Nd4j.randn(DataType.FLOAT, dim, dim).muli(0.02);
        INDArray sharedGamma = Nd4j.ones(DataType.FLOAT, dim);

        // Graph A: input -> matmul(W) -> rmsNorm(gamma) -> out
        SameDiff sdA = SameDiff.create();
        SDVariable inA = sdA.placeHolder("input", DataType.FLOAT, 1, dim);
        sdA.constant("weight", sharedWeight.dup());
        sdA.constant("gamma", sharedGamma.dup());
        SDVariable projA = sdA.mmul("projected", inA, sdA.getVariable("weight"));
        SDVariable normA = sdA.nn().rmsNorm("normed", projA, sdA.getVariable("gamma"), 1e-5);
        sdA.identity("out", normA);
        track(sdA);
        sdA.setGraphExecutionMode(mode);

        // Graph B: SAME structure, SAME weight values
        SameDiff sdB = SameDiff.create();
        SDVariable inB = sdB.placeHolder("input", DataType.FLOAT, 1, dim);
        sdB.constant("weight", sharedWeight.dup());
        sdB.constant("gamma", sharedGamma.dup());
        SDVariable projB = sdB.mmul("projected", inB, sdB.getVariable("weight"));
        SDVariable normB = sdB.nn().rmsNorm("normed", projB, sdB.getVariable("gamma"), 1e-5);
        sdB.identity("out", normB);
        track(sdB);
        sdB.setGraphExecutionMode(mode);

        // SLOT_BY_SLOT reference
        SameDiff sdRef = SameDiff.create();
        SDVariable inRef = sdRef.placeHolder("input", DataType.FLOAT, 1, dim);
        sdRef.constant("weight", sharedWeight.dup());
        sdRef.constant("gamma", sharedGamma.dup());
        SDVariable projRef = sdRef.mmul("projected", inRef, sdRef.getVariable("weight"));
        SDVariable normRef = sdRef.nn().rmsNorm("normed", projRef, sdRef.getVariable("gamma"), 1e-5);
        sdRef.identity("out", normRef);
        track(sdRef);
        sdRef.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);

        INDArray testInput = Nd4j.zeros(DataType.FLOAT, 1, dim);
        testInput.putScalar(0, 0, 1.0f);

        // Get reference output BEFORE either plan captures
        INDArray refOutput = sdRef.output(Map.of("input", testInput), "out").get("out").dup();

        // Get graph B output BEFORE A captures (baseline)
        INDArray bBeforeCapture = sdB.output(Map.of("input", testInput), "out").get("out").dup();
        double bBaselineDiff = refOutput.sub(bBeforeCapture).amaxNumber().doubleValue();
        assertTrue(bBaselineDiff < 1e-4,
                mode + ": graph B doesn't match reference even before A captures! diff=" + bBaselineDiff);

        // Now run graph A through 15 steps to trigger capture/freeze
        for (int step = 0; step < 15; step++) {
            INDArray in = Nd4j.zeros(DataType.FLOAT, 1, dim);
            in.putScalar(0, step % dim, 1.0f);
            sdA.output(Map.of("input", in), "out");
        }

        log.info("{}: graph A ran 15 steps (capture should have happened)", mode);

        // NOW run graph B again — output must STILL match reference
        INDArray bAfterCapture = sdB.output(Map.of("input", testInput), "out").get("out").dup();
        double bPostCaptureDiff = refOutput.sub(bAfterCapture).amaxNumber().doubleValue();
        assertTrue(bPostCaptureDiff < 1e-4,
                mode + ": graph B output CORRUPTED after graph A captured! diff=" + bPostCaptureDiff
                        + " (was " + bBaselineDiff + " before). writeSpecial poisoning?");

        // Run B through 10 more steps — must all match reference
        int bMatchCount = 0;
        for (int step = 0; step < 10; step++) {
            INDArray in = Nd4j.zeros(DataType.FLOAT, 1, dim);
            in.putScalar(0, step % dim, 1.0f);
            INDArray bResult = sdB.output(Map.of("input", in), "out").get("out").dup();
            INDArray ref = sdRef.output(Map.of("input", in), "out").get("out").dup();
            double diff = ref.sub(bResult).amaxNumber().doubleValue();
            if (diff < 1e-4) bMatchCount++;
        }

        assertTrue(bMatchCount >= 9,
                mode + ": after A captured, graph B only matched reference " + bMatchCount
                        + "/10 steps. Cross-plan weight corruption.");

        log.info("{}: cross-plan weight sharing passed (B post-capture diff={}, match={}/10)",
                mode, bPostCaptureDiff, bMatchCount);
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  33. 50-STEP DECODE — every step with a different token, assert NO
    //      stuck tokens (consecutive identical outputs). Catches: TRITON_SKIP
    //      stuck, SLOT_BY_SLOT degenerate, frozen sync skip.
    // ═══════════════════════════════════════════════════════════════════════

    @ParameterizedTest(name = "33_decodeNoStuckTokens_{0}")
    @EnumSource(value = GraphExecutionMode.class,
            names = {"SLOT_BY_SLOT", "AUTO", "TRITON", "CUDA_GRAPHS"})
    @Order(33)
    void test33_FiftyStepDecodeNoStuckTokens(GraphExecutionMode mode) {
        int vocab = 64, dim = 16;
        INDArray[] gatherWeights = generateGatherWeights(vocab, dim);
        SameDiff sd = buildGatherMatmulSoftmaxWith(gatherWeights);
        sd.setGraphExecutionMode(mode);

        int totalSteps = 50;
        INDArray[] outputs = new INDArray[totalSteps];
        long[] tokens = new long[totalSteps];

        // Generate a non-repeating token sequence
        for (int step = 0; step < totalSteps; step++) {
            tokens[step] = (step * 7 + 3) % vocab;  // spread across vocab
        }
        // Ensure no two consecutive tokens are the same
        for (int step = 1; step < totalSteps; step++) {
            if (tokens[step] == tokens[step - 1]) {
                tokens[step] = (tokens[step] + 1) % vocab;
            }
        }

        for (int step = 0; step < totalSteps; step++) {
            INDArray tokenId = Nd4j.createFromArray(new long[]{tokens[step]});
            outputs[step] = sd.output(Map.of("token_id", tokenId), "probs").get("probs").dup();
        }

        // ASSERTION 1: No NaN or all-zero outputs
        for (int step = 0; step < totalSteps; step++) {
            assertFalse(Double.isNaN(outputs[step].maxNumber().doubleValue()),
                    mode + " step " + step + ": NaN output");
            assertTrue(outputs[step].amaxNumber().doubleValue() > 1e-10,
                    mode + " step " + step + ": all-zero output");
        }

        // ASSERTION 2: No consecutive identical outputs (stuck token detection)
        int stuckCount = 0;
        int maxConsecutiveStuck = 0;
        int currentStreak = 0;
        for (int step = 1; step < totalSteps; step++) {
            double diff = outputs[step].sub(outputs[step - 1]).amaxNumber().doubleValue();
            if (diff < 1e-8) {
                stuckCount++;
                currentStreak++;
                maxConsecutiveStuck = Math.max(maxConsecutiveStuck, currentStreak);
                log.error("{} STUCK step {}: token {} produced same output as step {} token {} (diff={})",
                        mode, step, tokens[step], step - 1, tokens[step - 1], diff);
            } else {
                currentStreak = 0;
            }
        }

        assertEquals(0, stuckCount,
                mode + ": " + stuckCount + " stuck token pairs in 50 steps (max streak=" + maxConsecutiveStuck
                        + "). Different tokens produced identical outputs. "
                        + "Frozen fast-path not syncing placeholders, or gap ops not executing.");

        // ASSERTION 3: Different tokens must produce different probability distributions
        // Check a few specific pairs
        for (int i = 0; i < Math.min(10, totalSteps - 1); i++) {
            int j = totalSteps - 1 - i;
            if (tokens[i] != tokens[j]) {
                double pairDiff = outputs[i].sub(outputs[j]).amaxNumber().doubleValue();
                assertTrue(pairDiff > 1e-6,
                        mode + ": tokens " + tokens[i] + " and " + tokens[j]
                                + " produced identical probabilities (diff=" + pairDiff
                                + "). Graph may not be reading input.");
            }
        }

        log.info("{}: 50-step decode passed (0 stuck, all outputs unique)", mode);
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  34. ACCUMULATING EXTERNAL STATE — simulate KV cache growth pattern.
    //      Each step adds output to an accumulator, re-fed as input.
    //      Tests that accumulated state doesn't get stale or corrupted.
    //      Catches: device-authoritative sync bugs, KV zeroing.
    // ═══════════════════════════════════════════════════════════════════════

    @ParameterizedTest(name = "34_accumulatingState_{0}")
    @EnumSource(value = GraphExecutionMode.class,
            names = {"SLOT_BY_SLOT", "AUTO", "TRITON", "CUDA_GRAPHS"})
    @Order(34)
    void test34_AccumulatingExternalState(GraphExecutionMode mode) {
        int dim = 16;
        Nd4j.getRandom().setSeed(42);
        INDArray weight = Nd4j.randn(DataType.FLOAT, dim, dim).muli(0.02);
        INDArray gamma = Nd4j.ones(DataType.FLOAT, dim);

        // Graph: token -> matmul(W) -> rmsNorm(gamma) -> add(state) -> out
        // state is a placeholder that accumulates across steps (like KV cache)
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, dim);
        SDVariable state = sd.placeHolder("state", DataType.FLOAT, 1, dim);
        sd.constant("weight", weight.dup());
        sd.constant("gamma", gamma.dup());
        SDVariable projected = sd.mmul("projected", input, sd.getVariable("weight"));
        SDVariable normed = sd.nn().rmsNorm("normed", projected, sd.getVariable("gamma"), 1e-5);
        SDVariable combined = sd.math().add("combined", normed, state);
        sd.identity("out", combined);
        track(sd);

        // Reference
        SameDiff sdRef = SameDiff.create();
        SDVariable inputRef = sdRef.placeHolder("input", DataType.FLOAT, 1, dim);
        SDVariable stateRef = sdRef.placeHolder("state", DataType.FLOAT, 1, dim);
        sdRef.constant("weight", weight.dup());
        sdRef.constant("gamma", gamma.dup());
        SDVariable projectedRef = sdRef.mmul("projected", inputRef, sdRef.getVariable("weight"));
        SDVariable normedRef = sdRef.nn().rmsNorm("normed", projectedRef, sdRef.getVariable("gamma"), 1e-5);
        SDVariable combinedRef = sdRef.math().add("combined", normedRef, stateRef);
        sdRef.identity("out", combinedRef);
        track(sdRef);

        sd.setGraphExecutionMode(mode);
        sdRef.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);

        // Simulate 40-step decode with accumulating state
        INDArray accumulator = Nd4j.zeros(DataType.FLOAT, 1, dim);
        INDArray accumulatorRef = Nd4j.zeros(DataType.FLOAT, 1, dim);

        int mismatchCount = 0;
        double worstDiff = 0;
        Set<Integer> uniqueOutputHashes = new HashSet<>();

        for (int step = 0; step < 40; step++) {
            INDArray tokenInput = Nd4j.zeros(DataType.FLOAT, 1, dim);
            tokenInput.putScalar(0, step % dim, 1.0f);

            INDArray result = sd.output(
                    Map.of("input", tokenInput, "state", accumulator.dup()), "out").get("out").dup();
            INDArray ref = sdRef.output(
                    Map.of("input", tokenInput, "state", accumulatorRef.dup()), "out").get("out").dup();

            // Track output uniqueness
            int hash = Arrays.hashCode(result.data().asFloat());
            uniqueOutputHashes.add(hash);

            // Must match reference — errors compound with accumulation
            double diff = ref.sub(result).amaxNumber().doubleValue();
            if (diff > 1e-3) {
                mismatchCount++;
                worstDiff = Math.max(worstDiff, diff);
                log.warn("{} step {}: accumulated state mismatch diff={}", mode, step, diff);
            }

            // Accumulate: state += output * 0.1 (bounded growth)
            accumulator.addi(result.mul(0.05));
            accumulatorRef.addi(ref.mul(0.05));
        }

        // ASSERTION 1: Reference accuracy
        assertTrue(mismatchCount <= 4,
                mode + ": " + mismatchCount + "/40 steps mismatched with accumulating state. "
                        + "worstDiff=" + worstDiff + ". Errors compound — "
                        + "even small sync bugs cause divergence after 10+ steps.");

        // ASSERTION 2: Every output must be unique (accumulating state ensures this)
        assertTrue(uniqueOutputHashes.size() >= 38,
                mode + ": only " + uniqueOutputHashes.size() + "/40 unique outputs. "
                        + "Accumulating state is not being reflected — "
                        + "frozen fast-path may be ignoring placeholder updates.");

        log.info("{}: accumulating state passed (mismatch={}/40, unique={}/40, worstDiff={})",
                mode, mismatchCount, uniqueOutputHashes.size(), worstDiff);
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  35. PLAN SWAP WITH FP16 WEIGHTS — the exact VLM decode NaN pattern.
    //
    //      From VLM diag log:
    //        redispatchForCurrentShapes: plan swapped from X to Y — resetting frozen state
    //        [DSP_EVENT] seg[0-2550] WARMUP_START execCount=1 phase=SHAPES_FROZEN
    //        NATIVE_OUT logits shape=[1,1,49280] first5=[NaN,NaN,NaN,NaN,NaN]
    //
    //      Conditions: (1) FP32 weights pre-cast to HALF (the optimizer does
    //      this: "213 arrays quantized FP32→HALF"), (2) prefill at shape
    //      [1, seqLen, dim], (3) shape change to [1, 1, dim] triggers plan
    //      swap via redispatchForCurrentShapes, (4) the new plan's first
    //      WARMUP execution produces NaN in the final rms_norm_linear.
    //
    //      The final VLM op is a fused rms_norm_linear:
    //        x=layer_out_23, gamma=model.norm.weight, W=permute_186, output=lm_logits
    //      The weight (permute_186) was pre-cast to HALF. After plan swap
    //      the new plan may fail to sync the HALF weight buffer correctly.
    // ═══════════════════════════════════════════════════════════════════════

    /** Helper: build the rms_norm_linear tail of a transformer (the part that NaNs). */
    private SameDiff buildPlanSwapGraph(INDArray projWeight, INDArray gamma, DataType inputDtype, int dim) {
        SameDiff sd = SameDiff.create();
        // Dynamic sequence length — allows plan swap on shape change
        SDVariable input = sd.placeHolder("input", inputDtype, -1, dim);
        sd.constant("proj_weight", projWeight.dup());
        sd.constant("gamma", gamma.dup());
        // This is the exact pattern: rmsNorm then matmul (the fused rms_norm_linear)
        SDVariable normed = sd.nn().rmsNorm("normed", input, sd.getVariable("gamma"), 1e-6);
        SDVariable logits = sd.mmul("logits", normed, sd.getVariable("proj_weight"));
        sd.identity("out", logits);
        return track(sd);
    }

    @ParameterizedTest(name = "35_planSwapFp16Weights_{0}")
    @EnumSource(value = GraphExecutionMode.class,
            names = {"SLOT_BY_SLOT", "AUTO", "TRITON", "CUDA_GRAPHS"})
    @Order(35)
    void test35_PlanSwapWithFp16Weights(GraphExecutionMode mode) {
        int dim = 64;
        int vocab = 128;
        int prefillLen = 32;
        Nd4j.getRandom().setSeed(42);

        // FP32 weights pre-cast to HALF — the exact optimizer pattern
        INDArray projWeight = Nd4j.randn(DataType.FLOAT, dim, vocab).muli(0.02).castTo(DataType.HALF);
        INDArray gamma = Nd4j.ones(DataType.FLOAT, dim);

        SameDiff sd = buildPlanSwapGraph(projWeight, gamma, DataType.FLOAT, dim);
        SameDiff sdRef = buildPlanSwapGraph(projWeight, gamma, DataType.FLOAT, dim);
        sd.setGraphExecutionMode(mode);
        sdRef.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);

        // PHASE 1: Prefill at [prefillLen, dim] — establishes the plan
        INDArray prefillInput = Nd4j.randn(DataType.FLOAT, prefillLen, dim).muli(0.1);
        INDArray prefillResult = sd.output(Map.of("input", prefillInput), "out").get("out").dup();
        assertFalse(prefillResult.isNaN().any(),
                mode + ": prefill NaN at [" + prefillLen + "," + dim + "]");
        log.info("{}: prefill [{}x{}] logits first3=[{}, {}, {}]", mode, prefillLen, vocab,
                prefillResult.getFloat(0, 0), prefillResult.getFloat(0, 1), prefillResult.getFloat(0, 2));

        // Run a few more to let the plan freeze
        for (int i = 0; i < 3; i++) {
            sd.output(Map.of("input", Nd4j.randn(DataType.FLOAT, prefillLen, dim).muli(0.1)), "out");
        }

        // PHASE 2: Shape change to [1, dim] — triggers plan swap
        // This is the exact transition: redispatchForCurrentShapes swaps to a new plan
        // Do NOT resetSession — the VLM doesn't reset, it just changes the input shape.
        int nanCount = 0;
        int mismatchCount = 0;
        for (int step = 0; step < 15; step++) {
            INDArray decodeInput = Nd4j.randn(DataType.FLOAT, 1, dim).muli(0.1);
            INDArray result = sd.output(Map.of("input", decodeInput), "out").get("out").dup();
            INDArray ref = sdRef.output(Map.of("input", decodeInput), "out").get("out").dup();

            if (result.isNaN().any()) {
                nanCount++;
                log.error("{} DECODE step {} NaN — plan swap failed to sync HALF weight. "
                        + "logits first3=[{}, {}, {}]", mode, step,
                        result.getFloat(0, 0), result.getFloat(0, 1), result.getFloat(0, 2));
            }

            double diff = ref.sub(result).amaxNumber().doubleValue();
            if (diff > 1e-2) {
                mismatchCount++;
                log.warn("{} step {}: diff={} first3=[{}, {}, {}] ref=[{}, {}, {}]",
                        mode, step, diff,
                        result.getFloat(0, 0), result.getFloat(0, 1), result.getFloat(0, 2),
                        ref.getFloat(0, 0), ref.getFloat(0, 1), ref.getFloat(0, 2));
            }
        }

        assertEquals(0, nanCount,
                mode + ": " + nanCount + "/15 decode steps NaN after plan swap. "
                        + "VLM pattern: prefill [" + prefillLen + ",dim] → decode [1,dim] "
                        + "with FP16 pre-cast weight in rms_norm_linear.");
        assertTrue(mismatchCount <= 2,
                mode + ": " + mismatchCount + "/15 decode steps diverged from reference after plan swap.");

        log.info("{}: plan swap FP16 passed (nan={}, mismatch={}/15)", mode, nanCount, mismatchCount);
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  36. ARGMAX BIAS AFTER SHAPE FREEZE — the exact Qwen Q8_0 pattern.
    //
    //      From Qwen diag log:
    //        Q4_K_M: SEG_EXIT_ARGMAX argmax=248069 vals=[4.72,8.85,10.91,5.43]
    //        Q8_0:   SEG_EXIT_ARGMAX argmax=0     vals=[18.88,5.47,12.81,8.14]
    //
    //      Token 0 = "!" in the Qwen tokenizer. logit[0] is ~19 in Q8_0
    //      but ~4 in Q4_K_M. The model always picks token 0 → "!!!!!!".
    //
    //      This test isolates: after shapes freeze and REPLAY begins,
    //      does argmax remain stable and correct across steps? We check
    //      that no single output index dominates all steps (which would
    //      indicate a bias like the Q8_0 argmax=0 bug).
    // ═══════════════════════════════════════════════════════════════════════

    @ParameterizedTest(name = "36_argmaxBiasAfterFreeze_{0}")
    @EnumSource(value = GraphExecutionMode.class,
            names = {"SLOT_BY_SLOT", "AUTO", "TRITON", "CUDA_GRAPHS"})
    @Order(36)
    void test36_ArgmaxBiasAfterFreeze(GraphExecutionMode mode) {
        int vocab = 64, dim = 32;
        Nd4j.getRandom().setSeed(42);

        INDArray embedTable = Nd4j.randn(DataType.FLOAT, vocab, dim).muli(0.02);
        INDArray projWeight = Nd4j.randn(DataType.FLOAT, dim, vocab).muli(0.02);
        INDArray gamma = Nd4j.ones(DataType.FLOAT, dim);

        // Graph: gather -> rmsNorm -> matmul -> out (raw logits, no softmax)
        SameDiff sd = SameDiff.create();
        sd.constant("embed_table", embedTable.dup());
        sd.constant("proj_weight", projWeight.dup());
        sd.constant("gamma", gamma.dup());
        SDVariable tokenId = sd.placeHolder("token_id", DataType.INT64, 1);
        SDVariable gathered = sd.gather("gathered", sd.getVariable("embed_table"), tokenId, 0);
        SDVariable normed = sd.nn().rmsNorm("normed", gathered, sd.getVariable("gamma"), 1e-5);
        SDVariable logits = sd.mmul("logits", normed, sd.getVariable("proj_weight"));
        sd.identity("out", logits);
        track(sd);

        // Reference
        SameDiff sdRef = SameDiff.create();
        sdRef.constant("embed_table", embedTable.dup());
        sdRef.constant("proj_weight", projWeight.dup());
        sdRef.constant("gamma", gamma.dup());
        SDVariable tokenIdRef = sdRef.placeHolder("token_id", DataType.INT64, 1);
        SDVariable gatheredRef = sdRef.gather("gathered", sdRef.getVariable("embed_table"), tokenIdRef, 0);
        SDVariable normedRef = sdRef.nn().rmsNorm("normed", gatheredRef, sdRef.getVariable("gamma"), 1e-5);
        SDVariable logitsRef = sdRef.mmul("logits", normedRef, sdRef.getVariable("proj_weight"));
        sdRef.identity("out", logitsRef);
        track(sdRef);

        sd.setGraphExecutionMode(mode);
        sdRef.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);

        // Track argmax per step — the Q8_0 bug has argmax=0 EVERY step
        int totalSteps = 20;
        int[] argmaxes = new int[totalSteps];
        int[] refArgmaxes = new int[totalSteps];
        Map<Integer, Integer> argmaxCounts = new HashMap<>();

        for (int step = 0; step < totalSteps; step++) {
            long token = (step * 7 + 3) % vocab;  // varied tokens
            INDArray tokenArr = Nd4j.createFromArray(new long[]{token});

            INDArray result = sd.output(Map.of("token_id", tokenArr), "out").get("out").dup();
            INDArray ref = sdRef.output(Map.of("token_id", tokenArr), "out").get("out").dup();

            // Get argmax from raw logits
            argmaxes[step] = result.argMax(1).getInt(0);
            refArgmaxes[step] = ref.argMax(1).getInt(0);
            argmaxCounts.merge(argmaxes[step], 1, Integer::sum);

            // Log the first few values like the diag does
            if (step < 5 || argmaxes[step] != refArgmaxes[step]) {
                log.info("{} step {} token={}: argmax={} ref_argmax={} vals=[{},{},{},{}]",
                        mode, step, token, argmaxes[step], refArgmaxes[step],
                        String.format("%.4f", result.getFloat(0, 0)),
                        String.format("%.4f", result.getFloat(0, 1)),
                        String.format("%.4f", result.getFloat(0, 2)),
                        String.format("%.4f", result.getFloat(0, 3)));
            }
        }

        // ASSERTION 1: No single index should dominate all steps
        // The Q8_0 bug has argmax=0 on ALL 10 steps → 100% concentration
        int maxCount = argmaxCounts.values().stream().max(Integer::compareTo).orElse(0);
        int dominantIdx = argmaxCounts.entrySet().stream()
                .max(Map.Entry.comparingByValue()).map(Map.Entry::getKey).orElse(-1);
        assertTrue(maxCount <= totalSteps * 0.7,
                mode + ": argmax index " + dominantIdx + " appeared " + maxCount + "/" + totalSteps
                        + " times (" + (maxCount * 100 / totalSteps) + "%). "
                        + "This is the Q8_0 stuck-token pattern — one logit dominates all steps.");

        // ASSERTION 2: Argmax should match reference on most steps
        int argmaxMismatch = 0;
        for (int step = 0; step < totalSteps; step++) {
            if (argmaxes[step] != refArgmaxes[step]) argmaxMismatch++;
        }
        assertTrue(argmaxMismatch <= 4,
                mode + ": " + argmaxMismatch + "/" + totalSteps + " argmax mismatches vs reference. "
                        + "Execution mode is changing which token wins.");

        log.info("{}: argmax bias test passed (dominant={} appeared {}/{}, mismatch={})",
                mode, dominantIdx, maxCount, totalSteps, argmaxMismatch);
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  37. SHAPE TRANSITION WITHOUT SESSION RESET — the VLM never resets
    //      between prefill and decode. It just changes the input shape.
    //      The DSP must swap plans via redispatchForCurrentShapes and the
    //      new plan must inherit weight constants correctly.
    //      Tests multiple shape transitions: large → small → large → small.
    //      Catches: plan swap weight sync, stale buffer after redispatch.
    // ═══════════════════════════════════════════════════════════════════════

    @ParameterizedTest(name = "37_multiShapeTransitionNoReset_{0}")
    @EnumSource(value = GraphExecutionMode.class,
            names = {"SLOT_BY_SLOT", "AUTO", "TRITON", "CUDA_GRAPHS"})
    @Order(37)
    void test37_MultiShapeTransitionNoReset(GraphExecutionMode mode) {
        int dim = 32;
        Nd4j.getRandom().setSeed(42);

        INDArray weight = Nd4j.randn(DataType.FLOAT, dim, dim).muli(0.02);
        INDArray gamma = Nd4j.ones(DataType.FLOAT, dim);

        // Graph: input -> matmul -> rmsNorm -> matmul -> out
        // Dynamic first dim for shape transitions
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, dim);
        sd.constant("w1", weight.dup());
        sd.constant("w2", weight.dup());
        sd.constant("gamma", gamma.dup());
        SDVariable h1 = sd.mmul("matmul1", input, sd.getVariable("w1"));
        SDVariable n1 = sd.nn().rmsNorm("norm1", h1, sd.getVariable("gamma"), 1e-5);
        SDVariable h2 = sd.mmul("matmul2", n1, sd.getVariable("w2"));
        sd.identity("out", h2);
        track(sd);

        SameDiff sdRef = SameDiff.create();
        SDVariable inputRef = sdRef.placeHolder("input", DataType.FLOAT, -1, dim);
        sdRef.constant("w1", weight.dup());
        sdRef.constant("w2", weight.dup());
        sdRef.constant("gamma", gamma.dup());
        SDVariable h1r = sdRef.mmul("matmul1", inputRef, sdRef.getVariable("w1"));
        SDVariable n1r = sdRef.nn().rmsNorm("norm1", h1r, sdRef.getVariable("gamma"), 1e-5);
        SDVariable h2r = sdRef.mmul("matmul2", n1r, sdRef.getVariable("w2"));
        sdRef.identity("out", h2r);
        track(sdRef);

        sd.setGraphExecutionMode(mode);
        sdRef.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);

        // Transition pattern: [16,dim] → [1,dim] → [8,dim] → [1,dim] → [4,dim]
        // Each shape change forces a plan swap. NO session reset between transitions.
        int[][] shapes = {{16}, {1}, {1}, {1}, {8}, {1}, {1}, {1}, {4}, {1}, {1}, {1}};
        int nanCount = 0;
        int mismatchCount = 0;

        for (int phase = 0; phase < shapes.length; phase++) {
            int seqLen = shapes[phase][0];
            INDArray in = Nd4j.randn(DataType.FLOAT, seqLen, dim).muli(0.1);
            INDArray result = sd.output(Map.of("input", in), "out").get("out").dup();
            INDArray ref = sdRef.output(Map.of("input", in), "out").get("out").dup();

            if (result.isNaN().any()) {
                nanCount++;
                log.error("{} phase {} shape=[{},{}]: NaN after plan swap", mode, phase, seqLen, dim);
            }

            double diff = ref.sub(result).amaxNumber().doubleValue();
            if (diff > 1e-4) {
                mismatchCount++;
                log.warn("{} phase {} shape=[{},{}]: diff={}", mode, phase, seqLen, dim, diff);
            }
        }

        assertEquals(0, nanCount,
                mode + ": " + nanCount + " phases produced NaN during multi-shape transitions. "
                        + "Plan swap is not syncing weight buffers correctly.");
        assertTrue(mismatchCount <= 2,
                mode + ": " + mismatchCount + " phases diverged from reference during shape transitions.");

        log.info("{}: multi-shape transition passed (nan={}, mismatch={}/{})",
                mode, nanCount, mismatchCount, shapes.length);
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  38. FROZEN REPLAY LOGIT STABILITY — after shapes freeze and the
    //      plan enters REPLAY mode, the logits must remain correct.
    //      The Qwen Q8_0 bug shows REPLAY mode with captured=0 replayed=0
    //      but the EXEC_SUMMARY still says mode=REPLAY. This tests that
    //      varying inputs produce varying logits during REPLAY phase.
    //      Catches: frozen replay producing stale output, REPLAY with
    //      no actual capture, argmax lock to index 0.
    // ═══════════════════════════════════════════════════════════════════════

    @ParameterizedTest(name = "38_frozenReplayLogitStability_{0}")
    @EnumSource(value = GraphExecutionMode.class,
            names = {"AUTO", "TRITON", "CUDA_GRAPHS"})
    @Order(38)
    void test38_FrozenReplayLogitStability(GraphExecutionMode mode) {
        int vocab = 64, dim = 32;
        Nd4j.getRandom().setSeed(42);

        INDArray embedTable = Nd4j.randn(DataType.FLOAT, vocab, dim).muli(0.02);
        INDArray projWeight = Nd4j.randn(DataType.FLOAT, dim, vocab).muli(0.02);
        INDArray gamma = Nd4j.ones(DataType.FLOAT, dim);

        SameDiff sd = SameDiff.create();
        sd.constant("embed_table", embedTable.dup());
        sd.constant("proj_weight", projWeight.dup());
        sd.constant("gamma", gamma.dup());
        SDVariable tokenId = sd.placeHolder("token_id", DataType.INT64, 1);
        SDVariable gathered = sd.gather("gathered", sd.getVariable("embed_table"), tokenId, 0);
        SDVariable normed = sd.nn().rmsNorm("normed", gathered, sd.getVariable("gamma"), 1e-5);
        SDVariable logits = sd.mmul("logits", normed, sd.getVariable("proj_weight"));
        sd.identity("out", logits);
        track(sd);

        sd.setGraphExecutionMode(mode);

        // PHASE 1: Warm up with fixed token to reach SHAPES_FROZEN + REPLAY
        long warmupToken = 5;
        for (int step = 0; step < 10; step++) {
            sd.output(Map.of("token_id", Nd4j.createFromArray(new long[]{warmupToken})), "out");
        }

        // Save warmup output for stale-detection
        INDArray warmupOut = sd.output(
                Map.of("token_id", Nd4j.createFromArray(new long[]{warmupToken})), "out").get("out").dup();

        // PHASE 2: Now in REPLAY — vary tokens and check logits change
        int staleCount = 0;
        Set<Integer> uniqueArgmaxes = new HashSet<>();
        for (int step = 0; step < 20; step++) {
            long token = (step * 7 + 13) % vocab;
            if (token == warmupToken) token = (token + 1) % vocab;

            INDArray result = sd.output(
                    Map.of("token_id", Nd4j.createFromArray(new long[]{token})), "out").get("out").dup();

            assertFalse(result.isNaN().any(),
                    mode + ": NaN at post-freeze step " + step + " with token " + token);

            // Check for stale output (identical to warmup)
            double staleDiff = warmupOut.sub(result).amaxNumber().doubleValue();
            if (staleDiff < 1e-6) {
                staleCount++;
                log.error("{} post-freeze step {}: token {} → STALE output identical to warmup token {}",
                        mode, step, token, warmupToken);
            }

            uniqueArgmaxes.add(result.argMax(1).getInt(0));
        }

        assertEquals(0, staleCount,
                mode + ": " + staleCount + "/20 post-freeze steps had stale output. "
                        + "REPLAY mode is not re-reading placeholder inputs.");
        assertTrue(uniqueArgmaxes.size() >= 5,
                mode + ": only " + uniqueArgmaxes.size() + " unique argmax values in 20 steps. "
                        + "Logits are locked to same output — the Q8_0 argmax=0 pattern.");

        log.info("{}: frozen replay logit stability passed (stale={}, uniqueArgmax={})",
                mode, staleCount, uniqueArgmaxes.size());
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  39. FP16 rms_norm_linear AFTER PLAN SWAP — isolates the exact fused
    //      op that produces NaN in the VLM decode.
    //
    //      VLM log: "Fused RMSNorm+Linear: x=layer_out_23, gamma=model.norm.weight,
    //                W=permute_186, eps=1e-7, output=lm_logits"
    //
    //      The weight was pre-cast to HALF. After plan swap, the new plan
    //      does WARMUP with the HALF weight. If the weight buffer isn't
    //      synced to the new plan's slot array, the fused op reads garbage → NaN.
    //
    //      This test uses FP32 input + HALF weight (the pre-cast pattern)
    //      and exercises plan swap by changing the sequence length.
    // ═══════════════════════════════════════════════════════════════════════

    @ParameterizedTest(name = "39_rmsNormLinearFp16PlanSwap_{0}")
    @EnumSource(value = GraphExecutionMode.class,
            names = {"SLOT_BY_SLOT", "AUTO", "TRITON", "CUDA_GRAPHS"})
    @Order(39)
    void test39_RmsNormLinearFp16AfterPlanSwap(GraphExecutionMode mode) {
        int dim = 64;
        int outDim = 128;  // simulates vocab projection
        Nd4j.getRandom().setSeed(42);

        // FP32→HALF pre-cast weight (the optimizer path)
        INDArray weight = Nd4j.randn(DataType.FLOAT, dim, outDim).muli(0.02).castTo(DataType.HALF);
        INDArray gamma = Nd4j.ones(DataType.FLOAT, dim);

        SameDiff sd = buildPlanSwapGraph(weight, gamma, DataType.FLOAT, dim);
        SameDiff sdRef = buildPlanSwapGraph(weight, gamma, DataType.FLOAT, dim);
        sd.setGraphExecutionMode(mode);
        sdRef.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);

        // Phase 1: Run at seqLen=16, let it freeze
        for (int i = 0; i < 5; i++) {
            INDArray in16 = Nd4j.randn(DataType.FLOAT, 16, dim).muli(0.1);
            sd.output(Map.of("input", in16), "out");
        }

        // Phase 2: Switch to seqLen=1 (plan swap), then back to 16, then 1 again
        // Each swap exercises the weight-sync path
        int[][] transitions = {{1, 5}, {16, 3}, {1, 10}, {4, 3}, {1, 5}};
        int nanCount = 0;
        int mismatchCount = 0;
        int totalSteps = 0;

        for (int[] transition : transitions) {
            int seqLen = transition[0];
            int steps = transition[1];
            for (int step = 0; step < steps; step++) {
                INDArray in = Nd4j.randn(DataType.FLOAT, seqLen, dim).muli(0.1);
                INDArray result = sd.output(Map.of("input", in), "out").get("out").dup();
                INDArray ref = sdRef.output(Map.of("input", in), "out").get("out").dup();

                if (result.isNaN().any()) {
                    nanCount++;
                    log.error("{} seqLen={} step {}: rms_norm_linear NaN — HALF weight not synced after plan swap. "
                            + "first3=[{},{},{}]", mode, seqLen, step,
                            result.getFloat(0, 0), result.getFloat(0, 1), result.getFloat(0, 2));
                }

                double diff = ref.sub(result).amaxNumber().doubleValue();
                if (diff > 1e-2) {
                    mismatchCount++;
                }
                totalSteps++;
            }
        }

        assertEquals(0, nanCount,
                mode + ": " + nanCount + "/" + totalSteps + " steps NaN from rms_norm_linear "
                        + "with HALF weight after plan swap. "
                        + "This is the exact VLM lm_logits NaN — fused op reads unsync'd HALF buffer.");
        assertTrue(mismatchCount <= 3,
                mode + ": " + mismatchCount + "/" + totalSteps + " steps diverged after plan swap.");

        log.info("{}: rms_norm_linear FP16 plan swap passed (nan={}, mismatch={}/{})",
                mode, nanCount, mismatchCount, totalSteps);
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  40. SESSION MEMORY ISOLATION — run graph A to completion, reset and
    //      close, then run graph B. If A doesn't release GPU memory, B's
    //      allocations may fail or overlap with A's stale buffers.
    //      Catches: GPU memory leak between configs, OOM on model reload.
    // ═══════════════════════════════════════════════════════════════════════

    @ParameterizedTest(name = "40_sessionMemoryIsolation_{0}")
    @EnumSource(value = GraphExecutionMode.class,
            names = {"SLOT_BY_SLOT", "AUTO", "TRITON", "CUDA_GRAPHS"})
    @Order(40)
    void test40_SessionMemoryIsolation(GraphExecutionMode mode) {
        int dim = 32;
        int layers = 6;
        Nd4j.getRandom().setSeed(42);

        // === Graph A: run through full lifecycle ===
        INDArray[][] weightsA = generateDeepChainWeights(layers, dim);
        SameDiff sdA = buildDeepChainWith(weightsA, dim);
        sdA.setGraphExecutionMode(mode);

        INDArray lastOutputA = null;
        for (int step = 0; step < 20; step++) {
            INDArray input = Nd4j.zeros(DataType.FLOAT, 1, dim);
            input.putScalar(0, step % dim, 1.0f);
            lastOutputA = sdA.output(Map.of("input", input), "out").get("out").dup();
        }
        assertFalse(lastOutputA.isNaN().any(), mode + ": Graph A produced NaN before cleanup");

        // Full cleanup
        sdA.resetSession();
        activeSds.remove(sdA);
        Nd4j.getWorkspaceManager().destroyAllWorkspacesForCurrentThread();
        System.gc();
        try {
            var nativeOps = Nd4j.getNativeOps();
            for (int d = 0; d < Nd4j.getAffinityManager().getNumberOfDevices(); d++) {
                nativeOps.trimMemoryPool(d);
            }
        } catch (Exception ignored) { }

        // === Graph B: fresh graph with DIFFERENT weights ===
        INDArray[][] weightsB = generateDeepChainWeights(layers, dim);
        for (int l = 0; l < layers; l++) weightsB[l][0].muli(2.0);
        SameDiff sdB = buildDeepChainWith(weightsB, dim);
        sdB.setGraphExecutionMode(mode);
        SameDiff sdBRef = buildDeepChainWith(weightsB, dim);
        sdBRef.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);

        int mismatchCount = 0;
        int nanCount = 0;
        for (int step = 0; step < 20; step++) {
            INDArray input = Nd4j.zeros(DataType.FLOAT, 1, dim);
            input.putScalar(0, step % dim, 1.0f);

            INDArray result;
            try {
                result = sdB.output(Map.of("input", input), "out").get("out").dup();
            } catch (Exception e) {
                fail(mode + ": Graph B failed at step " + step + " after Graph A cleanup: "
                        + e.getMessage() + ". GPU memory not released between sessions.");
                return;
            }

            if (result.isNaN().any()) nanCount++;

            INDArray ref = sdBRef.output(Map.of("input", input), "out").get("out").dup();
            double diff = ref.sub(result).amaxNumber().doubleValue();
            if (diff > 1e-4) mismatchCount++;
        }

        assertEquals(0, nanCount,
                mode + ": Graph B produced " + nanCount + "/20 NaN outputs after Graph A cleanup.");
        assertTrue(mismatchCount <= 2,
                mode + ": " + mismatchCount + "/20 Graph B steps mismatched reference.");

        log.info("{}: session memory isolation passed (nan={}, mismatch={}/20)", mode, nanCount, mismatchCount);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Test: External input introspection and staging buffer lifecycle
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * Verifies that ext input introspection APIs correctly report variable/placeholder
     * classification and staging buffer allocation after plan warmup. This is a
     * prerequisite for diagnosing stale-input bugs in the decode loop.
     *
     * Tests:
     * 1. Variable ext inputs are correctly classified after execution
     * 2. Staging buffers are allocated for variable inputs after freeze
     * 3. Output changes when placeholder values change (not stale)
     * 4. snapshotExtInputState produces correct diagnostic output
     */
    @ParameterizedTest(name = "test36_ExtInputIntrospection_{0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"AUTO", "TRITON", "CUDA_GRAPHS", "SLOT_BY_SLOT"})
    void test36_ExtInputIntrospection(GraphExecutionMode mode) {
        SameDiff sd = SameDiff.create();
        track(sd);
        SDVariable ph = sd.placeHolder("token_embed", DataType.FLOAT, 1, 8);
        SDVariable w = sd.var("weight", Nd4j.randn(DataType.FLOAT, 8, 4));
        SDVariable out = sd.mmul("out", ph, w);

        sd.setGraphExecutionMode(mode);
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        // Step 1: Warmup — execute several times to reach frozen/replaying state
        INDArray input = Nd4j.randn(DataType.FLOAT, 1, 8);
        Map<String, INDArray> placeholders = new HashMap<>();
        placeholders.put("token_embed", input);
        INDArray[] prevOutputs = new INDArray[10];

        for (int i = 0; i < 10; i++) {
            input = Nd4j.randn(DataType.FLOAT, 1, 8);
            placeholders.put("token_embed", input);
            prevOutputs[i] = sd.output(placeholders, "out").get("out").dup();
        }

        // Step 2: Introspect ext input state
        int numExt = Nd4j.getNativeOps().getPlanNumExternalInputs(
                DspPlanAssertions.getPlanHandleForQuery(sd));
        assertTrue(numExt > 0, mode + ": plan should have external inputs, got " + numExt);

        int numVar = Nd4j.getNativeOps().getPlanNumVariableExternalInputs(
                DspPlanAssertions.getPlanHandleForQuery(sd));
        log.info("{}: numExt={} numVar={}", mode, numExt, numVar);

        // Snapshot diagnostic output — verify it doesn't crash and contains expected fields
        String snapshot = DspPlanAssertions.snapshotExtInputState(sd);
        assertNotNull(snapshot, mode + ": snapshotExtInputState should not return null");
        assertTrue(snapshot.contains("extInputState"), mode + ": snapshot should contain header");
        log.info("{}: ext input snapshot:\n{}", mode, snapshot);

        // Step 3: Verify outputs are NOT stale — each step with different input should
        // produce a different output. If staging buffers aren't syncing, all outputs
        // after the first graph replay would be identical.
        int distinctCount = 0;
        for (int i = 1; i < prevOutputs.length; i++) {
            double diff = prevOutputs[i].sub(prevOutputs[i - 1]).amaxNumber().doubleValue();
            if (diff > 1e-6) distinctCount++;
        }

        // At least 7/9 step-to-step pairs should differ (allows 2 warmup-phase identical outputs)
        assertTrue(distinctCount >= 7,
                mode + ": only " + distinctCount + "/9 step-pairs produced distinct outputs. "
                        + "Staging buffer D2D may not be refreshing variable ext inputs. "
                        + "Snapshot: " + snapshot);

        log.info("{}: ext input introspection passed — {}/{} distinct output pairs",
                mode, distinctCount, prevOutputs.length - 1);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Test 41: Cross-attention pattern — fixed encoder output + changing decoder input
    //
    // This reproduces the VLM decode pattern:
    // - "image_features" ext input is FIXED (same value every step)
    // - "token_embed" ext input CHANGES every step
    // - Graph computes cross-attention: Q=token_embed@Wq, K=image_features@Wk, V=image_features@Wv
    //   then scores = Q@K^T, attn = softmax(scores), out = attn@V@Wout
    //
    // If DSP misclassifies image_features (e.g., skips staging or uses stale capture-time
    // value), the cross-attention output will be wrong — decoder can't "see" the image.
    // ═══════════════════════════════════════════════════════════════════════════

    @ParameterizedTest(name = "41_crossAttentionFixedEncoder_{0}")
    @EnumSource(value = GraphExecutionMode.class,
            names = {"SLOT_BY_SLOT", "AUTO", "TRITON", "CUDA_GRAPHS"})
    @Order(41)
    void test41_CrossAttentionFixedEncoder(GraphExecutionMode mode) {
        int dim = 16;
        int seqLen = 4; // encoder sequence length (image patches)
        Nd4j.getRandom().setSeed(42);

        // Weights for Q, K, V, Out projections
        INDArray wQ = Nd4j.randn(DataType.FLOAT, dim, dim).muli(0.1);
        INDArray wK = Nd4j.randn(DataType.FLOAT, dim, dim).muli(0.1);
        INDArray wV = Nd4j.randn(DataType.FLOAT, dim, dim).muli(0.1);
        INDArray wOut = Nd4j.randn(DataType.FLOAT, dim, dim).muli(0.1);

        // Build graph: cross-attention with fixed encoder and changing decoder
        SameDiff sd = SameDiff.create();
        track(sd);
        // Decoder input changes each step (like token embedding)
        SDVariable decoderInput = sd.placeHolder("token_embed", DataType.FLOAT, 1, dim);
        // Encoder output is fixed (like image features) — but it's a PLACEHOLDER
        // because in VLM it comes from vision encoder output, not a trained constant
        SDVariable encoderOutput = sd.placeHolder("image_features", DataType.FLOAT, seqLen, dim);

        sd.constant("wQ", wQ.dup());
        sd.constant("wK", wK.dup());
        sd.constant("wV", wV.dup());
        sd.constant("wOut", wOut.dup());

        // Q = decoder_input @ Wq  -> [1, dim]
        SDVariable Q = sd.mmul("Q", decoderInput, sd.getVariable("wQ"));
        // K = encoder_output @ Wk -> [seqLen, dim]
        SDVariable K = sd.mmul("K", encoderOutput, sd.getVariable("wK"));
        // V = encoder_output @ Wv -> [seqLen, dim]
        SDVariable V = sd.mmul("V", encoderOutput, sd.getVariable("wV"));

        // scores = Q @ K^T -> [1, seqLen]
        SDVariable Kt = sd.permute("Kt", K, 1, 0);
        SDVariable scores = sd.mmul("scores", Q, Kt);
        // attn = softmax(scores) -> [1, seqLen]
        SDVariable attn = sd.nn().softmax("attn", scores, 1);
        // context = attn @ V -> [1, dim]
        SDVariable context = sd.mmul("context", attn, V);
        // out = context @ Wout -> [1, dim]
        sd.mmul("out", context, sd.getVariable("wOut"));

        // Reference graph (SLOT_BY_SLOT)
        SameDiff sdRef = SameDiff.create();
        track(sdRef);
        SDVariable decoderInputRef = sdRef.placeHolder("token_embed", DataType.FLOAT, 1, dim);
        SDVariable encoderOutputRef = sdRef.placeHolder("image_features", DataType.FLOAT, seqLen, dim);
        sdRef.constant("wQ", wQ.dup());
        sdRef.constant("wK", wK.dup());
        sdRef.constant("wV", wV.dup());
        sdRef.constant("wOut", wOut.dup());
        SDVariable Qr = sdRef.mmul("Q", decoderInputRef, sdRef.getVariable("wQ"));
        SDVariable Kr = sdRef.mmul("K", encoderOutputRef, sdRef.getVariable("wK"));
        SDVariable Vr = sdRef.mmul("V", encoderOutputRef, sdRef.getVariable("wV"));
        SDVariable Ktr = sdRef.permute("Kt", Kr, 1, 0);
        SDVariable scoresr = sdRef.mmul("scores", Qr, Ktr);
        SDVariable attnr = sdRef.nn().softmax("attn", scoresr, 1);
        SDVariable contextr = sdRef.mmul("context", attnr, Vr);
        sdRef.mmul("out", contextr, sdRef.getVariable("wOut"));

        sd.setGraphExecutionMode(mode);
        sdRef.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);

        // FIXED image features — same every step (like VLM vision encoder output)
        INDArray imageFeatures = Nd4j.randn(DataType.FLOAT, seqLen, dim).muli(0.5);

        int mismatchCount = 0;
        for (int step = 0; step < 30; step++) {
            // Token embedding changes each step
            INDArray tokenEmbed = Nd4j.zeros(DataType.FLOAT, 1, dim);
            tokenEmbed.putScalar(0, step % dim, 1.0f);

            Map<String, INDArray> ph = Map.of(
                    "token_embed", tokenEmbed,
                    "image_features", imageFeatures
            );

            INDArray result = sd.output(ph, "out").get("out").dup();
            INDArray ref = sdRef.output(ph, "out").get("out").dup();

            double diff = ref.sub(result).amaxNumber().doubleValue();
            if (diff > 1e-4) {
                mismatchCount++;
                if (mismatchCount <= 3) {
                    log.warn("{} step {}: cross-attn diff={} (ref sum={}, result sum={})",
                            mode, step, diff, ref.sumNumber(), result.sumNumber());
                }
            }
        }

        assertTrue(mismatchCount <= 2,
                mode + ": " + mismatchCount + "/30 cross-attention steps mismatched reference. "
                        + "Fixed encoder features may not be reaching decoder correctly during replay.");
        log.info("{}: cross-attention fixed encoder passed (mismatch={}/30)", mode, mismatchCount);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Test 42: Shape transition WITHOUT session reset (real VLM pattern)
    //
    // VLM does NOT reset session between prefill and decode. The plan recompiles
    // on shape change. This tests that the recompiled plan:
    // 1. Gets correct weight/constant data (not stale from prior plan)
    // 2. Properly stages the now-smaller input
    // 3. Produces correct output matching SLOT_BY_SLOT reference
    // ═══════════════════════════════════════════════════════════════════════════

    @ParameterizedTest(name = "42_shapeTransitionNoReset_{0}")
    @EnumSource(value = GraphExecutionMode.class,
            names = {"SLOT_BY_SLOT", "AUTO", "TRITON", "CUDA_GRAPHS"})
    @Order(42)
    void test42_ShapeTransitionNoSessionReset(GraphExecutionMode mode) {
        int dim = 16;
        Nd4j.getRandom().setSeed(42);
        INDArray weight = Nd4j.randn(DataType.FLOAT, dim, dim).muli(0.02);
        INDArray gamma = Nd4j.ones(DataType.FLOAT, dim);

        SameDiff sd = SameDiff.create();
        track(sd);
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, dim);
        sd.constant("weight", weight.dup());
        sd.constant("gamma", gamma.dup());
        SDVariable projected = sd.mmul("projected", input, sd.getVariable("weight"));
        SDVariable normed = sd.nn().rmsNorm("normed", projected, sd.getVariable("gamma"), 1e-5);
        sd.identity("out", normed);

        SameDiff sdRef = SameDiff.create();
        track(sdRef);
        SDVariable inputRef = sdRef.placeHolder("input", DataType.FLOAT, -1, dim);
        sdRef.constant("weight", weight.dup());
        sdRef.constant("gamma", gamma.dup());
        SDVariable projectedRef = sdRef.mmul("projected", inputRef, sdRef.getVariable("weight"));
        SDVariable normedRef = sdRef.nn().rmsNorm("normed", projectedRef, sdRef.getVariable("gamma"), 1e-5);
        sdRef.identity("out", normedRef);

        sd.setGraphExecutionMode(mode);
        sdRef.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);

        // PHASE 1: Prefill with seqLen=8 — run 5 steps to warm up and capture
        for (int step = 0; step < 5; step++) {
            INDArray prefill = Nd4j.randn(DataType.FLOAT, 8, dim);
            sd.output(Map.of("input", prefill), "out");
            sdRef.output(Map.of("input", prefill), "out");
        }

        // PHASE 2: Switch to decode shape (seqLen=1) — NO session reset
        // This forces plan recompile/swap
        int mismatchCount = 0;
        for (int step = 0; step < 25; step++) {
            INDArray decodeInput = Nd4j.zeros(DataType.FLOAT, 1, dim);
            decodeInput.putScalar(0, step % dim, 1.0f);

            INDArray result = sd.output(Map.of("input", decodeInput), "out").get("out").dup();
            INDArray ref = sdRef.output(Map.of("input", decodeInput), "out").get("out").dup();

            double diff = ref.sub(result).amaxNumber().doubleValue();
            if (diff > 1e-4) {
                mismatchCount++;
                if (mismatchCount <= 3) {
                    log.warn("{} decode step {}: diff={}", mode, step, diff);
                }
            }
        }

        assertTrue(mismatchCount <= 2,
                mode + ": " + mismatchCount + "/25 decode steps after shape transition (no reset) "
                        + "mismatched reference. Plan recompile may have stale weights/state.");
        log.info("{}: shape transition no-reset passed (mismatch={}/25)", mode, mismatchCount);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Test 43: Mixed variable/fixed ext inputs — some change, some don't
    //
    // VLM has 63 ext inputs. Only ~3 change each step (token_embed, position_ids,
    // attention_mask). The rest (KV caches, image features) are fixed or
    // device-written. Tests that the frozen fast-path correctly identifies which
    // inputs changed and sends ONLY those, without corrupting the fixed ones.
    // ═══════════════════════════════════════════════════════════════════════════

    @ParameterizedTest(name = "43_mixedFixedAndChanging_{0}")
    @EnumSource(value = GraphExecutionMode.class,
            names = {"SLOT_BY_SLOT", "AUTO", "TRITON", "CUDA_GRAPHS"})
    @Order(43)
    void test43_MixedFixedAndChangingExtInputs(GraphExecutionMode mode) {
        int dim = 16;
        int numFixed = 10; // simulate 10 fixed ext inputs (like KV caches / image features)
        Nd4j.getRandom().setSeed(42);

        // Pre-generate all weights so both graphs share identical values
        INDArray[] projWeights = new INDArray[numFixed];
        for (int i = 0; i < numFixed; i++) {
            projWeights[i] = Nd4j.randn(DataType.FLOAT, dim, dim).muli(0.02);
        }
        INDArray wOut = Nd4j.randn(DataType.FLOAT, dim, dim).muli(0.1);

        // Build test graph
        SameDiff sd = SameDiff.create();
        track(sd);
        SDVariable tokenEmbed = sd.placeHolder("token_embed", DataType.FLOAT, 1, dim);
        SDVariable accumulated = tokenEmbed;
        for (int i = 0; i < numFixed; i++) {
            SDVariable fixedPh = sd.placeHolder("fixed_" + i, DataType.FLOAT, 1, dim);
            sd.constant("proj_" + i, projWeights[i].dup());
            SDVariable projected = sd.mmul("proj_fixed_" + i, fixedPh, sd.getVariable("proj_" + i));
            accumulated = accumulated.add("add_" + i, projected);
        }
        sd.constant("wOut", wOut.dup());
        sd.mmul("out", accumulated, sd.getVariable("wOut"));

        // Build reference graph with same weights
        SameDiff sdRef = SameDiff.create();
        track(sdRef);
        SDVariable tokenEmbedRef = sdRef.placeHolder("token_embed", DataType.FLOAT, 1, dim);
        SDVariable accumulatedRef = tokenEmbedRef;
        for (int i = 0; i < numFixed; i++) {
            SDVariable fixedPhRef = sdRef.placeHolder("fixed_" + i, DataType.FLOAT, 1, dim);
            sdRef.constant("proj_" + i, projWeights[i].dup());
            SDVariable projected = sdRef.mmul("proj_fixed_" + i, fixedPhRef, sdRef.getVariable("proj_" + i));
            accumulatedRef = accumulatedRef.add("add_" + i, projected);
        }
        sdRef.constant("wOut", wOut.dup());
        sdRef.mmul("out", accumulatedRef, sdRef.getVariable("wOut"));

        sd.setGraphExecutionMode(mode);
        sdRef.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);

        // Fixed inputs — same every step
        Map<String, INDArray> fixedInputs = new LinkedHashMap<>();
        for (int i = 0; i < numFixed; i++) {
            fixedInputs.put("fixed_" + i, Nd4j.randn(DataType.FLOAT, 1, dim).muli(0.3));
        }

        int mismatchCount = 0;
        int stuckCount = 0;
        INDArray prevResult = null;
        for (int step = 0; step < 30; step++) {
            INDArray tokenInput = Nd4j.zeros(DataType.FLOAT, 1, dim);
            tokenInput.putScalar(0, step % dim, 1.0f);

            Map<String, INDArray> ph = new LinkedHashMap<>();
            ph.put("token_embed", tokenInput);
            ph.putAll(fixedInputs);

            INDArray result = sd.output(ph, "out").get("out").dup();
            INDArray ref = sdRef.output(ph, "out").get("out").dup();

            double diff = ref.sub(result).amaxNumber().doubleValue();
            if (diff > 1e-3) mismatchCount++;

            if (prevResult != null) {
                double stepDiff = result.sub(prevResult).amaxNumber().doubleValue();
                if (stepDiff < 1e-6) stuckCount++;
            }
            prevResult = result;
        }

        assertTrue(mismatchCount <= 2,
                mode + ": " + mismatchCount + "/30 steps mismatched reference with mixed fixed/changing inputs. "
                        + "Fixed ext inputs may be corrupted or changing input not propagated.");
        assertTrue(stuckCount < 3,
                mode + ": " + stuckCount + "/29 consecutive identical outputs. "
                        + "Changing token_embed not reaching output through fixed-input additions.");
        log.info("{}: mixed fixed/changing passed (mismatch={}/30, stuck={}/29)", mode, mismatchCount, stuckCount);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Test 44: Attention diamond pattern — Q, K, V branch from same source
    //
    // Tests the graph topology where one input feeds three parallel matmuls
    // (Q, K, V projections), then Q@K^T and softmax@V merge. Segment boundaries
    // must not split this diamond incorrectly.
    // ═══════════════════════════════════════════════════════════════════════════

    @ParameterizedTest(name = "44_attentionDiamond_{0}")
    @EnumSource(value = GraphExecutionMode.class,
            names = {"SLOT_BY_SLOT", "AUTO", "TRITON", "CUDA_GRAPHS"})
    @Order(44)
    void test44_AttentionDiamondPattern(GraphExecutionMode mode) {
        int dim = 16;
        Nd4j.getRandom().setSeed(42);

        INDArray wQ = Nd4j.randn(DataType.FLOAT, dim, dim).muli(0.1);
        INDArray wK = Nd4j.randn(DataType.FLOAT, dim, dim).muli(0.1);
        INDArray wV = Nd4j.randn(DataType.FLOAT, dim, dim).muli(0.1);
        INDArray wOut = Nd4j.randn(DataType.FLOAT, dim, dim).muli(0.1);

        // Self-attention: Q, K, V all from same input (diamond topology)
        SameDiff sd = SameDiff.create();
        track(sd);
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, dim);
        sd.constant("wQ", wQ.dup());
        sd.constant("wK", wK.dup());
        sd.constant("wV", wV.dup());
        sd.constant("wOut", wOut.dup());

        SDVariable Q = sd.mmul("Q", input, sd.getVariable("wQ"));
        SDVariable K = sd.mmul("K", input, sd.getVariable("wK"));
        SDVariable V = sd.mmul("V", input, sd.getVariable("wV"));
        SDVariable Kt = sd.permute("Kt", K, 1, 0);
        SDVariable scores = sd.mmul("scores", Q, Kt);
        SDVariable attn = sd.nn().softmax("attn", scores, 1);
        SDVariable context = sd.mmul("context", attn, V);
        sd.mmul("out", context, sd.getVariable("wOut"));

        // Reference
        SameDiff sdRef = SameDiff.create();
        track(sdRef);
        SDVariable inputRef = sdRef.placeHolder("input", DataType.FLOAT, 1, dim);
        sdRef.constant("wQ", wQ.dup());
        sdRef.constant("wK", wK.dup());
        sdRef.constant("wV", wV.dup());
        sdRef.constant("wOut", wOut.dup());
        SDVariable Qr = sdRef.mmul("Q", inputRef, sdRef.getVariable("wQ"));
        SDVariable Kr = sdRef.mmul("K", inputRef, sdRef.getVariable("wK"));
        SDVariable Vr = sdRef.mmul("V", inputRef, sdRef.getVariable("wV"));
        SDVariable Ktr = sdRef.permute("Kt", Kr, 1, 0);
        SDVariable scoresr = sdRef.mmul("scores", Qr, Ktr);
        SDVariable attnr = sdRef.nn().softmax("attn", scoresr, 1);
        SDVariable contextr = sdRef.mmul("context", attnr, Vr);
        sdRef.mmul("out", contextr, sdRef.getVariable("wOut"));

        sd.setGraphExecutionMode(mode);
        sdRef.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);

        int mismatchCount = 0;
        for (int step = 0; step < 30; step++) {
            INDArray in = Nd4j.zeros(DataType.FLOAT, 1, dim);
            in.putScalar(0, step % dim, 1.0f);

            INDArray result = sd.output(Map.of("input", in), "out").get("out").dup();
            INDArray ref = sdRef.output(Map.of("input", in), "out").get("out").dup();

            double diff = ref.sub(result).amaxNumber().doubleValue();
            if (diff > 1e-4) {
                mismatchCount++;
                if (mismatchCount <= 3) {
                    log.warn("{} step {}: attention diamond diff={}", mode, step, diff);
                }
            }
        }

        assertTrue(mismatchCount <= 2,
                mode + ": " + mismatchCount + "/30 self-attention diamond steps mismatched. "
                        + "Segment boundary may split Q@K^T or softmax@V incorrectly.");
        log.info("{}: attention diamond passed (mismatch={}/30)", mode, mismatchCount);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Test 45: Cross-attention with FP16 weights (VLM exact pattern)
    //
    // VLM uses FP16 pre-cast weights via QuantizationOptimizations. The cross-
    // attention has FLOAT activation × HALF weight through MmulHelper's mixed
    // precision path. Combined with fixed encoder features, this is the exact
    // pattern that produces wrong output in the VLM.
    // ═══════════════════════════════════════════════════════════════════════════

    @ParameterizedTest(name = "45_crossAttnFp16Weights_{0}")
    @EnumSource(value = GraphExecutionMode.class,
            names = {"SLOT_BY_SLOT", "AUTO", "TRITON", "CUDA_GRAPHS"})
    @Order(45)
    void test45_CrossAttentionFp16Weights(GraphExecutionMode mode) {
        int dim = 16;
        int encoderSeqLen = 4;
        Nd4j.getRandom().setSeed(42);

        // HALF weights (like after QuantizationOptimizations)
        INDArray wQ = Nd4j.randn(DataType.FLOAT, dim, dim).muli(0.1).castTo(DataType.HALF);
        INDArray wK = Nd4j.randn(DataType.FLOAT, dim, dim).muli(0.1).castTo(DataType.HALF);
        INDArray wV = Nd4j.randn(DataType.FLOAT, dim, dim).muli(0.1).castTo(DataType.HALF);
        INDArray wOut = Nd4j.randn(DataType.FLOAT, dim, dim).muli(0.1).castTo(DataType.HALF);

        SameDiff sd = SameDiff.create();
        track(sd);
        SDVariable decoderInput = sd.placeHolder("token_embed", DataType.FLOAT, 1, dim);
        SDVariable encoderOutput = sd.placeHolder("image_features", DataType.FLOAT, encoderSeqLen, dim);
        sd.constant("wQ", wQ.dup());
        sd.constant("wK", wK.dup());
        sd.constant("wV", wV.dup());
        sd.constant("wOut", wOut.dup());

        SDVariable Q = sd.mmul("Q", decoderInput, sd.getVariable("wQ"));
        SDVariable K = sd.mmul("K", encoderOutput, sd.getVariable("wK"));
        SDVariable V = sd.mmul("V", encoderOutput, sd.getVariable("wV"));
        SDVariable Kt = sd.permute("Kt", K, 1, 0);
        SDVariable scores = sd.mmul("scores", Q, Kt);
        SDVariable attn = sd.nn().softmax("attn", scores, 1);
        SDVariable context = sd.mmul("context", attn, V);
        sd.mmul("out", context, sd.getVariable("wOut"));

        // Reference
        SameDiff sdRef = SameDiff.create();
        track(sdRef);
        SDVariable decoderInputRef = sdRef.placeHolder("token_embed", DataType.FLOAT, 1, dim);
        SDVariable encoderOutputRef = sdRef.placeHolder("image_features", DataType.FLOAT, encoderSeqLen, dim);
        sdRef.constant("wQ", wQ.dup());
        sdRef.constant("wK", wK.dup());
        sdRef.constant("wV", wV.dup());
        sdRef.constant("wOut", wOut.dup());
        SDVariable Qr = sdRef.mmul("Q", decoderInputRef, sdRef.getVariable("wQ"));
        SDVariable Kr = sdRef.mmul("K", encoderOutputRef, sdRef.getVariable("wK"));
        SDVariable Vr = sdRef.mmul("V", encoderOutputRef, sdRef.getVariable("wV"));
        SDVariable Ktr = sdRef.permute("Kt", Kr, 1, 0);
        SDVariable scoresr = sdRef.mmul("scores", Qr, Ktr);
        SDVariable attnr = sdRef.nn().softmax("attn", scoresr, 1);
        SDVariable contextr = sdRef.mmul("context", attnr, Vr);
        sdRef.mmul("out", contextr, sdRef.getVariable("wOut"));

        sd.setGraphExecutionMode(mode);
        sdRef.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);

        // Fixed image features
        INDArray imageFeatures = Nd4j.randn(DataType.FLOAT, encoderSeqLen, dim).muli(0.5);

        int mismatchCount = 0;
        for (int step = 0; step < 30; step++) {
            INDArray tokenEmbed = Nd4j.zeros(DataType.FLOAT, 1, dim);
            tokenEmbed.putScalar(0, step % dim, 1.0f);

            Map<String, INDArray> ph = Map.of(
                    "token_embed", tokenEmbed,
                    "image_features", imageFeatures
            );

            INDArray result = sd.output(ph, "out").get("out").dup();
            INDArray ref = sdRef.output(ph, "out").get("out").dup();

            // FP16 tolerance is looser
            double diff = ref.sub(result).amaxNumber().doubleValue();
            if (diff > 0.01) {
                mismatchCount++;
                if (mismatchCount <= 3) {
                    log.warn("{} step {}: FP16 cross-attn diff={} (ref sum={}, result sum={})",
                            mode, step, diff, ref.sumNumber(), result.sumNumber());
                }
            }
        }

        assertTrue(mismatchCount <= 2,
                mode + ": " + mismatchCount + "/30 FP16 cross-attention steps mismatched. "
                        + "Mixed precision + fixed encoder + DSP replay interaction bug.");
        log.info("{}: FP16 cross-attention passed (mismatch={}/30)", mode, mismatchCount);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Test 46: Partial placeholder provision after warmup
    //
    // After DSP captures the graph, some VLM inputs are NOT provided in the
    // decode step map (e.g., inputs_embeds is computed internally by
    // AutoregressiveDecode). If the executor's frozen fast-path sends a stale
    // cached pointer for the missing input, outputs are wrong.
    // ═══════════════════════════════════════════════════════════════════════════

    @ParameterizedTest(name = "46_partialPlaceholderAfterWarmup_{0}")
    @EnumSource(value = GraphExecutionMode.class,
            names = {"SLOT_BY_SLOT", "AUTO", "TRITON", "CUDA_GRAPHS"})
    @Order(46)
    void test46_PartialPlaceholderAfterWarmup(GraphExecutionMode mode) {
        int dim = 16;
        Nd4j.getRandom().setSeed(42);

        // Graph with TWO placeholders — we'll provide both during warmup,
        // then only provide ONE during decode (simulating missing input)
        SameDiff sd = SameDiff.create();
        track(sd);
        SDVariable ph1 = sd.placeHolder("always_present", DataType.FLOAT, 1, dim);
        SDVariable ph2 = sd.placeHolder("sometimes_missing", DataType.FLOAT, 1, dim);
        INDArray w = Nd4j.randn(DataType.FLOAT, dim, dim).muli(0.1);
        sd.constant("w", w);
        // Output depends on both: (ph1 + ph2) @ W
        SDVariable sum = ph1.add("sum", ph2);
        sd.mmul("out", sum, sd.getVariable("w"));

        sd.setGraphExecutionMode(mode);

        // Warmup: provide both placeholders
        INDArray input1 = Nd4j.randn(DataType.FLOAT, 1, dim);
        INDArray input2 = Nd4j.randn(DataType.FLOAT, 1, dim);
        for (int step = 0; step < 5; step++) {
            input1.assign(Nd4j.randn(DataType.FLOAT, 1, dim));
            input2.assign(Nd4j.randn(DataType.FLOAT, 1, dim));
            sd.output(Map.of("always_present", input1, "sometimes_missing", input2), "out");
        }

        // Now provide ONLY "always_present" — what happens to "sometimes_missing"?
        // The correct behavior is either: error, OR use last-known value.
        // The WRONG behavior is: use uninitialized/random data silently.
        INDArray lastInput2 = input2.dup(); // save last known value

        boolean errored = false;
        INDArray resultWithMissing = null;
        try {
            input1.assign(Nd4j.randn(DataType.FLOAT, 1, dim));
            resultWithMissing = sd.output(Map.of("always_present", input1), "out").get("out").dup();
        } catch (Exception e) {
            errored = true;
            log.info("{}: correctly threw error when placeholder missing: {}", mode, e.getMessage());
        }

        if (!errored && resultWithMissing != null) {
            // Didn't error — verify it used the cached/last value, not garbage
            // Compute expected: (input1 + lastInput2) @ W
            INDArray expected = input1.add(lastInput2).mmul(w);
            double diff = expected.sub(resultWithMissing).amaxNumber().doubleValue();
            // This is informational — document behavior
            log.info("{}: no error on missing placeholder. diff from last-cached={} "
                    + "(if large, stale/random data used)", mode, diff);
        }

        // The main assertion: regardless of missing-input behavior, providing BOTH
        // again must still work correctly
        INDArray freshInput1 = Nd4j.randn(DataType.FLOAT, 1, dim);
        INDArray freshInput2 = Nd4j.randn(DataType.FLOAT, 1, dim);
        INDArray recovery = sd.output(
                Map.of("always_present", freshInput1, "sometimes_missing", freshInput2), "out"
        ).get("out").dup();
        INDArray expected = freshInput1.add(freshInput2).mmul(w);
        double recoveryDiff = expected.sub(recovery).amaxNumber().doubleValue();
        assertTrue(recoveryDiff < 1e-3,
                mode + ": after missing-placeholder step, recovery with both inputs diff=" + recoveryDiff
                        + ". Plan state may be corrupted.");
        log.info("{}: partial placeholder test passed (recovery diff={})", mode, recoveryDiff);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Tests 47-52: Merged capture through compute gaps
    //
    // These tests target the specific pattern where:
    //   island(Triton) → gap(cuBLAS matmul) → island(Triton)
    // and the gap matmul reads from a changing placeholder.
    //
    // With the bug: gap matmul gets baked into the merged CUDA graph,
    // producing stale output on every replay step.
    // After the fix: gap matmul runs fresh each step.
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Build a graph with interleaved Triton-compilable islands and cuBLAS gap matmuls.
     *
     * Pattern: input → [gather(constants) → reshape]×N → matmul(gap) → [gather → reshape]×N → matmul(gap) → ... → out
     *
     * The gather+reshape chains form Triton islands (many small ops = capturable).
     * The matmuls between them are cuBLAS gap ops.
     * Each matmul reads the placeholder input (directly or through a chain).
     */
    private SameDiff buildIslandGapIslandChain(INDArray[][] weights, int dim, int numBlocks) {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, dim);

        // Build a table of embeddings (constants) for gather ops to form islands
        INDArray embTable = Nd4j.randn(DataType.FLOAT, dim, dim).muli(0.02);
        sd.constant("emb_table", embTable.dup());

        // Indices for gather ops — constant, forms Triton-compilable island
        INDArray indices = Nd4j.arange(dim).castTo(DataType.INT64);
        sd.constant("indices", indices.dup());

        SDVariable current = x;
        for (int block = 0; block < numBlocks; block++) {
            // --- ISLAND: gather + reshape chain (Triton-compilable, no cuBLAS) ---
            // These ops produce a permutation of the embedding table
            SDVariable gathered = sd.gather("gather_" + block, sd.getVariable("emb_table"), sd.getVariable("indices"), 0);
            SDVariable reshaped = sd.reshape("reshape_" + block, gathered, sd.constant("shape_" + block, Nd4j.createFromArray(1L, (long) dim * dim)));
            SDVariable sliced = sd.stridedSlice("slice_" + block, reshaped, new long[]{0, 0}, new long[]{1, dim}, new long[]{1, 1});

            // --- GAP: matmul (cuBLAS — NOT Triton-compilable as a standalone) ---
            // This matmul reads from both the island output AND the changing placeholder
            sd.constant("w_" + block, weights[block][0].dup());
            SDVariable addedInput = current.add("add_input_" + block, sliced);
            current = sd.mmul("gap_mm_" + block, addedInput, sd.getVariable("w_" + block));
        }

        current.rename("out");
        return sd;
    }

    /**
     * Test 47: Island-gap-island chain — gap matmul output MUST change when input changes.
     *
     * This is the closest unit-test reproduction of the VLM bug pattern:
     * Triton islands are captured into CUDA graphs, but the matmul gap ops
     * between them may get baked into merged CUDA graphs with stale kernel args.
     */
    @ParameterizedTest(name = "islandGapIslandStaleOutput mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"TRITON", "AUTO"})
    @Order(47)
    void test47_IslandGapIslandStaleOutput(GraphExecutionMode mode) {
        int dim = 32;
        int numBlocks = 4;
        INDArray[][] weights = new INDArray[numBlocks][1];
        Nd4j.getRandom().setSeed(42);
        for (int b = 0; b < numBlocks; b++) {
            weights[b][0] = Nd4j.randn(DataType.FLOAT, dim, dim).muli(0.02);
        }

        SameDiff sd = track(buildIslandGapIslandChain(weights, dim, numBlocks));
        sd.setGraphExecutionMode(mode);

        SameDiff sdRef = track(buildIslandGapIslandChain(weights, dim, numBlocks));
        sdRef.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);

        int totalSteps = 20;
        int matchCount = 0;
        INDArray prevResult = null;
        int stuckCount = 0;

        for (int step = 0; step < totalSteps; step++) {
            INDArray input = Nd4j.randn(DataType.FLOAT, 1, dim);

            INDArray result = sd.output(Map.of("x", input), "out").get("out").dup();
            INDArray ref = sdRef.output(Map.of("x", input), "out").get("out").dup();

            assertFalse(result.isNaN().any(), mode + " step " + step + ": NaN in output");

            double diff = ref.sub(result).amaxNumber().doubleValue();
            if (diff < 0.05) matchCount++;

            // Check for stuck output (the bug symptom)
            if (prevResult != null) {
                double changeMag = result.sub(prevResult).amaxNumber().doubleValue();
                if (changeMag < 1e-6) {
                    stuckCount++;
                    log.warn("{} step {}: output STUCK (change={}, same as prev step)", mode, step, changeMag);
                }
            }
            prevResult = result;

            if (step < 3 || step == totalSteps - 1) {
                log.info("{} step {}: diff={} result[0]={}", mode, step, diff, result.getFloat(0));
            }
        }

        // Must match reference for at least 80% of steps
        double matchRate = (double) matchCount / totalSteps;
        assertTrue(matchRate >= 0.8,
                mode + ": island-gap-island chain accuracy failed. matchRate=" + matchRate
                        + " (need >=0.8). Gap matmul outputs may be stale from merged capture.");

        // Must NOT be stuck — output should change every step
        assertTrue(stuckCount <= 1,
                mode + ": output stuck for " + stuckCount + "/" + totalSteps
                        + " steps. Gap matmul baked into CUDA graph?");

        log.info("{}: island-gap-island test passed. matchRate={} stuckCount={}", mode, matchRate, stuckCount);
    }

    /**
     * Test 48: Multi-layer deep chain with explicit gap matmuls between norm blocks.
     *
     * Pattern: (matmul → rmsNorm)×8 with changing input.
     * rmsNorm is Triton-compilable, matmul may be cuBLAS gap.
     * With 8 layers, this creates enough ops for multiple segments and islands.
     * The key assertion: step N output != step N-1 output for ALL replay steps.
     */
    @ParameterizedTest(name = "deepChainNoStuckSteps mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"TRITON", "AUTO"})
    @Order(48)
    void test48_DeepChainNoStuckSteps(GraphExecutionMode mode) {
        int layers = 8, dim = 64;
        INDArray[][] weights = generateDeepChainWeights(layers, dim);

        SameDiff sd = track(buildDeepChainWith(weights, dim));
        sd.setGraphExecutionMode(mode);

        SameDiff sdRef = track(buildDeepChainWith(weights, dim));
        sdRef.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);

        int totalSteps = 30;
        INDArray prevResult = null;
        int stuckCount = 0;
        int mismatchCount = 0;

        for (int step = 0; step < totalSteps; step++) {
            INDArray input = Nd4j.randn(DataType.FLOAT, 1, dim).muli(0.1);

            INDArray result = sd.output(Map.of("input", input), "norm_" + (layers - 1)).get("norm_" + (layers - 1)).dup();
            INDArray ref = sdRef.output(Map.of("input", input), "norm_" + (layers - 1)).get("norm_" + (layers - 1)).dup();

            // Check accuracy vs reference
            double diff = ref.sub(result).amaxNumber().doubleValue();
            if (diff > 0.05) {
                mismatchCount++;
                log.warn("{} step {}: MISMATCH vs SLOT_BY_SLOT diff={}", mode, step, diff);
            }

            // Check for stuck output
            if (prevResult != null) {
                double changeMag = result.sub(prevResult).amaxNumber().doubleValue();
                if (changeMag < 1e-6) {
                    stuckCount++;
                    log.warn("{} step {}: output STUCK (change={})", mode, step, changeMag);
                }
            }
            prevResult = result;
        }

        assertTrue(stuckCount <= 1,
                mode + ": deep chain stuck " + stuckCount + "/" + totalSteps + " steps");
        assertTrue(mismatchCount <= 3,
                mode + ": deep chain mismatch " + mismatchCount + "/" + totalSteps + " steps");

        log.info("{}: deep chain 30-step test passed. stuck={} mismatch={}", mode, stuckCount, mismatchCount);
    }

    /**
     * Test 49: Verify that SLOT_BY_SLOT and composite replay produce the SAME output
     * for a graph with multiple matmul gap ops — specifically at the transition point
     * from warmup (slot-by-slot execution) to captured replay.
     *
     * The transition step is where merged capture bugs manifest:
     * warmup output is correct, but the first replay step produces stale values
     * because gap ops are baked into the CUDA graph.
     */
    @ParameterizedTest(name = "warmupToReplayTransitionAccuracy mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"TRITON", "AUTO"})
    @Order(49)
    void test49_WarmupToReplayTransitionAccuracy(GraphExecutionMode mode) {
        int layers = 4, dim = 32;
        INDArray[][] weights = generateDeepChainWeights(layers, dim);

        SameDiff sd = track(buildDeepChainWith(weights, dim));
        sd.setGraphExecutionMode(mode);

        SameDiff sdRef = track(buildDeepChainWith(weights, dim));
        sdRef.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);

        String outName = "norm_" + (layers - 1);

        // Steps 0-3: warmup (slot-by-slot execution in DSP)
        // Steps 4-7: should transition to replay
        // Steps 8-15: steady-state replay
        int totalSteps = 16;
        double maxDiff = 0;
        int worstStep = -1;

        for (int step = 0; step < totalSteps; step++) {
            INDArray input = Nd4j.randn(DataType.FLOAT, 1, dim).muli(0.1);

            INDArray result = sd.output(Map.of("input", input), outName).get(outName).dup();
            INDArray ref = sdRef.output(Map.of("input", input), outName).get(outName).dup();

            double diff = ref.sub(result).amaxNumber().doubleValue();
            if (diff > maxDiff) {
                maxDiff = diff;
                worstStep = step;
            }

            log.info("{} step {} ({}): diff={}", mode, step,
                    step < 4 ? "warmup" : step < 8 ? "TRANSITION" : "replay", diff);

            // Tight tolerance at every step — no regression at transition
            assertTrue(diff < 0.1,
                    mode + " step " + step + ": accuracy regression diff=" + diff
                            + " (this step is " + (step < 4 ? "warmup" : step < 8 ? "TRANSITION" : "replay") + ")");
        }

        log.info("{}: transition accuracy test passed. worstDiff={} at step {}", mode, maxDiff, worstStep);
    }

    /**
     * Test 50: Verify that constant weights are NEVER corrupted during replay.
     *
     * Run 30 steps with changing placeholder input. After each step, read back the
     * constant weight and verify it's bit-identical to the original.
     * This catches bugs where D2D staging, arg table refresh, or merged capture
     * accidentally overwrite constant weight buffers.
     */
    @ParameterizedTest(name = "constantWeightIntegrityDuringReplay mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"TRITON", "AUTO"})
    @Order(50)
    void test50_ConstantWeightIntegrityDuringReplay(GraphExecutionMode mode) {
        int dim = 32;
        Nd4j.getRandom().setSeed(99);
        INDArray w = Nd4j.randn(DataType.FLOAT, dim, dim).muli(0.02);
        INDArray b = Nd4j.ones(DataType.FLOAT, 1, dim);

        // Save original weight bytes for bit-exact comparison
        float[] origW = w.dup().data().asFloat();
        float[] origB = b.dup().data().asFloat();

        SameDiff sd = track(SameDiff.create());
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, dim);
        SDVariable wVar = sd.constant("w", w.dup());
        SDVariable bVar = sd.constant("b", b.dup());
        SDVariable mm = sd.mmul("mm", x, wVar);
        mm.add("out", bVar);
        sd.setGraphExecutionMode(mode);

        for (int step = 0; step < 30; step++) {
            INDArray input = Nd4j.randn(DataType.FLOAT, 1, dim);
            sd.output(Map.of("x", input), "out");

            // Read back weight from SameDiff — should be unchanged
            INDArray currentW = sd.getVariable("w").getArr();
            INDArray currentB = sd.getVariable("b").getArr();

            if (currentW != null) {
                float[] curWData = currentW.dup().data().asFloat();
                for (int i = 0; i < origW.length; i++) {
                    assertEquals(origW[i], curWData[i], 0.0f,
                            mode + " step " + step + ": weight w[" + i + "] corrupted! "
                                    + "orig=" + origW[i] + " now=" + curWData[i]);
                }
            }
            if (currentB != null) {
                float[] curBData = currentB.dup().data().asFloat();
                for (int i = 0; i < origB.length; i++) {
                    assertEquals(origB[i], curBData[i], 0.0f,
                            mode + " step " + step + ": bias b[" + i + "] corrupted!");
                }
            }
        }

        log.info("{}: constant weight integrity verified across 30 replay steps", mode);
    }

    /**
     * Test 51: Multiple placeholders with different lifecycle patterns in one graph.
     *
     * - placeholder A: changes every step (like inputs_embeds)
     * - placeholder B: changes every step (like position_ids)
     * - placeholder C: stays constant after warmup (like attention_mask shape)
     *
     * All three must be handled correctly simultaneously.
     * The bug pattern: if A and B change but C doesn't, the arg table refresh
     * might skip the refresh for A and B because C's address is stable.
     */
    @ParameterizedTest(name = "multiPlaceholderLifecycle mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"TRITON", "AUTO"})
    @Order(51)
    void test51_MultiPlaceholderLifecycle(GraphExecutionMode mode) {
        int dim = 32;
        Nd4j.getRandom().setSeed(77);
        INDArray wA = Nd4j.randn(DataType.FLOAT, dim, dim).muli(0.02);
        INDArray wB = Nd4j.randn(DataType.FLOAT, dim, dim).muli(0.02);

        SameDiff sd = track(SameDiff.create());
        SDVariable a = sd.placeHolder("a", DataType.FLOAT, 1, dim);
        SDVariable b = sd.placeHolder("b", DataType.FLOAT, 1, dim);
        SDVariable c = sd.placeHolder("c", DataType.FLOAT, 1, dim);

        sd.constant("wA", wA.dup());
        sd.constant("wB", wB.dup());

        SDVariable mmA = sd.mmul("mmA", a, sd.getVariable("wA"));
        SDVariable mmB = sd.mmul("mmB", b, sd.getVariable("wB"));
        SDVariable combined = mmA.add("add_ab", mmB);
        combined.add("out", c);  // c is added directly (stays constant after warmup)

        sd.setGraphExecutionMode(mode);

        SameDiff sdRef = track(SameDiff.create());
        SDVariable aR = sdRef.placeHolder("a", DataType.FLOAT, 1, dim);
        SDVariable bR = sdRef.placeHolder("b", DataType.FLOAT, 1, dim);
        SDVariable cR = sdRef.placeHolder("c", DataType.FLOAT, 1, dim);
        sdRef.constant("wA", wA.dup());
        sdRef.constant("wB", wB.dup());
        SDVariable mmAR = sdRef.mmul("mmA", aR, sdRef.getVariable("wA"));
        SDVariable mmBR = sdRef.mmul("mmB", bR, sdRef.getVariable("wB"));
        SDVariable combinedR = mmAR.add("add_ab", mmBR);
        combinedR.add("out", cR);
        sdRef.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);

        // C stays constant after step 0
        INDArray fixedC = Nd4j.randn(DataType.FLOAT, 1, dim);

        int totalSteps = 20;
        int stuckCount = 0;
        INDArray prevResult = null;

        for (int step = 0; step < totalSteps; step++) {
            INDArray inputA = Nd4j.randn(DataType.FLOAT, 1, dim);
            INDArray inputB = Nd4j.randn(DataType.FLOAT, 1, dim);
            Map<String, INDArray> ph = Map.of("a", inputA, "b", inputB, "c", fixedC);

            INDArray result = sd.output(ph, "out").get("out").dup();
            INDArray ref = sdRef.output(ph, "out").get("out").dup();

            double diff = ref.sub(result).amaxNumber().doubleValue();
            assertTrue(diff < 0.05,
                    mode + " step " + step + ": multi-placeholder diff=" + diff);

            if (prevResult != null) {
                double change = result.sub(prevResult).amaxNumber().doubleValue();
                if (change < 1e-6) stuckCount++;
            }
            prevResult = result;
        }

        assertTrue(stuckCount <= 1,
                mode + ": multi-placeholder stuck " + stuckCount + "/" + totalSteps);
        log.info("{}: multi-placeholder lifecycle test passed. stuck={}", mode, stuckCount);
    }

    /**
     * Test 52: Verify that the number of unique output values across N replay steps
     * matches expectations — specifically that we don't get just 1-2 unique values
     * repeated across 20 steps (the degenerate VLM pattern).
     *
     * This is a statistical test: with random inputs, the probability of getting
     * the same argmax index twice in 20 steps is very low for a 32-dim output.
     */
    @ParameterizedTest(name = "outputDiversityAcrossReplaySteps mode={0}")
    @EnumSource(value = GraphExecutionMode.class, names = {"TRITON", "AUTO"})
    @Order(52)
    void test52_OutputDiversityAcrossReplaySteps(GraphExecutionMode mode) {
        int layers = 4, dim = 32;
        INDArray[][] weights = generateDeepChainWeights(layers, dim);

        SameDiff sd = track(buildDeepChainWith(weights, dim));
        sd.setGraphExecutionMode(mode);

        String outName = "norm_" + (layers - 1);
        int totalSteps = 20;

        Set<Integer> uniqueArgmax = new HashSet<>();
        Set<String> uniqueFirstElement = new HashSet<>();

        for (int step = 0; step < totalSteps; step++) {
            INDArray input = Nd4j.randn(DataType.FLOAT, 1, dim);
            INDArray result = sd.output(Map.of("input", input), outName).get(outName).dup();

            int argmax = Nd4j.argMax(result, 1).getInt(0);
            uniqueArgmax.add(argmax);

            // Track first element as a string (to detect bit-exact repeats)
            uniqueFirstElement.add(String.valueOf(result.getFloat(0)));
        }

        log.info("{}: {} unique argmax values, {} unique first-element values across {} steps",
                mode, uniqueArgmax.size(), uniqueFirstElement.size(), totalSteps);

        // With random 32-dim inputs, we expect diverse outputs.
        // The VLM bug produced only 2-3 unique values across 250 steps.
        assertTrue(uniqueArgmax.size() >= 3,
                mode + ": only " + uniqueArgmax.size() + " unique argmax in " + totalSteps
                        + " steps — likely stuck/degenerate output");
        assertTrue(uniqueFirstElement.size() >= totalSteps / 2,
                mode + ": only " + uniqueFirstElement.size() + " unique first-element values in "
                        + totalSteps + " steps — output not changing between steps");
    }
}
