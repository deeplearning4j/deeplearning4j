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
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.autodiff.samediff.execution.PlanPhase;
import org.nd4j.common.config.ND4JSystemProperties;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for the DSP phase lifecycle after the modular refactoring.
 *
 * Verifies that phaseWarmup, phaseCompile, phaseFreeze, phaseReplay, and
 * phaseSlotBySlot produce correct results individually and during transitions.
 *
 * Uses simple SameDiff graphs (no VLM models) so tests run in seconds.
 *
 * Run from platform-tests:
 *   mvn test -Dtest=TestDSPPhaseLifecycle -Dbackend.artifactId=nd4j-cuda-12.9
 *   mvn test -Dtest=TestDSPPhaseLifecycle -Dbackend.artifactId=nd4j-native
 */
@Slf4j
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
public class TestDSPPhaseLifecycle {

    @BeforeAll
    static void setup() {
        System.setProperty(ND4JSystemProperties.DYNAMIC_SHAPE_PLAN_ENABLED, "true");
        InferenceSession.setDynamicShapePlanEnabled(true);
    }

    @AfterEach
    void cleanup() {
        Nd4j.getExecutioner().commit();
    }

    // Shared fixed weights — initialized once, reused across all graph instances
    private static INDArray FIXED_W1;
    private static INDArray FIXED_B1;
    private static INDArray FIXED_GAMMA;

    static {
        Nd4j.getRandom().setSeed(42);
        FIXED_W1 = Nd4j.randn(DataType.FLOAT, 64, 64).muli(0.02);
        FIXED_B1 = Nd4j.zeros(DataType.FLOAT, 64);
        FIXED_GAMMA = Nd4j.ones(DataType.FLOAT, 64);
    }

    // ─── Helper: build a small decoder-like graph ────────────────────────────
    // matmul → add → rms_norm → matmul (mimics a transformer layer)
    private static SameDiff buildSmallTransformerGraph() {
        SameDiff sd = SameDiff.create();

        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, 1, 64);
        SDVariable w1 = sd.var("w1", FIXED_W1.dup());
        SDVariable b1 = sd.var("b1", FIXED_B1.dup());
        SDVariable gamma = sd.var("gamma", FIXED_GAMMA.dup());

        // proj = input @ w1 + b1
        SDVariable proj = sd.mmul("proj", input, w1);
        SDVariable biased = proj.add("biased", b1);

        // norm = rms_norm(biased)
        SDVariable norm = sd.math.standardize("norm", biased, 2);
        SDVariable scaled = norm.mul("scaled", gamma);

        // output = scaled @ w1 (reuse weights for simplicity)
        SDVariable output = sd.mmul("output", scaled, w1);

        sd.setOutputs("output");
        return sd;
    }

    // ─── Helper: execute and return output ────────────────────────────────────
    private static INDArray executeAndGetOutput(SameDiff sd, INDArray input) {
        Map<String, INDArray> ph = new LinkedHashMap<>();
        ph.put("input", input);
        Map<String, INDArray> out = sd.output(ph, "output");
        return out.get("output").dup();
    }

    // ─── Helper: close output properly ───────────────────────────────────────
    private static void closeOutput(INDArray arr) {
        if (arr != null) {
            arr.setCloseable(true);
            arr.close();
        }
    }

    // =========================================================================
    // Test 1: Slot-by-slot produces stable results across executions
    // =========================================================================
    @Test
    @Order(1)
    @DisplayName("SLOT_BY_SLOT: stable output across multiple executions")
    public void testSlotBySlotStability() {
        SameDiff sd = buildSmallTransformerGraph();
        sd.compileNativeDynamicShapePlan(sd.outputs(), GraphExecutionMode.SLOT_BY_SLOT, true);

        INDArray input = Nd4j.randn(DataType.FLOAT, 1, 1, 64);
        INDArray ref = null;

        try {
            ref = executeAndGetOutput(sd, input);

            for (int i = 0; i < 5; i++) {
                INDArray result = executeAndGetOutput(sd, input);
                double maxDiff = ref.sub(result).amaxNumber().doubleValue();
                log.info("  SLOT_BY_SLOT exec {}: maxDiff={}", i + 1, maxDiff);
                assertTrue(maxDiff < 1e-5,
                        "SLOT_BY_SLOT should be deterministic, maxDiff=" + maxDiff);
                closeOutput(result);
            }
        } finally {
            closeOutput(ref);
            sd.resetSession();
            sd.close();
        }
    }

    // =========================================================================
    // Test 2: Phase progression: SLOT_BY_SLOT → SHAPES_FROZEN after freeze
    // =========================================================================
    @Test
    @Order(2)
    @DisplayName("Phase progression: unfrozen → frozen phase transition")
    public void testPhaseProgressionAfterFreeze() {
        SameDiff sd = buildSmallTransformerGraph();
        sd.compileNativeDynamicShapePlan(sd.outputs(), GraphExecutionMode.SLOT_BY_SLOT, true);

        INDArray input = Nd4j.randn(DataType.FLOAT, 1, 1, 64);
        INDArray result = null;

        try {
            // Execute once unfrozen to populate shapes
            result = executeAndGetOutput(sd, input);
            closeOutput(result);
            result = null;

            DynamicShapePlanExecutor executor = sd.getSessions().values().iterator().next()
                    .getDynamicShapePlanExecutor();
            assertNotNull(executor, "Executor should exist after execution");

            // Before freeze
            assertFalse(executor.isShapesFrozen(), "Should not be frozen yet");

            // Freeze shapes
            executor.setShapesFrozen(true);
            assertTrue(executor.isShapesFrozen(), "Should be frozen after setShapesFrozen(true)");

            PlanPhase phase = executor.getPlanPhase();
            log.info("  Phase after freeze: {}", phase);
            assertNotNull(phase, "Phase should not be null");
            assertTrue(phase.isAtLeast(PlanPhase.SHAPES_FROZEN),
                    "Phase should be at least SHAPES_FROZEN, got " + phase);

            // Execute while frozen (warmup)
            result = executeAndGetOutput(sd, input);
            int frozenCount = executor.getFrozenExecutionCount();
            log.info("  Frozen execution count after warmup: {}", frozenCount);
            assertTrue(frozenCount >= 1, "Frozen exec count should be >= 1");
            closeOutput(result);
            result = null;

            // Execute again (should advance past warmup)
            result = executeAndGetOutput(sd, input);
            int frozenCount2 = executor.getFrozenExecutionCount();
            log.info("  Frozen execution count after 2nd exec: {}", frozenCount2);
            assertTrue(frozenCount2 > frozenCount,
                    "Frozen exec count should increase");
            closeOutput(result);
            result = null;
        } finally {
            closeOutput(result);
            sd.resetSession();
            sd.close();
        }
    }

    // =========================================================================
    // Test 3: Warmup output matches slot-by-slot output
    // =========================================================================
    @Test
    @Order(3)
    @DisplayName("Warmup output matches slot-by-slot baseline")
    public void testWarmupMatchesSlotBySlot() {
        INDArray input = Nd4j.randn(DataType.FLOAT, 1, 1, 64);

        // Reference: slot-by-slot
        SameDiff sdRef = buildSmallTransformerGraph();
        sdRef.compileNativeDynamicShapePlan(sdRef.outputs(), GraphExecutionMode.SLOT_BY_SLOT, true);
        INDArray refOutput = null;

        // Test: auto mode with freeze (warmup path)
        SameDiff sdTest = buildSmallTransformerGraph();
        sdTest.compileNativeDynamicShapePlan(
                sdTest.outputs(),
                GraphExecutionMode.AUTO, true);
        INDArray testOutput = null;

        try {
            // Get reference
            refOutput = executeAndGetOutput(sdRef, input);

            // Execute test unfrozen first (populates shapes)
            testOutput = executeAndGetOutput(sdTest, input);
            closeOutput(testOutput);
            testOutput = null;

            // Freeze and execute (warmup phase)
            DynamicShapePlanExecutor executor = sdTest.getSessions().values().iterator().next()
                    .getDynamicShapePlanExecutor();
            executor.setShapesFrozen(true);

            testOutput = executeAndGetOutput(sdTest, input);

            double maxDiff = refOutput.sub(testOutput).amaxNumber().doubleValue();
            log.info("  Warmup vs slot-by-slot maxDiff={}", maxDiff);
            assertTrue(maxDiff < 1e-4,
                    "Warmup output should match slot-by-slot, maxDiff=" + maxDiff);
        } finally {
            closeOutput(refOutput);
            closeOutput(testOutput);
            sdRef.resetSession();
            sdRef.close();
            sdTest.resetSession();
            sdTest.close();
        }
    }

    // =========================================================================
    // Test 4: Post-warmup executions remain correct (multiple frozen steps)
    // =========================================================================
    @Test
    @Order(4)
    @DisplayName("Post-warmup frozen executions remain correct across 10 steps")
    public void testFrozenExecutionStability() {
        INDArray input = Nd4j.randn(DataType.FLOAT, 1, 1, 64);

        // Reference: slot-by-slot
        SameDiff sdRef = buildSmallTransformerGraph();
        sdRef.compileNativeDynamicShapePlan(sdRef.outputs(), GraphExecutionMode.SLOT_BY_SLOT, true);
        INDArray refOutput = null;

        // Test: auto mode
        SameDiff sdTest = buildSmallTransformerGraph();
        sdTest.compileNativeDynamicShapePlan(
                sdTest.outputs(),
                GraphExecutionMode.AUTO, true);
        INDArray testOutput = null;

        try {
            refOutput = executeAndGetOutput(sdRef, input);

            // Execute unfrozen
            testOutput = executeAndGetOutput(sdTest, input);
            closeOutput(testOutput);
            testOutput = null;

            // Freeze
            DynamicShapePlanExecutor executor = sdTest.getSessions().values().iterator().next()
                    .getDynamicShapePlanExecutor();
            executor.setShapesFrozen(true);

            // Execute 10 frozen steps, verify each matches reference
            for (int step = 0; step < 10; step++) {
                testOutput = executeAndGetOutput(sdTest, input);

                double maxDiff = refOutput.sub(testOutput).amaxNumber().doubleValue();
                int frozenExec = executor.getFrozenExecutionCount();
                PlanPhase phase = executor.getPlanPhase();
                log.info("  Frozen step {}: maxDiff={} frozenExec={} phase={}",
                        step, maxDiff, frozenExec, phase);

                assertTrue(maxDiff < 1e-4,
                        "Frozen step " + step + " diverged from ref, maxDiff=" + maxDiff);
                closeOutput(testOutput);
                testOutput = null;
            }
        } finally {
            closeOutput(refOutput);
            closeOutput(testOutput);
            sdRef.resetSession();
            sdRef.close();
            sdTest.resetSession();
            sdTest.close();
        }
    }

    // =========================================================================
    // Test 5: Triton mode output matches slot-by-slot (if Triton available)
    // =========================================================================
    @Test
    @Order(5)
    @DisplayName("TRITON mode output matches SLOT_BY_SLOT baseline")
    public void testTritonMatchesSlotBySlot() {
        if (!Nd4j.getNativeOps().isTritonAvailable()) {
            log.info("Triton not available, skipping");
            return;
        }

        INDArray input = Nd4j.randn(DataType.FLOAT, 1, 1, 64);

        // Reference: slot-by-slot
        SameDiff sdRef = buildSmallTransformerGraph();
        sdRef.compileNativeDynamicShapePlan(
                sdRef.outputs(),
                GraphExecutionMode.SLOT_BY_SLOT, true);
        INDArray refOutput = null;

        // Test: Triton mode
        SameDiff sdTest = buildSmallTransformerGraph();
        sdTest.compileNativeDynamicShapePlan(
                sdTest.outputs(),
                GraphExecutionMode.TRITON, false);
        INDArray testOutput = null;

        try {
            refOutput = executeAndGetOutput(sdRef, input);

            // Execute unfrozen
            testOutput = executeAndGetOutput(sdTest, input);
            closeOutput(testOutput);
            testOutput = null;

            // Freeze
            DynamicShapePlanExecutor executor = sdTest.getSessions().values().iterator().next()
                    .getDynamicShapePlanExecutor();
            executor.setShapesFrozen(true);

            // Execute frozen steps — warmup, compile, then Triton execution
            for (int step = 0; step < 5; step++) {
                testOutput = executeAndGetOutput(sdTest, input);

                double maxDiff = refOutput.sub(testOutput).amaxNumber().doubleValue();
                int frozenExec = executor.getFrozenExecutionCount();
                PlanPhase phase = executor.getPlanPhase();
                log.info("  TRITON step {}: maxDiff={} frozenExec={} phase={}",
                        step, maxDiff, frozenExec, phase);

                // Allow TF32 tolerance on CUDA (10-bit mantissa → ~1e-3 relative error)
                assertTrue(maxDiff < 0.05,
                        "TRITON step " + step + " diverged from SLOT_BY_SLOT, maxDiff=" + maxDiff);
                closeOutput(testOutput);
                testOutput = null;
            }
        } finally {
            closeOutput(refOutput);
            closeOutput(testOutput);
            sdRef.resetSession();
            sdRef.close();
            sdTest.resetSession();
            sdTest.close();
        }
    }

    // =========================================================================
    // Test 6: Freeze-unfreeze-refreeze cycle produces correct output
    // =========================================================================
    @Test
    @Order(6)
    @DisplayName("Freeze → unfreeze → refreeze cycle maintains correctness")
    public void testFreezeUnfreezeRefreezeCycle() {
        INDArray input = Nd4j.randn(DataType.FLOAT, 1, 1, 64);

        SameDiff sd = buildSmallTransformerGraph();
        sd.compileNativeDynamicShapePlan(
                sd.outputs(),
                GraphExecutionMode.AUTO, true);
        INDArray refOutput = null;
        INDArray testOutput = null;

        try {
            // Get reference
            refOutput = executeAndGetOutput(sd, input);

            DynamicShapePlanExecutor executor = sd.getSessions().values().iterator().next()
                    .getDynamicShapePlanExecutor();

            // Cycle 1: freeze → run → unfreeze
            executor.setShapesFrozen(true);
            testOutput = executeAndGetOutput(sd, input);
            double diff1 = refOutput.sub(testOutput).amaxNumber().doubleValue();
            log.info("  After first freeze: maxDiff={}", diff1);
            assertTrue(diff1 < 1e-4, "First freeze diverged, maxDiff=" + diff1);
            closeOutput(testOutput);
            testOutput = null;

            executor.setShapesFrozen(false);
            PlanPhase phaseAfterUnfreeze = executor.getPlanPhase();
            log.info("  Phase after unfreeze: {}", phaseAfterUnfreeze);

            // Cycle 2: refreeze → run
            testOutput = executeAndGetOutput(sd, input);
            closeOutput(testOutput);
            testOutput = null;

            executor.setShapesFrozen(true);
            testOutput = executeAndGetOutput(sd, input);
            double diff2 = refOutput.sub(testOutput).amaxNumber().doubleValue();
            log.info("  After refreeze: maxDiff={}", diff2);
            assertTrue(diff2 < 1e-4, "Refreeze diverged, maxDiff=" + diff2);
            closeOutput(testOutput);
            testOutput = null;

            // Run a few more frozen steps
            for (int i = 0; i < 3; i++) {
                testOutput = executeAndGetOutput(sd, input);
                double diff = refOutput.sub(testOutput).amaxNumber().doubleValue();
                log.info("  Post-refreeze step {}: maxDiff={}", i, diff);
                assertTrue(diff < 1e-4, "Post-refreeze step " + i + " diverged");
                closeOutput(testOutput);
                testOutput = null;
            }
        } finally {
            closeOutput(refOutput);
            closeOutput(testOutput);
            sd.resetSession();
            sd.close();
        }
    }

    // =========================================================================
    // Test 7: All execution modes produce same output on same graph
    // =========================================================================
    @Test
    @Order(7)
    @DisplayName("All execution modes produce consistent output")
    public void testAllModesConsistent() {
        INDArray input = Nd4j.randn(DataType.FLOAT, 1, 1, 64);

        // Modes to test
        List<GraphExecutionMode> modes = new ArrayList<>();
        modes.add(GraphExecutionMode.SLOT_BY_SLOT);
        modes.add(GraphExecutionMode.AUTO);
        if (Nd4j.getNativeOps().isTritonAvailable()) {
            modes.add(GraphExecutionMode.TRITON);
        }

        // Reference: slot-by-slot
        SameDiff sdRef = buildSmallTransformerGraph();
        sdRef.compileNativeDynamicShapePlan(
                sdRef.outputs(),
                GraphExecutionMode.SLOT_BY_SLOT, true);
        INDArray refOutput = executeAndGetOutput(sdRef, input);

        try {
            for (GraphExecutionMode mode : modes) {
                SameDiff sd = buildSmallTransformerGraph();
                sd.compileNativeDynamicShapePlan(
                        sd.outputs(),
                        mode, mode != GraphExecutionMode.TRITON);

                // Execute unfrozen
                INDArray out = executeAndGetOutput(sd, input);
                double unfrozenDiff = refOutput.sub(out).amaxNumber().doubleValue();
                log.info("  {} unfrozen: maxDiff={}", mode, unfrozenDiff);
                closeOutput(out);

                // Freeze and execute
                DynamicShapePlanExecutor executor = sd.getSessions().values().iterator().next()
                        .getDynamicShapePlanExecutor();
                executor.setShapesFrozen(true);

                // Warmup + a few more
                for (int step = 0; step < 4; step++) {
                    out = executeAndGetOutput(sd, input);
                    double frozenDiff = refOutput.sub(out).amaxNumber().doubleValue();
                    log.info("  {} frozen step {}: maxDiff={}", mode, step, frozenDiff);

                    double tolerance = (mode == GraphExecutionMode.TRITON) ? 0.05 : 1e-4;
                    assertTrue(frozenDiff < tolerance,
                            mode + " frozen step " + step + " diverged, maxDiff=" + frozenDiff);
                    closeOutput(out);
                }

                sd.resetSession();
                sd.close();
            }
        } finally {
            closeOutput(refOutput);
            sdRef.resetSession();
            sdRef.close();
        }
    }

    // =========================================================================
    // Test 8: Changing input values while frozen produces correct output
    // =========================================================================
    @Test
    @Order(8)
    @DisplayName("Different input values while frozen produce correct output")
    public void testFrozenWithVaryingInputs() {
        SameDiff sd = buildSmallTransformerGraph();
        sd.compileNativeDynamicShapePlan(
                sd.outputs(),
                GraphExecutionMode.AUTO, true);

        SameDiff sdRef = buildSmallTransformerGraph();
        sdRef.compileNativeDynamicShapePlan(
                sdRef.outputs(),
                GraphExecutionMode.SLOT_BY_SLOT, true);

        INDArray testOut = null;
        INDArray refOut = null;

        try {
            // Execute unfrozen with first input
            INDArray input1 = Nd4j.randn(DataType.FLOAT, 1, 1, 64);
            testOut = executeAndGetOutput(sd, input1);
            closeOutput(testOut);
            testOut = null;

            // Freeze
            DynamicShapePlanExecutor executor = sd.getSessions().values().iterator().next()
                    .getDynamicShapePlanExecutor();
            executor.setShapesFrozen(true);

            // Warmup with input1
            testOut = executeAndGetOutput(sd, input1);
            closeOutput(testOut);
            testOut = null;

            // Now try different input values — same shape, different data
            for (int i = 0; i < 5; i++) {
                INDArray input = Nd4j.randn(DataType.FLOAT, 1, 1, 64);

                refOut = executeAndGetOutput(sdRef, input);
                testOut = executeAndGetOutput(sd, input);

                double maxDiff = refOut.sub(testOut).amaxNumber().doubleValue();
                log.info("  Varying input {}: maxDiff={}", i, maxDiff);
                assertTrue(maxDiff < 1e-4,
                        "Frozen execution with input " + i + " diverged, maxDiff=" + maxDiff);
                closeOutput(refOut);
                closeOutput(testOut);
                refOut = null;
                testOut = null;
            }
        } finally {
            closeOutput(testOut);
            closeOutput(refOut);
            sd.resetSession();
            sd.close();
            sdRef.resetSession();
            sdRef.close();
        }
    }
}
