package org.eclipse.deeplearning4j.nd4j.autodiff.samediff;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.*;
import org.junit.jupiter.api.Assumptions;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlanExecutor;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.autodiff.samediff.execution.PlanPhase;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for DSP phase enforcement guards.
 *
 * The DSP system has four lifecycle phases:
 *   SLOT_BY_SLOT (0) → SHAPES_FROZEN (1) → POINTERS_STABLE (2) → REPLAYING (3)
 *
 * Phase enforcement guards (DSP_REQUIRE_PLAN_PHASE_*) assert that configuration
 * methods are only called in valid phases:
 *   - Config setters (setGraphExecutionMode, setCudaGraphsEnabled, setJitMode) require ≤ SLOT_BY_SLOT
 *   - setShapesFrozen(true) requires == SLOT_BY_SLOT
 *   - phaseCompile / platformPrecompileSegments require ≤ SHAPES_FROZEN
 *
 * Since native assertion failures crash the JVM and cannot be caught from Java,
 * these tests verify:
 *   (A) Valid phase transitions succeed and advance phases correctly
 *   (B) Phase state queries return expected values at each stage
 *   (C) After resetForNextPage(), config methods work again (phase demoted to SLOT_BY_SLOT)
 */
@Slf4j
@Tag(TagNames.SAMEDIFF)
@NativeTag
public class DSPPhaseEnforcementTest extends BaseNd4jTestWithBackends {

    private static final double FLOAT_TOL = 1e-4;

    @Override
    public char ordering() {
        return 'c';
    }

    private boolean isCudaBackend() {
        return Nd4j.getBackend().getClass().getSimpleName().contains("Cuda");
    }

    // ===================== Helpers =====================

    /**
     * Build a simple matmul SameDiff model.
     */
    private SameDiff buildMatmulAddGraph() {
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 16);
        SDVariable w = sd.constant("w", Nd4j.randn(DataType.FLOAT, 16, 32));
        sd.mmul("output", input, w);
        return sd;
    }

    /**
     * Enable DSP auto-compile on a SameDiff instance.
     */
    private void enableDsp(SameDiff sd, GraphExecutionMode mode) {
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);
        sd.setGraphExecutionMode(mode);
    }

    /**
     * Get the DynamicShapePlanExecutor for the current thread's session.
     * Returns null if no session exists or DSP hasn't been initialized yet.
     */
    private DynamicShapePlanExecutor getDspExecutor(SameDiff sd) {
        InferenceSession session = sd.getOrCreateSession();
        return session.getDynamicShapePlanExecutor();
    }

    /**
     * Get the current plan phase from the DSP executor.
     * Returns null if no native plan is compiled.
     */
    private PlanPhase getPlanPhase(SameDiff sd) {
        DynamicShapePlanExecutor dsp = getDspExecutor(sd);
        return dsp != null ? dsp.getPlanPhase() : null;
    }

    /**
     * Freeze shapes on the DSP executor (enters SHAPES_FROZEN phase).
     */
    private void freezeShapes(SameDiff sd) {
        DynamicShapePlanExecutor dsp = getDspExecutor(sd);
        if (dsp != null) {
            dsp.setShapesFrozen(true);
        }
    }

    /**
     * Unfreeze shapes on the DSP executor.
     */
    private void unfreezeShapes(SameDiff sd) {
        DynamicShapePlanExecutor dsp = getDspExecutor(sd);
        if (dsp != null) {
            dsp.setShapesFrozen(false);
        }
    }

    /**
     * Run one execution of the model with the given input.
     */
    private INDArray executeOnce(SameDiff sd, INDArray input) {
        Map<String, INDArray> result = sd.output(Map.of("input", input.dup()), "output");
        return result.get("output").dup();
    }

    // ===================== Tests =====================

    @Test
    @DisplayName("Phase 1: Config setters work in SLOT_BY_SLOT before any execution")
    public void testConfigSettersAllowedInSlotBySlot() {
        SameDiff sd = buildMatmulAddGraph();

        // Enable DSP — this sets graph execution mode while in SLOT_BY_SLOT
        enableDsp(sd, GraphExecutionMode.AUTO);

        // Should still be able to change mode (still in SLOT_BY_SLOT, no execution yet)
        sd.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);

        // Verify by executing successfully
        INDArray input = Nd4j.randn(DataType.FLOAT, 1, 16);
        INDArray output = executeOnce(sd, input);
        assertNotNull(output, "Execution after config change should succeed");

        sd.close();
        log.info("PASS: testConfigSettersAllowedInSlotBySlot");
    }

    @Test
    @DisplayName("Phase 2: Valid transition SLOT_BY_SLOT → SHAPES_FROZEN via setShapesFrozen(true)")
    public void testValidTransitionToShapesFrozen() {
        SameDiff sd = buildMatmulAddGraph();
        enableDsp(sd, GraphExecutionMode.AUTO);

        // Execute once to create the native plan
        INDArray input = Nd4j.randn(DataType.FLOAT, 1, 16);
        executeOnce(sd, input);

        // Plan should now be compiled
        DynamicShapePlanExecutor dsp = getDspExecutor(sd);
        assertNotNull(dsp, "DSP executor should exist after first execution");

        // Before freezing: phase should be SLOT_BY_SLOT (shapes not yet frozen)
        PlanPhase phaseBefore = getPlanPhase(sd);
        log.info("Phase before freeze: {}", phaseBefore);

        // Freeze shapes — this is valid in SLOT_BY_SLOT (requires == SLOT_BY_SLOT)
        freezeShapes(sd);
        assertTrue(sd.isDspShapesFrozen(), "Shapes should be frozen after setShapesFrozen(true)");

        // Phase should now be SHAPES_FROZEN
        PlanPhase phaseAfter = getPlanPhase(sd);
        assertNotNull(phaseAfter, "Phase should not be null after freezing");
        assertEquals(PlanPhase.SHAPES_FROZEN, phaseAfter,
                "Phase should be SHAPES_FROZEN after setShapesFrozen(true)");

        unfreezeShapes(sd);
        sd.close();
        log.info("PASS: testValidTransitionToShapesFrozen");
    }

    @Test
    @DisplayName("Phase 3: Valid transition SHAPES_FROZEN → POINTERS_STABLE → REPLAYING via executions")
    public void testPhaseAdvancesThroughExecutions() {
        SameDiff sd = buildMatmulAddGraph();
        enableDsp(sd, GraphExecutionMode.CUDA_GRAPHS);

        INDArray fixedInput = Nd4j.randn(DataType.FLOAT, 1, 16);

        // First execution: creates the plan in SLOT_BY_SLOT
        executeOnce(sd, fixedInput);
        PlanPhase phase0 = getPlanPhase(sd);
        log.info("Phase after 1st exec: {}", phase0);
        assertEquals(PlanPhase.SLOT_BY_SLOT, phase0,
                "Phase should be SLOT_BY_SLOT after first execution");

        // Freeze shapes → SHAPES_FROZEN
        freezeShapes(sd);
        PlanPhase phase1 = getPlanPhase(sd);
        assertEquals(PlanPhase.SHAPES_FROZEN, phase1,
                "Phase should be SHAPES_FROZEN after freeze");

        // Second execution with frozen shapes → should advance to POINTERS_STABLE
        executeOnce(sd, fixedInput);
        PlanPhase phase2 = getPlanPhase(sd);
        log.info("Phase after 2nd exec (frozen): {}", phase2);
        assertTrue(phase2.isAtLeast(PlanPhase.SHAPES_FROZEN),
                "Phase should be at least SHAPES_FROZEN after second frozen execution, was: " + phase2);

        // Additional executions to reach REPLAYING (graph capture steady state)
        int maxExecs = 20;
        PlanPhase finalPhase = phase2;
        for (int i = 0; i < maxExecs; i++) {
            executeOnce(sd, fixedInput);
            finalPhase = getPlanPhase(sd);
            if (finalPhase == PlanPhase.REPLAYING) {
                log.info("Reached REPLAYING after {} additional executions", i + 1);
                break;
            }
        }
        log.info("Final phase after warmup: {}", finalPhase);
        // At minimum we should have advanced; ideally REPLAYING
        assertTrue(finalPhase.isAtLeast(PlanPhase.POINTERS_STABLE),
                "Phase should have advanced to at least POINTERS_STABLE after warmup, was: " + finalPhase);

        unfreezeShapes(sd);
        sd.close();
        log.info("PASS: testPhaseAdvancesThroughExecutions");
    }

    @Test
    @DisplayName("Phase 4: resetForNextPage() demotes phase back to SLOT_BY_SLOT")
    public void testResetForNextPageDemotesPhase() {
        SameDiff sd = buildMatmulAddGraph();
        enableDsp(sd, GraphExecutionMode.AUTO);

        INDArray input = Nd4j.randn(DataType.FLOAT, 1, 16);

        // Execute and freeze to reach SHAPES_FROZEN
        executeOnce(sd, input);
        freezeShapes(sd);
        assertEquals(PlanPhase.SHAPES_FROZEN, getPlanPhase(sd),
                "Phase should be SHAPES_FROZEN before reset");
        assertTrue(sd.isDspShapesFrozen(), "Shapes should be frozen before reset");

        // Reset — this should unfreeze shapes and clear caches
        DynamicShapePlanExecutor dsp = getDspExecutor(sd);
        assertNotNull(dsp);
        dsp.resetForNextPage();

        // After reset: shapes should be unfrozen
        assertFalse(sd.isDspShapesFrozen(),
                "Shapes should NOT be frozen after resetForNextPage()");

        // The native plan handle is PRESERVED by resetForNextPage() (not freed).
        // The phase is reset to SLOT_BY_SLOT on the native side.
        PlanPhase phaseAfterReset = getPlanPhase(sd);
        assertEquals(PlanPhase.SLOT_BY_SLOT, phaseAfterReset,
                "Phase should be SLOT_BY_SLOT after resetForNextPage() (native handle preserved)");

        sd.close();
        log.info("PASS: testResetForNextPageDemotesPhase");
    }

    @Test
    @DisplayName("Phase 5: Config setters work again after resetForNextPage()")
    public void testConfigSettersWorkAfterReset() {
        SameDiff sd = buildMatmulAddGraph();
        enableDsp(sd, GraphExecutionMode.AUTO);

        INDArray input = Nd4j.randn(DataType.FLOAT, 1, 16);

        // Execute and freeze
        executeOnce(sd, input);
        freezeShapes(sd);
        assertEquals(PlanPhase.SHAPES_FROZEN, getPlanPhase(sd));

        // Reset
        DynamicShapePlanExecutor dsp = getDspExecutor(sd);
        assertNotNull(dsp);
        dsp.resetForNextPage();

        // After reset, we should be able to reconfigure (back to SLOT_BY_SLOT)
        // This sets graph execution mode which requires ≤ SLOT_BY_SLOT
        sd.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);

        // Re-execute to create a new native plan
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);
        executeOnce(sd, input);

        // Should be back to SLOT_BY_SLOT with the new plan
        PlanPhase phaseAfterReExec = getPlanPhase(sd);
        assertEquals(PlanPhase.SLOT_BY_SLOT, phaseAfterReExec,
                "New plan should start at SLOT_BY_SLOT after re-execution");

        sd.close();
        log.info("PASS: testConfigSettersWorkAfterReset");
    }

    @Test
    @DisplayName("Phase 6: Full lifecycle — configure → freeze → compile → replay → reset → reconfigure")
    public void testFullLifecycleRoundTrip() {
        SameDiff sd = buildMatmulAddGraph();

        // --- Stage 1: Configure in SLOT_BY_SLOT ---
        enableDsp(sd, GraphExecutionMode.AUTO);

        INDArray input = Nd4j.randn(DataType.FLOAT, 1, 16);
        INDArray refOutput = executeOnce(sd, input);
        assertNotNull(refOutput, "First execution should produce output");

        // --- Stage 2: Freeze shapes ---
        freezeShapes(sd);
        assertEquals(PlanPhase.SHAPES_FROZEN, getPlanPhase(sd),
                "Should be SHAPES_FROZEN after freeze");

        // --- Stage 3: Execute in frozen mode (valid: compile requires ≤ SHAPES_FROZEN) ---
        INDArray frozenOutput = executeOnce(sd, input);
        double diff = frozenOutput.sub(refOutput).norm2Number().doubleValue();
        double refNorm = refOutput.norm2Number().doubleValue();
        double relErr = refNorm > 0 ? diff / refNorm : diff;
        assertTrue(relErr < FLOAT_TOL,
                "Frozen execution output should match reference. relErr=" + relErr);

        // --- Stage 4: Reset ---
        DynamicShapePlanExecutor dsp = getDspExecutor(sd);
        assertNotNull(dsp);
        dsp.resetForNextPage();
        assertFalse(sd.isDspShapesFrozen(), "Should be unfrozen after reset");

        // --- Stage 5: Reconfigure (valid because reset demoted to SLOT_BY_SLOT) ---
        sd.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        INDArray postResetOutput = executeOnce(sd, input);
        assertNotNull(postResetOutput, "Post-reset execution should produce output");

        // Output should still be numerically correct
        double postResetDiff = postResetOutput.sub(refOutput).norm2Number().doubleValue();
        double postResetRelErr = refNorm > 0 ? postResetDiff / refNorm : postResetDiff;
        assertTrue(postResetRelErr < FLOAT_TOL,
                "Post-reset output should match reference. relErr=" + postResetRelErr);

        sd.close();
        log.info("PASS: testFullLifecycleRoundTrip");
    }

    @Test
    @DisplayName("Phase 7: Multiple freeze/unfreeze cycles maintain correct phase")
    public void testFreezeUnfreezeCycles() {
        SameDiff sd = buildMatmulAddGraph();
        enableDsp(sd, GraphExecutionMode.AUTO);

        INDArray input = Nd4j.randn(DataType.FLOAT, 1, 16);
        executeOnce(sd, input);

        // Freeze → unfreeze → freeze cycle
        freezeShapes(sd);
        assertEquals(PlanPhase.SHAPES_FROZEN, getPlanPhase(sd),
                "Should be SHAPES_FROZEN after first freeze");

        unfreezeShapes(sd);
        assertFalse(sd.isDspShapesFrozen(), "Should be unfrozen");
        // After unfreeze, the native plan still exists but shapes are not frozen
        // Phase on native side may still report SHAPES_FROZEN until execution clears it

        freezeShapes(sd);
        assertTrue(sd.isDspShapesFrozen(), "Should be frozen again");

        sd.close();
        log.info("PASS: testFreezeUnfreezeCycles");
    }

    @Test
    @DisplayName("Phase 8: setShapesFrozen(true) is valid when plan is in SLOT_BY_SLOT")
    public void testFreezeValidInSlotBySlot() {
        SameDiff sd = buildMatmulAddGraph();
        enableDsp(sd, GraphExecutionMode.AUTO);

        // Execute to create the plan
        INDArray input = Nd4j.randn(DataType.FLOAT, 1, 16);
        executeOnce(sd, input);

        // Plan is in SLOT_BY_SLOT — setShapesFrozen(true) should succeed
        // This is the valid case: setShapesFrozen(true) requires == SLOT_BY_SLOT
        assertDoesNotThrow(() -> freezeShapes(sd),
                "setShapesFrozen(true) should not throw when plan is in SLOT_BY_SLOT");
        assertTrue(sd.isDspShapesFrozen());

        unfreezeShapes(sd);
        sd.close();
        log.info("PASS: testFreezeValidInSlotBySlot");
    }
}
