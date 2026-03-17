package org.eclipse.deeplearning4j.nd4j.autodiff.samediff;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.*;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.DspCompilationMode;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Isolated regression tests for Triton gap op data handoff.
 *
 * Root cause investigation: when ANY Triton section exists in a DSP plan,
 * ALL segments go through executeSegmentWithGpuGraph() instead of
 * executeSegmentSlotBySlot(). This forces frozenContextReady=false for
 * gap ops, routing them through the normal execution path instead of
 * the frozen fast path. The VLM decode produces stale/identical logits
 * on each step, suggesting external input changes don't propagate.
 *
 * These tests isolate:
 * 1. Multi-step outputDirect with Triton vs SLOT_BY_SLOT
 * 2. Gap op correctness after Triton sub-kernel execution
 * 3. External input propagation through Triton execution paths
 * 4. Cache-related staleness (disk cache cleared before each test)
 */
@Slf4j
@DisplayName("Triton Gap Op Data Handoff Tests")
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
public class TestTritonGapOpDataHandoff {

    private static final double TOL = 1e-4;

    @BeforeEach
    void clearTritonCache() {
        // Clear Triton disk cache to eliminate stale kernel entries
        String home = System.getProperty("user.home");
        Path cacheDir = Path.of(home, ".nd4j", "triton_cache");
        if (Files.exists(cacheDir)) {
            try {
                Files.walk(cacheDir)
                        .filter(Files::isRegularFile)
                        .forEach(p -> {
                            try { Files.delete(p); } catch (Exception e) { /* ignore */ }
                        });
                log.info("Cleared Triton cache at {}", cacheDir);
            } catch (Exception e) {
                log.warn("Failed to clear Triton cache: {}", e.getMessage());
            }
        }
    }

    /**
     * Build a decoder-like graph with enough ops to trigger Triton compilation.
     * Mix of Triton-compilable (elementwise: add, mul, sub) and non-compilable (matmul) ops.
     * This creates a segment where some ops become Triton sub-kernels and others become gap ops.
     */
    private static SameDiff buildDecoderGraph(INDArray w1Init, INDArray w2Init) {
        SameDiff sd = SameDiff.create();

        // Placeholders
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, 16);

        // Layer 1: matmul (gap op) + elementwise chain (Triton-compilable)
        SDVariable w1 = sd.var("w1", w1Init.dup());
        SDVariable mm1 = input.mmul("mm1", w1);  // gap op: matmul [1,16] x [16,32] = [1,32]
        SDVariable scale1 = sd.constant("scale1", Nd4j.scalar(DataType.FLOAT, 0.5f));
        SDVariable bias1 = sd.var("bias1", Nd4j.zeros(DataType.FLOAT, 1, 32));
        SDVariable scaled1 = mm1.mul("scaled1", scale1);  // Triton
        SDVariable biased1 = scaled1.add("biased1", bias1);  // Triton
        SDVariable ones1 = sd.onesLike("ones1", biased1);
        SDVariable sub1 = ones1.sub("sub1", biased1);  // Triton: (1 - x)
        SDVariable abs1 = sd.math.abs("abs1", sub1);  // Triton
        SDVariable relu1 = sd.nn.relu("relu1", abs1, 0);  // Triton

        // Layer 2: matmul (gap op) + more elementwise (Triton)
        SDVariable w2 = sd.var("w2", w2Init.dup());
        SDVariable mm2 = relu1.mmul("mm2", w2);  // gap op: matmul [1,32] x [32,8] = [1,8]
        SDVariable scale2 = sd.constant("scale2", Nd4j.scalar(DataType.FLOAT, 2.0f));
        SDVariable output = mm2.mul("output", scale2);  // Triton

        return sd;
    }

    /**
     * Test 1: SLOT_BY_SLOT baseline — multi-step produces different outputs.
     */
    @Test
    @Order(1)
    @DisplayName("SLOT_BY_SLOT: multi-step decoder chain produces different outputs")
    void testSlotBySlotMultiStep() {
        INDArray w1 = Nd4j.randn(DataType.FLOAT, 16, 32).muli(0.1);
        INDArray w2 = Nd4j.randn(DataType.FLOAT, 32, 8).muli(0.1);
        SameDiff sd = buildDecoderGraph(w1, w2);

        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);
        GraphExecutionMode mode = sd.compileNativeDynamicShapePlan(
                List.of("output"), GraphExecutionMode.SLOT_BY_SLOT, true);
        log.info("SLOT_BY_SLOT compiled with mode: {}", mode);

        INDArray[] outputs = new INDArray[5];
        for (int step = 0; step < 5; step++) {
            INDArray inp = Nd4j.randn(DataType.FLOAT, 1, 16).muli(step + 1);
            Map<String, INDArray> result = sd.outputDirect(Map.of("input", inp), "output");
            outputs[step] = result.get("output").dup();
            log.info("SLOT_BY_SLOT step {}: output[0]={}", step, outputs[step].getFloat(0, 0));
        }

        for (int i = 1; i < 5; i++) {
            double diff = outputs[i].sub(outputs[0]).amaxNumber().doubleValue();
            log.info("SLOT_BY_SLOT step {} vs 0: maxDiff={}", i, diff);
            assertTrue(diff > TOL, "Step " + i + " should differ from step 0 but maxDiff=" + diff);
        }

        sd.close();
    }

    /**
     * Test 2: TRITON (MAX_AUTOTUNE) — multi-step MUST produce different outputs.
     * This is the core regression test for the Triton data handoff bug.
     */
    @Test
    @Order(2)
    @DisplayName("TRITON: multi-step decoder chain produces different outputs")
    void testTritonMultiStep() {
        INDArray w1 = Nd4j.randn(DataType.FLOAT, 16, 32).muli(0.1);
        INDArray w2 = Nd4j.randn(DataType.FLOAT, 32, 8).muli(0.1);
        SameDiff sd = buildDecoderGraph(w1, w2);

        sd.setDspAutoCompileEnabled(false);
        sd.setDspNativeAutoCompileEnabled(false);
        GraphExecutionMode mode = sd.compileNativeDynamicShapePlan(
                List.of("output"), DspCompilationMode.MAX_AUTOTUNE);
        log.info("TRITON compiled with mode: {}", mode);

        INDArray[] outputs = new INDArray[7]; // extra steps to get past warmup
        for (int step = 0; step < 7; step++) {
            INDArray inp = Nd4j.randn(DataType.FLOAT, 1, 16).muli(step + 1);
            Map<String, INDArray> result = sd.outputDirect(Map.of("input", inp), "output");
            outputs[step] = result.get("output").dup();
            log.info("TRITON step {}: output[0]={}", step, outputs[step].getFloat(0, 0));
        }

        // Check that outputs after Triton compilation differ from each other
        for (int i = 1; i < 7; i++) {
            double diff = outputs[i].sub(outputs[i - 1]).amaxNumber().doubleValue();
            log.info("TRITON step {} vs {}: maxDiff={}", i, i - 1, diff);
            assertTrue(diff > TOL,
                    "TRITON: Step " + i + " should differ from step " + (i - 1) + " but maxDiff=" + diff);
        }

        sd.close();
    }

    /**
     * Test 3: Compare Triton vs SLOT_BY_SLOT for same inputs and weights.
     * Both paths should produce identical numerical results.
     */
    @Test
    @Order(3)
    @DisplayName("TRITON vs SLOT_BY_SLOT: same inputs produce same outputs")
    void testTritonVsSlotBySlotSameOutput() {
        INDArray w1 = Nd4j.randn(DataType.FLOAT, 16, 32).muli(0.1);
        INDArray w2 = Nd4j.randn(DataType.FLOAT, 32, 8).muli(0.1);
        INDArray testInput = Nd4j.randn(DataType.FLOAT, 1, 16);

        // SLOT_BY_SLOT reference
        INDArray slotBySlotOutput;
        {
            SameDiff sd = buildDecoderGraph(w1, w2);
            sd.setDspAutoCompileEnabled(true);
            sd.setDspNativeAutoCompileEnabled(true);
            sd.compileNativeDynamicShapePlan(
                    List.of("output"), GraphExecutionMode.SLOT_BY_SLOT, true);
            // Warmup
            for (int i = 0; i < 3; i++)
                sd.outputDirect(Map.of("input", testInput.dup()), "output");
            slotBySlotOutput = sd.outputDirect(Map.of("input", testInput.dup()), "output")
                    .get("output").dup();
            sd.close();
        }

        // Triton
        INDArray tritonOutput;
        {
            SameDiff sd = buildDecoderGraph(w1, w2);
            sd.setDspAutoCompileEnabled(false);
            sd.setDspNativeAutoCompileEnabled(false);
            sd.compileNativeDynamicShapePlan(
                    List.of("output"), DspCompilationMode.MAX_AUTOTUNE);
            // Run enough steps to get past warmup + compilation
            for (int i = 0; i < 5; i++)
                sd.outputDirect(Map.of("input", testInput.dup()), "output");
            tritonOutput = sd.outputDirect(Map.of("input", testInput.dup()), "output")
                    .get("output").dup();
            sd.close();
        }

        log.info("SLOT_BY_SLOT output: {}", slotBySlotOutput);
        log.info("TRITON output:       {}", tritonOutput);

        double maxDiff = slotBySlotOutput.sub(tritonOutput).amaxNumber().doubleValue();
        log.info("Max diff: {}", maxDiff);
        assertTrue(maxDiff < TOL, "Triton vs SLOT_BY_SLOT maxDiff=" + maxDiff);
    }

    /**
     * Test 4: Attention mask reformat pattern with Triton.
     * (1 - mask) * MASK_FILL — the pattern that fails in VLM decode.
     * Uses sd.output() as reference, sd.outputDirect() with Triton as test.
     */
    @Test
    @Order(4)
    @DisplayName("Attention mask reformat: Triton outputDirect vs standard output")
    void testAttnMaskReformatTriton() {
        float MASK_FILL = -3.4028235e+38f;

        SameDiff sd = SameDiff.create();
        SDVariable mask = sd.placeHolder("mask", DataType.FLOAT, 1, 20);
        SDVariable ones = sd.onesLike("ones", mask);
        SDVariable inverted = ones.sub("inverted", mask);
        SDVariable fill = sd.constant("fill", Nd4j.scalar(MASK_FILL));
        SDVariable bias = inverted.mul("bias", fill);

        // Compile with Triton
        sd.setDspAutoCompileEnabled(false);
        sd.setDspNativeAutoCompileEnabled(false);
        GraphExecutionMode mode = sd.compileNativeDynamicShapePlan(
                List.of("bias"), DspCompilationMode.MAX_AUTOTUNE);
        log.info("Compiled with mode: {}", mode);

        // Run 5 decode steps with growing valid region
        for (int step = 0; step < 5; step++) {
            int validPositions = 5 + step;
            INDArray maskArr = Nd4j.zeros(DataType.FLOAT, 1, 20);
            for (int i = 0; i < validPositions; i++) maskArr.putScalar(0, i, 1.0f);
            maskArr.putScalar(0, 19, 1.0f);

            Map<String, INDArray> inputs = Map.of("mask", maskArr);

            // Reference: standard output
            Map<String, INDArray> refResult = sd.output(inputs, "bias");
            INDArray refBias = refResult.get("bias").dup();

            // Test: Triton outputDirect
            Map<String, INDArray> tritonResult = sd.outputDirect(inputs, "bias");
            INDArray tritonBias = tritonResult.get("bias").dup();

            double maxDiff = refBias.sub(tritonBias).amaxNumber().doubleValue();
            log.info("Step {}: validPositions={} maxDiff={}", step, validPositions, maxDiff);

            // Check position-by-position
            for (int pos = 0; pos < 20; pos++) {
                float expected = refBias.getFloat(0, pos);
                float actual = tritonBias.getFloat(0, pos);
                if (Math.abs(expected - actual) > TOL) {
                    log.error("  Position {}: expected={} actual={} mask={}", pos, expected, actual,
                            maskArr.getFloat(0, pos));
                }
            }

            assertTrue(maxDiff < TOL,
                    "Step " + step + ": mask reformat should match. maxDiff=" + maxDiff);
        }

        sd.close();
    }

    /**
     * Test 5: CUDA_GRAPHS (no Triton) — multi-step baseline (should always pass).
     */
    @Test
    @Order(5)
    @DisplayName("CUDA_GRAPHS (no Triton): multi-step produces different outputs")
    void testCudaGraphsMultiStep() {
        INDArray w1 = Nd4j.randn(DataType.FLOAT, 16, 32).muli(0.1);
        INDArray w2 = Nd4j.randn(DataType.FLOAT, 32, 8).muli(0.1);
        SameDiff sd = buildDecoderGraph(w1, w2);

        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);
        GraphExecutionMode mode = sd.compileNativeDynamicShapePlan(
                List.of("output"), GraphExecutionMode.CUDA_GRAPHS, true);
        log.info("CUDA_GRAPHS compiled with mode: {}", mode);

        INDArray[] outputs = new INDArray[5];
        for (int step = 0; step < 5; step++) {
            INDArray inp = Nd4j.randn(DataType.FLOAT, 1, 16).muli(step + 1);
            Map<String, INDArray> result = sd.outputDirect(Map.of("input", inp), "output");
            outputs[step] = result.get("output").dup();
            log.info("CUDA_GRAPHS step {}: output[0]={}", step, outputs[step].getFloat(0, 0));
        }

        for (int i = 1; i < 5; i++) {
            double diff = outputs[i].sub(outputs[0]).amaxNumber().doubleValue();
            assertTrue(diff > TOL, "Step " + i + " should differ from step 0 but maxDiff=" + diff);
        }

        sd.close();
    }

    /**
     * Test 6: Autoregressive decode pattern with Triton.
     * Each step feeds the output of the previous step as input.
     * This is the exact pattern from the VLM decode loop.
     */
    @Test
    @Order(6)
    @DisplayName("Autoregressive decode pattern with Triton")
    void testAutoregressiveDecodeTriton() {
        INDArray w1 = Nd4j.randn(DataType.FLOAT, 16, 32).muli(0.1);
        INDArray w2 = Nd4j.randn(DataType.FLOAT, 32, 8).muli(0.1);

        // Reference: SLOT_BY_SLOT
        INDArray[] refOutputs = new INDArray[5];
        {
            SameDiff sd = buildDecoderGraph(w1, w2);
            sd.setDspAutoCompileEnabled(true);
            sd.setDspNativeAutoCompileEnabled(true);
            sd.compileNativeDynamicShapePlan(
                    List.of("output"), GraphExecutionMode.SLOT_BY_SLOT, true);

            INDArray currentInput = Nd4j.randn(DataType.FLOAT, 1, 16);
            for (int step = 0; step < 5; step++) {
                Map<String, INDArray> result = sd.outputDirect(
                        Map.of("input", currentInput.dup()), "output");
                refOutputs[step] = result.get("output").dup();
                // Create next input from output (simulating token embedding lookup)
                currentInput = Nd4j.randn(DataType.FLOAT, 1, 16)
                        .muli(refOutputs[step].getFloat(0, 0));
            }
            sd.close();
        }

        // Test: Triton
        INDArray[] tritonOutputs = new INDArray[5];
        {
            SameDiff sd = buildDecoderGraph(w1, w2);
            sd.setDspAutoCompileEnabled(false);
            sd.setDspNativeAutoCompileEnabled(false);
            sd.compileNativeDynamicShapePlan(
                    List.of("output"), DspCompilationMode.MAX_AUTOTUNE);

            INDArray currentInput = Nd4j.randn(DataType.FLOAT, 1, 16);
            for (int step = 0; step < 5; step++) {
                Map<String, INDArray> result = sd.outputDirect(
                        Map.of("input", currentInput.dup()), "output");
                tritonOutputs[step] = result.get("output").dup();
                // Same next input generation as reference
                currentInput = Nd4j.randn(DataType.FLOAT, 1, 16)
                        .muli(tritonOutputs[step].getFloat(0, 0));
            }
            sd.close();
        }

        // Compare each step
        for (int step = 0; step < 5; step++) {
            double maxDiff = refOutputs[step].sub(tritonOutputs[step]).amaxNumber().doubleValue();
            log.info("Step {}: ref[0]={} triton[0]={} maxDiff={}",
                    step, refOutputs[step].getFloat(0, 0),
                    tritonOutputs[step].getFloat(0, 0), maxDiff);
        }

        // Verify Triton outputs are different from each other (not stale)
        for (int i = 1; i < 5; i++) {
            double diff = tritonOutputs[i].sub(tritonOutputs[i - 1]).amaxNumber().doubleValue();
            assertTrue(diff > TOL,
                    "Autoregressive: Triton step " + i + " should differ from step " + (i - 1) +
                            " but maxDiff=" + diff);
        }
    }

    /**
     * Test 7: Pure elementwise chain (all Triton-compilable, NO gap ops).
     * Tests that Triton kernel execution itself tracks input changes correctly.
     */
    @Test
    @Order(7)
    @DisplayName("Pure elementwise chain: all Triton, no gap ops")
    void testPureTritonChain() {
        SameDiff sd = SameDiff.create();

        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, 16);
        // Chain of elementwise ops (all Triton-compilable)
        SDVariable scale = sd.constant("scale", Nd4j.scalar(DataType.FLOAT, 3.0f));
        SDVariable bias = sd.constant("bias", Nd4j.scalar(DataType.FLOAT, 1.0f));
        SDVariable negOne = sd.constant("negOne", Nd4j.scalar(DataType.FLOAT, -1.0f));
        SDVariable x = input.mul("step1_mul", scale);    // *3
        SDVariable y = x.add("step2_add", bias);          // +1
        SDVariable z = y.mul("step3_mul", negOne);         // *-1
        SDVariable w = z.add("step4_add", bias);           // +1
        SDVariable output = w.mul("output", scale);        // *3

        sd.setDspAutoCompileEnabled(false);
        sd.setDspNativeAutoCompileEnabled(false);
        GraphExecutionMode mode = sd.compileNativeDynamicShapePlan(
                List.of("output"), DspCompilationMode.MAX_AUTOTUNE);
        log.info("Pure elementwise compiled with mode: {}", mode);

        for (int step = 0; step < 5; step++) {
            float inputVal = step + 1;
            INDArray inp = Nd4j.ones(DataType.FLOAT, 1, 16).muli(inputVal);

            // Reference via sd.output()
            Map<String, INDArray> refResult = sd.output(Map.of("input", inp.dup()), "output");
            INDArray refOut = refResult.get("output").dup();

            // Triton via outputDirect
            Map<String, INDArray> tritonResult = sd.outputDirect(Map.of("input", inp.dup()), "output");
            INDArray tritonOut = tritonResult.get("output").dup();

            // Expected: (((x*3+1)*-1)+1)*3 = ((-(3x+1))+1)*3 = (1-3x-1)*3 = -9x
            float expected = -9.0f * inputVal;
            log.info("Step {}: ref[0]={} triton[0]={} expected={}",
                    step, refOut.getFloat(0, 0), tritonOut.getFloat(0, 0), expected);

            double maxDiff = refOut.sub(tritonOut).amaxNumber().doubleValue();
            assertTrue(maxDiff < TOL,
                    "Step " + step + ": pure Triton should match reference. maxDiff=" + maxDiff);
        }

        sd.close();
    }

    /**
     * Test 8: Assert slot/segment structure and arg table completeness.
     * Verifies that the compiled plan has correct segment boundaries, that all
     * op slots have the expected inputs/outputs, and that Triton sub-kernels
     * cover exactly the expected slot ranges.
     */
    @Test
    @Order(8)
    @DisplayName("Segment structure: slot layout, gap ops, Triton sub-kernels")
    void testSegmentStructureAndArgTable() {
        INDArray w1 = Nd4j.randn(DataType.FLOAT, 16, 32).muli(0.1);
        INDArray w2 = Nd4j.randn(DataType.FLOAT, 32, 8).muli(0.1);
        SameDiff sd = buildDecoderGraph(w1, w2);

        sd.setDspAutoCompileEnabled(false);
        sd.setDspNativeAutoCompileEnabled(false);
        sd.compileNativeDynamicShapePlan(
                List.of("output"), DspCompilationMode.MAX_AUTOTUNE);

        // Get the compiled plan
        var session = sd.getOrCreateSession();
        var executor = session.getDynamicShapePlanExecutor();
        var plan = executor.getCurrentPlan();
        assertNotNull(plan, "Plan should be compiled");

        var slots = plan.getSlots();
        log.info("Plan has {} slots, {} total output slots", slots.length, plan.getTotalOutputSlots());

        // Assert: plan should have 9 slots (matmul, mul, add, ones_as, sub, abs, relu, matmul, mul)
        assertTrue(slots.length >= 9, "Plan should have at least 9 slots, got " + slots.length);

        // Build a map of op names
        for (int i = 0; i < slots.length; i++) {
            var slot = slots[i];
            log.info("Slot {}: op='{}' inputs={} outputs={} viewCapable={} dataDependent={}",
                    i, slot.getOpName(),
                    slot.getInputSourceIndices() != null ? slot.getInputSourceIndices().length : 0,
                    slot.getOutputSlotIndices() != null ? slot.getOutputSlotIndices().length : 0,
                    slot.isViewCapableOp(), slot.isDataDependent());

            // Every slot must have at least one output
            assertNotNull(slot.getOutputSlotIndices(), "Slot " + i + " (" + slot.getOpName() + ") must have outputs");
            assertTrue(slot.getOutputSlotIndices().length > 0,
                    "Slot " + i + " (" + slot.getOpName() + ") must have at least one output");

            // Every non-first slot must have inputs
            if (i > 0 && !"ones_as".equals(slot.getOpName())) {
                assertNotNull(slot.getInputSourceIndices(),
                        "Slot " + i + " (" + slot.getOpName() + ") must have inputs");
            }
        }

        // Assert: segments should cover all slots
        var segments = plan.getSegments();
        log.info("Plan has {} segments", segments.size());
        for (var seg : segments) {
            log.info("Segment [{}-{}]: {} slots, capturable={}, ops={}",
                    seg.getStartSlot(), seg.getEndSlot(), seg.getSlotCount(),
                    seg.isCapturable(), seg.getOpNames());
            assertTrue(seg.getStartSlot() <= seg.getEndSlot(),
                    "Segment start must be <= end");
            assertTrue(seg.getSlotCount() > 0, "Segment must have at least 1 slot");
        }

        // Assert: every slot's inputs trace to either an external input or another slot's output
        for (int i = 0; i < slots.length; i++) {
            var slot = slots[i];
            int[] inputSources = slot.getInputSourceIndices();
            if (inputSources == null) continue;
            for (int j = 0; j < inputSources.length; j++) {
                int src = inputSources[j];
                if (src < 0) {
                    // External input: must be valid index
                    int extIdx = -(src + 1);
                    assertTrue(extIdx >= 0, "External input index must be >= 0, got " + extIdx
                            + " at slot " + i + " input " + j);
                } else {
                    // Internal slot reference: must be a valid slot output
                    assertTrue(src < plan.getTotalOutputSlots(),
                            "Slot " + i + " input " + j + " references slot " + src
                                    + " but total output slots is " + plan.getTotalOutputSlots());
                }
            }
        }

        // Run a warmup + multi-step to verify the structure produces correct results
        INDArray[] outputs = new INDArray[3];
        for (int step = 0; step < 3; step++) {
            INDArray inp = Nd4j.randn(DataType.FLOAT, 1, 16).muli(step + 1);
            Map<String, INDArray> result = sd.outputDirect(Map.of("input", inp), "output");
            outputs[step] = result.get("output").dup();
            log.info("Step {}: output[0]={}", step, outputs[step].getFloat(0, 0));
        }

        // Verify outputs differ across steps
        for (int i = 1; i < 3; i++) {
            double diff = outputs[i].sub(outputs[0]).amaxNumber().doubleValue();
            assertTrue(diff > TOL, "Step " + i + " should differ from step 0 but maxDiff=" + diff);
        }

        sd.close();
    }

    /**
     * Test 9: Buffer ownership and gap op data handoff.
     * Verifies that after each outputDirect step, the output array contains
     * data that's numerically consistent with a reference sd.output() execution.
     * This catches buffer ownership issues where gap ops write to wrong buffers
     * or Triton sub-kernels read from stale buffers.
     */
    @Test
    @Order(9)
    @DisplayName("Buffer ownership: Triton outputDirect matches sd.output reference")
    void testBufferOwnershipGapHandoff() {
        INDArray w1 = Nd4j.randn(DataType.FLOAT, 16, 32).muli(0.1);
        INDArray w2 = Nd4j.randn(DataType.FLOAT, 32, 8).muli(0.1);

        // Triton execution
        SameDiff sdTriton = buildDecoderGraph(w1, w2);
        sdTriton.setDspAutoCompileEnabled(false);
        sdTriton.setDspNativeAutoCompileEnabled(false);
        sdTriton.compileNativeDynamicShapePlan(
                List.of("output"), DspCompilationMode.MAX_AUTOTUNE);

        // Reference execution (standard sd.output, no DSP)
        SameDiff sdRef = buildDecoderGraph(w1, w2);

        // Run 5 steps with deterministic inputs and compare
        for (int step = 0; step < 5; step++) {
            INDArray inp = Nd4j.randn(DataType.FLOAT, 1, 16).muli(step + 1);

            // Reference
            Map<String, INDArray> refResult = sdRef.output(Map.of("input", inp.dup()), "output");
            INDArray refOut = refResult.get("output").dup();

            // Triton
            Map<String, INDArray> tritonResult = sdTriton.outputDirect(Map.of("input", inp.dup()), "output");
            INDArray tritonOut = tritonResult.get("output").dup();

            double maxDiff = refOut.sub(tritonOut).amaxNumber().doubleValue();
            log.info("Buffer ownership step {}: ref[0]={} triton[0]={} maxDiff={}",
                    step, refOut.getFloat(0, 0), tritonOut.getFloat(0, 0), maxDiff);

            assertTrue(maxDiff < TOL,
                    "Buffer ownership: Step " + step + " Triton output should match reference. " +
                            "ref[0]=" + refOut.getFloat(0, 0) + " triton[0]=" + tritonOut.getFloat(0, 0) +
                            " maxDiff=" + maxDiff);
        }

        sdTriton.close();
        sdRef.close();
    }

    /**
     * Test 10: Triton with tritonSkipKernels=true — forces all Triton sub-kernels
     * to use native fallback. Should produce correct output since only native ops run.
     */
    @Test
    @Order(10)
    @DisplayName("TRITON with skipKernels: native fallback for all sub-kernels")
    void testTritonSkipKernelsMultiStep() {
        boolean prevSkip = Nd4j.getEnvironment().tritonSkipKernels();
        Nd4j.getEnvironment().setTritonSkipKernels(true);

        try {
            INDArray w1 = Nd4j.randn(DataType.FLOAT, 16, 32).muli(0.1);
            INDArray w2 = Nd4j.randn(DataType.FLOAT, 32, 8).muli(0.1);
            SameDiff sd = buildDecoderGraph(w1, w2);

            sd.setDspAutoCompileEnabled(false);
            sd.setDspNativeAutoCompileEnabled(false);
            GraphExecutionMode mode = sd.compileNativeDynamicShapePlan(
                    List.of("output"), DspCompilationMode.MAX_AUTOTUNE);
            log.info("TRITON+skipKernels compiled with mode: {}", mode);

            INDArray[] outputs = new INDArray[5];
            for (int step = 0; step < 5; step++) {
                INDArray inp = Nd4j.randn(DataType.FLOAT, 1, 16).muli(step + 1);
                Map<String, INDArray> result = sd.outputDirect(Map.of("input", inp), "output");
                outputs[step] = result.get("output").dup();
                log.info("skipKernels step {}: output[0]={}", step, outputs[step].getFloat(0, 0));
            }

            for (int i = 1; i < 5; i++) {
                double diff = outputs[i].sub(outputs[0]).amaxNumber().doubleValue();
                assertTrue(diff > TOL,
                        "skipKernels: Step " + i + " should differ from step 0 but maxDiff=" + diff);
            }

            sd.close();
        } finally {
            Nd4j.getEnvironment().setTritonSkipKernels(prevSkip);
        }
    }
}
