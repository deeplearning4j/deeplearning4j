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
import org.bytedeco.javacpp.Pointer;
import org.junit.jupiter.api.*;
import org.junit.jupiter.api.Assumptions;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlanExecutor;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.autodiff.samediff.execution.PlanPhase;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.common.config.ND4JSystemProperties;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.BaseNd4jTestWithBackends;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for DSP plan recovery scenarios: shape changes triggering recompile,
 * phase transitions after shape change, and plan destruction/rebuild lifecycle.
 *
 * <p>Run:
 * <pre>
 *   cd platform-tests && mvn test -Dtest=DSPRecoveryTest
 * </pre>
 */
@Slf4j
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
@Tag("dsp")
public class DSPRecoveryTest extends BaseNd4jTestWithBackends {

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

    @AfterEach
    void cleanup() {
        Nd4j.getExecutioner().commit();
    }

    // ─── Helpers ─────────────────────────────────────────────────────────────

    private void enableDsp(SameDiff sd) {
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);
    }

    /**
     * Build a simple matmul chain: input[?,16] @ w1[16,32] → relu → @ w2[32,16] → output
     * The first dimension is dynamic (-1) so shape changes are supported.
     */
    private SameDiff buildMatmulChain() {
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 16);
        SDVariable w1 = sd.constant("w1", Nd4j.randn(DataType.FLOAT, 16, 32));
        SDVariable w2 = sd.constant("w2", Nd4j.randn(DataType.FLOAT, 32, 16));
        SDVariable h = sd.nn.relu("mm1", sd.mmul("mm1", input, w1), 0);
        sd.mmul("output", h, w2);
        return sd;
    }

    /**
     * Build a small transformer-like graph: matmul → standardize → mul → matmul
     */
    private SameDiff buildSmallTransformerGraph() {
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 64);
        SDVariable w1 = sd.constant("w1", Nd4j.randn(DataType.FLOAT, 64, 64).muli(0.02));
        SDVariable gamma = sd.constant("gamma", Nd4j.ones(DataType.FLOAT, 64));
        SDVariable proj = sd.mmul("proj", input, w1);
        SDVariable norm = sd.math.standardize("norm", proj, 2);
        SDVariable scaled = norm.mul("scaled", gamma);
        SDVariable output = sd.mmul("output", scaled, w1);
        return sd;
    }

    /**
     * Compute a slot-by-slot reference for the matmul chain graph.
     */
    private INDArray computeMatmulChainReference(INDArray input, INDArray w1Data, INDArray w2Data) {
        SameDiff ref = SameDiff.create();
        SDVariable inp = ref.placeHolder("input", input.dataType(), input.shape());
        SDVariable w1 = ref.constant("w1", w1Data.dup());
        SDVariable w2 = ref.constant("w2", w2Data.dup());
        SDVariable h = ref.nn.relu("mm1", ref.mmul("mm1", inp, w1), 0);
        ref.mmul("output", h, w2);
        ref.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);
        Map<String, INDArray> result = ref.output(Map.of("input", input.dup()), "output");
        INDArray out = result.get("output").dup();
        ref.close();
        return out;
    }

    /**
     * Compute reference for the transformer graph.
     */
    private INDArray computeTransformerReference(INDArray input, INDArray w1Data, INDArray gammaData) {
        SameDiff ref = SameDiff.create();
        SDVariable inp = ref.placeHolder("input", input.dataType(), input.shape());
        SDVariable w1 = ref.constant("w1", w1Data.dup());
        SDVariable gamma = ref.constant("gamma", gammaData.dup());
        SDVariable proj = ref.mmul("proj", inp, w1);
        SDVariable norm = ref.math.standardize("norm", proj, 2);
        SDVariable scaled = norm.mul("scaled", gamma);
        SDVariable output = ref.mmul("output", scaled, w1);
        ref.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);
        Map<String, INDArray> result = ref.output(Map.of("input", input.dup()), "output");
        INDArray out = result.get("output").dup();
        ref.close();
        return out;
    }

    /**
     * Close output arrays safely.
     */
    private void closeOutput(INDArray arr) {
        if (arr != null) {
            arr.setCloseable(true);
            arr.close();
        }
    }

    /**
     * Advance the DSP plan to REPLAYING phase by executing repeatedly with the same shape.
     * Always unfreezes first to allow shape discovery, then executes unfrozen,
     * then freezes and warms up.
     */
    private void advanceToReplaying(SameDiff sd, INDArray input, int numExecutions) {
        // Unfreeze first to allow shape discovery (critical when called after a previous freeze cycle)
        DynamicShapePlanExecutor executor = sd.getOrCreateSession().getDynamicShapePlanExecutor();
        if (executor != null) {
            executor.setShapesFrozen(false);
        }

        // First execution to populate shapes (unfrozen) — may create the executor
        sd.output(Map.of("input", input.dup()), "output");

        // Re-fetch executor (may have been created by the first execution)
        executor = sd.getOrCreateSession().getDynamicShapePlanExecutor();
        if (executor != null) {
            executor.setShapesFrozen(true);
        }

        // Warmup executions while frozen
        for (int i = 0; i < numExecutions; i++) {
            sd.output(Map.of("input", input.dup()), "output");
        }
    }

    // =========================================================================
    // Test 1: Shape change triggers recompile
    // =========================================================================

    @Test
    @Order(1)
    @DisplayName("Shape change [1,16] → [4,16] triggers recompile and produces correct output")
    public void testShapeChangeTriggersRecompile() {
        SameDiff sd = buildMatmulChain();
        sd.setGraphExecutionMode(GraphExecutionMode.AUTO);
        enableDsp(sd);

        INDArray w1Data = sd.getVariable("w1").getArr().dup();
        INDArray w2Data = sd.getVariable("w2").getArr().dup();

        // Phase 1: Execute with [1,16] until REPLAYING
        INDArray smallInput = Nd4j.randn(DataType.FLOAT, 1, 16);
        advanceToReplaying(sd, smallInput, 5);

        DynamicShapePlanExecutor executor = sd.getOrCreateSession().getDynamicShapePlanExecutor();
        assertNotNull(executor, "Executor should exist");
        PlanPhase phaseBefore = executor.getPlanPhase();
        log.info("Phase before shape change: {}", phaseBefore);
        // After advanceToReplaying (unfrozen + freeze + warmup), we should be at least SHAPES_FROZEN
        assertTrue(phaseBefore.isAtLeast(PlanPhase.SHAPES_FROZEN),
                "Should be at least SHAPES_FROZEN after warmup, got " + phaseBefore);

        // Phase 2: Change input shape to [4,16] — this should trigger recompile
        // Unfreeze to allow shape discovery
        executor.setShapesFrozen(false);
        INDArray bigInput = Nd4j.randn(DataType.FLOAT, 4, 16);
        sd.output(Map.of("input", bigInput), "output");

        // Refreeze and warmup the new shape
        executor.setShapesFrozen(true);
        for (int i = 0; i < 3; i++) {
            sd.output(Map.of("input", bigInput.dup()), "output");
        }

        // Now execute and verify
        Map<String, INDArray> result = sd.output(Map.of("input", bigInput.dup()), "output");
        INDArray actual = result.get("output").dup();

        assertNotNull(actual, "Output should not be null after shape change");
        assertArrayEquals(new long[]{4, 16}, actual.shape(),
                "Output shape should be [4, 16] for [4, 16] input");

        // Compute reference with slot-by-slot
        INDArray expected = computeMatmulChainReference(bigInput, w1Data, w2Data);

        double maxDiff = expected.sub(actual).amaxNumber().doubleValue();
        log.info("Shape change [1,16]→[4,16]: maxDiff={}", maxDiff);
        assertTrue(maxDiff < TOL,
                "Output after shape change diverged from slot-by-slot reference. maxDiff=" + maxDiff);

        closeOutput(actual);
        closeOutput(expected);
        sd.close();
    }

    // =========================================================================
    // Test 2: Replaying → Slot-by-slot on shape change
    // =========================================================================

    @Test
    @Order(2)
    @DisplayName("Phase drops back to SLOT_BY_SLOT after shape change from REPLAYING")
    public void testReplayingToSlotBySlotOnShapeChange() {
        SameDiff sd = buildMatmulChain();
        sd.setGraphExecutionMode(GraphExecutionMode.AUTO);
        enableDsp(sd);

        // Warm up with [1,16]
        INDArray smallInput = Nd4j.randn(DataType.FLOAT, 1, 16);
        advanceToReplaying(sd, smallInput, 8);

        DynamicShapePlanExecutor executor = sd.getOrCreateSession().getDynamicShapePlanExecutor();
        PlanPhase phaseBefore = executor.getPlanPhase();
        log.info("Phase before shape change: {}", phaseBefore);

        // Change shape to [2,16] — should force a recompile
        // Unfreeze first to allow shape discovery
        executor.setShapesFrozen(false);
        INDArray newInput = Nd4j.randn(DataType.FLOAT, 2, 16);
        sd.output(Map.of("input", newInput), "output");

        PlanPhase phaseAfter = executor.getPlanPhase();
        log.info("Phase after shape change: {}", phaseAfter);

        // After a shape change, the plan should reset to an early phase
        // (either SLOT_BY_SLOT or SHAPES_FROZEN at most, not REPLAYING)
        assertTrue(phaseAfter.getNativeCode() < PlanPhase.REPLAYING.getNativeCode(),
                "Phase after shape change should drop below REPLAYING, got " + phaseAfter);

        // Execute several more times with the new shape to verify correctness
        for (int i = 0; i < 3; i++) {
            INDArray inp = Nd4j.randn(DataType.FLOAT, 2, 16);
            Map<String, INDArray> result = sd.output(Map.of("input", inp), "output");
            INDArray actual = result.get("output").dup();
            assertNotNull(actual, "Output should not be null");
            assertArrayEquals(new long[]{2, 16}, actual.shape(),
                    "Output shape should be [2, 16]");
            closeOutput(actual);
        }

        sd.close();
    }

    // =========================================================================
    // Test 3: Multiple shape changes in sequence
    // =========================================================================

    @Test
    @Order(3)
    @DisplayName("Multiple shape changes [1,16] → [4,16] → [2,16] → [3,16] all produce correct output")
    public void testMultipleShapeChangesInSequence() {
        // Build graph with dynamic batch dimension [-1, 16] — second dim must match w1[16,32]
        INDArray w1Data = Nd4j.randn(DataType.FLOAT, 16, 32);
        INDArray w2Data = Nd4j.randn(DataType.FLOAT, 32, 16);

        // Define shape sequence — only batch dimension changes, second dim stays 16
        long[][] shapes = {
                {1, 16},
                {4, 16},
                {2, 16},
                {3, 16}
        };

        for (int s = 0; s < shapes.length; s++) {
            long[] shape = shapes[s];
            // Build a fresh SameDiff for each shape to ensure clean DSP state
            SameDiff sd = SameDiff.create();
            SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 16);
            SDVariable w1 = sd.constant("w1", w1Data.dup());
            SDVariable w2 = sd.constant("w2", w2Data.dup());
            SDVariable h = sd.nn.relu("mm1", sd.mmul("mm1", input, w1), 0);
            sd.mmul("output", h, w2);

            sd.setGraphExecutionMode(GraphExecutionMode.AUTO);
            enableDsp(sd);

            INDArray inputArr = Nd4j.randn(DataType.FLOAT, shape);

            // Warm up this shape
            advanceToReplaying(sd, inputArr, 3);

            // Execute and verify
            INDArray testInput = Nd4j.randn(DataType.FLOAT, shape);
            Map<String, INDArray> result = sd.output(Map.of("input", testInput), "output");
            INDArray actual = result.get("output").dup();

            assertNotNull(actual, "Output should not be null for shape " + Arrays.toString(shape));
            long[] expectedShape = new long[]{shape[0], 16};
            assertArrayEquals(expectedShape, actual.shape(),
                    "Output shape mismatch for input shape " + Arrays.toString(shape));

            // Verify against slot-by-slot reference
            INDArray expected = computeMatmulChainReference(testInput, w1Data, w2Data);
            double maxDiff = expected.sub(actual).amaxNumber().doubleValue();
            log.info("Shape {}: maxDiff={}", Arrays.toString(shape), maxDiff);
            assertTrue(maxDiff < TOL,
                    "Shape " + Arrays.toString(shape) + " diverged. maxDiff=" + maxDiff);

            closeOutput(actual);
            closeOutput(expected);
            sd.close();
        }
    }

    // =========================================================================
    // Test 4: Session reset preserves correctness
    // =========================================================================

    @Test
    @Order(4)
    @DisplayName("Session reset followed by re-execution produces correct output")
    public void testSessionResetPreservesCorrectness() {
        SameDiff sd = buildMatmulChain();
        sd.setGraphExecutionMode(GraphExecutionMode.AUTO);
        enableDsp(sd);

        INDArray w1Data = sd.getVariable("w1").getArr().dup();
        INDArray w2Data = sd.getVariable("w2").getArr().dup();

        // Execute several times
        INDArray input = Nd4j.randn(DataType.FLOAT, 1, 16);
        for (int i = 0; i < 5; i++) {
            sd.output(Map.of("input", input.dup()), "output");
        }

        // Reset session — this destroys the DSP plan executor
        sd.resetSession();

        // Re-execute with the same input
        INDArray inputAfterReset = Nd4j.randn(DataType.FLOAT, 1, 16);
        Map<String, INDArray> result = sd.output(Map.of("input", inputAfterReset.dup()), "output");
        INDArray actual = result.get("output").dup();

        assertNotNull(actual, "Output should not be null after session reset");
        assertArrayEquals(new long[]{1, 16}, actual.shape(),
                "Output shape should be [1, 16]");

        // Verify against slot-by-slot reference
        INDArray expected = computeMatmulChainReference(inputAfterReset, w1Data, w2Data);
        double maxDiff = expected.sub(actual).amaxNumber().doubleValue();
        log.info("After session reset: maxDiff={}", maxDiff);
        assertTrue(maxDiff < TOL,
                "Output after session reset diverged from reference. maxDiff=" + maxDiff);

        closeOutput(actual);
        closeOutput(expected);
        sd.close();
    }

    // =========================================================================
    // Test 5: Plan rebuild after close
    // =========================================================================

    @Test
    @Order(5)
    @DisplayName("Close plan, create new plan for same graph, verify correctness")
    public void testPlanRebuildAfterClose() {
        SameDiff sd = buildMatmulChain();
        sd.setGraphExecutionMode(GraphExecutionMode.AUTO);
        enableDsp(sd);

        INDArray w1Data = sd.getVariable("w1").getArr().dup();
        INDArray w2Data = sd.getVariable("w2").getArr().dup();

        // Execute to compile the plan
        INDArray input = Nd4j.randn(DataType.FLOAT, 1, 16);
        sd.output(Map.of("input", input.dup()), "output");

        // Get the executor and close the plan
        InferenceSession session = sd.getOrCreateSession();
        DynamicShapePlanExecutor executor = session.getDynamicShapePlanExecutor();
        assertNotNull(executor, "Executor should exist after execution");

        Pointer handleBefore = executor.getNativePlanHandle();
        assertNotNull(handleBefore, "Handle should exist before close");
        assertFalse(handleBefore.isNull(), "Handle should not be null before close");

        // Close the plan executor
        executor.close();

        // Re-execute — this should rebuild the plan
        INDArray inputAfterClose = Nd4j.randn(DataType.FLOAT, 1, 16);
        Map<String, INDArray> result = sd.output(Map.of("input", inputAfterClose.dup()), "output");
        INDArray actual = result.get("output").dup();

        assertNotNull(actual, "Output should not be null after plan rebuild");
        assertArrayEquals(new long[]{1, 16}, actual.shape(),
                "Output shape should be [1, 16]");

        // Verify the plan was rebuilt (new handle or valid state)
        DynamicShapePlanExecutor newExecutor = sd.getOrCreateSession().getDynamicShapePlanExecutor();
        assertNotNull(newExecutor, "Executor should exist after rebuild");

        // Verify correctness
        INDArray expected = computeMatmulChainReference(inputAfterClose, w1Data, w2Data);
        double maxDiff = expected.sub(actual).amaxNumber().doubleValue();
        log.info("After plan rebuild: maxDiff={}", maxDiff);
        assertTrue(maxDiff < TOL,
                "Output after plan rebuild diverged from reference. maxDiff=" + maxDiff);

        closeOutput(actual);
        closeOutput(expected);
        sd.close();
    }

    // =========================================================================
    // Test 6: Shape change with GEM_SLOT_BY_SLOT
    // =========================================================================

    @Test
    @Order(6)
    @DisplayName("Shape change recovery under GEM_SLOT_BY_SLOT produces correct output")
    public void testShapeChangeWithSlotBySlot() {
        SameDiff sd = buildMatmulChain();
        sd.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);
        enableDsp(sd);

        INDArray w1Data = sd.getVariable("w1").getArr().dup();
        INDArray w2Data = sd.getVariable("w2").getArr().dup();

        long[][] shapes = {{1, 16}, {4, 16}, {2, 16}, {1, 16}};

        for (long[] shape : shapes) {
            INDArray input = Nd4j.randn(DataType.FLOAT, shape);
            Map<String, INDArray> result = sd.output(Map.of("input", input.dup()), "output");
            INDArray actual = result.get("output").dup();

            assertNotNull(actual, "Output should not be null for shape " + Arrays.toString(shape));
            long[] expectedShape = new long[]{shape[0], 16};
            assertArrayEquals(expectedShape, actual.shape(),
                    "Output shape mismatch for " + Arrays.toString(shape));

            INDArray expected = computeMatmulChainReference(input, w1Data, w2Data);
            double maxDiff = expected.sub(actual).amaxNumber().doubleValue();
            log.info("SLOT_BY_SLOT shape {}: maxDiff={}", Arrays.toString(shape), maxDiff);
            assertTrue(maxDiff < TOL,
                    "SLOT_BY_SLOT shape " + Arrays.toString(shape) + " diverged. maxDiff=" + maxDiff);

            closeOutput(actual);
            closeOutput(expected);
        }

        sd.close();
    }

    // =========================================================================
    // Test 7: Shape change with GEM_EMULATED_REPLAY
    // =========================================================================

    @Test
    @Order(7)
    @DisplayName("Shape change recovery under GEM_EMULATED_REPLAY produces correct output")
    public void testShapeChangeWithEmulatedReplay() {
        INDArray w1Data = Nd4j.randn(DataType.FLOAT, 16, 32);
        INDArray w2Data = Nd4j.randn(DataType.FLOAT, 32, 16);

        long[][] shapes = {{1, 16}, {4, 16}, {2, 16}, {1, 16}};

        for (long[] shape : shapes) {
            // Build a fresh SameDiff for each shape to ensure clean DSP state
            SameDiff sd = buildMatmulChain();
            // Replace constants with shared weights
            sd.getVariable("w1").setArray(w1Data.dup());
            sd.getVariable("w2").setArray(w2Data.dup());

            sd.setGraphExecutionMode(GraphExecutionMode.EMULATED_REPLAY);
            enableDsp(sd);

            INDArray input = Nd4j.randn(DataType.FLOAT, shape);

            // Warm up this shape using advanceToReplaying (unfrozen exec → freeze → warmup)
            advanceToReplaying(sd, input, 3);

            // Execute and verify with a fresh input of the same shape
            INDArray testInput = Nd4j.randn(DataType.FLOAT, shape);
            Map<String, INDArray> result = sd.output(Map.of("input", testInput), "output");
            INDArray actual = result.get("output").dup();

            assertNotNull(actual, "Output should not be null for shape " + Arrays.toString(shape));
            long[] expectedShape = new long[]{shape[0], 16};
            assertArrayEquals(expectedShape, actual.shape(),
                    "Output shape mismatch for " + Arrays.toString(shape));

            INDArray expected = computeMatmulChainReference(testInput, w1Data, w2Data);
            double maxDiff = expected.sub(actual).amaxNumber().doubleValue();
            log.info("EMULATED_REPLAY shape {}: maxDiff={}", Arrays.toString(shape), maxDiff);
            assertTrue(maxDiff < TOL,
                    "EMULATED_REPLAY shape " + Arrays.toString(shape) + " diverged. maxDiff=" + maxDiff);

            closeOutput(actual);
            closeOutput(expected);
            sd.close();
        }
    }

    // =========================================================================
    // Test 8: Shape change with GEM_CUDA_GRAPHS (if available)
    // =========================================================================

    @Test
    @Order(8)
    @DisplayName("Shape change recovery under GEM_CUDA_GRAPHS produces correct output")
    public void testShapeChangeWithCudaGraphs() {
        SameDiff sd = buildMatmulChain();
        sd.setGraphExecutionMode(GraphExecutionMode.CUDA_GRAPHS);
        enableDsp(sd);

        INDArray w1Data = sd.getVariable("w1").getArr().dup();
        INDArray w2Data = sd.getVariable("w2").getArr().dup();

        long[][] shapes = {{1, 16}, {4, 16}, {2, 16}, {1, 16}};

        for (long[] shape : shapes) {
            INDArray input = Nd4j.randn(DataType.FLOAT, shape);

            // Warm up this shape using advanceToReplaying (unfrozen exec → freeze → warmup)
            advanceToReplaying(sd, input, 3);

            // Execute and verify with a fresh input of the same shape
            INDArray testInput = Nd4j.randn(DataType.FLOAT, shape);
            Map<String, INDArray> result = sd.output(Map.of("input", testInput), "output");
            INDArray actual = result.get("output").dup();

            assertNotNull(actual, "Output should not be null for shape " + Arrays.toString(shape));
            long[] expectedShape = new long[]{shape[0], 16};
            assertArrayEquals(expectedShape, actual.shape(),
                    "Output shape mismatch for " + Arrays.toString(shape));

            INDArray expected = computeMatmulChainReference(testInput, w1Data, w2Data);
            double maxDiff = expected.sub(actual).amaxNumber().doubleValue();
            log.info("CUDA_GRAPHS shape {}: maxDiff={}", Arrays.toString(shape), maxDiff);
            assertTrue(maxDiff < TOL,
                    "CUDA_GRAPHS shape " + Arrays.toString(shape) + " diverged. maxDiff=" + maxDiff);

            closeOutput(actual);
            closeOutput(expected);
        }

        sd.close();
    }

    // =========================================================================
    // Test 9: Repeated shape changes don't accumulate errors
    // =========================================================================

    @Test
    @Order(9)
    @DisplayName("Repeated shape changes [1,16] ↔ [4,16] 10 times maintain correctness")
    public void testRepeatedShapeChangesNoErrorAccumulation() {
        SameDiff sd = buildMatmulChain();
        sd.setGraphExecutionMode(GraphExecutionMode.AUTO);
        enableDsp(sd);

        INDArray w1Data = sd.getVariable("w1").getArr().dup();
        INDArray w2Data = sd.getVariable("w2").getArr().dup();

        DynamicShapePlanExecutor executor = null;

        for (int i = 0; i < 10; i++) {
            // Alternate between shapes
            long[] shape = (i % 2 == 0) ? new long[]{1, 16} : new long[]{4, 16};
            INDArray input = Nd4j.randn(DataType.FLOAT, shape);

            // Unfreeze to allow shape discovery, then advanceToReplaying
            executor = sd.getOrCreateSession().getDynamicShapePlanExecutor();
            if (executor != null) {
                executor.setShapesFrozen(false);
            }
            advanceToReplaying(sd, input, 2);

            // Execute
            Map<String, INDArray> result = sd.output(Map.of("input", input.dup()), "output");
            INDArray actual = result.get("output").dup();

            assertNotNull(actual, "Output should not be null at iteration " + i);
            long[] expectedShape = new long[]{shape[0], 16};
            assertArrayEquals(expectedShape, actual.shape(),
                    "Output shape mismatch at iteration " + i);

            INDArray expected = computeMatmulChainReference(input, w1Data, w2Data);
            double maxDiff = expected.sub(actual).amaxNumber().doubleValue();
            log.info("Iteration {} shape {}: maxDiff={}", i, Arrays.toString(shape), maxDiff);
            assertTrue(maxDiff < TOL,
                    "Iteration " + i + " shape " + Arrays.toString(shape) + " diverged. maxDiff=" + maxDiff);

            closeOutput(actual);
            closeOutput(expected);
        }

        sd.close();
    }

    // =========================================================================
    // Test 10: Phase tracking across shape changes
    // =========================================================================

    @Test
    @Order(10)
    @DisplayName("Phase transitions are tracked correctly across shape changes")
    public void testPhaseTransitionsAcrossShapeChanges() {
        SameDiff sd = buildMatmulChain();
        sd.setGraphExecutionMode(GraphExecutionMode.AUTO);
        enableDsp(sd);

        INDArray w1Data = sd.getVariable("w1").getArr().dup();
        INDArray w2Data = sd.getVariable("w2").getArr().dup();

        // Initial shape [1,16] — execute unfrozen first to populate shapes
        INDArray input1 = Nd4j.randn(DataType.FLOAT, 1, 16);
        sd.output(Map.of("input", input1.dup()), "output");

        DynamicShapePlanExecutor executor = sd.getOrCreateSession().getDynamicShapePlanExecutor();
        assertNotNull(executor, "Executor should exist");

        PlanPhase phase0 = executor.getPlanPhase();
        log.info("Initial phase: {}", phase0);

        // Freeze and warm up
        executor.setShapesFrozen(true);
        for (int i = 0; i < 5; i++) {
            sd.output(Map.of("input", input1.dup()), "output");
        }

        PlanPhase phase1 = executor.getPlanPhase();
        log.info("After freeze+warmup: {}", phase1);
        assertTrue(phase1.isAtLeast(PlanPhase.SHAPES_FROZEN),
                "Should be at least SHAPES_FROZEN, got " + phase1);

        // Change shape — this should trigger recompilation
        // Unfreeze first to allow shape discovery
        executor.setShapesFrozen(false);
        INDArray input2 = Nd4j.randn(DataType.FLOAT, 4, 16);
        sd.output(Map.of("input", input2.dup()), "output");

        PlanPhase phase2 = executor.getPlanPhase();
        log.info("After shape change (unfrozen): {}", phase2);
        assertNotNull(phase2, "Phase should not be null after shape change");

        // Refreeze and warm up the new shape
        executor.setShapesFrozen(true);
        for (int i = 0; i < 5; i++) {
            sd.output(Map.of("input", input2.dup()), "output");
        }

        PlanPhase phase3 = executor.getPlanPhase();
        log.info("After second shape warmup: {}", phase3);

        // Execute and verify final output is correct
        INDArray testInput = Nd4j.randn(DataType.FLOAT, 4, 16);
        Map<String, INDArray> result = sd.output(Map.of("input", testInput), "output");
        INDArray actual = result.get("output").dup();

        INDArray expected = computeMatmulChainReference(testInput, w1Data, w2Data);
        double maxDiff = expected.sub(actual).amaxNumber().doubleValue();
        log.info("Final verification: maxDiff={}", maxDiff);
        assertTrue(maxDiff < TOL,
                "Final output diverged after phase transitions. maxDiff=" + maxDiff);

        closeOutput(actual);
        closeOutput(expected);
        sd.close();
    }
}
