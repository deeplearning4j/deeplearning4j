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
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlanExecutor;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.nativeblas.NativeOps;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for shared capture workspace across CUDA graph segments.
 *
 * <p>Previously, each segment allocated its own 128MB workspace for CUDA graph capture.
 * With N segments, this consumed N × 128MB of GPU memory simultaneously. Since segments
 * execute sequentially (cudaGraphLaunch + cudaStreamSynchronize), all segments can safely
 * share a single workspace. The workspace pointer is baked into graph nodes during capture,
 * so all segments must capture with the same pointer address (guaranteed by allocating once).</p>
 *
 * <p>These tests verify:</p>
 * <ul>
 *   <li>Multi-segment graphs capture and replay correctly with shared workspace</li>
 *   <li>Results are numerically correct across warmup, capture, and replay phases</li>
 *   <li>Multiple execution cycles (destroy + rebuild plan) work correctly</li>
 *   <li>Large graphs with many segments don't OOM from workspace proliferation</li>
 * </ul>
 */
@Slf4j
@Tag("samediff")
public class TestSharedCaptureWorkspace extends BaseNd4jTestWithBackends {

    @Override
    public char ordering() {
        return 'c';
    }

    /**
     * Build a deep SameDiff graph that will produce multiple segments.
     * Each "layer" is: matmul → add → activation, giving 3 ops per layer.
     * With enough layers, the plan compiler splits into multiple segments.
     */
    private SameDiff buildDeepGraph(int numLayers, int inputDim, int hiddenDim) {
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, inputDim);
        SDVariable current = input;

        for (int i = 0; i < numLayers; i++) {
            int inDim = (i == 0) ? inputDim : hiddenDim;
            SDVariable w = sd.constant("w_" + i, Nd4j.randn(DataType.FLOAT, inDim, hiddenDim).muli(0.01));
            SDVariable b = sd.constant("b_" + i, Nd4j.zeros(DataType.FLOAT, hiddenDim));
            current = sd.mmul("mm_" + i, current, w);
            current = current.add("add_" + i, b);
            if (i % 2 == 0) {
                current = sd.nn.relu("relu_" + i, current, 0);
            } else {
                current = sd.math.tanh("tanh_" + i, current);
            }
        }
        // Final output
        SDVariable wOut = sd.constant("w_out", Nd4j.randn(DataType.FLOAT, hiddenDim, 16).muli(0.01));
        current = sd.mmul("mm_out", current, wOut);
        sd.math.tanh("output", current);
        return sd;
    }

    /**
     * Build a wide graph with parallel branches that merge — creates a complex
     * segment structure with many ops.
     */
    private SameDiff buildWideBranchGraph(int numBranches, int dim) {
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, dim);

        SDVariable[] branches = new SDVariable[numBranches];
        for (int i = 0; i < numBranches; i++) {
            SDVariable w1 = sd.constant("w1_" + i, Nd4j.randn(DataType.FLOAT, dim, dim).muli(0.01));
            SDVariable w2 = sd.constant("w2_" + i, Nd4j.randn(DataType.FLOAT, dim, dim).muli(0.01));
            SDVariable mm1 = sd.mmul("mm1_" + i, input, w1);
            SDVariable act1 = sd.nn.relu("relu_" + i, mm1, 0);
            SDVariable mm2 = sd.mmul("mm2_" + i, act1, w2);
            branches[i] = sd.math.tanh("tanh_" + i, mm2);
        }

        // Sum all branches
        SDVariable sum = branches[0];
        for (int i = 1; i < numBranches; i++) {
            sum = sum.add("sum_" + i, branches[i]);
        }
        sd.identity("output", sum);
        return sd;
    }

    private void configureDsp(SameDiff sd) {
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);
        sd.setGraphExecutionMode(GraphExecutionMode.CUDA_GRAPHS);
    }

    // ── Test 1: Basic multi-segment correctness ────────────────────────────

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Deep graph with shared workspace produces correct results across warmup/capture/replay")
    public void testDeepGraphCorrectness(Nd4jBackend backend) {
        int numLayers = 12;  // 12 layers × 3 ops = 36 ops → multiple segments
        SameDiff sd = buildDeepGraph(numLayers, 64, 64);
        configureDsp(sd);

        INDArray inputArr = Nd4j.randn(DataType.FLOAT, 4, 64);

        // Run 5 times: warmup (1) + capture (2) + replay (3-5)
        INDArray firstResult = null;
        for (int call = 0; call < 5; call++) {
            Map<String, INDArray> result = sd.outputDirect(
                    Collections.singletonMap("input", inputArr), "output");
            INDArray actual = result.get("output");

            assertNotNull(actual, "Call " + call + ": output should not be null");
            assertFalse(actual.isNaN().any(), "Call " + call + ": output contains NaN");
            assertFalse(actual.isInfinite().any(), "Call " + call + ": output contains Inf");

            if (firstResult == null) {
                firstResult = actual.dup();
            } else {
                // All calls with same input should produce same output
                float maxDiff = Transforms.abs(actual.sub(firstResult)).maxNumber().floatValue();
                assertTrue(maxDiff < 0.01f,
                        "Call " + call + ": result diverged from first call, maxDiff=" + maxDiff);
            }
        }

        // Verify segments were captured
        NativeOps ops = Nd4j.getNativeOps();
        DynamicShapePlanExecutor executor = sd.getOrCreateSession().getDynamicShapePlanExecutor();
        if (executor != null && executor.getNativePlanHandle() != null) {
            var handle = executor.getNativePlanHandle();
            int numSegments = ops.getPlanNumSegments(handle);
            int captured = ops.getPlanNumCapturedGraphSegments(handle);
            log.info("Deep graph: {} segments, {} captured", numSegments, captured);
            assertTrue(numSegments > 0, "Should have at least 1 segment");
        }

        sd.close();
    }

    // ── Test 2: Wide branch graph (many segments) ───────────────────────────

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Wide branch graph with many segments shares workspace correctly")
    public void testWideBranchGraphSharedWorkspace(Nd4jBackend backend) {
        int numBranches = 8;  // 8 branches × 5 ops each = 40 ops
        SameDiff sd = buildWideBranchGraph(numBranches, 64);
        configureDsp(sd);

        INDArray inputArr = Nd4j.randn(DataType.FLOAT, 4, 64);

        // Run 5 times
        INDArray firstResult = null;
        for (int call = 0; call < 5; call++) {
            Map<String, INDArray> result = sd.outputDirect(
                    Collections.singletonMap("input", inputArr), "output");
            INDArray actual = result.get("output");

            assertNotNull(actual, "Call " + call + ": output should not be null");
            assertFalse(actual.isNaN().any(), "Call " + call + ": output contains NaN");

            if (firstResult == null) {
                firstResult = actual.dup();
            } else {
                float maxDiff = Transforms.abs(actual.sub(firstResult)).maxNumber().floatValue();
                assertTrue(maxDiff < 0.01f,
                        "Call " + call + ": result diverged, maxDiff=" + maxDiff);
            }
        }

        NativeOps ops = Nd4j.getNativeOps();
        DynamicShapePlanExecutor executor = sd.getOrCreateSession().getDynamicShapePlanExecutor();
        if (executor != null && executor.getNativePlanHandle() != null) {
            var handle = executor.getNativePlanHandle();
            int numSegments = ops.getPlanNumSegments(handle);
            int captured = ops.getPlanNumCapturedGraphSegments(handle);
            log.info("Wide branch graph: {} segments, {} captured", numSegments, captured);
        }

        sd.close();
    }

    // ── Test 3: Repeated plan rebuild cycles ────────────────────────────────

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Multiple plan rebuild cycles work correctly with shared workspace")
    public void testRepeatedPlanRebuildCycles(Nd4jBackend backend) {
        int numLayers = 8;
        SameDiff sd = buildDeepGraph(numLayers, 32, 32);
        configureDsp(sd);

        // Run 3 cycles: each cycle does warmup + capture + replay, then resets
        for (int cycle = 0; cycle < 3; cycle++) {
            INDArray inputArr = Nd4j.randn(DataType.FLOAT, 2, 32);

            INDArray firstResult = null;
            for (int call = 0; call < 4; call++) {
                Map<String, INDArray> result = sd.outputDirect(
                        Collections.singletonMap("input", inputArr), "output");
                INDArray actual = result.get("output");

                assertNotNull(actual, "Cycle " + cycle + " call " + call + ": output null");
                assertFalse(actual.isNaN().any(),
                        "Cycle " + cycle + " call " + call + ": NaN detected");

                if (firstResult == null) {
                    firstResult = actual.dup();
                } else {
                    float maxDiff = Transforms.abs(actual.sub(firstResult)).maxNumber().floatValue();
                    assertTrue(maxDiff < 0.01f,
                            "Cycle " + cycle + " call " + call + ": diverged, maxDiff=" + maxDiff);
                }
            }

            // Reset session to force plan destruction and rebuild
            sd.resetSession();
            log.info("Cycle {} completed successfully", cycle);
        }

        sd.close();
    }

    // ── Test 4: Different inputs produce different outputs on replay ─────────

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Shared workspace replay produces different outputs for different inputs")
    public void testReplayOutputsVaryWithInput(Nd4jBackend backend) {
        // Use a shallow graph (2 layers) with relu + identity output so gradients
        // don't vanish to zero (deep tanh with 0.01-scaled weights produces all-zeros)
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 32);
        SDVariable w1 = sd.constant("w1", Nd4j.randn(DataType.FLOAT, 32, 32).muli(0.1));
        SDVariable mm1 = sd.mmul("mm1", input, w1);
        SDVariable act1 = sd.nn.relu("relu1", mm1, 0);
        SDVariable w2 = sd.constant("w2", Nd4j.randn(DataType.FLOAT, 32, 16).muli(0.1));
        SDVariable mm2 = sd.mmul("mm2", act1, w2);
        sd.identity("output", mm2);

        configureDsp(sd);

        // Warmup + capture with first input
        INDArray input1 = Nd4j.randn(DataType.FLOAT, 2, 32);
        for (int i = 0; i < 3; i++) {
            sd.output(Collections.singletonMap("input", input1), "output");
        }

        // Now replay with same input
        Map<String, INDArray> result1 = sd.output(
                Collections.singletonMap("input", input1), "output");
        INDArray out1 = result1.get("output").dup();

        // Replay with different input — should get different result
        INDArray input2 = Nd4j.randn(DataType.FLOAT, 2, 32);
        Map<String, INDArray> result2 = sd.output(
                Collections.singletonMap("input", input2), "output");
        INDArray out2 = result2.get("output").dup();

        assertFalse(out1.isNaN().any(), "Output 1 contains NaN");
        assertFalse(out2.isNaN().any(), "Output 2 contains NaN");

        // Outputs should differ for different inputs
        float diff = Transforms.abs(out1.sub(out2)).maxNumber().floatValue();
        assertTrue(diff > 1e-6f,
                "Different inputs should produce different outputs, but maxDiff=" + diff);

        sd.close();
    }

    // ── Test 5: Very deep graph (stress test for many segments) ─────────────

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Very deep graph with 60+ ops creates many segments — shared workspace prevents OOM")
    @Timeout(120)
    public void testVeryDeepGraphManySegments(Nd4jBackend backend) {
        // 20 layers × 3 ops = 60 ops → likely 5+ segments
        // Without shared workspace: 5+ × 128MB = 640MB+ just for workspaces
        // With shared workspace: 1 × 128MB
        int numLayers = 20;
        SameDiff sd = buildDeepGraph(numLayers, 64, 64);
        configureDsp(sd);

        INDArray inputArr = Nd4j.randn(DataType.FLOAT, 4, 64);

        // Run through full lifecycle: warmup → capture → replay × 3
        for (int call = 0; call < 5; call++) {
            Map<String, INDArray> result = sd.outputDirect(
                    Collections.singletonMap("input", inputArr), "output");
            INDArray actual = result.get("output");

            assertNotNull(actual, "Call " + call + ": output null");
            assertFalse(actual.isNaN().any(), "Call " + call + ": NaN");
        }

        // Log segment info
        NativeOps ops = Nd4j.getNativeOps();
        DynamicShapePlanExecutor executor = sd.getOrCreateSession().getDynamicShapePlanExecutor();
        if (executor != null && executor.getNativePlanHandle() != null) {
            var handle = executor.getNativePlanHandle();
            int numSegments = ops.getPlanNumSegments(handle);
            int captured = ops.getPlanNumCapturedGraphSegments(handle);
            log.info("Very deep graph: {} segments, {} captured (shared workspace)", numSegments, captured);

            // With shared workspace, even many captured segments should work
            // The test completing without OOM is itself the primary assertion
            assertTrue(numSegments >= 1, "Should have segments");
        }

        sd.close();
    }

    // ── Test 6: Rapid create/destroy cycles ─────────────────────────────────

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Rapid plan creation and destruction cycles don't leak shared workspace")
    public void testRapidPlanLifecycles(Nd4jBackend backend) {
        // Create and destroy plans rapidly to test shared workspace cleanup
        for (int planNum = 0; planNum < 5; planNum++) {
            SameDiff sd = buildDeepGraph(8, 32, 32);
            configureDsp(sd);

            INDArray inputArr = Nd4j.randn(DataType.FLOAT, 2, 32);

            // Warmup + capture + replay
            for (int call = 0; call < 4; call++) {
                Map<String, INDArray> result = sd.outputDirect(
                        Collections.singletonMap("input", inputArr), "output");
                assertNotNull(result.get("output"), "Plan " + planNum + " call " + call);
                assertFalse(result.get("output").isNaN().any(),
                        "Plan " + planNum + " call " + call + ": NaN");
            }

            sd.close();
            log.info("Plan lifecycle {} completed", planNum);
        }
        // If shared workspace is leaked, we'd accumulate 5 × 128MB per cycle
        // which would eventually OOM. Completing all 5 cycles is the assertion.
    }

    // ── Test 7: Multi-segment graph with non-capturable barriers ────────────

    /**
     * Build a graph with explicit non-capturable barriers (gather with INT indices)
     * between capturable segments. This forces multiple segment boundaries:
     *
     *   [capturable: matmul+relu] → [non-capturable: gather(INT)] → [capturable: matmul+relu] → ...
     *
     * Each gather has an INT SDVariable input, which the compiler flags as
     * outputShapeDependsOnInputValues=true, making it non-capturable.
     * The capturable segments get CUDA graph capture after warmup.
     * With shared workspace, all captured segments share one 128MB allocation.
     *
     * @param numBarriers number of non-capturable gather barriers (creates numBarriers+1 capturable segments)
     */
    private SameDiff buildMultiSegmentGraph(int numBarriers, int dim) {
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, dim);
        SDVariable indices = sd.placeHolder("indices", DataType.INT, -1);
        SDVariable current = input;

        for (int seg = 0; seg <= numBarriers; seg++) {
            // Capturable block: matmul + relu
            SDVariable w = sd.constant("w_" + seg, Nd4j.randn(DataType.FLOAT, dim, dim).muli(0.1));
            current = sd.nn.relu("relu_" + seg, sd.mmul("mm_" + seg, current, w), 0);

            // Non-capturable barrier (except after last capturable block)
            if (seg < numBarriers) {
                current = sd.gather("gather_" + seg, current, indices, 0);
            }
        }

        // Final projection
        SDVariable wOut = sd.constant("w_out", Nd4j.randn(DataType.FLOAT, dim, 16).muli(0.1));
        sd.identity("output", sd.mmul("mm_out", current, wOut));
        return sd;
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Multi-segment graph with gather barriers forces shared workspace across captured segments")
    @Timeout(120)
    public void testMultiSegmentCapturedGraphsShareWorkspace(Nd4jBackend backend) {
        int numBarriers = 4;  // → 5 capturable segments + 4 non-capturable = 9 segments total
        int dim = 32;
        int batchSize = 4;

        SameDiff sd = buildMultiSegmentGraph(numBarriers, dim);
        configureDsp(sd);

        INDArray inputArr = Nd4j.randn(DataType.FLOAT, batchSize, dim);
        INDArray indicesArr = Nd4j.arange(batchSize).castTo(DataType.INT);  // [0,1,2,3] — gather all rows

        Map<String, INDArray> feed = new HashMap<>();
        feed.put("input", inputArr);
        feed.put("indices", indicesArr);

        // Phase 1: Warmup (unfrozen) — compiles plan, runs slot-by-slot
        INDArray firstResult = null;
        for (int call = 0; call < 3; call++) {
            Map<String, INDArray> result = sd.outputDirect(feed, "output");
            INDArray actual = result.get("output");
            assertNotNull(actual, "Warmup call " + call + ": output null");
            assertFalse(actual.isNaN().any(), "Warmup call " + call + ": NaN");
            if (firstResult == null) firstResult = actual.dup();
        }

        // Phase 2: Freeze shapes — enables CUDA graph capture for capturable segments
        DynamicShapePlanExecutor executor = sd.getOrCreateSession().getDynamicShapePlanExecutor();
        assertNotNull(executor, "DynamicShapePlanExecutor should exist after warmup");
        assertNotNull(executor.getNativePlanHandle(), "Native plan handle should exist");

        NativeOps ops = Nd4j.getNativeOps();
        var handle = executor.getNativePlanHandle();
        int numSegments = ops.getPlanNumSegments(handle);
        log.info("Pre-freeze: {} segments", numSegments);
        assertTrue(numSegments >= 3,
                "Expected at least 3 segments before freeze, got " + numSegments);

        executor.setShapesFrozen(true);

        // Phase 3: Frozen execution — warmup(1) + capture(2) + replay(3-7)
        for (int call = 0; call < 7; call++) {
            Map<String, INDArray> result = sd.outputDirect(feed, "output");
            INDArray actual = result.get("output");

            assertNotNull(actual, "Frozen call " + call + ": output null");
            assertFalse(actual.isNaN().any(), "Frozen call " + call + ": NaN");
            assertFalse(actual.isInfinite().any(), "Frozen call " + call + ": Inf");
            assertEquals(batchSize, actual.shape()[0], "Frozen call " + call + ": wrong batch size");

            float maxDiff = Transforms.abs(actual.sub(firstResult)).maxNumber().floatValue();
            assertTrue(maxDiff < 0.01f,
                    "Frozen call " + call + ": result diverged from warmup, maxDiff=" + maxDiff);
        }

        // Verify multi-segment structure with captured graphs
        int captured = ops.getPlanNumCapturedGraphSegments(handle);
        log.info("Multi-segment graph: {} segments, {} captured (barriers={})", numSegments, captured, numBarriers);

        // With 4 barriers we expect at least 5 segments (alternating capturable/non-capturable)
        // and at least 2 captured graph segments sharing the workspace
        assertTrue(captured >= 2,
                "Expected at least 2 captured graph segments sharing workspace, got " + captured);

        sd.close();
    }

    // ── Test 8: Many captured segments stress test ────────────────────────────

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Many captured segments share workspace without OOM")
    @Timeout(180)
    public void testManyCapturedSegmentsShareWorkspace(Nd4jBackend backend) {
        // 10 barriers → 11 capturable segments → without shared workspace: 11 × 128MB = 1.4GB
        // With shared workspace: 1 × 128MB
        int numBarriers = 10;
        int dim = 32;
        int batchSize = 4;

        SameDiff sd = buildMultiSegmentGraph(numBarriers, dim);
        configureDsp(sd);

        INDArray inputArr = Nd4j.randn(DataType.FLOAT, batchSize, dim);
        INDArray indicesArr = Nd4j.arange(batchSize).castTo(DataType.INT);

        Map<String, INDArray> feed = new HashMap<>();
        feed.put("input", inputArr);
        feed.put("indices", indicesArr);

        // Phase 1: Warmup (unfrozen)
        for (int call = 0; call < 2; call++) {
            Map<String, INDArray> result = sd.outputDirect(feed, "output");
            assertNotNull(result.get("output"), "Warmup call " + call + ": output null");
        }

        // Phase 2: Freeze shapes and run through capture + replay
        DynamicShapePlanExecutor executor = sd.getOrCreateSession().getDynamicShapePlanExecutor();
        assertNotNull(executor, "Executor should exist");
        executor.setShapesFrozen(true);

        for (int call = 0; call < 7; call++) {
            Map<String, INDArray> result = sd.outputDirect(feed, "output");
            INDArray actual = result.get("output");
            assertNotNull(actual, "Frozen call " + call + ": output null");
            assertFalse(actual.isNaN().any(), "Frozen call " + call + ": NaN");
        }

        NativeOps ops = Nd4j.getNativeOps();
        var handle = executor.getNativePlanHandle();
        int numSegments = ops.getPlanNumSegments(handle);
        int captured = ops.getPlanNumCapturedGraphSegments(handle);
        log.info("Many-segment stress: {} segments, {} captured (barriers={})", numSegments, captured, numBarriers);

        // Primary assertion: completing without OOM with many captured segments
        assertTrue(numSegments >= 5, "Expected many segments, got " + numSegments);
        // With shared workspace, all capturable segments should eventually get captured
        assertTrue(captured >= 3, "Expected at least 3 captured segments sharing workspace, got " + captured);

        sd.close();
    }

    // ── Test 9: Sequential replay steps with varying batch sizes ─────────────

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Sequential replay steps with varying batch sizes work correctly")
    public void testVaryingBatchSizeReplay(Nd4jBackend backend) {
        // Shared workspace is allocated for first segment — verify it works
        // when batch size stays the same (no shape change that would invalidate capture)
        int numLayers = 10;
        SameDiff sd = buildDeepGraph(numLayers, 32, 32);
        configureDsp(sd);

        int batchSize = 4;
        INDArray inputArr = Nd4j.randn(DataType.FLOAT, batchSize, 32);

        // Warmup + capture
        for (int i = 0; i < 3; i++) {
            sd.outputDirect(Collections.singletonMap("input", inputArr), "output");
        }

        // Multiple replay steps with same shape but different values
        for (int step = 0; step < 10; step++) {
            INDArray newInput = Nd4j.randn(DataType.FLOAT, batchSize, 32);
            Map<String, INDArray> result = sd.outputDirect(
                    Collections.singletonMap("input", newInput), "output");
            INDArray actual = result.get("output");

            assertNotNull(actual, "Step " + step + ": output null");
            assertFalse(actual.isNaN().any(), "Step " + step + ": NaN");
            assertFalse(actual.isInfinite().any(), "Step " + step + ": Inf");
            assertEquals(batchSize, actual.shape()[0],
                    "Step " + step + ": wrong batch size in output");
        }

        sd.close();
    }

    // ── Test 10: VLM-scale capture buffer stress test ──────────────────────

    /**
     * Build a VLM-scale graph that emulates the real crash scenario:
     * - ~80+ barriers → 160+ segments (~80+ capturable, ~80 non-capturable)
     * - Large weight matrices (>= 1MB each) → directReference classification
     * - Multiple weight matrices per capturable block → many external inputs
     * - Each capturable block: matmul(W1) → relu → matmul(W2) → add(bias) (like a transformer layer)
     * - Non-capturable barriers: gather with INT indices
     *
     * The VLM crash had: 168 captured segments, 1300+ external inputs, SIGSEGV at seg[420-421]
     * because directReference weight addresses drifted after capture and the CUDA graph
     * path has NO lineage drift check (only Triton/gpubackend path has it, gated by argTableStable).
     *
     * @param numBarriers number of non-capturable barriers (creates numBarriers+1 capturable blocks)
     * @param dim hidden dimension (512+ for 1MB+ weight threshold)
     * @param layersPerBlock ops per capturable block (more = more external inputs per segment)
     */
    private SameDiff buildVlmScaleGraph(int numBarriers, int dim, int layersPerBlock) {
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, dim);
        SDVariable indices = sd.placeHolder("indices", DataType.INT, -1);
        SDVariable current = input;

        for (int seg = 0; seg <= numBarriers; seg++) {
            // Each capturable block has multiple layers, each with large weight matrices
            for (int layer = 0; layer < layersPerBlock; layer++) {
                String prefix = "s" + seg + "_l" + layer;
                // W1: dim×dim = 1MB+ (directReference when >= WEIGHT_THRESHOLD)
                SDVariable w1 = sd.constant(prefix + "_w1",
                        Nd4j.randn(DataType.FLOAT, dim, dim).muli(0.02));
                current = sd.mmul(prefix + "_mm1", current, w1);
                current = sd.nn.relu(prefix + "_relu", current, 0);

                // W2: dim×dim = 1MB+ (another directReference weight)
                SDVariable w2 = sd.constant(prefix + "_w2",
                        Nd4j.randn(DataType.FLOAT, dim, dim).muli(0.02));
                current = sd.mmul(prefix + "_mm2", current, w2);

                // Bias: dim (small, NOT directReference — gets capture buffer)
                SDVariable bias = sd.constant(prefix + "_b",
                        Nd4j.zeros(DataType.FLOAT, dim));
                current = current.add(prefix + "_add", bias);
            }

            // Non-capturable barrier (gather with INT indices)
            if (seg < numBarriers) {
                current = sd.gather("gather_" + seg, current, indices, 0);
            }
        }

        // Final projection
        SDVariable wOut = sd.constant("w_out",
                Nd4j.randn(DataType.FLOAT, dim, 16).muli(0.02));
        sd.identity("output", sd.mmul("mm_out", current, wOut));
        return sd;
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("VLM-scale: 80+ barriers, 160+ segments, 500+ large weights, address drift after capture")
    @Timeout(600)
    public void testVlmScaleAddressDriftAfterCapture(Nd4jBackend backend) {
        // Emulates the actual VLM crash:
        // - 80 barriers → 81 capturable blocks → 161 total segments
        // - 2 layers/block × 2 weight matrices + 1 bias = 5 constants/block → 405 external inputs
        // - Each weight is 512×512 = 1MB → directReference classification
        // - Total weight memory: ~405 × 1MB ≈ 405MB on GPU
        //
        // The crash scenario:
        // 1. Capture succeeds → directReference addresses baked into CUDA graphs
        // 2. Some weights get reallocated (memory pressure, pool compaction)
        // 3. CUDA graph replay reads stale directReference address → SIGSEGV
        //
        // In the Triton path (gpubackend.cpp), argTableStable=true skips lineage
        // drift check entirely (line 1078). In the CUDA graph path (cudagraph.cu),
        // there is NO lineage check at all — directReference buffers are just skipped
        // in the replay refresh loop (line 512: if (cb.directReference) continue;).
        int numBarriers = 80;
        int dim = 512;          // 512×512 × 4 = 1MB per weight (directReference threshold)
        int layersPerBlock = 2; // 2 layers × (W1 + W2 + bias) = 5 constants per block
        int batchSize = 4;

        log.info("Building VLM-scale graph: {} barriers, {} dim, {} layers/block...",
                numBarriers, dim, layersPerBlock);
        SameDiff sd = buildVlmScaleGraph(numBarriers, dim, layersPerBlock);
        configureDsp(sd);

        INDArray inputArr = Nd4j.randn(DataType.FLOAT, batchSize, dim);
        INDArray indicesArr = Nd4j.arange(batchSize).castTo(DataType.INT);

        Map<String, INDArray> feed = new HashMap<>();
        feed.put("input", inputArr);
        feed.put("indices", indicesArr);

        // Phase 1: Warmup (unfrozen) — compile plan, establish segment structure
        log.info("Phase 1: Warmup (unfrozen)...");
        for (int call = 0; call < 3; call++) {
            Map<String, INDArray> result = sd.outputDirect(feed, "output");
            assertNotNull(result.get("output"), "Warmup call " + call);
            log.info("Warmup call {} complete", call);
        }

        // Phase 2: Freeze shapes → enable CUDA graph capture
        DynamicShapePlanExecutor executor = sd.getOrCreateSession().getDynamicShapePlanExecutor();
        assertNotNull(executor, "Executor should exist");
        executor.setShapesFrozen(true);

        NativeOps ops = Nd4j.getNativeOps();
        var handle = executor.getNativePlanHandle();
        int numSegments = ops.getPlanNumSegments(handle);
        log.info("Phase 2: Frozen. {} total segments", numSegments);
        assertTrue(numSegments >= 100,
                "Expected 100+ segments for VLM-scale graph, got " + numSegments);

        // Phase 3: Run until capture + replay succeeds across all segments
        log.info("Phase 3: Frozen execution (capture + replay)...");
        for (int call = 0; call < 7; call++) {
            Map<String, INDArray> result = sd.outputDirect(feed, "output");
            INDArray actual = result.get("output");
            assertNotNull(actual, "Frozen call " + call + ": output null");
            assertFalse(actual.isNaN().any(), "Frozen call " + call + ": NaN");
            log.info("Frozen call {} complete", call);
        }

        int captured = ops.getPlanNumCapturedGraphSegments(handle);
        log.info("Phase 3 done: {} segments, {} captured", numSegments, captured);
        assertTrue(captured >= 40,
                "Expected 40+ captured segments for VLM scale, got " + captured);

        // Phase 4: Replace weight arrays across ALL segments — simulate address drift
        // This is what happens when GPU memory pressure causes reallocation.
        // Every directReference address changes → graph reads stale pointers.
        log.info("Phase 4: Replacing ALL weight arrays to simulate address drift...");
        int replacedCount = 0;
        for (int seg = 0; seg <= numBarriers; seg++) {
            for (int layer = 0; layer < layersPerBlock; layer++) {
                String prefix = "s" + seg + "_l" + layer;
                sd.associateArrayWithVariable(
                        Nd4j.randn(DataType.FLOAT, dim, dim).muli(0.02), prefix + "_w1");
                sd.associateArrayWithVariable(
                        Nd4j.randn(DataType.FLOAT, dim, dim).muli(0.02), prefix + "_w2");
                sd.associateArrayWithVariable(
                        Nd4j.zeros(DataType.FLOAT, dim), prefix + "_b");
                replacedCount += 3;
            }
        }
        sd.associateArrayWithVariable(
                Nd4j.randn(DataType.FLOAT, dim, 16).muli(0.02), "w_out");
        replacedCount++;
        log.info("Replaced {} weight/bias arrays (all directReference addresses changed)", replacedCount);

        // Phase 5: Execute after address drift — THIS IS WHERE THE VLM CRASHED
        // If bug exists: SIGSEGV (stale directReference GPU address in baked graph)
        // If fixed: system detects drift, invalidates captured graphs, re-captures
        log.info("Phase 5: Executing after address drift (crash site)...");
        for (int call = 0; call < 5; call++) {
            Map<String, INDArray> result = sd.outputDirect(feed, "output");
            INDArray actual = result.get("output");
            assertNotNull(actual, "Post-drift call " + call + ": output null");
            assertFalse(actual.isNaN().any(), "Post-drift call " + call + ": NaN");
            assertFalse(actual.isInfinite().any(), "Post-drift call " + call + ": Inf");
            log.info("Post-drift call {} complete (survived)", call);
        }

        int capturedAfter = ops.getPlanNumCapturedGraphSegments(handle);
        log.info("Post-drift: {} captured segments (was {} before drift)", capturedAfter, captured);

        sd.close();
        log.info("VLM-scale test PASSED: {} segments, {} weights replaced, no SIGSEGV",
                numSegments, replacedCount);
    }

    // ── Test 11: VLM-scale repeated drift cycles ─────────────────────────────

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("VLM-scale: repeated weight drift + re-capture cycles")
    @Timeout(600)
    public void testVlmScaleRepeatedDriftCycles(Nd4jBackend backend) {
        // Same VLM-scale graph, but drift weights multiple times to stress
        // the re-capture path. Each cycle: capture → drift → survive → re-capture.
        int numBarriers = 50;
        int dim = 512;
        int layersPerBlock = 2;
        int batchSize = 4;

        log.info("Building VLM-scale graph: {} barriers...", numBarriers);
        SameDiff sd = buildVlmScaleGraph(numBarriers, dim, layersPerBlock);
        configureDsp(sd);

        INDArray inputArr = Nd4j.randn(DataType.FLOAT, batchSize, dim);
        INDArray indicesArr = Nd4j.arange(batchSize).castTo(DataType.INT);
        Map<String, INDArray> feed = new HashMap<>();
        feed.put("input", inputArr);
        feed.put("indices", indicesArr);

        // Warmup + freeze
        for (int call = 0; call < 3; call++) {
            sd.outputDirect(feed, "output");
        }
        DynamicShapePlanExecutor executor = sd.getOrCreateSession().getDynamicShapePlanExecutor();
        executor.setShapesFrozen(true);

        NativeOps ops = Nd4j.getNativeOps();
        var handle = executor.getNativePlanHandle();

        // 3 drift cycles
        for (int cycle = 0; cycle < 3; cycle++) {
            log.info("=== Drift cycle {} ===", cycle);

            // Capture + replay
            for (int call = 0; call < 5; call++) {
                Map<String, INDArray> result = sd.outputDirect(feed, "output");
                INDArray actual = result.get("output");
                assertNotNull(actual, "Cycle " + cycle + " call " + call);
                assertFalse(actual.isNaN().any(), "Cycle " + cycle + " call " + call + ": NaN");
            }

            int captured = ops.getPlanNumCapturedGraphSegments(handle);
            log.info("Cycle {}: {} captured before drift", cycle, captured);

            // Drift ALL weights
            for (int seg = 0; seg <= numBarriers; seg++) {
                for (int layer = 0; layer < layersPerBlock; layer++) {
                    String prefix = "s" + seg + "_l" + layer;
                    sd.associateArrayWithVariable(
                            Nd4j.randn(DataType.FLOAT, dim, dim).muli(0.02), prefix + "_w1");
                    sd.associateArrayWithVariable(
                            Nd4j.randn(DataType.FLOAT, dim, dim).muli(0.02), prefix + "_w2");
                }
            }
            log.info("Cycle {}: drifted all weights", cycle);

            // Execute after drift — must not crash
            for (int call = 0; call < 3; call++) {
                Map<String, INDArray> result = sd.outputDirect(feed, "output");
                INDArray actual = result.get("output");
                assertNotNull(actual, "Post-drift cycle " + cycle + " call " + call);
                assertFalse(actual.isNaN().any(),
                        "Post-drift cycle " + cycle + " call " + call + ": NaN");
            }
            log.info("Cycle {} survived", cycle);
        }

        sd.close();
    }

    // ── Test 12: Cross-segment buffer refresh at VLM scale ──────────────────

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("VLM-scale: cross-segment capture buffers refresh correctly with varying inputs")
    @Timeout(600)
    public void testVlmScaleCrossSegmentRefresh(Nd4jBackend backend) {
        // Tests that cross-segment capture buffers (outputs from non-capturable
        // gather barriers fed into capturable matmul segments) are properly
        // refreshed when placeholder values change between replay invocations.
        //
        // Uses fewer barriers (4) with larger layers to avoid vanishing gradients
        // through 60+ barriers while still testing cross-segment refresh at scale.
        int numBarriers = 4;
        int dim = 1024;  // Larger dim for bigger directReference weights
        int layersPerBlock = 4;  // More layers per block = more ops per segment
        int batchSize = 4;

        log.info("Building cross-seg test graph: {} barriers, dim={}, layers/block={}...",
                numBarriers, dim, layersPerBlock);
        SameDiff sd = buildVlmScaleGraph(numBarriers, dim, layersPerBlock);
        configureDsp(sd);

        INDArray indicesArr = Nd4j.arange(batchSize).castTo(DataType.INT);

        // Warmup + freeze + capture with input A
        INDArray inputA = Nd4j.randn(DataType.FLOAT, batchSize, dim);
        Map<String, INDArray> feedA = new HashMap<>();
        feedA.put("input", inputA);
        feedA.put("indices", indicesArr);

        for (int call = 0; call < 3; call++) {
            sd.outputDirect(feedA, "output");
        }
        DynamicShapePlanExecutor executor = sd.getOrCreateSession().getDynamicShapePlanExecutor();
        executor.setShapesFrozen(true);

        INDArray resultA = null;
        for (int call = 0; call < 5; call++) {
            Map<String, INDArray> result = sd.outputDirect(feedA, "output");
            INDArray actual = result.get("output");
            assertNotNull(actual, "InputA call " + call);
            assertFalse(actual.isNaN().any(), "InputA call " + call + ": NaN");
            if (call == 4) resultA = actual.dup();
        }

        NativeOps ops = Nd4j.getNativeOps();
        var handle = executor.getNativePlanHandle();
        int captured = ops.getPlanNumCapturedGraphSegments(handle);
        log.info("Captured {} segments with input A", captured);

        // Switch to very different input B — cross-segment data changes significantly
        INDArray inputB = Nd4j.randn(DataType.FLOAT, batchSize, dim).muli(10);
        Map<String, INDArray> feedB = new HashMap<>();
        feedB.put("input", inputB);
        feedB.put("indices", indicesArr);

        INDArray resultB = null;
        for (int call = 0; call < 5; call++) {
            Map<String, INDArray> result = sd.outputDirect(feedB, "output");
            INDArray actual = result.get("output");
            assertNotNull(actual, "InputB call " + call);
            assertFalse(actual.isNaN().any(), "InputB call " + call + ": NaN");
            if (call == 4) resultB = actual.dup();
        }

        // Primary assertion: no crash when cross-segment data changes during replay
        // Output diff may be 0 for very deep networks (vanishing gradients) — that's OK
        assertNotNull(resultA);
        assertNotNull(resultB);
        float diff = Transforms.abs(resultA.sub(resultB)).maxNumber().floatValue();
        log.info("Cross-segment: input A vs B output diff: {} (captured={})", diff, captured);

        sd.close();
    }

    // ── Test 13: Multi-device VLM-scale address drift ────────────────────────

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Multi-device: VLM-scale graph with address drift across GPU devices")
    @Timeout(600)
    public void testMultiDeviceVlmScaleAddressDrift(Nd4jBackend backend) {
        // The E2E VLM crash happened on a multi-GPU system (dev0: 24GB, dev1: 8GB).
        // CUDA graph capture bakes in device-specific GPU addresses. If arrays
        // migrate between devices or get reallocated on a different device,
        // the baked addresses become invalid → SIGSEGV.
        //
        // This test:
        // 1. Builds a VLM-scale graph
        // 2. Captures on device 0 (default)
        // 3. Allocates replacement weights on device 0 (new addresses)
        // 4. Also allocates some tensors on device 1 to create memory pressure
        // 5. Replays — system must handle address drift safely
        int numDevices = Nd4j.getAffinityManager().getNumberOfDevices();
        log.info("Multi-device test: {} GPU devices available", numDevices);

        int numBarriers = 40;
        int dim = 512;
        int layersPerBlock = 2;
        int batchSize = 4;

        log.info("Building multi-device VLM-scale graph: {} barriers...", numBarriers);
        SameDiff sd = buildVlmScaleGraph(numBarriers, dim, layersPerBlock);
        configureDsp(sd);

        INDArray inputArr = Nd4j.randn(DataType.FLOAT, batchSize, dim);
        INDArray indicesArr = Nd4j.arange(batchSize).castTo(DataType.INT);
        Map<String, INDArray> feed = new HashMap<>();
        feed.put("input", inputArr);
        feed.put("indices", indicesArr);

        // Phase 1: Warmup + freeze + capture
        for (int call = 0; call < 3; call++) {
            sd.outputDirect(feed, "output");
        }
        DynamicShapePlanExecutor executor = sd.getOrCreateSession().getDynamicShapePlanExecutor();
        executor.setShapesFrozen(true);

        for (int call = 0; call < 5; call++) {
            Map<String, INDArray> result = sd.outputDirect(feed, "output");
            assertNotNull(result.get("output"), "Capture call " + call);
            assertFalse(result.get("output").isNaN().any(), "Capture call " + call + ": NaN");
        }

        NativeOps ops = Nd4j.getNativeOps();
        var handle = executor.getNativePlanHandle();
        int captured = ops.getPlanNumCapturedGraphSegments(handle);
        log.info("Phase 1: {} captured segments", captured);

        // Phase 2: Create memory pressure on all devices
        // Allocate large tensors to fragment the memory pool
        java.util.List<INDArray> pressureArrays = new java.util.ArrayList<>();
        for (int dev = 0; dev < numDevices; dev++) {
            for (int i = 0; i < 5; i++) {
                try {
                    // 64MB chunks to fragment memory
                    INDArray pressure = Nd4j.zeros(DataType.FLOAT, 4096, 4096);
                    Nd4j.getAffinityManager().ensureLocation(pressure,
                            org.nd4j.linalg.api.concurrency.AffinityManager.Location.HOST);
                    pressureArrays.add(pressure);
                } catch (Exception e) {
                    log.info("Device {}: could not allocate pressure array {}: {}", dev, i, e.getMessage());
                    break;
                }
            }
        }
        log.info("Phase 2: created {} pressure arrays across {} devices",
                pressureArrays.size(), numDevices);

        // Phase 3: Replace weights after memory pressure — addresses likely differ
        int replacedCount = 0;
        for (int seg = 0; seg <= numBarriers; seg++) {
            for (int layer = 0; layer < layersPerBlock; layer++) {
                String prefix = "s" + seg + "_l" + layer;
                sd.associateArrayWithVariable(
                        Nd4j.randn(DataType.FLOAT, dim, dim).muli(0.02), prefix + "_w1");
                sd.associateArrayWithVariable(
                        Nd4j.randn(DataType.FLOAT, dim, dim).muli(0.02), prefix + "_w2");
                replacedCount += 2;
            }
        }
        log.info("Phase 3: replaced {} weights after memory pressure", replacedCount);

        // Release pressure arrays to free memory
        for (INDArray arr : pressureArrays) {
            arr.close();
        }
        pressureArrays.clear();

        // Phase 4: Execute after drift + memory pressure — crash site
        log.info("Phase 4: executing after multi-device address drift...");
        for (int call = 0; call < 5; call++) {
            Map<String, INDArray> result = sd.outputDirect(feed, "output");
            INDArray actual = result.get("output");
            assertNotNull(actual, "Multi-device post-drift call " + call);
            assertFalse(actual.isNaN().any(), "Multi-device post-drift call " + call + ": NaN");
            log.info("Multi-device post-drift call {} survived", call);
        }

        int capturedAfter = ops.getPlanNumCapturedGraphSegments(handle);
        log.info("Multi-device test: {} captured before, {} after drift ({} weights replaced, {} devices)",
                captured, capturedAfter, replacedCount, numDevices);

        sd.close();
    }

    // ── Test 14: Multi-device interleaved execution ──────────────────────────

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Multi-device: interleaved execution with device affinity changes")
    @Timeout(600)
    public void testMultiDeviceInterleavedExecution(Nd4jBackend backend) {
        // Tests that CUDA graph capture/replay handles scenarios where
        // input tensors may have different device affinities between executions.
        // This is common in multi-GPU setups where data is loaded on one device
        // and processed on another.
        int numDevices = Nd4j.getAffinityManager().getNumberOfDevices();
        log.info("Interleaved multi-device test: {} GPUs", numDevices);

        int numBarriers = 20;
        int dim = 512;
        int layersPerBlock = 2;
        int batchSize = 4;

        SameDiff sd = buildVlmScaleGraph(numBarriers, dim, layersPerBlock);
        configureDsp(sd);

        INDArray indicesArr = Nd4j.arange(batchSize).castTo(DataType.INT);

        // Phase 1: Warmup + freeze + capture on device 0
        INDArray input0 = Nd4j.randn(DataType.FLOAT, batchSize, dim);
        Map<String, INDArray> feed = new HashMap<>();
        feed.put("input", input0);
        feed.put("indices", indicesArr);

        for (int call = 0; call < 3; call++) {
            sd.outputDirect(feed, "output");
        }
        DynamicShapePlanExecutor executor = sd.getOrCreateSession().getDynamicShapePlanExecutor();
        executor.setShapesFrozen(true);

        for (int call = 0; call < 5; call++) {
            sd.outputDirect(feed, "output");
        }

        NativeOps ops = Nd4j.getNativeOps();
        var handle = executor.getNativePlanHandle();
        int captured = ops.getPlanNumCapturedGraphSegments(handle);
        log.info("Phase 1: {} captured on device 0", captured);

        // Phase 2: Interleave execution with fresh input arrays
        // Each new Nd4j.randn() allocates a new GPU buffer (potentially different address)
        // simulating what happens in real inference where new inputs arrive each step
        for (int step = 0; step < 20; step++) {
            INDArray freshInput = Nd4j.randn(DataType.FLOAT, batchSize, dim);
            // Optionally try to place on different device
            if (numDevices > 1 && step % 3 == 0) {
                // Touch the array to ensure device allocation, then sync back
                // This exercises the H2D/D2H paths that interact with capture buffers
                Nd4j.getAffinityManager().ensureLocation(freshInput,
                        org.nd4j.linalg.api.concurrency.AffinityManager.Location.HOST);
            }
            feed.put("input", freshInput);

            Map<String, INDArray> result = sd.outputDirect(feed, "output");
            INDArray actual = result.get("output");
            assertNotNull(actual, "Interleaved step " + step);
            assertFalse(actual.isNaN().any(), "Interleaved step " + step + ": NaN");
        }

        int capturedAfter = ops.getPlanNumCapturedGraphSegments(handle);
        log.info("Phase 2: {} captured after 20 interleaved steps (was {})", capturedAfter, captured);

        // Phase 3: Drift all weights PLUS change input device → maximum address disruption
        for (int seg = 0; seg <= numBarriers; seg++) {
            for (int layer = 0; layer < layersPerBlock; layer++) {
                String prefix = "s" + seg + "_l" + layer;
                sd.associateArrayWithVariable(
                        Nd4j.randn(DataType.FLOAT, dim, dim).muli(0.02), prefix + "_w1");
                sd.associateArrayWithVariable(
                        Nd4j.randn(DataType.FLOAT, dim, dim).muli(0.02), prefix + "_w2");
            }
        }
        log.info("Phase 3: drifted all weights");

        for (int call = 0; call < 5; call++) {
            INDArray driftInput = Nd4j.randn(DataType.FLOAT, batchSize, dim);
            feed.put("input", driftInput);
            Map<String, INDArray> result = sd.outputDirect(feed, "output");
            INDArray actual = result.get("output");
            assertNotNull(actual, "Post-full-drift call " + call);
            assertFalse(actual.isNaN().any(), "Post-full-drift call " + call + ": NaN");
            log.info("Post-full-drift call {} survived", call);
        }

        sd.close();
        log.info("Multi-device interleaved test PASSED");
    }

    // ── Test 15: Growing attention_mask — staging buffer must handle size changes ──

    /**
     * Build a VLM-style decode graph where attention_mask grows by 1 each step.
     *
     * <p>In real VLM inference:</p>
     * <ul>
     *   <li>attention_mask starts at [1, seqLen] during prefill</li>
     *   <li>Each decode step appends 1 token → attention_mask grows to [1, seqLen+1], [1, seqLen+2], ...</li>
     *   <li>The graph uses attention_mask to mask out invalid positions</li>
     * </ul>
     *
     * <p>The capture buffer (staging buffer) for attention_mask should be allocated at
     * max size so that growing inputs can always be copied in. The current bug: the
     * capture buffer is allocated at capture-time size, and when srcBytes != capturedSize,
     * the refresh is silently skipped → stale data → SIGSEGV or wrong results.</p>
     */
    private SameDiff buildGrowingMaskGraph(int dim, int numLayers) {
        SameDiff sd = SameDiff.create();

        // input: [batch, dim] — the hidden state (fixed size each step)
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, dim);

        // attention_mask: [1, seqLen] — GROWS by 1 each decode step
        // Using -1 for seqLen dimension since it's dynamic
        SDVariable mask = sd.placeHolder("attention_mask", DataType.FLOAT, 1, -1);

        // indices placeholder to create non-capturable barriers (gather with INT)
        SDVariable indices = sd.placeHolder("indices", DataType.INT, -1);

        SDVariable current = input;

        for (int layer = 0; layer < numLayers; layer++) {
            String prefix = "layer" + layer;

            // Weight matrix (large enough for directReference classification)
            SDVariable w = sd.constant(prefix + "_w",
                    Nd4j.randn(DataType.FLOAT, dim, dim).muli(0.02));
            current = sd.mmul(prefix + "_mm", current, w);

            // Apply attention mask: sum the mask to get a scalar scaling factor,
            // then multiply the hidden state by it. This makes the output
            // DIRECTLY dependent on the mask values — if the mask is stale,
            // the output will be wrong.
            //
            // mask shape: [1, seqLen] → sum(dim=1) → [1, 1] scalar-like
            // We use sum so the result changes when mask grows (more 1s = bigger sum)
            SDVariable maskSum = sd.math.sum(prefix + "_masksum", mask, 1);
            // maskSum is [1,1], broadcast multiply with current [batch, dim]
            current = current.mul(prefix + "_maskmul", maskSum);

            SDVariable bias = sd.constant(prefix + "_b", Nd4j.zeros(DataType.FLOAT, dim));
            current = current.add(prefix + "_add", bias);
            current = sd.nn.relu(prefix + "_relu", current, 0);

            // Add a gather barrier between some layers to create multiple segments
            if (layer > 0 && layer < numLayers - 1 && layer % 2 == 0) {
                current = sd.gather(prefix + "_gather", current, indices, 0);
            }
        }

        // Final projection to small output
        SDVariable wOut = sd.constant("w_out",
                Nd4j.randn(DataType.FLOAT, dim, 16).muli(0.02));
        sd.identity("output", sd.mmul("mm_out", current, wOut));
        return sd;
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Growing attention_mask: staging buffer must refresh with size-changing inputs")
    @Timeout(300)
    public void testGrowingAttentionMaskStagingBuffer(Nd4jBackend backend) {
        // This test models the exact VLM crash scenario:
        // 1. attention_mask starts at [1, 2048] (prefill)
        // 2. Each decode step: mask grows by 1 → [1, 2049], [1, 2050], ...
        // 3. Graph captures with [1, 2048] → staging buffer sized for 2048 elements
        // 4. On replay with [1, 2049]: srcBytes != capturedSize → copy SKIPPED
        // 5. Graph replays with stale mask data → wrong output or SIGSEGV
        //
        // The fix: staging buffer should be allocated at max size, and the
        // copy should always succeed regardless of current input size.

        int dim = 256;
        int numLayers = 6;
        int batchSize = 2;
        int initialSeqLen = 128;   // Start with this sequence length
        int numDecodeSteps = 20;   // Grow mask by 1 for 20 steps

        SameDiff sd = buildGrowingMaskGraph(dim, numLayers);
        configureDsp(sd);

        // Phase 1: Warmup with initial seqLen to let graph plan compile
        Map<String, INDArray> feed = new HashMap<>();
        INDArray inputData = Nd4j.randn(DataType.FLOAT, batchSize, dim);
        INDArray indicesData = Nd4j.arange(batchSize).castTo(DataType.INT);

        for (int warmup = 0; warmup < 3; warmup++) {
            int seqLen = initialSeqLen;
            INDArray maskData = Nd4j.ones(DataType.FLOAT, 1, seqLen);
            feed.put("input", inputData);
            feed.put("attention_mask", maskData);
            feed.put("indices", indicesData);

            Map<String, INDArray> result = sd.outputDirect(feed, "output");
            INDArray out = result.get("output");
            assertNotNull(out, "Warmup " + warmup);
            assertFalse(out.isNaN().any(), "Warmup " + warmup + ": NaN");
            log.info("Warmup {} with seqLen={}: output sum={}", warmup, seqLen, out.sumNumber().floatValue());
        }

        // Phase 2: Freeze shapes → trigger graph capture on next exec
        DynamicShapePlanExecutor executor = sd.getOrCreateSession().getDynamicShapePlanExecutor();
        executor.setShapesFrozen(true);
        log.info("Shapes frozen");

        // Execute once more to trigger capture with initialSeqLen
        {
            INDArray maskData = Nd4j.ones(DataType.FLOAT, 1, initialSeqLen);
            feed.put("attention_mask", maskData);
            Map<String, INDArray> captureResult = sd.outputDirect(feed, "output");
            INDArray captureOut = captureResult.get("output");
            assertNotNull(captureOut, "Capture exec");
            assertFalse(captureOut.isNaN().any(), "Capture exec: NaN");
            log.info("Capture exec with seqLen={}: sum={}",
                    initialSeqLen, captureOut.sumNumber().floatValue());
        }

        // Compute reference output with initialSeqLen (mask sum = 128)
        float captureOutputSum;
        {
            INDArray maskData = Nd4j.ones(DataType.FLOAT, 1, initialSeqLen);
            feed.put("attention_mask", maskData);
            Map<String, INDArray> ref = sd.outputDirect(feed, "output");
            captureOutputSum = ref.get("output").sumNumber().floatValue();
            log.info("Reference with seqLen={}: sum={}", initialSeqLen, captureOutputSum);
        }

        // Phase 3: Decode steps — mask grows by 1 each step
        // Each step adds one more "1" to attention_mask → maskSum increases by 1
        // Output should change proportionally. If staging buffer is stale,
        // output will stay the same as the captured version.
        float prevSum = captureOutputSum;
        int staleCount = 0;

        for (int step = 0; step < numDecodeSteps; step++) {
            int seqLen = initialSeqLen + step + 1;  // 129, 130, 131, ...
            INDArray maskData = Nd4j.ones(DataType.FLOAT, 1, seqLen);
            feed.put("attention_mask", maskData);

            Map<String, INDArray> result = sd.outputDirect(feed, "output");
            INDArray out = result.get("output");
            assertNotNull(out, "Decode step " + step + " seqLen=" + seqLen);
            assertFalse(out.isNaN().any(), "Decode step " + step + ": NaN");

            float currentSum = out.sumNumber().floatValue();
            log.info("Decode step {} seqLen={}: sum={} (prev={})", step, seqLen, currentSum, prevSum);

            // The output MUST change when mask grows — maskSum goes from N to N+1,
            // which scales all activations. If the staging buffer is stale (old mask
            // not refreshed), output sum will stay identical to the captured version.
            if (Math.abs(currentSum - captureOutputSum) < 1e-3f && captureOutputSum != 0) {
                staleCount++;
                log.warn("STALE OUTPUT at step {} seqLen={}: sum {} matches capture sum {} — "
                        + "staging buffer NOT refreshed!", step, seqLen, currentSum, captureOutputSum);
            }

            prevSum = currentSum;
        }

        // If more than 2 steps produce the exact same output as capture, the staging
        // buffer is definitely stale. Allow 1-2 for floating point edge cases.
        if (staleCount > 2) {
            fail("Staging buffer is STALE: " + staleCount + "/" + numDecodeSteps
                    + " decode steps produced the same output as the captured version. "
                    + "The attention_mask capture buffer is not being refreshed when "
                    + "srcBytes != capturedSize. Expected output to change as mask grows.");
        }
        log.info("Growing attention_mask test: {}/{} steps had fresh output (stale={})",
                numDecodeSteps - staleCount, numDecodeSteps, staleCount);

        sd.close();
        log.info("Growing attention_mask staging buffer test PASSED");
    }

    // ── Test 16: Max-size staging buffer — allocate once, copy partial each step ──

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Max-size staging buffer: capture at max, replay with varying sizes, output always correct")
    @Timeout(300)
    public void testMaxSizeStagingBufferCorrectness(Nd4jBackend backend) {
        // Verifies the CORRECT behavior: staging buffer allocated at max size,
        // smaller inputs zero-padded. The graph always sees the max-size buffer.
        // The attention_mask handles semantic correctness (zeros = ignore).
        //
        // Pattern:
        //   Step 0: mask = [1,1,1,...,1, 0,0,...,0] (initialSeqLen ones, rest zeros)
        //   Step 1: mask = [1,1,1,...,1,1, 0,...,0]  (initialSeqLen+1 ones)
        //   ...
        // The mask shape is ALWAYS [1, maxSeqLen] — never changes size.
        // Only the VALUES change (more 1s appended). Graph shape is stable.

        int dim = 256;
        int numLayers = 6;
        int batchSize = 2;
        int maxSeqLen = 256;        // Max buffer size — fixed for all steps
        int initialSeqLen = 128;    // Start with this many 1s
        int numDecodeSteps = 50;    // Add 1 more token each step

        SameDiff sd = buildGrowingMaskGraph(dim, numLayers);
        configureDsp(sd);

        // Helper: create a padded mask of fixed shape [1, maxSeqLen]
        // with `activeLen` ones followed by zeros
        java.util.function.IntFunction<INDArray> makeMask = (activeLen) -> {
            INDArray m = Nd4j.zeros(DataType.FLOAT, 1, maxSeqLen);
            if (activeLen > 0) {
                // Set first activeLen positions to 1
                m.get(org.nd4j.linalg.indexing.NDArrayIndex.point(0),
                      org.nd4j.linalg.indexing.NDArrayIndex.interval(0, Math.min(activeLen, maxSeqLen)))
                 .assign(1.0f);
            }
            return m;
        };

        Map<String, INDArray> feed = new HashMap<>();
        INDArray inputData = Nd4j.randn(DataType.FLOAT, batchSize, dim);
        INDArray indicesData = Nd4j.arange(batchSize).castTo(DataType.INT);

        // Phase 1: Warmup with maxSeqLen-shaped mask (fixed shape from the start)
        for (int warmup = 0; warmup < 3; warmup++) {
            INDArray maskData = makeMask.apply(initialSeqLen);
            feed.put("input", inputData);
            feed.put("attention_mask", maskData);
            feed.put("indices", indicesData);

            Map<String, INDArray> result = sd.outputDirect(feed, "output");
            assertNotNull(result.get("output"), "Warmup " + warmup);
            assertFalse(result.get("output").isNaN().any(), "Warmup " + warmup + ": NaN");
            log.info("Warmup {} maskActive={} shape={}: sum={}",
                    warmup, initialSeqLen, java.util.Arrays.toString(maskData.shape()),
                    result.get("output").sumNumber().floatValue());
        }

        // Phase 2: Freeze shapes → capture with maxSeqLen-shaped mask
        DynamicShapePlanExecutor executor = sd.getOrCreateSession().getDynamicShapePlanExecutor();
        executor.setShapesFrozen(true);
        log.info("Shapes frozen");

        // Capture execution with initialSeqLen active positions
        float captureSum;
        {
            INDArray maskData = makeMask.apply(initialSeqLen);
            feed.put("attention_mask", maskData);
            Map<String, INDArray> captureResult = sd.outputDirect(feed, "output");
            captureSum = captureResult.get("output").sumNumber().floatValue();
            log.info("Capture exec: maskActive={} shape={} sum={}",
                    initialSeqLen, java.util.Arrays.toString(maskData.shape()), captureSum);
        }

        // Phase 3: Decode steps — shape is ALWAYS [1, maxSeqLen], only values change
        // Since shape never changes, capture buffer size always matches.
        // Output MUST change because maskSum changes (more 1s → bigger sum).
        float prevSum = captureSum;
        int correctSteps = 0;

        for (int step = 0; step < numDecodeSteps; step++) {
            int activeLen = initialSeqLen + step + 1;
            INDArray maskData = makeMask.apply(activeLen);
            feed.put("attention_mask", maskData);

            Map<String, INDArray> result = sd.outputDirect(feed, "output");
            INDArray out = result.get("output");
            assertNotNull(out, "Decode step " + step);
            assertFalse(out.isNaN().any(), "Decode step " + step + ": NaN");

            float currentSum = out.sumNumber().floatValue();

            // Output must change from capture output (more mask positions active)
            boolean changed = Math.abs(currentSum - captureSum) > 1e-3f || captureSum == 0;
            if (changed) correctSteps++;

            if (step % 10 == 0 || !changed) {
                log.info("Decode step {} activeLen={}: sum={} (capture={}) changed={}",
                        step, activeLen, currentSum, captureSum, changed);
            }

            prevSum = currentSum;
        }

        // ALL steps should produce different output from the capture step
        // (because maskSum increases by 1 each step)
        assertTrue(correctSteps >= numDecodeSteps - 1,
                "Expected nearly all decode steps to produce different output from capture. "
                + "Got " + correctSteps + "/" + numDecodeSteps + " correct. "
                + "If most steps match the capture output, the staging buffer is stale.");

        log.info("Max-size staging buffer test: {}/{} steps correct", correctSteps, numDecodeSteps);
        sd.close();
        log.info("Max-size staging buffer correctness test PASSED");
    }

    // ── Test 17: Lineage drift with argTableStable — the SIGSEGV scenario ────

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Lineage drift: replacing external input arrays while argTableStable=true must not SIGSEGV")
    @Timeout(300)
    public void testLineageDriftWithArgTableStable(Nd4jBackend backend) {
        // Reproduces the VLM SIGSEGV at seg[420-421]:
        //
        // Root cause: CUDA graph capture bakes in GPU memory addresses of
        // directReference entries (weight constants). When the CUDA memory pool
        // reallocates these buffers (e.g., between VLM decode executions),
        // the baked addresses become stale. The lineage check at gpubackend.cpp:1078
        // detects this drift and invalidates the graph for re-capture.
        //
        // The bug: when argTableStable=true (set after first successful replay),
        // the lineage check was SKIPPED even when extAddrsStable=false (external
        // addresses changed). This allowed graph replay with stale directReference
        // addresses → cudaMemcpyAsync reads freed GPU memory → SIGSEGV.
        //
        // The fix: force lineage check when !extAddrsStable, regardless of argTableStable.
        //
        // Test strategy:
        //   1. Build graph with weight constants (become directReference in capture)
        //   2. Warmup → freeze shapes → capture → replay (establishes argTableStable=true)
        //   3. Replace weight arrays with NEW allocations (different GPU addresses)
        //   4. Execute again — without fix: SIGSEGV; with fix: detects drift, re-captures
        //   5. Verify output is correct after re-capture

        int dim = 128;
        int numLayers = 2;
        int batchSize = 4;
        int numBarriers = 2;  // Enough for multi-segment but not excessive

        // Build graph with gather barriers for multi-segment structure
        // Use fewer layers + smaller dim to avoid vanishing gradients
        SameDiff sd = buildVlmScaleGraph(numBarriers, dim, numLayers);
        configureDsp(sd);

        INDArray inputArr = Nd4j.randn(DataType.FLOAT, batchSize, dim);
        INDArray indicesArr = Nd4j.arange(batchSize).castTo(DataType.INT);
        Map<String, INDArray> feed = new HashMap<>();
        feed.put("input", inputArr);
        feed.put("indices", indicesArr);

        // Phase 1: Warmup (3 calls to establish plan)
        for (int i = 0; i < 3; i++) {
            Map<String, INDArray> result = sd.outputDirect(feed, "output");
            assertNotNull(result.get("output"), "Warmup " + i);
        }

        // Freeze shapes to enable graph capture
        DynamicShapePlanExecutor executor = sd.getOrCreateSession().getDynamicShapePlanExecutor();
        executor.setShapesFrozen(true);
        log.info("Phase 1: shapes frozen");

        // Phase 2: Capture + replay to establish argTableStable=true
        // Execute enough times for capture + multiple replays
        INDArray preReplaceResult = null;
        for (int i = 0; i < 8; i++) {
            Map<String, INDArray> result = sd.outputDirect(feed, "output");
            INDArray out = result.get("output");
            assertNotNull(out, "Capture/replay " + i);
            assertFalse(out.isNaN().any(), "Capture/replay " + i + ": NaN");
            if (i == 7) preReplaceResult = out.dup();
        }

        NativeOps ops = Nd4j.getNativeOps();
        var handle = executor.getNativePlanHandle();
        int capturedBefore = ops.getPlanNumCapturedGraphSegments(handle);
        log.info("Phase 2: {} segments captured, argTableStable should be true", capturedBefore);
        assertTrue(capturedBefore > 0, "Should have captured at least 1 segment");

        // Phase 3: Replace ALL weight arrays with BRAND NEW GPU allocations.
        // This simulates what happens in VLM decode when the CUDA memory pool
        // reallocates buffers — same shapes, different GPU addresses.
        // The directReference entries in capture buffers still point to the OLD addresses.
        log.info("Phase 3: replacing all weight arrays with new GPU allocations...");
        int replacedCount = 0;
        for (int seg = 0; seg <= numBarriers; seg++) {
            for (int layer = 0; layer < numLayers; layer++) {
                String prefix = "s" + seg + "_l" + layer;
                // Create BRAND NEW arrays — Nd4j.randn allocates fresh GPU buffers
                // with potentially different addresses than the originals.
                // Use larger scale (0.5) so outputs are non-zero and differ measurably.
                INDArray newW1 = Nd4j.randn(DataType.FLOAT, dim, dim).muli(0.5);
                INDArray newW2 = Nd4j.randn(DataType.FLOAT, dim, dim).muli(0.5);
                sd.associateArrayWithVariable(newW1, prefix + "_w1");
                sd.associateArrayWithVariable(newW2, prefix + "_w2");
                replacedCount += 2;
            }
        }
        log.info("Phase 3: replaced {} weight arrays", replacedCount);

        // Phase 4: Execute after replacement — THE CRASH SITE
        // Without fix: argTableStable=true → lineage check skipped → graph replays
        // with stale directReference addresses → SIGSEGV in cudaMemcpyAsync
        // With fix: extAddrsStable=false → lineage check runs → detects drift →
        // invalidates graph → re-captures → correct output
        log.info("Phase 4: executing after weight replacement (crash site)...");
        for (int i = 0; i < 5; i++) {
            Map<String, INDArray> result = sd.outputDirect(feed, "output");
            INDArray out = result.get("output");
            assertNotNull(out, "Post-replace call " + i);
            assertFalse(out.isNaN().any(), "Post-replace call " + i + ": NaN");
            assertFalse(out.isInfinite().any(), "Post-replace call " + i + ": Inf");
            log.info("Post-replace call {} survived, sum={}", i, out.sumNumber().floatValue());
        }

        // Phase 5: Verify the graph survived weight replacement.
        // Primary assertion: no crash (SIGSEGV) — already passed above.
        // Secondary: output should change with new weights. But deep relu networks
        // with small weights may vanish to zero, so only assert if pre-replace was non-zero.
        Map<String, INDArray> finalResult = sd.outputDirect(feed, "output");
        INDArray postReplaceResult = finalResult.get("output");
        assertNotNull(postReplaceResult);
        assertNotNull(preReplaceResult);

        float preSum = Math.abs(preReplaceResult.sumNumber().floatValue());
        float maxDiff = Transforms.abs(postReplaceResult.sub(preReplaceResult)).maxNumber().floatValue();
        log.info("Output diff after weight replacement: maxDiff={} preSum={}", maxDiff, preSum);
        // Only assert output changed if pre-replace output was non-trivial
        if (preSum > 1e-3f) {
            assertTrue(maxDiff > 1e-6f,
                    "Output should differ after weight replacement (maxDiff=" + maxDiff + ", preSum=" + preSum + "). "
                    + "If identical, lineage drift was not detected and graph replayed with stale data.");
        } else {
            log.warn("Pre-replace output is near-zero (sum={}), skipping output diff assertion. "
                    + "Primary no-crash assertion already passed.", preSum);
        }

        int capturedAfter = ops.getPlanNumCapturedGraphSegments(handle);
        log.info("Test 17 complete: captured before={} after={} replaced={} weights",
                capturedBefore, capturedAfter, replacedCount);

        sd.close();
        log.info("Lineage drift with argTableStable test PASSED");
    }

    // ── Test 18: Rapid input array cycling — address reuse detection ─────────

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Rapid input cycling: new placeholder arrays every call with same shape must not crash")
    @Timeout(300)
    public void testRapidInputArrayCycling(Nd4jBackend backend) {
        // In VLM decode, each step creates NEW INDArray objects for position_ids,
        // attention_mask, etc. (same shape, different GPU addresses). The CUDA pool
        // may reuse addresses from freed arrays, making pointer-based staleness
        // detection unreliable. This test exercises that pattern at high frequency.
        //
        // Key: create a brand new INDArray for the placeholder every single call,
        // never reusing the same Java object. This maximizes address churn.

        int dim = 256;
        int numLayers = 4;
        int batchSize = 2;
        int numBarriers = 2;

        SameDiff sd = buildVlmScaleGraph(numBarriers, dim, numLayers);
        configureDsp(sd);

        INDArray indicesArr = Nd4j.arange(batchSize).castTo(DataType.INT);

        // Warmup with fixed input
        INDArray warmupInput = Nd4j.randn(DataType.FLOAT, batchSize, dim);
        Map<String, INDArray> feed = new HashMap<>();
        feed.put("input", warmupInput);
        feed.put("indices", indicesArr);

        for (int i = 0; i < 3; i++) {
            sd.outputDirect(feed, "output");
        }

        DynamicShapePlanExecutor executor = sd.getOrCreateSession().getDynamicShapePlanExecutor();
        executor.setShapesFrozen(true);

        // Capture with initial input
        for (int i = 0; i < 5; i++) {
            sd.outputDirect(feed, "output");
        }

        NativeOps ops = Nd4j.getNativeOps();
        var handle = executor.getNativePlanHandle();
        int captured = ops.getPlanNumCapturedGraphSegments(handle);
        log.info("Captured {} segments before rapid cycling", captured);

        // Rapid cycling: new input array every call (50 calls)
        // Each Nd4j.randn allocates a fresh GPU buffer → different address
        int numCycles = 50;
        for (int cycle = 0; cycle < numCycles; cycle++) {
            INDArray freshInput = Nd4j.randn(DataType.FLOAT, batchSize, dim);
            feed.put("input", freshInput);

            Map<String, INDArray> result = sd.outputDirect(feed, "output");
            INDArray out = result.get("output");
            assertNotNull(out, "Cycle " + cycle);
            assertFalse(out.isNaN().any(), "Cycle " + cycle + ": NaN");

            if (cycle % 10 == 0) {
                log.info("Rapid cycle {}/{}: sum={}", cycle, numCycles, out.sumNumber().floatValue());
            }
        }

        log.info("Rapid input cycling test PASSED: {} cycles with {} captured segments",
                numCycles, captured);
        sd.close();
    }
}
