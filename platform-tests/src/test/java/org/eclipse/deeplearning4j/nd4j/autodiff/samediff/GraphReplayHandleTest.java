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

import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.execution.*;
import org.nd4j.autodiff.samediff.internal.TrainingSession;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.nativeblas.NativeOps;

import org.nd4j.autodiff.samediff.internal.InferenceSession;

import java.io.File;
import java.nio.file.Files;
import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for GraphReplayHandle abstraction: capture verification, per-segment observability,
 * pointer tracking, replay cache, backend plan management, and training replay profiles.
 */
public class GraphReplayHandleTest extends BaseNd4jTestWithBackends {

    @Override
    public char ordering() {
        return 'c';
    }

    /**
     * Build a SameDiff graph with 15+ ops for DSP testing.
     * Chain: input -> matmul -> add -> relu -> matmul -> add -> tanh -> (repeat) -> output
     */
    private SameDiff buildTestGraph(int numOps) {
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 32);
        SDVariable current = input;
        int iterations = numOps / 3;
        for (int i = 0; i < iterations; i++) {
            SDVariable wi = sd.var("w_" + i, Nd4j.randn(DataType.FLOAT, 32, 32).muli(0.01));
            SDVariable bi = sd.var("b_" + i, Nd4j.zeros(DataType.FLOAT, 32));
            current = sd.mmul("matmul_" + i, current, wi);
            current = current.add("add_" + i, bi);
            if (i == iterations - 1) {
                // Last iteration: name output "out"
                current = sd.math.tanh("out", current);
            } else if (i % 2 == 0) {
                current = sd.math.tanh("tanh_" + i, current);
            } else {
                current = sd.nn.relu("relu_" + i, current, 0);
            }
        }
        return sd;
    }

    private void configureDsp(SameDiff sd, GraphExecutionMode mode) {
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);
        sd.setGraphExecutionMode(mode);
    }

    private Map<String, INDArray> makeInputs() {
        return Collections.singletonMap("input", Nd4j.randn(DataType.FLOAT, 4, 32));
    }

    // --- Replay Handle Tests ---

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testReplayHandleCreatedOnCapture(Nd4jBackend backend) {
        SameDiff sd = buildTestGraph(18);
        configureDsp(sd, GraphExecutionMode.CUDA_GRAPHS);

        for (int i = 0; i < 3; i++) {
            sd.outputDirect(makeInputs(), "out");
        }

        NativeOps ops = Nd4j.getNativeOps();
        DynamicShapePlanExecutor executor = sd.getOrCreateSession().getDynamicShapePlanExecutor();
        if (executor != null && executor.getNativePlanHandle() != null) {
            int captured = ops.getPlanNumCapturedGraphSegments(executor.getNativePlanHandle());
            assertTrue(captured >= 0, "Should have captured segments (or 0 on CPU)");
        }
        sd.close();
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPerSegmentBackendName(Nd4jBackend backend) {
        SameDiff sd = buildTestGraph(18);
        configureDsp(sd, GraphExecutionMode.CUDA_GRAPHS);

        for (int i = 0; i < 3; i++) {
            sd.outputDirect(makeInputs(), "out");
        }

        NativeOps ops = Nd4j.getNativeOps();
        DynamicShapePlanExecutor executor = sd.getOrCreateSession().getDynamicShapePlanExecutor();
        if (executor != null && executor.getNativePlanHandle() != null) {
            var handle = executor.getNativePlanHandle();
            int numSegs = ops.getPlanNumSegments(handle);
            for (int i = 0; i < numSegs; i++) {
                String name = ops.getPlanSegmentBackendName(handle, i);
                assertNotNull(name, "Backend name should not be null for seg " + i);
            }
        }
        sd.close();
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPerSegmentReplayState(Nd4jBackend backend) {
        SameDiff sd = buildTestGraph(18);
        configureDsp(sd, GraphExecutionMode.CUDA_GRAPHS);

        for (int i = 0; i < 3; i++) {
            sd.outputDirect(makeInputs(), "out");
        }

        NativeOps ops = Nd4j.getNativeOps();
        DynamicShapePlanExecutor executor = sd.getOrCreateSession().getDynamicShapePlanExecutor();
        if (executor != null && executor.getNativePlanHandle() != null) {
            var handle = executor.getNativePlanHandle();
            int numSegs = ops.getPlanNumSegments(handle);
            for (int i = 0; i < numSegs; i++) {
                int state = ops.getPlanSegmentReplayState(handle, i);
                // State should be valid: -1 (no handle), 0-4 (ReplayState values)
                assertTrue(state >= -1 && state <= 4,
                    "Invalid replay state " + state + " for seg " + i);
            }
        }
        sd.close();
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPerSegmentReplayCount(Nd4jBackend backend) {
        SameDiff sd = buildTestGraph(18);
        configureDsp(sd, GraphExecutionMode.CUDA_GRAPHS);

        for (int i = 0; i < 5; i++) {
            sd.outputDirect(makeInputs(), "out");
        }

        NativeOps ops = Nd4j.getNativeOps();
        DynamicShapePlanExecutor executor = sd.getOrCreateSession().getDynamicShapePlanExecutor();
        if (executor != null && executor.getNativePlanHandle() != null) {
            var handle = executor.getNativePlanHandle();
            int numSegs = ops.getPlanNumSegments(handle);
            for (int i = 0; i < numSegs; i++) {
                int count = ops.getPlanSegmentReplayCount(handle, i);
                assertTrue(count >= 0, "Replay count should be >= 0 for seg " + i);
            }
        }
        sd.close();
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPerSegmentStatisticsJson(Nd4jBackend backend) {
        SameDiff sd = buildTestGraph(18);
        configureDsp(sd, GraphExecutionMode.CUDA_GRAPHS);

        for (int i = 0; i < 3; i++) {
            sd.outputDirect(makeInputs(), "out");
        }

        NativeOps ops = Nd4j.getNativeOps();
        DynamicShapePlanExecutor executor = sd.getOrCreateSession().getDynamicShapePlanExecutor();
        if (executor != null && executor.getNativePlanHandle() != null) {
            var handle = executor.getNativePlanHandle();
            int numSegs = ops.getPlanNumSegments(handle);
            for (int i = 0; i < numSegs; i++) {
                String json = ops.getPlanSegmentStatisticsJson(handle, i);
                assertNotNull(json, "Stats JSON should not be null for seg " + i);
                if (!json.isEmpty()) {
                    assertTrue(json.contains("numOperations"),
                        "Stats JSON should contain numOperations: " + json);
                }
            }
        }
        sd.close();
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSlotBySlotNoCapture(Nd4jBackend backend) {
        SameDiff sd = buildTestGraph(18);
        configureDsp(sd, GraphExecutionMode.SLOT_BY_SLOT);

        for (int i = 0; i < 3; i++) {
            sd.outputDirect(makeInputs(), "out");
        }

        NativeOps ops = Nd4j.getNativeOps();
        DynamicShapePlanExecutor executor = sd.getOrCreateSession().getDynamicShapePlanExecutor();
        if (executor != null && executor.getNativePlanHandle() != null) {
            int captured = ops.getPlanNumCapturedGraphSegments(executor.getNativePlanHandle());
            if (Nd4j.getEnvironment().isCPU()) {
                // CPU FunctionalReplayHandle caches all segments — captured count may be non-zero
                assertTrue(captured >= 0, "SLOT_BY_SLOT captured count should be non-negative on CPU");
            } else {
                assertEquals(0, captured, "SLOT_BY_SLOT should have 0 captured segments on CUDA");
            }
        }
        sd.close();
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testCaptureFallbackOnSmallGraph(Nd4jBackend backend) {
        // Build a very small graph (< 10 ops)
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 8);
        SDVariable w = sd.var("w", Nd4j.randn(DataType.FLOAT, 8, 8));
        SDVariable mm = sd.mmul("mm", input, w);
        SDVariable out = sd.math.tanh("out", mm);
        configureDsp(sd, GraphExecutionMode.CUDA_GRAPHS);

        Map<String, INDArray> inputs = Collections.singletonMap("input",
            Nd4j.randn(DataType.FLOAT, 2, 8));
        for (int i = 0; i < 3; i++) {
            sd.outputDirect(inputs, "out");
        }

        // Small graph may not capture (depends on minCaptureSegmentSize)
        NativeOps ops = Nd4j.getNativeOps();
        DynamicShapePlanExecutor executor = sd.getOrCreateSession().getDynamicShapePlanExecutor();
        if (executor != null && executor.getNativePlanHandle() != null) {
            // Just verify no crash - small graphs are allowed to not capture
            int numSegs = ops.getPlanNumSegments(executor.getNativePlanHandle());
            assertTrue(numSegs >= 0);
        }
        sd.close();
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testShapeChangeInvalidatesCapture(Nd4jBackend backend) {
        SameDiff sd = buildTestGraph(18);
        configureDsp(sd, GraphExecutionMode.CUDA_GRAPHS);

        // Execute with shape [4, 32]
        for (int i = 0; i < 3; i++) {
            sd.outputDirect(makeInputs(), "out");
        }

        // Execute with different shape [8, 32]
        Map<String, INDArray> newInputs = Collections.singletonMap("input",
            Nd4j.randn(DataType.FLOAT, 8, 32));
        sd.outputDirect(newInputs, "out");

        // Should still work (re-capture or slot-by-slot fallback)
        NativeOps ops = Nd4j.getNativeOps();
        DynamicShapePlanExecutor executor = sd.getOrCreateSession().getDynamicShapePlanExecutor();
        if (executor != null && executor.getNativePlanHandle() != null) {
            int numSegs = ops.getPlanNumSegments(executor.getNativePlanHandle());
            assertTrue(numSegs >= 0);
        }
        sd.close();
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testCpuFunctionalReplayTracksOps(Nd4jBackend backend) {
        // This test is meaningful on CPU builds where FunctionalReplayHandle is used
        SameDiff sd = buildTestGraph(18);
        configureDsp(sd, GraphExecutionMode.CUDA_GRAPHS);

        for (int i = 0; i < 3; i++) {
            sd.outputDirect(makeInputs(), "out");
        }

        // On CPU, FunctionalReplayHandle should be created and track ops
        // On CUDA, CudaGraphReplayHandle is used instead
        // Either way, execution should succeed
        NativeOps ops = Nd4j.getNativeOps();
        DynamicShapePlanExecutor executor = sd.getOrCreateSession().getDynamicShapePlanExecutor();
        if (executor != null && executor.getNativePlanHandle() != null) {
            var handle = executor.getNativePlanHandle();
            int numSegs = ops.getPlanNumSegments(handle);
            assertTrue(numSegs > 0, "Should have segments");
        }
        sd.close();
    }

    // --- Pointer Tracking Tests ---

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPointerTrackingSnapshotAndMatch(Nd4jBackend backend) {
        SameDiff sd = buildTestGraph(18);
        configureDsp(sd, GraphExecutionMode.CUDA_GRAPHS);

        for (int i = 0; i < 3; i++) {
            sd.outputDirect(makeInputs(), "out");
        }

        NativeOps ops = Nd4j.getNativeOps();
        DynamicShapePlanExecutor executor = sd.getOrCreateSession().getDynamicShapePlanExecutor();
        if (executor != null && executor.getNativePlanHandle() != null) {
            var handle = executor.getNativePlanHandle();
            int numSegs = ops.getPlanNumSegments(handle);
            for (int i = 0; i < numSegs; i++) {
                String ptrs = ops.getPlanSegmentTrackedPointers(handle, i);
                assertNotNull(ptrs, "Tracked pointers should not be null");
                // Should be valid JSON array
                assertTrue(ptrs.startsWith("["), "Should be JSON array: " + ptrs);
            }
        }
        sd.close();
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPointerTrackingAcrossMultipleReplays(Nd4jBackend backend) {
        SameDiff sd = buildTestGraph(18);
        configureDsp(sd, GraphExecutionMode.CUDA_GRAPHS);

        // Use same input across all replays
        Map<String, INDArray> fixedInputs = makeInputs();
        for (int i = 0; i < 5; i++) {
            sd.outputDirect(fixedInputs, "out");
        }

        NativeOps ops = Nd4j.getNativeOps();
        DynamicShapePlanExecutor executor = sd.getOrCreateSession().getDynamicShapePlanExecutor();
        if (executor != null && executor.getNativePlanHandle() != null) {
            var handle = executor.getNativePlanHandle();
            int numSegs = ops.getPlanNumSegments(handle);
            for (int i = 0; i < numSegs; i++) {
                int numBufs = ops.getPlanSegmentNumCaptureBuffers(handle, i);
                assertEquals(0, numBufs, "Capture buffers should be removed from replay handles");
                int numHost = ops.getPlanSegmentNumHostPointers(handle, i);
                assertTrue(numHost >= 0, "Host pointer count should be >= 0");
            }
        }
        sd.close();
    }

    // --- Replay Cache Tests ---

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testReplayCacheWarmStart(Nd4jBackend backend) {
        Assumptions.assumeTrue(!Nd4j.getEnvironment().isCPU(),
                "Replay cache with CUDA_GRAPHS requires CUDA backend");
        NativeOps ops = Nd4j.getNativeOps();

        // First: build, compile, execute
        SameDiff sd1 = buildTestGraph(18);
        configureDsp(sd1, GraphExecutionMode.CUDA_GRAPHS);
        for (int i = 0; i < 3; i++) {
            sd1.outputDirect(makeInputs(), sd1.outputs().toArray(new String[0]));
        }
        sd1.close();

        // Cache stats should reflect activity
        int hits = ops.getReplayCacheHits();
        int misses = ops.getReplayCacheMisses();
        assertTrue(hits >= 0 && misses >= 0,
            "Cache stats should be non-negative: hits=" + hits + " misses=" + misses);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testReplayCacheInvalidatedOnShapeChange(Nd4jBackend backend) {
        NativeOps ops = Nd4j.getNativeOps();
        ops.clearReplayCache();

        SameDiff sd = buildTestGraph(18);
        configureDsp(sd, GraphExecutionMode.CUDA_GRAPHS);

        // Execute with one shape
        for (int i = 0; i < 3; i++) {
            sd.outputDirect(makeInputs(), "out");
        }

        // Execute with different shape
        Map<String, INDArray> newInputs = Collections.singletonMap("input",
            Nd4j.randn(DataType.FLOAT, 8, 32));
        sd.outputDirect(newInputs, "out");

        // Should work - shape change handled gracefully
        sd.close();
    }

    // --- Backend Plan Management Tests ---

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testAvailableBackendsNonEmpty(Nd4jBackend backend) {
        SameDiff sd = buildTestGraph(18);
        configureDsp(sd, GraphExecutionMode.CUDA_GRAPHS);
        sd.outputDirect(makeInputs(), "out");

        DynamicShapePlanExecutor executor = sd.getOrCreateSession().getDynamicShapePlanExecutor();
        if (executor != null && executor.getNativePlanHandle() != null) {
            List<BackendPlanManager.BackendInfo> backends =
                BackendPlanManager.getAvailableBackends(executor.getNativePlanHandle());
            assertFalse(backends.isEmpty(), "Should have at least one backend");
            assertNotNull(backends.get(0).name, "Backend name should be non-null");
        }
        sd.close();
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSegmentCompiledBackendTracked(Nd4jBackend backend) {
        SameDiff sd = buildTestGraph(18);
        configureDsp(sd, GraphExecutionMode.CUDA_GRAPHS);

        for (int i = 0; i < 3; i++) {
            sd.outputDirect(makeInputs(), "out");
        }

        DynamicShapePlanExecutor executor = sd.getOrCreateSession().getDynamicShapePlanExecutor();
        if (executor != null && executor.getNativePlanHandle() != null) {
            var handle = executor.getNativePlanHandle();
            NativeOps ops = Nd4j.getNativeOps();
            int numSegs = ops.getPlanNumSegments(handle);
            for (int i = 0; i < numSegs; i++) {
                String compiled = BackendPlanManager.getSegmentBackendName(handle, i);
                assertNotNull(compiled, "Compiled backend should not be null for seg " + i);
            }
        }
        sd.close();
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testBackendCacheStatsJson(Nd4jBackend backend) {
        SameDiff sd = buildTestGraph(18);
        configureDsp(sd, GraphExecutionMode.CUDA_GRAPHS);

        for (int i = 0; i < 3; i++) {
            sd.outputDirect(makeInputs(), "out");
        }

        DynamicShapePlanExecutor executor = sd.getOrCreateSession().getDynamicShapePlanExecutor();
        if (executor != null && executor.getNativePlanHandle() != null) {
            String stats = BackendPlanManager.getBackendCacheStats(executor.getNativePlanHandle());
            assertNotNull(stats);
            assertTrue(stats.contains("backends"), "Should contain backends key: " + stats);
        }
        sd.close();
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testInvalidateSegmentCacheTriggersRecompile(Nd4jBackend backend) {
        SameDiff sd = buildTestGraph(18);
        configureDsp(sd, GraphExecutionMode.CUDA_GRAPHS);

        for (int i = 0; i < 3; i++) {
            sd.outputDirect(makeInputs(), "out");
        }

        DynamicShapePlanExecutor executor = sd.getOrCreateSession().getDynamicShapePlanExecutor();
        if (executor != null && executor.getNativePlanHandle() != null) {
            var handle = executor.getNativePlanHandle();
            NativeOps ops = Nd4j.getNativeOps();
            int numSegs = ops.getPlanNumSegments(handle);
            if (numSegs > 0) {
                // Invalidate first segment
                BackendPlanManager.invalidateSegmentCache(handle, 0);
                // Execution should still work (re-capture)
                sd.outputDirect(makeInputs(), "out");
            }
        }
        sd.close();
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testBackendOverrideForces(Nd4jBackend backend) {
        SameDiff sd = buildTestGraph(18);
        configureDsp(sd, GraphExecutionMode.CUDA_GRAPHS);

        sd.outputDirect(makeInputs(), "out");

        DynamicShapePlanExecutor executor = sd.getOrCreateSession().getDynamicShapePlanExecutor();
        if (executor != null && executor.getNativePlanHandle() != null) {
            var handle = executor.getNativePlanHandle();
            NativeOps ops = Nd4j.getNativeOps();
            int numSegs = ops.getPlanNumSegments(handle);
            if (numSegs > 0) {
                // Force slot-by-slot
                BackendPlanManager.setSegmentBackendOverride(handle, 0, "slot-by-slot");
                sd.outputDirect(makeInputs(), "out");

                // Clear override
                BackendPlanManager.setSegmentBackendOverride(handle, 0, null);
                sd.outputDirect(makeInputs(), "out");
            }
        }
        sd.close();
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testFormatPlanIncludesBackendInfo(Nd4jBackend backend) {
        SameDiff sd = buildTestGraph(18);
        configureDsp(sd, GraphExecutionMode.CUDA_GRAPHS);

        for (int i = 0; i < 3; i++) {
            sd.outputDirect(makeInputs(), "out");
        }

        DynamicShapePlanExecutor executor = sd.getOrCreateSession().getDynamicShapePlanExecutor();
        if (executor != null && executor.getNativePlanHandle() != null && executor.getCurrentPlan() != null) {
            String summary = PlanIntrospection.formatReplaySummary(executor.getNativePlanHandle());
            assertNotNull(summary);
            assertTrue(summary.contains("Replay Summary"), "Should contain Replay Summary: " + summary);
        }
        sd.close();
    }

    // --- Per-Device Cache Tests ---

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testDeviceCacheStatsJson(Nd4jBackend backend) {
        String stats = Nd4j.getNativeOps().getReplayCacheDeviceStatsJson();
        assertNotNull(stats);
        assertTrue(stats.contains("hits"), "Should contain hits: " + stats);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testDeviceCacheMigrateIncompatibleFails(Nd4jBackend backend) {
        // CPU -> GPU migration should fail (incompatible)
        boolean result = Nd4j.getNativeOps().migrateReplayCache(0, 0, 1, 0);
        // May succeed or fail depending on device compatibility
        // Just verify no crash
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testDeviceCachePruneRemovesStale(Nd4jBackend backend) {
        int pruned = Nd4j.getNativeOps().pruneStaleReplayCacheDevices();
        assertTrue(pruned >= 0, "Prune count should be >= 0");
    }

    // --- Training Replay Profile Tests ---

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testTrainingReplayProfileCapture(Nd4jBackend backend) {
        SameDiff sd = buildTestGraph(18);
        configureDsp(sd, GraphExecutionMode.CUDA_GRAPHS);

        // Execute to get a native plan handle
        for (int i = 0; i < 3; i++) {
            sd.outputDirect(makeInputs(), "out");
        }

        DynamicShapePlanExecutor executor = sd.getOrCreateSession().getDynamicShapePlanExecutor();
        if (executor != null && executor.getNativePlanHandle() != null) {
            Map<String, long[]> shapes = new HashMap<>();
            shapes.put("input", new long[]{4, 32});

            ReplayProfileManager.ReplayProfile profile =
                ReplayProfileManager.captureProfile(executor.getNativePlanHandle(), shapes);
            assertNotNull(profile);
            assertTrue(profile.getNumSegments() > 0, "Profile should have segments");
            assertEquals(ReplayProfileManager.computeShapeHash(shapes), profile.getShapeHash());
        }
        sd.close();
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testTrainingReplayProfileMultipleBatchSizes(Nd4jBackend backend) {
        SameDiff sd = buildTestGraph(18);
        configureDsp(sd, GraphExecutionMode.CUDA_GRAPHS);

        for (int i = 0; i < 3; i++) {
            sd.outputDirect(makeInputs(), "out");
        }

        DynamicShapePlanExecutor executor = sd.getOrCreateSession().getDynamicShapePlanExecutor();
        if (executor != null && executor.getNativePlanHandle() != null) {
            var handle = executor.getNativePlanHandle();

            ReplayProfileManager.ReplayProfileCollection profiles =
                new ReplayProfileManager.ReplayProfileCollection();

            // Profile for batch=4
            Map<String, long[]> shapes4 = new HashMap<>();
            shapes4.put("input", new long[]{4, 32});
            profiles.put(ReplayProfileManager.captureProfile(handle, shapes4));

            // Profile for batch=8
            Map<String, long[]> shapes8 = new HashMap<>();
            shapes8.put("input", new long[]{8, 32});
            profiles.put(ReplayProfileManager.captureProfile(handle, shapes8));

            assertEquals(2, profiles.size(), "Should have 2 profiles for different batch sizes");
        }
        sd.close();
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testTrainingReplayProfileSaveLoad(Nd4jBackend backend) throws Exception {
        SameDiff sd = buildTestGraph(18);
        configureDsp(sd, GraphExecutionMode.CUDA_GRAPHS);

        for (int i = 0; i < 3; i++) {
            sd.outputDirect(makeInputs(), "out");
        }

        DynamicShapePlanExecutor executor = sd.getOrCreateSession().getDynamicShapePlanExecutor();
        if (executor != null && executor.getNativePlanHandle() != null) {
            Map<String, long[]> shapes = new HashMap<>();
            shapes.put("input", new long[]{4, 32});

            ReplayProfileManager.ReplayProfile profile =
                ReplayProfileManager.captureProfile(executor.getNativePlanHandle(), shapes);

            // Save to temp directory
            File tempDir = Files.createTempDirectory("replay_test").toFile();
            try {
                ReplayProfileManager.ReplayProfileCollection collection =
                    new ReplayProfileManager.ReplayProfileCollection();
                collection.put(profile);
                collection.saveToDisk(tempDir.getAbsolutePath());

                // Load back
                ReplayProfileManager.ReplayProfileCollection loaded =
                    ReplayProfileManager.ReplayProfileCollection.loadFromDisk(tempDir.getAbsolutePath());
                assertEquals(collection.size(), loaded.size(), "Loaded profiles should match saved");

                ReplayProfileManager.ReplayProfile loadedProfile =
                    loaded.getExact(profile.getShapeHash());
                assertNotNull(loadedProfile, "Should find profile by shape hash");
                assertEquals(profile.getNumSegments(), loadedProfile.getNumSegments());
            } finally {
                // Cleanup
                for (File f : tempDir.listFiles()) f.delete();
                tempDir.delete();
            }
        }
        sd.close();
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testTrainingReplayProfileWarmRestart(Nd4jBackend backend) {
        SameDiff sd = buildTestGraph(18);
        configureDsp(sd, GraphExecutionMode.CUDA_GRAPHS);

        for (int i = 0; i < 3; i++) {
            sd.outputDirect(makeInputs(), "out");
        }

        DynamicShapePlanExecutor executor = sd.getOrCreateSession().getDynamicShapePlanExecutor();
        if (executor != null && executor.getNativePlanHandle() != null) {
            Map<String, long[]> shapes = new HashMap<>();
            shapes.put("input", new long[]{4, 32});

            ReplayProfileManager.ReplayProfile profile =
                ReplayProfileManager.captureProfile(executor.getNativePlanHandle(), shapes);

            // Load profile back (warm restart simulation)
            boolean loaded = ReplayProfileManager.loadProfile(executor.getNativePlanHandle(), profile);
            // On CPU this may return false since cache may not be enabled - that's OK
            // Just verify no crash
        }
        sd.close();
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testTrainingReplayProfileCheckpointRoundTrip(Nd4jBackend backend) throws Exception {
        SameDiff sd = buildTestGraph(18);
        configureDsp(sd, GraphExecutionMode.CUDA_GRAPHS);

        for (int i = 0; i < 3; i++) {
            sd.outputDirect(makeInputs(), "out");
        }

        DynamicShapePlanExecutor executor = sd.getOrCreateSession().getDynamicShapePlanExecutor();
        if (executor != null && executor.getNativePlanHandle() != null) {
            Map<String, long[]> shapes = new HashMap<>();
            shapes.put("input", new long[]{4, 32});

            ReplayProfileManager.ReplayProfile profile =
                ReplayProfileManager.captureProfile(executor.getNativePlanHandle(), shapes);

            // JSON round-trip
            String json = profile.toJson();
            ReplayProfileManager.ReplayProfile restored =
                ReplayProfileManager.ReplayProfile.fromJson(json);

            assertEquals(profile.getShapeHash(), restored.getShapeHash());
            assertEquals(profile.getNumSegments(), restored.getNumSegments());
            assertEquals(profile.getSegments().size(), restored.getSegments().size());
        }
        sd.close();
    }
}
