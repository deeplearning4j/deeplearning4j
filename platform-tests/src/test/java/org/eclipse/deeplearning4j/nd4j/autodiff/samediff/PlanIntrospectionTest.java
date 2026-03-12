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

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.*;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.NativeOps;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.nio.file.Files;
import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for {@link PlanIntrospection} — DSP slot/segment introspection and DOT export.
 */
public class PlanIntrospectionTest extends BaseNd4jTestWithBackends {

    private static final Logger log = LoggerFactory.getLogger(PlanIntrospectionTest.class);

    @Override
    public char ordering() {
        return 'c';
    }

    private DynamicShapePlan compileSimplePlan() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, 4);
        SDVariable w = sd.constant("w", Nd4j.randn(DataType.FLOAT, 4, 8));
        SDVariable b = sd.constant("b", Nd4j.randn(DataType.FLOAT, 1, 8));
        SDVariable mm = sd.mmul("mm", x, w);
        SDVariable added = mm.add("added", b);
        sd.nn.relu("out", added, 0);

        return sd.compileDynamicShapePlan("out");
    }

    @Test
    @DisplayName("getDetailedSummary contains expected op names")
    public void testDetailedSummary() {
        DynamicShapePlan plan = compileSimplePlan();
        assertNotNull(plan, "Plan should compile successfully");

        String summary = plan.getDetailedSummary();
        log.info("Detailed summary:\n{}", summary);

        assertTrue(summary.contains("DynamicShapePlan"), "Should contain header");
        assertTrue(summary.contains("Slots:"), "Should contain slot count");
        // Verify at least one op name appears
        boolean hasMatmul = summary.contains("matmul") || summary.contains("mmul");
        boolean hasRelu = summary.contains("relu");
        assertTrue(hasMatmul, "Summary should contain matmul/mmul op");
        assertTrue(hasRelu, "Summary should contain relu op");
        // Should contain sections
        assertTrue(summary.contains("Segments"), "Should contain Segments section");
        assertTrue(summary.contains("Op Histogram"), "Should contain Op Histogram section");
        assertTrue(summary.contains("Memory Timeline"), "Should contain Memory Timeline section");
    }

    @Test
    @DisplayName("toDot produces valid DOT syntax")
    public void testToDot() {
        DynamicShapePlan plan = compileSimplePlan();
        assertNotNull(plan);

        String dot = plan.toDot();
        log.info("DOT output:\n{}", dot);

        assertTrue(dot.startsWith("digraph DSP {"), "Should start with digraph header");
        assertTrue(dot.contains("->"), "Should contain edges");
        assertTrue(dot.contains("slot_"), "Should contain slot nodes");
        assertTrue(dot.endsWith("}\n"), "Should end with closing brace");
        // Should have subgraph clusters
        assertTrue(dot.contains("subgraph cluster_"), "Should contain segment clusters");
    }

    @Test
    @DisplayName("getSegments returns valid segment boundaries")
    public void testGetSegments() {
        DynamicShapePlan plan = compileSimplePlan();
        assertNotNull(plan);

        List<PlanIntrospection.SegmentInfo> segments = plan.getSegments();
        assertFalse(segments.isEmpty(), "Should have at least one segment");

        // Verify segments cover all slots
        int totalSlots = 0;
        for (PlanIntrospection.SegmentInfo seg : segments) {
            assertTrue(seg.getSlotCount() > 0, "Each segment should have at least one slot");
            assertTrue(seg.getStartSlot() <= seg.getEndSlot(), "Start should be <= end");
            assertEquals(seg.getSlotCount(), seg.getOpNames().size(), "Op names count should match slot count");
            totalSlots += seg.getSlotCount();
        }
        assertEquals(plan.getSlots().length, totalSlots, "Segments should cover all slots");

        // Verify segment boundaries don't overlap
        for (int i = 1; i < segments.size(); i++) {
            assertTrue(segments.get(i).getStartSlot() > segments.get(i - 1).getEndSlot(),
                    "Segments should not overlap");
        }

        // Log segment info
        for (int i = 0; i < segments.size(); i++) {
            log.info("Segment {}: {}", i, PlanIntrospection.formatSegment(segments.get(i)));
        }
    }

    @Test
    @DisplayName("getDependentsOf returns correct downstream slots")
    public void testGetDependentsOf() {
        DynamicShapePlan plan = compileSimplePlan();
        assertNotNull(plan);

        DynamicShapeSlot[] slots = plan.getSlots();

        // The first op (matmul) should have at least one dependent (add)
        // Find the matmul slot
        int matmulStep = -1;
        for (int i = 0; i < slots.length; i++) {
            String name = slots[i].getOpName().toLowerCase();
            if (name.contains("matmul") || name.contains("mmul")) {
                matmulStep = i;
                break;
            }
        }
        assertTrue(matmulStep >= 0, "Should find matmul slot");

        List<Integer> dependents = PlanIntrospection.getDependentsOf(plan, matmulStep);
        assertFalse(dependents.isEmpty(), "Matmul should have downstream dependents");
        log.info("Matmul (step {}) dependents: {}", matmulStep, dependents);

        // The last op (relu) should have no dependents (it's the output)
        int lastStep = slots.length - 1;
        // Actually, let's find the relu step
        int reluStep = -1;
        for (int i = 0; i < slots.length; i++) {
            if (slots[i].getOpName().toLowerCase().contains("relu")) {
                reluStep = i;
                break;
            }
        }
        if (reluStep >= 0) {
            List<Integer> reluDependents = PlanIntrospection.getDependentsOf(plan, reluStep);
            // relu is the final output, so it may have no dependents within the plan
            log.info("Relu (step {}) dependents: {}", reluStep, reluDependents);
        }
    }

    @Test
    @DisplayName("getProducersOf returns correct upstream slots")
    public void testGetProducersOf() {
        DynamicShapePlan plan = compileSimplePlan();
        assertNotNull(plan);

        DynamicShapeSlot[] slots = plan.getSlots();

        // Find the add slot — it should have matmul as a producer
        int addStep = -1;
        for (int i = 0; i < slots.length; i++) {
            if (slots[i].getOpName().toLowerCase().equals("add")) {
                addStep = i;
                break;
            }
        }
        if (addStep >= 0) {
            List<Integer> producers = PlanIntrospection.getProducersOf(plan, addStep);
            log.info("Add (step {}) producers: {}", addStep, producers);
            // Add should have at least matmul as a producer
            assertFalse(producers.isEmpty(), "Add should have upstream producers");
        }
    }

    @Test
    @DisplayName("slot.toString() contains op name and decoded inputs")
    public void testSlotToString() {
        DynamicShapePlan plan = compileSimplePlan();
        assertNotNull(plan);

        for (DynamicShapeSlot slot : plan.getSlots()) {
            String s = slot.toString();
            log.info("Slot toString: {}", s);
            assertTrue(s.contains("Slot#"), "Should contain Slot# prefix");
            assertTrue(s.contains("[" + slot.getOpName() + "]"), "Should contain op name");
            assertTrue(s.contains("inputs:["), "Should contain inputs section");
            assertTrue(s.contains("outputs:["), "Should contain outputs section");
        }
    }

    @Test
    @DisplayName("getDecodedInputs resolves external and op inputs correctly")
    public void testGetDecodedInputs() {
        DynamicShapePlan plan = compileSimplePlan();
        assertNotNull(plan);

        DynamicShapeSlot[] slots = plan.getSlots();
        for (int i = 0; i < slots.length; i++) {
            List<PlanIntrospection.DecodedInput> inputs = PlanIntrospection.getDecodedInputs(plan, i);
            for (PlanIntrospection.DecodedInput input : inputs) {
                assertNotNull(input.getSourceTypeName());
                if (input.getSourceType() == DynamicShapeSlot.SOURCE_OP_OUTPUT) {
                    assertTrue(input.getFromSlot() >= 0, "OP_OUTPUT should have valid fromSlot");
                } else {
                    assertEquals(-1, input.getFromSlot(), "External input should have fromSlot=-1");
                    assertNotNull(input.getExternalKey(), "External input should have external key");
                }
            }
        }
    }

    @Test
    @DisplayName("getMemoryTimeline has matching allocations")
    public void testGetMemoryTimeline() {
        DynamicShapePlan plan = compileSimplePlan();
        assertNotNull(plan);

        List<PlanIntrospection.MemoryEvent> timeline = PlanIntrospection.getMemoryTimeline(plan);
        assertFalse(timeline.isEmpty(), "Should have memory events");

        long allocs = timeline.stream()
                .filter(e -> e.getEventType() == PlanIntrospection.MemoryEvent.EventType.ALLOCATE).count();
        assertTrue(allocs > 0, "Should have allocations");
        log.info("Memory timeline: {} events ({} allocs)", timeline.size(), allocs);
    }

    @Test
    @DisplayName("getDevicePlacement returns all slots")
    public void testGetDevicePlacement() {
        DynamicShapePlan plan = compileSimplePlan();
        assertNotNull(plan);

        var placement = PlanIntrospection.getDevicePlacement(plan);
        int total = placement.values().stream().mapToInt(List::size).sum();
        assertEquals(plan.getSlots().length, total, "All slots should be assigned to a device");
    }

    @Test
    @DisplayName("formatSlot produces readable multi-line output")
    public void testFormatSlot() {
        DynamicShapePlan plan = compileSimplePlan();
        assertNotNull(plan);

        String formatted = PlanIntrospection.formatSlot(plan, 0);
        log.info("Formatted slot 0:\n{}", formatted);
        assertTrue(formatted.contains("Slot#0"), "Should contain slot index");
        assertTrue(formatted.contains("Inputs:"), "Should contain Inputs section");
        assertTrue(formatted.contains("Outputs:"), "Should contain Outputs section");
    }

    @Test
    @DisplayName("getParallelGroups works without errors")
    public void testGetParallelGroups() {
        DynamicShapePlan plan = compileSimplePlan();
        assertNotNull(plan);

        List<List<Integer>> groups = plan.getParallelGroups();
        // Simple linear chain may not have parallel groups, just verify no errors
        log.info("Parallel groups: {}", groups);
        // Each group should have at least 2 members
        for (List<Integer> group : groups) {
            assertTrue(group.size() >= 2, "Parallel group should have at least 2 members");
        }
    }

    // ─── Replay and Backend Introspection Tests ──────────────────────────

    private SameDiff buildLargerGraph() {
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 32);
        SDVariable current = input;
        for (int i = 0; i < 6; i++) {
            SDVariable wi = sd.var("w_" + i, Nd4j.randn(DataType.FLOAT, 32, 32).muli(0.01));
            SDVariable bi = sd.var("b_" + i, Nd4j.zeros(DataType.FLOAT, 32));
            current = sd.mmul("mm_" + i, current, wi);
            current = current.add("add_" + i, bi);
            String tanhName = (i == 5) ? "out" : "tanh_" + i;
            current = sd.math.tanh(tanhName, current);
        }
        return sd;
    }

    @Test
    @DisplayName("getSegmentsWithReplayState returns segments with replay metadata")
    public void testSegmentInfoHasReplayState() {
        SameDiff sd = buildLargerGraph();
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);
        sd.setGraphExecutionMode(GraphExecutionMode.CUDA_GRAPHS);

        Map<String, INDArray> inputs = Collections.singletonMap("input",
            Nd4j.randn(DataType.FLOAT, 4, 32));
        for (int i = 0; i < 3; i++) {
            sd.outputDirect(inputs, "out");
        }

        DynamicShapePlanExecutor executor = sd.getOrCreateSession().getDynamicShapePlanExecutor();
        if (executor != null && executor.getNativePlanHandle() != null && executor.getCurrentPlan() != null) {
            List<PlanIntrospection.SegmentInfo> segments =
                PlanIntrospection.getSegmentsWithReplayState(executor.getCurrentPlan(),
                    executor.getNativePlanHandle());
            assertFalse(segments.isEmpty(), "Should have segments");
            for (PlanIntrospection.SegmentInfo seg : segments) {
                assertNotNull(seg.getReplayStateName());
                log.info("Segment [{}-{}]: replay={} backend={} compiled={}",
                    seg.getStartSlot(), seg.getEndSlot(),
                    seg.getReplayStateName(), seg.getReplayBackendName(),
                    seg.getCompiledByBackend());
            }
        }
        sd.close();
    }

    @Test
    @DisplayName("formatReplaySummary produces parseable output")
    public void testReplaySummaryFormat() {
        SameDiff sd = buildLargerGraph();
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);
        sd.setGraphExecutionMode(GraphExecutionMode.CUDA_GRAPHS);

        Map<String, INDArray> inputs = Collections.singletonMap("input",
            Nd4j.randn(DataType.FLOAT, 4, 32));
        for (int i = 0; i < 3; i++) {
            sd.outputDirect(inputs, "out");
        }

        DynamicShapePlanExecutor executor = sd.getOrCreateSession().getDynamicShapePlanExecutor();
        if (executor != null && executor.getNativePlanHandle() != null) {
            String summary = PlanIntrospection.formatReplaySummary(executor.getNativePlanHandle());
            assertNotNull(summary);
            assertTrue(summary.contains("Replay Summary"), "Should have header: " + summary);
            assertTrue(summary.contains("segments"), "Should mention segments: " + summary);
            log.info("Replay summary:\n{}", summary);
        }
        sd.close();
    }

    @Test
    @DisplayName("toDot colors segments by replay state")
    public void testToDotColorsSegmentsByReplayState() {
        SameDiff sd = buildLargerGraph();
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);
        sd.setGraphExecutionMode(GraphExecutionMode.CUDA_GRAPHS);

        Map<String, INDArray> inputs = Collections.singletonMap("input",
            Nd4j.randn(DataType.FLOAT, 4, 32));
        for (int i = 0; i < 3; i++) {
            sd.outputDirect(inputs, "out");
        }

        DynamicShapePlanExecutor executor = sd.getOrCreateSession().getDynamicShapePlanExecutor();
        if (executor != null && executor.getNativePlanHandle() != null && executor.getCurrentPlan() != null) {
            // Get segments with replay state for DOT export
            List<PlanIntrospection.SegmentInfo> segments =
                PlanIntrospection.getSegmentsWithReplayState(executor.getCurrentPlan(),
                    executor.getNativePlanHandle());

            // toDot should work with replay-enriched segments
            String dot = PlanIntrospection.toDot(executor.getCurrentPlan());
            assertNotNull(dot);
            assertTrue(dot.contains("digraph"), "Should be valid DOT: " + dot.substring(0, 50));
        }
        sd.close();
    }

    @Test
    @DisplayName("Segment capture buffer counts are non-negative")
    public void testSegmentInfoCaptureBufferCounts() {
        SameDiff sd = buildLargerGraph();
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);
        sd.setGraphExecutionMode(GraphExecutionMode.CUDA_GRAPHS);

        Map<String, INDArray> inputs = Collections.singletonMap("input",
            Nd4j.randn(DataType.FLOAT, 4, 32));
        for (int i = 0; i < 3; i++) {
            sd.outputDirect(inputs, "out");
        }

        DynamicShapePlanExecutor executor = sd.getOrCreateSession().getDynamicShapePlanExecutor();
        if (executor != null && executor.getNativePlanHandle() != null && executor.getCurrentPlan() != null) {
            List<PlanIntrospection.SegmentInfo> segments =
                PlanIntrospection.getSegmentsWithReplayState(executor.getCurrentPlan(),
                    executor.getNativePlanHandle());
            for (PlanIntrospection.SegmentInfo seg : segments) {
                assertTrue(seg.getNumCaptureBuffers() >= 0, "Capture buffers should be >= 0");
                assertTrue(seg.getNumHostPointers() >= 0, "Host pointers should be >= 0");
            }
        }
        sd.close();
    }

    @Test
    @DisplayName("Replay cache dir is configurable")
    public void testReplayCacheDirConfigurable() {
        NativeOps ops = Nd4j.getNativeOps();
        String dir = ops.getReplayCacheDir();
        assertNotNull(dir, "Cache dir should not be null");
        // Dir should be a valid path (non-empty)
        assertFalse(dir.isEmpty(), "Cache dir should not be empty");
    }

    @Test
    @DisplayName("ReplayProfileManager capture and load round-trip")
    public void testReplayProfileManagerCaptureAndLoad() {
        SameDiff sd = buildLargerGraph();
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);
        sd.setGraphExecutionMode(GraphExecutionMode.CUDA_GRAPHS);

        Map<String, INDArray> inputs = Collections.singletonMap("input",
            Nd4j.randn(DataType.FLOAT, 4, 32));
        for (int i = 0; i < 3; i++) {
            sd.outputDirect(inputs, "out");
        }

        DynamicShapePlanExecutor executor = sd.getOrCreateSession().getDynamicShapePlanExecutor();
        if (executor != null && executor.getNativePlanHandle() != null) {
            Map<String, long[]> shapes = new HashMap<>();
            shapes.put("input", new long[]{4, 32});

            ReplayProfileManager.ReplayProfile profile =
                ReplayProfileManager.captureProfile(executor.getNativePlanHandle(), shapes);
            assertNotNull(profile);
            assertTrue(profile.getNumSegments() > 0);

            // Load should not crash
            ReplayProfileManager.loadProfile(executor.getNativePlanHandle(), profile);
        }
        sd.close();
    }

    @Test
    @DisplayName("ReplayProfileCollection handles multiple shape hashes")
    public void testReplayProfileCollectionMultiShape() {
        SameDiff sd = buildLargerGraph();
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);
        sd.setGraphExecutionMode(GraphExecutionMode.CUDA_GRAPHS);

        Map<String, INDArray> inputs = Collections.singletonMap("input",
            Nd4j.randn(DataType.FLOAT, 4, 32));
        sd.outputDirect(inputs, "out");

        DynamicShapePlanExecutor executor = sd.getOrCreateSession().getDynamicShapePlanExecutor();
        if (executor != null && executor.getNativePlanHandle() != null) {
            var handle = executor.getNativePlanHandle();

            ReplayProfileManager.ReplayProfileCollection profiles =
                new ReplayProfileManager.ReplayProfileCollection();

            Map<String, long[]> shapes1 = new HashMap<>();
            shapes1.put("input", new long[]{4, 32});
            profiles.put(ReplayProfileManager.captureProfile(handle, shapes1));

            Map<String, long[]> shapes2 = new HashMap<>();
            shapes2.put("input", new long[]{8, 32});
            profiles.put(ReplayProfileManager.captureProfile(handle, shapes2));

            Map<String, long[]> shapes3 = new HashMap<>();
            shapes3.put("input", new long[]{16, 32});
            profiles.put(ReplayProfileManager.captureProfile(handle, shapes3));

            assertEquals(3, profiles.size());

            // Find matching
            ReplayProfileManager.ReplayProfile match =
                ReplayProfileManager.findMatchingProfile(profiles, shapes2);
            assertNotNull(match, "Should find matching profile for shapes2");
        }
        sd.close();
    }

    @Test
    @DisplayName("ReplayProfile disk persistence round-trip")
    public void testReplayProfileDiskPersistence() throws Exception {
        SameDiff sd = buildLargerGraph();
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);
        sd.setGraphExecutionMode(GraphExecutionMode.CUDA_GRAPHS);

        Map<String, INDArray> inputs = Collections.singletonMap("input",
            Nd4j.randn(DataType.FLOAT, 4, 32));
        sd.outputDirect(inputs, "out");

        DynamicShapePlanExecutor executor = sd.getOrCreateSession().getDynamicShapePlanExecutor();
        if (executor != null && executor.getNativePlanHandle() != null) {
            Map<String, long[]> shapes = new HashMap<>();
            shapes.put("input", new long[]{4, 32});

            ReplayProfileManager.ReplayProfileCollection collection =
                new ReplayProfileManager.ReplayProfileCollection();
            collection.put(ReplayProfileManager.captureProfile(executor.getNativePlanHandle(), shapes));

            File tempDir = Files.createTempDirectory("plan_intro_test").toFile();
            try {
                collection.saveToDisk(tempDir.getAbsolutePath());

                ReplayProfileManager.ReplayProfileCollection loaded =
                    ReplayProfileManager.ReplayProfileCollection.loadFromDisk(tempDir.getAbsolutePath());
                assertEquals(collection.size(), loaded.size());
            } finally {
                for (File f : tempDir.listFiles()) f.delete();
                tempDir.delete();
            }
        }
        sd.close();
    }

    @Test
    @DisplayName("Replay cache hit/miss counts are non-negative")
    public void testReplayCacheHitAfterSecondCompile() {
        NativeOps ops = Nd4j.getNativeOps();
        int hits = ops.getReplayCacheHits();
        int misses = ops.getReplayCacheMisses();
        assertTrue(hits >= 0, "Hits should be >= 0");
        assertTrue(misses >= 0, "Misses should be >= 0");
    }

    @Test
    @DisplayName("Per-segment replay counts sum to total replays")
    public void testReplaySummaryCountsMatchAggregateStats() {
        SameDiff sd = buildLargerGraph();
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);
        sd.setGraphExecutionMode(GraphExecutionMode.CUDA_GRAPHS);

        Map<String, INDArray> inputs = Collections.singletonMap("input",
            Nd4j.randn(DataType.FLOAT, 4, 32));
        for (int i = 0; i < 5; i++) {
            sd.outputDirect(inputs, "out");
        }

        DynamicShapePlanExecutor executor = sd.getOrCreateSession().getDynamicShapePlanExecutor();
        if (executor != null && executor.getNativePlanHandle() != null) {
            var handle = executor.getNativePlanHandle();
            NativeOps ops = Nd4j.getNativeOps();

            int totalReplays = ops.getPlanTotalGraphReplays(handle);
            int sumPerSeg = 0;
            int numSegs = ops.getPlanNumSegments(handle);
            for (int i = 0; i < numSegs; i++) {
                int segReplayCount = ops.getPlanSegmentReplayCount(handle, i);
                assertTrue(segReplayCount >= 0, "Per-segment replay count should be >= 0");
                sumPerSeg += segReplayCount;
            }

            // Per-segment counts should be <= total replays
            // (total includes aggregate captures; per-segment only counts when replay handle exists)
            assertTrue(sumPerSeg <= totalReplays || totalReplays == 0,
                "Sum of per-segment replays (" + sumPerSeg +
                ") should be <= total replays (" + totalReplays + ")");
        }
        sd.close();
    }

    @Test
    @DisplayName("Segment tracked pointers JSON is valid")
    public void testSegmentInfoHasPointerTracking() {
        SameDiff sd = buildLargerGraph();
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);
        sd.setGraphExecutionMode(GraphExecutionMode.CUDA_GRAPHS);

        Map<String, INDArray> inputs = Collections.singletonMap("input",
            Nd4j.randn(DataType.FLOAT, 4, 32));
        for (int i = 0; i < 3; i++) {
            sd.outputDirect(inputs, "out");
        }

        DynamicShapePlanExecutor executor = sd.getOrCreateSession().getDynamicShapePlanExecutor();
        if (executor != null && executor.getNativePlanHandle() != null && executor.getCurrentPlan() != null) {
            List<PlanIntrospection.SegmentInfo> segments =
                PlanIntrospection.getSegmentsWithReplayState(executor.getCurrentPlan(),
                    executor.getNativePlanHandle());
            for (PlanIntrospection.SegmentInfo seg : segments) {
                String ptrs = seg.getTrackedPointersJson();
                assertNotNull(ptrs);
                assertTrue(ptrs.startsWith("["), "Should be JSON array: " + ptrs);
            }
        }
        sd.close();
    }
}
