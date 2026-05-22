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
package org.eclipse.deeplearning4j.nd4j.autodiff.samediff.execution;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlan;
import org.nd4j.autodiff.samediff.execution.DynamicShapeSlot;
import org.nd4j.autodiff.samediff.execution.PlanIntrospection;
import org.nd4j.autodiff.samediff.execution.PlanIntrospection.DecodedInput;
import org.nd4j.autodiff.samediff.execution.PlanIntrospection.SegmentInfo;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Isolated unit tests for {@link PlanIntrospection} — stateless query methods
 * on {@link DynamicShapePlan}. No native handle required; uses builder-based
 * slot construction.
 */
@DisplayName("PlanIntrospection unit tests")
public class PlanIntrospectionTest {

    // ─── Helpers ──────────────────────────────────────────────────────

    private static DynamicShapeSlot slot(String opName, int stepIndex,
                                          int[] inputSourceIndices, byte[] inputSourceTypes,
                                          String[] inputVarNames, int[] outputSlotIndices,
                                          String[] outputVarNames) {
        return DynamicShapeSlot.builder()
                .opName(opName)
                .stepIndex(stepIndex)
                .inputSourceIndices(inputSourceIndices)
                .inputSourceTypes(inputSourceTypes)
                .inputVarNames(inputVarNames)
                .outputSlotIndices(outputSlotIndices)
                .outputVarNames(outputVarNames)
                .build();
    }

    /**
     * Build a DynamicShapePlan with explicit external input keys.
     */
    private static DynamicShapePlan makePlan(DynamicShapeSlot[] slots, String[] extKeys) {
        int[][] releaseAtStep = new int[slots.length][];
        for (int i = 0; i < slots.length; i++) releaseAtStep[i] = new int[0];

        Set<String> requested = new LinkedHashSet<>();
        Map<String, Integer> outMap = new HashMap<>();
        for (int i = 0; i < slots.length; i++) {
            if (slots[i].getOutputVarNames() != null) {
                for (int j = 0; j < slots[i].getOutputVarNames().length; j++) {
                    String vn = slots[i].getOutputVarNames()[j];
                    if (vn != null && slots[i].getOutputSlotIndices() != null && j < slots[i].getOutputSlotIndices().length) {
                        outMap.put(vn, slots[i].getOutputSlotIndices()[j]);
                    }
                }
            }
        }
        if (extKeys.length > 0) {
            requested.add("out0");
        }

        return new DynamicShapePlan(
                slots, slots.length, releaseAtStep,
                null, extKeys, new byte[extKeys.length],
                requested, outMap,
                false, null, null, null, null, null, null
        );
    }

    /** Auto-extract ext keys from slot input var names, ordered by ext index */
    private static DynamicShapePlan makePlan(DynamicShapeSlot[] slots) {
        // Collect ext keys by their encoded index: idx<0 means extIdx = -(idx+1)
        TreeMap<Integer, String> extByIndex = new TreeMap<>();
        for (DynamicShapeSlot s : slots) {
            int[] srcIndices = s.getInputSourceIndices();
            String[] varNames = s.getInputVarNames();
            if (srcIndices == null || varNames == null) continue;
            for (int i = 0; i < srcIndices.length; i++) {
                if (srcIndices[i] < 0) {
                    int extIdx = -(srcIndices[i] + 1);
                    String name = i < varNames.length ? varNames[i] : "";
                    if (name != null && !name.isEmpty()) {
                        extByIndex.putIfAbsent(extIdx, name);
                    }
                }
            }
        }
        // Build ext keys array in order of ext index
        String[] extKeys;
        if (extByIndex.isEmpty()) {
            extKeys = new String[0];
        } else {
            int maxIdx = extByIndex.lastKey();
            extKeys = new String[maxIdx + 1];
            for (Map.Entry<Integer, String> e : extByIndex.entrySet()) {
                extKeys[e.getKey()] = e.getValue();
            }
            // Fill gaps with empty string
            for (int i = 0; i < extKeys.length; i++) {
                if (extKeys[i] == null) extKeys[i] = "";
            }
        }
        return makePlan(slots, extKeys);
    }

    // ─── getDecodedInputs ────────────────────────────────────────────

    @Test
    @DisplayName("getDecodedInputs: external constant inputs")
    void testDecodedInputs_ExternalConstant() {
        DynamicShapeSlot slot0 = slot("add", 0,
                new int[]{-1, -2},
                new byte[]{DynamicShapeSlot.SOURCE_CONSTANT, DynamicShapeSlot.SOURCE_CONSTANT},
                new String[]{"const_a", "const_b"},
                new int[]{0}, new String[]{"out0"});

        DynamicShapePlan plan = makePlan(new DynamicShapeSlot[]{slot0});

        List<DecodedInput> inputs = PlanIntrospection.getDecodedInputs(plan, 0);
        assertEquals(2, inputs.size());

        DecodedInput i0 = inputs.get(0);
        assertEquals(0, i0.getIndex());
        assertEquals(DynamicShapeSlot.SOURCE_CONSTANT, i0.getSourceType());
        assertEquals("const_a", i0.getSourceName());
        assertEquals("ext#0:\"const_a\"(CONSTANT)", i0.toString());

        DecodedInput i1 = inputs.get(1);
        assertEquals(1, i1.getIndex());
        assertEquals("const_b", i1.getSourceName());
        assertEquals("CONSTANT", i1.getSourceTypeName());
    }

    @Test
    @DisplayName("getDecodedInputs: op output as input")
    void testDecodedInputs_OpOutput() {
        DynamicShapeSlot slot0 = slot("matmul", 0,
                new int[]{-1, -2}, new byte[]{0, 0},
                new String[]{"a", "b"},
                new int[]{0}, new String[]{"out0"});
        DynamicShapeSlot slot1 = slot("relu", 1,
                new int[]{0}, new byte[]{3},
                new String[]{"", ""},
                new int[]{1}, new String[]{"out1"});

        DynamicShapePlan plan = makePlan(new DynamicShapeSlot[]{slot0, slot1});

        List<DecodedInput> inputs = PlanIntrospection.getDecodedInputs(plan, 1);
        assertEquals(1, inputs.size());
        DecodedInput inp = inputs.get(0);
        assertEquals(0, inp.getIndex());
        assertEquals(DynamicShapeSlot.SOURCE_OP_OUTPUT, inp.getSourceType());
        assertEquals("OP_OUTPUT", inp.getSourceTypeName());
        assertEquals("slot#0(OP)", inp.toString());
    }

    @Test
    @DisplayName("getDecodedInputs: mixed sources")
    void testDecodedInputs_MixedSources() {
        DynamicShapeSlot slot0 = slot("placeholder", 0,
                new int[]{-1}, new byte[]{2},
                new String[]{"input"},
                new int[]{0}, new String[]{"ph_out"});
        DynamicShapeSlot slot1 = slot("matmul", 1,
                new int[]{-1, -2}, new byte[]{0, 0},
                new String[]{"w", "x"},
                new int[]{1}, new String[]{"mm_out"});
        DynamicShapeSlot slot2 = slot("add", 2,
                new int[]{1, -3},
                new byte[]{DynamicShapeSlot.SOURCE_OP_OUTPUT, DynamicShapeSlot.SOURCE_PLACEHOLDER},
                new String[]{"", "bias"},
                new int[]{2}, new String[]{"out"});

        DynamicShapePlan plan = makePlan(new DynamicShapeSlot[]{slot0, slot1, slot2});

        List<DecodedInput> inputs = PlanIntrospection.getDecodedInputs(plan, 2);
        assertEquals(2, inputs.size());
        assertEquals("slot#1(OP)", inputs.get(0).toString());
        assertEquals("ext#2:\"bias\"(PLACEHOLDER)", inputs.get(1).toString());
    }

    // ─── getDependentsOf ─────────────────────────────────────────────

    @Test
    @DisplayName("getDependentsOf: simple chain")
    void testDependentsOf_SimpleChain() {
        DynamicShapeSlot s0 = slot("inp", 0, new int[]{-1}, new byte[]{0}, new String[]{"a"}, new int[]{0}, new String[]{"o0"});
        DynamicShapeSlot s1 = slot("relu", 1, new int[]{0}, new byte[]{3}, new String[]{""}, new int[]{1}, new String[]{"o1"});
        DynamicShapeSlot s2 = slot("out", 2, new int[]{1}, new byte[]{3}, new String[]{""}, new int[]{2}, new String[]{"o2"});

        DynamicShapePlan plan = makePlan(new DynamicShapeSlot[]{s0, s1, s2});

        assertEquals(List.of(1), PlanIntrospection.getDependentsOf(plan, 0));
        assertEquals(List.of(2), PlanIntrospection.getDependentsOf(plan, 1));
        assertTrue(PlanIntrospection.getDependentsOf(plan, 2).isEmpty());
    }

    @Test
    @DisplayName("getDependentsOf: fan-out")
    void testDependentsOf_FanOut() {
        DynamicShapeSlot s0 = slot("src", 0, new int[]{-1}, new byte[]{0}, new String[]{"a"}, new int[]{0}, new String[]{"o0"});
        DynamicShapeSlot s1 = slot("branch1", 1, new int[]{0}, new byte[]{3}, new String[]{""}, new int[]{1}, new String[]{"o1"});
        DynamicShapeSlot s2 = slot("branch2", 2, new int[]{0}, new byte[]{3}, new String[]{""}, new int[]{2}, new String[]{"o2"});

        DynamicShapePlan plan = makePlan(new DynamicShapeSlot[]{s0, s1, s2});

        List<Integer> deps = PlanIntrospection.getDependentsOf(plan, 0);
        assertEquals(2, deps.size());
        assertTrue(deps.contains(1));
        assertTrue(deps.contains(2));
    }

    // ─── getProducersOf ──────────────────────────────────────────────

    @Test
    @DisplayName("getProducersOf: simple chain upstream")
    void testProducersOf_SimpleChain() {
        DynamicShapeSlot s0 = slot("inp", 0, new int[]{-1}, new byte[]{0}, new String[]{"a"}, new int[]{0}, new String[]{"o0"});
        DynamicShapeSlot s1 = slot("relu", 1, new int[]{0}, new byte[]{3}, new String[]{""}, new int[]{1}, new String[]{"o1"});

        DynamicShapePlan plan = makePlan(new DynamicShapeSlot[]{s0, s1});

        assertEquals(List.of(0), PlanIntrospection.getProducersOf(plan, 1));
    }

    @Test
    @DisplayName("getProducersOf: multi-input slot")
    void testProducersOf_MultiInput() {
        DynamicShapeSlot s0 = slot("src0", 0, new int[]{-1}, new byte[]{0}, new String[]{"a"}, new int[]{0}, new String[]{"o0"});
        DynamicShapeSlot s1 = slot("src1", 1, new int[]{-2}, new byte[]{0}, new String[]{"b"}, new int[]{1}, new String[]{"o1"});
        DynamicShapeSlot s2 = slot("add", 2, new int[]{0, 1}, new byte[]{3, 3}, new String[]{"", ""}, new int[]{2}, new String[]{"o2"});

        DynamicShapePlan plan = makePlan(new DynamicShapeSlot[]{s0, s1, s2});

        List<Integer> producers = PlanIntrospection.getProducersOf(plan, 2);
        assertEquals(2, producers.size());
        assertTrue(producers.contains(0));
        assertTrue(producers.contains(1));
    }

    // ─── getMemoryTimeline ───────────────────────────────────────────

    @Test
    @DisplayName("getMemoryTimeline: basic allocation/release")
    void testMemoryTimeline_Basic() {
        DynamicShapeSlot s0 = slot("op0", 0, new int[]{-1}, new byte[]{0}, new String[]{"a"}, new int[]{0}, new String[]{"o0"});
        DynamicShapeSlot s1 = slot("op1", 1, new int[]{0}, new byte[]{3}, new String[]{""}, new int[]{1}, new String[]{"o1"});

        int[][] releaseAtStep = {{}, {0}};
        DynamicShapePlan plan = new DynamicShapePlan(
                new DynamicShapeSlot[]{s0, s1}, 2, releaseAtStep,
                null, new String[]{"a"}, new byte[]{0},
                new LinkedHashSet<>(List.of("out1")),
                Map.of("out1", 1), false, null, null, null, null, null, null
        );

        List<PlanIntrospection.MemoryEvent> timeline = PlanIntrospection.getMemoryTimeline(plan);
        assertEquals(3, timeline.size());
        assertEquals(0, timeline.get(0).getStep());
        assertEquals(PlanIntrospection.MemoryEvent.EventType.ALLOCATE, timeline.get(0).getEventType());
        assertEquals(0, timeline.get(0).getSlotIndex());
        assertEquals(1, timeline.get(1).getStep());
        assertEquals(PlanIntrospection.MemoryEvent.EventType.ALLOCATE, timeline.get(1).getEventType());
        assertEquals(1, timeline.get(2).getStep());
        assertEquals(PlanIntrospection.MemoryEvent.EventType.RELEASE, timeline.get(2).getEventType());
        assertEquals(0, timeline.get(2).getSlotIndex());
    }

    @Test
    @DisplayName("getMemoryTimeline: no releases")
    void testMemoryTimeline_NoReleases() {
        DynamicShapeSlot s0 = slot("op0", 0, new int[]{-1}, new byte[]{0}, new String[]{"a"}, new int[]{0}, new String[]{"o0"});
        int[][] releaseAtStep = {{}};
        DynamicShapePlan plan = new DynamicShapePlan(
                new DynamicShapeSlot[]{s0}, 1, releaseAtStep,
                null, new String[]{"a"}, new byte[]{0},
                new LinkedHashSet<>(List.of("out0")),
                Map.of("out0", 0), false, null, null, null, null, null, null
        );

        List<PlanIntrospection.MemoryEvent> timeline = PlanIntrospection.getMemoryTimeline(plan);
        assertEquals(1, timeline.size());
        assertEquals(PlanIntrospection.MemoryEvent.EventType.ALLOCATE, timeline.get(0).getEventType());
    }

    // ─── getDevicePlacement ──────────────────────────────────────────

    @Test
    @DisplayName("getDevicePlacement: all on default device")
    void testGetDevicePlacement_Default() {
        DynamicShapeSlot s0 = slot("op0", 0, new int[]{-1}, new byte[]{0}, new String[]{"a"}, new int[]{0}, new String[]{"o0"});
        DynamicShapeSlot s1 = slot("op1", 1, new int[]{0}, new byte[]{3}, new String[]{""}, new int[]{1}, new String[]{"o1"});

        DynamicShapePlan plan = makePlan(new DynamicShapeSlot[]{s0, s1});
        Map<Integer, List<Integer>> placement = PlanIntrospection.getDevicePlacement(plan);
        assertEquals(1, placement.size());
        assertEquals(List.of(0, 1), placement.get(0));
    }

    @Test
    @DisplayName("getDevicePlacement: mixed devices")
    void testGetDevicePlacement_Mixed() {
        DynamicShapeSlot s0 = DynamicShapeSlot.builder()
                .opName("op0").stepIndex(0).targetDeviceId(0)
                .inputSourceIndices(new int[]{-1}).inputSourceTypes(new byte[]{0})
                .inputVarNames(new String[]{"a"})
                .outputSlotIndices(new int[]{0}).outputVarNames(new String[]{"o0"})
                .build();
        DynamicShapeSlot s1 = DynamicShapeSlot.builder()
                .opName("op1").stepIndex(1).targetDeviceId(1)
                .inputSourceIndices(new int[]{0}).inputSourceTypes(new byte[]{3})
                .inputVarNames(new String[]{""})
                .outputSlotIndices(new int[]{1}).outputVarNames(new String[]{"o1"})
                .build();

        DynamicShapePlan plan = makePlan(new DynamicShapeSlot[]{s0, s1});
        Map<Integer, List<Integer>> placement = PlanIntrospection.getDevicePlacement(plan);
        assertEquals(2, placement.size());
        assertEquals(List.of(0), placement.get(0));
        assertEquals(List.of(1), placement.get(1));
    }

    // ─── getSegments ─────────────────────────────────────────────────

    @Test
    @DisplayName("getSegments: single device, all capturable")
    void testGetSegments_SingleDevice() {
        DynamicShapeSlot s0 = DynamicShapeSlot.builder()
                .opName("mmul").stepIndex(0).targetDeviceId(0)
                .outputShapeDependsOnInputValues(false)
                .inputSourceIndices(new int[]{-1, -2}).inputSourceTypes(new byte[]{0, 0})
                .inputVarNames(new String[]{"a", "b"})
                .outputSlotIndices(new int[]{0}).outputVarNames(new String[]{"o0"})
                .build();
        DynamicShapeSlot s1 = DynamicShapeSlot.builder()
                .opName("add").stepIndex(1).targetDeviceId(0)
                .outputShapeDependsOnInputValues(false)
                .inputSourceIndices(new int[]{0, -3}).inputSourceTypes(new byte[]{3, 0})
                .inputVarNames(new String[]{"", "c"})
                .outputSlotIndices(new int[]{1}).outputVarNames(new String[]{"o1"})
                .build();

        DynamicShapePlan plan = makePlan(new DynamicShapeSlot[]{s0, s1});

        List<SegmentInfo> segments = PlanIntrospection.getSegments(plan);
        assertEquals(1, segments.size());
        SegmentInfo seg = segments.get(0);
        assertEquals(0, seg.getStartSlot());
        assertEquals(1, seg.getEndSlot());
        assertEquals(0, seg.getDeviceId());
        assertTrue(seg.isCapturable());
        assertEquals(2, seg.getSlotCount());
        assertEquals(List.of("mmul", "add"), seg.getOpNames());
    }

    @Test
    @DisplayName("getSegments: device boundary creates separate segments")
    void testGetSegments_DeviceBoundary() {
        DynamicShapeSlot s0 = DynamicShapeSlot.builder()
                .opName("mmul").stepIndex(0).targetDeviceId(0)
                .outputShapeDependsOnInputValues(false)
                .inputSourceIndices(new int[]{-1}).inputSourceTypes(new byte[]{0})
                .inputVarNames(new String[]{"a"})
                .outputSlotIndices(new int[]{0}).outputVarNames(new String[]{"o0"})
                .build();
        DynamicShapeSlot s1 = DynamicShapeSlot.builder()
                .opName("add").stepIndex(1).targetDeviceId(1)
                .outputShapeDependsOnInputValues(false)
                .inputSourceIndices(new int[]{0}).inputSourceTypes(new byte[]{3})
                .inputVarNames(new String[]{""})
                .outputSlotIndices(new int[]{1}).outputVarNames(new String[]{"o1"})
                .build();

        DynamicShapePlan plan = makePlan(new DynamicShapeSlot[]{s0, s1});

        List<SegmentInfo> segments = PlanIntrospection.getSegments(plan);
        assertEquals(2, segments.size());
        assertEquals(0, segments.get(0).getDeviceId());
        assertEquals(1, segments.get(1).getDeviceId());
    }

    @Test
    @DisplayName("getSegments: value-dependent shape boundary")
    void testGetSegments_ValueDependentBoundary() {
        DynamicShapeSlot s0 = DynamicShapeSlot.builder()
                .opName("mmul").stepIndex(0).targetDeviceId(0)
                .outputShapeDependsOnInputValues(false)
                .inputSourceIndices(new int[]{-1}).inputSourceTypes(new byte[]{0})
                .inputVarNames(new String[]{"a"})
                .outputSlotIndices(new int[]{0}).outputVarNames(new String[]{"o0"})
                .build();
        DynamicShapeSlot s1 = DynamicShapeSlot.builder()
                .opName("reshape").stepIndex(1).targetDeviceId(0)
                .outputShapeDependsOnInputValues(true)
                .inputSourceIndices(new int[]{0}).inputSourceTypes(new byte[]{3})
                .inputVarNames(new String[]{""})
                .outputSlotIndices(new int[]{1}).outputVarNames(new String[]{"o1"})
                .build();

        DynamicShapePlan plan = makePlan(new DynamicShapeSlot[]{s0, s1});
        List<SegmentInfo> segments = PlanIntrospection.getSegments(plan);
        assertEquals(2, segments.size());
        assertTrue(segments.get(0).isCapturable());
        assertFalse(segments.get(1).isCapturable());
    }

    @Test
    @DisplayName("getSegments: empty plan returns empty list")
    void testGetSegments_EmptyPlan() {
        DynamicShapePlan plan = makePlan(new DynamicShapeSlot[0]);
        List<SegmentInfo> segments = PlanIntrospection.getSegments(plan);
        assertTrue(segments.isEmpty());
    }

    @Test
    @DisplayName("getSegments: null slots returns empty list")
    void testGetSegments_NullSlots() {
        DynamicShapePlan plan = new DynamicShapePlan(
                null, 0, new int[0][],
                null, new String[0], new byte[0],
                new LinkedHashSet<>(), Collections.emptyMap(),
                false, null, null, null, null, null, null
        );
        List<SegmentInfo> segments = PlanIntrospection.getSegments(plan);
        assertTrue(segments.isEmpty());
    }

    // ─── getParallelGroups ───────────────────────────────────────────

    @Test
    @DisplayName("getParallelGroups: independent branches")
    void testParallelGroups_IndependentBranches() {
        DynamicShapeSlot s0 = DynamicShapeSlot.builder()
                .opName("mmul").stepIndex(0).targetDeviceId(0)
                .inputSourceIndices(new int[]{-1, -2}).inputSourceTypes(new byte[]{0, 0})
                .inputVarNames(new String[]{"a", "b"})
                .outputSlotIndices(new int[]{0}).outputVarNames(new String[]{"mmul0"})
                .build();
        DynamicShapeSlot s1 = DynamicShapeSlot.builder()
                .opName("mmul").stepIndex(1).targetDeviceId(0)
                .inputSourceIndices(new int[]{-1, -2}).inputSourceTypes(new byte[]{0, 0})
                .inputVarNames(new String[]{"a", "b"})
                .outputSlotIndices(new int[]{1}).outputVarNames(new String[]{"mmul1"})
                .build();

        DynamicShapePlan plan = makePlan(new DynamicShapeSlot[]{s0, s1});
        List<List<Integer>> groups = PlanIntrospection.getParallelGroups(plan);
        assertEquals(1, groups.size());
        assertTrue(groups.get(0).containsAll(List.of(0, 1)));
    }

    @Test
    @DisplayName("getParallelGroups: no parallelism in chain")
    void testParallelGroups_Chain() {
        DynamicShapeSlot s0 = slot("op0", 0, new int[]{-1}, new byte[]{0}, new String[]{"a"}, new int[]{0}, new String[]{"o0"});
        DynamicShapeSlot s1 = slot("op1", 1, new int[]{0}, new byte[]{3}, new String[]{""}, new int[]{1}, new String[]{"o1"});

        DynamicShapePlan plan = makePlan(new DynamicShapeSlot[]{s0, s1});
        List<List<Integer>> groups = PlanIntrospection.getParallelGroups(plan);
        assertTrue(groups.isEmpty());
    }

    // ─── formatPlan / formatSlot ──────────────────────────────────────

    @Test
    @DisplayName("formatPlan: non-empty output")
    void testFormatPlan_NonEmpty() {
        DynamicShapeSlot s0 = slot("mmul", 0, new int[]{-1, -2}, new byte[]{0, 0},
                new String[]{"a", "b"}, new int[]{0}, new String[]{"out0"});
        DynamicShapePlan plan = makePlan(new DynamicShapeSlot[]{s0});

        String formatted = PlanIntrospection.formatPlan(plan);
        assertNotNull(formatted);
        assertTrue(formatted.contains("DynamicShapePlan"));
        assertTrue(formatted.contains("mmul"));
        assertTrue(formatted.contains("Op Histogram"));
    }

    @Test
    @DisplayName("formatPlan: empty plan")
    void testFormatPlan_Empty() {
        DynamicShapePlan plan = makePlan(new DynamicShapeSlot[0]);
        assertEquals("DynamicShapePlan{empty}", PlanIntrospection.formatPlan(plan));
    }

    @Test
    @DisplayName("formatSlot: non-null output")
    void testFormatSlot() {
        DynamicShapeSlot s0 = DynamicShapeSlot.builder()
                .opName("mmul").stepIndex(0).targetDeviceId(0)
                .inputSourceIndices(new int[]{-1}).inputSourceTypes(new byte[]{0})
                .inputVarNames(new String[]{"a"})
                .outputSlotIndices(new int[]{0}).outputVarNames(new String[]{"out"})
                .outputShapeDependsOnInputValues(false)
                .build();

        DynamicShapePlan plan = makePlan(new DynamicShapeSlot[]{s0});
        String formatted = PlanIntrospection.formatSlot(plan, 0);
        assertNotNull(formatted);
        assertTrue(formatted.contains("Slot#0"));
        assertTrue(formatted.contains("mmul"));
        assertTrue(formatted.contains("Inputs:"));
    }

    // ─── toDot ────────────────────────────────────────────────────────

    @Test
    @DisplayName("toDot: non-empty DOT output")
    void testToDot_NonEmpty() {
        DynamicShapeSlot s0 = slot("mmul", 0, new int[]{-1, -2}, new byte[]{0, 0},
                new String[]{"a", "b"}, new int[]{0}, new String[]{"out0"});
        DynamicShapePlan plan = makePlan(new DynamicShapeSlot[]{s0});

        String dot = PlanIntrospection.toDot(plan);
        assertNotNull(dot);
        assertTrue(dot.contains("digraph DSP"));
        assertTrue(dot.contains("ext_0"));
        assertTrue(dot.contains("slot_0"));
    }

    @Test
    @DisplayName("toDot: empty plan")
    void testToDot_Empty() {
        DynamicShapePlan plan = makePlan(new DynamicShapeSlot[0]);
        assertEquals("digraph DSP {\n}\n", PlanIntrospection.toDot(plan));
    }

    // ─── DecodedInput.toString ────────────────────────────────────────

    @Test
    @DisplayName("DecodedInput toString: OP_OUTPUT type")
    void testDecodedInputToString_OpOutput() {
        DecodedInput di = new DecodedInput(5, DynamicShapeSlot.SOURCE_OP_OUTPUT, "", 5, null);
        assertEquals("slot#5(OP)", di.toString());
    }

    @Test
    @DisplayName("DecodedInput toString: external input")
    void testDecodedInputToString_External() {
        DecodedInput di = new DecodedInput(0, DynamicShapeSlot.SOURCE_VARIABLE, "weights", -1, "weights");
        assertEquals("ext#0:\"weights\"(VARIABLE)", di.toString());
    }

    @Test
    @DisplayName("DecodedInput getSourceTypeName: unknown type")
    void testDecodedInputUnknownType() {
        DecodedInput di = new DecodedInput(0, (byte) 99, "unknown", -1, "x");
        assertEquals("UNKNOWN", di.getSourceTypeName());
    }

    // ─── getPlanPhase (via SegmentInfo) ───────────────────────────────

    @Test
    @DisplayName("getPlanPhase: minimum phase across segments")
    void testGetPlanPhase_Minimum() {
        List<SegmentInfo> segments = new ArrayList<>();
        SegmentInfo s0 = new SegmentInfo(0, 2, true, 0, 3, List.of("a", "b", "c"));
        s0.setExecutionPhaseCode(3); // REPLAYING
        SegmentInfo s1 = new SegmentInfo(3, 5, false, 0, 3, List.of("d", "e", "f"));
        s1.setExecutionPhaseCode(0); // WARMUP
        segments.add(s0);
        segments.add(s1);

        // Min is WARMUP (0)
        assertEquals(0, PlanIntrospection.getPlanPhase(segments).getNativeCode());
    }

    @Test
    @DisplayName("getPlanPhase: null for empty segments")
    void testGetPlanPhase_Empty() {
        assertNull(PlanIntrospection.getPlanPhase(Collections.emptyList()));
    }
}