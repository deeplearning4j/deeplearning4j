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

package org.nd4j.autodiff.samediff.execution;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Stateless utility class providing introspection, query, and visualization
 * capabilities for {@link DynamicShapePlan}.
 *
 * <p>All methods are static and take a {@code DynamicShapePlan} as the first argument,
 * keeping the plan class itself lean.</p>
 */
public final class PlanIntrospection {

    private PlanIntrospection() {}

    // ─── Inner data classes ──────────────────────────────────────────────────

    /** Decoded representation of a single input to a slot. */
    public static class DecodedInput {
        private final int index;
        private final byte sourceType;
        private final String sourceName;
        private final int fromSlot;
        private final String externalKey;

        public DecodedInput(int index, byte sourceType, String sourceName, int fromSlot, String externalKey) {
            this.index = index;
            this.sourceType = sourceType;
            this.sourceName = sourceName;
            this.fromSlot = fromSlot;
            this.externalKey = externalKey;
        }

        public int getIndex() { return index; }
        public byte getSourceType() { return sourceType; }
        public String getSourceName() { return sourceName; }
        public int getFromSlot() { return fromSlot; }
        public String getExternalKey() { return externalKey; }

        public String getSourceTypeName() {
            switch (sourceType) {
                case DynamicShapeSlot.SOURCE_CONSTANT: return "CONSTANT";
                case DynamicShapeSlot.SOURCE_VARIABLE: return "VARIABLE";
                case DynamicShapeSlot.SOURCE_PLACEHOLDER: return "PLACEHOLDER";
                case DynamicShapeSlot.SOURCE_OP_OUTPUT: return "OP_OUTPUT";
                default: return "UNKNOWN";
            }
        }

        @Override
        public String toString() {
            if (sourceType == DynamicShapeSlot.SOURCE_OP_OUTPUT) {
                return "slot#" + fromSlot + "(OP)";
            } else {
                int extIdx = -(index + 1);
                return "ext#" + extIdx + ":\"" + externalKey + "\"(" + getSourceTypeName() + ")";
            }
        }
    }

    /** Describes a contiguous segment of slots sharing properties. */
    public static class SegmentInfo {
        private final int startSlot;
        private final int endSlot;
        private final boolean capturable;
        private final int deviceId;
        private final int slotCount;
        private final List<String> opNames;

        public SegmentInfo(int startSlot, int endSlot, boolean capturable, int deviceId,
                           int slotCount, List<String> opNames) {
            this.startSlot = startSlot;
            this.endSlot = endSlot;
            this.capturable = capturable;
            this.deviceId = deviceId;
            this.slotCount = slotCount;
            this.opNames = opNames;
        }

        // Replay state
        private String replayBackendName = "";
        private int replayState = -1;     // 0=EMPTY, 1=CAPTURING, 2=CAPTURED, 3=READY, 4=ERROR, -1=no handle
        private int replayCount = 0;
        private int executionCount = 0;
        private boolean captureFailed = false;
        private String statisticsJson = "";

        // Pointer tracking
        private String trackedPointersJson = "[]";
        private int numCaptureBuffers = 0;
        private String captureBuffersJson = "[]";
        private int numHostPointers = 0;

        // Backend tracking
        private String compiledByBackend = "";

        public int getStartSlot() { return startSlot; }
        public int getEndSlot() { return endSlot; }
        public boolean isCapturable() { return capturable; }
        public int getDeviceId() { return deviceId; }
        public int getSlotCount() { return slotCount; }
        public List<String> getOpNames() { return opNames; }

        public String getReplayBackendName() { return replayBackendName; }
        public int getReplayState() { return replayState; }
        public int getReplayCount() { return replayCount; }
        public int getExecutionCount() { return executionCount; }
        public boolean isCaptureFailed() { return captureFailed; }
        public String getStatisticsJson() { return statisticsJson; }
        public String getTrackedPointersJson() { return trackedPointersJson; }
        public int getNumCaptureBuffers() { return numCaptureBuffers; }
        public String getCaptureBuffersJson() { return captureBuffersJson; }
        public int getNumHostPointers() { return numHostPointers; }
        public String getCompiledByBackend() { return compiledByBackend; }

        public String getReplayStateName() {
            switch (replayState) {
                case 0: return "EMPTY";
                case 1: return "CAPTURING";
                case 2: return "CAPTURED";
                case 3: return "READY";
                case 4: return "ERROR";
                default: return "NO_HANDLE";
            }
        }

        public boolean isReplaying() { return replayState == 3; }
        public boolean hasCaptureHandle() { return replayState >= 0; }
    }

    /** Describes a memory allocate or release event. */
    public static class MemoryEvent {
        public enum EventType { ALLOCATE, RELEASE }

        private final int step;
        private final int slotIndex;
        private final EventType eventType;

        public MemoryEvent(int step, int slotIndex, EventType eventType) {
            this.step = step;
            this.slotIndex = slotIndex;
            this.eventType = eventType;
        }

        public int getStep() { return step; }
        public int getSlotIndex() { return slotIndex; }
        public EventType getEventType() { return eventType; }

        @Override
        public String toString() {
            return "step " + step + ": " + eventType + " slot " + slotIndex;
        }
    }

    // ─── Structured query methods ────────────────────────────────────────────

    /**
     * Decode all inputs for a given slot, resolving negative indices to external
     * input names with source types.
     */
    public static List<DecodedInput> getDecodedInputs(DynamicShapePlan plan, int slotIndex) {
        DynamicShapeSlot slot = plan.getSlots()[slotIndex];
        int[] srcIndices = slot.getInputSourceIndices();
        byte[] srcTypes = slot.getInputSourceTypes();
        String[] varNames = slot.getInputVarNames();
        String[] extKeys = plan.getExternalInputKeys();

        List<DecodedInput> result = new ArrayList<>(srcIndices.length);
        for (int i = 0; i < srcIndices.length; i++) {
            int idx = srcIndices[i];
            byte type = srcTypes[i];
            String name = varNames != null && i < varNames.length ? varNames[i] : "";
            if (idx >= 0) {
                // OP_OUTPUT — fromSlot is the output slot index
                result.add(new DecodedInput(idx, type, name, idx, null));
            } else {
                int extIdx = -(idx + 1);
                String extKey = extIdx < extKeys.length ? extKeys[extIdx] : name;
                result.add(new DecodedInput(idx, type, name, -1, extKey));
            }
        }
        return result;
    }

    /**
     * Get downstream slot step indices that consume this slot's outputs.
     */
    public static List<Integer> getDependentsOf(DynamicShapePlan plan, int slotIndex) {
        int[][] successors = plan.getSuccessors();
        if (successors != null && slotIndex < successors.length) {
            List<Integer> result = new ArrayList<>();
            for (int s : successors[slotIndex]) {
                result.add(s);
            }
            return result;
        }
        // Fallback: scan all slots
        DynamicShapeSlot[] slots = plan.getSlots();
        DynamicShapeSlot target = slots[slotIndex];
        int[] targetOutputs = target.getOutputSlotIndices();
        Set<Integer> outputSet = new HashSet<>();
        for (int o : targetOutputs) outputSet.add(o);

        List<Integer> dependents = new ArrayList<>();
        for (int i = 0; i < slots.length; i++) {
            if (i == slotIndex) continue;
            for (int srcIdx : slots[i].getInputSourceIndices()) {
                if (srcIdx >= 0 && outputSet.contains(srcIdx)) {
                    dependents.add(i);
                    break;
                }
            }
        }
        return dependents;
    }

    /**
     * Get upstream slot step indices that feed into this slot (from inputSourceIndices >= 0).
     */
    public static List<Integer> getProducersOf(DynamicShapePlan plan, int slotIndex) {
        int[][] predecessors = plan.getPredecessors();
        if (predecessors != null && slotIndex < predecessors.length) {
            List<Integer> result = new ArrayList<>();
            for (int p : predecessors[slotIndex]) {
                result.add(p);
            }
            return result;
        }
        // Fallback: scan input source indices
        DynamicShapeSlot slot = plan.getSlots()[slotIndex];
        DynamicShapeSlot[] allSlots = plan.getSlots();
        Set<Integer> producers = new LinkedHashSet<>();
        for (int srcIdx : slot.getInputSourceIndices()) {
            if (srcIdx >= 0) {
                // Find which step produces this output slot
                for (int i = 0; i < allSlots.length; i++) {
                    for (int outIdx : allSlots[i].getOutputSlotIndices()) {
                        if (outIdx == srcIdx) {
                            producers.add(i);
                        }
                    }
                }
            }
        }
        return new ArrayList<>(producers);
    }

    /**
     * Compute an ordered timeline of memory allocate/release events.
     */
    public static List<MemoryEvent> getMemoryTimeline(DynamicShapePlan plan) {
        DynamicShapeSlot[] slots = plan.getSlots();
        int[][] releaseAtStep = plan.getReleaseAtStep();
        List<MemoryEvent> events = new ArrayList<>();

        for (int step = 0; step < slots.length; step++) {
            // Allocate events for each output of this step
            for (int outSlot : slots[step].getOutputSlotIndices()) {
                if (outSlot >= 0) {
                    events.add(new MemoryEvent(step, outSlot, MemoryEvent.EventType.ALLOCATE));
                }
            }
            // Release events
            if (releaseAtStep != null && step < releaseAtStep.length) {
                for (int relSlot : releaseAtStep[step]) {
                    events.add(new MemoryEvent(step, relSlot, MemoryEvent.EventType.RELEASE));
                }
            }
        }
        return events;
    }

    /**
     * Get device placement: device ID → list of step indices assigned to that device.
     */
    public static Map<Integer, List<Integer>> getDevicePlacement(DynamicShapePlan plan) {
        DynamicShapeSlot[] slots = plan.getSlots();
        Map<Integer, List<Integer>> result = new LinkedHashMap<>();
        for (int i = 0; i < slots.length; i++) {
            int dev = slots[i].getTargetDeviceId();
            if (dev < 0) dev = 0; // default device
            result.computeIfAbsent(dev, k -> new ArrayList<>()).add(i);
        }
        return result;
    }

    /**
     * Compute segment boundaries from slot data. Segments are contiguous runs of slots
     * sharing the same device and capturability (no data-dependent ops).
     */
    public static List<SegmentInfo> getSegments(DynamicShapePlan plan) {
        DynamicShapeSlot[] slots = plan.getSlots();
        if (slots == null || slots.length == 0) return Collections.emptyList();

        List<SegmentInfo> segments = new ArrayList<>();
        int segStart = 0;

        for (int i = 1; i <= slots.length; i++) {
            boolean boundary = (i == slots.length);
            if (!boundary) {
                int prevDev = slots[i - 1].getTargetDeviceId();
                int curDev = slots[i].getTargetDeviceId();
                boolean prevCapturable = !slots[i - 1].isDataDependent() && !slots[i - 1].isOutputShapeDependsOnInputValues();
                boolean curCapturable = !slots[i].isDataDependent() && !slots[i].isOutputShapeDependsOnInputValues();
                boundary = (prevDev != curDev) || (prevCapturable != curCapturable);
            }
            if (boundary) {
                boolean capturable = !slots[segStart].isDataDependent() && !slots[segStart].isOutputShapeDependsOnInputValues();
                int deviceId = slots[segStart].getTargetDeviceId();
                if (deviceId < 0) deviceId = 0;
                int count = i - segStart;
                List<String> opNames = new ArrayList<>(count);
                for (int j = segStart; j < i; j++) {
                    opNames.add(slots[j].getOpName());
                }
                segments.add(new SegmentInfo(segStart, i - 1, capturable, deviceId, count, opNames));
                segStart = i;
            }
        }
        return segments;
    }

    /**
     * Get segments with full replay state populated from native plan handle.
     */
    public static List<SegmentInfo> getSegmentsWithReplayState(DynamicShapePlan plan, org.bytedeco.javacpp.Pointer nativePlanHandle) {
        List<SegmentInfo> segments = getSegments(plan);
        if (nativePlanHandle == null) return segments;

        org.nd4j.nativeblas.NativeOps ops = org.nd4j.linalg.factory.Nd4j.getNativeOps();
        for (int i = 0; i < segments.size(); i++) {
            SegmentInfo seg = segments.get(i);
            seg.replayBackendName = ops.getPlanSegmentBackendName(nativePlanHandle, i);
            seg.replayState = ops.getPlanSegmentReplayState(nativePlanHandle, i);
            seg.replayCount = ops.getPlanSegmentReplayCount(nativePlanHandle, i);
            seg.executionCount = ops.getPlanSegmentExecutionCount(nativePlanHandle, i);
            seg.captureFailed = ops.isPlanSegmentCaptureFailed(nativePlanHandle, i);
            seg.statisticsJson = ops.getPlanSegmentStatisticsJson(nativePlanHandle, i);
            seg.trackedPointersJson = ops.getPlanSegmentTrackedPointers(nativePlanHandle, i);
            seg.numCaptureBuffers = ops.getPlanSegmentNumCaptureBuffers(nativePlanHandle, i);
            seg.captureBuffersJson = ops.getPlanSegmentCaptureBuffersJson(nativePlanHandle, i);
            seg.numHostPointers = ops.getPlanSegmentNumHostPointers(nativePlanHandle, i);
            seg.compiledByBackend = ops.getPlanSegmentCompiledBackend(nativePlanHandle, i);
        }
        return segments;
    }

    /**
     * Find groups of slots that share the same predecessor set and could execute concurrently.
     */
    public static List<List<Integer>> getParallelGroups(DynamicShapePlan plan) {
        int[][] predecessors = plan.getPredecessors();
        if (predecessors == null) return Collections.emptyList();

        Map<List<Integer>, List<Integer>> groups = new LinkedHashMap<>();
        for (int i = 0; i < predecessors.length; i++) {
            int[] preds = predecessors[i];
            List<Integer> key = Arrays.stream(preds).sorted().boxed().collect(Collectors.toList());
            groups.computeIfAbsent(key, k -> new ArrayList<>()).add(i);
        }

        // Only return groups with more than one member
        List<List<Integer>> result = new ArrayList<>();
        for (List<Integer> group : groups.values()) {
            if (group.size() > 1) {
                result.add(group);
            }
        }
        return result;
    }

    // ─── Human-readable formatting ──────────────────────────────────────────

    /**
     * Format a single slot as a multi-line string with decoded inputs, outputs, and flags.
     */
    public static String formatSlot(DynamicShapePlan plan, int slotIndex) {
        DynamicShapeSlot slot = plan.getSlots()[slotIndex];
        List<DecodedInput> inputs = getDecodedInputs(plan, slotIndex);
        List<Integer> dependents = getDependentsOf(plan, slotIndex);

        StringBuilder sb = new StringBuilder();
        sb.append("Slot#").append(slotIndex).append(" [").append(slot.getOpName()).append("]\n");

        // Inputs
        sb.append("  Inputs: ");
        if (inputs.isEmpty()) {
            sb.append("(none)");
        } else {
            sb.append(inputs.stream().map(DecodedInput::toString).collect(Collectors.joining(", ")));
        }
        sb.append("\n");

        // Outputs
        sb.append("  Outputs: slots ");
        sb.append(Arrays.stream(slot.getOutputSlotIndices())
                .mapToObj(Integer::toString).collect(Collectors.joining(", ")));
        if (slot.getOutputVarNames() != null) {
            sb.append(" (").append(String.join(", ", slot.getOutputVarNames())).append(")");
        }
        sb.append("\n");

        // Dependents
        if (!dependents.isEmpty()) {
            sb.append("  Dependents: steps ").append(dependents).append("\n");
        }

        // Flags
        List<String> flags = new ArrayList<>();
        if (slot.isDataDependent()) flags.add("data-dependent");
        if (slot.isOutputShapeDependsOnInputValues()) flags.add("value-dependent-shape");
        if (slot.isNeedsZeroedOutput()) flags.add("needs-zeroed");
        if (slot.isNeedsIntLongSync()) flags.add("int-long-sync");
        if (slot.getTargetDeviceId() >= 0) flags.add("device:" + slot.getTargetDeviceId());
        if (!flags.isEmpty()) {
            sb.append("  Flags: ").append(String.join(", ", flags)).append("\n");
        }

        return sb.toString();
    }

    /**
     * Format the entire plan as a human-readable multi-line summary.
     */
    public static String formatPlan(DynamicShapePlan plan) {
        DynamicShapeSlot[] slots = plan.getSlots();
        if (slots == null || slots.length == 0) return "DynamicShapePlan{empty}";

        StringBuilder sb = new StringBuilder();
        sb.append("=== DynamicShapePlan ===\n");
        sb.append("Slots: ").append(slots.length)
          .append(", Output slots: ").append(plan.getTotalOutputSlots())
          .append(", External inputs: ").append(plan.getExternalInputKeys().length)
          .append(", Requested outputs: ").append(plan.getRequestedOutputs().size())
          .append("\n\n");

        // Per-slot one-liners
        sb.append("--- Slots ---\n");
        for (int i = 0; i < slots.length; i++) {
            DynamicShapeSlot slot = slots[i];
            List<DecodedInput> inputs = getDecodedInputs(plan, i);
            sb.append(String.format("  %4d: %-25s inputs:[%s] → outputs:[%s]",
                    i, slot.getOpName(),
                    inputs.stream().map(DecodedInput::toString).collect(Collectors.joining(", ")),
                    Arrays.stream(slot.getOutputSlotIndices()).mapToObj(Integer::toString)
                            .collect(Collectors.joining(", "))));
            if (slot.getTargetDeviceId() >= 0) {
                sb.append(" device:").append(slot.getTargetDeviceId());
            }
            if (slot.isDataDependent()) sb.append(" [data-dep]");
            if (slot.isOutputShapeDependsOnInputValues()) sb.append(" [val-dep]");
            sb.append("\n");
        }

        // Segments
        List<SegmentInfo> segments = getSegments(plan);
        sb.append("\n--- Segments (").append(segments.size()).append(") ---\n");
        for (int i = 0; i < segments.size(); i++) {
            SegmentInfo seg = segments.get(i);
            sb.append(String.format("  Segment %d: slots [%d..%d] (%d ops) device:%d %s\n",
                    i, seg.startSlot, seg.endSlot, seg.slotCount, seg.deviceId,
                    seg.capturable ? "capturable" : "non-capturable"));
            if (seg.replayState >= 0) {
                sb.append("    Replay: ").append(seg.getReplayStateName());
                if (!seg.replayBackendName.isEmpty()) sb.append(" (").append(seg.replayBackendName).append(")");
                sb.append(" | ").append(seg.replayCount).append(" replays");
                sb.append("\n");
            }
            if (!seg.compiledByBackend.isEmpty()) {
                sb.append("    Backend: ").append(seg.compiledByBackend).append("\n");
            }
        }

        // Op histogram
        Map<String, Integer> opHist = new TreeMap<>();
        for (DynamicShapeSlot slot : slots) {
            opHist.merge(slot.getOpName(), 1, Integer::sum);
        }
        sb.append("\n--- Op Histogram ---\n");
        opHist.entrySet().stream()
                .sorted((a, b) -> Integer.compare(b.getValue(), a.getValue()))
                .forEach(e -> sb.append(String.format("  %-25s %d\n", e.getKey(), e.getValue())));

        // Device placement
        Map<Integer, List<Integer>> devicePlacement = getDevicePlacement(plan);
        sb.append("\n--- Device Placement ---\n");
        for (Map.Entry<Integer, List<Integer>> entry : devicePlacement.entrySet()) {
            sb.append("  Device ").append(entry.getKey()).append(": ")
              .append(entry.getValue().size()).append(" ops\n");
        }

        // Memory timeline summary
        List<MemoryEvent> timeline = getMemoryTimeline(plan);
        long allocCount = timeline.stream().filter(e -> e.eventType == MemoryEvent.EventType.ALLOCATE).count();
        long releaseCount = timeline.stream().filter(e -> e.eventType == MemoryEvent.EventType.RELEASE).count();
        sb.append("\n--- Memory Timeline ---\n");
        sb.append("  Allocations: ").append(allocCount)
          .append(", Releases: ").append(releaseCount)
          .append(", Peak live: ").append(computePeakLive(timeline))
          .append(" slots\n");

        // Parallel groups
        List<List<Integer>> parallelGroups = getParallelGroups(plan);
        if (!parallelGroups.isEmpty()) {
            sb.append("\n--- Parallel Groups (").append(parallelGroups.size()).append(") ---\n");
            for (int i = 0; i < parallelGroups.size(); i++) {
                List<Integer> group = parallelGroups.get(i);
                sb.append("  Group ").append(i).append(": steps ")
                  .append(group).append(" [");
                sb.append(group.stream()
                        .map(idx -> slots[idx].getOpName())
                        .collect(Collectors.joining(", ")));
                sb.append("]\n");
            }
        }

        return sb.toString();
    }

    /**
     * Format a segment as a human-readable string.
     */
    public static String formatSegment(SegmentInfo seg) {
        return String.format("Segment[%d..%d] %d ops, device:%d, %s, ops: %s",
                seg.startSlot, seg.endSlot, seg.slotCount, seg.deviceId,
                seg.capturable ? "capturable" : "non-capturable",
                seg.opNames);
    }

    // ─── DOT/Graphviz export ─────────────────────────────────────────────────

    /**
     * Export the plan as a DOT format string for Graphviz visualization.
     *
     * <p>Nodes = slots (box shaped, labeled with slot# and opName).
     * Edges = data dependencies from inputSourceIndices >= 0.
     * External inputs = diamond-shaped nodes.
     * Segments = cluster subgraphs.
     * Color coding: data-dependent (red border), value-dependent-shape (orange),
     * needs-zeroed (dashed), device coloring (different fills).</p>
     */
    public static String toDot(DynamicShapePlan plan) {
        DynamicShapeSlot[] slots = plan.getSlots();
        if (slots == null || slots.length == 0) return "digraph DSP {\n}\n";

        String[] extKeys = plan.getExternalInputKeys();
        List<SegmentInfo> segments = getSegments(plan);

        // Device colors
        String[] deviceColors = {"#FFFFFF", "#E8F5E9", "#E3F2FD", "#FFF3E0", "#F3E5F5",
                "#FFEBEE", "#E0F7FA", "#FFF9C4"};

        StringBuilder sb = new StringBuilder();
        sb.append("digraph DSP {\n");
        sb.append("  rankdir=TB;\n");
        sb.append("  node [fontname=\"Helvetica\", fontsize=10];\n");
        sb.append("  edge [fontname=\"Helvetica\", fontsize=8];\n\n");

        // External input nodes
        Set<Integer> externalIndicesUsed = new LinkedHashSet<>();
        for (DynamicShapeSlot slot : slots) {
            for (int srcIdx : slot.getInputSourceIndices()) {
                if (srcIdx < 0) {
                    externalIndicesUsed.add(-(srcIdx + 1));
                }
            }
        }
        if (!externalIndicesUsed.isEmpty()) {
            sb.append("  // External inputs\n");
            for (int extIdx : externalIndicesUsed) {
                String label = extIdx < extKeys.length ? extKeys[extIdx] : "ext#" + extIdx;
                // Truncate long names
                if (label.length() > 30) {
                    label = label.substring(0, 27) + "...";
                }
                // Determine fill color by source type (check first slot that uses it)
                String fillColor = "#D3D3D3"; // gray default (constant)
                for (DynamicShapeSlot slot : slots) {
                    for (int i = 0; i < slot.getInputSourceIndices().length; i++) {
                        if (slot.getInputSourceIndices()[i] == -(extIdx + 1)) {
                            switch (slot.getInputSourceTypes()[i]) {
                                case DynamicShapeSlot.SOURCE_VARIABLE: fillColor = "#ADD8E6"; break;   // lightblue
                                case DynamicShapeSlot.SOURCE_PLACEHOLDER: fillColor = "#FFFFE0"; break; // lightyellow
                                default: fillColor = "#D3D3D3"; break;  // gray for constants
                            }
                            break;
                        }
                    }
                    if (!fillColor.equals("#D3D3D3")) break;
                }
                sb.append(String.format("  ext_%d [label=\"%s\", shape=diamond, style=filled, fillcolor=\"%s\"];\n",
                        extIdx, escDot(label), fillColor));
            }
            sb.append("\n");
        }

        // Slot nodes within segment clusters
        for (int segIdx = 0; segIdx < segments.size(); segIdx++) {
            SegmentInfo seg = segments.get(segIdx);
            sb.append(String.format("  subgraph cluster_%d {\n", segIdx));
            String segLabel = String.format("Segment %d (device %d%s)", segIdx, seg.deviceId,
                    seg.capturable ? ", capturable" : "");
            if (seg.replayState >= 0) {
                segLabel += "\\n" + seg.getReplayStateName();
                if (!seg.compiledByBackend.isEmpty()) segLabel += " [" + seg.compiledByBackend + "]";
            }
            sb.append(String.format("    label=\"%s\";\n", segLabel));

            String bgColor = "#F5F5F5";  // default gray
            if (seg.replayState == 3) bgColor = "#E8F5E9";      // READY -> green
            else if (seg.replayState == 2) bgColor = "#FFF9C4";  // CAPTURED -> yellow
            else if (seg.replayState == 4 || seg.captureFailed) bgColor = "#FFEBEE"; // ERROR -> red
            else if (seg.replayState == 1) bgColor = "#E3F2FD";  // CAPTURING -> blue

            sb.append("    style=\"dashed,filled\";\n");
            sb.append("    fillcolor=\"").append(bgColor).append("\";\n");
            sb.append("    color=gray;\n");

            for (int si = seg.startSlot; si <= seg.endSlot; si++) {
                DynamicShapeSlot slot = slots[si];
                String label = "slot#" + si + "\\n" + slot.getOpName();

                // Border color
                String borderColor = "black";
                if (slot.isDataDependent()) borderColor = "red";
                else if (slot.isOutputShapeDependsOnInputValues()) borderColor = "orange";

                // Fill color by device
                int dev = slot.getTargetDeviceId();
                if (dev < 0) dev = 0;
                String fill = deviceColors[dev % deviceColors.length];

                // Style
                String style = "filled";
                if (slot.isNeedsZeroedOutput()) style = "filled,dashed";

                sb.append(String.format("    slot_%d [label=\"%s\", shape=box, style=\"%s\", fillcolor=\"%s\", color=\"%s\"];\n",
                        si, label, style, fill, borderColor));
            }
            sb.append("  }\n\n");
        }

        // Edges
        sb.append("  // Data dependency edges\n");
        for (int stepIdx = 0; stepIdx < slots.length; stepIdx++) {
            DynamicShapeSlot slot = slots[stepIdx];
            int[] srcIndices = slot.getInputSourceIndices();
            for (int i = 0; i < srcIndices.length; i++) {
                int srcIdx = srcIndices[i];
                if (srcIdx >= 0) {
                    // Find producing step
                    int producerStep = findProducerStep(slots, srcIdx);
                    if (producerStep >= 0) {
                        // Add edge label for multi-output producers
                        String edgeLabel = "";
                        if (slots[producerStep].getOutputSlotIndices().length > 1) {
                            edgeLabel = String.format(" [label=\"out:%d\"]", srcIdx);
                        }
                        sb.append(String.format("  slot_%d -> slot_%d%s;\n",
                                producerStep, stepIdx, edgeLabel));
                    }
                } else {
                    int extIdx = -(srcIdx + 1);
                    sb.append(String.format("  ext_%d -> slot_%d;\n", extIdx, stepIdx));
                }
            }
        }

        sb.append("}\n");
        return sb.toString();
    }

    // ─── Internal helpers ────────────────────────────────────────────────────

    private static int computePeakLive(List<MemoryEvent> timeline) {
        int live = 0;
        int peak = 0;
        for (MemoryEvent event : timeline) {
            if (event.eventType == MemoryEvent.EventType.ALLOCATE) {
                live++;
            } else {
                live--;
            }
            if (live > peak) peak = live;
        }
        return peak;
    }

    private static int findProducerStep(DynamicShapeSlot[] slots, int outputSlotIdx) {
        for (int i = 0; i < slots.length; i++) {
            for (int outIdx : slots[i].getOutputSlotIndices()) {
                if (outIdx == outputSlotIdx) return i;
            }
        }
        return -1;
    }

    private static String escDot(String s) {
        return s.replace("\"", "\\\"").replace("\n", "\\n");
    }

    // ─── Replay observability ───────────────────────────────────────────────

    /**
     * Get a compact replay summary for logging.
     */
    public static String formatReplaySummary(org.bytedeco.javacpp.Pointer nativePlanHandle) {
        if (nativePlanHandle == null) return "No native plan handle";

        org.nd4j.nativeblas.NativeOps ops = org.nd4j.linalg.factory.Nd4j.getNativeOps();
        int numSegs = ops.getPlanNumSegments(nativePlanHandle);
        int captured = ops.getPlanNumCapturedGraphSegments(nativePlanHandle);
        int totalReplays = ops.getPlanTotalGraphReplays(nativePlanHandle);

        StringBuilder sb = new StringBuilder();
        sb.append("Replay Summary: ").append(numSegs).append(" segments | ")
          .append(captured).append(" captured | ")
          .append(totalReplays).append(" total replays\n");

        for (int i = 0; i < numSegs; i++) {
            int state = ops.getPlanSegmentReplayState(nativePlanHandle, i);
            String backend = ops.getPlanSegmentBackendName(nativePlanHandle, i);
            int replays = ops.getPlanSegmentReplayCount(nativePlanHandle, i);
            boolean capturable = ops.isPlanSegmentCapturable(nativePlanHandle, i);
            String compiled = ops.getPlanSegmentCompiledBackend(nativePlanHandle, i);
            int numBufs = ops.getPlanSegmentNumCaptureBuffers(nativePlanHandle, i);
            int numHost = ops.getPlanSegmentNumHostPointers(nativePlanHandle, i);

            String stateName;
            switch (state) {
                case 0: stateName = "EMPTY"; break;
                case 1: stateName = "CAPTURING"; break;
                case 2: stateName = "CAPTURED"; break;
                case 3: stateName = "READY"; break;
                case 4: stateName = "ERROR"; break;
                default: stateName = "-"; break;
            }

            sb.append(String.format("  seg[%d]: %-8s %-8s %d replays  %d bufs  %d host",
                i, compiled != null && !compiled.isEmpty() ? compiled : "-",
                backend != null && !backend.isEmpty() ? backend + "/" + stateName : stateName,
                replays, numBufs, numHost));
            if (!capturable) sb.append("  (non-capturable)");
            sb.append("\n");
        }

        // Cache stats
        if (ops.isReplayCacheEnabled()) {
            sb.append("Cache: ").append(ops.getReplayCacheHits()).append(" hits / ")
              .append(ops.getReplayCacheMisses()).append(" misses | dir: ")
              .append(ops.getReplayCacheDir()).append("\n");
        }

        return sb.toString();
    }
}
