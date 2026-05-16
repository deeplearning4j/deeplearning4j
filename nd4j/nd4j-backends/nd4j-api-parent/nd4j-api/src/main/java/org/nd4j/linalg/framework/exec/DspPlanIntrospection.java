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

package org.nd4j.linalg.framework.exec;

import lombok.extern.slf4j.Slf4j;
import org.bytedeco.javacpp.Pointer;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlan;
import org.nd4j.autodiff.samediff.execution.DynamicShapeSlot;
import org.nd4j.autodiff.samediff.execution.PlanIntrospection;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * DSP plan introspection - provides detailed inspection of DynamicShapePlan models.
 * 
 * Usage:
 * <pre>
 *   DspPlanIntrospection introspection = Nd4j.framework.execution().dsp().introspection();
 *   
 *   // Get plan from current session/executor
 *   DynamicShapePlan plan = getCurrentPlan();
 *   
 *   // Get plan summary
 *   String summary = introspection.getSummary(plan);
 *   
 *   // Get segment information
 *   List<SegmentInfo> segments = introspection.getSegments(plan);
 *   
 *   // Get memory timeline
 *   List<MemoryEvent> timeline = introspection.getMemoryTimeline(plan);
 *   
 *   // Get device placement
 *   Map<Integer, List<Integer>> devicePlacement = introspection.getDevicePlacement(plan);
 * </pre>
 * 
 * @author ND4J Team
 */
@Slf4j
public final class DspPlanIntrospection {
    
    private final NativeOps nativeOps;
    
    public DspPlanIntrospection() {
        this.nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
    }
    
    // =========================================================================
    // Plan Summary and Overview
    // =========================================================================
    
    /**
     * Get human-readable summary of a DSP plan
     */
    public String getSummary(DynamicShapePlan plan) {
        if (plan == null) {
            return "No plan available";
        }
        return plan.getSummary();
    }
    
    /**
     * Get detailed plan information as a map
     */
    public Map<String, Object> getPlanInfo(DynamicShapePlan plan) {
        Map<String, Object> info = new HashMap<>();
        
        if (plan == null) {
            info.put("error", "No plan available");
            return info;
        }
        
        info.put("numSlots", plan.getSlots().length);
        info.put("totalOutputSlots", plan.getTotalOutputSlots());
        info.put("numExternalInputs", plan.getExternalInputKeys().length);
        info.put("requestedOutputs", new ArrayList<>(plan.getRequestedOutputs()));
        info.put("hasControlFlow", plan.isHasControlFlowOps());
        info.put("numLoopRegions", plan.getLoopRegions() != null ? plan.getLoopRegions().length : 0);
        info.put("numRootSlots", plan.getRootSlots() != null ? plan.getRootSlots().length : 0);
        
        return info;
    }
    
    // =========================================================================
    // Slot Introspection
    // =========================================================================
    
    /**
     * Get information about a specific slot
     */
    public SlotInfo getSlotInfo(DynamicShapePlan plan, int slotIndex) {
        if (plan == null || slotIndex < 0 || slotIndex >= plan.getSlots().length) {
            return null;
        }
        
        DynamicShapeSlot slot = plan.getSlots()[slotIndex];

        return SlotInfo.builder()
            .slotIndex(slotIndex)
            .opName(slot.getOpName())
            .opType(slot.getOpType())
            .numInputs(slot.getInputSourceIndices() != null ? slot.getInputSourceIndices().length : 0)
            .numOutputs(slot.getOutputSlotIndices() != null ? slot.getOutputSlotIndices().length : 0)
            .deviceId(slot.getDeviceId())
            .capturable(slot.isCapturable())
            .build();
    }
    
    /**
     * Get decoded inputs for a slot
     */
    public List<PlanIntrospection.DecodedInput> getDecodedInputs(DynamicShapePlan plan, int slotIndex) {
        return PlanIntrospection.getDecodedInputs(plan, slotIndex);
    }
    
    /**
     * Get producers of a slot (slots that produce inputs for this slot)
     */
    public List<Integer> getProducersOf(DynamicShapePlan plan, int slotIndex) {
        return PlanIntrospection.getProducersOf(plan, slotIndex);
    }
    
    /**
     * Get dependents of a slot (slots that depend on this slot's outputs)
     */
    public List<Integer> getDependentsOf(DynamicShapePlan plan, int slotIndex) {
        return PlanIntrospection.getDependentsOf(plan, slotIndex);
    }
    
    // =========================================================================
    // Segment Introspection
    // =========================================================================
    
    /**
     * Get segment information for a plan
     */
    public List<SegmentInfo> getSegments(DynamicShapePlan plan) {
        List<PlanIntrospection.SegmentInfo> planSegments = PlanIntrospection.getSegments(plan);
        List<SegmentInfo> result = new ArrayList<>();
        for (PlanIntrospection.SegmentInfo seg : planSegments) {
            SegmentInfo newSeg = new SegmentInfo(
                seg.getStartSlot(), seg.getEndSlot(), seg.isCapturable(),
                seg.getDeviceId(), seg.getSlotCount(), seg.getOpNames());
            newSeg.setReplayBackendName(seg.getReplayBackendName());
            newSeg.setReplayState(seg.getReplayState());
            newSeg.setReplayCount(seg.getReplayCount());
            newSeg.setExecutionCount(seg.getExecutionCount());
            newSeg.setCompilationFailed(seg.isCompilationFailed());
            newSeg.setStatisticsJson(seg.getStatisticsJson());
            result.add(newSeg);
        }
        return result;
    }

    /**
     * Get segment information with replay state from native handle
     */
    public List<SegmentInfo> getSegmentsWithReplayState(DynamicShapePlan plan, Pointer nativeHandle) {
        if (nativeHandle == null) {
            return getSegments(plan);
        }
        List<PlanIntrospection.SegmentInfo> planSegments = PlanIntrospection.getSegmentsWithReplayState(plan, nativeHandle);
        List<SegmentInfo> result = new ArrayList<>();
        for (PlanIntrospection.SegmentInfo seg : planSegments) {
            SegmentInfo newSeg = new SegmentInfo(
                seg.getStartSlot(), seg.getEndSlot(), seg.isCapturable(),
                seg.getDeviceId(), seg.getSlotCount(), seg.getOpNames());
            newSeg.setReplayBackendName(seg.getReplayBackendName());
            newSeg.setReplayState(seg.getReplayState());
            newSeg.setReplayCount(seg.getReplayCount());
            newSeg.setExecutionCount(seg.getExecutionCount());
            newSeg.setCompilationFailed(seg.isCompilationFailed());
            newSeg.setStatisticsJson(seg.getStatisticsJson());
            result.add(newSeg);
        }
        return result;
    }
    
    /**
     * Get segment information from Framework API (uses current plan from executor)
     */
    public List<SegmentInfo> getSegments() {
        // Try to get current plan from executor
        // This would need integration with DynamicShapePlanExecutor
        return new ArrayList<>();
    }
    
    /**
     * Get number of segments in a plan
     */
    public int getNumSegments(DynamicShapePlan plan) {
        return getSegments(plan).size();
    }
    
    /**
     * Get segment summary as JSON
     */
    public String getSegmentsSummaryJson(DynamicShapePlan plan, Pointer nativeHandle) {
        if (nativeHandle == null) {
            return "[]";
        }
        return nativeOps.getPlanSegmentsSummaryJson(nativeHandle);
    }
    
    // =========================================================================
    // Native Plan Handle Introspection
    // =========================================================================
    
    /**
     * Get number of segments from native plan handle
     */
    public int getPlanNumSegments(Pointer planHandle) {
        return nativeOps.getPlanNumSegments(planHandle);
    }
    
    /**
     * Get number of slots from native plan handle
     */
    public int getPlanNumSlots(Pointer planHandle) {
        return nativeOps.getPlanNumSlots(planHandle);
    }
    
    /**
     * Get number of external inputs from native plan handle
     */
    public int getPlanNumExternalInputs(Pointer planHandle) {
        return nativeOps.getPlanNumExternalInputs(planHandle);
    }

    // ── External input introspection ─────────────────────────────────────

    public boolean isExternalInputVariable(Pointer planHandle, int extIdx) {
        return nativeOps.getPlanIsExternalInputVariable(planHandle, extIdx);
    }

    public boolean isExternalInputPlaceholder(Pointer planHandle, int extIdx) {
        return nativeOps.getPlanIsExternalInputPlaceholder(planHandle, extIdx);
    }

    public int getNumVariableExternalInputs(Pointer planHandle) {
        return nativeOps.getPlanNumVariableExternalInputs(planHandle);
    }

    public long getStagingBufferAddress(Pointer planHandle, int extIdx) {
        return nativeOps.getPlanStagingBufferAddress(planHandle, extIdx);
    }

    public long getEffectiveExternalAddress(Pointer planHandle, int extIdx) {
        return nativeOps.getPlanEffectiveExternalAddress(planHandle, extIdx);
    }

    public int getNumStagingBuffers(Pointer planHandle) {
        return nativeOps.getPlanNumStagingBuffers(planHandle);
    }

    /**
     * Get number of requested outputs from native plan handle
     */
    public int getPlanNumRequestedOutputs(Pointer planHandle) {
        return nativeOps.getPlanNumRequestedOutputs(planHandle);
    }
    
    /**
     * Get number of captured graph segments
     */
    public int getPlanNumCapturedGraphSegments(Pointer planHandle) {
        return nativeOps.getPlanNumCapturedGraphSegments(planHandle);
    }
    
    /**
     * Get total graph replays
     */
    public int getPlanTotalGraphReplays(Pointer planHandle) {
        return nativeOps.getPlanTotalGraphReplays(planHandle);
    }
    
    /**
     * Get number of host-only ops
     */
    public int getPlanNumHostOnlyOps(Pointer planHandle) {
        return nativeOps.getPlanNumHostOnlyOps(planHandle);
    }
    
    /**
     * Get host-only op names
     */
    public String getPlanHostOnlyOpNames(Pointer planHandle) {
        return nativeOps.getPlanHostOnlyOpNames(planHandle);
    }
    
    /**
     * Get segment backend name
     */
    public String getPlanSegmentBackendName(Pointer planHandle, int segmentIdx) {
        return nativeOps.getPlanSegmentBackendName(planHandle, segmentIdx);
    }
    
    /**
     * Get segment replay state (0=EMPTY, 1=CAPTURING, 2=CAPTURED, 3=READY, 4=ERROR)
     */
    public int getPlanSegmentReplayState(Pointer planHandle, int segmentIdx) {
        return nativeOps.getPlanSegmentReplayState(planHandle, segmentIdx);
    }
    
    /**
     * Get segment replay count
     */
    public int getPlanSegmentReplayCount(Pointer planHandle, int segmentIdx) {
        return nativeOps.getPlanSegmentReplayCount(planHandle, segmentIdx);
    }
    
    /**
     * Get segment execution count
     */
    public int getPlanSegmentExecutionCount(Pointer planHandle, int segmentIdx) {
        return nativeOps.getPlanSegmentExecutionCount(planHandle, segmentIdx);
    }
    
    /**
     * Get segment execution phase
     */
    public int getPlanSegmentExecutionPhase(Pointer planHandle, int segmentIdx) {
        return nativeOps.getPlanSegmentExecutionPhase(planHandle, segmentIdx);
    }
    
    /**
     * Get segment statistics as JSON
     */
    public String getPlanSegmentStatisticsJson(Pointer planHandle, int segmentIdx) {
        return nativeOps.getPlanSegmentStatisticsJson(planHandle, segmentIdx);
    }
    
    /**
     * Get segment compiled backend
     */
    public String getPlanSegmentCompiledBackend(Pointer planHandle, int segmentIdx) {
        return nativeOps.getPlanSegmentCompiledBackend(planHandle, segmentIdx);
    }
    
    /**
     * Get segment compilation audit as JSON
     */
    public String getPlanSegmentCompilationAudit(Pointer planHandle, int segmentIdx) {
        return nativeOps.getPlanSegmentCompilationAudit(planHandle, segmentIdx);
    }
    
    /**
     * Get segment tracked pointers as JSON
     */
    public String getPlanSegmentTrackedPointers(Pointer planHandle, int segmentIdx) {
        return nativeOps.getPlanSegmentTrackedPointers(planHandle, segmentIdx);
    }
    
    /**
     * Get segment capture buffers as JSON
     */
    public String getPlanSegmentCaptureBuffersJson(Pointer planHandle, int segmentIdx) {
        return nativeOps.getPlanSegmentCaptureBuffersJson(planHandle, segmentIdx);
    }
    
    /**
     * Get available backends for a plan
     */
    public String getPlanAvailableBackends(Pointer planHandle) {
        return nativeOps.getPlanAvailableBackends(planHandle);
    }
    
    /**
     * Get capture stats for a plan
     */
    public String getPlanCaptureStats(Pointer planHandle) {
        return nativeOps.getPlanCaptureStats(planHandle);
    }
    
    // =========================================================================
    // Memory Timeline
    // =========================================================================
    
    /**
     * Get memory allocation/release timeline for a plan
     */
    public List<PlanIntrospection.MemoryEvent> getMemoryTimeline(DynamicShapePlan plan) {
        return PlanIntrospection.getMemoryTimeline(plan);
    }
    
    /**
     * Get memory timeline summary
     */
    public String getMemoryTimelineSummary(DynamicShapePlan plan) {
        List<PlanIntrospection.MemoryEvent> timeline = getMemoryTimeline(plan);
        int allocCount = 0;
        int releaseCount = 0;
        
        for (PlanIntrospection.MemoryEvent event : timeline) {
            if (event.getEventType() == PlanIntrospection.MemoryEvent.EventType.ALLOCATE) {
                allocCount++;
            } else {
                releaseCount++;
            }
        }
        
        return String.format("Memory Timeline: %d allocations, %d releases across %d steps",
            allocCount, releaseCount, plan.getSlots().length);
    }
    
    // =========================================================================
    // Device Placement
    // =========================================================================
    
    /**
     * Get device placement map (deviceId -> list of slot indices)
     */
    public Map<Integer, List<Integer>> getDevicePlacement(DynamicShapePlan plan) {
        return PlanIntrospection.getDevicePlacement(plan);
    }
    
    /**
     * Get device placement summary
     */
    public String getDevicePlacementSummary(DynamicShapePlan plan) {
        Map<Integer, List<Integer>> placement = getDevicePlacement(plan);
        StringBuilder sb = new StringBuilder();
        
        for (Map.Entry<Integer, List<Integer>> entry : placement.entrySet()) {
            sb.append("Device ").append(entry.getKey()).append(": ")
              .append(entry.getValue().size()).append(" slots\n");
        }
        
        return sb.toString();
    }
    
    // =========================================================================
    // Parallel Groups
    // =========================================================================
    
    /**
     * Get parallel execution groups
     */
    public List<List<Integer>> getParallelGroups(DynamicShapePlan plan) {
        return PlanIntrospection.getParallelGroups(plan);
    }
    
    /**
     * Get parallel groups summary
     */
    public String getParallelGroupsSummary(DynamicShapePlan plan) {
        List<List<Integer>> groups = getParallelGroups(plan);
        return String.format("Parallel Groups: %d groups identified", groups.size());
    }
    
    // =========================================================================
    // Plan Visualization
    // =========================================================================
    
    /**
     * Get ASCII art representation of the plan
     */
    public String toAsciiArt(DynamicShapePlan plan) {
        StringBuilder sb = new StringBuilder();
        sb.append("=== DSP Plan Visualization ===\n\n");
        
        DynamicShapeSlot[] slots = plan.getSlots();
        for (int i = 0; i < slots.length; i++) {
            DynamicShapeSlot slot = slots[i];
            sb.append(String.format("[%03d] %-30s (device=%d)\n",
                i, slot.getOpName(), slot.getDeviceId()));
            
            // Show inputs
            List<PlanIntrospection.DecodedInput> inputs = getDecodedInputs(plan, i);
            if (!inputs.isEmpty()) {
                sb.append("      Inputs: ");
                for (int j = 0; j < inputs.size(); j++) {
                    if (j > 0) sb.append(", ");
                    sb.append(inputs.get(j).toString());
                }
                sb.append("\n");
            }
        }
        
        return sb.toString();
    }
    
    // =========================================================================
    // Inner Classes
    // =========================================================================
    
    /**
     * Slot information
     */
    @lombok.Data
    @lombok.Builder
    public static class SlotInfo {
        private int slotIndex;
        private String opName;
        private String opType;
        private int numInputs;
        private int numOutputs;
        private int deviceId;
        private boolean capturable;
    }
    
    /**
     * Segment information (extends PlanIntrospection.SegmentInfo)
     */
    public static class SegmentInfo extends PlanIntrospection.SegmentInfo {
        public SegmentInfo(int startSlot, int endSlot, boolean capturable, int deviceId,
                           int slotCount, List<String> opNames) {
            super(startSlot, endSlot, capturable, deviceId, slotCount, opNames);
        }
    }
}