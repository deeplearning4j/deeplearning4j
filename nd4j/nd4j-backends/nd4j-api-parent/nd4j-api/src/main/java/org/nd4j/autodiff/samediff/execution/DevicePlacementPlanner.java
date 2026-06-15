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

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.common.config.ND4JSystemProperties;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Plans device placement for DSP execution across multiple devices.
 *
 * <p>Takes a compiled {@link DynamicShapePlan} and a {@link SameDiff} model, analyzes
 * memory requirements, and assigns {@code targetDeviceId} for each slot. The executor
 * already fully supports per-slot device targeting — this planner only decides
 * where to place ops.</p>
 *
 * <p>Disabled by default (zero overhead when off). Enable via
 * {@code -Dnd4j.placement.enabled=true} or by calling
 * {@code sd.setPlacementStrategy(strategy)}.</p>
 *
 * @see DynamicShapePlan#assignDevices()
 * @see DynamicShapePlanExecutor
 */
@Slf4j
public class DevicePlacementPlanner {

    /**
     * Device placement strategy.
     */
    public enum PlacementStrategy {
        /** All ops on one GPU (default device 0). Zero cross-device transfers. */
        SINGLE_DEVICE,
        /** Greedy: fill current device until estimated full, then spill to next device with most free memory. */
        MEMORY_FIT,
        /** Partition slots into N contiguous segments (N = device count), minimizing cross-segment live variables. */
        PIPELINE_PARALLEL,
        /** User provides explicit variable name → device ID mapping. */
        CUSTOM
    }

    /**
     * Result of device placement planning.
     */
    @Data
    @AllArgsConstructor
    public static class PlacementPlan {
        /** Per-slot device assignment. Index = slot index, value = device ID. */
        private final int[] slotDeviceIds;
        /** Where to pre-replicate constants. Key = variable name, value = device ID. */
        private final Map<String, Integer> constantDeviceIds;
        /** Which strategy produced this plan. */
        private final PlacementStrategy strategy;
        /** Estimated bytes per device after placement. */
        private final Map<Integer, Long> deviceMemoryEstimates;
        /** False if the planner fell back to single device due to errors or insufficient info. */
        private final boolean valid;
    }

    /**
     * Plan device placement for a compiled DSP plan.
     *
     * @param plan     compiled DSP plan
     * @param sd       the SameDiff model
     * @param strategy placement strategy to use
     * @return placement plan with per-slot device assignments
     */
    public static PlacementPlan plan(DynamicShapePlan plan, SameDiff sd, PlacementStrategy strategy) {
        if (strategy == null) strategy = PlacementStrategy.SINGLE_DEVICE;

        DynamicShapeSlot[] slots = plan.getSlots();
        if (slots == null || slots.length == 0) {
            return singleDevicePlan(0, 0, strategy);
        }

        int defaultDevice = Integer.getInteger(ND4JSystemProperties.PLACEMENT_DEFAULT_DEVICE, 0);
        boolean logDecisions = Boolean.getBoolean(ND4JSystemProperties.PLACEMENT_LOG_DECISIONS);

        switch (strategy) {
            case SINGLE_DEVICE:
                return planSingleDevice(plan, sd, defaultDevice, logDecisions);
            case MEMORY_FIT:
                return planMemoryFit(plan, sd, defaultDevice, logDecisions);
            case PIPELINE_PARALLEL:
                return planPipelineParallel(plan, sd, logDecisions);
            case CUSTOM:
                Map<String, Integer> customMap = sd.getCustomDevicePlacement();
                if (customMap == null || customMap.isEmpty()) {
                    log.warn("CUSTOM placement strategy but no custom mapping provided; falling back to SINGLE_DEVICE");
                    return planSingleDevice(plan, sd, defaultDevice, logDecisions);
                }
                return planCustom(plan, sd, customMap);
            default:
                return planSingleDevice(plan, sd, defaultDevice, logDecisions);
        }
    }

    /**
     * Plan with explicit variable-to-device mapping.
     *
     * @param plan           compiled DSP plan
     * @param sd             the SameDiff model
     * @param variableToDevice mapping of variable names to device IDs
     * @return placement plan
     */
    public static PlacementPlan planCustom(DynamicShapePlan plan, SameDiff sd,
                                           Map<String, Integer> variableToDevice) {
        DynamicShapeSlot[] slots = plan.getSlots();
        int numSlots = slots != null ? slots.length : 0;
        int[] slotDevices = new int[numSlots];
        int defaultDevice = Integer.getInteger(ND4JSystemProperties.PLACEMENT_DEFAULT_DEVICE, 0);
        Map<Integer, Long> deviceEstimates = new HashMap<>();

        for (int i = 0; i < numSlots; i++) {
            DynamicShapeSlot slot = slots[i];
            int device = defaultDevice;

            // Check output variable names for custom mapping
            String[] outputNames = slot.getOutputVarNames();
            if (outputNames != null) {
                for (String name : outputNames) {
                    if (variableToDevice.containsKey(name)) {
                        device = variableToDevice.get(name);
                        break;
                    }
                }
            }

            // Also check input variable names
            String[] inputNames = slot.getInputVarNames();
            if (inputNames != null && device == defaultDevice) {
                for (String name : inputNames) {
                    if (variableToDevice.containsKey(name)) {
                        device = variableToDevice.get(name);
                        break;
                    }
                }
            }

            slotDevices[i] = device;
            deviceEstimates.merge(device, 4L, Long::sum); // minimal estimate
        }

        // Build constant device map
        Map<String, Integer> constantDevices = buildConstantDeviceMap(sd, variableToDevice, defaultDevice);

        return new PlacementPlan(slotDevices, constantDevices, PlacementStrategy.CUSTOM, deviceEstimates, true);
    }

    /**
     * Apply a placement plan to a DSP plan by setting targetDeviceId on each slot.
     *
     * @param plan      the DSP plan to modify
     * @param placement the placement plan to apply
     */
    public static void applyPlan(DynamicShapePlan plan, PlacementPlan placement) {
        if (placement == null || !placement.isValid()) return;

        DynamicShapeSlot[] slots = plan.getSlots();
        int[] slotDevices = placement.getSlotDeviceIds();
        if (slots == null || slotDevices == null) return;

        int len = Math.min(slots.length, slotDevices.length);
        for (int i = 0; i < len; i++) {
            slots[i].setTargetDeviceId(slotDevices[i]);
        }

        if (log.isDebugEnabled()) {
            log.debug("Applied {} placement: {} slots across {} devices",
                    placement.getStrategy(), len, placement.getDeviceMemoryEstimates().size());
        }
    }

    // ---- Private strategy implementations ----

    private static PlacementPlan planSingleDevice(DynamicShapePlan plan, SameDiff sd,
                                                   int deviceId, boolean logDecisions) {
        DynamicShapeSlot[] slots = plan.getSlots();
        int numSlots = slots.length;
        int[] slotDevices = new int[numSlots];
        for (int i = 0; i < numSlots; i++) {
            slotDevices[i] = deviceId;
        }

        long totalMem = ModelMemoryEstimator.estimateWeightMemory(sd);
        Map<Integer, Long> deviceEstimates = new HashMap<>();
        deviceEstimates.put(deviceId, totalMem);

        Map<String, Integer> constantDevices = new HashMap<>();
        for (SDVariable var : sd.variables()) {
            if (var.getVariableType() == VariableType.CONSTANT || var.getVariableType() == VariableType.VARIABLE) {
                constantDevices.put(var.name(), deviceId);
            }
        }

        if (logDecisions) {
            log.info("SINGLE_DEVICE placement: all {} slots on device {}, weight memory: {}MB",
                    numSlots, deviceId, totalMem / (1024 * 1024));
        }

        return new PlacementPlan(slotDevices, constantDevices, PlacementStrategy.SINGLE_DEVICE, deviceEstimates, true);
    }

    private static PlacementPlan planMemoryFit(DynamicShapePlan plan, SameDiff sd,
                                                int defaultDevice, boolean logDecisions) {
        DynamicShapeSlot[] slots = plan.getSlots();
        int numSlots = slots.length;
        int[] slotDevices = new int[numSlots];

        // Query available devices and their free memory
        Map<Integer, Long> deviceFreeMemory = queryDeviceMemory();
        if (deviceFreeMemory.isEmpty()) {
            // No device info available — fall back to single device
            return planSingleDevice(plan, sd, defaultDevice, logDecisions);
        }

        double headroom = Double.parseDouble(
                System.getProperty(ND4JSystemProperties.PLACEMENT_MEMORY_HEADROOM, "0.1"));

        // Apply headroom reduction to available memory
        Map<Integer, Long> deviceBudget = new LinkedHashMap<>();
        for (Map.Entry<Integer, Long> entry : deviceFreeMemory.entrySet()) {
            long budget = (long) (entry.getValue() * (1.0 - headroom));
            if (budget > 0) deviceBudget.put(entry.getKey(), budget);
        }

        if (deviceBudget.isEmpty()) {
            return planSingleDevice(plan, sd, defaultDevice, logDecisions);
        }

        // Track running memory usage per device
        Map<Integer, Long> deviceUsed = new HashMap<>();
        for (int d : deviceBudget.keySet()) deviceUsed.put(d, 0L);

        int currentDevice = deviceBudget.containsKey(defaultDevice) ? defaultDevice :
                deviceBudget.keySet().iterator().next();

        Map<Integer, Long> deviceEstimates = new HashMap<>();

        for (int i = 0; i < numSlots; i++) {
            // Estimate this slot's output memory (conservative: 4 bytes per output slot)
            DynamicShapeSlot slot = slots[i];
            int[] outIndices = slot.getOutputSlotIndices();
            long slotBytes = outIndices != null ? (long) outIndices.length * 1024 : 1024; // 1KB per output

            long currentUsed = deviceUsed.getOrDefault(currentDevice, 0L);
            long currentBudget = deviceBudget.getOrDefault(currentDevice, 0L);

            // If current device is estimated full, find device with most free space
            if (currentUsed + slotBytes > currentBudget) {
                int bestDevice = currentDevice;
                long bestFree = 0;
                for (Map.Entry<Integer, Long> entry : deviceBudget.entrySet()) {
                    long free = entry.getValue() - deviceUsed.getOrDefault(entry.getKey(), 0L);
                    if (free > bestFree) {
                        bestFree = free;
                        bestDevice = entry.getKey();
                    }
                }
                currentDevice = bestDevice;
            }

            slotDevices[i] = currentDevice;
            deviceUsed.merge(currentDevice, slotBytes, Long::sum);
            deviceEstimates.merge(currentDevice, slotBytes, Long::sum);
        }

        Map<String, Integer> constantDevices = new HashMap<>();
        for (SDVariable var : sd.variables()) {
            if (var.getVariableType() == VariableType.CONSTANT || var.getVariableType() == VariableType.VARIABLE) {
                constantDevices.put(var.name(), defaultDevice);
            }
        }

        if (logDecisions) {
            log.info("MEMORY_FIT placement: {} slots across {} devices, estimates: {}",
                    numSlots, deviceEstimates.size(), formatMemoryMap(deviceEstimates));
        }

        return new PlacementPlan(slotDevices, constantDevices, PlacementStrategy.MEMORY_FIT,
                deviceEstimates, deviceEstimates.size() > 0);
    }

    private static PlacementPlan planPipelineParallel(DynamicShapePlan plan, SameDiff sd,
                                                      boolean logDecisions) {
        DynamicShapeSlot[] slots = plan.getSlots();
        int numSlots = slots.length;

        Map<Integer, Long> deviceFreeMemory = queryDeviceMemory();
        int numDevices = Math.max(1, deviceFreeMemory.size());
        if (numDevices <= 1) {
            int defaultDevice = Integer.getInteger(ND4JSystemProperties.PLACEMENT_DEFAULT_DEVICE, 0);
            return planSingleDevice(plan, sd, defaultDevice, logDecisions);
        }

        int[] slotDevices = new int[numSlots];
        int[][] releaseAtStep = plan.getReleaseAtStep();

        // Find optimal partition boundaries that minimize cross-segment live variables.
        // Use a simple approach: partition into N roughly equal segments, then adjust
        // boundaries to minimize live variables at cut points.
        int[] boundaries = findOptimalBoundaries(numSlots, numDevices, releaseAtStep);

        int[] deviceIds = deviceFreeMemory.keySet().stream().mapToInt(Integer::intValue).toArray();
        Map<Integer, Long> deviceEstimates = new HashMap<>();

        int deviceIdx = 0;
        for (int i = 0; i < numSlots; i++) {
            // Find which segment this slot belongs to
            while (deviceIdx < boundaries.length - 1 && i >= boundaries[deviceIdx + 1]) {
                deviceIdx++;
            }
            int device = deviceIds[Math.min(deviceIdx, deviceIds.length - 1)];
            slotDevices[i] = device;
            deviceEstimates.merge(device, 1L, Long::sum);
        }

        Map<String, Integer> constantDevices = new HashMap<>();
        int defaultDevice = deviceIds.length > 0 ? deviceIds[0] : 0;
        for (SDVariable var : sd.variables()) {
            if (var.getVariableType() == VariableType.CONSTANT || var.getVariableType() == VariableType.VARIABLE) {
                constantDevices.put(var.name(), defaultDevice);
            }
        }

        if (logDecisions) {
            log.info("PIPELINE_PARALLEL placement: {} slots across {} devices, boundaries: {}",
                    numSlots, numDevices, java.util.Arrays.toString(boundaries));
        }

        return new PlacementPlan(slotDevices, constantDevices, PlacementStrategy.PIPELINE_PARALLEL,
                deviceEstimates, true);
    }

    /**
     * Find optimal boundaries for pipeline parallel partitioning.
     * Minimizes cross-segment live variables using releaseAtStep liveness data.
     */
    private static int[] findOptimalBoundaries(int numSlots, int numDevices, int[][] releaseAtStep) {
        int[] boundaries = new int[numDevices + 1];
        boundaries[0] = 0;
        boundaries[numDevices] = numSlots;

        if (releaseAtStep == null || releaseAtStep.length == 0) {
            // No liveness data — use equal partitioning
            for (int d = 1; d < numDevices; d++) {
                boundaries[d] = (int) ((long) numSlots * d / numDevices);
            }
            return boundaries;
        }

        // Count live variables at each step
        int[] liveCount = new int[numSlots];
        int currentLive = 0;
        for (int step = 0; step < numSlots; step++) {
            currentLive++; // each step produces at least one output
            if (step < releaseAtStep.length && releaseAtStep[step] != null) {
                currentLive -= releaseAtStep[step].length;
            }
            liveCount[step] = Math.max(0, currentLive);
        }

        // For each partition boundary, find the point with minimum live variables
        // within a window around the equal partition point
        for (int d = 1; d < numDevices; d++) {
            int equalPoint = (int) ((long) numSlots * d / numDevices);
            int windowStart = Math.max(1, equalPoint - numSlots / (numDevices * 4));
            int windowEnd = Math.min(numSlots - 1, equalPoint + numSlots / (numDevices * 4));

            int bestPoint = equalPoint;
            int minLive = Integer.MAX_VALUE;
            for (int p = windowStart; p <= windowEnd; p++) {
                if (liveCount[p] < minLive) {
                    minLive = liveCount[p];
                    bestPoint = p;
                }
            }
            boundaries[d] = bestPoint;
        }

        // Ensure boundaries are strictly increasing
        for (int d = 1; d < numDevices; d++) {
            if (boundaries[d] <= boundaries[d - 1]) {
                boundaries[d] = boundaries[d - 1] + 1;
            }
        }

        return boundaries;
    }

    /**
     * Query available device memory. Returns empty map on CPU-only backends.
     */
    private static Map<Integer, Long> queryDeviceMemory() {
        Map<Integer, Long> result = new LinkedHashMap<>();
        try {
            org.nd4j.nativeblas.NativeOps nativeOps =
                    org.nd4j.nativeblas.NativeOpsHolder.getInstance().getDeviceNativeOps();
            int numDevices = nativeOps.getAvailableDevices();
            for (int d = 0; d < numDevices; d++) {
                long free = nativeOps.getDeviceFreeMemory(d);
                if (free > 0) result.put(d, free);
            }
        } catch (Exception e) {
            // CPU backend or unavailable — return empty
        }
        return result;
    }

    private static Map<String, Integer> buildConstantDeviceMap(SameDiff sd,
                                                                Map<String, Integer> variableToDevice,
                                                                int defaultDevice) {
        Map<String, Integer> constantDevices = new HashMap<>();
        for (SDVariable var : sd.variables()) {
            if (var.getVariableType() == VariableType.CONSTANT || var.getVariableType() == VariableType.VARIABLE) {
                int device = variableToDevice.getOrDefault(var.name(), defaultDevice);
                constantDevices.put(var.name(), device);
            }
        }
        return constantDevices;
    }

    private static PlacementPlan singleDevicePlan(int numSlots, int deviceId, PlacementStrategy strategy) {
        int[] slotDevices = new int[numSlots];
        for (int i = 0; i < numSlots; i++) slotDevices[i] = deviceId;
        Map<Integer, Long> estimates = new HashMap<>();
        estimates.put(deviceId, 0L);
        return new PlacementPlan(slotDevices, new HashMap<>(), strategy, estimates, true);
    }

    private static String formatMemoryMap(Map<Integer, Long> memMap) {
        StringBuilder sb = new StringBuilder("{");
        boolean first = true;
        for (Map.Entry<Integer, Long> entry : memMap.entrySet()) {
            if (!first) sb.append(", ");
            sb.append("dev").append(entry.getKey()).append("=")
                    .append(entry.getValue() / (1024 * 1024)).append("MB");
            first = false;
        }
        sb.append("}");
        return sb.toString();
    }

    private DevicePlacementPlanner() {
        // Utility class
    }
}
