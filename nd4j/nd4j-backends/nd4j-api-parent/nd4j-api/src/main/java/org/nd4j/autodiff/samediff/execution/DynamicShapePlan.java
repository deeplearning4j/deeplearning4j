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

import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.ops.OpContext;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;

import org.bytedeco.javacpp.LongPointer;

import java.io.Closeable;
import java.util.*;

/**
 * A compiled execution plan for autoregressive inference with dynamic shapes.
 *
 * <p>Unlike {@link ExecutionPlan} which pre-allocates a fixed workspace for static shapes,
 * DynamicShapePlan handles the case where shapes change every step (e.g., growing KV cache).
 * It pre-compiles the graph wiring (input/output index mapping, liveness schedule) once,
 * then uses flat array-indexed slots instead of string-keyed HashMaps on each step.</p>
 *
 * <p>Key components:</p>
 * <ul>
 *   <li>{@code slots} — pre-compiled per-op descriptors with integer-indexed wiring</li>
 *   <li>{@code releaseAtStep} — pre-computed liveness: which output slots to release after each step</li>
 *   <li>{@code opContextPool} — pre-allocated OpContext pool (one per slot, reused across calls)</li>
 *   <li>{@code externalInputKeys} — ordered array of constant/variable/placeholder names for external resolution</li>
 * </ul>
 *
 * @see DynamicShapeSlot
 * @see DynamicShapePlanCompiler
 * @see DynamicShapePlanExecutor
 */
@Slf4j
@Data
public class DynamicShapePlan implements Closeable {

    /** Pre-compiled per-op descriptors in execution order. */
    private final DynamicShapeSlot[] slots;

    /** Total number of flat output slots (for INDArray[] allocation). */
    private final int totalOutputSlots;

    /**
     * Release schedule: {@code releaseAtStep[i]} contains the flat output slot indices
     * that become dead after step {@code i} completes. Empty array if nothing to release.
     */
    private final int[][] releaseAtStep;

    /** Pre-allocated OpContext pool, one per slot. Reused across execute() calls. */
    private final OpContext[] opContextPool;

    /**
     * Ordered array of external input variable names (constants, variables, placeholders).
     * External inputs are referenced by negative indices in DynamicShapeSlot.inputSourceIndices:
     * {@code -(index + 1)} maps into this array.
     */
    private final String[] externalInputKeys;

    /** The set of output variable names this plan was compiled for. */
    private final Set<String> requestedOutputs;

    /**
     * Pre-built mapping from requested output variable name to its flat output slot index.
     * Enables O(1) output collection instead of O(outputs * slots) linear search.
     */
    private final Map<String, Integer> outputNameToSlotIndex;

    /** Whether any slot has control flow ops (plan is invalid if true — should not be created). */
    private final boolean hasControlFlowOps;

    /**
     * Get a human-readable summary of this plan.
     */
    public String getSummary() {
        int totalReleasable = 0;
        for (int[] releases : releaseAtStep) {
            totalReleasable += releases.length;
        }
        return "DynamicShapePlan{slots=" + slots.length +
                ", outputSlots=" + totalOutputSlots +
                ", externalInputs=" + externalInputKeys.length +
                ", releasableSlots=" + totalReleasable +
                ", outputs=" + requestedOutputs +
                "}";
    }

    /**
     * Clear all per-slot shape caches.
     *
     * <p>This MUST be called when a session resets or when a new executor initializes with this plan.
     * The cached DataBuffer references in each slot's {@code cachedOutputShapes} may point to freed
     * GPU memory from a previous session. Reusing these stale references causes memory corruption.</p>
     *
     * <p>The plan itself survives session resets (cached at SameDiff level) because the graph wiring
     * doesn't change. Only the runtime shape cache contents are session-specific.</p>
     */
    public void clearAllShapeCaches() {
        if (slots != null) {
            for (DynamicShapeSlot slot : slots) {
                if (slot != null) {
                    slot.clearShapeCache();
                }
            }
        }
    }

    /**
     * Auto-assign target devices to plan slots for multi-GPU model parallelism.
     * Queries available CUDA devices and their free memory, then distributes
     * ops across devices proportionally to available memory.
     *
     * <p>For single-GPU or CPU backends, this is a no-op (all slots stay at
     * targetDeviceId=-1, meaning "use current thread's device").</p>
     */
    public void assignDevices() {
        int numDevices;
        try {
            numDevices = NativeOpsHolder.getInstance().getDeviceNativeOps().getAvailableDevices();
        } catch (Exception e) {
            return; // CPU backend or no devices
        }
        if (numDevices <= 1 || slots == null || slots.length == 0) return;
        if (Boolean.getBoolean("nd4j.dsp.singleGpu")) {
            log.info("DSP single-GPU mode forced via nd4j.dsp.singleGpu=true");
            return;
        }

        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();

        // Include ALL devices. Cross-device data transfer for non-P2P devices
        // stages through host memory via replicateToDevice() (views auto-dup'd).
        // Use pool-aware memory accounting: cudaMemGetInfo reports free memory MINUS
        // pool reserved, but cudaMallocAsync can reuse reserved pool memory. Without
        // accounting for this, devices that previously ran a large graph (e.g., vision
        // encoder) appear nearly full even though their pool has GB of reusable memory.
        // available = cudaFree + (poolReserved - poolUsed)
        //
        // For non-P2P secondary devices, use only a fraction of available memory as
        // the budget for op assignment.
        Map<Integer, Long> freeMemory = new LinkedHashMap<>();
        for (int d = 0; d < numDevices; d++) {
            long cudaFree = nativeOps.getDeviceFreeMemory(d);
            long total = nativeOps.getDeviceTotalMemory(d);
            boolean p2p = (d == 0) || nativeOps.isPeerAccessSupported(0, d);

            // Add reusable pool memory (reserved but not used by live allocations).
            // cudaMallocAsync can allocate from this without going to the driver.
            long poolReusable = 0;
            try {
                LongPointer usedPtr = new LongPointer(1);
                LongPointer reservedPtr = new LongPointer(1);
                nativeOps.getMemoryPoolStats(d, usedPtr, reservedPtr);
                long poolUsed = usedPtr.get();
                long poolReserved = reservedPtr.get();
                poolReusable = Math.max(0, poolReserved - poolUsed);
            } catch (Exception ignored) {}

            long available = cudaFree + poolReusable;
            // All devices participate in op assignment. Non-P2P secondary devices use
            // host-staged transfers (D2H + H2D) via replicateToDevice() which auto-dups
            // views before replication. The intermediate contiguous copy leak that caused
            // ~30MB/step growth has been fixed (contiguous.close() in replicateToDevice).
            // Use a smaller fraction for non-P2P secondary devices since host-staged
            // transfers are slower than P2P and the small GPU has limited memory.
            long budget = (d == 0 || p2p) ? available : (long)(available * 0.15);
            freeMemory.put(d, budget);
            log.info("  Device {}: {}MB cudaFree + {}MB poolReusable = {}MB available / {}MB total (P2P: {}){}",
                    d, cudaFree / (1024 * 1024), poolReusable / (1024 * 1024),
                    available / (1024 * 1024), total / (1024 * 1024),
                    d == 0 ? "self" : p2p ? "yes" : "no (host-staged transfers)",
                    (!p2p && d != 0) ? " [budget: " + (budget / (1024 * 1024)) + "MB @ 15%]" : "");
        }
        if (freeMemory.size() <= 1) return; // Only one usable device
        assignDevices(freeMemory);
    }

    /**
     * Assign target devices using provided memory budgets per device.
     * Devices with more memory get proportionally more ops. Ops are assigned
     * in execution order, so early ops (typically early layers) go to the
     * device with most memory.
     *
     * @param deviceMemoryBudgets map of deviceId to available bytes for computation
     */
    public void assignDevices(Map<Integer, Long> deviceMemoryBudgets) {
        if (deviceMemoryBudgets == null || deviceMemoryBudgets.size() <= 1
                || slots == null || slots.length == 0) return;

        long totalMem = 0;
        for (long mem : deviceMemoryBudgets.values()) {
            totalMem += mem;
        }
        if (totalMem <= 0) return;

        // Sort devices largest-first so the primary GPU gets the first (most) slots
        List<Map.Entry<Integer, Long>> sorted = new ArrayList<>(deviceMemoryBudgets.entrySet());
        sorted.sort((a, b) -> Long.compare(b.getValue(), a.getValue()));

        int assigned = 0;
        for (int i = 0; i < sorted.size(); i++) {
            int deviceId = sorted.get(i).getKey();
            long deviceMem = sorted.get(i).getValue();

            int slotsForDevice;
            if (i == sorted.size() - 1) {
                slotsForDevice = slots.length - assigned; // Last device gets remainder
            } else {
                slotsForDevice = (int) Math.round((double) deviceMem / totalMem * slots.length);
            }

            for (int s = 0; s < slotsForDevice && assigned < slots.length; s++, assigned++) {
                slots[assigned].setTargetDeviceId(deviceId);
            }
        }

        log.info("Device placement: {} slots across {} devices — {}",
                slots.length, deviceMemoryBudgets.size(), getDeviceAssignmentSummary());
    }

    /**
     * Assign a specific device to a range of slots [startSlot, endSlot).
     * Useful for manual pipeline parallelism where layer boundaries are known.
     *
     * @param startSlot inclusive start index
     * @param endSlot   exclusive end index
     * @param deviceId  CUDA device ID to assign
     */
    public void assignDeviceToRange(int startSlot, int endSlot, int deviceId) {
        if (slots == null) return;
        for (int i = Math.max(0, startSlot); i < Math.min(endSlot, slots.length); i++) {
            slots[i].setTargetDeviceId(deviceId);
        }
    }

    /**
     * Get a human-readable summary of device assignments across slots.
     */
    public String getDeviceAssignmentSummary() {
        if (slots == null) return "No slots";
        Map<Integer, Integer> counts = new LinkedHashMap<>();
        for (DynamicShapeSlot slot : slots) {
            counts.merge(slot.getTargetDeviceId(), 1, Integer::sum);
        }
        StringBuilder sb = new StringBuilder("DevicePlacement{");
        boolean first = true;
        for (Map.Entry<Integer, Integer> entry : counts.entrySet()) {
            if (!first) sb.append(", ");
            int devId = entry.getKey();
            sb.append("device").append(devId < 0 ? "Default" : String.valueOf(devId));
            sb.append("=").append(entry.getValue()).append(" ops");
            first = false;
        }
        sb.append("}");
        return sb.toString();
    }

    @Override
    public void close() {
        // Clear shape caches first to release DataBuffer references
        clearAllShapeCaches();

        if (opContextPool != null) {
            for (OpContext ctx : opContextPool) {
                if (ctx != null) {
                    try {
                        ctx.close();
                    } catch (Exception e) {
                        // ignore cleanup errors
                    }
                }
            }
        }
    }
}
