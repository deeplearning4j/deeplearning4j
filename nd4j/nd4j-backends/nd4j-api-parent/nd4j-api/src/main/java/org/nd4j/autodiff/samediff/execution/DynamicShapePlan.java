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
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.common.config.ND4JSystemProperties;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.device.MultiGpuTracer;
import org.nd4j.linalg.api.ops.OpContext;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;

import org.bytedeco.javacpp.LongPointer;

import java.io.Closeable;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.*;
import java.util.stream.Collectors;

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
@Getter
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

    /**
     * Source type for each external input, parallel to {@code externalInputKeys}.
     * Values: {@link DynamicShapeSlot#SOURCE_CONSTANT}, {@link DynamicShapeSlot#SOURCE_VARIABLE},
     * {@link DynamicShapeSlot#SOURCE_PLACEHOLDER}.
     */
    @Getter private final byte[] externalInputSourceTypes;

    /** The set of output variable names this plan was compiled for. */
    private final Set<String> requestedOutputs;

    /**
     * Pre-built mapping from requested output variable name to its flat output slot index.
     * Enables O(1) output collection instead of O(outputs * slots) linear search.
     */
    private final Map<String, Integer> outputNameToSlotIndex;

    /** Whether this plan contains control flow ops (Switch/Merge/Enter/Exit/NextIteration/LoopCond). */
    private final boolean hasControlFlowOps;

    /**
     * Loop regions detected during compilation. Each region describes a while-loop
     * with Merge/LoopCond/Switch/NextIteration/Exit slot indices.
     */
    @Getter private final LoopRegion[] loopRegions;

    /**
     * Describes a while-loop region in the DSP plan.
     */
    @Data
    public static class LoopRegion {
        /** Step index of the Merge op (loop entry / back-edge target). */
        public final int mergeSlot;
        /** Step index of the Switch op (loop gate). */
        public final int switchSlot;
        /** Step index of the NextIteration op (triggers jump-back). */
        public final int nextIterSlot;
        /** Step index of the Exit op (output when loop ends). */
        public final int exitSlot;
        /** First body slot after Switch. */
        public final int bodyStartSlot;
        /** Last body slot (= nextIterSlot). */
        public final int bodyEndSlot;
    }

    // --- Dependency graph for async parallel execution ---

    /** Initial predecessor count per step. A step is ready when its count reaches 0. */
    @Getter private final int[] predecessorCounts;

    /** Predecessor step indices per step (for building predecessor sets for parallel group detection). */
    @Getter private final int[][] predecessors;

    /** Successor step indices per step. When step i completes, decrement predecessorCounts for each successor. */
    @Getter private final int[][] successors;

    /** Consumer count per output slot. A slot is freeable when its consumer count reaches 0. */
    @Getter private final int[] consumerCounts;

    /** Steps with 0 predecessors — execution entry points for async mode. */
    @Getter private final int[] rootSlots;

    /** Number of distinct devices in slot assignments. Computed by assignDevices(). */
    @Getter private int numDistinctDevices = 0;

    /**
     * Full constructor with dependency graph and loop regions.
     */
    public DynamicShapePlan(DynamicShapeSlot[] slots, int totalOutputSlots, int[][] releaseAtStep,
                            OpContext[] opContextPool, String[] externalInputKeys,
                            byte[] externalInputSourceTypes,
                            Set<String> requestedOutputs, Map<String, Integer> outputNameToSlotIndex,
                            boolean hasControlFlowOps, LoopRegion[] loopRegions,
                            int[] predecessorCounts, int[][] predecessors, int[][] successors,
                            int[] consumerCounts, int[] rootSlots) {
        this.slots = slots;
        this.totalOutputSlots = totalOutputSlots;
        this.releaseAtStep = releaseAtStep;
        this.opContextPool = opContextPool;
        this.externalInputKeys = externalInputKeys;
        this.externalInputSourceTypes = externalInputSourceTypes;
        this.requestedOutputs = requestedOutputs;
        this.outputNameToSlotIndex = outputNameToSlotIndex;
        this.hasControlFlowOps = hasControlFlowOps;
        this.loopRegions = loopRegions;
        this.predecessorCounts = predecessorCounts;
        this.predecessors = predecessors;
        this.successors = successors;
        this.consumerCounts = consumerCounts;
        this.rootSlots = rootSlots;
    }

    /**
     * Backward-compatible constructor without dependency graph.
     */
    public DynamicShapePlan(DynamicShapeSlot[] slots, int totalOutputSlots, int[][] releaseAtStep,
                            OpContext[] opContextPool, String[] externalInputKeys,
                            Set<String> requestedOutputs, Map<String, Integer> outputNameToSlotIndex,
                            boolean hasControlFlowOps) {
        this(slots, totalOutputSlots, releaseAtStep, opContextPool, externalInputKeys,
                new byte[externalInputKeys.length],
                requestedOutputs, outputNameToSlotIndex, hasControlFlowOps, null,
                null, null, null, null, null);
    }

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
        if (Boolean.getBoolean(ND4JSystemProperties.DSP_SINGLE_GPU)) {
            log.debug("DSP single-GPU mode forced via {}=true", ND4JSystemProperties.DSP_SINGLE_GPU);
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
            // Non-P2P secondary devices: when nd4j.dsp.multiGpuShard=true the caller
            // explicitly requests op-segment sharding across all GPUs.  In that mode give
            // non-P2P devices their full available budget so the placement policy assigns
            // tail segments to the secondary GPU (fitting the oversized model).  Outputs
            // produced on the secondary device are migrated back to device-0 asynchronously
            // after each execution (platformGetOutputForDevice0 in the CUDA backend).
            // Without the flag (default) the budget is zero: non-P2P devices receive no ops
            // and behaviour is byte-identical to single-GPU.  The explicit fraction override
            // (nd4j.dsp.nonP2pBudgetFraction) still applies on top of the flag.
            boolean multiGpuShard = Boolean.getBoolean(ND4JSystemProperties.DSP_MULTI_GPU_SHARD);
            // Helper: parse double property, returning defaultVal when absent or empty string.
            // Surefire can inject empty strings for <systemPropertyVariables> entries that have
            // no corresponding -D flag on the Maven command line; Double.parseDouble("") throws.
            String nonP2pPropRaw = System.getProperty(ND4JSystemProperties.DSP_NON_P2P_BUDGET_FRACTION);
            String nonP2pProp = (nonP2pPropRaw != null) ? nonP2pPropRaw.trim() : null;
            double nonP2pFraction = multiGpuShard ? 1.0 :
                    ((nonP2pProp != null && !nonP2pProp.isEmpty()) ? Double.parseDouble(nonP2pProp) : 0.0);
            // Allow explicit fraction override even in shard mode (e.g. 0.8 to leave headroom)
            if (multiGpuShard && nonP2pProp != null && !nonP2pProp.isEmpty()) {
                nonP2pFraction = Double.parseDouble(nonP2pProp);
            }
            long budget = (d == 0 || p2p) ? available : (long)(available * nonP2pFraction);
            if (budget > 0) {
                freeMemory.put(d, budget);
            }
            log.debug("  Device {}: {}MB cudaFree + {}MB poolReusable = {}MB available / {}MB total (P2P: {}){}",
                    d, cudaFree / (1024 * 1024), poolReusable / (1024 * 1024),
                    available / (1024 * 1024), total / (1024 * 1024),
                    d == 0 ? "self" : p2p ? "yes" : "no (host-staged transfers)",
                    budget <= 0 ? " [excluded from compute]" :
                    ((!p2p && d != 0 && nonP2pFraction < 1.0) ? " [budget: " + (budget / (1024 * 1024)) + "MB @ " + (int)(nonP2pFraction * 100) + "%]" : ""));
        }
        if (freeMemory.size() <= 1) return; // Only one usable device
        MultiGpuTracer.traceParallelExec("device-discovery",
                "found " + numDevices + " devices, " + freeMemory.size() + " usable");
        assignDevices(freeMemory);
    }

    /**
     * Assign target devices using provided memory budgets per device.
     * Devices with more memory get proportionally more ops. Ops are assigned
     * in execution order, so early ops (typically early layers) go to the
     * device with most memory.
     *
     * After proportional assignment, parallel groups (ops sharing the same
     * predecessor set) are split across devices so both GPUs have concurrent work.
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

        // Sort devices largest-first so the primary GPU gets the bulk of ops.
        // With a 24GB RTX 4090 + 8GB RTX 3070 Ti (30% budget = 2.4GB), device 0 gets
        // ~91% of ops. The secondary GPU only gets overflow ops at the end of the graph.
        // This minimizes cross-device data transfers and ensures the primary GPU (which
        // holds all model constants) runs most ops without needing constant replication.
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
            MultiGpuTracer.traceDeviceAssignment(deviceId, slotsForDevice, slots.length,
                    deviceMem / (1024 * 1024), totalMem / (1024 * 1024),
                    true /* P2P status not tracked here, logged in assignDevices() */);
        }

        // Split parallel groups across devices. Ops sharing the same predecessor set
        // can execute concurrently — assign alternate members to different devices.
        // This ensures both GPUs have work within each transformer layer's parallel
        // operations (Q/K/V projections, gate/up projections).
        if (predecessors != null && deviceMemoryBudgets.size() > 1) {
            int[] deviceIds = deviceMemoryBudgets.keySet().stream().mapToInt(Integer::intValue).toArray();
            splitParallelGroups(deviceIds);
        }

        // Compute numDistinctDevices from actual assignments
        computeNumDistinctDevices();

        log.debug("Device placement: {} slots across {} devices — {}",
                slots.length, numDistinctDevices, getDeviceAssignmentSummary());
    }

    /**
     * Split parallel groups across devices. Ops with identical predecessor sets
     * form a parallel group and are distributed round-robin across available devices.
     */
    private void splitParallelGroups(int[] deviceIds) {
        if (deviceIds.length <= 1 || predecessors == null) return;

        // Group steps by their predecessor set
        Map<List<Integer>, List<Integer>> parallelGroups = new HashMap<>();
        for (int i = 0; i < slots.length; i++) {
            int[] preds = predecessors[i];
            if (preds == null || preds.length == 0) continue;
            List<Integer> predKey = Arrays.stream(preds).sorted().boxed().collect(Collectors.toList());
            parallelGroups.computeIfAbsent(predKey, k -> new ArrayList<>()).add(i);
        }

        int reassigned = 0;
        for (List<Integer> group : parallelGroups.values()) {
            if (group.size() > 1) {
                for (int j = 0; j < group.size(); j++) {
                    int slotIdx = group.get(j);
                    int newDevice = deviceIds[j % deviceIds.length];
                    if (slots[slotIdx].getTargetDeviceId() != newDevice) {
                        slots[slotIdx].setTargetDeviceId(newDevice);
                        reassigned++;
                    }
                }
            }
        }
        if (reassigned > 0) {
            log.debug("Parallel group splitting: reassigned {} ops across {} devices",
                    reassigned, deviceIds.length);
        }
    }

    /**
     * Compute numDistinctDevices from actual slot assignments.
     */
    private void computeNumDistinctDevices() {
        if (slots == null || slots.length == 0) {
            numDistinctDevices = 0;
            return;
        }
        Set<Integer> devices = new HashSet<>();
        for (DynamicShapeSlot slot : slots) {
            int dev = slot.getTargetDeviceId();
            devices.add(dev >= 0 ? dev : 0);
        }
        numDistinctDevices = devices.size();
    }

    /**
     * Re-run device assignment with fresh memory budgets. Called after memory-intensive
     * phases (e.g., vision encoder) complete and free GPU memory on secondary devices.
     * Resets all slot assignments to default (-1) before re-running assignDevices().
     */
    public void reassignDevices() {
        if (slots == null || slots.length == 0) return;
        // Reset all slots to default device
        for (DynamicShapeSlot slot : slots) {
            slot.setTargetDeviceId(-1);
        }
        assignDevices(); // Re-runs with fresh memory budgets
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

    // ─── Binary Serialization for Native C++ Executor ─────────────────────────

    /** Magic bytes identifying a serialized DSP plan. */
    private static final int DSP_MAGIC = 0x44535031;  // "DSP1" in big-endian
    /** Serialization format version. V2 adds legacy ops. V3 adds control flow. V4 adds external names. V5 adds string args. V6 adds static zero-input output shapes. */
    private static final int DSP_VERSION = 6;

    /**
     * Serialize this plan into a compact binary format for the native C++ executor.
     *
     * <p>Binary format:</p>
     * <pre>
     *   Header: magic("DSP1") + version(int32) + numSlots(int32) + totalOutputSlots(int32)
     *           + numExternalInputs(int32) + numRequestedOutputs(int32)
     *   Per-slot: opNameHash(int64), opName(int32 len + UTF-8), numInputs(int32), numOutputs(int32),
     *             inputSourceIndices[numInputs](int32),
     *             inputSourceTypes[numInputs](int8),
     *             outputSlotIndices[numOutputs](int32),
     *             numIArgs(int32), iArgs[](int64),
     *             numTArgs(int32), tArgs[](double),
     *             numBArgs(int32), bArgs[](byte),
     *             numDArgs(int32), dArgs[](int32),
     *             numSArgs(int32), sArgs[](int32 len + UTF-8),
     *             flags: needsZeroedOutput(byte), isDataDependent(byte),
     *                    outputShapeDependsOnInputValues(byte), needsIntLongSync(byte),
     *                    isCustomOp(byte), targetDeviceId(int32),
     *             legacyOpType(int32), legacyOpNum(int32),
     *             controlFlowType(byte), loopBackTarget(int32), loopRegionIndex(int32),
     *             numStaticOutputShapes(int32),
     *             staticOutputShapes[numStaticOutputShapes](int32 length + int64[length])
     *   Release schedule: for each step: count(int32) + slotIndices[count](int32)
     *   Requested outputs: for each output: slotIndex(int32)
     * </pre>
     *
     * @return serialized plan bytes, or null if serialization fails
     */
    public byte[] serialize() {
        if (slots == null || slots.length == 0) return null;

        // Validate release schedule matches slot count
        if (releaseAtStep.length != slots.length) {
            throw new IllegalStateException(
                "Release schedule length (" + releaseAtStep.length +
                ") does not match slot count (" + slots.length +
                "). The release schedule must have exactly one entry per execution step.");
        }

        // Calculate buffer size
        int size = 24; // Header: magic + version + numSlots + totalOutputSlots + numExternalInputs + numRequestedOutputs
        for (DynamicShapeSlot slot : slots) {
            size += 8;  // opNameHash (long)
            // opName string: 4-byte length + UTF-8 bytes
            byte[] opNameBytes = slot.getOpName() != null ? slot.getOpName().getBytes(java.nio.charset.StandardCharsets.UTF_8) : new byte[0];
            size += 4 + opNameBytes.length;
            size += 8;  // numInputs + numOutputs (2 ints)
            size += slot.getInputSourceIndices().length * 4;  // inputSourceIndices
            size += slot.getInputSourceTypes().length;          // inputSourceTypes (bytes)
            size += slot.getOutputSlotIndices().length * 4;     // outputSlotIndices
            size += 20; // 5 arg counts (int each)
            size += (slot.getIArgs() != null ? slot.getIArgs().length : 0) * 8;  // iArgs (long)
            size += (slot.getTArgs() != null ? slot.getTArgs().length : 0) * 8;  // tArgs (double)
            size += (slot.getBArgs() != null ? slot.getBArgs().length : 0);       // bArgs (byte)
            size += (slot.getDArgs() != null ? slot.getDArgs().length : 0) * 4;  // dArgs (int toInt/C++ enum)
            String[] sArgs = slot.getSArgs();
            size += 4;
            if (sArgs != null) {
                for (String sArg : sArgs) {
                    byte[] sArgBytes = sArg != null ? sArg.getBytes(java.nio.charset.StandardCharsets.UTF_8) : new byte[0];
                    size += 4 + sArgBytes.length;
                }
            }
            size += 5 + 4; // 5 flag bytes + targetDeviceId int
            size += 8; // legacyOpType (int32) + legacyOpNum (int32)
            size += 1 + 4 + 4; // controlFlowType (byte) + loopBackTarget (int32) + loopRegionIndex (int32)
            long[][] staticOutputShapeInfos = slot.getStaticOutputShapeInfos();
            size += 4; // numStaticOutputShapes
            if (staticOutputShapeInfos != null) {
                for (long[] shapeInfo : staticOutputShapeInfos) {
                    size += 4 + (shapeInfo != null ? shapeInfo.length : 0) * 8;
                }
            }
        }
        // Release schedule
        for (int[] releases : releaseAtStep) {
            size += 4 + releases.length * 4; // count + indices
        }
        // Loop regions: count + 6 ints per region
        int numLoopRegions = (loopRegions != null) ? loopRegions.length : 0;
        size += 4 + numLoopRegions * 24; // count(int32) + 6 ints per region

        // Requested outputs — one slot-index per requestedOutputs entry (may be -1 if unresolved).
        // Header numRequestedOutputs and body must match; iterate on requestedOutputs in both.
        size += requestedOutputs.size() * 4;

        // External input names (v4+): count + per-name (int32 len + UTF-8 bytes)
        byte[][] extNameBytes = new byte[externalInputKeys.length][];
        for (int i = 0; i < externalInputKeys.length; i++) {
            extNameBytes[i] = externalInputKeys[i] != null
                ? externalInputKeys[i].getBytes(java.nio.charset.StandardCharsets.UTF_8) : new byte[0];
            size += 4 + extNameBytes[i].length;
        }

        ByteBuffer buf = ByteBuffer.allocate(size);
        buf.order(ByteOrder.LITTLE_ENDIAN);

        // Header
        buf.putInt(DSP_MAGIC);
        buf.putInt(DSP_VERSION);
        buf.putInt(slots.length);
        buf.putInt(totalOutputSlots);
        buf.putInt(externalInputKeys.length);
        buf.putInt(requestedOutputs.size());

        // Per-slot
        for (DynamicShapeSlot slot : slots) {
            buf.putLong(slot.getOpNameHash());
            // Write opName string: int32 length + UTF-8 bytes
            byte[] opNameBytes = slot.getOpName() != null ? slot.getOpName().getBytes(java.nio.charset.StandardCharsets.UTF_8) : new byte[0];
            buf.putInt(opNameBytes.length);
            buf.put(opNameBytes);
            buf.putInt(slot.getInputSourceIndices().length);
            buf.putInt(slot.getOutputSlotIndices().length);

            for (int idx : slot.getInputSourceIndices()) buf.putInt(idx);
            for (byte type : slot.getInputSourceTypes()) buf.put(type);
            for (int idx : slot.getOutputSlotIndices()) buf.putInt(idx);

            long[] iArgs = slot.getIArgs();
            double[] tArgs = slot.getTArgs();
            boolean[] bArgs = slot.getBArgs();
            DataType[] dArgs = slot.getDArgs();
            String[] sArgs = slot.getSArgs();

            // Write args interleaved: count then data for each arg type.
            buf.putInt(iArgs != null ? iArgs.length : 0);
            if (iArgs != null) for (long a : iArgs) buf.putLong(a);

            buf.putInt(tArgs != null ? tArgs.length : 0);
            if (tArgs != null) for (double a : tArgs) buf.putDouble(a);

            buf.putInt(bArgs != null ? bArgs.length : 0);
            if (bArgs != null) for (boolean a : bArgs) buf.put(a ? (byte) 1 : (byte) 0);

            buf.putInt(dArgs != null ? dArgs.length : 0);
            if (dArgs != null) for (DataType a : dArgs) buf.putInt(a.toInt());

            buf.putInt(sArgs != null ? sArgs.length : 0);
            if (sArgs != null) {
                for (String sArg : sArgs) {
                    byte[] sArgBytes = sArg != null ? sArg.getBytes(java.nio.charset.StandardCharsets.UTF_8) : new byte[0];
                    buf.putInt(sArgBytes.length);
                    buf.put(sArgBytes);
                }
            }

            buf.put((byte) 0);  // needsZeroedOutput — format placeholder, derived from OpTraitTable in C++
            buf.put((byte) 0);  // isDataDependent — format placeholder, derived from OpTraitTable in C++
            buf.put(slot.isOutputShapeDependsOnInputValues() ? (byte) 1 : (byte) 0);
            buf.put(slot.isNeedsIntLongSync() ? (byte) 1 : (byte) 0);
            buf.put(slot.isCustomOp() ? (byte) 1 : (byte) 0);
            buf.putInt(slot.getTargetDeviceId());
            buf.putInt(slot.getLegacyOpType());
            buf.putInt(slot.getLegacyOpNum());

            // V3: control flow metadata
            buf.put(slot.getControlFlowType());
            buf.putInt(slot.getLoopBackTarget());
            buf.putInt(slot.getLoopRegionIndex());

            // V6: full shape-info buffers for zero-input ops with declared static outputs.
            long[][] staticOutputShapeInfos = slot.getStaticOutputShapeInfos();
            buf.putInt(staticOutputShapeInfos != null ? staticOutputShapeInfos.length : 0);
            if (staticOutputShapeInfos != null) {
                for (long[] shapeInfo : staticOutputShapeInfos) {
                    buf.putInt(shapeInfo != null ? shapeInfo.length : 0);
                    if (shapeInfo != null) {
                        for (long value : shapeInfo) buf.putLong(value);
                    }
                }
            }
        }

        // Release schedule
        for (int[] releases : releaseAtStep) {
            buf.putInt(releases.length);
            for (int idx : releases) buf.putInt(idx);
        }

        // Loop regions
        buf.putInt(numLoopRegions);
        if (loopRegions != null) {
            for (LoopRegion lr : loopRegions) {
                buf.putInt(lr.mergeSlot);
                buf.putInt(lr.switchSlot);
                buf.putInt(lr.nextIterSlot);
                buf.putInt(lr.exitSlot);
                buf.putInt(lr.bodyStartSlot);
                buf.putInt(lr.bodyEndSlot);
            }
        }

        // Requested output slot indices — MUST match the order of requestedOutputs iteration,
        // because Java executeNative() maps results using new ArrayList<>(getRequestedOutputs()).
        for (String outputName : requestedOutputs) {
            Integer slotIdx = outputNameToSlotIndex.get(outputName);
            buf.putInt(slotIdx != null ? slotIdx : -1);
        }

        // External input names (v4+)
        for (byte[] nameBytes : extNameBytes) {
            buf.putInt(nameBytes.length);
            buf.put(nameBytes);
        }

        buf.flip();
        byte[] result = new byte[buf.remaining()];
        buf.get(result);
        return result;
    }

    // ─── Structure Hash for Disk Cache ────────────────────────────────────────

    /**
     * Compute the FNV-1a 64-bit hash of serialized plan bytes.
     * This hash serves as the disk cache key — it uniquely identifies the
     * graph structure (ops, wiring, args, control flow, outputs).
     *
     * <p>Matches the FNV-1a implementation in {@code DspHashUtils.h} (C++ side)
     * and {@code TritonGraphBackend_cache.cpp::computeDiskCacheHash()}.</p>
     *
     * @param serializedPlanBytes output of {@link #serialize()}
     * @return FNV-1a 64-bit hash, or 0 if input is null/empty
     */
    public static long computeStructureHash(byte[] serializedPlanBytes) {
        if (serializedPlanBytes == null || serializedPlanBytes.length == 0) return 0L;
        long hash = 0xcbf29ce484222325L;  // FNV-1a offset basis
        for (byte b : serializedPlanBytes) {
            hash ^= (b & 0xFFL);
            hash *= 0x100000001b3L;       // FNV-1a prime
        }
        return hash;
    }

    /**
     * Compute the structure hash of this plan by serializing it first.
     *
     * @return FNV-1a 64-bit hash of the serialized plan bytes
     */
    public long computeStructureHash() {
        return computeStructureHash(serialize());
    }

    /**
     * Deserialize a plan from binary bytes (for testing/validation).
     * This is the inverse of {@link #serialize()}.
     *
     * @param data serialized plan bytes
     * @return true if the data has a valid DSP1 header
     */
    public static boolean isValidSerializedPlan(byte[] data) {
        if (data == null || data.length < 24) return false;
        ByteBuffer buf = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);
        int magic = buf.getInt();
        int version = buf.getInt();
        return magic == DSP_MAGIC && (version >= 1 && version <= 5);
    }

    /**
     * Get a detailed multi-line summary of this plan including per-slot info,
     * segment boundaries, op histogram, device placement, and memory timeline.
     */
    public String getDetailedSummary() {
        return PlanIntrospection.formatPlan(this);
    }

    /**
     * Export this plan as a DOT format string for Graphviz visualization.
     */
    public String toDot() {
        return PlanIntrospection.toDot(this);
    }

    /**
     * Compute segment boundaries from slot data.
     */
    public List<PlanIntrospection.SegmentInfo> getSegments() {
        return PlanIntrospection.getSegments(this);
    }

    /**
     * Find groups of slots that share the same predecessor set and could execute concurrently.
     */
    public List<List<Integer>> getParallelGroups() {
        return PlanIntrospection.getParallelGroups(this);
    }

    /**
     * Get segments with full replay state (requires native plan handle).
     */
    public List<PlanIntrospection.SegmentInfo> getSegmentsWithReplayState(org.bytedeco.javacpp.Pointer nativePlanHandle) {
        if (nativePlanHandle == null) return PlanIntrospection.getSegments(this);
        return PlanIntrospection.getSegmentsWithReplayState(this, nativePlanHandle);
    }

    /**
     * Get compact replay summary string for logging.
     */
    public String getReplaySummary(org.bytedeco.javacpp.Pointer nativePlanHandle) {
        if (nativePlanHandle == null) return "No native plan handle";
        return PlanIntrospection.formatReplaySummary(nativePlanHandle);
    }

    /**
     * Returns the pre-compiled slot descriptors in execution order.
     */
    public DynamicShapeSlot[] getSlots() {
        return slots;
    }

    /**
     * Returns the mapping from requested output variable name to its flat output slot index.
     */
    public Map<String, Integer> getOutputNameToSlotIndex() {
        return outputNameToSlotIndex;
    }

    /**
     * Returns the ordered array of external input variable names
     * (constants, variables, placeholders).
     */
    public String[] getExternalInputKeys() {
        return externalInputKeys;
    }

    /**
     * Returns the total number of flat output slots.
     */
    public int getTotalOutputSlots() {
        return totalOutputSlots;
    }

    /**
     * Returns the set of output variable names this plan was compiled for.
     */
    public Set<String> getRequestedOutputs() {
        return requestedOutputs;
    }

    /**
     * Returns the release schedule.
     */
    public int[][] getReleaseAtStep() {
        return releaseAtStep;
    }

    /**
     * Returns whether this plan contains control flow ops.
     */
    public boolean hasControlFlowOps() {
        return hasControlFlowOps;
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
