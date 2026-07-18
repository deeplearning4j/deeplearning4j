/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.linalg.vulkan;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.RandomUtils;
import org.bytedeco.javacpp.Pointer;
import org.nd4j.allocator.impl.MemoryTracker;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.memory.AllocationsTracker;
import org.nd4j.linalg.api.memory.Deallocator;
import org.nd4j.linalg.api.memory.abstracts.Nd4jWorkspace;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.AllocationKind;
import org.nd4j.linalg.api.memory.enums.DebugMode;
import org.nd4j.linalg.api.memory.enums.LocationPolicy;
import org.nd4j.linalg.api.memory.enums.MemoryKind;
import org.nd4j.linalg.api.memory.enums.MirroringPolicy;
import org.nd4j.linalg.api.memory.enums.ResetPolicy;
import org.nd4j.linalg.api.memory.enums.SpillPolicy;
import org.nd4j.linalg.api.memory.pointers.PagedPointer;
import org.nd4j.linalg.api.memory.pointers.PointersPair;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import java.util.ArrayList;
import java.util.List;
import java.util.Queue;

import static org.nd4j.linalg.workspace.WorkspaceUtils.getAligned;

/**
 * Vulkan-aware workspace implementation.
 *
 * <p>The lifecycle and spill policies follow {@code CudaWorkspace}. Vulkan
 * device pointers are opaque allocation tokens, however, so byte arithmetic on
 * one large device pointer is invalid. The Vulkan equivalent keeps reusable
 * logical slots whose allocations come from {@code VulkanMemoryPool}; the pool
 * still suballocates real {@code VkDeviceMemory} blocks, while every returned
 * token remains independently registered for transfers, dispatch, and release.</p>
 *
 * <p>The host plane is staging/mirroring memory. Its presence is not an
 * execution fallback: a DEVICE request either returns a Vulkan allocation or
 * fails according to the configured spill policy.</p>
 */
@Slf4j
public class VulkanWorkspace extends Nd4jWorkspace {

    private static final long BASE_VULKAN_WORKSPACE_ID = RandomUtils.nextLong();
    private static final String STANDALONE_WORKSPACE_MANAGER_ID =
            "vulkan:" + Long.toUnsignedString(BASE_VULKAN_WORKSPACE_ID);

    /**
     * Reusable Vulkan allocations in logical workspace order. Each entry owns
     * one pool token and records its capacity in bytes.
     */
    private final List<PointersPair> deviceSlots = new ArrayList<>();
    private int deviceSlotCursor;
    private VulkanWorkspaceManager workspaceManager;

    public VulkanWorkspace(@NonNull WorkspaceConfiguration configuration) {
        this(configuration, DEFAULT_ID, VulkanRuntime.getInstance().currentDevice());
    }

    public VulkanWorkspace(
            @NonNull WorkspaceConfiguration configuration, @NonNull String workspaceId) {
        this(configuration, workspaceId, VulkanRuntime.getInstance().currentDevice());
    }

    public VulkanWorkspace(
            @NonNull WorkspaceConfiguration configuration,
            @NonNull String workspaceId,
            Integer deviceId) {
        super(
                configuration,
                workspaceId,
                VulkanRuntime.getInstance().memoryManager(),
                requireVulkanDevice(deviceId),
                STANDALONE_WORKSPACE_MANAGER_ID);
    }

    private static int requireVulkanDevice(Integer deviceId) {
        if (deviceId == null || deviceId < 0) {
            throw new IllegalArgumentException("Vulkan workspace device id must be non-negative");
        }
        return deviceId;
    }

    void attachWorkspaceManager(@NonNull VulkanWorkspaceManager manager) {
        workspaceManager = manager;
        guid = manager.getUUID();
    }

    private DebugMode workspaceDebugMode() {
        return workspaceManager == null ? DebugMode.DISABLED : workspaceManager.getDebugMode();
    }

    int workspaceDeviceId() {
        if (deviceId < 0) {
            throw new ND4JIllegalStateException("Vulkan workspace has no assigned device");
        }
        return deviceId;
    }

    @Override
    protected void recordHostAllocation(long bytes) {
        // Vulkan host memory is a staging domain, not a compute-device allocation.
    }

    VulkanMemoryManager vulkanMemoryManager() {
        if (!(memoryManager instanceof VulkanMemoryManager)) {
            throw new ND4JIllegalStateException(
                    "VulkanWorkspace requires VulkanMemoryManager, got "
                            + memoryManager.getClass().getName());
        }
        return (VulkanMemoryManager) memoryManager;
    }

    @Override
    protected void init() {
        if (workspaceConfiguration.getPolicyLocation() == LocationPolicy.MMAP) {
            throw new ND4JIllegalStateException("Vulkan does not support MMAP workspaces");
        }

        super.init();
        if (currentSize.get() <= 0) {
            return;
        }

        isInit.set(true);
        long bytes = currentSize.get() + SAFETY_OFFSET;
        Pointer host = memoryManager.allocate(bytes, MemoryKind.HOST, false);
        if (host == null || host.address() == 0L) {
            throw new ND4JIllegalStateException(
                    "Unable to allocate the Vulkan workspace host plane");
        }

        workspace.setHostPointer(new PagedPointer(host));
        // Device slots are allocated lazily. A single opaque Vulkan token cannot
        // be sliced with Pointer.withOffset as a CUDA address can.
    }

    @Override
    public PagedPointer alloc(long requiredMemory, DataType type, boolean initialize) {
        return alloc(requiredMemory, MemoryKind.DEVICE, type, initialize);
    }

    @Override
    public long requiredMemoryPerArray(INDArray array) {
        return getAligned(array.length() * array.dataType().width());
    }

    @Override
    public synchronized PagedPointer alloc(
            long requiredMemory, MemoryKind kind, DataType type, boolean initialize) {
        if (kind == MemoryKind.HOST) {
            return super.alloc(requiredMemory, kind, type, initialize);
        }
        if (kind != MemoryKind.DEVICE) {
            throw new ND4JIllegalStateException("Unknown MemoryKind: " + kind);
        }

        long numElements = requiredMemory / type.width();
        requiredMemory = alignMemory(requiredMemory);
        AllocationsTracker.getInstance()
                .getTracker(id)
                .allocate(type, kind, numElements, requiredMemory);

        if (!isUsed.get()) {
            if (disabledCounter.incrementAndGet() % 10 == 0) {
                log.warn(
                        "Workspace [{}] was disabled and remained disabled for {} allocations",
                        id,
                        disabledCounter.get());
            }
            return allocateExternalDevice(requiredMemory, numElements, initialize, false);
        }

        boolean trimmer =
                (workspaceConfiguration.getPolicyReset() == ResetPolicy.ENDOFBUFFER_REACHED
                                && requiredMemory + cycleAllocations.get()
                                        > initialBlockSize.get()
                                && initialBlockSize.get() > 0)
                        || trimmedMode.get();

        if (trimmer
                && workspaceConfiguration.getPolicySpill() == SpillPolicy.REALLOCATE
                && !trimmedMode.get()) {
            trimmedMode.set(true);
            trimmedStep.set(stepsCount.get());
        }

        if (deviceOffset.get() + requiredMemory <= currentSize.get()
                && !trimmer
                && workspaceDebugMode()
                        != DebugMode.SPILL_EVERYTHING) {
            cycleAllocations.addAndGet(requiredMemory);
            deviceOffset.addAndGet(requiredMemory);

            if (workspaceConfiguration.getPolicyMirroring() == MirroringPolicy.HOST_ONLY) {
                return null;
            }

            return allocateReusableDeviceSlot(requiredMemory, numElements, initialize);
        }

        if (workspaceConfiguration.getPolicyReset() == ResetPolicy.ENDOFBUFFER_REACHED
                && currentSize.get() > 0
                && !trimmer
                && workspaceDebugMode()
                        != DebugMode.SPILL_EVERYTHING) {
            deviceOffset.set(0);
            deviceSlotCursor = 0;
            resetPlanned.set(true);
            return alloc(requiredMemory, kind, type, initialize);
        }

        cycleAllocations.addAndGet(requiredMemory);
        if (workspaceConfiguration.getPolicyMirroring() == MirroringPolicy.HOST_ONLY) {
            return null;
        }

        if (!trimmer) {
            spilledAllocationsSize.addAndGet(requiredMemory);
            AllocationsTracker.getInstance()
                    .getTracker(id)
                    .allocateSpilled(type, kind, numElements, requiredMemory);
        } else {
            pinnedAllocationsSize.addAndGet(requiredMemory);
            AllocationsTracker.getInstance()
                    .getTracker(id)
                    .allocatePinned(type, kind, numElements, requiredMemory);
        }

        switch (workspaceConfiguration.getPolicySpill()) {
            case REALLOCATE:
            case EXTERNAL:
                return allocateExternalDevice(
                        requiredMemory, numElements, initialize, trimmer);
            case FAIL:
            default:
                throw new ND4JIllegalStateException(
                        "Vulkan workspace [" + id + "] is full");
        }
    }

    private PagedPointer allocateReusableDeviceSlot(
            long requiredMemory, long numElements, boolean initialize) {
        int slotIndex = deviceSlotCursor++;
        PointersPair slot;

        if (slotIndex < deviceSlots.size()) {
            slot = deviceSlots.get(slotIndex);
            if (slot.getRequiredMemory() < requiredMemory) {
                releaseTrackedDevice(slot);
                slot = newTrackedDeviceAllocation(requiredMemory, numElements, initialize);
                deviceSlots.set(slotIndex, slot);
            } else if (initialize) {
                vulkanMemoryManager()
                        .initializeDevice(
                                slot.getDevicePointer(),
                                requiredMemory,
                                workspaceDeviceId());
            }
        } else {
            slot = newTrackedDeviceAllocation(requiredMemory, numElements, initialize);
            deviceSlots.add(slot);
        }

        return new PagedPointer(slot.getDevicePointer(), numElements);
    }

    private PagedPointer allocateExternalDevice(
            long requiredMemory,
            long numElements,
            boolean initialize,
            boolean pinned) {
        PointersPair allocation =
                newTrackedDeviceAllocation(requiredMemory, numElements, initialize);
        PagedPointer pointer = allocation.getDevicePointer();
        pointer.setLeaked(true);

        if (pinned) {
            pinnedCount.incrementAndGet();
            allocation.setAllocationCycle(stepsCount.get());
            pinnedAllocations.add(allocation);
        } else {
            externalCount.incrementAndGet();
            externalAllocations.add(allocation);
        }
        return pointer;
    }

    private PointersPair newTrackedDeviceAllocation(
            long requiredMemory, long numElements, boolean initialize) {
        Pointer raw =
                vulkanMemoryManager()
                        .allocateDevice(requiredMemory, initialize, workspaceDeviceId());
        PagedPointer pointer = new PagedPointer(raw, numElements);
        AllocationsTracker.getInstance()
                .markAllocated(
                        AllocationKind.GENERAL, workspaceDeviceId(), requiredMemory);
        MemoryTracker.getInstance()
                .incrementWorkspaceAllocatedAmount(
                        workspaceDeviceId(), requiredMemory);
        return new PointersPair(null, requiredMemory, null, pointer);
    }

    @Override
    public synchronized void destroyWorkspace(boolean extended) {
        currentSize.set(0);
        reset();

        if (extended) {
            clearExternalAllocations();
        }
        clearPinnedAllocations(extended);

        if (workspace.getHostPointer() != null) {
            memoryManager.release(workspace.getHostPointer(), MemoryKind.HOST);
            workspace.setHostPointer(null);
        }

        for (PointersPair slot : deviceSlots) {
            releaseTrackedDevice(slot);
        }
        deviceSlots.clear();
        deviceSlotCursor = 0;
        workspace.setDevicePointer(null);
    }

    @Override
    protected synchronized void clearPinnedAllocations(boolean extended) {
        while (!pinnedAllocations.isEmpty()) {
            PointersPair allocation = pinnedAllocations.peek();
            if (allocation == null) {
                throw new ND4JIllegalStateException(
                        "Null Vulkan pinned workspace allocation");
            }

            long allocationStep =
                    allocation.getAllocationCycle() == null
                            ? 0
                            : allocation.getAllocationCycle();
            if (!extended && allocationStep + 2 >= stepsCount.get()) {
                break;
            }

            pinnedAllocations.remove();
            pinnedCount.decrementAndGet();
            releasePair(allocation, true);
            if (allocation.getRequiredMemory() != null) {
                pinnedAllocationsSize.addAndGet(
                        -allocation.getRequiredMemory());
            }
        }
    }

    @Override
    protected synchronized void clearExternalAllocations() {
        VulkanRuntime.getInstance().executioner().commit();
        for (PointersPair allocation : externalAllocations) {
            releasePair(allocation, false);
        }
        externalAllocations.clear();
        spilledAllocationsSize.set(0);
        externalCount.set(0);
    }

    private void releasePair(PointersPair allocation, boolean pinned) {
        if (allocation == null) {
            return;
        }

        Long bytes = allocation.getRequiredMemory();
        if (allocation.getHostPointer() != null) {
            if (bytes != null) {
                if (pinned) {
                    AllocationsTracker.getInstance()
                            .getTracker(id)
                            .deallocatePinned(MemoryKind.HOST, bytes);
                }
            }
            memoryManager.release(allocation.getHostPointer(), MemoryKind.HOST);
        }

        if (allocation.getDevicePointer() != null) {
            releaseTrackedDevice(allocation);
            if (pinned && bytes != null) {
                AllocationsTracker.getInstance()
                        .getTracker(id)
                        .deallocatePinned(MemoryKind.DEVICE, bytes);
            }
        }
    }

    private void releaseTrackedDevice(PointersPair allocation) {
        if (allocation == null || allocation.getDevicePointer() == null) {
            return;
        }

        long bytes =
                allocation.getRequiredMemory() == null
                        ? 0
                        : allocation.getRequiredMemory();
        if (bytes > 0) {
            AllocationsTracker.getInstance()
                    .markReleased(AllocationKind.GENERAL, workspaceDeviceId(), bytes);
            MemoryTracker.getInstance()
                    .decrementWorkspaceAmount(workspaceDeviceId(), bytes);
        }
        vulkanMemoryManager()
                .releaseDevice(allocation.getDevicePointer(), workspaceDeviceId());
        allocation.setDevicePointer(null);
    }

    @Override
    public synchronized void reset() {
        super.reset();
        deviceSlotCursor = 0;
    }

    @Override
    protected void resetWorkspace() {
        reset();
    }

    PointersPair workspacePointers() {
        return workspace;
    }

    Queue<PointersPair> pinnedPointers() {
        return pinnedAllocations;
    }

    List<PointersPair> externalPointers() {
        return externalAllocations;
    }

    List<PointersPair> reusableDevicePointers() {
        return deviceSlots;
    }

    @Override
    public Deallocator deallocator() {
        return new VulkanWorkspaceDeallocator(this);
    }

    @Override
    public long getUniqueId() {
        return BASE_VULKAN_WORKSPACE_ID
                + VulkanRuntime.getInstance().deallocatorService().nextValue();
    }

    @Override
    public int targetDevice() {
        return workspaceDeviceId();
    }

    @Override
    public long getPrimaryOffset() {
        return getDeviceOffset();
    }
}
