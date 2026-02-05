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

package org.nd4j.linalg.api.memory.abstracts;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.device.DeviceDescriptor;
import org.nd4j.linalg.api.device.DeviceType;
import org.nd4j.linalg.api.memory.*;
import org.nd4j.linalg.api.memory.conf.DeviceAwareWorkspaceConfiguration;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.MemoryKind;
import org.nd4j.linalg.api.memory.pointers.PagedPointer;
import org.nd4j.linalg.factory.BackendRegistry;
import org.nd4j.linalg.factory.Nd4j;

import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Default implementation of MultiBackendWorkspace.
 * Manages workspace memory across multiple devices with coherence tracking.
 */
@Slf4j
public class DefaultMultiBackendWorkspace implements MultiBackendWorkspace {

    protected final String id;
    protected final DeviceAwareWorkspaceConfiguration configuration;
    protected final DeviceDescriptor primaryDevice;

    // Per-device workspaces
    protected final Map<String, MemoryWorkspace> deviceWorkspaces = new ConcurrentHashMap<>();
    protected final Map<String, AtomicLong> deviceOffsets = new ConcurrentHashMap<>();
    protected final Map<String, AtomicLong> deviceAllocatedSizes = new ConcurrentHashMap<>();
    protected final Map<String, AtomicLong> deviceCycleAllocations = new ConcurrentHashMap<>();

    // Coherence tracking per offset per device
    protected final Map<String, Map<Long, CoherenceState>> deviceCoherenceStates = new ConcurrentHashMap<>();

    // State tracking
    protected final AtomicBoolean scopeActive = new AtomicBoolean(false);
    protected final AtomicLong generationId = new AtomicLong(0);
    protected final AtomicLong cyclesCount = new AtomicLong(0);
    protected volatile boolean mirroringEnabled;

    // Transfer executor
    protected final ExecutorService transferExecutor;

    // Statistics
    protected final AtomicLong totalAllocations = new AtomicLong(0);
    protected final AtomicLong totalTransfers = new AtomicLong(0);
    protected final AtomicLong totalBytesTransferred = new AtomicLong(0);

    // Parent workspace for nesting
    protected volatile MemoryWorkspace parentWorkspace;

    // Thread tracking
    protected final long threadId;

    // Stack traces for debugging
    protected volatile StackTraceElement[] lastEntered;
    protected volatile StackTraceElement[] lastClosed;
    protected volatile StackTraceElement[] lastBorrowed;

    public DefaultMultiBackendWorkspace(@NonNull DeviceAwareWorkspaceConfiguration configuration, @NonNull String id) {
        this.configuration = configuration;
        this.id = id;
        this.threadId = Thread.currentThread().getId();
        this.mirroringEnabled = configuration.isCrossDeviceMirroring();

        // Determine primary device
        this.primaryDevice = selectPrimaryDevice(configuration);

        // Initialize transfer executor
        this.transferExecutor = Executors.newFixedThreadPool(2, r -> {
            Thread t = new Thread(r, "workspace-transfer-" + id);
            t.setDaemon(true);
            return t;
        });

        log.debug("Created MultiBackendWorkspace {} with primary device {}", id, primaryDevice.getDeviceId());
    }

    private DeviceDescriptor selectPrimaryDevice(DeviceAwareWorkspaceConfiguration config) {
        // If explicit device is set, use it
        if (config.getPrimaryDevice() != null) {
            return config.getPrimaryDevice();
        }

        // Otherwise, select based on preferences
        BackendRegistry registry = BackendRegistry.getInstance();
        for (DeviceType preferredType : config.getPreferredDeviceTypes()) {
            List<DeviceDescriptor> devices = registry.getDevices(preferredType);
            if (!devices.isEmpty()) {
                return devices.get(0);
            }
        }

        // Fallback to default CPU
        DeviceDescriptor cpu = registry.getDefaultCpuDevice();
        if (cpu != null) {
            return cpu;
        }

        throw new IllegalStateException("No devices available for workspace");
    }

    @Override
    public DeviceAwareWorkspaceConfiguration getMultiBackendConfiguration() {
        return configuration;
    }

    @Override
    public DeviceDescriptor getPrimaryDevice() {
        return primaryDevice;
    }

    @Override
    public List<DeviceDescriptor> getActiveDevices() {
        List<DeviceDescriptor> devices = new ArrayList<>();
        BackendRegistry registry = BackendRegistry.getInstance();
        for (String deviceId : deviceWorkspaces.keySet()) {
            DeviceDescriptor device = registry.getDevice(deviceId);
            if (device != null) {
                devices.add(device);
            }
        }
        return devices;
    }

    @Override
    public PagedPointer allocOnDevice(long requiredMemory, DataType dataType,
                                       DeviceDescriptor device, boolean initialize) {
        String deviceId = device.getDeviceId();

        // Ensure we have a workspace for this device
        MemoryWorkspace deviceWorkspace = ensureDeviceWorkspace(device);

        // Track allocations
        deviceCycleAllocations.computeIfAbsent(deviceId, k -> new AtomicLong(0))
                .addAndGet(requiredMemory);
        totalAllocations.addAndGet(requiredMemory);

        // Allocate from device workspace
        MemoryKind kind = device.getDeviceType().isGpu() ? MemoryKind.DEVICE : MemoryKind.HOST;
        PagedPointer pointer = deviceWorkspace.alloc(requiredMemory, kind, dataType, initialize);

        // Update offset tracking
        deviceOffsets.computeIfAbsent(deviceId, k -> new AtomicLong(0))
                .addAndGet(requiredMemory);

        // Initialize coherence state
        long offset = deviceOffsets.get(deviceId).get() - requiredMemory;
        setCoherenceState(device, offset, CoherenceState.EXCLUSIVE);

        // If mirroring is enabled, mirror to other devices
        if (mirroringEnabled && configuration.getMirrorDevices() != null) {
            for (DeviceDescriptor mirrorDevice : configuration.getMirrorDevices()) {
                if (!mirrorDevice.equals(device)) {
                    mirrorAllocation(device, mirrorDevice, offset, requiredMemory);
                }
            }
        }

        return pointer;
    }

    private void mirrorAllocation(DeviceDescriptor source, DeviceDescriptor target,
                                   long offset, long size) {
        try {
            // Allocate on mirror device
            MemoryWorkspace targetWorkspace = ensureDeviceWorkspace(target);
            MemoryKind kind = target.getDeviceType().isGpu() ? MemoryKind.DEVICE : MemoryKind.HOST;
            targetWorkspace.alloc(size, kind, DataType.BYTE, false);

            // Mark as shared
            setCoherenceState(source, offset, CoherenceState.SHARED);
            setCoherenceState(target, offset, CoherenceState.SHARED);

            totalTransfers.incrementAndGet();
        } catch (Exception e) {
            log.warn("Failed to mirror allocation to device {}", target.getDeviceId(), e);
        }
    }

    private MemoryWorkspace ensureDeviceWorkspace(DeviceDescriptor device) {
        String deviceId = device.getDeviceId();

        return deviceWorkspaces.computeIfAbsent(deviceId, k -> {
            // Create workspace configuration for this device
            WorkspaceConfiguration deviceConfig = WorkspaceConfiguration.builder()
                    .initialSize(configuration.getInitialSizeForDevice(device.getDeviceType()))
                    .maxSize(configuration.getMaxSizeForDevice(device.getDeviceType()))
                    .policyAllocation(configuration.getPolicyAllocation())
                    .policySpill(configuration.getPolicySpill())
                    .policyReset(configuration.getPolicyReset())
                    .policyLearning(configuration.getPolicyLearning())
                    .overallocationLimit(configuration.getOverallocationLimit())
                    .build();

            // Get workspace from backend
            MemoryWorkspaceManager manager = Nd4j.getWorkspaceManager();
            String deviceWorkspaceId = id + "_" + deviceId;
            return manager.createNewWorkspace(deviceConfig, deviceWorkspaceId, device.getDeviceIndex());
        });
    }

    @Override
    public long getOffsetForDevice(DeviceDescriptor device) {
        AtomicLong offset = deviceOffsets.get(device.getDeviceId());
        return offset != null ? offset.get() : 0;
    }

    @Override
    public long getAllocatedSizeForDevice(DeviceDescriptor device) {
        MemoryWorkspace workspace = deviceWorkspaces.get(device.getDeviceId());
        return workspace != null ? workspace.getCurrentSize() : 0;
    }

    @Override
    public long getAvailableSpaceForDevice(DeviceDescriptor device) {
        long allocated = getAllocatedSizeForDevice(device);
        long maxSize = configuration.getMaxSizeForDevice(device.getDeviceType());
        long offset = getOffsetForDevice(device);
        return Math.max(0, allocated - offset);
    }

    @Override
    public Map<DeviceDescriptor, Long> getPerDeviceCycleAllocations() {
        Map<DeviceDescriptor, Long> result = new HashMap<>();
        BackendRegistry registry = BackendRegistry.getInstance();
        for (Map.Entry<String, AtomicLong> entry : deviceCycleAllocations.entrySet()) {
            DeviceDescriptor device = registry.getDevice(entry.getKey());
            if (device != null) {
                result.put(device, entry.getValue().get());
            }
        }
        return result;
    }

    @Override
    public Map<DeviceDescriptor, Long> getPerDeviceTotalAllocations() {
        Map<DeviceDescriptor, Long> result = new HashMap<>();
        BackendRegistry registry = BackendRegistry.getInstance();
        for (Map.Entry<String, MemoryWorkspace> entry : deviceWorkspaces.entrySet()) {
            DeviceDescriptor device = registry.getDevice(entry.getKey());
            if (device != null) {
                result.put(device, entry.getValue().getThisCycleAllocations());
            }
        }
        return result;
    }

    @Override
    public void transferTo(DeviceDescriptor sourceDevice, DeviceDescriptor targetDevice) {
        throw new UnsupportedOperationException(
                "DefaultMultiBackendWorkspace does not implement data movement; " +
                "use buffer-level transfers or NativeMultiBackendWorkspace");
    }

    @Override
    public Future<Void> transferToAsync(DeviceDescriptor sourceDevice, DeviceDescriptor targetDevice) {
        throw new UnsupportedOperationException(
                "DefaultMultiBackendWorkspace does not implement async data movement; " +
                "use buffer-level transfers or NativeMultiBackendWorkspace");
    }

    @Override
    public void prefetchTo(DeviceDescriptor device) {
        if (configuration.isAsyncTransfers()) {
            transferToAsync(primaryDevice, device);
        } else {
            transferTo(primaryDevice, device);
        }
    }

    @Override
    public void syncDevice(DeviceDescriptor device) {
        MemoryWorkspace workspace = deviceWorkspaces.get(device.getDeviceId());
        if (workspace != null) {
            // Trigger sync through commit
            Nd4j.getExecutioner().commit();
        }
    }

    @Override
    public void syncAllDevices() {
        Nd4j.getExecutioner().commit();
    }

    @Override
    public CoherenceState getCoherenceState(DeviceDescriptor device, long offset) {
        Map<Long, CoherenceState> states = deviceCoherenceStates.get(device.getDeviceId());
        if (states == null) {
            return CoherenceState.INVALID;
        }
        return states.getOrDefault(offset, CoherenceState.INVALID);
    }

    @Override
    public void setCoherenceState(DeviceDescriptor device, long offset, CoherenceState state) {
        deviceCoherenceStates.computeIfAbsent(device.getDeviceId(), k -> new ConcurrentHashMap<>())
                .put(offset, state);
    }

    @Override
    public void invalidateDevice(DeviceDescriptor device) {
        Map<Long, CoherenceState> states = deviceCoherenceStates.get(device.getDeviceId());
        if (states != null) {
            for (Long offset : states.keySet()) {
                states.put(offset, CoherenceState.INVALID);
            }
        }
    }

    @Override
    public void invalidateAllExcept(DeviceDescriptor exceptDevice) {
        for (String deviceId : deviceCoherenceStates.keySet()) {
            if (!deviceId.equals(exceptDevice.getDeviceId())) {
                Map<Long, CoherenceState> states = deviceCoherenceStates.get(deviceId);
                if (states != null) {
                    for (Long offset : states.keySet()) {
                        states.put(offset, CoherenceState.INVALID);
                    }
                }
            }
        }
    }

    @Override
    public boolean hasMemoryOnDevice(DeviceDescriptor device) {
        return deviceWorkspaces.containsKey(device.getDeviceId());
    }

    @Override
    public void extendOnDevice(DeviceDescriptor device, long additionalBytes) {
        // This would need to trigger a reallocation in the underlying workspace
        log.warn("Workspace extension not fully implemented for device {}", device.getDeviceId());
    }

    @Override
    public void releaseOnDevice(DeviceDescriptor device) {
        String deviceId = device.getDeviceId();
        MemoryWorkspace workspace = deviceWorkspaces.remove(deviceId);
        if (workspace != null) {
            workspace.destroyWorkspace();
        }
        deviceOffsets.remove(deviceId);
        deviceCycleAllocations.remove(deviceId);
        deviceCoherenceStates.remove(deviceId);
    }

    @Override
    public MemoryWorkspace getDeviceWorkspace(DeviceDescriptor device) {
        return deviceWorkspaces.get(device.getDeviceId());
    }

    @Override
    public boolean isMirroringEnabled() {
        return mirroringEnabled;
    }

    @Override
    public void setMirroringEnabled(boolean enabled) {
        this.mirroringEnabled = enabled;
    }

    @Override
    public Map<String, Object> getStatistics() {
        Map<String, Object> stats = new LinkedHashMap<>();
        stats.put("id", id);
        stats.put("primaryDevice", primaryDevice.getDeviceId());
        stats.put("activeDevices", deviceWorkspaces.size());
        stats.put("totalAllocations", totalAllocations.get());
        stats.put("totalTransfers", totalTransfers.get());
        stats.put("totalBytesTransferred", totalBytesTransferred.get());
        stats.put("cyclesCount", cyclesCount.get());
        stats.put("mirroringEnabled", mirroringEnabled);
        return stats;
    }

    @Override
    public void resetStatistics() {
        totalAllocations.set(0);
        totalTransfers.set(0);
        totalBytesTransferred.set(0);
    }

    @Override
    public void compact() {
        for (DeviceDescriptor device : getActiveDevices()) {
            compactOnDevice(device);
        }
    }

    @Override
    public void compactOnDevice(DeviceDescriptor device) {
        // Compaction is typically handled by the underlying workspace implementation
        log.debug("Compact requested for device {}", device.getDeviceId());
    }

    // ================== MemoryWorkspace Interface Implementation ==================

    @Override
    public String getId() {
        return id;
    }

    @Override
    public int getDeviceId() {
        return primaryDevice.getDeviceIndex();
    }

    @Override
    public Long getThreadId() {
        return threadId;
    }

    @Override
    public long getGenerationId() {
        return generationId.get();
    }

    @Override
    public Type getWorkspaceType() {
        return Type.SCOPED;
    }

    @Override
    public MemoryWorkspace getParentWorkspace() {
        return parentWorkspace;
    }

    @Override
    public PagedPointer alloc(long requiredMemory, DataType dataType, boolean initialize) {
        return allocOnDevice(requiredMemory, dataType, primaryDevice, initialize);
    }

    @Override
    public PagedPointer alloc(long requiredMemory, MemoryKind kind, DataType dataType, boolean initialize) {
        DeviceDescriptor targetDevice = primaryDevice;
        if (kind == MemoryKind.DEVICE && !primaryDevice.getDeviceType().isGpu()) {
            // Try to find a GPU
            DeviceDescriptor gpu = BackendRegistry.getInstance().getFirstGpu();
            if (gpu != null) {
                targetDevice = gpu;
            }
        }
        return allocOnDevice(requiredMemory, dataType, targetDevice, initialize);
    }

    @Override
    public MemoryWorkspace notifyScopeEntered() {
        scopeActive.set(true);
        generationId.incrementAndGet();
        lastEntered = Thread.currentThread().getStackTrace();

        // Enter scope on all device workspaces
        for (MemoryWorkspace workspace : deviceWorkspaces.values()) {
            workspace.notifyScopeEntered();
        }

        return this;
    }

    @Override
    public MemoryWorkspace notifyScopeBorrowed() {
        scopeActive.set(true);
        lastBorrowed = Thread.currentThread().getStackTrace();
        return this;
    }

    @Override
    public MemoryWorkspace notifyScopeLeft() {
        scopeActive.set(false);
        cyclesCount.incrementAndGet();
        lastClosed = Thread.currentThread().getStackTrace();

        // Reset cycle allocations
        for (AtomicLong allocation : deviceCycleAllocations.values()) {
            allocation.set(0);
        }

        // Leave scope on all device workspaces
        for (MemoryWorkspace workspace : deviceWorkspaces.values()) {
            workspace.notifyScopeLeft();
        }

        // Reset offsets based on policy
        if (configuration.getPolicyReset() == org.nd4j.linalg.api.memory.enums.ResetPolicy.BLOCK_LEFT) {
            for (AtomicLong offset : deviceOffsets.values()) {
                offset.set(0);
            }
        }

        return this;
    }

    @Override
    public void initializeWorkspace() {
        ensureDeviceWorkspace(primaryDevice);
    }

    @Override
    public void destroyWorkspace() {
        destroyWorkspace(false);
    }

    @Override
    public void destroyWorkspace(boolean extended) {
        // Shutdown transfer executor
        transferExecutor.shutdown();
        try {
            transferExecutor.awaitTermination(5, TimeUnit.SECONDS);
        } catch (InterruptedException e) {
            transferExecutor.shutdownNow();
        }

        // Destroy all device workspaces
        for (MemoryWorkspace workspace : deviceWorkspaces.values()) {
            workspace.destroyWorkspace(extended);
        }
        deviceWorkspaces.clear();
        deviceOffsets.clear();
        deviceCycleAllocations.clear();
        deviceCoherenceStates.clear();
    }

    @Override
    public void close() {
        notifyScopeLeft();
    }

    @Override
    public boolean isScopeActive() {
        return scopeActive.get();
    }

    @Override
    public long getThisCycleAllocations() {
        long total = 0;
        for (AtomicLong allocation : deviceCycleAllocations.values()) {
            total += allocation.get();
        }
        return total;
    }

    @Override
    public long getLastCycleAllocations() {
        // Return from primary device workspace
        MemoryWorkspace primary = deviceWorkspaces.get(primaryDevice.getDeviceId());
        return primary != null ? primary.getLastCycleAllocations() : 0;
    }

    @Override
    public long getMaxCycleAllocations() {
        MemoryWorkspace primary = deviceWorkspaces.get(primaryDevice.getDeviceId());
        return primary != null ? primary.getMaxCycleAllocations() : 0;
    }

    @Override
    public long getCurrentSize() {
        long total = 0;
        for (MemoryWorkspace workspace : deviceWorkspaces.values()) {
            total += workspace.getCurrentSize();
        }
        return total;
    }

    @Override
    public long getCurrentOffset() {
        return getOffsetForDevice(primaryDevice);
    }

    @Override
    public long getPrimaryOffset() {
        return getCurrentOffset();
    }

    @Override
    public WorkspaceConfiguration getWorkspaceConfiguration() {
        return configuration;
    }

    @Override
    public void toggleWorkspaceUse(boolean isEnabled) {
        for (MemoryWorkspace workspace : deviceWorkspaces.values()) {
            workspace.toggleWorkspaceUse(isEnabled);
        }
    }

    @Override
    public void enableDebug(boolean reallyEnable) {
        for (MemoryWorkspace workspace : deviceWorkspaces.values()) {
            workspace.enableDebug(reallyEnable);
        }
    }

    @Override
    public StackTraceElement[] lastEntered() {
        return lastEntered;
    }

    @Override
    public StackTraceElement[] lastClosed() {
        return lastClosed;
    }

    @Override
    public StackTraceElement[] lastBorrowed() {
        return lastBorrowed;
    }

    @Override
    public void setPreviousWorkspace(MemoryWorkspace memoryWorkspace) {
        this.parentWorkspace = memoryWorkspace;
    }

    @Override
    public MemoryWorkspace tagOutOfScopeUse() {
        return this;
    }

    @Override
    public Enum getAssociatedEnumType() {
        return null;
    }

    @Override
    public void setAssociatedEnumType(Enum associatedEnumType) {
        // Not used in multi-backend workspace
    }

    @Override
    public void setWorkspaceMgr(org.nd4j.linalg.workspace.WorkspaceMgr mgr) {
        // Not used in multi-backend workspace
    }

    @Override
    public int targetDevice() {
        return primaryDevice != null ? primaryDevice.getDeviceIndex() : 0;
    }

    @Override
    public Deallocator deallocator() {
        return null; // Multi-backend workspace does not use Deallocator pattern
    }

    @Override
    public long getUniqueId() {
        return generationId.get();
    }
}
