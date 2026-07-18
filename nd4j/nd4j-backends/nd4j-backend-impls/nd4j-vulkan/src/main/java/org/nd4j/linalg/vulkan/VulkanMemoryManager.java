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

package org.nd4j.linalg.vulkan;

import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.Pointer;
import org.nd4j.linalg.api.memory.BasicMemoryManager;
import org.nd4j.linalg.api.memory.AllocationsTracker;
import org.nd4j.linalg.api.memory.enums.AllocationKind;
import org.nd4j.linalg.api.memory.enums.MemoryKind;
import org.nd4j.linalg.api.device.DeviceContext;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.vulkan.bindings.Nd4jVulkan;
import org.nd4j.linalg.vulkan.ops.executioner.VulkanExecutioner;
import org.nd4j.nativeblas.OpaqueDataBuffer;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

/**
 * MemoryManager implementation for the Vulkan compute backend.
 *
 * <p>Mirrors the CudaMemoryManager allocation contract (ADR-0112 §1,
 * ADR-0111 §7): a DEVICE request either returns Vulkan device memory or fails.
 * It never changes the requested memory kind.</p>
 * <ul>
 *   <li><b>DEVICE allocations:</b> routed through
 *       {@code NativeOps.mallocDevice(bytes, deviceId, 0)} which calls into the
 *       VulkanMemoryPool's suballocator (P2 concern; the native layer handles
 *       the pool contract — Java does not duplicate pool logic here).</li>
 *   <li><b>HOST allocations:</b> unchanged — {@code NativeOps.mallocHost}.</li>
 *   <li><b>Safe frees:</b> Java's {@code release()} calls
 *       {@code NativeOps.freeDevice}; the native pool synchronizes the owning
 *       Vulkan device before reclaiming the allocation. DEVICE requests are
 *       never redirected to host memory.</li>
 *   <li><b>memset:</b> arrays use the backend execution path so device storage
 *       is never dereferenced as a host pointer.</li>
 * </ul>
 */
public class VulkanMemoryManager extends BasicMemoryManager {
    private static final int INFER_DEVICE_FROM_ALLOCATION = -1;

    private final Nd4jVulkan nativeOps;
    private final VulkanAffinityManager affinityManager;
    private final VulkanExecutioner executioner;
    private final Map<Integer, AtomicLong> ownedBufferBytes = new ConcurrentHashMap<>();

    public VulkanMemoryManager() {
        this(VulkanRuntime.getInstance().nativeOps(),
                VulkanRuntime.getInstance().affinityManager(),
                VulkanRuntime.getInstance().executioner());
    }

    VulkanMemoryManager(Nd4jVulkan nativeOps, VulkanAffinityManager affinityManager,
                        VulkanExecutioner executioner) {
        if (nativeOps == null || affinityManager == null || executioner == null) {
            throw new IllegalArgumentException("Vulkan memory services must not be null");
        }
        this.nativeOps = nativeOps;
        this.affinityManager = affinityManager;
        this.executioner = executioner;
    }

    /**
     * Returns a Pointer to an allocated memory region.
     * HOST path: pinned host allocation via NativeOps.mallocHost.
     * DEVICE path: Vulkan device allocation via NativeOps.mallocDevice on the
     * current thread's Vulkan device. Allocation failure is reported directly.
     */
    @Override
    public Pointer allocate(long bytes, MemoryKind kind, boolean initialize) {
        if (kind == MemoryKind.HOST) {
            return allocateHost(bytes, initialize);
        }
        if (kind == MemoryKind.DEVICE) {
            return allocateDevice(
                    bytes, initialize, affinityManager.getDeviceForCurrentThread());
        }
        throw new IllegalArgumentException(
                "VulkanMemoryManager: unknown MemoryKind: " + kind);
    }

    private Pointer allocateHost(long bytes, boolean initialize) {
        Pointer pointer = nativeOps.mallocHost(bytes, 0);
        if (pointer == null || pointer.address() == 0L) {
            throw new OutOfMemoryError(
                    "VulkanMemoryManager: failed to allocate " + bytes + " bytes from HOST memory");
        }
        if (initialize) {
            Pointer.memset(pointer, 0, bytes);
        }
        return pointer;
    }

    Pointer allocateDevice(long bytes, boolean initialize, int deviceId) {
        if (deviceId < 0) {
            throw new IllegalArgumentException(
                    "VulkanMemoryManager: invalid Vulkan device id " + deviceId);
        }

        nativeOps.clearLastError();
        Pointer pointer = nativeOps.mallocDevice(bytes, deviceId, 0);
        int errorCode = nativeOps.lastErrorCode();

        if (errorCode != 0 || pointer == null || pointer.address() == 0L) {
            String errorMessage = nativeOps.lastErrorMessage();
            if (pointer != null && pointer.address() != 0L) {
                nativeOps.clearLastError();
                nativeOps.freeDevice(pointer, deviceId);
                pointer.setNull();
            }
            nativeOps.clearLastError();
            throw new OutOfMemoryError(
                    "VulkanMemoryManager: failed to allocate " + bytes
                            + " bytes from DEVICE memory on device " + deviceId
                            + " with native error " + errorCode + ": " + errorMessage);
        }

        if (initialize) {
            try {
                initializeDevice(pointer, bytes, deviceId);
            } catch (RuntimeException initializationFailure) {
                nativeOps.clearLastError();
                nativeOps.freeDevice(pointer, deviceId);
                pointer.setNull();
                nativeOps.clearLastError();
                throw initializationFailure;
            }
        }

        return pointer;
    }

    void initializeDevice(Pointer pointer, long bytes, int deviceId) {
        nativeOps.clearLastError();
        int initialized = nativeOps.memsetSync(pointer, 0, bytes, 0, null);
        int errorCode = nativeOps.lastErrorCode();
        if (initialized != 1 || errorCode != 0) {
            String errorMessage = nativeOps.lastErrorMessage();
            nativeOps.clearLastError();
            throw new IllegalStateException(
                    "VulkanMemoryManager: failed to initialize DEVICE allocation on device "
                            + deviceId + " with native error " + errorCode + ": " + errorMessage);
        }
    }

    /**
     * Releases a previously allocated memory chunk. The native Vulkan pool
     * synchronizes the allocation's owning device before reclaiming it.
     */
    @Override
    public void release(Pointer pointer, MemoryKind kind) {
        if (kind == MemoryKind.DEVICE) {
            releaseDevice(pointer, INFER_DEVICE_FROM_ALLOCATION);
            return;
        }
        if (kind != MemoryKind.HOST) {
            throw new IllegalArgumentException(
                    "VulkanMemoryManager: unknown MemoryKind: " + kind);
        }
        releaseHost(pointer);
    }

    void releaseDevice(Pointer pointer, int deviceId) {
        releaseNative(pointer, MemoryKind.DEVICE, deviceId);
    }

    private void releaseHost(Pointer pointer) {
        releaseNative(pointer, MemoryKind.HOST, -1);
    }

    private void releaseNative(Pointer pointer, MemoryKind kind, int deviceId) {
        if (pointer == null || pointer.address() == 0L) {
            return;
        }

        nativeOps.clearLastError();
        int released = kind == MemoryKind.DEVICE
                ? nativeOps.freeDevice(pointer, deviceId)
                : nativeOps.freeHost(pointer);
        int errorCode = nativeOps.lastErrorCode();
        if (released != 1 || errorCode != 0) {
            String errorMessage = nativeOps.lastErrorMessage();
            nativeOps.clearLastError();
            throw new IllegalStateException(
                    "VulkanMemoryManager: failed to release " + kind
                            + " memory on device " + deviceId
                            + " with native error " + errorCode + ": " + errorMessage);
        }
        pointer.setNull();
    }

    /**
     * Zero-fills an array through the Vulkan execution path. This handles views
     * and device-resident storage without treating a device token as a host pointer.
     */
    @Override
    public void memset(INDArray array) {
        if (array == null || array.isEmpty()) {
            return;
        }
        if (!(array.data() instanceof VulkanDataBuffer)) {
            throw new IllegalArgumentException(
                    "Vulkan memset requires VulkanDataBuffer storage");
        }

        if (array.elementWiseStride() != 1) {
            array.assign(0.0);
            executioner.commit();
            return;
        }

        VulkanDataBuffer buffer = (VulkanDataBuffer) array.data();
        OpaqueDataBuffer opaque = buffer.opaqueBuffer();
        if (opaque.backendOwner().nativeOps() != nativeOps) {
            throw new IllegalArgumentException(
                    "Vulkan memset buffer belongs to a different backend");
        }
        nativeOps.dbAllocateSpecialBuffer(opaque);
        Pointer special = opaque.specialBuffer();
        if (special == null || special.isNull()) {
            throw new IllegalStateException("Vulkan memset target has no device allocation");
        }

        long width = array.dataType().width();
        long byteOffset = Math.multiplyExact(array.offset(), width);
        long bytes = Math.multiplyExact(array.length(), width);
        Pointer target = new BytePointer(special).position(byteOffset);
        nativeOps.clearLastError();
        int status = nativeOps.memsetSync(target, 0, bytes, 0, null);
        int errorCode = nativeOps.lastErrorCode();
        if (status != 1 || errorCode != 0) {
            String message = nativeOps.lastErrorMessage();
            nativeOps.clearLastError();
            throw new IllegalStateException(
                    "Vulkan memset failed with native error " + errorCode + ": " + message);
        }
        buffer.markDeviceDirty();
    }

    // -------------------------------------------------------------------------
    // Collect / purge — delegate to base
    // -------------------------------------------------------------------------

    @Override
    public void collect(INDArray... arrays) {
        super.collect(arrays);
    }

    @Override
    public synchronized void purgeCaches() {
        int deviceCount = affinityManager.getNumberOfDevices();
        if (deviceCount == 0) {
            return;
        }

        executioner.commit();
        for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
            nativeOps.trimMemoryPool(deviceId);
        }
    }

    // -------------------------------------------------------------------------
    // Statistics
    // -------------------------------------------------------------------------

    @Override
    public Map<Integer, Long> getBandwidthUse() {
        return null;
    }

    void recordBufferAllocation(int deviceId, long bytes) {
        if (bytes > 0) {
            ownedBufferBytes.computeIfAbsent(deviceId, ignored -> new AtomicLong()).addAndGet(bytes);
        }
    }

    void recordBufferDeallocation(int deviceId, long bytes) {
        if (bytes <= 0) {
            return;
        }
        AtomicLong allocated = ownedBufferBytes.get(deviceId);
        if (allocated != null) {
            long remaining = allocated.addAndGet(-bytes);
            if (remaining < 0) {
                allocated.addAndGet(bytes);
                throw new IllegalStateException(
                        "Vulkan buffer accounting underflow on device " + deviceId
                                + ": releasing " + bytes + " bytes");
            }
        }
    }

    @Override
    public long allocatedMemory(Integer deviceId) {
        AtomicLong buffers = ownedBufferBytes.get(deviceId);
        return (buffers != null ? buffers.get() : 0L)
                + AllocationsTracker.getInstance().bytesOnDevice(AllocationKind.GENERAL, deviceId)
                + AllocationsTracker.getInstance().bytesOnDevice(AllocationKind.WORKSPACE, deviceId);
    }

    // -------------------------------------------------------------------------
    // Periodic GC hooks — no-op (same as CUDA manager)
    // -------------------------------------------------------------------------

    @Override
    public boolean isPeriodicGcActive() {
        return false;
    }

    @Override
    public void startPeriodicGc() { /* no-op */ }

    @Override
    public void stopPeriodicGc() { /* no-op */ }

    @Override
    public void setFrequency(int frequency) { /* no-op */ }

    @Override
    public int getFrequency() { return 0; }

    @Override
    public void gcIfHeapPressured() { /* no-op */ }

    @Override
    public org.nd4j.linalg.api.memory.deallocation.DeallocatorService getDeallocatorService() {
        return VulkanRuntime.getInstance().deallocatorService();
    }

    @Override
    public void releaseCurrentContext() {
        if (affinityManager.getNumberOfDevices() == 0) {
            return;
        }

        DeviceContext context = affinityManager.contextProvider().getCurrentContext();
        Pointer executionStream = context.getExecutionStream();
        Pointer copyStream = context.getCopyStream();

        synchronizeStream(executionStream, "execution");
        if (copyStream != null && copyStream.address() != executionStream.address()) {
            synchronizeStream(copyStream, "copy");
        }
    }

    private void synchronizeStream(Pointer stream, String streamName) {
        nativeOps.clearLastError();
        int status = nativeOps.streamSynchronize(stream);
        int errorCode = nativeOps.lastErrorCode();
        if (status != 1 || errorCode != 0) {
            String errorMessage = nativeOps.lastErrorMessage();
            nativeOps.clearLastError();
            throw new IllegalStateException(
                    "VulkanMemoryManager: failed to synchronize current " + streamName
                            + " stream with native status " + status
                            + " and error " + errorCode + ": " + errorMessage);
        }
    }
}
