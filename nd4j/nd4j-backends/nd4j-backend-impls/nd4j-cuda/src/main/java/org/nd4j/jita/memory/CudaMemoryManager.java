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

package org.nd4j.jita.memory;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.bytedeco.javacpp.Pointer;
import org.nd4j.jita.allocator.enums.AllocationStatus;
import org.nd4j.jita.allocator.impl.AllocationPoint;
import org.nd4j.jita.allocator.impl.AtomicAllocator;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.device.DeviceMemoryManager;
import org.nd4j.linalg.api.memory.AllocationsTracker;
import org.nd4j.linalg.api.memory.enums.AllocationKind;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.executioner.CpuBackendLoader;
import org.nd4j.linalg.compression.CompressedDataBuffer;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.buffer.BaseCudaDataBuffer;
import org.nd4j.linalg.jcublas.context.CudaContext;
import org.nd4j.linalg.api.memory.BasicMemoryManager;
import org.nd4j.linalg.api.memory.enums.MemoryKind;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * @author raver119@gmail.com
 */
@Slf4j
public class CudaMemoryManager extends BasicMemoryManager {

    /**
     * Tracks allocations that were requested as DEVICE but fell back to HOST due to GPU memory exhaustion.
     * Key: pointer address, Value: true if this was a fallback HOST allocation for a DEVICE request.
     * This is needed to properly route release() calls to freeHost() instead of freeDevice().
     */
    private static final ConcurrentHashMap<Long, Boolean> hostFallbackAllocations = new ConcurrentHashMap<>();

    /**
     * Whether to enable CPU fallback when CUDA allocation fails.
     * Can be disabled via system property: nd4j.cuda.memory.fallback.enabled=false
     */
    private static final boolean CPU_FALLBACK_ENABLED = Boolean.parseBoolean(
            System.getProperty("nd4j.cuda.memory.fallback.enabled", "true"));

    /**
     * This method returns Pointer to allocated memory chunk
     *
     * PLEASE NOTE: Cache options depend on specific implementations
     *
     * @param bytes
     * @param kind
     * @param initialize
     */
    @Override
    public Pointer allocate(long bytes, MemoryKind kind, boolean initialize) {
        val allocator = AtomicAllocator.getInstance();

        //log.info("Allocating {} bytes in {} memory...", bytes, kind);

        if (kind == MemoryKind.HOST) {
            val ptr = NativeOpsHolder.getInstance().getDeviceNativeOps().mallocHost(bytes, 0);

            if (ptr == null)
                throw new RuntimeException("Failed to allocate " + bytes + " bytes from HOST memory");

            if (initialize)
                Pointer.memset(ptr, 0, bytes);

            return ptr;
        } else if (kind == MemoryKind.DEVICE) {
            // Try CUDA device allocation first
            Pointer ptr = tryAllocateDevice(bytes);

            if (ptr != null) {
                // Successful CUDA allocation
                if (initialize) {
                    val context = AtomicAllocator.getInstance().getDeviceContext();
                    int i = NativeOpsHolder.getInstance().getDeviceNativeOps().memsetAsync(ptr, 0, bytes, 0, context.getSpecialStream());
                    if (i == 0)
                        throw new ND4JIllegalStateException("memset failed on device_" + Nd4j.getAffinityManager().getDeviceForCurrentThread());
                    context.getSpecialStream().synchronize();
                }
                return ptr;
            }

            // CUDA allocation failed - attempt recovery
            log.warn("CUDA device allocation failed for {} bytes on device_{}, attempting recovery...",
                    bytes, Nd4j.getAffinityManager().getDeviceForCurrentThread());

            // Step 1: Trigger garbage collection and synchronize
            triggerMemoryReclamation();

            // Step 2: Retry CUDA allocation on same device
            ptr = tryAllocateDevice(bytes);
            if (ptr != null) {
                log.info("CUDA allocation succeeded after memory reclamation for {} bytes", bytes);
                if (initialize) {
                    val context = AtomicAllocator.getInstance().getDeviceContext();
                    int i = NativeOpsHolder.getInstance().getDeviceNativeOps().memsetAsync(ptr, 0, bytes, 0, context.getSpecialStream());
                    if (i == 0)
                        throw new ND4JIllegalStateException("memset failed on device_" + Nd4j.getAffinityManager().getDeviceForCurrentThread());
                    context.getSpecialStream().synchronize();
                }
                return ptr;
            }

            // Step 3: Try other GPUs with available memory before falling back to CPU
            ptr = tryAllocateOnAlternateGpu(bytes, initialize);
            if (ptr != null) {
                return ptr;
            }

            // Step 4: Check if CPU fallback is available and enabled
            if (CPU_FALLBACK_ENABLED && CpuBackendLoader.isCpuBackendAvailable()) {
                log.warn("CUDA allocation still failed after GC. Falling back to HOST memory for {} bytes. " +
                        "CPU backend (nd4j-native) is available for execution.", bytes);

                // Allocate in HOST memory instead
                ptr = allocateHostFallback(bytes, initialize);
                if (ptr != null) {
                    // Track this as a fallback allocation so release() knows to use freeHost
                    hostFallbackAllocations.put(ptr.address(), Boolean.TRUE);
                    log.info("Successfully allocated {} bytes in HOST memory as CUDA fallback", bytes);
                    return ptr;
                }
            }

            // All recovery attempts failed - throw with detailed info
            val deviceId = Nd4j.getAffinityManager().getDeviceForCurrentThread();
            String cpuStatus = CpuBackendLoader.isCpuBackendAvailable() ?
                    "CPU backend available but allocation failed" :
                    "CPU backend NOT available (add nd4j-native to classpath for fallback)";
            throw new RuntimeException(String.format(
                    "CUDA memory allocation failed and recovery unsuccessful. " +
                    "Requested: %d bytes (%.2f MB); Device: %d; %s. " +
                    "Consider: (1) reducing batch size, (2) adding nd4j-native for CPU fallback, " +
                    "(3) freeing unused arrays, or (4) using a GPU with more VRAM.",
                    bytes, bytes / (1024.0 * 1024.0), deviceId, cpuStatus));
        } else
            throw new RuntimeException("Unknown MemoryKind requested: " + kind);
    }

    /**
     * Attempt to allocate device memory without throwing on failure.
     * Uses the current thread's device. If this fails, tryAllocateOnAlternateGpu()
     * handles failover to other GPUs via DeviceMemoryManager.
     * @return pointer if successful, null if allocation failed
     */
    private Pointer tryAllocateDevice(long bytes) {
        int deviceId = Nd4j.getAffinityManager().getDeviceForCurrentThread();
        val ptr = NativeOpsHolder.getInstance().getDeviceNativeOps().mallocDevice(bytes, deviceId, 0);
        log.trace("Attempting allocation of {} bytes for device_{}", bytes, deviceId);

        val ec = NativeOpsHolder.getInstance().getDeviceNativeOps().lastErrorCode();
        if (ec != 0) {
            // Clear the error state
            NativeOpsHolder.getInstance().getDeviceNativeOps().lastErrorMessage();
            return null;
        }

        if (ptr == null || ptr.address() == 0L) {
            return null;
        }

        return ptr;
    }

    /**
     * Try to allocate device memory on an alternate GPU when the current GPU is exhausted.
     * Uses DeviceMemoryManager.selectBestGpu() to find the GPU with the most free memory,
     * then switches to it and retries allocation. If the best GPU fails, tries all remaining
     * GPUs in order of free memory.
     *
     * @param bytes number of bytes to allocate
     * @param initialize whether to zero-initialize the memory
     * @return pointer if successful, null if all GPUs failed
     */
    private Pointer tryAllocateOnAlternateGpu(long bytes, boolean initialize) {
        // Use nativeOps.getAvailableDevices() to get the REAL device count.
        // Nd4j.getAffinityManager().getNumberOfDevices() returns 1 when autoMultiGpuEnabled
        // is false (default), making this failover path dead code on multi-GPU systems.
        int numDevices = NativeOpsHolder.getInstance().getDeviceNativeOps().getAvailableDevices();
        if (numDevices <= 1) {
            return null;
        }

        int currentDevice = Nd4j.getAffinityManager().getDeviceForCurrentThread();

        // Query free memory for all GPUs and find the best one
        int bestDevice = DeviceMemoryManager.getInstance().selectBestGpu();

        if (bestDevice != currentDevice) {
            Pointer ptr = tryAllocateOnDevice(bestDevice, bytes, initialize);
            if (ptr != null) {
                log.info("GPU failover succeeded: allocated {} bytes on device_{} (was device_{})",
                        bytes, bestDevice, currentDevice);
                return ptr;
            }
        }

        // Best GPU failed too - try all remaining GPUs sorted by free memory
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        int[] devices = new int[numDevices];
        long[] freeMemory = new long[numDevices];
        for (int i = 0; i < numDevices; i++) {
            devices[i] = i;
            freeMemory[i] = nativeOps.getDeviceFreeMemory(i);
        }

        // Simple sort by free memory descending
        for (int i = 0; i < numDevices - 1; i++) {
            for (int j = i + 1; j < numDevices; j++) {
                if (freeMemory[j] > freeMemory[i]) {
                    long tmpMem = freeMemory[i]; freeMemory[i] = freeMemory[j]; freeMemory[j] = tmpMem;
                    int tmpDev = devices[i]; devices[i] = devices[j]; devices[j] = tmpDev;
                }
            }
        }

        for (int idx = 0; idx < numDevices; idx++) {
            int d = devices[idx];
            if (d == currentDevice || d == bestDevice) continue;
            if (freeMemory[idx] < bytes) continue;

            Pointer ptr = tryAllocateOnDevice(d, bytes, initialize);
            if (ptr != null) {
                log.info("GPU failover succeeded: allocated {} bytes on device_{} (was device_{})",
                        bytes, d, currentDevice);
                return ptr;
            }
        }

        // All GPUs exhausted - restore original device
        DeviceMemoryManager.getInstance().switchDevice(currentDevice, "CudaMemoryManager", "failover-restore");
        log.warn("GPU failover failed: no alternate GPU could allocate {} bytes", bytes);
        return null;
    }

    /**
     * Switch to the specified device, attempt allocation, and initialize if requested.
     * Returns null on failure without throwing.
     */
    private Pointer tryAllocateOnDevice(int deviceId, long bytes, boolean initialize) {
        try {
            DeviceMemoryManager.getInstance().switchDevice(deviceId, "CudaMemoryManager.tryAllocateOnDevice", "failover-attempt");
            Pointer ptr = tryAllocateDevice(bytes);
            if (ptr != null) {
                if (initialize) {
                    val context = AtomicAllocator.getInstance().getDeviceContext();
                    int i = NativeOpsHolder.getInstance().getDeviceNativeOps()
                            .memsetAsync(ptr, 0, bytes, 0, context.getSpecialStream());
                    if (i == 0) {
                        // memset failed - free the allocation and return null
                        NativeOpsHolder.getInstance().getDeviceNativeOps().freeDevice(ptr, 0);
                        return null;
                    }
                    context.getSpecialStream().synchronize();
                }
                return ptr;
            }
        } catch (Exception e) {
            log.debug("Allocation attempt on device_{} failed: {}", deviceId, e.getMessage());
        }
        return null;
    }

    /**
     * Allocate memory in HOST as a fallback when DEVICE allocation fails.
     * Uses CPU NativeOps if available for proper host allocation.
     */
    private Pointer allocateHostFallback(long bytes, boolean initialize) {
        try {
            // Try using CPU backend's NativeOps for host allocation if available
            NativeOps cpuNativeOps = CpuBackendLoader.loadCpuNativeOps();
            Pointer ptr;

            if (cpuNativeOps != null) {
                // Use CPU backend for host allocation - this ensures proper CPU-side memory management
                ptr = cpuNativeOps.mallocHost(bytes, 0);
            } else {
                // Fall back to CUDA's host allocation (pinned memory)
                ptr = NativeOpsHolder.getInstance().getDeviceNativeOps().mallocHost(bytes, 0);
            }

            if (ptr == null || ptr.address() == 0L) {
                log.error("HOST memory allocation also failed for {} bytes", bytes);
                return null;
            }

            if (initialize) {
                Pointer.memset(ptr, 0, bytes);
            }

            return ptr;
        } catch (Exception e) {
            log.error("Exception during HOST fallback allocation: {}", e.getMessage());
            return null;
        }
    }

    /**
     * Trigger memory reclamation across the system.
     * This synchronizes CUDA operations, runs GC, and gives time for async deallocations.
     */
    private void triggerMemoryReclamation() {
        try {
            // Synchronize all pending CUDA operations
            Nd4j.getExecutioner().commit();

            // Run Java garbage collector
            System.gc();

            // Brief pause for async deallocations to complete
            Thread.sleep(50);

            // Synchronize again after GC
            Nd4j.getExecutioner().commit();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        } catch (Exception e) {
            log.debug("Exception during memory reclamation: {}", e.getMessage());
        }
    }

    /**
     * This method detaches off-heap memory from passed INDArray instances, and optionally stores them in cache for future reuse
     * PLEASE NOTE: Cache options depend on specific implementations
     *
     * @param arrays
     */
    @Override
    public void collect(INDArray... arrays) {
        // we basically want to free memory, without touching INDArray itself.
        // so we don't care when gc is going to release object: memory is already cached

        Nd4j.getExecutioner().commit();

        int cnt = -1;
        AtomicAllocator allocator = AtomicAllocator.getInstance();
        for (INDArray array : arrays) {
            cnt++;
            // we don't collect views, since they don't have their own memory
            if (array == null || array.isView())
                continue;

            AllocationPoint point = allocator.getAllocationPoint(array);

            if (point.getAllocationStatus() == AllocationStatus.HOST)
                allocator.getMemoryHandler().free(point, AllocationStatus.HOST);
            else if (point.getAllocationStatus() == AllocationStatus.DEVICE) {
                allocator.getMemoryHandler().free(point, AllocationStatus.DEVICE);
                allocator.getMemoryHandler().free(point, AllocationStatus.HOST);
            } else if (point.getAllocationStatus() == AllocationStatus.DEALLOCATED) {
                // do nothing
            } else
                throw new RuntimeException(
                                "Unknown AllocationStatus: " + point.getAllocationStatus() + " for argument: " + cnt);

            point.setAllocationStatus(AllocationStatus.DEALLOCATED);
        }
    }

    /**
     * This method purges all cached memory chunks
     * PLEASE NOTE: This method SHOULD NOT EVER BE USED without being 146% clear of all consequences.
     */
    @Override
    public synchronized void purgeCaches() {
        // reset device cache offset
        //        Nd4j.getConstantHandler().purgeConstants();

        // reset TADs
        //        ((CudaGridExecutioner) Nd4j.getExecutioner()).getTadManager().purgeBuffers();

        // purge shapes
        //        Nd4j.getShapeInfoProvider().purgeCache();

        // purge memory cache
        //AtomicAllocator.getInstance().getMemoryHandler().getMemoryProvider().purgeCache();

    }

    protected void allocateHostPointers(DataBuffer... dataBuffers) {
        for (val v:dataBuffers) {
            if (v != null && v instanceof BaseCudaDataBuffer) {
                ((BaseCudaDataBuffer) v).lazyAllocateHostPointer();
            }
        }
    }

    /**
     * This method provides basic memcpy functionality with respect to target environment
     *
     * @param dstBuffer
     * @param srcBuffer
     */
    @Override
    public void memcpy(DataBuffer dstBuffer, DataBuffer srcBuffer) {
        val context = AtomicAllocator.getInstance().getDeviceContext();


        if (dstBuffer instanceof CompressedDataBuffer && !(srcBuffer instanceof CompressedDataBuffer)) {
            // destination is compressed, source isn't
            AllocationPoint srcPoint = AtomicAllocator.getInstance().getAllocationPoint(srcBuffer);

            allocateHostPointers(dstBuffer, srcBuffer);

            long size = srcBuffer.getElementSize() * srcBuffer.length();
            if (!srcPoint.isActualOnHostSide()) {
                // copying device -> host

                AtomicAllocator.getInstance().synchronizeHostData(srcBuffer);

                // Pointer src = AtomicAllocator.getInstance().getPointer(srcBuffer, context);

                // NativeOpsHolder.getInstance().getDeviceNativeOps().memcpyAsync(dstBuffer.addressPointer(), src, size, 2, context.getSpecialStream());
                // context.syncSpecialStream();

            } // else {
              // copying host -> host
            val src = AtomicAllocator.getInstance().getHostPointer(srcBuffer);

            Pointer.memcpy(dstBuffer.addressPointer(), src, size);
            // }

        } else if (!(dstBuffer instanceof CompressedDataBuffer) && srcBuffer instanceof CompressedDataBuffer) {
            allocateHostPointers(dstBuffer, srcBuffer);

            // destination is NOT compressed, source is compressed
            AllocationPoint dstPoint = AtomicAllocator.getInstance().getAllocationPoint(dstBuffer);
            long size = srcBuffer.getElementSize() * srcBuffer.length();

            Pointer.memcpy(dstBuffer.addressPointer(), srcBuffer.addressPointer(), size);
            dstPoint.tickHostWrite();

        } else if (dstBuffer instanceof CompressedDataBuffer && srcBuffer instanceof CompressedDataBuffer) {
            // both buffers are compressed, just fire memcpy

            allocateHostPointers(dstBuffer, srcBuffer);

            Pointer.memcpy(dstBuffer.addressPointer(), srcBuffer.addressPointer(),
                            srcBuffer.length() * srcBuffer.getElementSize());
        } else {
            // both buffers are NOT compressed
            AtomicAllocator.getInstance().memcpy(dstBuffer, srcBuffer);
        }
    }

    /**
     * This method releases previously allocated memory chunk
     *
     * @param pointer
     * @param kind
     * @return
     */
    @Override
    public void release(Pointer pointer, MemoryKind kind) {
        if (pointer == null || pointer.address() == 0L) {
            return;
        }

        if (kind == MemoryKind.DEVICE) {
            // Check if this was a HOST fallback allocation that was tracked
            Boolean wasFallback = hostFallbackAllocations.remove(pointer.address());
            if (wasFallback != null && wasFallback) {
                // This was actually allocated in HOST memory as a CUDA fallback
                log.trace("Releasing HOST fallback allocation at address {}", pointer.address());
                releaseHostFallback(pointer);
            } else {
                // Normal CUDA device allocation
                NativeOpsHolder.getInstance().getDeviceNativeOps().freeDevice(pointer, 0);
            }
            pointer.setNull();
        } else if (kind == MemoryKind.HOST) {
            NativeOpsHolder.getInstance().getDeviceNativeOps().freeHost(pointer);
            pointer.setNull();
        }
    }

    /**
     * Release memory that was allocated in HOST as a CUDA fallback.
     * Uses CPU NativeOps if that was used for allocation, otherwise uses CUDA's freeHost.
     */
    private void releaseHostFallback(Pointer pointer) {
        try {
            NativeOps cpuNativeOps = CpuBackendLoader.getCpuNativeOps();
            if (cpuNativeOps != null) {
                // Free using CPU backend
                cpuNativeOps.freeHost(pointer);
            } else {
                // Free using CUDA's host free
                NativeOpsHolder.getInstance().getDeviceNativeOps().freeHost(pointer);
            }
        } catch (Exception e) {
            log.warn("Exception releasing HOST fallback allocation: {}", e.getMessage());
            // Try the other method as last resort
            try {
                NativeOpsHolder.getInstance().getDeviceNativeOps().freeHost(pointer);
            } catch (Exception e2) {
                log.error("Failed to release HOST fallback allocation: {}", e2.getMessage());
            }
        }
    }

    @Override
    public void setAutoGcWindow(int windowMillis) {
        super.setAutoGcWindow(windowMillis);
        CudaEnvironment.getInstance().getConfiguration().setNoGcWindowMs(windowMillis);
    }

    @Override
    public void memset(INDArray array) {
        if (array.isView()) {
            array.assign(0.0);

            // we don't want any mGRID activations here
            Nd4j.getExecutioner().commit();
            return;
        }

        // we want to be sure we have no trails left in mGRID
        Nd4j.getExecutioner().push();

        AllocationPoint point = AtomicAllocator.getInstance().getAllocationPoint(array);

        if (point.getAllocationStatus() == AllocationStatus.DEVICE) {
            CudaContext context = AtomicAllocator.getInstance().getDeviceContext();
            NativeOpsHolder.getInstance().getDeviceNativeOps().memsetAsync(AtomicAllocator.getInstance().getPointer(array, context),0, array.data().length() * Nd4j.sizeOfDataType(array.data().dataType()),0, context.getOldStream());

            // we also memset host pointer
            Pointer.memset(AtomicAllocator.getInstance().getHostPointer(array), 0, array.data().length() * Nd4j.sizeOfDataType(array.data().dataType()));

            // better be safe then sorry
            context.getOldStream().synchronize();
            point.tickDeviceWrite();
            point.tickHostRead();
        } else if (point.getAllocationStatus() == AllocationStatus.HOST) {
            Nd4j.getExecutioner().commit();

            // just casual memset
            Pointer.memset(AtomicAllocator.getInstance().getHostPointer(array), 0, array.data().length() * Nd4j.sizeOfDataType(array.data().dataType()));
            point.tickHostWrite();
        }
    }

    @Override
    public Map<Integer, Long> getBandwidthUse() {
        return null;
    }

    @Override
    public long allocatedMemory(Integer deviceId) {
        return AllocationsTracker.getInstance().bytesOnDevice(AllocationKind.GENERAL, deviceId) + AllocationsTracker.getInstance().bytesOnDevice(AllocationKind.WORKSPACE, deviceId);
    }

    @Override
    public void releaseCurrentContext() {
        // IMPORTANT: During JVM shutdown, CUDA resources may already be freed by JavaCPP's
        // Deallocator thread. Calling cudaStreamSynchronize on freed stream pointers causes
        // SIGSEGV crashes that cannot be caught by Java exception handlers.
        //
        // Shutdown detection: If this method is called from a shutdown hook thread,
        // we skip synchronization entirely since:
        // 1. The GPU driver will clean up when the process exits
        // 2. Any pending operations will complete or be aborted naturally
        // 3. Native memory may already be freed, making synchronization unsafe
        //
        // Thread name patterns for shutdown contexts:
        // - "SpringApplicationShutdownHook" (Spring Boot)
        // - "ShutdownHook" (generic JVM shutdown hooks)
        // - "DestroyJavaVM" (JVM termination)
        String threadName = Thread.currentThread().getName();
        if (threadName != null && (threadName.contains("Shutdown") ||
                                    threadName.contains("DestroyJavaVM") ||
                                    threadName.contains("shutdown"))) {
            log.trace("Skipping CUDA stream synchronization during shutdown (thread: {})", threadName);
            return;
        }

        // For non-shutdown contexts, synchronize streams to ensure pending operations complete.
        // Use the sync methods that get fresh stream pointers from native code.
        try {
            val context = AtomicAllocator.getInstance().getDeviceContext();
            if (context != null) {
                try {
                    context.syncOldStream();
                } catch (Exception e) {
                    log.trace("Could not sync execution stream during context release: {}", e.getMessage());
                }
                try {
                    context.syncSpecialStream();
                } catch (Exception e) {
                    log.trace("Could not sync special stream during context release: {}", e.getMessage());
                }
            }
        } catch (Exception e) {
            log.trace("Error during context release synchronization: {}", e.getMessage());
        } catch (Error e) {
            // Catch native errors like UnsatisfiedLinkError
            log.trace("Native error during context release: {}", e.getMessage());
        }
    }

    // =========================================================================
    // Fallback Allocation Monitoring
    // =========================================================================

    /**
     * Get the number of active HOST fallback allocations (DEVICE requests that fell back to HOST).
     * This is useful for monitoring memory pressure situations.
     *
     * @return count of active fallback allocations
     */
    public static int getHostFallbackAllocationCount() {
        return hostFallbackAllocations.size();
    }

    /**
     * Check if any HOST fallback allocations are currently active.
     * When this returns true, it indicates the GPU ran out of memory at some point
     * and some allocations are being served from HOST memory.
     *
     * @return true if there are active fallback allocations
     */
    public static boolean hasHostFallbackAllocations() {
        return !hostFallbackAllocations.isEmpty();
    }

    /**
     * Check if CPU fallback is enabled and available.
     * CPU fallback requires: (1) nd4j.cuda.memory.fallback.enabled=true (default)
     * and (2) nd4j-native on the classpath.
     *
     * @return true if CPU fallback can be used when CUDA memory is exhausted
     */
    public static boolean isCpuFallbackAvailable() {
        return CPU_FALLBACK_ENABLED && CpuBackendLoader.isCpuBackendAvailable();
    }

    /**
     * Clear the fallback allocation tracking (for testing purposes).
     * WARNING: Only call this if you know what you're doing - calling this
     * while fallback allocations are still in use will cause memory leaks.
     */
    public static void clearFallbackTracking() {
        hostFallbackAllocations.clear();
    }
}
