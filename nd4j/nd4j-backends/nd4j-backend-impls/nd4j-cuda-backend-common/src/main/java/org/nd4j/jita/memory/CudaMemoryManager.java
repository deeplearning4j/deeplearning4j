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
import org.nd4j.common.config.ND4JSystemProperties;
import org.nd4j.jita.allocator.enums.AllocationStatus;
import org.nd4j.jita.allocator.impl.AllocationPoint;
import org.nd4j.jita.allocator.impl.AtomicAllocator;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.api.buffer.DataBuffer;

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
            System.getProperty(ND4JSystemProperties.CUDA_MEMORY_FALLBACK_ENABLED, "true"));

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
            // Allocate on the current thread's device. Device selection (best GPU by
            // free memory) is handled upstream by OpaqueDataBuffer.allocateDataBuffer().
            val ptr = tryAllocateDevice(bytes);

            if (ptr == null)
                throw new RuntimeException("Failed to allocate " + bytes + " bytes from DEVICE memory");

            if (initialize) {
                val context = AtomicAllocator.getInstance().getDeviceContext();
                int ret = NativeOpsHolder.getInstance().getDeviceNativeOps()
                        .memsetAsync(ptr, 0, bytes, 0, context.getSpecialStream());
                if (ret == 0)
                    throw new ND4JIllegalStateException("memset failed on device_" +
                            Nd4j.getAffinityManager().getDeviceForCurrentThread());
                context.getSpecialStream().synchronize();
            }

            return ptr;
        } else
            throw new RuntimeException("Unknown MemoryKind requested: " + kind);
    }

    /**
     * Attempt to allocate device memory on a specific device without throwing on failure.
     * Uses mallocDevice which routes to the CUDA memory pool for the target device directly —
     * no thread-based device switching is needed.
     *
     * @param bytes number of bytes to allocate
     * @param deviceId target device to allocate on
     * @return pointer if successful, null if allocation failed
     */
    private Pointer tryAllocateDevice(long bytes, int deviceId) {
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
     * Convenience: allocate on the current thread's device.
     */
    private Pointer tryAllocateDevice(long bytes) {
        return tryAllocateDevice(bytes, Nd4j.getAffinityManager().getDeviceForCurrentThread());
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

    /**
     * Get the deallocator service.
     * @return deallocator service instance
     */
    @Override
    public org.nd4j.linalg.api.memory.deallocation.DeallocatorService getDeallocatorService() {
        return org.nd4j.linalg.api.memory.deallocation.DeallocatorService.getInstance();
    }

    /**
     * Calls GC if heap memory is pressured (above threshold).
     * No-op in CUDA memory manager - GC is handled by the allocator.
     */
    @Override
    public void gcIfHeapPressured() {
        // No-op - CUDA memory management doesn't use heap pressure GC
    }

    /**
     * Get the periodic GC frequency.
     * @return frequency value (no-op in CUDA)
     */
    @Override
    public int getFrequency() {
        return 0;
    }

    /**
     * Set the periodic GC frequency.
     * No-op in CUDA memory manager - GC is handled by the allocator.
     * @param frequency the frequency value
     */
    @Override
    public void setFrequency(int frequency) {
        // No-op - CUDA memory management doesn't use periodic GC
    }

    /**
     * Start periodic GC.
     * No-op in CUDA memory manager - GC is handled by the allocator.
     */
    @Override
    public void startPeriodicGc() {
        // No-op - CUDA memory management doesn't use periodic GC
    }

    /**
     * Stop periodic GC.
     * No-op in CUDA memory manager - GC is handled by the allocator.
     */
    @Override
    public void stopPeriodicGc() {
        // No-op - CUDA memory management doesn't use periodic GC
    }
}
