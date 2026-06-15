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

package org.nd4j.nativeblas;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.bytedeco.javacpp.Pointer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.device.DeviceDescriptor;
import org.nd4j.linalg.api.device.DeviceMemoryManager;
import org.nd4j.linalg.api.device.DeviceType;
import org.nd4j.linalg.api.memory.deallocation.DeallocatorService;
import org.nd4j.linalg.api.memory.deallocation.OpaqueDataBufferDeallocator;
import org.nd4j.linalg.api.ops.executioner.CpuBackendLoader;
import org.nd4j.linalg.api.ops.executioner.DeviceAwareOpExecutioner;
import org.nd4j.linalg.factory.BackendRegistry;

import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * OpaqueDataBuffer is a JavaCPP wrapper for the native InteropDataBuffer.
 * This class manages lifecycle of native DataBuffer allocations.
 *
 * <p><b>Memory Management:</b> As of this version, OpaqueDataBuffer is integrated 
 * with {@link DeallocatorService} for reliable memory cleanup. Previously relied on 
 * JavaCPP finalizers which were unreliable. Now uses {@link OpaqueDataBufferDeallocator} 
 * for deterministic cleanup.</p>
 *
 * @see DeallocatorService
 * @see OpaqueDataBufferDeallocator
 */
@Slf4j
public class OpaqueDataBuffer extends Pointer {
    private static final int MAX_TRIES = 5;
    private String allocationTrace = null;
    public static AtomicBoolean currentlyExecuting = new AtomicBoolean(false);

    // Track the deallocator for this instance
    private OpaqueDataBufferDeallocator deallocator;

    /**
     * Whether to enable CPU fallback when CUDA/GPU allocation fails.
     * Can be disabled via system property: nd4j.memory.fallback.enabled=false
     */
    private static final boolean CPU_FALLBACK_ENABLED = Boolean.parseBoolean(
            System.getProperty("nd4j.memory.fallback.enabled", "true"));

    /**
     * Tracks buffers that were allocated using CPU NativeOps as a fallback.
     * Key: buffer address, Value: the CPU NativeOps used for allocation (needed for deallocation).
     */
    private static final ConcurrentHashMap<Long, NativeOps> cpuFallbackBuffers = new ConcurrentHashMap<>();

    // Track if buffer has been explicitly closed to prevent double-free
    private volatile boolean explicitlyClosed = false;

    // Atomic flag to coordinate between multiple deallocators (CudaDeallocator and OpaqueDataBufferDeallocator)
    // Only the first deallocator to claim this flag should perform the actual deallocation
    private final AtomicBoolean markedForDeallocation = new AtomicBoolean(false);

    /**
     * Record the current allocation stack trace.
     * This is mainly used when {@link NativeOps#isFuncTrace()}
     * is true. A build of the c++ library has to be generated with the library
     * in order for this to return true.
     *
     * Please do not use this in production. Only use func trace with debug builds.
     */
    public void captureTrace() {
        if(currentlyExecuting.get())
            return;
        
        try {
            currentlyExecuting.set(true);
            allocationTrace = currentTrace();
        } finally {
            // LEAK FIX: Always reset the flag
            currentlyExecuting.set(false);
        }
    }

    public void printNativeAllocationTrace() {
        // Placeholder for native trace printing
    }

    private String currentTrace() {
        return Arrays.toString(Thread.currentThread().getStackTrace()).replace(',', '\n');
    }

    /**
     * Only invoke System.gc() if Java heap usage exceeds 75% of max.
     * GPU OOM is not helped by Java GC — calling System.gc() unconditionally
     * on every native allocation retry causes massive GC pressure (1000+ Full GCs)
     * while providing zero benefit for GPU memory recovery.
     */
    private static void gcIfHeapPressured() {
        Runtime rt = Runtime.getRuntime();
        long used = rt.totalMemory() - rt.freeMemory();
        long max = rt.maxMemory();
        if (used > max * 3 / 4) {
            System.gc();
        }
    }

    /**
     * Constructor for wrapping native pointers.
     * IMPORTANT: This constructor does NOT register with DeallocatorService.
     * Caller is responsible for cleanup via closeBuffer() or the buffer must
     * be registered manually.
     *
     * Consider using factory methods (allocateDataBuffer, externalizedDataBuffer, createView)
     * which handle registration automatically.
     */
    public OpaqueDataBuffer(Pointer p) {
        super(p);
        // WARNING: Not registered with DeallocatorService - caller must manage lifecycle
    }

    /**
     * Internal constructor that optionally registers with DeallocatorService.
     * Use this for buffers that should be automatically cleaned up.
     */
    private OpaqueDataBuffer(Pointer p, boolean autoRegister) {
        super(p);
        if (autoRegister && p != null && !((OpaqueDataBuffer)p).isNull()) {
            try {
                registerWithDeallocatorService(this);
                if(Nd4j.getNativeOps().isFuncTrace()) {
                    captureTrace();
                }
            } catch (Exception e) {
                // Clean up if registration fails
                Nd4j.getNativeOps().dbClose(this);
                throw e;
            }
        }
    }

    public static void tracingSetExecuting(boolean executing) {
        currentlyExecuting.set(executing);
    }

    /**
     * Registers this OpaqueDataBuffer with the DeallocatorService for automatic cleanup.
     *
     * @param buffer The buffer to register
     * @throws RuntimeException if registration fails (buffer must be cleaned up by caller)
     */
    private static void registerWithDeallocatorService(OpaqueDataBuffer buffer) {
        registerWithDeallocatorService(buffer, false, 0L);
    }

    /**
     * Registers this OpaqueDataBuffer with the DeallocatorService for automatic cleanup.
     *
     * This overload allows marking the buffer as constant before the deallocator is registered,
     * preventing the race condition where GC could trigger deallocation between buffer
     * creation and setConstant() being called.
     *
     * @param buffer The buffer to register
     * @param isConstant If true, marks the deallocator as constant immediately to prevent deallocation
     * @throws RuntimeException if registration fails (buffer must be cleaned up by caller)
     */
    private static void registerWithDeallocatorService(OpaqueDataBuffer buffer, boolean isConstant, long allocationBytes) {
        try {
            DeallocatorService service = Nd4j.getDeallocatorService();
            long uniqueId = service.nextValue();
            int targetDevice = Nd4j.getAffinityManager().getDeviceForCurrentThread();

            OpaqueDataBufferDeallocator deallocator = new OpaqueDataBufferDeallocator(
                buffer, uniqueId, targetDevice, allocationBytes
            );

            if (isConstant) {
                deallocator.setConstant(true);
                // Also set on native side immediately - MUST check return value!
                // If this fails, the buffer was already closed (use-after-free race condition)
                boolean nativeSuccess = Nd4j.getNativeOps().dbSetConstant(buffer, true);
                if (!nativeSuccess) {
                    throw new IllegalStateException(
                        "RACE CONDITION DETECTED in registerWithDeallocatorService: Failed to set constant flag on buffer at " +
                        buffer.address() + " because the native buffer was already closed. " +
                        "This indicates a timing issue where the buffer was freed before it could be marked constant. " +
                        "Buffer allocation trace: " + (buffer.allocationTrace != null ? buffer.allocationTrace : "trace not available"));
                }
                buffer.retainReference();
            }

            buffer.deallocator = deallocator;

            // Only register with DeallocatorService if not constant
            // Constants should never be deallocated by GC
            if (!isConstant) {
                service.pickObject(deallocator);
            }

            if (log.isTraceEnabled()) {
                log.trace("Registered OpaqueDataBuffer {} with DeallocatorService, isConstant={}", uniqueId, isConstant);
            }
        } catch (Exception e) {
            // LEAK FIX: If registration fails, caller must clean up the buffer
            log.error("Failed to register OpaqueDataBuffer with DeallocatorService - buffer must be manually cleaned", e);
            throw new RuntimeException("Failed to register buffer with DeallocatorService", e);
        }
    }

    public static OpaqueDataBuffer externalizedDataBuffer(long numElements, @NonNull DataType dataType, Pointer primary, Pointer special) {
        return externalizedDataBuffer(numElements, dataType, primary, special, false);
    }

    /**
     * Creates an externalized data buffer that wraps existing native pointers.
     * The buffer is automatically registered with DeallocatorService for cleanup.
     *
     * This overload allows marking the buffer as constant immediately during registration,
     * before GC can run and deallocate it. This prevents the race condition where:
     * 1. Buffer is created
     * 2. GC runs and frees buffer (because isConstant is false)
     * 3. setConstant(true) is called but fails (buffer already freed)
     *
     * Note: Device assignment must happen before native buffer wrapper creation. Even though
     * this wraps existing pointers, the device assignment is needed to ensure the thread
     * is bound to a device before any native operations.
     *
     * @param numElements Number of elements
     * @param dataType Data type
     * @param primary Primary (host) pointer
     * @param special Special (device) pointer
     * @param isConstant If true, marks as constant immediately to prevent GC deallocation
     * @return Externalized buffer with appropriate constant protection
     */
    public static OpaqueDataBuffer externalizedDataBuffer(long numElements, @NonNull DataType dataType, Pointer primary, Pointer special, boolean isConstant) {
        // Ensure device is assigned for this thread before any native operations.
        Nd4j.getAffinityManager().getDeviceForCurrentThread();

        OpaqueDataBuffer ret;

        if (isConstant) {
            // For constant buffers, use the native function that marks the buffer constant
            // in native code before returning to Java. This eliminates the race condition
            // where GC can finalize the buffer before we call setConstant() on the Java side.
            ret = Nd4j.getNativeOps().dbCreateConstantExternalDataBuffer(numElements, dataType.toInt(), primary, special);

            if (ret != null && !ret.isNull()) {
                // Prevent JavaCPP from attaching a deallocator (extra safety)
                ret.retainReference();

                if(NativeOpsHolder.getInstance().getDeviceNativeOps().isFuncTrace())
                    ret.captureTrace();
            }

            // Register with DeallocatorService - buffer is already constant on native side
            if (ret != null && !ret.isNull()) {
                try {
                    // Pass isConstant=true so Java side knows it's constant,
                    // but dbSetConstant will see it's already constant and succeed
                    registerWithDeallocatorService(ret, true, 0L);
                } catch (Exception e) {
                    // Constant buffers should never fail registration since they're
                    // already protected on native side, but handle just in case
                    log.error("Failed to register constant buffer with DeallocatorService", e);
                    // Don't call dbClose - constant buffers should never be closed
                    throw e;
                }
            }
        } else {
            // Non-constant buffers use the regular path
            long bytes = allocationBytes(numElements, dataType);
            ret = Nd4j.getNativeOps().dbCreateExternalDataBuffer(numElements, dataType.toInt(), primary, special);

            if (ret != null && !ret.isNull()) {
                ret.retainReference();

                if(NativeOpsHolder.getInstance().getDeviceNativeOps().isFuncTrace())
                    ret.captureTrace();

                // Register with DeallocatorService
                try {
                    registerWithDeallocatorService(ret, false, bytes);

                    // Track externalized buffer allocation for accurate memory accounting
                    DeviceDescriptor actualDevice = resolveAllocationDevice(ret);
                    if (actualDevice != null) {
                        DeviceMemoryManager.getInstance().recordAllocation(actualDevice, bytes);
                    }
                } catch (Exception e) {
                    // LEAK FIX: Clean up buffer if registration fails
                    Nd4j.getNativeOps().dbClose(ret);
                    throw e;
                }
            }
        }

        // If ret is null, it means allocation failed - throw an exception with context
        if (ret == null || ret.isNull()) {
            throw new IllegalStateException("Failed to allocate external data buffer with " + numElements + " elements of type " + dataType);
        }

        return ret;
    }

    /**
     * Creates a workspace-backed data buffer that does NOT register with DeallocatorService.
     * The workspace owns the memory lifecycle; the buffer must not outlive the workspace scope.
     *
     * <p>Use this for buffers allocated from workspace memory pools. Since the workspace
     * manages all memory via bump allocation and scope-based recycling, individual buffer
     * deallocation is unnecessary and would cause double-free errors.</p>
     *
     * @param numElements Number of elements
     * @param dataType Data type of the buffer
     * @param primary Primary (host) pointer from workspace, or null
     * @param special Special (device) pointer from workspace, or null
     * @return Buffer wrapping workspace memory, NOT registered with DeallocatorService
     */
    public static OpaqueDataBuffer workspaceDataBuffer(long numElements, @NonNull DataType dataType, Pointer primary, Pointer special) {
        OpaqueDataBuffer ret = Nd4j.getNativeOps().dbCreateExternalDataBuffer(numElements, dataType.toInt(), primary, special);

        if (ret != null && !ret.isNull()) {
            ret.retainReference();

            if (NativeOpsHolder.getInstance().getDeviceNativeOps().isFuncTrace())
                ret.captureTrace();

            // Do NOT register with DeallocatorService - workspace owns this memory
        }

        if (ret == null || ret.isNull()) {
            throw new IllegalStateException("Failed to allocate workspace data buffer with " + numElements + " elements of type " + dataType);
        }

        return ret;
    }

    /**
     * Allocates a new InteropDataBuffer on the GPU with the most free memory.
     * The buffer is automatically registered with DeallocatorService for cleanup.
     *
     * @param numElements Number of elements
     * @param dataType Data type
     * @param allocateBoth Whether to allocate both host and device buffers
     * @return Allocated buffer registered with DeallocatorService
     */
    public static OpaqueDataBuffer allocateDataBuffer(long numElements, @NonNull DataType dataType, boolean allocateBoth) {
        OpaqueDataBuffer buffer = null;
        int ec = 0;
        String em = null;
        long bytes = allocationBytes(numElements, dataType);
        DeviceDescriptor selectedDevice = null;

        DeviceMemoryManager memoryManager = DeviceMemoryManager.getInstance();

        // Select the best GPU based on free memory before allocating.
        // This ensures allocations route to a device that can actually fit them,
        // e.g., after a large model has consumed most memory on the primary GPU.
        // Save and restore CUDA context — switchDevice changes the active device
        // for ALL CUDA calls on this thread, which would corrupt DSP streams/graphs.
        int savedDevice = -1;
        selectedDevice = selectDeviceForAllocation(bytes, memoryManager);
        if (selectedDevice != null && selectedDevice.getDeviceType().isGpu()) {
            int targetDevice = selectedDevice.getDeviceIndex();
            int currentDevice = Nd4j.getAffinityManager().getDeviceForCurrentThread();
            if (targetDevice >= 0 && currentDevice != targetDevice) {
                savedDevice = currentDevice;
                // Auto-install device-aware op executioner on first cross-device allocation
                // so that subsequent ops handle cross-device data correctly
                if (!DeviceAwareOpExecutioner.isInstalled()) {
                    DeviceAwareOpExecutioner.install();
                }
                DeviceMemoryManager.getInstance().switchDevice(targetDevice, "OpaqueDataBuffer", "allocate");
            }
        }

        try {
            for (int t = 0; t < MAX_TRIES; t++) {
                try {
                    // try to allocate data buffer
                    buffer = Nd4j.getNativeOps().allocateDataBuffer(numElements, dataType.toInt(), allocateBoth);

                    // Check if allocation succeeded
                    if(buffer != null && !buffer.isNull()) {
                        buffer.retainReference();

                        // Register with DeallocatorService
                        try {
                            registerWithDeallocatorService(buffer, false, bytes);

                            // Always track allocation in DeviceMemoryManager so device selection
                            // and failover have accurate memory accounting
                            DeviceDescriptor actualDevice = resolveAllocationDevice(buffer);
                            if ((actualDevice == null || actualDevice.getDeviceType() == DeviceType.CPU)
                                    && selectedDevice != null && selectedDevice.getDeviceType().isGpu()) {
                                actualDevice = selectedDevice;
                            }
                            if (actualDevice != null) {
                                memoryManager.recordAllocation(actualDevice, bytes);
                            }

                            // Capture trace if needed
                            if(Nd4j.getNativeOps().isFuncTrace())
                                buffer.captureTrace();

                            // Success - return the buffer
                            return buffer;
                        } catch (Exception regEx) {
                            // LEAK FIX: Clean up buffer if registration fails
                            Nd4j.getNativeOps().dbClose(buffer);
                            throw regEx;
                        }
                    }

                    // check error code
                    ec = Nd4j.getNativeOps().lastErrorCode();
                    if (ec != 0) {
                        em = Nd4j.getNativeOps().lastErrorMessage();

                        // Only invoke GC if Java heap is under pressure — GPU OOM is not helped by Java GC
                        gcIfHeapPressured();

                        // sleeping for 50ms to let any pending async frees complete
                        Thread.sleep(50);
                    } else {
                        // Buffer is null but no error - shouldn't happen, but break to avoid infinite loop
                        break;
                    }
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    throw new RuntimeException("Allocation interrupted", e);
                }
            }

            // if MAX_TRIES is over, we'll just throw an exception
            throw new RuntimeException("Allocation failed: [" + em + "] for amount of memory " + numElements * dataType.width() + " bytes");
        } finally {
            // Restore CUDA context to the original device — leaving it on a different
            // device would corrupt DSP streams, CUDA graphs, and all subsequent ops.
            if (savedDevice >= 0) {
                DeviceMemoryManager.getInstance().switchDevice(savedDevice, "OpaqueDataBuffer", "restore");
            }
        }
    }

    /**
     * Allocates a new InteropDataBuffer on the GPU with the most free memory,
     * and optionally marks it as constant to prevent deallocation before
     * DeallocatorService registration.
     *
     * @param numElements Number of elements
     * @param dataType Data type
     * @param allocateBoth Whether to allocate both host and device buffers
     * @param isConstant If true, marks the buffer as constant immediately to prevent deallocation
     * @return Allocated buffer with appropriate constant protection
     */
    public static OpaqueDataBuffer allocateDataBuffer(long numElements, @NonNull DataType dataType, boolean allocateBoth, boolean isConstant) {
        OpaqueDataBuffer buffer = null;
        int ec = 0;
        String em = null;
        long bytes = allocationBytes(numElements, dataType);
        DeviceDescriptor selectedDevice = null;

        DeviceMemoryManager memoryManager = DeviceMemoryManager.getInstance();

        // Select the best GPU based on free memory before allocating.
        // Save and restore CUDA context around allocation.
        int savedDevice = -1;
        selectedDevice = selectDeviceForAllocation(bytes, memoryManager);
        if (selectedDevice != null && selectedDevice.getDeviceType().isGpu()) {
            int targetDevice = selectedDevice.getDeviceIndex();
            int currentDevice = Nd4j.getAffinityManager().getDeviceForCurrentThread();
            if (targetDevice >= 0 && currentDevice != targetDevice) {
                savedDevice = currentDevice;
                // Auto-install device-aware op executioner on first cross-device allocation
                // so that subsequent ops handle cross-device data correctly
                if (!DeviceAwareOpExecutioner.isInstalled()) {
                    DeviceAwareOpExecutioner.install();
                }
                DeviceMemoryManager.getInstance().switchDevice(targetDevice, "OpaqueDataBuffer", "allocate");
            }
        }

        try {
            for (int t = 0; t < MAX_TRIES; t++) {
                try {
                    // try to allocate data buffer
                    buffer = Nd4j.getNativeOps().allocateDataBuffer(numElements, dataType.toInt(), allocateBoth);

                    // Check if allocation succeeded
                    if(buffer != null && !buffer.isNull()) {
                        buffer.retainReference();

                        // Register with DeallocatorService, marking as constant if requested
                        try {
                            registerWithDeallocatorService(buffer, isConstant, isConstant ? 0L : bytes);

                            // Always track non-constant allocations in DeviceMemoryManager
                            if (!isConstant) {
                                DeviceDescriptor actualDevice = resolveAllocationDevice(buffer);
                                if ((actualDevice == null || actualDevice.getDeviceType() == DeviceType.CPU)
                                        && selectedDevice != null && selectedDevice.getDeviceType().isGpu()) {
                                    actualDevice = selectedDevice;
                                }
                                if (actualDevice != null) {
                                    memoryManager.recordAllocation(actualDevice, bytes);
                                }
                            }

                            // Capture trace if needed
                            if(Nd4j.getNativeOps().isFuncTrace())
                                buffer.captureTrace();

                            // Success - return the buffer
                            return buffer;
                        } catch (Exception regEx) {
                            // LEAK FIX: Clean up buffer if registration fails
                            Nd4j.getNativeOps().dbClose(buffer);
                            throw regEx;
                        }
                    }

                    // check error code
                    ec = Nd4j.getNativeOps().lastErrorCode();
                    if (ec != 0) {
                        em = Nd4j.getNativeOps().lastErrorMessage();

                        // Only invoke GC if Java heap is under pressure — GPU OOM is not helped by Java GC
                        gcIfHeapPressured();

                        // sleeping for 50ms to let any pending async frees complete
                        Thread.sleep(50);
                    } else {
                        // Buffer is null but no error - shouldn't happen, but break to avoid infinite loop
                        break;
                    }
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    throw new RuntimeException("Allocation interrupted", e);
                }
            }

            // if MAX_TRIES is over, we'll just throw an exception
            throw new RuntimeException("Allocation failed: [" + em + "] for amount of memory " + numElements * dataType.width() + " bytes");
        } finally {
            if (savedDevice >= 0) {
                DeviceMemoryManager.getInstance().switchDevice(savedDevice, "OpaqueDataBuffer", "restore");
            }
        }
    }

    /**
     * This method expands buffer, and copies content to the new buffer
     *
     * PLEASE NOTE: if InteropDataBuffer doesn't own actual buffers - original pointers won't be released
     * @param numElements
     */
    public void expand(long numElements) {
        int ec = 0;
        String em = null;

        for (int t = 0; t < MAX_TRIES; t++) {
            try {
                // try to expand the buffer
                Nd4j.getNativeOps().dbExpand(this, numElements);

                // check error code
                ec = Nd4j.getNativeOps().lastErrorCode();
                if (ec == 0) {
                    // Success
                    return;
                }
                
                em = Nd4j.getNativeOps().lastErrorMessage();

                // Only invoke GC if Java heap is under pressure — GPU OOM is not helped by Java GC
                gcIfHeapPressured();

                Thread.sleep(50);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new RuntimeException("Expansion interrupted", e);
            }
        }

        // if MAX_TRIES is over, we'll just throw an exception
        throw new RuntimeException("DataBuffer expansion failed: [" + em + "]");
    }

    /**
     * This method creates a view out of this InteropDataBuffer
     *
     * MEMORY LEAK FIX: Clean up failed view buffers in retry loop
     *
     * @param bytesLength Length in bytes
     * @return View buffer registered with DeallocatorService
     */
    public OpaqueDataBuffer createView(long bytesLength) {
        OpaqueDataBuffer buffer = null;
        int ec = 0;
        String em = null;

        for (int t = 0; t < MAX_TRIES; t++) {
            try {
                buffer = Nd4j.getNativeOps().dbCreateView(this, bytesLength);

                // Check if view creation succeeded
                if(buffer != null && !buffer.isNull()) {
                    // Prevent JavaCPP's NativeDeallocator from running on this buffer.
                    // Without retainReference(), BOTH JavaCPP's DeallocatorThread AND our
                    // DeallocatorService would try to free this buffer, causing a double-free
                    // that corrupts glibc heap metadata (manifests as "corrupted size vs.
                    // prev_size in fastbins" or SIGSEGV in _int_free).
                    // retainReference() does NOT prevent GC - it only disables JavaCPP's
                    // own deallocator. Our DeallocatorService (PhantomReference-based) still works.
                    buffer.retainReference();

                    // Register with DeallocatorService
                    try {
                        registerWithDeallocatorService(buffer);
                        
                        if(NativeOpsHolder.getInstance().getDeviceNativeOps().isFuncTrace())
                            buffer.captureTrace();
                        
                        // Success - return the buffer
                        return buffer;
                    } catch (Exception regEx) {
                        // LEAK FIX: Clean up buffer if registration fails
                        Nd4j.getNativeOps().dbClose(buffer);
                        throw regEx;
                    }
                }
                
                // check error code
                ec = Nd4j.getNativeOps().lastErrorCode();

                if (ec != 0) {
                    em = Nd4j.getNativeOps().lastErrorMessage();

                    // Only invoke GC if Java heap is under pressure — GPU OOM is not helped by Java GC
                    gcIfHeapPressured();

                    // sleeping to let any pending async frees complete
                    Thread.sleep(50);
                } else {
                    // Buffer is null but no error - shouldn't happen, but break to avoid infinite loop
                    break;
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new RuntimeException("View creation interrupted", e);
            }
        }

        // if MAX_TRIES is over, we'll just throw an exception
        throw new RuntimeException("DataBuffer view creation failed: [" + em + "]");
    }

    public long numElements() {
        return Nd4j.getNativeOps().dbBufferLength(this);
    }

    /**
     * This method returns pointer to linear buffer, primary one.
     * @return
     */
    public Pointer primaryBuffer() {
        return Nd4j.getNativeOps().dbPrimaryBuffer(this);
    }

    /**
     * This method returns pointer to special buffer, device one, if any.
     * @return
     */
    public Pointer specialBuffer() {
        return Nd4j.getNativeOps().dbSpecialBuffer(this);
    }

    /**
     * This method returns deviceId of this DataBuffer
     * @return
     */
    public int deviceId() {
        return Nd4j.getNativeOps().dbDeviceId(this);
    }

    /**
     * This method allows to set external pointer as primary buffer.
     *
     * PLEASE NOTE: if InteropDataBuffer owns current memory buffer, it will be released
     * @param ptr
     * @param numElements
     */
    public void setPrimaryBuffer(Pointer ptr, long numElements) {
        //note we call print here because dbSetPrimaryBuffer can deallocate on the c++ side
        printAllocationTraceIfNeeded();
        Nd4j.getNativeOps().dbSetPrimaryBuffer(this, ptr, numElements);
    }

    /**
     * This method allows to set external pointer as special buffer.
     *
     * PLEASE NOTE: if InteropDataBuffer owns current memory buffer, it will be released
     * @param ptr
     * @param numElements
     */
    public void setSpecialBuffer(Pointer ptr, long numElements) {
        //note we call print here because dbSetSpecialBuffer can deallocate on the c++ side
        printAllocationTraceIfNeeded();
        Nd4j.getNativeOps().dbSetSpecialBuffer(this, ptr, numElements);
    }

    /**
     * This method synchronizes device memory
     */
    public void syncToSpecial() {
        Nd4j.getNativeOps().dbSyncToSpecial(this);
    }
    public void migrate() {
        Nd4j.getNativeOps().dbMigrate(this);
    }

    /**
     * This method synchronizes host memory
     */
    public void syncToPrimary() {
        Nd4j.getNativeOps().dbSyncToPrimary(this);
    }

    public void printAllocationTraceIfNeeded() {
        if(allocationTrace != null && Nd4j.getEnvironment().isFuncTracePrintAllocate()) {
            System.out.println("Java side allocation trace: \n " + allocationTrace);
        }
    }

    // Diagnostics: track closeBuffer outcomes
    private static final java.util.concurrent.atomic.AtomicLong dbCloseCallCount = new java.util.concurrent.atomic.AtomicLong();
    private static final java.util.concurrent.atomic.AtomicLong dbCloseSkipShutdown = new java.util.concurrent.atomic.AtomicLong();
    private static final java.util.concurrent.atomic.AtomicLong dbCloseSkipNull = new java.util.concurrent.atomic.AtomicLong();
    private static final java.util.concurrent.atomic.AtomicLong dbCloseSkipExplicit = new java.util.concurrent.atomic.AtomicLong();
    private static final java.util.concurrent.atomic.AtomicLong dbCloseSkipMark = new java.util.concurrent.atomic.AtomicLong();
    private static final java.util.concurrent.atomic.AtomicLong dbCloseSuccess = new java.util.concurrent.atomic.AtomicLong();

    public static String getCloseBufferStats() {
        return String.format("closeBuffer stats: calls=%d, dbClose=%d, skipShutdown=%d, skipNull=%d, skipExplicit=%d, skipMark=%d",
                dbCloseCallCount.get(), dbCloseSuccess.get(), dbCloseSkipShutdown.get(),
                dbCloseSkipNull.get(), dbCloseSkipExplicit.get(), dbCloseSkipMark.get());
    }

    public static void resetCloseBufferStats() {
        dbCloseCallCount.set(0);
        dbCloseSkipShutdown.set(0);
        dbCloseSkipNull.set(0);
        dbCloseSkipExplicit.set(0);
        dbCloseSkipMark.set(0);
        dbCloseSuccess.set(0);
    }

    public void closeBuffer() {
        dbCloseCallCount.incrementAndGet();

        // During JVM shutdown, use GPU-only free (no host free) to avoid SIGABRT
        // from corrupted heap metadata caused by C++ op buffer overruns.
        // freeGpuOnly releases GPU memory via cudaFreeAsync and uses madvise(MADV_DONTNEED)
        // to release host physical pages without calling free().
        if (DeallocatorService.getShutdownInProgress().get()) {
            dbCloseSkipShutdown.incrementAndGet();
            if (!this.isNull() && tryMarkForDeallocation()) {
                try {
                    Nd4j.getNativeOps().dbFreeBuffersOnly(this);
                    this.setNull();
                    if (deallocator != null) {
                        deallocator.markDeallocated();
                    }
                } catch (Throwable t) {
                    // Ignore - JVM is shutting down, OS will reclaim all memory
                }
            }
            return;
        }

        if (this.isNull() || explicitlyClosed) {
            if (this.isNull()) dbCloseSkipNull.incrementAndGet();
            else dbCloseSkipExplicit.incrementAndGet();
            return;
        }

        synchronized (this) {
            if (explicitlyClosed) {
                dbCloseSkipExplicit.incrementAndGet();
                return;
            }
            explicitlyClosed = true;
        }

        // Use tryMarkForDeallocation() directly to coordinate with any GC-based deallocator.
        // This is the atomic flag that prevents double-free. We call dbClose directly here
        // instead of delegating to OpaqueDataBufferDeallocator.deallocate() because:
        // 1. The deallocator may have been marked "deallocated" without actually calling dbClose
        //    (e.g. if buffer.isNull() was true at the time)
        // 2. The DeallocatorService PhantomReference path has a strong reference cycle
        //    (OpaqueDataBufferDeallocator implements both Deallocatable and Deallocator with
        //    deallocator() returning this), so GC-based cleanup never fires
        if (!this.isNull() && tryMarkForDeallocation()) {
            try {
                printAllocationTraceIfNeeded();
                if (Nd4j.getEnvironment().isFuncTracePrintDeallocate()) {
                    System.out.println("Java side deallocation current trace: \n " + currentTrace());
                }
                long allocationBytes = deallocator != null ? deallocator.getAllocationBytes() : 0L;
                DeviceDescriptor allocationDevice = allocationBytes > 0 ? resolveAllocationDevice(this) : null;
                // Second shutdown check: if shutdown started after the initial check,
                // use GPU-only free to avoid host free() discovering corrupted metadata.
                if (DeallocatorService.getShutdownInProgress().get()) {
                    Nd4j.getNativeOps().dbFreeBuffersOnly(this);
                } else {
                    Nd4j.getNativeOps().dbClose(this);
                }
                dbCloseSuccess.incrementAndGet();
                this.setNull();

                if (deallocator != null) {
                    deallocator.markDeallocated();
                }
                if (allocationBytes > 0 && allocationDevice != null) {
                    DeviceMemoryManager.getInstance().recordDeallocation(allocationDevice, allocationBytes);
                }
            } catch (Exception e) {
                log.error("Error in closeBuffer dbClose", e);
            }
        } else {
            dbCloseSkipMark.incrementAndGet();
        }
    }

    /**
     * Atomically attempts to mark this buffer for deallocation.
     * This method ensures that only one deallocator (either CudaDeallocator or
     * OpaqueDataBufferDeallocator) can successfully claim the right to deallocate
     * this buffer, preventing double-free errors.
     *
     * @return true if this call successfully marked the buffer for deallocation,
     *         false if the buffer was already marked (another deallocator claimed it)
     */
    public boolean tryMarkForDeallocation() {
        // Guard against null - can happen if JavaCPP creates instances
        // via Pointer(Pointer) without running field initializers
        if (markedForDeallocation == null) {
            return true; // treat as first claim so caller proceeds with cleanup
        }
        return markedForDeallocation.compareAndSet(false, true);
    }

    /**
     * Checks if this buffer has been marked for deallocation.
     *
     * For JavaCPP-created instances (via native pointer wrapping), the markedForDeallocation
     * field may be null because Java field initializers don't run for JavaCPP-allocated
     * objects. In that case, fall back to checking explicitlyClosed which IS set correctly
     * during closeBuffer().
     *
     * @return true if the buffer has been marked for deallocation
     */
    public boolean isMarkedForDeallocation() {
        // Check explicitlyClosed first — this is correctly set during closeBuffer()
        // even for JavaCPP-created instances where markedForDeallocation may be null.
        if (explicitlyClosed) {
            return true;
        }
        if (markedForDeallocation == null) {
            return false;
        }
        return markedForDeallocation.get();
    }

    /**
     * Gets the deallocator associated with this OpaqueDataBuffer.
     *
     * @return The deallocator or null if not registered
     */
    public OpaqueDataBufferDeallocator getDeallocator() {
        return deallocator;
    }

    /**
     * Marks this buffer as constant (immutable).
     * Constant buffers are never freed by the DeallocatorService because they
     * wrap cached/shared data that has a different lifecycle.
     *
     * This should be called for buffers that wrap cached shape info pointers
     * or other constant data that should not be deallocated when the Java
     * wrapper is garbage collected.
     *
     * @param isConstant true to mark as constant, false otherwise
     */
    public void setConstant(boolean isConstant) {
        if (this.isNull()) {
            return;
        }

        boolean nativeSuccess = Nd4j.getNativeOps().dbSetConstant(this, isConstant);

        if (!nativeSuccess) {
            // The native buffer was already closed - this is a race condition!
            // The buffer was deallocated by GC before we could mark it constant.
            // This typically means the buffer was not protected by registerPendingConstant().
            //
            // We throw an exception here instead of silently continuing because:
            // 1. Silently continuing would cause use-after-free later
            // 2. The caller needs to know their buffer was freed
            // 3. This helps diagnose race conditions in buffer lifecycle management
            throw new IllegalStateException(
                "RACE CONDITION DETECTED: Failed to set constant flag on buffer at " + this.address() +
                " because it was already freed by GC. This indicates a bug in buffer lifecycle management. " +
                "The buffer should have been created with isConstant=true to prevent this race condition.");
        }

        if (isConstant) {
            this.retainReference();
        }

        // Also set the constant flag on the Java-side deallocator.
        // This prevents ND4J's DeallocatorService from trying to deallocate this buffer.
        if (deallocator != null) {
            deallocator.setConstant(isConstant);
        }
    }

    private static long allocationBytes(long numElements, DataType dataType) {
        if (numElements == 0) {
            return dataType.width();
        }
        return numElements * dataType.width();
    }

    private static DeviceDescriptor resolveAllocationDevice(OpaqueDataBuffer buffer) {
        int deviceId;
        try {
            deviceId = buffer.deviceId();
        } catch (Exception e) {
            deviceId = -1;
        }

        BackendRegistry registry = BackendRegistry.getInstance();
        if (deviceId < 0) {
            try {
                int currentDevice = Nd4j.getAffinityManager().getDeviceForCurrentThread();
                if (currentDevice >= 0) {
                    boolean gpuBackend = false;
                    try {
                        String backendName = Nd4j.getBackend().getClass().getSimpleName().toLowerCase();
                        gpuBackend = backendName.contains("cuda") || backendName.contains("jcublas")
                                || backendName.contains("rocm") || backendName.contains("metal");
                    } catch (Exception e) {
                        // ignore and fall back to registry check
                    }
                    if (gpuBackend || registry.hasGpu()) {
                        deviceId = currentDevice;
                    }
                }
            } catch (Exception e) {
                // ignore and fall back to CPU
            }
        }

        if (deviceId >= 0) {
            DeviceDescriptor byId = registry.getDevice(DeviceDescriptor.cuda(deviceId).getDeviceId());
            return byId != null ? byId : DeviceDescriptor.cuda(deviceId);
        }

        DeviceDescriptor cpu = registry.getDefaultCpuDevice();
        return cpu != null ? cpu : DeviceDescriptor.cpu();
    }

    /**
     * Minimum device headroom in bytes (128MB). Allocations will not be placed on a device
     * if doing so would leave less than this amount free. This reserves space for CUDA
     * ContextBuffers (2x8MB workspace + streams), DSP capture workspace (up to 512MB),
     * CudaMemoryPool reserved blocks, and other C++ runtime overhead that bypasses Java routing.
     */
    private static final long MIN_DEVICE_HEADROOM = 128L * 1024 * 1024;

    /**
     * Thread-local flag to suppress cross-device routing during CUDA graph
     * capture/replay. When true, allocations stay on the current device even
     * if memory is low — the C++ layer handles OOM via its own failover.
     * Routing to another device during graph execution would invalidate the
     * captured graph (different device pointers).
     */
    private static final ThreadLocal<Boolean> tl_suppressCrossDeviceRouting = ThreadLocal.withInitial(() -> false);
    // Global suppress flag for async transfers that run on different threads.
    // Set by InferenceSession during executeOperations() to prevent spurious
    // multi-GPU routing caused by CUDA pool holding freed entries.
    private static volatile boolean globalSuppressCrossDeviceRouting = false;

    public static void suppressCrossDeviceRouting(boolean suppress) {
        tl_suppressCrossDeviceRouting.set(suppress);
        globalSuppressCrossDeviceRouting = suppress;
    }

    /**
     * Check if cross-device routing is currently suppressed (either thread-local or global).
     * Used by CudaExecutioner's selectTargetDevice to avoid unnecessary cross-device migration
     * when the CUDA pool has reusable freed entries.
     */
    public static boolean isCrossDeviceRoutingSuppressed() {
        return tl_suppressCrossDeviceRouting.get() || globalSuppressCrossDeviceRouting;
    }

    private static DeviceDescriptor selectDeviceForAllocation(long bytes, DeviceMemoryManager memoryManager) {
        // Always allocate on the current device. The C++ CudaMemoryPool handles OOM
        // correctly: trim pool (releases reserved-but-unused memory back to driver) →
        // retry on same device → failover to other devices only as last resort.
        //
        // Java-level routing based on cudaMemGetInfo is WRONG because it doesn't
        // account for pool-reserved memory that is reclaimable by trim. cudaMemGetInfo
        // reports pool-reserved as "used" even though it's instantly reclaimable,
        // causing false OOM detection and unnecessary cross-device routing that then
        // fails with "invalid argument" on non-peer transfers.
        int currentDeviceId = Nd4j.getAffinityManager().getDeviceForCurrentThread();
        DeviceDescriptor currentDevice = memoryManager.getRegisteredDevice(currentDeviceId);
        return currentDevice != null ? currentDevice :
            memoryManager.selectDevice(bytes, DeviceMemoryManager.DeviceRoutingPolicy.MOST_FREE);
    }

    /**
     * Returns whether this buffer is marked as constant.
     *
     * @return true if constant, false otherwise
     */
    public boolean isConstant() {
        return deallocator != null && deallocator.isConstant();
    }
}
