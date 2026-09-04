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
import org.nd4j.linalg.api.concurrency.AffinityManager;
import org.nd4j.linalg.api.device.DeviceDescriptor;
import org.nd4j.linalg.api.device.DeviceMemoryManager;
import org.nd4j.linalg.api.device.DeviceType;
import org.nd4j.linalg.api.memory.deallocation.DeallocatorService;
import org.nd4j.linalg.api.memory.deallocation.OpaqueDataBufferDeallocator;
import org.nd4j.linalg.api.ops.executioner.DeviceAwareOpExecutioner;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;
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

    // Track the deallocator and exact backend owner for this instance.
    private OpaqueDataBufferDeallocator deallocator;
    private NativeBufferOwner backendOwner;
    private DeviceDescriptor allocationDevice;

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
        // Native factories attach their exact owner before first use. Legacy callers
        // that wrap a raw pointer resolve the process-primary owner lazily.
        // WARNING: Not registered with DeallocatorService - caller must manage lifecycle.
    }

    /**
     * Internal constructor that optionally registers with DeallocatorService.
     * Use this for buffers that should be automatically cleaned up.
     */
    private OpaqueDataBuffer(Pointer p, boolean autoRegister) {
        super(p);
        NativeBufferOwner owner = primaryBackendOwner();
        attachOwner(owner, null);
        if (autoRegister && p != null && !((OpaqueDataBuffer)p).isNull()) {
            try {
                allocationDevice = resolveAllocationDevice(owner, this);
                registerWithDeallocatorService(this);
                if(owner.nativeOps().isFuncTrace()) {
                    captureTrace();
                }
            } catch (Exception e) {
                // Clean up if registration fails
                owner.nativeOps().dbClose(this);
                throw e;
            }
        }
    }

    /**
     * Attaches the backend authority that created this buffer.
     *
     * <p>This is intentionally explicit: a missing owner is an error, never a
     * request to consult the process-primary backend.</p>
     */
    public OpaqueDataBuffer attachOwner(@NonNull NativeBufferOwner owner, DeviceDescriptor device) {
        if (backendOwner != null && backendOwner != owner) {
            if (deallocator != null || allocationDevice != null) {
                throw new IllegalStateException("OpaqueDataBuffer already belongs to a different backend");
            }
            // JavaCPP may invoke the public pointer constructor before a native factory
            // can attach the backend that actually produced the pointer. Replacement is
            // legal only while the wrapper has no established allocation lifecycle.
            allocationDevice = null;
        }
        backendOwner = owner;
        if (device != null || allocationDevice == null) {
            allocationDevice = device;
        }
        return this;
    }

    public NativeBufferOwner backendOwner() {
        return requireBackendOwner();
    }

    public DeviceDescriptor allocationDevice() {
        return allocationDevice;
    }

    private NativeBufferOwner requireBackendOwner() {
        NativeBufferOwner owner = backendOwner;
        if (owner == null) {
            synchronized (this) {
                owner = backendOwner;
                if (owner == null) {
                    owner = primaryBackendOwner();
                    backendOwner = owner;
                }
            }
        }
        return owner;
    }

    private NativeOps nativeOps() {
        return requireBackendOwner().nativeOps();
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
            NativeBufferOwner owner = buffer.requireBackendOwner();
            DeallocatorService service = owner.deallocatorService();
            long uniqueId = service.nextValue();
            DeviceDescriptor allocationDevice = buffer.allocationDevice;
            int targetDevice = allocationDevice != null && allocationDevice.getDeviceType() != DeviceType.CPU
                    ? allocationDevice.getDeviceIndex()
                    : owner.currentDevice();

            OpaqueDataBufferDeallocator deallocator = new OpaqueDataBufferDeallocator(
                    buffer, uniqueId, targetDevice, allocationBytes, owner, allocationDevice);

            if (isConstant) {
                deallocator.setConstant(true);
                // Also set on native side immediately - MUST check return value!
                // If this fails, the buffer was already closed (use-after-free race condition)
                boolean nativeSuccess = owner.nativeOps().dbSetConstant(buffer, true);
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
        return externalizedDataBuffer(numElements, dataType, primary, special, false, primaryBackendOwner());
    }

    public static OpaqueDataBuffer externalizedDataBuffer(long numElements, @NonNull DataType dataType,
                                                          Pointer primary, Pointer special,
                                                          @NonNull NativeBufferOwner owner) {
        return externalizedDataBuffer(numElements, dataType, primary, special, false, owner);
    }

    /**
     * Creates an externalized data buffer that wraps existing native pointers.
     * The buffer is automatically registered with DeallocatorService for cleanup.
     *
     * @param numElements Number of elements
     * @param dataType Data type
     * @param primary Primary (host) pointer
     * @param special Special (device) pointer
     * @param isConstant If true, marks as constant immediately to prevent GC deallocation
     * @return Externalized buffer with appropriate constant protection
     */
    public static OpaqueDataBuffer externalizedDataBuffer(long numElements, @NonNull DataType dataType,
                                                          Pointer primary, Pointer special,
                                                          boolean isConstant) {
        return externalizedDataBuffer(
                numElements, dataType, primary, special, isConstant, primaryBackendOwner());
    }

    public static OpaqueDataBuffer externalizedDataBuffer(long numElements, @NonNull DataType dataType,
                                                          Pointer primary, Pointer special,
                                                          boolean isConstant,
                                                          @NonNull NativeBufferOwner owner) {
        owner.currentDevice();
        NativeOps ops = owner.nativeOps();
        OpaqueDataBuffer ret = isConstant
                ? ops.dbCreateConstantExternalDataBuffer(numElements, dataType.toInt(), primary, special)
                : ops.dbCreateExternalDataBuffer(numElements, dataType.toInt(), primary, special);

        if (ret == null || ret.isNull()) {
            throw new IllegalStateException("Failed to allocate external data buffer with "
                    + numElements + " elements of type " + dataType);
        }

        ret.attachOwner(owner, resolveAllocationDevice(owner, ret));
        ret.retainReference();
        if (ops.isFuncTrace()) {
            ret.captureTrace();
        }

        long bytes = isConstant ? 0L : allocationBytes(numElements, dataType);
        try {
            registerWithDeallocatorService(ret, isConstant, bytes);
            if (!isConstant && ret.allocationDevice != null) {
                owner.recordAllocation(ret.allocationDevice, bytes);
            }
        } catch (Exception e) {
            if (!isConstant) {
                ops.dbClose(ret);
            }
            throw e;
        }

        return ret;
    }

    /**
     * Creates a workspace-backed data buffer that does NOT register with DeallocatorService.
     * The workspace owns the memory lifecycle; the buffer must not outlive the workspace scope.
     */
    public static OpaqueDataBuffer workspaceDataBuffer(long numElements, @NonNull DataType dataType,
                                                       Pointer primary, Pointer special) {
        return workspaceDataBuffer(
                numElements, dataType, primary, special, primaryBackendOwner());
    }

    public static OpaqueDataBuffer workspaceDataBuffer(long numElements, @NonNull DataType dataType,
                                                       Pointer primary, Pointer special,
                                                       @NonNull NativeBufferOwner owner) {
        owner.currentDevice();
        NativeOps ops = owner.nativeOps();
        OpaqueDataBuffer ret =
                ops.dbCreateExternalDataBuffer(numElements, dataType.toInt(), primary, special);

        if (ret == null || ret.isNull()) {
            throw new IllegalStateException("Failed to allocate workspace data buffer with "
                    + numElements + " elements of type " + dataType);
        }

        ret.attachOwner(owner, resolveAllocationDevice(owner, ret));
        ret.retainReference();
        if (ops.isFuncTrace()) {
            ret.captureTrace();
        }
        // Do NOT register with DeallocatorService - workspace owns this memory.
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
        NativeBufferOwner owner = primaryBackendOwner();
        NativeOps ops = owner.nativeOps();
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

        // Tracks whether this allocation has already failed an OOM over to another GPU.
        boolean failedOver = false;

        try {
            for (int t = 0; t < MAX_TRIES; t++) {
                try {
                    // try to allocate data buffer
                    buffer = ops.allocateDataBuffer(numElements, dataType.toInt(), allocateBoth);

                    // Check if allocation succeeded
                    if(buffer != null && !buffer.isNull()) {
                        DeviceDescriptor ownedDevice = selectedDevice != null
                                ? selectedDevice
                                : resolveAllocationDevice(owner, buffer);
                        buffer.attachOwner(owner, ownedDevice);
                        buffer.retainReference();

                        // Register with DeallocatorService
                        try {
                            registerWithDeallocatorService(buffer, false, bytes);

                            // Track only an actual backend device allocation. Host memory has
                            // no compute-device descriptor and must not be synthesized as CPU.
                            if (buffer.allocationDevice != null) {
                                memoryManager.recordAllocation(buffer.allocationDevice, bytes);
                            }

                            // Capture trace if needed
                            if(ops.isFuncTrace())
                                buffer.captureTrace();

                            // Success - return the buffer
                            return buffer;
                        } catch (Exception regEx) {
                            // LEAK FIX: Clean up buffer if registration fails
                            ops.dbClose(buffer);
                            throw regEx;
                        }
                    }

                    // check error code
                    ec = ops.lastErrorCode();
                    if (ec != 0) {
                        em = ops.lastErrorMessage();

                        // Only invoke GC if Java heap is under pressure — GPU OOM is not helped by Java GC
                        gcIfHeapPressured();

                        // sleeping for 50ms to let any pending async frees complete
                        Thread.sleep(50);

                        // FAILOVER: the current GPU is out of memory. Instead of spinning on the
                        // same full device for MAX_TRIES and then throwing, ask DeviceMemoryManager
                        // (live, pool-aware) for the GPU with the most reclaimable free memory and
                        // retry there. This is the automatic "one GPU full -> use the other"
                        // failover; it fires only on the slow OOM path, so the common case is
                        // unaffected. Cross-device data is handled by DeviceAwareOpExecutioner.
                        if (!failedOver) {
                            int failedDevice = (selectedDevice != null && selectedDevice.getDeviceType().isGpu())
                                    ? selectedDevice.getDeviceIndex()
                                    : memoryManager.getCurrentDeviceId();
                            DeviceDescriptor failoverTarget = memoryManager.selectFailoverDevice(bytes, failedDevice);
                            if (failoverTarget != null && failoverTarget.getDeviceType().isGpu()
                                    && failoverTarget.getDeviceIndex() != failedDevice) {
                                if (savedDevice < 0) savedDevice = failedDevice; // restore original in finally
                                if (!DeviceAwareOpExecutioner.isInstalled()) DeviceAwareOpExecutioner.install();
                                memoryManager.switchDevice(failoverTarget.getDeviceIndex(), "OpaqueDataBuffer", "oom-failover");
                                selectedDevice = failoverTarget;
                                failedOver = true;
                            }
                        }
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
        NativeBufferOwner owner = primaryBackendOwner();
        NativeOps ops = owner.nativeOps();
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

        // Tracks whether this allocation has already failed an OOM over to another GPU.
        boolean failedOver = false;

        try {
            for (int t = 0; t < MAX_TRIES; t++) {
                try {
                    // try to allocate data buffer
                    buffer = ops.allocateDataBuffer(numElements, dataType.toInt(), allocateBoth);

                    // Check if allocation succeeded
                    if(buffer != null && !buffer.isNull()) {
                        DeviceDescriptor ownedDevice = selectedDevice != null
                                ? selectedDevice
                                : resolveAllocationDevice(owner, buffer);
                        buffer.attachOwner(owner, ownedDevice);
                        buffer.retainReference();

                        // Register with DeallocatorService, marking as constant if requested
                        try {
                            registerWithDeallocatorService(buffer, isConstant, isConstant ? 0L : bytes);

                            // Track only an actual backend device allocation. Host memory has
                            // no compute-device descriptor and must not be synthesized as CPU.
                            if (!isConstant && buffer.allocationDevice != null) {
                                memoryManager.recordAllocation(buffer.allocationDevice, bytes);
                            }

                            // Capture trace if needed
                            if(ops.isFuncTrace())
                                buffer.captureTrace();

                            // Success - return the buffer
                            return buffer;
                        } catch (Exception regEx) {
                            // LEAK FIX: Clean up buffer if registration fails
                            ops.dbClose(buffer);
                            throw regEx;
                        }
                    }

                    // check error code
                    ec = ops.lastErrorCode();
                    if (ec != 0) {
                        em = ops.lastErrorMessage();

                        // Only invoke GC if Java heap is under pressure — GPU OOM is not helped by Java GC
                        gcIfHeapPressured();

                        // sleeping for 50ms to let any pending async frees complete
                        Thread.sleep(50);

                        // FAILOVER: the current GPU is out of memory. Instead of spinning on the
                        // same full device for MAX_TRIES and then throwing, ask DeviceMemoryManager
                        // (live, pool-aware) for the GPU with the most reclaimable free memory and
                        // retry there. This is the automatic "one GPU full -> use the other"
                        // failover; it fires only on the slow OOM path, so the common case is
                        // unaffected. Cross-device data is handled by DeviceAwareOpExecutioner.
                        if (!failedOver) {
                            int failedDevice = (selectedDevice != null && selectedDevice.getDeviceType().isGpu())
                                    ? selectedDevice.getDeviceIndex()
                                    : memoryManager.getCurrentDeviceId();
                            DeviceDescriptor failoverTarget = memoryManager.selectFailoverDevice(bytes, failedDevice);
                            if (failoverTarget != null && failoverTarget.getDeviceType().isGpu()
                                    && failoverTarget.getDeviceIndex() != failedDevice) {
                                if (savedDevice < 0) savedDevice = failedDevice; // restore original in finally
                                if (!DeviceAwareOpExecutioner.isInstalled()) DeviceAwareOpExecutioner.install();
                                memoryManager.switchDevice(failoverTarget.getDeviceIndex(), "OpaqueDataBuffer", "oom-failover");
                                selectedDevice = failoverTarget;
                                failedOver = true;
                            }
                        }
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

    public static OpaqueDataBuffer allocateDataBuffer(long numElements, @NonNull DataType dataType,
                                                      boolean allocateBoth,
                                                      @NonNull NativeBufferOwner owner) {
        return allocateOwnedDataBuffer(numElements, dataType, allocateBoth, false, owner);
    }

    public static OpaqueDataBuffer allocateDataBuffer(long numElements, @NonNull DataType dataType,
                                                      boolean allocateBoth, boolean isConstant,
                                                      @NonNull NativeBufferOwner owner) {
        return allocateOwnedDataBuffer(numElements, dataType, allocateBoth, isConstant, owner);
    }

    private static OpaqueDataBuffer allocateOwnedDataBuffer(long numElements, DataType dataType,
                                                            boolean allocateBoth, boolean isConstant,
                                                            NativeBufferOwner owner) {
        owner.currentDevice();
        NativeOps ops = owner.nativeOps();
        long bytes = allocationBytes(numElements, dataType);
        String errorMessage = null;

        for (int attempt = 0; attempt < MAX_TRIES; attempt++) {
            OpaqueDataBuffer buffer =
                    ops.allocateDataBuffer(numElements, dataType.toInt(), allocateBoth);
            if (buffer != null && !buffer.isNull()) {
                buffer.attachOwner(owner, resolveAllocationDevice(owner, buffer));
                buffer.retainReference();
                try {
                    registerWithDeallocatorService(
                            buffer, isConstant, isConstant ? 0L : bytes);
                    if (!isConstant && buffer.allocationDevice != null) {
                        owner.recordAllocation(buffer.allocationDevice, bytes);
                    }
                    if (ops.isFuncTrace()) {
                        buffer.captureTrace();
                    }
                    return buffer;
                } catch (RuntimeException registrationFailure) {
                    ops.dbClose(buffer);
                    throw registrationFailure;
                }
            }

            int errorCode = ops.lastErrorCode();
            if (errorCode == 0) {
                break;
            }
            errorMessage = ops.lastErrorMessage();
            gcIfHeapPressured();
            try {
                Thread.sleep(50);
            } catch (InterruptedException interrupted) {
                Thread.currentThread().interrupt();
                throw new RuntimeException("Allocation interrupted", interrupted);
            }
        }

        throw new RuntimeException("Allocation failed: [" + errorMessage + "] for amount of memory "
                + bytes + " bytes");
    }

    /**
     * This method expands buffer, and copies content to the new buffer
     *
     * PLEASE NOTE: if InteropDataBuffer doesn't own actual buffers - original pointers won't be released
     * @param numElements
     */
    public void expand(long numElements) {
        NativeOps ops = nativeOps();
        int ec = 0;
        String em = null;

        for (int t = 0; t < MAX_TRIES; t++) {
            try {
                // try to expand the buffer
                ops.dbExpand(this, numElements);

                // check error code
                ec = ops.lastErrorCode();
                if (ec == 0) {
                    // Success
                    return;
                }
                
                em = ops.lastErrorMessage();

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
        NativeOps ops = nativeOps();
        OpaqueDataBuffer buffer = null;
        int ec = 0;
        String em = null;

        for (int t = 0; t < MAX_TRIES; t++) {
            try {
                buffer = ops.dbCreateView(this, bytesLength);

                // Check if view creation succeeded
                if(buffer != null && !buffer.isNull()) {
                    buffer.attachOwner(requireBackendOwner(), allocationDevice);
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
                        
                        if(ops.isFuncTrace())
                            buffer.captureTrace();
                        
                        // Success - return the buffer
                        return buffer;
                    } catch (Exception regEx) {
                        // LEAK FIX: Clean up buffer if registration fails
                        ops.dbClose(buffer);
                        throw regEx;
                    }
                }
                
                // check error code
                ec = ops.lastErrorCode();

                if (ec != 0) {
                    em = ops.lastErrorMessage();

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
        return nativeOps().dbBufferLength(this);
    }

    /**
     * This method returns pointer to linear buffer, primary one.
     * @return
     */
    public Pointer primaryBuffer() {
        return nativeOps().dbPrimaryBuffer(this);
    }

    /**
     * This method returns pointer to special buffer, device one, if any.
     * @return
     */
    public Pointer specialBuffer() {
        return nativeOps().dbSpecialBuffer(this);
    }

    /**
     * This method returns deviceId of this DataBuffer
     * @return
     */
    public int deviceId() {
        return nativeOps().dbDeviceId(this);
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
        nativeOps().dbSetPrimaryBuffer(this, ptr, numElements);
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
        nativeOps().dbSetSpecialBuffer(this, ptr, numElements);
    }

    /**
     * This method synchronizes device memory
     */
    public void syncToSpecial() {
        nativeOps().dbSyncToSpecial(this);
    }
    public void migrate() {
        NativeBufferOwner owner = requireBackendOwner();
        owner.nativeOps().dbMigrate(this);
        allocationDevice = resolveAllocationDevice(owner, this);
    }

    /**
     * This method synchronizes host memory
     */
    public void syncToPrimary() {
        nativeOps().dbSyncToPrimary(this);
    }

    public void printAllocationTraceIfNeeded() {
        if(allocationTrace != null && Nd4j.getEnvironment().isFuncTracePrintAllocate()) {
            log.debug("Java side allocation trace: \n {}", allocationTrace);
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
                    nativeOps().dbFreeBuffersOnly(this);
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
        // This is the atomic flag that prevents double-free. Explicit close uses this public
        // facade while the PhantomReference cleanup action uses a detached raw-address facade;
        // both ultimately meet the native InteropDataBuffer tryClose guard.
        if (!this.isNull() && tryMarkForDeallocation()) {
            try {
                printAllocationTraceIfNeeded();
                if (Nd4j.getEnvironment().isFuncTracePrintDeallocate()) {
                    log.debug("Java side deallocation current trace: \n {}", currentTrace());
                }
                long allocationBytes = deallocator != null ? deallocator.getAllocationBytes() : 0L;
                DeviceDescriptor deallocationDevice = allocationBytes > 0 ? allocationDevice : null;
                // Second shutdown check: if shutdown started after the initial check,
                // use GPU-only free to avoid host free() discovering corrupted metadata.
                if (DeallocatorService.getShutdownInProgress().get()) {
                    nativeOps().dbFreeBuffersOnly(this);
                } else {
                    nativeOps().dbClose(this);
                }
                dbCloseSuccess.incrementAndGet();
                this.setNull();

                if (deallocator != null) {
                    deallocator.markDeallocated();
                }
                if (allocationBytes > 0 && deallocationDevice != null) {
                    requireBackendOwner().recordDeallocation(deallocationDevice, allocationBytes);
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

        boolean nativeSuccess = nativeOps().dbSetConstant(this, isConstant);

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
        return resolveAllocationDevice(buffer.requireBackendOwner(), buffer);
    }

    private static DeviceDescriptor resolveAllocationDevice(NativeBufferOwner owner,
                                                            OpaqueDataBuffer buffer) {
        int deviceId = owner.nativeOps().dbDeviceId(buffer);
        return deviceId >= 0 ? owner.deviceDescriptor(deviceId) : null;
    }

    /**
     * Process-primary backend owner. Public so callers that wrap natively
     * returned handles on inherently primary-backend-scoped paths (e.g.
     * DspHandle staging reads through NativeOpsHolder) can attach the same
     * owner the buffer factories use.
     */
    public static NativeBufferOwner primaryOwner() {
        return primaryBackendOwner();
    }

    private static NativeBufferOwner primaryBackendOwner() {
        final NativeOps nativeOps = Nd4j.getNativeOps();
        final AffinityManager affinityManager = Nd4j.getAffinityManager();
        final OpExecutioner executioner = Nd4j.getExecutioner();
        final DeallocatorService deallocatorService = Nd4j.getDeallocatorService();
        final DeviceMemoryManager deviceMemoryManager = DeviceMemoryManager.getInstance();

        return new NativeBufferOwner() {
            @Override
            public NativeOps nativeOps() {
                return nativeOps;
            }

            @Override
            public DeallocatorService deallocatorService() {
                return deallocatorService;
            }

            @Override
            public int currentDevice() {
                return affinityManager.getDeviceForCurrentThread();
            }

            @Override
            public int deviceCount() {
                return affinityManager.getNumberOfDevices();
            }

            @Override
            public void setDevice(int deviceId) {
                affinityManager.setDeviceForCurrentThread(deviceId);
            }

            @Override
            public void commit() {
                executioner.commit();
            }

            @Override
            public DeviceDescriptor deviceDescriptor(int deviceId) {
                return affinityManager.getDeviceDescriptor(deviceId);
            }

            @Override
            public void recordAllocation(DeviceDescriptor device, long bytes) {
                deviceMemoryManager.recordAllocation(device, bytes);
            }

            @Override
            public void recordDeallocation(DeviceDescriptor device, long bytes) {
                deviceMemoryManager.recordDeallocation(device, bytes);
            }
        };
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
