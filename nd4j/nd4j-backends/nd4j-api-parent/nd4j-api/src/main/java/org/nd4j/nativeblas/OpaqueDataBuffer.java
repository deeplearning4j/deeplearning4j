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
import org.nd4j.common.primitives.AtomicBoolean;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.memory.deallocation.DeallocatorService;
import org.nd4j.linalg.api.memory.deallocation.OpaqueDataBufferDeallocator;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;

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

    // Track if buffer has been explicitly closed to prevent double-free
    private volatile boolean explicitlyClosed = false;

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
        registerWithDeallocatorService(buffer, false);
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
    private static void registerWithDeallocatorService(OpaqueDataBuffer buffer, boolean isConstant) {
        try {
            DeallocatorService service = Nd4j.getDeallocatorService();
            long uniqueId = service.nextValue();
            int targetDevice = Nd4j.getAffinityManager().getDeviceForCurrentThread();

            OpaqueDataBufferDeallocator deallocator = new OpaqueDataBufferDeallocator(
                buffer, uniqueId, targetDevice
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
                    registerWithDeallocatorService(ret, true);
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
            ret = Nd4j.getNativeOps().dbCreateExternalDataBuffer(numElements, dataType.toInt(), primary, special);

            if (ret != null && !ret.isNull()) {
                ret.retainReference();

                if(NativeOpsHolder.getInstance().getDeviceNativeOps().isFuncTrace())
                    ret.captureTrace();

                // Register with DeallocatorService
                try {
                    registerWithDeallocatorService(ret, false);
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
     * This method allocates new InteropDataBuffer and returns pointer to it.
     * The buffer is automatically registered with DeallocatorService for cleanup.
     *
     * MEMORY LEAK FIXES:
     * - Clean up failed buffers in retry loop
     * - Clean up buffer if registration fails
     *
     * Note: Device assignment must happen before native allocation. Without this, multiple threads
     * starting simultaneously could all see CUDA device 0 (default), but then get different device
     * assignments from Java's round-robin, leading to illegal memory access when operations try to
     * access buffers on the wrong device.
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

        // Ensure device is assigned for this thread before any native allocation.
        // This ensures the native code allocates the buffer on the correct device.
        // Without this, there's a race condition where:
        // 1. Native code uses CUDA's current device (often device 0)
        // 2. Java later assigns a different device via round-robin
        // 3. Subsequent operations fail with CUDA error 700 (illegal memory access)
        Nd4j.getAffinityManager().getDeviceForCurrentThread();

        for (int t = 0; t < MAX_TRIES; t++) {
            try {
                // try to allocate data buffer
                buffer = Nd4j.getNativeOps().allocateDataBuffer(numElements, dataType.toInt(), allocateBoth);

                // Check if allocation succeeded
                if(buffer != null && !buffer.isNull()) {
                    buffer.retainReference();

                    // Register with DeallocatorService
                    try {
                        registerWithDeallocatorService(buffer);
                        
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

                    // if allocation failed it might be caused by casual OOM, so we'll try GC
                    System.gc();

                    // sleeping for 50ms
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
    }

    /**
     * Allocates a new InteropDataBuffer and optionally marks it as constant.
     *
     * This method allows creating constant buffers that are protected from deallocation
     * before the deallocator is registered with DeallocatorService. This prevents the race
     * condition where GC could trigger deallocation between buffer creation and setConstant()
     * being called.
     *
     * Note: Device assignment must happen before native allocation. Without this, multiple threads
     * starting simultaneously could all see CUDA device 0 (default), but then get different device
     * assignments from Java's round-robin, leading to illegal memory access when operations try to
     * access buffers on the wrong device.
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

        // Ensure device is assigned for this thread before any native allocation.
        // This ensures the native code allocates the buffer on the correct device.
        Nd4j.getAffinityManager().getDeviceForCurrentThread();

        for (int t = 0; t < MAX_TRIES; t++) {
            try {
                // try to allocate data buffer
                buffer = Nd4j.getNativeOps().allocateDataBuffer(numElements, dataType.toInt(), allocateBoth);

                // Check if allocation succeeded
                if(buffer != null && !buffer.isNull()) {
                    buffer.retainReference();

                    // Register with DeallocatorService, marking as constant if requested
                    try {
                        registerWithDeallocatorService(buffer, isConstant);

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

                    // if allocation failed it might be caused by casual OOM, so we'll try GC
                    System.gc();

                    // sleeping for 50ms
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

                // if expansion failed it might be caused by casual OOM, so we'll try GC
                System.gc();

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
                // NOTE: Do NOT call retainReference() - it prevents DeallocatorService from working!
                // DeallocatorService relies on the Java object becoming garbage-collectible
                buffer = Nd4j.getNativeOps().dbCreateView(this, bytesLength);
                
                // Check if view creation succeeded
                if(buffer != null && !buffer.isNull()) {
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

                    // if view creation failed it might be caused by casual OOM, so we'll try GC
                    System.gc();

                    // sleeping to let gc kick in
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

    public void closeBuffer() {
        // Check if already closed or null
        if (this.isNull() || explicitlyClosed) {
            if (log.isTraceEnabled()) {
                log.trace("Attempted to close already closed or null OpaqueDataBuffer");
            }
            return;
        }

        synchronized (this) {
            if (explicitlyClosed) {
                return;
            }
            explicitlyClosed = true;
        }

        if (deallocator != null) {
            // Only deallocate if not already done - prevents double-free
            if (!deallocator.isDeallocated()) {
                deallocator.deallocate();
            }
            // If deallocator exists but is already deallocated, do nothing
        } else {
            // Fallback ONLY if not registered with DeallocatorService at all
            printAllocationTraceIfNeeded();
            if(Nd4j.getEnvironment().isFuncTracePrintDeallocate()) {
                System.out.println("Java side deallocation current trace: \n " + currentTrace());
            }
            Nd4j.getNativeOps().dbClose(this);
        }
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

    /**
     * Returns whether this buffer is marked as constant.
     *
     * @return true if constant, false otherwise
     */
    public boolean isConstant() {
        return deallocator != null && deallocator.isConstant();
    }
}
