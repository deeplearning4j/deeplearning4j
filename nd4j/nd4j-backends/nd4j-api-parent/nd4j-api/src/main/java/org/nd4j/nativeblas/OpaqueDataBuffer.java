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
        try {
            DeallocatorService service = Nd4j.getDeallocatorService();
            long uniqueId = service.nextValue();
            int targetDevice = Nd4j.getAffinityManager().getDeviceForCurrentThread();
            
            OpaqueDataBufferDeallocator deallocator = new OpaqueDataBufferDeallocator(
                buffer, uniqueId, targetDevice
            );
            
            buffer.deallocator = deallocator;
            service.pickObject(deallocator);
            
            if (log.isTraceEnabled()) {
                log.trace("Registered OpaqueDataBuffer {} with DeallocatorService", uniqueId);
            }
        } catch (Exception e) {
            // LEAK FIX: If registration fails, caller must clean up the buffer
            log.error("Failed to register OpaqueDataBuffer with DeallocatorService - buffer must be manually cleaned", e);
            throw new RuntimeException("Failed to register buffer with DeallocatorService", e);
        }
    }

    public static OpaqueDataBuffer externalizedDataBuffer(long numElements, @NonNull DataType dataType, Pointer primary, Pointer special) {
        // NOTE: Do NOT call retainReference() - it prevents DeallocatorService from working!
        // DeallocatorService relies on the Java object becoming garbage-collectible
        OpaqueDataBuffer ret = Nd4j.getNativeOps().dbCreateExternalDataBuffer(numElements, dataType.toInt(), primary, special);
        
        if(NativeOpsHolder.getInstance().getDeviceNativeOps().isFuncTrace())
            ret.captureTrace();
        
        // Register with DeallocatorService
        if (ret != null && !ret.isNull()) {
            try {
                registerWithDeallocatorService(ret);
            } catch (Exception e) {
                // LEAK FIX: Clean up buffer if registration fails
                Nd4j.getNativeOps().dbClose(ret);
                throw e;
            }
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
     * @param numElements Number of elements
     * @param dataType Data type
     * @param allocateBoth Whether to allocate both host and device buffers
     * @return Allocated buffer registered with DeallocatorService
     */
    public static OpaqueDataBuffer allocateDataBuffer(long numElements, @NonNull DataType dataType, boolean allocateBoth) {
        OpaqueDataBuffer buffer = null;
        int ec = 0;
        String em = null;

        for (int t = 0; t < MAX_TRIES; t++) {
            try {
                // try to allocate data buffer
                buffer = Nd4j.getNativeOps().allocateDataBuffer(numElements, dataType.toInt(), allocateBoth);
                
                // Check if allocation succeeded
                if(buffer != null && !buffer.isNull()) {
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

        if (deallocator != null && !deallocator.isDeallocated()) {
            deallocator.deallocate();
        } else {
            // Fallback if not registered with DeallocatorService
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
}
