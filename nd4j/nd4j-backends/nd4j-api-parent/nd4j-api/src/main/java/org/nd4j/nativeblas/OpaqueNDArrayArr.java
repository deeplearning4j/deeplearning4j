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

import lombok.extern.slf4j.Slf4j;
import org.bytedeco.javacpp.LongPointer;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.PointerPointer;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.device.DeviceDescriptor;
import org.nd4j.linalg.api.memory.deallocation.DeallocatorService;
import org.nd4j.linalg.api.memory.deallocation.OpaqueNDArrayArrDeallocator;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Arrays;
import java.util.List;

/**
 * OpaqueNDArrayArr is a PointerPointer wrapper for arrays of OpaqueNDArray instances.
 * It maintains references to parent INDArrays to ensure they remain alive while
 * the OpaqueNDArrayArr is in use, preventing use-after-free issues.
 *
 * <p><b>Memory Management:</b> This class is integrated with {@link DeallocatorService}
 * for reliable memory cleanup. Parent INDArray references are held to ensure
 * the underlying OpaqueNDArray pointers remain valid.</p>
 *
 * <p><b>Usage Pattern:</b> Use try-with-resources for explicit cleanup:
 * <pre>{@code
 * try (OpaqueNDArrayArr arr = OpaqueNDArrayArr.createFrom(array1, array2)) {
 *     // Use arr...
 * }
 * }</pre>
 * Or rely on automatic cleanup via DeallocatorService when the object becomes unreachable.
 * </p>
 *
 * @see OpaqueNDArray
 * @see OpaqueNDArrayArrDeallocator
 * @see DeallocatorService
 */
@Slf4j
public class OpaqueNDArrayArr extends PointerPointer<OpaqueNDArray> implements AutoCloseable {

    // Keep parent arrays alive to prevent use-after-free
    private INDArray[] parentArrays;

    private OpaqueNDArray[] opaqueArrays;
    private boolean[] opaqueReferencesReleased;
    private boolean ownsOpaqueArrays;

    // Track the deallocator for this instance
    private OpaqueNDArrayArrDeallocator deallocator;

    // Track the number of arrays stored
    private int numArrays;

    /**
     * Contiguous C-array of sd::NDArray* values (N longs).
     *
     * <p>WHY: The JNI thunk generated for {@code @ByVal OpaqueNDArrayArr} performs ONE
     * pointer dereference: it reads slot[position] (default position=0) from the
     * PointerPointer's native buffer and passes that VALUE as the {@code sd::NDArray**}
     * argument to C++.  With the old N-slot layout each slot held one
     * {@code sd::NDArray*}, so slot[0] was a single NDArray pointer — C++ then
     * mis-indexed into NDArray's internal fields instead of treating it as an array
     * of pointers, producing the {@code ConstantShapeBuffer::primary(): corrupted this}
     * crash at {@code x[0]->shapeInfo()} and null-NDArray crashes at {@code x[1+]}.</p>
     *
     * <p>With the 1-slot layout:
     * <ul>
     *   <li>This {@code LongPointer} holds N contiguous 8-byte {@code sd::NDArray*} VALUES.</li>
     *   <li>Each slot is populated via {@code new LongPointer(opaqueArr[i]).get(0)} which
     *       dereferences the JavaCPP wrapper to extract the actual {@code sd::NDArray*} pointer
     *       value (NOT {@code opaqueArr[i].address()}, which is the wrapper address, not the value).</li>
     *   <li>The PointerPointer has exactly 1 slot: slot[0] = {@code ndPtrBuffer.address()}.</li>
     *   <li>JNI dereferences once → C++ receives {@code x = ndPtrBuffer.address() = sd::NDArray**}.</li>
     *   <li>C++ does {@code x[i]} → reads the i-th {@code sd::NDArray*} value from the buffer. Correct.</li>
     * </ul></p>
     */
    private LongPointer ndPtrBuffer;

    /** Owns the one-slot native pointer table viewed by this facade. */
    private PointerPointer<OpaqueNDArray> pointerStorage;

    /**
     * Default constructor for internal use.
     * Creates an uninitialized OpaqueNDArrayArr that must be set up via direct field access.
     */
    public OpaqueNDArrayArr() {
        super();
    }

    /**
     * Constructor that wraps an existing Pointer's memory.
     * This is used to share native memory with another Pointer (like LongPointer).
     *
     * @param p The Pointer whose memory to wrap
     */
    public OpaqueNDArrayArr(Pointer p) {
        super(p);
    }

    /**
     * Constructor that allocates native memory for the specified number of pointers.
     * Uses PointerPointer's native memory allocation.
     *
     * @param size The number of pointers to allocate space for
     */
    public OpaqueNDArrayArr(long size) {
        super(size);
    }

    /**
     * Constructor that creates from an array of OpaqueNDArray objects.
     * Uses PointerPointer's native put() method to properly store the pointers.
     *
     * <p><b>Note:</b> The caller is responsible for keeping the OpaqueNDArray objects
     * alive while this OpaqueNDArrayArr is in use.</p>
     *
     * @param opaqueArrays Array of OpaqueNDArray objects
     */
    public OpaqueNDArrayArr(OpaqueNDArray[] opaqueArrays) {
        // This facade is a non-owning view. Detached pointerStorage owns the
        // one-slot table and can therefore be cleaned without retaining this object.
        super();
        if (opaqueArrays == null || opaqueArrays.length == 0) {
            throw new IllegalArgumentException("Cannot create OpaqueNDArrayArr from null or empty array");
        }

        for (int i = 0; i < opaqueArrays.length; i++) {
            if (opaqueArrays[i] == null) {
                throw new IllegalArgumentException("OpaqueNDArray at index " + i + " is null");
            }
        }

        // Allocate a contiguous C-array of N sd::NDArray* values.
        // CRITICAL: opaqueArrays[i].address() is the JavaCPP WRAPPER address (OpaqueNDArray* rptr
        // from "new OpaqueNDArray(createOpaqueNDArray(...))").  The actual sd::NDArray* value is
        // stored AT that wrapper address (the first 8 bytes of the wrapper struct).
        // We must dereference once to get the sd::NDArray* value that C++ shuffle() expects.
        LongPointer buf = new LongPointer(opaqueArrays.length);
        for (int i = 0; i < opaqueArrays.length; i++) {
            buf.put(i, new LongPointer(opaqueArrays[i]).get(0));
        }

        PointerPointer<OpaqueNDArray> storage = new PointerPointer<>(1L);
        storage.put(0, buf);
        setPointerView(storage);

        this.ndPtrBuffer = buf;
        this.pointerStorage = storage;
        this.numArrays = opaqueArrays.length;
        this.opaqueArrays = opaqueArrays;
        this.opaqueReferencesReleased = new boolean[opaqueArrays.length];
        Arrays.fill(this.opaqueReferencesReleased, true);
        this.ownsOpaqueArrays = false;
    }

    private void setPointerView(Pointer pointer) {
        this.address = pointer.address();
        this.position = pointer.position();
        this.limit = pointer.limit();
        this.capacity = pointer.capacity();
    }

    /**
     * Gets the number of arrays stored in this OpaqueNDArrayArr.
     * @return The number of arrays
     */
    public int getNumArrays() {
        return numArrays;
    }

    /**
     * Creates an OpaqueNDArrayArr from a list of INDArrays.
     * Parent INDArray references are held to ensure validity of the OpaqueNDArray pointers.
     *
     * <p><b>Memory Management:</b> The created OpaqueNDArrayArr is automatically registered
     * with {@link DeallocatorService} for cleanup. You can also explicitly call {@link #close()}
     * for immediate cleanup.</p>
     *
     * @param array List of INDArrays to convert
     * @return A new OpaqueNDArrayArr registered with DeallocatorService
     * @see #createFrom(INDArray...)
     */
    public static OpaqueNDArrayArr createFrom(List<INDArray> array) {
        return createFrom(true, array);
    }

    /**
     * Creates an OpaqueNDArrayArr from a list of INDArrays with optional DeallocatorService registration.
     *
     * @param registerWithDeallocator Whether to register with DeallocatorService
     * @param array List of INDArrays to convert
     * @return A new OpaqueNDArrayArr, optionally registered with DeallocatorService
     * @see #createFrom(boolean, INDArray...)
     */
    public static OpaqueNDArrayArr createFrom(boolean registerWithDeallocator, List<INDArray> array) {
        INDArray[] arrayArr = array.toArray(new INDArray[0]);
        return createFrom(registerWithDeallocator, arrayArr);
    }

    /**
     * Creates an array wrapper through an explicitly selected native backend.
     */
    public static OpaqueNDArrayArr createFrom(
            NativeBufferOwner owner, List<INDArray> array) {
        return createFrom(owner, true, array);
    }

    /**
     * Creates an array wrapper through an explicitly selected native backend.
     */
    public static OpaqueNDArrayArr createFrom(
            NativeBufferOwner owner, boolean registerWithDeallocator, List<INDArray> array) {
        if (array == null) {
            throw new IllegalArgumentException("Cannot create OpaqueNDArrayArr from a null list");
        }
        return createFrom(owner, registerWithDeallocator, array.toArray(new INDArray[0]));
    }

    /**
     * Creates an OpaqueNDArrayArr from an array of INDArrays.
     * Parent INDArray references are held to ensure validity of the OpaqueNDArray pointers.
     *
     * <p><b>Memory Management:</b> The created OpaqueNDArrayArr is automatically registered
     * with {@link DeallocatorService} for cleanup. You can also explicitly call {@link #close()}
     * for immediate cleanup.</p>
     *
     * <p><b>Important:</b> This method creates owned, uncached OpaqueNDArray wrappers from
     * the parent INDArrays. The parent arrays must remain alive
     * while this OpaqueNDArrayArr is in use. This is ensured by storing strong references
     * to the parent arrays.</p>
     *
     * @param array Array of INDArrays to convert
     * @return A new OpaqueNDArrayArr registered with DeallocatorService
     */
    public static OpaqueNDArrayArr createFrom(INDArray... array) {
        return createFrom(true, array);
    }

    /**
     * Creates an array wrapper through an explicitly selected native backend.
     */
    public static OpaqueNDArrayArr createFrom(
            NativeBufferOwner owner, INDArray... array) {
        return createFrom(owner, true, array);
    }

    /**
     * Creates an OpaqueNDArrayArr from an array of INDArrays with optional DeallocatorService registration.
     *
     * <p><b>Memory Management:</b> If registerWithDeallocator is true, the created OpaqueNDArrayArr
     * is automatically registered with {@link DeallocatorService} for cleanup. If false, the caller
     * is responsible for calling {@link #close()} to avoid memory leaks.</p>
     *
     * <p><b>When to skip registration:</b> Set registerWithDeallocator=false when the OpaqueNDArrayArr
     * is owned by another object (like CudaOpContext) that will handle its lifecycle. This prevents
     * race conditions where both the owner and DeallocatorService try to clean up the same arrays.</p>
     *
     * @param registerWithDeallocator Whether to register with DeallocatorService
     * @param array Array of INDArrays to convert
     * @return A new OpaqueNDArrayArr, optionally registered with DeallocatorService
     */
    public static OpaqueNDArrayArr createFrom(boolean registerWithDeallocator, INDArray... array) {
        return createFromInternal(null, registerWithDeallocator, array);
    }

    /**
     * Creates an array wrapper without consulting ND4J's primary backend.
     */
    public static OpaqueNDArrayArr createFrom(
            NativeBufferOwner owner, boolean registerWithDeallocator, INDArray... array) {
        if (owner == null) {
            throw new IllegalArgumentException("NativeBufferOwner cannot be null");
        }
        return createFromInternal(owner, registerWithDeallocator, array);
    }

    private static OpaqueNDArrayArr createFromInternal(
            NativeBufferOwner owner, boolean registerWithDeallocator, INDArray... array) {
        if (array == null || array.length == 0) {
            throw new IllegalArgumentException("Cannot create OpaqueNDArrayArr from null or empty array");
        }

        // Add a comprehensive null check here, right before the native pointers are created.
        // This is the last line of defense. If a null gets this far, it means upstream checks
        // in OpContext/InferenceSession were bypassed.
        for (int i = 0; i < array.length; i++) {
            if (array[i] == null) {
                throw new org.nd4j.linalg.exception.ND4JIllegalStateException(
                        "OpaqueNDArrayArr.createFrom received a null INDArray at index " + i + " of " + array.length +
                        ". This indicates a null array was passed to an OpContext, which should be prevented by upstream checks." +
                        " Array contents: " + Arrays.toString(array));
            }
        }

        OpaqueNDArray[] inputs = new OpaqueNDArray[array.length];
        boolean[] opaqueReferencesReleased = new boolean[array.length];
        Arrays.fill(opaqueReferencesReleased, true);
        LongPointer buf = null;
        PointerPointer<OpaqueNDArray> storage = null;
        OpaqueNDArrayArr inputsOpaque = null;
        try {
            for (int i = 0; i < array.length; i++) {
                INDArray indArray = array[i];
                OpaqueNDArray opaque = owner == null
                        ? OpaqueNDArray.fromINDArrayUncached(indArray)
                        : OpaqueNDArray.fromINDArrayUncached(owner, indArray);
                if (opaque == null || opaque.isNull()) {
                    throw new org.nd4j.linalg.exception.ND4JIllegalStateException(
                            "Failed to create OpaqueNDArray at index " + i
                                    + " (id=" + indArray.getId() + ", shape="
                                    + java.util.Arrays.toString(indArray.shape()) + ")");
                }

                // Publish the wrapper to rollback state before retention can fail.
                inputs[i] = opaque;
                opaque.retainReference();
                opaqueReferencesReleased[i] = false;
            }

            buf = new LongPointer(inputs.length);
            for (int i = 0; i < inputs.length; i++) {
                buf.put(i, new LongPointer(inputs[i]).get(0));
            }

            storage = new PointerPointer<>(1L);
            storage.put(0, buf);

            inputsOpaque = new OpaqueNDArrayArr();
            inputsOpaque.setPointerView(storage);
            inputsOpaque.pointerStorage = storage;
            inputsOpaque.ndPtrBuffer = buf;
            inputsOpaque.parentArrays = array;
            inputsOpaque.opaqueArrays = inputs;
            inputsOpaque.opaqueReferencesReleased = opaqueReferencesReleased;
            inputsOpaque.ownsOpaqueArrays = true;
            inputsOpaque.numArrays = inputs.length;

            if (registerWithDeallocator) {
                registerWithDeallocatorService(inputsOpaque, array, owner);
            } else if (log.isTraceEnabled()) {
                log.trace("OpaqueNDArrayArr created without DeallocatorService registration (caller manages lifecycle)");
            }

            return inputsOpaque;
        } catch (RuntimeException | Error failure) {
            try {
                if (inputsOpaque != null) {
                    inputsOpaque.cleanupResources();
                } else {
                    RuntimeException cleanupFailure = cleanupOpaqueResources(
                            inputs, opaqueReferencesReleased, buf, storage, true);
                    if (cleanupFailure != null) {
                        failure.addSuppressed(cleanupFailure);
                    }
                }
            } catch (RuntimeException cleanupFailure) {
                failure.addSuppressed(cleanupFailure);
            }
            throw failure;
        }
    }

    /**
     * Registers this OpaqueNDArrayArr with the DeallocatorService for automatic cleanup.
     * This ensures parent INDArrays remain alive and provides reliable cleanup.
     *
     * @param arrayArr The array to register
     * @param parentArrays The parent INDArrays to keep alive
     * @throws RuntimeException if registration fails
     */
    private static void registerWithDeallocatorService(
            OpaqueNDArrayArr arrayArr, INDArray[] parentArrays, NativeBufferOwner owner) {
        try {
            AllocationContext allocation = resolveAllocationContext(parentArrays, owner);
            DeallocatorService service = allocation.owner.deallocatorService();
            long uniqueId = service.nextValue();

            OpaqueNDArrayArrDeallocator.ResourceState resources =
                    new OpaqueNDArrayArrDeallocator.ResourceState(
                            parentArrays,
                            arrayArr.opaqueArrays,
                            arrayArr.opaqueReferencesReleased,
                            arrayArr.ndPtrBuffer,
                            arrayArr.pointerStorage,
                            arrayArr.ownsOpaqueArrays);
            OpaqueNDArrayArrDeallocator deallocator =
                    new OpaqueNDArrayArrDeallocator(
                            resources, uniqueId,
                            allocation.device.getDeviceIndex(), allocation.owner);

            // Tie the phantom referent to the facade before publishing it.
            arrayArr.deallocator = deallocator;
            try {
                service.pickObject(deallocator, allocation.owner);
            } catch (RuntimeException | Error registrationFailure) {
                arrayArr.deallocator = null;
                throw registrationFailure;
            }

            if (log.isTraceEnabled()) {
                log.trace("Registered OpaqueNDArrayArr {} with DeallocatorService (parent count: {})",
                        uniqueId, parentArrays.length);
            }
        } catch (RuntimeException e) {
            log.error("Failed to register OpaqueNDArrayArr with DeallocatorService", e);
            throw e;
        }
    }

    private static AllocationContext resolveAllocationContext(
            INDArray[] arrays, NativeBufferOwner requestedOwner) {
        NativeBufferOwner allocationOwner = null;
        DeviceDescriptor allocationDevice = null;

        for (int i = 0; i < arrays.length; i++) {
            DataBuffer data = arrays[i].data();
            OpaqueDataBuffer buffer = data != null ? data.opaqueBuffer() : null;
            if (buffer == null) {
                throw new IllegalArgumentException(
                        "INDArray at index " + i + " has no native data buffer");
            }

            NativeBufferOwner bufferOwner = buffer.backendOwner();
            DeviceDescriptor bufferDevice = buffer.allocationDevice();
            if (bufferDevice == null) {
                throw new IllegalStateException(
                        "INDArray at index " + i + " has no allocation device");
            }
            if (requestedOwner != null && bufferOwner != requestedOwner) {
                throw new IllegalArgumentException(
                        "INDArray at index " + i + " belongs to a different native owner");
            }
            if (allocationOwner == null) {
                allocationOwner = bufferOwner;
                allocationDevice = bufferDevice;
            } else if (bufferOwner != allocationOwner || !allocationDevice.equals(bufferDevice)) {
                throw new IllegalArgumentException(
                        "All INDArrays must share the same native owner and allocation device");
            }
        }

        if (allocationOwner == null || allocationDevice == null) {
            throw new IllegalArgumentException("At least one INDArray is required");
        }
        allocationOwner.deviceDescriptor(allocationDevice.getDeviceIndex());
        return new AllocationContext(allocationOwner, allocationDevice);
    }

    private static final class AllocationContext {
        private final NativeBufferOwner owner;
        private final DeviceDescriptor device;

        private AllocationContext(NativeBufferOwner owner, DeviceDescriptor device) {
            this.owner = owner;
            this.device = device;
        }
    }

    /**
     * Closes the current OpaqueNDArrayArr, releasing any allocated resources.
     * This method provides explicit cleanup and is preferred over waiting for
     * automatic cleanup via DeallocatorService.
     *
     * <p><b>Note:</b> After calling close(), this OpaqueNDArrayArr should not be used.</p>
     */
    @Override
    public void close() {
        OpaqueNDArrayArrDeallocator currentDeallocator = deallocator;
        if (currentDeallocator != null) {
            currentDeallocator.deallocate();
            clearFacadeAfterCleanup();
        } else {
            cleanupResources();
        }
    }

    /**
     * Shared idempotent cleanup for explicitly managed instances and
     * construction rollback. Registered instances use the detached state in
     * {@link OpaqueNDArrayArrDeallocator}.
     */
    public synchronized void cleanupResources() {
        boolean detachedStorage = pointerStorage != null;
        RuntimeException failure = cleanupOpaqueResources(
                opaqueArrays,
                opaqueReferencesReleased,
                ndPtrBuffer,
                pointerStorage,
                ownsOpaqueArrays);

        // Legacy pointer/size constructors still own their JavaCPP allocation.
        if (!detachedStorage && !isNull()) {
            try {
                super.close();
            } catch (RuntimeException e) {
                failure = appendCleanupFailure(failure, e);
            }
        }

        if (failure != null) {
            throw failure;
        }
        clearFacadeAfterCleanup();
    }

    private synchronized void clearFacadeAfterCleanup() {
        pointerStorage = null;
        ndPtrBuffer = null;
        opaqueArrays = null;
        opaqueReferencesReleased = null;
        parentArrays = null;
        deallocator = null;
        ownsOpaqueArrays = false;
        numArrays = 0;
        setNull();
    }

    private static RuntimeException cleanupOpaqueResources(
            OpaqueNDArray[] arrays,
            boolean[] referencesReleased,
            LongPointer pointerBuffer,
            PointerPointer<OpaqueNDArray> storage,
            boolean ownsArrays) {
        RuntimeException failure = null;
        if (ownsArrays && arrays != null) {
            for (int i = 0; i < arrays.length; i++) {
                OpaqueNDArray opaque = arrays[i];
                if (opaque == null) {
                    continue;
                }

                if (!referencesReleased[i]) {
                    try {
                        opaque.releaseReference();
                        referencesReleased[i] = true;
                    } catch (RuntimeException e) {
                        failure = appendCleanupFailure(failure, e);
                        continue;
                    }
                }

                try {
                    opaque.close();
                    arrays[i] = null;
                } catch (RuntimeException e) {
                    failure = appendCleanupFailure(failure, e);
                }
            }
        }

        if (storage != null) {
            try {
                storage.close();
            } catch (RuntimeException e) {
                failure = appendCleanupFailure(failure, e);
            }
        }

        if (pointerBuffer != null) {
            try {
                pointerBuffer.close();
            } catch (RuntimeException e) {
                failure = appendCleanupFailure(failure, e);
            }
        }
        return failure;
    }

    private static RuntimeException appendCleanupFailure(
            RuntimeException failure, RuntimeException next) {
        if (failure == null) {
            return next;
        }
        failure.addSuppressed(next);
        return failure;
    }

    /**
     * Gets the deallocator associated with this OpaqueNDArrayArr.
     *
     * @return The deallocator or null if not registered
     */
    public OpaqueNDArrayArrDeallocator getDeallocator() {
        return deallocator;
    }

    /**
     * Gets the parent INDArrays being kept alive by this OpaqueNDArrayArr.
     *
     * @return The parent arrays or null if deallocated
     */
    public INDArray[] getParentArrays() {
        return parentArrays;
    }

    /**
     * Gets the OpaqueNDArray instances being kept alive by this OpaqueNDArrayArr.
     * These are the uncached OpaqueNDArray instances created from parent INDArrays.
     *
     * @return The OpaqueNDArray arrays or null if deallocated
     */
    public OpaqueNDArray[] getOpaqueArrays() {
        return opaqueArrays;
    }

    /**
     * Clears the opaqueArrays reference (used by deallocator after closing them).
     */
    public void clearOpaqueArrays() {
        opaqueArrays = null;
    }
}
