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
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.PointerPointer;
import org.nd4j.linalg.api.memory.deallocation.DeallocatorService;
import org.nd4j.linalg.api.memory.deallocation.OpaqueNDArrayArrDeallocator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

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

    // CRITICAL: Keep OpaqueNDArray instances alive to prevent use-after-free!
    // The PointerPointer only stores raw native pointers. Without this field,
    // the OpaqueNDArray Java objects become GC-eligible immediately after createFrom()
    // returns. When GC runs, DeallocatorService deletes the native sd::NDArray* objects
    // that the PointerPointer's raw pointers still reference -> heap corruption!
    private OpaqueNDArray[] opaqueArrays;

    // Track the deallocator for this instance
    private OpaqueNDArrayArrDeallocator deallocator;

    // Track the number of arrays stored
    private int numArrays;

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
        // Use PointerPointer's native allocation - allocates sizeof(void*) * length bytes
        super(opaqueArrays.length);
        if (opaqueArrays == null || opaqueArrays.length == 0) {
            throw new IllegalArgumentException("Cannot create OpaqueNDArrayArr from null or empty array");
        }

        // Use PointerPointer's put() method to properly store each pointer in native memory
        for (int i = 0; i < opaqueArrays.length; i++) {
            if (opaqueArrays[i] == null) {
                throw new IllegalArgumentException("OpaqueNDArray at index " + i + " is null");
            }
            this.put(i, opaqueArrays[i]);
        }

        this.numArrays = opaqueArrays.length;
        this.opaqueArrays = opaqueArrays;
        this.retainReference();
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
     * Creates an OpaqueNDArrayArr from an array of INDArrays.
     * Parent INDArray references are held to ensure validity of the OpaqueNDArray pointers.
     *
     * <p><b>Memory Management:</b> The created OpaqueNDArrayArr is automatically registered
     * with {@link DeallocatorService} for cleanup. You can also explicitly call {@link #close()}
     * for immediate cleanup.</p>
     *
     * <p><b>Important:</b> This method uses cached OpaqueNDArray instances from parent INDArrays
     * via {@link OpaqueNDArray#fromINDArray(INDArray)}. The parent arrays must remain alive
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

        // Convert INDArrays to OpaqueNDArrays.
        // We create new OpaqueNDArray wrappers for each INDArray to ensure we have
        // independent native sd::NDArray* objects that we fully control.
        OpaqueNDArray[] inputs = new OpaqueNDArray[array.length];
        for (int i = 0; i < array.length; i++) {
            INDArray indArray = array[i];
            // Create a new OpaqueNDArray wrapper - this creates a new sd::NDArray* in native code
            // that wraps the same DataBuffer but is independent of any cached OpaqueNDArray
            OpaqueNDArray opaque = OpaqueNDArray.fromINDArrayUncached(indArray);
            if (opaque == null) {
                throw new org.nd4j.linalg.exception.ND4JIllegalStateException(
                    "Failed to create OpaqueNDArray for INDArray at index " + i +
                    " (id=" + indArray.getId() + ", shape=" + java.util.Arrays.toString(indArray.shape()) + ")");
            }
            if (opaque.isNull()) {
                throw new org.nd4j.linalg.exception.ND4JIllegalStateException(
                    "OpaqueNDArray.fromINDArrayUncached returned object with null native pointer at index " + i +
                    " (id=" + indArray.getId() + ", shape=" + java.util.Arrays.toString(indArray.shape()) + ")");
            }
            opaque.retainReference(); // Explicitly retain reference for this usage
            inputs[i] = opaque;
        }

        // Allocate our own PointerPointer and write addresses directly to native memory
        OpaqueNDArrayArr inputsOpaque = new OpaqueNDArrayArr(inputs.length);
        inputsOpaque.retainReference();

        // Write addresses directly using Pointer.put(long, Pointer) which writes to native memory
        for (int i = 0; i < inputs.length; i++) {
            inputsOpaque.put(i, inputs[i]);
        }

        // Store references to prevent GC from freeing memory while we're using it
        inputsOpaque.parentArrays = array;
        inputsOpaque.opaqueArrays = inputs;
        inputsOpaque.numArrays = inputs.length;

        // Register with DeallocatorService for proper lifecycle management (if requested).
        // When registerWithDeallocator=false, the caller (e.g., CudaOpContext) is responsible
        // for cleanup, which prevents race conditions between multiple deallocators.
        if (registerWithDeallocator) {
            registerWithDeallocatorService(inputsOpaque, array);
        } else {
            if (log.isTraceEnabled()) {
                log.trace("OpaqueNDArrayArr created without DeallocatorService registration (caller manages lifecycle)");
            }
        }

        return inputsOpaque;
    }

    /**
     * Registers this OpaqueNDArrayArr with the DeallocatorService for automatic cleanup.
     * This ensures parent INDArrays remain alive and provides reliable cleanup.
     *
     * @param arrayArr The array to register
     * @param parentArrays The parent INDArrays to keep alive
     * @throws RuntimeException if registration fails
     */
    private static void registerWithDeallocatorService(OpaqueNDArrayArr arrayArr, INDArray[] parentArrays) {
        try {
            DeallocatorService service = Nd4j.getDeallocatorService();
            long uniqueId = service.nextValue();
            int targetDevice = Nd4j.getAffinityManager().getDeviceForCurrentThread();
            
            OpaqueNDArrayArrDeallocator deallocator = new OpaqueNDArrayArrDeallocator(
                arrayArr, parentArrays, uniqueId, targetDevice
            );
            
            arrayArr.deallocator = deallocator;
            service.pickObject(deallocator);
            
            if (log.isTraceEnabled()) {
                log.trace("Registered OpaqueNDArrayArr {} with DeallocatorService (parent count: {})", 
                        uniqueId, parentArrays.length);
            }
        } catch (Exception e) {
            // LEAK FIX: If registration fails, caller must clean up the array
            log.error("Failed to register OpaqueNDArrayArr with DeallocatorService", e);
            throw new RuntimeException("Failed to register array with DeallocatorService", e);
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
        if (deallocator != null) {
            // Only deallocate if not already done - prevents double-free
            if (!deallocator.isDeallocated()) {
                deallocator.deallocate();
            }
            // If deallocator exists but is already deallocated, do nothing - already cleaned up
        } else {
            // Fallback cleanup ONLY if not registered with DeallocatorService at all
            if (log.isTraceEnabled()) {
                log.trace("Fallback cleanup for unregistered OpaqueNDArrayArr");
            }
            // Close the uncached OpaqueNDArrays we created
            // Since we use fromINDArrayUncached(), these are owned by this OpaqueNDArrayArr
            // and must be explicitly closed to free the native sd::NDArray* objects.
            if (opaqueArrays != null) {
                for (OpaqueNDArray opaque : opaqueArrays) {
                    if (opaque != null) {
                        opaque.releaseReference(); // Release the reference added in createFrom
                        opaque.close(); // Close the uncached OpaqueNDArray to free native memory
                    }
                }
            }
            // Deallocate the PointerPointer's native memory
            deallocate();
            parentArrays = null;
            opaqueArrays = null;
        }
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
