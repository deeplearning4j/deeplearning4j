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
    
    // Track the deallocator for this instance
    private OpaqueNDArrayArrDeallocator deallocator;

    /**
     * Basic constructor for internal use.
     * 
     * @param array Array of OpaqueNDArray pointers
     */
    public OpaqueNDArrayArr(OpaqueNDArray... array) { 
        super(array); 
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
        INDArray[] arrayArr = array.toArray(new INDArray[0]);
        return createFrom(arrayArr);
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
        if (array == null || array.length == 0) {
            throw new IllegalArgumentException("Cannot create OpaqueNDArrayArr from null or empty array");
        }

        // Convert INDArrays to OpaqueNDArrays (using cached instances)
        OpaqueNDArray[] inputs = Arrays.stream(array)
                .map(OpaqueNDArray::fromINDArray)
                .toArray(OpaqueNDArray[]::new);
        
        // Create the OpaqueNDArrayArr
        OpaqueNDArrayArr inputsOpaque = (OpaqueNDArrayArr) new OpaqueNDArrayArr().capacity(inputs.length);
        inputsOpaque.put(inputs);
        
        // Store parent references to keep them alive
        inputsOpaque.parentArrays = array.clone(); // Clone to prevent external modification
        
        // Register with DeallocatorService for automatic cleanup
        registerWithDeallocatorService(inputsOpaque, array);
        
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
        if (deallocator != null && !deallocator.isDeallocated()) {
            deallocator.deallocate();
        } else {
            // Fallback cleanup if not registered with DeallocatorService
            if (log.isTraceEnabled()) {
                log.trace("Fallback cleanup for unregistered OpaqueNDArrayArr");
            }
            if (!isNull()) {
                deallocate();
                setNull();
            }
            parentArrays = null;
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
}
