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

package org.nd4j.linalg.api.memory.deallocation;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.memory.Deallocatable;
import org.nd4j.linalg.api.memory.Deallocator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.nativeblas.OpaqueNDArrayArr;

/**
 * Deallocator for OpaqueNDArrayArr instances.
 * This class integrates OpaqueNDArrayArr with the DeallocatorService,
 * ensuring reliable cleanup and maintaining parent INDArray references
 * to prevent use-after-free issues.
 *
 * <p>The deallocator holds references to parent INDArrays to ensure they
 * remain alive for the lifetime of the OpaqueNDArrayArr, since the array
 * contains pointers to OpaqueNDArray instances owned by those parents.</p>
 *
 * <p>When an OpaqueNDArrayArr is created, an instance of this deallocator
 * is registered with the DeallocatorService, which will call deallocate()
 * when the Java object becomes unreachable.</p>
 *
 * @author Adam Gibson
 * @see DeallocatorService
 * @see OpaqueNDArrayArr
 * @see OpaqueNDArrayDeallocator
 */
@Slf4j
public class OpaqueNDArrayArrDeallocator implements Deallocatable, Deallocator {
    private INDArray[] parentArrays;
    private OpaqueNDArrayArr arrayArr;
    private final long uniqueId;
    private final int targetDevice;
    private volatile boolean deallocated = false;
    private volatile boolean constant = false;

    /**
     * Creates a new deallocator for the given OpaqueNDArrayArr.
     *
     * @param arrayArr The OpaqueNDArrayArr to manage
     * @param parentArrays The parent INDArrays whose lifetime must match the OpaqueNDArrayArr
     * @param uniqueId Unique identifier for tracking
     * @param targetDevice The device this array is allocated on
     */
    public OpaqueNDArrayArrDeallocator(OpaqueNDArrayArr arrayArr, INDArray[] parentArrays, 
                                       long uniqueId, int targetDevice) {
        if (arrayArr == null) {
            throw new IllegalArgumentException("OpaqueNDArrayArr cannot be null");
        }
        if (parentArrays == null) {
            throw new IllegalArgumentException("parentArrays cannot be null");
        }
        this.arrayArr = arrayArr;
        this.parentArrays = parentArrays;
        this.uniqueId = uniqueId;
        this.targetDevice = targetDevice;
    }

    @Override
    public void deallocate() {
        if (deallocated) {
            return;
        }

        synchronized (this) {
            if (deallocated) {
                return;
            }

            try {
                if (arrayArr != null && !arrayArr.isNull()) {
                    if (log.isTraceEnabled()) {
                        log.trace("Deallocating OpaqueNDArrayArr with uniqueId: {} (parent count: {})", 
                                uniqueId, parentArrays != null ? parentArrays.length : 0);
                    }
                    
                    // Note: OpaqueNDArrayArr is a PointerPointer, it doesn't own the individual
                    // OpaqueNDArray pointers (those are owned by parent INDArrays).
                    // We just need to release our reference and let the parent arrays be GC'd.
                    // The PointerPointer itself will be cleaned up by JavaCPP.
                    arrayArr.deallocate();
                    arrayArr.setNull();
                }
            } catch (Exception e) {
                log.error("Error deallocating OpaqueNDArrayArr with uniqueId: " + uniqueId, e);
            } finally {
                arrayArr = null;
                parentArrays = null;
                deallocated = true;
            }
        }
    }

    @Override
    public long getUniqueId() {
        return uniqueId;
    }

    @Override
    public Deallocator deallocator() {
        return this;
    }

    @Override
    public int targetDevice() {
        return targetDevice;
    }

    @Override
    public boolean isConstant() {
        return constant;
    }

    @Override
    public void setConstant(boolean constant) {
        this.constant = constant;
    }

    /**
     * Returns whether this deallocator has already been invoked.
     *
     * @return true if deallocate() has been called
     */
    public boolean isDeallocated() {
        return deallocated;
    }

    /**
     * Returns the OpaqueNDArrayArr being managed (may be null if deallocated).
     *
     * @return The managed array or null
     */
    public OpaqueNDArrayArr getArrayArr() {
        return arrayArr;
    }

    /**
     * Returns the parent INDArrays being kept alive (may be null if deallocated).
     *
     * @return The parent arrays or null
     */
    public INDArray[] getParentArrays() {
        return parentArrays;
    }
}
