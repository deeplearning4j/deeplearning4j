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

import lombok.Data;
import org.nd4j.linalg.api.memory.Deallocatable;
import org.nd4j.linalg.api.memory.Deallocator;
import org.nd4j.linalg.factory.Nd4j;

import java.lang.ref.PhantomReference;
import java.lang.ref.ReferenceQueue;

@Data
public class DeallocatableReference extends PhantomReference<Deallocatable> {
    private long id;
    private Deallocator deallocator;
    private final DeallocatorService service;
    private final long timestamp = System.currentTimeMillis();

    public DeallocatableReference(Deallocatable referent, ReferenceQueue<? super Deallocatable> q) {
        this(referent, q, Nd4j.getDeallocatorService());
    }

    /**
     * Creates a reference owned by the supplied service. Owner-scoped registration
     * uses this constructor so listener handling never consults the process-primary
     * backend.
     */
    public DeallocatableReference(Deallocatable referent,
                                  ReferenceQueue<? super Deallocatable> q,
                                  DeallocatorService service) {
        super(referent, q);
        if (service == null) {
            throw new IllegalArgumentException("DeallocatorService cannot be null");
        }

        this.id = referent.getUniqueId();
        this.deallocator = referent.deallocator();
        this.service = service;
        if (!service.getListeners().isEmpty()) {
            service.registerDeallocatbleToListener(this);
        }
    }

    /**
     * Get the timestamp when this reference was created.
     * @return timestamp in milliseconds
     */
    public long getTimestamp() {
        return timestamp;
    }

    /**
     * Get the number of bytes associated with this reference.
     * @return byte count
     */
    public long getBytes() {
        return deallocator != null ? deallocator.getBytes() : 0;
    }

    public void deallocate() {
        if (deallocator == null) {
            return;
        }

        // Constant buffers (like shape info) should never be deallocated.
        // Instead of throwing an exception which could crash the deallocator thread,
        // silently return. The buffer will remain in memory, which is the intended
        // behavior for constant/cached buffers.
        if(deallocator.isConstant()) {
            return;
        }

        // During JVM shutdown, skip native deallocation to avoid calling free()
        // on potentially corrupted heap metadata. The OS reclaims all process
        // memory on exit.
        if (DeallocatorService.getShutdownInProgress().get()) {
            return;
        }

        if (!service.getListeners().isEmpty()) {
            service.registerDeallocatbleToListener(this);
        }
        deallocator.deallocate();
    }

}
