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
package org.eclipse.deeplearning4j.nd4j.autodiff.samediff;

import org.bytedeco.javacpp.Pointer;
import org.nd4j.linalg.api.memory.Deallocatable;
import org.nd4j.linalg.api.memory.Deallocator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.NativeOps;

/**
 * Owns one native plan cache of a SameDiff graph and frees it when the graph is closed or, for a
 * graph dropped without close(), once the graph has been collected.
 *
 * <p>The cache's plans hold device memory (slot buffers, captured graphs, a cuBLAS workspace each),
 * so a graph that is never closed has to give it back the way its arrays do: through the
 * DeallocatorService, when this owner becomes unreachable. The graph keeps its owners reachable for
 * as long as it lives, and every borrower of the cache (executors and their execution bindings, which
 * hold plan leases) references the graph. Once the owner is collected no borrower is left, so the
 * leases they never released are dropped with the cache.</p>
 *
 * <p>{@link #free()} keeps the stricter rule for a live graph: a cache a borrower still leases is not
 * destroyed, and its release at collection stays armed.</p>
 */
public final class NativePlanCacheOwner implements Deallocatable {

    private final long id = Nd4j.getDeallocatorService().nextValue();
    private final int device;
    private final Release release;

    private NativePlanCacheOwner(NativeOps nativeOps, Pointer cache, int device) {
        this.device = device;
        this.release = new Release(nativeOps, cache);
    }

    /**
     * Creates a native plan cache on the current thread's device and registers its release at
     * collection. When memory is low, the caches of graphs dropped without close() are reclaimed
     * first ({@link LowMemoryReclaim}).
     */
    public static NativePlanCacheOwner create(NativeOps nativeOps) {
        int device = Nd4j.getAffinityManager().getDeviceForCurrentThread();
        LowMemoryReclaim.reclaimIfLow(nativeOps, device);
        Pointer cache = nativeOps.createNativePlanCache();
        if (cache == null || cache.isNull()) {
            throw new IllegalStateException("createNativePlanCache returned null — native DSP cache is unavailable");
        }
        NativePlanCacheOwner owner = new NativePlanCacheOwner(nativeOps, cache, device);
        try {
            Nd4j.getDeallocatorService().pickObject(owner);
        } catch (RuntimeException e) {
            owner.release.free();
            throw e;
        }
        return owner;
    }

    /** The native cache handle, or null once the cache was destroyed. */
    public Pointer cache() {
        return release.cache();
    }

    /**
     * Frees the cache of a graph being closed.
     *
     * @return true when the cache was destroyed; false when a borrower's lease kept it, in which
     *         case the cache is freed once this owner is collected
     */
    public boolean free() {
        if (!release.free()) {
            return false;
        }
        Nd4j.getDeallocatorService().getReferenceMap().remove(id);
        return true;
    }

    @Override
    public long getUniqueId() {
        return id;
    }

    @Override
    public Deallocator deallocator() {
        return release;
    }

    @Override
    public int targetDevice() {
        return device;
    }

    /** The cache handle, shared by the owner and its collection-time release; never references the owner. */
    private static final class Release implements Deallocator {
        private final NativeOps nativeOps;
        private Pointer cache;

        private Release(NativeOps nativeOps, Pointer cache) {
            this.nativeOps = nativeOps;
            this.cache = cache;
        }

        synchronized Pointer cache() {
            return cache;
        }

        /** Destroys the cache unless a borrower leases one of its plans; true once it is gone. */
        synchronized boolean free() {
            if (cache == null) {
                return true;
            }
            if (nativeOps.freeNativePlanCache(cache) == 0) {
                return false;
            }
            cache = null;
            return true;
        }

        /** The owner was collected: so was every borrower, and their leases go with the cache. */
        @Override
        public synchronized void deallocate() {
            if (cache == null) {
                return;
            }
            nativeOps.freeAbandonedNativePlanCache(cache);
            cache = null;
        }

        @Override
        public boolean isConstant() {
            return false;
        }
    }
}
