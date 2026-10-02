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

import org.nd4j.linalg.api.memory.Deallocatable;
import org.nd4j.linalg.api.memory.Deallocator;
import org.nd4j.linalg.factory.Nd4j;

import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * One unit of a shared count, given back exactly once: by {@link #release()}, or through the
 * DeallocatorService once its holder, the only object referencing it, has been collected.
 */
public final class HeldCount implements Deallocatable {

    private final long id = Nd4j.getDeallocatorService().nextValue();
    private final int device = Nd4j.getAffinityManager().getDeviceForCurrentThread();
    private final Release release;

    private HeldCount(AtomicInteger count) {
        this.release = new Release(count);
    }

    /** Increments {@code count} and returns the unit its holder now owns. */
    public static HeldCount acquire(AtomicInteger count) {
        HeldCount held = new HeldCount(count);
        count.incrementAndGet();
        try {
            Nd4j.getDeallocatorService().pickObject(held);
        } catch (RuntimeException e) {
            held.release.deallocate();
            throw e;
        }
        return held;
    }

    /** Gives the unit back; later calls, and the release at collection, do nothing. */
    public void release() {
        release.deallocate();
        Nd4j.getDeallocatorService().getReferenceMap().remove(id);
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

    /** Shared by the unit and its collection-time release; never references the unit. */
    private static final class Release implements Deallocator {
        private final AtomicInteger count;
        private final AtomicBoolean released = new AtomicBoolean();

        private Release(AtomicInteger count) {
            this.count = count;
        }

        @Override
        public void deallocate() {
            if (released.compareAndSet(false, true)) {
                count.decrementAndGet();
            }
        }

        @Override
        public boolean isConstant() {
            return false;
        }
    }
}
