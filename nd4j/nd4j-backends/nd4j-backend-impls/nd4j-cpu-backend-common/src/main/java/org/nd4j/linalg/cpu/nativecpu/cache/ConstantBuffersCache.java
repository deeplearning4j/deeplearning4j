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

package org.nd4j.linalg.cpu.nativecpu.cache;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.memory.AllocationsTracker;
import org.nd4j.linalg.api.memory.enums.AllocationKind;
import org.nd4j.linalg.cache.ArrayDescriptor;
import org.nd4j.linalg.cache.BasicConstantHandler;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Supplier;

public class ConstantBuffersCache extends BasicConstantHandler {
    protected Map<ArrayDescriptor, DataBuffer> buffersCache = new ConcurrentHashMap<>();
    private AtomicInteger counter = new AtomicInteger(0);
    private AtomicLong bytes = new AtomicLong(0);
    private static final int MAX_ENTRIES = 1000;

    /**
     * This method removes all cached constants
     */
    @Override
    public void purgeConstants() {
        buffersCache = new ConcurrentHashMap<>();
        // The entry limit applies to the new, empty cache; without the reset, a cache purged after
        // MAX_ENTRIES insertions in total would never cache again
        counter.set(0);
        AllocationsTracker.getInstance().markReleased(AllocationKind.CONSTANT, 0, bytes.getAndSet(0));
    }

    @Override
    public DataBuffer getConstantBuffer(int[] array, DataType dataType) {
        return cached(new ArrayDescriptor(array, dataType), array.length, dataType,
                () -> Nd4j.createTypedBufferDetached(array, dataType));
    }

    @Override
    public DataBuffer getConstantBuffer(boolean[] array, DataType dataType) {
        return cached(new ArrayDescriptor(array, dataType), array.length, dataType,
                () -> Nd4j.createTypedBufferDetached(array, dataType));
    }

    @Override
    public DataBuffer getConstantBuffer(double[] array, DataType dataType) {
        return cached(new ArrayDescriptor(array, dataType), array.length, dataType,
                () -> Nd4j.createTypedBufferDetached(array, dataType));
    }

    @Override
    public DataBuffer getConstantBuffer(float[] array, DataType dataType) {
        return cached(new ArrayDescriptor(array, dataType), array.length, dataType,
                () -> Nd4j.createTypedBufferDetached(array, dataType));
    }

    @Override
    public DataBuffer getConstantBuffer(long[] array, DataType dataType) {
        return cached(new ArrayDescriptor(array, dataType), array.length, dataType,
                () -> Nd4j.createTypedBufferDetached(array, dataType));
    }

    /**
     * Returns the cached buffer for this content, or creates one and caches it while the cache has room.
     * A new entry is keyed by a copy of the descriptor: the descriptor wraps the caller's array, and a key
     * over it changed whenever the caller later modified the array, after which it could match lookups for
     * the modified content and return the buffer holding the original values.
     */
    private DataBuffer cached(ArrayDescriptor descriptor, int length, DataType dataType, Supplier<DataBuffer> create) {
        DataBuffer cached = buffersCache.get(descriptor);
        if (cached != null)
            return cached;

        DataBuffer buffer = create.get();
        if (counter.get() < MAX_ENTRIES) {
            DataBuffer existing = buffersCache.putIfAbsent(descriptor.copy(), buffer);
            if (existing != null)
                return existing;

            counter.incrementAndGet();
            long size = (long) length * Nd4j.sizeOfDataType(dataType);
            bytes.addAndGet(size);
            AllocationsTracker.getInstance().markAllocated(AllocationKind.CONSTANT, 0, size);
        }
        return buffer;
    }

    @Override
    public long getCachedBytes() {
        return bytes.get();
    }
}
