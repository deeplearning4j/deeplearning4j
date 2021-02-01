/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
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

/**
 * @author raver119@gmail.com
 */
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
    }

    @Override
    public DataBuffer getConstantBuffer(int[] array, DataType dataType) {
        ArrayDescriptor descriptor = new ArrayDescriptor(array, dataType);

        if (!buffersCache.containsKey(descriptor)) {
            DataBuffer buffer = Nd4j.createTypedBufferDetached(array, dataType);

            if (counter.get() < MAX_ENTRIES) {
                counter.incrementAndGet();
                buffersCache.put(descriptor, buffer);

                bytes.addAndGet(array.length * Nd4j.sizeOfDataType(dataType));
                AllocationsTracker.getInstance().markAllocated(AllocationKind.CONSTANT, 0, array.length * Nd4j.sizeOfDataType(dataType));
            }
            return buffer;
        }

        return buffersCache.get(descriptor);
    }

    @Override
    public DataBuffer getConstantBuffer(boolean[] array, DataType dataType) {
        ArrayDescriptor descriptor = new ArrayDescriptor(array, dataType);

        if (!buffersCache.containsKey(descriptor)) {
            DataBuffer buffer = Nd4j.createTypedBufferDetached(array, dataType);

            if (counter.get() < MAX_ENTRIES) {
                counter.incrementAndGet();
                buffersCache.put(descriptor, buffer);

                bytes.addAndGet(array.length * Nd4j.sizeOfDataType(dataType));
                AllocationsTracker.getInstance().markAllocated(AllocationKind.CONSTANT, 0, array.length * Nd4j.sizeOfDataType(dataType));
            }
            return buffer;
        }

        return buffersCache.get(descriptor);
    }

    @Override
    public DataBuffer getConstantBuffer(double[] array, DataType dataType) {
        ArrayDescriptor descriptor = new ArrayDescriptor(array, dataType);

        if (!buffersCache.containsKey(descriptor)) {
            DataBuffer buffer = Nd4j.createTypedBufferDetached(array, dataType);

            if (counter.get() < MAX_ENTRIES) {
                counter.incrementAndGet();
                buffersCache.put(descriptor, buffer);

                bytes.addAndGet(array.length * Nd4j.sizeOfDataType(dataType));
                AllocationsTracker.getInstance().markAllocated(AllocationKind.CONSTANT, 0, array.length * Nd4j.sizeOfDataType(dataType));
            }
            return buffer;
        }

        return buffersCache.get(descriptor);
    }

    @Override
    public DataBuffer getConstantBuffer(float[] array, DataType dataType) {
        ArrayDescriptor descriptor = new ArrayDescriptor(array, dataType);

        if (!buffersCache.containsKey(descriptor)) {
            DataBuffer buffer = Nd4j.createTypedBufferDetached(array, dataType);

            if (counter.get() < MAX_ENTRIES) {
                counter.incrementAndGet();
                buffersCache.put(descriptor, buffer);

                bytes.addAndGet(array.length * Nd4j.sizeOfDataType(dataType));
                AllocationsTracker.getInstance().markAllocated(AllocationKind.CONSTANT, 0, array.length * Nd4j.sizeOfDataType(dataType));
            }
            return buffer;
        }

        return buffersCache.get(descriptor);
    }

    @Override
    public DataBuffer getConstantBuffer(long[] array, DataType dataType) {
        ArrayDescriptor descriptor = new ArrayDescriptor(array, dataType);

        if (!buffersCache.containsKey(descriptor)) {
            DataBuffer buffer = Nd4j.createBufferDetached(array);

            if (counter.get() < MAX_ENTRIES) {
                counter.incrementAndGet();
                buffersCache.put(descriptor, buffer);

                bytes.addAndGet(array.length * Nd4j.sizeOfDataType(dataType));
                AllocationsTracker.getInstance().markAllocated(AllocationKind.CONSTANT, 0, array.length * Nd4j.sizeOfDataType(dataType));
            }
            return buffer;
        }

        return buffersCache.get(descriptor);
    }

    @Override
    public long getCachedBytes() {
        return bytes.get();
    }
}
