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

package org.nd4j.linalg.cpu.nativecpu;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.memory.AllocationsTracker;
import org.nd4j.linalg.api.memory.enums.AllocationKind;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.api.shape.options.ArrayOptionsHelper;
import org.nd4j.common.primitives.Pair;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.BaseShapeInfoProvider;

import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

@Slf4j
public class DirectShapeInfoProvider extends BaseShapeInfoProvider {
    private Map<LongShapeDescriptor, Pair<DataBuffer, long[]>> longCache = new ConcurrentHashMap<>();
    private AtomicInteger counter = new AtomicInteger(0);
    private static final int MAX_ENTRIES = 1000;

    @Override
    public Pair<DataBuffer, long[]> createShapeInformation(long[] shapeInfo) {
        // Route through the cached path to avoid workspace allocation for shape info
        long[] shape = Shape.shape(shapeInfo);
        long[] stride = Shape.stride(shapeInfo);
        long ews = Shape.elementWiseStride(shapeInfo);
        char order = Shape.order(shapeInfo);
        long extras = Shape.extras(shapeInfo);
        return createShapeInformation(shape, stride, ews, order, extras);
    }

    public Pair<DataBuffer, long[]> createShapeInformation(long[] shape, long[] stride,  long elementWiseStride, char order, DataType dataType) {
        long extras = 0;
        extras = ArrayOptionsHelper.setOptionBit(extras, dataType);
        return createShapeInformation(shape, stride, elementWiseStride, order, extras);
    }

    @Override
    public Pair<DataBuffer, long[]> createShapeInformation(long[] shape, long[] stride,  long elementWiseStride, char order, long extras) {
        // We enforce offset to 0 in shapeBuffer, since we need it for cache efficiency + we don't actually use offset value @ native side
        // We also enforce elementWiseStride = 0
        if (elementWiseStride < 0)
            elementWiseStride = 0;

        LongShapeDescriptor descriptor = new LongShapeDescriptor(shape, stride, 0, elementWiseStride, order, extras);
        if (!longCache.containsKey(descriptor)) {
            if (counter.get() < MAX_ENTRIES) {
                synchronized (this) {
                    if (!longCache.containsKey(descriptor)) {
                        counter.incrementAndGet();
                        Pair<DataBuffer, long[]> buffer;
                        // Scope out of any active workspace so the cached shape info buffer
                        // is allocated from regular memory, not from workspace memory that gets recycled
                        try (MemoryWorkspace ws = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
                            buffer = super.createShapeInformation(shape, stride, elementWiseStride, order, extras);
                        }
                        buffer.getFirst().setConstant(true);
                        longCache.put(descriptor, buffer);

                        bytes.addAndGet(buffer.getFirst().length() * 8 * 2);
                        AllocationsTracker.getInstance().markAllocated(AllocationKind.CONSTANT,0, buffer.getFirst().length() * 8 * 2);
                        return buffer;
                    } else
                        return longCache.get(descriptor);
                }
            } else {
                // Cache is full, but we MUST still mark shape buffers as constant!
                // Without this, the DeallocatorService will free the shape buffer while
                // NDArrays are still using it, causing use-after-free crashes.
                Pair<DataBuffer, long[]> buffer;
                // Scope out of any active workspace so the constant shape info buffer
                // is allocated from regular memory, not from workspace memory that gets recycled
                try (MemoryWorkspace ws = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
                    buffer = super.createShapeInformation(shape, stride, elementWiseStride, order, extras);
                }
                buffer.getFirst().setConstant(true);
                return buffer;
            }
        }

        return longCache.get(descriptor);
    }

    @Override
    public void purgeCache() {
    }
}
