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
package org.nd4j.linalg.api.memory;

import org.nd4j.common.primitives.AtomicDouble;
import org.nd4j.common.primitives.CounterMap;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.memory.enums.MemoryKind;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicLong;

public class WorkspaceAllocationsTracker {

    private Map<MemoryKind,AtomicLong> bytesTracked = new HashMap<>();
    private Map<MemoryKind,AtomicLong> pinnedBytesTracked = new HashMap<>();
    private Map<MemoryKind,AtomicLong> spilledBytesTracked = new HashMap<>();

    private Map<DataType, CounterMap<Long,Long>> pinnedTypeCounts = new HashMap<>();

    private Map<DataType, CounterMap<Long,Long>> dataTypeCounts = new HashMap<>();

    private Map<DataType, CounterMap<Long,Long>> spilledTypeCounts = new HashMap<>();


    public WorkspaceAllocationsTracker() {
        Arrays.stream(DataType.values()).forEach(dataType -> {
            dataTypeCounts.put(dataType,new CounterMap<>());
            spilledTypeCounts.put(dataType,new CounterMap<>());
            pinnedTypeCounts.put(dataType,new CounterMap<>());

        });

        Arrays.stream(MemoryKind.values()).forEach(memoryKind -> {
            bytesTracked.put(memoryKind,new AtomicLong(0));
            spilledBytesTracked.put(memoryKind,new AtomicLong(0));
            pinnedBytesTracked.put(memoryKind,new AtomicLong(0));
        });

    }

    public long currentSpilledBytes(MemoryKind memoryKind) {
        return spilledBytesTracked.get(memoryKind).get();
    }

    public CounterMap<Long,Long> currentDataTypeSpilledCount(DataType toCount) {
        return spilledTypeCounts.get(toCount);
    }

    public long currentPinnedBytes(MemoryKind memoryKind) {
        return pinnedBytesTracked.get(memoryKind).get();
    }

    public long totalAllocationCount() {
     return sumAll(dataTypeCounts);
    }

    public long totalSpilledAllocationCount() {
        return sumAll(spilledTypeCounts);
    }

    public long totalPinnedAllocationCount() {
        return sumAll(pinnedTypeCounts);
    }

    private long sumAll(Map<DataType,CounterMap<Long,Long>> toSum) {
        AtomicDouble count = new AtomicDouble(0);
        toSum.keySet().forEach(dataType -> {
            toSum.get(dataType).getIterator().forEachRemaining(iter -> {
                count.addAndGet(pinnedTypeCounts.get(dataType).getCount(iter.getFirst(),iter.getSecond()));
            });
        });

        return count.longValue();
    }

    public CounterMap<Long,Long> currentDataTypePinnedCount(DataType toCount) {
        return pinnedTypeCounts.get(toCount);
    }


    public long currentBytes(MemoryKind memoryKind) {
        return bytesTracked.get(memoryKind).get();
    }

    public CounterMap<Long,Long> currentDataTypeCount(DataType toCount) {
        return dataTypeCounts.get(toCount);
    }


    /**
     * Allocate bytes in the workspace tracking
     * @param dataType the data type to add
     * @param memoryKind  the kind of memory to add allocation for
     * @param bytes the bytes to add to the workspace
     */
    public void allocate(DataType dataType, MemoryKind memoryKind,long size, long bytes) {
        dataTypeCounts.get(dataType).incrementCount(size,bytes,1.0);
        bytesTracked.get(memoryKind).addAndGet(bytes);
    }

    public void deallocate(MemoryKind memoryKind,long bytes) {
        bytesTracked.get(memoryKind).addAndGet(-bytes);
    }


    /**
     * Allocate pinned bytes in the workspace tracking
     * @param dataType the data type to add
     * @param memoryKind  the kind of memory to add allocation for
     * @param bytes the bytes to add to the workspace
     */
    public void allocatePinned(DataType dataType, MemoryKind memoryKind,long size, long bytes) {
        pinnedTypeCounts.get(dataType).incrementCount(size,bytes,1.0);
        pinnedBytesTracked.get(memoryKind).addAndGet(bytes);
    }

    public void deallocatePinned(MemoryKind memoryKind,long bytes) {
        pinnedBytesTracked.get(memoryKind).addAndGet(-bytes);
    }


    /**
     * Allocate spilled bytes in the workspace tracking
     * @param dataType the data type to add
     * @param memoryKind  the kind of memory to add allocation for
     * @param bytes the bytes to add to the workspace
     */
    public void allocateSpilled(DataType dataType, MemoryKind memoryKind,long size, long bytes) {
        spilledTypeCounts.get(dataType).incrementCount(size,bytes,1.0);
        spilledBytesTracked.get(memoryKind).addAndGet(bytes);
    }

    public void deallocateSpilled(MemoryKind memoryKind,long bytes) {
        spilledBytesTracked.get(memoryKind).addAndGet(-bytes);
    }

}
