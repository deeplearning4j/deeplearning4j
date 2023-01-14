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
import org.nd4j.linalg.api.memory.abstracts.Nd4jWorkspace;
import org.nd4j.linalg.api.memory.enums.MemoryKind;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Supplementary tracking of cross thread workspace allocations.
 *
 * @author Adam Gibson
 */
public class WorkspaceAllocationsTracker {

    private Map<MemoryKind,AtomicLong> bytesTracked = new HashMap<>();
    private Map<MemoryKind,AtomicLong> pinnedBytesTracked = new HashMap<>();
    private Map<MemoryKind,AtomicLong> spilledBytesTracked = new HashMap<>();
    private Map<MemoryKind,AtomicLong> externalBytesTracked = new HashMap<>();

    private Map<DataType, CounterMap<Long,Long>> pinnedTypeCounts = new HashMap<>();

    private Map<DataType, CounterMap<Long,Long>> dataTypeCounts = new HashMap<>();

    private Map<DataType, CounterMap<Long,Long>> spilledTypeCounts = new HashMap<>();
    private Map<DataType, CounterMap<Long,Long>> externalTypeCounts = new HashMap<>();


    public WorkspaceAllocationsTracker() {
        Arrays.stream(DataType.values()).forEach(dataType -> {
            dataTypeCounts.put(dataType,new CounterMap<>());
            spilledTypeCounts.put(dataType,new CounterMap<>());
            pinnedTypeCounts.put(dataType,new CounterMap<>());
            externalTypeCounts.put(dataType,new CounterMap<>());

        });

        Arrays.stream(MemoryKind.values()).forEach(memoryKind -> {
            bytesTracked.put(memoryKind,new AtomicLong(0));
            spilledBytesTracked.put(memoryKind,new AtomicLong(0));
            pinnedBytesTracked.put(memoryKind,new AtomicLong(0));
            externalBytesTracked.put(memoryKind,new AtomicLong(0));
        });

    }

    /**
     * Current number of bytes allocated externally for a given device.
     *
     * @see {@link Nd4jWorkspace#getNumberOfExternalAllocations()}
     * @param memoryKind the kind of memory to check for
     * @return
     */
    public long currentExternalBytes(MemoryKind memoryKind) {
        return externalBytesTracked.get(memoryKind).get();
    }

    /**
     * The current number of spilled bytes for a
     * given device.
     * @see {@link Nd4jWorkspace#getSpilledSize()}
     * @param memoryKind
     * @return
     */
    public long currentSpilledBytes(MemoryKind memoryKind) {
        return spilledBytesTracked.get(memoryKind).get();
    }


    /**
     * Current number of external allocations broken down by data type.
     * @param toCount the data type to get the counts for
     * @return
     */
    public CounterMap<Long,Long> currentDataTypeExternalCount(DataType toCount) {
        return spilledTypeCounts.get(toCount);
    }
    /**
     * Current number of spilled allocations broken down by data type.
     * @param toCount the data type to get the counts for
     * @return
     */
    public CounterMap<Long,Long> currentDataTypeSpilledCount(DataType toCount) {
        return spilledTypeCounts.get(toCount);
    }

    /**
     * Current number of pinned bytes for a given memory type
     * @see {@link Nd4jWorkspace#getPinnedSize()}
     * @param memoryKind
     * @return
     */
    public long currentPinnedBytes(MemoryKind memoryKind) {
        return pinnedBytesTracked.get(memoryKind).get();
    }


    /**
     * Get the total number of external allocations
     * @see {@link Nd4jWorkspace#getNumberOfExternalAllocations()}
     * @return
     */
    public long totalExternalAllocationCount() {
        return sumAll(externalTypeCounts);
    }

    /**
     * Get the total number of allocations across all data types
     * and workspaces
     * @see {@link Nd4jWorkspace#getPrimaryOffset()}
     * @return
     */
    public long totalAllocationCount() {
     return sumAll(dataTypeCounts);
    }

    /**
     * Get the total number of spilled allocations
     * aggregated by data type.
     *
     * @see {@link Nd4jWorkspace#getSpilledSize()}
     * @return
     */
    public long totalSpilledAllocationCount() {
        return sumAll(spilledTypeCounts);
    }

    /**
     * Get the total number of pinned allocations
     * aggregated by data type.
     * @see {@link Nd4jWorkspace#getNumberOfPinnedAllocations()}
     * @return
     */
    public long totalPinnedAllocationCount() {
        return sumAll(pinnedTypeCounts);
    }

    private long sumAll(Map<DataType,CounterMap<Long,Long>> toSum) {
        AtomicDouble count = new AtomicDouble(0);
        toSum.keySet().forEach(dataType -> {
            toSum.get(dataType).getIterator().forEachRemaining(iter -> {
                count.addAndGet(toSum.get(dataType).getCount(iter.getFirst(),iter.getSecond()));
            });
        });

        return count.longValue();
    }

    /**
     * Get the number of pinned allocations for a given data type.
     *
     * @see {@link Nd4jWorkspace#getNumberOfPinnedAllocations()}
     * @param toCount
     * @return
     */
    public CounterMap<Long,Long> currentDataTypePinnedCount(DataType toCount) {
        return pinnedTypeCounts.get(toCount);
    }


    /**
     * Get the current number of bytes allocated
     * for a given memory type
     * @param memoryKind the type of memory to get the bytes allocated for
     * @return
     */
    public long currentBytes(MemoryKind memoryKind) {
        return bytesTracked.get(memoryKind).get();
    }

    /**
     * Get the current number of allocations for a given data type
     * and requested aligned memory size.
     * @see {@link Nd4jWorkspace#alignMemory(long)}
     * @see {@link Nd4jWorkspace#getPrimaryOffset()}
     * @param toCount the data type to get the counts for
     * @return
     */
    public CounterMap<Long,Long> currentDataTypeCount(DataType toCount) {
        return dataTypeCounts.get(toCount);
    }


    /**
     * Add bytes in the workspace tracking
     * @param dataType the data type to add
     * @param memoryKind  the kind of memory to add allocation for
     * @param bytes the bytes to add to the workspace
     */
    public void allocate(DataType dataType, MemoryKind memoryKind,long size, long bytes) {
        dataTypeCounts.get(dataType).incrementCount(size,bytes,1.0);
        bytesTracked.get(memoryKind).addAndGet(bytes);
    }

    /**
     * Decrements memory count of a given kind
     * @param memoryKind the kind of memory to deallocate
     * @param bytes the bytes to add to the workspace
     */
    public void deallocate(MemoryKind memoryKind,long bytes) {
        bytesTracked.get(memoryKind).addAndGet(-bytes);
    }

    /**
     * Add external bytes in the workspace tracking
     * @param dataType the data type to add
     * @param memoryKind  the kind of memory to add allocation for
     * @param bytes the bytes to add to the workspace
     */
    public void allocateExternal(DataType dataType, MemoryKind memoryKind,long size, long bytes) {
        externalTypeCounts.get(dataType).incrementCount(size,bytes,1.0);
        externalBytesTracked.get(memoryKind).addAndGet(bytes);
    }

    /**
     * Add pinned bytes in the workspace tracking
     * @param dataType the data type to add
     * @param memoryKind  the kind of memory to add allocation for
     * @param bytes the bytes to add to the workspace
     */
    public void allocatePinned(DataType dataType, MemoryKind memoryKind,long size, long bytes) {
        pinnedTypeCounts.get(dataType).incrementCount(size,bytes,1.0);
        pinnedBytesTracked.get(memoryKind).addAndGet(bytes);
    }

    /**
     * Deallocates pinned memory
     *
     * @param memoryKind the kind of memory to deallocate
     * @param bytes the size in bytes to deallocate
     */
    public void deallocatePinned(MemoryKind memoryKind,long bytes) {
        pinnedBytesTracked.get(memoryKind).addAndGet(-bytes);
    }


    /**
     * Add spilled bytes in the workspace tracking
     * @param dataType the data type to add
     * @param memoryKind  the kind of memory to add allocation for
     * @param bytes the bytes to add to the workspace
     */
    public void allocateSpilled(DataType dataType, MemoryKind memoryKind,long size, long bytes) {
        spilledTypeCounts.get(dataType).incrementCount(size,bytes,1.0);
        spilledBytesTracked.get(memoryKind).addAndGet(bytes);
    }

    /**
     * Subtracts spilled memory
     * @param memoryKind the type of memory to
     *                   de allocate
     * @param bytes the size in bytes to deallocate
     */
    public void deallocateSpilled(MemoryKind memoryKind,long bytes) {
        spilledBytesTracked.get(memoryKind).addAndGet(-bytes);
    }

    /**
     * Subtracts external memory
     * @param memoryKind the type of memory to
     *                   de allocate
     * @param bytes the size in bytes to deallocate
     */
    public void deallocateExternal(MemoryKind memoryKind,long bytes) {
        externalBytesTracked.get(memoryKind).addAndGet(-bytes);
    }

}
