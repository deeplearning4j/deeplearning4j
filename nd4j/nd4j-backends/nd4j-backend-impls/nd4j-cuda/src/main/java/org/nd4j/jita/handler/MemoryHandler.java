/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.nd4j.jita.handler;

import com.google.common.collect.Table;
import org.bytedeco.javacpp.Pointer;
import org.nd4j.jita.allocator.Allocator;
import org.nd4j.jita.allocator.context.ContextPool;
import org.nd4j.jita.allocator.context.ExternalContext;
import org.nd4j.jita.allocator.enums.AllocationStatus;
import org.nd4j.jita.allocator.impl.AllocationPoint;
import org.nd4j.jita.allocator.impl.AllocationShape;
import org.nd4j.jita.allocator.pointers.PointersPair;
import org.nd4j.jita.conf.Configuration;
import org.nd4j.jita.flow.FlowController;
import org.nd4j.jita.memory.MemoryProvider;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.jcublas.context.CudaContext;

import java.util.Set;

/**
 * MemoryHandler interface describes methods for data access
 *
 * @author raver119@gmail.com
 */
public interface MemoryHandler {

    /**
     * This method gets called from Allocator, during Allocator/MemoryHandler initialization
     *
     * @param configuration
     * @param allocator
     */
    void init(Configuration configuration, Allocator allocator);

    /**
     * This method returns if this MemoryHandler instance is device-dependant (i.e. CUDA)
     *
     * @return TRUE if dependant, FALSE otherwise
     */
    boolean isDeviceDependant();


    CudaContext getCudaContext();


    /**
     * This method removes AllocationPoint from corresponding device/host trackers
     * @param point
     * @param location
     */
    void forget(AllocationPoint point, AllocationStatus location);

    /**
     * This method causes memory synchronization on host side.
     *  Viable only for Device-dependant MemoryHandlers
     *
     * @param threadId
     * @param deviceId
     * @param point
     */
    void synchronizeThreadDevice(Long threadId, Integer deviceId, AllocationPoint point);

    /**
     * Allocate specified memory chunk on specified device/host
     *
     * @param targetMode valid arguments are DEVICE, ZERO
     * @return
     */
    PointersPair alloc(AllocationStatus targetMode, AllocationPoint point, AllocationShape shape, boolean initialize);

    /**
     * This method checks if specified device has free memory
     *
     * @return
     */
    boolean pingDeviceForFreeMemory(Integer deviceId, long requiredMemory);

    /**
     *  Relocates specific chunk of memory from one storage to another
     *
     * @param currentStatus
     * @param targetStatus
     * @param point
     */
    void relocate(AllocationStatus currentStatus, AllocationStatus targetStatus, AllocationPoint point,
                    AllocationShape shape, CudaContext context);

    /**
     * Copies memory from device to host, if needed.
     * Device copy is preserved as is.
     *
     * @param point
     */
    void copyback(AllocationPoint point, AllocationShape shape);


    /**
     * Copies memory from host buffer to device.
     * Host copy is preserved as is.
     *
     * @param point
     */
    void copyforward(AllocationPoint point, AllocationShape shape);

    /**
     * Copies memory from device to zero-copy memory
     *
     * @param point
     * @param shape
     */
    void fallback(AllocationPoint point, AllocationShape shape);

    /**
     * This method frees memory chunk specified by pointer
     *
     * @param point
     */
    void free(AllocationPoint point, AllocationStatus target);

    /**
     * This method returns initial allocation location. So, it can be HOST, or DEVICE if environment allows that.
     *
     * @return
     */
    AllocationStatus getInitialLocation();

    /**
     * This method initializes specific device for current thread
     */
    void initializeDevice(Long threadId, Integer deviceId);

    /**
     *  Synchronous version of memcpy.
     *
     *
     * @param dstBuffer
     * @param srcPointer
     * @param length
     * @param dstOffset
     */
    void memcpyBlocking(DataBuffer dstBuffer, Pointer srcPointer, long length, long dstOffset);

    /**
     * Asynchronous version of memcpy
     *
     * PLEASE NOTE: This is device-dependent method, if it's not supported in your environment, blocking call will be used instead.
     *
     * @param dstBuffer
     * @param srcPointer
     * @param length
     * @param dstOffset
     */
    void memcpyAsync(DataBuffer dstBuffer, Pointer srcPointer, long length, long dstOffset);

    void memcpySpecial(DataBuffer dstBuffer, Pointer srcPointer, long length, long dstOffset);

    void memcpyDevice(DataBuffer dstBuffer, Pointer srcPointer, long length, long dstOffset, CudaContext context);

    /**
     * Synchronous version of memcpy
     *
     * @param dstBuffer
     * @param srcBuffer
     */
    void memcpy(DataBuffer dstBuffer, DataBuffer srcBuffer);

    /**
     * PLEASE NOTE: Specific implementation, on systems without special devices can return HostPointer here
     * @return
     */
    Pointer getDevicePointer(DataBuffer buffer, CudaContext context);

    /**
     * PLEASE NOTE: This method always returns pointer valid within OS memory space
     * @return
     */
    Pointer getHostPointer(DataBuffer buffer);


    /**
     * This method returns total amount of memory allocated within system
     *
     * @return
     */
    Table<AllocationStatus, Integer, Long> getAllocationStatistics();

    /**
     * This method returns total amount of host memory allocated within this MemoryHandler
     *
     * @return
     */
    long getAllocatedHostMemory();

    /**
     * This method returns total amount of memory allocated at specified device
     *
     * @param device
     * @return
     */
    long getAllocatedDeviceMemory(Integer device);

    /**
     * This method returns number of allocated objects within specific bucket
     *
     * @param bucketId
     * @return
     */
    long getAllocatedHostObjects(Long bucketId);

    /**
     * This method returns total number of allocated objects in host memory
     * @return
     */
    long getAllocatedHostObjects();

    /**
     * This method returns total number of object allocated on specified device
     *
     * @param deviceId
     * @return
     */
    long getAllocatedDeviceObjects(Integer deviceId);

    /**
     * This method returns set of allocation tracking IDs for specific device
     *
     * @param deviceId
     * @return
     */
    Set<Long> getDeviceTrackingPoints(Integer deviceId);

    /**
     * This method returns sets of allocation tracking IDs for specific bucket
     *
     * @param bucketId
     * @return
     */
    Set<Long> getHostTrackingPoints(Long bucketId);

    /**
     * This method removes specific previously allocated object from device memory
     *
     * @param threadId
     * @param deviceId
     * @param objectId
     * @param point
     * @param copyback
     */
    void purgeDeviceObject(Long threadId, Integer deviceId, Long objectId, AllocationPoint point, boolean copyback);

    /**
     * This method removes specific previously allocated object from host memory
     *
     * @param bucketId
     * @param objectId
     * @param point
     * @param copyback
     */
    void purgeZeroObject(Long bucketId, Long objectId, AllocationPoint point, boolean copyback);

    /**
     * This method returns set of available devices
     * @return
     */
    Set<Integer> getAvailableDevices();

    /**
     * This method returns device ID for current thread
     *
     * @return
     */
    Integer getDeviceId();

    /** Returns {@link #getDeviceId()} wrapped as a {@link Pointer}. */
    Pointer getDeviceIdPointer();

    /**
     * This method returns ExternalContext wrapper (if applicable)
     * @return
     */
    ExternalContext getDeviceContext();

    void registerAction(CudaContext context, INDArray result, INDArray... operands);

    FlowController getFlowController();

    MemoryProvider getMemoryProvider();

    boolean promoteObject(DataBuffer buffer);

    void relocateObject(DataBuffer buffer);

    ContextPool getContextPool();


}
