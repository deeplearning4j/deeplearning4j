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

package org.nd4j.jita.allocator.impl;

import lombok.Getter;
import lombok.NonNull;
import lombok.Setter;
import lombok.val;
import org.bytedeco.javacpp.Pointer;
import org.nd4j.jita.allocator.enums.AllocationStatus;
import org.nd4j.jita.allocator.garbage.GarbageBufferReference;
import org.nd4j.jita.allocator.pointers.cuda.cudaEvent_t;
import org.nd4j.jita.allocator.time.TimeProvider;
import org.nd4j.jita.allocator.time.providers.OperativeProvider;
import org.nd4j.linalg.api.buffer.BaseDataBuffer;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.context.CudaContext;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;
import org.nd4j.nativeblas.OpaqueDataBuffer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.atomic.AtomicBoolean;

/**
 * This class describes top-level allocation unit.
 * Every buffer passed into CUDA wii have allocation point entry, describing allocation state.
 *
 * @author raver119@gmail.com
 */
// DO NOT EVER MAKE THIS CLASS SERIALIZABLE.
public class AllocationPoint {
    private static Logger log = LoggerFactory.getLogger(AllocationPoint.class);

    @Getter
    private OpaqueDataBuffer ptrDataBuffer;

    @Getter
    @Setter
    private Long objectId;

    @Getter
    @Setter
    private Long bucketId;

    @Getter
    @Setter
    private boolean isAttached = false;

    @Getter
    @Setter
    private volatile boolean released = false;

    // thread safety is guaranteed by allocLock
    private AllocationStatus allocationStatus = AllocationStatus.UNDEFINED;

    private transient TimeProvider timeProvider = new OperativeProvider();

    // corresponding access times in TimeProvider quants
    private long accessHostRead = 0L;
    private long accessDeviceRead = 0L;

    private long accessHostWrite = 0L;
    private long accessDeviceWrite = 0L;

    protected static final NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
/*
    @Getter
    @Setter
    protected volatile cudaEvent_t writeLane;

    @Getter
    protected Queue<cudaEvent_t> readLane = new ConcurrentLinkedQueue<>();
*/
    @Getter
    @Setter
    private boolean constant;

    /*
     device, where memory was/will be allocated.
    Valid integer >= 0 is deviceId, null for undefined
    */
    private volatile int deviceId;

    private long bytes;

    public AllocationPoint(@NonNull OpaqueDataBuffer opaqueDataBuffer, long bytes) {
        ptrDataBuffer = opaqueDataBuffer;
        this.bytes = bytes;
        objectId = Nd4j.getDeallocatorService().nextValue();
    }

    public void setPointers(Pointer primary, Pointer special, long numberOfElements) {
        NativeOpsHolder.getInstance().getDeviceNativeOps().dbSetPrimaryBuffer(ptrDataBuffer, primary, numberOfElements);
        NativeOpsHolder.getInstance().getDeviceNativeOps().dbSetSpecialBuffer(ptrDataBuffer, special, numberOfElements);
    }

    public int getDeviceId() {
        return ptrDataBuffer.deviceId();
    }

    public void setDeviceId(int deviceId) {
        NativeOpsHolder.getInstance().getDeviceNativeOps().dbSetDeviceId(ptrDataBuffer, deviceId);
    }

    private AtomicBoolean enqueued = new AtomicBoolean(false);

    @Getter
    @Setter
    private cudaEvent_t lastWriteEvent;

    @Getter
    @Setter
    private cudaEvent_t lastReadEvent;

    private volatile CudaContext currentContext;

    public boolean isEnqueued() {
        return enqueued.get();
    }

    public void markEnqueued(boolean reallyEnqueued) {
        enqueued.set(reallyEnqueued);
    }

    public CudaContext getCurrentContext() {
        synchronized (this) {
            return currentContext;
        }
    }

    public void setCurrentContext(CudaContext context) {
        synchronized (this) {
            this.currentContext = context;
        }
    }

    public long getNumberOfBytes() {
        return bytes;
    }

    /*
    public void addReadLane(cudaEvent_t event) {
        readLane.add(event);
    }
    */

    /**
     * This method stores WeakReference to original BaseCudaDataBuffer
     *
     * @param buffer
     */
    public void attachBuffer(@NonNull BaseDataBuffer buffer) {
        //originalDataBufferReference = new WeakReference<BaseDataBuffer>(buffer);
    }

    public void attachReference(GarbageBufferReference reference) {
        //garbageBufferReference = reference;
    }

    /**
     * This method returns previously stored BaseCudaDataBuffer instance
     *
     * PLEASE NOTE: Return value CAN be null
     *
     * @return
     */
    public DataBuffer getBuffer() {
        //if (originalDataBufferReference != null) {
        //    return originalDataBufferReference.get();
        //} else
        return null;
    }

    /**
     * This method returns current AllocationStatus for this point
     * @return
     */
    public AllocationStatus getAllocationStatus() {
        return allocationStatus;
    }

    /**
     * This method sets specified AllocationStatus for this point
     * @param status
     */
    public void setAllocationStatus(@NonNull AllocationStatus status) {
        allocationStatus = status;
    }

    /**
     * This method returns CUDA pointer object for this allocation.
     * It can be either device pointer or pinned memory pointer, or null.
     *
     * @return
     */
    public Pointer getDevicePointer() {
        return NativeOpsHolder.getInstance().getDeviceNativeOps().dbSpecialBuffer(ptrDataBuffer);
    }

    /**
     * This method returns CUDA pointer object for this allocation.
     * It can be either device pointer or pinned memory pointer, or null.
     *
     * @return
     */
    public Pointer getHostPointer() {
        return NativeOpsHolder.getInstance().getDeviceNativeOps().dbPrimaryBuffer(ptrDataBuffer);
    }


    public synchronized void tickDeviceRead() {
        NativeOpsHolder.getInstance().getDeviceNativeOps().dbTickDeviceRead(ptrDataBuffer);
    }

    /**
     * Returns time, in milliseconds, when this point was accessed on device side
     *
     * @return
     */
    public synchronized long getDeviceAccessTime() {
        return accessDeviceRead;
    }

    /**
     * Returns time when point was written on device last time
     *
     * @return
     */
    public synchronized long getDeviceWriteTime() {
        return accessDeviceWrite;
    }

    public synchronized void tickHostRead() {
        NativeOpsHolder.getInstance().getDeviceNativeOps().dbTickHostRead(ptrDataBuffer);
    }

    /**
     * This method sets time when this point was changed on device
     *
     */
    public synchronized void tickDeviceWrite() {
        NativeOpsHolder.getInstance().getDeviceNativeOps().dbTickDeviceWrite(ptrDataBuffer);
    }

    /**
     * This method sets time when this point was changed on host
     */
    public synchronized void tickHostWrite() {
        NativeOpsHolder.getInstance().getDeviceNativeOps().dbTickHostWrite(ptrDataBuffer);
    }

    /**
     * This method returns, if host side has actual copy of data
     *
     * @return true, if data is actual, false otherwise
     */
    public synchronized boolean isActualOnHostSide() {
        val s = NativeOpsHolder.getInstance().getDeviceNativeOps().dbLocality(ptrDataBuffer);
        return s <= 0;
    }

    /**
     * This method returns, if device side has actual copy of data
     *
     * @return
     */
    public synchronized boolean isActualOnDeviceSide() {
        val s = NativeOpsHolder.getInstance().getDeviceNativeOps().dbLocality(ptrDataBuffer);
        return s >= 0;
    }

    /**
     * This method sets device access time equal to host write time
     */
    public synchronized void tickDeviceToHost() {
        accessDeviceRead = (accessHostRead);
    }

    @Override
    public String toString() {
        return "AllocationPoint{" + "deviceId=" + deviceId + ", objectId=" + objectId + "}";
    }
}
