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

package org.nd4j.jita.concurrency;

import lombok.NonNull;
import lombok.val;
import org.nd4j.jita.allocator.impl.AllocationPoint;
import org.nd4j.jita.allocator.impl.AtomicAllocator;
import org.nd4j.jita.allocator.pointers.CudaPointer;
import org.nd4j.jita.conf.Configuration;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.concurrency.AffinityManager;
import org.nd4j.linalg.api.concurrency.BasicAffinityManager;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.buffer.BaseCudaDataBuffer;
import org.nd4j.linalg.jcublas.context.CudaContext;
import org.nd4j.nativeblas.NativeOpsHolder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * AffinityManager implementation for CUDA
 *
 * @author raver119@gmail.com
 */
public class CudaAffinityManager extends BasicAffinityManager {
    private static Logger logger = LoggerFactory.getLogger(CudaAffinityManager.class);

    private Map<Long, Integer> affinityMap = new ConcurrentHashMap<>();
    private AtomicInteger devPtr = new AtomicInteger(0);
    private ThreadLocal<AtomicBoolean> affiliated = new ThreadLocal<>();

    private AtomicInteger numberOfDevices = new AtomicInteger(-1);

    public CudaAffinityManager() {
        super();

    }

    /**
     * This method returns deviceId for current thread.
     *
     * If no device was assigned to this thread before this call, it'll be assinged here.
     *
     * @return
     */
    @Override
    public Integer getDeviceForCurrentThread() {
        return NativeOpsHolder.getInstance().getDeviceNativeOps().getDevice();
    }

    /**
     * This method returns deviceId for a given thread
     * @return
     */
    @Override
    public Integer getDeviceForThread(long threadId) {
        val id = affinityMap.get(threadId);
        if (id == null)
            throw new RuntimeException("Affinity for thread [" + threadId + "] wasn't defined yet");

        return id;
    }


    /**
     * This method returns device id available. Round-robin balancing used here.
     *
     * @param threadId this parameter can be anything, it's used for logging only.
     * @return
     */
    protected Integer getNextDevice(long threadId) {
        Integer device = null;
        if (!CudaEnvironment.getInstance().getConfiguration().isForcedSingleGPU() && getNumberOfDevices() > 0) {
            // simple round-robin here
            synchronized (this) {
                device = CudaEnvironment.getInstance().getConfiguration().getAvailableDevices().get(devPtr.getAndIncrement());

                // We check only for number of entries here, not their actual values
                if (devPtr.get() >= CudaEnvironment.getInstance().getConfiguration().getAvailableDevices().size())
                    devPtr.set(0);

                val t = Thread.currentThread();
                val n = t.getId() == threadId ? t.getName() : "N/A";

                logger.debug("Mapping thread [{} - {}] to device [{}], out of [{}] devices...", threadId, n, device, CudaEnvironment.getInstance().getConfiguration().getAvailableDevices().size());
            }
        } else {
            device = CudaEnvironment.getInstance().getConfiguration().getAvailableDevices().get(0);
            logger.debug("Single device is forced, mapping to device [{}]", device);
        }

        return device;
    }

    /**
     * This method returns number of available devices in system.
     *
     * Please note: returned value might be different from actual number of used devices.
     *
     * @return total number of devices
     */
    @Override
    public int getNumberOfDevices() {
        if (numberOfDevices.get() < 0) {
            synchronized (this) {
                if (numberOfDevices.get() < 1) {
                    numberOfDevices.set(NativeOpsHolder.getInstance().getDeviceNativeOps().getAvailableDevices());
                }
            }
        }

        return numberOfDevices.get();
    }

    /**
     * Utility method, to associate INDArray with specific device (backend-specific)
     *
     * @param array
     */
    @Override
    public void touch(INDArray array) {
        if (array == null)
            return;

        touch(array.data());
        touch(array.shapeInfoDataBuffer());
    }

    /**
     * Utility method, to associate INDArray with specific device (backend-specific)
     *
     * @param buffer
     */
    @Override
    public void touch(DataBuffer buffer) {
        if (buffer == null)
            return;

        AllocationPoint point = AtomicAllocator.getInstance().getAllocationPoint(buffer);

        if (point.isConstant()) {
            Nd4j.getConstantHandler().relocateConstantSpace(buffer);
        } else {
            AtomicAllocator.getInstance().getMemoryHandler().relocateObject(buffer);
        }
    }

    /**
     * This method replicates given INDArray, and places it to target device.
     *
     * @param deviceId target deviceId
     * @param array    INDArray to replicate
     * @return
     */
    @Override
    public synchronized INDArray replicateToDevice(Integer deviceId, INDArray array) {
        if (array == null)
            return null;

        // string arrays are stored in host memory only atm
        if (array.isS())
            return array.dup(array.ordering());

        if (array.isView())
            throw new UnsupportedOperationException("It's impossible to replicate View");

        val shape = array.shape();
        val stride = array.stride();
        val elementWiseStride = array.elementWiseStride();
        val ordering = array.ordering();
        val length = array.length();
        val dtype = array.dataType();
        val empty = array.isEmpty();

        // we use this call to get device memory updated
        AtomicAllocator.getInstance().getPointer(array, AtomicAllocator.getInstance().getDeviceContext());

        int currentDeviceId = getDeviceForCurrentThread();

        if (currentDeviceId != deviceId.intValue()) {
            unsafeSetDevice(deviceId);
        }


        DataBuffer newDataBuffer = replicateToDevice(deviceId, array.data());
        DataBuffer newShapeBuffer = Nd4j.getShapeInfoProvider().createShapeInformation(shape, stride, elementWiseStride, ordering, dtype, empty).getFirst();
        INDArray result = Nd4j.createArrayFromShapeBuffer(newDataBuffer, newShapeBuffer);

        if (currentDeviceId != deviceId.intValue()) {
            unsafeSetDevice(currentDeviceId);
        }


        return result;
    }

    /**
     * This method replicates given DataBuffer, and places it to target device.
     *
     * @param deviceId target deviceId
     * @param buffer
     * @return
     */
    @Override
    public DataBuffer replicateToDevice(Integer deviceId, DataBuffer buffer) {
        if (buffer == null)
            return null;

        int currentDeviceId = Nd4j.getAffinityManager().getDeviceForCurrentThread();

        if (currentDeviceId != deviceId) {
            Nd4j.getAffinityManager().unsafeSetDevice(deviceId);
        }

        DataBuffer dstBuffer = Nd4j.createBuffer(buffer.dataType(), buffer.length(), false);
        AtomicAllocator.getInstance().memcpy(dstBuffer, buffer);

        if (currentDeviceId != deviceId) {
            Nd4j.getAffinityManager().unsafeSetDevice(currentDeviceId);
        }

        return dstBuffer;
    }

    /**
     * This method marks given INDArray as actual in specific location (either host, device, or both)
     *
     * @param array
     * @param location
     */
    @Override
    public void tagLocation(INDArray array, Location location) {
        // we can't tag empty arrays.
        if (array.isEmpty())
            return;

        if (location == Location.HOST)
            AtomicAllocator.getInstance().getAllocationPoint(array).tickHostWrite();
        else if (location == Location.DEVICE)
            AtomicAllocator.getInstance().getAllocationPoint(array).tickDeviceWrite();
        else if (location == Location.EVERYWHERE) {
            AtomicAllocator.getInstance().getAllocationPoint(array).tickDeviceWrite();
            AtomicAllocator.getInstance().getAllocationPoint(array).tickHostRead();
        }
    }

    /**
     * This method marks given DataBuffer as actual in specific location (either host, device, or both)
     *
     * @param buffer
     * @param location
     */
    @Override
    public void tagLocation(DataBuffer buffer, Location location) {
        if (location == Location.HOST)
            AtomicAllocator.getInstance().getAllocationPoint(buffer).tickHostWrite();
        else if (location == Location.DEVICE)
            AtomicAllocator.getInstance().getAllocationPoint(buffer).tickDeviceWrite();
        else if (location == Location.EVERYWHERE) {
            AtomicAllocator.getInstance().getAllocationPoint(buffer).tickDeviceWrite();
            AtomicAllocator.getInstance().getAllocationPoint(buffer).tickHostRead();
        }
    }

    @Override
    public Integer getDeviceForArray(@NonNull INDArray array) {
        return AtomicAllocator.getInstance().getDeviceId(array);
    }

    @Override
    public void unsafeSetDevice(Integer deviceId) {
        NativeOpsHolder.getInstance().getDeviceNativeOps().setDevice(deviceId);
    }

    @Override
    public void ensureLocation(INDArray array, Location location) {
        // to location to ensure for empty array
        if (array.isEmpty() || array.isS())
            return;

        // let's make sure host pointer actually exists
        ((BaseCudaDataBuffer) array.data()).lazyAllocateHostPointer();

        val point = AtomicAllocator.getInstance().getAllocationPoint(array);
        switch (location) {
            case HOST: {
                    AtomicAllocator.getInstance().synchronizeHostData(array);
                }
                break;
            case DEVICE:{
                    AtomicAllocator.getInstance().getFlowController().synchronizeToDevice(point);
                }
                break;
            case EVERYWHERE:
            default: {
                AtomicAllocator.getInstance().synchronizeHostData(array);
                AtomicAllocator.getInstance().getFlowController().synchronizeToDevice(point);
            }
        }
    }

    @Override
    public Location getActiveLocation(INDArray array) {
        if (array.isEmpty())
            return Location.EVERYWHERE;

        val point = AtomicAllocator.getInstance().getAllocationPoint(array);

        if (point.isActualOnDeviceSide() && point.isActualOnHostSide()) {
            return Location.EVERYWHERE;
        } else if (point.isActualOnDeviceSide()) {
            return Location.DEVICE;
        } else {
            return Location.HOST;
        }
    }

    @Override
    public boolean isCrossDeviceAccessSupported() {
        return NativeOpsHolder.getInstance().getDeviceNativeOps().isP2PAvailable() && CudaEnvironment.getInstance().getConfiguration().isCrossDeviceAccessAllowed();
    }

    @Override
    public void allowCrossDeviceAccess(boolean reallyAllow) {
        CudaEnvironment.getInstance().getConfiguration().allowCrossDeviceAccess(reallyAllow);
    }
}
