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

package org.nd4j.jita.concurrency;

import lombok.NonNull;
import lombok.val;
import org.nd4j.jita.allocator.impl.AllocationPoint;
import org.nd4j.jita.allocator.impl.AtomicAllocator;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.concurrency.BasicAffinityManager;
import org.nd4j.linalg.api.device.DeviceContext;
import org.nd4j.linalg.api.device.DeviceMemoryManager;
import org.nd4j.linalg.api.device.MultiGpuTracer;
import org.nd4j.linalg.api.device.DeviceType;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.buffer.BaseCudaDataBuffer;
import org.nd4j.nativeblas.NativeOpsHolder;
import org.nd4j.nativeblas.OpaqueDataBuffer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * AffinityManager implementation for CUDA.
 *
 * This implementation recognizes both GPU devices (device IDs >= 0) and the CPU
 * as a valid device (CPU_DEVICE_ID = -1). This allows seamless data transfer between
 * CPU and GPU memory, enabling hybrid CPU-GPU computation patterns.
 *
 * @author raver119@gmail.com
 */
public class CudaAffinityManager extends BasicAffinityManager {
    private static Logger logger = LoggerFactory.getLogger(CudaAffinityManager.class);

    private final Map<Long, Integer> affinityMap = new ConcurrentHashMap<>();
    private ThreadLocal<AtomicBoolean> affiliated = new ThreadLocal<>();

    private AtomicInteger numberOfDevices = new AtomicInteger(-1);

    private final CudaDeviceContextProvider contextProvider;

    public CudaAffinityManager() {
        super();
        contextProvider = new CudaDeviceContextProvider(affinityMap);
        DeviceMemoryManager.getInstance().setContextProvider(contextProvider);
        logger.debug("CudaAffinityManager initialized - CPU device ID: {}", CPU_DEVICE_ID);
    }

    /**
     * Checks if the given device ID represents the CPU.
     *
     * @param deviceId the device ID to check
     * @return true if the device is CPU (deviceId == -1), false otherwise
     */
    @Override
    public boolean isCpuDevice(int deviceId) {
        return deviceId == CPU_DEVICE_ID;
    }

    /**
     * Returns the device type for the given device ID.
     *
     * @param deviceId the device ID to query
     * @return CPU for deviceId == -1, CUDA_GPU for deviceId >= 0
     */
    @Override
    public DeviceType getDeviceType(int deviceId) {
        if (deviceId == CPU_DEVICE_ID) {
            return DeviceType.CPU;
        }
        return DeviceType.CUDA_GPU;
    }

    /**
     * Returns deviceId for current thread.
     *
     * First call selects the GPU with the most free memory via DeviceMemoryManager.
     * Subsequent calls return the cached device. Device can be changed explicitly via
     * {@link DeviceMemoryManager#switchDevice(int, String, String)} or {@link #refreshDeviceForCurrentThread()}.
     *
     * @return device ID for current thread
     */
    @Override
    public Integer getDeviceForCurrentThread() {
        long threadId = Thread.currentThread().getId();
        Integer deviceId = affinityMap.get(threadId);
        if (deviceId != null) {
            return deviceId;
        }

        RuntimeException lastException = null;
        int attempts = Math.max(1, CudaEnvironment.getInstance().getConfiguration().getAvailableDevices().size());
        for (int i = 0; i < attempts; i++) {
            deviceId = getNextDevice(threadId);
            affinityMap.put(threadId, deviceId);
            try {
                contextProvider.switchDevice(deviceId, "CudaAffinityManager.getDeviceForCurrentThread", "first-time-init");
                return deviceId;
            } catch (RuntimeException e) {
                lastException = e;
                affinityMap.remove(threadId);
                logger.warn("Failed to set CUDA device [{}] for thread [{}]: {}", deviceId, threadId, e.getMessage());
                if (deviceId != null && deviceId >= 0) {
                    CudaEnvironment.getInstance().getConfiguration().banDevice(deviceId);
                }
            }
        }

        throw new RuntimeException("No usable CUDA devices available for thread " + threadId, lastException);
    }

    /**
     * Force re-evaluation of device assignment for the current thread.
     * Selects the GPU with the most free memory. Call this at strategic points
     * (e.g., between inference batches) to rebalance across GPUs.
     *
     * @return newly selected device ID
     */
    public Integer refreshDeviceForCurrentThread() {
        long threadId = Thread.currentThread().getId();
        affinityMap.remove(threadId);
        return getDeviceForCurrentThread();
    }

    /**
     * This method returns deviceId for a given thread
     * @return
     */
    @Override
    public Integer getDeviceForThread(long threadId) {
        Integer id = affinityMap.get(threadId);
        if (id == null) {
            // if this is current thread - we're still able to fetch id from native side, and update map
            if (threadId == Thread.currentThread().getId()) {
                id = getDeviceForCurrentThread();
            } else
                // TODO: we should get rid of this method, and forbid such kind of queries
                throw new RuntimeException("Affinity for thread [" + threadId + "] wasn't defined yet");
        }

        return id;
    }

    /**
     * Set the device ID for the current thread.
     * @param deviceId the device ID to set
     */
    @Override
    public void setDeviceForCurrentThread(int deviceId) {
        long threadId = Thread.currentThread().getId();
        affinityMap.put(threadId, deviceId);
    }


    /**
     * Selects the best available GPU device for a new thread by delegating to
     * {@link DeviceMemoryManager}. The device with the most actual free memory
     * is chosen.
     *
     * @param threadId used for logging only.
     * @return device id
     */
    protected Integer getNextDevice(long threadId) {
        if (CudaEnvironment.getInstance().getConfiguration().getAvailableDevices().isEmpty()) {
            logger.warn("No available CUDA devices configured; defaulting to device [0]");
            return 0;
        }

        if (CudaEnvironment.getInstance().getConfiguration().isForcedSingleGPU() || getNumberOfDevices() <= 1) {
            int device = CudaEnvironment.getInstance().getConfiguration().getAvailableDevices().get(0);
            logger.debug("Single device is forced, mapping to device [{}]", device);
            return device;
        }

        // Delegate to DeviceMemoryManager - single mechanism for all device selection
        int device = DeviceMemoryManager.getInstance().selectBestGpu();

        val t = Thread.currentThread();
        val n = t.getId() == threadId ? t.getName() : "N/A";
        logger.debug("Mapping thread [{} - {}] to device [{}], out of [{}] devices...",
                threadId, n, device, getNumberOfDevices());

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
                    // Ensure configuration has initialized the device list
                    CudaEnvironment.getInstance().notifyConfigurationApplied();
                    numberOfDevices.set(CudaEnvironment.getInstance().getConfiguration().getAvailableDevices().size());
                }
            }
        }

        return numberOfDevices.get();
    }

    @Override
    public List<Integer> getAvailableDeviceIds() {
        CudaEnvironment.getInstance().notifyConfigurationApplied();
        return Collections.unmodifiableList(
                CudaEnvironment.getInstance().getConfiguration().getAvailableDevices());
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
     * Supports both GPU devices (deviceId >= 0) and CPU (deviceId == CPU_DEVICE_ID).
     * When targeting CPU, data is synchronized to host memory and a host-resident copy is created.
     *
     * @param deviceId target deviceId (use CPU_DEVICE_ID for CPU, >= 0 for GPU)
     * @param array    INDArray to replicate
     * @return the replicated array on the target device
     */
    @Override
    public synchronized INDArray replicateToDevice(Integer deviceId, INDArray array) {
        if (array == null)
            return null;

        if (MultiGpuTracer.ENABLED) {
            int srcDev = -1;
            try {
                srcDev = AtomicAllocator.getInstance().getAllocationPoint(array).getDeviceId();
            } catch (Exception ignored) {}
            MultiGpuTracer.traceDataTransfer(srcDev, deviceId, array.shape(), array.dataType(),
                    array.length() * (array.data() != null ? array.data().getElementSize() : 0),
                    array.isView(),
                    array.data() != null && array.data().isConstant());
        }

        // string arrays are stored in host memory only atm
        if (array.isS()) {
            Nd4j.getExecutioner().setSkipDeviceCoherency(true);
            try (MemoryWorkspace ws = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
                return array.dup(array.ordering());
            } finally {
                Nd4j.getExecutioner().setSkipDeviceCoherency(false);
            }
        }

        if (array.isView()) {
            // Views can't be replicated directly (they reference parent buffer).
            // Create a contiguous copy first, then replicate to target device.
            // The contiguous copy must be explicitly closed after replication —
            // GC cleanup is broken (PhantomRef strong reference cycle), so without
            // this, every cross-device migration of a view leaks a contiguous copy
            // on the source device (~30MB/step for VLM KV cache with 60 views).
            //
            // CRITICAL: dup() must run on the SOURCE device (where the array data
            // actually resides). If the current thread is on a different device
            // (e.g., DeviceWorker[1] on device 1, but array data is on device 0),
            // dup() would try a direct GPU-to-GPU copy which fails without P2P
            // (error 700: illegal memory access). Switch to source device first.
            //
            // Skip device coherency during dup() to prevent infinite recursion:
            // replicateToDevice -> dup -> assign -> exec -> ensureDeviceCoherency -> replicateToDevice
            int sourceDeviceId = AtomicAllocator.getInstance().getAllocationPoint(array).getDeviceId();
            int currentDeviceId = getDeviceForCurrentThread();
            if (sourceDeviceId >= 0 && sourceDeviceId != currentDeviceId) {
                DeviceMemoryManager.getInstance().switchDevice(sourceDeviceId, "CudaAffinityManager.replicateToDevice", "view-dup-source");
            }
            Nd4j.getExecutioner().setSkipDeviceCoherency(true);
            INDArray contiguous;
            try (MemoryWorkspace ws = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
                contiguous = array.dup(array.ordering());
            } finally {
                Nd4j.getExecutioner().setSkipDeviceCoherency(false);
            }
            // Restore original device before recursive replicateToDevice call
            if (sourceDeviceId >= 0 && sourceDeviceId != currentDeviceId) {
                DeviceMemoryManager.getInstance().switchDevice(currentDeviceId, "CudaAffinityManager.replicateToDevice", "restore-after-dup");
            }
            INDArray result = replicateToDevice(deviceId, contiguous);
            contiguous.close();
            return result;
        }

        val shape = array.shape();
        val stride = array.stride();
        val elementWiseStride = -1;
        val ordering = array.ordering();
        val length = array.length();
        val dtype = array.dataType();
        val empty = array.isEmpty();

        int currentDeviceId = getDeviceForCurrentThread();

        // Handle CPU device target - sync to host and create host-resident copy
        if (isCpuDevice(deviceId)) {
            logger.debug("Replicating array to CPU device from GPU device {}", currentDeviceId);

            // Ensure data is synchronized to host memory
            AtomicAllocator.getInstance().synchronizeHostData(array);
            Nd4j.getExecutioner().commit();

            // Create a copy that's marked as host-resident
            INDArray result;
            Nd4j.getExecutioner().setSkipDeviceCoherency(true);
            try (MemoryWorkspace ws = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
                result = array.dup(array.ordering());
                // Tag as host-resident
                tagLocation(result, Location.HOST);
            } finally {
                Nd4j.getExecutioner().setSkipDeviceCoherency(false);
            }

            logger.debug("Array replicated to CPU, shape: {}", java.util.Arrays.toString(shape));
            return result;
        }

        // GPU-to-GPU transfer
        // Get the device where the SOURCE array actually resides
        int sourceDeviceId = AtomicAllocator.getInstance().getAllocationPoint(array).getDeviceId();

        // Ensure source data is synced to host memory before any device switching.
        // This uses native code that handles the device context properly internally.
        // synchronizeHostData uses dbSyncToPrimary which knows the buffer's device.
        AtomicAllocator.getInstance().synchronizeHostData(array);

        // Also commit any pending operations on current device
        Nd4j.getExecutioner().commit();

        // Switch to target device for buffer creation
        if (currentDeviceId != deviceId.intValue()) {
            DeviceMemoryManager.getInstance().switchDevice(deviceId, "CudaAffinityManager.replicateToDevice", "target-device-create");
        }

        // Create new data buffer on target device and copy data
        DataBuffer newDataBuffer = replicateToDevice(deviceId, array.data());

        INDArray result;
        try (MemoryWorkspace ws = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
            // Create shape buffer on TARGET device (we're already on it)
            DataBuffer newShapeBuffer = Nd4j.getShapeInfoProvider().createShapeInformation(shape, stride, elementWiseStride, ordering, dtype, empty).getFirst();
            result = Nd4j.createArrayFromShapeBuffer(newDataBuffer, newShapeBuffer);
        }

        // Sync target device before switching back
        Nd4j.getExecutioner().commit();

        // Switch back to original device
        if (getDeviceForCurrentThread() != currentDeviceId) {
            DeviceMemoryManager.getInstance().switchDevice(currentDeviceId, "CudaAffinityManager.replicateToDevice", "restore-after-array-replicate");
        }

        return result;
    }

    /**
     * This method replicates given DataBuffer, and places it to target device.
     *
     * Supports both GPU devices (deviceId >= 0) and CPU (deviceId == CPU_DEVICE_ID).
     * When targeting CPU, data is synchronized to host memory and a host-resident copy is created.
     *
     * @param deviceId target deviceId (use CPU_DEVICE_ID for CPU, >= 0 for GPU)
     * @param buffer   the DataBuffer to replicate
     * @return the replicated buffer on the target device
     */
    @Override
    public DataBuffer replicateToDevice(Integer deviceId, DataBuffer buffer) {
        if (buffer == null)
            return null;

        int currentDeviceId = Nd4j.getAffinityManager().getDeviceForCurrentThread();

        // Ensure source data is synced to host memory before any device switching.
        // This uses native code that handles the device context properly internally.
        // synchronizeHostData uses dbSyncToPrimary which knows the buffer's device.
        AtomicAllocator.getInstance().synchronizeHostData(buffer);

        // Commit any pending operations on current device
        Nd4j.getExecutioner().commit();

        // Handle CPU device target
        if (isCpuDevice(deviceId)) {
            logger.debug("Replicating DataBuffer to CPU device from GPU device {}", currentDeviceId);

            // Create a new buffer and copy host data
            DataBuffer dstBuffer;
            try (MemoryWorkspace ws = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
                dstBuffer = Nd4j.createBuffer(buffer.dataType(), buffer.length(), false);
            }

            // memcpy from source (already synced to host) to destination
            AtomicAllocator.getInstance().memcpy(dstBuffer, buffer);

            // Tag the destination as host-resident
            tagLocation(dstBuffer, Location.HOST);

            return dstBuffer;
        }

        // GPU-to-GPU transfer
        // Switch to target device for buffer creation
        if (currentDeviceId != deviceId.intValue()) {
            DeviceMemoryManager.getInstance().switchDevice(deviceId, "CudaAffinityManager.replicateToDevice", "target-device-buffer");
        }

        // Suppress OpaqueDataBuffer's cross-device routing during replicateToDevice.
        // The caller explicitly chose deviceId as the target — the Java-level routing
        // must not override this by rerouting to a device with more free memory.
        // If the target device truly OOMs, the C++ allocateFailover handles it.
        DataBuffer dstBuffer;
        OpaqueDataBuffer.suppressCrossDeviceRouting(true);
        try (MemoryWorkspace ws = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
            dstBuffer = Nd4j.createBuffer(buffer.dataType(), buffer.length(), false);
        } finally {
            OpaqueDataBuffer.suppressCrossDeviceRouting(false);
        }

        // memcpy handles cross-device copying (source is already synced to host)
        AtomicAllocator.getInstance().memcpy(dstBuffer, buffer);

        // Sync target device after copy
        Nd4j.getExecutioner().commit();

        // Switch back to original device
        if (Nd4j.getAffinityManager().getDeviceForCurrentThread() != currentDeviceId) {
            DeviceMemoryManager.getInstance().switchDevice(currentDeviceId, "CudaAffinityManager.replicateToDevice", "restore-after-buffer-replicate");
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
    public void ensureLocation(INDArray array, Location location) {
        // to location to ensure for empty array
        if (array == null || array.isEmpty() || array.isS())
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
