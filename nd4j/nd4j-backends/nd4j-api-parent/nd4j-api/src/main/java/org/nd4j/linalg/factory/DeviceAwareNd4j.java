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

package org.nd4j.linalg.factory;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.device.DeviceDescriptor;
import org.nd4j.linalg.api.device.DeviceMemoryManager;
import org.nd4j.linalg.api.device.DeviceRoutingConfiguration;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.executioner.DeviceAwareOpExecutioner;
import org.nd4j.linalg.api.ops.executioner.OpExecutionDelegator;

/**
 * Device-aware extension of Nd4j factory methods.
 *
 * This class provides seamless device routing for array creation, ensuring arrays
 * are allocated on optimal devices based on:
 * - Available memory
 * - Device priorities
 * - Memory caps
 * - Current routing configuration
 *
 * <p>When device routing is enabled, standard Nd4j.create() calls automatically
 * route to the best available device. Arrays are also automatically transferred
 * when passed to ops on different devices.</p>
 *
 * <h3>Basic Usage:</h3>
 * <pre>{@code
 * // Enable device routing (done once at startup)
 * DeviceAwareNd4j.enableDeviceRouting();
 *
 * // Configure device preferences
 * DeviceAwareNd4j.setMemoryCap(DeviceDescriptor.cuda(0), 8L * 1024 * 1024 * 1024); // 8GB
 * DeviceAwareNd4j.preferDevice(DeviceDescriptor.cuda(0)); // Prefer GPU 0
 *
 * // Now Nd4j.create() automatically routes to best device
 * INDArray arr = Nd4j.create(1000, 1000); // Routed based on config
 *
 * // Or explicitly create on a device
 * INDArray gpuArr = DeviceAwareNd4j.createOnDevice(new long[]{1000, 1000}, DeviceDescriptor.cuda(0));
 * }</pre>
 *
 * <h3>Automatic Transfer:</h3>
 * <pre>{@code
 * INDArray cpuArray = Nd4j.create(new float[]{1, 2, 3, 4});
 * INDArray gpuArray = DeviceAwareNd4j.createOnDevice(new long[]{4}, DeviceDescriptor.cuda(0));
 *
 * // When executing ops, cpuArray is automatically transferred to GPU
 * INDArray result = cpuArray.add(gpuArray); // cpuArray copied to GPU transparently
 * }</pre>
 *
 * @author Eclipse Deeplearning4j Contributors
 * @see DeviceMemoryManager
 * @see DeviceRoutingConfiguration
 * @see OpExecutionDelegator
 */
@Slf4j
public class DeviceAwareNd4j {

    private static volatile boolean routingEnabled = false;
    private static volatile AllocationHook allocationHook;

    private DeviceAwareNd4j() {
        // Static utility class
    }

    // =========================================================================
    // Configuration
    // =========================================================================

    /**
     * Enable device-aware routing for all Nd4j.create() calls.
     * Arrays will be allocated on the best available device.
     */
    public static void enableDeviceRouting() {
        enableDeviceRouting(DeviceRoutingConfiguration.gpuOptimized());
    }

    /**
     * Enable device routing with custom configuration.
     *
     * @param config the routing configuration
     */
    public static void enableDeviceRouting(DeviceRoutingConfiguration config) {
        config.apply();
        routingEnabled = true;

        // Install allocation hook
        allocationHook = new AllocationHook();

        // Install the device-aware op executioner for automatic cross-device transfers
        DeviceAwareOpExecutioner.install();

        log.info("Device routing enabled with policy: {}", config.getDefaultPolicy());
    }

    /**
     * Disable device-aware routing.
     */
    public static void disableDeviceRouting() {
        routingEnabled = false;
        allocationHook = null;

        // Uninstall the device-aware op executioner
        DeviceAwareOpExecutioner.uninstall();
    }

    /**
     * Check if device routing is enabled.
     */
    public static boolean isRoutingEnabled() {
        return routingEnabled;
    }

    /**
     * Set a memory cap for a device.
     *
     * @param device the device
     * @param maxBytes maximum bytes
     */
    public static void setMemoryCap(DeviceDescriptor device, long maxBytes) {
        DeviceMemoryManager.getInstance().setMemoryCap(device, maxBytes);
    }

    /**
     * Set memory cap as fraction of total device memory.
     *
     * @param device the device
     * @param fraction fraction (0.0 to 1.0)
     */
    public static void setMemoryCapFraction(DeviceDescriptor device, double fraction) {
        DeviceMemoryManager.getInstance().setMemoryCapFraction(device, fraction);
    }

    /**
     * Set a device as the preferred/default device.
     *
     * @param device the preferred device
     */
    public static void preferDevice(DeviceDescriptor device) {
        DeviceMemoryManager.getInstance().setDefaultDevice(device);
    }

    /**
     * Set device priority (higher = more preferred).
     *
     * @param device the device
     * @param priority priority value
     */
    public static void setDevicePriority(DeviceDescriptor device, int priority) {
        DeviceMemoryManager.getInstance().setDevicePriority(device, priority);
    }

    /**
     * Register a device with the memory manager.
     *
     * @param device the device to register
     */
    public static void registerDevice(DeviceDescriptor device) {
        DeviceMemoryManager.getInstance().registerDevice(device);
    }

    // =========================================================================
    // Device-Aware Creation (Explicit Device)
    // =========================================================================

    /**
     * Create an array on a specific device.
     *
     * @param shape the array shape
     * @param device the target device
     * @return array on the specified device
     */
    public static INDArray createOnDevice(long[] shape, DeviceDescriptor device) {
        return createOnDevice(shape, DataType.FLOAT, device);
    }

    /**
     * Create an array with specified type on a device.
     *
     * @param shape the array shape
     * @param dataType the data type
     * @param device the target device
     * @return array on the specified device
     */
    public static INDArray createOnDevice(long[] shape, DataType dataType, DeviceDescriptor device) {
        INDArray array = Nd4j.create(dataType, shape);
        ensureOnDevice(array, device);
        recordAllocation(array, device);
        return array;
    }

    /**
     * Create zeros array on a device.
     */
    public static INDArray zerosOnDevice(long[] shape, DeviceDescriptor device) {
        INDArray array = Nd4j.zeros(shape);
        ensureOnDevice(array, device);
        recordAllocation(array, device);
        return array;
    }

    /**
     * Create ones array on a device.
     */
    public static INDArray onesOnDevice(long[] shape, DeviceDescriptor device) {
        INDArray array = Nd4j.ones(shape);
        ensureOnDevice(array, device);
        recordAllocation(array, device);
        return array;
    }

    /**
     * Create random array on a device.
     */
    public static INDArray randOnDevice(long[] shape, DeviceDescriptor device) {
        INDArray array = Nd4j.rand(shape);
        ensureOnDevice(array, device);
        recordAllocation(array, device);
        return array;
    }

    /**
     * Create from float array on a device.
     */
    public static INDArray createOnDevice(float[] data, long[] shape, DeviceDescriptor device) {
        INDArray array = Nd4j.create(data, shape);
        ensureOnDevice(array, device);
        recordAllocation(array, device);
        return array;
    }

    /**
     * Create from double array on a device.
     */
    public static INDArray createOnDevice(double[] data, long[] shape, DeviceDescriptor device) {
        INDArray array = Nd4j.create(data, shape);
        ensureOnDevice(array, device);
        recordAllocation(array, device);
        return array;
    }

    /**
     * Create from int array on a device.
     */
    public static INDArray createOnDevice(int[] data, long[] shape, DeviceDescriptor device) {
        INDArray array = Nd4j.createFromArray(data).reshape(shape);
        ensureOnDevice(array, device);
        recordAllocation(array, device);
        return array;
    }

    // =========================================================================
    // Device-Routed Creation (Automatic Device Selection)
    // =========================================================================

    /**
     * Create an array on the best available device.
     * Uses current routing configuration to select device.
     *
     * @param shape the array shape
     * @return array on optimal device
     */
    public static INDArray createRouted(long[] shape) {
        return createRouted(shape, DataType.FLOAT);
    }

    /**
     * Create array on best device with specified type.
     *
     * @param shape the array shape
     * @param dataType the data type
     * @return array on optimal device
     */
    public static INDArray createRouted(long[] shape, DataType dataType) {
        long bytes = calculateBytes(shape, dataType);
        DeviceDescriptor device = DeviceMemoryManager.getInstance()
                .selectDeviceForAllocation(bytes);
        return createOnDevice(shape, dataType, device);
    }

    /**
     * Create zeros on best device.
     */
    public static INDArray zerosRouted(long[] shape) {
        long bytes = calculateBytes(shape, DataType.FLOAT);
        DeviceDescriptor device = DeviceMemoryManager.getInstance()
                .selectDeviceForAllocation(bytes);
        return zerosOnDevice(shape, device);
    }

    /**
     * Create random array on best device.
     */
    public static INDArray randRouted(long[] shape) {
        long bytes = calculateBytes(shape, DataType.FLOAT);
        DeviceDescriptor device = DeviceMemoryManager.getInstance()
                .selectDeviceForAllocation(bytes);
        return randOnDevice(shape, device);
    }

    // =========================================================================
    // Convenience Methods for Common Devices
    // =========================================================================

    /**
     * Create array on GPU 0.
     */
    public static INDArray createOnGpu(long[] shape) {
        return createOnDevice(shape, DeviceDescriptor.cuda(0));
    }

    /**
     * Create array on specified GPU.
     */
    public static INDArray createOnGpu(long[] shape, int gpuIndex) {
        return createOnDevice(shape, DeviceDescriptor.cuda(gpuIndex));
    }

    /**
     * Create array on CPU.
     */
    public static INDArray createOnCpu(long[] shape) {
        return createOnDevice(shape, DeviceDescriptor.cpu());
    }

    // =========================================================================
    // Transfer Methods
    // =========================================================================

    /**
     * Transfer an array to a device.
     *
     * @param array the array to transfer
     * @param device the target device
     */
    public static void transferTo(INDArray array, DeviceDescriptor device) {
        ensureOnDevice(array, device);
    }

    /**
     * Transfer array to GPU 0.
     */
    public static void transferToGpu(INDArray array) {
        transferTo(array, DeviceDescriptor.cuda(0));
    }

    /**
     * Transfer array to CPU.
     */
    public static void transferToCpu(INDArray array) {
        transferTo(array, DeviceDescriptor.cpu());
    }

    /**
     * Transfer multiple arrays to a device.
     */
    public static void transferAllTo(DeviceDescriptor device, INDArray... arrays) {
        for (INDArray array : arrays) {
            ensureOnDevice(array, device);
        }
    }

    /**
     * Get the effective device for an array.
     * Returns the pinned device if set, otherwise the owner device.
     */
    public static DeviceDescriptor getDevice(INDArray array) {
        if (array == null) return null;

        DataBuffer buffer = array.data();
        if (buffer != null && buffer.isHybrid()) {
            return buffer.asHybrid().getEffectiveDevice();
        }
        return DeviceDescriptor.cpu();
    }

    // =========================================================================
    // Pinning Methods
    // =========================================================================

    /**
     * Pin an array to a specific device.
     *
     * @param array the array to pin
     * @param device the device to pin to
     */
    public static void pin(INDArray array, DeviceDescriptor device) {
        if (array == null || device == null) {
            return;
        }
        DataBuffer buffer = array.data();
        if (buffer != null && buffer.isHybrid()) {
            if (device.getDeviceType().isGpu() && !DeviceAwareOpExecutioner.isInstalled()) {
                DeviceAwareOpExecutioner.install();
            }
            buffer.asHybrid().pinTo(device);
        } else if (device.getDeviceType().isGpu()) {
            throw new IllegalArgumentException("Cannot pin non-hybrid array to " + device.getDeviceId());
        }
    }

    /**
     * Remove pinning from an array.
     *
     * @param array the array to unpin
     */
    public static void unpin(INDArray array) {
        if (array == null) {
            return;
        }
        DataBuffer buffer = array.data();
        if (buffer != null && buffer.isHybrid()) {
            buffer.asHybrid().unpin();
        }
    }

    /**
     * Get the pinned device for an array, if any.
     *
     * @param array the array to query
     * @return pinned device or null if not pinned
     */
    public static DeviceDescriptor getPinnedDevice(INDArray array) {
        if (array == null) {
            return null;
        }
        DataBuffer buffer = array.data();
        if (buffer != null && buffer.isHybrid()) {
            return buffer.asHybrid().getPinnedDevice();
        }
        return null;
    }

    /**
     * Check if an array is pinned to a device.
     *
     * @param array the array to query
     * @return true if pinned
     */
    public static boolean isPinned(INDArray array) {
        return getPinnedDevice(array) != null;
    }

    /**
     * Check if array is on specified device.
     */
    public static boolean isOnDevice(INDArray array, DeviceDescriptor device) {
        DeviceDescriptor current = getDevice(array);
        return current != null && current.getDeviceId().equals(device.getDeviceId());
    }

    // =========================================================================
    // Query Methods
    // =========================================================================

    /**
     * Get available memory on a device.
     */
    public static long getAvailableMemory(DeviceDescriptor device) {
        return DeviceMemoryManager.getInstance().getAvailableMemory(device);
    }

    /**
     * Get allocated memory on a device.
     */
    public static long getAllocatedMemory(DeviceDescriptor device) {
        return DeviceMemoryManager.getInstance().getAllocatedMemory(device);
    }

    /**
     * Get memory utilization for a device.
     */
    public static double getMemoryUtilization(DeviceDescriptor device) {
        return DeviceMemoryManager.getInstance().getMemoryUtilization(device);
    }

    /**
     * Check if device can accommodate allocation.
     */
    public static boolean canAllocate(DeviceDescriptor device, long bytes) {
        return DeviceMemoryManager.getInstance().canAllocate(device, bytes);
    }

    /**
     * Get the best device for an allocation.
     */
    public static DeviceDescriptor selectDevice(long bytes) {
        return DeviceMemoryManager.getInstance().selectDeviceForAllocation(bytes);
    }

    // =========================================================================
    // Statistics
    // =========================================================================

    /**
     * Log memory statistics for all devices.
     */
    public static void logMemoryStats() {
        DeviceMemoryManager.getInstance().logMemoryStats();
    }

    /**
     * Log transfer statistics.
     */
    public static void logTransferStats() {
        OpExecutionDelegator.getInstance().logStatistics();
    }

    /**
     * Reset statistics.
     */
    public static void resetStats() {
        DeviceMemoryManager.getInstance().resetPeakMemory();
        OpExecutionDelegator.getInstance().resetStatistics();
    }

    // =========================================================================
    // Internal Methods
    // =========================================================================

    /**
     * Ensure array is on the specified device.
     */
    private static void ensureOnDevice(INDArray array, DeviceDescriptor device) {
        if (array == null || device == null) {
            return;
        }

        DataBuffer buffer = array.data();
        if (buffer != null && buffer.isHybrid()) {
            DeviceDescriptor pinned = buffer.asHybrid().getPinnedDevice();
            if (pinned != null && !pinned.getDeviceId().equals(device.getDeviceId())) {
                throw new IllegalStateException("Array is pinned to " + pinned.getDeviceId()
                        + " and cannot be transferred to " + device.getDeviceId());
            }
            buffer.asHybrid().ensureAvailableOn(device);
        }
    }

    /**
     * Record an allocation for memory tracking.
     */
    private static void recordAllocation(INDArray array, DeviceDescriptor device) {
        if (array == null || device == null) {
            return;
        }

        long bytes = array.length() * array.data().getElementSize();
        DeviceMemoryManager.getInstance().recordAllocation(device, bytes);
    }

    /**
     * Calculate bytes for a shape and type.
     */
    private static long calculateBytes(long[] shape, DataType dataType) {
        long elements = 1;
        for (long dim : shape) {
            elements *= dim;
        }
        return elements * dataType.width();
    }

    // =========================================================================
    // Allocation Hook
    // =========================================================================

    /**
     * Hook for intercepting allocations when routing is enabled.
     * This would integrate with the native allocation path.
     */
    private static class AllocationHook {

        /**
         * Called before allocation to determine device.
         */
        public DeviceDescriptor beforeAllocation(long bytes) {
            if (!routingEnabled) {
                return null;
            }

            return DeviceMemoryManager.getInstance().selectDeviceForAllocation(bytes);
        }

        /**
         * Called after allocation to record it.
         */
        public void afterAllocation(INDArray array, DeviceDescriptor device) {
            if (!routingEnabled || array == null || device == null) {
                return;
            }

            long bytes = array.length() * array.data().getElementSize();
            DeviceMemoryManager.getInstance().recordAllocation(device, bytes);
        }
    }

    /**
     * Get the current allocation hook (for internal use).
     */
    public static AllocationHook getAllocationHook() {
        return allocationHook;
    }
}
