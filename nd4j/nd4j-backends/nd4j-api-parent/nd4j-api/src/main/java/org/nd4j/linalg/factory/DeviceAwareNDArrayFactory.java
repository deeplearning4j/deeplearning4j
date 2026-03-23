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
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.MultiBackendWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Factory utility for creating device-aware NDArrays.
 *
 * This class provides convenient methods for creating NDArrays on specific devices,
 * with support for:
 * - Direct device placement
 * - Workspace-based allocation
 * - Asynchronous prefetching
 * - Data transfer between devices
 *
 * <p>Example usage:</p>
 * <pre>{@code
 * // Create array on GPU 0
 * INDArray arr = DeviceAwareNDArrayFactory.createOnDevice(
 *     new long[]{3, 4}, DataType.FLOAT, DeviceDescriptor.cuda(0)
 * );
 *
 * // Create from data on specific device
 * float[] data = {1, 2, 3, 4};
 * INDArray arr = DeviceAwareNDArrayFactory.createFromArrayOnDevice(
 *     data, new long[]{2, 2}, DeviceDescriptor.cuda(0)
 * );
 *
 * // Transfer existing array to device
 * DeviceAwareNDArrayFactory.ensureOnDevice(existingArray, DeviceDescriptor.cuda(1));
 * }</pre>
 *
 * Adam Gibson
 */
@Slf4j
public class DeviceAwareNDArrayFactory {

    private static final ExecutorService prefetchExecutor = Executors.newFixedThreadPool(2, r -> {
        Thread t = new Thread(r, "DeviceAwareNDArrayFactory-Prefetch");
        t.setDaemon(true);
        return t;
    });

    private DeviceAwareNDArrayFactory() {
        // Static utility class
    }

    // =========================================================================
    // Create on Device
    // =========================================================================

    /**
     * Create a new NDArray on a specific device.
     *
     * @param shape the array shape
     * @param dataType the data type
     * @param device the target device
     * @return array allocated on the device
     */
    public static INDArray createOnDevice(long[] shape, DataType dataType, DeviceDescriptor device) {
        INDArray array = Nd4j.create(dataType, shape);
        ensureOnDevice(array, device);
        return array;
    }

    /**
     * Create a new NDArray with float type on a specific device.
     *
     * @param shape the array shape
     * @param device the target device
     * @return float array on the device
     */
    public static INDArray createOnDevice(long[] shape, DeviceDescriptor device) {
        return createOnDevice(shape, DataType.FLOAT, device);
    }

    /**
     * Create a zeros array on a specific device.
     *
     * @param shape the array shape
     * @param dataType the data type
     * @param device the target device
     * @return zeros array on the device
     */
    public static INDArray zerosOnDevice(long[] shape, DataType dataType, DeviceDescriptor device) {
        INDArray array = Nd4j.zeros(dataType, shape);
        ensureOnDevice(array, device);
        return array;
    }

    /**
     * Create a ones array on a specific device.
     *
     * @param shape the array shape
     * @param dataType the data type
     * @param device the target device
     * @return ones array on the device
     */
    public static INDArray onesOnDevice(long[] shape, DataType dataType, DeviceDescriptor device) {
        INDArray array = Nd4j.ones(dataType, shape);
        ensureOnDevice(array, device);
        return array;
    }

    /**
     * Create a random array on a specific device.
     *
     * @param shape the array shape
     * @param device the target device
     * @return random array on the device
     */
    public static INDArray randOnDevice(long[] shape, DeviceDescriptor device) {
        INDArray array = Nd4j.rand(shape);
        ensureOnDevice(array, device);
        return array;
    }

    /**
     * Create a random normal array on a specific device.
     *
     * @param shape the array shape
     * @param device the target device
     * @return random normal array on the device
     */
    public static INDArray randnOnDevice(long[] shape, DeviceDescriptor device) {
        INDArray array = Nd4j.randn(shape);
        ensureOnDevice(array, device);
        return array;
    }

    // =========================================================================
    // Create from Data on Device
    // =========================================================================

    /**
     * Create an array from float data on a specific device.
     *
     * @param data the float data
     * @param shape the array shape
     * @param device the target device
     * @return array on the device
     */
    public static INDArray createFromArrayOnDevice(float[] data, long[] shape, DeviceDescriptor device) {
        INDArray array = Nd4j.create(data, shape);
        ensureOnDevice(array, device);
        return array;
    }

    /**
     * Create an array from double data on a specific device.
     *
     * @param data the double data
     * @param shape the array shape
     * @param device the target device
     * @return array on the device
     */
    public static INDArray createFromArrayOnDevice(double[] data, long[] shape, DeviceDescriptor device) {
        INDArray array = Nd4j.create(data, shape);
        ensureOnDevice(array, device);
        return array;
    }

    /**
     * Create an array from int data on a specific device.
     *
     * @param data the int data
     * @param shape the array shape
     * @param device the target device
     * @return array on the device
     */
    public static INDArray createFromArrayOnDevice(int[] data, long[] shape, DeviceDescriptor device) {
        INDArray array = Nd4j.createFromArray(data).reshape(shape);
        ensureOnDevice(array, device);
        return array;
    }

    /**
     * Create an array from long data on a specific device.
     *
     * @param data the long data
     * @param shape the array shape
     * @param device the target device
     * @return array on the device
     */
    public static INDArray createFromArrayOnDevice(long[] data, long[] shape, DeviceDescriptor device) {
        INDArray array = Nd4j.createFromArray(data).reshape(shape);
        ensureOnDevice(array, device);
        return array;
    }

    // =========================================================================
    // Workspace-based Creation
    // =========================================================================

    /**
     * Create an array within a workspace on a specific device.
     *
     * @param shape the array shape
     * @param dataType the data type
     * @param workspace the workspace
     * @param device the target device
     * @return array in the workspace on the device
     */
    public static INDArray createInWorkspace(long[] shape, DataType dataType,
                                              MultiBackendWorkspace workspace, DeviceDescriptor device) {
        try (MemoryWorkspace ws = workspace.notifyScopeEntered()) {
            INDArray array = Nd4j.create(dataType, shape);
            ensureOnDevice(array, device);
            return array;
        }
    }

    /**
     * Create a zeros array within a workspace on a specific device.
     *
     * @param shape the array shape
     * @param workspace the workspace
     * @param device the target device
     * @return zeros array in the workspace on the device
     */
    public static INDArray zerosInWorkspace(long[] shape, MultiBackendWorkspace workspace,
                                             DeviceDescriptor device) {
        try (MemoryWorkspace ws = workspace.notifyScopeEntered()) {
            INDArray array = Nd4j.zeros(shape);
            ensureOnDevice(array, device);
            return array;
        }
    }

    // =========================================================================
    // Device Transfer
    // =========================================================================

    /**
     * Ensure an array is available on the specified device.
     * Triggers data transfer if necessary.
     *
     * @param array the array to transfer
     * @param device the target device
     */
    public static void ensureOnDevice(INDArray array, DeviceDescriptor device) {
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
     * Prefetch an array to a device asynchronously.
     *
     * @param array the array to prefetch
     * @param device the target device
     * @return future that completes when prefetch is done
     */
    public static CompletableFuture<Void> prefetchToDevice(INDArray array, DeviceDescriptor device) {
        return CompletableFuture.runAsync(() -> {
            DataBuffer buffer = array.data();
            if (buffer != null && buffer.isHybrid()) {
                DeviceDescriptor pinned = buffer.asHybrid().getPinnedDevice();
                if (pinned != null && !pinned.getDeviceId().equals(device.getDeviceId())) {
                    throw new IllegalStateException("Array is pinned to " + pinned.getDeviceId()
                            + " and cannot be prefetched to " + device.getDeviceId());
                }
                buffer.asHybrid().prefetch(device);
            }
        }, prefetchExecutor);
    }

    /**
     * Transfer an array to a new device, returning a copy.
     * The original array is not modified.
     *
     * @param array the source array
     * @param device the target device
     * @return copy of the array on the new device
     */
    public static INDArray copyToDevice(INDArray array, DeviceDescriptor device) {
        INDArray copy = array.dup();
        ensureOnDevice(copy, device);
        return copy;
    }

    /**
     * Transfer multiple arrays to a device.
     *
     * @param arrays the arrays to transfer
     * @param device the target device
     */
    public static void transferAllToDevice(INDArray[] arrays, DeviceDescriptor device) {
        for (INDArray array : arrays) {
            ensureOnDevice(array, device);
        }
    }

    /**
     * Asynchronously transfer multiple arrays to a device.
     *
     * @param arrays the arrays to transfer
     * @param device the target device
     * @return future that completes when all transfers are done
     */
    public static CompletableFuture<Void> transferAllToDeviceAsync(INDArray[] arrays,
                                                                    DeviceDescriptor device) {
        CompletableFuture<?>[] futures = new CompletableFuture[arrays.length];
        for (int i = 0; i < arrays.length; i++) {
            final INDArray array = arrays[i];
            futures[i] = CompletableFuture.runAsync(() -> ensureOnDevice(array, device), prefetchExecutor);
        }
        return CompletableFuture.allOf(futures);
    }

    // =========================================================================
    // Device Queries
    // =========================================================================

    /**
     * Check if an array is valid on a specific device.
     *
     * @param array the array to check
     * @param device the device to check
     * @return true if data is up-to-date on the device
     */
    public static boolean isValidOnDevice(INDArray array, DeviceDescriptor device) {
        if (array == null || device == null) {
            return false;
        }

        DataBuffer buffer = array.data();
        if (buffer != null && buffer.isHybrid()) {
            return buffer.asHybrid().isValidOn(device);
        }
        return true; // Non-hybrid buffers are always valid
    }

    /**
     * Get the effective device for an array.
     * Returns the pinned device if set, otherwise the owner device.
     *
     * @param array the array
     * @return the effective device, or null if not hybrid
     */
    public static DeviceDescriptor getOwnerDevice(INDArray array) {
        if (array == null) {
            return null;
        }

        DataBuffer buffer = array.data();
        if (buffer != null && buffer.isHybrid()) {
            return buffer.asHybrid().getEffectiveDevice();
        }
        return null;
    }

    /**
     * Mark an array as valid on a device.
     *
     * @param array the array
     * @param device the device to mark valid
     */
    public static void markValidOnDevice(INDArray array, DeviceDescriptor device) {
        if (array == null || device == null) {
            return;
        }

        DataBuffer buffer = array.data();
        if (buffer != null && buffer.isHybrid()) {
            buffer.asHybrid().markValidOn(device);
        }
    }

    /**
     * Mark an array as invalid on a device.
     *
     * @param array the array
     * @param device the device to invalidate
     */
    public static void markInvalidOnDevice(INDArray array, DeviceDescriptor device) {
        if (array == null || device == null) {
            return;
        }

        DataBuffer buffer = array.data();
        if (buffer != null && buffer.isHybrid()) {
            buffer.asHybrid().markInvalidOn(device);
        }
    }

    // =========================================================================
    // Convenience Methods for Common Device Types
    // =========================================================================

    /**
     * Create an array on CUDA device.
     *
     * @param shape the array shape
     * @param gpuIndex the GPU index
     * @return array on the GPU
     */
    public static INDArray createOnCuda(long[] shape, int gpuIndex) {
        return createOnDevice(shape, DataType.FLOAT, DeviceDescriptor.cuda(gpuIndex));
    }

    /**
     * Create an array on CPU.
     *
     * @param shape the array shape
     * @return array on CPU
     */
    public static INDArray createOnCpu(long[] shape) {
        return createOnDevice(shape, DataType.FLOAT, DeviceDescriptor.cpu());
    }

    /**
     * Transfer an array to CUDA device.
     *
     * @param array the array to transfer
     * @param gpuIndex the GPU index
     */
    public static void transferToCuda(INDArray array, int gpuIndex) {
        ensureOnDevice(array, DeviceDescriptor.cuda(gpuIndex));
    }

    /**
     * Transfer an array to CPU.
     *
     * @param array the array to transfer
     */
    public static void transferToCpu(INDArray array) {
        ensureOnDevice(array, DeviceDescriptor.cpu());
    }

    // =========================================================================
    // Batch Operations
    // =========================================================================

    /**
     * Create a batch of arrays distributed across devices.
     *
     * @param shape the shape for each array
     * @param count number of arrays to create
     * @param devices devices to distribute across (round-robin)
     * @return array of created arrays
     */
    public static INDArray[] createBatchOnDevices(long[] shape, int count, DeviceDescriptor... devices) {
        INDArray[] result = new INDArray[count];
        for (int i = 0; i < count; i++) {
            DeviceDescriptor device = devices[i % devices.length];
            result[i] = createOnDevice(shape, DataType.FLOAT, device);
        }
        return result;
    }

    /**
     * Concatenate arrays from different devices onto a target device.
     *
     * @param arrays the arrays to concatenate
     * @param dimension the dimension to concatenate along
     * @param targetDevice the device for the result
     * @return concatenated array on the target device
     */
    public static INDArray concatOnDevice(INDArray[] arrays, int dimension, DeviceDescriptor targetDevice) {
        // First transfer all to target device
        for (INDArray array : arrays) {
            ensureOnDevice(array, targetDevice);
        }

        // Then concatenate
        INDArray result = Nd4j.concat(dimension, arrays);
        ensureOnDevice(result, targetDevice);
        return result;
    }

    /**
     * Shutdown the prefetch executor.
     * Should be called during application shutdown.
     */
    public static void shutdown() {
        prefetchExecutor.shutdown();
    }
}
