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

package org.nd4j.linalg.api.ops.executioner;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.device.DeviceDescriptor;
import org.nd4j.linalg.api.device.DeviceMemoryManager;
import org.nd4j.linalg.api.device.DeviceRoutingConfiguration;
import org.nd4j.linalg.api.device.DeviceType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.CustomOp;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.nativeblas.MultiBackendNativeOpsHolder;
import org.nd4j.nativeblas.NativeOps;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Default implementation of BackendRoutingStrategy that routes operations
 * based on input array locations and device preferences.
 *
 * <p>This strategy:
 * <ul>
 *   <li>Routes ops to where the majority of input data resides</li>
 *   <li>Automatically transfers arrays to target device when needed</li>
 *   <li>Falls back to CPU when GPU is unavailable</li>
 *   <li>Respects device memory constraints</li>
 * </ul>
 *
 * Adam Gibson
 */
@Slf4j
public class DefaultBackendRoutingStrategy implements BackendRoutingStrategy {

    private static final DefaultBackendRoutingStrategy INSTANCE = new DefaultBackendRoutingStrategy();

    // Thread-local current device
    private final ThreadLocal<DeviceDescriptor> currentDevice = new ThreadLocal<>();

    // Statistics
    private final AtomicLong cpuRoutingCount = new AtomicLong(0);
    private final AtomicLong cudaRoutingCount = new AtomicLong(0);
    private final AtomicLong transferCount = new AtomicLong(0);
    private final AtomicLong transferBytes = new AtomicLong(0);

    private DefaultBackendRoutingStrategy() {
        // Singleton
    }

    public static DefaultBackendRoutingStrategy getInstance() {
        return INSTANCE;
    }

    @Override
    public DeviceDescriptor selectTargetDevice(Op op) {
        if (op == null) {
            return getDefaultDevice();
        }

        List<INDArray> inputs = new ArrayList<>();
        if (op.x() != null) inputs.add(op.x());
        if (op.y() != null) inputs.add(op.y());

        return selectTargetDevice(inputs);
    }

    @Override
    public DeviceDescriptor selectTargetDevice(CustomOp op) {
        if (op == null) {
            return getDefaultDevice();
        }

        List<INDArray> inputs = new ArrayList<>();
        if (op.inputArguments() != null) {
            inputs.addAll(op.inputArguments());
        }

        return selectTargetDevice(inputs);
    }

    @Override
    public DeviceDescriptor selectTargetDevice(List<INDArray> inputs) {
        DeviceRoutingConfiguration config = DeviceRoutingConfiguration.current();

        // If no inputs, use default device
        if (inputs == null || inputs.isEmpty()) {
            return getDefaultDevice();
        }

        // If routing by input location is disabled, use default
        if (!config.isRouteOpsByInputLocation()) {
            return getDefaultDevice();
        }

        // Calculate total bytes per device type
        Map<DeviceType, Long> bytesPerDevice = new HashMap<>();
        for (INDArray arr : inputs) {
            if (arr == null) continue;

            DeviceDescriptor device = getArrayDevice(arr);
            DeviceType type = device.getDeviceType();
            long bytes = arr.length() * arr.data().getElementSize();

            bytesPerDevice.merge(type, bytes, Long::sum);
        }

        // Find device type with most data
        DeviceType targetType = DeviceType.CPU;
        long maxBytes = 0;
        for (Map.Entry<DeviceType, Long> entry : bytesPerDevice.entrySet()) {
            if (entry.getValue() > maxBytes) {
                maxBytes = entry.getValue();
                targetType = entry.getKey();
            }
        }

        // Check if the target backend is available
        if (!isBackendAvailable(targetType)) {
            // Fall back to CPU
            targetType = DeviceType.CPU;
        }

        // Record routing decision
        if (targetType == DeviceType.CUDA_GPU) {
            cudaRoutingCount.incrementAndGet();
        } else {
            cpuRoutingCount.incrementAndGet();
        }

        if (config.isLogRoutingDecisions()) {
            log.debug("Routing op to {} (inputs: {}, bytes by device: {})",
                    targetType, inputs.size(), bytesPerDevice);
        }

        // Return appropriate device descriptor
        if (targetType == DeviceType.CUDA_GPU) {
            // Use first CUDA device or current device if it's CUDA
            DeviceDescriptor current = getCurrentDevice();
            if (current != null && current.getDeviceType() == DeviceType.CUDA_GPU) {
                return current;
            }
            return DeviceDescriptor.cuda(0);
        }

        return DeviceDescriptor.cpu();
    }

    @Override
    public NativeOps getNativeOpsForDevice(DeviceDescriptor device) {
        return MultiBackendNativeOpsHolder.getInstance().getOpsForDevice(device);
    }

    @Override
    public NativeOps getNativeOpsForDeviceType(DeviceType deviceType) {
        return MultiBackendNativeOpsHolder.getInstance().getOpsForDeviceType(deviceType);
    }

    @Override
    public boolean isBackendAvailable(DeviceType deviceType) {
        return MultiBackendNativeOpsHolder.getInstance().isBackendAvailable(deviceType);
    }

    @Override
    public DeviceDescriptor getCurrentDevice() {
        DeviceDescriptor device = currentDevice.get();
        if (device == null) {
            device = getDefaultDevice();
            currentDevice.set(device);
        }
        return device;
    }

    @Override
    public void setCurrentDevice(DeviceDescriptor device) {
        currentDevice.set(device);
    }

    @Override
    public INDArray ensureOnDevice(INDArray array, DeviceDescriptor targetDevice) {
        if (array == null || targetDevice == null) {
            return array;
        }

        DeviceDescriptor currentDevice = getArrayDevice(array);
        if (currentDevice.getDeviceId().equals(targetDevice.getDeviceId())) {
            return array; // Already on target device
        }

        // Track transfer
        long bytes = array.length() * array.data().getElementSize();
        transferCount.incrementAndGet();
        transferBytes.addAndGet(bytes);

        DeviceRoutingConfiguration config = DeviceRoutingConfiguration.current();
        if (config.isLogDataTransfers()) {
            log.debug("Transferring {} bytes from {} to {}",
                    bytes, currentDevice.getDeviceId(), targetDevice.getDeviceId());
        }

        // Use HybridDataBuffer's ensureAvailableOn if supported
        DataBuffer buffer = array.data();
        if (buffer != null && buffer.isHybrid()) {
            DeviceDescriptor pinned = buffer.asHybrid().getPinnedDevice();
            if (pinned != null && !pinned.getDeviceId().equals(targetDevice.getDeviceId())) {
                throw new IllegalStateException("Array is pinned to " + pinned.getDeviceId()
                        + " and cannot be transferred to " + targetDevice.getDeviceId());
            }
            buffer.asHybrid().ensureAvailableOn(targetDevice);
        }

        return array;
    }

    @Override
    public String getRoutingStats() {
        long total = cpuRoutingCount.get() + cudaRoutingCount.get();
        double cudaPercent = total > 0 ? 100.0 * cudaRoutingCount.get() / total : 0;

        return String.format(
                "Backend Routing Stats:\n" +
                        "  Total routing decisions: %d\n" +
                        "  CPU routes: %d\n" +
                        "  CUDA routes: %d (%.1f%%)\n" +
                        "  Transfers: %d\n" +
                        "  Transfer bytes: %d (%.2f MB)",
                total,
                cpuRoutingCount.get(),
                cudaRoutingCount.get(), cudaPercent,
                transferCount.get(),
                transferBytes.get(), transferBytes.get() / (1024.0 * 1024.0)
        );
    }

    /**
     * Reset statistics.
     */
    public void resetStats() {
        cpuRoutingCount.set(0);
        cudaRoutingCount.set(0);
        transferCount.set(0);
        transferBytes.set(0);
    }

    /**
     * Get the device an array currently resides on.
     */
    private DeviceDescriptor getArrayDevice(INDArray array) {
        if (array == null) {
            return DeviceDescriptor.cpu();
        }

        DataBuffer buffer = array.data();
        if (buffer != null && buffer.isHybrid()) {
            DeviceDescriptor effective = buffer.asHybrid().getEffectiveDevice();
            return effective != null ? effective : DeviceDescriptor.cpu();
        }

        // Default to CPU for non-hybrid buffers
        return DeviceDescriptor.cpu();
    }

    /**
     * Get the default device based on configuration.
     */
    private DeviceDescriptor getDefaultDevice() {
        // Check if there's a configured default
        DeviceMemoryManager mgr = DeviceMemoryManager.getInstance();
        DeviceDescriptor defaultDev = mgr.getDefaultDevice();
        if (defaultDev != null) {
            return defaultDev;
        }

        // Prefer GPU if available
        if (isBackendAvailable(DeviceType.CUDA_GPU)) {
            return DeviceDescriptor.cuda(0);
        }

        return DeviceDescriptor.cpu();
    }
}
