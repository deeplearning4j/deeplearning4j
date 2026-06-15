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

package org.nd4j.linalg.api.ops.routing;

import org.nd4j.linalg.api.device.DeviceDescriptor;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.CustomOp;

import java.util.List;
import java.util.Objects;

/**
 * Routing policy that allows manual/explicit device selection via thread-local.
 * Useful for debugging, testing, or when the user knows best.
 */
public class ManualPolicy implements RoutingPolicy {

    private static final ThreadLocal<DeviceDescriptor> targetDevice = new ThreadLocal<>();
    private static final ThreadLocal<String> targetDeviceId = new ThreadLocal<>();

    @Override
    public String getName() {
        return "Manual";
    }

    @Override
    public RoutingDecision selectDevice(Op op, List<DeviceDescriptor> availableDevices) {
        return selectTargetDevice(availableDevices);
    }

    @Override
    public RoutingDecision selectDevice(CustomOp op, List<DeviceDescriptor> availableDevices) {
        return selectTargetDevice(availableDevices);
    }

    private RoutingDecision selectTargetDevice(List<DeviceDescriptor> availableDevices) {
        if (availableDevices == null || availableDevices.isEmpty()) {
            throw new IllegalArgumentException("No available devices");
        }

        // Check for explicitly set device
        DeviceDescriptor device = targetDevice.get();
        if (device != null) {
            // Verify it's still available
            for (DeviceDescriptor available : availableDevices) {
                if (available.getDeviceId().equals(device.getDeviceId())) {
                    return RoutingDecision.builder(available)
                            .reason("manually selected device")
                            .confidence(1.0)
                            .build();
                }
            }
            // Device not available, fall through
        }

        // Check for device ID
        String deviceId = targetDeviceId.get();
        if (deviceId != null) {
            for (DeviceDescriptor available : availableDevices) {
                if (available.getDeviceId().equals(deviceId)) {
                    return RoutingDecision.builder(available)
                            .reason("manually selected device by ID")
                            .confidence(1.0)
                            .build();
                }
            }
            // ID not found, fall through
        }

        // No manual selection, use first available
        return RoutingDecision.builder(availableDevices.get(0))
                .reason("no manual selection, using default")
                .confidence(0.5)
                .build();
    }

    @Override
    public boolean canHandle(Op op) {
        return targetDevice.get() != null || targetDeviceId.get() != null;
    }

    @Override
    public boolean canHandle(CustomOp op) {
        return targetDevice.get() != null || targetDeviceId.get() != null;
    }

    @Override
    public int getPriority() {
        return 1000;  // Highest priority when active
    }

    // Static methods for setting the target device

    /**
     * Set the target device for the current thread
     * @param device the device to use
     */
    public static void setDevice(DeviceDescriptor device) {
        targetDevice.set(device);
        if (device != null) {
            targetDeviceId.set(device.getDeviceId());
        }
    }

    /**
     * Set the target device by ID for the current thread
     * @param deviceId the device ID to use
     */
    public static void setDeviceId(String deviceId) {
        targetDeviceId.set(deviceId);
        targetDevice.remove();  // Clear device reference
    }

    /**
     * Clear the manual device selection for the current thread
     */
    public static void clearDevice() {
        targetDevice.remove();
        targetDeviceId.remove();
    }

    /**
     * Get the currently selected device for this thread
     * @return the device, or null if not set
     */
    public static DeviceDescriptor getDevice() {
        return targetDevice.get();
    }

    /**
     * Get the currently selected device ID for this thread
     * @return the device ID, or null if not set
     */
    public static String getDeviceId() {
        return targetDeviceId.get();
    }

    /**
     * Check if a manual device is set for this thread
     * @return true if a device is set
     */
    public static boolean isDeviceSet() {
        return targetDevice.get() != null || targetDeviceId.get() != null;
    }

    /**
     * Execute a runnable with a specific device
     * @param device the device to use
     * @param runnable the code to execute
     */
    public static void withDevice(DeviceDescriptor device, Runnable runnable) {
        DeviceDescriptor previous = targetDevice.get();
        try {
            setDevice(device);
            runnable.run();
        } finally {
            if (previous != null) {
                setDevice(previous);
            } else {
                clearDevice();
            }
        }
    }

    /**
     * Execute a callable with a specific device
     * @param device the device to use
     * @param callable the code to execute
     * @param <T> the return type
     * @return the result of the callable
     */
    public static <T> T withDevice(DeviceDescriptor device, java.util.concurrent.Callable<T> callable) throws Exception {
        DeviceDescriptor previous = targetDevice.get();
        try {
            setDevice(device);
            return callable.call();
        } finally {
            if (previous != null) {
                setDevice(previous);
            } else {
                clearDevice();
            }
        }
    }
}
