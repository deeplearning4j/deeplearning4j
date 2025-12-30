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

package org.nd4j.linalg.api.memory.conf;

import lombok.Builder;
import lombok.Getter;
import lombok.experimental.SuperBuilder;
import org.nd4j.linalg.api.device.DeviceDescriptor;
import org.nd4j.linalg.api.device.DeviceType;
import org.nd4j.linalg.api.memory.enums.*;

import java.util.*;

/**
 * Extended workspace configuration for multi-backend/multi-device support.
 * Allows specifying device preferences, cross-device mirroring, and transfer policies.
 */
@Getter
@SuperBuilder
public class DeviceAwareWorkspaceConfiguration extends WorkspaceConfiguration {

    /**
     * Policy for handling cross-device data transfers
     */
    public enum TransferPolicy {
        /** Transfer on-demand when data is accessed on different device */
        ON_DEMAND,
        /** Pre-transfer to all configured devices at allocation time */
        EAGER,
        /** Never transfer - fail if wrong device accesses data */
        NONE
    }

    /**
     * Policy for selecting which device to allocate on
     */
    public enum DeviceSelectionPolicy {
        /** Use the first available device */
        FIRST_AVAILABLE,
        /** Use the device with most free memory */
        MOST_FREE_MEMORY,
        /** Round-robin across devices */
        ROUND_ROBIN,
        /** Explicitly specified device */
        EXPLICIT
    }

    /**
     * Primary device for this workspace
     */
    @Builder.Default
    private DeviceDescriptor primaryDevice = null;

    /**
     * List of devices this workspace can use
     */
    @Builder.Default
    private List<DeviceDescriptor> allowedDevices = new ArrayList<>();

    /**
     * Preferred device types in priority order
     */
    @Builder.Default
    private List<DeviceType> preferredDeviceTypes = Arrays.asList(DeviceType.CUDA_GPU, DeviceType.CPU);

    /**
     * Policy for cross-device transfers
     */
    @Builder.Default
    private TransferPolicy transferPolicy = TransferPolicy.ON_DEMAND;

    /**
     * Policy for device selection
     */
    @Builder.Default
    private DeviceSelectionPolicy deviceSelectionPolicy = DeviceSelectionPolicy.FIRST_AVAILABLE;

    /**
     * Whether to mirror allocations across multiple devices
     */
    @Builder.Default
    private boolean crossDeviceMirroring = false;

    /**
     * Devices to mirror allocations to (if crossDeviceMirroring is true)
     */
    @Builder.Default
    private Set<DeviceDescriptor> mirrorDevices = new HashSet<>();

    /**
     * Whether to enable async transfers between devices
     */
    @Builder.Default
    private boolean asyncTransfers = true;

    /**
     * Maximum pending async transfers before blocking
     */
    @Builder.Default
    private int maxPendingTransfers = 16;

    /**
     * Whether to track device-specific memory usage
     */
    @Builder.Default
    private boolean trackPerDeviceMemory = true;

    /**
     * Initial size per device (overrides base initialSize for multi-device)
     */
    @Builder.Default
    private Map<DeviceType, Long> perDeviceInitialSize = new HashMap<>();

    /**
     * Max size per device (overrides base maxSize for multi-device)
     */
    @Builder.Default
    private Map<DeviceType, Long> perDeviceMaxSize = new HashMap<>();

    /**
     * Whether to use unified memory where supported (CUDA managed memory)
     */
    @Builder.Default
    private boolean useUnifiedMemory = false;

    /**
     * Whether to enable coherence tracking for this workspace
     */
    @Builder.Default
    private boolean enableCoherenceTracking = true;

    /**
     * Get the initial size for a specific device type
     * @param deviceType the device type
     * @return the initial size, or base initialSize if not specified
     */
    public long getInitialSizeForDevice(DeviceType deviceType) {
        Long size = perDeviceInitialSize.get(deviceType);
        return size != null ? size : getInitialSize();
    }

    /**
     * Get the max size for a specific device type
     * @param deviceType the device type
     * @return the max size, or base maxSize if not specified
     */
    public long getMaxSizeForDevice(DeviceType deviceType) {
        Long size = perDeviceMaxSize.get(deviceType);
        return size != null ? size : getMaxSize();
    }

    /**
     * Check if a device is allowed for this workspace
     * @param device the device to check
     * @return true if allowed
     */
    public boolean isDeviceAllowed(DeviceDescriptor device) {
        if (allowedDevices == null || allowedDevices.isEmpty()) {
            return true;  // No restrictions
        }
        return allowedDevices.contains(device);
    }

    /**
     * Create a builder from an existing WorkspaceConfiguration
     * @param base the base configuration
     * @return builder pre-populated with base values
     */
    public static DeviceAwareWorkspaceConfigurationBuilder<?, ?> fromBase(WorkspaceConfiguration base) {
        return DeviceAwareWorkspaceConfiguration.builder()
                .policyAllocation(base.getPolicyAllocation())
                .policySpill(base.getPolicySpill())
                .policyMirroring(base.getPolicyMirroring())
                .policyLearning(base.getPolicyLearning())
                .policyReset(base.getPolicyReset())
                .policyLocation(base.getPolicyLocation())
                .initialSize(base.getInitialSize())
                .minSize(base.getMinSize())
                .maxSize(base.getMaxSize())
                .overallocationLimit(base.getOverallocationLimit())
                .cyclesBeforeInitialization(base.getCyclesBeforeInitialization());
    }

    /**
     * Create a simple GPU-preferred configuration
     * @param initialSize initial workspace size
     * @return configuration preferring GPU
     */
    public static DeviceAwareWorkspaceConfiguration gpuPreferred(long initialSize) {
        return DeviceAwareWorkspaceConfiguration.builder()
                .initialSize(initialSize)
                .preferredDeviceTypes(Arrays.asList(DeviceType.CUDA_GPU, DeviceType.CPU))
                .deviceSelectionPolicy(DeviceSelectionPolicy.FIRST_AVAILABLE)
                .transferPolicy(TransferPolicy.ON_DEMAND)
                .build();
    }

    /**
     * Create a simple CPU-only configuration
     * @param initialSize initial workspace size
     * @return configuration for CPU only
     */
    public static DeviceAwareWorkspaceConfiguration cpuOnly(long initialSize) {
        return DeviceAwareWorkspaceConfiguration.builder()
                .initialSize(initialSize)
                .preferredDeviceTypes(Collections.singletonList(DeviceType.CPU))
                .deviceSelectionPolicy(DeviceSelectionPolicy.EXPLICIT)
                .transferPolicy(TransferPolicy.NONE)
                .build();
    }

    /**
     * Create a mirrored configuration that keeps data on both CPU and GPU
     * @param initialSize initial workspace size
     * @return configuration with cross-device mirroring
     */
    public static DeviceAwareWorkspaceConfiguration mirrored(long initialSize) {
        return DeviceAwareWorkspaceConfiguration.builder()
                .initialSize(initialSize)
                .preferredDeviceTypes(Arrays.asList(DeviceType.CUDA_GPU, DeviceType.CPU))
                .crossDeviceMirroring(true)
                .transferPolicy(TransferPolicy.EAGER)
                .build();
    }
}
