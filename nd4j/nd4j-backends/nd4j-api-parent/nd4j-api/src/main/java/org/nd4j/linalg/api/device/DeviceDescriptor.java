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

package org.nd4j.linalg.api.device;

import java.util.Map;
import java.util.Set;

/**
 * Interface representing a compute device (CPU, GPU, TPU, etc.) in ND4J.
 * Provides device properties, capabilities, and performance characteristics
 * used for routing decisions and resource management.
 */
public interface DeviceDescriptor {

    /**
     * Get the unique identifier for this device
     * Format: "{backend}:{type}:{index}" e.g., "native:cpu:0" or "cuda:gpu:1"
     * @return unique device identifier
     */
    String getDeviceId();

    /**
     * Get the type of this device
     * @return device type
     */
    DeviceType getDeviceType();

    /**
     * Get the device index within its type (e.g., GPU 0, GPU 1)
     * @return device index
     */
    int getDeviceIndex();

    /**
     * Get the human-readable name of the device
     * @return device name
     */
    String getDeviceName();

    /**
     * Get the human-readable name of the device (alias for getDeviceName)
     * @return device name
     */
    default String getName() {
        return getDeviceName();
    }

    /**
     * Get the type of this device (alias for getDeviceType)
     * @return device type
     */
    default DeviceType getType() {
        return getDeviceType();
    }

    /**
     * Get the total memory available on this device in bytes
     * @return total memory in bytes
     */
    long getTotalMemory();

    /**
     * Get the currently available (free) memory on this device in bytes
     * @return available memory in bytes
     */
    long getAvailableMemory();

    /**
     * Get the memory bandwidth in bytes per second
     * @return memory bandwidth
     */
    long getMemoryBandwidth();

    /**
     * Get the estimated transfer bandwidth to another device in bytes per second
     * @param target the target device
     * @return transfer bandwidth, or 0 if transfer not possible
     */
    long getTransferBandwidthTo(DeviceDescriptor target);

    /**
     * Check if this device can transfer data directly to another device
     * @param target the target device
     * @return true if direct transfer is possible
     */
    boolean canTransferTo(DeviceDescriptor target);

    /**
     * Get the compute capability/performance rating of this device
     * Higher values indicate more capable devices
     * @return compute capability score
     */
    double getComputeCapability();

    /**
     * Get the number of compute units (cores, SMs, etc.)
     * @return number of compute units
     */
    int getComputeUnits();

    /**
     * Check if this device supports a specific capability
     * @param capability the capability to check
     * @return true if supported
     */
    boolean hasCapability(String capability);

    /**
     * Get all capabilities supported by this device
     * @return set of capability strings
     */
    Set<String> getCapabilities();

    /**
     * Get device-specific properties
     * @return map of property name to value
     */
    Map<String, Object> getProperties();

    /**
     * Get a specific property value
     * @param key the property key
     * @return the property value, or null if not present
     */
    Object getProperty(String key);

    /**
     * Check if this device is currently available for computation
     * @return true if available
     */
    boolean isAvailable();

    /**
     * Get the backend identifier that owns this device
     * @return backend identifier
     */
    String getBackendId();

    /**
     * Get the estimated FLOPS (floating point operations per second) for this device
     * @return estimated FLOPS
     */
    long getEstimatedFlops();

    /**
     * Check if this is the default device for its backend
     * @return true if default device
     */
    boolean isDefault();

    // Common capability constants
    String CAPABILITY_FP16 = "fp16";
    String CAPABILITY_FP64 = "fp64";
    String CAPABILITY_INT8 = "int8";
    String CAPABILITY_TENSOR_CORES = "tensor_cores";
    String CAPABILITY_UNIFIED_MEMORY = "unified_memory";
    String CAPABILITY_PEER_ACCESS = "peer_access";
    String CAPABILITY_ASYNC_COPY = "async_copy";
    String CAPABILITY_AVX = "avx";
    String CAPABILITY_AVX2 = "avx2";
    String CAPABILITY_AVX512 = "avx512";
    String CAPABILITY_NEON = "neon";
    String CAPABILITY_SVE = "sve";

    // =========================================================================
    // Static Factory Methods for Common Device Types
    // =========================================================================

    /**
     * Create a CPU device descriptor.
     *
     * @return CPU device descriptor
     */
    static DeviceDescriptor cpu() {
        return new CpuDeviceDescriptor(0);
    }

    /**
     * Create a CPU device descriptor with a specific index.
     *
     * @param index the CPU index (for multi-socket systems)
     * @return CPU device descriptor
     */
    static DeviceDescriptor cpu(int index) {
        return new CpuDeviceDescriptor(index);
    }

    /**
     * Create a CUDA GPU device descriptor.
     *
     * @param gpuIndex the GPU index
     * @return CUDA device descriptor
     */
    static DeviceDescriptor cuda(int gpuIndex) {
        return new CudaDeviceDescriptor(gpuIndex);
    }

    /**
     * Create a generic accelerator descriptor when the backend does not expose
     * richer hardware metadata through a specialized descriptor.
     */
    static DeviceDescriptor accelerator(String backendId, DeviceType deviceType, int deviceIndex) {
        return new GenericAcceleratorDeviceDescriptor(backendId, deviceType, deviceIndex);
    }

    /**
     * Create a device descriptor from a device ID string.
     *
     * @param deviceId the device ID (e.g., "cuda:gpu:0", "native:cpu:0")
     * @return device descriptor
     */
    static DeviceDescriptor fromId(String deviceId) {
        if (deviceId == null || deviceId.isEmpty()) {
            return cpu();
        }

        String[] parts = deviceId.split(":");
        if (parts.length < 2) {
            return cpu();
        }

        String backendId = parts[0].toLowerCase();
        int index = parts.length >= 3 ? Integer.parseInt(parts[2]) : 0;
        if ("native".equals(backendId) || "cpu".equals(parts[1])) {
            return cpu(index);
        }
        if ("cuda".equals(backendId)) {
            return cuda(index);
        }
        return accelerator(backendId, DeviceType.fromIdentifier(backendId), index);
    }

    /**
     * Create a device descriptor for the default device.
     *
     * @return default device descriptor (usually CPU)
     */
    static DeviceDescriptor defaultDevice() {
        return cpu();
    }
}

/**
 * Descriptor for accelerators whose backend currently exposes identity and
 * device count, but not detailed hardware telemetry.
 */
final class GenericAcceleratorDeviceDescriptor extends BaseDeviceDescriptor {

    GenericAcceleratorDeviceDescriptor(String backendId, DeviceType deviceType, int deviceIndex) {
        super(backendId, deviceType, deviceIndex,
                deviceType.getIdentifier().toUpperCase() + " device " + deviceIndex,
                deviceIndex == 0);
    }

    @Override
    public long getTotalMemory() {
        return 0;
    }

    @Override
    public long getAvailableMemory() {
        return 0;
    }

    @Override
    public long getMemoryBandwidth() {
        return 0;
    }

    @Override
    public double getComputeCapability() {
        return 0;
    }

    @Override
    public int getComputeUnits() {
        return 0;
    }

    @Override
    public long getEstimatedFlops() {
        return 0;
    }
}
