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

import java.util.Set;

/**
 * Device descriptor for NVIDIA CUDA GPUs.
 * Provides GPU-specific information like compute capability, SM count, and memory.
 */
public class CudaDeviceDescriptor extends BaseDeviceDescriptor {

    private final int majorVersion;
    private final int minorVersion;
    private final int smCount;
    private final long totalMemory;
    private final long memoryBandwidth;
    private final int maxThreadsPerBlock;
    private final int warpSize;
    private final long estimatedFlops;

    // Peer access tracking
    private Set<Integer> peerAccessDevices;

    /**
     * Create a simple CUDA device descriptor with the given GPU index.
     * Uses default/estimated values for GPU properties.
     * For accurate properties, use the full constructor or fromNative().
     *
     * @param gpuIndex the GPU index
     */
    public CudaDeviceDescriptor(int gpuIndex) {
        // Use reasonable defaults for a modern GPU (e.g., RTX 3080-like)
        this("cuda", gpuIndex, "CUDA GPU " + gpuIndex, gpuIndex == 0,
                8, 6,  // Ampere compute capability 8.6
                68,    // ~68 SMs
                10L * 1024 * 1024 * 1024,  // 10 GB
                760L * 1024 * 1024 * 1024); // ~760 GB/s
    }

    /**
     * Create a new CUDA device descriptor
     * @param backendId the backend identifier
     * @param deviceIndex the GPU index
     * @param deviceName the GPU name
     * @param isDefault whether this is the default device
     * @param majorVersion CUDA compute capability major version
     * @param minorVersion CUDA compute capability minor version
     * @param smCount number of streaming multiprocessors
     * @param totalMemory total GPU memory in bytes
     * @param memoryBandwidth memory bandwidth in bytes per second
     */
    public CudaDeviceDescriptor(String backendId, int deviceIndex, String deviceName,
                                boolean isDefault, int majorVersion, int minorVersion,
                                int smCount, long totalMemory, long memoryBandwidth) {
        super(backendId, DeviceType.CUDA_GPU, deviceIndex, deviceName, isDefault);
        this.majorVersion = majorVersion;
        this.minorVersion = minorVersion;
        this.smCount = smCount;
        this.totalMemory = totalMemory;
        this.memoryBandwidth = memoryBandwidth;
        this.maxThreadsPerBlock = 1024;  // Standard CUDA limit
        this.warpSize = 32;

        // Estimate FLOPS based on SM count and architecture
        // Ampere (8.x): 128 FP32 cores per SM
        // Turing (7.5): 64 FP32 cores per SM
        // Volta (7.0): 64 FP32 cores per SM
        int coresPerSm = getCoresPerSm(majorVersion, minorVersion);
        // Assume 1.5 GHz clock for conservative estimate
        this.estimatedFlops = (long) smCount * coresPerSm * 2 * 1_500_000_000L;

        // Set properties
        setProperty("computeCapability", majorVersion + "." + minorVersion);
        setProperty("smCount", smCount);
        setProperty("maxThreadsPerBlock", maxThreadsPerBlock);
        setProperty("warpSize", warpSize);

        // Detect capabilities
        detectCapabilities();
    }

    private int getCoresPerSm(int major, int minor) {
        if (major >= 8) {
            return 128;  // Ampere and later
        } else if (major >= 7) {
            if (minor >= 5) {
                return 64;  // Turing
            }
            return 64;  // Volta
        } else if (major >= 6) {
            return 128;  // Pascal
        } else if (major >= 5) {
            return 128;  // Maxwell
        }
        return 64;  // Older architectures
    }

    private void detectCapabilities() {
        // All CUDA GPUs support these
        addCapability(CAPABILITY_FP64);
        addCapability(CAPABILITY_ASYNC_COPY);

        // FP16 support (compute capability 5.3+)
        if (majorVersion > 5 || (majorVersion == 5 && minorVersion >= 3)) {
            addCapability(CAPABILITY_FP16);
        }

        // INT8 support (compute capability 6.1+)
        if (majorVersion > 6 || (majorVersion == 6 && minorVersion >= 1)) {
            addCapability(CAPABILITY_INT8);
        }

        // Tensor Cores (compute capability 7.0+)
        if (majorVersion >= 7) {
            addCapability(CAPABILITY_TENSOR_CORES);
        }

        // Unified memory (compute capability 6.0+)
        if (majorVersion >= 6) {
            addCapability(CAPABILITY_UNIFIED_MEMORY);
        }

        // All CUDA GPUs support peer access (if NVLink or PCIe)
        addCapability(CAPABILITY_PEER_ACCESS);
    }

    @Override
    public long getTotalMemory() {
        return totalMemory;
    }

    @Override
    public long getAvailableMemory() {
        // This should be queried from CUDA at runtime
        // For now, return a conservative estimate
        return (long) (totalMemory * 0.8);
    }

    @Override
    public long getMemoryBandwidth() {
        return memoryBandwidth;
    }

    @Override
    public double getComputeCapability() {
        return majorVersion + (minorVersion / 10.0);
    }

    @Override
    public int getComputeUnits() {
        return smCount;
    }

    @Override
    public long getEstimatedFlops() {
        return estimatedFlops;
    }

    /**
     * Get the CUDA compute capability major version
     * @return major version
     */
    public int getMajorVersion() {
        return majorVersion;
    }

    /**
     * Get the CUDA compute capability minor version
     * @return minor version
     */
    public int getMinorVersion() {
        return minorVersion;
    }

    /**
     * Get the number of streaming multiprocessors
     * @return SM count
     */
    public int getSmCount() {
        return smCount;
    }

    /**
     * Get the maximum threads per block
     * @return max threads per block
     */
    public int getMaxThreadsPerBlock() {
        return maxThreadsPerBlock;
    }

    /**
     * Get the warp size
     * @return warp size
     */
    public int getWarpSize() {
        return warpSize;
    }

    /**
     * Check if this GPU has tensor cores
     * @return true if tensor cores are available
     */
    public boolean hasTensorCores() {
        return hasCapability(CAPABILITY_TENSOR_CORES);
    }

    /**
     * Set the devices that support peer access from this GPU
     * @param deviceIndices set of device indices with peer access
     */
    public void setPeerAccessDevices(Set<Integer> deviceIndices) {
        this.peerAccessDevices = deviceIndices;
    }

    /**
     * Check if peer access is available to another GPU
     * @param otherDeviceIndex the other GPU index
     * @return true if peer access is available
     */
    public boolean hasPeerAccessTo(int otherDeviceIndex) {
        return peerAccessDevices != null && peerAccessDevices.contains(otherDeviceIndex);
    }

    @Override
    protected long computeTransferBandwidth(DeviceDescriptor target) {
        if (target.getDeviceType() == DeviceType.CUDA_GPU) {
            CudaDeviceDescriptor cudaTarget = (CudaDeviceDescriptor) target;
            // Check for NVLink (assuming ~300 GB/s for NVLink 3.0)
            if (hasPeerAccessTo(cudaTarget.getDeviceIndex())) {
                return 300L * 1024 * 1024 * 1024;
            }
            // PCIe peer transfer (~12 GB/s)
            return 12L * 1024 * 1024 * 1024;
        } else if (target.getDeviceType() == DeviceType.CPU) {
            // GPU to CPU: PCIe bandwidth
            return 16L * 1024 * 1024 * 1024;
        }
        return 8L * 1024 * 1024 * 1024;
    }

    @Override
    public String toString() {
        return String.format("CudaDevice[%d: %s, CC %d.%d, %d SMs, %d MB]",
                getDeviceIndex(), getDeviceName(),
                majorVersion, minorVersion,
                smCount, totalMemory / (1024 * 1024));
    }
}
