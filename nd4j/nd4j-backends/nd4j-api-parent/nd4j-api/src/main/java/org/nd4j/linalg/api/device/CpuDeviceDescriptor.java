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

/**
 * Device descriptor for CPU devices.
 * Provides CPU-specific information like core count, cache sizes, and SIMD capabilities.
 */
public class CpuDeviceDescriptor extends BaseDeviceDescriptor {

    private final int coreCount;
    private final int threadCount;
    private final long totalMemory;
    private final long memoryBandwidth;
    private final String architecture;
    private final long estimatedFlops;

    /**
     * Create a simple CPU device descriptor with the given index.
     * Uses system information for other properties.
     *
     * @param deviceIndex the device index
     */
    public CpuDeviceDescriptor(int deviceIndex) {
        this("native", deviceIndex, "CPU-" + deviceIndex, deviceIndex == 0,
                Runtime.getRuntime().availableProcessors(),
                Runtime.getRuntime().availableProcessors(),
                Runtime.getRuntime().maxMemory(),
                System.getProperty("os.arch", "unknown"));
    }

    /**
     * Create a new CPU device descriptor
     * @param backendId the backend identifier
     * @param deviceIndex the device index
     * @param deviceName the device name
     * @param isDefault whether this is the default device
     * @param coreCount number of physical cores
     * @param threadCount number of logical threads
     * @param totalMemory total system memory in bytes
     * @param architecture CPU architecture string
     */
    public CpuDeviceDescriptor(String backendId, int deviceIndex, String deviceName,
                               boolean isDefault, int coreCount, int threadCount,
                               long totalMemory, String architecture) {
        super(backendId, DeviceType.CPU, deviceIndex, deviceName, isDefault);
        this.coreCount = coreCount;
        this.threadCount = threadCount;
        this.totalMemory = totalMemory;
        this.architecture = architecture;

        // Estimate memory bandwidth (DDR4-3200 ~ 25 GB/s per channel, assume dual channel)
        this.memoryBandwidth = 50L * 1024 * 1024 * 1024;

        // Estimate FLOPS based on core count and frequency
        // Conservative estimate: 8 FLOPS/cycle * 3 GHz * cores
        this.estimatedFlops = 8L * 3_000_000_000L * coreCount;

        // Set properties
        setProperty("coreCount", coreCount);
        setProperty("threadCount", threadCount);
        setProperty("architecture", architecture);

        // Detect capabilities
        detectCapabilities();
    }

    /**
     * Create a CPU device descriptor from system information
     * @param backendId the backend identifier
     * @return a new CPU device descriptor
     */
    public static CpuDeviceDescriptor fromSystem(String backendId) {
        Runtime runtime = Runtime.getRuntime();
        int cores = runtime.availableProcessors();
        long memory = runtime.maxMemory();

        String arch = System.getProperty("os.arch", "unknown");
        String cpuName = System.getProperty("os.name", "CPU") + " " + arch;

        // Try to get better CPU name on Linux
        try {
            String vendor = System.getenv("PROCESSOR_IDENTIFIER");
            if (vendor != null && !vendor.isEmpty()) {
                cpuName = vendor;
            }
        } catch (Exception e) {
            // Ignore
        }

        return new CpuDeviceDescriptor(backendId, 0, cpuName, true,
                cores, cores, memory, arch);
    }

    private void detectCapabilities() {
        String arch = architecture.toLowerCase();

        // Always support FP64 on CPU
        addCapability(CAPABILITY_FP64);

        // Detect SIMD capabilities based on architecture
        if (arch.contains("amd64") || arch.contains("x86_64") || arch.contains("x86")) {
            // x86 architecture - assume at least SSE2
            addCapability("sse2");

            // Check for AVX support via system property or environment
            String avxSupport = System.getProperty("nd4j.cpu.avx", System.getenv("ND4J_CPU_AVX"));
            if (avxSupport != null) {
                if (avxSupport.contains("512")) {
                    addCapability(CAPABILITY_AVX);
                    addCapability(CAPABILITY_AVX2);
                    addCapability(CAPABILITY_AVX512);
                } else if (avxSupport.contains("2")) {
                    addCapability(CAPABILITY_AVX);
                    addCapability(CAPABILITY_AVX2);
                } else if (avxSupport.contains("1") || avxSupport.equalsIgnoreCase("true")) {
                    addCapability(CAPABILITY_AVX);
                }
            } else {
                // Assume AVX2 on modern x86_64 systems
                addCapability(CAPABILITY_AVX);
                addCapability(CAPABILITY_AVX2);
            }
        } else if (arch.contains("aarch64") || arch.contains("arm64")) {
            // ARM64 architecture
            addCapability(CAPABILITY_NEON);

            // Check for SVE support
            String sveSupport = System.getProperty("nd4j.cpu.sve", System.getenv("ND4J_CPU_SVE"));
            if ("true".equalsIgnoreCase(sveSupport)) {
                addCapability(CAPABILITY_SVE);
            }
        } else if (arch.contains("arm")) {
            // 32-bit ARM
            addCapability(CAPABILITY_NEON);
        }

        // Always support async operations on CPU
        addCapability(CAPABILITY_ASYNC_COPY);
    }

    @Override
    public long getTotalMemory() {
        return totalMemory;
    }

    @Override
    public long getAvailableMemory() {
        Runtime runtime = Runtime.getRuntime();
        return runtime.maxMemory() - runtime.totalMemory() + runtime.freeMemory();
    }

    @Override
    public long getMemoryBandwidth() {
        return memoryBandwidth;
    }

    @Override
    public double getComputeCapability() {
        // Return a score based on cores and capabilities
        double score = coreCount * 1.0;
        if (hasCapability(CAPABILITY_AVX512)) {
            score *= 4.0;  // 512-bit SIMD
        } else if (hasCapability(CAPABILITY_AVX2)) {
            score *= 2.0;  // 256-bit SIMD
        } else if (hasCapability(CAPABILITY_AVX)) {
            score *= 1.5;  // 256-bit SIMD (partial)
        }
        return score;
    }

    @Override
    public int getComputeUnits() {
        return coreCount;
    }

    @Override
    public long getEstimatedFlops() {
        return estimatedFlops;
    }

    /**
     * Get the number of physical CPU cores
     * @return core count
     */
    public int getCoreCount() {
        return coreCount;
    }

    /**
     * Get the number of logical threads (with hyperthreading)
     * @return thread count
     */
    public int getThreadCount() {
        return threadCount;
    }

    /**
     * Get the CPU architecture string
     * @return architecture
     */
    public String getArchitecture() {
        return architecture;
    }

    @Override
    protected long computeTransferBandwidth(DeviceDescriptor target) {
        if (target.getDeviceType() == DeviceType.CPU) {
            // CPU to CPU: memory bandwidth
            return memoryBandwidth;
        } else if (target.getDeviceType() == DeviceType.CUDA_GPU) {
            // CPU to GPU: PCIe bandwidth (Gen3 x16 ~ 16 GB/s, Gen4 x16 ~ 32 GB/s)
            return 16L * 1024 * 1024 * 1024;
        }
        // Default PCIe bandwidth
        return 8L * 1024 * 1024 * 1024;
    }
}
