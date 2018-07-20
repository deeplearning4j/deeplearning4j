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

package org.nd4j.jita.conf;

import lombok.Getter;
import lombok.NonNull;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.jita.allocator.enums.Aggressiveness;
import org.nd4j.jita.allocator.enums.AllocationStatus;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * @author raver119@gmail.com
 */
@Slf4j
public class Configuration implements Serializable {

    public enum ExecutionModel {
        SEQUENTIAL, ASYNCHRONOUS, OPTIMIZED,
    }

    public enum AllocationModel {
        DIRECT, CACHE_HOST, CACHE_ALL,
    }

    public enum MemoryModel {
        IMMEDIATE, DELAYED
    }

    @Getter
    @Deprecated //Only SEQUENTIAL is supported
    private ExecutionModel executionModel = ExecutionModel.SEQUENTIAL;

    @Getter
    private AllocationModel allocationModel = AllocationModel.CACHE_ALL;

    @Getter
    private AllocationStatus firstMemory = AllocationStatus.DEVICE;

    @Getter
    private MemoryModel memoryModel = MemoryModel.IMMEDIATE;

    @Getter
    private boolean debug = false;

    @Getter
    private boolean verbose = false;

    @Getter
    private boolean fillDashboard = false;

    private boolean forceSingleGPU = false;

    @Getter
    private long noGcWindowMs = 100;

    /**
     * Keep this value between 0.01 and 0.95 please
     */
    @Getter
    private double maximumDeviceMemoryUsed = 0.85;

    /**
     * Minimal number of activations for relocation threshold
     */
    @Getter
    private int minimumRelocationThreshold = 5;

    /**
     * Minimal guaranteed TTL for memory chunk
     */
    @Getter
    private long minimumTTLMilliseconds = 10 * 1000L;

    /**
     * Number of buckets/garbage collectors for host memory
     */
    @Getter
    private int numberOfGcThreads = 6;

    /**
     * Deallocation aggressiveness
     */
    @Deprecated
    @Getter
    private Aggressiveness hostDeallocAggressiveness = Aggressiveness.REASONABLE;

    @Deprecated
    @Getter
    private Aggressiveness gpuDeallocAggressiveness = Aggressiveness.REASONABLE;

    /**
     * Allocation aggressiveness
     */
    @Deprecated
    @Getter
    private Aggressiveness gpuAllocAggressiveness = Aggressiveness.REASONABLE;


    /**
     * Maximum allocated per-device memory, in bytes
     */
    @Getter
    private long maximumDeviceAllocation = 4 * 1024 * 1024 * 1024L;


    /**
     * Maximum allocatable zero-copy/pinned/pageable memory
     */
    @Getter
    private long maximumZeroAllocation = Runtime.getRuntime().maxMemory() + (500 * 1024 * 1024L);

    /**
     * True if allowed, false if relocation required
     */
    @Getter
    private boolean crossDeviceAccessAllowed = true;

    /**
     * True, if allowed, false otherwise
     */
    @Getter
    private boolean zeroCopyFallbackAllowed = false;

    /**
     * Maximum length of single memory chunk
     */
    @Getter
    private long maximumSingleHostAllocation = Long.MAX_VALUE;

    @Getter
    private long maximumSingleDeviceAllocation = 1024 * 1024 * 1024L;

    @Getter
    private List<Integer> availableDevices = new ArrayList<>();

    @Getter
    private List<Integer> bannedDevices = new ArrayList<>();

    @Getter
    private int maximumGridSize = 4096;

    @Getter
    private int maximumBlockSize = 256;

    @Getter
    private int minimumBlockSize = 32;

    @Getter
    private long maximumHostCache = 3 * 1024 * 1024 * 1024L;

    @Getter
    private long maximumDeviceCache = 512L * 1024L * 1024L;

    @Getter
    private boolean usePreallocation = false;

    @Getter
    private int preallocationCalls = 10;

    @Getter
    private long maximumHostCacheableLength = 100663296;

    @Getter
    private long maximumDeviceCacheableLength = 16L * 1024L * 1024L;

    @Getter
    private int commandQueueLength = 3;

    @Getter
    private int commandLanesNumber = 4;

    @Getter
    private int debugTriggered = 0;

    @Getter
    private int poolSize = 32;

    private final AtomicBoolean initialized = new AtomicBoolean(false);

    public boolean isInitialized() {
        return initialized.get();
    }

    public void setInitialized() {
        this.initialized.compareAndSet(false, true);
    }

    /**
     * Environment variables for
     * controlling cuda.
     */
    private static final String MAX_BLOCK_SIZE = "ND4J_CUDA_MAX_BLOCK_SIZE";
    private static final String MIN_BLOCK_SIZE = "ND4J_CUDA_MIN_BLOCK_SIZE";
    private static final String MAX_GRID_SIZE = "ND4J_CUDA_MAX_GRID_SIZE";
    private static final String DEBUG_ENABLED = "ND4J_DEBUG";
    private static final String VERBOSE = "ND4J_VERBOSE";
    private static final String USE_PREALLOCATION = "ND4J_CUDA_USE_PREALLOCATION";
    private static final String MAX_DEVICE_CACHE = "ND4J_CUDA_MAX_DEVICE_CACHE";
    private static final String MAX_HOST_CACHE = "ND4J_CUDA_MAX_HOST_CACHE";
    private static final String MAX_DEVICE_ALLOCATION = "ND4J_CUDA_MAX_DEVICE_ALLOCATION";
    private static final String FORCE_SINGLE_GPU = "ND4J_CUDA_FORCE_SINGLE_GPU";


    private void parseEnvironmentVariables() {

        // Do not call System.getenv(): Accessing all variables requires higher security privileges
        if (System.getenv(MAX_BLOCK_SIZE) != null) {
            try {
                int var = Integer.parseInt(System.getenv(MAX_BLOCK_SIZE));
                setMaximumBlockSize(var);
            } catch (Exception e) {
                log.error("Can't parse {}: [{}]", MAX_BLOCK_SIZE, System.getenv(MAX_BLOCK_SIZE));
            }
        }

        if (System.getenv(MIN_BLOCK_SIZE) != null) {
            try {
                int var = Integer.parseInt(System.getenv(MIN_BLOCK_SIZE));
                setMinimumBlockSize(var);
            } catch (Exception e) {
                log.error("Can't parse {}: [{}]", MIN_BLOCK_SIZE, System.getenv(MIN_BLOCK_SIZE));
            }
        }

        if (System.getenv(MAX_GRID_SIZE) != null) {
            try {
                int var = Integer.parseInt(System.getenv(MAX_GRID_SIZE));
                setMaximumGridSize(var);
            } catch (Exception e) {
                log.error("Can't parse {}: [{}]", MAX_GRID_SIZE, System.getenv(MAX_GRID_SIZE));
            }
        }

        if (System.getenv(DEBUG_ENABLED) != null) {
            try {
                boolean var = Boolean.parseBoolean(System.getenv(DEBUG_ENABLED));
                enableDebug(var);
            } catch (Exception e) {
                log.error("Can't parse {}: [{}]", DEBUG_ENABLED, System.getenv(DEBUG_ENABLED));
            }
        }

        if (System.getenv(FORCE_SINGLE_GPU) != null) {
            try {
                boolean var = Boolean.parseBoolean(System.getenv(FORCE_SINGLE_GPU));
                allowMultiGPU(!var);
            } catch (Exception e) {
                log.error("Can't parse {}: [{}]", FORCE_SINGLE_GPU, System.getenv(FORCE_SINGLE_GPU));
            }
        }

        if (System.getenv(VERBOSE) != null) {
            try {
                boolean var = Boolean.parseBoolean(System.getenv(VERBOSE));
                setVerbose(var);
            } catch (Exception e) {
                log.error("Can't parse {}: [{}]", VERBOSE, System.getenv(VERBOSE));
            }
        }

        if (System.getenv(USE_PREALLOCATION) != null) {
            try {
                boolean var = Boolean.parseBoolean(System.getenv(USE_PREALLOCATION));
                allowPreallocation(var);
            } catch (Exception e) {
                log.error("Can't parse {}: [{}]", USE_PREALLOCATION, System.getenv(USE_PREALLOCATION));
            }
        }

        if (System.getenv(MAX_DEVICE_CACHE) != null) {
            try {
                long var = Long.parseLong(System.getenv(MAX_DEVICE_CACHE));
                setMaximumDeviceCache(var);
            } catch (Exception e) {
                log.error("Can't parse {}: [{}]", MAX_DEVICE_CACHE, System.getenv(MAX_DEVICE_CACHE));
            }
        }


        if (System.getenv(MAX_HOST_CACHE) != null) {
            try {
                long var = Long.parseLong(System.getenv(MAX_HOST_CACHE));
                setMaximumHostCache(var);
            } catch (Exception e) {
                log.error("Can't parse {}: [{}]", MAX_HOST_CACHE, System.getenv(MAX_HOST_CACHE));
            }
        }

        if (System.getenv(MAX_DEVICE_ALLOCATION) != null) {
            try {
                long var = Long.parseLong(System.getenv(MAX_DEVICE_ALLOCATION));
                setMaximumSingleDeviceAllocation(var);
            } catch (Exception e) {
                log.error("Can't parse {}: [{}]", MAX_DEVICE_ALLOCATION, System.getenv(MAX_DEVICE_ALLOCATION));
            }
        }

    }

    /**
     * This method enables/disables
     *
     * @param reallyEnable
     * @return
     */
    public Configuration enableDashboard(boolean reallyEnable) {
        fillDashboard = reallyEnable;
        return this;
    }

    /**
     * Per-device resources pool size. Streams, utility memory
     *
     * @param poolSize
     * @return
     */
    public Configuration setPoolSize(int poolSize) {
        if (poolSize < 8)
            throw new IllegalStateException("poolSize can't be lower then 8");
        this.poolSize = poolSize;
        return this;
    }

    public Configuration triggerDebug(int code) {
        this.debugTriggered = code;
        return this;
    }

    public Configuration setMinimumRelocationThreshold(int threshold) {
        this.maximumDeviceAllocation = Math.max(2, threshold);

        return this;
    }

    /**
     * This method allows you to specify maximum memory cache for host memory
     *
     * @param maxCache
     * @return
     */
    public Configuration setMaximumHostCache(long maxCache) {
        this.maximumHostCache = maxCache;
        return this;
    }

    /**
     * This method allows you to specify maximum memory cache per device
     *
     * @param maxCache
     * @return
     */
    public Configuration setMaximumDeviceCache(long maxCache) {
        this.maximumDeviceCache = maxCache;
        return this;
    }

    /**
     * This method allows you to specify max per-device memory use.
     *
     * PLEASE NOTE: Accepted value range is 0.01 > x < 0.95
     *
     * @param percentage
     */
    public Configuration setMaximumDeviceMemoryUsed(double percentage) {
        if (percentage < 0.02 || percentage > 0.95) {
            this.maximumDeviceMemoryUsed = 0.85;
        } else
            this.maximumDeviceMemoryUsed = percentage;

        return this;
    }

    public Configuration() {
        parseEnvironmentVariables();
    }


    void updateDevice() {
        int cnt = Nd4j.getAffinityManager().getNumberOfDevices();

        if (cnt == 0)
            throw new RuntimeException("No CUDA devices were found in system");

        for (int i = 0; i < cnt; i++) {
            availableDevices.add(i);
        }
    }



    /**
     * This method checks, if GPU subsystem supports cross-device P2P access over PCIe.
     *
     * PLEASE NOTE: This method also returns TRUE if system has only one device. This is done to guarantee reallocation avoidance within same device.
     *
     * @return
     */
    public boolean isP2PSupported() {
        return NativeOpsHolder.getInstance().getDeviceNativeOps().isP2PAvailable();
    }

    /**
     * This method allows you to ban specific device.
     *
     * PLEASE NOTE: This method
     *
     * @param deviceId
     * @return
     */
    public Configuration banDevice(@NonNull Integer deviceId) {
        if (!availableDevices.contains(deviceId))
            return this;

        if (!bannedDevices.contains(deviceId)) {
            bannedDevices.add(deviceId);
        }

        availableDevices.remove(deviceId);

        return this;
    }

    /**
     * This method forces specific device to be used. All other devices present in system will be ignored.
     *
     * @param deviceId
     * @return
     */
    public Configuration useDevice(@NonNull Integer deviceId) {
        return useDevices(deviceId);
    }

    /**
     * This method forces specific devices to be used. All other devices present in system will be ignored.
     *
     * @param devices
     * @return
     */
    public Configuration useDevices(@NonNull int... devices) {
        List<Integer> usableDevices = new ArrayList<>();
        for (int device : devices) {
            if (!availableDevices.contains(device)) {
                log.warn("Non-existent device [{}] requested, ignoring...", device);
            } else {
                if (!usableDevices.contains(device))
                    usableDevices.add(device);
            }

        }

        if (usableDevices.size() > 0) {
            availableDevices.clear();
            availableDevices.addAll(usableDevices);
        }

        return this;
    }

    /**
     * This method allows you to set maximum host allocation. However, it's recommended to leave it as default: Xmx + something.
     *
     * @param max amount of memory in bytes
     */
    public Configuration setMaximumZeroAllocation(long max) {
        long xmx = Runtime.getRuntime().maxMemory();
        if (max < xmx)
            log.warn("Setting maximum memory below -Xmx value can cause problems");

        if (max <= 0)
            throw new IllegalStateException("You can't set maximum host memory <= 0");

        maximumZeroAllocation = max;

        return this;
    }

    /**
     * This method allows you to set maximum device allocation. It's recommended to keep it equal to MaximumZeroAllocation
     * @param max
     */
    public Configuration setMaximumDeviceAllocation(long max) {
        if (max < 0)
            throw new IllegalStateException("You can't set maximum device memory < 0");

        return this;
    }

    /**
     * This method allows to specify maximum single allocation on host.
     *
     * Default value: Long.MAX_VALUE
     *
     * @param max
     * @return
     */
    public Configuration setMaximumSingleHostAllocation(long max) {
        this.maximumSingleHostAllocation = max;

        return this;
    }

    /**
     * This method allows to specify maximum single allocation on device.
     *
     * Default value: Long.MAX_VALUE
     *
     * @param max
     * @return
     */
    public Configuration setMaximumSingleDeviceAllocation(long max) {
        this.maximumSingleDeviceAllocation = max;

        return this;
    }

    /**
     * This method allows to specify max gridDim for kernel launches.
     *
     * Default value: 128
     *
     * @param gridDim
     * @return
     */
    public Configuration setMaximumGridSize(int gridDim) {
        if (gridDim <= 7 || gridDim > 8192)
            throw new IllegalStateException("Please keep gridDim in range [8...8192]");

        this.maximumGridSize = gridDim;

        return this;
    }

    /**
     * This methos allows to specify max blockSize for kernel launches
     *
     * Default value: -1 (that means pick value automatically, device occupancy dependent)
     *
     * @param blockDim
     * @return
     */
    public Configuration setMaximumBlockSize(int blockDim) {
        if (blockDim < 32 || blockDim > 768)
            throw new IllegalStateException("Please keep blockDim in range [32...768]");


        this.maximumBlockSize = blockDim;

        return this;
    }

    public Configuration setMinimumBlockSize(int blockDim) {
        if (blockDim < 32 || blockDim > 768)
            throw new IllegalStateException("Please keep blockDim in range [32...768]");


        this.minimumBlockSize = blockDim;

        return this;
    }

    /**
     * With debug enabled all CUDA launches will become synchronous, with forced stream synchronizations after calls.
     *
     * Default value: false;
     *
     * @return
     */
    public Configuration enableDebug(boolean debug) {
        this.debug = debug;
        return this;
    }

    public Configuration setVerbose(boolean verbose) {
        this.verbose = verbose;
        return this;
    }

    /**
     * Enables/disables P2P memory access for multi-gpu
     *
     * @param reallyAllow
     * @return
     */
    public Configuration allowCrossDeviceAccess(boolean reallyAllow) {
        this.crossDeviceAccessAllowed = reallyAllow;

        return this;
    }

    /**
     * This method allows to specify execution model for matrix/blas operations
     *
     * SEQUENTIAL: Issue commands in order Java compiler sees them.
     * ASYNCHRONOUS: Issue commands asynchronously, if that's possible.
     * OPTIMIZED: Not implemented yet. Equals to asynchronous for now.
     *
     * Default value: SEQUENTIAL
     *
     * @param executionModel
     * @return
     * @deprecated Only ExecutionModel.SEQUENTIAL is supported
     */
    @Deprecated
    public Configuration setExecutionModel(@NonNull ExecutionModel executionModel) {
        if(executionModel != ExecutionModel.SEQUENTIAL){
            throw new IllegalArgumentException("Only ExecutionModel.SEQUENTIAL is supported");
        }
        this.executionModel = ExecutionModel.SEQUENTIAL;
        return this;
    }

    /**
     * This method allows to specify allocation model for memory.
     *
     * DIRECT: Do not cache anything, release memory as soon as it's not used.
     * CACHE_HOST: Cache host memory only, Device memory (if any) will use DIRECT mode.
     * CACHE_ALL: All memory will be cached.
     *
     * Defailt value: CACHE_ALL
     *
     * @param allocationModel
     * @return
     */
    public Configuration setAllocationModel(@NonNull AllocationModel allocationModel) {
        this.allocationModel = allocationModel;

        return this;
    }

    /**
     * This method allows to specify initial memory to be used within system.
     * HOST: all data is located on host memory initially, and gets into DEVICE, if used frequent enough
     * DEVICE: all memory is located on device.
     * DELAYED: memory allocated on HOST first, and on first use gets moved to DEVICE
     *
     * PLEASE NOTE: For device memory all data still retains on host side as well.
     *
     * Default value: DEVICE
     * @param initialMemory
     * @return
     */
    public Configuration setFirstMemory(@NonNull AllocationStatus initialMemory) {
        if (initialMemory != AllocationStatus.DEVICE && initialMemory != AllocationStatus.HOST
                        && initialMemory != AllocationStatus.DELAYED)
            throw new IllegalStateException("First memory should be either [HOST], [DEVICE] or [DELAYED]");

        this.firstMemory = initialMemory;

        return this;
    }

    /**
     * NOT IMPLEMENTED YET
     * @param reallyAllow
     * @return
     */
    public Configuration allowFallbackFromDevice(boolean reallyAllow) {
        this.zeroCopyFallbackAllowed = reallyAllow;
        return this;
    }

    /**
     * This method allows you to set number of threads that'll handle memory releases on native side.
     *
     * Default value: 4
     * @return
     */
    public Configuration setNumberOfGcThreads(int numThreads) {
        if (numThreads <= 0 || numThreads > 20)
            throw new IllegalStateException("Please, use something in range of [1..20] as number of GC threads");

        if (!isInitialized())
            this.numberOfGcThreads = numThreads;

        return this;
    }

    /**
     * This method allows to specify maximum length of single memory chunk that's allowed to be cached.
     * Please note: -1 value totally disables limits here.
     *
     * Default value: 96 MB
     * @param maxLen
     * @return
     */
    public Configuration setMaximumHostCacheableLength(long maxLen) {
        this.maximumHostCacheableLength = maxLen;

        return this;
    }

    /**
     * This method allows to specify maximum length of single memory chunk that's allowed to be cached.
     * Please note: -1 value totally disables limits here.
     *
     * Default value: 96 MB
     * @param maxLen
     * @return
     */
    public Configuration setMaximumDeviceCacheableLength(long maxLen) {
        this.maximumDeviceCacheableLength = maxLen;

        return this;
    }

    /**
     * If set to true, each non-cached allocation request will cause few additional allocations,
     *
     * Default value: true
     *
     * @param reallyAllow
     * @return
     */
    public Configuration allowPreallocation(boolean reallyAllow) {
        this.usePreallocation = reallyAllow;

        return this;
    }

    /**
     * This method allows to specify number of preallocation calls done by cache subsystem in parallel, to serve later requests.
     *
     * Default value: 25
     *
     * @param numCalls
     * @return
     */
    public Configuration setPreallocationCalls(int numCalls) {
        if (numCalls < 0 || numCalls > 100)
            throw new IllegalStateException("Please use preallocation calls in range of [1..100]");
        this.preallocationCalls = numCalls;

        return this;
    }

    /**
     * This method allows you to specify command queue length, as primary argument for asynchronous execution controller
     *
     * Default value: 3
     *
     * @param length
     * @return
     */
    public Configuration setCommandQueueLength(int length) {
        if (length <= 0)
            throw new IllegalStateException("Command queue length can't be <= 0");
        this.commandQueueLength = length;

        return this;
    }

    /**
     * This option specifies minimal time gap between two subsequent System.gc() calls
     * Set to 0 to disable this option.
     *
     * @param windowMs
     * @return
     */
    public Configuration setNoGcWindowMs(long windowMs) {
        if (windowMs < 1)
            throw new IllegalStateException("No-GC window should have positive value");

        this.noGcWindowMs = windowMs;
        return this;
    }

    /**
     * This method allows you to specify maximum number of probable parallel cuda processes
     *
     * Default value: 4
     *
     * PLEASE NOTE: This parameter has effect only for ASYNCHRONOUS execution model
     *
     * @param length
     * @return
     */
    public Configuration setCommandLanesNumber(int length) {
        if (length < 1)
            throw new IllegalStateException("Command Lanes number can't be < 1");
        if (length > 8)
            length = 8;
        this.commandLanesNumber = length;

        return this;
    }

    public boolean isForcedSingleGPU() {
        return forceSingleGPU;
    }

    /**
     * This method allows you to enable or disable multi-GPU mode.
     *
     * PLEASE NOTE: This is NOT magic method, that will automatically scale your application performance.
     *
     * @param reallyAllow
     * @return
     */
    public Configuration allowMultiGPU(boolean reallyAllow) {
        forceSingleGPU = !reallyAllow;
        return this;
    }

    public Configuration setMemoryModel(@NonNull MemoryModel model) {
        memoryModel = model;
        return this;
    }
}
