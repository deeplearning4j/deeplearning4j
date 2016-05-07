package org.nd4j.jita.conf;

import lombok.Getter;
import lombok.NonNull;
import org.nd4j.jita.allocator.enums.Aggressiveness;
import org.nd4j.jita.allocator.enums.AllocationStatus;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * @author raver119@gmail.com
 */
public class Configuration implements Serializable {
    private static Logger logger = LoggerFactory.getLogger(Configuration.class);

    public enum ExecutionModel {
        SEQUENTIAL,
        ASYNCHRONOUS,
        OPTIMIZED,
    }

    public enum AllocationModel {
        DIRECT,
        CACHE_HOST,
        CACHE_ALL,
    }

    @Getter private ExecutionModel executionModel = ExecutionModel.ASYNCHRONOUS;

    @Getter private AllocationModel allocationModel = AllocationModel.CACHE_ALL;

    @Getter private AllocationStatus firstMemory = AllocationStatus.DEVICE;

    @Getter private boolean debug = false;

    @Getter private boolean verbose = false;

    /**
     * Keep this value between 0.01 and 0.95 please
     */
    @Getter private double maximumDeviceMemoryUsed = 0.85;

    /**
     * Minimal number of activations for relocation threshold
     */
    @Getter private int minimumRelocationThreshold = 5;

    /**
     * Minimal guaranteed TTL for memory chunk
     */
    @Getter private long minimumTTLMilliseconds = 10 * 1000L;

    /**
     * Number of buckets/garbage collectors for host memory
     */
    @Getter private int numberOfGcThreads= 4;

    /**
     * Deallocation aggressiveness
     */
    @Getter private Aggressiveness hostDeallocAggressiveness = Aggressiveness.REASONABLE;

    @Getter private Aggressiveness gpuDeallocAggressiveness = Aggressiveness.REASONABLE;

    /**
     * Allocation aggressiveness
     */
    @Getter private Aggressiveness gpuAllocAggressiveness = Aggressiveness.REASONABLE;


    /**
     * Maximum allocated per-device memory, in bytes
     */
    @Getter private long maximumDeviceAllocation = 4 * 1024 * 1024 * 1024L;


    /**
     * Maximum allocatable zero-copy/pinned/pageable memory
     */
    @Getter private long maximumZeroAllocation = Runtime.getRuntime().maxMemory() + (500 * 1024 * 1024L);

    /**
     * True if allowed, false if relocation required
     */
    @Getter private boolean crossDeviceAccessAllowed = false;

    /**
     * True, if allowed, false otherwise
     */
    @Getter private boolean zeroCopyFallbackAllowed = false;

    /**
     * Maximum length of single memory chunk
     */
    @Getter private long maximumSingleHostAllocation = Long.MAX_VALUE;

    @Getter private long maximumSingleDeviceAllocation = Long.MAX_VALUE;

    @Getter private List<Integer> availableDevices = new ArrayList<>();

    @Getter private List<Integer> bannedDevices = new ArrayList<>();

    @Getter private int maximumGridSize = 128;

    @Getter private int maximumBlockSize = 512;

    @Getter private long maximumHostCache = Long.MAX_VALUE;

    @Getter private long maximumDeviceCache = Long.MAX_VALUE;

    @Getter private boolean usePreallocation = true;

    @Getter private int preallocationCalls = 10;

    @Getter private long maximumHostCacheableLength = 100663296;

    @Getter private long maximumDeviceCacheableLength = 100663296;

    @Getter private int commandQueueLength = 3;

    private final AtomicBoolean initialized = new AtomicBoolean(false);

    private NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();

    public boolean isInitialized() {
        return initialized.get();
    }

    public void setInitialized() {
        this.initialized.compareAndSet(false, true);
    }

    public Configuration setMinimumRelocationThreshold(int threshold) {
        this.maximumDeviceAllocation = Math.max(2, threshold);

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
        } else this.maximumDeviceMemoryUsed = percentage;

        return this;
    }

    public Configuration() {
        int cnt = (int) nativeOps.getAvailableDevices();
        if (cnt == 0)
            throw new RuntimeException("No CUDA devices were found in system");

        for (int i = 0; i < cnt; i++) {
            availableDevices.add(i);
        }

        nativeOps.setOmpNumThreads(maximumBlockSize);
        nativeOps.setGridLimit(maximumGridSize);
    }

    /**
     * This method allows you to ban specific device.
     *
     * @param deviceId
     * @return
     */
    public Configuration banDevice(@NonNull Integer deviceId) {
        if (!bannedDevices.contains(deviceId)) {
            bannedDevices.add(deviceId);
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
            logger.warn("Setting maximum memory below -Xmx value can cause problems");

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
    public Configuration setMaximumSingleHostAllocation(long max){
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
    public Configuration setMaximumSingleDeviceAllocation(long max){
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
        if (gridDim <= 0 || gridDim > 512)
            throw new IllegalStateException("Please keep gridDim in range [64...512]");

        this.maximumGridSize = gridDim;

        nativeOps.setGridLimit(maximumGridSize);

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
        if (blockDim < 64 || blockDim > 512)
            throw new IllegalStateException("Please keep blockDim in range [64...512]");


        this.maximumBlockSize = blockDim;

        nativeOps.setOmpNumThreads(blockDim);

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

        nativeOps.enableDebugMode(debug);

        return this;
    }

    public Configuration setVerbose(boolean verbose) {
        this.verbose = verbose;

        nativeOps.enableVerboseMode(verbose);

        return this;
    }

    /**
     * NOT IMPLEMENTED YET
     *
     * @param reallyAllow
     * @return
     */
    @Deprecated
    public Configuration allowCrossDeviceAccess(boolean reallyAllow) {
        // TODO:  this thing should be implemented for specific algebra-related tasks
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
     * Default value: ASYNCHRONOUS
     *
     * @param executionModel
     * @return
     */
    public Configuration setExecutionModel(@NonNull ExecutionModel executionModel) {
        this.executionModel = executionModel;

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
     *
     * PLEASE NOTE: For device memory all data still retains on host side as well.
     *
     * Default value: DEVICE
     * @param initialMemory
     * @return
     */
    public Configuration setFirstMemory(@NonNull AllocationStatus initialMemory) {
        if (initialMemory != AllocationStatus.DEVICE && initialMemory != AllocationStatus.HOST)
            throw new IllegalStateException("First memory should be either [HOST] or [DEVICE]");

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
        if (numThreads <= 0 || numThreads >20)
            throw new IllegalStateException("Please, use something in range of [1..20] as number of GC threads");

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
}
