package org.nd4j.jita.conf;

import lombok.Getter;
import lombok.NonNull;
import org.nd4j.jita.allocator.enums.Aggressiveness;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

/**
 * @author raver119@gmail.com
 */
public class Configuration implements Serializable {
    private static Logger logger = LoggerFactory.getLogger(Configuration.class);

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
    @Getter private int numberOfHostMemoryBuckets = 8;

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
    @Getter private long maximumDeviceAllocation = 1 * 1024 * 1024 * 1024L;


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

    @Getter private boolean debug = false;

    @Getter private int maximumGridSize = 128;

    @Getter private int maximumBlockSize = -1;

    @Getter private long maximumHostCache = Long.MAX_VALUE;

    @Getter private long maximumDeviceCache = Long.MAX_VALUE;

    private NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();


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
        if (gridDim <= 0 || gridDim >= 1024)
            throw new IllegalStateException("Please keep gridDim in range [64...512]");

        this.maximumGridSize = gridDim;

        nativeOps.setOmpNumThreads(gridDim);

        return this;
    }

    /**
     * This methos allows to specify max blockSize for kernel launches
     *
     * Default value: -1 (that means pick value automatically)
     *
     * @param blockDim
     * @return
     */
    public Configuration setMaximumBlockSize(int blockDim) {
        if (blockDim <= 0 || blockDim > 1024)
            throw new IllegalStateException("Please keep blockDim in range [64...512]");

        this.maximumBlockSize = blockDim;

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
}
