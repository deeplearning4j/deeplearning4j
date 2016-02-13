package org.nd4j.jita.conf;

import lombok.Data;
import org.nd4j.jita.allocator.enums.Aggressiveness;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

/**
 * @author raver119@gmail.com
 */
@Data
public class Configuration implements Serializable {

    /**
     * Minimal number of activations for relocation threshold
     */
    private int minimumRelocationThreshold = 10;

    /**
     * How long allocated chunk have right to exist
     */
    private long decay = 10 * 1000L;

    /**
     * Minimal guaranteed TTL for memory chunk
     */
    private long minimumTTLMilliseconds = 5 * 1000L;

    /**
     * Deallocation aggressiveness
     */
    private Aggressiveness hostDeallocAggressiveness = Aggressiveness.REASONABLE;

    private Aggressiveness gpuDeallocAggressiveness = Aggressiveness.REASONABLE;

    /**
     * Allocation aggressiveness
     */
    private Aggressiveness gpuAllocAggressiveness = Aggressiveness.REASONABLE;

    private Aggressiveness hostAllocAggressiveness = Aggressiveness.REASONABLE;

    /**
     * Maximum allocated per-device memory, in bytes
     */
    private long maximumDeviceAllocation = 256 * 1024 * 1024L;


    /**
     * Maximum allocatable zero-copy/pinned/pageable memory
     */
    private long maximumZeroAllocation = Runtime.getRuntime().maxMemory();

    /**
     * True if allowed, false if relocation required
     */
    private boolean crossDeviceAccessAllowed = false;

    /**
     * True, if allowed, false otherwise
     */
    private boolean zeroCopyFallbackAllowed = false;

    /**
     * List of forbidden devices that can't be used for allocation and calculation
     */
    private List<Integer> forbiddenDevices = new ArrayList<>();

}
