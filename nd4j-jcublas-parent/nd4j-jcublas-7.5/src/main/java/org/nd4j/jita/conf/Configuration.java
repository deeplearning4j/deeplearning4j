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
    private long minimumTtl = 5 * 1000L;

    /**
     * Deallocation aggressiveness
     */
    private Aggressiveness deallocAggressiveness = Aggressiveness.REASONABLE;

    /**
     * Allocation aggressiveness
     */
    private Aggressiveness allocAggressiveness = Aggressiveness.REASONABLE;

    /**
     * Maximum allocated ram, in bytes
     */
    private long maximumAllocation = 256 * 1024 * 1024L;


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
