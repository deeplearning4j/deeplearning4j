package org.nd4j.jita.conf;

import lombok.Data;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

/**
 * @author raver119@gmail.com
 */
@Data
public class Configuration implements Serializable {
    public enum Aggressiveness {
        PEACEFUL,
        REASONABLE,
        URGENT,
        IMMEDIATE
    }

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
     * Maximum allocated ram, in bytes
     */
    private long maximumAllocation = 256 * 1024 * 1024;


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
