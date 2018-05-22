package org.nd4j.linalg.api.memory.enums;

/**
 * This enum describes where workspace memory is located
 *
 * @author raver119@gmail.com
 */
public enum LocationPolicy {
    /**
     * Allocations will be in RAM
     */
    RAM,

    /**
     * Allocations will be in memory-mapped file
     */
    MMAP
}
