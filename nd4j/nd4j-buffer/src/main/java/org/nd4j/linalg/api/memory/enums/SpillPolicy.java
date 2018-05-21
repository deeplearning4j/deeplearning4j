package org.nd4j.linalg.api.memory.enums;

/**
 * @author raver119@gmail.com
 */
public enum SpillPolicy {
    /**
     * This policy means - use external allocation for spills.
     *
     * PLEASE NOTE: external allocations will be released after end of loop is reached.
     */
    EXTERNAL,

    /**
     * This policy means - use external allocation for spills + reallocate at the end of loop.
     */
    REALLOCATE,

    /**
     * This policy means - no spills will be ever possible, exception will be thrown.
     *
     * PLEASE NOTE: basically useful for debugging.
     */
    FAIL,
}
