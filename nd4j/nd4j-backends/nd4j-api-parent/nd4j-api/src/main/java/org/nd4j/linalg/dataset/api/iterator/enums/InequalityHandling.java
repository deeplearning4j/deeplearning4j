package org.nd4j.linalg.dataset.api.iterator.enums;

/**
 * This enum describes different handling options for situations once one of producer runs out of data
 *
 * @author raver119@gmail.com
 */
public enum InequalityHandling {
    /**
     * Parallel iterator will stop everything once one of producers runs out of data
     */
    STOP_EVERYONE,

    /**
     * Parallel iterator will keep returning true on hasNext(), but next() will return null instead of DataSet
     */
    PASS_NULL,

    /**
     * Parallel iterator will silently reset underlying producer
     */
    RESET,

    /**
     * Parallel iterator will ignore this producer, and will use other producers.
     *
     * PLEASE NOTE: This option will invoke cross-device relocation in multi-device systems.
     */
    RELOCATE,
}
