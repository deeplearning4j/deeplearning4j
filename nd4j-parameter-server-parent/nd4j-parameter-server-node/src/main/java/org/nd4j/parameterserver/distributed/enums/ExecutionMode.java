package org.nd4j.parameterserver.distributed.enums;

/**
 * @author raver119@gmail.com
 */
public enum ExecutionMode {
    /**
     * This option assumes data (parameters) are split over multiple hosts
     */
    SHARDED,

    /**
     * This option assumes data stored on multiple shards at the same time
     */
    AVERAGING,

    /**
     * This option means data storage is controlled by application, and out of VoidParameterServer control
     */
    MANAGED,
}
