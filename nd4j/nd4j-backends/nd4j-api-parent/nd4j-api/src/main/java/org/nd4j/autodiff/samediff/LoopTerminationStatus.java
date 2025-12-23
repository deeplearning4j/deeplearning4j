package org.nd4j.autodiff.samediff;

/**
 * Possible loop termination statuses
 */
public enum LoopTerminationStatus {
    ACTIVE,                  // Loop is currently executing
    TERMINATED_NORMAL,       // Loop terminated normally (condition became false)
    TERMINATED_EARLY,        // Loop terminated earlier than expected
    TERMINATED_ERROR,        // Loop terminated due to an error
    TERMINATED_TIMEOUT,      // Loop terminated due to timeout
    TERMINATED_MANUAL,       // Loop was manually terminated
    TERMINATED_RESOURCE,     // Loop terminated due to resource exhaustion
    PAUSED,                  // Loop execution is paused
    UNKNOWN                  // Status is unknown
}
