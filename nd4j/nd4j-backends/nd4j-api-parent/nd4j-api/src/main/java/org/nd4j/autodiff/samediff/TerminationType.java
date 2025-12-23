package org.nd4j.autodiff.samediff;

public enum TerminationType {
    CONDITION_FALSE,        // Normal termination via loop condition
    CONDITION_TRUE_EXIT,    // Exit operation triggered by true condition
    SWITCH_TERMINATION,     // Switch operation caused termination
    ERROR_TERMINATION,      // Error/exception during loop
    TIMEOUT_TERMINATION,    // Maximum iterations exceeded
    EARLY_BREAK,           // Early termination before expected completion
    RESOURCE_EXHAUSTION,   // Memory or other resource limits
    MANUAL_TERMINATION     // Explicitly terminated by user/system
}
