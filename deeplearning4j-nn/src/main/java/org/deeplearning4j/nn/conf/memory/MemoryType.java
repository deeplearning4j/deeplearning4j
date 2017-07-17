package org.deeplearning4j.nn.conf.memory;

/**
 * Created by Alex on 13/07/2017.
 */
public enum MemoryType {
    PARAMETERS,
    PARAMATER_GRADIENTS,
    ACTIVATIONS,
    ACTIVATION_GRADIENTS,
    UPDATER_STATE,
    WORKING_MEMORY_FIXED,
    WORKING_MEMORY_VARIABLE,
    CACHED_MEMORY_FIXED,
    CACHED_MEMORY_VARIABLE
}
