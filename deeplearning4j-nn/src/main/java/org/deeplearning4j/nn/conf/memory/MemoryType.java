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
    CACHED_MEMORY_VARIABLE;

    public boolean isInference(){
        switch (this){
            case PARAMETERS:
            case ACTIVATIONS:
            case WORKING_MEMORY_FIXED:
            case WORKING_MEMORY_VARIABLE:
                return true;
            case PARAMATER_GRADIENTS:
            case ACTIVATION_GRADIENTS:
            case UPDATER_STATE:
            case CACHED_MEMORY_FIXED:
            case CACHED_MEMORY_VARIABLE:
                return false;
        }
        throw new RuntimeException("Unknown memory type: " + this );
    }
}
