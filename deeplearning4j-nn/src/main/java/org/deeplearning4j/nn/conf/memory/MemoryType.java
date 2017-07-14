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
    INFERENCE_WORKING_MEM,
    TRAINING_WORKING_MEM

}
