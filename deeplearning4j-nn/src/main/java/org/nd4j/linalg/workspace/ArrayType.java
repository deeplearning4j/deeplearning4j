package org.nd4j.linalg.workspace;

public enum ArrayType {
    INPUT,
    ACTIVATIONS,
    ACTIVATION_GRAD,
    FF_WORKING_MEM,
    BP_WORKING_MEM,
    RNN_FF_LOOP_WORKING_MEM,
    RNN_BP_LOOP_WORKING_MEM,
    UPDATER_WORKING_MEM
}
