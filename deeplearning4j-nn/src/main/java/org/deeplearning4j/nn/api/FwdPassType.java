package org.deeplearning4j.nn.api;

/**
 * Type of forward pass to do. Used internally in MultiLayerNetwork and ComputationGraph
 *
 * @author Alex Black
 */
public enum FwdPassType {
    STANDARD,
    RNN_TIMESTEP,
    RNN_ACTIVATE_WITH_STORED_STATE
}
