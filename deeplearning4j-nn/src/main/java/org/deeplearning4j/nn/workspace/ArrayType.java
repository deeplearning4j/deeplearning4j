package org.deeplearning4j.nn.workspace;

/**
 * Array type enumeration for use with {@link LayerWorkspaceMgr}<br>
 * <br>
 * Array types:<br>
 * INPUT: The array set to the input field of a layer (i.e., input activations)<br>
 * ACTIVATIONS: The output activations for a layer's feed-forward method<br>
 * ACTIVATION_GRAD
 * FF_WORKING_MEM
 * BP_WORKING_MEM
 * RNN_FF_LOOP_WORKING_MEM
 * RNN_BP_LOOP_WORKING_MEM
 * UPDATER_WORKING_MEM
 *
 * @author Alex Black
 */
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
