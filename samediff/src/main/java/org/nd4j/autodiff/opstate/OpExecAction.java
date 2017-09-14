package org.nd4j.autodiff.opstate;

import lombok.Builder;
import lombok.Data;

import java.io.Serializable;

/**
 * An action in the graph
 *
 * @author Adam Gibson
 */
@Data
@Builder
public class OpExecAction implements Serializable {
    private OpState opState;
    private OpExecAction forwardAction;
    private OpExecAction backwardAction;
    private NDArrayInformation[] inputs;
    private NDArrayInformation output;
    private int[] inputsIds;
    private int outputId;


    /**
     * Links the forward and backward ops
     * to each other.
     * Note here that you have to invoke this
     * on the *forward* operation
     * passing in the *backward* operation
     *
     * @param backwardAction
     */
    public void setupForwardBackward(OpExecAction backwardAction) {
        setBackwardAction(backwardAction);
        backwardAction.setForwardAction(this);
    }

    public boolean isInPlace() {
        return opState.isInPlace();
    }

}
