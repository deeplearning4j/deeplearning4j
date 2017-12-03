package org.nd4j.autodiff.opstate;

import lombok.Builder;
import lombok.Data;

import java.io.Serializable;
import java.util.Arrays;

/**
 * An action in the graph
 *
 * @author Adam Gibson
 */
@Data
@Builder
public class OpExecAction implements Serializable {
    private OpExecAction forwardAction;
    private OpExecAction backwardAction;
    private int[] inputsIds;
    private int[] outputId;


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



    @Override
    public String toString() {
        return "OpExecAction{" +
                ", inputsIds=" + Arrays.toString(inputsIds) +
                ", outputId=" + Arrays.toString(outputId) +
                '}';
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        OpExecAction action = (OpExecAction) o;

        if (outputId != action.outputId) return false;
        // Probably incorrect - comparing Object[] arrays with Arrays.equals
        return Arrays.equals(inputsIds, action.inputsIds);
    }

    @Override
    public int hashCode() {
        int result = super.hashCode();
        result = 31 * result + Arrays.hashCode(inputsIds);
        result = 31 * result + Arrays.hashCode(outputId);
        return result;
    }
}
