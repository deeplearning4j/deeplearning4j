package org.nd4j.autodiff.opstate;

import lombok.Builder;
import lombok.Data;
import org.nd4j.autodiff.samediff.impl.SDVariable;

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
    private OpState opState;
    private OpExecAction forwardAction;
    private OpExecAction backwardAction;
    private SDVariable[] inputs;
    private SDVariable output;
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

    public boolean isInPlace() {
        return opState.isInPlace();
    }

    @Override
    public String toString() {
        return "OpExecAction{" +
                "opState=" + opState +
                ", inputs=" + Arrays.toString(inputs) +
                ", output=" + output +
                ", inputsIds=" + Arrays.toString(inputsIds) +
                ", outputId=" + outputId +
                '}';
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        OpExecAction action = (OpExecAction) o;

        if (outputId != action.outputId) return false;
        if (opState != null ? !opState.equals(action.opState) : action.opState != null) return false;
        // Probably incorrect - comparing Object[] arrays with Arrays.equals
        if (!Arrays.equals(inputs, action.inputs)) return false;
        if (output != null ? !output.equals(action.output) : action.output != null) return false;
        return Arrays.equals(inputsIds, action.inputsIds);
    }

    @Override
    public int hashCode() {
        int result = super.hashCode();
        result = 31 * result + (opState != null ? opState.hashCode() : 0);
        result = 31 * result + Arrays.hashCode(inputs);
        result = 31 * result + (output != null ? output.hashCode() : 0);
        result = 31 * result + Arrays.hashCode(inputsIds);
        result = 31 * result + Arrays.hashCode(outputId);
        return result;
    }
}
