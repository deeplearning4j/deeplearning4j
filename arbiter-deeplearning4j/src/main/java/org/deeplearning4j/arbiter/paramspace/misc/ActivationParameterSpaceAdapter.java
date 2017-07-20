package org.deeplearning4j.arbiter.paramspace.misc;

import lombok.Data;
import org.deeplearning4j.arbiter.optimize.api.ParameterSpace;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.util.List;

/**
 * Created by Alex on 30/06/2017.
 */
@Data
public class ActivationParameterSpaceAdapter implements ParameterSpace<IActivation> {

    private ParameterSpace<Activation> activation;

    public ActivationParameterSpaceAdapter(@JsonProperty("activation") ParameterSpace<Activation> activation) {
        this.activation = activation;
    }

    @Override
    public IActivation getValue(double[] parameterValues) {
        return activation.getValue(parameterValues).getActivationFunction();
    }

    @Override
    public int numParameters() {
        return activation.numParameters();
    }

    @Override
    public List<ParameterSpace> collectLeaves() {
        return activation.collectLeaves();
    }

    @Override
    public boolean isLeaf() {
        return activation.isLeaf();
    }

    @Override
    public void setIndices(int... indices) {
        activation.setIndices(indices);
    }

    @Override
    public String toString(){
        return activation.toString();
    }
}
