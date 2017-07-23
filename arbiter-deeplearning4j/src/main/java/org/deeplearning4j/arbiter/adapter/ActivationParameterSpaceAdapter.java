package org.deeplearning4j.arbiter.adapter;

import lombok.Data;
import org.deeplearning4j.arbiter.optimize.api.ParameterSpace;
import org.deeplearning4j.arbiter.optimize.api.adapter.ParameterSpaceAdapter;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.util.List;
import java.util.Map;

/**
 * Created by Alex on 30/06/2017.
 */
@Data
public class ActivationParameterSpaceAdapter extends ParameterSpaceAdapter<Activation, IActivation> {

    private ParameterSpace<Activation> activation;

    public ActivationParameterSpaceAdapter(@JsonProperty("activation") ParameterSpace<Activation> activation) {
        this.activation = activation;
    }

    @Override
    public IActivation convertValue(Activation from) {
        return from.getActivationFunction();
    }

    @Override
    protected ParameterSpace<Activation> underlying() {
        return activation;
    }
}
