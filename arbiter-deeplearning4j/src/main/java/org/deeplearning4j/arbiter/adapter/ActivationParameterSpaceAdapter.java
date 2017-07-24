package org.deeplearning4j.arbiter.adapter;

import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import org.deeplearning4j.arbiter.optimize.api.ParameterSpace;
import org.deeplearning4j.arbiter.optimize.api.adapter.ParameterSpaceAdapter;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.shade.jackson.annotation.JsonProperty;

/**
 * A simple class to adapt a {@link Activation} parameter space to a {@link IActivation} parameter space
 *
 * @author Alex Black
 */
@Data
@NoArgsConstructor
@EqualsAndHashCode(callSuper = false)
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
