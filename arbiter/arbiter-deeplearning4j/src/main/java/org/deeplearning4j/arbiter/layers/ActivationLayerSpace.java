package org.deeplearning4j.arbiter.layers;

import lombok.AccessLevel;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import org.deeplearning4j.arbiter.adapter.ActivationParameterSpaceAdapter;
import org.deeplearning4j.arbiter.optimize.api.ParameterSpace;
import org.deeplearning4j.arbiter.optimize.parameter.FixedValue;
import org.deeplearning4j.arbiter.util.LeafUtils;
import org.deeplearning4j.nn.conf.layers.ActivationLayer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.IActivation;

/**
 * Layer space for {@link ActivationLayer}
 *
 * @author Alex Black
 */
@Data
@EqualsAndHashCode(callSuper = true)
@NoArgsConstructor(access = AccessLevel.PRIVATE) //For Jackson JSON/YAML deserialization
public class ActivationLayerSpace extends LayerSpace<ActivationLayer> {

    private ParameterSpace<IActivation> activationFunction;

    protected ActivationLayerSpace(Builder builder) {
        super(builder);
        this.activationFunction = builder.activationFunction;
        this.numParameters = LeafUtils.countUniqueParameters(collectLeaves());
    }


    @Override
    public ActivationLayer getValue(double[] parameterValues) {
        ActivationLayer.Builder b = new ActivationLayer.Builder();
        super.setLayerOptionsBuilder(b, parameterValues);
        b.activation(activationFunction.getValue(parameterValues));
        return b.build();
    }

    public static class Builder extends LayerSpace.Builder<Builder> {

        private ParameterSpace<IActivation> activationFunction;

        public Builder activation(Activation activation) {
            return activation(new FixedValue<>(activation));
        }

        public Builder activation(IActivation iActivation) {
            return activationFn(new FixedValue<>(iActivation));
        }

        public Builder activation(ParameterSpace<Activation> activationFunction) {
            return activationFn(new ActivationParameterSpaceAdapter(activationFunction));
        }

        public Builder activationFn(ParameterSpace<IActivation> activationFunction) {
            this.activationFunction = activationFunction;
            return this;
        }

        @SuppressWarnings("unchecked")
        public ActivationLayerSpace build() {
            return new ActivationLayerSpace(this);
        }
    }

    @Override
    public String toString() {
        return toString(", ");
    }

    @Override
    public String toString(String delim) {
        return "ActivationLayerSpace(" + super.toString(delim) + ")";
    }
}
