package org.deeplearning4j.arbiter.layers;

import lombok.AccessLevel;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import org.deeplearning4j.arbiter.adapter.ActivationParameterSpaceAdapter;
import org.deeplearning4j.arbiter.adapter.LossFunctionParameterSpaceAdapter;
import org.deeplearning4j.arbiter.optimize.api.ParameterSpace;
import org.deeplearning4j.arbiter.optimize.parameter.FixedValue;
import org.deeplearning4j.arbiter.util.LeafUtils;
import org.deeplearning4j.nn.conf.layers.LossLayer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.lossfunctions.LossFunctions;

@Data
@EqualsAndHashCode(callSuper = true)
@NoArgsConstructor(access = AccessLevel.PROTECTED) //For Jackson JSON/YAML deserialization
public class LossLayerSpace extends LayerSpace<LossLayer> {

    private ParameterSpace<IActivation> activationFunction;
    protected ParameterSpace<ILossFunction> lossFunction;

    public LossLayerSpace(Builder builder){
        super(builder);
        this.activationFunction = builder.activationFunction;
        this.lossFunction = builder.lossFunction;

        this.numParameters = LeafUtils.countUniqueParameters(collectLeaves());
    }

    @Override
    public LossLayer getValue(double[] parameterValues) {
        LossLayer.Builder b = new LossLayer.Builder();
        if(activationFunction != null)
            b.activation(activationFunction.getValue(parameterValues));
        if(lossFunction != null)
            b.lossFunction(lossFunction.getValue(parameterValues));
        return b.build();
    }


    public static class Builder extends LayerSpace.Builder<Builder>{
        
        private ParameterSpace<IActivation> activationFunction;
        protected ParameterSpace<ILossFunction> lossFunction;

        public Builder lossFunction(LossFunctions.LossFunction lossFunction) {
            return lossFunction(new FixedValue<>(lossFunction));
        }

        public Builder lossFunction(ParameterSpace<LossFunctions.LossFunction> lossFunction) {
            return iLossFunction(new LossFunctionParameterSpaceAdapter(lossFunction));
        }

        public Builder iLossFunction(ILossFunction lossFunction) {
            return iLossFunction(new FixedValue<>(lossFunction));
        }

        public Builder iLossFunction(ParameterSpace<ILossFunction> lossFunction) {
            this.lossFunction = lossFunction;
            return this;
        }

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

        @Override
        public LossLayerSpace build() {
            return new LossLayerSpace(this);
        }
    }
}
