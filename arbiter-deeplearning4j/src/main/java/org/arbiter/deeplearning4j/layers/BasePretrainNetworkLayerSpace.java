package org.arbiter.deeplearning4j.layers;

import org.arbiter.optimize.parameter.FixedValue;
import org.arbiter.optimize.parameter.ParameterSpace;
import org.deeplearning4j.nn.conf.layers.BasePretrainNetwork;
import org.nd4j.linalg.api.ops.LossFunction;

public abstract class BasePretrainNetworkLayerSpace<L extends BasePretrainNetwork> extends FeedForwardLayerSpace<L> {

    protected ParameterSpace<LossFunction> lossFunction;
    protected BasePretrainNetworkLayerSpace(Builder builder){
        super(builder);
        this.lossFunction = builder.lossFunction;
    }


    public static abstract class Builder<T> extends FeedForwardLayerSpace.Builder<T>{
        protected ParameterSpace<LossFunction> lossFunction;

        public T lossFunction(LossFunction lossFunction){
            return lossFunction(new FixedValue<LossFunction>(lossFunction));
        }

        public T lossFunction(ParameterSpace<LossFunction> lossFunction){
            this.lossFunction = lossFunction;
            return (T)this;
        }

    }

}
