package org.arbiter.deeplearning4j.layers;

import org.arbiter.optimize.parameter.FixedValue;
import org.arbiter.optimize.parameter.ParameterSpace;
import org.deeplearning4j.nn.conf.layers.BaseOutputLayer;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;


public abstract class BaseOutputLayerSpace<L extends BaseOutputLayer> extends FeedForwardLayerSpace<L>{

    protected ParameterSpace<LossFunction> lossFunction;

    protected BaseOutputLayerSpace(Builder builder){
        super(builder);
        this.lossFunction = builder.lossFunction;
    }

    protected void setLayerOptionsBuilder(BaseOutputLayer.Builder builder){
        super.setLayerOptionsBuilder(builder);
        if(lossFunction != null) builder.lossFunction(lossFunction.randomValue());
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
