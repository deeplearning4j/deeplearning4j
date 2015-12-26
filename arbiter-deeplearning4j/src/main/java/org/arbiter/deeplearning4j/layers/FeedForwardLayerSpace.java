package org.arbiter.deeplearning4j.layers;

import org.arbiter.optimize.parameter.FixedValue;
import org.arbiter.optimize.parameter.ParameterSpace;
import org.deeplearning4j.nn.conf.layers.FeedForwardLayer;

public abstract class FeedForwardLayerSpace<L extends FeedForwardLayer> extends LayerSpace<L> {

    protected ParameterSpace<Integer> nOut;


    protected FeedForwardLayerSpace(Builder builder){
        super(builder);
        nOut = builder.nOut;
    }

    protected void setLayerOptionsBuilder(FeedForwardLayer.Builder builder){
        super.setLayerOptionsBuilder(builder);
        if(nOut != null) builder.nOut(nOut.randomValue());
    }


    public abstract static class Builder<T> extends LayerSpace.Builder<T> {

        protected ParameterSpace<Integer> nOut;

        public T nOut(int nOut){
            return nOut(new FixedValue<Integer>(nOut));
        }

        public T nOut(ParameterSpace<Integer> nOut){
            this.nOut = nOut;
            return (T)this;
        }

    }

}
