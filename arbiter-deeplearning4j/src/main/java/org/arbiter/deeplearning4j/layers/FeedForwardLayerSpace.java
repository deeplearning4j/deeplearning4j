package org.arbiter.deeplearning4j.layers;

import org.arbiter.optimize.parameter.FixedValue;
import org.arbiter.optimize.parameter.ParameterSpace;
import org.deeplearning4j.nn.conf.layers.FeedForwardLayer;

public abstract class FeedForwardLayerSpace<L extends FeedForwardLayer> extends LayerSpace<L> {

    protected ParameterSpace<Integer> nIn;
    protected ParameterSpace<Integer> nOut;


    protected FeedForwardLayerSpace(Builder builder){
        super(builder);
        nIn = builder.nIn;
        nOut = builder.nOut;
    }

    protected void setLayerOptionsBuilder(FeedForwardLayer.Builder builder){
        super.setLayerOptionsBuilder(builder);
        if(nIn != null) builder.nIn(nIn.randomValue());
        if(nOut != null) builder.nOut(nOut.randomValue());
    }


    public abstract static class Builder<T> extends LayerSpace.Builder<T> {

        protected ParameterSpace<Integer> nIn;
        protected ParameterSpace<Integer> nOut;

        public T nIn(int nIn){
            return nIn(new FixedValue<Integer>(nIn));
        }

        public T nIn(ParameterSpace<Integer> nIn){
            this.nIn = nIn;
            return (T)this;
        }

        public T nOut(int nOut){
            return nOut(new FixedValue<Integer>(nOut));
        }

        public T nOut(ParameterSpace<Integer> nOut){
            this.nOut = nOut;
            return (T)this;
        }
    }

    @Override
    public String toString(){
        return toString(", ");
    }

    @Override
    protected String toString(String delim){
        StringBuilder sb = new StringBuilder();
        if(nIn != null) sb.append("nIn: ").append(nIn).append(delim);
        if(nOut != null) sb.append("nOut: ").append(nOut).append(delim);
        sb.append(super.toString(delim));
        return sb.toString();
    }

}
