package org.arbiter.deeplearning4j.layers;

import org.deeplearning4j.nn.conf.layers.BaseOutputLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;

public class OutputLayerSpace extends BaseOutputLayerSpace<OutputLayer> {

    private OutputLayerSpace(Builder builder){
        super(builder);
    }

    @Override
    public OutputLayer randomLayer() {
        OutputLayer.Builder o = new OutputLayer.Builder();
        setLayerOptionsBuilder(o);
        return o.build();
    }

    protected void setLayerOptionsBuilder(OutputLayer.Builder builder){
        super.setLayerOptionsBuilder(builder);
    }

    public static class Builder extends BaseOutputLayerSpace.Builder<Builder>{

        @Override
        @SuppressWarnings("unchecked")
        public OutputLayerSpace build(){
            return new OutputLayerSpace(this);
        }
    }

    @Override
    public String toString(){
        return toString(", ");
    }

    @Override
    public String toString(String delim){
        return "OutputLayerSpace(" + super.toString(delim) + ")";
    }
}
