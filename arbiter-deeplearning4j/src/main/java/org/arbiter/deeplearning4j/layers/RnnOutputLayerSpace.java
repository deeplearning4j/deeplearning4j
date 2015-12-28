package org.arbiter.deeplearning4j.layers;

import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;

public class RnnOutputLayerSpace extends BaseOutputLayerSpace<RnnOutputLayer> {

    private RnnOutputLayerSpace(Builder builder){
        super(builder);
    }

    @Override
    public RnnOutputLayer randomLayer() {
        RnnOutputLayer.Builder b = new RnnOutputLayer.Builder();
        setLayerOptionsBuilder(b);
        return b.build();
    }

    protected void setLayerOptionsBuilder(RnnOutputLayer.Builder builder){
        super.setLayerOptionsBuilder(builder);
    }

    @Override
    public String toString(){
        return toString(", ");
    }

    @Override
    public String toString(String delim){
        return "RnnOutputLayerSpace(" + super.toString(delim) + ")";
    }

    public static class Builder extends BaseOutputLayerSpace.Builder<RnnOutputLayer>{

        @Override
        @SuppressWarnings("unchecked")
        public RnnOutputLayerSpace build(){
            return new RnnOutputLayerSpace(this);
        }
    }


}
