package org.arbiter.deeplearning4j.layers;

import org.deeplearning4j.nn.conf.layers.DenseLayer;

public class DenseLayerSpace extends FeedForwardLayerSpace<DenseLayer> {

    private DenseLayerSpace(Builder builder){
        super(builder);
    }

    @Override
    public DenseLayer randomLayer() {
        //Using the builder here, to get default options
        DenseLayer.Builder b = new DenseLayer.Builder();
        setLayerOptionsBuilder(b);
        return b.build();
    }

    protected void setLayerOptionsBuilder(DenseLayer.Builder builder){
        super.setLayerOptionsBuilder(builder);
    }

//    public static class Builder<T extends Builder<T>> extends FeedForwardLayerSpace.Builder<T>{
    public static class Builder extends FeedForwardLayerSpace.Builder<Builder>{

        @Override
        @SuppressWarnings("unchecked")
        public DenseLayerSpace build(){
            return new DenseLayerSpace(this);
        }
    }

}
