package org.deeplearning4j.nn.graph.multioutput.testlayers;

import lombok.NoArgsConstructor;
import org.deeplearning4j.nn.conf.layers.DenseLayer;

public class SplitDenseLayerConf extends DenseLayer {

    public SplitDenseLayerConf(Builder builder){
        super(builder);
    }

    @NoArgsConstructor
    public static class Builder extends DenseLayer.Builder {

        @Override
        @SuppressWarnings("unchecked")
        public SplitDenseLayerConf build(){
            return new SplitDenseLayerConf(this);
        }
    }
}
