package org.deeplearning4j.nn.layers.local;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.layers.BaseLayer;

public class LocallyConnected2DLayer extends BaseLayer<org.deeplearning4j.nn.conf.layers.LocallyConnected2D> {

    public LocallyConnected2DLayer(NeuralNetConfiguration conf) {
        super(conf);
    }

    @Override
    public boolean isPretrainLayer() {
        return false;
    }
}
