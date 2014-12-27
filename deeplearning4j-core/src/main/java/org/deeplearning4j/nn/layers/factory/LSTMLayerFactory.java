package org.deeplearning4j.nn.layers.factory;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.params.LSTMParamInitializer;

/**
 *  LSTM layer initializer
 *  @author Adam Gibson
 */
public class LSTMLayerFactory extends DefaultLayerFactory {

    public LSTMLayerFactory(Class<? extends Layer> layerClazz) {
        super(layerClazz);
    }

    @Override
    public ParamInitializer getInitializer() {
        return new LSTMParamInitializer();
    }
}
