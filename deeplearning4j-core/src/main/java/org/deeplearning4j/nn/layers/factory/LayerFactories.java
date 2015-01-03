package org.deeplearning4j.nn.layers.factory;

import org.deeplearning4j.models.classifiers.lstm.LSTM;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.LayerFactory;
import org.deeplearning4j.nn.layers.BasePretrainNetwork;
import org.deeplearning4j.nn.layers.ConvolutionDownSampleLayer;

/**
 * Static method for finding which layer factory to use
 * @author Adam Gibson
 */
public class LayerFactories {

    public static LayerFactory getFactory(Class<? extends Layer> clazz) {
        if(clazz.equals(ConvolutionDownSampleLayer.class))
            return new ConvolutionLayerFactory(clazz);
        else if(clazz.equals(LSTM.class)) {
            return new LSTMLayerFactory(LSTM.class);
        }
        else if(BasePretrainNetwork.class.isAssignableFrom(clazz))
            return new PretrainLayerFactory(clazz);
        return new DefaultLayerFactory(clazz);
    }


}
