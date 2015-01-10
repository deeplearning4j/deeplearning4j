package org.deeplearning4j.nn.layers.factory;

import org.deeplearning4j.models.classifiers.lstm.LSTM;
import org.deeplearning4j.models.featuredetectors.autoencoder.recursive.RecursiveAutoEncoder;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.LayerFactory;
import org.deeplearning4j.nn.layers.BasePretrainNetwork;
import org.deeplearning4j.nn.layers.convolution.ConvolutionDownSampleLayer;

/**
 * Static method for finding which layer factory to use
 * @author Adam Gibson
 */
public class LayerFactories {
    /**
     * Get the factory based on the passed in class
     * @param clazz the clazz to get the layer factory for
     * @return the layer factory for the particular layer
     */
    public static LayerFactory getFactory(Class<? extends Layer> clazz) {
        if(clazz.equals(ConvolutionDownSampleLayer.class))
            return new ConvolutionLayerFactory(clazz);
        else if(clazz.equals(LSTM.class))
            return new LSTMLayerFactory(LSTM.class);
        else if(RecursiveAutoEncoder.class.isAssignableFrom(clazz))
            return new RecursiveAutoEncoderLayerFactory(RecursiveAutoEncoder.class);
        else if(BasePretrainNetwork.class.isAssignableFrom(clazz))
            return new PretrainLayerFactory(clazz);
        return new DefaultLayerFactory(clazz);
    }


}
