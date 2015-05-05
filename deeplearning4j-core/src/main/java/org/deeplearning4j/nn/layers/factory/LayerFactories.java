/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.deeplearning4j.nn.layers.factory;

import org.deeplearning4j.nn.layers.convolution.ConvolutionLayer;
import org.deeplearning4j.nn.layers.convolution.subsampling.SubsamplingLayer;
import org.deeplearning4j.nn.layers.recurrent.LSTM;
import org.deeplearning4j.nn.layers.feedforward.autoencoder.recursive.RecursiveAutoEncoder;
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
        else if(ConvolutionLayer.class.isAssignableFrom(clazz))
            return new ConvolutionLayerFactory(clazz);
        else if(SubsamplingLayer.class.isAssignableFrom(clazz))
            return new SubsampleLayerFactory(clazz);
        return new DefaultLayerFactory(clazz);
    }


    /**
     * Get the type for the layer factory
     * @param layerFactory the layer factory
     * @return the type
     */
    public static Layer.Type typeForFactory(LayerFactory layerFactory) {
        if(layerFactory instanceof ConvolutionLayerFactory || layerFactory instanceof SubsampleLayerFactory)
            return Layer.Type.CONVOLUTIONAL;
        else if(layerFactory instanceof LSTMLayerFactory)
            return Layer.Type.RECURRENT;
        else if(layerFactory instanceof RecursiveAutoEncoderLayerFactory)
            return Layer.Type.RECURSIVE;
        return Layer.Type.FEED_FORWARD;
    }

}
