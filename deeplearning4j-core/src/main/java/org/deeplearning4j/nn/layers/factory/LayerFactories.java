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


import org.deeplearning4j.nn.api.LayerFactory;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.*;

/**
 * Static method for finding which layer factory to use
 * @author Adam Gibson
 */
public class LayerFactories {
    /**
     * Get the factory based on the passed in class
     * @param conf the clazz to get the layer factory for
     * @return the layer factory for the particular layer
     */
    public static LayerFactory getFactory(NeuralNetConfiguration conf) {
        return getFactory(conf.getLayer());
    }

    /**
     * Get the factory based on the passed in class
     * @param layer the clazz to get the layer factory for
     * @return the layer factory for the particular layer
     */
    public static LayerFactory getFactory(Layer layer) {
        Class<? extends Layer> clazz = layer.getClass();
        if(clazz.equals(GravesLSTM.class))
        	return new GravesLSTMLayerFactory(GravesLSTM.class);
        else if (clazz.equals(GravesBidirectionalLSTM.class))
            return new GravesBidirectionalLSTMLayerFactory(GravesBidirectionalLSTM.class);
        else if(clazz.equals(GRU.class))
        	return new GRULayerFactory(GRU.class);
        else if(BasePretrainNetwork.class.isAssignableFrom(clazz))
            return new PretrainLayerFactory(clazz);
        else if(ConvolutionLayer.class.isAssignableFrom(clazz))
            return new ConvolutionLayerFactory(clazz);
        else if(SubsamplingLayer.class.isAssignableFrom(clazz))
            return new SubsampleLayerFactory(clazz);
        else if(BatchNormalization.class.isAssignableFrom(clazz))
            return new BatchNormalizationLayerFactory(clazz);
        else if(LocalResponseNormalization.class.isAssignableFrom(clazz))
            return new EmptyFactory(clazz);
        else if(ActivationLayer.class.isAssignableFrom(clazz))
            return new EmptyFactory(clazz);
        return new DefaultLayerFactory(clazz);
    }


}
