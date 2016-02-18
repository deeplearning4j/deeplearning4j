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

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.LayerFactory;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.layers.feedforward.embedding.EmbeddingLayer;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.optimize.api.IterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.*;

/**
 * Default layer factory: create a bias and a weight matrix
 * @author Adam Gibson
 */
public class DefaultLayerFactory implements LayerFactory {
   
    protected org.deeplearning4j.nn.conf.layers.Layer layerConfig;

    public DefaultLayerFactory(Class<? extends org.deeplearning4j.nn.conf.layers.Layer> layerConfig) {
        try {
            this.layerConfig = layerConfig.newInstance();
        } catch (Exception e) {
           throw new RuntimeException(e);
        }
    }

    @Override
    public <E extends Layer> E create(NeuralNetConfiguration conf, int index, int numLayers, Collection<IterationListener> iterationListeners) {
        return create(conf, iterationListeners, index);
    }

    @Override
    public <E extends Layer> E create(NeuralNetConfiguration conf) {
        return create(conf,new ArrayList<IterationListener>(),0);
    }

    @Override
    public <E extends Layer> E create(NeuralNetConfiguration conf, Collection<IterationListener> iterationListeners, int index) {
        Layer ret = getInstance(conf);
        ret.setListeners(iterationListeners);
        ret.setIndex(index);
        ret.setParamTable(getParams(conf));
        ret.setConf(conf);
        return (E) ret;
    }
    
    protected Layer getInstance(NeuralNetConfiguration conf) {
        if(layerConfig instanceof DenseLayer)
            return new org.deeplearning4j.nn.layers.feedforward.dense.DenseLayer(conf);
        if(layerConfig instanceof org.deeplearning4j.nn.conf.layers.AutoEncoder)
            return new org.deeplearning4j.nn.layers.feedforward.autoencoder.AutoEncoder(conf);
        if(layerConfig instanceof org.deeplearning4j.nn.conf.layers.RBM)
            return new org.deeplearning4j.nn.layers.feedforward.rbm.RBM(conf);
        if(layerConfig instanceof org.deeplearning4j.nn.conf.layers.ImageLSTM)
            return new org.deeplearning4j.nn.layers.recurrent.ImageLSTM(conf);
        if(layerConfig instanceof org.deeplearning4j.nn.conf.layers.GravesLSTM)
        	return new org.deeplearning4j.nn.layers.recurrent.GravesLSTM(conf);
        if (layerConfig instanceof org.deeplearning4j.nn.conf.layers.GravesBidirectionalLSTM )
            return new org.deeplearning4j.nn.layers.recurrent.GravesBidirectionalLSTM(conf);
        if(layerConfig instanceof org.deeplearning4j.nn.conf.layers.GRU )
        	return new org.deeplearning4j.nn.layers.recurrent.GRU(conf);
        if(layerConfig instanceof org.deeplearning4j.nn.conf.layers.OutputLayer)
            return new org.deeplearning4j.nn.layers.OutputLayer(conf);
        if(layerConfig instanceof org.deeplearning4j.nn.conf.layers.RnnOutputLayer)
        	return new org.deeplearning4j.nn.layers.recurrent.RnnOutputLayer(conf);
        if(layerConfig instanceof org.deeplearning4j.nn.conf.layers.RecursiveAutoEncoder)
            return new org.deeplearning4j.nn.layers.feedforward.autoencoder.recursive.RecursiveAutoEncoder(conf);   
        if(layerConfig instanceof org.deeplearning4j.nn.conf.layers.ConvolutionLayer)
            return new org.deeplearning4j.nn.layers.convolution.ConvolutionLayer(conf);
        if(layerConfig instanceof org.deeplearning4j.nn.conf.layers.SubsamplingLayer)
            return new org.deeplearning4j.nn.layers.convolution.subsampling.SubsamplingLayer(conf);
         if(layerConfig instanceof org.deeplearning4j.nn.conf.layers.BatchNormalization)
             return new org.deeplearning4j.nn.layers.normalization.BatchNormalization(conf);
        if(layerConfig instanceof org.deeplearning4j.nn.conf.layers.LocalResponseNormalization)
            return new org.deeplearning4j.nn.layers.normalization.LocalResponseNormalization(conf);
        if(layerConfig instanceof org.deeplearning4j.nn.conf.layers.EmbeddingLayer)
            return new EmbeddingLayer(conf);
        if(layerConfig instanceof  org.deeplearning4j.nn.conf.layers.ActivationLayer)
            return new org.deeplearning4j.nn.layers.ActivationLayer(conf);
        throw new RuntimeException("unknown layer type: " + layerConfig);
    }


    protected Map<String,INDArray> getParams(NeuralNetConfiguration conf) {
        ParamInitializer init = initializer();
        Map<String,INDArray> params = Collections.synchronizedMap(new LinkedHashMap<String,INDArray>());
        init.init(params,conf);
        return params;
    }

    @Override
    public ParamInitializer initializer() {
        return new DefaultParamInitializer();
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof DefaultLayerFactory)) return false;

        DefaultLayerFactory that = (DefaultLayerFactory) o;

        return !(layerConfig != null ? !layerConfig.equals(that.layerConfig) : that.layerConfig != null);
    }

    @Override
    public int hashCode() {
        return layerConfig != null ? layerConfig.hashCode() : 0;
    }
}
