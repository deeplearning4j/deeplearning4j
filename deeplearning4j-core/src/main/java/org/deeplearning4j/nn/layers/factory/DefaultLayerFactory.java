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
import org.deeplearning4j.nn.conf.distribution.Distributions;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.optimize.api.IterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.distribution.Distribution;

import java.lang.reflect.Constructor;
import java.util.Collection;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Default layer factory: create a bias and a weight matrix
 * @author Adam Gibson
 */
public class DefaultLayerFactory implements LayerFactory {
    protected Class<? extends Layer> layerClazz;

    public DefaultLayerFactory(Class<? extends Layer> layerClazz) {
        this.layerClazz = layerClazz;
    }

    @Override
    public <E extends Layer> E create(NeuralNetConfiguration conf, int index, int numLayers, Collection<IterationListener> iterationListeners) {
        return create(conf, iterationListeners);
    }

    @Override
    public <E extends Layer> E create(NeuralNetConfiguration conf) {
        return create(conf, Collections.<IterationListener>emptyList());
    }

    @Override
    public <E extends Layer> E create(NeuralNetConfiguration conf, Collection<IterationListener> iterationListeners) {
        Distribution dist = Distributions.createDistribution(conf.getDist());
        Layer ret = getInstance(conf);
        ret.setIterationListeners(iterationListeners);
        Map<String,INDArray> params = getParams(conf, dist);
        ret.setParamTable(params);
        ret.setConf(conf);
        return (E) ret;
    }
    
    protected Layer getInstance(NeuralNetConfiguration conf) {
        try {
            Constructor<?> constructor = layerClazz.getConstructor(NeuralNetConfiguration.class);
            return (Layer) constructor.newInstance(conf);
        } catch (Exception e) {
            throw new IllegalStateException(e);
        }

    }


    protected Map<String,INDArray> getParams(NeuralNetConfiguration conf, Distribution dist) {
        ParamInitializer init = initializer();
        Map<String,INDArray> params = Collections.synchronizedMap(new LinkedHashMap<String,INDArray>());
        init.init(params,conf);
        return params;
    }

    @Override
    public String layerClazzName() {
        return layerClazz.getName();
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

        return !(layerClazz != null ? !layerClazz.equals(that.layerClazz) : that.layerClazz != null);
    }

    @Override
    public int hashCode() {
        return layerClazz != null ? layerClazz.hashCode() : 0;
    }
}
