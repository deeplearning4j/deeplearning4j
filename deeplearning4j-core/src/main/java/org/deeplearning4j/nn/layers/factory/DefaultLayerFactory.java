package org.deeplearning4j.nn.layers.factory;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.LayerFactory;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.lang.reflect.Constructor;
import java.util.HashMap;
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
    public Layer create(NeuralNetConfiguration conf, int index, int numLayers) {
        return create(conf);
    }

    @Override
    public Layer create(NeuralNetConfiguration conf) {
        Layer ret = getInstance(conf);
        Map<String,INDArray> params = getParams(conf);
        ret.setParamTable(params);
        ret.setConfiguration(conf);
        return ret;
    }


    protected Layer getInstance(NeuralNetConfiguration conf) {
        try {
            Constructor<?> constructor = layerClazz.getConstructor(NeuralNetConfiguration.class);
            Layer ret = (Layer) constructor.newInstance(conf);
            return ret;
        } catch (Exception e) {
            throw new IllegalStateException(e);
        }

    }


    protected Map<String,INDArray> getParams(NeuralNetConfiguration conf) {
        ParamInitializer init = getInitializer();
        Map<String,INDArray> params = new HashMap<>();
        init.init(params,conf);
        return params;
    }

    @Override
    public ParamInitializer getInitializer() {
        return new DefaultParamInitializer();
    }
}
