package org.deeplearning4j.nn.params;

import org.canova.api.conf.Configuration;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.BatchNormalization;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.Map;

/**
 * Batch normalization variable init
 *
 * @author Adam Gibson
 */

public class BatchNormalizationParamInitializer implements ParamInitializer {
    public final static String GAMMA = "gamma"; // equivalent to weights
    public final static String BETA = DefaultParamInitializer.BIAS_KEY; // equivalent to bias

    @Override
    public void init(Map<String, INDArray> params, NeuralNetConfiguration conf) {
        params.put(GAMMA,createGamma(conf));
        conf.addVariable(GAMMA);
        params.put(BETA, createBeta(conf));
        conf.addVariable(BETA);
    }

    @Override
    public void init(Map<String, INDArray> params, NeuralNetConfiguration conf, Configuration extraConf) {
        init(params,conf);
    }

    protected INDArray createBeta(NeuralNetConfiguration conf) {
        BatchNormalization layer = (BatchNormalization) conf.getLayer();
        INDArray ret = Nd4j.valueArrayOf(1, layer.getNOut(), layer.getBeta());
        ret.data().persist();
        return ret;
    }

    protected INDArray createGamma(NeuralNetConfiguration conf) {
        BatchNormalization layer = (BatchNormalization) conf.getLayer();
        INDArray ret = Nd4j.valueArrayOf(1, layer.getNOut(), layer.getGamma());
        ret.data().persist();
        return ret;
    }

}
