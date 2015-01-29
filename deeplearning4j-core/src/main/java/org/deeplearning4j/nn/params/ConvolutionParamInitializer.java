package org.deeplearning4j.nn.params;


import org.canova.api.conf.Configuration;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.weights.WeightInitUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Map;

/**
 * Initialize convolution params.
 * @author Adam Gibson
 */
public class ConvolutionParamInitializer implements ParamInitializer {

    public final static String CONVOLUTION_BIAS = "convbias";
    public final static String CONVOLUTION_WEIGHTS = "convweights";
    @Override
    public void init(Map<String, INDArray> params, NeuralNetConfiguration conf) {
        if(conf.getFilterSize().length < 4)
            throw new IllegalArgumentException("Filter size must be == 4");

        params.put(CONVOLUTION_BIAS,createBias(conf));
        params.put(CONVOLUTION_WEIGHTS,createWeightMatrix(conf));
        conf.addVariable(CONVOLUTION_WEIGHTS);
        conf.addVariable(CONVOLUTION_BIAS);

    }

    @Override
    public void init(Map<String, INDArray> params, NeuralNetConfiguration conf, Configuration extraConf) {
        init(params,conf);
    }


    protected INDArray createBias(NeuralNetConfiguration conf) {
        return Nd4j.zeros(conf.getFilterSize()[0]);
    }


    protected INDArray createWeightMatrix(NeuralNetConfiguration conf) {
       return WeightInitUtil.initWeights(conf.getFilterSize(),conf.getWeightInit(),conf.getActivationFunction(),conf.getDist());
    }

}
