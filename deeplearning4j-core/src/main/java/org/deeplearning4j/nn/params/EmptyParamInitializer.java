package org.deeplearning4j.nn.params;

import org.canova.api.conf.Configuration;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Map;

/**
 * @author Adam Gibson
 */
public class EmptyParamInitializer implements ParamInitializer {
    @Override
    public void init(Map<String, INDArray> params, NeuralNetConfiguration conf) {

    }

    @Override
    public void init(Map<String, INDArray> params, NeuralNetConfiguration conf, Configuration extraConf) {

    }
}
