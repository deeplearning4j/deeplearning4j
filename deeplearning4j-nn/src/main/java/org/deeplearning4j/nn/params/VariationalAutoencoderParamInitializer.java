package org.deeplearning4j.nn.params;

import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Map;

/**
 * Created by Alex on 25/11/2016.
 */
public class VariationalAutoencoderParamInitializer implements ParamInitializer {

    private static final VariationalAutoencoderParamInitializer INSTANCE = new VariationalAutoencoderParamInitializer();
    public static VariationalAutoencoderParamInitializer getInstance(){
        return INSTANCE;
    }

    @Override
    public int numParams(NeuralNetConfiguration conf, boolean backprop) {
        return 0;
    }

    @Override
    public Map<String, INDArray> init(NeuralNetConfiguration conf, INDArray paramsView, boolean initializeParams) {
        return null;
    }

    @Override
    public Map<String, INDArray> getGradientsFromFlattened(NeuralNetConfiguration conf, INDArray gradientView) {
        return null;
    }
}
