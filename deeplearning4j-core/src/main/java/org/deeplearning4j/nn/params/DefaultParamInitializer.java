package org.deeplearning4j.nn.params;

import org.deeplearning4j.nn.WeightInitUtil;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Map;

/**
 * Static weight initializer with just a weight matrix and a bias
 * @author Adam Gibson
 */
public class DefaultParamInitializer implements ParamInitializer {

    public final static String WEIGHT_KEY = "W";
    public final static String BIAS_KEY = "b";

    @Override
    public void init(Map<String, INDArray> params, NeuralNetConfiguration conf) {
        params.put(WEIGHT_KEY,createWeightMatrix(conf));
        params.put(BIAS_KEY,createBias(conf));


    }



    protected INDArray createBias(NeuralNetConfiguration conf) {
        return Nd4j.zeros(conf.getnOut());
    }


    protected INDArray createWeightMatrix(NeuralNetConfiguration conf) {
        INDArray W = WeightInitUtil.initWeights(
                conf.getnIn(),
                conf.getnOut(),
                conf.getWeightInit(),
                conf.getActivationFunction(),
                conf.getDist());
        return W;
    }


}
