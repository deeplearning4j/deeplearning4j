package org.deeplearning4j.nn.params;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.weights.WeightInitUtil;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Map;

/**
 * Recursive autoencoder initializer
 * @author Adam Gibson
 */
public class RecursiveParamInitializer extends DefaultParamInitializer {


    public final static String W = "w";
    public final static String U = "u";
    public final static String BIAS = "b";
    public final static String C = "c";

    @Override
    public void init(Map<String, INDArray> params, NeuralNetConfiguration conf) {
        int vis = conf.getnIn();
        int out = vis * 2;
        params.put(W, WeightInitUtil.initWeights(new int[]{out,vis},conf.getWeightInit(),conf.getActivationFunction(),conf.getDist()));
        params.put(U, WeightInitUtil.initWeights(new int[]{vis,out},conf.getWeightInit(),conf.getActivationFunction(),conf.getDist()));
        params.put(BIAS, WeightInitUtil.initWeights(new int[]{vis},conf.getWeightInit(),conf.getActivationFunction(),conf.getDist()));
        params.put(C, WeightInitUtil.initWeights(new int[]{out},conf.getWeightInit(),conf.getActivationFunction(),conf.getDist()));
        conf.getGradientList().add(W);
        conf.getGradientList().add(U);
        conf.getGradientList().add(BIAS);
        conf.getGradientList().add(C);


    }


}
