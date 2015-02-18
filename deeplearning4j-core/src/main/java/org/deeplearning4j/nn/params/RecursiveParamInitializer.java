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

    //encoder weights
    public final static String W = "w";
    //decoder weights
    public final static String U = "u";
    //hidden bias
    public final static String BIAS = "b";
    //visible bias
    public final static String C = "c";

    @Override
    public void init(Map<String, INDArray> params, NeuralNetConfiguration conf) {
        int vis = conf.getnIn();
        int out = vis * 2;

        params.put(W, WeightInitUtil.initWeights(new int[]{out,vis},conf.getWeightInit(), conf.getDist()));
        params.put(U, WeightInitUtil.initWeights(new int[]{vis,out},conf.getWeightInit(), conf.getDist()));
        params.put(BIAS, WeightInitUtil.initWeights(new int[]{out},conf.getWeightInit(), conf.getDist()));
        params.put(C, WeightInitUtil.initWeights(new int[]{vis},conf.getWeightInit(), conf.getDist()));

        conf.addVariable(W);
        conf.addVariable(U);
        conf.addVariable(BIAS);
        conf.addVariable(C);



    }


}
