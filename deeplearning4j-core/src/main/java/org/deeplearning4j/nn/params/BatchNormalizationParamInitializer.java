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
    public final static String GAMMA = "gamma";
    public final static String BETA = "beta";

    public final static String AVG_MEAN = "avgMean";
    public final static String AVG_VAR = "avgVar";

    public final static String GAMMA_GRADIENT = "gammaGradient";
    public final static String BETA_GRADIENT = "betaGradient";



    @Override
    public void init(Map<String, INDArray> params, NeuralNetConfiguration conf) {
        BatchNormalization normalization = (BatchNormalization) conf.getLayer();
//        int size = ArrayUtil.prod(normalization.getShape());
//
//        params.put(AVG_MEAN, Nd4j.zeros(1,size));
//        params.put(AVG_VAR,Nd4j.zerosLike(params.get(AVG_MEAN)));
//
//
//        params.put(GAMMA,Nd4j.onesLike(params.get(AVG_MEAN)));
//        params.put(GAMMA_GRADIENT,Nd4j.zerosLike(params.get(AVG_MEAN)));
//
//        params.put(BETA,Nd4j.zerosLike(params.get(AVG_MEAN)));
//        params.put(BETA_GRADIENT,Nd4j.zerosLike(params.get(AVG_MEAN)));


    }

    @Override
    public void init(Map<String, INDArray> params, NeuralNetConfiguration conf, Configuration extraConf) {

    }
}
