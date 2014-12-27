package org.deeplearning4j.nn.params;

import org.deeplearning4j.nn.weights.WeightInitUtil;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Map;

/**
 * LSTM Parameters
 * @author Adam Gibson
 */
public class LSTMParamInitializer implements ParamInitializer {
    public final static String RECURRENT_WEIGHTS = "recurrentweights";
    public final static String DECODER_BIAS = "decoderbias";
    public final static String DECODER_WEIGHTS = "decoderweights";

    @Override
    public void init(Map<String, INDArray> params, NeuralNetConfiguration conf) {
        params.put(RECURRENT_WEIGHTS,WeightInitUtil.initWeights(conf.getnIn() + conf.getnOut() + 1, conf.getnOut() * 4, conf.getWeightInit(), conf.getActivationFunction(), conf.getDist()));
        params.put(DECODER_WEIGHTS,WeightInitUtil.initWeights(conf.getnOut(),conf.getRecurrentOutput(),conf.getWeightInit(),conf.getActivationFunction(),conf.getDist()));
        params.put(DECODER_BIAS, Nd4j.zeros(conf.getRecurrentOutput()));

    }
}
