/*
 * Copyright 2015 Skymind,Inc.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

package org.deeplearning4j.nn.params;

import org.canova.api.conf.Configuration;
import org.deeplearning4j.nn.weights.WeightInitUtil;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import java.util.Map;

/**
 * LSTM Parameters.
 * Recurrent weights represent the all of the related parameters for the recurrent net.
 * The decoder weights are used for predictions.
 * @author Adam Gibson
 */
public class LSTMParamInitializer implements ParamInitializer {
    public final static String RECURRENT_WEIGHTS = "recurrentweights";
    public final static String DECODER_BIAS = "decoderbias";
    public final static String DECODER_WEIGHTS = "decoderweights";

    @Override
    public void init(Map<String, INDArray> params, NeuralNetConfiguration conf) {
        int inputSize = conf.getnIn();
        int hiddenSize = conf.getnIn();
        int outputSize = conf.getnOut();
        conf.addVariable(RECURRENT_WEIGHTS);
        conf.addVariable(DECODER_WEIGHTS);
        conf.addVariable(DECODER_BIAS);
        params.put(RECURRENT_WEIGHTS,WeightInitUtil.initWeights(inputSize + hiddenSize + 1, 4 * hiddenSize, conf.getWeightInit(), conf.getDist()));
        params.put(DECODER_WEIGHTS,WeightInitUtil.initWeights(hiddenSize,outputSize,conf.getWeightInit(),conf.getDist()));
        params.put(DECODER_BIAS, Nd4j.zeros(outputSize));

    }

    @Override
    public void init(Map<String, INDArray> params, NeuralNetConfiguration conf, Configuration extraConf) {
        init(params,conf);
    }
}
