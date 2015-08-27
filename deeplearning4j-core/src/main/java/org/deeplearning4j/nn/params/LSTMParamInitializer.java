/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.deeplearning4j.nn.params;

import org.canova.api.conf.Configuration;
import org.deeplearning4j.nn.weights.WeightInitUtil;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.Distributions;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.api.rng.distribution.Distribution;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Map;

/**
 * LSTM Parameters.
 * Recurrent weights represent the all of the related parameters for the recurrent net.
 * The decoder weights are used for predictions.
 * @author Adam Gibson
 */
public class LSTMParamInitializer implements ParamInitializer {
    public final static String RECURRENT_WEIGHT_KEY = "RW";
    public final static String INPUT_WEIGHT_KEY = DefaultParamInitializer.WEIGHT_KEY;
    public final static String BIAS_KEY = DefaultParamInitializer.BIAS_KEY;
    @Override
    public void init(Map<String, INDArray> params, NeuralNetConfiguration conf) {
        org.deeplearning4j.nn.conf.layers.LSTM layerConf =
                (org.deeplearning4j.nn.conf.layers.LSTM) conf.getLayer();

        Distribution dist = Distributions.createDistribution(layerConf.getDist());

        int inputSize = layerConf.getNIn();
        int hiddenSize = 8; //layerConf.getNIn(); // TODO add attribute to pass in hiddenSize
        int outputSize = layerConf.getNOut();

        conf.addVariable(RECURRENT_WEIGHT_KEY);
        conf.addVariable(INPUT_WEIGHT_KEY);
        conf.addVariable(BIAS_KEY);

        params.put(RECURRENT_WEIGHT_KEY,WeightInitUtil.initWeights(inputSize + hiddenSize, 4 * hiddenSize, layerConf.getWeightInit(), dist));
        params.put(INPUT_WEIGHT_KEY,WeightInitUtil.initWeights(hiddenSize,outputSize,layerConf.getWeightInit(), dist));
        params.put(BIAS_KEY, Nd4j.zeros(outputSize));
        params.get(RECURRENT_WEIGHT_KEY).data().persist();
        params.get(BIAS_KEY).data().persist();
        params.get(INPUT_WEIGHT_KEY).data().persist();

    }

    @Override
    public void init(Map<String, INDArray> params, NeuralNetConfiguration conf, Configuration extraConf) {
        init(params,conf);
    }
}
