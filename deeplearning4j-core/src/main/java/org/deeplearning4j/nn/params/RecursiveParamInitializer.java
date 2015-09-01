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

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.Distributions;
import org.deeplearning4j.nn.weights.WeightInitUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.distribution.Distribution;

import java.util.Map;

/**
 * Recursive autoencoder initializer
 * @author Adam Gibson
 */
public class RecursiveParamInitializer extends DefaultParamInitializer {

    //encoder weights
    public final static String ENCODER_WEIGHT_KEY = DefaultParamInitializer.WEIGHT_KEY;
    //decoder weights
    public final static String DECODER_WEIGHT_KEY = "U";
    //hidden bias
    public final static String HIDDEN_BIAS_KEY = DefaultParamInitializer.BIAS_KEY;
    //visible bias
    public final static String VISIBLE_BIAS_KEY = PretrainParamInitializer.VISIBLE_BIAS_KEY;

    @Override
    public void init(Map<String, INDArray> params, NeuralNetConfiguration conf) {
        org.deeplearning4j.nn.conf.layers.RecursiveAutoEncoder layerConf =
                (org.deeplearning4j.nn.conf.layers.RecursiveAutoEncoder) conf.getLayer();

        Distribution dist = Distributions.createDistribution(layerConf.getDist());

        int vis = layerConf.getNIn();
        int out = vis * 2;

        params.put(ENCODER_WEIGHT_KEY, WeightInitUtil.initWeights(new int[]{vis, out},layerConf.getWeightInit(), dist));
        params.put(DECODER_WEIGHT_KEY, WeightInitUtil.initWeights(new int[]{out, vis},layerConf.getWeightInit(), dist));
        params.put(HIDDEN_BIAS_KEY, WeightInitUtil.initWeights(new int[]{1, out},layerConf.getWeightInit(), dist));
        params.put(VISIBLE_BIAS_KEY, WeightInitUtil.initWeights(new int[]{1, vis},layerConf.getWeightInit(), dist));

        conf.addVariable(ENCODER_WEIGHT_KEY);
        conf.addVariable(DECODER_WEIGHT_KEY);
        conf.addVariable(HIDDEN_BIAS_KEY);
        conf.addVariable(VISIBLE_BIAS_KEY);
    }


}
