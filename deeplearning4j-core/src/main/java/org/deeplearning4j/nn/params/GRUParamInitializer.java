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
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.Distributions;
import org.deeplearning4j.nn.weights.WeightInitUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.distribution.Distribution;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Map;


public class GRUParamInitializer implements ParamInitializer {
	/** Weights for previous time step -> current time step connections */
    public final static String RECURRENT_WEIGHTS = "RW";
    public final static String BIAS = DefaultParamInitializer.BIAS_KEY;
    public final static String INPUT_WEIGHTS = DefaultParamInitializer.WEIGHT_KEY;

    @Override
    public void init(Map<String, INDArray> params, NeuralNetConfiguration conf) {
        Distribution dist = Distributions.createDistribution(conf.getDist());

        int nL = conf.getNOut();	//i.e., n neurons in this layer
        int nLast = conf.getNIn();	//i.e., n neurons in previous layer
        
        
        conf.addVariable(RECURRENT_WEIGHTS);
        conf.addVariable(INPUT_WEIGHTS);
        conf.addVariable(BIAS);
        
        
        //Order: RUC - i.e., reset, update, candidate
        params.put(INPUT_WEIGHTS,WeightInitUtil.initWeights(nLast, 3 * nL, conf.getWeightInit(), dist));
        params.put(RECURRENT_WEIGHTS,WeightInitUtil.initWeights(nL, 3 * nL, conf.getWeightInit(), dist));
        params.put(BIAS, Nd4j.zeros(1,3*nL));

        params.get(RECURRENT_WEIGHTS).data().persist();
        params.get(INPUT_WEIGHTS).data().persist();
        params.get(BIAS).data().persist();
    }

    @Override
    public void init(Map<String, INDArray> params, NeuralNetConfiguration conf, Configuration extraConf) {
        init(params,conf);
    }
}
