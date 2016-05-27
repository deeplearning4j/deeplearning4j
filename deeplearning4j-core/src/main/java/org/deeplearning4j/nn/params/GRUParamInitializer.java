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
    public final static String RECURRENT_WEIGHT_KEY = "RW";
    public final static String BIAS_KEY = DefaultParamInitializer.BIAS_KEY;
    public final static String INPUT_WEIGHT_KEY = DefaultParamInitializer.WEIGHT_KEY;

    @Override
    public int numParams(NeuralNetConfiguration conf, boolean backprop) {
        throw new UnsupportedOperationException("Not yet implemented"); //TODO
    }

    @Override
    public void init(Map<String, INDArray> params, NeuralNetConfiguration conf, INDArray paramsView) {
    	org.deeplearning4j.nn.conf.layers.GRU layerConf =
                (org.deeplearning4j.nn.conf.layers.GRU) conf.getLayer();
//        Distribution dist = Distributions.createDistribution(layerConf.getDist());
//
//        int nL = layerConf.getNOut();	//i.e., n neurons in this layer
//        int nLast = layerConf.getNIn();	//i.e., n neurons in previous layer
//
//        conf.addVariable(INPUT_WEIGHT_KEY);
//        conf.addVariable(RECURRENT_WEIGHT_KEY);
//        conf.addVariable(BIAS_KEY);
//
//        //Order: RUC - i.e., reset, update, candidate
//        params.put(INPUT_WEIGHT_KEY,WeightInitUtil.initWeights(nLast, 3 * nL, layerConf.getWeightInit(), dist));
//        params.put(RECURRENT_WEIGHT_KEY,WeightInitUtil.initWeights(nL, 3 * nL, layerConf.getWeightInit(), dist));
//        params.put(BIAS_KEY, Nd4j.zeros(1,3*nL));
//
//        params.get(INPUT_WEIGHT_KEY).data().persist();
//        params.get(RECURRENT_WEIGHT_KEY).data().persist();
//        params.get(BIAS_KEY).data().persist();

        throw new UnsupportedOperationException("Not yet implemented"); //TODO
    }

    @Override
    public Map<String, INDArray> getGradientsFromFlattened(NeuralNetConfiguration conf, INDArray gradientView) {
        throw new UnsupportedOperationException("Not yet implemented");
    }
}
