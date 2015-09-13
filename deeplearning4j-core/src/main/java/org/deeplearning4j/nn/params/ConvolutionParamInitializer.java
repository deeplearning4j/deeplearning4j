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


import com.google.common.primitives.Ints;
import org.canova.api.conf.Configuration;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.Distributions;
import org.deeplearning4j.nn.weights.WeightInitUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.distribution.Distribution;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Map;

/**
 * Initialize convolution params.
 * @author Adam Gibson
 */
public class ConvolutionParamInitializer implements ParamInitializer {

    public final static String WEIGHT_KEY = DefaultParamInitializer.WEIGHT_KEY;
    public final static String BIAS_KEY = DefaultParamInitializer.BIAS_KEY;

    @Override
    public void init(Map<String, INDArray> params, NeuralNetConfiguration conf) {
        if(((org.deeplearning4j.nn.conf.layers.ConvolutionLayer) conf.getLayer()).getKernelSize().length < 2)
            throw new IllegalArgumentException("Filter size must be == 2");

        params.put(BIAS_KEY,createBias(conf));
        params.put(WEIGHT_KEY,createWeightMatrix(conf));
        conf.addVariable(WEIGHT_KEY);
        conf.addVariable(BIAS_KEY);

    }

    @Override
    public void init(Map<String, INDArray> params, NeuralNetConfiguration conf, Configuration extraConf) {
        init(params,conf);
    }

    //1 bias per feature map
    protected INDArray createBias(NeuralNetConfiguration conf) {
        //the bias is a 1D tensor -- one bias per output feature map
        org.deeplearning4j.nn.conf.layers.ConvolutionLayer layerConf =
                (org.deeplearning4j.nn.conf.layers.ConvolutionLayer) conf.getLayer();
        return Nd4j.valueArrayOf(layerConf.getNOut(), layerConf.getBiasInit());
    }


    protected INDArray createWeightMatrix(NeuralNetConfiguration conf) {
        /**
         * Create a 4d weight matrix of:
         *   (number of kernels, num input channels,
         kernel height, kernel width)
         Inputs to the convolution layer are:
         (batch size, num input feature maps,
         image height, image width)

         */
        org.deeplearning4j.nn.conf.layers.ConvolutionLayer layerConf =
                (org.deeplearning4j.nn.conf.layers.ConvolutionLayer) conf.getLayer();

        Distribution dist = Distributions.createDistribution(conf.getLayer().getDist());
        return WeightInitUtil.initWeights(
                Ints.concat(new int[] {layerConf.getNOut(), layerConf.getNIn()}, layerConf.getKernelSize()),
                layerConf.getWeightInit(),
                dist);
    }

}
