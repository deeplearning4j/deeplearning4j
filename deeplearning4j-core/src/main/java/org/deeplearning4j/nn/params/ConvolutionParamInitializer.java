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
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Initialize convolution params.
 *
 * @author Adam Gibson
 */
public class ConvolutionParamInitializer implements ParamInitializer {

    public final static String WEIGHT_KEY = DefaultParamInitializer.WEIGHT_KEY;
    public final static String BIAS_KEY = DefaultParamInitializer.BIAS_KEY;

    @Override
    public int numParams(NeuralNetConfiguration conf, boolean backprop) {
        org.deeplearning4j.nn.conf.layers.ConvolutionLayer layerConf =
                (org.deeplearning4j.nn.conf.layers.ConvolutionLayer) conf.getLayer();

        int[] kernel = layerConf.getKernelSize();
        int nIn = layerConf.getNIn();
        int nOut = layerConf.getNOut();
        return nIn * nOut * kernel[0] * kernel[1] + nOut;
    }

    @Override
    public void init(Map<String, INDArray> params, NeuralNetConfiguration conf, INDArray paramsView) {
        if (((org.deeplearning4j.nn.conf.layers.ConvolutionLayer) conf.getLayer()).getKernelSize().length != 2)
            throw new IllegalArgumentException("Filter size must be == 2");

        org.deeplearning4j.nn.conf.layers.ConvolutionLayer layerConf =
                (org.deeplearning4j.nn.conf.layers.ConvolutionLayer) conf.getLayer();

        int[] kernel = layerConf.getKernelSize();
        int nIn = layerConf.getNIn();
        int nOut = layerConf.getNOut();

        INDArray biasView = paramsView.get(NDArrayIndex.point(0), NDArrayIndex.interval(0, nOut));
        INDArray weightView = paramsView.get(NDArrayIndex.point(0), NDArrayIndex.interval(nOut, numParams(conf,true)));

        params.put(BIAS_KEY, createBias(conf, biasView));
        params.put(WEIGHT_KEY, createWeightMatrix(conf, weightView));
        conf.addVariable(WEIGHT_KEY);
        conf.addVariable(BIAS_KEY);

    }

    @Override
    public Map<String, INDArray> getGradientsFromFlattened(NeuralNetConfiguration conf, INDArray gradientView) {

        org.deeplearning4j.nn.conf.layers.ConvolutionLayer layerConf =
                (org.deeplearning4j.nn.conf.layers.ConvolutionLayer) conf.getLayer();

        int[] kernel = layerConf.getKernelSize();
        int nIn = layerConf.getNIn();
        int nOut = layerConf.getNOut();

        INDArray biasGradientView = gradientView.get(NDArrayIndex.point(0), NDArrayIndex.interval(0, nOut));
        INDArray weightGradientView = gradientView.get(NDArrayIndex.point(0), NDArrayIndex.interval(nOut, numParams(conf,true)))
                .reshape('c',nOut, nIn, kernel[0], kernel[1]);

        Map<String,INDArray> out = new LinkedHashMap<>();
        out.put(BIAS_KEY, biasGradientView);
        out.put(WEIGHT_KEY, weightGradientView);
        return out;
    }

    //1 bias per feature map
    protected INDArray createBias(NeuralNetConfiguration conf, INDArray biasView) {
        //the bias is a 1D tensor -- one bias per output feature map
        org.deeplearning4j.nn.conf.layers.ConvolutionLayer layerConf =
                (org.deeplearning4j.nn.conf.layers.ConvolutionLayer) conf.getLayer();
        biasView.assign(layerConf.getBiasInit());
        return biasView;
    }


    protected INDArray createWeightMatrix(NeuralNetConfiguration conf, INDArray weightView) {
        /*
         Create a 4d weight matrix of:
           (number of kernels, num input channels, kernel height, kernel width)
         Note c order is used specifically for the CNN weights, as opposed to f order elsewhere
         Inputs to the convolution layer are:
         (batch size, num input feature maps, image height, image width)
         */
        org.deeplearning4j.nn.conf.layers.ConvolutionLayer layerConf =
                (org.deeplearning4j.nn.conf.layers.ConvolutionLayer) conf.getLayer();

        Distribution dist = Distributions.createDistribution(conf.getLayer().getDist());
        int[] kernel = layerConf.getKernelSize();
        return WeightInitUtil.initWeights(new int[]{layerConf.getNOut(), layerConf.getNIn(), kernel[0], kernel[1]},
                layerConf.getWeightInit(), dist, 'c', weightView);
    }
}
