/*-
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


import lombok.val;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.Distributions;
import org.deeplearning4j.nn.conf.layers.DepthwiseConvolution2D;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.weights.WeightInitUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.distribution.Distribution;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.*;

/**
 * Initialize depth-wise convolution parameters.
 *
 * @author Max Pumperla
 */
public class DepthwiseConvolutionParamInitializer implements ParamInitializer {

    private static final DepthwiseConvolutionParamInitializer INSTANCE = new DepthwiseConvolutionParamInitializer();

    public static DepthwiseConvolutionParamInitializer getInstance() {
        return INSTANCE;
    }

    public final static String WEIGHT_KEY = DefaultParamInitializer.WEIGHT_KEY;
    public final static String BIAS_KEY = DefaultParamInitializer.BIAS_KEY;

    @Override
    public long numParams(NeuralNetConfiguration conf) {
        return numParams(conf.getLayer());
    }

    @Override
    public long numParams(Layer l) {
        DepthwiseConvolution2D layerConf = (DepthwiseConvolution2D) l;

        val depthWiseParams = numDepthWiseParams(layerConf);
        val biasParams = numBiasParams(layerConf);

        return depthWiseParams + biasParams;
    }

    private long numBiasParams(DepthwiseConvolution2D layerConf) {
        val nOut = layerConf.getNOut();
        return (layerConf.hasBias() ? nOut : 0);
    }

    /**
     * For each input feature we separately compute depthMultiplier many
     * output maps for the given kernel size
     *
     * @param layerConf layer configuration of the separable conv2d layer
     * @return number of parameters of the channels-wise convolution operation
     */
    private long numDepthWiseParams(DepthwiseConvolution2D layerConf) {
        int[] kernel = layerConf.getKernelSize();
        val nIn = layerConf.getNIn();
        val depthMultiplier = layerConf.getDepthMultiplier();

        return nIn * depthMultiplier * kernel[0] * kernel[1];
    }

    @Override
    public List<String> paramKeys(Layer layer) {
        DepthwiseConvolution2D layerConf =
                (DepthwiseConvolution2D) layer;
        if(layerConf.hasBias()){
            return Arrays.asList(WEIGHT_KEY, BIAS_KEY);
        } else {
            return weightKeys(layer);
        }
    }

    @Override
    public List<String> weightKeys(Layer layer) {
        return Arrays.asList(WEIGHT_KEY);
    }

    @Override
    public List<String> biasKeys(Layer layer) {
        DepthwiseConvolution2D layerConf =
                (DepthwiseConvolution2D) layer;
        if(layerConf.hasBias()){
            return Collections.singletonList(BIAS_KEY);
        } else {
            return Collections.emptyList();
        }
    }

    @Override
    public boolean isWeightParam(Layer layer, String key) {
        return WEIGHT_KEY.equals(key);
    }

    @Override
    public boolean isBiasParam(Layer layer, String key) {
        return BIAS_KEY.equals(key);
    }


    @Override
    public Map<String, INDArray> init(NeuralNetConfiguration conf, INDArray paramsView, boolean initializeParams) {
        DepthwiseConvolution2D layer = (DepthwiseConvolution2D) conf.getLayer();
        if (layer.getKernelSize().length != 2) throw new IllegalArgumentException("Filter size must be == 2");

        Map<String, INDArray> params = Collections.synchronizedMap(new LinkedHashMap<String, INDArray>());
        DepthwiseConvolution2D layerConf = (DepthwiseConvolution2D) conf.getLayer();

        val depthWiseParams = numDepthWiseParams(layerConf);
        val biasParams = numBiasParams(layerConf);

        INDArray depthWiseWeightView = paramsView.get(
                NDArrayIndex.point(0), NDArrayIndex.interval(biasParams, biasParams + depthWiseParams));

        params.put(WEIGHT_KEY, createDepthWiseWeightMatrix(conf, depthWiseWeightView, initializeParams));
        conf.addVariable(WEIGHT_KEY);

        if(layer.hasBias()){
            INDArray biasView = paramsView.get(NDArrayIndex.point(0), NDArrayIndex.interval(0, biasParams));
            params.put(BIAS_KEY, createBias(conf, biasView, initializeParams));
            conf.addVariable(BIAS_KEY);
        }

        return params;
    }

    @Override
    public Map<String, INDArray> getGradientsFromFlattened(NeuralNetConfiguration conf, INDArray gradientView) {

        DepthwiseConvolution2D layerConf = (DepthwiseConvolution2D) conf.getLayer();

        int[] kernel = layerConf.getKernelSize();
        val nIn = layerConf.getNIn();
        val depthMultiplier = layerConf.getDepthMultiplier();
        val nOut = layerConf.getNOut();

        Map<String, INDArray> out = new LinkedHashMap<>();

        val depthWiseParams = numDepthWiseParams(layerConf);
        val biasParams = numBiasParams(layerConf);

        INDArray depthWiseWeightGradientView = gradientView.get(
                NDArrayIndex.point(0), NDArrayIndex.interval(biasParams, biasParams + depthWiseParams))
                .reshape('c', depthMultiplier, nIn, kernel[0], kernel[1]);
        out.put(WEIGHT_KEY, depthWiseWeightGradientView);

        if(layerConf.hasBias()){
            INDArray biasGradientView = gradientView.get(NDArrayIndex.point(0), NDArrayIndex.interval(0, nOut));
            out.put(BIAS_KEY, biasGradientView);
        }
        return out;
    }

    protected INDArray createBias(NeuralNetConfiguration conf, INDArray biasView, boolean initializeParams) {
        DepthwiseConvolution2D layerConf = (DepthwiseConvolution2D) conf.getLayer();
        if (initializeParams)
            biasView.assign(layerConf.getBiasInit());
        return biasView;
    }


    protected INDArray createDepthWiseWeightMatrix(NeuralNetConfiguration conf, INDArray weightView, boolean initializeParams) {
        /*
         Create a 4d weight matrix of: (channels multiplier, num input channels, kernel height, kernel width)
         Inputs to the convolution layer are: (batch size, num input feature maps, image height, image width)
         */
        DepthwiseConvolution2D layerConf =
                        (DepthwiseConvolution2D) conf.getLayer();
        int depthMultiplier = layerConf.getDepthMultiplier();

        if (initializeParams) {
            Distribution dist = Distributions.createDistribution(layerConf.getDist());
            int[] kernel = layerConf.getKernelSize();
            int[] stride = layerConf.getStride();

            val inputDepth = layerConf.getNIn();

            double fanIn = inputDepth * kernel[0] * kernel[1];
            double fanOut = depthMultiplier * kernel[0] * kernel[1] / ((double) stride[0] * stride[1]);

            val weightsShape = new long[] {depthMultiplier, inputDepth, kernel[0], kernel[1]};

            return WeightInitUtil.initWeights(fanIn, fanOut, weightsShape, layerConf.getWeightInit(), dist, 'c',
                            weightView);
        } else {
            int[] kernel = layerConf.getKernelSize();
            return WeightInitUtil.reshapeWeights(
                            new long[] {depthMultiplier, layerConf.getNIn(), kernel[0], kernel[1]}, weightView, 'c');
        }
    }
}
