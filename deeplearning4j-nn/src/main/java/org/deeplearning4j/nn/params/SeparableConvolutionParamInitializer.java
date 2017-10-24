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


import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.Distributions;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.layers.SeparableConvolution2D;
import org.deeplearning4j.nn.weights.WeightInitUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.distribution.Distribution;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.*;

/**
 * Initialize separable convolution params.
 *
 * @author Max Pumperla
 */
public class SeparableConvolutionParamInitializer implements ParamInitializer {

    private static final SeparableConvolutionParamInitializer INSTANCE = new SeparableConvolutionParamInitializer();

    public static SeparableConvolutionParamInitializer getInstance() {
        return INSTANCE;
    }

    public final static String DEPTH_WISE_WEIGHT_KEY = DefaultParamInitializer.WEIGHT_KEY;
    public final static String POINT_WISE_WEIGHT_KEY = "pW";
    public final static String BIAS_KEY = DefaultParamInitializer.BIAS_KEY;

    @Override
    public int numParams(NeuralNetConfiguration conf) {
        return numParams(conf.getLayer());
    }

    @Override
    public int numParams(Layer l) {
        SeparableConvolution2D layerConf = (SeparableConvolution2D) l;

        int depthWiseParams = numDepthWiseParams(layerConf);
        int pointWiseParams = numPointWiseParams(layerConf);
        int biasParams = numBiasParams(layerConf);

        return depthWiseParams + pointWiseParams + biasParams;
    }

    private int numBiasParams(SeparableConvolution2D layerConf) {
        int nOut = layerConf.getNOut();
        return (layerConf.hasBias() ? nOut : 0);
    }

    /**
     * For each input feature we separately compute depthMultiplier many
     * output maps for the given kernel size
     *
     * @param layerConf layer configuration of the separable conv2d layer
     * @return number of parameters of the depth-wise convolution operation
     */
    private int numDepthWiseParams(SeparableConvolution2D layerConf) {
        int[] kernel = layerConf.getKernelSize();
        int nIn = layerConf.getNIn();
        int depthMultiplier = layerConf.getDepthMultiplier();

        return nIn * depthMultiplier * kernel[0] * kernel[1];
    }

    /**
     * For the point-wise convolution part we have (nIn * depthMultiplier) many
     * input maps and nOut output maps. Kernel size is (1, 1) for this operation.
     *
     * @param layerConf layer configuration of the separable conv2d layer
     * @return number of parameters of the point-wise convolution operation
     */
    private int numPointWiseParams(SeparableConvolution2D layerConf) {
        int nIn = layerConf.getNIn();
        int nOut = layerConf.getNOut();
        int depthMultiplier = layerConf.getDepthMultiplier();

        return (nIn * depthMultiplier) * nOut;
    }

    @Override
    public List<String> paramKeys(Layer layer) {
        SeparableConvolution2D layerConf =
                (SeparableConvolution2D) layer;
        if(layerConf.hasBias()){
            return Arrays.asList(DEPTH_WISE_WEIGHT_KEY, POINT_WISE_WEIGHT_KEY, BIAS_KEY);
        } else {
            return weightKeys(layer);
        }
    }

    @Override
    public List<String> weightKeys(Layer layer) {
        return Arrays.asList(DEPTH_WISE_WEIGHT_KEY, POINT_WISE_WEIGHT_KEY);
    }

    @Override
    public List<String> biasKeys(Layer layer) {
        SeparableConvolution2D layerConf =
                (SeparableConvolution2D) layer;
        if(layerConf.hasBias()){
            return Collections.singletonList(BIAS_KEY);
        } else {
            return Collections.emptyList();
        }
    }

    @Override
    public boolean isWeightParam(String key) {
        return DEPTH_WISE_WEIGHT_KEY.equals(key) || POINT_WISE_WEIGHT_KEY.equals(key);
    }

    @Override
    public boolean isBiasParam(String key) {
        return BIAS_KEY.equals(key);
    }


    @Override
    public Map<String, INDArray> init(NeuralNetConfiguration conf, INDArray paramsView, boolean initializeParams) {
        SeparableConvolution2D layer = (SeparableConvolution2D) conf.getLayer();
        if (layer.getKernelSize().length != 2) throw new IllegalArgumentException("Filter size must be == 2");

        Map<String, INDArray> params = Collections.synchronizedMap(new LinkedHashMap<String, INDArray>());
        SeparableConvolution2D layerConf = (SeparableConvolution2D) conf.getLayer();

        int depthWiseParams = numDepthWiseParams(layerConf);
        int biasParams = numBiasParams(layerConf);

        INDArray depthWiseWeightView = paramsView.get(
                NDArrayIndex.point(0), NDArrayIndex.interval(biasParams, biasParams + depthWiseParams));
        INDArray pointWiseWeightView = paramsView.get(
                NDArrayIndex.point(0), NDArrayIndex.interval(biasParams + depthWiseParams, numParams(conf)));

        params.put(DEPTH_WISE_WEIGHT_KEY, createDepthWiseWeightMatrix(conf, depthWiseWeightView, initializeParams));
        conf.addVariable(DEPTH_WISE_WEIGHT_KEY);
        params.put(POINT_WISE_WEIGHT_KEY, createPointWiseWeightMatrix(conf, pointWiseWeightView, initializeParams));
        conf.addVariable(POINT_WISE_WEIGHT_KEY);

        if(layer.hasBias()){
            INDArray biasView = paramsView.get(NDArrayIndex.point(0), NDArrayIndex.interval(0, biasParams));
            params.put(BIAS_KEY, createBias(conf, biasView, initializeParams));
            conf.addVariable(BIAS_KEY);
        }

        return params;
    }

    @Override
    public Map<String, INDArray> getGradientsFromFlattened(NeuralNetConfiguration conf, INDArray gradientView) {

        SeparableConvolution2D layerConf =
                        (SeparableConvolution2D) conf.getLayer();

        int[] kernel = layerConf.getKernelSize();
        int nIn = layerConf.getNIn();
        int depthMultiplier = layerConf.getDepthMultiplier();
        int nOut = layerConf.getNOut();

        Map<String, INDArray> out = new LinkedHashMap<>();

        int depthWiseParams = numDepthWiseParams(layerConf);
        int biasParams = numBiasParams(layerConf);

        INDArray depthWiseWeightGradientView = gradientView.get(
                NDArrayIndex.point(0), NDArrayIndex.interval(biasParams, biasParams + depthWiseParams))
                .reshape('c', depthMultiplier, nIn, kernel[0], kernel[1]);
        INDArray pointWiseWeightGradientView = gradientView.get(
                NDArrayIndex.point(0), NDArrayIndex.interval(biasParams + depthWiseParams, numParams(conf)))
                .reshape('c', nOut, nIn * depthMultiplier, 1, 1);
        out.put(DEPTH_WISE_WEIGHT_KEY, depthWiseWeightGradientView);
        out.put(POINT_WISE_WEIGHT_KEY, pointWiseWeightGradientView);

        if(layerConf.hasBias()){
            INDArray biasGradientView = gradientView.get(NDArrayIndex.point(0), NDArrayIndex.interval(0, nOut));
            out.put(BIAS_KEY, biasGradientView);
        }
        return out;
    }

    protected INDArray createBias(NeuralNetConfiguration conf, INDArray biasView, boolean initializeParams) {
        SeparableConvolution2D layerConf =
                        (SeparableConvolution2D) conf.getLayer();
        if (initializeParams)
            biasView.assign(layerConf.getBiasInit());
        return biasView;
    }


    protected INDArray createDepthWiseWeightMatrix(NeuralNetConfiguration conf, INDArray weightView, boolean initializeParams) {
        /*
         Create a 4d weight matrix of: (depth multiplier, num input channels, kernel height, kernel width)
         Inputs to the convolution layer are: (batch size, num input feature maps, image height, image width)
         */
        SeparableConvolution2D layerConf =
                        (SeparableConvolution2D) conf.getLayer();
        int depthMultiplier = layerConf.getDepthMultiplier();

        if (initializeParams) {
            Distribution dist = Distributions.createDistribution(layerConf.getDist());
            int[] kernel = layerConf.getKernelSize();
            int[] stride = layerConf.getStride();

            int inputDepth = layerConf.getNIn();

            double fanIn = inputDepth * kernel[0] * kernel[1];
            double fanOut = depthMultiplier * kernel[0] * kernel[1] / ((double) stride[0] * stride[1]);

            int[] weightsShape = new int[] {depthMultiplier, inputDepth, kernel[0], kernel[1]};

            return WeightInitUtil.initWeights(fanIn, fanOut, weightsShape, layerConf.getWeightInit(), dist, 'c',
                            weightView);
        } else {
            int[] kernel = layerConf.getKernelSize();
            return WeightInitUtil.reshapeWeights(
                            new int[] {depthMultiplier, layerConf.getNIn(), kernel[0], kernel[1]}, weightView, 'c');
        }
    }

    protected INDArray createPointWiseWeightMatrix(NeuralNetConfiguration conf, INDArray weightView,
                                                   boolean initializeParams) {
        /*
         Create a 4d weight matrix of: (num output channels, depth multiplier * num input channels,
         kernel height, kernel width)
         */
        SeparableConvolution2D layerConf =
                (SeparableConvolution2D) conf.getLayer();
        int depthMultiplier = layerConf.getDepthMultiplier();

        if (initializeParams) {
            Distribution dist = Distributions.createDistribution(layerConf.getDist());

            int inputDepth = layerConf.getNIn();
            int outputDepth = layerConf.getNOut();

            double fanIn = inputDepth * depthMultiplier;
            double fanOut = fanIn;

            int[] weightsShape = new int[] {outputDepth, depthMultiplier * inputDepth, 1, 1};

            return WeightInitUtil.initWeights(fanIn, fanOut, weightsShape, layerConf.getWeightInit(), dist, 'c',
                    weightView);
        } else {
            return WeightInitUtil.reshapeWeights(
                    new int[] {layerConf.getNOut(), depthMultiplier * layerConf.getNIn(), 1, 1}, weightView, 'c');
        }
    }
}
