/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.nn.params;


import lombok.val;
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
    public long numParams(NeuralNetConfiguration conf) {
        return numParams(conf.getLayer());
    }

    @Override
    public long numParams(Layer l) {
        SeparableConvolution2D layerConf = (SeparableConvolution2D) l;

        val depthWiseParams = numDepthWiseParams(layerConf);
        val pointWiseParams = numPointWiseParams(layerConf);
        val biasParams = numBiasParams(layerConf);

        return depthWiseParams + pointWiseParams + biasParams;
    }

    private long numBiasParams(SeparableConvolution2D layerConf) {
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
    private long numDepthWiseParams(SeparableConvolution2D layerConf) {
        int[] kernel = layerConf.getKernelSize();
        val nIn = layerConf.getNIn();
        val depthMultiplier = layerConf.getDepthMultiplier();

        return nIn * depthMultiplier * kernel[0] * kernel[1];
    }

    /**
     * For the point-wise convolution part we have (nIn * depthMultiplier) many
     * input maps and nOut output maps. Kernel size is (1, 1) for this operation.
     *
     * @param layerConf layer configuration of the separable conv2d layer
     * @return number of parameters of the point-wise convolution operation
     */
    private long numPointWiseParams(SeparableConvolution2D layerConf) {
        val nIn = layerConf.getNIn();
        val nOut = layerConf.getNOut();
        val depthMultiplier = layerConf.getDepthMultiplier();

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
    public boolean isWeightParam(Layer layer, String key) {
        return DEPTH_WISE_WEIGHT_KEY.equals(key) || POINT_WISE_WEIGHT_KEY.equals(key);
    }

    @Override
    public boolean isBiasParam(Layer layer, String key) {
        return BIAS_KEY.equals(key);
    }


    @Override
    public Map<String, INDArray> init(NeuralNetConfiguration conf, INDArray paramsView, boolean initializeParams) {
        SeparableConvolution2D layer = (SeparableConvolution2D) conf.getLayer();
        if (layer.getKernelSize().length != 2) throw new IllegalArgumentException("Filter size must be == 2");

        Map<String, INDArray> params = Collections.synchronizedMap(new LinkedHashMap<String, INDArray>());
        SeparableConvolution2D layerConf = (SeparableConvolution2D) conf.getLayer();

        val depthWiseParams = numDepthWiseParams(layerConf);
        val biasParams = numBiasParams(layerConf);

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
        val nIn = layerConf.getNIn();
        val depthMultiplier = layerConf.getDepthMultiplier();
        val nOut = layerConf.getNOut();

        Map<String, INDArray> out = new LinkedHashMap<>();

        val depthWiseParams = numDepthWiseParams(layerConf);
        val biasParams = numBiasParams(layerConf);

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
         Create a 4d weight matrix of: (channels multiplier, num input channels, kernel height, kernel width)
         Inputs to the convolution layer are: (batch size, num input feature maps, image height, image width)
         */
        SeparableConvolution2D layerConf =
                        (SeparableConvolution2D) conf.getLayer();
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

    protected INDArray createPointWiseWeightMatrix(NeuralNetConfiguration conf, INDArray weightView,
                                                   boolean initializeParams) {
        /*
         Create a 4d weight matrix of: (num output channels, channels multiplier * num input channels,
         kernel height, kernel width)
         */
        SeparableConvolution2D layerConf =
                (SeparableConvolution2D) conf.getLayer();
        int depthMultiplier = layerConf.getDepthMultiplier();

        if (initializeParams) {
            Distribution dist = Distributions.createDistribution(layerConf.getDist());

            val inputDepth = layerConf.getNIn();
            val outputDepth = layerConf.getNOut();

            double fanIn = inputDepth * depthMultiplier;
            double fanOut = fanIn;

            val weightsShape = new long[] {outputDepth, depthMultiplier * inputDepth, 1, 1};

            return WeightInitUtil.initWeights(fanIn, fanOut, weightsShape, layerConf.getWeightInit(), dist, 'c',
                    weightView);
        } else {
            return WeightInitUtil.reshapeWeights(
                    new long[] {layerConf.getNOut(), depthMultiplier * layerConf.getNIn(), 1, 1}, weightView, 'c');
        }
    }
}
