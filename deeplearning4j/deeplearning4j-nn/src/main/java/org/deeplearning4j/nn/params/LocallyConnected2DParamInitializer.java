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
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.weights.WeightInitUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.distribution.Distribution;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.*;

/**
 * Initialize locally connected 2D parameters.
 *
 * @author Max Pumperla
 */
public class LocallyConnected2DParamInitializer implements ParamInitializer {

    private static final LocallyConnected2DParamInitializer INSTANCE = new LocallyConnected2DParamInitializer();

    public static LocallyConnected2DParamInitializer getInstance() {
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
        ConvolutionLayer layerConf =
                        (ConvolutionLayer) l;

        int[] kernel = layerConf.getKernelSize();
        val nIn = layerConf.getNIn();
        val nOut = layerConf.getNOut();
        return nIn * nOut * kernel[0] * kernel[1] + (layerConf.hasBias() ? nOut : 0);
    }

    @Override
    public List<String> paramKeys(Layer layer) {
        ConvolutionLayer layerConf =
                (ConvolutionLayer) layer;
        if(layerConf.hasBias()){
            return Arrays.asList(WEIGHT_KEY, BIAS_KEY);
        } else {
            return weightKeys(layer);
        }
    }

    @Override
    public List<String> weightKeys(Layer layer) {
        return Collections.singletonList(WEIGHT_KEY);
    }

    @Override
    public List<String> biasKeys(Layer layer) {
        ConvolutionLayer layerConf =
                (ConvolutionLayer) layer;
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
        ConvolutionLayer layer = (ConvolutionLayer) conf.getLayer();
        if (layer.getKernelSize().length != 2) throw new IllegalArgumentException("Filter size must be == 2");

        Map<String, INDArray> params = Collections.synchronizedMap(new LinkedHashMap<String, INDArray>());

        ConvolutionLayer layerConf =
                        (ConvolutionLayer) conf.getLayer();

        val nOut = layerConf.getNOut();

        if(layer.hasBias()){
            //Standard case
            INDArray biasView = paramsView.get(NDArrayIndex.point(0), NDArrayIndex.interval(0, nOut));
            INDArray weightView = paramsView.get(NDArrayIndex.point(0), NDArrayIndex.interval(nOut, numParams(conf)));
            params.put(BIAS_KEY, createBias(conf, biasView, initializeParams));
            params.put(WEIGHT_KEY, createWeightMatrix(conf, weightView, initializeParams));
            conf.addVariable(WEIGHT_KEY);
            conf.addVariable(BIAS_KEY);
        } else {
            INDArray weightView = paramsView;
            params.put(WEIGHT_KEY, createWeightMatrix(conf, weightView, initializeParams));
            conf.addVariable(WEIGHT_KEY);
        }

        return params;
    }

    @Override
    public Map<String, INDArray> getGradientsFromFlattened(NeuralNetConfiguration conf, INDArray gradientView) {

        ConvolutionLayer layerConf =
                        (ConvolutionLayer) conf.getLayer();

        int[] kernel = layerConf.getKernelSize();
        val nIn = layerConf.getNIn();
        val nOut = layerConf.getNOut();

        Map<String, INDArray> out = new LinkedHashMap<>();
        if(layerConf.hasBias()){
            //Standard case
            INDArray biasGradientView = gradientView.get(NDArrayIndex.point(0), NDArrayIndex.interval(0, nOut));
            INDArray weightGradientView =
                    gradientView.get(NDArrayIndex.point(0), NDArrayIndex.interval(nOut, numParams(conf)))
                            .reshape('c', nOut, nIn, kernel[0], kernel[1]);
            out.put(BIAS_KEY, biasGradientView);
            out.put(WEIGHT_KEY, weightGradientView);
        } else {
            INDArray weightGradientView = gradientView.reshape('c', nOut, nIn, kernel[0], kernel[1]);
            out.put(WEIGHT_KEY, weightGradientView);
        }
        return out;
    }

    //1 bias per feature map
    protected INDArray createBias(NeuralNetConfiguration conf, INDArray biasView, boolean initializeParams) {
        //the bias is a 1D tensor -- one bias per output feature map
        ConvolutionLayer layerConf =
                        (ConvolutionLayer) conf.getLayer();
        if (initializeParams)
            biasView.assign(layerConf.getBiasInit());
        return biasView;
    }


    protected INDArray createWeightMatrix(NeuralNetConfiguration conf, INDArray weightView, boolean initializeParams) {
        /*
         Create a 4d weight matrix of:
           (number of kernels, num input channels, kernel height, kernel width)
         Note c order is used specifically for the CNN weights, as opposed to f order elsewhere
         Inputs to the convolution layer are:
         (batch size, num input feature maps, image height, image width)
         */
        ConvolutionLayer layerConf =
                        (ConvolutionLayer) conf.getLayer();
        if (initializeParams) {
            Distribution dist = Distributions.createDistribution(layerConf.getDist());
            int[] kernel = layerConf.getKernelSize();
            int[] stride = layerConf.getStride();

            val inputDepth = layerConf.getNIn();
            val outputDepth = layerConf.getNOut();

            double fanIn = inputDepth * kernel[0] * kernel[1];
            double fanOut = outputDepth * kernel[0] * kernel[1] / ((double) stride[0] * stride[1]);

            val weightsShape = new long[] {outputDepth, inputDepth, kernel[0], kernel[1]};

            return WeightInitUtil.initWeights(fanIn, fanOut, weightsShape, layerConf.getWeightInit(), dist, 'c',
                            weightView);
        } else {
            int[] kernel = layerConf.getKernelSize();
            return WeightInitUtil.reshapeWeights(
                            new long[] {layerConf.getNOut(), layerConf.getNIn(), kernel[0], kernel[1]}, weightView, 'c');
        }
    }
}
