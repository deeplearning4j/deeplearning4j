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
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.Distributions;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.layers.LocallyConnected2D;
import org.deeplearning4j.nn.weights.WeightInitUtil;
import org.deeplearning4j.util.ConvolutionUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.distribution.Distribution;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.*;

/**
 * Initialize locally connected 2D parameters. A regular convolution has nIn * nOut * kernelHeight * kernelWeight
 * many parameters (excluding bias). For a locally connected layer this has to be multiplied by the number of patches
 * of each convolution, i.e. by outputHeight * outputWidth.
 * <p>
 * Internally we represent the weights of a locally connected 2D layer as follows: the weights are a 3-tensor of
 * shape: [outputHeight * outputWidth,
 * kernelHeight * kernelWidth * nIn,
 * nOut]
 * <p>
 * i.e. the first dimension determines the patch, the second is for input channels, and the last for output channels.
 *
 * @author Max Pumperla
 */
public class LocallyConnected2DParamInitializer extends ConvolutionParamInitializer {

    private static final LocallyConnected2DParamInitializer INSTANCE = new LocallyConnected2DParamInitializer();

    public static LocallyConnected2DParamInitializer getInstance() {
        return INSTANCE;
    }


    public final static String WEIGHT_KEY = DefaultParamInitializer.WEIGHT_KEY;
    public final static String BIAS_KEY = DefaultParamInitializer.BIAS_KEY;

    @Override
    public long numParams(Layer l) {
        LocallyConnected2D layerConf = (LocallyConnected2D) l;

        int nIn = (int) layerConf.getNIn();
        int nOut = (int) layerConf.getNOut();
        int[] kernel = layerConf.getKernelSize();
        int[] outputSize = layerConf.getOutputSize();

        return nIn * nOut * kernel[0] * kernel[1] * outputSize[0] * outputSize[1] + (layerConf.hasBias() ? nOut : 0);
    }

    @Override
    public Map<String, INDArray> getGradientsFromFlattened(NeuralNetConfiguration conf, INDArray gradientView) {

        LocallyConnected2D layerConf = (LocallyConnected2D) conf.getLayer();

        int[] kernel = layerConf.getKernelSize();
        int[] outputSize = layerConf.getOutputSize();
        val nIn = layerConf.getNIn();
        val nOut = layerConf.getNOut();

        Map<String, INDArray> out = new LinkedHashMap<>();
        if (layerConf.hasBias()) {
            INDArray biasGradientView = gradientView.get(NDArrayIndex.point(0), NDArrayIndex.interval(0, nOut));
            INDArray weightGradientView =
                    gradientView.get(NDArrayIndex.point(0), NDArrayIndex.interval(nOut, numParams(conf)))
                            .reshape('c', outputSize[0] * outputSize[1],
                                    nIn * kernel[0] * kernel[1], nOut);
            out.put(BIAS_KEY, biasGradientView);
            out.put(WEIGHT_KEY, weightGradientView);
        } else {
            INDArray weightGradientView = gradientView
                    .reshape('c', outputSize[0] * outputSize[1],
                            nIn * kernel[0] * kernel[1], nOut);
            out.put(WEIGHT_KEY, weightGradientView);
        }
        return out;
    }

    protected INDArray createBias(NeuralNetConfiguration conf, INDArray biasView, boolean initializeParams) {
        LocallyConnected2D layerConf = (LocallyConnected2D) conf.getLayer();
        if (initializeParams)
            biasView.assign(layerConf.getBiasInit());
        return biasView;
    }


    protected INDArray createWeightMatrix(NeuralNetConfiguration conf, INDArray weightView, boolean initializeParams) {
        /** Create a 3d weight matrix in c order of shape:
         ([outputHeight * outputWidth, kernelHeight * kernelWidth * nIn, nOut]
         Inputs to the convolution layer are:
         (batch size, num input feature maps, image height, image width)
         */
        LocallyConnected2D layerConf = (LocallyConnected2D) conf.getLayer();

        int[] kernel = layerConf.getKernelSize();
        int[] stride = layerConf.getStride();
        int[] outputSize = layerConf.getOutputSize();

        val inputDepth = layerConf.getNIn();
        val outputDepth = layerConf.getNOut();

        val weightsShape = new long[]{
                outputSize[0] * outputSize[1],
                inputDepth * kernel[0] * kernel[1],
                outputDepth};

        if (initializeParams) {
            Distribution dist = Distributions.createDistribution(layerConf.getDist());

            double fanIn = inputDepth * kernel[0] * kernel[1];
            double fanOut = outputDepth * kernel[0] * kernel[1] / ((double) stride[0] * stride[1]);

            return WeightInitUtil.initWeights(fanIn, fanOut, weightsShape, layerConf.getWeightInit(), dist, 'c',
                    weightView);
        } else {
            return WeightInitUtil.reshapeWeights(weightsShape, weightView, 'c');
        }
    }
}
