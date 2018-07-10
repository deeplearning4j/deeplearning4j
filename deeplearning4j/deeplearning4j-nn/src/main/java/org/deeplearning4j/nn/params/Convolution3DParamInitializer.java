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
import org.deeplearning4j.nn.conf.layers.Convolution3D;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.weights.WeightInitUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.distribution.Distribution;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.*;

/**
 * Initialize 3D convolution parameters.
 *
 * @author Adam Gibson
 */
public class Convolution3DParamInitializer extends ConvolutionParamInitializer {

    private static final Convolution3DParamInitializer INSTANCE = new Convolution3DParamInitializer();

    public static Convolution3DParamInitializer getInstance() {
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
        Convolution3D layerConf =
                (Convolution3D) l;

        int[] kernel = layerConf.getKernelSize();
        val nIn = layerConf.getNIn();
        val nOut = layerConf.getNOut();
        return nIn * nOut * kernel[0] * kernel[1] * kernel[2] + (layerConf.hasBias() ? nOut : 0);
    }


    @Override
    public Map<String, INDArray> init(NeuralNetConfiguration conf, INDArray paramsView, boolean initializeParams) {
        Convolution3D layer = (Convolution3D) conf.getLayer();
        if (layer.getKernelSize().length != 3) throw new IllegalArgumentException("Filter size must be == 3");

        Map<String, INDArray> params = Collections.synchronizedMap(new LinkedHashMap<String, INDArray>());

        Convolution3D layerConf = (Convolution3D) conf.getLayer();
        val nOut = layerConf.getNOut();

        if (layer.hasBias()) {
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

        Convolution3D layerConf = (Convolution3D) conf.getLayer();

        int[] kernel = layerConf.getKernelSize();
        val nIn = layerConf.getNIn();
        val nOut = layerConf.getNOut();

        Map<String, INDArray> out = new LinkedHashMap<>();
        if (layerConf.hasBias()) {
            INDArray biasGradientView = gradientView.get(NDArrayIndex.point(0), NDArrayIndex.interval(0, nOut));
            INDArray weightGradientView =
                    gradientView.get(NDArrayIndex.point(0), NDArrayIndex.interval(nOut, numParams(conf)))
                            .reshape('c', nOut, nIn, kernel[0], kernel[1], kernel[2]);
            out.put(BIAS_KEY, biasGradientView);
            out.put(WEIGHT_KEY, weightGradientView);
        } else {
            INDArray weightGradientView = gradientView.reshape('c', nOut, nIn, kernel[0], kernel[1], kernel[2]);
            out.put(WEIGHT_KEY, weightGradientView);
        }
        return out;
    }


    protected INDArray createWeightMatrix(NeuralNetConfiguration conf, INDArray weightView, boolean initializeParams) {
        /*
         Create a 5d weight matrix of:
           (number of kernels, num input channels, kernel depth, kernel height, kernel width)
         Note c order is used specifically for the CNN weights, as opposed to f order elsewhere
         Inputs to the convolution layer are:
         (batch size, num input feature maps, image depth, image height, image width)
         */
        Convolution3D layerConf = (Convolution3D) conf.getLayer();

        if (initializeParams) {
            Distribution dist = Distributions.createDistribution(layerConf.getDist());
            int[] kernel = layerConf.getKernelSize();
            int[] stride = layerConf.getStride();

            val inputDepth = layerConf.getNIn();
            val outputDepth = layerConf.getNOut();

            double fanIn = inputDepth * kernel[0] * kernel[1] * kernel[2];
            double fanOut = outputDepth * kernel[0] * kernel[1] * kernel[2] /
                    ((double) stride[0] * stride[1] * stride[2]);

            val weightsShape = new long[]{outputDepth, inputDepth, kernel[0], kernel[1], kernel[2]};

            return WeightInitUtil.initWeights(fanIn, fanOut, weightsShape, layerConf.getWeightInit(), dist, 'c',
                    weightView);
        } else {
            int[] kernel = layerConf.getKernelSize();
            return WeightInitUtil.reshapeWeights(
                    new long[]{layerConf.getNOut(), layerConf.getNIn(), kernel[0], kernel[1], kernel[2]},
                    weightView, 'c');
        }
    }
}
