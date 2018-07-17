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
import org.apache.commons.math3.util.FastMath;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.FeedForwardLayer;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.distribution.Distribution;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

import static org.deeplearning4j.nn.weights.WeightInitUtil.DEFAULT_WEIGHT_INIT_ORDER;
import static org.deeplearning4j.nn.weights.WeightInitUtil.initWeights;

/**
 * created by jingshu
 */
public class ElementWiseParamInitializer extends DefaultParamInitializer{

    private static final ElementWiseParamInitializer INSTANCE = new ElementWiseParamInitializer();

    public static ElementWiseParamInitializer getInstance() {
        return INSTANCE;
    }

    @Override
    public long numParams(Layer layer) {
        FeedForwardLayer layerConf = (FeedForwardLayer) layer;
        val nIn = layerConf.getNIn();
        return nIn*2; //weights + bias
    }

    /**
     * Initialize the parameters
     *
     * @param conf             the configuration
     * @param paramsView       a view of the full network (backprop) parameters
     * @param initializeParams if true: initialize the parameters according to the configuration. If false: don't modify the
     *                         values in the paramsView array (but do select out the appropriate subset, reshape etc as required)
     * @return Map of parameters keyed by type (view of the 'paramsView' array)
     */
    @Override
    public Map<String, INDArray> init(NeuralNetConfiguration conf, INDArray paramsView, boolean initializeParams) {
        if (!(conf.getLayer() instanceof org.deeplearning4j.nn.conf.layers.FeedForwardLayer))
            throw new IllegalArgumentException("unsupported layer type: " + conf.getLayer().getClass().getName());

        Map<String, INDArray> params = Collections.synchronizedMap(new LinkedHashMap<String, INDArray>());

        val length = numParams(conf);
        if (paramsView.length() != length)
            throw new IllegalStateException(
                    "Expected params view of length " + length + ", got length " + paramsView.length());

        org.deeplearning4j.nn.conf.layers.FeedForwardLayer layerConf =
                (org.deeplearning4j.nn.conf.layers.FeedForwardLayer) conf.getLayer();
        val nIn = layerConf.getNIn();

        val nWeightParams = nIn ;
        INDArray weightView = paramsView.get(NDArrayIndex.point(0), NDArrayIndex.interval(0, nWeightParams));
        INDArray biasView = paramsView.get(NDArrayIndex.point(0),
                NDArrayIndex.interval(nWeightParams, nWeightParams + nIn));


        params.put(WEIGHT_KEY, createWeightMatrix(conf, weightView, initializeParams));
        params.put(BIAS_KEY, createBias(conf, biasView, initializeParams));
        conf.addVariable(WEIGHT_KEY);
        conf.addVariable(BIAS_KEY);

        return params;
    }

    /**
     * Return a map of gradients (in their standard non-flattened representation), taken from the flattened (row vector) gradientView array.
     * The idea is that operates in exactly the same way as the the paramsView does in
     * thus the position in the view (and, the array orders) must match those of the parameters
     *
     * @param conf         Configuration
     * @param gradientView The flattened gradients array, as a view of the larger array
     * @return A map containing an array by parameter type, that is a view of the full network gradients array
     */
    @Override
    public Map<String, INDArray> getGradientsFromFlattened(NeuralNetConfiguration conf, INDArray gradientView) {
        org.deeplearning4j.nn.conf.layers.FeedForwardLayer layerConf =
                (org.deeplearning4j.nn.conf.layers.FeedForwardLayer) conf.getLayer();
        val nIn = layerConf.getNIn();
        val nOut = layerConf.getNOut();
        val nWeightParams = nIn ;

        INDArray weightGradientView = gradientView.get(NDArrayIndex.point(0), NDArrayIndex.interval(0, nWeightParams));
        INDArray biasView = gradientView.get(NDArrayIndex.point(0),
                NDArrayIndex.interval(nWeightParams, nWeightParams + nOut)); //Already a row vector

        Map<String, INDArray> out = new LinkedHashMap<>();
        out.put(WEIGHT_KEY, weightGradientView);
        out.put(BIAS_KEY, biasView);

        return out;
    }

    protected INDArray createWeightMatrix(long nIn, long nOut, WeightInit weightInit, Distribution dist,
                                          INDArray weightParamView, boolean initializeParameters) {
        val shape = new long[] {1,nIn};

        if (initializeParameters) {
            INDArray ret = initWeights(nIn, //Fan in
                    nOut, //Fan out
                    shape, weightInit, dist, DEFAULT_WEIGHT_INIT_ORDER, weightParamView);
            return ret;
        } else {
            return weightParamView;
        }
    }


}

