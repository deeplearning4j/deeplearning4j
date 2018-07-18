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
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Initialize Center Loss params.
 *
 * @author Justin Long (@crockpotveggies)
 * @author Alex Black (@AlexDBlack)
 */
public class CenterLossParamInitializer extends DefaultParamInitializer {

    private static final CenterLossParamInitializer INSTANCE = new CenterLossParamInitializer();

    public static CenterLossParamInitializer getInstance() {
        return INSTANCE;
    }

    public final static String WEIGHT_KEY = DefaultParamInitializer.WEIGHT_KEY;
    public final static String BIAS_KEY = DefaultParamInitializer.BIAS_KEY;
    public final static String CENTER_KEY = "cL";

    @Override
    public long numParams(NeuralNetConfiguration conf) {
        org.deeplearning4j.nn.conf.layers.FeedForwardLayer layerConf =
                        (org.deeplearning4j.nn.conf.layers.FeedForwardLayer) conf.getLayer();
        val nIn = layerConf.getNIn();
        val nOut = layerConf.getNOut(); // also equal to numClasses
        return nIn * nOut + nOut + nIn * nOut; //weights + bias + embeddings
    }

    @Override
    public Map<String, INDArray> init(NeuralNetConfiguration conf, INDArray paramsView, boolean initializeParams) {
        Map<String, INDArray> params = Collections.synchronizedMap(new LinkedHashMap<String, INDArray>());

        org.deeplearning4j.nn.conf.layers.CenterLossOutputLayer layerConf =
                        (org.deeplearning4j.nn.conf.layers.CenterLossOutputLayer) conf.getLayer();

        val nIn = layerConf.getNIn();
        val nOut = layerConf.getNOut(); // also equal to numClasses

        val wEndOffset = nIn * nOut;
        val bEndOffset = wEndOffset + nOut;
        val cEndOffset = bEndOffset + nIn * nOut;

        INDArray weightView = paramsView.get(NDArrayIndex.point(0), NDArrayIndex.interval(0, wEndOffset));
        INDArray biasView = paramsView.get(NDArrayIndex.point(0), NDArrayIndex.interval(wEndOffset, bEndOffset));
        INDArray centerLossView = paramsView.get(NDArrayIndex.point(0), NDArrayIndex.interval(bEndOffset, cEndOffset))
                        .reshape('c', nOut, nIn);

        params.put(WEIGHT_KEY, createWeightMatrix(conf, weightView, initializeParams));
        params.put(BIAS_KEY, createBias(conf, biasView, initializeParams));
        params.put(CENTER_KEY, createCenterLossMatrix(conf, centerLossView, initializeParams));
        conf.addVariable(WEIGHT_KEY);
        conf.addVariable(BIAS_KEY);
        conf.addVariable(CENTER_KEY);

        return params;
    }

    @Override
    public Map<String, INDArray> getGradientsFromFlattened(NeuralNetConfiguration conf, INDArray gradientView) {
        org.deeplearning4j.nn.conf.layers.CenterLossOutputLayer layerConf =
                        (org.deeplearning4j.nn.conf.layers.CenterLossOutputLayer) conf.getLayer();

        val nIn = layerConf.getNIn();
        val nOut = layerConf.getNOut(); // also equal to numClasses

        val wEndOffset = nIn * nOut;
        val bEndOffset = wEndOffset + nOut;
        val cEndOffset = bEndOffset + nIn * nOut; // note: numClasses == nOut

        INDArray weightGradientView = gradientView.get(NDArrayIndex.point(0), NDArrayIndex.interval(0, wEndOffset))
                        .reshape('f', nIn, nOut);
        INDArray biasView = gradientView.get(NDArrayIndex.point(0), NDArrayIndex.interval(wEndOffset, bEndOffset)); //Already a row vector
        INDArray centerLossView = gradientView.get(NDArrayIndex.point(0), NDArrayIndex.interval(bEndOffset, cEndOffset))
                        .reshape('c', nOut, nIn);

        Map<String, INDArray> out = new LinkedHashMap<>();
        out.put(WEIGHT_KEY, weightGradientView);
        out.put(BIAS_KEY, biasView);
        out.put(CENTER_KEY, centerLossView);

        return out;
    }


    protected INDArray createCenterLossMatrix(NeuralNetConfiguration conf, INDArray centerLossView,
                    boolean initializeParameters) {
        org.deeplearning4j.nn.conf.layers.CenterLossOutputLayer layerConf =
                        (org.deeplearning4j.nn.conf.layers.CenterLossOutputLayer) conf.getLayer();

        if (initializeParameters) {
            centerLossView.assign(0.0);
        }
        return centerLossView;
    }
}
