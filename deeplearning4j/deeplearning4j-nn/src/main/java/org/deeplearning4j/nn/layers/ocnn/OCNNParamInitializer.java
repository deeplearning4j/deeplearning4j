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

package org.deeplearning4j.nn.layers.ocnn;

import static  org.nd4j.linalg.indexing.NDArrayIndex.*;

import lombok.val;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.Distributions;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.nn.weights.WeightInitUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.distribution.Distribution;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.*;

/**
 * Param initializer for {@link OCNNOutputLayer}
 *
 * @author Adam Gibson
 */
public class OCNNParamInitializer extends DefaultParamInitializer {

    private final static OCNNParamInitializer INSTANCE = new OCNNParamInitializer();


    public final static String NU_KEY = "nu";
    public final static String K_KEY = "k";

    public final static String V_KEY = "v";
    public final static String W_KEY = "w";

    public final static String R_KEY = "r";


    private final static List<String> WEIGHT_KEYS = Arrays.asList(W_KEY,V_KEY,R_KEY);
    private final static List<String> PARAM_KEYS = Arrays.asList(W_KEY,V_KEY,R_KEY);

    public static OCNNParamInitializer getInstance() {
        return INSTANCE;
    }

    @Override
    public long numParams(NeuralNetConfiguration conf) {
        return numParams(conf.getLayer());
    }


    @Override
    public long numParams(Layer layer) {
        org.deeplearning4j.nn.conf.ocnn.OCNNOutputLayer ocnnOutputLayer = ( org.deeplearning4j.nn.conf.ocnn.OCNNOutputLayer) layer;
        val nIn = ocnnOutputLayer.getNIn();
        val hiddenLayer = ocnnOutputLayer.getHiddenSize();

        val firstLayerWeightLength =  hiddenLayer;
        val secondLayerLength = nIn * hiddenLayer;
        val rLength = 1;
        return firstLayerWeightLength + secondLayerLength + rLength;
    }

    @Override
    public List<String> paramKeys(Layer layer) {
        return PARAM_KEYS;
    }

    @Override
    public List<String> weightKeys(Layer layer) {
        return WEIGHT_KEYS;
    }

    @Override
    public List<String> biasKeys(Layer layer) {
        return Collections.emptyList();
    }

    @Override
    public boolean isWeightParam(Layer layer, String key) {
        return WEIGHT_KEYS.contains(key);
    }

    @Override
    public boolean isBiasParam(Layer layer, String key) {
        return false;
    }

    @Override
    public Map<String, INDArray> init(NeuralNetConfiguration conf, INDArray paramsView, boolean initializeParams) {
        org.deeplearning4j.nn.conf.ocnn.OCNNOutputLayer ocnnOutputLayer = ( org.deeplearning4j.nn.conf.ocnn.OCNNOutputLayer) conf.getLayer();
        Map<String, INDArray> params = Collections.synchronizedMap(new LinkedHashMap<String, INDArray>());
        val nIn = ocnnOutputLayer.getNIn();
        int hiddenLayer = ocnnOutputLayer.getHiddenSize();

        val firstLayerWeightLength =  hiddenLayer;
        val secondLayerLength = nIn * hiddenLayer;
        int rLength = 1;
        INDArray weightView = paramsView.get(point(0),interval(0, firstLayerWeightLength))
                .reshape(1,hiddenLayer);
        INDArray weightsTwoView = paramsView.get(point(0),
                NDArrayIndex.interval(firstLayerWeightLength,
                        firstLayerWeightLength + secondLayerLength))
                .reshape('f',nIn,hiddenLayer);
        INDArray rView = paramsView.get(point(0),point(paramsView.length() - rLength));


        INDArray paramViewPut = createWeightMatrix(conf, weightView, initializeParams);
        params.put(W_KEY, paramViewPut);
        conf.addVariable(W_KEY);
        INDArray paramIvewPutTwo = createWeightMatrix(conf,weightsTwoView,initializeParams);
        params.put(V_KEY,paramIvewPutTwo);
        conf.addVariable(V_KEY);
        INDArray rViewPut = createWeightMatrix(conf,rView,initializeParams);
        params.put(R_KEY,rViewPut);
        conf.addVariable(R_KEY);

        return params;
    }

    @Override
    public Map<String, INDArray> getGradientsFromFlattened(NeuralNetConfiguration conf, INDArray gradientView) {
        org.deeplearning4j.nn.conf.ocnn.OCNNOutputLayer ocnnOutputLayer = ( org.deeplearning4j.nn.conf.ocnn.OCNNOutputLayer) conf.getLayer();
        Map<String, INDArray> params = Collections.synchronizedMap(new LinkedHashMap<String, INDArray>());
        val nIn = ocnnOutputLayer.getNIn();
        val hiddenLayer = ocnnOutputLayer.getHiddenSize();

        val firstLayerWeightLength =  hiddenLayer;
        val secondLayerLength = nIn * hiddenLayer;

        INDArray weightView = gradientView.get(point(0),interval(0, firstLayerWeightLength))
                .reshape('f',1,hiddenLayer);
        INDArray vView = gradientView.get(point(0),
                NDArrayIndex.interval(firstLayerWeightLength,firstLayerWeightLength + secondLayerLength))
                .reshape('f',nIn,hiddenLayer);
        params.put(W_KEY, weightView);
        params.put(V_KEY,vView);
        params.put(R_KEY,gradientView.get(point(0),point(gradientView.length() - 1)));
        return params;

    }


    protected INDArray createWeightMatrix(NeuralNetConfiguration configuration,
                                          INDArray weightParamView,
                                          boolean initializeParameters) {

        org.deeplearning4j.nn.conf.ocnn.OCNNOutputLayer ocnnOutputLayer = ( org.deeplearning4j.nn.conf.ocnn.OCNNOutputLayer) configuration.getLayer();
        WeightInit weightInit = ocnnOutputLayer.getWeightInit();
        Distribution dist = Distributions.createDistribution(ocnnOutputLayer.getDist());
        if (initializeParameters) {
            INDArray ret = WeightInitUtil.initWeights(weightParamView.size(0), //Fan in
                    weightParamView.size(1), //Fan out
                    weightParamView.shape(), weightInit, dist, weightParamView);
            return ret;
        } else {
            return WeightInitUtil.reshapeWeights(weightParamView.shape(), weightParamView);
        }
    }
}
