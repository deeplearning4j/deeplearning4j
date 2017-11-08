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

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.Map;

/**
 * Pretrain weight initializer.
 * Has the visible bias as well as hidden and weight matrix.
 *
 * @author Adam Gibson
 */
public class PretrainParamInitializer extends DefaultParamInitializer {

    private static final PretrainParamInitializer INSTANCE = new PretrainParamInitializer();

    public static PretrainParamInitializer getInstance() {
        return INSTANCE;
    }

    public final static String VISIBLE_BIAS_KEY = "v" + DefaultParamInitializer.BIAS_KEY;

    @Override
    public int numParams(Layer layer) {
        org.deeplearning4j.nn.conf.layers.BasePretrainNetwork layerConf =
                        (org.deeplearning4j.nn.conf.layers.BasePretrainNetwork) layer;
        return super.numParams(layer) + layerConf.getNIn();
    }

    @Override
    public Map<String, INDArray> init(Layer layer, INDArray paramsView, boolean initializeParams) {
        Map<String, INDArray> params = super.init(layer, paramsView, initializeParams);

        org.deeplearning4j.nn.conf.layers.BasePretrainNetwork layerConf =
                        (org.deeplearning4j.nn.conf.layers.BasePretrainNetwork) layer;
        int nIn = layerConf.getNIn();
        int nOut = layerConf.getNOut();
        int nWeightParams = nIn * nOut;

        INDArray visibleBiasView = paramsView.get(NDArrayIndex.point(0),
                        NDArrayIndex.interval(nWeightParams + nOut, nWeightParams + nOut + nIn));
        params.put(VISIBLE_BIAS_KEY, createVisibleBias(layer, visibleBiasView, initializeParams));

        return params;
    }

    protected INDArray createVisibleBias(Layer layer, INDArray visibleBiasView,
                    boolean initializeParameters) {
        org.deeplearning4j.nn.conf.layers.BasePretrainNetwork layerConf =
                        (org.deeplearning4j.nn.conf.layers.BasePretrainNetwork) layer;
        if (initializeParameters) {
            INDArray ret = Nd4j.valueArrayOf(layerConf.getNIn(), layerConf.getVisibleBiasInit());
            visibleBiasView.assign(ret);
        }
        return visibleBiasView;
    }


    @Override
    public Map<String, INDArray> getGradientsFromFlattened(Layer layer, INDArray gradientView) {
        Map<String, INDArray> out = super.getGradientsFromFlattened(layer, gradientView);
        org.deeplearning4j.nn.conf.layers.FeedForwardLayer layerConf =
                        (org.deeplearning4j.nn.conf.layers.FeedForwardLayer) layer;

        int nIn = layerConf.getNIn();
        int nOut = layerConf.getNOut();
        int nWeightParams = nIn * nOut;

        INDArray vBiasView = gradientView.get(NDArrayIndex.point(0),
                        NDArrayIndex.interval(nWeightParams + nOut, nWeightParams + nOut + nIn));

        out.put(VISIBLE_BIAS_KEY, vBiasView);

        return out;
    }
}
