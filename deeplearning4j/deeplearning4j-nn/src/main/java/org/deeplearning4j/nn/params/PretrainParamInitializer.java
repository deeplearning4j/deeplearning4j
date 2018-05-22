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
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
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
    public long numParams(NeuralNetConfiguration conf) {
        org.deeplearning4j.nn.conf.layers.BasePretrainNetwork layerConf =
                        (org.deeplearning4j.nn.conf.layers.BasePretrainNetwork) conf.getLayer();
        return super.numParams(conf) + layerConf.getNIn();
    }

    @Override
    public Map<String, INDArray> init(NeuralNetConfiguration conf, INDArray paramsView, boolean initializeParams) {
        Map<String, INDArray> params = super.init(conf, paramsView, initializeParams);

        org.deeplearning4j.nn.conf.layers.BasePretrainNetwork layerConf =
                        (org.deeplearning4j.nn.conf.layers.BasePretrainNetwork) conf.getLayer();
        val nIn = layerConf.getNIn();
        val nOut = layerConf.getNOut();
        val nWeightParams = nIn * nOut;

        INDArray visibleBiasView = paramsView.get(NDArrayIndex.point(0),
                        NDArrayIndex.interval(nWeightParams + nOut, nWeightParams + nOut + nIn));
        params.put(VISIBLE_BIAS_KEY, createVisibleBias(conf, visibleBiasView, initializeParams));
        conf.addVariable(VISIBLE_BIAS_KEY);

        return params;
    }

    protected INDArray createVisibleBias(NeuralNetConfiguration conf, INDArray visibleBiasView,
                    boolean initializeParameters) {
        org.deeplearning4j.nn.conf.layers.BasePretrainNetwork layerConf =
                        (org.deeplearning4j.nn.conf.layers.BasePretrainNetwork) conf.getLayer();
        if (initializeParameters) {
            INDArray ret = Nd4j.valueArrayOf(new long[]{1, layerConf.getNIn()}, layerConf.getVisibleBiasInit());
            visibleBiasView.assign(ret);
        }
        return visibleBiasView;
    }


    @Override
    public Map<String, INDArray> getGradientsFromFlattened(NeuralNetConfiguration conf, INDArray gradientView) {
        Map<String, INDArray> out = super.getGradientsFromFlattened(conf, gradientView);
        org.deeplearning4j.nn.conf.layers.FeedForwardLayer layerConf =
                        (org.deeplearning4j.nn.conf.layers.FeedForwardLayer) conf.getLayer();

        val nIn = layerConf.getNIn();
        val nOut = layerConf.getNOut();
        val nWeightParams = nIn * nOut;

        INDArray vBiasView = gradientView.get(NDArrayIndex.point(0),
                        NDArrayIndex.interval(nWeightParams + nOut, nWeightParams + nOut + nIn));

        out.put(VISIBLE_BIAS_KEY, vBiasView);

        return out;
    }
}
