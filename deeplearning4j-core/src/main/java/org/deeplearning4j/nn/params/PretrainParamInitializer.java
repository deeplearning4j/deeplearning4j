/*
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
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Map;

/**
 * Pretrain weight initializer.
 * Has the visible bias as well as hidden and weight matrix.
 *
 * @author Adam Gibson
 */
public class PretrainParamInitializer extends DefaultParamInitializer {
    public final static String VISIBLE_BIAS_KEY = DefaultParamInitializer.BIAS_KEY + "B";

    @Override
    public void init(Map<String, INDArray> params, NeuralNetConfiguration conf) {
        super.init(params, conf);
        org.deeplearning4j.nn.conf.layers.BasePretrainNetwork layerConf =
                (org.deeplearning4j.nn.conf.layers.BasePretrainNetwork) conf.getLayer();

        params.put(VISIBLE_BIAS_KEY, Nd4j.valueArrayOf(layerConf.getNIn(),0.0));
        conf.addVariable(VISIBLE_BIAS_KEY);
        params.get(VISIBLE_BIAS_KEY).data().persist();
    }


}
