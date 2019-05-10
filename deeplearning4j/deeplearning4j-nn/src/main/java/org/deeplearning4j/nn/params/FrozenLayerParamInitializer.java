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

import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.layers.misc.FrozenLayer;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Collections;
import java.util.List;
import java.util.Map;

/**
 * Parameter initializer for {@link FrozenLayer} instances. Relies on underlying layer's param initializer.
 *
 * @author Alex Black
 */
public class FrozenLayerParamInitializer implements ParamInitializer {

    private static final FrozenLayerParamInitializer INSTANCE = new FrozenLayerParamInitializer();

    public static FrozenLayerParamInitializer getInstance() {
        return INSTANCE;
    }

    @Override
    public long numParams(NeuralNetConfiguration conf) {
        return numParams(conf.getLayer());
    }

    @Override
    public long numParams(Layer layer) {
        FrozenLayer fl = (FrozenLayer) layer;
        ParamInitializer initializer = fl.getLayer().initializer();
        return initializer.numParams(fl.getLayer());
    }

    @Override
    public List<String> paramKeys(Layer layer) {
        return Collections.emptyList();
    }

    @Override
    public List<String> weightKeys(Layer layer) {
        return Collections.emptyList();
    }

    @Override
    public List<String> biasKeys(Layer layer) {
        return Collections.emptyList();
    }

    @Override
    public boolean isWeightParam(Layer layer, String key) {
        return false;
    }

    @Override
    public boolean isBiasParam(Layer layer, String key) {
        return false;
    }

    @Override
    public Map<String, INDArray> init(NeuralNetConfiguration conf, INDArray paramsView, boolean initializeParams) {
        FrozenLayer fl = (FrozenLayer) conf.getLayer();
        Layer innerLayer = fl.getLayer();
        ParamInitializer initializer = innerLayer.initializer();
        conf.setLayer(innerLayer);
        Map<String, INDArray> m = initializer.init(conf, paramsView, initializeParams);
        conf.setLayer(fl);

        return m;
    }

    @Override
    public Map<String, INDArray> getGradientsFromFlattened(NeuralNetConfiguration conf, INDArray gradientView) {
        FrozenLayer fl = (FrozenLayer) conf.getLayer();
        Layer innerLayer = fl.getLayer();
        ParamInitializer initializer = innerLayer.initializer();
        conf.setLayer(innerLayer);
        Map<String, INDArray> m = initializer.getGradientsFromFlattened(conf, gradientView);
        conf.setLayer(fl);
        return m;
    }
}
