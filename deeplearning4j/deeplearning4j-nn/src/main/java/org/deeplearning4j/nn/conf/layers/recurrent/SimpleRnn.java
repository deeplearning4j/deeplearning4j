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

package org.deeplearning4j.nn.conf.layers.recurrent;

import lombok.Data;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.BaseRecurrentLayer;
import org.deeplearning4j.nn.conf.layers.LayerValidation;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.nn.params.SimpleRnnParamInitializer;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Collection;
import java.util.Map;

/**
 * Simple RNN - aka "vanilla" RNN is the simplest type of recurrent neural network layer.
 * It implements {@code out_t = activationFn( in_t * inWeight + out_(t-1) * recurrentWeights + bias)}.
 *
 * Note that other architectures (LSTM, etc) are usually much more effective, especially for longer time series;
 * however SimpleRnn is very fast to compute, and hence may be considered where the length of the temporal dependencies
 * in the dataset are only a few steps long.
 *
 * @author Alex Black
 */
@Data
public class SimpleRnn extends BaseRecurrentLayer {

    protected SimpleRnn(Builder builder){
        super(builder);
    }

    private SimpleRnn(){

    }

    @Override
    public Layer instantiate(NeuralNetConfiguration conf, Collection<TrainingListener> trainingListeners,
                             int layerIndex, INDArray layerParamsView, boolean initializeParams) {
        LayerValidation.assertNInNOutSet("SimpleRnn", getLayerName(), layerIndex, getNIn(), getNOut());

        org.deeplearning4j.nn.layers.recurrent.SimpleRnn ret =
                new org.deeplearning4j.nn.layers.recurrent.SimpleRnn(conf);
        ret.setListeners(trainingListeners);
        ret.setIndex(layerIndex);
        ret.setParamsViewArray(layerParamsView);
        Map<String, INDArray> paramTable = initializer().init(conf, layerParamsView, initializeParams);
        ret.setParamTable(paramTable);
        ret.setConf(conf);
        return ret;
    }

    @Override
    public ParamInitializer initializer() {
        return SimpleRnnParamInitializer.getInstance();
    }

    @Override
    public double getL1ByParam(String paramName){
        switch (paramName){
            case SimpleRnnParamInitializer.WEIGHT_KEY:
            case SimpleRnnParamInitializer.RECURRENT_WEIGHT_KEY:
                return l1;
            case SimpleRnnParamInitializer.BIAS_KEY:
                return l1Bias;
            default:
                throw new IllegalStateException("Unknown parameter: \"" + paramName + "\"");
        }
    }

    @Override
    public double getL2ByParam(String paramName){
        switch (paramName){
            case SimpleRnnParamInitializer.WEIGHT_KEY:
            case SimpleRnnParamInitializer.RECURRENT_WEIGHT_KEY:
                return l2;
            case SimpleRnnParamInitializer.BIAS_KEY:
                return l2Bias;
            default:
                throw new IllegalStateException("Unknown parameter: \"" + paramName + "\"");
        }
    }

    @Override
    public LayerMemoryReport getMemoryReport(InputType inputType) {
        return null;
    }

    public static class Builder extends BaseRecurrentLayer.Builder<Builder>{


        @Override
        public SimpleRnn build() {
            return new SimpleRnn(this);
        }
    }
}
