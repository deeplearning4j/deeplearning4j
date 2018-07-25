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

package org.deeplearning4j.nn.conf.layers;

import lombok.*;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.nn.conf.memory.MemoryReport;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Collection;
import java.util.Map;

/** Parametrized Rectified Linear Unit (PReLU)
 *
 * f(x) = alpha * x for x < 0, f(x) = x for x >= 0
 *
 * alpha has the same shape as x and is a learned parameter.
 *
 * @author Max Pumperla
 */
@Data
@NoArgsConstructor
@ToString(callSuper = true)
@EqualsAndHashCode(callSuper = true)
public class PReLULayer extends FeedForwardLayer {

    private long[] inputShape = null;

    private PReLULayer(Builder builder) {
        super(builder);
        this.inputShape = builder.inputShape;
        this.nOut = inputShape[0]; // mostly here for layer validation
        this.nIn = inputShape[0];
        initializeConstraints(builder);
    }

    @Override
    public Layer instantiate(NeuralNetConfiguration conf, Collection<TrainingListener> trainingListeners,
                    int layerIndex, INDArray layerParamsView, boolean initializeParams) {
        LayerValidation.assertNInNOutSet("PReLULayer", getLayerName(), layerIndex, getNIn(), getNOut());

        org.deeplearning4j.nn.layers.feedforward.dense.DenseLayer ret =
                        new org.deeplearning4j.nn.layers.feedforward.dense.DenseLayer(conf);
        ret.setListeners(trainingListeners);
        ret.setIndex(layerIndex);
        ret.setParamsViewArray(layerParamsView);
        Map<String, INDArray> paramTable = initializer().init(conf, layerParamsView, initializeParams);
        ret.setParamTable(paramTable);
        ret.setConf(conf);
        return ret;
    }

    @Override
    public InputType getOutputType(int layerIndex, InputType inputType) {
        if (inputType == null)
            throw new IllegalStateException("Invalid input type: null for layer name \"" + getLayerName() + "\"");
        return inputType;
    }

    @Override
    public ParamInitializer initializer() {
        return DefaultParamInitializer.getInstance();
    }

    @Override
    public LayerMemoryReport getMemoryReport(InputType inputType) {
        InputType outputType = getOutputType(-1, inputType);

        val numParams = initializer().numParams(this);
        val updaterStateSize = (int) getIUpdater().stateSize(numParams);

        return new LayerMemoryReport.Builder(layerName, PReLULayer.class, inputType, outputType)
                        .standardMemory(numParams, updaterStateSize)
                        .workingMemory(0, 0, 0, 0)
                        .cacheMemory(MemoryReport.CACHE_MODE_ALL_ZEROS, MemoryReport.CACHE_MODE_ALL_ZEROS)
                        .build();
    }

    @NoArgsConstructor
    public static class Builder extends FeedForwardLayer.Builder<PReLULayer.Builder> {

        private long[] inputShape = null;


        /**
         * Explicitly set input shape of incoming activations so that parameters
         * can be initialized properly.
         *
         * @param shape
         * @return
         */
        public Builder inputShape(long... shape){
            this.inputShape = shape;
            return this;
        }

        @Override
        @SuppressWarnings("unchecked")
        public PReLULayer build() {
            return new PReLULayer(this);
        }
    }

}
