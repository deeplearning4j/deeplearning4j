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
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.nn.conf.memory.MemoryReport;
import org.deeplearning4j.nn.params.EmptyParamInitializer;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Collection;
import java.util.Map;

/**
 * Activation layer is a simple layer that applies the specified activation function to the input activations
 */
@Data
@NoArgsConstructor
@ToString(callSuper = true)
@EqualsAndHashCode(callSuper = true)
public class ActivationLayer extends NoParamLayer {

    protected IActivation activationFn;

    protected ActivationLayer(Builder builder) {
        super(builder);
        this.activationFn = builder.activationFn;
        initializeConstraints(builder);
    }

    /**
     * @param activation Activation function for the layer
     */
    public ActivationLayer(Activation activation){
        this(new Builder().activation(activation));
    }

    /**
     * @param activationFn Activation function for the layer
     */
    public ActivationLayer(IActivation activationFn){
        this(new Builder().activation(activationFn));
    }

    @Override
    public ActivationLayer clone() {
        ActivationLayer clone = (ActivationLayer) super.clone();
        return clone;
    }

    @Override
    public Layer instantiate(NeuralNetConfiguration conf, Collection<TrainingListener> trainingListeners,
                    int layerIndex, INDArray layerParamsView, boolean initializeParams) {
        org.deeplearning4j.nn.layers.ActivationLayer ret = new org.deeplearning4j.nn.layers.ActivationLayer(conf);
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
        return EmptyParamInitializer.getInstance();
    }

    @Override
    public InputType getOutputType(int layerIndex, InputType inputType) {
        if (inputType == null)
            throw new IllegalStateException("Invalid input type: null for layer name \"" + getLayerName() + "\"");
        return inputType;
    }

    @Override
    public InputPreProcessor getPreProcessorForInputType(InputType inputType) {
        //No input preprocessor required for any input
        return null;
    }

    @Override
    public LayerMemoryReport getMemoryReport(InputType inputType) {
        val actElementsPerEx = inputType.arrayElementsPerExample();

        return new LayerMemoryReport.Builder(layerName, ActivationLayer.class, inputType, inputType)
                        .standardMemory(0, 0) //No params
                        //During inference: modify input activation in-place
                        //During  backprop: dup the input for later re-use
                        .workingMemory(0, 0, 0, actElementsPerEx)
                        .cacheMemory(MemoryReport.CACHE_MODE_ALL_ZEROS, MemoryReport.CACHE_MODE_ALL_ZEROS) //No caching
                        .build();
    }

    @Override
    public void setNIn(InputType inputType, boolean override) {
        //No op
    }

    @AllArgsConstructor
    @NoArgsConstructor
    public static class Builder extends org.deeplearning4j.nn.conf.layers.Layer.Builder<Builder> {

        private IActivation activationFn = null;

        /**
         * Layer activation function.
         * Typical values include:<br>
         * "relu" (rectified linear), "tanh", "sigmoid", "softmax",
         * "hardtanh", "leakyrelu", "maxout", "softsign", "softplus"
         * @deprecated Use {@link #activation(Activation)} or {@link @activation(IActivation)}
         */
        @Deprecated
        public Builder activation(String activationFunction) {
            return activation(Activation.fromString(activationFunction));
        }

        /**
         * @param activationFunction Activation function for the layer
         */
        public Builder activation(IActivation activationFunction) {
            this.activationFn = activationFunction;
            return this;
        }

        /**
         * @param activation Activation function for the layer
         */
        public Builder activation(Activation activation) {
            return activation(activation.getActivationFunction());
        }

        @Override
        @SuppressWarnings("unchecked")
        public ActivationLayer build() {
            return new ActivationLayer(this);
        }
    }
}
