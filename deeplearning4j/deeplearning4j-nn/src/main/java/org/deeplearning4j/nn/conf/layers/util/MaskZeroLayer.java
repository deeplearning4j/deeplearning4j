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

package org.deeplearning4j.nn.conf.layers.util;

import lombok.Data;
import lombok.NoArgsConstructor;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.layers.wrapper.BaseWrapperLayer;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Collection;

/**
 * Wrapper which masks timesteps with activation equal to the specified masking value (0.0 default).
 * Assumes that the input shape is [batch_size, input_size, timesteps].
 * @author Martin Boyanov mboyanov@gmail.com
 */
@Data
public class MaskZeroLayer extends BaseWrapperLayer {

    private double maskingValue = 0.0;

    private static final long serialVersionUID = 9074525846200921839L;

    public MaskZeroLayer(Builder builder) {
        super(builder);
        this.underlying = builder.underlying;
        this.maskingValue = builder.maskValue;
    }


    public MaskZeroLayer(Layer underlying, double maskingValue) {
        this.underlying = underlying;
        this.maskingValue = maskingValue;
    }


    @Override
    public org.deeplearning4j.nn.api.Layer instantiate(NeuralNetConfiguration conf, Collection<TrainingListener> trainingListeners,
                                                       int layerIndex, INDArray layerParamsView, boolean initializeParams) {

        NeuralNetConfiguration conf2 = conf.clone();
        conf2.setLayer(((BaseWrapperLayer)conf2.getLayer()).getUnderlying());

        org.deeplearning4j.nn.api.Layer underlyingLayer =
                underlying.instantiate(conf2, trainingListeners, layerIndex, layerParamsView, initializeParams);
        return new org.deeplearning4j.nn.layers.recurrent.MaskZeroLayer(underlyingLayer, maskingValue);
    }

    @Override
    public InputType getOutputType(int layerIndex, InputType inputType) {
        return underlying.getOutputType(layerIndex, inputType);
    }

    @Override
    public void setNIn(InputType inputType, boolean override) {
        underlying.setNIn(inputType, override);
    }

    @Override
    public InputPreProcessor getPreProcessorForInputType(InputType inputType) {
        return underlying.getPreProcessorForInputType(inputType);    //No op
    }

    @Override
    public double getL1ByParam(String paramName) {
        return underlying.getL1ByParam(paramName);   //No params
    }

    @Override
    public double getL2ByParam(String paramName) {
        return underlying.getL2ByParam(paramName);   //No params
    }

    @Override
    public boolean isPretrainParam(String paramName) {
        return false;
    }

    @Override
    public LayerMemoryReport getMemoryReport(InputType inputType) {
        return underlying.getMemoryReport(inputType);
    }

    @Override
    public String toString(){
        return "MaskZeroLayer(" + underlying.toString() + ")";
    }


    @NoArgsConstructor
    public static class Builder extends Layer.Builder<Builder> {

        private Layer underlying;
        private double maskValue;

        public Builder setUnderlying(Layer underlying) {
            this.underlying = underlying;
            return this;
        }

        public Builder setMaskValue(double maskValue) {
            this.maskValue = maskValue;
            return this;
        }

        @Override
        @SuppressWarnings("unchecked")
        public MaskZeroLayer build() {
            return new MaskZeroLayer(this);
        }
    }

}