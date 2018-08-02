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

import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.nn.conf.memory.MemoryReport;
import org.deeplearning4j.nn.params.EmptyParamInitializer;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Arrays;
import java.util.Collection;
import java.util.Map;

/**
 * Zero padding 1D layer for convolutional neural networks.
 * Allows padding to be done separately for top and bottom.
 *
 * @author Max Pumperla
 */
@Data
@NoArgsConstructor
@EqualsAndHashCode(callSuper = true)
public class ZeroPadding1DLayer extends NoParamLayer {

    private int[] padding; // [padLeft, padRight]

    private ZeroPadding1DLayer(Builder builder) {
        super(builder);
        this.padding = builder.padding;
    }

    public ZeroPadding1DLayer(int padding){
        this(new Builder(padding));
    }

    public ZeroPadding1DLayer(int padLeft, int padRight){
        this(new Builder(padLeft, padRight));
    }

    public ZeroPadding1DLayer(int[] padding){
        this(new Builder(padding));
    }

    @Override
    public org.deeplearning4j.nn.api.Layer instantiate(NeuralNetConfiguration conf,
                                                       Collection<TrainingListener> trainingListeners, int layerIndex, INDArray layerParamsView,
                                                       boolean initializeParams) {
        org.deeplearning4j.nn.layers.convolution.ZeroPadding1DLayer ret =
                new org.deeplearning4j.nn.layers.convolution.ZeroPadding1DLayer(conf);
        ret.setListeners(trainingListeners);
        ret.setIndex(layerIndex);
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
        if (inputType == null || inputType.getType() != InputType.Type.RNN) {
            throw new IllegalStateException("Invalid input for 1D CNN layer (layer index = " + layerIndex
                    + ", layer name = \"" + getLayerName() + "\"): expect RNN input type with size > 0. Got: "
                    + inputType);
        }
        InputType.InputTypeRecurrent recurrent = (InputType.InputTypeRecurrent) inputType;
        return InputType.recurrent(recurrent.getSize(),
                recurrent.getTimeSeriesLength() + padding[0] + padding[1]);
    }

    @Override
    public void setNIn(InputType inputType, boolean override) {
        //No op
    }

    @Override
    public InputPreProcessor getPreProcessorForInputType(InputType inputType) {
        if (inputType == null) {
            throw new IllegalStateException("Invalid input for ZeroPadding1DLayer layer (layer name=\"" + getLayerName()
                    + "\"): input is null");
        }

        return InputTypeUtil.getPreprocessorForInputTypeRnnLayers(inputType, getLayerName());
    }

    @Override
    public double getL1ByParam(String paramName) {
        return 0;
    }

    @Override
    public double getL2ByParam(String paramName) {
        return 0;
    }

    @Override
    public boolean isPretrainParam(String paramName) {
        throw new UnsupportedOperationException("ZeroPaddingLayer does not contain parameters");
    }

    @Override
    public LayerMemoryReport getMemoryReport(InputType inputType) {
        InputType outputType = getOutputType(-1, inputType);

        return new LayerMemoryReport.Builder(layerName, ZeroPaddingLayer.class, inputType, outputType)
                .standardMemory(0, 0) //No params
                .workingMemory(0, 0, MemoryReport.CACHE_MODE_ALL_ZEROS, MemoryReport.CACHE_MODE_ALL_ZEROS)
                .cacheMemory(MemoryReport.CACHE_MODE_ALL_ZEROS, MemoryReport.CACHE_MODE_ALL_ZEROS) //No caching
                .build();
    }

    public static class Builder extends Layer.Builder<Builder> {

        private int[] padding = new int[] {0, 0}; //Padding: left, right

        /**
         * @param padding  Padding for both the left and right
         */
        public Builder(int padding) {
            this(padding, padding);
        }

        public Builder(int padLeft, int padRight) {
            this(new int[] {padLeft, padRight});
        }

        public Builder(int[] padding) {
            this.padding = padding;
        }

        @Override
        @SuppressWarnings("unchecked")
        public ZeroPadding1DLayer build() {
            for (int p : padding) {
                if (p < 0) {
                    throw new IllegalStateException(
                            "Invalid zero padding layer config: padding [left, right]"
                                    + " must be > 0 for all elements. Got: "
                                    + Arrays.toString(padding));
                }
            }
            return new ZeroPadding1DLayer(this);
        }
    }
}
