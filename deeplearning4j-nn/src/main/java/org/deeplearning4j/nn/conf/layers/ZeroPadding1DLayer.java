/*-
 *
 *  * Copyright 2017 Skymind,Inc.
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
package org.deeplearning4j.nn.conf.layers;

import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.nn.conf.memory.MemoryReport;
import org.deeplearning4j.nn.params.EmptyParamInitializer;
import org.deeplearning4j.optimize.api.IterationListener;
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
public class ZeroPadding1DLayer extends Layer {

    private int[] padding; // [padLeft, padRight]

    private ZeroPadding1DLayer(Builder builder) {
        super(builder);
        this.padding = builder.padding;
    }

    @Override
    public org.deeplearning4j.nn.api.Layer instantiate(Collection<IterationListener> iterationListeners,
                                                       String name, int layerIndex, int numInputs, INDArray layerParamsView,
                                                       boolean initializeParams) {
        org.deeplearning4j.nn.layers.convolution.ZeroPadding1DLayer ret =
                new org.deeplearning4j.nn.layers.convolution.ZeroPadding1DLayer(this);
        ret.setIndex(layerIndex);
        Map<String, INDArray> paramTable = initializer().init(this, layerParamsView, initializeParams);
        ret.setParamTable(paramTable);
        ret.setConf(this);
        return ret;
    }

    @Override
    public ParamInitializer initializer() {
        return EmptyParamInitializer.getInstance();
    }

    @Override
    public InputType[] getOutputType(int layerIndex, InputType... inputType) {
        if (inputType == null || inputType.length != 1 || inputType[0].getType() != InputType.Type.RNN) {
            throw new IllegalStateException("Invalid input for 1D CNN layer (layer index = " + layerIndex
                    + ", layer name = \"" + getLayerName() + "\"): expect RNN input type with size > 0. Got: "
                    + inputType);
        }
        InputType.InputTypeRecurrent recurrent = (InputType.InputTypeRecurrent) inputType[0];
        return new InputType[]{InputType.recurrent(recurrent.getSize(),
                recurrent.getTimeSeriesLength() + padding[0] + padding[1])};
    }

    @Override
    public void setNIn(InputType[] inputType, boolean override) {
        //No op
    }

    @Override
    public InputPreProcessor getPreProcessorForInputType(InputType... inputType) {
        if (inputType == null || inputType.length != 1) {
            throw new IllegalStateException("Invalid input for ZeroPadding1DLayer (layer name = \"" + getLayerName()
                    + "\"): input type should be length 1 (got: " + (inputType == null ? null : Arrays.toString(inputType)) + ")");
        }

        return InputTypeUtil.getPreprocessorForInputTypeRnnLayers(inputType[0], getLayerName());
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
    public LayerMemoryReport getMemoryReport(InputType... inputTypes) {
        if (inputTypes == null || inputTypes.length != 1) {
            throw new IllegalArgumentException("Expected 1 input type: got " + (inputTypes == null ? null : Arrays.toString(inputTypes)));
        }
        InputType inputType = inputTypes[0];
        InputType outputType = getOutputType(-1, inputType)[0];

        return new LayerMemoryReport.Builder(layerName, ZeroPaddingLayer.class, inputType, outputType)
                .standardMemory(0, 0) //No params
                .workingMemory(0, 0, MemoryReport.CACHE_MODE_ALL_ZEROS, MemoryReport.CACHE_MODE_ALL_ZEROS)
                .cacheMemory(MemoryReport.CACHE_MODE_ALL_ZEROS, MemoryReport.CACHE_MODE_ALL_ZEROS) //No caching
                .build();
    }

    public static class Builder extends Layer.Builder<Builder> {

        private int[] padding = new int[]{0, 0}; //Padding: left, right

        /**
         * @param padding Padding for both the left and right
         */
        public Builder(int padding) {
            this(padding, padding);
        }

        public Builder(int padLeft, int padRight) {
            this(new int[]{padLeft, padRight});
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
