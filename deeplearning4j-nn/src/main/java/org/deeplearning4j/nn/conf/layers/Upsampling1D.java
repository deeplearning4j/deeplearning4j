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
import lombok.ToString;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.nn.conf.memory.MemoryReport;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Collection;
import java.util.Map;

/**
 * Upsampling 1D layer
 *
 * @author Max Pumperla
 */

@Data
@NoArgsConstructor
@ToString(callSuper = true)
@EqualsAndHashCode(callSuper = true)
public class Upsampling1D extends BaseUpsamplingLayer {

    protected int[] size;

    protected Upsampling1D(UpsamplingBuilder builder) {
        super(builder);
        this.size = builder.size;
    }

    @Override
    public org.deeplearning4j.nn.api.Layer instantiate(NeuralNetConfiguration conf,
                                                       Collection<TrainingListener> trainingListeners, int layerIndex, INDArray layerParamsView,
                                                       boolean initializeParams) {
        org.deeplearning4j.nn.layers.convolution.upsampling.Upsampling1D ret =
                new org.deeplearning4j.nn.layers.convolution.upsampling.Upsampling1D(conf);
        ret.setListeners(trainingListeners);
        ret.setIndex(layerIndex);
        ret.setParamsViewArray(layerParamsView);
        Map<String, INDArray> paramTable = initializer().init(conf, layerParamsView, initializeParams);
        ret.setParamTable(paramTable);
        ret.setConf(conf);
        return ret;
    }

    @Override
    public Upsampling1D clone() {
        Upsampling1D clone = (Upsampling1D) super.clone();
        return clone;
    }

    @Override
    public InputType getOutputType(int layerIndex, InputType inputType) {
        if (inputType == null || inputType.getType() != InputType.Type.RNN) {
            throw new IllegalStateException("Invalid input for 1D Upsampling layer (layer index = " + layerIndex
                    + ", layer name = \"" + getLayerName() + "\"): expect RNN input type with size > 0. Got: "
                    + inputType);
        }
        InputType.InputTypeRecurrent recurrent = (InputType.InputTypeRecurrent) inputType;
        return InputType.recurrent(recurrent.getSize(), recurrent.getTimeSeriesLength());
    }

    @Override
    public InputPreProcessor getPreProcessorForInputType(InputType inputType) {
        if (inputType == null) {
            throw new IllegalStateException("Invalid input for Upsampling layer (layer name=\"" + getLayerName()
                    + "\"): input is null");
        }
        return InputTypeUtil.getPreProcessorForInputTypeCnnLayers(inputType, getLayerName());
    }

    @Override
    public LayerMemoryReport getMemoryReport(InputType inputType) {
        InputType.InputTypeRecurrent recurrent = (InputType.InputTypeRecurrent) inputType;
        InputType.InputTypeRecurrent outputType = (InputType.InputTypeRecurrent) getOutputType(-1, inputType);

        int im2colSizePerEx = recurrent.getSize() * outputType.getTimeSeriesLength() * size[0];
        int trainingWorkingSizePerEx = im2colSizePerEx;
        if (getIDropout() != null) {
            trainingWorkingSizePerEx += inputType.arrayElementsPerExample();
        }

        return new LayerMemoryReport.Builder(layerName, Upsampling1D.class, inputType, outputType)
                .standardMemory(0, 0) //No params
                .workingMemory(0, im2colSizePerEx, 0, trainingWorkingSizePerEx)
                .cacheMemory(MemoryReport.CACHE_MODE_ALL_ZEROS, MemoryReport.CACHE_MODE_ALL_ZEROS) //No caching
                .build();
    }

    @NoArgsConstructor
    public static class Builder extends UpsamplingBuilder<Builder> {

        public Builder(int size) {
            super(new int[] {size});
        }

        /**
         * Upsampling size int
         *
         * @param size    upsampling size in single spatial dimension of this 1D layer
         */
        public Builder size(int size) {

            this.size = new int[] {size};
            return this;
        }

        /**
         * Upsampling size int array with a single element
         *
         * @param size    upsampling size in single spatial dimension of this 1D layer
         */
        public Builder size(int[] size) {
            Preconditions.checkArgument(size.length == 1);
            this.size = size;
            return this;
        }

        @Override
        @SuppressWarnings("unchecked")
        public Upsampling1D build() {
            return new Upsampling1D(this);
        }
    }

}
