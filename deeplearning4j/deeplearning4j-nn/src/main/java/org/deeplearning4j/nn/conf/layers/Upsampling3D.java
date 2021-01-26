/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.nn.conf.layers;

import lombok.*;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.nn.conf.memory.MemoryReport;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.deeplearning4j.util.ValidationUtils;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Collection;
import java.util.Map;

/**
 * Upsampling 3D layer<br> Repeats each value (all channel values for each x/y/z location) by size[0], size[1] and
 * size[2]<br> If input has shape {@code [minibatch, channels, depth, height, width]} then output has shape {@code
 * [minibatch, channels, size[0] * depth, size[1] * height, size[2] * width]}
 *
 * @author Max Pumperla
 */
@Data
@NoArgsConstructor
@ToString(callSuper = true)
@EqualsAndHashCode(callSuper = true)
public class Upsampling3D extends BaseUpsamplingLayer {

    protected int[] size;
    protected Convolution3D.DataFormat dataFormat = Convolution3D.DataFormat.NCDHW; //Default to NCDHW for 1.0.0-beta4 and earlier, when no config existed (NCDHW only)



    protected Upsampling3D(Builder builder) {
        super(builder);
        this.size = builder.size;
        this.dataFormat = builder.dataFormat;
    }

    @Override
    public Upsampling3D clone() {
        return (Upsampling3D) super.clone();
    }

    @Override
    public org.deeplearning4j.nn.api.Layer instantiate(NeuralNetConfiguration conf,
                                                       Collection<TrainingListener> iterationListeners, int layerIndex, INDArray layerParamsView,
                                                       boolean initializeParams, DataType networkDataType) {
        org.deeplearning4j.nn.layers.convolution.upsampling.Upsampling3D ret =
                        new org.deeplearning4j.nn.layers.convolution.upsampling.Upsampling3D(conf, networkDataType);
        ret.setListeners(iterationListeners);
        ret.setIndex(layerIndex);
        ret.setParamsViewArray(layerParamsView);
        Map<String, INDArray> paramTable = initializer().init(conf, layerParamsView, initializeParams);
        ret.setParamTable(paramTable);
        ret.setConf(conf);
        return ret;
    }

    @Override
    public InputType getOutputType(int layerIndex, InputType inputType) {
        if (inputType == null || inputType.getType() != InputType.Type.CNN3D) {
            throw new IllegalStateException("Invalid input for Upsampling 3D layer (layer name=\"" + getLayerName()
                            + "\"): Expected CNN3D input, got " + inputType);
        }
        InputType.InputTypeConvolutional3D i = (InputType.InputTypeConvolutional3D) inputType;

        long inHeight = (int) i.getHeight();
        long inWidth = (int) i.getWidth();
        long inDepth = (int) i.getDepth();
        long inChannels = (int) i.getChannels();

        return InputType.convolutional3D(size[0] * inDepth, size[1] * inHeight, size[2] * inWidth, inChannels);
    }

    @Override
    public InputPreProcessor getPreProcessorForInputType(InputType inputType) {
        if (inputType == null) {
            throw new IllegalStateException("Invalid input for Upsampling 3D layer (layer name=\"" + getLayerName()
                            + "\"): input is null");
        }
        return InputTypeUtil.getPreProcessorForInputTypeCnn3DLayers(inputType, getLayerName());
    }

    @Override
    public LayerMemoryReport getMemoryReport(InputType inputType) {
        InputType.InputTypeConvolutional3D c = (InputType.InputTypeConvolutional3D) inputType;
        InputType.InputTypeConvolutional3D outputType =
                        (InputType.InputTypeConvolutional3D) getOutputType(-1, inputType);

        // During forward pass: im2col array + reduce. Reduce is counted as activations, so only im2col is working mem
        val im2colSizePerEx = c.getChannels() & outputType.getDepth() * outputType.getHeight() * outputType.getWidth()
                        * size[0] * size[1] * size[2];

        // Current implementation does NOT cache im2col etc... which means: it's recalculated on each backward pass
        long trainingWorkingSizePerEx = im2colSizePerEx;
        if (getIDropout() != null) {
            //Dup on the input before dropout, but only for training
            trainingWorkingSizePerEx += inputType.arrayElementsPerExample();
        }

        return new LayerMemoryReport.Builder(layerName, Upsampling3D.class, inputType, outputType).standardMemory(0, 0) //No params
                        .workingMemory(0, im2colSizePerEx, 0, trainingWorkingSizePerEx)
                        .cacheMemory(MemoryReport.CACHE_MODE_ALL_ZEROS, MemoryReport.CACHE_MODE_ALL_ZEROS) //No caching
                        .build();
    }


    @NoArgsConstructor
    public static class Builder extends UpsamplingBuilder<Builder> {

        protected Convolution3D.DataFormat dataFormat = Convolution3D.DataFormat.NCDHW;

        /**
         * @param size Upsampling layer size (most common value: 2)
         */
        public Builder(int size) {
            super(new int[] {size, size, size});
        }

        /**
         * @param dataFormat Data format - see {@link Convolution3D.DataFormat} for more details
         * @param size Upsampling layer size (most common value: 2)
         */
        public Builder(@NonNull Convolution3D.DataFormat dataFormat, int size){
            super(new int[]{size, size, size});
            this.dataFormat = dataFormat;
        }

        /**
         * Sets the DataFormat. See {@link Convolution3D.DataFormat} for more details
         */
        public Builder dataFormat(@NonNull Convolution3D.DataFormat dataFormat){
            this.dataFormat = dataFormat;
            return this;
        }

        /**
         * Upsampling size as int, so same upsampling size is used for depth, width and height
         *
         * @param size upsampling size in height, width and depth dimensions
         */
        public Builder size(int size) {

            this.setSize(new int[] {size, size, size});
            return this;
        }

        /**
         * Upsampling size as int, so same upsampling size is used for depth, width and height
         *
         * @param size upsampling size in height, width and depth dimensions
         */
        public Builder size(int[] size) {
            Preconditions.checkArgument(size.length == 3);
            this.setSize(size);
            return this;
        }

        @Override
        @SuppressWarnings("unchecked")
        public Upsampling3D build() {
            return new Upsampling3D(this);
        }

        @Override
        public void setSize(int... size) {
            this.size = ValidationUtils.validate3NonNegative(size, "size");
        }
    }

}
