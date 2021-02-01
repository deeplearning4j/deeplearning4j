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
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.layers.convolution.Convolution3DLayer;
import org.deeplearning4j.nn.params.Convolution3DParamInitializer;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.deeplearning4j.util.Convolution3DUtils;
import org.deeplearning4j.util.ConvolutionUtils;
import org.deeplearning4j.util.ValidationUtils;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Collection;
import java.util.Map;

/**
 * 3D convolution layer configuration
 *
 * @author Max Pumperla
 */
@Data
@NoArgsConstructor
@ToString(callSuper = true)
@EqualsAndHashCode(callSuper = true)
public class Convolution3D extends ConvolutionLayer {

    /**
     * An optional dataFormat: "NDHWC" or "NCDHW". Defaults to "NCDHW".<br> The data format of the input and output
     * data. <br> For "NCDHW" (also known as 'channels first' format), the data storage order is: [batchSize,
     * inputChannels, inputDepth, inputHeight, inputWidth].<br> For "NDHWC" ('channels last' format), the data is stored
     * in the order of: [batchSize, inputDepth, inputHeight, inputWidth, inputChannels].
     */
    public enum DataFormat {
        NCDHW, NDHWC
    }

    private ConvolutionMode mode = ConvolutionMode.Same; // in libnd4j: 0 - same mode, 1 - valid mode
    private DataFormat dataFormat = DataFormat.NCDHW; // in libnd4j: 1 - NCDHW, 0 - NDHWC

    /**
     * 3-dimensional convolutional layer configuration nIn in the input layer is the number of channels nOut is the
     * number of filters to be used in the net or in other words the depth The builder specifies the filter/kernel size,
     * the stride and padding The pooling layer takes the kernel size
     */
    public Convolution3D(Builder builder) {
        super(builder);
        this.dataFormat = builder.dataFormat;
        this.convolutionMode = builder.convolutionMode;
    }

    public boolean hasBias() {
        return hasBias;
    }


    @Override
    public Convolution3D clone() {
        Convolution3D clone = (Convolution3D) super.clone();
        if (clone.kernelSize != null) {
            clone.kernelSize = clone.kernelSize.clone();
        }
        if (clone.stride != null) {
            clone.stride = clone.stride.clone();
        }
        if (clone.padding != null) {
            clone.padding = clone.padding.clone();
        }
        if (clone.dilation != null) {
            clone.dilation = clone.dilation.clone();
        }
        return clone;
    }

    @Override
    public Layer instantiate(NeuralNetConfiguration conf, Collection<TrainingListener> iterationListeners,
                             int layerIndex, INDArray layerParamsView, boolean initializeParams, DataType networkDataType) {
        LayerValidation.assertNInNOutSet("Convolution3D", getLayerName(), layerIndex, getNIn(), getNOut());

        Convolution3DLayer ret = new Convolution3DLayer(conf, networkDataType);
        ret.setListeners(iterationListeners);
        ret.setIndex(layerIndex);
        ret.setParamsViewArray(layerParamsView);
        Map<String, INDArray> paramTable = initializer().init(conf, layerParamsView, initializeParams);
        ret.setParamTable(paramTable);
        ret.setConf(conf);
        return ret;
    }

    @Override
    public ParamInitializer initializer() {
        return Convolution3DParamInitializer.getInstance();
    }

    @Override
    public InputType getOutputType(int layerIndex, InputType inputType) {
        if (inputType == null || inputType.getType() != InputType.Type.CNN3D) {
            throw new IllegalStateException("Invalid input for Convolution3D layer (layer name=\"" + getLayerName()
                            + "\"): Expected CNN3D input, got " + inputType);
        }
        return InputTypeUtil.getOutputTypeCnn3DLayers(inputType, dataFormat, kernelSize, stride, padding, dilation, convolutionMode,
                        nOut, layerIndex, getLayerName(), Convolution3DLayer.class);
    }

    @Override
    public InputPreProcessor getPreProcessorForInputType(InputType inputType) {
        if (inputType == null) {
            throw new IllegalStateException("Invalid input for Convolution3D layer (layer name=\"" + getLayerName()
                            + "\"): input is null");
        }

        return InputTypeUtil.getPreProcessorForInputTypeCnn3DLayers(inputType, getLayerName());
    }


    @Override
    public void setNIn(InputType inputType, boolean override) {
        if (inputType == null || inputType.getType() != InputType.Type.CNN3D) {
            throw new IllegalStateException("Invalid input for Convolution 3D layer (layer name=\"" + getLayerName()
                            + "\"): Expected CNN3D input, got " + inputType);
        }

        if (nIn <= 0 || override) {
            InputType.InputTypeConvolutional3D c = (InputType.InputTypeConvolutional3D) inputType;
            this.nIn = c.getChannels();
        }
    }

    @AllArgsConstructor
    @Getter
    @Setter
    public static class Builder extends ConvolutionLayer.BaseConvBuilder<Builder> {

        /**
         * The data format for input and output activations.<br> NCDHW: activations (in/out) should have shape
         * [minibatch, channels, depth, height, width]<br> NDHWC: activations (in/out) should have shape [minibatch,
         * depth, height, width, channels]<br>
         */
        private DataFormat dataFormat = DataFormat.NCDHW;

        public Builder() {
            super(new int[] {2, 2, 2}, new int[] {1, 1, 1}, new int[] {0, 0, 0}, new int[] {1, 1, 1}, 3);
        }

        @Override
        protected boolean allowCausal() {
            //Causal convolution - allowed for 1D only
            return false;
        }

        public Builder(int[] kernelSize, int[] stride, int[] padding, int[] dilation) {
            super(kernelSize, stride, padding, dilation, 3);
        }

        public Builder(int[] kernelSize, int[] stride, int[] padding) {
            this(kernelSize, stride, padding, new int[] {1, 1, 1});
        }

        public Builder(int[] kernelSize, int[] stride) {
            this(kernelSize, stride, new int[] {0, 0, 0});
        }

        public Builder(int... kernelSize) {
            this(kernelSize, new int[] {1, 1, 1});
        }

        /**
         * Set kernel size for 3D convolutions in (depth, height, width) order
         *
         * @param kernelSize kernel size
         * @return 3D convolution layer builder
         */
        public Builder kernelSize(int... kernelSize) {
            this.setKernelSize(kernelSize);
            return this;
        }

        /**
         * Set stride size for 3D convolutions in (depth, height, width) order
         *
         * @param stride kernel size
         * @return 3D convolution layer builder
         */
        public Builder stride(int... stride) {
            this.setStride(stride);
            return this;
        }

        /**
         * Set padding size for 3D convolutions in (depth, height, width) order
         *
         * @param padding kernel size
         * @return 3D convolution layer builder
         */
        public Builder padding(int... padding) {
            this.setPadding(padding);
            return this;
        }

        /**
         * Set dilation size for 3D convolutions in (depth, height, width) order
         *
         * @param dilation kernel size
         * @return 3D convolution layer builder
         */
        public Builder dilation(int... dilation) {
            this.setDilation(dilation);
            return this;
        }

        public Builder convolutionMode(ConvolutionMode mode) {
            this.setConvolutionMode(mode);
            return this;
        }

        /**
         * The data format for input and output activations.<br> NCDHW: activations (in/out) should have shape
         * [minibatch, channels, depth, height, width]<br> NDHWC: activations (in/out) should have shape [minibatch,
         * depth, height, width, channels]<br>
         *
         * @param dataFormat Data format to use for activations
         */
        public Builder dataFormat(DataFormat dataFormat) {
            this.setDataFormat(dataFormat);
            return this;
        }

        /**
         * Set kernel size for 3D convolutions in (depth, height, width) order
         *
         * @param kernelSize kernel size
         */
        @Override
        public void setKernelSize(int... kernelSize) {
            this.kernelSize = ValidationUtils.validate3NonNegative(kernelSize, "kernelSize");
        }

        /**
         * Set stride size for 3D convolutions in (depth, height, width) order
         *
         * @param stride kernel size
         */
        @Override
        public void setStride(int... stride) {
            this.stride = ValidationUtils.validate3NonNegative(stride, "stride");
        }

        /**
         * Set padding size for 3D convolutions in (depth, height, width) order
         *
         * @param padding kernel size
         */
        @Override
        public void setPadding(int... padding) {
            this.padding = ValidationUtils.validate3NonNegative(padding, "padding");
        }

        /**
         * Set dilation size for 3D convolutions in (depth, height, width) order
         *
         * @param dilation kernel size
         */
        @Override
        public void setDilation(int... dilation) {
            this.dilation = ValidationUtils.validate3NonNegative(dilation, "dilation");
        }



        @Override
        @SuppressWarnings("unchecked")
        public Convolution3D build() {
            ConvolutionUtils.validateConvolutionModePadding(convolutionMode, padding);
            Convolution3DUtils.validateCnn3DKernelStridePadding(kernelSize, stride, padding);

            return new Convolution3D(this);
        }
    }

}
