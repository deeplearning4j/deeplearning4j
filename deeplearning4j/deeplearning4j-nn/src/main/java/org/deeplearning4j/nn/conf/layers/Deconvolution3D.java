/* ******************************************************************************
 * Copyright (c) 2019 Konduit K.K.
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
import lombok.ToString;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.layers.convolution.Deconvolution3DLayer;
import org.deeplearning4j.nn.params.Deconvolution3DParamInitializer;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.deeplearning4j.util.ValidationUtils;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Collection;
import java.util.Map;

/**
 * 3D deconvolution layer configuration<br>
 *
 * Deconvolutions are also known as transpose convolutions or fractionally strided convolutions. In essence,
 * deconvolutions swap forward and backward pass with regular 3D convolutions.
 *
 * See the paper by Matt Zeiler for details: <a href="http://www.matthewzeiler.com/wp-content/uploads/2017/07/cvpr2010.pdf">http://www.matthewzeiler.com/wp-content/uploads/2017/07/cvpr2010.pdf</a>
 *
 * For an intuitive guide to convolution arithmetic and shapes, see:
 * <a href="https://arxiv.org/abs/1603.07285v1">https://arxiv.org/abs/1603.07285v1</a>
 *
 * @author Alex Black
 */
@Data
@NoArgsConstructor
@ToString(callSuper = true)
@EqualsAndHashCode(callSuper = true)
public class Deconvolution3D extends ConvolutionLayer {

    private Convolution3D.DataFormat dataFormat = Convolution3D.DataFormat.NCDHW; // in libnd4j: 1 - NCDHW, 0 - NDHWC

    /**
     * Deconvolution3D layer nIn in the input layer is the number of channels nOut is the number of filters to be used
     * in the net or in other words the channels The builder specifies the filter/kernel size, the stride and padding
     * The pooling layer takes the kernel size
     */
    protected Deconvolution3D(Builder builder) {
        super(builder);
        this.dataFormat = builder.dataFormat;
        initializeConstraints(builder);
    }

    public boolean hasBias() {
        return hasBias;
    }

    @Override
    public Deconvolution3D clone() {
        Deconvolution3D clone = (Deconvolution3D) super.clone();
        if (clone.kernelSize != null) {
            clone.kernelSize = clone.kernelSize.clone();
        }
        if (clone.stride != null) {
            clone.stride = clone.stride.clone();
        }
        if (clone.padding != null) {
            clone.padding = clone.padding.clone();
        }
        return clone;
    }

    @Override
    public Layer instantiate(NeuralNetConfiguration conf, Collection<TrainingListener> trainingListeners,
                             int layerIndex, INDArray layerParamsView, boolean initializeParams, DataType networkDataType) {
        LayerValidation.assertNInNOutSet("Deconvolution2D", getLayerName(), layerIndex, getNIn(), getNOut());

        Deconvolution3DLayer ret =
                        new Deconvolution3DLayer(conf, networkDataType);
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
        return Deconvolution3DParamInitializer.getInstance();
    }

    @Override
    public InputPreProcessor getPreProcessorForInputType(InputType inputType) {
        if (inputType == null) {
            throw new IllegalStateException("Invalid input for Deconvolution3D layer (layer name=\"" + getLayerName() + "\"): input is null");
        }

        return InputTypeUtil.getPreProcessorForInputTypeCnn3DLayers(inputType, getLayerName());
    }

    @Override
    public void setNIn(InputType inputType, boolean override) {
        if (inputType == null || inputType.getType() != InputType.Type.CNN3D) {
            throw new IllegalStateException("Invalid input for Deconvolution 3D layer (layer name=\"" + getLayerName() + "\"): Expected CNN3D input, got " + inputType);
        }

        if (nIn <= 0 || override) {
            InputType.InputTypeConvolutional3D c = (InputType.InputTypeConvolutional3D) inputType;
            this.nIn = c.getChannels();
        }
    }

    @Override
    public InputType getOutputType(int layerIndex, InputType inputType) {
        if (inputType == null || inputType.getType() != InputType.Type.CNN3D) {
            throw new IllegalStateException("Invalid input for Deconvolution layer (layer name=\"" + getLayerName()
                            + "\"): Expected CNN input, got " + inputType);
        }

        return InputTypeUtil.getOutputTypeDeconv3dLayer(inputType, kernelSize, stride, padding, dilation, convolutionMode,
                        dataFormat, nOut, layerIndex, getLayerName(), Deconvolution3DLayer.class);
    }

    public static class Builder extends BaseConvBuilder<Builder> {

        private Convolution3D.DataFormat dataFormat = Convolution3D.DataFormat.NCDHW; // in libnd4j: 1 - NCDHW, 0 - NDHWC

        public Builder() {
            super(new int[] {2, 2, 2}, new int[] {1, 1, 1}, new int[] {0, 0, 0}, new int[] {1, 1, 1}, 3);
        }

        @Override
        protected boolean allowCausal() {
            //Causal convolution - allowed for 1D only
            return false;
        }

        /**
         * Set the convolution mode for the Convolution layer. See {@link ConvolutionMode} for more details
         *
         * @param convolutionMode Convolution mode for layer
         */
        public Builder convolutionMode(ConvolutionMode convolutionMode) {
            return super.convolutionMode(convolutionMode);
        }

        /**
         * Size of the convolution rows/columns
         *
         * @param kernelSize the height and width of the kernel
         */
        public Builder kernelSize(int... kernelSize) {
            this.setKernelSize(kernelSize);
            return this;
        }

        public Builder stride(int... stride) {
            this.setStride(stride);
            return this;
        }

        public Builder padding(int... padding) {
            this.setPadding(padding);
            return this;
        }

        @Override
        public void setKernelSize(int... kernelSize) {
            this.kernelSize = ValidationUtils.validate3NonNegative(kernelSize, "kernelSize");
        }

        @Override
        public void setStride(int... stride) {
            this.stride = ValidationUtils.validate3NonNegative(stride, "stride");
        }

        @Override
        public void setPadding(int... padding) {
            this.padding = ValidationUtils.validate3NonNegative(padding, "padding");
        }

        @Override
        public void setDilation(int... dilation) {
            this.dilation = ValidationUtils.validate3NonNegative(dilation, "dilation");
        }

        public Builder dataFormat(Convolution3D.DataFormat dataFormat){
            this.dataFormat = dataFormat;
            return this;
        }

        @Override
        public Deconvolution3D build() {
            return new Deconvolution3D(this);
        }
    }

}
