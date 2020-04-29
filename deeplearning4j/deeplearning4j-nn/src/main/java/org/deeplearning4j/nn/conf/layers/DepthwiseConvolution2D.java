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
import org.deeplearning4j.nn.conf.CNN2DFormat;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.layers.convolution.DepthwiseConvolution2DLayer;
import org.deeplearning4j.nn.params.DepthwiseConvolutionParamInitializer;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.deeplearning4j.util.ConvolutionUtils;
import org.deeplearning4j.util.ValidationUtils;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.*;

/**
 * 2D depth-wise convolution layer configuration.
 * <p>
 * Performs a channels-wise convolution, which operates on each of the input maps separately. A channel multiplier is
 * used to specify the number of outputs per input map. This convolution is carried out with the specified kernel sizes,
 * stride and padding values.
 *
 * @author Max Pumperla
 */
@Data
@NoArgsConstructor
@ToString(callSuper = true)
@EqualsAndHashCode(callSuper = true)
public class DepthwiseConvolution2D extends ConvolutionLayer {

    protected int depthMultiplier;

    protected DepthwiseConvolution2D(Builder builder) {
        super(builder);
        Preconditions.checkState(builder.depthMultiplier > 0, "Depth multiplier must be > 0,  got %s", builder.depthMultiplier);
        this.depthMultiplier = builder.depthMultiplier;
        this.nOut = this.nIn * this.depthMultiplier;
        this.cnn2dDataFormat = builder.cnn2DFormat;

        initializeConstraints(builder);
    }

    @Override
    public DepthwiseConvolution2D clone() {
        DepthwiseConvolution2D clone = (DepthwiseConvolution2D) super.clone();
        clone.depthMultiplier = depthMultiplier;
        return clone;
    }


    @Override
    public Layer instantiate(NeuralNetConfiguration conf, Collection<TrainingListener> trainingListeners,
                             int layerIndex, INDArray layerParamsView, boolean initializeParams, DataType networkDataType) {
        LayerValidation.assertNInNOutSet("DepthwiseConvolution2D", getLayerName(), layerIndex, getNIn(), getNOut());

        DepthwiseConvolution2DLayer ret = new DepthwiseConvolution2DLayer(conf, networkDataType);
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
        return DepthwiseConvolutionParamInitializer.getInstance();
    }

    @Override
    public InputType getOutputType(int layerIndex, InputType inputType) {
        if (inputType == null || inputType.getType() != InputType.Type.CNN) {
            throw new IllegalStateException("Invalid input for  depth-wise convolution layer (layer name=\""
                            + getLayerName() + "\"): Expected CNN input, got " + inputType);
        }

        return InputTypeUtil.getOutputTypeCnnLayers(inputType, kernelSize, stride, padding, dilation, convolutionMode,
                        nOut, layerIndex, getLayerName(), cnn2dDataFormat, DepthwiseConvolution2DLayer.class);
    }

    @Override
    public void setNIn(InputType inputType, boolean override) {
        super.setNIn(inputType, override);

        if(nOut == 0 || override){
            nOut = this.nIn * this.depthMultiplier;
        }
        this.cnn2dDataFormat = ((InputType.InputTypeConvolutional)inputType).getFormat();
    }

    @Getter
    @Setter
    public static class Builder extends BaseConvBuilder<Builder> {

        /**
         * Set channels multiplier for depth-wise convolution
         *
         */
        protected int depthMultiplier = 1;
        protected CNN2DFormat cnn2DFormat = CNN2DFormat.NCHW;


        public Builder(int[] kernelSize, int[] stride, int[] padding) {
            super(kernelSize, stride, padding);
        }

        public Builder(int[] kernelSize, int[] stride) {
            super(kernelSize, stride);
        }

        public Builder(int... kernelSize) {
            super(kernelSize);
        }

        public Builder() {
            super();
        }

        @Override
        protected boolean allowCausal() {
            //Causal convolution - allowed for 1D only
            return false;
        }

        /**
         * Set the data format for the CNN activations - NCHW (channels first) or NHWC (channels last).
         * See {@link CNN2DFormat} for more details.<br>
         * Default: NCHW
         * @param format Format for activations (in and out)
         */
        public Builder dataFormat(CNN2DFormat format){
            this.cnn2DFormat = format;
            return this;
        }

        /**
         * Set channels multiplier for depth-wise convolution
         *
         * @param depthMultiplier integer value, for each input map we get depthMultiplier outputs in channels-wise
         * step.
         * @return Builder
         */
        public Builder depthMultiplier(int depthMultiplier) {
            this.setDepthMultiplier(depthMultiplier);
            return this;
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

        /**
         * Stride of the convolution in rows/columns (height/width) dimensions
         *
         * @param stride Stride of the layer
         */
        public Builder stride(int... stride) {
            this.setStride(stride);
            return this;
        }

        /**
         * Padding of the convolution in rows/columns (height/width) dimensions
         *
         * @param padding Padding of the layer
         */
        public Builder padding(int... padding) {
            this.setPadding(padding);
            return this;
        }

        @Override
        public void setKernelSize(int... kernelSize) {
            this.kernelSize = ValidationUtils.validate2NonNegative(kernelSize, false, "kernelSize");
        }

        @Override
        public void setStride(int... stride) {
            this.stride = ValidationUtils.validate2NonNegative(stride, false, "stride");
        }

        @Override
        public void setPadding(int... padding) {
            this.padding = ValidationUtils.validate2NonNegative(padding, false, "padding");
        }

        @Override
        public void setDilation(int... dilation) {
            this.dilation = ValidationUtils.validate2NonNegative(dilation, false, "dilation");
        }

        @Override
        @SuppressWarnings("unchecked")
        public DepthwiseConvolution2D build() {
            ConvolutionUtils.validateConvolutionModePadding(convolutionMode, padding);
            ConvolutionUtils.validateCnnKernelStridePadding(kernelSize, stride, padding);

            return new DepthwiseConvolution2D(this);
        }
    }

}
