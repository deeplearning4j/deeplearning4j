/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
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
import org.deeplearning4j.nn.api.layers.LayerConstraint;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.RNNFormat;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.layers.convolution.SeparableConvolution1DLayer;
import org.deeplearning4j.nn.params.SeparableConvolutionParamInitializer;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.deeplearning4j.util.Convolution1DUtils;
import org.deeplearning4j.util.ConvolutionUtils;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.*;

/**
 * 1D Separable convolution layer. Separable convolutions split a regular convolution into two
 * simpler operations: a depthwise convolution and a pointwise convolution.
 *
 * @author Adam Gibson
 */
@Data
@NoArgsConstructor
@ToString(callSuper = true)
@EqualsAndHashCode(callSuper = true)
public class SeparableConvolution1D extends ConvolutionLayer {

    protected int depthMultiplier;
    protected RNNFormat rnnDataFormat = RNNFormat.NCW;

    protected SeparableConvolution1D(Builder builder) {
        super(builder);
        this.depthMultiplier = builder.depthMultiplier;
        this.rnnDataFormat = builder.rnnDataFormat;
        this.hasBias = builder.hasBias;
        this.convolutionMode = builder.convolutionMode;
        this.dilation = builder.dilation;
        this.kernelSize = builder.kernelSize;
        this.stride = builder.stride;
        this.padding = builder.padding;
        initializeConstraints(builder);
    }

    @Override
    protected void initializeConstraints(org.deeplearning4j.nn.conf.layers.Layer.Builder<?> builder) {
        super.initializeConstraints(builder);
        if (((Builder) builder).pointWiseConstraints != null) {
            if (constraints == null) {
                constraints = new ArrayList<>();
            }
            for (LayerConstraint constraint : ((Builder) builder).pointWiseConstraints) {
                LayerConstraint clonedConstraint = constraint.clone();
                clonedConstraint.setParams(
                        Collections.singleton(SeparableConvolutionParamInitializer.POINT_WISE_WEIGHT_KEY));
                constraints.add(clonedConstraint);
            }
        }
    }

    @Override
    public SeparableConvolution1D clone() {
        SeparableConvolution1D clone = (SeparableConvolution1D) super.clone();
        return clone;
    }

    @Override
    public Layer instantiate(NeuralNetConfiguration conf, Collection<TrainingListener> trainingListeners,
                             int layerIndex, INDArray layerParamsView, boolean initializeParams, DataType networkDataType) {
        LayerValidation.assertNInNOutSet("SeparableConvolution1D", getLayerName(), layerIndex, getNIn(), getNOut());

        SeparableConvolution1DLayer ret = new SeparableConvolution1DLayer(conf, networkDataType);
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
        return SeparableConvolutionParamInitializer.getInstance();
    }

    @Override
    public InputType getOutputType(int layerIndex, InputType inputType) {
        if (inputType == null || inputType.getType() != InputType.Type.RNN) {
            throw new IllegalStateException("Invalid input for SeparableConvolution1D layer (layer name=\"" + getLayerName()
                    + "\"): Expected RNN input, got " + inputType);
        }
        InputType.InputTypeRecurrent r = (InputType.InputTypeRecurrent) inputType;
        long inputTsLength = r.getTimeSeriesLength();
        long outLength;
        if (inputTsLength < 0) {
            outLength = -1;
        } else {
            outLength = Convolution1DUtils.getOutputSizeLong(inputTsLength, kernelSize[0], stride[0], padding[0],
                    convolutionMode, dilation[0]);
        }
        return InputType.recurrent(nOut, outLength, rnnDataFormat);
    }

    public RNNFormat getRnnDataFormat() {
        return rnnDataFormat;
    }

    @Getter
    @Setter
    public static class Builder extends BaseConvBuilder<Builder> {

        /**
         * Set channels multiplier for depthwise convolution
         */
        protected int depthMultiplier = 1;

        /**
         * RNN data format - NCW (channels first) or NWC (channels last)
         */
        protected RNNFormat rnnDataFormat = RNNFormat.NCW;

        /**
         * Constraints for point-wise convolution weights
         */
        protected List<LayerConstraint> pointWiseConstraints;

        public Builder() {
            this.kernelSize = new long[]{1, 1};
            this.stride = new long[]{1, 1};
            this.padding = new long[]{0, 0};
            this.dilation = new long[]{1, 1};
        }

        public Builder(int kernelSize) {
            this();
            this.setKernelSize(kernelSize);
        }

        public Builder(int kernelSize, int stride) {
            this();
            this.setKernelSize(kernelSize);
            this.setStride(stride);
        }

        public Builder(int kernelSize, int stride, int padding) {
            this();
            this.setKernelSize(kernelSize);
            this.setStride(stride);
            this.setPadding(padding);
        }

        @Override
        protected boolean allowCausal() {
            return true;
        }

        /**
         * Set channels multiplier for depthwise convolution
         *
         * @param depthMultiplier Depth multiplier
         */
        public Builder depthMultiplier(int depthMultiplier) {
            this.depthMultiplier = depthMultiplier;
            return this;
        }

        /**
         * Set the data format for the RNN activations - NCW (channels first) or NWC (channels last).
         *
         * @param rnnDataFormat Format for activations
         */
        public Builder rnnDataFormat(RNNFormat rnnDataFormat) {
            this.rnnDataFormat = rnnDataFormat;
            return this;
        }

        /**
         * Set constraints for point-wise convolution weights
         *
         * @param constraints Constraints to apply
         */
        public Builder constrainPointWise(LayerConstraint... constraints) {
            this.pointWiseConstraints = Arrays.asList(constraints);
            return this;
        }

        /**
         * Size of the convolution kernel
         *
         * @param kernelSize Kernel size
         */
        public Builder kernelSize(int kernelSize) {
            this.setKernelSize(kernelSize);
            return this;
        }

        /**
         * Stride for the convolution
         *
         * @param stride Stride
         */
        public Builder stride(int stride) {
            this.setStride(stride);
            return this;
        }

        /**
         * Padding for the convolution
         *
         * @param padding Padding
         */
        public Builder padding(int padding) {
            this.setPadding(padding);
            return this;
        }

        /**
         * Dilation for the convolution
         *
         * @param dilation Dilation
         */
        public Builder dilation(int dilation) {
            this.setDilation(dilation);
            return this;
        }

        @Override
        public void setKernelSize(long... kernelSize) {
            if (kernelSize == null) {
                this.kernelSize = null;
                return;
            }
            this.kernelSize = ConvolutionUtils.getLongConfig(kernelSize, 1);
        }

        @Override
        public void setStride(long... stride) {
            if (stride == null) {
                this.stride = null;
                return;
            }
            this.stride = ConvolutionUtils.getLongConfig(stride, 1);
        }

        @Override
        public void setPadding(long... padding) {
            if (padding == null) {
                this.padding = null;
                return;
            }
            this.padding = ConvolutionUtils.getLongConfig(padding, 0);
        }

        @Override
        public void setDilation(long... dilation) {
            if (dilation == null) {
                this.dilation = null;
                return;
            }
            this.dilation = ConvolutionUtils.getLongConfig(dilation, 1);
        }

        @Override
        public SeparableConvolution1D build() {
            ConvolutionUtils.validateConvolutionModePadding(convolutionMode, padding);
            return new SeparableConvolution1D(this);
        }
    }
}
