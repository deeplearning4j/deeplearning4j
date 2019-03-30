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

import java.util.Map;
import lombok.AccessLevel;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.inputs.InputType.InputTypeConvolutional;
import org.deeplearning4j.nn.conf.inputs.InputType.Type;
import org.deeplearning4j.nn.conf.layers.samediff.SDLayerParams;
import org.deeplearning4j.nn.conf.layers.samediff.SameDiffLayer;
import org.deeplearning4j.nn.weights.WeightInitUtil;
import org.deeplearning4j.util.CapsuleUtils;
import org.deeplearning4j.util.ValidationUtils;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv2DConfig;
import org.nd4j.linalg.factory.Nd4j;

/**
 * An implementation of the PrimaryCaps layer from Dynamic Routing Between Capsules
 *
 * Is a reshaped 2D convolution, and the input should be 2D convolutional ([mb, c, h, w]).
 *
 * From <a href="http://papers.nips.cc/paper/6975-dynamic-routing-between-capsules.pdf">Dynamic Routing Between Capsules</a>
 *
 * @author Ryan Nett
 */
@Data
@NoArgsConstructor
@EqualsAndHashCode(callSuper = true)
public class PrimaryCapsules extends SameDiffLayer {

    private int[] kernelSize;
    private int[] stride;
    private int[] padding;
    private int[] dilation;
    private int inputChannels;
    private int channels;

    private boolean hasBias;

    private int capsules;
    private int capsuleDimensions;

    private ConvolutionMode convolutionMode = ConvolutionMode.Truncate;

    private boolean useRelu = false;
    private double leak = 0;

    private static final String WEIGHT_PARAM = "weight";
    private static final String BIAS_PARAM = "bias";

    public PrimaryCapsules(Builder builder){
        super(builder);

        this.kernelSize = builder.kernelSize;
        this.stride = builder.stride;
        this.padding = builder.padding;
        this.dilation = builder.dilation;
        this.channels = builder.channels;
        this.hasBias = builder.hasBias;
        this.capsules = builder.capsules;
        this.capsuleDimensions = builder.capsuleDimensions;
        this.convolutionMode = builder.convolutionMode;
        this.useRelu = builder.useRelu;
        this.leak = builder.leak;

        if(capsuleDimensions <= 0 || channels <= 0){
            throw new IllegalArgumentException("Invalid configuration for Primary Capsules (layer name = \""
                    + layerName + "\"):"
                    + " capsuleDimensions and channels must be > 0.  Got: "
                    + capsuleDimensions + ", " + channels);
        }

        if(capsules < 0){
            throw new IllegalArgumentException("Invalid configuration for Capsule Layer (layer name = \""
                    + layerName + "\"):"
                    + " capsules must be >= 0 if set.  Got: "
                    + capsules);
        }

    }

    @Override
    public SDVariable defineLayer(SameDiff SD, SDVariable input, Map<String, SDVariable> paramTable) {

        Conv2DConfig conf = Conv2DConfig.builder()
                .kH(kernelSize[0]).kW(kernelSize[1])
                .sH(stride[0]).sW(stride[1])
                .pH(padding[0]).pW(padding[1])
                .dH(dilation[0]).dW(dilation[1])
                .isSameMode(convolutionMode == ConvolutionMode.Same)
                .build();

        SDVariable conved;

        if(hasBias){
            conved = SD.cnn.conv2d(input, paramTable.get(WEIGHT_PARAM), paramTable.get(BIAS_PARAM), conf);
        } else {
            conved = SD.cnn.conv2d(input, paramTable.get(WEIGHT_PARAM), conf);
        }

        if(useRelu){
            if(leak == 0) {
                conved = SD.nn.relu(conved, 0);
            } else {
                conved = SD.nn.leakyRelu(conved, leak);
            }
        }

        SDVariable reshaped = conved.reshape(-1, capsules, capsuleDimensions);
        return CapsuleUtils.squash(SD, reshaped, 2);
    }

    @Override
    public void defineParameters(SDLayerParams params) {
        params.clear();
        params.addWeightParam(WEIGHT_PARAM,
                kernelSize[0], kernelSize[1], inputChannels, channels);

        if(hasBias){
            params.addBiasParam(BIAS_PARAM, channels);
        }
    }

    @Override
    public void initializeParameters(Map<String, INDArray> params) {
        try (MemoryWorkspace ws = Nd4j.getWorkspaceManager().scopeOutOfWorkspaces()) {
            for (Map.Entry<String, INDArray> e : params.entrySet()) {
                if (BIAS_PARAM.equals(e.getKey())) {
                    e.getValue().assign(0);
                } else if(WEIGHT_PARAM.equals(e.getKey())){
                    double fanIn = inputChannels * kernelSize[0] * kernelSize[1];
                    double fanOut = channels * kernelSize[0] * kernelSize[1] / ((double) stride[0] * stride[1]);
                    WeightInitUtil.initWeights(fanIn, fanOut, e.getValue().shape(), weightInit, null, 'c',
                            e.getValue());
                }
            }
        }
    }

    @Override
    public InputType getOutputType(int layerIndex, InputType inputType) {
        if (inputType == null || inputType.getType() != Type.CNN) {
            throw new IllegalStateException("Invalid input for Primary Capsules layer (layer name = \""
                    + layerName + "\"): expect CNN input.  Got: " + inputType);
        }

        if(capsules > 0){
            return InputType.recurrent(capsules, capsuleDimensions);
        } else {

            InputTypeConvolutional out = (InputTypeConvolutional) InputTypeUtil
                    .getOutputTypeCnnLayers(inputType, kernelSize, stride, padding, dilation, convolutionMode,
                            channels, -1, getLayerName(), PrimaryCapsules.class);

            return InputType.recurrent((int) (out.getChannels() * out.getHeight() * out.getWidth() / capsuleDimensions),
                    capsuleDimensions);
        }
    }

    @Override
    public void setNIn(InputType inputType, boolean override) {
        if (inputType == null || inputType.getType() != Type.CNN) {
            throw new IllegalStateException("Invalid input for Primary Capsules layer (layer name = \""
                    + layerName + "\"): expect CNN input.  Got: " + inputType);
        }

        InputTypeConvolutional ci = (InputTypeConvolutional) inputType;

        this.inputChannels = (int) ci.getChannels();

        if(channels <= 0 || override) {

            InputTypeConvolutional out = (InputTypeConvolutional) InputTypeUtil
                    .getOutputTypeCnnLayers(inputType, kernelSize, stride, padding, dilation, convolutionMode,
                            channels, -1, getLayerName(), PrimaryCapsules.class);

            this.capsules = (int) (out.getChannels() * out.getHeight() * out.getWidth() / capsuleDimensions);
        }
    }

    @Getter
    @Setter
    public static class Builder extends SameDiffLayer.Builder<Builder>{

        @Setter(AccessLevel.NONE)
        private int[] kernelSize = new int[]{9, 9};

        @Setter(AccessLevel.NONE)
        private int[] stride = new int[]{2, 2};

        @Setter(AccessLevel.NONE)
        private int[] padding = new int[]{0, 0};

        @Setter(AccessLevel.NONE)
        private int[] dilation = new int[]{1, 1};

        private int channels = 32;

        private boolean hasBias = true;

        private int capsules;
        private int capsuleDimensions;

        private ConvolutionMode convolutionMode = ConvolutionMode.Truncate;

        private boolean useRelu = false;
        private double leak = 0;


        public void setKernelSize(int... kernelSize){
            this.kernelSize = ValidationUtils.validate2NonNegative(kernelSize, true, "kernelSize");
        }

        public void setStride(int... stride){
            this.stride = ValidationUtils.validate2NonNegative(stride, true, "stride");
        }

        public void setPadding(int... padding){
            this.padding = ValidationUtils.validate2NonNegative(padding, true, "padding");
        }

        public void setDilation(int... dilation){
            this.dilation = ValidationUtils.validate2NonNegative(dilation, true, "dilation");
        }


        public Builder(int capsuleDimensions, int channels,
                int[] kernelSize, int[] stride, int[] padding, int[] dilation,
                ConvolutionMode convolutionMode){
            this.capsuleDimensions = capsuleDimensions;
            this.channels = channels;
            this.setKernelSize(kernelSize);
            this.setStride(stride);
            this.setPadding(padding);
            this.setDilation(dilation);
            this.convolutionMode = convolutionMode;
        }

        public Builder(int capsuleDimensions, int channels,
                int[] kernelSize, int[] stride, int[] padding, int[] dilation){
            this(capsuleDimensions, channels, kernelSize, stride, padding, dilation, ConvolutionMode.Truncate);
        }

        public Builder(int capsuleDimensions, int channels,
                int[] kernelSize, int[] stride, int[] padding){
            this(capsuleDimensions, channels, kernelSize, stride, padding, new int[]{1, 1}, ConvolutionMode.Truncate);
        }

        public Builder(int capsuleDimensions, int channels,
                int[] kernelSize, int[] stride){
            this(capsuleDimensions, channels, kernelSize, stride, new int[]{0, 0}, new int[]{1, 1}, ConvolutionMode.Truncate);
        }

        public Builder(int capsuleDimensions, int channels,
                int[] kernelSize){
            this(capsuleDimensions, channels, kernelSize, new int[]{2, 2}, new int[]{0, 0}, new int[]{1, 1}, ConvolutionMode.Truncate);
        }

        public Builder(int capsuleDimensions, int channels){
            this(capsuleDimensions, channels, new int[]{9, 9}, new int[]{2, 2}, new int[]{0, 0}, new int[]{1, 1}, ConvolutionMode.Truncate);
        }

        /**
         * Sets the kernel size of the 2d convolution
         *
         * @see ConvolutionLayer.Builder#kernelSize(int...)
         * @param kernelSize
         * @return
         */
        public Builder kernelSize(int... kernelSize){
            this.setKernelSize(kernelSize);
            return this;
        }

        /**
         * Sets the stride of the 2d convolution
         *
         * @see ConvolutionLayer.Builder#stride(int...)
         * @param stride
         * @return
         */
        public Builder stride(int... stride){
            this.setStride(stride);
            return this;
        }

        /**
         * Sets the padding of the 2d convolution
         *
         * @see ConvolutionLayer.Builder#padding(int...)
         * @param padding
         * @return
         */
        public Builder padding(int... padding){
            this.setPadding(padding);
            return this;
        }

        /**
         * Sets the dilation of the 2d convolution
         *
         * @see ConvolutionLayer.Builder#dilation(int...)
         * @param dilation
         * @return
         */
        public Builder dilation(int... dilation){
            this.setDilation(dilation);
            return this;
        }

        /**
         * Sets the number of channels to use in the 2d convolution.
         *
         * Note that the actual number of channels is channels * capsuleDimensions
         *
         * Does the same thing as nOut()
         *
         * @param channels
         * @return
         */
        public Builder channels(int channels){
            this.channels = channels;
            return this;
        }

        /**
         * Sets the number of channels to use in the 2d convolution.
         *
         * Note that the actual number of channels is channels * capsuleDimensions
         *
         * Does the same thing as channels()
         *
         * @param nOut
         * @return
         */
        public Builder nOut(int nOut){
            return channels(nOut);
        }

        /**
         * Sets the number of dimensions to use in the capsules.
         * @param capsuleDimensions
         * @return
         */
        public Builder capsuleDimensions(int capsuleDimensions){
            this.capsuleDimensions = capsuleDimensions;
            return this;
        }

        /**
         * Usually inferred automatically.
         * @param capsules
         * @return
         */
        public Builder capsules(int capsules){
            this.capsules = capsules;
            return this;
        }

        public Builder hasBias(boolean hasBias){
            this.hasBias = hasBias;
            return this;
        }

        /**
         * The convolution mode to use in the 2d convolution
         * @param convolutionMode
         * @return
         */
        public Builder convolutionMode(ConvolutionMode convolutionMode){
            this.convolutionMode = convolutionMode;
            return this;
        }

        /**
         * Whether to use a ReLU activation on the 2d convolution
         * @param useRelu
         * @return
         */
        public Builder useReLU(boolean useRelu){
            this.useRelu = useRelu;
            return this;
        }

        /**
         * Use a ReLU activation on the 2d convolution
         * @return
         */
        public Builder useReLU(){
            return useReLU(true);
        }

        /**
         * Use a LeakyReLU activation on the 2d convolution
         * @param leak the alpha value for the LeakyReLU activation.
         * @return
         */
        public Builder useLeakyReLU(double leak){
            this.useRelu = true;
            this.leak = leak;
            return this;
        }

        @Override
        public <E extends Layer> E build() {
            return (E) new PrimaryCapsules(this);
        }
    }
}
