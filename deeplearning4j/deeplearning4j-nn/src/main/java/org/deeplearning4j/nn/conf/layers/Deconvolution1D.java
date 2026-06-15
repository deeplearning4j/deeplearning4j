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
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.RNNFormat;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.samediff.SDLayerParams;
import org.deeplearning4j.nn.conf.layers.samediff.SameDiffLayer;
import org.deeplearning4j.nn.weights.WeightInitUtil;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.DeConv2DConfig;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.IActivation;

import java.util.Map;

/**
 * Deconvolution1D layer (also known as Conv1DTranspose or transposed convolution).
 *
 * This layer performs the transpose of a 1D convolution, useful for:
 * - Upsampling in autoencoders
 * - Generating sequences in generative models
 * - Feature map expansion
 *
 * This implementation uses SameDiff to apply deconv2d with height=1.
 *
 * @author Adam Gibson
 */
@Data
@EqualsAndHashCode(callSuper = true)
public class Deconvolution1D extends SameDiffLayer {

    private static final String WEIGHT_KEY = "weight";
    private static final String BIAS_KEY = "bias";

    private long nIn;
    private long nOut;
    private long kernelSize;
    private long stride;
    private long padding;
    private long dilation;
    private boolean hasBias;
    private ConvolutionMode convolutionMode;
    private IActivation activation;

    private Deconvolution1D() {
        // No-arg constructor for serialization
    }

    protected Deconvolution1D(Builder builder) {
        super(builder);
        this.nIn = builder.nIn;
        this.nOut = builder.nOut;
        this.kernelSize = builder.kernelSize;
        this.stride = builder.stride;
        this.padding = builder.padding;
        this.dilation = builder.dilation;
        this.hasBias = builder.hasBias;
        this.convolutionMode = builder.convolutionMode;
        this.activation = builder.activation;
    }

    @Override
    public InputPreProcessor getPreProcessorForInputType(InputType inputType) {
        return InputTypeUtil.getPreprocessorForInputTypeRnnLayers(inputType, RNNFormat.NCW, getLayerName());
    }

    @Override
    public void setNIn(InputType inputType, boolean override) {
        if (inputType == null || inputType.getType() != InputType.Type.RNN) {
            throw new IllegalStateException("Invalid input for Deconvolution1D layer (layer name = \""
                    + getLayerName() + "\"): expect RNN input type with size > 0. Got: " + inputType);
        }
        if (nIn <= 0 || override) {
            InputType.InputTypeRecurrent r = (InputType.InputTypeRecurrent) inputType;
            this.nIn = r.getSize();
        }
    }

    @Override
    public InputType getOutputType(int layerIndex, InputType inputType) {
        if (inputType == null || inputType.getType() != InputType.Type.RNN) {
            throw new IllegalStateException("Invalid input for Deconvolution1D layer (layer index = "
                    + layerIndex + ", layer name = \"" + getLayerName() + "\"): expect RNN input type. Got: "
                    + inputType);
        }
        InputType.InputTypeRecurrent itr = (InputType.InputTypeRecurrent) inputType;

        long outputLength;
        if (convolutionMode == ConvolutionMode.Same) {
            outputLength = itr.getTimeSeriesLength() * stride;
        } else {
            // Valid/Truncate mode - transpose convolution expands
            outputLength = (itr.getTimeSeriesLength() - 1) * stride - 2 * padding + dilation * (kernelSize - 1) + 1;
        }

        return InputType.recurrent(nOut, outputLength);
    }

    @Override
    public void defineParameters(SDLayerParams params) {
        // Weights: [kernelSize, 1, nOut, nIn] for deconv (note: nOut and nIn swapped compared to conv)
        // Using format [kH, kW, oC, iC] for deconv2d
        params.addWeightParam(WEIGHT_KEY, kernelSize, 1, nOut, nIn);

        if (hasBias) {
            params.addBiasParam(BIAS_KEY, 1, nOut);
        }
    }

    @Override
    public void initializeParameters(Map<String, INDArray> params) {
        try (MemoryWorkspace ws = Nd4j.getWorkspaceManager().scopeOutOfWorkspaces()) {
            // Initialize weights
            INDArray weight = params.get(WEIGHT_KEY);
            WeightInitUtil.initWeights(nIn, nOut, weight.shape(), weightInit, null, 'c', weight);

            // Initialize bias
            if (hasBias) {
                INDArray bias = params.get(BIAS_KEY);
                bias.assign(0);
            }
        }
    }

    @Override
    public SDVariable defineLayer(SameDiff sameDiff, SDVariable layerInput,
                                  Map<String, SDVariable> paramTable, SDVariable mask) {
        // Input shape: [batch, channels, length] (NCW format)
        // Reshape to 4D for deconv2d: [batch, channels, 1, length] (NCHW format)
        SDVariable input4D = sameDiff.expandDims(layerInput, 2);

        SDVariable weights = paramTable.get(WEIGHT_KEY);
        SDVariable bias = hasBias ? paramTable.get(BIAS_KEY) : null;

        // Apply deconv2d
        DeConv2DConfig config = DeConv2DConfig.builder()
                .kH(1).kW(kernelSize)
                .sH(1).sW(stride)
                .pH(0).pW(padding)
                .dH(1).dW(dilation)
                .isSameMode(convolutionMode == ConvolutionMode.Same)
                .dataFormat("NCHW")
                .build();

        SDVariable output;
        if (hasBias) {
            output = sameDiff.cnn().deconv2d(input4D, weights, bias, config);
        } else {
            output = sameDiff.cnn().deconv2d(input4D, weights, config);
        }

        // Squeeze back to 3D: [batch, nOut, length]
        SDVariable output3D = sameDiff.squeeze(output, 2);

        // Note: Activation should be applied by adding a separate activation layer
        // SameDiff layers don't directly support IActivation interface

        return output3D;
    }

    @Getter
    @Setter
    public static class Builder extends SameDiffLayer.Builder<Builder> {
        private long nIn;
        private long nOut;
        private long kernelSize = 3;
        private long stride = 1;
        private long padding = 0;
        private long dilation = 1;
        private boolean hasBias = true;
        private ConvolutionMode convolutionMode = ConvolutionMode.Same;
        private IActivation activation = Activation.IDENTITY.getActivationFunction();

        public Builder nIn(long nIn) {
            this.nIn = nIn;
            return this;
        }

        public Builder nOut(long nOut) {
            this.nOut = nOut;
            return this;
        }

        public Builder kernelSize(long kernelSize) {
            this.kernelSize = kernelSize;
            return this;
        }

        public Builder stride(long stride) {
            this.stride = stride;
            return this;
        }

        public Builder padding(long padding) {
            this.padding = padding;
            return this;
        }

        public Builder dilation(long dilation) {
            this.dilation = dilation;
            return this;
        }

        public Builder hasBias(boolean hasBias) {
            this.hasBias = hasBias;
            return this;
        }

        public Builder convolutionMode(ConvolutionMode mode) {
            this.convolutionMode = mode;
            return this;
        }

        public Builder activation(IActivation activation) {
            this.activation = activation;
            return this;
        }

        @Override
        public Deconvolution1D build() {
            return new Deconvolution1D(this);
        }
    }
}
