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
import lombok.val;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.samediff.SameDiffLayer;
import org.deeplearning4j.nn.conf.layers.samediff.SDLayerParams;
import org.deeplearning4j.nn.conf.layers.samediff.SameDiffLayerUtils;
import org.deeplearning4j.nn.params.ConvolutionParamInitializer;
import org.deeplearning4j.nn.weights.WeightInitUtil;
import org.deeplearning4j.util.ConvolutionUtils;
import org.nd4j.autodiff.samediff.SDIndex;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.shade.jackson.annotation.JsonIgnoreProperties;

import java.util.*;

@Data
@EqualsAndHashCode(callSuper = true)
@JsonIgnoreProperties({"paramShapes"})
/**
 * SameDiff version of a 2D locally connected layer.
 *
 *
 * @author Max Pumperla
 */
public class LocallyConnected2D extends SameDiffLayer {

    private static final List<String> WEIGHT_KEYS = Collections.singletonList(ConvolutionParamInitializer.WEIGHT_KEY);
    private static final List<String> BIAS_KEYS = Collections.singletonList(ConvolutionParamInitializer.BIAS_KEY);
    private static final List<String> PARAM_KEYS = Arrays.asList(
            ConvolutionParamInitializer.BIAS_KEY,
            ConvolutionParamInitializer.WEIGHT_KEY);

    private long nIn;
    private long nOut;
    private Activation activation;
    private int[] kernel;
    private int[] stride;
    private int[] padding;
    private ConvolutionMode cm;
    private int[] dilation;
    private boolean hasBias;
    private int[] inputSize;
    private int[] outputSize;
    private int featureDim;

    protected LocallyConnected2D(Builder builder) {
        super(builder);
        this.nIn = builder.nIn;
        this.nOut = builder.nOut;
        this.activation = builder.activation;
        this.kernel = builder.kernel;
        this.stride = builder.stride;
        this.padding = builder.padding;
        this.cm = builder.cm;
        this.dilation = builder.dilation;
        this.hasBias = builder.hasBias;
        this.inputSize = builder.inputSize;
        this.featureDim = kernel[0] * kernel[1] * (int) nIn;
    }

    private LocallyConnected2D(){
        //No arg constructor for Jackson/JSON serialization
    }

    public void computeOutputSize() {
        int nIn = (int) getNIn();

        if (inputSize == null) {
            throw new IllegalArgumentException("Input size has to be specified for locally connected layers.");
        }

        int[] inputShape = new int[] {1, nIn, inputSize[0], inputSize[1]};
        INDArray dummyInputForShapeInference = Nd4j.ones(inputShape);

        if (cm == ConvolutionMode.Same) {
            this.outputSize = ConvolutionUtils.getOutputSize(
                    dummyInputForShapeInference, kernel, stride, null, cm, dilation);
            this.padding = ConvolutionUtils.getSameModeTopLeftPadding(outputSize, inputSize, kernel, stride, dilation);
        } else {
            this.outputSize = ConvolutionUtils.getOutputSize(
                    dummyInputForShapeInference, kernel, stride, padding, cm, dilation);
        }
    }

    @Override
    public InputType getOutputType(int layerIndex, InputType inputType) {
        if (inputType == null || inputType.getType() != InputType.Type.CNN) {
            throw new IllegalArgumentException("Provided input type for locally connected 2D layers has to be " +
                    "of CNN type, got: " + inputType);
        }
        // dynamically compute input size from input type
        InputType.InputTypeConvolutional cnnType = (InputType.InputTypeConvolutional) inputType;
        this.inputSize = new int[] { (int) cnnType.getHeight(), (int) cnnType.getWidth()};
        computeOutputSize();

        return InputTypeUtil.getOutputTypeCnnLayers(inputType, kernel, stride, padding, new int[]{1, 1},
                cm, nOut, layerIndex, getLayerName(), LocallyConnected2D.class);
    }

    @Override
    public void setNIn(InputType inputType, boolean override) {
        if (nIn <= 0 || override) {
            InputType.InputTypeConvolutional c = (InputType.InputTypeConvolutional) inputType;
            this.nIn = c.getChannels();
        }
    }

    @Override
    public InputPreProcessor getPreProcessorForInputType(InputType inputType) {
        return InputTypeUtil.getPreProcessorForInputTypeCnnLayers(inputType, getLayerName());
    }

    @Override
    public void defineParameters(SDLayerParams params) {
        params.clear();
        val weightsShape = new long[]{outputSize[0] * outputSize[1], featureDim, nOut};
        params.addWeightParam(ConvolutionParamInitializer.WEIGHT_KEY, weightsShape);
        if(hasBias) {
            val biasShape = new long[]{1, nOut};
            params.addBiasParam(ConvolutionParamInitializer.BIAS_KEY, biasShape);
        }
    }

    @Override
    public void initializeParameters(Map<String, INDArray> params) {
        try(MemoryWorkspace ws = Nd4j.getWorkspaceManager().scopeOutOfWorkspaces()) {
            for (Map.Entry<String, INDArray> e : params.entrySet()) {
                if (ConvolutionParamInitializer.BIAS_KEY.equals(e.getKey())) {
                    e.getValue().assign(0);
                } else {
                    double fanIn = nIn * kernel[0] * kernel[1];
                    double fanOut = nOut * kernel[0] * kernel[1] / ((double) stride[0] * stride[1]);
                    WeightInitUtil.initWeights(
                            fanIn, fanOut, e.getValue().shape(), weightInit, null, 'c', e.getValue());
                }
            }
        }
    }

    @Override
    public SDVariable defineLayer(SameDiff sameDiff, SDVariable layerInput, Map<String, SDVariable> paramTable) {

        SDVariable w = paramTable.get(ConvolutionParamInitializer.WEIGHT_KEY);

        long[] inputShape = layerInput.getShape();
        long miniBatch = inputShape[0];
        int outH = outputSize[0];
        int outW = outputSize[1];
        int sH = stride[0];
        int sW = stride[1];
        int kH = kernel[0];
        int kW = kernel[1];

        SDVariable[] inputArray = new SDVariable[outH * outW];
        for (int i = 0; i < outH; i++) {
            for (int j = 0; j < outW; j++) {
                SDVariable slice = layerInput.get(
                        SDIndex.all(), // miniBatch
                        SDIndex.all(), // nIn
                        SDIndex.interval(i * sH, i * sH + kH), // kernel height
                        SDIndex.interval(j * sW, j * sW + kW) // kernel width
                );
                inputArray[i * outH + j] = sameDiff.reshape(slice, 1, miniBatch, featureDim);
            }
        }
        SDVariable concatOutput = sameDiff.concat(0, inputArray); // (outH * outW, miniBatch, featureDim)

        SDVariable mmulResult = sameDiff.mmul(concatOutput, w); // (outH * outW, miniBatch, nOut)

        SDVariable reshapeResult = sameDiff.reshape(mmulResult, outH, outW, miniBatch, nOut);

        SDVariable permutedResult = sameDiff.permute(reshapeResult,2, 3, 0, 1); // (mb, nOut, outH, outW)

        SDVariable b = sameDiff.zero("bias", new long[] {1, nOut});
        if(hasBias){
            b = paramTable.get(ConvolutionParamInitializer.BIAS_KEY);
        }
        SDVariable biasAddedResult = sameDiff.biasAdd(permutedResult, b);
        return activation.asSameDiff("out", sameDiff, biasAddedResult);

    }

    @Override
    public void applyGlobalConfigToLayer(NeuralNetConfiguration.Builder globalConfig) {
        if (activation == null) {
            activation = SameDiffLayerUtils.fromIActivation(globalConfig.getActivationFn());
        }
        if (cm == null) {
            cm = globalConfig.getConvolutionMode();
        }
    }

    public static class Builder extends SameDiffLayer.Builder<Builder> {

        private int nIn;
        private int nOut;
        private Activation activation = Activation.TANH;
        private int[] kernel = new int[]{2, 2};
        private int[] stride = new int[]{1, 1};
        private int[] padding = new int[]{0, 0};
        private int[] dilation = new int[]{1, 1};
        private int[] inputSize;
        private ConvolutionMode cm = ConvolutionMode.Same;
        private boolean hasBias = false;

        public Builder nIn(int nIn) {
            this.nIn = nIn;
            return this;
        }

        public Builder nOut(int nOut) {
            this.nOut = nOut;
            return this;
        }


        public Builder activation(Activation activation) {
            this.activation = activation;
            return this;
        }

        public Builder kernelSize(int... k) {
            this.kernel = k;
            return this;
        }

        public Builder stride(int... s) {
            this.stride = s;
            return this;
        }

        public Builder padding(int... p) {
            this.padding = p;
            return this;
        }

        public Builder convolutionMode(ConvolutionMode cm) {
            this.cm = cm;
            return this;
        }

        public Builder dilation(int... d) {
            this.dilation = d;
            return this;
        }

        public Builder hasBias(boolean hasBias){
            this.hasBias = hasBias;
            return this;
        }

        /**
         * Set input filter size (h,w) for this locally connected 2D layer
         *
         * @param inputSize pair of height and width of the input filters to this layer
         * @return Builder
         */
        public Builder setInputSize(int... inputSize){
            Preconditions.checkState(inputSize.length == 2, "Input size argument of a locally connected" +
                    "layer has to have length 2, got " + inputSize.length);
            this.inputSize = inputSize;
            return this;
        }

        @Override
        @SuppressWarnings("unchecked")
        public LocallyConnected2D build() {
            ConvolutionUtils.validateConvolutionModePadding(cm, padding);
            ConvolutionUtils.validateCnnKernelStridePadding(kernel, stride, padding);
            return new LocallyConnected2D(this);
        }
    }
}
