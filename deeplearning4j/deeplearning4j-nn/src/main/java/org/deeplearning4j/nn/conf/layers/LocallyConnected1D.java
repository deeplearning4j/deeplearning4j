/*
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
import org.deeplearning4j.nn.conf.layers.samediff.SDLayerParams;
import org.deeplearning4j.nn.conf.layers.samediff.SameDiffLayer;
import org.deeplearning4j.nn.conf.layers.samediff.SameDiffLayerUtils;
import org.deeplearning4j.nn.params.ConvolutionParamInitializer;
import org.deeplearning4j.nn.weights.WeightInitUtil;
import org.deeplearning4j.util.Convolution1DUtils;
import org.nd4j.autodiff.samediff.SDIndex;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.shade.jackson.annotation.JsonIgnoreProperties;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;

/**
 * SameDiff version of a 1D locally connected layer.
 *
 * @author Max Pumperla
 */
@Data
@EqualsAndHashCode(callSuper = true)
@JsonIgnoreProperties({"paramShapes"})
public class LocallyConnected1D extends SameDiffLayer {

    private static final List<String> WEIGHT_KEYS = Collections.singletonList(ConvolutionParamInitializer.WEIGHT_KEY);
    private static final List<String> BIAS_KEYS = Collections.singletonList(ConvolutionParamInitializer.BIAS_KEY);
    private static final List<String> PARAM_KEYS = Arrays.asList(
            ConvolutionParamInitializer.BIAS_KEY,
            ConvolutionParamInitializer.WEIGHT_KEY);

    private long nIn;
    private long nOut;
    private Activation activation;
    private int kernel;
    private int stride;
    private int padding;
    private ConvolutionMode cm;
    private int dilation;
    private boolean hasBias;
    private int inputSize;
    private int outputSize;
    private int featureDim;

    protected LocallyConnected1D(Builder builder) {
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
        this.featureDim = kernel * (int) nIn;
    }

    private LocallyConnected1D(){
        //No arg constructor for Jackson/JSON serialization
    }

    public void computeOutputSize() {
        int nIn = (int) getNIn();
        if (inputSize == 0) {
            throw new IllegalArgumentException("Input size has to be set for Locally connected layers");
        }
        int[] inputShape = new int[] {1, nIn, inputSize};
        INDArray dummyInputForShapeInference = Nd4j.ones(inputShape);

        if (cm == ConvolutionMode.Same) {
            this.outputSize = Convolution1DUtils.getOutputSize(
                    dummyInputForShapeInference, kernel, stride, 0, cm, dilation);
            this.padding = Convolution1DUtils.getSameModeTopLeftPadding(outputSize, inputSize, kernel, stride, dilation);
        } else {
            this.outputSize = Convolution1DUtils.getOutputSize(
                    dummyInputForShapeInference, kernel, stride, padding, cm, dilation);
        }
    }

    @Override
    public InputType getOutputType(int layerIndex, InputType inputType) {
        if (inputType == null || inputType.getType() != InputType.Type.RNN) {
            throw new IllegalArgumentException("Provided input type for locally connected 1D layers has to be " +
                    "of CNN1D/RNN type, got: " + inputType);
        }
        // dynamically compute input size from input type
        InputType.InputTypeRecurrent rnnType = (InputType.InputTypeRecurrent) inputType;
        this.inputSize = (int) rnnType.getTimeSeriesLength();
        computeOutputSize();

        return InputTypeUtil.getOutputTypeCnn1DLayers(inputType, kernel, stride, padding, 1,
                cm, nOut, layerIndex, getLayerName(), LocallyConnected1D.class);
    }

    @Override
    public void setNIn(InputType inputType, boolean override) {
        if (nIn <= 0 || override) {
            InputType.InputTypeRecurrent c = (InputType.InputTypeRecurrent) inputType;
            this.nIn = c.getSize();
        }
    }

    @Override
    public InputPreProcessor getPreProcessorForInputType(InputType inputType) {
        return InputTypeUtil.getPreprocessorForInputTypeRnnLayers(inputType, getLayerName());
    }

    @Override
    public void defineParameters(SDLayerParams params) {
        params.clear();
        val weightsShape = new long[]{outputSize, featureDim, nOut};
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
                    double fanIn = nIn * kernel;
                    double fanOut = nOut * kernel / ((double) stride);
                    WeightInitUtil.initWeights(
                            fanIn, fanOut, e.getValue().shape(), weightInit, null, 'c', e.getValue());
                }
            }
        }
    }

    @Override
    public SDVariable defineLayer(SameDiff sameDiff, SDVariable layerInput, Map<String, SDVariable> paramTable) {

        SDVariable w = paramTable.get(ConvolutionParamInitializer.WEIGHT_KEY); // (outH, featureDim, nOut)
        // System.out.println(Arrays.toString(w.getShape()));

        long[] inputShape = layerInput.getShape();
        long miniBatch = inputShape[0];
        int outH = outputSize;
        int sH = stride;
        int kH = kernel;

        SDVariable[] inputArray = new SDVariable[outH];
        for (int i = 0; i < outH; i++) {
                SDVariable slice = layerInput.get(
                        SDIndex.all(), // miniBatch
                        SDIndex.all(), // nIn
                        SDIndex.interval(i * sH, i * sH + kH) // kernel
                );
                inputArray[i] = sameDiff.reshape(slice, 1, miniBatch, featureDim);
        }
        SDVariable concatOutput = sameDiff.concat(0, inputArray); // (outH, miniBatch, featureDim)
        sameDiff.exec();

        System.out.println(Arrays.toString(concatOutput.getShape()));

        SDVariable mmulResult = sameDiff.mmul(concatOutput, w); // (outH, miniBatch, nOut)
        System.out.println(Arrays.toString(mmulResult.getShape()));

        SDVariable result = sameDiff.permute(mmulResult,1, 2, 0); // (miniBatch, nOut, outH)

        SDVariable b = sameDiff.zero("bias", new long[] {1, nOut});
        if(hasBias){
            b = paramTable.get(ConvolutionParamInitializer.BIAS_KEY);
        }
        SDVariable biasAddedResult = sameDiff.biasAdd(result, b);
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
        private int kernel = 2;
        private int stride = 1;
        private int padding = 0;
        private int dilation = 1;
        private int inputSize;
        private ConvolutionMode cm = ConvolutionMode.Same;
        private boolean hasBias = false;

        /**
         * @param nIn Number of inputs to the layer (input size)
         */
        public Builder nIn(int nIn) {
            this.nIn = nIn;
            return this;
        }

        /**
         * @param nOut Number of outputs (output size)
         */
        public Builder nOut(int nOut) {
            this.nOut = nOut;
            return this;
        }

        /**
         * @param activation Activation function for the layer
         */
        public Builder activation(Activation activation) {
            this.activation = activation;
            return this;
        }

        /**
         * @param k Kernel size for the layer
         */
        public Builder kernelSize(int k) {
            this.kernel = k;
            return this;
        }

        /**
         * @param s Stride for the layer
         */
        public Builder stride(int s) {
            this.stride = s;
            return this;
        }

        /**
         * @param p Padding for the layer. Not used if {@link ConvolutionMode#Same} is set
         */
        public Builder padding(int p) {
            this.padding = p;
            return this;
        }

        /**
         * @param cm Convolution mode for the layer. See {@link ConvolutionMode} for details
         */
        public Builder convolutionMode(ConvolutionMode cm) {
            this.cm = cm;
            return this;
        }

        /**
         * @param d Dilation for the layer
         */
        public Builder dilation(int d) {
            this.dilation = d;
            return this;
        }

        /**
         * @param hasBias If true (default is false) the layer will have a bias
         */
        public Builder hasBias(boolean hasBias){
            this.hasBias = hasBias;
            return this;
        }

        /**
         * Set input filter size for this locally connected 1D layer
         *
         * @param inputSize height of the input filters
         * @return Builder
         */
        public Builder setInputSize(int inputSize){ ;
            this.inputSize = inputSize;
            return this;
        }

        @Override
        @SuppressWarnings("unchecked")
        public LocallyConnected1D build() {
            Convolution1DUtils.validateConvolutionModePadding(cm, padding);
            Convolution1DUtils.validateCnn1DKernelStridePadding(kernel, stride, padding);
            return new LocallyConnected1D(this);
        }
    }
}
