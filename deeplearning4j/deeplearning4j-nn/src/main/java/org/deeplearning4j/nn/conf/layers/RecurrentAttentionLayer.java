/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.RNNFormat;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.samediff.SDLayerParams;
import org.deeplearning4j.nn.conf.layers.samediff.SameDiffLayer;
import org.deeplearning4j.nn.conf.layers.samediff.SameDiffLayerUtils;
import org.deeplearning4j.nn.params.SimpleRnnParamInitializer;
import org.deeplearning4j.nn.weights.WeightInitUtil;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Map;

/**
 * Implements Recurrent Dot Product Attention
 *
 * Takes in RNN style input in the shape of [batchSize, features, timesteps]
 * and applies dot product attention using the hidden state as the query and
 * <b>all</b> time steps as keys/values.
 *
 * a_i = σ(W*x_i + R*attention(a_i, x, x) + b)
 *
 * The output will be in the shape of [batchSize, nOut, timesteps].
 *
 * Attention implemented as in
 * Attention is all you need by Vaswani et al. [arXiv:1706.03762], pp. 4,5
 *
 * <b>Note: At the moment this is limited to equal-length mini-batch input. Mixing mini-batches of differing timestep
 * counts will not work.</b>
 *
 * @see LearnedSelfAttentionLayer
 * @see SelfAttentionLayer
 * @see org.nd4j.linalg.api.ops.impl.transforms.custom.MultiHeadDotProductAttention
 * @author Paul Dubs
 */
@Data
@EqualsAndHashCode(callSuper = true)
public class RecurrentAttentionLayer extends SameDiffLayer {
    private long nIn;
    private long nOut;
    private int nHeads;
    private long headSize;
    private boolean projectInput;
    private Activation activation;
    private boolean hasBias;

    private static final String WEIGHT_KEY_QUERY_PROJECTION = "Wq";
    private static final String WEIGHT_KEY_KEY_PROJECTION = "Wk";
    private static final String WEIGHT_KEY_VALUE_PROJECTION = "Wv";
    private static final String WEIGHT_KEY_OUT_PROJECTION = "Wo";
    private static final String WEIGHT_KEY = SimpleRnnParamInitializer.WEIGHT_KEY;
    private static final String BIAS_KEY = SimpleRnnParamInitializer.BIAS_KEY;
    private static final String RECURRENT_WEIGHT_KEY = SimpleRnnParamInitializer.RECURRENT_WEIGHT_KEY;
    private int timeSteps;

    private RecurrentAttentionLayer(){/*No arg constructor for serialization*/}

    protected RecurrentAttentionLayer(Builder builder){
        super(builder);
        nIn = builder.nIn;
        nOut = builder.nOut;
        nHeads = builder.nHeads;
        headSize = builder.headSize == 0 ? nOut / nHeads : builder.headSize;
        projectInput = builder.projectInput;
        activation = builder.activation;
        hasBias = builder.hasBias;
    }

    @Override
    public InputPreProcessor getPreProcessorForInputType(InputType inputType) {
        return InputTypeUtil.getPreprocessorForInputTypeRnnLayers(inputType, RNNFormat.NCW, getLayerName());
    }

    @Override
    public void setNIn(InputType inputType, boolean override) {
        if (inputType == null || inputType.getType() != InputType.Type.RNN) {
            throw new IllegalStateException("Invalid input for Recurrent Attention layer (layer name = \"" + getLayerName()
                    + "\"): expect RNN input type with size > 0. Got: " + inputType);
        }

        if (nIn <= 0 || override) {
            InputType.InputTypeRecurrent r = (InputType.InputTypeRecurrent) inputType;
            this.nIn = r.getSize();
        }
    }

    @Override
    public InputType getOutputType(int layerIndex, InputType inputType) {
        if (inputType == null || inputType.getType() != InputType.Type.RNN) {
            throw new IllegalStateException("Invalid input for Recurrent Attention layer (layer index = " + layerIndex
                    + ", layer name = \"" + getLayerName() + "\"): expect RNN input type with size > 0. Got: "
                    + inputType);
        }

        InputType.InputTypeRecurrent itr = (InputType.InputTypeRecurrent) inputType;


        return InputType.recurrent(nOut, itr.getTimeSeriesLength());
    }

    @Override
    public void defineParameters(SDLayerParams params) {
        params.clear();

        params.addWeightParam(WEIGHT_KEY, nIn, nOut);
        params.addWeightParam(RECURRENT_WEIGHT_KEY, nOut, nOut);
        if(hasBias){
            params.addBiasParam(BIAS_KEY, nOut);
        }

        if(projectInput){
            params.addWeightParam(WEIGHT_KEY_QUERY_PROJECTION, nHeads, headSize, nOut);
            params.addWeightParam(WEIGHT_KEY_KEY_PROJECTION,   nHeads, headSize, nIn);
            params.addWeightParam(WEIGHT_KEY_VALUE_PROJECTION, nHeads, headSize, nIn);
            params.addWeightParam(WEIGHT_KEY_OUT_PROJECTION, nHeads * headSize, nOut);
        }
    }

    @Override
    public void initializeParameters(Map<String, INDArray> params) {
        try (MemoryWorkspace ws = Nd4j.getWorkspaceManager().scopeOutOfWorkspaces()) {
            for (Map.Entry<String, INDArray> e : params.entrySet()) {
                final String keyName = e.getKey();
                switch (keyName) {
                    case WEIGHT_KEY:
                        WeightInitUtil.initWeights(nIn, nOut, e.getValue().shape(), weightInit, null, 'c', e.getValue());
                        break;
                    case RECURRENT_WEIGHT_KEY:
                        WeightInitUtil.initWeights(nOut, nOut, e.getValue().shape(), weightInit, null, 'c', e.getValue());
                        break;
                    case BIAS_KEY:
                        e.getValue().assign(0);
                        break;
                    case WEIGHT_KEY_OUT_PROJECTION:
                        WeightInitUtil.initWeights(nIn, headSize, e.getValue().shape(), weightInit, null, 'c', e.getValue());
                        break;
                    default:
                        WeightInitUtil.initWeights(nHeads * headSize, nOut, e.getValue().shape(), weightInit, null, 'c', e.getValue());
                        break;
                }
            }
        }
    }

    @Override
    public void applyGlobalConfigToLayer(NeuralNetConfiguration.Builder globalConfig) {
        if (activation == null) {
            activation = SameDiffLayerUtils.fromIActivation(globalConfig.getActivationFn());
        }
    }

    @Override
    public void validateInput(INDArray input) {
        final long inputLength = input.size(2);
        Preconditions.checkArgument(inputLength == (long) this.timeSteps, "This layer only supports fixed length mini-batches. Expected %s time steps but got %s.", this.timeSteps, inputLength);
    }

    @Override
    public SDVariable defineLayer(SameDiff sameDiff, SDVariable layerInput, Map<String, SDVariable> paramTable, SDVariable mask) {
        final val W = paramTable.get(WEIGHT_KEY);
        final val R = paramTable.get(RECURRENT_WEIGHT_KEY);
        final val b = paramTable.get(BIAS_KEY);

        long[] shape = layerInput.getShape();
        Preconditions.checkState(shape != null, "Null shape for input placeholder");
        SDVariable[] inputSlices = sameDiff.unstack(layerInput, 2, (int)shape[2]);
        this.timeSteps = inputSlices.length;
        SDVariable[] outputSlices = new SDVariable[timeSteps];
        SDVariable prev = null;
        for (int i = 0; i < timeSteps; i++) {
            final val x_i = inputSlices[i];
            outputSlices[i] = x_i.mmul(W);
            if(hasBias){
                outputSlices[i] = outputSlices[i].add(b);
            }

            if(prev != null){
                SDVariable attn;
                if(projectInput){
                    val Wq = paramTable.get(WEIGHT_KEY_QUERY_PROJECTION);
                    val Wk = paramTable.get(WEIGHT_KEY_KEY_PROJECTION);
                    val Wv = paramTable.get(WEIGHT_KEY_VALUE_PROJECTION);
                    val Wo = paramTable.get(WEIGHT_KEY_OUT_PROJECTION);

                    attn = sameDiff.nn.multiHeadDotProductAttention(getLayerName()+"_attention_"+i, prev, layerInput, layerInput, Wq, Wk, Wv, Wo, mask, true);
                }else{
                    attn = sameDiff.nn.dotProductAttention(getLayerName()+"_attention_"+i, prev, layerInput, layerInput, mask, true);
                }

                attn = sameDiff.squeeze(attn, 2);

                outputSlices[i] = outputSlices[i].add(attn.mmul(R));
            }

            outputSlices[i] = activation.asSameDiff(sameDiff, outputSlices[i]);
            outputSlices[i] = sameDiff.expandDims(outputSlices[i], 2);
            prev = outputSlices[i];
        }
        return sameDiff.concat(2, outputSlices);
    }

    @Getter
    @Setter
    public static class Builder extends SameDiffLayer.Builder<RecurrentAttentionLayer.Builder> {

        /**
         * Number of inputs to the layer (input size)
         */
        private int nIn;

        /**
         * Number of outputs (output size)
         */
        private int nOut;

        /**
         * Number of Attention Heads
         */
        private int nHeads;

        /**
         * Size of attention heads
         */
        private int headSize;

        /**
         * Project input before applying attention or not.
         */
        private boolean projectInput = true;

        /**
         * If true (default is true) the layer will have a bias
         */
        private boolean hasBias = true;

        /**
         * Activation function for the layer
         */
        private Activation activation = Activation.TANH;

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
         * Number of Attention Heads
         */
        public Builder nHeads(int nHeads){
            this.nHeads = nHeads;
            return this;
        }

        /**
         * Size of attention heads
         */
        public Builder headSize(int headSize){
            this.headSize = headSize;
            return this;
        }

        /**
         * Project input before applying attention or not.
         */
        public Builder projectInput(boolean projectInput){
            this.projectInput = projectInput;
            return this;
        }

        /**
         * @param hasBias If true (default is true) the layer will have a bias
         */
        public Builder hasBias(boolean hasBias) {
            this.hasBias = hasBias;
            return this;
        }

        /**
         * @param activation Activation function for the layer
         */
        public Builder activation(Activation activation) {
            this.activation = activation;
            return this;
        }

        @Override
        @SuppressWarnings("unchecked")
        public RecurrentAttentionLayer build() {
            Preconditions.checkArgument(this.projectInput || this.nHeads == 1, "projectInput must be true when nHeads != 1");
            Preconditions.checkArgument(this.projectInput || nIn == nOut, "nIn must be equal to nOut when projectInput is false");
            Preconditions.checkArgument(!this.projectInput || nOut != 0, "nOut must be specified when projectInput is true");
            Preconditions.checkArgument(this.nOut % nHeads == 0 || headSize > 0, "nOut isn't divided by nHeads cleanly. Specify the headSize manually.");
            return new RecurrentAttentionLayer(this);
        }
    }
}
