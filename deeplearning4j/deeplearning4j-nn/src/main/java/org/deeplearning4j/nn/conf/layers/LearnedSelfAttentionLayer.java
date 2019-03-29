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
import org.deeplearning4j.nn.api.MaskState;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.samediff.SDLayerParams;
import org.deeplearning4j.nn.conf.layers.samediff.SameDiffLayer;
import org.deeplearning4j.nn.weights.WeightInitUtil;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;

import java.util.Map;

/**
 * Implements Dot Product Self Attention with learned queries
 *
 * Takes in RNN style input in the shape of [batchSize, features, timesteps]
 * and applies dot product attention using learned queries.
 *
 * @author Paul Dubs
 */
@Data
@EqualsAndHashCode(callSuper = true)
public class LearnedSelfAttentionLayer extends SameDiffLayer {
    private long nIn;
    private long nOut;
    private int nHeads;
    private long headSize;
    private boolean projectInput;
    private int numQueries;

    private static final String WEIGHT_KEY_QUERY_PROJECTION = "Wq";
    private static final String WEIGHT_KEY_KEY_PROJECTION = "Wk";
    private static final String WEIGHT_KEY_VALUE_PROJECTION = "Wv";
    private static final String WEIGHT_KEY_OUT_PROJECTION = "Wo";
    private static final String WEIGHT_QUERIES = "Q";

    private LearnedSelfAttentionLayer(){/*No arg constructor for serialization*/}

    protected LearnedSelfAttentionLayer(Builder builder){
        super(builder);
        nIn = builder.nIn;
        nOut = builder.nOut;
        nHeads = builder.nHeads;
        headSize = builder.headSize == 0 ? nOut / nHeads : builder.headSize;
        projectInput = builder.projectInput;
        numQueries = builder.numQueries;
    }

    @Override
    public InputPreProcessor getPreProcessorForInputType(InputType inputType) {
        return InputTypeUtil.getPreprocessorForInputTypeRnnLayers(inputType, getLayerName());
    }

    @Override
    public void setNIn(InputType inputType, boolean override) {
        if (inputType == null || inputType.getType() != InputType.Type.RNN) {
            throw new IllegalStateException("Invalid input for Learned Self Attention layer (layer name = \"" + getLayerName()
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
            throw new IllegalStateException("Invalid input for Learned Self Attention layer (layer index = " + layerIndex
                    + ", layer name = \"" + getLayerName() + "\"): expect RNN input type with size > 0. Got: "
                    + inputType);
        }

        if(projectInput){
            return InputType.recurrent(nOut, numQueries);
        }else{
            return InputType.recurrent(nIn, numQueries);
        }
    }

    @Override
    public void defineParameters(SDLayerParams params) {
        params.clear();

        params.addWeightParam(WEIGHT_QUERIES, 1, nIn, numQueries);

        if(projectInput){
            params.addWeightParam(WEIGHT_KEY_QUERY_PROJECTION, nHeads, headSize, nIn);
            params.addWeightParam(WEIGHT_KEY_KEY_PROJECTION,   nHeads, headSize, nIn);
            params.addWeightParam(WEIGHT_KEY_VALUE_PROJECTION, nHeads, headSize, nIn);
            params.addWeightParam(WEIGHT_KEY_OUT_PROJECTION, nHeads * headSize, nOut);
        }
    }

    @Override
    public void initializeParameters(Map<String, INDArray> params) {
        try (MemoryWorkspace ws = Nd4j.getWorkspaceManager().scopeOutOfWorkspaces()) {
            for (Map.Entry<String, INDArray> e : params.entrySet()) {
                if(e.getKey().equals(WEIGHT_KEY_OUT_PROJECTION)){
                    WeightInitUtil.initWeights(nIn, headSize, e.getValue().shape(), weightInit, null, 'c', e.getValue());
                }else if(e.getKey().equals(WEIGHT_QUERIES)){
                    WeightInitUtil.initWeights(nIn, numQueries, e.getValue().shape(), weightInit, null, 'c', e.getValue());
                }else{
                    WeightInitUtil.initWeights(nHeads * headSize, nOut, e.getValue().shape(), weightInit, null, 'c', e.getValue());
                }
            }
        }
    }


    @Override
    public SDVariable defineLayer(SameDiff sameDiff, SDVariable layerInput, Map<String, SDVariable> paramTable, SDVariable mask) {
        val baseQueries = paramTable.get(WEIGHT_QUERIES);
        val queries = sameDiff.f().tile(baseQueries, new int[]{(int) layerInput.getShape()[0], 1, 1});

        if(projectInput){
            val Wq = paramTable.get(WEIGHT_KEY_QUERY_PROJECTION);
            val Wk = paramTable.get(WEIGHT_KEY_KEY_PROJECTION);
            val Wv = paramTable.get(WEIGHT_KEY_VALUE_PROJECTION);
            val Wo = paramTable.get(WEIGHT_KEY_OUT_PROJECTION);

            return sameDiff.nn.multiHeadDotProductAttention(getLayerName(), queries, layerInput, layerInput, Wq, Wk, Wv, Wo, mask, true);
        }else{
            return sameDiff.nn.dotProductAttention(getLayerName(), queries, layerInput, layerInput, mask, true);
        }
    }

    @Override
    public Pair<INDArray, MaskState> feedForwardMaskArray(INDArray maskArray, MaskState currentMaskState, int minibatchSize) {
        // No further mask propagation here, as the results have taken any mask into account, like in a global pooling layer
        return null;
    }

    public static class Builder extends SameDiffLayer.Builder<LearnedSelfAttentionLayer.Builder> {

        /**
         * Number of inputs to the layer (input size)
         */
        @Getter
        @Setter
        private int nIn;

        /**
         * Number of outputs (output size)
         */
        @Getter
        @Setter
        private int nOut;

        /**
         * Number of Attention Heads
         */
        @Getter
        @Setter
        private int nHeads;

        /**
         * Size of attention heads
         */
        @Getter
        @Setter
        private int headSize;

        /**
         * Project input before applying attention or not.
         */
        @Getter
        @Setter
        private boolean projectInput;


        /**
         * Number of queries to learn
         */
        @Getter
        @Setter
        private int numQueries;

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
         * Number of queries to learn
         */
        public Builder numQueries(int numQueries){
            this.numQueries = numQueries;
            return this;
        }

        @Override
        @SuppressWarnings("unchecked")
        public LearnedSelfAttentionLayer build() {
            if(!this.projectInput && this.nHeads != 1){ throw new IllegalArgumentException("projectInput must be true when nHeads != 1"); }
            if(!this.projectInput && nIn != nOut){ throw new IllegalArgumentException("nIn must be equal to nOut when projectInput is false"); }
            if(this.projectInput && nOut == 0){ throw new IllegalArgumentException("nOut must be specified when projectInput is true"); }
            if(this.nOut % nHeads != 0 && headSize == 0){ throw new IllegalArgumentException("nOut isn't divided by nHeads cleanly. Specify the headSize manually."); }
            if(this.numQueries == 0){ throw new IllegalArgumentException("You must set numQueries."); }

            return new LearnedSelfAttentionLayer(this);
        }
    }
}
