/*
 *  ******************************************************************************
 *  *
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
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.RNNFormat;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.samediff.SDLayerParams;
import org.deeplearning4j.nn.conf.layers.samediff.SameDiffLayer;
import org.deeplearning4j.nn.weights.WeightInitUtil;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Map;

/**
 * Implements Dot Product Self Attention
 *
 * Takes in RNN style input in the shape of [batchSize, features, timesteps]
 * and applies dot product attention using each timestep as the query.
 *
 * The output will be in the shape of [batchSize, nOut, timesteps].
 *
 * Attention implemented as in
 * Attention is all you need by Vaswani et al. [arXiv:1706.03762], pp. 4,5
 *
 * @see LearnedSelfAttentionLayer
 * @see RecurrentAttentionLayer
 * @see org.nd4j.linalg.api.ops.impl.transforms.custom.MultiHeadDotProductAttention
 *
 * @author Paul Dubs
 */
@Data
@EqualsAndHashCode(callSuper = true)
public class SelfAttentionLayer extends SameDiffLayer {
    private long nIn;
    private long nOut;
    private int nHeads;
    private long headSize;
    private boolean projectInput;

    private static final String WEIGHT_KEY_QUERY_PROJECTION = "Wq";
    private static final String WEIGHT_KEY_KEY_PROJECTION = "Wk";
    private static final String WEIGHT_KEY_VALUE_PROJECTION = "Wv";
    private static final String WEIGHT_KEY_OUT_PROJECTION = "Wo";

    private SelfAttentionLayer(){/*No arg constructor for serialization*/}

    protected SelfAttentionLayer(Builder builder){
        super(builder);
        nIn = builder.nIn;
        nOut = builder.nOut;
        nHeads = builder.nHeads;
        headSize = builder.headSize == 0 ? nOut / nHeads : builder.headSize;
        projectInput = builder.projectInput;
    }

    @Override
    public InputPreProcessor getPreProcessorForInputType(InputType inputType) {
        return InputTypeUtil.getPreprocessorForInputTypeRnnLayers(inputType, RNNFormat.NCW,getLayerName());
    }

    @Override
    public void setNIn(InputType inputType, boolean override) {
        if (inputType == null || inputType.getType() != InputType.Type.RNN) {
            throw new IllegalStateException("Invalid input for Self Attention layer (layer name = \"" + getLayerName()
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
            throw new IllegalStateException("Invalid input for Self Attention layer (layer index = " + layerIndex
                    + ", layer name = \"" + getLayerName() + "\"): expect RNN input type with size > 0. Got: "
                    + inputType);
        }

        InputType.InputTypeRecurrent itr = (InputType.InputTypeRecurrent) inputType;

        if(projectInput){
            return InputType.recurrent(nOut, itr.getTimeSeriesLength());
        }else{
            return InputType.recurrent(nIn, itr.getTimeSeriesLength());
        }
    }

    @Override
    public void defineParameters(SDLayerParams params) {
        params.clear();

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
                }else{
                    WeightInitUtil.initWeights(nHeads * headSize, nOut, e.getValue().shape(), weightInit, null, 'c', e.getValue());
                }
            }
        }
    }


    @Override
    public SDVariable defineLayer(SameDiff sameDiff, SDVariable layerInput, Map<String, SDVariable> paramTable, SDVariable mask) {
        if(projectInput){
            val Wq = paramTable.get(WEIGHT_KEY_QUERY_PROJECTION);
            val Wk = paramTable.get(WEIGHT_KEY_KEY_PROJECTION);
            val Wv = paramTable.get(WEIGHT_KEY_VALUE_PROJECTION);
            val Wo = paramTable.get(WEIGHT_KEY_OUT_PROJECTION);

            return sameDiff.nn.multiHeadDotProductAttention(getLayerName(), layerInput, layerInput, layerInput, Wq, Wk, Wv, Wo, mask, true);
        }else{
            return sameDiff.nn.dotProductAttention(getLayerName(), layerInput, layerInput, layerInput, mask, true);
        }
    }


    @Getter
    @Setter
    public static class Builder extends SameDiffLayer.Builder<SelfAttentionLayer.Builder> {

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
        private boolean projectInput;

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

        @Override
        @SuppressWarnings("unchecked")
        public SelfAttentionLayer build() {
            Preconditions.checkArgument(this.projectInput || this.nHeads == 1, "projectInput must be true when nHeads != 1");
            Preconditions.checkArgument(this.projectInput || nIn == nOut, "nIn must be equal to nOut when projectInput is false");
            Preconditions.checkArgument(!this.projectInput || nOut != 0, "nOut must be specified when projectInput is true");
            Preconditions.checkArgument(this.nOut % nHeads == 0 || headSize > 0, "nOut isn't divided by nHeads cleanly. Specify the headSize manually.");
            return new SelfAttentionLayer(this);
        }
    }
}
