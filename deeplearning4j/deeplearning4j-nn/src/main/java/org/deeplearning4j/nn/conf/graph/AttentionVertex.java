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
package org.deeplearning4j.nn.conf.graph;

import org.nd4j.shade.guava.base.Preconditions;
import lombok.*;
import org.deeplearning4j.nn.api.MaskState;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.inputs.InvalidInputTypeException;
import org.deeplearning4j.nn.conf.layers.samediff.SDVertexParams;
import org.deeplearning4j.nn.conf.layers.samediff.SameDiffVertex;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.nn.weights.WeightInitUtil;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.common.primitives.Pair;

import java.util.Map;

/**
 * Implements Dot Product Attention using the given inputs.
 * For Timestep-wise Self-Attention use the same value for all three inputs.
 *
 * @see org.nd4j.autodiff.samediff.ops.SDNN#multiHeadDotProductAttention(String, SDVariable, SDVariable, SDVariable, SDVariable, SDVariable, SDVariable, SDVariable, SDVariable, boolean, boolean)
 * @see org.nd4j.autodiff.samediff.ops.SDNN#dotProductAttention(String, SDVariable, SDVariable, SDVariable, SDVariable, boolean, boolean)
 *
 * @author Paul Dubs
 */
@NoArgsConstructor
@Data
@EqualsAndHashCode(callSuper = true)
@ToString
public class AttentionVertex extends SameDiffVertex {
    private long nInKeys = 0;
    private long nInValues = 0;
    private long nInQueries = 0;
    private long nOut = 0;
    private long headSize = 0;
    private int nHeads = 1;
    private boolean projectInput;
    protected WeightInit weightInit;

    private static final String WEIGHT_KEY_QUERY_PROJECTION = "Wq";
    private static final String WEIGHT_KEY_KEY_PROJECTION = "Wk";
    private static final String WEIGHT_KEY_VALUE_PROJECTION = "Wv";
    private static final String WEIGHT_KEY_OUT_PROJECTION = "Wo";

    protected AttentionVertex(Builder builder) {
        this.nInKeys = builder.nInKeys;
        this.nInValues = builder.nInValues;
        this.nInQueries = builder.nInQueries;
        this.nOut = builder.nOut;
        this.headSize = builder.headSize;
        this.projectInput = builder.projectInput;
        this.nHeads = builder.nHeads;
        this.weightInit = builder.weightInit;
    }

    @Override
    public AttentionVertex clone() {
        AttentionVertex av = new AttentionVertex();
        av.nInKeys = nInKeys;
        av.nInValues = nInValues;
        av.nInQueries = nInQueries;
        av.nOut = nOut;
        av.headSize = headSize;
        av.nHeads = nHeads;
        av.projectInput = projectInput;
        av.weightInit = weightInit;
        return av;
    }

    @Override
    public InputType getOutputType(int layerIndex, InputType... vertexInputs) throws InvalidInputTypeException {
        InputType.InputTypeRecurrent queries = (InputType.InputTypeRecurrent) vertexInputs[0];

        if(projectInput){
            return InputType.recurrent(nOut, queries.getTimeSeriesLength());
        }else{
            return InputType.recurrent(nInValues, queries.getTimeSeriesLength());
        }
    }

    @Override
    public void defineParametersAndInputs(SDVertexParams params) {
        params.clear();

        params.defineInputs("queries", "keys", "values");

        if(projectInput){
            params.addWeightParam(WEIGHT_KEY_QUERY_PROJECTION, nHeads, headSize, nInQueries);
            params.addWeightParam(WEIGHT_KEY_KEY_PROJECTION,   nHeads, headSize, nInKeys);
            params.addWeightParam(WEIGHT_KEY_VALUE_PROJECTION, nHeads, headSize, nInValues);
            params.addWeightParam(WEIGHT_KEY_OUT_PROJECTION, nHeads * headSize, nOut);
        }
    }

    @Override
    public void initializeParameters(Map<String, INDArray> params) {
        try (MemoryWorkspace ws = Nd4j.getWorkspaceManager().scopeOutOfWorkspaces()) {
            for (Map.Entry<String, INDArray> e : params.entrySet()) {
                switch (e.getKey()) {
                    case WEIGHT_KEY_QUERY_PROJECTION:
                        WeightInitUtil.initWeights(nInQueries, headSize, e.getValue().shape(), weightInit, null, 'c', e.getValue());
                        break;
                    case WEIGHT_KEY_KEY_PROJECTION:
                        WeightInitUtil.initWeights(nInKeys, headSize, e.getValue().shape(), weightInit, null, 'c', e.getValue());
                        break;
                    case WEIGHT_KEY_VALUE_PROJECTION:
                        WeightInitUtil.initWeights(nInValues, headSize, e.getValue().shape(), weightInit, null, 'c', e.getValue());
                        break;
                    case WEIGHT_KEY_OUT_PROJECTION:
                        WeightInitUtil.initWeights(nHeads * headSize, nOut, e.getValue().shape(), weightInit, null, 'c', e.getValue());
                        break;
                }
            }
        }
    }

    @Override
    public Pair<INDArray, MaskState> feedForwardMaskArrays(INDArray[] maskArrays, MaskState currentMaskState, int minibatchSize) {
        if(maskArrays != null) {
            if(maskArrays[0] == null) {
                // Queries are unmasked, we don't need to pass on any mask
                return null;
            }else{
                // Queries are masked, keep the masking going
                return Pair.of(maskArrays[0], currentMaskState);
            }
        }else {
            return Pair.of(null, currentMaskState);
        }
    }

    @Override
    public SDVariable defineVertex(SameDiff sameDiff, Map<String, SDVariable> layerInput, Map<String, SDVariable> paramTable, Map<String, SDVariable> maskVars) {
        final SDVariable queries = layerInput.get("queries");
        final SDVariable keys = layerInput.get("keys");
        final SDVariable values = layerInput.get("values");
        final SDVariable mask = maskVars != null ? sameDiff.min(maskVars.get("keys"), maskVars.get("values")): null;

        SDVariable attention;
        if(projectInput){
            val Wq = paramTable.get(WEIGHT_KEY_QUERY_PROJECTION);
            val Wk = paramTable.get(WEIGHT_KEY_KEY_PROJECTION);
            val Wv = paramTable.get(WEIGHT_KEY_VALUE_PROJECTION);
            val Wo = paramTable.get(WEIGHT_KEY_OUT_PROJECTION);

            attention = sameDiff.nn.multiHeadDotProductAttention(getLayerName(), queries, keys, values, Wq, Wk, Wv, Wo, mask, true);
        }else{
            attention = sameDiff.nn.dotProductAttention(getLayerName(), queries, keys, values, mask, true);
        }

        if(maskVars != null){
            return attention.mul(sameDiff.expandDims(maskVars.get("queries"), 1));
        }else{
            return attention;
        }
    }

    @Getter
    @Setter
    public static class Builder {
        /**
         * Size of Keys
         */
        private long nInKeys = 0;

        /**
         * Size of Values
         */
        private long nInValues = 0;

        /**
         * Size of Queries
         */
        private long nInQueries = 0;

        /**
         * Output Size
         */
        private long nOut = 0;

        /**
         * Size of Attention Heads
         */
        private long headSize = 0;

        /**
         * Number of Attention Heads
         */
        private int nHeads = 1;

        /**
         * Toggle to enable / disable projection of inputs (key, values, queries).
         *
         * Works only if input size is identical for all AND only one head is used AND output size is
         * identical to input size
         */
        private boolean projectInput;

        /**
         * Weight initialization scheme
         */
        protected WeightInit weightInit;


        /**
         * Size of Keys
         */
        public Builder nInKeys(long nInKeys) {
            this.nInKeys = nInKeys;
            return this;
        }

        /**
         * Size of Queries
         */
        public Builder nInQueries(long nInQueries) {
            this.nInQueries = nInQueries;
            return this;
        }

        /**
         * Size of Values
         */
        public Builder nInValues(long nInValues) {
            this.nInValues = nInValues;
            return this;
        }

        /**
         * Size of Attention Heads
         */
        public Builder headSize(long headSize){
            this.headSize = headSize;
            return this;
        }

        /**
         * Number of Attention Heads
         */
        public Builder nHeads(int nHeads) {
            this.nHeads = nHeads;
            return this;
        }

        /**
         * Output Size
         */
        public Builder nOut(long nOut) {
            this.nOut = nOut;
            return this;
        }

        /**
         * Weight initialization scheme
         */
        public Builder weightInit(WeightInit weightInit){
            this.weightInit = weightInit;
            return this;
        }

        /**
         * Toggle to enable / disable projection of inputs (key, values, queries).
         *
         * Works only if input size is identical for all AND only one head is used AND output size is
         * identical to input size
         */
        public Builder projectInput(boolean projectInput){
            this.projectInput = projectInput;
            return this;
        }

        public AttentionVertex build(){
            this.nHeads = nHeads == 0 ? 1 : nHeads;
            this.weightInit = weightInit == null ? WeightInit.XAVIER : weightInit;
            Preconditions.checkArgument(nOut > 0, "You have to set nOut");
            Preconditions.checkArgument(nInKeys > 0, "You have to set nInKeys");
            Preconditions.checkArgument(nInQueries > 0, "You have to set nInQueries");
            Preconditions.checkArgument(nInValues > 0, "You have to set nInValues");
            Preconditions.checkArgument(headSize > 0 || nOut % this.nHeads == 0, "You have to set a head size if nOut isn't cleanly divided by nHeads");
            Preconditions.checkArgument(projectInput || (nInQueries == nInKeys && nInKeys == nInValues  && nInValues == nOut && nHeads == 1), "You may only disable projectInput if all nIn* equal to nOut and you want to use only a single attention head");
            this.headSize = headSize == 0 ? nOut / nHeads : headSize;

            return new AttentionVertex(this);
        }
    }
}
