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
package org.deeplearning4j.nn.conf.graph;

import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.Getter;
import lombok.Setter;
import org.deeplearning4j.nn.api.MaskState;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.inputs.InvalidInputTypeException;
import org.deeplearning4j.nn.conf.layers.samediff.SDVertexParams;
import org.deeplearning4j.nn.conf.layers.samediff.SameDiffVertex;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.primitives.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Map;
@Data
@EqualsAndHashCode(callSuper = false)
public class DotProductAttentionVertex extends SameDiffVertex {

    private double scaleFactor;
    private double dropoutProbability;
    private boolean useCausalMask;
    private boolean training;
    private long nIn;
    private long nOut;

    public DotProductAttentionVertex() {
    }

    public DotProductAttentionVertex(Builder builder) {
        this.scaleFactor = builder.scaleFactor;
        this.dropoutProbability = builder.dropoutProbability;
        this.useCausalMask = builder.useCausalMask;
        this.training = builder.training;
        this.nIn = builder.nIn;
        this.nOut = builder.nOut;
    }

    @Override
    public GraphVertex clone() {
        DotProductAttentionVertex dotProductAttentionVertex = new DotProductAttentionVertex();
        dotProductAttentionVertex.scaleFactor = scaleFactor;
        dotProductAttentionVertex.dropoutProbability = dropoutProbability;
        dotProductAttentionVertex.useCausalMask = useCausalMask;
        dotProductAttentionVertex.training = training;
        dotProductAttentionVertex.nIn = nIn;
        dotProductAttentionVertex.nOut = nOut;
        return dotProductAttentionVertex;
    }

    @Override
    public SDVariable defineVertex(SameDiff sameDiff, Map<String, SDVariable> layerInput, Map<String, SDVariable> paramTable, Map<String, SDVariable> maskVars) {
        final SDVariable queries = layerInput.get("queries");
        final SDVariable keys = layerInput.get("keys");
        final SDVariable values = layerInput.get("values");
        final SDVariable qMask = maskVars  != null && maskVars.containsKey("qMask") ? maskVars.get("qMask") : null;
        final SDVariable vMask = maskVars != null  && maskVars.containsKey("vMask")? maskVars.get("vMask") : null;
        return sameDiff.nn.dotProductAttentionV2(queries,values,keys,qMask,vMask,scaleFactor,dropoutProbability,useCausalMask,training);

    }

    @Override
    public void defineParametersAndInputs(SDVertexParams params) {
        params.clear();

        params.defineInputs("queries", "keys", "values");

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
    public void initializeParameters(Map<String, INDArray> params) {

    }

    @Override
    public InputType getOutputType(int layerIndex, InputType... vertexInputs) throws InvalidInputTypeException {
        InputType.InputTypeRecurrent queries = (InputType.InputTypeRecurrent) vertexInputs[0];
        return InputType.recurrent(nIn, queries.getTimeSeriesLength());

    }
    @Getter
    @Setter
    public static class Builder  {

        private double scaleFactor;
        private double dropoutProbability;
        private boolean useCausalMask;
        private boolean training;
        private long nIn;
        private long nOut;

        public Builder nIn(long nIn) {
            this.nIn = nIn;
            return this;
        }

        public Builder nOut(long nOut) {
            this.nOut = nOut;
            return this;
        }

        public Builder training(boolean training) {
            this.training = training;
            return this;
        }

        public Builder dropoutProbability(double dropoutProbability) {
            this.dropoutProbability = dropoutProbability;
            return this;
        }

        /**
         * @param scaleFactor Whether to scale the input or not.
         *               Defaults to true.
         */
        public Builder scale(double scaleFactor) {
            this.scaleFactor = scaleFactor;
            return this;
        }

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


        public Builder useCausalMask(boolean useCausalMask) {
            this.useCausalMask = useCausalMask;
            return this;
        }


        @SuppressWarnings("unchecked")
        public DotProductAttentionVertex build() {
            return new DotProductAttentionVertex(this);
        }
    }
}
