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
package org.deeplearning4j.nn.conf.graph;

import com.google.common.base.Preconditions;
import lombok.*;
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

    @Builder
    public AttentionVertex(long nInKeys, long nInValues, long nInQueries, long nOut, long headSize, int nHeads, boolean projectInput, WeightInit weightInit) {
        this.nHeads = nHeads == 0 ? 1 : nHeads;
        this.weightInit = weightInit == null ? WeightInit.XAVIER : weightInit;
        Preconditions.checkArgument(nOut > 0, "You have to set nOut");
        Preconditions.checkArgument(nInKeys > 0, "You have to set nInKeys");
        Preconditions.checkArgument(nInQueries > 0, "You have to set nInQueries");
        Preconditions.checkArgument(nInValues > 0, "You have to set nInValues");
        Preconditions.checkArgument(headSize > 0 || nOut % this.nHeads == 0, "You have to set a head size if nOut isn't cleanly divided by nHeads");
        Preconditions.checkArgument(projectInput || (nInQueries == nInKeys && nInKeys == nInValues  && nInValues == nOut && nHeads == 1), "You may only disable projectInput if all nIn* equal to nOut and you want to use only a single attention head");
        this.nInKeys = nInKeys;
        this.nInValues = nInValues;
        this.nInQueries = nInQueries;
        this.nOut = nOut;
        this.headSize = headSize == 0 ? nOut / nHeads : headSize;
        this.projectInput = projectInput;
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
    public SDVariable defineVertex(SameDiff sameDiff, Map<String, SDVariable> layerInput, Map<String, SDVariable> paramTable) {
        final SDVariable queries = layerInput.get("queries");
        final SDVariable keys = layerInput.get("keys");
        final SDVariable values = layerInput.get("values");

        if(projectInput){
            val Wq = paramTable.get(WEIGHT_KEY_QUERY_PROJECTION);
            val Wk = paramTable.get(WEIGHT_KEY_KEY_PROJECTION);
            val Wv = paramTable.get(WEIGHT_KEY_VALUE_PROJECTION);
            val Wo = paramTable.get(WEIGHT_KEY_OUT_PROJECTION);

            return sameDiff.nn.multiHeadDotProductAttention(getLayerName(), queries, keys, values, Wq, Wk, Wv, Wo, null, true);
        }else{
            return sameDiff.nn.dotProductAttention(getLayerName(), queries, keys, values, null, true);
        }
    }
}
