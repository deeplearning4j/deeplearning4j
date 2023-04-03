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

package org.nd4j.linalg.api.ops.impl.transforms.custom;

import lombok.NoArgsConstructor;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


@NoArgsConstructor
public class DotProductAttentionV2Bp extends DynamicCustomOp {


    public DotProductAttentionV2Bp(SameDiff sameDiff,
                                   SDVariable queries,
                                   SDVariable values,
                                   SDVariable keys,
                                   SDVariable eps,
                                   SDVariable queryMask,
                                   SDVariable valueMask,
                                   SDVariable attentionScoresOut,
                                   SDVariable attentionScoreWeights,
                                   SDVariable attentionScoresLogits,
                                   SDVariable dropoutWeights,
                                   double scaleFactor,
                                   double dropout,
                                   int scoreMode,
                                   boolean useCausalMask,
                                   boolean withWeights,
                                   boolean training) {
        super(null, sameDiff,inputs(sameDiff,queries,values,keys,attentionScoresOut,attentionScoreWeights,attentionScoresLogits,eps,queryMask,valueMask,dropoutWeights), false);
        addIArgument(scoreMode);

        addTArgument(scaleFactor);
        addTArgument(dropout);
        addBArgument(useCausalMask);
        addBArgument(training);
    }

    private static SDVariable[] inputs(SameDiff sd,
                                       SDVariable queries,
                                       SDVariable values,
                                       SDVariable keys,
                                       SDVariable attentionScoresOut,
                                       SDVariable attentionScoreWeights,
                                       SDVariable attentionScoresLogits,
                                       SDVariable eps,
                                       SDVariable queryMask,
                                       SDVariable valueMask,
                                       SDVariable dropoutWeights) {
        List<SDVariable> inputs = new ArrayList<>();
        inputs.add(queries);
        inputs.add(values);
        inputs.add(keys == null ? values : keys);
        inputs.add(attentionScoresOut);
        inputs.add(attentionScoreWeights);
        inputs.add(attentionScoresLogits);
        inputs.add(eps);
        if(dropoutWeights != null) {
            inputs.add(dropoutWeights);
        }
        inputs.add(queryMask == null ? sd.constant(Nd4j.empty(queries.dataType())) : queryMask);
        inputs.add(valueMask == null ? sd.constant(Nd4j.empty(queries.dataType())) : valueMask);
        return inputs.toArray(new SDVariable[inputs.size()]);

    }


    @Override
    public String opName() {
        return "dot_product_attention_v2_bp";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> grad){
        throw new UnsupportedOperationException("Differentiation of " + getClass().getName() + " not supported");
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        DataType first = dataTypes.get(0);
        for( int i = 0; i < dataTypes.size(); i++) {
            Preconditions.checkState(dataTypes.get(i).isFPType(), "Input %s datatype must be a floating point type, got datypes %s", dataTypes);
            if(i > 0){
                Preconditions.checkState(first == dataTypes.get(i), "All datatypes must be same type, got input datatypes %s", dataTypes);
            }
        }

        return Arrays.asList(first, first, first);
    }
}