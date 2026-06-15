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


    /**
     * Create a backpropagation op for dot product attention.
     *
     * @param sameDiff The SameDiff instance
     * @param queries Query tensor [batch, Tq, dim]
     * @param values Value tensor [batch, Tv, dim]
     * @param keys Key tensor [batch, Tv, dim]
     * @param eps Gradient from upstream [batch, Tq, dim]
     * @param queryMask Query mask [batch, Tq] or null
     * @param valueMask Value mask [batch, Tv] or null
     * @param attentionScoresOut Forward pass attention output
     * @param attentionScoreWeights Forward pass attention weights (after softmax)
     * @param attentionScoresLogits Forward pass attention logits (before softmax)
     * @param dropoutMask Forward pass dropout mask or null
     * @param scaleFactor Scale factor for attention scores
     * @param dropout Dropout probability
     * @param useCausalMask Whether causal mask was applied
     * @param training Whether in training mode
     */
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
                                   SDVariable dropoutMask,
                                   double scaleFactor,
                                   double dropout,
                                   boolean useCausalMask,
                                   boolean training) {
        super(null, sameDiff, inputs(sameDiff, queries, values, keys, attentionScoresOut,
                attentionScoreWeights, attentionScoresLogits, eps, dropoutMask, queryMask, valueMask), false);
        // T_ARG order: scale, dropout (same as forward pass)
        addTArgument(scaleFactor, dropout);
        // B_ARG order: useCausalMask, training (same as forward pass)
        addBArgument(useCausalMask, training);
    }

    private static SDVariable[] inputs(SameDiff sd,
                                       SDVariable queries,
                                       SDVariable values,
                                       SDVariable keys,
                                       SDVariable attentionScoresOut,
                                       SDVariable attentionScoreWeights,
                                       SDVariable attentionScoresLogits,
                                       SDVariable eps,
                                       SDVariable dropoutMask,
                                       SDVariable queryMask,
                                       SDVariable valueMask) {
        // C++ expects inputs in this order:
        // 0: queries, 1: values, 2: keys
        // 3: attentionScoresOut, 4: attentionScoreWeights, 5: attentionScoreLogits
        // 6: eps
        // 7: dropoutMask (if exists, else empty)
        // 8: queryMask, 9: valueMask
        List<SDVariable> inputs = new ArrayList<>();
        inputs.add(queries);                                    // 0
        inputs.add(values);                                     // 1
        inputs.add(keys == null ? values : keys);               // 2
        inputs.add(attentionScoresOut);                         // 3
        inputs.add(attentionScoreWeights);                      // 4
        inputs.add(attentionScoresLogits);                      // 5
        inputs.add(eps);                                        // 6
        // Always add dropout mask at position 7 (empty if null) to keep mask positions consistent
        inputs.add(dropoutMask == null ? sd.constant(Nd4j.empty(queries.dataType())) : dropoutMask);  // 7
        inputs.add(queryMask == null ? sd.constant(Nd4j.empty(queries.dataType())) : queryMask);      // 8
        inputs.add(valueMask == null ? sd.constant(Nd4j.empty(queries.dataType())) : valueMask);      // 9
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
    public int getNumOutputs() {
        return 3;
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