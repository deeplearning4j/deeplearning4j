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
import lombok.NonNull;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;


@NoArgsConstructor
public class DotProductAttentionV2 extends DynamicCustomOp {
    private boolean useCausalMask;
    private boolean training;
    private double scaleFactor = 1.0;
    private double dropout = 0.0;

    private SDVariable queryMask;
    private SDVariable valueMask;

    /**
     * Create a dot product attention operation.
     *
     * @param queries Query tensor [batch, Tq, dim] or [Tq, dim]
     * @param values Value tensor [batch, Tv, dim] or [Tv, dim]
     * @param keys Key tensor [batch, Tv, dim] or [Tv, dim], or null to use values
     * @param queryMask Query mask [batch, Tq] or null
     * @param valueMask Value mask [batch, Tv] or null
     * @param scaleFactor Scale factor for attention scores (typically 1/sqrt(dim))
     * @param dropoutProbability Dropout probability (0 = no dropout)
     * @param useCausalMask Whether to apply causal (lower triangular) mask
     * @param training Whether in training mode (affects dropout)
     */
    public DotProductAttentionV2(INDArray queries, INDArray values, INDArray keys, INDArray queryMask, INDArray valueMask,
                                 double scaleFactor, double dropoutProbability, boolean useCausalMask, boolean training) {
        super(wrapFilterNull(queries, values, keys, queryMask, valueMask), null);
        this.scaleFactor = scaleFactor;
        this.dropout = dropoutProbability;
        this.useCausalMask = useCausalMask;
        this.training = training;
        // T_ARG order: scale, dropout
        addTArgument(scaleFactor, dropoutProbability);
        // B_ARG order: useCausalMask, training
        addBArgument(useCausalMask, training);
    }

    /**
     * Create a dot product attention operation for SameDiff.
     */
    public DotProductAttentionV2(SameDiff sd, SDVariable queries, SDVariable values, SDVariable keys,
                                 SDVariable queryMask, SDVariable valueMask,
                                 double scaleFactor, double dropoutProbability,
                                 boolean useCausalMask, boolean training) {
        super(null, sd, inputs(sd, queries, values, keys, queryMask, valueMask), false);
        this.scaleFactor = scaleFactor;
        this.dropout = dropoutProbability;
        this.useCausalMask = useCausalMask;
        this.training = training;
        this.queryMask = queryMask;
        this.valueMask = valueMask;
        // T_ARG order: scale, dropout
        addTArgument(scaleFactor, dropoutProbability);
        // B_ARG order: useCausalMask, training
        addBArgument(useCausalMask, training);
    }

    private static SDVariable[] inputs(SameDiff sd,
                                       SDVariable queries,
                                       SDVariable values,
                                       SDVariable keys,
                                       SDVariable queryMask,
                                       SDVariable valueMask) {
        List<SDVariable> inputs = new ArrayList<>();
        inputs.add(queries);
        inputs.add(values);
        inputs.add(keys == null ? values : keys);
        inputs.add(queryMask == null ? sd.constant(Nd4j.empty(queries.dataType())) : queryMask);
        inputs.add(valueMask == null ? sd.constant(Nd4j.empty(queries.dataType())) : valueMask);
        return inputs.toArray(new SDVariable[inputs.size()]);

    }

    @Override
    public void configureFromArguments() {
        super.configureFromArguments();
        // B_ARG order: useCausalMask, training
        if(bArguments.size() > 0)
            this.useCausalMask = bArguments.get(0);

        if(bArguments.size() > 1)
            this.training = bArguments.get(1);

        // T_ARG order: scale, dropout
        if(tArguments.size() > 0)
            this.scaleFactor = tArguments.get(0);

        if(tArguments.size() > 1)
            this.dropout = tArguments.get(1);
    }

    @Override
    public void configureWithSameDiff(SameDiff sameDiff) {
        super.configureWithSameDiff(sameDiff);
        if(args().length > 3)
            this.queryMask = arg(3);

        if(args().length > 4)
            this.valueMask = arg(4);
    }

    @Override
    public String opName() {
        return "dot_product_attention_v2";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> gradient) {
        // Forward outputs: [attentionOutput, attentionWeights, attentionLogits, dropoutMask?]
        SDVariable[] outs = outputVariables();
        // BP inputs order: queries, values, keys, eps, queryMask, valueMask,
        //                  attentionScoresOut, attentionScoreWeights, attentionScoresLogits, dropoutMask,
        //                  scaleFactor, dropout, useCausalMask, training
        return Arrays.asList(new DotProductAttentionV2Bp(sameDiff,
                arg(0),                              // queries
                arg(1),                              // values
                arg(2),                              // keys
                gradient.get(0),                     // eps (gradient from upstream)
                queryMask,                           // queryMask
                valueMask,                           // valueMask
                outs[0],                             // attentionScoresOut (attention output)
                outs[1],                             // attentionScoreWeights (after softmax)
                outs[2],                             // attentionScoresLogits (before softmax)
                dropout > 0.0 ? outs[3] : null,      // dropoutMask
                scaleFactor,
                dropout,
                useCausalMask,
                training).outputVariables());
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        DataType first = dataTypes.get(0);
        for( int i = 0; i < dataTypes.size(); i++) {
            Preconditions.checkState(dataTypes.get(i).isFPType(), "Input %s datatype must be a floating point type, got datypes %s", dataTypes);
            if(i > 0) {
                Preconditions.checkState(first == dataTypes.get(i), "All datatypes must be same type, got input datatypes %s", dataTypes);
            }
        }
        if(dropout > 0)
            return Arrays.asList(first, first,first,first);

        return Arrays.asList(first, first,first);


    }

    @Override
    public int getNumOutputs() {
        return dropout > 0 ? 4 : 3;

    }
}