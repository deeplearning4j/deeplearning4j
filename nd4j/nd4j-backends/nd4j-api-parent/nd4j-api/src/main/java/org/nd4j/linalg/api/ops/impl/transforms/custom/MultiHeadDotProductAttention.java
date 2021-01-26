/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
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

package org.nd4j.linalg.api.ops.impl.transforms.custom;

import lombok.NoArgsConstructor;
import lombok.NonNull;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;


/**
 * (optionally scaled) multi head dot product attention
 *
 * See also "Attention is all you need" (https://arxiv.org/abs/1706.03762, pp. 4,5, "3.2.2 Multi-Head Attention")
 *
 * @author Paul Dubs
 */
@NoArgsConstructor
public class MultiHeadDotProductAttention extends DynamicCustomOp {
    private boolean withWeights;
    private boolean scaled;

    public MultiHeadDotProductAttention(SameDiff sameDiff, SDVariable queries, SDVariable keys, SDVariable values,
                                                           SDVariable Wq, SDVariable Wk, SDVariable Wv, SDVariable Wo,
                                                           SDVariable mask,
                                        boolean scaled, boolean withWeights) {
        super(null, sameDiff,
                mask == null ? new SDVariable[] {queries, keys, values, Wq, Wk, Wv, Wo}
                : new SDVariable[] {queries, keys, values, Wq, Wk, Wv, Wo, mask},
                false);
        this.scaled = scaled;
        this.withWeights = withWeights;
        addIArgument(scaled ? 1 : 0);
        addIArgument(withWeights ? 1 : 0);
    }

    public MultiHeadDotProductAttention(@NonNull INDArray queries, @NonNull INDArray keys, @NonNull INDArray values,
                                        @NonNull INDArray Wq, @NonNull INDArray Wk, @NonNull INDArray Wv, @NonNull INDArray Wo,
                                        INDArray mask, boolean scaled) {
        this(queries, keys, values, Wq, Wk, Wv, Wo, mask, scaled, false);
    }

    public MultiHeadDotProductAttention(@NonNull INDArray queries, @NonNull INDArray keys, @NonNull INDArray values,
                                        @NonNull INDArray Wq, @NonNull INDArray Wk, @NonNull INDArray Wv, @NonNull INDArray Wo,
                                        INDArray mask, boolean scaled, boolean withWeights) {
        super(wrapFilterNull(queries, keys, values, Wq, Wk, Wv, Wo, mask), null);
        this.scaled = scaled;
        this.withWeights = withWeights;
        addIArgument(scaled ? 1 : 0);
        addIArgument(withWeights ? 1 : 0);
    }

    @Override
    public String opName() {
        return "multi_head_dot_product_attention";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> gradient) {
        return Arrays.asList(new MultiHeadDotProductAttentionBp(sameDiff, arg(0), arg(1), arg(2), arg(3), arg(4), arg(5), arg(6), gradient.get(0), args().length > 7 ? arg(7) : null, scaled).outputVariables());
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes){
        Preconditions.checkState(dataTypes != null && (dataTypes.size() == 7 || dataTypes.size() == 8), "Expected 7 or 8 input datatypes, got %s", dataTypes);
        DataType first = dataTypes.get(0);
        for( int i=0; i<dataTypes.size(); i++ ) {
            Preconditions.checkState(dataTypes.get(i).isFPType(), "Input %s datatype must be a floating point type, got datypes %s", dataTypes);
            if(i > 0){
                Preconditions.checkState(first == dataTypes.get(i), "All datatypes must be same type, got input datatypes %s", dataTypes);
            }
        }
        if(withWeights){
            return Arrays.asList(first, first);
        }else{
            return Collections.singletonList(first);
        }
    }

    @Override
    public int getNumOutputs() {
        if(withWeights){
            return 2;
        }else{
            return 1;
        }
    }
}
