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
    private boolean withWeights;
    private boolean scaled;

    private int scoreMode;




    public DotProductAttentionV2(@NonNull INDArray queries, @NonNull INDArray keys, @NonNull INDArray values, INDArray mask, boolean scaled){
        this(queries, keys, values, mask, scaled, false);
    }

    public DotProductAttentionV2(@NonNull INDArray queries, @NonNull INDArray keys, @NonNull INDArray values, INDArray mask, boolean scaled, boolean withWeights){
        super(wrapFilterNull(queries, keys, values, mask), null);
        this.scaled = scaled;
        this.withWeights = withWeights;
        addIArgument(0);
        addBArgument(withWeights);
    }

    public DotProductAttentionV2(SameDiff sd, SDVariable queries, SDVariable values, SDVariable keys, SDVariable queryMask, SDVariable valueMask, boolean scaled, int scoreMode, boolean b, boolean b1) {
        super(null, sd, inputs(sd,queries, values, keys, queryMask, valueMask), false);
        this.scaled = scaled;
        this.scoreMode = scoreMode;
        this.withWeights = withWeights;
        addIArgument(scoreMode);
        addBArgument(withWeights);
    }

    public DotProductAttentionV2(INDArray queries, INDArray values, INDArray keys, INDArray queryMask, INDArray valueMask, boolean scaled, int scoreMode, boolean useCausalMask, boolean withWeights) {
        super(wrapFilterNull(queries, keys, values, queryMask,valueMask), null);
        this.scaled = scaled;
        this.withWeights = withWeights;
        addIArgument(scoreMode);
        addBArgument(useCausalMask);
        addBArgument(withWeights);
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
    public String opName() {
        return "dot_product_attention_v2";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> gradient) {
        SDVariable mask = args().length == 4 ? arg(3) : null;
        return Arrays.asList(new DotProductAttentionV2Bp(sameDiff, arg(0), arg(1), arg(2), gradient.get(0), mask, scaled).outputVariables());
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        DataType first = dataTypes.get(0);
        for( int i = 0; i<dataTypes.size(); i++) {
            Preconditions.checkState(dataTypes.get(i).isFPType(), "Input %s datatype must be a floating point type, got datypes %s", dataTypes);
            if(i > 0){
                Preconditions.checkState(first == dataTypes.get(i), "All datatypes must be same type, got input datatypes %s", dataTypes);
            }
        }
        if(withWeights) {
            return Arrays.asList(first, first);
        }else{
            return Collections.singletonList(first);
        }
    }

    @Override
    public int getNumOutputs() {
        if(withWeights) {
            return 2;
        }else{
            return 1;
        }
    }
}
