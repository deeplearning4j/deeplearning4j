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

package org.nd4j.linalg.api.ops.impl.transforms.custom;

import lombok.NoArgsConstructor;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.Arrays;
import java.util.List;


/**
 * (optionally scaled) multi head dot product attention Backprop
 *
 * See also "Attention is all you need" (https://arxiv.org/abs/1706.03762, pp. 4,5, "3.2.2 Multi-Head Attention")
 *
 * @author Paul Dubs
 */
@NoArgsConstructor
public class MultiHeadDotProductAttentionBp extends DynamicCustomOp {

    private boolean scaled;

    public MultiHeadDotProductAttentionBp(SameDiff sameDiff, SDVariable queries, SDVariable keys, SDVariable values,
                                                             SDVariable Wq, SDVariable Wk, SDVariable Wv, SDVariable Wo,
                                                             SDVariable eps, SDVariable mask,
                                          boolean scaled) {
        super(null, sameDiff,
                mask == null ? new SDVariable[] {queries, keys, values, Wq, Wk, Wv, Wo, eps}
                : new SDVariable[] {queries, keys, values, Wq, Wk, Wv, Wo, eps, mask}
                , false);
        this.scaled = scaled;
        addIArgument(scaled ? 1 : 0);
    }

    @Override
    public String opName() {
        return "multi_head_dot_product_attention_bp";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> grad){
        throw new UnsupportedOperationException("Differentiation of " + getClass().getName() + " not supported");
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes){
        Preconditions.checkState(dataTypes != null && (dataTypes.size() == 8 || dataTypes.size() == 9), "Expected 8 or 9 input datatypes, got %s", dataTypes);
        DataType first = dataTypes.get(0);
        for( int i=0; i<dataTypes.size(); i++ ) {
            Preconditions.checkState(dataTypes.get(i).isFPType(), "Input %s datatype must be a floating point type, got datypes %s", dataTypes);
            if(i > 0){
                Preconditions.checkState(first == dataTypes.get(i), "All datatypes must be same type, got input datatypes %s", dataTypes);
            }
        }

        return Arrays.asList(first, first, first, first, first, first, first);
    }
}
