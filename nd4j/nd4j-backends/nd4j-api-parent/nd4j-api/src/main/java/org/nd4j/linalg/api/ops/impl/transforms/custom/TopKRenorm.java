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

import lombok.Getter;
import lombok.NoArgsConstructor;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.Collections;
import java.util.List;

/**
 * Top-K filtering with renormalization.
 *
 * Keeps only the top-K highest-probability tokens, zeros the rest,
 * then renormalizes so the kept probabilities sum to 1.
 *
 * Input:
 *   0: logits [batch, vocabSize] or [vocabSize]
 *
 * Output:
 *   0: renormalized probabilities (same shape as input)
 *
 * Int args:
 *   0: k — number of top tokens to keep
 */
@NoArgsConstructor
public class TopKRenorm extends DynamicCustomOp {

    @Getter private int k;

    public TopKRenorm(INDArray logits, int k) {
        super(new INDArray[]{logits}, null);
        this.k = k;
        addIArgument((long) k);
    }

    public TopKRenorm(SameDiff sd, SDVariable logits, int k) {
        super(null, sd, new SDVariable[]{logits}, false);
        this.k = k;
        addIArgument((long) k);
    }

    @Override
    public void configureFromArguments() {
        super.configureFromArguments();
        if (iArguments.size() > 0) this.k = iArguments.get(0).intValue();
    }

    @Override
    public String opName() {
        return "top_k_renorm";
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes) {
        return Collections.singletonList(inputDataTypes.get(0));
    }

    @Override
    public int getNumOutputs() {
        return 1;
    }
}
