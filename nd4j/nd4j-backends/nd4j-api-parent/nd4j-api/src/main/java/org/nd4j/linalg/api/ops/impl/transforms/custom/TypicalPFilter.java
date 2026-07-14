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
 * Typical-p (entropy-deviation) logit filter.
 *
 * Masks tokens whose information content -log(p) deviates most from the distribution
 * entropy H, keeping the most typical tokens (smallest |−log(p) − H|) until their
 * cumulative probability mass reaches typicalP. Masked positions are set to -inf.
 *
 * Input:
 *   0: logits [vocabSize] or [batch, vocabSize]
 *
 * Output:
 *   0: filtered logits (same shape and type as input)
 *
 * Float args:
 *   0: typicalP — cumulative mass to keep (1.0 = off / no-op)
 *
 * Adam Gibson
 */
@NoArgsConstructor
public class TypicalPFilter extends DynamicCustomOp {

    @Getter private double typicalP = 1.0;

    public TypicalPFilter(INDArray logits, double typicalP) {
        super(new INDArray[]{logits}, null);
        this.typicalP = typicalP;
        addTArgument(typicalP);
    }

    public TypicalPFilter(SameDiff sameDiff, SDVariable logits, double typicalP) {
        super(null, sameDiff, new SDVariable[]{logits}, false);
        this.typicalP = typicalP;
        addTArgument(typicalP);
    }

    @Override
    public void configureFromArguments() {
        super.configureFromArguments();
        if (tArguments.size() > 0) this.typicalP = tArguments.get(0);
    }

    @Override
    public String opName() {
        return "typical_p_filter";
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
