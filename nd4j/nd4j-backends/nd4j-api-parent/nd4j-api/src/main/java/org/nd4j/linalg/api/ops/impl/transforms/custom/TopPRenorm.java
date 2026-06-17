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
 * Top-P (nucleus) filtering with renormalization.
 *
 * Sorts tokens by descending probability, accumulates until cumulative
 * probability >= p, zeros the rest, then renormalizes.
 *
 * Input:
 *   0: logits [batch, vocabSize] or [vocabSize]
 *
 * Output:
 *   0: renormalized probabilities (same shape as input)
 *
 * Float args:
 *   0: p — cumulative probability threshold (0.0-1.0)
 */
@NoArgsConstructor
public class TopPRenorm extends DynamicCustomOp {

    @Getter private double p;

    public TopPRenorm(INDArray logits, double p) {
        super(new INDArray[]{logits}, null);
        this.p = p;
        addTArgument(p);
    }

    public TopPRenorm(SameDiff sd, SDVariable logits, double p) {
        super(null, sd, new SDVariable[]{logits}, false);
        this.p = p;
        addTArgument(p);
    }

    @Override
    public void configureFromArguments() {
        super.configureFromArguments();
        if (tArguments.size() > 0) this.p = tArguments.get(0);
    }

    @Override
    public String opName() {
        return "top_p_renorm";
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
