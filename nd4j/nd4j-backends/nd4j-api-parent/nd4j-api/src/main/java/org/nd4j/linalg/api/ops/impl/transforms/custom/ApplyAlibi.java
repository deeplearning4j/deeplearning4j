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
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.Collections;
import java.util.List;

/**
 * Apply ALiBi (Attention with Linear Biases) to attention scores.
 *
 * Adds position-dependent linear biases to attention scores, allowing the model
 * to learn position information without explicit positional embeddings.
 *
 * Reference: "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation"
 * https://arxiv.org/abs/2108.12409
 *
 * Inputs:
 *   0: scores tensor (attention scores)
 *
 * Integer args:
 *   0: numHeads (number of attention heads)
 *
 * Output:
 *   0: scores with ALiBi biases applied
 *
 * @author Eclipse Deeplearning4j Contributors
 */
@NoArgsConstructor
public class ApplyAlibi extends DynamicCustomOp {

    private int numHeads = 0;

    public ApplyAlibi(SameDiff sameDiff, SDVariable scores, int numHeads) {
        super(null, sameDiff, new SDVariable[]{scores}, false);
        this.numHeads = numHeads;
        addIArgument(numHeads);
    }

    public ApplyAlibi(INDArray scores, int numHeads) {
        super(new INDArray[]{scores}, null);
        this.numHeads = numHeads;
        addIArgument(numHeads);
    }

    public ApplyAlibi(INDArray scores, INDArray output, int numHeads) {
        super(new INDArray[]{scores}, output != null ? new INDArray[]{output} : null);
        this.numHeads = numHeads;
        addIArgument(numHeads);
    }

    @Override
    public void configureFromArguments() {
        super.configureFromArguments();
        if (!iArguments.isEmpty()) {
            this.numHeads = iArguments.get(0).intValue();
        }
    }

    @Override
    public String opName() {
        return "apply_alibi";
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        DataType first = dataTypes.get(0);
        Preconditions.checkState(first.isFPType(),
                "Input datatype must be a floating point type, got %s", first);
        return Collections.singletonList(first);
    }

    @Override
    public int getNumOutputs() {
        return 1;
    }
}
