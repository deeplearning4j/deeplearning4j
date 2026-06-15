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
 * Backpropagation for SiLU (Sigmoid Linear Unit) activation function.
 *
 * Computes the gradient of silu(x) = x * sigmoid(x).
 *
 * Inputs:
 *   0: input tensor (original forward input)
 *   1: gradient of the output (gradOut)
 *
 * Output:
 *   0: gradient with respect to input
 *
 * Adam Gibson
 */
@NoArgsConstructor
public class SiLUBp extends DynamicCustomOp {

    public SiLUBp(SameDiff sameDiff, SDVariable input, SDVariable gradOut) {
        super(null, sameDiff, new SDVariable[]{input, gradOut}, false);
    }

    public SiLUBp(SameDiff sameDiff, SDVariable input, SDVariable gradOut, boolean inPlace) {
        super(null, sameDiff, new SDVariable[]{input, gradOut}, inPlace);
    }

    public SiLUBp(INDArray input, INDArray gradOut, INDArray output) {
        super(new INDArray[]{input, gradOut}, output != null ? new INDArray[]{output} : null);
    }

    public SiLUBp(INDArray input, INDArray gradOut) {
        super(new INDArray[]{input, gradOut}, null);
    }

    @Override
    public void configureFromArguments() {
        super.configureFromArguments();
    }

    @Override
    public String opName() {
        return "silu_bp";
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
