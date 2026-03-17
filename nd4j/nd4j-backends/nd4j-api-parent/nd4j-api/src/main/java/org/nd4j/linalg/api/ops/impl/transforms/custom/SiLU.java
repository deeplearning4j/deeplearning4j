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

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * SiLU (Sigmoid Linear Unit) activation function.
 *
 * Computes: silu(x) = x * sigmoid(x)
 *
 * Also known as the Swish activation function.
 *
 * Inputs:
 *   0: input tensor
 *
 * Output:
 *   0: silu(input)
 *
 * @author Eclipse Deeplearning4j Contributors
 */
@NoArgsConstructor
public class SiLU extends DynamicCustomOp {

    public SiLU(SameDiff sameDiff, SDVariable input) {
        super(null, sameDiff, new SDVariable[]{input}, false);
    }

    public SiLU(SameDiff sameDiff, SDVariable input, boolean inPlace) {
        super(null, sameDiff, new SDVariable[]{input}, inPlace);
    }

    public SiLU(INDArray input, INDArray output) {
        super(new INDArray[]{input}, output != null ? new INDArray[]{output} : null);
    }

    public SiLU(INDArray input) {
        super(new INDArray[]{input}, null);
    }

    @Override
    public void configureFromArguments() {
        super.configureFromArguments();
    }

    @Override
    public String opName() {
        return "silu";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> gradients) {
        return Arrays.asList(new SiLUBp(sameDiff, arg(0), gradients.get(0)).outputVariables());
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
