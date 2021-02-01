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
package org.nd4j.linalg.api.ops.custom;

import lombok.NonNull;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.Collections;
import java.util.List;

public class BetaInc extends DynamicCustomOp {

    public BetaInc() {}

    public BetaInc(@NonNull INDArray a_input, @NonNull INDArray b_input, @NonNull INDArray x_input,
                   INDArray output) {
        addInputArgument(a_input, b_input, x_input);
        if (output != null) {
            addOutputArgument(output);
        }
    }

    public BetaInc(@NonNull INDArray a_input, @NonNull INDArray b_input, @NonNull INDArray x_input) {
        inputArguments.add(a_input);
        inputArguments.add(b_input);
        inputArguments.add(x_input);
    }

    public BetaInc(@NonNull SameDiff sameDiff, @NonNull SDVariable a, @NonNull SDVariable b, @NonNull SDVariable x) {
        super(sameDiff, new SDVariable[]{a,b,x});
    }

    @Override
    public String opName() {
        return "betainc";
    }

    @Override
    public String tensorflowName() {
        return "Betainc";
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes){
        int n = args().length;
        Preconditions.checkState(inputDataTypes != null && inputDataTypes.size() == n, "Expected %s input data types for %s, got %s", n, getClass(), inputDataTypes);
        return Collections.singletonList(inputDataTypes.get(0));
    }
}
