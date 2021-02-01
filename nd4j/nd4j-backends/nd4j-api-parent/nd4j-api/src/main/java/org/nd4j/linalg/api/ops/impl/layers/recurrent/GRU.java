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
package org.nd4j.linalg.api.ops.impl.layers.recurrent;

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

@NoArgsConstructor
public class GRU extends DynamicCustomOp {


    public GRU(@NonNull SameDiff sameDiff, @NonNull SDVariable x, @NonNull SDVariable hI, @NonNull SDVariable Wx, @NonNull SDVariable Wh, @NonNull SDVariable biases) {
        super(null, sameDiff, new SDVariable[]{x, hI, Wx, Wh, biases});

    }

    public GRU(@NonNull INDArray x, @NonNull INDArray hI, @NonNull INDArray Wx, @NonNull INDArray Wh, @NonNull INDArray biases) {
        super(new INDArray[]{x, hI, Wx, Wh, biases}, null);

    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes) {
        Preconditions.checkState(inputDataTypes != null && inputDataTypes.size() == 5, "Expected 5 inputs to GRU: initial cell output, input-to-hidden weights, hidden-to-hidden weights and biases got %s", inputDataTypes);
        DataType dt = inputDataTypes.get(1);
        for (int i = 0; i < inputDataTypes.size(); i++) {
            Preconditions.checkState(inputDataTypes.get(i).isFPType(), "All input types must be a floating point type, got %s", dt);
        }
        Preconditions.checkState(dt.isFPType(), "Input type 1 must be a floating point type, got %s", dt);
        return Collections.singletonList(dt);
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> grads) {
        return Arrays.asList(new GRUBp(sameDiff, arg(0), arg(1), arg(2), arg(3),
                arg(4), grads.get(0)).outputVariables());
    }


    @Override
    public String opName() {
        return "gru";
    }
}
