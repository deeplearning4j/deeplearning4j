/* ******************************************************************************
 * Copyright (c) 2020 Konduit K.K.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
package org.nd4j.linalg.api.ops.impl.shape;

import lombok.NoArgsConstructor;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.weights.LSTMLayerWeights;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;


@NoArgsConstructor
public class MergeMaxIndex extends DynamicCustomOp {
    public MergeMaxIndex(SameDiff sameDiff, SDVariable... inputs){
        super("mergemaxindex", sameDiff, inputs);
        Preconditions.checkArgument(AreEqualShapes(inputs), "All inputs have to be equal shapes");

    }

    public MergeMaxIndex(INDArray... inputs) {
        super("mergemaxindex", inputs, null);
        Preconditions.checkArgument(AreEqualShapes(inputs), "All inputs have to be equal shapes");
    }


    protected <T> boolean AreEqualShapes(T... inputs) {
        if (inputs instanceof SDVariable[]) {
            SDVariable[] in = (SDVariable[]) inputs;
            for (SDVariable input : in) {
                if (!Arrays.equals( in[0].getShape(),input.getShape())) {
                    return false;
                }
            }
        }

        if (inputs instanceof INDArray[]) {
            INDArray[] in = (INDArray[]) inputs;
            for (INDArray input : in) {

                if  (!Arrays.equals( in[0].shape() ,input.shape())) {
                    return false;

                }
            }
        }

    return true;
    }

    @Override
    public String opName() {
        return "mergemaxindex";
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        return Collections.singletonList(DataType.UINT16);
    }
}