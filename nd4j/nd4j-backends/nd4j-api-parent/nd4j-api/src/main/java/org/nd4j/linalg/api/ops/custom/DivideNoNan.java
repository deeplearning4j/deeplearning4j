/* ******************************************************************************
 * Copyright (c) 2019 Konduit K.K.
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
package org.nd4j.linalg.api.ops.custom;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.shape.Shape;

import java.util.Collections;
import java.util.List;

public class DivideNoNan extends DynamicCustomOp {
    public DivideNoNan() {
    }

    public DivideNoNan(INDArray in1, INDArray in2) {
        inputArguments.add(in1);
        inputArguments.add(in2);
    }

    public DivideNoNan(INDArray in1, INDArray in2, INDArray out) {
        this(in1,in2);
        outputArguments.add(out);
    }

    public DivideNoNan(SameDiff sameDiff, SDVariable in1, SDVariable in2) {
        super("", sameDiff, new SDVariable[]{in1, in2});
    }

    @Override
    public String opName() {
        return "divide_no_nan";
    }

    @Override
    public String tensorflowName() {
        return "DivNoNan";
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes){
        Preconditions.checkState(dataTypes != null && dataTypes.size() == 2, "Expected exactly 2 input datatypes for %s, got input %s", getClass(), dataTypes);

        DataType z = Shape.pickPairwiseDataType(dataTypes.get(0), dataTypes.get(1));
        return Collections.singletonList(z);
    }
}