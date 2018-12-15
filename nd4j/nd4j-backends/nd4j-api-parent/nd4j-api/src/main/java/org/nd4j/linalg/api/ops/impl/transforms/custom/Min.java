/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
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

package org.nd4j.linalg.api.ops.impl.transforms.custom;

import lombok.NonNull;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.BaseDynamicTransformOp;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * Min function
 *
 * @author Adam Gibson
 */
public class Min extends BaseDynamicTransformOp {
    public Min() {}

    public Min(SameDiff sameDiff, @NonNull SDVariable first, @NonNull SDVariable second){
        this(sameDiff, new SDVariable[]{first, second}, false);
    }

    public Min( SameDiff sameDiff, SDVariable[] args, boolean inPlace) {
        super(sameDiff, args, inPlace);
    }

    public Min( INDArray[] inputs, INDArray[] outputs) {
        super(inputs, outputs);
    }


    @Override
    public String opName() {
        return "minimum";
    }

    @Override
    public String onnxName() {
       return "Min";
    }

    @Override
    public String tensorflowName() {
        return "Minimum";
    }


    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        SDVariable min = outputVariables()[0];
        SDVariable eq1 = sameDiff.eq(larg(), min);
        SDVariable eq2 = sameDiff.eq(rarg(), min);

        return Arrays.asList(eq1.mul(f1.get(0)), eq2.mul(f1.get(0)));
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes){
        Preconditions.checkState(dataTypes != null && dataTypes.size() == 2, "Expected exactly 2 input datatype, got %s", dataTypes);
        Preconditions.checkState(dataTypes.get(0) == dataTypes.get(1), "Input datatypes must be the same, got %s", dataTypes);
        return Collections.singletonList(dataTypes.get(0));
    }
}
