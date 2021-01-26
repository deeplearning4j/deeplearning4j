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

package org.nd4j.linalg.api.ops.impl.shape;

import lombok.NonNull;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.Op;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class ReductionShape extends DynamicCustomOp {

    private boolean keepDims;

    public ReductionShape(){ }

    public ReductionShape(@NonNull SameDiff sameDiff, @NonNull SDVariable shape, @NonNull SDVariable axis, boolean keepDims){
        super(sameDiff, new SDVariable[]{shape, axis});
        this.keepDims = keepDims;
        addBArgument(keepDims);
    }


    @Override
    public String opName() {
        return "evaluate_reduction_shape";
    }

    @Override
    public Op.Type opType() {
        return Op.Type.CUSTOM;
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v) {
        return Arrays.asList(sameDiff.zerosLike(arg(0)), sameDiff.zerosLike(arg(1)));
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes){
        Preconditions.checkState(dataTypes.size() == 2, "Expected list with exactly 2 datatypes for %s, got %s", getClass(), dataTypes);
        Preconditions.checkState(dataTypes.get(0).isIntType(), "Input 0 (shape) must be integer datatype, is %s", dataTypes.get(0));
        Preconditions.checkState(dataTypes.get(0).isIntType(), "Input 1 (axis) must be an integer datatype, is %s", dataTypes.get(1));
        return Collections.singletonList(dataTypes.get(0));
    }

}
