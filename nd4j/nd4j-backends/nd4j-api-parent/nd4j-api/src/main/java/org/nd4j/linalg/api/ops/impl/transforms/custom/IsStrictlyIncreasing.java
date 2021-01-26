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

package org.nd4j.linalg.api.ops.impl.transforms.custom;

import lombok.NonNull;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.Collections;
import java.util.List;

/**
 * This op takes 1 n-dimensional array as input,
 * and returns true if for every adjacent pair we have x[i] < x[i+1].
 *
 */
public class IsStrictlyIncreasing extends DynamicCustomOp {
    public IsStrictlyIncreasing() {}

    public IsStrictlyIncreasing( SameDiff sameDiff, SDVariable[] args, boolean inPlace) {
        super(null, sameDiff, args, inPlace);
    }

    public IsStrictlyIncreasing( SameDiff sameDiff, SDVariable input) {
        super(null, sameDiff, new SDVariable[]{input});
    }

    public IsStrictlyIncreasing(@NonNull INDArray input){
        this(input, null);
    }

    public IsStrictlyIncreasing(@NonNull INDArray input, INDArray output) {
        super(null, new INDArray[]{input}, wrapOrNull(output));
    }


    @Override
    public String opName() {
        return "is_strictly_increasing";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        return Collections.singletonList(sameDiff.zerosLike(arg()));
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes){
        Preconditions.checkState(dataTypes != null && dataTypes.size() == 1, "Expected exactly 1 input datatypes for %s, got %s", getClass(), dataTypes);
        return Collections.singletonList(DataType.BOOL);
    }
}
