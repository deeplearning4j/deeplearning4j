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

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Collections;
import java.util.List;

public class CompareAndBitpack extends DynamicCustomOp {
    public CompareAndBitpack() {}

    public CompareAndBitpack(INDArray in, double threshold) {
        inputArguments.add(in);
        inputArguments.add(Nd4j.scalar(threshold));
    }

    public CompareAndBitpack(INDArray in, double threshold, INDArray out) {
        this(in, threshold);
        outputArguments.add(out);
    }

    public CompareAndBitpack(SameDiff sameDiff, SDVariable threshold) {
        super("", sameDiff, new SDVariable[]{threshold});
    }

    @Override
    public String opName() {
        return "compare_and_bitpack";
    }

    @Override
    public String tensorflowName() {
        return "CompareAndBitpack";
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes){
        Preconditions.checkState(dataTypes != null && dataTypes.size() == 2, "Expected exactly 2 input datatypes for %s, got input %s", getClass(), dataTypes);
        Preconditions.checkState(dataTypes.get(0) == dataTypes.get(1), "Input data types must be the same: got %s", dataTypes);
        return Collections.singletonList(DataType.UINT8);
    }
}