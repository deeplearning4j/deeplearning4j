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

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class LogicalNot extends DynamicCustomOp {

    public LogicalNot(){ }

    public LogicalNot(SameDiff sd, SDVariable in1, SDVariable in2){
        super(null, sd, new SDVariable[]{in1, in2});
    }

    @Override
    public String opName(){
        return "boolean_not";
    }

    @Override
    public String tensorflowName() {
        return "LogicalNot";
    }


    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        return Arrays.asList( sameDiff.zerosLike(larg()), sameDiff.zerosLike(rarg()));
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes){
        Preconditions.checkState(dataTypes != null && dataTypes.size() == 1, "Expected exactly 1 input datatype, got %s", dataTypes);
        Preconditions.checkState(dataTypes.get(0) == DataType.BOOL, "Datatype for input must be BOOL: got %s", dataTypes.get(0));
        return Collections.singletonList(DataType.BOOL);
    }
}
