/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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

package org.nd4j.linalg.api.ops.impl.controlflow;

import lombok.NoArgsConstructor;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.Collections;
import java.util.List;

/**
 *
 */
@NoArgsConstructor
public class Where extends DynamicCustomOp {
    public Where(SameDiff sameDiff, SDVariable[] args) {
        super(null, sameDiff, args);
    }

    public Where(String opName, INDArray[] inputs, INDArray[] outputs, List<Double> tArguments, List<Integer> iArguments) {
        super(opName, inputs, outputs, tArguments, iArguments);
    }

    public Where(INDArray[] inputs, INDArray[] outputs) {
        super(null, inputs, outputs);
    }

    public Where(SameDiff sameDiff, SDVariable[] args, boolean inPlace) {
        super(null, sameDiff, args, inPlace);
    }

    @Override
    public String opName() {
        return "Where";
    }

    @Override
    public String tensorflowName() {
        return "Where";
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputTypes) {
        Preconditions.checkState(inputTypes != null && (inputTypes.size() == 1 || inputTypes.size() == 3),
                "Expected 1 or 3 input types, got %s for op %s",inputTypes, getClass());
        if(inputTypes.size() == 3) {
            Preconditions.checkState(inputTypes.get(1) == inputTypes.get(2), "X and Y input must be same type, got inputs %s for op %s", inputTypes, getClass());
            //Output type same as x/y types
            return Collections.singletonList(inputTypes.get(1));
        } else {
            //Coordinates of true elements
            //TODO allow this to be configured
            return Collections.singletonList(DataType.LONG);
        }
    }
}
