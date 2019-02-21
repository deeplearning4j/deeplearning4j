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
import java.util.List;

public class StandardizeBp extends DynamicCustomOp {

    public StandardizeBp(SameDiff sameDiff, SDVariable i_v, SDVariable grad, int... dimensions) {
        super(null, sameDiff, new SDVariable[]{i_v, grad}, false);
        this.dimensions = dimensions;
        addIArgument(dimensions);
    }

    public StandardizeBp() {
    }

    @Override
    public String opName() {
        return "standardize_bp";
    }


    @Override
    public List<SDVariable> doDiff(List<SDVariable> grad) {
            throw new UnsupportedOperationException();
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes){
        Preconditions.checkState(dataTypes != null && dataTypes.size() == 2, "Expected exactly 2 input datatype for %s, got %s", getClass(), dataTypes);
        Preconditions.checkState(dataTypes.get(0).isFPType(), "Input 0 must be a floating point type, got %s", dataTypes.get(0));
        Preconditions.checkState(dataTypes.get(1).isFPType(), "Input 1 must be a floating point type, got %s", dataTypes.get(1));
        return  Arrays.asList(dataTypes.get(0));
    }
}
