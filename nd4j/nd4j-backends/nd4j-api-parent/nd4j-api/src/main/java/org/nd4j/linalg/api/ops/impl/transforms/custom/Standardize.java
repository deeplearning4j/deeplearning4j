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

package org.nd4j.linalg.api.ops.impl.transforms.custom;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class Standardize extends DynamicCustomOp {

    public Standardize(SameDiff sameDiff, SDVariable i_v, int... dimensions) {
        super(null, sameDiff, new SDVariable[]{i_v}, false);
        setDimensions(dimensions);
    }

    public Standardize(INDArray input, int... dimensions){
        this(input, null, dimensions);
    }

    public Standardize(INDArray input, INDArray result, int... dimensions){
        super("standardize", new INDArray[]{input},wrapOrNull(result));
        setDimensions(dimensions);
    }

    public Standardize() {
    }

    @Override
    public void setDimensions(int[] dimensions) {
        Preconditions.checkArgument(dimensions != null, "Standardize: You have to provide dimensions");
        Preconditions.checkArgument(dimensions.length > 0, "Standardize: You have to provide dimensions");

        this.dimensions = dimensions;
        addIArgument(dimensions);
    }

    @Override
    public String opName() {
        return "standardize";
    }


    @Override
    public List<SDVariable> doDiff(List<SDVariable> grad) {
        return new StandardizeBp(sameDiff, arg(0), grad.get(0), dimensions).outputs();
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes){
        Preconditions.checkState(dataTypes != null && dataTypes.size() == 1, "Expected exactly 1 input datatype for %s, got %s", getClass(), dataTypes);
        Preconditions.checkState(dataTypes.get(0).isFPType(), "Input must be a floating point type, got %s", dataTypes.get(0));
        return Collections.singletonList(dataTypes.get(0));
    }

}
