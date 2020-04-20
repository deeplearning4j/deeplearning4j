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
import org.nd4j.linalg.api.ops.impl.transforms.gradient.SoftmaxBp;

import java.util.Collections;
import java.util.List;

/**
 * Soft max function
 * row_maxes is a row vector (max for each row)
 * row_maxes = rowmaxes(input)
 * diff = exp(input - max) / diff.rowSums()
 * Outputs a probability distribution.
 * Note that this is a parameterized model and requires
 * the sum and max for the vector being calculated
 *
 * @author Adam Gibson
 */

public class SoftMax extends BaseDynamicTransformOp {
    public SoftMax() {
        super();
    }

    private int dimension = 1;

    public SoftMax(SameDiff sameDiff, SDVariable[] args) {
        super(sameDiff, args, false);
    }

    public SoftMax(SameDiff sameDiff, SDVariable x, int dimension) {
        this(sameDiff, new SDVariable[]{x}, dimension);
    }

    public SoftMax(SameDiff sameDiff, SDVariable[] args, boolean inPlace) {
        super(sameDiff, args, inPlace);
    }

    public SoftMax(SameDiff sameDiff, SDVariable[] args, int dimension) {
        super(sameDiff, args, false);
        this.dimension = dimension;
        addIArgument(dimension);
    }

    public SoftMax(SameDiff sameDiff, SDVariable[] args, int dimension, boolean inPlace) {
        super(sameDiff, args, inPlace);
        this.dimension = dimension;
        addIArgument(dimension);
    }

    public SoftMax(@NonNull INDArray input, int dimension){
        this(input, null, dimension);
    }

    public SoftMax(INDArray input, INDArray result, int dimension){
        super(new INDArray[]{input}, wrapOrNull(result));
        this.dimension = dimension;
        addIArgument(dimension);
    }

    public SoftMax(INDArray input){
        this(input, input);
    }

    public SoftMax(INDArray input, INDArray result){
        this(input, result, -1);
    }

    @Override
    public String opName() {
        return "softmax";
    }

    @Override
    public String onnxName() {
        return "Softmax";
    }

    @Override
    public String tensorflowName() {
        return "Softmax";
    }


    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v) {
        return new SoftmaxBp(sameDiff, arg(), i_v.get(0), this.dimension).outputs();
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes){
        Preconditions.checkState(dataTypes != null && dataTypes.size() == 1, "Expected exactly 1 input datatype for %s, got %s", getClass(), dataTypes);
        Preconditions.checkState(dataTypes.get(0).isFPType(), "Input must be a floating point type, got %s", dataTypes.get(0));
        return Collections.singletonList(dataTypes.get(0));
    }
}
