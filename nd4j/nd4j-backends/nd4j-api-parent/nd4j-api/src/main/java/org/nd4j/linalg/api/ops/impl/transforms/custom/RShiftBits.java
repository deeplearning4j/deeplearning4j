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
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.BaseDynamicTransformOp;

import java.util.Collections;
import java.util.List;

/**
 * Element-wise shift operation, shift bits to the right, >>
 *
 * @author raver119@gmail.com
 */
public class RShiftBits extends BaseDynamicTransformOp {

    public RShiftBits(SameDiff sameDiff, SDVariable x, SDVariable y) {
        super(sameDiff, new SDVariable[] {x, y} ,false);
    }

    public RShiftBits(INDArray input, INDArray shift, INDArray output) {
        super(new INDArray[]{input, shift}, new INDArray[]{output});
    }

    public RShiftBits(INDArray input, INDArray shift) {
        this(input, shift,input.ulike());
    }

    public RShiftBits() {}

    @Override
    public String opName() {
        return "rshift_bits";
    }


    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx op opName found for " +  opName());
    }

    @Override
    public String tensorflowName() {
        throw new NoOpNameFoundException("No TensorFlow op opName found for " +  opName());
    }


    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v) {
        throw new UnsupportedOperationException("Not yet implemented: " + opName());
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes){
        Preconditions.checkState(dataTypes.get(0).isIntType(), "Input 0 datatype must be a integer type, got %s", dataTypes.get(0));
        return Collections.singletonList(dataTypes.get(0));
    }
}
