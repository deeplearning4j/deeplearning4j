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

package org.nd4j.linalg.api.ops.impl.transforms.any;

import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseTransformAnyOp;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Collections;
import java.util.List;

/**
 * [1, 2, 3, 1] -> [0, 0, 1, 0]
 * @author Adam Gibson
 */
public class IsMax extends DynamicCustomOp {
    public IsMax(SameDiff sameDiff, SDVariable i_v) {
        super(sameDiff, i_v);
    }


    public IsMax(INDArray x, INDArray z) {
        super(new INDArray[]{x}, new INDArray[]{z});
    }

    public IsMax() {}

    public IsMax(INDArray x) {
        this(x, Nd4j.createUninitialized(DataType.BOOL, x.shape(), x.ordering()));
    }

    public IsMax(INDArray x, INDArray z, int... dimensions) {
        this(x, z);
        this.addIArgument(dimensions);
    }

    public IsMax(INDArray x, int... dimensions) {
        this(x, Nd4j.createUninitialized(DataType.BOOL, x.shape(), x.ordering()), dimensions);
    }

    @Override
    public String opName() {
        return "ismax";
    }

    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx op opName found for " +  opName());
    }

    @Override
    public String tensorflowName() {
        throw new NoOpNameFoundException("No tensorflow op opName found for " +  opName());
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        return Collections.singletonList(f().zerosLike(arg()));
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes){
        //Also supports other types if say float array is provided as output array
        return Collections.singletonList(DataType.BOOL);
    }
}
