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

package org.nd4j.linalg.api.ops.impl.reduce3;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.base.Preconditions;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseReduceFloatOp;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * Manhattan distance
 *
 * @author Adam Gibson
 */
public abstract class BaseReduce3Op extends BaseReduceFloatOp {
    public BaseReduce3Op(SameDiff sameDiff, SDVariable i_v, int[] dimensions) {
        super(sameDiff, i_v, dimensions);
    }

    public BaseReduce3Op(SameDiff sameDiff, SDVariable i_v, SDVariable i_v2, int... dimensions) {
        super(sameDiff, i_v, i_v2, dimensions);
    }

    public BaseReduce3Op() {}


    public BaseReduce3Op(INDArray x, INDArray y, int... dimensions) {
        this(x, y, false, dimensions);
    }

    public BaseReduce3Op(INDArray x, INDArray y, boolean allDistances, int... dimensions) {
        this(x, y, null, true, false, dimensions);
        this.isComplex = allDistances;
    }

    public BaseReduce3Op(INDArray x, INDArray y, INDArray z) {
        this(x, y, z, false, false, (int[])null);
    }

    public BaseReduce3Op(INDArray x, INDArray y, INDArray z, boolean keepDims, int... dimensions){
        this(x,y,z,keepDims, false);
    }

    public BaseReduce3Op(INDArray x, INDArray y, INDArray z, boolean keepDims, boolean allDistances, int... dimensions){
        super(x, y, z, keepDims, dimensions);
        this.isComplex = allDistances;
        extraArgs = new Object[]{0.0f, 0.0f};
    }

    public BaseReduce3Op(INDArray x, INDArray y, INDArray z, int... dimensions) {
        super(x, y, z, false, dimensions);
    }

    @Override
    public Type opType() {
        return Type.REDUCE3;
    }

    @Override
    public Type getOpType() {
        return opType();
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
    public DataType resultType() {
        if(x.dataType().isFPType())
            return x.dataType();
        return Nd4j.defaultFloatingPointType();
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes){
        //Second input is dynamic axis arg
        Preconditions.checkState(dataTypes != null && (dataTypes.size() == 2 || dataTypes.size() == 3),
                "Expected 2 or 3 input datatype for %s, got input %s", getClass(), dataTypes);
        Preconditions.checkState(dataTypes.size() == 2 || dataTypes.get(2).isIntType(), "When executing distance reductions" +
                "with 3 inputs, third input (axis) must be an integer datatype for %s, got %s", getClass(), dataTypes);
        //Output data type: always float. TODO let's allow configuration...
        if(dataTypes.get(0).isFPType()){
            return Collections.singletonList(dataTypes.get(0));
        }
        return Collections.singletonList(Nd4j.defaultFloatingPointType());
    }
}
