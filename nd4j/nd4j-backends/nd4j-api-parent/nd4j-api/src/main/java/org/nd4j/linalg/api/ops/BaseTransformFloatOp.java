/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.linalg.api.ops;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public abstract class BaseTransformFloatOp extends BaseTransformOp implements TransformFloatOp {

    public BaseTransformFloatOp(SameDiff sameDiff, SDVariable i_v, boolean inPlace) {
        super(sameDiff, i_v, inPlace);
    }

    public BaseTransformFloatOp(SameDiff sameDiff, SDVariable i_v,  boolean inPlace, Object[] extraArgs) {
        super(sameDiff, i_v, inPlace, extraArgs);
    }

    public BaseTransformFloatOp(SameDiff sameDiff, SDVariable i_v, Object[] extraArgs) {
        super(sameDiff, i_v, extraArgs);
    }

    public BaseTransformFloatOp(INDArray x, INDArray z) {
        super(x, z);
    }

    public BaseTransformFloatOp() {
        super();
    }

    public BaseTransformFloatOp(INDArray x) {
        super(x);
    }

    @Override
    public Type getOpType() {
        return Type.TRANSFORM_FLOAT;
    }

    @Override
    public Type opType() {
        return Type.TRANSFORM_FLOAT;
    }

    @Override
    public DataType resultType() {
        if (this.x() != null && this.x().isR())
            return this.x().dataType();

        return Nd4j.defaultFloatingPointType();
    }

    @Override
    public DataType resultType(OpContext oc) {
        if (oc.getInputArray(0) != null && oc.getInputArray(0).isR())
            return oc.getInputArray(0).dataType();

        return Nd4j.defaultFloatingPointType();
    }

    @Override
    public boolean validateDataTypes(OpContext oc, boolean experimentalMode) {
        INDArray x = oc != null ? oc.getInputArray(0) : x();
        INDArray y = oc != null ? oc.getInputArray(1) : y();
        INDArray z = oc != null ? oc.getOutputArray(0) : z();

        if (y != null && !experimentalMode) {
            Preconditions.checkArgument(x.dataType() == y.dataType(), "Op.X must have same data type as Op.Y");
        }

        if (z != null)
            Preconditions.checkArgument(z.isR(),"Op.Z must be one of floating types: z.datatype=%s for op %s", z.dataType(), getClass());

        return true;
    }

    @Override
    public List<DataBuffer> calculateOutputShape() {
        return calculateOutputShape(null);
    }

    @Override
    public List<DataBuffer> calculateOutputShape(OpContext oc) {
        INDArray x = oc != null ? oc.getInputArray(0) : x();
        if(x == null)
            return Collections.emptyList();
        if(x.isEmpty()) {
            List<DataBuffer> ret = new ArrayList<>();
            LongShapeDescriptor longShapeDescriptor = LongShapeDescriptor.emptyWithShape(x.shape(),x.dataType());
            ret.add(Nd4j.createBuffer(longShapeDescriptor.toShapeInfo()));
            return ret;
        }
        return Collections.singletonList(Nd4j.createBuffer(LongShapeDescriptor.fromShape(x.shape(), x.isR() ? x.dataType() : Nd4j.defaultFloatingPointType()).toShapeInfo()));
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes){
        Preconditions.checkState(dataTypes != null && dataTypes.size() == 1, "Expected exactly 1 input datatype for %s, got input %s", getClass(), dataTypes);
        if(dataTypes.get(0).isFPType())
            return Collections.singletonList(dataTypes.get(0));
        //TODO is this what we want for all cases?
        return Collections.singletonList(DataType.FLOAT);
    }
}
