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
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Collections;
import java.util.List;

public abstract class BaseReduceSameOp extends BaseReduceOp implements ReduceSameOp {
    public BaseReduceSameOp(SameDiff sameDiff, SDVariable i_v, SDVariable i_v2, long[] dimensions) {
        super(sameDiff, i_v, i_v2, dimensions);
    }

    public BaseReduceSameOp(SameDiff sameDiff, SDVariable i_v, boolean keepDims) {
        super(sameDiff, i_v, keepDims);
    }

    public BaseReduceSameOp(SameDiff sameDiff, SDVariable i_v, SDVariable dimensions, boolean keepDims) {
        super(sameDiff, i_v, dimensions, keepDims);
    }

    public BaseReduceSameOp(SameDiff sameDiff, SDVariable i_v, SDVariable i_v2) {
        super(sameDiff, i_v, i_v2);
    }

    protected BaseReduceSameOp(SameDiff sameDiff, SDVariable input, long[] dimensions, boolean keepDims) {
        super(sameDiff, input, dimensions, keepDims);
    }

    public BaseReduceSameOp(SameDiff sameDiff, SDVariable i_v, SDVariable i_v2, long[] dimensions, boolean keepDims) {
        super(sameDiff, i_v, i_v2, dimensions, keepDims);
    }

    public BaseReduceSameOp(SameDiff sameDiff, SDVariable i_v) {
        super(sameDiff, i_v);
    }

    protected BaseReduceSameOp(SameDiff sameDiff, SDVariable input, long... dimensions) {
        super(sameDiff, input, dimensions);
    }

    public BaseReduceSameOp(INDArray x, INDArray z, boolean keepDims, long[] dimensions) {
        super(x, null, z, keepDims, dimensions);
    }

    public BaseReduceSameOp(INDArray x, INDArray y, INDArray z, long... dimensions) {
        super(x, y, z, dimensions);
    }

    public BaseReduceSameOp(SameDiff sameDiff) {
        super(sameDiff);
    }

    public BaseReduceSameOp(SameDiff sameDiff, SDVariable i_v, SDVariable i_v2, SDVariable dimensions) {
        super(sameDiff, i_v, i_v2, dimensions);
    }

    public BaseReduceSameOp(INDArray x, long... dimensions) {
        super(x, dimensions);
    }

    public BaseReduceSameOp(INDArray x, boolean keepDims, long... dimensions) {
        super(x, keepDims, dimensions);
    }

    public BaseReduceSameOp(INDArray x, INDArray y, long... dimensions) {
        super(x, y, dimensions);
    }

    protected BaseReduceSameOp() {
        super();
    }

    public BaseReduceSameOp(INDArray x, INDArray y, INDArray z, boolean keepDims, long[] dimensions) {
        super(x, y, z, keepDims, dimensions);
    }

    @Override
    public Type opType() {
        return Type.REDUCE_SAME;
    }

    @Override
    public Type getOpType() {
        return opType();
    }

    @Override
    public DataType resultType() {
        return this.x().dataType();
    }

    @Override
    public DataType resultType(OpContext oc){
        return oc.getInputArray(0).dataType();
    }

    @Override
    public boolean validateDataTypes(OpContext oc) {
        INDArray x = oc != null ? oc.getInputArray(0) : x();
        INDArray y = oc != null ? oc.getInputArray(1) : y();

        INDArray z = oc != null ? oc.getOutputArray(0) : z();
        if (z != null)
            Preconditions.checkArgument(z.dataType() == x.dataType(), "Op.Z must be the same as Op.X type. Op.X.datatype=%s, " +
                    "Op.Z.datatype=%s", x.dataType(), z.dataType());

        return true;
    }

    @Override
    public INDArray dimensions() {
        if(dimensionz == null && dimensions != null) {
            this.dimensionz = Nd4j.create(Nd4j.createBuffer(dimensions));
        } else if(dimensionz == null && y != null) {
            this.dimensionz = y;
        }
        return dimensionz;
    }

    @Override
    public List<LongShapeDescriptor> calculateOutputShape() {
        return calculateOutputShape(null);
    }

    @Override
    public List<LongShapeDescriptor> calculateOutputShape(OpContext oc) {
        INDArray x = oc != null ? oc.getInputArray(0) : x();

        if(x == null)
            return Collections.emptyList();

        //Calculate reduction shape. Note that reduction on scalar - returns a scalar
        long[] reducedShape = x.rank() == 0 ? x.shape() : Shape.getReducedShape(x.shape(),dimensions, isKeepDims());
        DataType rt = oc != null ? resultType(oc) : resultType();
        return Collections.singletonList(LongShapeDescriptor.fromShape(reducedShape, rt));
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        //Output type: same as input type for BaseReduceSameOp
        //Note TF uses 2 inputs - i.e., axis arg is a variable or constant
        Preconditions.checkState(dataTypes != null && (dataTypes.size() == 1 || dataTypes.size() == 2),
                "Expected 1 or 2 input datatypes for %s, got %s", getClass(), dataTypes);
        Preconditions.checkState(dataTypes.size() == 1 || dataTypes.get(1).isIntType(), "When executing reductions" +
                " with 2 inputs, second input (axis) must be an integer datatype for %s, got %s", getClass(), dataTypes);
        return Collections.singletonList(dataTypes.get(0));
    }
}
