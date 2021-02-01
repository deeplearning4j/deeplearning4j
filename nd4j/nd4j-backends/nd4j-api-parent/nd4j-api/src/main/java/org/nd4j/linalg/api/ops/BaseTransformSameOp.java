/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
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

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public abstract class BaseTransformSameOp extends BaseTransformOp implements TransformSameOp {

    public BaseTransformSameOp(SameDiff sameDiff, SDVariable i_v1, SDVariable i_v2) {
        super(sameDiff, i_v1, i_v2);
    }

    public BaseTransformSameOp(SameDiff sameDiff, SDVariable i_v1, SDVariable i_v2, boolean inPlace) {
        super(sameDiff, i_v1, i_v2, inPlace);
    }

    public BaseTransformSameOp(SameDiff sameDiff) {
        super(sameDiff);
    }

    public BaseTransformSameOp(SameDiff sameDiff, SDVariable i_v1, SDVariable i_v2, Object[] extraArgs) {
        super(sameDiff, i_v1, i_v2, extraArgs);
    }

    public BaseTransformSameOp(SameDiff sameDiff, SDVariable i_v, boolean inPlace) {
        super(sameDiff, i_v, inPlace);
    }

    public BaseTransformSameOp(SameDiff sameDiff, SDVariable i_v, long[] shape, boolean inPlace, Object[] extraArgs) {
        super(sameDiff, i_v, shape, inPlace, extraArgs);
    }

    public BaseTransformSameOp(SameDiff sameDiff, SDVariable i_v, Object[] extraArgs) {
        super(sameDiff, i_v, extraArgs);
    }

    public BaseTransformSameOp(INDArray x, INDArray z) {
        super(x, z);
    }

    public BaseTransformSameOp(INDArray x, INDArray y, INDArray z){
        super(x,y,z);
    }

    public BaseTransformSameOp() {
        super();
    }


    public BaseTransformSameOp(INDArray x) {
        super(x);
    }

    @Override
    public Type getOpType() {
        return Type.TRANSFORM_SAME;
    }

    @Override
    public Type opType() {
        return Type.TRANSFORM_SAME;
    }

    @Override
    public DataType resultType() {
        return this.x().dataType();
    }

    @Override
    public DataType resultType(OpContext oc) {
        return oc.getInputArray(0).dataType();
    }

    @Override
    public boolean validateDataTypes(OpContext oc, boolean experimentalMode) {
        INDArray x = oc != null ? oc.getInputArray(0) : x();
        INDArray y = oc != null ? oc.getInputArray(1) : y();
        INDArray z = oc != null ? oc.getOutputArray(0) : z();
        if (y != null) {
            Preconditions.checkArgument(x.dataType() == y.dataType(), "Op.X type must be the same as Op.Y type: x.datatype=%s, y.datatype=%s for op %s",
                    x.dataType(), y.dataType(), getClass());
        }


        if (z != null)
            Preconditions.checkArgument(z.dataType() == x.dataType(), "Op.Z must be the same as Op.X type: x.datatype=%s, z.datatype=%s for op %s",
                    x.dataType(), z.dataType(), getClass());

        return true;
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

        return Collections.singletonList(LongShapeDescriptor.fromShape(x.shape(), x.dataType()));
    }

    @Override
    public List<org.nd4j.linalg.api.buffer.DataType> calculateOutputDataTypes(List<org.nd4j.linalg.api.buffer.DataType> dataTypes){
        //All same transform ops: always same output type as input type
        Preconditions.checkState(dataTypes != null, "Expected exactly 1 or more input datatype for %s, got input %s", getClass(), dataTypes);

        org.nd4j.linalg.api.buffer.DataType check = null;
        for(org.nd4j.linalg.api.buffer.DataType dataType : dataTypes) {
            if(check != null) {
                Preconditions.checkState(dataType == check,"Data types must all be the same!");
            } else {
                check = dataType;
            }
        }
        return Arrays.asList(dataTypes.get(0));
    }
}
