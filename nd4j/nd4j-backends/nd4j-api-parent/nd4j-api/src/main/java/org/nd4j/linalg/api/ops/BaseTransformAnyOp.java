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

import java.util.Collections;
import java.util.List;

public abstract class BaseTransformAnyOp extends BaseTransformOp implements TransformSameOp {

    public BaseTransformAnyOp(SameDiff sameDiff, SDVariable i_v1, SDVariable i_v2) {
        super(sameDiff, i_v1, i_v2);
    }

    public BaseTransformAnyOp(SameDiff sameDiff, SDVariable i_v1, SDVariable i_v2, boolean inPlace) {
        super(sameDiff, i_v1, i_v2, inPlace);
    }

    public BaseTransformAnyOp(SameDiff sameDiff) {
        super(sameDiff);
    }

    public BaseTransformAnyOp(SameDiff sameDiff, SDVariable i_v1, SDVariable i_v2, Object[] extraArgs) {
        super(sameDiff, i_v1, i_v2, extraArgs);
    }

    public BaseTransformAnyOp(SameDiff sameDiff, SDVariable i_v, boolean inPlace) {
        super(sameDiff, i_v, inPlace);
    }

    public BaseTransformAnyOp(SameDiff sameDiff, SDVariable i_v,  boolean inPlace, Object[] extraArgs) {
        super(sameDiff, i_v, inPlace, extraArgs);
    }

    public BaseTransformAnyOp(SameDiff sameDiff, SDVariable i_v, Object[] extraArgs) {
        super(sameDiff, i_v, extraArgs);
    }

    public BaseTransformAnyOp(INDArray x, INDArray z) {
        super(x, z);
    }

    public BaseTransformAnyOp(INDArray x, INDArray y, INDArray z) {
        super(x, y, z);
    }

    public BaseTransformAnyOp() {
        super();
    }


    public BaseTransformAnyOp(INDArray x) {
        super(x);
    }

    @Override
    public Type getOpType() {
        return Type.TRANSFORM_ANY;
    }

    @Override
    public Type opType() {
        return Type.TRANSFORM_ANY;
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
        return true;
    }

    @Override
    public List<DataBuffer> calculateOutputShape() {
        if(x == null)
            return Collections.emptyList();
        return Collections.singletonList(Nd4j.createBuffer(LongShapeDescriptor.fromShape(x.shape(), x.dataType()).toShapeInfo()));
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes){
        //Transform any: for the purposes of samediff datatype calculation, treat as same in/out
        Preconditions.checkState(dataTypes != null && dataTypes.size() >= 1, "Expected at least 1 input datatype for %s, got input %s", getClass(), dataTypes);
        return dataTypes;
    }
}
