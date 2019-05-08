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

package org.nd4j.linalg.api.ops;

import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Index based reduction algo
 *
 * @author Adam Gibson
 */
@Slf4j
@Data
public abstract class BaseIndexAccumulation extends BaseOp implements IndexAccumulation {
    protected boolean keepDims = false;

    public BaseIndexAccumulation(SameDiff sameDiff,
                                 SDVariable i_v,
                                 boolean keepDims,
                                 int[] dimensions) {
        super(sameDiff,new Object[]{dimensions});
        if (i_v != null) {
            this.dimensions = dimensions;
            f().validateDifferentialFunctionsameDiff(i_v);
            sameDiff.addArgsFor(new SDVariable[]{i_v},this);
            if(Shape.isPlaceholderShape(i_v.getShape())) {
                sameDiff.addPropertyToResolve(this,i_v.getVarName());
            }

            this.xVertexId = i_v.getVarName();
        } else {
            throw new IllegalArgumentException("Input not null variable.");
        }
        this.keepDims = keepDims;
        defineDimensions(dimensions);
    }

    public BaseIndexAccumulation(SameDiff sameDiff,
                                 SDVariable i_v,
                                 SDVariable i_v2,
                                 boolean keepDims,
                                 int[] dimensions) {
        super(sameDiff,new Object[]{dimensions});
        if (i_v != null) {
            this.dimensions = dimensions;
            f().validateDifferentialFunctionsameDiff(i_v);
            f().validateDifferentialFunctionsameDiff(i_v2);
            this.xVertexId = i_v.getVarName();
            this.yVertexId = i_v2.getVarName();
            sameDiff.addArgsFor(new SDVariable[]{i_v,i_v2},this);

            if(Shape.isPlaceholderShape(i_v.getShape())) {
                sameDiff.addPropertyToResolve(this,i_v.getVarName());
            }

            if(Shape.isPlaceholderShape(i_v2.getShape())) {
                sameDiff.addPropertyToResolve(this,i_v2.getVarName());
            }
        } else {
            throw new IllegalArgumentException("Input not null variable.");
        }
        this.keepDims = keepDims;
        defineDimensions(dimensions);
    }


    public BaseIndexAccumulation() {}


    public BaseIndexAccumulation(INDArray x, int[] dimensions) {
        this(x, null, dimensions);
    }

    public BaseIndexAccumulation(INDArray x, INDArray z, int[] dimensions) {
        super(x, z);
        defineDimensions(dimensions);
    }


    @Override
    public List<LongShapeDescriptor> calculateOutputShape() {
        if(x == null)
            return Collections.emptyList();

        long[] reducedShape = Shape.getReducedShape(x.shape(), dimensions, keepDims);
        return Collections.singletonList(LongShapeDescriptor.fromShape(reducedShape, DataType.LONG));
    }

    @Override
    public Type opType() {
        return Type.INDEXREDUCE;
    }

    @Override
    public boolean validateDataTypes() {

        if (z() != null)
            Preconditions.checkArgument(z().dataType() == DataType.LONG, "IndexReduce operations require LONG output: " +
                    "got result array of type %s for op %s", z.dataType(), getClass());

        return true;
    }

    @Override
    public List<org.nd4j.linalg.api.buffer.DataType> calculateOutputDataTypes(List<org.nd4j.linalg.api.buffer.DataType> dataTypes){
        //All index accumulation ops: always long output type
        Preconditions.checkState(dataTypes != null && dataTypes.size() == 1, "Expected exactly 1 input datatype for %s, got input %s", getClass(), dataTypes);
        return Collections.singletonList(DataType.LONG);
    }
}
