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

import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.util.SameDiffUtils;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Collections;
import java.util.List;

@Slf4j
@Data
public abstract class BaseIndexAccumulation extends BaseOp implements IndexAccumulation {
    protected boolean keepDims = false;

    public BaseIndexAccumulation(SameDiff sameDiff,
                                 SDVariable i_v,
                                 boolean keepDims,
                                 long[] dimensions) {
        super(sameDiff,null);
        if (i_v != null) {
            this.dimensions = dimensions;
            sameDiff.addArgsFor(new SDVariable[]{i_v},this);

            this.xVertexId = i_v.name();
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
                                 long[] dimensions) {
        super(sameDiff,null);
        if (i_v != null) {
            this.dimensions = dimensions;
            SameDiffUtils.validateDifferentialFunctionSameDiff(sameDiff, i_v, this);
            SameDiffUtils.validateDifferentialFunctionSameDiff(sameDiff, i_v2, this);
            this.xVertexId = i_v.name();
            this.yVertexId = i_v2.name();
            sameDiff.addArgsFor(new SDVariable[]{i_v,i_v2},this);
        } else {
            throw new IllegalArgumentException("Input not null variable.");
        }
        this.keepDims = keepDims;
        defineDimensions(dimensions);
    }


    public BaseIndexAccumulation() {}


    public BaseIndexAccumulation(INDArray x, long[] dimensions) {
        this(x, null, dimensions);
    }

    public BaseIndexAccumulation(INDArray x, boolean keepDims, long[] dimensions) {
        this(x, null, dimensions);
        this.keepDims = keepDims;
        defineDimensions(dimensions);
    }

    public BaseIndexAccumulation(INDArray x, INDArray z, long[] dimensions) {
        super(x, z);
        defineDimensions(dimensions);
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


        long[] reducedShape = Shape.getReducedShape(x.shape(), dimensions, keepDims);
        return Collections.singletonList(Nd4j.createBuffer(LongShapeDescriptor.fromShape(reducedShape, DataType.INT64).toShapeInfo()));
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
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes){
        //All index accumulation ops: always long output type
        Preconditions.checkState(dataTypes != null && dataTypes.size() == 1, "Expected exactly 1 input datatype for %s, got input %s", getClass(), dataTypes);
        return Collections.singletonList(DataType.LONG);
    }
}
