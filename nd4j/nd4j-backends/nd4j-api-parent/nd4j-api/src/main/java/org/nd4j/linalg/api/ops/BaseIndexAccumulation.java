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

package org.nd4j.linalg.api.ops;

import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
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
    protected int finalResult;
    protected boolean keepDims = false;
    protected boolean newFormat = false;

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
        this.newFormat = true;
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
        this.newFormat = true;
    }


    public BaseIndexAccumulation() {}

    /**
     * Initialize with the given
     * input, pairwise transform, result, and number
     * of elements
     *
     * @param x the input
     * @param y the pairwise transform
     * @param z the result
     * @param n the number of elements
     */
    public BaseIndexAccumulation(INDArray x, INDArray y, INDArray z, long n) {
        super(x, y, z, n);
        init(x,y,z,n);
    }

    public BaseIndexAccumulation(INDArray x, INDArray y, long n) {
        this(x, y, x, n);
    }

    public BaseIndexAccumulation(INDArray x) {
        this(x, null, x, x.lengthLong());
    }

    public BaseIndexAccumulation(INDArray x, INDArray y) {
        this(x, y, x, x.lengthLong());
    }

    @Override
    public double zeroDouble() {
        return 0.0;
    }

    @Override
    public float zeroFloat() {
        return 0.0f;
    }

    @Override
    public Pair<Double, Integer> zeroPair() {
        return new Pair<>(zeroDouble(), -1);
    }

    private void init() {
        init(x, y, x, x.lengthLong());
    }

    @Override
    public void init(INDArray x, INDArray y, INDArray z, long n) {
        super.init(x, y, z, n);
        if (Nd4j.dataType() == DataBuffer.Type.DOUBLE) {
            this.extraArgs = new Object[] {zeroDouble()};
        } else if (Nd4j.dataType() == DataBuffer.Type.FLOAT) {
            this.extraArgs = new Object[] {zeroFloat()};
        } else if (Nd4j.dataType() == DataBuffer.Type.HALF) {
            this.extraArgs = new Object[] {zeroHalf()};
        }
    }


    @Override
    public List<long[]> calculateOutputShape() {
        if(arg().getShape() == null)
            return Collections.emptyList();

        List<long[]> ret = new ArrayList<>(1);
        val reducedShape = Shape.getReducedShape(arg().getShape(),dimensions, keepDims, newFormat);
        ret.add(reducedShape);
        return ret;
    }

    @Override
    public Type opType() {
        return Type.INDEXREDUCE;
    }


    @Override
    public void setFinalResult(int idx) {
        this.finalResult = idx;
    }

    @Override
    public int getFinalResult() {
        return finalResult;
    }


}
