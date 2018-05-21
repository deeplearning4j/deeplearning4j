/*-
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 *
 */

package org.nd4j.linalg.api.ops.impl.accum;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseAccumulation;
import org.nd4j.linalg.api.shape.Shape;

import java.util.Arrays;
import java.util.List;

/**
 * Calculate the min over a vector
 *
 * @author Adam Gibson
 */
public class Min extends BaseAccumulation {
    public Min(SameDiff sameDiff, SDVariable i_v, int[] dimensions) {
        super(sameDiff, i_v, dimensions);
    }

    public Min(SameDiff sameDiff, SDVariable i_v, SDVariable i_v2, int[] dimensions) {
        super(sameDiff, i_v, i_v2, dimensions);
    }

    public Min() {
    }

    public Min(INDArray x, INDArray y, INDArray z, long n) {
        super(x, y, z, n);
    }

    public Min(INDArray x, INDArray y, long n) {
        super(x, y, n);
    }

    public Min(INDArray x) {
        super(x);
    }

    public Min(INDArray x, INDArray y) {
        super(x, y);
    }


    @Override
    public int opNum() {
        return 4;
    }

    @Override
    public String opName() {
        return "min";
    }

    @Override
    public double zeroDouble() {
        return Double.MAX_VALUE;
    }

    @Override
    public float zeroFloat() {
        return Float.MAX_VALUE;
    }

    @Override
    public float zeroHalf() {
        return 65503.0f;
    }

    @Override
    public String onnxName() {
        return "ReduceMin";
    }

    @Override
    public String tensorflowName() {
        return "ReduceMin";
    }


    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v1) {
        //TODO do we need to handle the "multiple equal minimums" case?
        //TODO code duplication (min/max)

        SDVariable out = outputVariables()[0];

        int origRank = Shape.rankFromShape(arg().getShape());
        SDVariable expandedOut = sameDiff.f().reductionBroadcastableWithOrigShape(origRank, dimensions, out);
        expandedOut = sameDiff.onesLike("temp0", arg()).mul("tempmul", expandedOut);
        SDVariable expandedGrad = sameDiff.f().reductionBroadcastableWithOrigShape(origRank, dimensions, i_v1.get(0));

        SDVariable eq = sameDiff.eq(arg(), expandedOut);
        SDVariable ret = eq.mul(expandedGrad);
        return Arrays.asList(ret);
    }

    @Override
    public Type getOpType() {
        return Type.REDUCE;
    }
}
