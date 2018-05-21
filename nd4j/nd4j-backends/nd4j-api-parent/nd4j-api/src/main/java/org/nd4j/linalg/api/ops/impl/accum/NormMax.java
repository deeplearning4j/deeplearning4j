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
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.Arrays;
import java.util.List;

/**
 * The max absolute value
 *
 * @author Adam Gibson
 */
public class NormMax extends BaseAccumulation {
    public NormMax(SameDiff sameDiff, SDVariable i_v, int[] dimensions) {
        super(sameDiff, i_v, dimensions);
    }

    public NormMax(SameDiff sameDiff, SDVariable i_v, SDVariable i_v2, int[] dimensions) {
        super(sameDiff, i_v, i_v2, dimensions);
    }

    public NormMax() {
    }

    public NormMax(INDArray x, INDArray y, INDArray z, long n) {
        super(x, y, z, n);
    }

    public NormMax(INDArray x, INDArray y, long n) {
        super(x, y, n);
    }

    public NormMax(INDArray x) {
        super(x);
    }

    public NormMax(INDArray x, INDArray y) {
        super(x, y);
    }

    @Override
    public INDArray noOp() {
        return Transforms.abs(x());
    }


    @Override
    public int opNum() {
        return 7;
    }

    @Override
    public String opName() {
        return "normmax";
    }


    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v1) {
        //maxnorm(in) = max_i |x_i|
        //d maxnorm(in)/dx = 0 if x_i is not the max, or d|x|/dx otherwise

        SDVariable absIn = sameDiff.abs(arg());
        SDVariable maxnorm = outputVariables()[0];
        int origRank = Shape.rankFromShape(arg().getShape());   //TODO shape may not always be defined?
        SDVariable maxnormBc = f().reductionBroadcastableWithOrigShape(origRank, dimensions, maxnorm);
        maxnormBc = sameDiff.onesLike(arg()).mul(maxnormBc);
        SDVariable eq = sameDiff.eq(absIn, maxnormBc);
        SDVariable dAbsXdX = sameDiff.sign(arg());
        SDVariable dNormmaxDx = eq.mul(dAbsXdX);
        SDVariable broadcastableGrad = f().reductionBroadcastableWithOrigShape(origRank, dimensions, i_v1.get(0));
        SDVariable ret = dNormmaxDx.mul(broadcastableGrad);
        return Arrays.asList(ret);
    }

    @Override
    public String onnxName() {
        return "Norm";
    }

    @Override
    public String tensorflowName() {
        return "norm";
    }

    @Override
    public Type getOpType() {
        return Type.REDUCE;
    }

}
