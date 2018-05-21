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

import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseAccumulation;
import org.nd4j.linalg.api.shape.Shape;

import java.util.Arrays;
import java.util.List;

/**
 * Sum the components
 *
 * @author Adam Gibson
 */
@Slf4j
public class Sum extends BaseAccumulation {
    public Sum(SameDiff sameDiff, SDVariable i_v, int[] dimensions) {
        super(sameDiff, i_v, dimensions);
    }

    public Sum(SameDiff sameDiff, SDVariable i_v, SDVariable i_v2, int[] dimensions) {
        super(sameDiff, i_v, i_v2, dimensions);
    }


    public Sum() {
    }

    public Sum(INDArray x, INDArray y, INDArray z, long n) {
        super(x, y, z, n);
    }

    public Sum(INDArray x, INDArray y, long n) {
        super(x, y, n);
    }

    public Sum(INDArray x) {
        super(x);
    }

    public Sum(INDArray x, INDArray y) {
        super(x, y);
    }

    public Sum(INDArray x, INDArray y, INDArray z) {
        super(x, y, z, x.lengthLong());
    }

    @Override
    public int opNum() {
        return 1;
    }

    @Override
    public String opName() {
        return "sum";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v1) {
        //Out = sum(in)
        // dL/dIn = dL/dOut * dOut/dIn
        //        = dL/dOut * 1
        // But broadcast to shape of the input

        int origRank = Shape.rankFromShape(arg().getShape());   //TODO shape may not always be defined?
        SDVariable broadcastable = sameDiff.f().reductionBroadcastableWithOrigShape(origRank, dimensions, i_v1.get(0));
        SDVariable ret = sameDiff.onesLike(arg()).mul(broadcastable);
        return Arrays.asList(ret);
    }


    @Override
    public String onnxName() {
        return "Sum";
    }

    @Override
    public String tensorflowName() {
        return "Sum";
    }

    @Override
    public Type getOpType() {
        return Type.REDUCE;
    }
}
