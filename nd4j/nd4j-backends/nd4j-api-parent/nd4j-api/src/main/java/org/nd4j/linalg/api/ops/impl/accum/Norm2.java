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
import java.util.Collections;
import java.util.List;

/**
 * Sum of squared values (real)
 * Sum of squared complex modulus (complex)
 *
 * @author Adam Gibson
 */
public class Norm2 extends BaseAccumulation {
    public Norm2(SameDiff sameDiff, SDVariable i_v, boolean keepDims, int[] dimensions) {
        super(sameDiff, i_v, dimensions, keepDims);
    }

    public Norm2() {
    }

    public Norm2(INDArray x, INDArray y, INDArray z, long n) {
        super(x, y, z, n);
    }

    public Norm2(INDArray x, INDArray y, long n) {
        super(x, y, n);
    }

    public Norm2(INDArray x) {
        super(x);
    }

    public Norm2(INDArray x, INDArray y) {
        super(x, y);
    }

    @Override
    public INDArray noOp() {
        return Transforms.abs(x());
    }


    @Override
    public int opNum() {
        return 6;
    }

    @Override
    public String opName() {
        return "reduce_norm2";
    }


    @Override
    public List<SDVariable> doDiff(List<SDVariable> grad) {
        //d norm2(in)/dx = x / norm2(in)
        return Collections.singletonList(f().norm2Bp(arg(), grad.get(0), keepDims, dimensions));
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
