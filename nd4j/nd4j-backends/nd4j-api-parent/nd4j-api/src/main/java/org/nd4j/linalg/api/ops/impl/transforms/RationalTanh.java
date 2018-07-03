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

package org.nd4j.linalg.api.ops.impl.transforms;

import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseTransformOp;

import java.util.Collections;
import java.util.List;

/**
 * Rational Tanh Approximation elementwise function, as described at https://github.com/deeplearning4j/libnd4j/issues/351
 *
 * @author raver119@gmail.com
 */
public class RationalTanh extends BaseTransformOp {
    public RationalTanh(SameDiff sameDiff, SDVariable i_v, boolean inPlace) {
        super(sameDiff, i_v, inPlace);
    }

    public RationalTanh() {}

    public RationalTanh(INDArray x, INDArray z) {
        super(x, z);
    }

    public RationalTanh(INDArray x, INDArray z, long n) {
        super(x, z, n);
    }

    public RationalTanh(INDArray x, INDArray y, INDArray z, long n) {
        super(x, y, z, n);
    }

    public RationalTanh(INDArray x, INDArray y, INDArray z) {
        super(x, y, z, x.lengthLong());
    }

    public RationalTanh(INDArray x) {
        super(x);
    }

    @Override
    public int opNum() {
        return 53;
    }

    @Override
    public String opName() {
        return "rational_tanh";
    }

    @Override
    public String onnxName() {
        return "Tanh";
    }

    @Override
    public String tensorflowName() {
        return "RationalTanh";
    }


    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        return Collections.singletonList(f().tanhRationalDerivative(arg()).mul(f1.get(0)));
    }
}
