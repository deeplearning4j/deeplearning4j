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

package org.nd4j.linalg.api.ops.impl.transforms.gradient;


import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseTransformOp;

import java.util.List;

/**
 * Rational Tanh Derivative, as described at https://github.com/deeplearning4j/libnd4j/issues/351
 * Calculates dOut/dIn given input, not dL/dIn given dL/dOut and input
 *
 * @author raver119@gmail.com
 * @author AlexDBlack
 */
public class RationalTanhDerivative extends BaseTransformOp {
    public RationalTanhDerivative(SameDiff sameDiff, SDVariable in, boolean inPlace) {
        super(sameDiff, in, inPlace);
    }

    public RationalTanhDerivative() {}

    public RationalTanhDerivative(INDArray x, INDArray z) {
        super(x, z);
    }

    public RationalTanhDerivative(INDArray x, INDArray z, long n) {
        super(x, z, n);
    }

    public RationalTanhDerivative(INDArray x, INDArray y, INDArray z, long n) {
        super(x, y, z, n);
    }

    public RationalTanhDerivative(INDArray x) {
        super(x);
    }

    @Override
    public int opNum() {
        return 54;
    }

    @Override
    public String opName() {
        return "rational_tanh_derivative";
    }

    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx op opName found for " +  opName());
    }

    @Override
    public String tensorflowName() {
        throw new NoOpNameFoundException("No tensorflow op opName found for " +  opName());
    }


    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        throw new UnsupportedOperationException("Differentiation not supported: " + getClass().getSimpleName());
    }
}
