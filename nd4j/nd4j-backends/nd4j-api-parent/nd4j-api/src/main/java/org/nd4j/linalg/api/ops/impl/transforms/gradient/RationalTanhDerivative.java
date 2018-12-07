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

package org.nd4j.linalg.api.ops.impl.transforms.gradient;


import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseTransformOp;
import org.nd4j.linalg.api.ops.BaseTransformStrictOp;

import java.util.List;

/**
 * Rational Tanh Derivative, as described at https://github.com/deeplearning4j/libnd4j/issues/351
 * Calculates dOut/dIn given input, not dL/dIn given dL/dOut and input
 *
 * @author raver119@gmail.com
 * @author AlexDBlack
 */
public class RationalTanhDerivative extends BaseTransformStrictOp {
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

    public RationalTanhDerivative(INDArray x) {
        super(x);
    }

    @Override
    public int opNum() {
        return 11;
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
