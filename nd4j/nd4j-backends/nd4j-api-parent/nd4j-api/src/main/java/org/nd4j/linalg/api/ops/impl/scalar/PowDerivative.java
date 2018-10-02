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

package org.nd4j.linalg.api.ops.impl.scalar;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseTransformOp;

import java.util.List;

/**
 * Pow derivative
 *
 * z = n * x ^ (n-1)
 *
 * @author raver119@gmail.com
 */
public class PowDerivative extends BaseTransformOp {
    private double pow;

    public PowDerivative(SameDiff sameDiff, SDVariable i_v, boolean inPlace, double pow) {
        super(sameDiff, i_v, inPlace);
        this.pow = pow;
        this.extraArgs = new Object[] {pow};
    }

    public PowDerivative() {}

    public PowDerivative(INDArray x, INDArray z, double pow) {
        super(x, z);
        this.pow = pow;
        init(x, null, z, x.lengthLong());
    }

    public PowDerivative(INDArray x, INDArray z, long n, double pow) {
        super(x, z, n);
        this.pow = pow;
        init(x, null, z, x.lengthLong());
    }

    public PowDerivative(INDArray x, INDArray y, INDArray z, long n, double pow) {
        super(x, y, z, n);
        this.pow = pow;
        init(x, null, z, x.lengthLong());
    }

    public PowDerivative(INDArray x, double pow) {
        super(x);
        this.pow = pow;
        init(x, null, z, x.lengthLong());
    }

    @Override
    public void init(INDArray x, INDArray y, INDArray z, long n) {
        super.init(x, y, z, n);
        this.extraArgs = new Object[] {pow};
    }

    @Override
    public int opNum() {
        return 92;
    }

    @Override
    public String opName() {
        return "_powderivative";
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
       throw new UnsupportedOperationException();
    }
}
