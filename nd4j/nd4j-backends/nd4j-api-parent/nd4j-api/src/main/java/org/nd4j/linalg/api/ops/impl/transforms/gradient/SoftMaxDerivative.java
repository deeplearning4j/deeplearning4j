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


import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseGradientOp;
import org.nd4j.linalg.api.ops.impl.transforms.OldSoftMax;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;

/**
 * @deprecated To be replaced by {@link SoftmaxBp}
 */
@Deprecated
public class SoftMaxDerivative extends BaseGradientOp  {
    public SoftMaxDerivative(INDArray x, INDArray z) {
        super(x, z);
    }

    public SoftMaxDerivative() {
    }

    public SoftMaxDerivative(INDArray x, INDArray z, long n) {
        super(x, z, n);
    }

    public SoftMaxDerivative(INDArray x, INDArray y, INDArray z) {
        super(x, y, z, z.lengthLong());
    }

    public SoftMaxDerivative(INDArray x) {
        super(x);
    }

    /**
     * An op number
     *
     * @return
     */
    @Override
    public int opNum() {
        return 0;
    }

    /**
     * The opName of this operation
     *
     * @return the opName of this operation
     */
    @Override
    public String opName() {
        return "softmaxderivative";
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
    public void exec() {
        INDArray softmaxed = Nd4j.getExecutioner().execAndReturn(new OldSoftMax(x));
        INDArray mulled = softmaxed.muli(y);
        INDArray summed = mulled.sum(-1);
        softmaxed.muliColumnVector(summed);
        mulled.subi(softmaxed);

    }

    @Override
    public void exec(int... dimensions) {
        super.exec(dimensions);
    }



    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v) {
        throw new UnsupportedOperationException();
    }
}
