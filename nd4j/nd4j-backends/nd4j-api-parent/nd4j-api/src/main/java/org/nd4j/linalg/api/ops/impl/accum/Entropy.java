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

package org.nd4j.linalg.api.ops.impl.accum;

import org.nd4j.autodiff.functions.DifferentialFunctionFactory;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseReduceOp;

import java.util.Collections;
import java.util.List;

/**
 * Entropy Op - returns the entropy (information gain, or uncertainty of a random variable).
 * -sum(x * log(x))
 *
 * @author raver119@gmail.com
 */
public class Entropy extends BaseReduceOp {
    public Entropy(SameDiff sameDiff, SDVariable i_v, int[] dimensions) {
        super(sameDiff, i_v, dimensions);
    }

    public Entropy() {}

    public Entropy(INDArray x, INDArray y, INDArray z, long n) {
        super(x, y, z, n);
    }

    public Entropy(INDArray x, INDArray y, long n) {
        super(x, y, n);
    }

    public Entropy(INDArray x) {
        super(x);
    }

    public Entropy(INDArray x, INDArray y) {
        super(x, y);
    }

    public Entropy(INDArray x, INDArray y, INDArray z) {
        super(x, y, z, x.lengthLong());
    }

    @Override
    public int opNum() {
        return 16;
    }

    @Override
    public String opName() {
        return "entropy";
    }

    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx op opName found for " +  opName());
    }

    @Override
    public String tensorflowName() {
        return "entropy_shannon";
    }

    @Override
    public Type getOpType() {
        return Type.REDUCE;
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        //dL/dx = dL/dOut * dOut/dIn
        //out = -sum(x*log(x))
        // let z = x * log(x)
        //Then we can do sumBp(z, -dL/dOut)
        //Note d/dx(x*log(x)) = log(x)+1

        return grad(f(), arg(), f1.get(0), dimensions);
    }

    public static List<SDVariable> grad(DifferentialFunctionFactory f, SDVariable arg, SDVariable grad, int[] dimensions){
        SDVariable logx = f.log(arg);
        SDVariable xLogX = arg.mul(logx);
        SDVariable sumBp = f.sumBp(xLogX, grad.neg(), false, dimensions);
        return Collections.singletonList(sumBp.mul(logx.add(1.0)));
    }
}
