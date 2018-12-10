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

package org.nd4j.linalg.api.ops.impl.reduce.floating;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseReduceFloatOp;

import java.util.Collections;
import java.util.List;

/**
 * Non-normalized Shannon Entropy Op - returns the entropy (information gain, or uncertainty of a random variable).
 *
 * @author raver119@gmail.com
 */
public class ShannonEntropy extends BaseReduceFloatOp {
    public ShannonEntropy(SameDiff sameDiff, SDVariable i_v, int[] dimensions) {
        super(sameDiff, i_v, dimensions);
    }

    public ShannonEntropy(SameDiff sameDiff, SDVariable i_v, SDVariable i_v2, int[] dimensions) {
        super(sameDiff, i_v, i_v2, dimensions);
    }

    public ShannonEntropy() {}

    public ShannonEntropy(INDArray x, INDArray z, long n) {
        super(x, null, z, n);
    }

    public ShannonEntropy(INDArray x) {
        super(x);
    }

    public ShannonEntropy(INDArray x, INDArray z) {
        super(x, null, z, x.lengthLong());
    }
    @Override
    public int opNum() {
        return 10;
    }

    @Override
    public String opName() {
        return "shannonentropy";
    }


    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        //dL/dx = dL/dOut * dOut/dIn
        //out = -sum(x*log2(x))
        // let z = x * log2(x)
        //Then we can do sumBp(z, -dL/dOut)
        //Note d/dx(x*log2(x)) = (log(x)+1)/log(2)

        SDVariable log2x = f().log(arg(),2);
        SDVariable logx = f().log(arg());
        SDVariable xLog2X = arg().mul(log2x);
        SDVariable sumBp = f().sumBp(xLog2X, f1.get(0).neg(), false, dimensions);
        return Collections.singletonList(sumBp.mul(logx.add(1.0)).div(Math.log(2.0)));
    }

    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx op opName found for " +  opName());
    }

    @Override
    public String tensorflowName() {
        return "entropy_shannon";
    }
}
