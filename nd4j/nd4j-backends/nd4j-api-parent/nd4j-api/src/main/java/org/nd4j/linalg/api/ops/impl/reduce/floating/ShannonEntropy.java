/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.linalg.api.ops.impl.reduce.floating;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseReduceFloatOp;
import org.nd4j.linalg.api.ops.impl.reduce.bp.SumBp;

import java.util.Collections;
import java.util.List;

public class ShannonEntropy extends BaseReduceFloatOp {
    public ShannonEntropy(SameDiff sameDiff, SDVariable i_v, int[] dimensions) {
        super(sameDiff, i_v, dimensions);
    }

    public ShannonEntropy(SameDiff sameDiff, SDVariable i_v, boolean keepDims, SDVariable dimensions) {
        super(sameDiff, i_v, keepDims, dimensions);
    }

    public ShannonEntropy(SameDiff sameDiff, SDVariable i_v, SDVariable i_v2, SDVariable dimensions) {
        super(sameDiff, i_v, i_v2, dimensions);
    }

    public ShannonEntropy(SameDiff sameDiff, SDVariable input, SDVariable dimensions, boolean keepDims) {
        super(sameDiff, input, dimensions, keepDims);
    }

    public ShannonEntropy(SameDiff sameDiff, SDVariable input, SDVariable dimensions) {
        super(sameDiff, input, dimensions);
    }

    public ShannonEntropy(INDArray input, INDArray output, boolean keepDims, int... dimensions) {
        super(input, output, keepDims, dimensions);
    }

    public ShannonEntropy(INDArray x, INDArray y, INDArray z, int... dimensions) {
        super(x, y, z, dimensions);
    }

    public ShannonEntropy(SameDiff sameDiff, SDVariable i_v, SDVariable i_v2, int[] dimensions) {
        super(sameDiff, i_v, i_v2, dimensions);
    }

    public ShannonEntropy(SameDiff sameDiff, SDVariable input, int[] dimensions, boolean keepDims) {
        super(sameDiff, input, dimensions, keepDims);
    }

    public ShannonEntropy() {}

    public ShannonEntropy(INDArray x, INDArray z, int... dimensions) {
        super(x, null, z, dimensions);
    }

    public ShannonEntropy(INDArray x, int[] dimensions, INDArray in, INDArray indArray, boolean keepDims) {
        super(x, dimensions);
    }

    public ShannonEntropy(INDArray in, boolean keepDims, int[] dimensions) {
        super(in,keepDims,dimensions);
    }

    public ShannonEntropy(INDArray x, int... dimensions) {
        super(x, dimensions);
    }

    public ShannonEntropy(INDArray in, INDArray dimensions, boolean keepDims) {
        super(in,keepDims,dimensions.toIntVector());
    }

    public ShannonEntropy(INDArray x, INDArray y, INDArray z, boolean keepDims, int... dimensions) {
        super(x, y, z, keepDims, dimensions);
    }

    public ShannonEntropy(SameDiff sameDiff, SDVariable i_v, boolean keepDims, int[] dimensions) {
        super(sameDiff, i_v, keepDims, dimensions);
    }

    public ShannonEntropy(INDArray in, int[] dimensions, boolean keepDims) {
        super(in,keepDims,dimensions);
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

        SDVariable log2x = sameDiff.math.log(arg(),2);
        SDVariable logx = sameDiff.math.log(arg());
        SDVariable xLog2X = arg().mul(log2x);
        SDVariable sumBp = new SumBp(sameDiff, xLog2X, f1.get(0).neg(), false, dimensions).outputVariable();
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
