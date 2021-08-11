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

package org.nd4j.linalg.api.ops.impl.reduce.floating.custom;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseReduceFloatOp;
import org.nd4j.linalg.api.ops.impl.reduce.bp.SumBp;
import org.nd4j.linalg.api.ops.impl.reduce.custom.BaseDynamicCustomReduction;

import java.util.Collections;
import java.util.List;

public class Entropy extends BaseDynamicCustomReduction {
    public Entropy() {
        super();
    }

    public Entropy(SameDiff sameDiff, SDVariable[] args, boolean keepDims) {
        super(sameDiff, args, keepDims);
    }

    public Entropy(SameDiff sameDiff, SDVariable[] args, boolean keepDims, int[] dimensions) {
        super(sameDiff, args, keepDims, dimensions);
    }

    public Entropy(SameDiff sameDiff, SDVariable[] args, boolean keepDims, boolean isComplex) {
        super(sameDiff, args, keepDims, isComplex);
    }

    public Entropy(SameDiff sameDiff, SDVariable[] args, boolean keepDims, boolean isComplex, int[] dimensions) {
        super(sameDiff, args, keepDims, isComplex, dimensions);
    }

    public Entropy(INDArray[] inputs, INDArray[] outputs) {
        super(inputs, outputs);
    }

    public Entropy(INDArray[] inputs, INDArray[] outputs, boolean keepDims) {
        super(inputs, outputs, keepDims);
    }

    public Entropy(INDArray[] inputs, INDArray[] outputs, boolean keepDims, int[] dimensions) {
        super(inputs, outputs, keepDims, dimensions);
    }

    public Entropy(INDArray[] inputs, boolean keepDims, int[] dimensions) {
        super(inputs, keepDims, dimensions);
    }

    public Entropy(boolean keepDims, boolean isComplex, boolean isEmptyReduce, int[] dimensions) {
        super(keepDims, isComplex, isEmptyReduce, dimensions);
    }

    public Entropy(SameDiff sameDiff, SDVariable arg, boolean keepDims, boolean isComplex, boolean isEmptyReduce, int[] dimensions) {
        super(sameDiff, arg, keepDims, isComplex, isEmptyReduce, dimensions);
    }

    public Entropy(SameDiff sameDiff, SDVariable[] args, boolean keepDims, boolean isComplex, boolean isEmptyReduce, int[] dimensions) {
        super(sameDiff, args, keepDims, isComplex, isEmptyReduce, dimensions);
    }

    public Entropy(String opName, SameDiff sameDiff, SDVariable[] args, boolean keepDims, boolean isComplex, boolean isEmptyReduce, int[] dimensions) {
        super(opName, sameDiff, args, keepDims, isComplex, isEmptyReduce, dimensions);
    }

    public Entropy(String opName, INDArray input, INDArray output, List<Double> tArguments, int[] iArguments, boolean keepDims, boolean isComplex, boolean isEmptyReduce, int[] dimensions) {
        super(opName, input, output, tArguments, iArguments, keepDims, isComplex, isEmptyReduce, dimensions);
    }

    public Entropy(String opName, INDArray[] inputs, INDArray[] outputs, List<Double> tArguments, int[] iArguments, boolean keepDims, boolean isComplex, boolean isEmptyReduce, int[] dimensions) {
        super(opName, inputs, outputs, tArguments, iArguments, keepDims, isComplex, isEmptyReduce, dimensions);
    }

    public Entropy(String opName, INDArray[] inputs, INDArray[] outputs, List<Double> tArguments, List<Integer> iArguments, boolean keepDims, boolean isComplex, boolean isEmptyReduce, int[] dimensions) {
        super(opName, inputs, outputs, tArguments, iArguments, keepDims, isComplex, isEmptyReduce, dimensions);
    }

    public Entropy(INDArray[] inputs, INDArray[] outputs, boolean keepDims, boolean isComplex, boolean isEmptyReduce, int[] dimensions) {
        super(inputs, outputs, keepDims, isComplex, isEmptyReduce, dimensions);
    }

    public Entropy(String opName, INDArray[] inputs, INDArray[] outputs, boolean keepDims, boolean isComplex, boolean isEmptyReduce, int[] dimensions) {
        super(opName, inputs, outputs, keepDims, isComplex, isEmptyReduce, dimensions);
    }

    public Entropy(String opName, SameDiff sameDiff, SDVariable[] args, boolean inPlace, boolean keepDims, boolean isComplex, boolean isEmptyReduce, int[] dimensions) {
        super(opName, sameDiff, args, inPlace, keepDims, isComplex, isEmptyReduce, dimensions);
    }

    public Entropy(SameDiff sameDiff, SDVariable[] args, boolean inPlace, boolean keepDims, boolean isComplex, boolean isEmptyReduce, int[] dimensions) {
        super(sameDiff, args, inPlace, keepDims, isComplex, isEmptyReduce, dimensions);
    }

    public Entropy(String opName, boolean keepDims, boolean isComplex, boolean isEmptyReduce, int[] dimensions) {
        super(opName, keepDims, isComplex, isEmptyReduce, dimensions);
    }

    public Entropy(INDArray[] input, INDArray[] output, boolean keepDims, boolean isComplex, int[] dimensions) {
        super(input, output, keepDims, isComplex, dimensions);
    }

    public Entropy(SameDiff sameDiff, SDVariable args, boolean keepDims) {
        super(sameDiff, new SDVariable[]{args}, keepDims);
    }

    public Entropy(SameDiff sameDiff, SDVariable args, boolean keepDims, int[] dimensions) {
        super(sameDiff, new SDVariable[]{args}, keepDims, dimensions);
    }

    public Entropy(INDArray in, boolean keepDims, int[] dimensions) {
        this(new INDArray[]{in},keepDims,dimensions);
    }

    @Override
    public int opNum() {
        return 8;
    }

    @Override
    public String opName() {
        return "entropy";
    }


    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        //dL/dx = dL/dOut * dOut/dIn
        //out = -sum(x*log(x))
        // let z = x * log(x)
        //Then we can do sumBp(z, -dL/dOut)
        //Note d/dx(x*log(x)) = log(x)+1

        return grad(sameDiff, arg(), f1.get(0), dimensions);
    }

    public static List<SDVariable> grad(SameDiff sd, SDVariable arg, SDVariable grad, int[] dimensions){
        SDVariable logx = sd.math.log(arg);
        SDVariable xLogX = arg.mul(logx);
        SDVariable sumBp = new SumBp(sd, xLogX, grad.neg(), false, dimensions).outputVariable();
        return Collections.singletonList(sumBp.mul(logx.add(1.0)));
    }
}
