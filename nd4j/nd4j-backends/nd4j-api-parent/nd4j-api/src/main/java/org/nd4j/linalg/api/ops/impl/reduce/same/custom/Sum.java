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

package org.nd4j.linalg.api.ops.impl.reduce.same.custom;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseReduceSameOp;
import org.nd4j.linalg.api.ops.impl.reduce.bp.SumBp;
import org.nd4j.linalg.api.ops.impl.reduce.custom.BaseDynamicCustomReduction;

import java.util.List;

@Slf4j
public class Sum extends BaseDynamicCustomReduction {

    public Sum() {
    }

    public Sum(SameDiff sameDiff, SDVariable[] args, boolean keepDims) {
        super(sameDiff, args, keepDims);
    }

    public Sum(SameDiff sameDiff, SDVariable[] args, boolean keepDims, int[] dimensions) {
        super(sameDiff, args, keepDims, dimensions);
    }

    public Sum(SameDiff sameDiff, SDVariable[] args, boolean keepDims, boolean isComplex) {
        super(sameDiff, args, keepDims, isComplex);
    }

    public Sum(SameDiff sameDiff, SDVariable[] args, boolean keepDims, boolean isComplex, int[] dimensions) {
        super(sameDiff, args, keepDims, isComplex, dimensions);
    }

    public Sum(INDArray[] inputs, INDArray[] outputs) {
        super(inputs, outputs);
    }

    public Sum(INDArray[] inputs, INDArray[] outputs, boolean keepDims) {
        super(inputs, outputs, keepDims);
    }

    public Sum(INDArray[] inputs, INDArray[] outputs, boolean keepDims, int[] dimensions) {
        super(inputs, outputs, keepDims, dimensions);
    }

    public Sum(INDArray[] inputs, boolean keepDims, int[] dimensions) {
        super(inputs, keepDims, dimensions);
    }

    public Sum(boolean keepDims, boolean isComplex, boolean isEmptyReduce, int[] dimensions) {
        super(keepDims, isComplex, isEmptyReduce, dimensions);
    }

    public Sum(SameDiff sameDiff, SDVariable arg, boolean keepDims, boolean isComplex, boolean isEmptyReduce, int[] dimensions) {
        super(sameDiff, arg, keepDims, isComplex, isEmptyReduce, dimensions);
    }

    public Sum(SameDiff sameDiff, SDVariable[] args, boolean keepDims, boolean isComplex, boolean isEmptyReduce, int[] dimensions) {
        super(sameDiff, args, keepDims, isComplex, isEmptyReduce, dimensions);
    }

    public Sum(String opName, SameDiff sameDiff, SDVariable[] args, boolean keepDims, boolean isComplex, boolean isEmptyReduce, int[] dimensions) {
        super(opName, sameDiff, args, keepDims, isComplex, isEmptyReduce, dimensions);
    }

    public Sum(String opName, INDArray input, INDArray output, List<Double> tArguments, int[] iArguments, boolean keepDims, boolean isComplex, boolean isEmptyReduce, int[] dimensions) {
        super(opName, input, output, tArguments, iArguments, keepDims, isComplex, isEmptyReduce, dimensions);
    }

    public Sum(String opName, INDArray[] inputs, INDArray[] outputs, List<Double> tArguments, int[] iArguments, boolean keepDims, boolean isComplex, boolean isEmptyReduce, int[] dimensions) {
        super(opName, inputs, outputs, tArguments, iArguments, keepDims, isComplex, isEmptyReduce, dimensions);
    }

    public Sum(String opName, INDArray[] inputs, INDArray[] outputs, List<Double> tArguments, List<Integer> iArguments, boolean keepDims, boolean isComplex, boolean isEmptyReduce, int[] dimensions) {
        super(opName, inputs, outputs, tArguments, iArguments, keepDims, isComplex, isEmptyReduce, dimensions);
    }

    public Sum(INDArray[] inputs, INDArray[] outputs, boolean keepDims, boolean isComplex, boolean isEmptyReduce, int[] dimensions) {
        super(inputs, outputs, keepDims, isComplex, isEmptyReduce, dimensions);
    }

    public Sum(String opName, INDArray[] inputs, INDArray[] outputs, boolean keepDims, boolean isComplex, boolean isEmptyReduce, int[] dimensions) {
        super(opName, inputs, outputs, keepDims, isComplex, isEmptyReduce, dimensions);
    }

    public Sum(String opName, SameDiff sameDiff, SDVariable[] args, boolean inPlace, boolean keepDims, boolean isComplex, boolean isEmptyReduce, int[] dimensions) {
        super(opName, sameDiff, args, inPlace, keepDims, isComplex, isEmptyReduce, dimensions);
    }

    public Sum(SameDiff sameDiff, SDVariable[] args, boolean inPlace, boolean keepDims, boolean isComplex, boolean isEmptyReduce, int[] dimensions) {
        super(sameDiff, args, inPlace, keepDims, isComplex, isEmptyReduce, dimensions);
    }

    public Sum(String opName, boolean keepDims, boolean isComplex, boolean isEmptyReduce, int[] dimensions) {
        super(opName, keepDims, isComplex, isEmptyReduce, dimensions);
    }

    public Sum(INDArray[] input, INDArray[] output, boolean keepDims, boolean isComplex, int[] dimensions) {
        super(input, output, keepDims, isComplex, dimensions);
    }

    public Sum(SameDiff sd, SDVariable in, boolean keepDims, int[] dimensions) {
        this(sd,new SDVariable[]{in},keepDims,dimensions);
    }

    public Sum(SameDiff sd, SDVariable in, SDVariable dimensions, boolean keepDims) {
        this(sd,new SDVariable[]{in,dimensions},keepDims);
    }

    public Sum(INDArray in, boolean keepDims, int[] dimensions) {
        this(new INDArray[]{in},keepDims,dimensions);
    }

    public Sum(INDArray in, INDArray dimensions, boolean keepDims) {
        this(new INDArray[]{in,dimensions},null,keepDims);
    }

    @Override
    public int opNum() {
        return 0;
    }

    @Override
    public String opName() {
        return "reduce_sum";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v1) {
        //Out = sum(in)
        // dL/dIn = dL/dOut * dOut/dIn
        //        = dL/dOut * 1
        // But broadcast to shape of the input
        return new SumBp(sameDiff, arg(), i_v1.get(0), keepDims, dimensions).outputs();
    }


    @Override
    public String onnxName() {
        return "Sum";
    }

    @Override
    public String tensorflowName() {
        return "Sum";
    }

}
