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

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseReduceSameOp;
import org.nd4j.linalg.api.ops.impl.reduce.bp.MinBp;
import org.nd4j.linalg.api.ops.impl.reduce.custom.BaseDynamicCustomReduction;

import java.util.Collections;
import java.util.List;

public class AMin extends BaseDynamicCustomReduction {
    public AMin() {
    }

    public AMin(SameDiff sameDiff, SDVariable[] args, boolean keepDims) {
        super(sameDiff, args, keepDims);
    }

    public AMin(SameDiff sameDiff, SDVariable[] args, boolean keepDims, int[] dimensions) {
        super(sameDiff, args, keepDims, dimensions);
    }

    public AMin(SameDiff sameDiff, SDVariable[] args, boolean keepDims, boolean isComplex) {
        super(sameDiff, args, keepDims, isComplex);
    }

    public AMin(SameDiff sameDiff, SDVariable[] args, boolean keepDims, boolean isComplex, int[] dimensions) {
        super(sameDiff, args, keepDims, isComplex, dimensions);
    }

    public AMin(INDArray[] inputs, INDArray[] outputs) {
        super(inputs, outputs);
    }

    public AMin(INDArray[] inputs, INDArray[] outputs, boolean keepDims) {
        super(inputs, outputs, keepDims);
    }

    public AMin(INDArray[] inputs, INDArray[] outputs, boolean keepDims, int[] dimensions) {
        super(inputs, outputs, keepDims, dimensions);
    }

    public AMin(INDArray[] inputs, boolean keepDims, int[] dimensions) {
        super(inputs, keepDims, dimensions);
    }

    public AMin(boolean keepDims, boolean isComplex, boolean isEmptyReduce, int[] dimensions) {
        super(keepDims, isComplex, isEmptyReduce, dimensions);
    }

    public AMin(SameDiff sameDiff, SDVariable arg, boolean keepDims, boolean isComplex, boolean isEmptyReduce, int[] dimensions) {
        super(sameDiff, arg, keepDims, isComplex, isEmptyReduce, dimensions);
    }

    public AMin(SameDiff sameDiff, SDVariable[] args, boolean keepDims, boolean isComplex, boolean isEmptyReduce, int[] dimensions) {
        super(sameDiff, args, keepDims, isComplex, isEmptyReduce, dimensions);
    }

    public AMin(String opName, SameDiff sameDiff, SDVariable[] args, boolean keepDims, boolean isComplex, boolean isEmptyReduce, int[] dimensions) {
        super(opName, sameDiff, args, keepDims, isComplex, isEmptyReduce, dimensions);
    }

    public AMin(String opName, INDArray input, INDArray output, List<Double> tArguments, int[] iArguments, boolean keepDims, boolean isComplex, boolean isEmptyReduce, int[] dimensions) {
        super(opName, input, output, tArguments, iArguments, keepDims, isComplex, isEmptyReduce, dimensions);
    }

    public AMin(String opName, INDArray[] inputs, INDArray[] outputs, List<Double> tArguments, int[] iArguments, boolean keepDims, boolean isComplex, boolean isEmptyReduce, int[] dimensions) {
        super(opName, inputs, outputs, tArguments, iArguments, keepDims, isComplex, isEmptyReduce, dimensions);
    }

    public AMin(String opName, INDArray[] inputs, INDArray[] outputs, List<Double> tArguments, List<Integer> iArguments, boolean keepDims, boolean isComplex, boolean isEmptyReduce, int[] dimensions) {
        super(opName, inputs, outputs, tArguments, iArguments, keepDims, isComplex, isEmptyReduce, dimensions);
    }

    public AMin(INDArray[] inputs, INDArray[] outputs, boolean keepDims, boolean isComplex, boolean isEmptyReduce, int[] dimensions) {
        super(inputs, outputs, keepDims, isComplex, isEmptyReduce, dimensions);
    }

    public AMin(String opName, INDArray[] inputs, INDArray[] outputs, boolean keepDims, boolean isComplex, boolean isEmptyReduce, int[] dimensions) {
        super(opName, inputs, outputs, keepDims, isComplex, isEmptyReduce, dimensions);
    }

    public AMin(String opName, SameDiff sameDiff, SDVariable[] args, boolean inPlace, boolean keepDims, boolean isComplex, boolean isEmptyReduce, int[] dimensions) {
        super(opName, sameDiff, args, inPlace, keepDims, isComplex, isEmptyReduce, dimensions);
    }

    public AMin(SameDiff sameDiff, SDVariable[] args, boolean inPlace, boolean keepDims, boolean isComplex, boolean isEmptyReduce, int[] dimensions) {
        super(sameDiff, args, inPlace, keepDims, isComplex, isEmptyReduce, dimensions);
    }

    public AMin(String opName, boolean keepDims, boolean isComplex, boolean isEmptyReduce, int[] dimensions) {
        super(opName, keepDims, isComplex, isEmptyReduce, dimensions);
    }

    public AMin(INDArray[] input, INDArray[] output, boolean keepDims, boolean isComplex, int[] dimensions) {
        super(input, output, keepDims, isComplex, dimensions);
    }

    public AMin(SameDiff sd, SDVariable in, SDVariable dimensions, boolean keepDims) {
        this(sd,new SDVariable[]{in,dimensions},keepDims);
    }

    public AMin(SameDiff sd, SDVariable in, boolean keepDims, int[] dimensions) {
        this(sd,new SDVariable[]{in},keepDims,dimensions);
    }

    public AMin(INDArray in, boolean keepDims, int[] dimensions) {
        this(new INDArray[]{in},keepDims,dimensions);
    }

    public AMin(INDArray in, INDArray dimensions, boolean keepDims) {
        this(new INDArray[]{in,dimensions},null,keepDims);
    }

    @Override
    public int opNum() {
        return 6;
    }

    @Override
    public String opName() {
        return "amin";
    }


    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        SDVariable sgn = sameDiff.math().sign(arg());
        SDVariable minBp = new MinBp(sameDiff, sameDiff.math().abs(arg()), f1.get(0), false, dimensions).outputVariable();
        return Collections.singletonList(sgn.mul(minBp));
    }


    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx op opName found for " +  opName());
    }

    @Override
    public String tensorflowName() {
        throw new NoOpNameFoundException("No tensorflow op opName found for " +  opName());
    }
}
