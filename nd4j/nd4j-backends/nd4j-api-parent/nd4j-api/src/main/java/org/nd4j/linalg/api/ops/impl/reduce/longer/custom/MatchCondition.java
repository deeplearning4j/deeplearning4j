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

package org.nd4j.linalg.api.ops.impl.reduce.longer.custom;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseReduceLongOp;
import org.nd4j.linalg.api.ops.impl.reduce.custom.BaseDynamicCustomLongReduction;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.conditions.Condition;

import java.util.Collections;
import java.util.List;

public class MatchCondition extends BaseDynamicCustomLongReduction {
    private double compare;
    private double eps;
    private int mode;

    public MatchCondition(double compare, double eps, int mode) {
        this.compare = compare;
        this.eps = eps;
        this.mode = mode;
    }

    public MatchCondition(SameDiff sameDiff, SDVariable[] args, boolean keepDims, double compare, double eps, int mode) {
        super(sameDiff, args, keepDims);
        this.compare = compare;
        this.eps = eps;
        this.mode = mode;
    }

    public MatchCondition(SameDiff sameDiff, SDVariable[] args, boolean keepDims, int[] dimensions, double compare, double eps, int mode) {
        super(sameDiff, args, keepDims, dimensions);
        this.compare = compare;
        this.eps = eps;
        this.mode = mode;
    }

    public MatchCondition(SameDiff sameDiff, SDVariable[] args, boolean keepDims, boolean isComplex, double compare, double eps, int mode) {
        super(sameDiff, args, keepDims, isComplex);
        this.compare = compare;
        this.eps = eps;
        this.mode = mode;
    }

    public MatchCondition(SameDiff sameDiff, SDVariable[] args, boolean keepDims, boolean isComplex, int[] dimensions, double compare, double eps, int mode) {
        super(sameDiff, args, keepDims, isComplex, dimensions);
        this.compare = compare;
        this.eps = eps;
        this.mode = mode;
    }

    public MatchCondition(INDArray[] inputs, INDArray[] outputs, double compare, double eps, int mode) {
        super(inputs, outputs);
        this.compare = compare;
        this.eps = eps;
        this.mode = mode;
    }

    public MatchCondition(INDArray[] inputs, INDArray[] outputs, boolean keepDims, double compare, double eps, int mode) {
        super(inputs, outputs, keepDims);
        this.compare = compare;
        this.eps = eps;
        this.mode = mode;
    }

    public MatchCondition(INDArray[] inputs, INDArray[] outputs, boolean keepDims, int[] dimensions, double compare, double eps, int mode) {
        super(inputs, outputs, keepDims, dimensions);
        this.compare = compare;
        this.eps = eps;
        this.mode = mode;
    }

    public MatchCondition(INDArray[] inputs, boolean keepDims, int[] dimensions, double compare, double eps, int mode) {
        super(inputs, keepDims, dimensions);
        this.compare = compare;
        this.eps = eps;
        this.mode = mode;
    }

    public MatchCondition(boolean keepDims, boolean isComplex, boolean isEmptyReduce, int[] dimensions, double compare, double eps, int mode) {
        super(keepDims, isComplex, isEmptyReduce, dimensions);
        this.compare = compare;
        this.eps = eps;
        this.mode = mode;
    }

    public MatchCondition(SameDiff sameDiff, SDVariable arg, boolean keepDims, boolean isComplex, boolean isEmptyReduce, int[] dimensions, double compare, double eps, int mode) {
        super(sameDiff, arg, keepDims, isComplex, isEmptyReduce, dimensions);
        this.compare = compare;
        this.eps = eps;
        this.mode = mode;
    }

    public MatchCondition(SameDiff sameDiff, SDVariable[] args, boolean keepDims, boolean isComplex, boolean isEmptyReduce, int[] dimensions, double compare, double eps, int mode) {
        super(sameDiff, args, keepDims, isComplex, isEmptyReduce, dimensions);
        this.compare = compare;
        this.eps = eps;
        this.mode = mode;
    }

    public MatchCondition(String opName, SameDiff sameDiff, SDVariable[] args, boolean keepDims, boolean isComplex, boolean isEmptyReduce, int[] dimensions, double compare, double eps, int mode) {
        super(opName, sameDiff, args, keepDims, isComplex, isEmptyReduce, dimensions);
        this.compare = compare;
        this.eps = eps;
        this.mode = mode;
    }

    public MatchCondition(String opName, INDArray input, INDArray output, List<Double> tArguments, int[] iArguments, boolean keepDims, boolean isComplex, boolean isEmptyReduce, int[] dimensions, double compare, double eps, int mode) {
        super(opName, input, output, tArguments, iArguments, keepDims, isComplex, isEmptyReduce, dimensions);
        this.compare = compare;
        this.eps = eps;
        this.mode = mode;
    }

    public MatchCondition(String opName, INDArray[] inputs, INDArray[] outputs, List<Double> tArguments, int[] iArguments, boolean keepDims, boolean isComplex, boolean isEmptyReduce, int[] dimensions, double compare, double eps, int mode) {
        super(opName, inputs, outputs, tArguments, iArguments, keepDims, isComplex, isEmptyReduce, dimensions);
        this.compare = compare;
        this.eps = eps;
        this.mode = mode;
    }

    public MatchCondition(String opName, INDArray[] inputs, INDArray[] outputs, List<Double> tArguments, List<Integer> iArguments, boolean keepDims, boolean isComplex, boolean isEmptyReduce, int[] dimensions, double compare, double eps, int mode) {
        super(opName, inputs, outputs, tArguments, iArguments, keepDims, isComplex, isEmptyReduce, dimensions);
        this.compare = compare;
        this.eps = eps;
        this.mode = mode;
    }

    public MatchCondition(INDArray[] inputs, INDArray[] outputs, boolean keepDims, boolean isComplex, boolean isEmptyReduce, int[] dimensions, double compare, double eps, int mode) {
        super(inputs, outputs, keepDims, isComplex, isEmptyReduce, dimensions);
        this.compare = compare;
        this.eps = eps;
        this.mode = mode;
    }

    public MatchCondition(String opName, INDArray[] inputs, INDArray[] outputs, boolean keepDims, boolean isComplex, boolean isEmptyReduce, int[] dimensions, double compare, double eps, int mode) {
        super(opName, inputs, outputs, keepDims, isComplex, isEmptyReduce, dimensions);
        this.compare = compare;
        this.eps = eps;
        this.mode = mode;
    }

    public MatchCondition(String opName, SameDiff sameDiff, SDVariable[] args, boolean inPlace, boolean keepDims, boolean isComplex, boolean isEmptyReduce, int[] dimensions, double compare, double eps, int mode) {
        super(opName, sameDiff, args, inPlace, keepDims, isComplex, isEmptyReduce, dimensions);
        this.compare = compare;
        this.eps = eps;
        this.mode = mode;
    }

    public MatchCondition(SameDiff sameDiff, SDVariable[] args, boolean inPlace, boolean keepDims, boolean isComplex, boolean isEmptyReduce, int[] dimensions, double compare, double eps, int mode) {
        super(sameDiff, args, inPlace, keepDims, isComplex, isEmptyReduce, dimensions);
        this.compare = compare;
        this.eps = eps;
        this.mode = mode;
    }

    public MatchCondition(String opName, boolean keepDims, boolean isComplex, boolean isEmptyReduce, int[] dimensions, double compare, double eps, int mode) {
        super(opName, keepDims, isComplex, isEmptyReduce, dimensions);
        this.compare = compare;
        this.eps = eps;
        this.mode = mode;
    }

    public MatchCondition(INDArray[] input, INDArray[] output, boolean keepDims, boolean isComplex, int[] dimensions, double compare, double eps, int mode) {
        super(input, output, keepDims, isComplex, dimensions);
        this.compare = compare;
        this.eps = eps;
        this.mode = mode;
    }

    @Override
    public int opNum() {
        return 2;
    }

    @Override
    public String opName() {
        return "match_condition";
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
        return Collections.singletonList(sameDiff.zerosLike(arg()));
    }

    @Override
    protected void addArgs() {
        addBArgument(keepDims);
        addIArgument(mode);
        addTArgument(compare);
        addTArgument(eps);
    }
}
