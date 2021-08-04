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

package org.nd4j.linalg.api.ops.impl.reduce3.custom;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.reduce.custom.BaseDynamicCustomReduction;
import org.nd4j.linalg.api.ops.impl.reduce3.BaseReduce3Op;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;
import java.util.List;

public class EqualsWithEps extends BaseDynamicCustomReduction {

    private double eps = Nd4j.EPS_THRESHOLD;


    public EqualsWithEps() {
    }

    public EqualsWithEps(SameDiff sameDiff, SDVariable[] args, boolean keepDims) {
        super(sameDiff, args, keepDims);
        this.eps = eps;
    }

    public EqualsWithEps(SameDiff sameDiff, SDVariable[] args, boolean keepDims, int[] dimensions) {
        super(sameDiff, args, keepDims, dimensions);
        this.eps = eps;
    }

    public EqualsWithEps(SameDiff sameDiff, SDVariable[] args, boolean keepDims, boolean isComplex) {
        super(sameDiff, args, keepDims, isComplex);
        this.eps = eps;
    }

    public EqualsWithEps(SameDiff sameDiff, SDVariable[] args, boolean keepDims, boolean isComplex, int[] dimensions) {
        super(sameDiff, args, keepDims, isComplex, dimensions);
        this.eps = eps;
    }

    public EqualsWithEps(INDArray[] inputs, INDArray[] outputs) {
        super(inputs, outputs);
        this.eps = eps;
    }

    public EqualsWithEps(INDArray[] inputs, INDArray[] outputs, boolean keepDims) {
        super(inputs, outputs, keepDims);
        this.eps = eps;
    }


    public EqualsWithEps(double eps) {
        this.eps = eps;
    }

    public EqualsWithEps(SameDiff sameDiff, SDVariable[] args, boolean keepDims, double eps) {
        super(sameDiff, args, keepDims);
        this.eps = eps;
    }

    public EqualsWithEps(SameDiff sameDiff, SDVariable[] args, boolean keepDims, int[] dimensions, double eps) {
        super(sameDiff, args, keepDims, dimensions);
        this.eps = eps;
    }

    public EqualsWithEps(SameDiff sameDiff, SDVariable[] args, boolean keepDims, boolean isComplex, double eps) {
        super(sameDiff, args, keepDims, isComplex);
        this.eps = eps;
    }

    public EqualsWithEps(SameDiff sameDiff, SDVariable[] args, boolean keepDims, boolean isComplex, int[] dimensions, double eps) {
        super(sameDiff, args, keepDims, isComplex, dimensions);
        this.eps = eps;
    }

    public EqualsWithEps(INDArray[] inputs, INDArray[] outputs, double eps) {
        super(inputs, outputs);
        this.eps = eps;
    }

    public EqualsWithEps(INDArray[] inputs, INDArray[] outputs, boolean keepDims, double eps) {
        super(inputs, outputs, keepDims);
        this.eps = eps;
    }

    public EqualsWithEps(INDArray[] inputs, INDArray[] outputs, boolean keepDims, int[] dimensions, double eps) {
        super(inputs, outputs, keepDims, dimensions);
        this.eps = eps;
    }

    public EqualsWithEps(INDArray[] inputs, boolean keepDims, int[] dimensions, double eps) {
        super(inputs, keepDims, dimensions);
        this.eps = eps;
    }

    public EqualsWithEps(boolean keepDims, boolean isComplex, boolean isEmptyReduce, int[] dimensions, double eps) {
        super(keepDims, isComplex, isEmptyReduce, dimensions);
        this.eps = eps;
    }

    public EqualsWithEps(SameDiff sameDiff, SDVariable arg, boolean keepDims, boolean isComplex, boolean isEmptyReduce, int[] dimensions, double eps) {
        super(sameDiff, arg, keepDims, isComplex, isEmptyReduce, dimensions);
        this.eps = eps;
    }

    public EqualsWithEps(SameDiff sameDiff, SDVariable[] args, boolean keepDims, boolean isComplex, boolean isEmptyReduce, int[] dimensions, double eps) {
        super(sameDiff, args, keepDims, isComplex, isEmptyReduce, dimensions);
        this.eps = eps;
    }

    public EqualsWithEps(String opName, SameDiff sameDiff, SDVariable[] args, boolean keepDims, boolean isComplex, boolean isEmptyReduce, int[] dimensions, double eps) {
        super(opName, sameDiff, args, keepDims, isComplex, isEmptyReduce, dimensions);
        this.eps = eps;
    }

    public EqualsWithEps(String opName, INDArray input, INDArray output, List<Double> tArguments, int[] iArguments, boolean keepDims, boolean isComplex, boolean isEmptyReduce, int[] dimensions, double eps) {
        super(opName, input, output, tArguments, iArguments, keepDims, isComplex, isEmptyReduce, dimensions);
        this.eps = eps;
    }

    public EqualsWithEps(String opName, INDArray[] inputs, INDArray[] outputs, List<Double> tArguments, int[] iArguments, boolean keepDims, boolean isComplex, boolean isEmptyReduce, int[] dimensions, double eps) {
        super(opName, inputs, outputs, tArguments, iArguments, keepDims, isComplex, isEmptyReduce, dimensions);
        this.eps = eps;
    }

    public EqualsWithEps(String opName, INDArray[] inputs, INDArray[] outputs, List<Double> tArguments, List<Integer> iArguments, boolean keepDims, boolean isComplex, boolean isEmptyReduce, int[] dimensions, double eps) {
        super(opName, inputs, outputs, tArguments, iArguments, keepDims, isComplex, isEmptyReduce, dimensions);
        this.eps = eps;
    }

    public EqualsWithEps(INDArray[] inputs, INDArray[] outputs, boolean keepDims, boolean isComplex, boolean isEmptyReduce, int[] dimensions, double eps) {
        super(inputs, outputs, keepDims, isComplex, isEmptyReduce, dimensions);
        this.eps = eps;
    }

    public EqualsWithEps(String opName, INDArray[] inputs, INDArray[] outputs, boolean keepDims, boolean isComplex, boolean isEmptyReduce, int[] dimensions, double eps) {
        super(opName, inputs, outputs, keepDims, isComplex, isEmptyReduce, dimensions);
        this.eps = eps;
    }

    public EqualsWithEps(String opName, SameDiff sameDiff, SDVariable[] args, boolean inPlace, boolean keepDims, boolean isComplex, boolean isEmptyReduce, int[] dimensions, double eps) {
        super(opName, sameDiff, args, inPlace, keepDims, isComplex, isEmptyReduce, dimensions);
        this.eps = eps;
    }

    public EqualsWithEps(SameDiff sameDiff, SDVariable[] args, boolean inPlace, boolean keepDims, boolean isComplex, boolean isEmptyReduce, int[] dimensions, double eps) {
        super(sameDiff, args, inPlace, keepDims, isComplex, isEmptyReduce, dimensions);
        this.eps = eps;
    }

    public EqualsWithEps(String opName, boolean keepDims, boolean isComplex, boolean isEmptyReduce, int[] dimensions, double eps) {
        super(opName, keepDims, isComplex, isEmptyReduce, dimensions);
        this.eps = eps;
    }

    public EqualsWithEps(INDArray[] input, INDArray[] output, boolean keepDims, boolean isComplex, int[] dimensions, double eps) {
        super(input, output, keepDims, isComplex, dimensions);
        this.eps = eps;
    }

    protected void addArgs() {
        super.addArgs();
        addTArgument(eps);
    }

    @Override
    public int opNum() {
        return 4;
    }

    @Override
    public String opName() {
        return "equals_with_eps";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        return Arrays.asList(outputVariables()[0]);
    }
}
