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

import java.util.Arrays;
import java.util.List;

public class HammingDistance extends BaseDynamicCustomReduction {

    public HammingDistance() {
        super();
    }

    public HammingDistance(SameDiff sameDiff, SDVariable[] args, boolean keepDims) {
        super(sameDiff, args, keepDims);
    }

    public HammingDistance(SameDiff sameDiff, SDVariable[] args, boolean keepDims, int[] dimensions) {
        super(sameDiff, args, keepDims, dimensions);
    }

    public HammingDistance(SameDiff sameDiff, SDVariable[] args, boolean keepDims, boolean isComplex) {
        super(sameDiff, args, keepDims, isComplex);
    }

    public HammingDistance(SameDiff sameDiff, SDVariable[] args, boolean keepDims, boolean isComplex, int[] dimensions) {
        super(sameDiff, args, keepDims, isComplex, dimensions);
    }

    public HammingDistance(INDArray[] inputs, INDArray[] outputs) {
        super(inputs, outputs);
    }

    public HammingDistance(INDArray[] inputs, INDArray[] outputs, boolean keepDims) {
        super(inputs, outputs, keepDims);
    }

    public HammingDistance(INDArray[] inputs, INDArray[] outputs, boolean keepDims, int[] dimensions) {
        super(inputs, outputs, keepDims, dimensions);
    }

    public HammingDistance(INDArray[] inputs, boolean keepDims, int[] dimensions) {
        super(inputs, keepDims, dimensions);
    }

    public HammingDistance(boolean keepDims, boolean isComplex, boolean isEmptyReduce, int[] dimensions) {
        super(keepDims, isComplex, isEmptyReduce, dimensions);
    }

    public HammingDistance(SameDiff sameDiff, SDVariable arg, boolean keepDims, boolean isComplex, boolean isEmptyReduce, int[] dimensions) {
        super(sameDiff, arg, keepDims, isComplex, isEmptyReduce, dimensions);
    }

    public HammingDistance(SameDiff sameDiff, SDVariable[] args, boolean keepDims, boolean isComplex, boolean isEmptyReduce, int[] dimensions) {
        super(sameDiff, args, keepDims, isComplex, isEmptyReduce, dimensions);
    }

    public HammingDistance(String opName, SameDiff sameDiff, SDVariable[] args, boolean keepDims, boolean isComplex, boolean isEmptyReduce, int[] dimensions) {
        super(opName, sameDiff, args, keepDims, isComplex, isEmptyReduce, dimensions);
    }

    public HammingDistance(String opName, INDArray input, INDArray output, List<Double> tArguments, int[] iArguments, boolean keepDims, boolean isComplex, boolean isEmptyReduce, int[] dimensions) {
        super(opName, input, output, tArguments, iArguments, keepDims, isComplex, isEmptyReduce, dimensions);
    }

    public HammingDistance(String opName, INDArray[] inputs, INDArray[] outputs, List<Double> tArguments, int[] iArguments, boolean keepDims, boolean isComplex, boolean isEmptyReduce, int[] dimensions) {
        super(opName, inputs, outputs, tArguments, iArguments, keepDims, isComplex, isEmptyReduce, dimensions);
    }

    public HammingDistance(String opName, INDArray[] inputs, INDArray[] outputs, List<Double> tArguments, List<Integer> iArguments, boolean keepDims, boolean isComplex, boolean isEmptyReduce, int[] dimensions) {
        super(opName, inputs, outputs, tArguments, iArguments, keepDims, isComplex, isEmptyReduce, dimensions);
    }

    public HammingDistance(INDArray[] inputs, INDArray[] outputs, boolean keepDims, boolean isComplex, boolean isEmptyReduce, int[] dimensions) {
        super(inputs, outputs, keepDims, isComplex, isEmptyReduce, dimensions);
    }

    public HammingDistance(String opName, INDArray[] inputs, INDArray[] outputs, boolean keepDims, boolean isComplex, boolean isEmptyReduce, int[] dimensions) {
        super(opName, inputs, outputs, keepDims, isComplex, isEmptyReduce, dimensions);
    }

    public HammingDistance(String opName, SameDiff sameDiff, SDVariable[] args, boolean inPlace, boolean keepDims, boolean isComplex, boolean isEmptyReduce, int[] dimensions) {
        super(opName, sameDiff, args, inPlace, keepDims, isComplex, isEmptyReduce, dimensions);
    }

    public HammingDistance(SameDiff sameDiff, SDVariable[] args, boolean inPlace, boolean keepDims, boolean isComplex, boolean isEmptyReduce, int[] dimensions) {
        super(sameDiff, args, inPlace, keepDims, isComplex, isEmptyReduce, dimensions);
    }

    public HammingDistance(String opName, boolean keepDims, boolean isComplex, boolean isEmptyReduce, int[] dimensions) {
        super(opName, keepDims, isComplex, isEmptyReduce, dimensions);
    }

    public HammingDistance(INDArray[] input, INDArray[] output, boolean keepDims, boolean isComplex, int[] dimensions) {
        super(input, output, keepDims, isComplex, dimensions);
    }

    public HammingDistance(SameDiff sd, SDVariable x, SDVariable y, boolean keepDims, boolean isComplex, int[] dimensions) {
        this(sd,new SDVariable[]{x,y},keepDims,isComplex,dimensions);
    }

    public HammingDistance(INDArray x, INDArray y, boolean keepDims, boolean isComplex, int[] dimensions) {
        this(new INDArray[]{x,y},keepDims,false,dimensions);
    }

    public HammingDistance(INDArray[] inputs, boolean keepDims, boolean isComplex, int[] dimensions) {
        this(inputs,null,keepDims,isComplex,false,dimensions);
    }

    @Override
    public int opNum() {
        return 7;
    }

    @Override
    public String opName() {
        return "hammingdistance";
    }

    protected void addArgs() {
        super.addArgs();
        addBArgument(isComplex);
    }


    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        //Hamming distance: "the Hamming distance between two strings of equal length is the number of positions at
        // which the corresponding symbols are different."
        //Consequently: it's not continuously differentiable, and gradients are 0 almost everywhere (but undefined
        // when x_i == y_i)
        return Arrays.asList(sameDiff.zerosLike(larg()), sameDiff.zerosLike(rarg()));
    }
}
