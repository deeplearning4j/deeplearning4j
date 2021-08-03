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
import org.nd4j.autodiff.util.SameDiffUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.reduce.custom.BaseDynamicCustomReduction;
import org.nd4j.linalg.api.ops.impl.reduce3.BaseReduce3Op;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;
import java.util.List;

public class CosineSimilarity extends BaseDynamicCustomReduction {
    public static final String OP_NAME = "cosinesimilarity";

    public CosineSimilarity() {
    }

    public CosineSimilarity(SameDiff sameDiff, SDVariable[] args, boolean keepDims) {
        super(sameDiff, args, keepDims);
    }

    public CosineSimilarity(SameDiff sameDiff, SDVariable[] args, boolean keepDims, int[] dimensions) {
        super(sameDiff, args, keepDims, dimensions);
    }

    public CosineSimilarity(SameDiff sameDiff, SDVariable[] args, boolean keepDims, boolean isComplex) {
        super(sameDiff, args, keepDims, isComplex);
    }

    public CosineSimilarity(SameDiff sameDiff, SDVariable[] args, boolean keepDims, boolean isComplex, int[] dimensions) {
        super(sameDiff, args, keepDims, isComplex, dimensions);
    }

    public CosineSimilarity(INDArray[] inputs, INDArray[] outputs) {
        super(inputs, outputs);
    }

    public CosineSimilarity(INDArray[] inputs, INDArray[] outputs, boolean keepDims) {
        super(inputs, outputs, keepDims);
    }

    public CosineSimilarity(INDArray[] inputs, INDArray[] outputs, boolean keepDims, int[] dimensions) {
        super(inputs, outputs, keepDims, dimensions);
    }

    public CosineSimilarity(INDArray[] inputs, boolean keepDims, int[] dimensions) {
        super(inputs, keepDims, dimensions);
    }

    public CosineSimilarity(boolean keepDims, boolean isComplex, boolean isEmptyReduce, int[] dimensions) {
        super(keepDims, isComplex, isEmptyReduce, dimensions);
    }

    public CosineSimilarity(SameDiff sameDiff, SDVariable arg, boolean keepDims, boolean isComplex, boolean isEmptyReduce, int[] dimensions) {
        super(sameDiff, arg, keepDims, isComplex, isEmptyReduce, dimensions);
    }

    public CosineSimilarity(SameDiff sameDiff, SDVariable[] args, boolean keepDims, boolean isComplex, boolean isEmptyReduce, int[] dimensions) {
        super(sameDiff, args, keepDims, isComplex, isEmptyReduce, dimensions);
    }

    public CosineSimilarity(String opName, SameDiff sameDiff, SDVariable[] args, boolean keepDims, boolean isComplex, boolean isEmptyReduce, int[] dimensions) {
        super(opName, sameDiff, args, keepDims, isComplex, isEmptyReduce, dimensions);
    }

    public CosineSimilarity(String opName, INDArray input, INDArray output, List<Double> tArguments, int[] iArguments, boolean keepDims, boolean isComplex, boolean isEmptyReduce, int[] dimensions) {
        super(opName, input, output, tArguments, iArguments, keepDims, isComplex, isEmptyReduce, dimensions);
    }

    public CosineSimilarity(String opName, INDArray[] inputs, INDArray[] outputs, List<Double> tArguments, int[] iArguments, boolean keepDims, boolean isComplex, boolean isEmptyReduce, int[] dimensions) {
        super(opName, inputs, outputs, tArguments, iArguments, keepDims, isComplex, isEmptyReduce, dimensions);
    }

    public CosineSimilarity(String opName, INDArray[] inputs, INDArray[] outputs, List<Double> tArguments, List<Integer> iArguments, boolean keepDims, boolean isComplex, boolean isEmptyReduce, int[] dimensions) {
        super(opName, inputs, outputs, tArguments, iArguments, keepDims, isComplex, isEmptyReduce, dimensions);
    }

    public CosineSimilarity(INDArray[] inputs, INDArray[] outputs, boolean keepDims, boolean isComplex, boolean isEmptyReduce, int[] dimensions) {
        super(inputs, outputs, keepDims, isComplex, isEmptyReduce, dimensions);
    }

    public CosineSimilarity(String opName, INDArray[] inputs, INDArray[] outputs, boolean keepDims, boolean isComplex, boolean isEmptyReduce, int[] dimensions) {
        super(opName, inputs, outputs, keepDims, isComplex, isEmptyReduce, dimensions);
    }

    public CosineSimilarity(String opName, SameDiff sameDiff, SDVariable[] args, boolean inPlace, boolean keepDims, boolean isComplex, boolean isEmptyReduce, int[] dimensions) {
        super(opName, sameDiff, args, inPlace, keepDims, isComplex, isEmptyReduce, dimensions);
    }

    public CosineSimilarity(SameDiff sameDiff, SDVariable[] args, boolean inPlace, boolean keepDims, boolean isComplex, boolean isEmptyReduce, int[] dimensions) {
        super(sameDiff, args, inPlace, keepDims, isComplex, isEmptyReduce, dimensions);
    }

    public CosineSimilarity(String opName, boolean keepDims, boolean isComplex, boolean isEmptyReduce, int[] dimensions) {
        super(opName, keepDims, isComplex, isEmptyReduce, dimensions);
    }

    public CosineSimilarity(INDArray[] input, INDArray[] output, boolean keepDims, boolean isComplex, int[] dimensions) {
        super(input, output, keepDims, isComplex, dimensions);
    }

    @Override
    public int opNum() {
        return 2;
    }

    @Override
    public String opName() {
        return OP_NAME;
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v1) {
        //Let cosine(x,y) = a / b
        //a = sum_i (x_i * y_i)
        //b = sqrt(sum_i x_i^2) * sqrt(sum_i y_i^2) = l2(x) * l2(y)
        //Then:
        // dc(x,y)/dx_i = 1/b * (y - x * a / (l2(x))^2)

        return doDiff(sameDiff, larg(), rarg(), i_v1.get(0), keepDims, dimensions);
    }

    protected void addArgs() {
        super.addArgs();
        addBArgument(isComplex);
    }



    public static List<SDVariable> doDiff(SameDiff sameDiff, SDVariable x, SDVariable y,
                                          SDVariable gradOut, boolean keepDims, int... dimensions){
        SDVariable a = sameDiff.sum(x.mul(y),true, dimensions);
        SDVariable l2x = sameDiff.norm2(x, true, dimensions);
        SDVariable l2y = sameDiff.norm2(y, true, dimensions);
        SDVariable b = l2x.mul(l2y);

        SDVariable l2xSq = sameDiff.math().square(l2x);
        SDVariable l2ySq = sameDiff.math().square(l2y);
        SDVariable broadcastableGrad;
        if(keepDims || dimensions == null || dimensions.length == 0 || (dimensions.length == 1 && dimensions[0] == Integer.MAX_VALUE)){
            //keepDims or full array reduction
            broadcastableGrad = gradOut;
        } else {
            broadcastableGrad = SameDiffUtils.reductionBroadcastableWithOrigShape(x, sameDiff.constant(Nd4j.createFromArray(dimensions)), gradOut);
        }

        SDVariable dcdx = y.sub(x.mul(a).div(l2xSq)).div(b);
        SDVariable dcdy = x.sub(y.mul(a).div(l2ySq)).div(b);

        return Arrays.asList(dcdx.mul(broadcastableGrad), dcdy.mul(broadcastableGrad));
    }
}
