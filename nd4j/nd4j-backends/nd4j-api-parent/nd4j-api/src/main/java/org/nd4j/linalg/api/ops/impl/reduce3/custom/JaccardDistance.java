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
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.reduce.custom.BaseDynamicCustomReduction;
import org.nd4j.linalg.api.ops.impl.reduce3.BaseReduce3Op;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;
import java.util.List;

public class JaccardDistance extends BaseDynamicCustomReduction {

    public JaccardDistance() {
        super();
    }

    public JaccardDistance(SameDiff sameDiff, SDVariable[] args, boolean keepDims) {
        super(sameDiff, args, keepDims);
    }

    public JaccardDistance(SameDiff sameDiff, SDVariable[] args, boolean keepDims, int[] dimensions) {
        super(sameDiff, args, keepDims, dimensions);
    }

    public JaccardDistance(SameDiff sameDiff, SDVariable[] args, boolean keepDims, boolean isComplex) {
        super(sameDiff, args, keepDims, isComplex);
    }

    public JaccardDistance(SameDiff sameDiff, SDVariable[] args, boolean keepDims, boolean isComplex, int[] dimensions) {
        super(sameDiff, args, keepDims, isComplex, dimensions);
    }

    public JaccardDistance(INDArray[] inputs, INDArray[] outputs) {
        super(inputs, outputs);
    }

    public JaccardDistance(INDArray[] inputs, INDArray[] outputs, boolean keepDims) {
        super(inputs, outputs, keepDims);
    }

    public JaccardDistance(INDArray[] inputs, INDArray[] outputs, boolean keepDims, int[] dimensions) {
        super(inputs, outputs, keepDims, dimensions);
    }

    public JaccardDistance(INDArray[] inputs, boolean keepDims, int[] dimensions) {
        super(inputs, keepDims, dimensions);
    }

    public JaccardDistance(boolean keepDims, boolean isComplex, boolean isEmptyReduce, int[] dimensions) {
        super(keepDims, isComplex, isEmptyReduce, dimensions);
    }

    public JaccardDistance(SameDiff sameDiff, SDVariable arg, boolean keepDims, boolean isComplex, boolean isEmptyReduce, int[] dimensions) {
        super(sameDiff, arg, keepDims, isComplex, isEmptyReduce, dimensions);
    }

    public JaccardDistance(SameDiff sameDiff, SDVariable[] args, boolean keepDims, boolean isComplex, boolean isEmptyReduce, int[] dimensions) {
        super(sameDiff, args, keepDims, isComplex, isEmptyReduce, dimensions);
    }

    public JaccardDistance(String opName, SameDiff sameDiff, SDVariable[] args, boolean keepDims, boolean isComplex, boolean isEmptyReduce, int[] dimensions) {
        super(opName, sameDiff, args, keepDims, isComplex, isEmptyReduce, dimensions);
    }

    public JaccardDistance(String opName, INDArray input, INDArray output, List<Double> tArguments, int[] iArguments, boolean keepDims, boolean isComplex, boolean isEmptyReduce, int[] dimensions) {
        super(opName, input, output, tArguments, iArguments, keepDims, isComplex, isEmptyReduce, dimensions);
    }

    public JaccardDistance(String opName, INDArray[] inputs, INDArray[] outputs, List<Double> tArguments, int[] iArguments, boolean keepDims, boolean isComplex, boolean isEmptyReduce, int[] dimensions) {
        super(opName, inputs, outputs, tArguments, iArguments, keepDims, isComplex, isEmptyReduce, dimensions);
    }

    public JaccardDistance(String opName, INDArray[] inputs, INDArray[] outputs, List<Double> tArguments, List<Integer> iArguments, boolean keepDims, boolean isComplex, boolean isEmptyReduce, int[] dimensions) {
        super(opName, inputs, outputs, tArguments, iArguments, keepDims, isComplex, isEmptyReduce, dimensions);
    }

    public JaccardDistance(INDArray[] inputs, INDArray[] outputs, boolean keepDims, boolean isComplex, boolean isEmptyReduce, int[] dimensions) {
        super(inputs, outputs, keepDims, isComplex, isEmptyReduce, dimensions);
    }

    public JaccardDistance(String opName, INDArray[] inputs, INDArray[] outputs, boolean keepDims, boolean isComplex, boolean isEmptyReduce, int[] dimensions) {
        super(opName, inputs, outputs, keepDims, isComplex, isEmptyReduce, dimensions);
    }

    public JaccardDistance(String opName, SameDiff sameDiff, SDVariable[] args, boolean inPlace, boolean keepDims, boolean isComplex, boolean isEmptyReduce, int[] dimensions) {
        super(opName, sameDiff, args, inPlace, keepDims, isComplex, isEmptyReduce, dimensions);
    }

    public JaccardDistance(SameDiff sameDiff, SDVariable[] args, boolean inPlace, boolean keepDims, boolean isComplex, boolean isEmptyReduce, int[] dimensions) {
        super(sameDiff, args, inPlace, keepDims, isComplex, isEmptyReduce, dimensions);
    }

    public JaccardDistance(String opName, boolean keepDims, boolean isComplex, boolean isEmptyReduce, int[] dimensions) {
        super(opName, keepDims, isComplex, isEmptyReduce, dimensions);
    }

    public JaccardDistance(INDArray[] input, INDArray[] output, boolean keepDims, boolean isComplex, int[] dimensions) {
        super(input, output, keepDims, isComplex, dimensions);
    }


    @Override
    public int opNum() {
        return 6;
    }

    @Override
    public String opName() {
        return "jaccarddistance";
    }

    protected void addArgs() {
        super.addArgs();
        addBArgument(isComplex);
    }


    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        //Jaccard distance: https://en.wikipedia.org/wiki/Jaccard_index#Generalized_Jaccard_similarity_and_distance
        //J(x,y) = 1 - sum_i min(x_i, y_i) / sum_i max(x_i, y_i)

        SDVariable min = sameDiff.math.min(larg(), rarg());
        SDVariable max = sameDiff.math.max(larg(), rarg());
        SDVariable sumMax = max.sum(true, dimensions);
        SDVariable sumMin = min.sum(true, dimensions);

        DataType d = arg().dataType();
        SDVariable xIsMin = sameDiff.eq(min, larg()).castTo(d);
        SDVariable xIsMax = sameDiff.eq(max, larg()).castTo(d);
        SDVariable yIsMin = sameDiff.eq(min, rarg()).castTo(d);
        SDVariable yIsMax = sameDiff.eq(max, rarg()).castTo(d);

        SDVariable sqSumMax = sameDiff.math.square(sumMax);
        SDVariable dldx = xIsMax.mul(sumMin).sub(xIsMin.mul(sumMax)).div(sqSumMax);
        SDVariable dldy = yIsMax.mul(sumMin).sub(yIsMin.mul(sumMax)).div(sqSumMax);

        SDVariable bcGradOut;
        if(keepDims || dimensions == null || dimensions.length == 0 || (dimensions.length == 1 && dimensions[0] == Integer.MAX_VALUE)){
            //KeepDims or full array reduction - already broadcastable
            bcGradOut = f1.get(0);
        } else {
            bcGradOut = SameDiffUtils.reductionBroadcastableWithOrigShape(arg(), sameDiff.constant(Nd4j.createFromArray(dimensions)), f1.get(0));
        }
        return Arrays.asList(dldx.mul(bcGradOut), dldy.mul(bcGradOut));
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
