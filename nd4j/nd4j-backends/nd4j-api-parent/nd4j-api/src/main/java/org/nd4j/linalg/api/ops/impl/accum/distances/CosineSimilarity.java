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

package org.nd4j.linalg.api.ops.impl.accum.distances;

import org.nd4j.autodiff.functions.DifferentialFunctionFactory;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseReduceOp;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;
import java.util.List;

/**
 * Cosine similarity
 * Note that you need to initialize
 * a scaling constant equal to the norm2 of the
 * vector
 *
 * @author Adam Gibson
 */
public class CosineSimilarity extends BaseReduceOp {
    public static final String OP_NAME = "cosinesimilarity";

    public CosineSimilarity(SameDiff sameDiff, SDVariable i_v, int[] dimensions) {
        super(sameDiff, i_v, dimensions);
    }

    public CosineSimilarity(SameDiff sameDiff, SDVariable i_v, SDVariable i_v2, int[] dimensions) {
        super(sameDiff, i_v, i_v2, dimensions);
    }

    public CosineSimilarity() {
        passThrough = true;
    }

    public CosineSimilarity(INDArray x, INDArray y, INDArray z, long n) {
        super(x, y, z, n);
        passThrough = Nd4j.getExecutioner().executionMode() == OpExecutioner.ExecutionMode.JAVA;
        extraArgs = new Object[]{0.0f, 0.0f};
    }

    public CosineSimilarity(INDArray x, INDArray y, long n) {
        super(x, y, n);
        passThrough = Nd4j.getExecutioner().executionMode() == OpExecutioner.ExecutionMode.JAVA;
        extraArgs = new Object[]{0.0f, 0.0f};
    }

    public CosineSimilarity(INDArray x) {
        super(x);
        passThrough = Nd4j.getExecutioner().executionMode() == OpExecutioner.ExecutionMode.JAVA;
        extraArgs = new Object[]{0.0f, 0.0f};
    }

    public CosineSimilarity(INDArray x, INDArray y) {
        super(x, y);
        passThrough = Nd4j.getExecutioner().executionMode() == OpExecutioner.ExecutionMode.JAVA;
        extraArgs = new Object[]{0.0f, 0.0f};
    }

    public CosineSimilarity(INDArray x, INDArray y, INDArray z, boolean allDistances) {
        this(x, y, z, x.lengthLong());
        this.isComplex = allDistances;
    }

    public CosineSimilarity(INDArray x, INDArray y, boolean allDistances) {
        this(x, y);
        this.isComplex = allDistances;
    }

    public CosineSimilarity(INDArray x, INDArray y, INDArray z, boolean newFormat, boolean keepDims, int... dimensions){
        super(x, y, z, newFormat, keepDims, dimensions);
        extraArgs = new Object[]{0.0f, 0.0f};
    }

    @Override
    public Type opType() {
        return Type.REDUCE3;
    }

    @Override
    public Type getOpType() {
        return opType();
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

        return doDiff(sameDiff, f(), larg(), rarg(), i_v1.get(0), dimensions);
    }

    public static List<SDVariable> doDiff(SameDiff sameDiff, DifferentialFunctionFactory f, SDVariable x, SDVariable y,
                                          SDVariable gradOut, int... dimensions){
        SDVariable a = sameDiff.sum(x.mul(y),true, dimensions);
        SDVariable l2x = f.norm2(x, true, dimensions);
        SDVariable l2y = f.norm2(y, true, dimensions);
        SDVariable b = l2x.mul(l2y);

        int origRank = Shape.rankFromShape(x.getShape());
        SDVariable l2xSq = sameDiff.square(l2x);
        SDVariable l2ySq = sameDiff.square(l2y);
        SDVariable broadcastableGrad = f.reductionBroadcastableWithOrigShape(origRank, dimensions, gradOut);

        SDVariable dcdx = y.sub(x.mul(a).div(l2xSq)).div(b);
        SDVariable dcdy = x.sub(y.mul(a).div(l2ySq)).div(b);

        return Arrays.asList(dcdx.mul(broadcastableGrad), dcdy.mul(broadcastableGrad));
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
