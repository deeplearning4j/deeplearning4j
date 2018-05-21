/*-
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 *
 */

package org.nd4j.linalg.api.ops.impl.accum.distances;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseAccumulation;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;
import java.util.List;

/**
 * Jaccard distance (dissimilarity)
 *
 * @author raver119@gmail.com
 */
public class JaccardDistance extends BaseAccumulation {

    public JaccardDistance(SameDiff sameDiff, SDVariable i_v, SDVariable i_v2, int... dimensions) {
        super(sameDiff, i_v, i_v2, dimensions);
        extraArgs = new Object[2];
        extraArgs[0] = 0.0f;
        extraArgs[1] = 0.0f;
    }

    public JaccardDistance() {
        passThrough = false;
    }

    public JaccardDistance(INDArray x, INDArray y, INDArray z, long n) {
        super(x, y, z, n);
        passThrough = Nd4j.getExecutioner().executionMode() == OpExecutioner.ExecutionMode.JAVA;
        extraArgs = new Object[2];
        extraArgs[0] = 0.0f;
        extraArgs[1] = 0.0f;
    }

    public JaccardDistance(INDArray x, INDArray y, long n) {
        super(x, y, n);
        passThrough = Nd4j.getExecutioner().executionMode() == OpExecutioner.ExecutionMode.JAVA;
        extraArgs = new Object[2];
        extraArgs[0] = 0.0f;
        extraArgs[1] = 0.0f;
    }

    public JaccardDistance(INDArray x) {
        super(x);
        passThrough = Nd4j.getExecutioner().executionMode() == OpExecutioner.ExecutionMode.JAVA;
        extraArgs = new Object[2];
        extraArgs[0] = 0.0f;
        extraArgs[1] = 0.0f;
    }

    public JaccardDistance(INDArray x, INDArray y) {
        super(x, y);
        passThrough = Nd4j.getExecutioner().executionMode() == OpExecutioner.ExecutionMode.JAVA;
        extraArgs = new Object[2];
        extraArgs[0] = 0.0f;
        extraArgs[1] = 0.0f;
    }

    public JaccardDistance(INDArray x, INDArray y, INDArray z, boolean allDistances) {
        this(x, y, z, x.lengthLong());
        this.isComplex = allDistances;
    }

    public JaccardDistance(INDArray x, INDArray y, boolean allDistances) {
        this(x, y);
        this.isComplex = allDistances;
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
        return 6;
    }

    @Override
    public String opName() {
        return "jaccarddistance";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        //Jaccard distance: https://en.wikipedia.org/wiki/Jaccard_index#Generalized_Jaccard_similarity_and_distance
        //J(x,y) = 1 - sum_i min(x_i, y_i) / sum_i max(x_i, y_i)

        int rank = Shape.rankFromShape(larg().getShape());

        SDVariable jSim = outputVariables()[0].rsub(1.0);   //jaccard similarity = 1 - jaccard distance
        SDVariable min = f().min(larg(), rarg());
        SDVariable max = f().max(larg(), rarg());
        SDVariable sumMax = f().sum(max, dimensions);
        SDVariable broadcastableSumMax = f().reductionBroadcastableWithOrigShape(rank, dimensions, sumMax);
        SDVariable broadcastableJSim = f().reductionBroadcastableWithOrigShape(rank, dimensions, jSim);

        SDVariable xIsMin = f().eq(min, larg());
        SDVariable xIsMax = f().eq(max, larg());
        SDVariable yIsMin = f().eq(min, rarg());
        SDVariable yIsMax = f().eq(max, rarg());

        SDVariable dldx = xIsMax.mul(broadcastableJSim).sub(xIsMin).div(broadcastableSumMax);
        SDVariable dldy = yIsMax.mul(broadcastableJSim).sub(yIsMin).div(broadcastableSumMax);

        return Arrays.asList(dldx.mul(f1.get(0)), dldy.mul(f1.get(0)));
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
