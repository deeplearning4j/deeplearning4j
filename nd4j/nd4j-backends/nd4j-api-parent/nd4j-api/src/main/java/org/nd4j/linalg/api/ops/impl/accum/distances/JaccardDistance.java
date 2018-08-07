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
        extraArgs = new Object[]{0.0f, 0.0f};
    }

    public JaccardDistance() {
        passThrough = false;
    }

    public JaccardDistance(INDArray x, INDArray y, INDArray z, long n) {
        super(x, y, z, n);
        passThrough = Nd4j.getExecutioner().executionMode() == OpExecutioner.ExecutionMode.JAVA;
        extraArgs = new Object[]{0.0f, 0.0f};
    }

    public JaccardDistance(INDArray x, INDArray y, long n) {
        super(x, y, n);
        passThrough = Nd4j.getExecutioner().executionMode() == OpExecutioner.ExecutionMode.JAVA;
        extraArgs = new Object[]{0.0f, 0.0f};
    }

    public JaccardDistance(INDArray x) {
        super(x);
        passThrough = Nd4j.getExecutioner().executionMode() == OpExecutioner.ExecutionMode.JAVA;
        extraArgs = new Object[]{0.0f, 0.0f};
    }

    public JaccardDistance(INDArray x, INDArray y) {
        super(x, y);
        passThrough = Nd4j.getExecutioner().executionMode() == OpExecutioner.ExecutionMode.JAVA;
        extraArgs = new Object[]{0.0f, 0.0f};
    }

    public JaccardDistance(INDArray x, INDArray y, INDArray z, boolean allDistances) {
        this(x, y, z, x.lengthLong());
        this.isComplex = allDistances;
    }

    public JaccardDistance(INDArray x, INDArray y, boolean allDistances) {
        this(x, y);
        this.isComplex = allDistances;
    }

    public JaccardDistance(INDArray x, INDArray y, INDArray z, boolean newFormat, boolean keepDims, int... dimensions){
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

        SDVariable min = f().min(larg(), rarg());
        SDVariable max = f().max(larg(), rarg());
        SDVariable sumMax = max.sum(true, dimensions);
        SDVariable sumMin = min.sum(true, dimensions);

        SDVariable xIsMin = f().eq(min, larg());
        SDVariable xIsMax = f().eq(max, larg());
        SDVariable yIsMin = f().eq(min, rarg());
        SDVariable yIsMax = f().eq(max, rarg());

        SDVariable sqSumMax = f().square(sumMax);
        SDVariable dldx = xIsMax.mul(sumMin).sub(xIsMin.mul(sumMax)).div(sqSumMax);
        SDVariable dldy = yIsMax.mul(sumMin).sub(yIsMin.mul(sumMax)).div(sqSumMax);

        SDVariable bcGradOut;
        if(dimensions == null){
            bcGradOut = f1.get(0);
        } else {
            int inRank = arg().getArr().rank();
            bcGradOut = f().reductionBroadcastableWithOrigShape(inRank, dimensions, f1.get(0));
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
