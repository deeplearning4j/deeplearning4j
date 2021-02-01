/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.linalg.api.ops.impl.reduce3;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.util.SameDiffUtils;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;
import java.util.List;

/**
 * Jaccard distance (dissimilarity)
 *
 * @author raver119@gmail.com
 */
public class JaccardDistance extends BaseReduce3Op {

    public JaccardDistance(SameDiff sameDiff, SDVariable i_v, SDVariable i_v2, int... dimensions) {
        super(sameDiff, i_v, i_v2, dimensions);
        extraArgs = new Object[]{0.0f, 0.0f};
    }

    public JaccardDistance() {

    }

    public JaccardDistance(INDArray x, INDArray y, int... dimensions) {
        this(x, y, null, false, dimensions);
    }

    public JaccardDistance(INDArray x, INDArray y, INDArray z, boolean allDistances, int... dimensions) {
        this(x, y, z, false, allDistances, dimensions);
        this.isComplex = allDistances;
    }

    public JaccardDistance(INDArray x, INDArray y, INDArray z) {
        this(x, y, z, false, null);
    }

    public JaccardDistance(INDArray x, INDArray y, boolean allDistances) {
        this(x, y);
        this.isComplex = allDistances;
    }

    public JaccardDistance(INDArray x, INDArray y, INDArray z, boolean keepDims, boolean allDistances, int... dimensions){
        super(x, y, z, keepDims, allDistances, dimensions);
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

    @Override
    public DataType resultType() {
        return Nd4j.defaultFloatingPointType();
    }
}
