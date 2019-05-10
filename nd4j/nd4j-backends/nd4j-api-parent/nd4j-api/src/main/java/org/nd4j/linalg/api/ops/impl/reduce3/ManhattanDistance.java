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

package org.nd4j.linalg.api.ops.impl.reduce3;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseReduceFloatOp;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;
import java.util.List;

/**
 * Manhattan distance
 *
 * @author Adam Gibson
 */
public class ManhattanDistance extends BaseReduce3Op {
    public static final String OP_NAME = "manhattan";

    public ManhattanDistance(SameDiff sameDiff, SDVariable i_v, int[] dimensions) {
        super(sameDiff, i_v, dimensions);
    }

    public ManhattanDistance(SameDiff sameDiff, SDVariable i_v, SDVariable i_v2, int... dimensions) {
        super(sameDiff, i_v, i_v2, dimensions);
    }

    public ManhattanDistance() {}


    public ManhattanDistance(INDArray x, INDArray y, int... dimensions) {
        this(x, y, false, dimensions);
    }

    public ManhattanDistance(INDArray x, INDArray y, boolean allDistances, int... dimensions) {
        this(x, y, null, false, allDistances, dimensions);
        this.isComplex = allDistances;
    }

    public ManhattanDistance(INDArray x, INDArray y, INDArray z) {
        this(x, y, z, false, null);
    }

    public ManhattanDistance(INDArray x, INDArray y, INDArray z, boolean allDistances, int... dimensions) {
        this(x, y, z, false, allDistances, dimensions);
    }

    public ManhattanDistance(INDArray x, INDArray y, INDArray z, boolean keepDims, boolean allDistances, int... dimensions){
        super(x, y, z, keepDims, allDistances, dimensions);
        extraArgs = new Object[]{0.0f, 0.0f};
    }

    @Override
    public int opNum() {
        return 0;
    }

    @Override
    public String opName() {
        return OP_NAME;
    }



    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v1) {
        //ddist(x,y)/dxi = sign(xi-yi)
        SDVariable difference = larg().sub(rarg());
        SDVariable gradBroadcastable;
        if(keepDims || dimensions == null || dimensions.length == 0 || (dimensions.length == 1 && dimensions[0] == Integer.MAX_VALUE)){
            //keepDims or full array reduction
            gradBroadcastable = i_v1.get(0);
        } else {
            gradBroadcastable = sameDiff.f().reductionBroadcastableWithOrigShape(arg(), sameDiff.constant(Nd4j.createFromArray(dimensions)), i_v1.get(0));
        }

        SDVariable gradX = sameDiff.math().sign(difference).mul(gradBroadcastable);
        SDVariable gradY = f().neg(gradX);
        return Arrays.asList(gradX, gradY);
    }
}
