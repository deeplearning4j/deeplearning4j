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
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.indexing.SpecifiedIndex;

import java.util.Arrays;
import java.util.List;

/**
 * Manhattan distance
 *
 * @author Adam Gibson
 */
public class ManhattanDistance extends BaseAccumulation {
    public static final String OP_NAME = "manhattan";

    public ManhattanDistance(SameDiff sameDiff, SDVariable i_v, int[] dimensions) {
        super(sameDiff, i_v, dimensions);
    }

    public ManhattanDistance(SameDiff sameDiff, SDVariable i_v, SDVariable i_v2, int... dimensions) {
        super(sameDiff, i_v, i_v2, dimensions);
    }

    public ManhattanDistance() {}

    public ManhattanDistance(INDArray x, INDArray y, INDArray z, long n) {
        super(x, y, z, n);
        extraArgs = new Object[2];
        extraArgs[0] = 0.0f;
        extraArgs[1] = 0.0f;
    }

    public ManhattanDistance(INDArray x, INDArray y, long n) {
        super(x, y, n);
        extraArgs = new Object[2];
        extraArgs[0] = 0.0f;
        extraArgs[1] = 0.0f;
    }

    public ManhattanDistance(INDArray x) {
        super(x);
        extraArgs = new Object[2];
        extraArgs[0] = 0.0f;
        extraArgs[1] = 0.0f;
    }

    public ManhattanDistance(INDArray x, INDArray y) {
        super(x, y);
        extraArgs = new Object[2];
        extraArgs[0] = 0.0f;
        extraArgs[1] = 0.0f;
    }

    public ManhattanDistance(INDArray x, INDArray y, boolean allDistances) {
        this(x, y);
        this.isComplex = allDistances;
    }

    public ManhattanDistance(INDArray x, INDArray y, INDArray z, boolean allDistances) {
        this(x, y, z, x.lengthLong());
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
        int origRank = Shape.rankFromShape(arg().getShape());   //TODO shape may not always be defined?
        if(!(dimensions.length == 1 && dimensions[0] == Integer.MAX_VALUE) ){
            //1x1 output case
            gradBroadcastable = i_v1.get(0);
        } else {
            gradBroadcastable = f().reductionBroadcastableWithOrigShape(origRank, dimensions, i_v1.get(0));
        }

        SDVariable gradX = sameDiff.sign(difference).mul(gradBroadcastable);
        SDVariable gradY = f().neg(gradX);
        return Arrays.asList(gradX, gradY);
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
