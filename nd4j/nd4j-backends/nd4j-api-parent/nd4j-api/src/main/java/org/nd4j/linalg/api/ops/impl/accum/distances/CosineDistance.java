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
import org.nd4j.linalg.api.ops.BaseReduceOp;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;
import java.util.List;

/**
 * Cosine distance
 * Note that you need to initialize
 * a scaling constant equal to the norm2 of the
 * vector
 *
 * @author raver119@gmail.com
 */
public class CosineDistance extends BaseReduceOp {

    public CosineDistance(SameDiff sameDiff, SDVariable i_v, int[] dimensions, Number constantNormalizedByNorm2X, Number constantNormalizedByNorm2Y) {
        super(sameDiff, i_v, dimensions);
    }

    public CosineDistance(SameDiff sameDiff, SDVariable i_v, SDVariable i_v2, int... dimensions) {
        super(sameDiff, i_v, i_v2, dimensions);
    }

    public CosineDistance() {
        passThrough = true;
    }

    public CosineDistance(INDArray x, INDArray y, INDArray z, long n) {
        super(x, y, z, n);
        passThrough = Nd4j.getExecutioner().executionMode() == OpExecutioner.ExecutionMode.JAVA;
        extraArgs = new Object[]{0.0f, 0.0f};
    }

    public CosineDistance(INDArray x, INDArray y, long n) {
        super(x, y, n);
        passThrough = Nd4j.getExecutioner().executionMode() == OpExecutioner.ExecutionMode.JAVA;
        extraArgs = new Object[]{0.0f, 0.0f};
    }

    public CosineDistance(INDArray x) {
        super(x);
        passThrough = Nd4j.getExecutioner().executionMode() == OpExecutioner.ExecutionMode.JAVA;
        extraArgs = new Object[]{0.0f, 0.0f};
    }

    public CosineDistance(INDArray x, INDArray y) {
        super(x, y);
        passThrough = Nd4j.getExecutioner().executionMode() == OpExecutioner.ExecutionMode.JAVA;
        extraArgs = new Object[]{0.0f, 0.0f};
    }

    public CosineDistance(INDArray x, INDArray y, INDArray z, boolean allDistances) {
        this(x, y, z, x.lengthLong());
        this.isComplex = allDistances;
    }

    public CosineDistance(INDArray x, INDArray y, boolean allDistances) {
        this(x, y);
        this.isComplex = allDistances;
    }

    public CosineDistance(INDArray x, INDArray y, INDArray z, boolean newFormat, boolean keepDims, int... dimensions){
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
        return 5;
    }

    @Override
    public String opName() {
        return "cosinedistance";
    }



    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v1) {
        //Cosine distance = 1 - cosine similarity
        //Therefore: just need to negate gradients from cosine similarity...

        List<SDVariable> diff = CosineSimilarity.doDiff(sameDiff, f(), larg(), rarg(), i_v1.get(0), dimensions);
        return Arrays.asList(f().neg(diff.get(0)), f().neg(diff.get(1)));
    }

    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx op opName found for " +  opName());
    }


    @Override
    public String tensorflowName() {
        return "cosine_distance";
    }


}
