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

package org.nd4j.linalg.api.ops.impl.accum;

import lombok.val;
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
 * Variance with bias correction.
 * Bias can either be divided by n or adjusted with:
 * (currentResult - (pow(bias, 2.0) / n())) / (n() - 1.0);
 *
 * @author Adam Gibson
 */
public class Variance extends BaseAccumulation {
    protected double mean, bias;
    protected boolean biasCorrected = true;

    public Variance(SameDiff sameDiff, SDVariable i_v, int[] dimensions, boolean biasCorrected) {
        super(sameDiff, i_v, dimensions);
        this.biasCorrected = biasCorrected;
    }

    public Variance(SameDiff sameDiff, SDVariable i_v, SDVariable i_v2, int[] dimensions, boolean biasCorrected) {
        super(sameDiff, i_v, i_v2, dimensions);
        this.biasCorrected = biasCorrected;
    }

    public Variance() {
    }

    public Variance(boolean biasCorrected) {
        this.biasCorrected = biasCorrected;
    }

    public Variance(INDArray x, INDArray y, INDArray z, long n) {
        super(x, y, z, n);
        init(x, y, z, n);
    }

    public Variance(INDArray x, INDArray y, long n) {
        this(x, y, x, n);
    }

    public Variance(INDArray x) {
        this(x, null, x, x.lengthLong(), true);
    }

    public Variance(INDArray x, INDArray y) {
        super(x, y);
    }

    public Variance(INDArray x, INDArray y, INDArray z, long n, boolean biasCorrected) {
        super(x, y, z, n);
        this.biasCorrected = biasCorrected;
        init(x, y, z, n);
    }

    public Variance(INDArray x, INDArray y, long n, boolean biasCorrected) {
        super(x, y, n);
        this.biasCorrected = biasCorrected;
        init(x, y, z, n);
    }

    public Variance(INDArray x, boolean biasCorrected) {
        super(x);
        this.biasCorrected = biasCorrected;
        init(x, y, z, n);
    }

    public Variance(INDArray x, INDArray y, boolean biasCorrected) {
        super(x, y);
        this.biasCorrected = biasCorrected;
        init(x, y, x, x.lengthLong());
    }

    @Override
    public INDArray noOp() {
        return Nd4j.zerosLike(x());
    }

    @Override
    public int opNum() {
        return 0;
    }

    @Override
    public String opName() {
        return "var";
    }


    @Override
    public void init(INDArray x, INDArray y, INDArray z, long n) {
        super.init(x, y, z, n);
        if (Nd4j.executionMode == OpExecutioner.ExecutionMode.JAVA) {
            if (biasCorrected)
                this.bias = Nd4j.getExecutioner().execAndReturn(new Bias(x)).getFinalResult().doubleValue();
            mean = Nd4j.getExecutioner().execAndReturn(new Mean(x)).getFinalResult().doubleValue();
        }

    }

    @Override
    public boolean isPassThrough() {
        return true;
    }


    public boolean isBiasCorrected() {
        return biasCorrected;
    }

    public void setBiasCorrected(boolean biasCorrected) {
        this.biasCorrected = biasCorrected;
    }


    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v1) {
        //If out = var(in) then:
        //dL/dIn = dL/dOut * dOut/dIn
        // with dOut/dIn = (in-mean) * 2/(n-1)
        val n = f().getReductionLength(this);
        int origRank = Shape.rankFromShape(arg().getShape());
        SDVariable broadcastableMean = f().reductionBroadcastableWithOrigShape(origRank, dimensions, f().mean(arg(), dimensions));
        SDVariable broadcastableGrad = f().reductionBroadcastableWithOrigShape(origRank, dimensions, i_v1.get(0));
        SDVariable dOutdIn = arg().sub(broadcastableMean).mul(2.0 / (biasCorrected ? (n - 1) : n));

        SDVariable dLdIn = dOutdIn.mul(broadcastableGrad);
        return Arrays.asList(dLdIn);
    }

    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx opName found for " + opName());
    }

    @Override
    public String tensorflowName() {
        return "moments";
    }


    @Override
    public Type getOpType() {
        return Type.VARIANCE;
    }
}
