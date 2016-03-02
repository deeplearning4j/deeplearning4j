/*
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

package org.nd4j.linalg.api.ops.impl.transforms;

import org.apache.commons.math3.util.FastMath;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseTransformOp;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.factory.Nd4j;

/**Log(softmax(X))
 * @author Alex Black
 */

public class LogSoftMax extends BaseTransformOp {
    public LogSoftMax() {
    }

    public LogSoftMax(INDArray x, INDArray z) {
        super(x, z);
    }

    public LogSoftMax(INDArray x, INDArray z, int n) {
        super(x, z, n);
    }

    public LogSoftMax(INDArray x, INDArray y, INDArray z, int n) {
        super(x, y, z, n);
    }

    public LogSoftMax(INDArray x, INDArray y, INDArray z) {
        super(x, y, z, x.length());
    }

    public LogSoftMax(INDArray x) {
        super(x);
    }

    @Override
    public int opNum() {
        throw new UnsupportedOperationException();
    }


    @Override
    public String name() {
        return "logsoftmax";
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, double other) {
        return origin;
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, float other) {
        return origin;
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, IComplexNumber other) {
        return origin;
    }

    @Override
    public float op(float origin, float other) {
        return origin;
    }

    @Override
    public double op(double origin, double other) {
        return origin;
    }

    @Override
    public double op(double origin) {
        return origin;
    }

    @Override
    public float op(float origin) {
        return origin;
    }

    @Override
    public IComplexNumber op(IComplexNumber origin) {
        return origin;
    }


    @Override
    public Op opForDimension(int index, int dimension) {
        INDArray xAlongDimension = x.vectorAlongDimension(index, dimension);
        if (y() != null)
            return new LogSoftMax(xAlongDimension, y.vectorAlongDimension(index, dimension), z.vectorAlongDimension(index, dimension), xAlongDimension.length());
        else
            return new LogSoftMax(xAlongDimension, z.vectorAlongDimension(index, dimension), xAlongDimension.length());

    }

    @Override
    public Op opForDimension(int index, int... dimension) {
        INDArray xAlongDimension = x.tensorAlongDimension(index, dimension);
        if (y() != null)
            return new LogSoftMax(xAlongDimension, y.tensorAlongDimension(index, dimension), z.tensorAlongDimension(index, dimension), xAlongDimension.length());
        else
            return new LogSoftMax(xAlongDimension, z.tensorAlongDimension(index, dimension), xAlongDimension.length());

    }

    @Override
    public void exec() {
        exec(1);
    }

    @Override
    public void init(INDArray x, INDArray y, INDArray z, int n) {
        super.init(x, y, z, n);
        passThrough = true;
    }


    @Override
    public void exec(int... dimensions) {
        if(dimensions[0] != 1)
            throw new IllegalArgumentException("Only supports row wise calculations");
        if(x.isMatrix()) {

            INDArray rowMax = x.max(1);
            INDArray xMinusRowMax = x.subColumnVector(rowMax);
            INDArray expXMinusRowMax = Nd4j.getExecutioner().execAndReturn(new Exp(xMinusRowMax.dup()));
            INDArray logRowSumExp = expXMinusRowMax.sum(1);
            Nd4j.getExecutioner().exec(new Log(logRowSumExp));

            INDArray logsoftmax = xMinusRowMax.subiColumnVector(logRowSumExp);
            if(this.z != null)
                z.assign(logsoftmax);
            else
                this.z = logsoftmax;
        }
        else if(x.isVector()) {
           double max = x.maxNumber().doubleValue();
            INDArray xMinusMax = x.sub(max);
            INDArray expXMinusMax = Nd4j.getExecutioner().execAndReturn(new Exp(xMinusMax.dup()));
            double logRowSumExp = FastMath.log(expXMinusMax.sumNumber().doubleValue());

            INDArray logsoftmax = xMinusMax.subi(logRowSumExp);
            if(this.z != null) z.assign(logsoftmax);
            else this.z = logsoftmax;
        }
    }
}
