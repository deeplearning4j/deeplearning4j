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
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseTransformOp;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ComplexNDArrayUtil;

import static org.nd4j.linalg.ops.transforms.Transforms.exp;


/**
 * Encapsulated vector operation
 *
 * @author Adam Gibson
 */
public class VectorFFT extends BaseTransformOp {
    protected int fftLength;
    private int originalN = -1;
    protected boolean executed = false;

    public VectorFFT() {
    }

    public VectorFFT(INDArray x, INDArray z,int fftLength) {
        super(x, z);
        this.fftLength = fftLength;
        this.n = fftLength;
        exec();
    }

    public VectorFFT(INDArray x, INDArray z, long n,int fftLength) {
        super(x, z, n);
        this.fftLength = fftLength;
        this.n = fftLength;
        exec();
    }

    public VectorFFT(INDArray x, INDArray y, INDArray z, long n,int fftLength) {
        super(x, y, z, n);
        this.z = z;
        this.fftLength = fftLength;
        exec();
    }

    public VectorFFT(INDArray x,int fftLength) {
        super(x);
        this.z = x;
        this.fftLength = fftLength;
        exec();
    }

    public VectorFFT(INDArray x) {
        this(x,x.length());
    }


    @Override
    public int opNum() {
        return 1;
    }

    @Override
    public String name() {
        return "fft";
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
            return new VectorFFT(xAlongDimension, y.vectorAlongDimension(index, dimension), z.vectorAlongDimension(index, dimension), xAlongDimension.length(),fftLength);
        else
            return new VectorFFT(xAlongDimension, z.vectorAlongDimension(index, dimension), xAlongDimension.length(),fftLength);

    }

    @Override
    public Op opForDimension(int index, int... dimension) {
        INDArray xAlongDimension = x.tensorAlongDimension(index, dimension);
        if (y() != null)
            return new VectorFFT(xAlongDimension, y.tensorAlongDimension(index, dimension), z.tensorAlongDimension(index, dimension), xAlongDimension.length(),fftLength);
        else
            return new VectorFFT(xAlongDimension, z.tensorAlongDimension(index, dimension), xAlongDimension.length(),fftLength);

    }

    @Override
    public void exec() {
        if(!x.isVector())
            return;
        if(executed)
            return;

        executed = true;

        double len = fftLength;
        IComplexNDArray ret = x instanceof IComplexNDArray ? (IComplexNDArray) x : Nd4j.createComplex(x);
        int desiredElementsAlongDimension = ret.length();

        if (len > desiredElementsAlongDimension) {
            ret = ComplexNDArrayUtil.padWithZeros(ret, new int[]{fftLength});
        } else if (len < desiredElementsAlongDimension) {
            ret = ComplexNDArrayUtil.truncate(ret, fftLength, 0);
        }


        IComplexNumber c2 = Nd4j.createDouble(0, -2).muli(FastMath.PI);
        //row vector
        INDArray n2 = Nd4j.arange(0, this.fftLength).reshape(1, this.fftLength);

        //column vector
        INDArray k = n2.reshape(n2.length(),1);
        INDArray kTimesN = k.mmul(n2);
        //here
        IComplexNDArray c1 = kTimesN.muli(c2);
        c1.divi(len);
        IComplexNDArray M = (IComplexNDArray) exp(c1);


        IComplexNDArray reshaped = ret.reshape(new int[]{1,ret.length()});
        IComplexNDArray matrix = reshaped.mmul(M);
        if (originalN > 0)
            matrix = ComplexNDArrayUtil.truncate(matrix, originalN, 0);


        //completely pass through
        this.x = matrix;
        this.z = matrix;
    }

    @Override
    public boolean isPassThrough() {
        return true;
    }

    @Override
    public void setX(INDArray x) {
        this.x = x;
        this.fftLength = x.length();
        this.n = x.length();
        executed = false;

    }

    @Override
    public void setZ(INDArray z) {
        this.z = z;
        this.fftLength = z.length();
        this.n = fftLength;
        executed = false;

    }

    @Override
    public void setY(INDArray y) {
        this.y = y;
    }
}
