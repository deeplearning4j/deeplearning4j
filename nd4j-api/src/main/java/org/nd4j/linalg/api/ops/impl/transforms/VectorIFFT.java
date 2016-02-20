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

import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseTransformOp;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ComplexNDArrayUtil;

/**
 * Single ifft operation
 *
 * @author Adam Gibson
 */
public class VectorIFFT extends BaseTransformOp {


    protected int fftLength;
    private int originalN = -1;
    protected boolean executed = false;

    public VectorIFFT() {
    }

    public VectorIFFT(INDArray x, INDArray z,int fftLength) {
        super(x, z);
        this.fftLength = fftLength;
        exec();
    }

    public VectorIFFT(INDArray x, INDArray z, int n,int fftLength) {
        super(x, z, n);
        this.fftLength = fftLength;
        exec();
    }

    public VectorIFFT(INDArray x, INDArray y, INDArray z, int n,int fftLength) {
        super(x, y, z, n);
        this.fftLength = fftLength;
        exec();
    }

    public VectorIFFT(INDArray x,int fftLength) {
        super(x);
        this.fftLength = fftLength;
        exec();
    }

    public VectorIFFT(INDArray x) {
        this(x,x.length());
    }

    @Override
    public int opNum() {
        return 0;
    }

    @Override
    public String name() {
        return "ifft";
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
            return new VectorIFFT(xAlongDimension, y.vectorAlongDimension(index, dimension), z.vectorAlongDimension(index, dimension), xAlongDimension.length(),fftLength);
        else
            return new VectorIFFT(xAlongDimension, z.vectorAlongDimension(index, dimension), xAlongDimension.length(),fftLength);

    }

    @Override
    public Op opForDimension(int index, int... dimension) {
        INDArray xAlongDimension = x.tensorAlongDimension(index, dimension);
        if (y() != null)
            return new VectorIFFT(xAlongDimension, y.tensorAlongDimension(index, dimension), z.tensorAlongDimension(index, dimension), xAlongDimension.length(),fftLength);
        else
            return new VectorIFFT(xAlongDimension, z.tensorAlongDimension(index, dimension), xAlongDimension.length(),fftLength);

    }

    @Override
    public boolean isPassThrough() {
        return true;
    }



    @Override
    public void setX(INDArray x) {
        this.x = x;
        executed = false;
        this.fftLength = z.length();
        this.n = fftLength;
    }

    @Override
    public void setZ(INDArray z) {
        this.z = z;
        executed = false;
        this.fftLength = z.length();
        this.n = fftLength;

    }

    @Override
    public void exec() {
        if(!x.isVector())
            return;
        if(executed)
            return;

        executed = true;



        //ifft(x) = conj(fft(conj(x)) / length(x)
        IComplexNDArray ndArray = x instanceof IComplexNDArray ? (IComplexNDArray) x : Nd4j.createComplex(x);
        IComplexNDArray fft = (IComplexNDArray) Nd4j.getExecutioner().execAndReturn(new VectorFFT(ndArray.conj(),y,z,x.length(),fftLength));
        IComplexNDArray ret = fft.conj().divi(Nd4j.complexScalar(fftLength));
        //completely pass through
        this.z = originalN > 0 ? ComplexNDArrayUtil.truncate(ret, originalN, 0) : ret;
        this.x = this.z;
    }
}
