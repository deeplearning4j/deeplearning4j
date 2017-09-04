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

import org.nd4j.linalg.api.blas.params.MMulTranspose;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseAccumulation;
import org.nd4j.linalg.api.ops.Op;

/**
 * Matrix multiplication/dot product
 *
 * @author Adam Gibson
 */
public class Mmul extends BaseAccumulation {

    private MMulTranspose mMulTranspose;

    public Mmul() {}


    public Mmul(INDArray x,
                INDArray y,
                INDArray z,
                MMulTranspose mMulTranspose) {
        this(x,y,z,x.lengthLong(),mMulTranspose);
    }

    public Mmul(INDArray x,
                INDArray y,
                INDArray z,
                long n,
                MMulTranspose mMulTranspose) {
        super(x, y, z, n);
        this.mMulTranspose = mMulTranspose;
    }

    public Mmul(INDArray x, INDArray y, INDArray z, long n) {
        this(x,y,z,n,MMulTranspose.allFalse());
    }

    public Mmul(INDArray x, INDArray y, INDArray z) {
        this(x,y,z,x.lengthLong());
    }



    public Mmul(INDArray x, INDArray y) {
        super(x, y);
    }

    @Override
    public int opNum() {
        return 3;
    }

    @Override
    public String name() {
        return "mmul";
    }

    @Override
    public Op opForDimension(int index, int dimension) {
        throw new UnsupportedOperationException();


    }

    @Override
    public long n() {
        return 0;
    }

    @Override
    public boolean isPassThrough() {
        return true;
    }

    @Override
    public Op opForDimension(int index, int... dimension) {
        throw new UnsupportedOperationException();

    }

    @Override
    public IComplexNumber op(IComplexNumber origin, double other) {
        numProcessed++;
        return origin.mul(other);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, float other) {
        numProcessed++;
        return origin.mul(other);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, IComplexNumber other) {
        numProcessed++;
        return origin.mul(other);
    }

    @Override
    public float op(float origin, float other) {
        numProcessed++;
        return origin * other;
    }

    @Override
    public double op(double origin, double other) {
        numProcessed++;
        return origin * other;
    }

    @Override
    public double op(double origin) {
        numProcessed++;
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
    public double update(double accum, double x) {
        return accum + x;
    }

    @Override
    public double update(double accum, double x, double y) {
        return accum + x * y;
    }

    @Override
    public float update(float accum, float x) {
        return accum + x;
    }

    @Override
    public float update(float accum, float x, float y) {
        return accum + x * y;
    }

    @Override
    public IComplexNumber update(IComplexNumber accum, double x) {
        return accum.add(x);
    }

    @Override
    public IComplexNumber update(IComplexNumber accum, double x, double y) {
        return accum.add(x * y);
    }

    @Override
    public IComplexNumber update(IComplexNumber accum, IComplexNumber x) {
        return accum.add(x);
    }

    @Override
    public IComplexNumber update(IComplexNumber accum, IComplexNumber x, IComplexNumber y) {
        return accum.add(x.mul(y));
    }

    @Override
    public IComplexNumber update(IComplexNumber accum, IComplexNumber x, double y) {
        return accum.add(x.mul(y));
    }

    @Override
    public double combineSubResults(double first, double second) {
        return first + second;
    }

    @Override
    public float combineSubResults(float first, float second) {
        return first + second;
    }

    @Override
    public IComplexNumber combineSubResults(IComplexNumber first, IComplexNumber second) {
        return first.add(second);
    }

    @Override
    public boolean isExecSpecial() {
        return true;
    }

    @Override
    public void exec() {
        if(this.z != null)
            x.mmul(y,z,mMulTranspose);
        else
            this.z = x.mmul(y,mMulTranspose);
    }
}
