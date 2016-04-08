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

package org.nd4j.linalg.api.ops.impl.accum;

import org.apache.commons.math3.util.FastMath;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseAccumulation;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ArrayUtil;

/**
 * Variance with bias correction.
 * Bias can either be divided by n or adjusted with:
 * (currentResult - (pow(bias, 2.0) / n())) / (n() - 1.0);
 *
 * @author Adam Gibson
 */
public class Variance extends BaseAccumulation {
    protected double mean, bias;
    protected  boolean biasCorrected = true;

    public Variance() {
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
    public double op(double origin) {
        return origin - mean;
    }

    @Override
    public float op(float origin) {
        return (float) (origin - mean);
    }

    @Override
    public double update(double accum, double x){
        return accum + x * x; //variance = 1/(n-1) * sum (x-mean)^2
    }

    @Override
    public double update(double accum, double x, double y){
        return accum + x * x;
    }

    @Override
    public float update(float accum, float x) {
        return accum + x * x;
    }

    @Override
    public float update(float accum, float x, float y) {
        return accum +  x * x;
    }

    @Override
    public IComplexNumber update( IComplexNumber accum, double x) {
        double dev = x - mean;
        return accum.add(dev*dev);
    }

    @Override
    public IComplexNumber update( IComplexNumber accum, double x, double y) {
        double dev = x - mean;
        return accum.add(dev*dev);
    }

    @Override
    public IComplexNumber update( IComplexNumber accum, IComplexNumber x) {
        IComplexNumber dev = x.sub(mean);
        return accum.add(dev.mul(dev));
    }

    @Override
    public IComplexNumber update( IComplexNumber accum, IComplexNumber x, IComplexNumber y){
        IComplexNumber dev = x.sub(mean);
        return accum.add(dev.mul(dev));
    }

    @Override
    public IComplexNumber update(IComplexNumber accum, IComplexNumber x, double y) {
        IComplexNumber dev = x.sub(mean);
        return accum.add(dev.mul(dev));
    }

    @Override
    public int opNum() {
        return 0;
    }

    @Override
    public String name() {
        return "var";
    }


    @Override
    public Op opForDimension(int index, int dimension) {
        INDArray xAlongDimension = x.vectorAlongDimension(index, dimension);

        Variance ret;
        if (y() != null)
            ret = new Variance(xAlongDimension, y.vectorAlongDimension(index, dimension), xAlongDimension.length());
        else
            ret = new Variance(x.vectorAlongDimension(index, dimension));
        ret.setBiasCorrected(biasCorrected);
        ret.setApplyFinalTransform(applyFinalTransform());
        return ret;
    }

    @Override
    public Variance opForDimension(int index, int... dimension) {
        INDArray xAlongDimension = x.tensorAlongDimension(index, dimension);

        Variance ret;
        if (y() != null)
            ret = new Variance(xAlongDimension, y.tensorAlongDimension(index, dimension), xAlongDimension.length());
        else
            ret = new Variance(x.tensorAlongDimension(index, dimension),biasCorrected);
        ret.setApplyFinalTransform(applyFinalTransform());
        return ret;
    }

    @Override
    public void init(INDArray x, INDArray y, INDArray z, long n) {
        super.init(x, y, z, n);
        if(Nd4j.executionMode == OpExecutioner.ExecutionMode.JAVA) {
            if (biasCorrected)
                this.bias = Nd4j.getExecutioner().execAndReturn(new Bias(x)).getFinalResult().doubleValue();
            mean = Nd4j.getExecutioner().execAndReturn(new Mean(x)).getFinalResult().doubleValue();
        }

    }

    @Override
    public boolean isPassThrough() {
        return true;
    }

    @Override
    public void exec(){
        if (biasCorrected)
            this.bias = Nd4j.getExecutioner().execAndReturn(new Bias(x)).getFinalResult().doubleValue();
        this.mean = Nd4j.getExecutioner().execAndReturn(new Mean(x)).getFinalResult().doubleValue();


        INDArray xSubMean = x.sub(mean);
        INDArray squared = xSubMean.muli(xSubMean);
        double accum = Nd4j.getExecutioner().execAndReturn(new Sum(squared)).getFinalResult().doubleValue();
        getAndSetFinalResult(accum);
        this.z = Nd4j.scalar(this.finalResult);

    }

    @Override
    public void exec(int... dimension) {
        if(dimension.length == 1 && dimension[0] == Integer.MAX_VALUE) {
            exec();
            return;
        }
        int[] retShape = ArrayUtil.removeIndex(x.shape(), dimension);
        long nOps = x.tensorssAlongDimension(dimension);
        z = Nd4j.create(retShape);
        for( int i = 0; i < nOps; i++ ) {
            double d = Nd4j.getExecutioner().execAndReturn(opForDimension(i, dimension)).getFinalResult().doubleValue();
            z.putScalar(i, d);
        }
    }

    @Override
    public double combineSubResults(double first, double second){
        return first + second;
    }

    @Override
    public float combineSubResults(float first, float second){
        return first + second;
    }

    @Override
    public IComplexNumber combineSubResults(IComplexNumber first, IComplexNumber second){
        return first.add(second);
    }


    @Override
    public double getAndSetFinalResult(double accum) {
        //accumulation is sum_i (x_i-mean)^2
        double result;
        if (biasCorrected)
            result = (accum - (FastMath.pow(bias, 2.0) / n())) / (n() - 1.0);
        else
            result = accum / (double) n();
        this.finalResult = result;
        return result;
    }

    @Override
    public float getAndSetFinalResult(float accum) {
        return (float) getAndSetFinalResult((double) accum);
    }

    @Override
    public IComplexNumber getAndSetFinalResult(IComplexNumber accum) {
   /*     if (biasCorrected)
            finalResultComplex = (accum.sub(ComplexUtil.pow(Nd4j.createComplexNumber(bias, 0), 2.0).div(Nd4j.createComplexNumber(n(), 0))).div(Nd4j.createComplexNumber(n() - 1.0, 0.0)));
        else finalResultComplex = accum.divi(n - 1);
        return finalResultComplex;*/
        throw new UnsupportedOperationException();
    }

    @Override
    public double calculateFinalResult(double accum, long n) {
        //accumulation is sum_i (x_i-mean)^2
        double result;
        if (biasCorrected)
            result = (accum - (FastMath.pow(bias, 2.0) / n)) / (n - 1.0);
        else
            result = accum / (double) n;
        this.finalResult = result;
        return result;
    }

    @Override
    public float calculateFinalResult(float accum, long n) {
        //accumulation is sum_i (x_i-mean)^2
        return (float) calculateFinalResult((double) accum, n);
    }

    public boolean isBiasCorrected() {
        return biasCorrected;
    }

    public void setBiasCorrected(boolean biasCorrected) {
        this.biasCorrected = biasCorrected;
    }
}
