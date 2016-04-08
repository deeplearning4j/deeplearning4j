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

import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseAccumulation;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ArrayUtil;

/**
 * Calculate a bias
 *
 * @author Adam Gibson
 */
public class Bias extends BaseAccumulation {

    private double mean;

    public Bias() {
    }

    public Bias(INDArray x, INDArray y, INDArray z, long n) {
        super(x, y, z, n);
    }

    public Bias(INDArray x, INDArray y, long n) {
        this(x, y, x, n);
    }

    public Bias(INDArray x) {
        super(x);
    }

    public Bias(INDArray x, INDArray y) {
        super(x, y);
    }

    @Override
    public int opNum() {
        return 2;
    }

    @Override
    public String name() {
        return "bias";
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, IComplexNumber other) {
        throw new UnsupportedOperationException();
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, float other) {
        throw new UnsupportedOperationException();
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, double other) {
        throw new UnsupportedOperationException();

    }

    @Override
    public Op opForDimension(int index, int dimension) {
        INDArray xAlongDimension = x.vectorAlongDimension(index, dimension);
        if (y() != null)
            return new Bias(xAlongDimension, y.vectorAlongDimension(index, dimension), xAlongDimension.length());
        else
            return new Bias(x.vectorAlongDimension(index, dimension));
    }

    @Override
    public Op opForDimension(int index, int... dimension) {
        INDArray xAlongDimension = x.tensorAlongDimension(index, dimension);
        if (y() != null)
            return new Bias(xAlongDimension, y.tensorAlongDimension(index, dimension), xAlongDimension.length());
        else
            return new Bias(x.tensorAlongDimension(index, dimension));
    }

    @Override
    public IComplexNumber op(IComplexNumber origin) {
        return origin.sub(mean);
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
        return accum + x;
    }

    @Override
    public double update(double accum, double x, double y){
        return accum + x;
    }

    @Override
    public float update(float accum, float x){
        return accum + x;
    }

    @Override
    public float update(float accum, float x, float y) {
        return accum + x;
    }

    @Override
    public IComplexNumber update( IComplexNumber accum, double x) {
        return accum.add(x);
    }

    @Override
    public IComplexNumber update( IComplexNumber accum, double x, double y){
        return accum.add(x);
    }

    @Override
    public IComplexNumber update( IComplexNumber accum, IComplexNumber x){
        return accum.add(x);
    }

    @Override
    public IComplexNumber update( IComplexNumber accum, IComplexNumber x, IComplexNumber y) {
        return accum.add(x);
    }

    @Override
    public IComplexNumber update(IComplexNumber accum, IComplexNumber x, double y) {
        return accum.add(x);
    }

    @Override
    public IComplexNumber zeroComplex() {
        return Nd4j.createComplexNumber(0.0, 0.0);
    }

    @Override
    public void init(INDArray x, INDArray y, INDArray z, long n) {
        super.init(x, y, z, n);
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
    public boolean isPassThrough() {
       return false;
    }

    @Override
    public void exec() {
        this.mean = Nd4j.getExecutioner().execAndReturn(new Mean(x)).getFinalResult().doubleValue();
        INDArray xMinusMean = x.sub(mean);
        double sum = Nd4j.getExecutioner().execAndReturn(new Sum(xMinusMean)).getFinalResult().doubleValue();
        this.finalResult = sum;
    }

    @Override
    public void exec(int... dimension) {
        int[] retShape = ArrayUtil.removeIndex(x.shape(), dimension);
        long nOps = x.tensorssAlongDimension(dimension);
        z = Nd4j.create(retShape);
        for( int i = 0; i < nOps; i++ ){
            double d = Nd4j.getExecutioner().execAndReturn((Bias)opForDimension(i,dimension)).getFinalResult().doubleValue();
            z.putScalar(i, d);
        }
    }
}
