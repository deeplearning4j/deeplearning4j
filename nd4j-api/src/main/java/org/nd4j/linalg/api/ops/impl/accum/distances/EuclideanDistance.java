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

package org.nd4j.linalg.api.ops.impl.accum.distances;

import org.apache.commons.math3.util.FastMath;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseAccumulation;
import org.nd4j.linalg.api.ops.Op;

/**
 * Euclidean distance
 *
 * @author Adam Gibson
 */
public class EuclideanDistance extends BaseAccumulation {
    public EuclideanDistance() {
    }

    public EuclideanDistance(INDArray x, INDArray y, INDArray z, int n) {
        super(x, y, z, n);
    }

    public EuclideanDistance(INDArray x, INDArray y, int n) {
        super(x, y, n);
    }

    public EuclideanDistance(INDArray x) {
        super(x);
    }

    public EuclideanDistance(INDArray x, INDArray y) {
        super(x, y);
    }

    @Override
    public double update(double accum, double x){
        return accum + (x * x);
    }

    @Override
    public double update(double accum, double x, double y) {
        double d = (x-y);
        return accum + d*d;
    }

    @Override
    public float update(float accum, float x){
        return accum + (x * x);
    }

    @Override
    public float update(float accum, float x, float y) {
        float f = (x-y);
        return accum + f * f;
    }

    @Override
    public IComplexNumber update( IComplexNumber accum, double x){
        return accum;
    }

    @Override
    public IComplexNumber update( IComplexNumber accum, double x, double y) {
        double d = (x-y);
        return accum.add(d * d);
    }

    @Override
    public IComplexNumber update(IComplexNumber accum, IComplexNumber x){
        return accum;
    }

    @Override
    public IComplexNumber update( IComplexNumber accum, IComplexNumber x, IComplexNumber y) {
        IComplexNumber c = x.sub(y);
        return accum.add(c.mul(c));
    }

    @Override
    public IComplexNumber update(IComplexNumber accum, IComplexNumber x, double y) {
        IComplexNumber c = x.sub(y);
        return accum.add(c.mul(c));
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
    public int opNum() {
        return 1;
    }

    @Override
    public String name() {
        return "euclidean";
    }


    @Override
    public IComplexNumber op(IComplexNumber origin, double other) {
        numProcessed++;
        return origin.sub(other);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, float other) {
        numProcessed++;
        return origin.sub(other);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, IComplexNumber other) {
        numProcessed++;
        return origin.sub(other);
    }

    @Override
    public float op(float origin, float other) {
        numProcessed++;
        return origin - other;
    }

    @Override
    public double op(double origin, double other) {
        numProcessed++;
        return origin - other;
    }

    @Override
    public double op(double origin) {
        numProcessed++;
        return origin;
    }

    @Override
    public float op(float origin) {
        numProcessed++;
        return origin;
    }

    @Override
    public IComplexNumber op(IComplexNumber origin) {
        numProcessed++;
        return origin;
    }

    @Override
    public Op opForDimension(int index, int dimension) {
        INDArray xForDimension = x.vectorAlongDimension(index, dimension);
        EuclideanDistance ret;
        if (y() != null)
            ret = new EuclideanDistance(xForDimension, y.vectorAlongDimension(index, dimension), xForDimension.length());
        else
            ret = new EuclideanDistance(x.vectorAlongDimension(index, dimension));
        ret.setApplyFinalTransform(applyFinalTransform());
        return ret;
    }

    @Override
    public Op opForDimension(int index, int... dimension) {
        INDArray xForDimension = x.tensorAlongDimension(index, dimension);
        EuclideanDistance ret;
        if (y() != null)
            ret = new EuclideanDistance(xForDimension, y.tensorAlongDimension(index, dimension), xForDimension.length());
        else
            ret = new EuclideanDistance(x.tensorAlongDimension(index, dimension));
        ret.setApplyFinalTransform(applyFinalTransform());
        return ret;
    }

    @Override
    public double getAndSetFinalResult(double accum) {
        if(applyFinalTransform()) {
            double d = FastMath.sqrt(accum);
            this.finalResult = d;
            return d;
        }
        else {
            this.finalResult = accum;
            return accum;
        }

    }

    @Override
    public float getAndSetFinalResult(float accum) {
        if(applyFinalTransform) {
            float f = (float) FastMath.sqrt(accum);
            this.finalResult = f;
            return f;
        }
        else {
            this.finalResult = accum;
            return accum;
        }

    }

    @Override
    public IComplexNumber getAndSetFinalResult(IComplexNumber accum){
        this.finalResultComplex = accum.sqrt();
        return finalResultComplex;
    }

    @Override
    public double calculateFinalResult(double accum, int n){
        return FastMath.sqrt(accum);
    }

    @Override
    public float calculateFinalResult(float accum, int n){
        return (float)FastMath.sqrt(accum);
    }
}
