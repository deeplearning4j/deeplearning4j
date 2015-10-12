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
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ComplexUtil;

/**
 * Sum of squared values (real)
 * Sum of squared complex modulus (complex)
 *
 * @author Adam Gibson
 */
public class Norm2 extends BaseAccumulation {

    public Norm2() {
    }

    public Norm2(INDArray x, INDArray y, INDArray z, int n) {
        super(x, y, z, n);
    }

    public Norm2(INDArray x, INDArray y, int n) {
        super(x, y, n);
    }

    public Norm2(INDArray x) {
        super(x);
    }

    public Norm2(INDArray x, INDArray y) {
        super(x, y);
    }

    @Override
    public double update(double accum, double x){
        return accum + x*x;
    }

    @Override
    public double update(double accum, double x, double y){
        return accum + x*x;
    }

    @Override
    public float update(float accum, float x){
        return accum + x*x;
    }

    @Override
    public float update(float accum, float x, float y){
        return accum + x*x;
    }

    @Override
    public IComplexNumber update( IComplexNumber accum, double x){
        return accum.add(x*x);
    }

    @Override
    public IComplexNumber update( IComplexNumber accum, double x, double y){
        return accum.add(x*x);
    }

    @Override
    public IComplexNumber update( IComplexNumber accum, IComplexNumber x){
        return accum.add(x.mul(x));
    }

    @Override
    public IComplexNumber update( IComplexNumber accum, IComplexNumber x, IComplexNumber y){
        return accum.add(x.mul(x));
    }

    @Override
    public IComplexNumber update(IComplexNumber accum, IComplexNumber x, double y) {
        return accum.add(x.mul(x));
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
    public String name() {
        return "norm2";
    }

    @Override
    public Op opForDimension(int index, int dimension) {
        INDArray xAlongDimension = x.vectorAlongDimension(index, dimension);

        if (y() != null)
            return new Norm2(xAlongDimension, y.vectorAlongDimension(index, dimension), xAlongDimension.length());
        else
            return new Norm2(x.vectorAlongDimension(index, dimension));

    }

    @Override
    public Op opForDimension(int index, int... dimension) {
        INDArray xAlongDimension = x.tensorAlongDimension(index, dimension);

        if (y() != null)
            return new Norm2(xAlongDimension, y.tensorAlongDimension(index, dimension), xAlongDimension.length());
        else
            return new Norm2(x.tensorAlongDimension(index, dimension));
    }

    @Override
    public double getAndSetFinalResult(double accum){
        double d = FastMath.sqrt(accum);
        this.finalResult = d;
        return d;
    }

    @Override
    public float getAndSetFinalResult(float accum){
        float f = (float)FastMath.sqrt(accum);
        this.finalResult = f;
        return f;
    }

    @Override
    public IComplexNumber getAndSetFinalResult(IComplexNumber accum){
        this.finalResultComplex = accum.sqrt();
        return finalResultComplex;
    }
}
