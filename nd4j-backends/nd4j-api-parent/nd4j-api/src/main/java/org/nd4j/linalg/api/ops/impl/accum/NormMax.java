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
import org.nd4j.linalg.ops.transforms.Transforms;

/**
 * The max absolute value
 *
 * @author Adam Gibson
 */
public class NormMax extends BaseAccumulation {
    public NormMax() {
    }

    public NormMax(INDArray x, INDArray y, INDArray z, int n) {
        super(x, y, z, n);
    }

    public NormMax(INDArray x, INDArray y, int n) {
        super(x, y, n);
    }

    public NormMax(INDArray x) {
        super(x);
    }

    public NormMax(INDArray x, INDArray y) {
        super(x, y);
    }

    @Override
    public INDArray noOp() {
        return Transforms.abs(x());
    }

    @Override
    public double update(double accum, double x) {
        numProcessed++;
        return FastMath.max(FastMath.abs(x),accum);
    }

    @Override
    public double update(double accum, double x, double y) {
        return update(accum,x);
    }


    @Override
    public float update(float accum, float x){
        return (x >= 0 ? (x > accum ? x : accum) : (-x > accum ? -x : accum));
    }

    @Override
    public float update(float accum, float x, float y) {
        return (x>=0 ? (x>accum?x:accum) : (-x > accum ? -x : accum));
    }

    @Override
    public IComplexNumber update( IComplexNumber accum, double x){
        return (accum.absoluteValue().doubleValue() >= FastMath.abs(x) ? accum : Nd4j.createComplexNumber(FastMath.abs(x),0));
    }

    @Override
    public IComplexNumber update( IComplexNumber accum, double x, double y){
        return (accum.absoluteValue().doubleValue() >= FastMath.abs(x) ? accum : Nd4j.createComplexNumber(FastMath.abs(x),0));
    }

    @Override
    public IComplexNumber update( IComplexNumber accum, IComplexNumber x){
        return (accum.absoluteValue().doubleValue() >= x.absoluteValue().doubleValue() ? accum : Nd4j.createComplexNumber(x.absoluteValue(),0));
    }

    @Override
    public IComplexNumber update( IComplexNumber accum, IComplexNumber x, IComplexNumber y){
        return (accum.absoluteValue().doubleValue() >= x.absoluteValue().doubleValue() ? accum : Nd4j.createComplexNumber(x.absoluteValue(),0));
    }

    @Override
    public IComplexNumber update(IComplexNumber accum, IComplexNumber x, double y) {
        return (accum.absoluteValue().doubleValue() >= x.absoluteValue().doubleValue() ? accum : Nd4j.createComplexNumber(x.absoluteValue(),0));
    }

    @Override
    public int opNum() {
        return 7;
    }

    @Override
    public String name() {
        return "normmax";
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
    public IComplexNumber op(IComplexNumber origin) {
        throw new UnsupportedOperationException();

    }

    @Override
    public Op opForDimension(int index, int dimension) {
        INDArray xAlongDimension = x.vectorAlongDimension(index, dimension);

        if (y() != null)
            return new NormMax(xAlongDimension, y.vectorAlongDimension(index, dimension), xAlongDimension.length());
        else
            return new NormMax(x.vectorAlongDimension(index, dimension));
    }

    @Override
    public Op opForDimension(int index, int... dimension) {
        INDArray xAlongDimension = x.tensorAlongDimension(index, dimension);

        if (y() != null)
            return new NormMax(xAlongDimension, y.tensorAlongDimension(index, dimension), xAlongDimension.length());
        else
            return new NormMax(x.tensorAlongDimension(index, dimension));
    }
}
