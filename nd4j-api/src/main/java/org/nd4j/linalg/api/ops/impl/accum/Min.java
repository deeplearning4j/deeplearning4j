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
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseAccumulation;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Calculate the min over a vector
 *
 * @author Adam Gibson
 */
public class Min extends BaseAccumulation {

    public Min() {
    }

    public Min(INDArray x, INDArray y, INDArray z, int n) {
        super(x, y, z, n);
    }

    public Min(INDArray x, INDArray y, int n) {
        super(x, y, n);
    }

    public Min(INDArray x) {
        super(x);
    }

    public Min(INDArray x, INDArray y) {
        super(x, y);
    }


    @Override
    public int opNum() {
        return 4;
    }

    @Override
    public String name() {
        return "min";
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
    public double update(double accum, double x) {
        return FastMath.min(accum,x);
    }

    @Override
    public double update(double accum, double x, double y){
        return FastMath.min(accum,x);
    }

    @Override
    public float update(float accum, float x){
        return FastMath.min(accum,x);
    }

    @Override
    public float update(float accum, float x, float y){
        return FastMath.min(accum,x);
    }

    @Override
    public IComplexNumber update( IComplexNumber accum, double x){
        return (accum.absoluteValue().doubleValue() < x ? accum : Nd4j.createComplexNumber(x, 0));
    }

    @Override
    public IComplexNumber update( IComplexNumber accum, double x, double y){
        return (accum.absoluteValue().doubleValue() < x ? accum : Nd4j.createComplexNumber(x,0));
    }

    @Override
    public IComplexNumber update( IComplexNumber accum, IComplexNumber x){
        return (accum.absoluteValue().doubleValue() < x.absoluteValue().doubleValue() ? accum : x);
    }

    @Override
    public IComplexNumber update( IComplexNumber accum, IComplexNumber x, IComplexNumber y){
        return (accum.absoluteValue().doubleValue() < x.absoluteValue().doubleValue() ? accum : x);
    }

    @Override
    public IComplexNumber update(IComplexNumber accum, IComplexNumber x, double y) {
        return (accum.absoluteValue().doubleValue() < x.absoluteValue().doubleValue() ? accum : x);
    }

    @Override
    public double zeroDouble() {
        return Double.MAX_VALUE;
    }

    @Override
    public float zeroFloat(){
        return Float.MAX_VALUE;
    }

    @Override
    public Op opForDimension(int index, int dimension) {
        INDArray xAlongDimension = x.vectorAlongDimension(index, dimension);

        if (y() != null)
            return new Min(xAlongDimension, y.vectorAlongDimension(index, dimension), xAlongDimension.length());
        else
            return new Min(x.vectorAlongDimension(index, dimension));

    }

    @Override
    public Op opForDimension(int index, int... dimension) {
        INDArray xAlongDimension = x.tensorAlongDimension(index, dimension);

        if (y() != null)
            return new Min(xAlongDimension, y.tensorAlongDimension(index, dimension), xAlongDimension.length());
        else
            return new Min(x.tensorAlongDimension(index, dimension));
    }


}
