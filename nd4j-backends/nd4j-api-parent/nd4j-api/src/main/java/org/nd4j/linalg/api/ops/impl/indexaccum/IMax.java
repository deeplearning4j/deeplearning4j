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

package org.nd4j.linalg.api.ops.impl.indexaccum;

import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseIndexAccumulation;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Calculate the index
 * of max value over a vector
 * @author Alex Black
 */
public class IMax extends BaseIndexAccumulation {
    public IMax() {
    }

    public IMax(INDArray x, INDArray y, long n) {
        super(x, y, n);
    }

    public IMax(INDArray x) {
        super(x);
    }

    public IMax(INDArray x, INDArray y) {
        super(x, y);
    }

    @Override
    public int update(double accum, int accumIdx, double x, int xIdx){
        return (accum >= x ? accumIdx : xIdx);
    }

    @Override
    public int update(float accum, int accumIdx, float x, int xIdx){
        return (accum >= x ? accumIdx : xIdx);
    }

    @Override
    public int update(double accum, int accumIdx, double x, double y, int idx){
        return (accum >=x ? accumIdx : idx);
    }

    @Override
    public int update(float accum, int accumIdx, float x, float y, int idx){
        return (accum >= x ? accumIdx : idx);
    }

    @Override
    public int update(IComplexNumber accum, int accumIdx, IComplexNumber x, int xIdx){
        return (accum.absoluteValue().doubleValue()>=x.absoluteValue().doubleValue() ? accumIdx : xIdx);
    }

    @Override
    public int update(IComplexNumber accum, int accumIdx, double x, int idx) {
        return (accum.absoluteValue().doubleValue() >= x ? accumIdx : idx);
    }

    @Override
    public int update(IComplexNumber accum, int accumIdx, double x, double y, int idx) {
        return (accum.absoluteValue().doubleValue() >= x ? accumIdx : idx);
    }

    @Override
    public int update(IComplexNumber accum, int accumIdx, IComplexNumber x, IComplexNumber y, int idx){
        return (accum.absoluteValue().doubleValue() >= x.absoluteValue().doubleValue() ? accumIdx : idx);
    }


    @Override
    public int opNum() {
        return 0;
    }

    @Override
    public String name() {
        return "imax";
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
    public float zeroFloat(){
        return -Float.MAX_VALUE;
    }

    @Override
    public double zeroDouble(){
        return -Double.MAX_VALUE;
    }

    @Override
    public IComplexNumber zeroComplex(){
        return Nd4j.createComplexNumber(-Double.MAX_VALUE,0);
    }

    @Override
    public Op opForDimension(int index, int dimension) {
        INDArray xAlongDimension = x.vectorAlongDimension(index, dimension);

        if (y() != null)
            return new IMax(xAlongDimension, y.vectorAlongDimension(index, dimension), xAlongDimension.length());
        else
            return new IMax(x.vectorAlongDimension(index, dimension));

    }

    @Override
    public Op opForDimension(int index, int... dimension) {
        INDArray xAlongDimension = x.tensorAlongDimension(index, dimension);

        if (y() != null)
            return new IMax(xAlongDimension, y.tensorAlongDimension(index, dimension), xAlongDimension.length());
        else
            return new IMax(x.tensorAlongDimension(index, dimension));
    }
}
