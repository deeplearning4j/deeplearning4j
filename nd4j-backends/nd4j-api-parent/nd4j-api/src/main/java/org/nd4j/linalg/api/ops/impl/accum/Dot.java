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

/**
 * Dot product
 * @author Adam Gibson
 */
public class Dot extends BaseAccumulation {

    public Dot() {
    }

    public Dot(INDArray x, INDArray y, INDArray z, int n) {
        super(x, y, z, n);
    }

    public Dot(INDArray x, INDArray y, int n) {
        super(x, y, n);
    }

    public Dot(INDArray x) {
        super(x);
    }

    public Dot(INDArray x, INDArray y) {
        super(x, y);
    }

    @Override
    public int opNum() {
        return 3;
    }

    @Override
    public String name() {
        return "dot";
    }

    @Override
    public Op opForDimension(int index, int dimension) {
        INDArray xAlongDimension = x.vectorAlongDimension(index, dimension);
        if (y() != null)
            return new Dot(xAlongDimension, y.vectorAlongDimension(index, dimension), xAlongDimension.length());
        else
            return new Dot(x.vectorAlongDimension(index, dimension));

    }

    @Override
    public Op opForDimension(int index, int... dimension) {
        INDArray xAlongDimension = x.tensorAlongDimension(index, dimension);
        if (y() != null)
            return new Dot(xAlongDimension, y.tensorAlongDimension(index, dimension), xAlongDimension.length());
        else
            return new Dot(x.tensorAlongDimension(index, dimension));
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
        return origin  * other;
    }

    @Override
    public double op(double origin, double other) {
        numProcessed++;
        return origin  * other;
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
    public double update(double accum, double x){
        return accum + x;
    }

    @Override
    public double update(double accum, double x, double y){
        return accum + x*y;
    }

    @Override
    public float update(float accum, float x){
        return accum + x;
    }

    @Override
    public float update(float accum, float x, float y){
        return accum + x*y;
    }

    @Override
    public IComplexNumber update( IComplexNumber accum, double x){
        return accum.add(x);
    }

    @Override
    public IComplexNumber update( IComplexNumber accum, double x, double y){
        return accum.add(x*y);
    }

    @Override
    public IComplexNumber update( IComplexNumber accum, IComplexNumber x){
        return accum.add(x);
    }

    @Override
    public IComplexNumber update( IComplexNumber accum, IComplexNumber x, IComplexNumber y){
        return accum.add(x.mul(y));
    }

    @Override
    public IComplexNumber update(IComplexNumber accum, IComplexNumber x, double y) {
        return accum.add(x.mul(y));
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
}
