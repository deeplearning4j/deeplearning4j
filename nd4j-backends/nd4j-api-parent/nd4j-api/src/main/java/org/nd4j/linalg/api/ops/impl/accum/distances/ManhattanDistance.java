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
import org.nd4j.linalg.util.ComplexUtil;

/**
 * Manhattan distance
 *
 * @author Adam Gibson
 */
public class ManhattanDistance extends BaseAccumulation {

    public ManhattanDistance() {
    }

    public ManhattanDistance(INDArray x, INDArray y, INDArray z, int n) {
        super(x, y, z, n);
    }

    public ManhattanDistance(INDArray x, INDArray y, int n) {
        super(x, y, n);
    }

    public ManhattanDistance(INDArray x) {
        super(x);
    }

    public ManhattanDistance(INDArray x, INDArray y) {
        super(x, y);
    }

    @Override
    public double update(double accum, double x) {
        return accum;
    }

    @Override
    public double update(double accum, double x, double y){
        return accum + FastMath.abs(x - y);
    }

    @Override
    public float update(float accum, float x){
        return accum + x;
    }

    @Override
    public float update(float accum, float x, float y){
        return accum + FastMath.abs(x-y);
    }

    @Override
    public IComplexNumber update( IComplexNumber accum, double x) {
        return accum.add(x);
    }

    @Override
    public IComplexNumber update( IComplexNumber accum, double x, double y){
        return accum.add(FastMath.abs(x - y));
    }

    @Override
    public IComplexNumber update( IComplexNumber accum, IComplexNumber x){
        return accum;
    }

    @Override
    public IComplexNumber update( IComplexNumber accum, IComplexNumber x, IComplexNumber y){
        return accum.add(x.sub(y).absoluteValue());
    }

    @Override
    public IComplexNumber update(IComplexNumber accum, IComplexNumber x, double y) {
        return accum.add(x.sub(y).absoluteValue());
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
        return 0;
    }

    @Override
    public String name() {
        return "manhattan";
    }


    @Override
    public IComplexNumber op(IComplexNumber origin, double other) {
        numProcessed++;
        return origin.sub(other);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, float other) {
        numProcessed++;
        return ComplexUtil.abs(origin.sub(other));
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, IComplexNumber other) {
        numProcessed++;
        return ComplexUtil.abs(origin.sub(other));
    }

    @Override
    public float op(float origin, float other) {
        return FastMath.abs(origin - other);
    }

    @Override
    public double op(double origin, double other) {
        numProcessed++;
        return FastMath.abs(origin - other);
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
        ManhattanDistance ret;
        if (y() != null)
            ret = new ManhattanDistance(x.vectorAlongDimension(index, dimension), y.vectorAlongDimension(index, dimension), x.length());
        else
            ret = new ManhattanDistance(x.vectorAlongDimension(index, dimension));
        ret.setApplyFinalTransform(applyFinalTransform());
        return ret;
    }

    @Override
    public Op opForDimension(int index, int... dimension) {
        ManhattanDistance ret;
        if (y() != null)
            ret = new ManhattanDistance(x.tensorAlongDimension(index, dimension), y.tensorAlongDimension(index, dimension), x.length());
        else
            ret = new ManhattanDistance(x.tensorAlongDimension(index, dimension));
        ret.setApplyFinalTransform(applyFinalTransform());
        return ret;
    }
}
