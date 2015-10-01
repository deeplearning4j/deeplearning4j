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

import org.apache.commons.math3.util.FastMath;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseIndexAccumulation;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.parallel.bufferops.IndexAccumulationDataBufferTask;
import org.nd4j.linalg.api.parallel.bufferops.impl.indexaccum.IAMaxOpDataBufferTask;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Calculate the index of the max absolute value over a vector
 *
 * @author Adam Gibson
 */
public class IAMax extends BaseIndexAccumulation {
    public IAMax() {
    }

    public IAMax(INDArray x, INDArray y, int n) {
        super(x, y, n);
    }

    public IAMax(INDArray x) {
        super(x);
    }

    public IAMax(INDArray x, INDArray y) {
        super(x, y);
    }


    public int update(double accum, int accumIdx, double x, int xIdx){
        return (FastMath.abs(accum)>=FastMath.abs(x) ? accumIdx : xIdx);
    }

    public int update(float accum, int accumIdx, float x, int xIdx){
        return (FastMath.abs(accum)>=FastMath.abs(x) ? accumIdx : xIdx);
    }

    public int update(double accum, int accumIdx, double x, double y, int idx){
        return (FastMath.abs(accum)>=FastMath.abs(x) ? accumIdx : idx);
    }

    public int update(float accum, int accumIdx, float x, float y, int idx){
        return (FastMath.abs(accum)>=FastMath.abs(x) ? accumIdx : idx);
    }

    public int update(IComplexNumber accum, int accumIdx, IComplexNumber x, int xIdx){
        return (accum.absoluteValue().doubleValue()>=x.absoluteValue().doubleValue() ? accumIdx : xIdx);
    }

    @Override
    public int update(IComplexNumber accum, int accumIdx, double x, int idx) {
        return (accum.absoluteValue().doubleValue()>=FastMath.abs(x) ? accumIdx : idx);
    }

    @Override
    public int update(IComplexNumber accum, int accumIdx, double x, double y, int idx) {
        return (accum.absoluteValue().doubleValue()>=FastMath.abs(x) ? accumIdx : idx);
    }

    public int update(IComplexNumber accum, int accumIdx, IComplexNumber x, IComplexNumber y, int idx){
        return (accum.absoluteValue().doubleValue()>=x.absoluteValue().doubleValue() ? accumIdx : idx);
    }


    @Override
    public String name() {
        return "iamax";
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, double other) {
        return (origin.absoluteValue().doubleValue()>=FastMath.abs(other) ? origin : Nd4j.createComplexNumber(other,0));
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, float other) {
        return (origin.absoluteValue().doubleValue()>=FastMath.abs(other) ? origin : Nd4j.createComplexNumber(other,0));
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, IComplexNumber other) {
        return (origin.absoluteValue().doubleValue()>=other.absoluteValue().doubleValue() ? origin : other);
    }

    @Override
    public float op(float origin, float other) {
        return (FastMath.abs(origin)>=FastMath.abs(other) ? origin : other);
    }

    @Override
    public double op(double origin, double other) {
        return (FastMath.abs(origin)>=FastMath.abs(other) ? origin : other);
    }

    @Override
    public double op(double origin) {
        return 0;
    }

    @Override
    public float op(float origin) {
        return 0;
    }

    @Override
    public IComplexNumber op(IComplexNumber origin) {
        return null;
    }

    @Override
    public Op opForDimension(int index, int dimension) {
        INDArray xAlongDimension = x.vectorAlongDimension(index, dimension);

        if (y() != null)
            return new IAMax(xAlongDimension, y.vectorAlongDimension(index, dimension), xAlongDimension.length());
        else
            return new IAMax(x.vectorAlongDimension(index, dimension));

    }

    @Override
    public Op opForDimension(int index, int... dimension) {
        INDArray xAlongDimension = x.tensorAlongDimension(index, dimension);

        if (y() != null)
            return new IAMax(xAlongDimension, y.tensorAlongDimension(index, dimension), xAlongDimension.length());
        else
            return new IAMax(x.tensorAlongDimension(index, dimension));
    }

    @Override
    public IndexAccumulationDataBufferTask getIndexAccumulationOpDataBufferTask(int threshold, int n, DataBuffer x, DataBuffer y,
                                                                                int offsetX, int offsetY, int incrX, int incrY,
                                                                                int elementOffset, boolean outerTask){
        return new IAMaxOpDataBufferTask(this,threshold,n,x,y,offsetX,offsetY,incrX,incrY,elementOffset,outerTask);
    }

    @Override
    public IndexAccumulationDataBufferTask getIndexAccumulationOpDataBufferTask(int tensorNum, int tensorDim, int parallelThreshold,
                                                                                INDArray x, INDArray y, boolean outerTask){
        return new IAMaxOpDataBufferTask(this,tensorNum,tensorDim,parallelThreshold,x,y,outerTask);
    }
}
