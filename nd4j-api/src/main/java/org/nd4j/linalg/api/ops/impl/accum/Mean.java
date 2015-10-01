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

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseAccumulation;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.parallel.bufferops.AccumulationDataBufferTask;
import org.nd4j.linalg.api.parallel.bufferops.impl.accum.MeanOpDataBufferTask;

/**
 * Calculate the mean of the vector
 *
 * @author Adam Gibson
 */
public class Mean extends Sum {

    public Mean() {
    }

    public Mean(INDArray x, INDArray y, INDArray z, int n) {
        super(x, y, z, n);
    }

    public Mean(INDArray x, INDArray y, int n) {
        super(x, y, n);
    }

    public Mean(INDArray x) {
        super(x);
    }

    public Mean(INDArray x, INDArray y) {
        super(x, y);
    }

    @Override
    public String name() {
        return "mean";
    }

    @Override
    public Op opForDimension(int index, int dimension) {
        INDArray xAlongDimension = x.vectorAlongDimension(index, dimension);

        if (y() != null)
            return new Mean(xAlongDimension, y.vectorAlongDimension(index, dimension), xAlongDimension.length());
        else
            return new Mean(x.vectorAlongDimension(index, dimension));
    }

    @Override
    public Op opForDimension(int index, int... dimension) {
        INDArray xAlongDimension = x.tensorAlongDimension(index, dimension);


        if (y() != null)
            return new Mean(xAlongDimension, y.tensorAlongDimension(index, dimension), xAlongDimension.length());
        else
            return new Mean(x.tensorAlongDimension(index, dimension));
    }

    @Override
    public double getAndSetFinalResult(double accum){
        double d = accum / n();
        this.finalResult = d;
        return d;
    }

    @Override
    public float getAndSetFinalResult(float accum){
        float f = accum / n();
        this.finalResult = f;
        return f;
    }

    @Override
    public IComplexNumber getAndSetFinalResult(IComplexNumber accum){
        finalResultComplex = accum.div(n());
        return finalResultComplex;
    }

    @Override
    public AccumulationDataBufferTask getAccumulationOpDataBufferTask(int threshold, int n, DataBuffer x, DataBuffer y,
                                                                      int offsetX, int offsetY, int incrX, int incrY, boolean outerTask){
        return new MeanOpDataBufferTask(this,threshold,n,x,y,offsetX,offsetY,incrX,incrY,outerTask);
    }

    @Override
    public AccumulationDataBufferTask getAccumulationOpDataBufferTask(int tensorNum, int tensorDim, int parallelThreshold,
                                                                      INDArray x, INDArray y, boolean outerTask){
        return new MeanOpDataBufferTask(this,tensorNum,tensorDim,parallelThreshold,x,y,outerTask);
    }
}
