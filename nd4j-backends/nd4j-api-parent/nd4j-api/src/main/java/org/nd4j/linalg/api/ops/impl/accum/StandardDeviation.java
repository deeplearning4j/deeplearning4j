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
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ArrayUtil;

/**
 * Standard deviation (sqrt of variance)
 *
 * @author Adam Gibson
 */
public class StandardDeviation extends Variance {
    public StandardDeviation(INDArray x, boolean biasCorrected) {
        super(x, biasCorrected);
    }

    public StandardDeviation() {
    }

    public StandardDeviation(INDArray x, INDArray y, long n) {
        super(x, y, n);
    }

    public StandardDeviation(INDArray x) {
        super(x);
    }

    public StandardDeviation(INDArray x, INDArray y) {
        super(x, y);
    }

    @Override
    public int opNum() {
        return 1;
    }

    @Override
    public String name() {
        return "std";
    }

    @Override
    public Op opForDimension(int index, int dimension) {
        INDArray xAlongDimension = x.vectorAlongDimension(index, dimension);

        if (y() != null)
            return new StandardDeviation(xAlongDimension, y.vectorAlongDimension(index, dimension), xAlongDimension.length());
        else
            return new StandardDeviation(xAlongDimension);

    }

    @Override
    public Variance opForDimension(int index, int... dimension) {
        INDArray xAlongDimension = x.tensorAlongDimension(index, dimension);

        if (y() != null)
            return new StandardDeviation(xAlongDimension, y.tensorAlongDimension(index, dimension), xAlongDimension.length());
        else
            return new StandardDeviation(xAlongDimension);
    }

    @Override
    public void exec(){
        super.exec();   //variance = sqrt(stdev) -> sqrt is done in getAndSetFinalResult(...)
    }

    @Override
    public void exec(int... dimension) {
        if(dimension.length == 1 && dimension[0] == Integer.MAX_VALUE) {
            exec();
            this.z = Nd4j.scalar(this.finalResult);
            return;
        }

        int[] retShape = ArrayUtil.removeIndex(x.shape(), dimension);
        long nOps = x.tensorssAlongDimension(dimension);
        z = Nd4j.create(retShape);
        for( int i = 0; i < nOps; i++) {
            double d = Nd4j.getExecutioner().execAndReturn((StandardDeviation) opForDimension(i,dimension)).getFinalResult().doubleValue();
            z.putScalar(i,d);
        }
    }

    @Override
    public double getAndSetFinalResult(double accum) {
        //stdev is sqrt of variance:
        double d = FastMath.sqrt(super.getAndSetFinalResult(accum));
        this.finalResult = d;
        return d;
    }

    @Override
    public float getAndSetFinalResult(float accum){
        float f = (float)FastMath.sqrt(super.getAndSetFinalResult(accum));
        this.finalResult = f;
        return f;
    }

    @Override
    public IComplexNumber getAndSetFinalResult(IComplexNumber accum){
        finalResultComplex = super.getAndSetFinalResult(accum).sqrt();
        return finalResultComplex;
    }

    @Override
    public double calculateFinalResult(double accum, long n) {
        return FastMath.sqrt(super.calculateFinalResult(accum,n));
    }

    @Override
    public float calculateFinalResult(float accum, long n) {
        return (float) FastMath.sqrt(super.calculateFinalResult(accum,n));
    }
}
