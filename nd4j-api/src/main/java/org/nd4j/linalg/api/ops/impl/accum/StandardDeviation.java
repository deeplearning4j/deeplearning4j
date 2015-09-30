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

/**
 * Standard deviation (sqrt of variance)
 *
 * @author Adam Gibson
 */
public class StandardDeviation extends Variance {

    public StandardDeviation() {
    }

    public StandardDeviation(INDArray x, INDArray y, int n) {
        super(x, y, n);
    }

    public StandardDeviation(INDArray x) {
        super(x);
    }

    public StandardDeviation(INDArray x, INDArray y) {
        super(x, y);
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
    public Op opForDimension(int index, int... dimension) {
        INDArray xAlongDimension = x.tensorAlongDimension(index, dimension);

        if (y() != null)
            return new StandardDeviation(xAlongDimension, y.tensorAlongDimension(index, dimension), xAlongDimension.length());
        else
            return new StandardDeviation(xAlongDimension);
    }

    @Override
    public double getAndSetFinalResult(double accum){
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
}
