/*
 * Copyright 2015 Skymind,Inc.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

package org.nd4j.linalg.api.ops.impl.accum;

import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseAccumulation;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Calculate a bias
 *
 * @author Adam Gibson
 */
public class Bias extends BaseAccumulation {

    private double mean;

    public Bias(INDArray x, INDArray y, INDArray z, int n) {
        super(x, y, z, n);
    }

    public Bias(INDArray x, INDArray y, int n) {
        this(x, y, x, n);
    }

    public Bias(INDArray x) {
        super(x);
    }

    public Bias(INDArray x, INDArray y) {
        super(x, y);
    }

    @Override
    public String name() {
        return "bias";
    }

    @Override
    public Op opForDimension(int index, int dimension) {
        INDArray xAlongDimension = x.vectorAlongDimension(index, dimension);


        if (y() != null)
            return new Bias(xAlongDimension, y.vectorAlongDimension(index, dimension), xAlongDimension.length());
        else
            return new Bias(x.vectorAlongDimension(index, dimension));
    }

    @Override
    public void update(Number result) {
        double dev = result.doubleValue() - mean;
        currentResult = currentResult().doubleValue() + dev;
        numProcessed++;

    }

    @Override
    public void update(IComplexNumber result) {
        IComplexNumber dev = result.sub(mean);
        currentComplexResult.addi(dev);
        numProcessed++;
    }

    @Override
    public Number zero() {
        return 0.0;
    }

    @Override
    public IComplexNumber zeroComplex() {
        return Nd4j.createComplexNumber(0.0, 0.0);
    }

    @Override
    public void init(INDArray x, INDArray y, INDArray z, int n) {
        super.init(x, y, z, n);
        this.mean = Nd4j.getExecutioner().execAndReturn(new Mean(x)).currentResult().doubleValue();
        this.extraArgs = new Object[]{zero(), mean};

    }


}
