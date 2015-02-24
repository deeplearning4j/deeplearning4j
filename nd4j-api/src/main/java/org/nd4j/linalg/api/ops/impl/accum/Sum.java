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
 * Sum the components
 * @author Adam Gibson
 */
public class Sum extends BaseAccumulation {
    public Sum(INDArray x, INDArray y, int n) {
        super(x, y, n);
    }

    public Sum(INDArray x) {
        super(x);
    }

    public Sum(INDArray x, INDArray y) {
        super(x, y);
    }

    @Override
    public void update(Number result) {
        currentResult = currentResult.doubleValue() + result.doubleValue();
        numProcessed++;
    }

    @Override
    public void update(IComplexNumber result) {
        currentComplexResult.addi(result);
        numProcessed++;
    }

    @Override
    public Number zero() {
        return 0.0;
    }

    @Override
    public IComplexNumber zeroComplex() {
        return Nd4j.createComplexNumber(0.0,0.0);
    }

    @Override
    public String name() {
        return "sum";
    }
    @Override
    public Op opForDimension(int index,int dimension) {
        INDArray xAlongDimension = x.vectorAlongDimension(index,dimension);

        if(y() != null)
            return new Sum(xAlongDimension,y.vectorAlongDimension(index,dimension),xAlongDimension.length());
        else
            return new Sum(x.vectorAlongDimension(index,dimension));

    }
}
