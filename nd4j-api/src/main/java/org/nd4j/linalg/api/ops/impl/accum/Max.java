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
import org.nd4j.linalg.factory.Nd4j;

/**
 *  Calculate the max over a vector
 *  @author Adam Gibson
 */
public class Max extends BaseAccumulation {
    public Max(INDArray x, INDArray y, int n) {
        super(x, y, n);
    }

    public Max(INDArray x) {
        super(x);
    }

    @Override
    public void update(Number result) {
        if(result.doubleValue() > currentResult().doubleValue())
            this.currentResult = result;
        numProcessed++;
    }

    @Override
    public void update(IComplexNumber result) {
        if(result.absoluteValue().doubleValue() > currentResultComplex().absoluteValue().doubleValue())
            this.currentComplexResult = result;
        numProcessed++;
    }

    @Override
    public Number zero() {
        return Double.MIN_VALUE;
    }

    @Override
    public IComplexNumber zeroComplex() {
        return Nd4j.createComplexNumber(zero(), zero());
    }

    @Override
    public String name() {
        return "max";
    }


}
