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

import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseAccumulation;
import org.nd4j.linalg.api.ops.Op;

/**
 * Calculate the max over a vector
 *
 * @author Adam Gibson
 */
public class Max extends BaseAccumulation {
    public Max(INDArray x, INDArray y, int n) {
        super(x, y, n);
    }

    public Max(INDArray x) {
        super(x);
    }

    public Max(INDArray x, INDArray y) {
        super(x, y);
    }

    @Override
    public void update(Number result) {
        if (result.doubleValue() > currentResult().doubleValue())
            this.currentResult = result;
        numProcessed++;
    }

    @Override
    public void update(IComplexNumber result) {
        if (result.absoluteValue().doubleValue() > currentResultComplex().absoluteValue().doubleValue())
            this.currentComplexResult = result;
        numProcessed++;
    }


    @Override
    public String name() {
        return "max";
    }

    @Override
    public void init(INDArray x, INDArray y, INDArray z, int n) {
        super.init(x, y, z, n);
        if (x instanceof IComplexNDArray) {
            IComplexNDArray complexX = (IComplexNDArray) x;
            currentComplexResult = complexX.getComplex(0);
        } else {
            currentResult = x.getDouble(0);
            initial = x.getDouble(0);
        }


    }

    @Override
    public Op opForDimension(int index, int dimension) {
        INDArray xAlongDimension = x.vectorAlongDimension(index, dimension);

        if (y() != null)
            return new Max(xAlongDimension, y.vectorAlongDimension(index, dimension), xAlongDimension.length());
        else
            return new Max(x.vectorAlongDimension(index, dimension));

    }
}
