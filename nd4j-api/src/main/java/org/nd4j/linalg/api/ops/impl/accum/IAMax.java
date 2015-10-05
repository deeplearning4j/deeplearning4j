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

import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseAccumulation;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Calculate the max over a vector
 *
 * @author Adam Gibson
 */
public class IAMax extends BaseAccumulation {
    private int currIndexOfMax = 0;
    private double currentResult = Double.MIN_VALUE;

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

    @Override
    public void update(Number result) {
        if(result.doubleValue() > currentResult) {
            currentResult = result.doubleValue();
            currIndexOfMax = numProcessed;
        }

        //done; accumulate final result
        if(numProcessed() == n()) {
            currentResult = currIndexOfMax;
        }
    }

    @Override
    public void update(IComplexNumber result) {
        IComplexNDArray complexX = (IComplexNDArray) x;
        currentComplexResult = Nd4j.createComplexNumber(Nd4j.getBlasWrapper().iamax(x),0.0);
    }


    @Override
    public String name() {
        return "iamax";
    }

    @Override
    public void init(INDArray x, INDArray y, INDArray z, int n) {
        super.init(x, y, z, n);
        // exec();
    }


    @Override
    public void exec(int... dimensions) {

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
    public void exec() {
        int idx = 0;
        double max = Double.MIN_VALUE;
        for(int i = 0; i < x.length(); i++) {
            double val = x.getDouble(i);
            if(val > max) {
                max = val;
                idx = i;
            }

        }
        currentResult = idx;
    }

    @Override
    public Number currentResult() {
        if(currIndexOfMax == 0)
            return 0;
        return currIndexOfMax - 1;
    }

    @Override
    public Op opForDimension(int index, int... dimension) {
        INDArray xAlongDimension = x.tensorAlongDimension(index, dimension);

        if (y() != null)
            return new IAMax(xAlongDimension, y.tensorAlongDimension(index, dimension), xAlongDimension.length());
        else
            return new IAMax(x.tensorAlongDimension(index, dimension));
    }
}
