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

import org.apache.commons.math3.util.FastMath;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.util.ComplexUtil;

/**
 * Standard deviation (sqrt of variance)
 *
 * @author Adam Gibson
 */
public class StandardDeviation extends Variance {
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
    public void update(Number result) {
        super.update(result);
        if (n() == numProcessed()) {
            currentResult = FastMath.sqrt(currentResult.doubleValue());
        }
    }

    @Override
    public void update(IComplexNumber result) {
        super.update(result);
        if (n() == numProcessed())
            currentComplexResult = ComplexUtil.sqrt(currentComplexResult);

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
            return new StandardDeviation(x.vectorAlongDimension(index, dimension));

    }
}
