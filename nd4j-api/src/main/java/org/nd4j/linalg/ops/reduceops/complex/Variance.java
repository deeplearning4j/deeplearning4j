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

package org.nd4j.linalg.ops.reduceops.complex;

import org.apache.commons.math3.stat.StatUtils;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Return the variance of an ndarray
 *
 * @author Adam Gibson
 */
public class Variance extends BaseScalarOp {

    public Variance() {
        super(Nd4j.createDouble(0, 0));
    }


    public IComplexNumber var(IComplexNDArray arr) {
        IComplexNumber mean = new Mean().apply(arr);
        return Nd4j.createDouble(StatUtils.variance(arr.ravel().data().asDouble(), mean.absoluteValue().floatValue()), 0);
    }


    @Override
    public IComplexNumber apply(IComplexNDArray input) {
        return var(input);
    }

    @Override
    public IComplexNumber accumulate(IComplexNDArray arr, int i, IComplexNumber soFar) {
        return Nd4j.createDouble(0, 0);
    }
}
