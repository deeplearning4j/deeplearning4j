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

import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Return the overall standard deviation of an ndarray
 *
 * @author Adam Gibson
 */
public class StandardDeviation extends BaseScalarOp {
    public StandardDeviation() {
        super(Nd4j.createDouble(0, 0));
    }

    public IComplexNumber std(IComplexNDArray arr) {
        org.apache.commons.math3.stat.descriptive.moment.StandardDeviation dev = new org.apache.commons.math3.stat.descriptive.moment.StandardDeviation();
        double std = dev.evaluate(arr.ravel().data().asDouble());
        return Nd4j.createDouble(std, 0);
    }


    @Override
    public IComplexNumber apply(IComplexNDArray input) {
        return std(input);
    }

    @Override
    public IComplexNumber accumulate(IComplexNDArray arr, int i, IComplexNumber soFar) {
        return Nd4j.createDouble(0, 0);
    }
}
