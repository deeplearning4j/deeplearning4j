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
import org.nd4j.linalg.util.ComplexUtil;

/**
 * Over all normmax of an ndarray
 *
 * @author Adam Gibson1
 */
public class NormMax extends BaseScalarOp {
    public NormMax() {
        super(Nd4j.createDouble(0, 0));
    }

    @Override
    public IComplexNumber accumulate(IComplexNDArray arr, int i, IComplexNumber soFar) {
        IComplexNumber abs = ComplexUtil.abs(arr.getComplex(i));
        return abs.absoluteValue().doubleValue() > soFar.absoluteValue().doubleValue() ? abs : soFar;
    }
}
