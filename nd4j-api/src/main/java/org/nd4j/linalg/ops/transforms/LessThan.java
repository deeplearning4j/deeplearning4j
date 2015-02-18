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

package org.nd4j.linalg.ops.transforms;

import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.BaseElementWiseOp;

/**
 * Less than operator
 *
 * @author Adam Gibson
 */
public class LessThan extends BaseElementWiseOp {
    /**
     * The transformation for a given value (a scalar ndarray)
     *
     * @param value the value to applyTransformToOrigin (a scalar ndarray)
     * @param i     the index of the element being acted upon
     * @return the transformed value based on the input
     */
    @Override
    public Object apply(INDArray from, Object value, int i) {
        Object curr = getFromOrigin(from, i);
        double originValue = (curr instanceof IComplexNDArray ? ((IComplexNumber) curr).absoluteValue().doubleValue() : (double) curr);
        double otherValue = (value instanceof IComplexNDArray ? ((IComplexNumber) value).absoluteValue().doubleValue() : (double) value);
        if (originValue < otherValue) {
            if (value instanceof IComplexNumber)
                return Nd4j.createDouble(1, 0);

            else
                return 1;


        } else {
            if (value instanceof IComplexNDArray)
                return Nd4j.createDouble(0, 0);

            else
                return 0;


        }
    }
}
