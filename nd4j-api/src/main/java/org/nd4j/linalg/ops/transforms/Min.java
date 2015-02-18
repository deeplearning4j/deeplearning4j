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

import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.BaseElementWiseOp;

/**
 * Minyou  function
 *
 * @author Adam Gibson
 */
public class Min extends BaseElementWiseOp {
    private Number min = 0;

    public Min(Double min) {
        this.min = min;
    }

    public Min(Float min) {
        this.min = min;
    }

    public Min(Number min) {
        this.min = min;
    }

    public Min() {
    }

    /**
     * The transformation for a given value (a scalar ndarray)
     *
     * @param value the value to apply (a scalar ndarray)
     * @param i     the index of the element being acted upon
     * @return the transformed value based on the input
     */
    @Override
    public Object apply(INDArray from, Object value, int i) {
        if (value instanceof IComplexNumber) {
            IComplexNumber num = (IComplexNumber) value;
            if (num.realComponent().doubleValue() < min.doubleValue())
                return num;
            return num.set(min, num.imaginaryComponent());
        }

        double val = (double) value;
        return Math.min(min.doubleValue(), val);
    }
}
