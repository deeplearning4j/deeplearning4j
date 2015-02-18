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
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.BaseElementWiseOp;
import org.nd4j.linalg.util.ComplexUtil;

/**
 * Hard Tanh
 * <p/>
 * values range: -1 < tanh(x) < 1
 *
 * @author Adam Gibson
 */
public class HardTanh extends BaseElementWiseOp {


    /**
     * The transformation for a given value (a scalar ndarray)
     *
     * @param value the value to applyTransformToOrigin (a scalar ndarray)
     * @param i     the index of the element being acted upon
     * @return the transformed value based on the input
     */
    @Override
    public Object apply(INDArray from, Object value, int i) {
        if (value instanceof IComplexNumber) {
            IComplexNumber element = (IComplexNumber) value;
            IComplexNumber ret = ComplexUtil.tanh(element);
            if (ret.realComponent().doubleValue() < -1)
                ret.set(-1, ret.imaginaryComponent().doubleValue());
            if (ret.realComponent().doubleValue() > 1)
                ret.set(1, ret.imaginaryComponent().doubleValue());
            return Nd4j.scalar(ret);
        } else {
            double d = (double) value;
            double ret = Math.tanh(d);
            if (ret < -1)
                return -1;
            else if (ret > 1)
                return (double) 1;
            else
                return ret;
        }
    }
}