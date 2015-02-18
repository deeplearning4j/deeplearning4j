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

import org.apache.commons.math3.util.FastMath;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.BaseElementWiseOp;
import org.nd4j.linalg.util.ComplexUtil;

/**
 * Exponential of an ndarray
 *
 * @author Adam Gibson
 */
public class Exp extends BaseElementWiseOp {
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
            IComplexNumber c = (IComplexNumber) value;
            return ComplexUtil.exp(c);
        } else {
            if (from.data().dataType() == (DataBuffer.FLOAT)) {
                double val = (double) value;
                return FastMath.exp(val);
            } else {
                double val = (double) value;
                if (val < 0) {
                    double ret = FastMath.exp(val);
                    return ret;
                } else
                    return Math.exp(val);


            }

        }

    }


    public double exp(double val) {
        final long tmp = (long) (1512775 * val) + 1072693248;
        final long mantissa = tmp & 0x000FFFFF;
        int error = (int) mantissa >> 7;   // remove chance of overflow
        error = (int) (error - mantissa * mantissa) / 186; // subtract mantissa^2 * 64
        // 64 / 186 = 1/2.90625
        return Double.longBitsToDouble((tmp - error) << 32);
    }
}
