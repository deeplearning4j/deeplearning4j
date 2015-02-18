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
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.BaseElementWiseOp;

/**
 * Ensures numerical stability.
 * Clips values of input such that
 * exp(k * in) is within single numerical precision
 *
 * @author Adam Gibson
 */
public class Stabilize extends BaseElementWiseOp {
    private double k = 1;

    public Stabilize(Double k) {
        this.k = k;
    }

    public Stabilize(Float k) {
        this.k = k;
    }

    public Stabilize(double k) {
        this.k = k;
    }

    public Stabilize() {
    }

    /**
     * The transformation for a given value (a scalar ndarray)
     *
     * @param value the value to applyTransformToOrigin (a scalar ndarray)
     * @param i     the index of the element being acted upon
     * @return the transformed value based on the input
     */
    @Override
    public Object apply(INDArray from, Object value, int i) {
        double realMin = 1.1755e-38f;
        double cutOff = FastMath.log(realMin);
        if (value instanceof IComplexNumber) {
            IComplexNumber c = (IComplexNumber) value;
            double curr = c.realComponent().doubleValue();
            if (curr * k > -cutOff)
                return Nd4j.createDouble(-cutOff / k, c.imaginaryComponent().doubleValue());
            else if (curr * k < cutOff)
                return Nd4j.createDouble(cutOff / k, c.imaginaryComponent().doubleValue());


        } else {
            double curr = (double) value;
            if (curr * k > -cutOff)
                return -cutOff / k;
            else if (curr * k < cutOff)
                return cutOff / k;

        }


        return value;
    }
}
