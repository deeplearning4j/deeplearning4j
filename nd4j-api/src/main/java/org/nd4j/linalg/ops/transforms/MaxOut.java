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
 * Max out activation:
 * http://arxiv.org/pdf/1302.4389.pdf
 *
 * @author Adam Gibson
 */
public class MaxOut extends BaseElementWiseOp {

    /**
     * Apply the transformation
     */
    @Override
    public void exec() {
        INDArray linear = from.linearView();
        if (linear instanceof IComplexNDArray) {
            IComplexNDArray cLinear = (IComplexNDArray) linear;
            int max = Nd4j.getBlasWrapper().iamax(cLinear);
            IComplexNumber max2 = cLinear.getComplex(max);
            for (int i = 0; i < cLinear.length(); i++) {
                cLinear.putScalar(i, max2);
            }

        } else {
            int max = Nd4j.getBlasWrapper().iamax(linear);
            double maxNum = linear.getFloat(max);
            for (int i = 0; i < linear.length(); i++) {
                from.putScalar(i, maxNum);
            }
        }
    }

    @Override
    public String name() {
        return "maxout";
    }

    /**
     * The transformation for a given value (a scalar)
     *
     * @param origin the origin ndarray
     * @param value  the value to apply (a scalar)
     * @param i      the index of the element being acted upon
     * @return the transformed value based on the input
     */
    @Override
    public <E> E apply(INDArray origin, Object value, int i) {
        return null;
    }
}
