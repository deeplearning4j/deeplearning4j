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

package org.nd4j.linalg.ops.elementwise;


import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.ops.BaseTwoArrayElementWiseOp;

/**
 * Add a scalar or a matrix
 *
 * @author Adam Gibson
 */
public class AddOp extends BaseTwoArrayElementWiseOp {


    @Override
    protected IComplexNumber complexComplex(IComplexNumber num1, IComplexNumber num2) {
        return num1.add(num2);
    }

    @Override
    protected IComplexNumber realComplex(double real, IComplexNumber other) {
        return other.add(real);
    }

    @Override
    protected IComplexNumber complexReal(IComplexNumber origin, double secondValue) {
        return origin.add(secondValue);
    }

    @Override
    protected double realReal(double firstElement, double secondElement) {
        return firstElement + secondElement;
    }
}
