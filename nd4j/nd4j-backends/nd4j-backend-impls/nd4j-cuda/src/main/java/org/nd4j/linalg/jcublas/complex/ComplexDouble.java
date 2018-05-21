/*-
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 *
 */

package org.nd4j.linalg.jcublas.complex;

import org.nd4j.linalg.api.complex.BaseComplexDouble;
import org.nd4j.linalg.api.complex.IComplexFloat;
import org.nd4j.linalg.api.complex.IComplexNumber;

/**
 * Double implementation of a complex number.
 * Based on the jblas api by mikio braun
 *
 * @author Adam Gibson
 */
public class ComplexDouble extends BaseComplexDouble {

    public final static ComplexDouble UNIT = new ComplexDouble(1, 0);
    public final static ComplexDouble NEG = new ComplexDouble(-1, 0);
    public final static ComplexDouble ZERO = new ComplexDouble(0, 0);

    public ComplexDouble(double real, double imag) {
        super(real, imag);
    }

    public ComplexDouble(double real) {
        super(real);
    }


    @Override
    public IComplexNumber dup() {
        return new ComplexDouble(real, imag);
    }

    /**
     * Convert to a float
     *
     * @return this complex number as a float
     */
    @Override
    public IComplexFloat asFloat() {
        return new ComplexFloat((float) real, (float) imag);
    }
}
