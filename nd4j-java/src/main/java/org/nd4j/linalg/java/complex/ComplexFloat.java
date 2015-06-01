/*
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

package org.nd4j.linalg.java.complex;

import org.nd4j.linalg.api.complex.BaseComplexFloat;
import org.nd4j.linalg.api.complex.IComplexDouble;
import org.nd4j.linalg.api.complex.IComplexFloat;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Complex float
 *
 * @author Adam Gibson
 */
public class ComplexFloat extends BaseComplexFloat {

    public ComplexFloat() {
    }

    public ComplexFloat(float real) {
        super(real);
    }

    public ComplexFloat(Float real, Float imag) {
        super(real, imag);
    }

    public ComplexFloat(float real, float imag) {
        super(real, imag);
    }

    @Override
    public IComplexNumber dup() {
        return Nd4j.createFloat(real,imag);
    }
}
