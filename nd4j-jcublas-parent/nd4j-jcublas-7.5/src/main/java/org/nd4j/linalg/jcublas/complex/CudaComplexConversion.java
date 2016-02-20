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

package org.nd4j.linalg.jcublas.complex;

import jcuda.cuComplex;
import jcuda.cuDoubleComplex;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Conversion between IComplexNumber and cuda types
 *
 * @author Adam Gibson
 */
public class CudaComplexConversion {

    /**
     * Create a complex doube from a cuda complex double
     *
     * @param cuDoubleComplex the double to create from
     * @return the created double
     */
    public static IComplexDouble toCuDouble(cuDoubleComplex cuDoubleComplex) {
        return Nd4j.createDouble(cuDoubleComplex.x, cuDoubleComplex.y);
    }

    /**
     * Create a complex float from a cuda float
     *
     * @param cuComplex the cuda float to convert
     * @return the create float
     */
    public static IComplexFloat toCuFloat(cuComplex cuComplex) {
        return Nd4j.createFloat(cuComplex.x, cuComplex.y);
    }

    /**
     * Convert a complex float to a cuda complex float
     *
     * @param float1 the float to convert
     * @return
     */
    public static cuComplex toComplex(IComplexFloat float1) {
        return cuComplex.cuCmplx(float1.realComponent().floatValue(), float1.imaginaryComponent().floatValue());
    }


    /**
     * Convert a complex double to a cuda complex double
     *
     * @param num the number to convert
     * @return the cuda complex double
     */
    public static cuDoubleComplex toComplexDouble(IComplexDouble num) {
        return cuDoubleComplex.cuCmplx(num.realComponent().doubleValue(), num.imaginaryComponent().doubleValue());
    }
}
