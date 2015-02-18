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

package org.nd4j.linalg.ops.reduceops.complex;

import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.complex.IComplexNumber;

/**
 * Scalar ops for complex ndarrays
 *
 * @author Adam Gibson
 */
public class ComplexOps {


    public static IComplexNumber std(IComplexNDArray arr) {
        return new StandardDeviation().apply(arr);
    }

    public static IComplexNumber norm1(IComplexNDArray arr) {
        return new Norm1().apply(arr);
    }

    public static IComplexNumber norm2(IComplexNDArray arr) {
        return new Norm2().apply(arr);
    }

    public static IComplexNumber normmax(IComplexNDArray arr) {
        return new NormMax().apply(arr);
    }


    public static IComplexNumber max(IComplexNDArray arr) {
        return new Max().apply(arr);
    }

    public static IComplexNumber min(IComplexNDArray arr) {
        return new Min().apply(arr);
    }

    public static IComplexNumber mean(IComplexNDArray arr) {
        return new Mean().apply(arr);
    }

    public static IComplexNumber sum(IComplexNDArray arr) {
        return new Sum().apply(arr);
    }

    public static IComplexNumber var(IComplexNDArray arr) {
        return new Variance().apply(arr);
    }

    public static IComplexNumber prod(IComplexNDArray arr) {
        return new Prod().apply(arr);
    }

}
