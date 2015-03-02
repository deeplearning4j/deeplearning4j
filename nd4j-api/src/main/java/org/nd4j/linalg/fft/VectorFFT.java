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

package org.nd4j.linalg.fft;


import com.google.common.base.Function;
import org.apache.commons.math3.util.FastMath;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ComplexNDArrayUtil;

import static org.nd4j.linalg.ops.transforms.Transforms.exp;


/**
 * Encapsulated vector operation
 *
 * @author Adam Gibson
 */
public class VectorFFT implements Function<IComplexNDArray, IComplexNDArray> {
    private int n;
    private int originalN = -1;

    /**
     * Create a vector fft operation.
     * If initialized with  a nonzero number, this will
     * find the next power of 2 for the element and truncate the
     * return matrix to the original n
     *
     * @param n
     */
    public VectorFFT(int n) {
        this.n = n;
    }

    @Override
    public IComplexNDArray apply(IComplexNDArray ndArray) {
        double len = n;

        int desiredElementsAlongDimension = ndArray.length();

        if (len > desiredElementsAlongDimension) {
            ndArray = ComplexNDArrayUtil.padWithZeros(ndArray, new int[]{n});
        } else if (len < desiredElementsAlongDimension) {
            ndArray = ComplexNDArrayUtil.truncate(ndArray, n, 0);
        }


        IComplexNumber c2 = Nd4j.createDouble(0, -2).muli(FastMath.PI);
        //row vector
        INDArray n = Nd4j.arange(0, this.n).reshape(1, this.n);

        //column vector
        INDArray k = n.reshape(new int[]{n.length(), 1});
        INDArray kTimesN = k.mmul(n);
        //here
        IComplexNDArray c1 = kTimesN.muli(c2);
        c1.divi(len);
        IComplexNDArray M = (IComplexNDArray) exp(c1);


        IComplexNDArray reshaped = ndArray.reshape(new int[]{ndArray.length()});
        IComplexNDArray matrix = reshaped.mmul(M);
        if (originalN > 0) {
            matrix = ComplexNDArrayUtil.truncate(matrix, originalN, 0);

        }

        return matrix;
    }


}
