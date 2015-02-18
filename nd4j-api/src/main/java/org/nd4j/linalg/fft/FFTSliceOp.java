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


import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ndarray.SliceOp;
import org.nd4j.linalg.factory.Nd4j;

/**
 * FFT Slice operation
 *
 * @author Adam Gibson
 */
public class FFTSliceOp implements SliceOp {
    private int n; //number of elements to operate on per dimension


    /**
     * FFT operation on a given slice.
     * Will throw an {@link IllegalArgumentException}
     * if n < 1
     *
     * @param n the number of elements per dimension in fft
     */
    public FFTSliceOp(int n) {
        if (n < 1)
            throw new IllegalArgumentException("Number of elements per dimension must be at least 1");
        this.n = n;
    }


    /**
     * Operates on an ndarray slice
     *
     * @param nd the result to operate on
     */
    @Override
    public void operate(INDArray nd) {
        if (nd instanceof INDArray) {
            INDArray a = nd;

            int n = this.n < 1 ? a.length() : this.n;

            INDArray result = new VectorFFT(n).apply(Nd4j.createComplex(a)).getReal();
            for (int i = 0; i < result.length(); i++)
                a.putScalar(i, result.getDouble(i));
        } else if (nd instanceof IComplexNDArray) {
            IComplexNDArray a = (IComplexNDArray) nd;
            IComplexNDArray result = new VectorFFT(n).apply(a);
            for (int i = 0; i < result.length(); i++) {
                a.putScalar(i, result.getComplex(i));
            }

        }
    }


}
