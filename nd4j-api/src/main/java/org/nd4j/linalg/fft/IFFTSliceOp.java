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
 * Dimension wise IFFT
 *
 * @author Adam Gibson
 */
public class IFFTSliceOp implements SliceOp {


    private int n; //number of elements per dimension


    /**
     * IFFT on the given nd array
     *
     * @param n number of elements per dimension for the ifft
     */
    public IFFTSliceOp(int n) {
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
        if (nd instanceof IComplexNDArray) {
            IComplexNDArray a = (IComplexNDArray) nd;
            int n = this.n < 1 ? a.length() : this.n;
            INDArray result = new VectorIFFT(n).apply(a).getReal();
            for (int i = 0; i < result.length(); i++) {
                a.put(i, result.getScalar(i));
            }
        } else {
            INDArray a = nd;
            int n = this.n < 1 ? a.length() : this.n;

            INDArray result = new VectorIFFT(n).apply(Nd4j.createComplex(a)).getReal();
            for (int i = 0; i < result.length(); i++) {
                a.put(i, result.getScalar(i));
            }

        }

    }


}
