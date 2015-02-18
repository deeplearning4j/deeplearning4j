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
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ComplexNDArrayUtil;

/**
 * Single ifft operation
 *
 * @author Adam Gibson
 */
public class VectorIFFT implements Function<IComplexNDArray, IComplexNDArray> {


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
    public VectorIFFT(int n) {
        this.n = n;
    }


    @Override
    public IComplexNDArray apply(IComplexNDArray ndArray) {
        //ifft(x) = conj(fft(conj(x)) / length(x)
        IComplexNDArray ret = new VectorFFT(n).apply(ndArray.conj()).conj().divi(Nd4j.complexScalar(n));
        return originalN > 0 ? ComplexNDArrayUtil.truncate(ret, originalN, 0) : ret;

    }
}
