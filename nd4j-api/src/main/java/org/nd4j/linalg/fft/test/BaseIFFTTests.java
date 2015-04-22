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

package org.nd4j.linalg.fft.test;

import org.junit.Test;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.ops.impl.transforms.VectorFFT;
import org.nd4j.linalg.api.ops.impl.transforms.VectorIFFT;
import org.nd4j.linalg.factory.Nd4j;


import static org.junit.Assert.assertEquals;

/**
 * Created by agibsonccc on 9/6/14.
 */
public abstract class BaseIFFTTests {
    @Test
    public void testIfft() {
        double[] ffted = {10.2, 5., -3.0, -1.};
        double[] orig = {3.5999999999999996, 2, 6.5999999999999996, 3};
        IComplexNDArray c = Nd4j.createComplex(orig, new int[]{2});
        IComplexNDArray assertion = Nd4j.createComplex(ffted, new int[]{2});

        assertEquals(assertion, Nd4j.getExecutioner().execAndReturn(new VectorFFT(c,2)));
        IComplexNDArray iffted = (IComplexNDArray) Nd4j.getExecutioner().execAndReturn(new VectorIFFT(assertion.dup(),2));
        assertEquals(iffted, c);


        double[] ffted2 = {17.8, 9., -1, -8.6, 4.6, 3.};
        double[] orig2 = {3.6, 2, 6.6, 3, 7.6, 4};
        double[] fftOrig2 = {17.8000000, 9, -4.3660254, -0.6339746, -2.6339746, -2.3660254};
        IComplexNDArray c2 = Nd4j.createComplex(orig2, new int[]{3});

        IComplexNDArray fftOrig2Arr = Nd4j.createComplex(fftOrig2, new int[]{fftOrig2.length / 2});
        IComplexNDArray fftOrig2Test = (IComplexNDArray) Nd4j.getExecutioner().execAndReturn(new VectorFFT(assertion.dup(),fftOrig2Arr.length()));
        assertEquals(fftOrig2Arr, fftOrig2Test);

        IComplexNDArray ifftTestFor = Nd4j.createComplex(new double[]{3.6, 2, 6.6, 3, 7.6, 4}, new int[]{3});
        IComplexNDArray ifftTest =  (IComplexNDArray) Nd4j.getExecutioner().execAndReturn(new VectorFFT(fftOrig2Arr,fftOrig2Arr.length()));
        assertEquals(ifftTestFor, ifftTest);
    }


}
