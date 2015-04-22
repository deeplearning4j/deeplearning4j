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
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.VectorFFT;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.fft.FFT;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static org.junit.Assert.assertEquals;

/**
 * Base class for FFTs
 */
public abstract class BaseFFTTest {

    private static Logger log = LoggerFactory.getLogger(BaseFFTTest.class);


    @Test
    public void testColumnVector() {
        IComplexNDArray n = (IComplexNDArray) Nd4j.getExecutioner().execAndReturn(new VectorFFT(Nd4j.complexLinSpace(1,8,8),8));
        IComplexNDArray assertion = Nd4j.createComplex(new double[]
                {36., 0., -4., 9.65685425, -4., 4, -4., 1.65685425, -4., 0., -4., -1.65685425, -4., -4., -4., -9.65685425
                }, new int[]{8});
        assertEquals(n, assertion);

    }


    @Test
    public void testWithOffset() {
        Nd4j.factory().setOrder('f');
        INDArray n = Nd4j.create(Nd4j.linspace(1, 30, 30).data(), new int[]{3, 5, 2});
        INDArray swapped = n.swapAxes(n.shape().length - 1, 1);
        INDArray firstSlice = swapped.slice(0).slice(0);
        IComplexNDArray test = Nd4j.createComplex(firstSlice);
        IComplexNDArray testNoOffset = Nd4j.createComplex(new double[]{1, 0, 4, 0, 7, 0, 10, 0, 13, 0}, new int[]{5});
        assertEquals(Nd4j.getExecutioner().execAndReturn(new VectorFFT(testNoOffset,5)), Nd4j.getExecutioner().execAndReturn(new VectorFFT(test,5)));

    }

    @Test
    public void testSimple() {
        IComplexNDArray arr = Nd4j.createComplex(new IComplexNumber[]{Nd4j.createComplexNumber(5, 0), Nd4j.createComplexNumber(1, 0)});
        IComplexNDArray arr2 = Nd4j.createComplex(new IComplexNumber[]{Nd4j.createComplexNumber(1, 0), Nd4j.createComplexNumber(5, 0)});

        IComplexNDArray assertion = Nd4j.createComplex(new IComplexNumber[]{Nd4j.createComplexNumber(6,0),Nd4j.createComplexNumber(4,0)});
        IComplexNDArray assertion2 = Nd4j.createComplex(new IComplexNumber[]{Nd4j.createComplexNumber(6,0),Nd4j.createComplexNumber(-4,4.371139E-7)});

        assertEquals(assertion,Nd4j.getFFt().fft(arr));
        assertEquals(assertion2,Nd4j.getFFt().fft(arr2));
    }

    @Test
    public void testMultiDimFFT() {
        INDArray a = Nd4j.linspace(1,8,8).reshape(2,2,2);
        INDArray fftedAnswer = Nd4j.create(new int[]{2, 2}, new double[]{6, 10, -4, -4}, new double[]{8, 12, -4, -4});
        INDArray ffted = FFT.fft(a);
        assertEquals(fftedAnswer,ffted);

    }



}


