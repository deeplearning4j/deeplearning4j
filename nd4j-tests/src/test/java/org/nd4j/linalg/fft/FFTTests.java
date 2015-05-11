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


import org.junit.Test;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.VectorFFT;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static org.junit.Assert.assertEquals;

/**
 * Base class for FFTs
 */
public  class FFTTests extends BaseNd4jTest {

    private static Logger log = LoggerFactory.getLogger(FFTTests.class);

    public FFTTests(Nd4jBackend backend) {
        super(backend);
    }

    public FFTTests(String name, Nd4jBackend backend) {
        super(name, backend);
    }


    @Test
    public void testColumnVector() {
        Nd4j.EPS_THRESHOLD = 1e-1;
        IComplexNDArray n = (IComplexNDArray) Nd4j.getExecutioner().execAndReturn(new VectorFFT(Nd4j.complexLinSpace(1,8,8),8));
        IComplexNDArray assertion = Nd4j.createComplex(new double[]
                {36., 0., -4., 9.65685425, -4., 4, -4., 1.65685425, -4., 0., -4., -1.65685425, -4., -4., -4., -9.65685425
                }, new int[]{8});
        assertEquals(n, assertion);

    }


    @Test
    public void testWithOffset() {
        Nd4j.dtype = DataBuffer.Type.DOUBLE;
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
        Nd4j.EPS_THRESHOLD = 1e-1;
        INDArray a = Nd4j.linspace(1,8,8).reshape(2,2,2);
        IComplexNDArray fftedAnswer = Nd4j.createComplex(2, 2, 2);
        IComplexNDArray matrix1 = Nd4j.createComplex(new IComplexNumber[][] {
                {Nd4j.createComplexNumber(36,0),Nd4j.createComplexNumber(-16,0)}
                ,{Nd4j.createComplexNumber(-8,0),Nd4j.createComplexNumber(0,0)}
        });

        IComplexNDArray matrix2 = Nd4j.createComplex(new IComplexNumber[][] {
                {Nd4j.createComplexNumber(-4,0),Nd4j.createComplexNumber(0,0)}
                ,{Nd4j.createComplexNumber(0,0),Nd4j.createComplexNumber(0,0)}
        });

        fftedAnswer.putSlice(0,matrix1);
        fftedAnswer.putSlice(1, matrix2);

        IComplexNDArray ffted = FFT.fftn(a);
        assertEquals(fftedAnswer,ffted);






        Nd4j.EPS_THRESHOLD = 1e-12;

    }

    @Test
    public void testNoSwap() {
        Nd4j.EPS_THRESHOLD = 1e-1;

        IComplexNDArray linspaced = Nd4j.complexLinSpace(1,6,6).reshape(2,3);
        IComplexNDArray assertion = Nd4j.createComplex(2,3);
        assertion.putSlice(0, Nd4j.createComplex(new IComplexNumber[] {
                Nd4j.createComplexNumber(21,0),Nd4j.createComplexNumber(-6,3.46),Nd4j.createComplexNumber(-6,-3.46)
        }));
        assertion.putSlice(1,Nd4j.createComplex(new IComplexNumber[] {
                Nd4j.createComplexNumber(-3,0),Nd4j.createComplexNumber(0,0),Nd4j.createComplexNumber(0,0)
        }));

        IComplexNDArray fftLinspaced = Nd4j.getFFt().fftn(linspaced);

        assertEquals(assertion,fftLinspaced);

        fftLinspaced = Nd4j.getFFt().rawfftn(linspaced,new int[]{2,2},null);
        assertion = Nd4j.createComplex(new IComplexNumber[][]{
                {Nd4j.createComplexNumber(10,0),Nd4j.createComplexNumber(-4,0)}
                ,{Nd4j.createComplexNumber(-2,0),Nd4j.createComplexNumber(0,0)}
        });
        assertEquals(assertion,fftLinspaced);



    }

    @Test
    public void testOnes() {
        Nd4j.EPS_THRESHOLD = 1e-1;
        IComplexNDArray ones = Nd4j.complexOnes(5, 5);
        IComplexNDArray ffted = FFT.fftn(ones);
        IComplexNDArray zeros = Nd4j.createComplex(5, 5);
        zeros.putScalar(0, 0, Nd4j.createComplexNumber(25, 0));
        assertEquals(zeros, ffted);

        IComplexNDArray threeOnes = Nd4j.complexOnes(3, 3);
        IComplexNDArray threeComp = Nd4j.createComplex(3, 3);
        threeComp.putScalar(0, 0, Nd4j.createComplexNumber(9, 0));
        assertEquals(FFT.fftn(threeOnes), threeComp);



    }

    @Test
    public void testOnesDifferentShapes() {
        Nd4j.EPS_THRESHOLD = 1e-1;
        IComplexNDArray ones = Nd4j.complexOnes(5, 5);
        IComplexNDArray ffted = Nd4j.getFFt().rawfftn(ones,new int[]{3,3},new int[]{0,1});
        System.out.println(ffted);

    }




    @Test
    public void testRawfft() {
        Nd4j.EPS_THRESHOLD = 1e-1;

        IComplexNDArray test = Nd4j.complexOnes(5,5);
        IComplexNDArray result = Nd4j.getFFt().rawfft(test, 3, 1);
        IComplexNDArray assertion = Nd4j.createComplex(5,3);
        for(int i = 0; i < assertion.rows(); i++)
            assertion.slice(i).putScalar(0, Nd4j.createComplexNumber(3, 0));
        for(int i = 0; i < result.slices(); i++) {
            IComplexNDArray assertionSlice = assertion.slice(i);
            IComplexNDArray resultSlice = result.slice(i);
            assertEquals("Failed on iteration " + i, assertionSlice, resultSlice);
        }
    }





}


