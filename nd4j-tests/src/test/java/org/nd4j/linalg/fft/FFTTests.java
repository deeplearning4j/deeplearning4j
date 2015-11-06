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

package org.nd4j.linalg.fft;


import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.VectorFFT;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;


/**
 * Base class for FFTs
 *
 * @author Adam Gibson
 */
@Ignore
public  class FFTTests extends BaseNd4jTest {


    public FFTTests(String name) {
        super(name);
    }

    public FFTTests(Nd4jBackend backend) {
        super(backend);
    }

    public FFTTests(String name, Nd4jBackend backend) {
        super(name, backend);
    }

    public FFTTests() {
    }

    @Test
    public void testVectorFftOnes() {
        INDArray arr = Nd4j.ones(5);
        VectorFFT fft = new VectorFFT(arr);
        fft.exec();
        INDArray assertion = Nd4j.create(5);
        assertion.putScalar(0,5);
        assertEquals(getFailureMessage(),assertion,fft.z());
    }


    @Test
    public void testColumnVector() {
        Nd4j.EPS_THRESHOLD = 1e-1;
        Nd4j.MAX_ELEMENTS_PER_SLICE = Integer.MAX_VALUE;
        Nd4j.MAX_SLICES_TO_PRINT = Integer.MAX_VALUE;
        IComplexNDArray complexLinSpace = Nd4j.complexLinSpace(1, 8, 8);
        IComplexNDArray n = (IComplexNDArray) Nd4j.getExecutioner().execAndReturn(new VectorFFT(complexLinSpace,8));
        IComplexNDArray assertion = Nd4j.createComplex(new double[]
                {36., 0., -4., 9.65685425, -4., 4, -4., 1.65685425, -4., 0., -4., -1.65685425, -4., -4., -4., -9.65685425
                }, new int[]{1, 8});
        assertEquals(getFailureMessage(),n, assertion);

    }


    @Test
    public void testWithOffset() {
        INDArray n = Nd4j.create(Nd4j.linspace(1, 30, 30).data(), new int[]{3, 5, 2});
        INDArray swapped = n.swapAxes(n.shape().length - 1, 1);
        INDArray firstSlice = swapped.slice(0).slice(0);
        IComplexNDArray test = Nd4j.createComplex(firstSlice);
        IComplexNDArray testNoOffset = Nd4j.createComplex(new double[]{1, 0, 4, 0, 7, 0, 10, 0, 13, 0}, new int[]{1,5});
        assertEquals(getFailureMessage(),Nd4j.getExecutioner().execAndReturn(new VectorFFT(testNoOffset,5)), Nd4j.getExecutioner().execAndReturn(new VectorFFT(test,5)));


    }



    @Test
    public void testSimple() {
        Nd4j.EPS_THRESHOLD = 1e-1;

        IComplexNDArray arr = Nd4j.createComplex(new IComplexNumber[]{Nd4j.createComplexNumber(5, 0), Nd4j.createComplexNumber(1, 0)});
        IComplexNDArray arr2 = Nd4j.createComplex(new IComplexNumber[]{Nd4j.createComplexNumber(1, 0), Nd4j.createComplexNumber(5, 0)});

        IComplexNDArray assertion = Nd4j.createComplex(new IComplexNumber[]{Nd4j.createComplexNumber(6,0),Nd4j.createComplexNumber(4,0)});
        IComplexNDArray assertion2 = Nd4j.createComplex(new IComplexNumber[]{Nd4j.createComplexNumber(6,0),Nd4j.createComplexNumber(-4,4.371139E-7)});

        assertEquals(getFailureMessage(),assertion,Nd4j.getFFt().fft(arr));
        assertEquals(getFailureMessage(),assertion2,Nd4j.getFFt().fft(arr2));
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
        assertEquals(getFailureMessage(),fftedAnswer,ffted);


        Nd4j.EPS_THRESHOLD = 1e-1;

    }

    @Test
    public void testOnes() {
        Nd4j.EPS_THRESHOLD = 1e-1;
        IComplexNDArray ones = Nd4j.complexOnes(5, 5);
        IComplexNDArray ffted = FFT.fftn(ones);
        IComplexNDArray zeros = Nd4j.createComplex(5, 5);
        zeros.putScalar(0, 0, Nd4j.createComplexNumber(25, 0));
        assertEquals(getFailureMessage(),zeros, ffted);
    }



    @Test
    public void testConv4d() {
        IComplexNDArray test = Nd4j.complexOnes(new int[]{5,5,5,5});
        Nd4j.getFFt().fftn(test);

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
            assertEquals(getFailureMessage() + " Failed on iteration " + i, assertionSlice, resultSlice);
        }
    }

    @Override
    public char ordering() {
        return 'f';
    }
}


