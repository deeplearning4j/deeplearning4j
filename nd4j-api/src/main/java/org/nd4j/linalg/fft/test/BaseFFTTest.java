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
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.fft.VectorFFT;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static org.junit.Assert.assertEquals;

/**
 * Base class for FFTs
 */
public abstract class BaseFFTTest {

    private static Logger log = LoggerFactory.getLogger(BaseFFTTest.class);

    private double[] testVector = new double[]{
            55.00000000
            , 0.00000000e+00
            , -26.37586651
            , -2.13098631e+01
            , 12.07106781
            , 2.58578644e+00
            , -9.44674873
            , 1.75576651e+00
            , 5.00000000
            , -6.00000000e+00
            , -0.89639702
            , 5.89790214e+00
            , -2.07106781
            , -5.41421356e+00
            , 4.71901226
            , 2.83227249e+00
            , -5.00000000
            , -6.12323400e-15
            , 4.71901226
            , -2.83227249e+00
            , -2.07106781
            , 5.41421356e+00
            , -0.89639702
            , -5.89790214e+00
            , 5.00000000
            , 6.00000000e+00
            , -9.44674873
            , -1.75576651e+00
            , 12.07106781
            , -2.58578644e+00
            , -26.37586651
            , 2.13098631e+01
    };

    private float[] testFloatVector = new float[]{55f, 0f, -5, 1.53884177e01f, -5f, 6.88190960e00f, -5f, 3.63271264e00f, -5f, 1.62459848e00f, -5f, 4.44089210e-16f, -5.f, -1.62459848e00f, -5.f, -3.63271264e00f, -5.f, -6.88190960e00f, -5.f, -1.53884177e01f};


    @Test
    public void testColumnVector() {
        IComplexNDArray n = new VectorFFT(8).apply(Nd4j.complexLinSpace(1, 8, 8));
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
        VectorFFT op = new VectorFFT(5);
        INDArray firstSlice = swapped.slice(0).slice(0);
        IComplexNDArray test = Nd4j.createComplex(firstSlice);
        IComplexNDArray testNoOffset = Nd4j.createComplex(new double[]{1, 0, 4, 0, 7, 0, 10, 0, 13, 0}, new int[]{5});
        assertEquals(op.apply(testNoOffset), op.apply(test));

    }


}


