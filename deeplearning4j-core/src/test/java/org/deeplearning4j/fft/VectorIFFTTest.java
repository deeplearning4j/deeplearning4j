package org.deeplearning4j.fft;


import static org.junit.Assert.*;

import org.deeplearning4j.nn.linalg.ComplexNDArray;
import org.junit.Test;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Vector IFFT Tests
 *
 * @author Adam Gibson
 */
public class VectorIFFTTest {

    private static Logger log = LoggerFactory.getLogger(VectorIFFT.class);

    @Test
    public void testIfft() {
        double[] ffted = {10.2,5.,-3.0,-1.};
        double[] orig = {3.5999999999999996,2, 6.5999999999999996 ,3};
        ComplexNDArray c = new ComplexNDArray(orig,new int[]{2});
        ComplexNDArray assertion = new ComplexNDArray(ffted,new int[]{2});

        assertEquals(assertion,new VectorFFT(2).apply(c));
        ComplexNDArray iffted = new VectorIFFT(2).apply(assertion.dup());
        assertEquals(iffted,c);



        double[] ffted2 = {17.8,9.,-1,-8.6,4.6,3.};
        double[] orig2 = {3.6,2, 6.6 ,3,7.6 ,4};
        double[] fftOrig2 = { 17.8000000,9,-4.3660254,-0.6339746,-2.6339746,-2.3660254};
        ComplexNDArray c2 = new ComplexNDArray(orig2,new int[]{3});

        ComplexNDArray fftOrig2Arr = new ComplexNDArray(fftOrig2,new int[]{fftOrig2.length / 2});
        ComplexNDArray fftOrig2Test = new VectorFFT(fftOrig2Arr.length).apply(c2);
        assertEquals(fftOrig2Arr,fftOrig2Test);

        ComplexNDArray ifftTestFor = new ComplexNDArray(new double[]{3.6,2,6.6,3,7.6,4},new int[]{3});
        ComplexNDArray ifftTest = new VectorIFFT(fftOrig2Arr.length).apply(fftOrig2Arr);
        assertEquals(ifftTestFor,ifftTest);
    }



}
