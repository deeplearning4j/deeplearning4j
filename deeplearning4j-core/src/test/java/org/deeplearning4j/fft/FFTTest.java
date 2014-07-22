package org.deeplearning4j.fft;

import static org.junit.Assert.*;

import org.deeplearning4j.nn.linalg.NDArray;
import org.jblas.ComplexDoubleMatrix;
import org.jblas.DoubleMatrix;
import org.junit.Test;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Testing FFTs
 */
public class FFTTest {

    private static Logger log = LoggerFactory.getLogger(FFTTest.class);

    @Test
    public void testBasicFFT() {
        DoubleMatrix d = DoubleMatrix.linspace(1,6,6);
        ComplexDoubleMatrix d2 = new ComplexDoubleMatrix(d);
        ComplexDoubleMatrix fft = FFT.fft(d2);
        assertEquals(6,fft.length);
        log.info("FFT " + fft);



    }

    @Test
    public void testFFTOp() {
        NDArray arr = new NDArray(DoubleMatrix.linspace(1,24,24).data,new int[]{4,3,2});
        log.info("Before " + arr);
        arr.iterateOverDimension(0,new FFTSliceOp(arr));
        log.info("After " + arr);
    }

    @Test
    public void testBasicIFFT() {
        DoubleMatrix d = DoubleMatrix.linspace(1,6,6);
        ComplexDoubleMatrix d2 = new ComplexDoubleMatrix(d);
        ComplexDoubleMatrix fft = FFT.ifft(FFT.fft(d2));
        assertEquals(6,fft.length);
        log.info("IFFT " + fft);

        assertEquals(d2,fft);



    }

    @Test
    public void testIFFT() {
        NDArray arr = new NDArray(DoubleMatrix.linspace(1,24,24).data,new int[]{4,3,2});
        log.info("Before " + arr);
        arr.iterateOverDimension(0,new IFFTSliceOp(arr,arr.shape()[0]));
        log.info("After " + arr);
    }



}
