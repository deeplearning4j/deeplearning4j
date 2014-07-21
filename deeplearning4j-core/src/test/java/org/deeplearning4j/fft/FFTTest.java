package org.deeplearning4j.fft;

import static org.junit.Assert.*;

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
        DoubleMatrix d = DoubleMatrix.linspace(1,5,5);
        ComplexDoubleMatrix d2 = new ComplexDoubleMatrix(d);
        ComplexDoubleMatrix fft = FFT.fft(d2,1);
        assertEquals(1,fft.length);
        log.info("FFT " + fft);
    }


}
