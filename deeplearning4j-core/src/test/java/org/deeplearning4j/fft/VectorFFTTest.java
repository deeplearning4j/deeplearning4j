package org.deeplearning4j.fft;

import static org.junit.Assert.*;

import org.deeplearning4j.nn.linalg.ComplexNDArray;
import org.deeplearning4j.nn.linalg.NDArray;
import org.junit.Test;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Created by agibsonccc on 8/5/14.
 */
public class VectorFFTTest {
    private static Logger log = LoggerFactory.getLogger(VectorFFTTest.class);

    @Test
    public void testRowVector() {
        ComplexNDArray n = new VectorFFT(10).apply(ComplexNDArray.linspace(1,10,10));
        ComplexNDArray assertion = new ComplexNDArray(new double[]{ 36.,0.,-4.,9.65685425 ,-4.,4,-4.,1.65685425,-4.,0.,-4.,-1.65685425,-4.,-4.,-4.,-9.65685425},new int[]{10});
        assertEquals(n,assertion);



    }

    @Test
    public void testColumnVector() {
        ComplexNDArray n = new VectorFFT(10).apply(ComplexNDArray.linspace(1,10,10).transpose());
        ComplexNDArray assertion = new ComplexNDArray(new double[]{ 36.,0.,-4.,9.65685425 ,-4.,4,-4.,1.65685425,-4.,0.,-4.,-1.65685425,-4.,-4.,-4.,-9.65685425},new int[]{10});
        assertEquals(n,assertion);

    }


}
