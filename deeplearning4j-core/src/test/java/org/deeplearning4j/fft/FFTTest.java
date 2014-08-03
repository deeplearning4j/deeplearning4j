package org.deeplearning4j.fft;

import static org.junit.Assert.*;

import org.deeplearning4j.nn.linalg.ComplexNDArray;
import org.deeplearning4j.nn.linalg.NDArray;
import org.deeplearning4j.nn.linalg.Shape;
import org.deeplearning4j.util.ArrayUtil;
import org.jblas.ComplexDoubleMatrix;
import org.jblas.DoubleMatrix;
import org.junit.Test;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;

/**
 * Testing FFTs
 */
public class FFTTest {

    private static Logger log = LoggerFactory.getLogger(FFTTest.class);

    @Test
    public void testBasicFFT() {
        DoubleMatrix d = DoubleMatrix.linspace(1,8,8);
        ComplexNDArray d2 = ComplexNDArray.wrap(new ComplexDoubleMatrix(d));
        ComplexDoubleMatrix fft = FFT.fft(d2);
        assertEquals(8, fft.length);
        ComplexDoubleMatrix test = new ComplexDoubleMatrix(new double[]{
                36,
                0,
                -4,
                9.65685425,
                -4,
                4,
                -4,
                1.65685425,
                -4,
                0,
                -4,
                -1.65685425,
                -4,
                -4,
                -4,
                -9.65685425
        }).reshape(1,8);


        assertEquals(fft.rows,test.rows);
        assertEquals(fft.columns,test.columns);
        assertEquals(fft,test);


        ComplexNDArray three = new ComplexNDArray(new NDArray(new double[]{3,4},new int[]{2}));
        ComplexNDArray threeAnswer = new ComplexNDArray(new double[]{7,0,-1,0},new int[]{2});
        ComplexNDArray fftedThree = FFT.fft(three);
        assertEquals(threeAnswer,fftedThree);





    }

    @Test
    public void testMultiDimFFT() {
        //1d case: these should be equal
        DoubleMatrix d = DoubleMatrix.linspace(1,8,8);
        NDArray arr = new NDArray(d.data,new int[]{1});
        ComplexNDArray arr2 = FFT.fftn(arr, 0, 1);
        assertEquals(arr,arr2.getReal());

        ComplexNDArray other = FFT.fftn(arr,0,1);
        assertEquals(1,other.length);
        NDArray single = NDArray.scalar(1.0);
        NDArray real = other.getReal();
        assertEquals(single,real);


        NDArray n = new NDArray(DoubleMatrix.linspace(1,24,24).data,new int[]{4,3,2});
        ComplexNDArray fftedResult = FFT.fftn(n,1,1);
        ComplexNDArray test = new ComplexNDArray(new NDArray(new double[]{1,2,7,8,13,14,19,20},new int[]{4,1,2}));
        assertEquals(test,fftedResult);


    }

    @Test
    public void testRawFftn() {
        ComplexNDArray test = new ComplexNDArray(new NDArray(DoubleMatrix.linspace(1,30,30).data,new int[]{3,5,2}));
        ComplexNDArray result = new ComplexNDArray(new double[]{
                465.,
                0.,
                -15.,
                0.,
                -30.,
                41.29145761,
                0.,
                0.,
                -30.,
                9.74759089,
                0.,
                0.,
                -30,
                -9.74759089,
                0.,
                0.,
                -30,
                -41.29145761,
                0.,
                0.,
                -150.,
                86.60254038,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                -150,
                -86.60254038,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.

        },new int[]{3,5,2});


        ComplexNDArray ffted = FFT.rawfftn(test,test.shape(), ArrayUtil.range(0,3));
        assertEquals(true,Shape.shapeEquals(result.shape(),ffted.shape()));
        assertEquals(result,ffted);

    }


    @Test
    public void testFFTDifferentDimensions() {
        ComplexNDArray fftTest = new ComplexNDArray(new NDArray(DoubleMatrix.linspace(1,30,30).data,new int[]{3,5,2}));
        ComplexNDArray result = FFT.fft(fftTest);

        ComplexNDArray assertion = new ComplexNDArray(new double[] {
                3 , 0 ,
                -1 , 0 ,
                7 , 0 ,
                -1 , 0 ,
                11 , 0 ,
                -1 , 0 ,
                15 , 0 ,
                -1 , 0 ,
                19 , 0 ,
                -1 , 0 ,
                23 , 0 ,
                -1 , 0 ,
                27 , 0 ,
                -1 , 0 ,
                31 , 0 ,
                -1 , 0 ,
                35 , 0 ,
                -1 , 0 ,
                39 , 0 ,
                -1 , 0 ,
                43 , 0 ,
                -1 , 0 ,
                47 , 0 ,
                -1 , 0 ,
                51 , 0 ,
                -1 , 0 ,
                55 , 0 ,
                -1 , 0 ,
                59 , 0 ,
                -1 , 0 ,
        },new int[]{3,5,2});

        assertEquals(assertion.length,ArrayUtil.prod(assertion.shape()));
        assertEquals(assertion,result);
    }


    @Test
    public void testFFTOp() {
        NDArray arr = new NDArray(DoubleMatrix.linspace(1,24,24).data,new int[]{4,3,2});
        log.info("Before " + arr);
        arr.iterateOverDimension(0,new FFTSliceOp(arr.shape()[0]),true);
        log.info("After " + arr);
        arr.iterateOverDimension(1,new FFTSliceOp(arr.shape()[1]),true);

    }

    @Test
    public void testBasicIFFT() {
        DoubleMatrix d = DoubleMatrix.linspace(1,6,6);
        ComplexNDArray d2 = ComplexNDArray.wrap(new ComplexDoubleMatrix(d));
        ComplexDoubleMatrix fft = FFT.ifft(FFT.fft(d2));
        assertEquals(6, fft.length);

        assertEquals(d2,fft);



    }

    @Test
    public void testIFFT() {
        NDArray arr = new NDArray(DoubleMatrix.linspace(1,24,24).data,new int[]{4,3,2});
        log.info("Before " + arr);
        arr.iterateOverDimension(1,new IFFTSliceOp(arr.shape()[1]),true);
        log.info("After " + arr);

    }



}
