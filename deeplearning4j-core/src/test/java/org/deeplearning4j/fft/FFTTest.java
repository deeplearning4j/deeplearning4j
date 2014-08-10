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


        ComplexNDArray anotherTest = new ComplexNDArray(new NDArray(new double[]{3,7,11,15,19},new int[]{5}));
        ComplexNDArray assertion = new ComplexNDArray(new double[]{55,0,-10,13.7638192,-10,3.24919696,-10,-3.24919696,-10,-13.7638192},new int[]{5});
        assertEquals(FFT.fft(anotherTest),assertion);


        ComplexNDArray one = ComplexNDArray.scalar(1);
        ComplexNDArray fftedOne = FFT.fft(d2,1);
        assertEquals(one,fftedOne);



    }


    @Test
    public void testAllDimensionFFT() {
        NDArray n = new NDArray(DoubleMatrix.linspace(1,30,30).data,new int[]{3,5,2});
        ComplexNDArray afterLastDimension = new ComplexNDArray(new double[]{
                3.,0,-1.,0,7.,0,-1.,0, 11.,0,-1.,0, 15.,0,-1.,0, 19.,0,-1.,0,23.,0,-1.,0, 27.,0,-1.,0, 31.,0,-1.,0, 35.,0,-1.,0, 39.,0,-1.,0,43.,0,-1.,0, 47.,0,-1.,0, 51.,0,-1.,0, 55.,0,-1.,0, 59.,0,-1.,0
        },new int[]{3,5,2});

        ComplexNDArray afterLastDimensionTest = FFT.fft(n,2,n.shape().length - 1);
        assertEquals(afterLastDimension,afterLastDimensionTest);

        ComplexNDArray afterSecondDimension = new ComplexNDArray(new double[]{
                25.,0.,30.,0.,-5.,6.8819096,-5.,6.8819096,-5.,1.62459848,-5.,1.62459848,-5.,-1.62459848,-5.,-1.62459848,-5.,-6.8819096,-5.,-6.8819096,75.,0.,80.,0.,-5.,6.8819096,-5.,6.8819096,-5.,1.62459848,-5.,1.62459848,-5.,-1.62459848,-5.,-1.62459848,-5.,-6.8819096,-5.,-6.8819096,125.,0.,130.,0.,-5.,6.8819096,-5.,6.8819096,-5.,1.62459848,-5.,1.62459848,-5.,-1.62459848,-5.,-1.62459848,-5.,-6.8819096,-5.,-6.8819096,        },new int[]{3,5,2});


        ComplexNDArray afterSecondDimensionTest = FFT.fft(n,5,n.shape().length  - 2);
        assertEquals(afterSecondDimension,afterSecondDimensionTest);


        ComplexNDArray afterFirstDimension = new ComplexNDArray(new double[]{
                33.,0.,36.,0.,39.,0.,42.,0.,45.,0.,48.,0.,51.,0.,54.,0.,57.,0.,60.,0.,-15.,8.66025404,-15.,8.66025404,-15.,8.66025404,-15.,8.66025404,-15.,8.66025404,-15.,8.66025404,-15.,8.66025404,-15.,8.66025404,-15.,8.66025404,-15.,8.66025404,-15.,-8.66025404,-15.,-8.66025404,-15.,-8.66025404,-15.,-8.66025404,-15.,-8.66025404,-15.,-8.66025404,-15.,-8.66025404,-15.,-8.66025404,-15.,-8.66025404,-15.,-8.66025404,465.,0.,-15.,0.,-30.,41.29145761,0.,0.,-30.,9.74759089,0.,0.,-30.,-9.74759089,0.,0.,-30.-41.29145761,0.,0.,-150.,86.60254038,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,-150.-86.60254038,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,
        },new int[]{3,5,2});


        ComplexNDArray afterFirstDimensionTest = FFT.fft(n,3,n.shape().length  - 3);
        assertEquals(afterFirstDimension,afterFirstDimensionTest);



    }


    @Test
    public void testMultiDimFFT() {
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
        ComplexNDArray fftedResult = FFT.fftn(n);
        ComplexNDArray test = new ComplexNDArray(new NDArray(new double[]{1,2,7,8,13,14,19,20},new int[]{4,1,2}));
        //assertEquals(test,fftedResult);


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
        ComplexNDArray ffted = FFT.ifft(d2);
        double[] data = new double[]{
                3.5
                ,0.
                ,-0.5
                ,-8.66025404e-01
                ,-0.5
                ,-2.88675135e-01
                ,-0.5
                ,-5.18104078e-16
                ,-0.5
                ,2.88675135e-01
                ,-0.5
                ,8.66025404e-01
        };

        ComplexNDArray assertion = new ComplexNDArray(data,new int[]{ffted.length});

        assertEquals(ffted,assertion);


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
