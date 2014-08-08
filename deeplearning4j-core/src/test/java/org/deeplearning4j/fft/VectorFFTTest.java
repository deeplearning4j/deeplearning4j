package org.deeplearning4j.fft;

import static org.junit.Assert.*;

import org.deeplearning4j.nn.linalg.ComplexNDArray;
import org.deeplearning4j.nn.linalg.NDArray;
import org.jblas.ComplexFloat;
import org.jblas.ComplexFloatMatrix;
import org.jblas.FloatMatrix;
import org.junit.Test;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 1d fft tests
 *
 * @author Adam Gibson
 */
public class VectorFFTTest {
    private static Logger log = LoggerFactory.getLogger(VectorFFTTest.class);
    private double[] testVector = new double[]{ 55. , 0.00000000e00  , -5.,  1.53884177e01 ,  -5. , 6.88190960e00,   -5. , 3.63271264e00 ,  -5. , 1.62459848e00  , -5., 4.44089210e-16, -5. , -1.62459848e00  , -5. , -3.63271264e00 ,  -5. , -6.88190960e00,  -5. , -1.53884177e01};
    private float[] testFloatVector = new float[]{ 55f , 0f  , -5,  1.53884177e01f ,  -5f , 6.88190960e00f,   -5f , 3.63271264e00f ,  -5f , 1.62459848e00f  , -5f, 4.44089210e-16f, -5.f , -1.62459848e00f  , -5.f , -3.63271264e00f ,  -5.f , -6.88190960e00f,  -5.f , -1.53884177e01f};

    @Test
    public void testRowVector() {
        ComplexNDArray n = new VectorFFT(10).apply(ComplexNDArray.linspace(1,10,10));
        ComplexNDArray assertion = new ComplexNDArray(testVector,new int[]{10});
        assertEquals(n,assertion);



    }

    @Test
    public void testColumnVector() {
        ComplexNDArray n = new VectorFFT(8).apply(ComplexNDArray.linspace(1, 8,8));
        ComplexNDArray assertion = new ComplexNDArray(new double[]
                { 36.,0.,-4.,9.65685425 ,-4.,4,-4.,1.65685425,-4.,0.,-4.,-1.65685425,-4.,-4.,-4.,-9.65685425
                },new int[]{8});
        assertEquals(n,assertion);

    }


    @Test
    public void testRowVectorFloat() {
        ComplexFloatMatrix n = new VectorFloatFFT(10).apply(new ComplexFloatMatrix(FloatMatrix.linspace(1,10,10)));
        ComplexFloatMatrix assertion = new ComplexFloatMatrix(testFloatVector);
        assertEquals(n,assertion);



    }

    @Test
    public void testColumnVectorFloat() {
        ComplexNDArray n = new VectorFFT(10).apply(ComplexNDArray.linspace(1,10,10));
        ComplexNDArray assertion = new ComplexNDArray(new double[]{ 36.,0.,-4.,9.65685425 ,-4.,4,-4.,1.65685425,-4.,0.,-4.,-1.65685425,-4.,-4.,-4.,-9.65685425},new int[]{10});
        assertEquals(n,assertion);

    }


}
