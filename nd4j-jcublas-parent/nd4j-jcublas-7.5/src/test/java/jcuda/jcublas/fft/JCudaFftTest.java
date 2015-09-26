package jcuda.jcublas.fft;

import static org.junit.Assert.*;

import org.junit.Test;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.fft.FFTInstance;
import org.nd4j.linalg.jcublas.fft.JcudaFft;

/**
 * @author Adam Gibson
 */
public class JCudaFftTest {
   private  FFTInstance instance = new JcudaFft();

    @Test
    public void test1d() {
        Nd4j.EPS_THRESHOLD = 1e-1;
        Nd4j.MAX_ELEMENTS_PER_SLICE = Integer.MAX_VALUE;
        Nd4j.MAX_SLICES_TO_PRINT = Integer.MAX_VALUE;
        IComplexNDArray complexLinSpace = Nd4j.complexLinSpace(1,8,8);
        IComplexNDArray n = instance.fft(complexLinSpace,8);
        IComplexNDArray assertion = Nd4j.createComplex(new double[]
                {36., 0., -4., 9.65685425, -4., 4, -4., 1.65685425, -4., 0., -4., -1.65685425, -4., -4., -4., -9.65685425
                }, new int[]{1,8});
       assertEquals(assertion,n);

    }




    @Test
    public void testOnes() {
        Nd4j.EPS_THRESHOLD = 1e-1;
        IComplexNDArray ones = Nd4j.complexOnes(5, 5);
        IComplexNDArray ffted = instance.fftn(ones);
        IComplexNDArray zeros = Nd4j.createComplex(5, 5);
        zeros.putScalar(0, 0, Nd4j.createComplexNumber(25, 0));
        assertEquals(zeros, ffted);




    }



}
