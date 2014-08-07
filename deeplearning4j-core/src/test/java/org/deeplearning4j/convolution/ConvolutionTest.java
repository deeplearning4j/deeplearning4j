package org.deeplearning4j.convolution;

import static org.junit.Assert.*;
import static org.deeplearning4j.util.Convolution.*;

import org.deeplearning4j.nn.linalg.NDArray;
import org.deeplearning4j.util.Convolution;
import org.jblas.DoubleMatrix;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * @author Adam Gibson
 */
public class ConvolutionTest {

    private static Logger log = LoggerFactory.getLogger(ConvolutionTest.class);
    DoubleMatrix image = new DoubleMatrix(new double[][]{
            {3,2,5,6,7,8},
            {5,4,2,10,8,1}
    });

    DoubleMatrix kernel = new DoubleMatrix(new double[][] {
            {4,5},
            {1,2}
    });

    @Test
    public void convNTest() {
        NDArray arr = new NDArray(DoubleMatrix.linspace(1,8,8).data,new int[]{8});
        NDArray kernel = new NDArray(DoubleMatrix.linspace(1,3,3).data,new int[]{3});
        NDArray answer = new NDArray(new double[]{10,16,22,28,34,40},new int[]{6,1});
        NDArray test = Convolution.convn(arr,kernel,Type.VALID);
        assertEquals(answer,test);
    }







    @Test
    public void testConvolution() {
        DoubleMatrix image = new DoubleMatrix(new double[][]{
                {3,2,5,6,7,8},
                {5,4,2,10,8,1}
        });

        DoubleMatrix kernel = new DoubleMatrix(new double[][] {
                {4,5},
                {1,2}
        });



        log.info(Convolution.conv2d(image, kernel, Type.FULL).toString());

        log.info(Convolution.conv2d(image, kernel, Type.VALID).toString());
    }

}
