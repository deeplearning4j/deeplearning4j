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
       NDArray arr = new NDArray(DoubleMatrix.linspace(1,8,8).data,new int[]{1,8},0);
       NDArray kernel = new NDArray(DoubleMatrix.linspace(1,3,3).data,new int[]{1,3},0);
       NDArray answer = new NDArray(new double[]{10,16,22,28,34,40},new int[]{1,6},0);
       NDArray test = Convolution.convn(arr,kernel,Type.VALID);
       assertEquals(answer,test);
    }

    @Test
    public void testDiscreteFourierTransform() {





        DoubleMatrix imageOutput = new DoubleMatrix(new double[][] {
                { 61.,-19.74943104 , 11.57755637  , 5.67187467 ,  5.67187467,11.57755637, -19.74943104},
                { 16., -9.96703093 , -9.01450594 , 4.26549287 ,  9.2946051, -5.71985961 , -1.35870149},
                { 16.,  -1.35870149  ,-5.71985961,  9.2946051, 4.26549287,-9.01450594 , -9.96703093}
        });

        DoubleMatrix testOutput = Convolution.disceteFourierTransform(image, 3, 7);

        assertEquals(testOutput,imageOutput);

        DoubleMatrix testInverseOutput = new DoubleMatrix(new double[][] {
                { 12., 9.36442861  , 3.44235346 , -1.30678208 , -1.30678208, 3.44235346 ,  9.36442861},
                {  7.5 , 4.63978736  , 0.92129159 , -0.85538415   ,0.64763321, 4.29854094  , 7.34813106},
                {  7.5, 7.34813106  , 4.29854094 ,  0.64763321 , -0.85538415, 0.92129159 ,  4.63978736}
        });

        DoubleMatrix testInverseImage = Convolution.disceteFourierTransform(kernel, 3, 7);

        assertEquals(testInverseOutput,testInverseImage);





    }


    @Test
    public void testInverseDiscreteFourierTransform() {
        DoubleMatrix rightOutput = new DoubleMatrix(  new double[][]
                {
                        {2.90476, -0.94045, 0.551310, 0.27009, 0.27009, 0.55131, -0.94045},
                        {0.76190, -0.47462, -0.42926, 0.20312, 0.44260, -0.27237, -0.06470},
                        {0.76190, -0.06470, -0.27237, 0.44260, 0.20312, -0.42926, -0.47462}
                });
        DoubleMatrix testOutput = Convolution.inverseDisceteFourierTransform(image,3,7);
        assertEquals(rightOutput,testOutput);
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
