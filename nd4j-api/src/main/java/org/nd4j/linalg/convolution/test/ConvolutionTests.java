package org.nd4j.linalg.convolution.test;

import static org.junit.Assert.*;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.convolution.Convolution;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Created by agibsonccc on 9/6/14.
 */
public abstract class ConvolutionTests {


    private static Logger log = LoggerFactory.getLogger(ConvolutionTests.class);


    INDArray image = Nd4j.create(new double[][]{
            {3, 2, 5, 6, 7, 8},
            {5, 4, 2, 10, 8, 1}
    });

    INDArray kernel =  Nd4j.create((new double[][] {
            {4,5},
            {1,2}
    }));

    @Test
    public void convNTest() {
        INDArray arr = Nd4j.linspace(1,8,8);
        INDArray kernel = Nd4j.linspace(1,3,3);
        INDArray answer = Nd4j.create(new double[]{1,4,10, 16, 22, 28, 34, 40,37,24}, new int[]{10, 1});
        INDArray test = Convolution.convn(arr, kernel, Convolution.Type.VALID);
        //technically close enough...may look in to this if its a problem later.
        assertEquals(answer,test);
    }







    @Test
    public void testConvolution() {
        INDArray image = Nd4j.create(new double[][]{
                {3, 2, 5, 6, 7, 8},
                {5, 4, 2, 10, 8, 1}
        });

        INDArray kernel = Nd4j.create(new double[][] {
                {4,5},
                {1,2}
        });



        log.info(Convolution.convn(image, kernel, Convolution.Type.FULL).toString());

        log.info(Convolution.convn(image, kernel, Convolution.Type.VALID).toString());
    }


}
