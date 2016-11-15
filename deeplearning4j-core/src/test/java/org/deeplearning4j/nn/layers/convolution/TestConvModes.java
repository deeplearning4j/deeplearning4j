package org.deeplearning4j.nn.layers.convolution;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.convolution.Convolution;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import static org.junit.Assert.assertEquals;

/**
 * Created by Alex on 15/11/2016.
 */
public class TestConvModes {

    @Test
    public void testIm2ColModes() {

        //
        int miniBatch = 2;
        int inDepth = 3;

        int[] kernel = {2, 2};
        int[] strides = {2, 2};
        int[] padding = {0, 0};

        INDArray in1 = Nd4j.rand(new int[]{miniBatch, inDepth, 4, 4});
        INDArray in2 = Nd4j.rand(new int[]{miniBatch, inDepth, 5, 5});

        in2.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.interval(0, 4), NDArrayIndex.interval(0, 4)).assign(in1);

        int hOut1 = (in1.size(2) - kernel[0] + 2 * padding[0]) / strides[0] + 1;
        int wOut1 = (in1.size(3) - kernel[1] + 2 * padding[1]) / strides[1] + 1;

        int hOut2 = (in2.size(2) - kernel[0] + 2 * padding[0]) / strides[0] + 1;
        int wOut2 = (in2.size(3) - kernel[1] + 2 * padding[1]) / strides[1] + 1;


        System.out.println(hOut1 + "\t" + wOut1);
        System.out.println(hOut2 + "\t" + wOut2);

        INDArray cola = Nd4j.createUninitialized(new int[]{miniBatch, hOut1, wOut1, inDepth, kernel[0], kernel[1]}, 'c');
        INDArray col2a = cola.permute(0, 3, 4, 5, 1, 2);
        Convolution.im2col(in1, kernel[0], kernel[1], strides[0], strides[1], padding[0], padding[1], false, col2a);

        INDArray colb = Nd4j.createUninitialized(new int[]{miniBatch, hOut2, wOut2, inDepth, kernel[0], kernel[1]}, 'c');
        INDArray col2b = colb.permute(0, 3, 4, 5, 1, 2);
        Convolution.im2col(in2, kernel[0], kernel[1], strides[0], strides[1], padding[0], padding[1], false, col2b);

        assertEquals(col2a, col2b);
    }


    @Test
    public void testIm2ColModes2() {

        //
        int miniBatch = 2;
        int inDepth = 3;

        //Test 1: passes
//        int[] kernel = {2, 2};
//        int[] strides = {2, 2};
//        int[] padding = {0, 0};
//
//        int inH = 4;
//        int inW = 4;


        //Test 2: passes
//        int[] kernel = {3, 3};
//        int[] strides = {3, 3};
//        int[] padding = {0, 0};
//
//        int inH = 9;
//        int inW = 9;

        //Test 3: passes (with i<strides[0])
        int[] kernel = {3, 3};
        int[] strides = {2, 2};
        int[] padding = {0, 0};

        int inH = 9;
        int inW = 9;

//        for( int i=1; i<kernel[0]; i++ ) {
        for( int i=1; i<strides[0]; i++ ) {

            INDArray in1 = Nd4j.rand(new int[]{miniBatch, inDepth, inH, inW});
            INDArray in2 = Nd4j.rand(new int[]{miniBatch, inDepth, inH+i, inW+i});

            in2.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.interval(0, inH), NDArrayIndex.interval(0, inH)).assign(in1);

            int hOut1 = (in1.size(2) - kernel[0] + 2 * padding[0]) / strides[0] + 1;
            int wOut1 = (in1.size(3) - kernel[1] + 2 * padding[1]) / strides[1] + 1;

            int hOut2 = (in2.size(2) - kernel[0] + 2 * padding[0]) / strides[0] + 1;
            int wOut2 = (in2.size(3) - kernel[1] + 2 * padding[1]) / strides[1] + 1;


            System.out.println(hOut1 + "\t" + wOut1);
            System.out.println(hOut2 + "\t" + wOut2);

            INDArray cola = Nd4j.createUninitialized(new int[]{miniBatch, hOut1, wOut1, inDepth, kernel[0], kernel[1]}, 'c');
            INDArray col2a = cola.permute(0, 3, 4, 5, 1, 2);
            Convolution.im2col(in1, kernel[0], kernel[1], strides[0], strides[1], padding[0], padding[1], false, col2a);

            INDArray colb = Nd4j.createUninitialized(new int[]{miniBatch, hOut2, wOut2, inDepth, kernel[0], kernel[1]}, 'c');
            INDArray col2b = colb.permute(0, 3, 4, 5, 1, 2);
            Convolution.im2col(in2, kernel[0], kernel[1], strides[0], strides[1], padding[0], padding[1], false, col2b);

            assertEquals(col2a, col2b);
        }
    }

}
