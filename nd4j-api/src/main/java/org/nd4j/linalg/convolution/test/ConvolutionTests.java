/*
 * Copyright 2015 Skymind,Inc.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

package org.nd4j.linalg.convolution.test;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.convolution.Convolution;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Created by agibsonccc on 9/6/14.
 */
public abstract class ConvolutionTests {


    private static Logger log = LoggerFactory.getLogger(ConvolutionTests.class);


    @Test
    public void convNTest() {
        INDArray arr = Nd4j.linspace(1, 8, 8);
        INDArray kernel = Nd4j.linspace(1, 3, 3);
        INDArray answer = Nd4j.create(new double[]{10, 16, 22, 28, 34, 40});
        INDArray test = Convolution.convn(arr, kernel, Convolution.Type.VALID);
        assertEquals(answer, test);
    }


    @Test
    public void testConv2d() {
        INDArray input = Nd4j.create(Nd4j.linspace(1, 8, 8).data(), new int[]{2, 2, 2});
        INDArray kernel = input.dup();
        INDArray convolution = Convolution.conv2d(input, kernel, Convolution.Type.FULL);
        assertTrue(Arrays.equals(new int[]{3, 3}, convolution.shape()));

        INDArray input2 = Nd4j.create(Nd4j.linspace(1, 16, 16).data(), new int[]{2, 4, 2});
        INDArray kernel2 = input2.dup();
        INDArray convolution2 = Convolution.conv2d(input2, kernel2, Convolution.Type.VALID);
        assertTrue(Arrays.equals(new int[]{2, 4}, convolution2.shape()));
    }


    @Test
    public void testConvolution() {
        INDArray image = Nd4j.create(new double[][]{
                {3, 2, 5, 6, 7, 8},
                {5, 4, 2, 10, 8, 1}
        });

        INDArray kernel = Nd4j.create(new double[][]{
                {4, 5},
                {1, 2}
        });


        log.info(Convolution.convn(image, kernel, Convolution.Type.FULL).toString());

        log.info(Convolution.convn(image, kernel, Convolution.Type.VALID).toString());
    }


}
