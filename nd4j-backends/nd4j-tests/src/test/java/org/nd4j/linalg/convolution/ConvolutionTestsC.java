/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 *
 */

package org.nd4j.linalg.convolution;
import static org.junit.Assert.*;

import org.junit.Ignore;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.AllocUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.util.Arrays;

/**
 * Created by agibsonccc on 9/6/14.
 */
@RunWith(Parameterized.class)
public  class ConvolutionTestsC extends BaseNd4jTest {

    public ConvolutionTestsC(Nd4jBackend backend) {
        super(backend);
    }


    @Test
    public void testConvOutWidthAndHeight() {
        int outSize = Convolution.outSize(2,1,1,2,false);
        assertEquals(6, outSize);
    }

    @Test
    public void testIm2Col() {
        INDArray linspaced = Nd4j.linspace(1,16,16).reshape(2, 2, 2, 2);
        INDArray ret = Convolution.im2col(linspaced, 1, 1, 1, 1, 2, 2, 0, false);
        INDArray im2colAssertion = Nd4j.create(new double[]{
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 6.0, 0.0, 0.0, 0.0, 0.0, 7.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.0, 10.0, 0.0, 0.0, 0.0, 0.0, 11.0, 12.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 13.0, 14.0, 0.0, 0.0, 0.0, 0.0, 15.0, 16.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        }, new int[]{2, 2, 1, 1, 6, 6});
        assertEquals(im2colAssertion, ret);
        INDArray col2ImAssertion = Nd4j.create(new double[] {
                1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,16.0

        }, new int[]{2,2,2,2});

        INDArray otherConv = Convolution.col2im(ret, 1, 1, 2, 2, 2, 2);
        assertEquals(col2ImAssertion,otherConv);

    }

    @Test
    public void testIm2Col2() {
        // n, c, h, w = new_val.shape
        int kh = 2;
        int kw = 2;
        int ph = 0;
        int pw = 0;
        int sy = 2;
        int sx = 2;
        int depth = 2;
        INDArray assertion  = Nd4j.create(new double[]{
                1, 1, 1, 1, 3, 3, 3, 3, 1, 1, 1, 1, 3, 3, 3, 3, 1, 1, 1, 1, 3, 3, 3, 3, 1, 1, 1, 1, 3, 3, 3, 3, 2, 2, 2, 2, 4, 4, 4, 4, 2, 2, 2, 2, 4, 4, 4, 4, 2, 2, 2, 2, 4, 4, 4, 4, 2, 2, 2, 2, 4, 4, 4, 4
        }, new int[]{1, 1, 2, 2, 4, 4});
        INDArray ret = Nd4j.create(new double[]{
                1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3,
                3, 4, 4, 4, 4, 4, 4, 4, 4, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2,
                2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4
        }, new int[]{1, 1, 8, 8});

        INDArray test = Convolution.im2col(ret, kh, kw, sy, sx, ph, pw, 0, false);
        assertEquals(assertion,test);

    }



    @Test
    public void testMoreIm2Col2() {
        int kh = 2;
        int kw = 2;
        int ph = 0;
        int pw = 0;
        int sy = 2;
        int sx = 2;

        INDArray ret = Nd4j.create(new double[]{
                1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3,
                3, 4, 4, 4, 4, 4, 4, 4, 4, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2,
                2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4
        }, new int[]{1, 1, 8, 8});

        INDArray assertion = Nd4j.create(new double[]{1,1,1,1,3,3,3,3,1,1,1,1,3,3,3,3,1,1,1,1,3,3,3,3,1,1,1,1,3,3,3,3
                ,2,2,2,2,4,4,4,4,2,2,
                2,2,4,4,4,4,2,2,2,2,4,4,4,4,2,2,2,2,4,4,4,4},new int[] {1, 1, 2, 2, 4, 4});
        INDArray im2colTest = Convolution.im2col(ret, kh, kw, sy, sx, ph, pw, 0, false);
        assertEquals(assertion,im2colTest);
    }





    @Override
    public char ordering() {
        return 'c';
    }
}
