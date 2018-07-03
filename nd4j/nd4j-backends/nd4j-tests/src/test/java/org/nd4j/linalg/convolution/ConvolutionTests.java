/*-
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

import org.junit.Ignore;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.AllocUtil;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.layers.convolution.Pooling2D;
import org.nd4j.linalg.checkutil.NDArrayCreationUtil;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.primitives.Pair;

import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.nd4j.linalg.indexing.NDArrayIndex.all;
import static org.nd4j.linalg.indexing.NDArrayIndex.point;

/**
 * Created by agibsonccc on 9/6/14.
 */
@RunWith(Parameterized.class)
public class ConvolutionTests extends BaseNd4jTest {

    public ConvolutionTests(Nd4jBackend backend) {
        super(backend);
    }


    @Test
    public void testIm2ColKnownValues() {
        //Input: w=3, h=3, depth=2, minibatch = 2
        //kH=2, kW=2
        /*
        ----- Input images -----
        example 0:
        depth 0     depth 1
        [ 0  1  2      [ 9 10 11
          3  4  5       12 13 14
          6  7  8]      15 16 17]
        example 1:
        [18 19 20      [27 28 29
         21 22 23       30 31 32
         24 25 26]      33 34 35]
        
         ----- Expected Output -----
         Shape: [miniBatch,depth,kH,kW,outH,outW]
         - example 0 -
         depth 0                        depth 1
         h0,w0      h0,w1               h0,w0      h0,w1
           0  1     1  2                 9 10      10 11
           3  4     4  5                12 13      13 14
        
         h1,w0      h1,w1               h1,w0      h1,w1
           3  4     4  5                12 13      13 14
           6  7     7  8                15 16      16 17
        
         - example 1 -
         depth 0                        depth 1
         h0,w0      h0,w1               h0,w0      h0,w1
          18 19     19 20               27 28      28 29
          21 22     22 23               30 31      31 32
        
         h1,w0      h1,w1               h1,w0      h1,w1
          21 22     22 23               30 31      31 32
          24 25     25 26               33 34      34 35
         */

        int miniBatch = 2;
        int depth = 2;
        int height = 3;
        int width = 3;

        int outH = 2;
        int outW = 2;
        int kH = 2;
        int kW = 2;
        int sX = 1;
        int sY = 1;
        int pX = 0;
        int pY = 0;

        //Input data: shape [miniBatch,depth,height,width]
        INDArray input = Nd4j.create(new int[] {miniBatch, depth, height, width}, 'c');
        input.put(new INDArrayIndex[] {point(0), point(0), all(),
                        all()}, Nd4j.create(new double[][] {{0, 1, 2}, {3, 4, 5}, {6, 7, 8}}));
        input.put(new INDArrayIndex[] {point(0), point(1), all(),
                        all()}, Nd4j.create(new double[][] {{9, 10, 11}, {12, 13, 14}, {15, 16, 17}}));
        input.put(new INDArrayIndex[] {point(1), point(0), all(),
                        all()}, Nd4j.create(new double[][] {{18, 19, 20}, {21, 22, 23}, {24, 25, 26}}));
        input.put(new INDArrayIndex[] {point(1), point(1), all(),
                        all()}, Nd4j.create(new double[][] {{27, 28, 29}, {30, 31, 32}, {33, 34, 35}}));

        //Expected data:
        INDArray expected = Nd4j.create(new int[] {miniBatch, depth, kH, kW, outH, outW}, 'c');

        //Example 0
        //depth 0
        expected.put(new INDArrayIndex[] {point(0), point(0), all(),
                        all(), point(0), point(0)},
                        Nd4j.create(new double[][] {{0, 1}, {3, 4}}));
        expected.put(new INDArrayIndex[] {point(0), point(0), all(),
                        all(), point(0), point(1)},
                        Nd4j.create(new double[][] {{1, 2}, {4, 5}}));
        expected.put(new INDArrayIndex[] {point(0), point(0), all(),
                        all(), point(1), point(0)},
                        Nd4j.create(new double[][] {{3, 4}, {6, 7}}));
        expected.put(new INDArrayIndex[] {point(0), point(0), all(),
                        all(), point(1), point(1)},
                        Nd4j.create(new double[][] {{4, 5}, {7, 8}}));
        //depth 1
        expected.put(new INDArrayIndex[] {point(0), point(1), all(),
                        all(), point(0), point(0)},
                        Nd4j.create(new double[][] {{9, 10}, {12, 13}}));
        expected.put(new INDArrayIndex[] {point(0), point(1), all(),
                        all(), point(0), point(1)},
                        Nd4j.create(new double[][] {{10, 11}, {13, 14}}));
        expected.put(new INDArrayIndex[] {point(0), point(1), all(),
                        all(), point(1), point(0)},
                        Nd4j.create(new double[][] {{12, 13}, {15, 16}}));
        expected.put(new INDArrayIndex[] {point(0), point(1), all(),
                        all(), point(1), point(1)},
                        Nd4j.create(new double[][] {{13, 14}, {16, 17}}));

        //Example 1
        //depth 0
        expected.put(new INDArrayIndex[] {point(1), point(0), all(),
                        all(), point(0), point(0)},
                        Nd4j.create(new double[][] {{18, 19}, {21, 22}}));
        expected.put(new INDArrayIndex[] {point(1), point(0), all(),
                        all(), point(0), point(1)},
                        Nd4j.create(new double[][] {{19, 20}, {22, 23}}));
        expected.put(new INDArrayIndex[] {point(1), point(0), all(),
                        all(), point(1), point(0)},
                        Nd4j.create(new double[][] {{21, 22}, {24, 25}}));
        expected.put(new INDArrayIndex[] {point(1), point(0), all(),
                        all(), point(1), point(1)},
                        Nd4j.create(new double[][] {{22, 23}, {25, 26}}));
        //depth 1
        expected.put(new INDArrayIndex[] {point(1), point(1), all(),
                        all(), point(0), point(0)},
                        Nd4j.create(new double[][] {{27, 28}, {30, 31}}));
        expected.put(new INDArrayIndex[] {point(1), point(1), all(),
                        all(), point(0), point(1)},
                        Nd4j.create(new double[][] {{28, 29}, {31, 32}}));
        expected.put(new INDArrayIndex[] {point(1), point(1), all(),
                        all(), point(1), point(0)},
                        Nd4j.create(new double[][] {{30, 31}, {33, 34}}));
        expected.put(new INDArrayIndex[] {point(1), point(1), all(),
                        all(), point(1), point(1)},
                        Nd4j.create(new double[][] {{31, 32}, {34, 35}}));

        INDArray out = Convolution.im2col(input, kH, kW, sY, sX, pY, pX, false);
        assertEquals(expected, out);

        //Now: test with a provided results array, where the results array has weird strides
        INDArray out2 = Nd4j.create(new int[] {miniBatch, depth, outH, outW, kH, kW}, 'c');
        INDArray out2p = out2.permute(0, 1, 4, 5, 2, 3);
        Convolution.im2col(input, kH, kW, sY, sX, pY, pX, false, out2p);
        assertEquals(expected, out2p);

        INDArray out3 = Nd4j.create(new int[] {miniBatch, outH, outW, depth, kH, kW}, 'c');
        INDArray out3p = out3.permute(0, 3, 4, 5, 1, 2);
        Convolution.im2col(input, kH, kW, sY, sX, pY, pX, false, out3p);
        assertEquals(expected, out3p);
    }


    @Test
    public void testIm2ColKnownValuesDilated() {
        //Input: w=4, h=4, depth=1, minibatch = 2, dilation=2, stride 1
        //kH=2, kW=2
        /*
        ----- Input images -----
        example 0:
        depth 0
        [ 0  1  2  3
          4  5  6  7
          8  9 10 11
         12 13 14 15 ]

        example 1:
        [16 17 18 19
         20 21 22 23
         24 25 26 27
         28 29 30 31 ]

         ----- Expected Output -----
         Shape: [miniBatch,depth,kH,kW,outH,outW]
         - example 0 -
         depth 0
         h0,w0      h0,w1
           0  2     1  3
           8 10     9 11

         h1,w0      h1,w1
           4  6     5  7
          12 14    13 15

         - example 1 -
         depth 0
         h0,w0      h0,w1
          16 18     17 19
          24 26     25 27

         h1,w0      h1,w1
          20 22     21 23
          28 30     29 31
         */

        int miniBatch = 2;
        int depth = 1;
        int height = 4;
        int width = 4;

        int outH = 2;
        int outW = 2;
        int kH = 2;
        int kW = 2;
        int sX = 1;
        int sY = 1;
        int pX = 0;
        int pY = 0;
        int dh = 2;
        int dw = 2;

        //Input data: shape [miniBatch,depth,height,width]
        INDArray input = Nd4j.create(new int[] {miniBatch, depth, height, width}, 'c');
        input.put(new INDArrayIndex[] {point(0), point(0), all(),
                all()}, Nd4j.create(new double[][] {{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}, {12, 13, 14, 15}}));
        input.put(new INDArrayIndex[] {point(1), point(0), all(),
                all()}, Nd4j.create(new double[][] {{16, 17, 18, 19}, {20, 21, 22, 23}, {24, 25, 26, 27}, {28, 29, 30, 31}}));

        //Expected data:
        INDArray expected = Nd4j.create(new int[] {miniBatch, depth, kH, kW, outH, outW}, 'c');

        //Example 0
        //depth 0
        expected.put(new INDArrayIndex[] {point(0), point(0), all(),
                        all(), point(0), point(0)},
                Nd4j.create(new double[][] {{0, 2}, {8, 10}}));
        expected.put(new INDArrayIndex[] {point(0), point(0), all(),
                        all(), point(0), point(1)},
                Nd4j.create(new double[][] {{1, 3}, {9, 11}}));
        expected.put(new INDArrayIndex[] {point(0), point(0), all(),
                        all(), point(1), point(0)},
                Nd4j.create(new double[][] {{4, 6}, {12, 14}}));
        expected.put(new INDArrayIndex[] {point(0), point(0), all(),
                        all(), point(1), point(1)},
                Nd4j.create(new double[][] {{5, 7}, {13, 15}}));

        //Example 1
        //depth 0
        expected.put(new INDArrayIndex[] {point(1), point(0), all(),
                        all(), point(0), point(0)},
                Nd4j.create(new double[][] {{16, 18}, {24, 26}}));
        expected.put(new INDArrayIndex[] {point(1), point(0), all(),
                        all(), point(0), point(1)},
                Nd4j.create(new double[][] {{17, 19}, {25, 27}}));
        expected.put(new INDArrayIndex[] {point(1), point(0), all(),
                        all(), point(1), point(0)},
                Nd4j.create(new double[][] {{20, 22}, {28, 30}}));
        expected.put(new INDArrayIndex[] {point(1), point(0), all(),
                        all(), point(1), point(1)},
                Nd4j.create(new double[][] {{21, 23}, {29, 31}}));

        INDArray out = Convolution.im2col(input, kH, kW, sY, sX, pY, pX, dh, dw, false);
        assertEquals(expected, out);

        //Now: test with a provided results array, where the results array has weird strides
        INDArray out2 = Nd4j.create(new int[] {miniBatch, depth, outH, outW, kH, kW}, 'c');
        INDArray out2p = out2.permute(0, 1, 4, 5, 2, 3);
        Convolution.im2col(input, kH, kW, sY, sX, pY, pX, dh, dw, false, out2p);
        assertEquals(expected, out2p);

        INDArray out3 = Nd4j.create(new int[] {miniBatch, outH, outW, depth, kH, kW}, 'c');
        INDArray out3p = out3.permute(0, 3, 4, 5, 1, 2);
        Convolution.im2col(input, kH, kW, sY, sX, pY, pX, dh, dw, false, out3p);
        assertEquals(expected, out3p);
    }

    @Test
    public void testIm2ColKnownValuesDilatedStrided() {
        //Input: w=5, h=5, depth=1, minibatch = 1, dilation=2, stride 2
        //kH=2, kW=2
        /*
        ----- Input images -----
        example 0:
        depth 0
        [ 0  1  2  3  4
          5  6  7  8  9
         10 11 12 13 14
         15 16 17 18 19
         20 21 22 23 24 ]

         ----- Expected Output -----
         Shape: [miniBatch,depth,kH,kW,outH,outW]
         - example 0 -
         depth 0
         h0,w0      h0,w1
           0  2     2  4
          10 12    12 14

         h1,w0      h1,w1
          10 12    12 14
          20 22    22 24
         */

        int miniBatch = 1;
        int depth = 1;
        int height = 5;
        int width = 5;

        int outH = 2;
        int outW = 2;
        int kH = 2;
        int kW = 2;
        int sX = 2;
        int sY = 2;
        int pX = 0;
        int pY = 0;
        int dh = 2;
        int dw = 2;

        //Input data: shape [miniBatch,depth,height,width]
        INDArray input = Nd4j.create(new int[] {miniBatch, depth, height, width}, 'c');
        input.put(new INDArrayIndex[] {point(0), point(0), all(),
                all()}, Nd4j.create(new double[][] {{0, 1, 2, 3, 4}, {5, 6, 7, 8, 9}, {10, 11, 12, 13, 14},
                {15, 16, 17, 18, 19}, {20, 21, 22, 23, 24}}));

        //Expected data:
        INDArray expected = Nd4j.create(new int[] {miniBatch, depth, kH, kW, outH, outW}, 'c');

        //Example 0
        //depth 0
        expected.put(new INDArrayIndex[] {point(0), point(0), all(),
                        all(), point(0), point(0)},
                Nd4j.create(new double[][] {{0, 2}, {10, 12}}));
        expected.put(new INDArrayIndex[] {point(0), point(0), all(),
                        all(), point(0), point(1)},
                Nd4j.create(new double[][] {{2, 4}, {12, 14}}));
        expected.put(new INDArrayIndex[] {point(0), point(0), all(),
                        all(), point(1), point(0)},
                Nd4j.create(new double[][] {{10, 12}, {20, 22}}));
        expected.put(new INDArrayIndex[] {point(0), point(0), all(),
                        all(), point(1), point(1)},
                Nd4j.create(new double[][] {{12, 14}, {22, 24}}));

        INDArray out = Convolution.im2col(input, kH, kW, sY, sX, pY, pX, dh, dw, false);
        assertEquals(expected, out);

        //Now: test with a provided results array, where the results array has weird strides
        INDArray out2 = Nd4j.create(new int[] {miniBatch, depth, outH, outW, kH, kW}, 'c');
        INDArray out2p = out2.permute(0, 1, 4, 5, 2, 3);
        Convolution.im2col(input, kH, kW, sY, sX, pY, pX, dh, dw, false, out2p);
        assertEquals(expected, out2p);

        INDArray out3 = Nd4j.create(new int[] {miniBatch, outH, outW, depth, kH, kW}, 'c');
        INDArray out3p = out3.permute(0, 3, 4, 5, 1, 2);
        Convolution.im2col(input, kH, kW, sY, sX, pY, pX, dh, dw, false, out3p);
        assertEquals(expected, out3p);
    }

    @Test
    public void testIm2ColKnownValuesMiniBatch3() {
        //Input: w=3, h=3, depth=2, minibatch = 3
        //kH=2, kW=2
        /*
        ----- Input images -----
        example 0:
        depth 0     depth 1
        [ 0  1  2      [ 9 10 11
          3  4  5       12 13 14
          6  7  8]      15 16 17]
        example 1:
        [18 19 20      [27 28 29
         21 22 23       30 31 32
         24 25 26]      33 34 35]
        example 2:
        [36 37 38      [45 46 47
         39 40 41       48 49 50
         42 43 44]      51 52 53]
        
        
         ----- Expected Output -----
         Shape: [miniBatch,depth,kH,kW,outH,outW]
         - example 0 -
         depth 0                        depth 1
         h0,w0      h0,w1               h0,w0      h0,w1
           0  1     1  2                 9 10      10 11
           3  4     4  5                12 13      13 14
        
         h1,w0      h1,w1               h1,w0      h1,w1
           3  4     4  5                12 13      13 14
           6  7     7  8                15 16      16 17
        
         - example 1 -
         depth 0                        depth 1
         h0,w0      h0,w1               h0,w0      h0,w1
          18 19     19 20               27 28      28 29
          21 22     22 23               30 31      31 32
        
         h1,w0      h1,w1               h1,w0      h1,w1
          21 22     22 23               30 31      31 32
          24 25     25 26               33 34      34 35
        
         - example 2 -
         depth 0                        depth 1
         h0,w0      h0,w1               h0,w0      h0,w1
          36 37     37 38               45 46      46 47
          39 40     40 41               48 49      49 50
        
         h1,w0      h1,w1               h1,w0      h1,w1
          39 40     40 41               48 49      49 50
          42 43     43 44               51 52      52 53
         */

        int miniBatch = 3;
        int depth = 2;
        int height = 3;
        int width = 3;

        int outH = 2;
        int outW = 2;
        int kH = 2;
        int kW = 2;
        int sX = 1;
        int sY = 1;
        int pX = 0;
        int pY = 0;

        //Input data: shape [miniBatch,depth,height,width]
        INDArray input = Nd4j.create(new int[] {miniBatch, depth, height, width}, 'c');
        input.put(new INDArrayIndex[] {point(0), point(0), all(),
                        all()}, Nd4j.create(new double[][] {{0, 1, 2}, {3, 4, 5}, {6, 7, 8}}));
        input.put(new INDArrayIndex[] {point(0), point(1), all(),
                        all()}, Nd4j.create(new double[][] {{9, 10, 11}, {12, 13, 14}, {15, 16, 17}}));
        input.put(new INDArrayIndex[] {point(1), point(0), all(),
                        all()}, Nd4j.create(new double[][] {{18, 19, 20}, {21, 22, 23}, {24, 25, 26}}));
        input.put(new INDArrayIndex[] {point(1), point(1), all(),
                        all()}, Nd4j.create(new double[][] {{27, 28, 29}, {30, 31, 32}, {33, 34, 35}}));
        input.put(new INDArrayIndex[] {point(2), point(0), all(),
                        all()}, Nd4j.create(new double[][] {{36, 37, 38}, {39, 40, 41}, {42, 43, 44}}));
        input.put(new INDArrayIndex[] {point(2), point(1), all(),
                        all()}, Nd4j.create(new double[][] {{45, 46, 47}, {48, 49, 50}, {51, 52, 53}}));

        //Expected data:
        INDArray expected = Nd4j.create(new int[] {miniBatch, depth, kH, kW, outH, outW}, 'c');

        //Example 0
        //depth 0
        expected.put(new INDArrayIndex[] {point(0), point(0), all(),
                        all(), point(0), point(0)},
                        Nd4j.create(new double[][] {{0, 1}, {3, 4}}));
        expected.put(new INDArrayIndex[] {point(0), point(0), all(),
                        all(), point(0), point(1)},
                        Nd4j.create(new double[][] {{1, 2}, {4, 5}}));
        expected.put(new INDArrayIndex[] {point(0), point(0), all(),
                        all(), point(1), point(0)},
                        Nd4j.create(new double[][] {{3, 4}, {6, 7}}));
        expected.put(new INDArrayIndex[] {point(0), point(0), all(),
                        all(), point(1), point(1)},
                        Nd4j.create(new double[][] {{4, 5}, {7, 8}}));
        //depth 1
        expected.put(new INDArrayIndex[] {point(0), point(1), all(),
                        all(), point(0), point(0)},
                        Nd4j.create(new double[][] {{9, 10}, {12, 13}}));
        expected.put(new INDArrayIndex[] {point(0), point(1), all(),
                        all(), point(0), point(1)},
                        Nd4j.create(new double[][] {{10, 11}, {13, 14}}));
        expected.put(new INDArrayIndex[] {point(0), point(1), all(),
                        all(), point(1), point(0)},
                        Nd4j.create(new double[][] {{12, 13}, {15, 16}}));
        expected.put(new INDArrayIndex[] {point(0), point(1), all(),
                        all(), point(1), point(1)},
                        Nd4j.create(new double[][] {{13, 14}, {16, 17}}));

        //Example 1
        //depth 0
        expected.put(new INDArrayIndex[] {point(1), point(0), all(),
                        all(), point(0), point(0)},
                        Nd4j.create(new double[][] {{18, 19}, {21, 22}}));
        expected.put(new INDArrayIndex[] {point(1), point(0), all(),
                        all(), point(0), point(1)},
                        Nd4j.create(new double[][] {{19, 20}, {22, 23}}));
        expected.put(new INDArrayIndex[] {point(1), point(0), all(),
                        all(), point(1), point(0)},
                        Nd4j.create(new double[][] {{21, 22}, {24, 25}}));
        expected.put(new INDArrayIndex[] {point(1), point(0), all(),
                        all(), point(1), point(1)},
                        Nd4j.create(new double[][] {{22, 23}, {25, 26}}));
        //depth 1
        expected.put(new INDArrayIndex[] {point(1), point(1), all(),
                        all(), point(0), point(0)},
                        Nd4j.create(new double[][] {{27, 28}, {30, 31}}));
        expected.put(new INDArrayIndex[] {point(1), point(1), all(),
                        all(), point(0), point(1)},
                        Nd4j.create(new double[][] {{28, 29}, {31, 32}}));
        expected.put(new INDArrayIndex[] {point(1), point(1), all(),
                        all(), point(1), point(0)},
                        Nd4j.create(new double[][] {{30, 31}, {33, 34}}));
        expected.put(new INDArrayIndex[] {point(1), point(1), all(),
                        all(), point(1), point(1)},
                        Nd4j.create(new double[][] {{31, 32}, {34, 35}}));

        //Example 2
        //depth 0
        expected.put(new INDArrayIndex[] {point(2), point(0), all(),
                        all(), point(0), point(0)},
                        Nd4j.create(new double[][] {{36, 37}, {39, 40}}));
        expected.put(new INDArrayIndex[] {point(2), point(0), all(),
                        all(), point(0), point(1)},
                        Nd4j.create(new double[][] {{37, 38}, {40, 41}}));
        expected.put(new INDArrayIndex[] {point(2), point(0), all(),
                        all(), point(1), point(0)},
                        Nd4j.create(new double[][] {{39, 40}, {42, 43}}));
        expected.put(new INDArrayIndex[] {point(2), point(0), all(),
                        all(), point(1), point(1)},
                        Nd4j.create(new double[][] {{40, 41}, {43, 44}}));
        //depth 1
        expected.put(new INDArrayIndex[] {point(2), point(1), all(),
                        all(), point(0), point(0)},
                        Nd4j.create(new double[][] {{45, 46}, {48, 49}}));
        expected.put(new INDArrayIndex[] {point(2), point(1), all(),
                        all(), point(0), point(1)},
                        Nd4j.create(new double[][] {{46, 47}, {49, 50}}));
        expected.put(new INDArrayIndex[] {point(2), point(1), all(),
                        all(), point(1), point(0)},
                        Nd4j.create(new double[][] {{48, 49}, {51, 52}}));
        expected.put(new INDArrayIndex[] {point(2), point(1), all(),
                        all(), point(1), point(1)},
                        Nd4j.create(new double[][] {{49, 50}, {52, 53}}));

        INDArray out = Convolution.im2col(input, kH, kW, sY, sX, pY, pX, false);
        assertEquals(expected, out);

        //Now: test with a provided results array, where the results array has weird strides
        INDArray out2 = Nd4j.create(new int[] {miniBatch, depth, outH, outW, kH, kW}, 'c');
        INDArray out2p = out2.permute(0, 1, 4, 5, 2, 3);
        Convolution.im2col(input, kH, kW, sY, sX, pY, pX, false, out2p);
        assertEquals(expected, out2p);

        INDArray out3 = Nd4j.create(new int[] {miniBatch, outH, outW, depth, kH, kW}, 'c');
        INDArray out3p = out3.permute(0, 3, 4, 5, 1, 2);
        Convolution.im2col(input, kH, kW, sY, sX, pY, pX, false, out3p);
        assertEquals(expected, out3p);
    }


    @Test
    public void testIm2ColSamePadding() {
        //Input: w=3, h=3, depth=2, minibatch = 2, kH/kW = 2, stride=1

        //Idea with same padding:
        //outH = ceil(inH / strideH)
        //outW = ceil(inW / strideW)

        int miniBatch = 2;
        int depth = 2;
        int inH = 3;
        int inW = 3;
        int strideH = 1;
        int strideW = 1;

        int kH = 2;
        int kW = 2;

        int outH = (int) Math.ceil(inH / ((double) strideH));
        int outW = (int) Math.ceil(inW / ((double) strideW));

        assertEquals(outH, inH);
        assertEquals(outW, inW);

        int sumPadHeight = ((outH - 1) * strideH + kH - inH);
        int padTop = sumPadHeight / 2;
        int padBottom = sumPadHeight - padTop;

        int sumPadWidth = ((outW - 1) * strideW + kW - inW);
        int padLeft = sumPadWidth / 2;
        int padRight = sumPadWidth - padLeft;

        System.out.println("Output size: " + outH + ", " + outW);
        System.out.println("Pad top/bottom: " + padTop + "\t" + padBottom);
        System.out.println("Pad left/right: " + padLeft + "\t" + padRight);


        /*
        ----- Input images -----
        example 0:
        depth 0     depth 1
        [ 0  1  2      [ 9 10 11
          3  4  5       12 13 14
          6  7  8]      15 16 17]
        example 1:
        [18 19 20      [27 28 29
         21 22 23       30 31 32
         24 25 26]      33 34 35]
        
         ----- Expected Output -----
         Shape: [miniBatch,depth,kH,kW,outH,outW]
         - example 0 -
         depth 0                        depth 1
          h0,w0    h0,w1    h0,w2        h0,w0    h0,w1    h0,w2
           0  1     1  2     2  0         9 10    10 11    11  0
           3  4     4  5     5  0        12 13    13 14    14  0
        
          h1,w0    h1,w1    h1,w2        h1,w0    h1,w1    h1,w2
           3  4     4  5     5  0        12 13    13 14    14  0
           6  7     7  8     8  0        15 16    16 17    17  0
        
          h2,w0    h2,w1    h2,w2        h2,w0    h2,w1    h2,w2
           6  7     7  8     8  0        15 16    16 17    17  0
           0  0     0  0     0  0         0  0     0  0     0  0
        
         - example 1 -
         depth 0                        depth 1
         h0,w0     h0,w1    h0,w2        h0,w0    h0,w1    h0,w2
          18 19    19 20    20  0        27 28    28 29    29  0
          21 22    22 23    23  0        30 31    31 32    32  0
        
         h1,w0     h1,w1    h1,w2        h1,w0    h1,w1    h1,w2
          21 22    22 23    23  0        30 31    31 32    32  0
          24 25    25 26    26  0        33 34    34 35    35  0
        
         h2,w0     h2,w1    h2,w2        h2,w0    h2,w1    h2,w2
          24 25    25 26    26  0        33 34    34 35    35  0
           0  0     0  0     0  0         0  0     0  0     0  0
         */

        //Input data: shape [miniBatch,depth,height,width]
        INDArray input = Nd4j.create(new int[] {miniBatch, depth, inH, inW}, 'c');
        input.put(new INDArrayIndex[] {point(0), point(0), all(),
                        all()}, Nd4j.create(new double[][] {{0, 1, 2}, {3, 4, 5}, {6, 7, 8}}));
        input.put(new INDArrayIndex[] {point(0), point(1), all(),
                        all()}, Nd4j.create(new double[][] {{9, 10, 11}, {12, 13, 14}, {15, 16, 17}}));
        input.put(new INDArrayIndex[] {point(1), point(0), all(),
                        all()}, Nd4j.create(new double[][] {{18, 19, 20}, {21, 22, 23}, {24, 25, 26}}));
        input.put(new INDArrayIndex[] {point(1), point(1), all(),
                        all()}, Nd4j.create(new double[][] {{27, 28, 29}, {30, 31, 32}, {33, 34, 35}}));

        //Expected data:
        INDArray expected = Nd4j.create(new int[] {miniBatch, depth, kH, kW, outH, outW}, 'c');

        //Example 0
        //depth 0
        expected.put(new INDArrayIndex[] {point(0), point(0), all(),
                        all(), point(0), point(0)},
                        Nd4j.create(new double[][] {{0, 1}, {3, 4}}));
        expected.put(new INDArrayIndex[] {point(0), point(0), all(),
                        all(), point(0), point(1)},
                        Nd4j.create(new double[][] {{1, 2}, {4, 5}}));
        expected.put(new INDArrayIndex[] {point(0), point(0), all(),
                        all(), point(0), point(2)},
                        Nd4j.create(new double[][] {{2, 0}, {5, 0}}));
        expected.put(new INDArrayIndex[] {point(0), point(0), all(),
                        all(), point(1), point(0)},
                        Nd4j.create(new double[][] {{3, 4}, {6, 7}}));
        expected.put(new INDArrayIndex[] {point(0), point(0), all(),
                        all(), point(1), point(1)},
                        Nd4j.create(new double[][] {{4, 5}, {7, 8}}));
        expected.put(new INDArrayIndex[] {point(0), point(0), all(),
                        all(), point(1), point(2)},
                        Nd4j.create(new double[][] {{5, 0}, {8, 0}}));
        expected.put(new INDArrayIndex[] {point(0), point(0), all(),
                        all(), point(2), point(0)},
                        Nd4j.create(new double[][] {{6, 7}, {0, 0}}));
        expected.put(new INDArrayIndex[] {point(0), point(0), all(),
                        all(), point(2), point(1)},
                        Nd4j.create(new double[][] {{7, 8}, {0, 0}}));
        expected.put(new INDArrayIndex[] {point(0), point(0), all(),
                        all(), point(2), point(2)},
                        Nd4j.create(new double[][] {{8, 0}, {0, 0}}));
        //depth 1
        expected.put(new INDArrayIndex[] {point(0), point(1), all(),
                        all(), point(0), point(0)},
                        Nd4j.create(new double[][] {{9, 10}, {12, 13}}));
        expected.put(new INDArrayIndex[] {point(0), point(1), all(),
                        all(), point(0), point(1)},
                        Nd4j.create(new double[][] {{10, 11}, {13, 14}}));
        expected.put(new INDArrayIndex[] {point(0), point(1), all(),
                        all(), point(0), point(2)},
                        Nd4j.create(new double[][] {{11, 0}, {14, 0}}));
        expected.put(new INDArrayIndex[] {point(0), point(1), all(),
                        all(), point(1), point(0)},
                        Nd4j.create(new double[][] {{12, 13}, {15, 16}}));
        expected.put(new INDArrayIndex[] {point(0), point(1), all(),
                        all(), point(1), point(1)},
                        Nd4j.create(new double[][] {{13, 14}, {16, 17}}));
        expected.put(new INDArrayIndex[] {point(0), point(1), all(),
                        all(), point(1), point(2)},
                        Nd4j.create(new double[][] {{14, 0}, {17, 0}}));
        expected.put(new INDArrayIndex[] {point(0), point(1), all(),
                        all(), point(2), point(0)},
                        Nd4j.create(new double[][] {{15, 16}, {0, 0}}));
        expected.put(new INDArrayIndex[] {point(0), point(1), all(),
                        all(), point(2), point(1)},
                        Nd4j.create(new double[][] {{16, 17}, {0, 0}}));
        expected.put(new INDArrayIndex[] {point(0), point(1), all(),
                        all(), point(2), point(2)},
                        Nd4j.create(new double[][] {{17, 0}, {0, 0}}));

        //Example 1
        //depth 0
        expected.put(new INDArrayIndex[] {point(1), point(0), all(),
                        all(), point(0), point(0)},
                        Nd4j.create(new double[][] {{18, 19}, {21, 22}}));
        expected.put(new INDArrayIndex[] {point(1), point(0), all(),
                        all(), point(0), point(1)},
                        Nd4j.create(new double[][] {{19, 20}, {22, 23}}));
        expected.put(new INDArrayIndex[] {point(1), point(0), all(),
                        all(), point(0), point(2)},
                        Nd4j.create(new double[][] {{20, 0}, {23, 0}}));
        expected.put(new INDArrayIndex[] {point(1), point(0), all(),
                        all(), point(1), point(0)},
                        Nd4j.create(new double[][] {{21, 22}, {24, 25}}));
        expected.put(new INDArrayIndex[] {point(1), point(0), all(),
                        all(), point(1), point(1)},
                        Nd4j.create(new double[][] {{22, 23}, {25, 26}}));
        expected.put(new INDArrayIndex[] {point(1), point(0), all(),
                        all(), point(1), point(2)},
                        Nd4j.create(new double[][] {{23, 0}, {26, 0}}));
        expected.put(new INDArrayIndex[] {point(1), point(0), all(),
                        all(), point(2), point(0)},
                        Nd4j.create(new double[][] {{24, 25}, {0, 0}}));
        expected.put(new INDArrayIndex[] {point(1), point(0), all(),
                        all(), point(2), point(1)},
                        Nd4j.create(new double[][] {{25, 26}, {0, 0}}));
        expected.put(new INDArrayIndex[] {point(1), point(0), all(),
                        all(), point(2), point(2)},
                        Nd4j.create(new double[][] {{26, 0}, {0, 0}}));

        //depth 1
        expected.put(new INDArrayIndex[] {point(1), point(1), all(),
                        all(), point(0), point(0)},
                        Nd4j.create(new double[][] {{27, 28}, {30, 31}}));
        expected.put(new INDArrayIndex[] {point(1), point(1), all(),
                        all(), point(0), point(1)},
                        Nd4j.create(new double[][] {{28, 29}, {31, 32}}));
        expected.put(new INDArrayIndex[] {point(1), point(1), all(),
                        all(), point(0), point(2)},
                        Nd4j.create(new double[][] {{29, 0}, {32, 0}}));
        expected.put(new INDArrayIndex[] {point(1), point(1), all(),
                        all(), point(1), point(0)},
                        Nd4j.create(new double[][] {{30, 31}, {33, 34}}));
        expected.put(new INDArrayIndex[] {point(1), point(1), all(),
                        all(), point(1), point(1)},
                        Nd4j.create(new double[][] {{31, 32}, {34, 35}}));
        expected.put(new INDArrayIndex[] {point(1), point(1), all(),
                        all(), point(1), point(2)},
                        Nd4j.create(new double[][] {{32, 0}, {35, 0}}));
        expected.put(new INDArrayIndex[] {point(1), point(1), all(),
                        all(), point(2), point(0)},
                        Nd4j.create(new double[][] {{33, 34}, {0, 0}}));
        expected.put(new INDArrayIndex[] {point(1), point(1), all(),
                        all(), point(2), point(1)},
                        Nd4j.create(new double[][] {{34, 35}, {0, 0}}));
        expected.put(new INDArrayIndex[] {point(1), point(1), all(),
                        all(), point(2), point(2)},
                        Nd4j.create(new double[][] {{35, 0}, {0, 0}}));

        //[miniBatch,depth,kH,kW,outH,outW]
        INDArray outAlloc = Nd4j.create(miniBatch, depth, kH, kW, outH, outW);
        INDArray out = Convolution.im2col(input, kH, kW, strideH, strideW, padTop, padLeft, true, outAlloc);
        //        System.out.println("Output shape: " + Arrays.toString(out.shape()));
        //
        //        for( int mb = 0; mb<2; mb++ ){
        //            for( int d = 0; d<2; d++ ){
        //                for( int h=0; h<3; h++ ){
        //                    for( int w=0; w<3; w++ ){
        //                        INDArrayIndex[] indx = new INDArrayIndex[]{NDArrayIndex.point(mb),NDArrayIndex.point(d),NDArrayIndex.all(),NDArrayIndex.all(), NDArrayIndex.point(h), NDArrayIndex.point(w)};
        //                        INDArray e = expected.get(indx);
        //                        INDArray a = out.get(indx);
        //
        //                        System.out.println("minibatch = " + mb + ", depth = " + depth + ", outY = " + h + ", outX = " + w + "\t" + (e.equals(a) ? "ok" : "FAILED"));
        //                        System.out.println(e);
        //                        System.out.println(a);
        //                        System.out.println("\n-------------------------");
        //                    }
        //                }
        //
        //            }
        //        }


        assertEquals(expected, out);

        //Now: test with a provided results array, where the results array has weird strides
        INDArray out2 = Nd4j.create(new int[] {miniBatch, depth, outH, outW, kH, kW}, 'c');
        INDArray out2p = out2.permute(0, 1, 4, 5, 2, 3);
        Convolution.im2col(input, kH, kW, strideH, strideW, padTop, padLeft, true, out2p);
        assertEquals(expected, out2p);

        INDArray out3 = Nd4j.create(new int[] {miniBatch, outH, outW, depth, kH, kW}, 'c');
        INDArray out3p = out3.permute(0, 3, 4, 5, 1, 2);
        Convolution.im2col(input, kH, kW, strideH, strideW, padTop, padLeft, true, out3p);
        assertEquals(expected, out3p);



        ///////////
        //Finally: Check col2im with the same shapes. This doesn't check the results, more 'does it crash or not'

        INDArray col2imResult = Nd4j.create(input.shape());
        INDArray col2im = Convolution.col2im(out, col2imResult, strideH, strideW, padTop, padLeft, inH, inW, 1, 1);
        System.out.println(Arrays.toString(col2im.data().asDouble()));
    }


    @Test
    public void testIm2ColSamePaddingStride2() {
        //Input: h=3, w=4, depth=2, minibatch = 1, kH/kW = 3, stride=2

        //Idea with same padding:
        //outH = ceil(inH / strideH)
        //outW = ceil(inW / strideW)

        int miniBatch = 1;
        int depth = 2;
        int inH = 3;
        int inW = 4;
        int strideH = 2;
        int strideW = 2;

        int kH = 3;
        int kW = 3;

        int outH = (int) Math.ceil(inH / ((double) strideH));
        int outW = (int) Math.ceil(inW / ((double) strideW));

        assertEquals(2, outH); //ceil(3/2) = 2
        assertEquals(2, outW); //ceil(4/2) = 2

        int sumPadHeight = ((outH - 1) * strideH + kH - inH);
        int padTop = sumPadHeight / 2;
        int padBottom = sumPadHeight - padTop;

        assertEquals(1, padTop);
        assertEquals(1, padBottom);

        int sumPadWidth = ((outW - 1) * strideW + kW - inW);
        int padLeft = sumPadWidth / 2;
        int padRight = sumPadWidth - padLeft;

        assertEquals(0, padLeft);
        assertEquals(1, padRight);

        System.out.println("Output size: " + outH + ", " + outW);
        System.out.println("Pad top/bottom: " + padTop + "\t" + padBottom);
        System.out.println("Pad left/right: " + padLeft + "\t" + padRight);


        /*
        ----- Input images -----
        example 0:
        depth 0       depth 1
        [ 0  1  2  3      [12 13 14 15
          4  5  6  7       16 17 18 19
          8  9 10 11]      20 21 22 23]
        
         ----- Expected Output -----
         Shape: [miniBatch,depth,kH,kW,outH,outW]
         - example 0 -
         depth 0                        depth 1
          h0,w0        h0,w1            h0,w0       h0,w1
           0  0  0     0  0  0           0  0  0    0  0  0
           0  1  2     2  3  0          12 13 14   14 15  0
           4  5  6     6  7  0          16 17 18   18 19  0
        
          h1,w0
           4  5  6     6  7  0          16 17 18   18 19  0
           8  9 10    10 11  0          20 21 22   22 23  0
           0  0  0     0  0  0           0  0  0    0  0  0
         */

        //Input data: shape [miniBatch,depth,height,width]
        INDArray input = Nd4j.create(new int[] {miniBatch, depth, inH, inW}, 'c');
        input.put(new INDArrayIndex[] {point(0), point(0), all(),
                        all()}, Nd4j.create(new double[][] {{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}}));
        input.put(new INDArrayIndex[] {point(0), point(1), all(),
                        all()},
                        Nd4j.create(new double[][] {{12, 13, 14, 15}, {16, 17, 18, 19}, {20, 21, 22, 23}}));

        //Expected data:
        INDArray expected = Nd4j.create(new int[] {miniBatch, depth, kH, kW, outH, outW}, 'c');

        //Example 0
        //depth 0
        expected.put(new INDArrayIndex[] {point(0), point(0), all(),
                        all(), point(0), point(0)},
                        Nd4j.create(new double[][] {{0, 0, 0}, {0, 1, 2}, {4, 5, 6}}));
        expected.put(new INDArrayIndex[] {point(0), point(0), all(),
                        all(), point(0), point(1)},
                        Nd4j.create(new double[][] {{0, 0, 0}, {2, 3, 0}, {6, 7, 0}}));
        expected.put(new INDArrayIndex[] {point(0), point(0), all(),
                        all(), point(1), point(0)},
                        Nd4j.create(new double[][] {{4, 5, 6}, {8, 9, 10}, {0, 0, 0}}));
        expected.put(new INDArrayIndex[] {point(0), point(0), all(),
                        all(), point(1), point(1)},
                        Nd4j.create(new double[][] {{6, 7, 0}, {10, 11, 0}, {0, 0, 0}}));
        //depth 1
        expected.put(new INDArrayIndex[] {point(0), point(1), all(),
                        all(), point(0), point(0)},
                        Nd4j.create(new double[][] {{0, 0, 0}, {12, 13, 14}, {16, 17, 18}}));
        expected.put(new INDArrayIndex[] {point(0), point(1), all(),
                        all(), point(0), point(1)},
                        Nd4j.create(new double[][] {{0, 0, 0}, {14, 15, 0}, {18, 19, 0}}));
        expected.put(new INDArrayIndex[] {point(0), point(1), all(),
                        all(), point(1), point(0)},
                        Nd4j.create(new double[][] {{16, 17, 18}, {20, 21, 22}, {0, 0, 0}}));
        expected.put(new INDArrayIndex[] {point(0), point(1), all(),
                        all(), point(1), point(1)},
                        Nd4j.create(new double[][] {{18, 19, 0}, {22, 23, 0}, {0, 0, 0}}));

        //[miniBatch,depth,kH,kW,outH,outW]
        INDArray outAlloc = Nd4j.create(miniBatch, depth, kH, kW, outH, outW);
        INDArray out = Convolution.im2col(input, kH, kW, strideH, strideW, padTop, padLeft, true, outAlloc);
        //        System.out.println("Output shape: " + Arrays.toString(out.shape()));
        //
        //        for( int mb = 0; mb<2; mb++ ){
        //            for( int d = 0; d<2; d++ ){
        //                for( int h=0; h<3; h++ ){
        //                    for( int w=0; w<3; w++ ){
        //                        INDArrayIndex[] indx = new INDArrayIndex[]{NDArrayIndex.point(mb),NDArrayIndex.point(d),NDArrayIndex.all(),NDArrayIndex.all(), NDArrayIndex.point(h), NDArrayIndex.point(w)};
        //                        INDArray e = expected.get(indx);
        //                        INDArray a = out.get(indx);
        //
        //                        System.out.println("minibatch = " + mb + ", depth = " + depth + ", outY = " + h + ", outX = " + w + "\t" + (e.equals(a) ? "ok" : "FAILED"));
        //                        System.out.println(e);
        //                        System.out.println(a);
        //                        System.out.println("\n-------------------------");
        //                    }
        //                }
        //
        //            }
        //        }


        assertEquals(expected, out);

        //Now: test with a provided results array, where the results array has weird strides
        INDArray out2 = Nd4j.create(new int[] {miniBatch, depth, outH, outW, kH, kW}, 'c');
        INDArray out2p = out2.permute(0, 1, 4, 5, 2, 3);
        Convolution.im2col(input, kH, kW, strideH, strideW, padTop, padLeft, true, out2p);
        assertEquals(expected, out2p);

        INDArray out3 = Nd4j.create(new int[] {miniBatch, outH, outW, depth, kH, kW}, 'c');
        INDArray out3p = out3.permute(0, 3, 4, 5, 1, 2);
        Convolution.im2col(input, kH, kW, strideH, strideW, padTop, padLeft, true, out3p);
        assertEquals(expected, out3p);


        ///////////
        //Finally: Check col2im with the same shapes. This doesn't check the results, more 'does it crash or not'

        INDArray col2imResult = Nd4j.create(input.shape());
        INDArray col2im = Convolution.col2im(out, col2imResult, strideH, strideW, padTop, padLeft, inH, inW, 1, 1);
        System.out.println(Arrays.toString(col2im.data().asDouble()));
    }



    @Test
    public void testCol2ImSamePaddingStride2() {
        //Input: h=3, w=4, depth=2, minibatch = 1, kH/kW = 3, stride=2

        //Idea with same padding:
        //outH = ceil(inH / strideH)
        //outW = ceil(inW / strideW)

        int miniBatch = 1;
        int depth = 2;
        int inH = 3;
        int inW = 4;
        int strideH = 2;
        int strideW = 2;

        int kH = 3;
        int kW = 3;

        int outH = (int) Math.ceil(inH / ((double) strideH));
        int outW = (int) Math.ceil(inW / ((double) strideW));

        assertEquals(2, outH); //ceil(3/2) = 2
        assertEquals(2, outW); //ceil(4/2) = 2

        int sumPadHeight = ((outH - 1) * strideH + kH - inH);
        int padTop = sumPadHeight / 2;
        int padBottom = sumPadHeight - padTop;

        assertEquals(1, padTop);
        assertEquals(1, padBottom);

        int sumPadWidth = ((outW - 1) * strideW + kW - inW);
        int padLeft = sumPadWidth / 2;
        int padRight = sumPadWidth - padLeft;

        assertEquals(0, padLeft);
        assertEquals(1, padRight);

//        System.out.println("Output size: " + outH + ", " + outW);
//        System.out.println("Pad top/bottom: " + padTop + "\t" + padBottom);
//        System.out.println("Pad left/right: " + padLeft + "\t" + padRight);


        /*
        ----- Input images -----
        example 0:
        depth 0       depth 1
        [ 0  1  2  3      [12 13 14 15
          4  5  6  7       16 17 18 19
          8  9 10 11]      20 21 22 23]
        
         ----- Expected Output -----
         Shape: [miniBatch,depth,kH,kW,outH,outW]
         - example 0 -
         depth 0                        depth 1
          h0,w0        h0,w1            h0,w0       h0,w1
           0  0  0     0  0  0           0  0  0    0  0  0
           0  1  2     2  3  0          12 13 14   14 15  0
           4  5  6     6  7  0          16 17 18   18 19  0
        
          h1,w0
           4  5  6     6  7  0          16 17 18   18 19  0
           8  9 10    10 11  0          20 21 22   22 23  0
           0  0  0     0  0  0           0  0  0    0  0  0
         */

        /*
        Col2im result:
        
        example 0:
        depth 0           depth 1
        [ 0  1  4  3      [12 13 28 15
          8 10 24 14       32 34 72 38
          8  9 20 11]      20 21 44 23]
         */

        //Input data: shape [miniBatch,depth,height,width]
        //        INDArray input = Nd4j.create(new int[]{miniBatch,depth,inH,inW},'c');
        //        input.put(new INDArrayIndex[]{NDArrayIndex.point(0), NDArrayIndex.point(0),NDArrayIndex.all(), NDArrayIndex.all()}, Nd4j.create(new double[][]{{0,1,2,3},{4,5,6,7},{8,9,10,11}}));
        //        input.put(new INDArrayIndex[]{NDArrayIndex.point(0), NDArrayIndex.point(1),NDArrayIndex.all(), NDArrayIndex.all()}, Nd4j.create(new double[][]{{12,13,14,15},{16,17,18,19},{20,21,22,23}}));

        INDArray col6d = Nd4j.create(new int[] {miniBatch, depth, kH, kW, outH, outW}, 'c');

        //Example 0
        //depth 0
        col6d.put(new INDArrayIndex[] {point(0), point(0), all(),
                        all(), point(0), point(0)},
                        Nd4j.create(new double[][] {{0, 0, 0}, {0, 1, 2}, {4, 5, 6}}));
        col6d.put(new INDArrayIndex[] {point(0), point(0), all(),
                        all(), point(0), point(1)},
                        Nd4j.create(new double[][] {{0, 0, 0}, {2, 3, 0}, {6, 7, 0}}));
        col6d.put(new INDArrayIndex[] {point(0), point(0), all(),
                        all(), point(1), point(0)},
                        Nd4j.create(new double[][] {{4, 5, 6}, {8, 9, 10}, {0, 0, 0}}));
        col6d.put(new INDArrayIndex[] {point(0), point(0), all(),
                        all(), point(1), point(1)},
                        Nd4j.create(new double[][] {{6, 7, 0}, {10, 11, 0}, {0, 0, 0}}));
        //depth 1
        col6d.put(new INDArrayIndex[] {point(0), point(1), all(),
                        all(), point(0), point(0)},
                        Nd4j.create(new double[][] {{0, 0, 0}, {12, 13, 14}, {16, 17, 18}}));
        col6d.put(new INDArrayIndex[] {point(0), point(1), all(),
                        all(), point(0), point(1)},
                        Nd4j.create(new double[][] {{0, 0, 0}, {14, 15, 0}, {18, 19, 0}}));
        col6d.put(new INDArrayIndex[] {point(0), point(1), all(),
                        all(), point(1), point(0)},
                        Nd4j.create(new double[][] {{16, 17, 18}, {20, 21, 22}, {0, 0, 0}}));
        col6d.put(new INDArrayIndex[] {point(0), point(1), all(),
                        all(), point(1), point(1)},
                        Nd4j.create(new double[][] {{18, 19, 0}, {22, 23, 0}, {0, 0, 0}}));


        //Expected result:
        INDArray expected = Nd4j.create(miniBatch, depth, inH, inW);
        expected.put(new INDArrayIndex[] {point(0), point(0), all(),
                        all()},
                        Nd4j.create(new double[][] {{0, 1, 4, 3}, {8, 10, 24, 14}, {8, 9, 20, 11}}));
        expected.put(new INDArrayIndex[] {point(0), point(1), all(),
                        all()},
                        Nd4j.create(new double[][] {{12, 13, 28, 15}, {32, 34, 72, 38}, {20, 21, 44, 23}}));


        INDArray col2imResult = Nd4j.create(miniBatch, depth, inH, inW);
        INDArray col2im = Convolution.col2im(col6d, col2imResult, strideH, strideW, padTop, padLeft, inH, inW, 1, 1);

        assertEquals(expected, col2im);
    }


    @Test
    public void testCol2ImSamePaddingStride1Dilation2() {
        //Input: h=4, w=5, depth=1, minibatch = 1, kH/kW = 2, stride=1, dilation 2

        //Idea with same padding:
        //outH = ceil(inH / strideH)
        //outW = ceil(inW / strideW)

        int miniBatch = 1;
        int depth = 1;
        int inH = 4;
        int inW = 5;
        int strideH = 1;
        int strideW = 1;
        int dH = 2;
        int dW = 2;

        int kH = 2;
        int kW = 2;

        int effectiveKH = kH + (kH-1)*(dH-1);
        int effectiveKW = kW + (kW-1)*(dW-1);

        int outH = (int) Math.ceil(inH / ((double) strideH));
        int outW = (int) Math.ceil(inW / ((double) strideW));

        assertEquals(5, outW); //ceil(5/1) = 5
        assertEquals(4, outH); //ceil(4/1) = 5

        int sumPadHeight = ((outH - 1) * strideH + effectiveKH - inH);
        int padTop = sumPadHeight / 2;
        int padBottom = sumPadHeight - padTop;

        assertEquals(1, padTop);
        assertEquals(1, padBottom);

        int sumPadWidth = ((outW - 1) * strideW + effectiveKW - inW);
        int padLeft = sumPadWidth / 2;
        int padRight = sumPadWidth - padLeft;

        assertEquals(1, padLeft);
        assertEquals(1, padRight);

//        System.out.println("Output size: " + outH + ", " + outW);
//        System.out.println("Pad top/bottom: " + padTop + "\t" + padBottom);
//        System.out.println("Pad left/right: " + padLeft + "\t" + padRight);


        /*
        ----- Input images -----
        example 0:
        depth 0
        [ 0  1  2  3  4
          5  6  7  8  9
         10 11 12 13 14
         15 16 17 18 19 ]

         Effective input, with padding:
        [ 0  0  0  0  0  0  0
          0  0  1  2  3  4  0
          0  5  6  7  8  9  0
          0 10 11 12 13 14  0
          0 15 16 17 18 19  0
          0  0  0  0  0  0  0]

         ----- Expected Output -----
         Shape: [miniBatch,depth,kH,kW,outH,outW]
         - example 0 -
         depth 0
          h0,w0     h0,w1    h0,w2    h0,w3    h0,w4
           0  0     0  0     0  0     0  0     0  0
           0  6     5  7     6  8     7  9     8  0

          h0,w0     h0,w1    h0,w2    h0,w3    h0,w4
           0  1     0  2     1  3     2  4     3  0
           0 11    10 12    11 13    12 14    13  0

          h0,w0     h0,w1    h0,w2    h0,w3    h0,w4
           0  6     5  7     6  8     7  9     8  0
           0 16    15 17    16 18    17 19    18  0

          h0,w0     h0,w1    h0,w2    h0,w3    h0,w4
           0 11    10 12    11 13    12 14    13  0
           0  0     0  0     0  0     0  0     0  0
         */

        /*
        Col2im result:

        example 0:
        depth 0
        [ 0  2  4  6  4
         10 24 28 32 18
         20 44 48 52 28
         15 32 34 36 19]
         */

        //Input data: shape [miniBatch,depth,height,width]
        INDArray input = Nd4j.create(new int[]{miniBatch,depth,inH,inW},'c');
        input.put(new INDArrayIndex[]{point(0), point(0),all(), all()}, Nd4j.create(new double[][]{{0,1,2,3,4},{5,6,7,8,9},{10,11,12,13,14},{15,16,17,18,19}}));

        INDArray col6d = Nd4j.create(new int[] {miniBatch, depth, kH, kW, outH, outW}, 'c');

        //Example 0
        //depth 0
        //Iterate over width, then height
        col6d.put(new INDArrayIndex[] {point(0), point(0), all(),all(), point(0), point(0)},
                Nd4j.create(new double[][] {{0, 0}, {0, 6}}));
        col6d.put(new INDArrayIndex[] {point(0), point(0), all(),all(), point(0), point(1)},
                Nd4j.create(new double[][] {{0, 0}, {5, 7}}));
        col6d.put(new INDArrayIndex[] {point(0), point(0), all(),all(), point(0), point(2)},
                Nd4j.create(new double[][] {{0, 0}, {6, 8}}));
        col6d.put(new INDArrayIndex[] {point(0), point(0), all(),all(), point(0), point(3)},
                Nd4j.create(new double[][] {{0, 0}, {7, 9}}));
        col6d.put(new INDArrayIndex[] {point(0), point(0), all(),all(), point(0), point(4)},
                Nd4j.create(new double[][] {{0, 0}, {8, 0}}));

        col6d.put(new INDArrayIndex[] {point(0), point(0), all(),all(), point(1), point(0)},
                Nd4j.create(new double[][] {{0, 1}, {0, 11}}));
        col6d.put(new INDArrayIndex[] {point(0), point(0), all(),all(), point(1), point(1)},
                Nd4j.create(new double[][] {{0, 2}, {10, 12}}));
        col6d.put(new INDArrayIndex[] {point(0), point(0), all(),all(), point(1), point(2)},
                Nd4j.create(new double[][] {{1, 3}, {11, 13}}));
        col6d.put(new INDArrayIndex[] {point(0), point(0), all(),all(), point(1), point(3)},
                Nd4j.create(new double[][] {{2, 4}, {12, 14}}));
        col6d.put(new INDArrayIndex[] {point(0), point(0), all(),all(), point(1), point(4)},
                Nd4j.create(new double[][] {{3, 0}, {13, 0}}));

        col6d.put(new INDArrayIndex[] {point(0), point(0), all(),all(), point(2), point(0)},
                Nd4j.create(new double[][] {{0, 6}, {0, 16}}));
        col6d.put(new INDArrayIndex[] {point(0), point(0), all(),all(), point(2), point(1)},
                Nd4j.create(new double[][] {{5, 7}, {15, 17}}));
        col6d.put(new INDArrayIndex[] {point(0), point(0), all(),all(), point(2), point(2)},
                Nd4j.create(new double[][] {{6, 8}, {16, 18}}));
        col6d.put(new INDArrayIndex[] {point(0), point(0), all(),all(), point(2), point(3)},
                Nd4j.create(new double[][] {{7, 9}, {17, 19}}));
        col6d.put(new INDArrayIndex[] {point(0), point(0), all(),all(), point(2), point(4)},
                Nd4j.create(new double[][] {{8, 0}, {18, 0}}));

        col6d.put(new INDArrayIndex[] {point(0), point(0), all(),all(), point(3), point(0)},
                Nd4j.create(new double[][] {{0, 11}, {0, 0}}));
        col6d.put(new INDArrayIndex[] {point(0), point(0), all(),all(), point(3), point(1)},
                Nd4j.create(new double[][] {{10, 12}, {0, 0}}));
        col6d.put(new INDArrayIndex[] {point(0), point(0), all(),all(), point(3), point(2)},
                Nd4j.create(new double[][] {{11, 13}, {0, 0}}));
        col6d.put(new INDArrayIndex[] {point(0), point(0), all(),all(), point(3), point(3)},
                Nd4j.create(new double[][] {{12, 14}, {0, 0}}));
        col6d.put(new INDArrayIndex[] {point(0), point(0), all(),all(), point(3), point(4)},
                Nd4j.create(new double[][] {{13, 0}, {0, 0}}));



        //Check im2col:
        INDArray im2col = Convolution.im2col(input, kH, kW, strideH, strideW, padTop, padLeft, dH, dW, true);


        for( int j=0; j<outH; j++ ){
            for(int i=0; i<outW; i++ ){
                INDArray exp = col6d.get(point(0), point(0), all(), all(), point(j), point(i));
                INDArray act = im2col.get(point(0), point(0), all(), all(), point(j), point(i));
                if(!exp.equals(act)) {
                    System.out.println(i + "\t" + j);
                    System.out.println(exp);
                    System.out.println();
                    System.out.println(act);
                    System.out.println("\n");
                }
            }
        }

        assertEquals(col6d, im2col);


        //Expected result:
        INDArray expected = Nd4j.create(miniBatch, depth, inH, inW);
        expected.put(new INDArrayIndex[] {point(0), point(0), all(), all()},
                Nd4j.create(new double[][] {{0, 2, 4, 6, 4}, {10, 24, 28, 32, 18}, {20, 44, 48, 52, 28}, {15, 32, 34, 36, 19}}));


        INDArray col2imResult = Nd4j.create(miniBatch, depth, inH, inW);
        INDArray col2im = Convolution.col2im(col6d, col2imResult, strideH, strideW, padTop, padLeft, inH, inW, dH, dW);

        assertEquals(expected, col2im);
    }



    @Test
    public void testConvOutWidthAndHeight() {
        int outSize = Convolution.outSize(2, 1, 1, 2, 1, false);
        assertEquals(6, outSize);
    }

    @Test
    public void testIm2Col() {
        INDArray linspaced = Nd4j.linspace(1, 16, 16).reshape(2, 2, 2, 2);
        INDArray ret = Convolution.im2col(linspaced, 1, 1, 1, 1, 2, 2, 0, false);
        System.out.println(ret);
    }



    @Test
    @Ignore
    public void testCompareIm2ColImpl() {

        int[] miniBatches = {1, 3, 5};
        int[] depths = {1, 3, 5};
        int[] inHeights = {5, 21};
        int[] inWidths = {5, 21};
        int[] strideH = {1, 2};
        int[] strideW = {1, 2};
        int[] sizeW = {1, 2, 3};
        int[] sizeH = {1, 2, 3};
        int[] padH = {0, 1, 2};
        int[] padW = {0, 1, 2};
        boolean[] coverall = {false, true};

        DataBuffer.Type[] types = new DataBuffer.Type[] {DataBuffer.Type.FLOAT, DataBuffer.Type.DOUBLE,
                        DataBuffer.Type.FLOAT, DataBuffer.Type.DOUBLE};
        DataBuffer.AllocationMode[] modes =
                        new DataBuffer.AllocationMode[] {DataBuffer.AllocationMode.HEAP, DataBuffer.AllocationMode.HEAP,
                                        DataBuffer.AllocationMode.DIRECT, DataBuffer.AllocationMode.DIRECT};

        String factoryClassName = Nd4j.factory().getClass().toString().toLowerCase();
        if (factoryClassName.contains("jcublas") || factoryClassName.contains("cuda")) {
            //Only test direct for CUDA; test all for CPU
            types = new DataBuffer.Type[] {DataBuffer.Type.FLOAT, DataBuffer.Type.DOUBLE};
            modes = new DataBuffer.AllocationMode[] {DataBuffer.AllocationMode.DIRECT,
                            DataBuffer.AllocationMode.DIRECT};
        }

        DataBuffer.Type initialType = Nd4j.dataType();
        for (int i = 0; i < types.length; i++) {
            DataBuffer.Type type = types[i];
            DataBuffer.AllocationMode mode = modes[i];

            DataTypeUtil.setDTypeForContext(type);
            Nd4j.alloc = mode;

            AllocUtil.setAllocationModeForContext(mode);

            for (int m : miniBatches) {
                for (int d : depths) {
                    for (int h : inHeights) {
                        for (int w : inWidths) {
                            for (int sh : strideH) {
                                for (int sw : strideW) {
                                    for (int kh : sizeH) {
                                        for (int kw : sizeW) {
                                            for (int ph : padH) {
                                                for (int pw : padW) {
                                                    if ((w - kw + 2 * pw) % sw != 0 || (h - kh + 2 * ph) % sh != 0)
                                                        continue; //(w-kp+2*pW)/sw + 1 is not an integer,  i.e., number of outputs doesn't fit

                                                    System.out.println("Running " + m + " " + d + " " + h + " " + w);
                                                    for (boolean cAll : coverall) {

                                                        INDArray in = Nd4j.rand(new int[] {m, d, h, w});
                                                        //assertEquals(in.data().allocationMode(), mode);
                                                        //assertEquals(in.data().dataType(), opType);

                                                        INDArray outOrig = OldConvolution.im2col(in, kh, kw, sh, sw, ph,
                                                                        pw, -1, cAll); //Old implementation
                                                        INDArray outNew = Convolution.im2col(in, kh, kw, sh, sw, ph, pw,
                                                                        cAll); //Current implementation

                                                        assertArrayEquals(outOrig.data().asFloat(),
                                                                        outNew.data().asFloat(), 0.01f);
                                                        assertEquals(outOrig, outNew);
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        DataTypeUtil.setDTypeForContext(initialType);
    }

    @Test
    @Ignore
    public void testCompareIm2Col() throws Exception {

        int[] miniBatches = {1, 3, 5};
        int[] depths = {1, 3, 5};
        int[] inHeights = {5, 21};
        int[] inWidths = {5, 21};
        int[] strideH = {1, 2};
        int[] strideW = {1, 2};
        int[] sizeW = {1, 2, 3};
        int[] sizeH = {1, 2, 3};
        int[] padH = {0, 1, 2};
        int[] padW = {0, 1, 2};

        DataBuffer.Type[] types = new DataBuffer.Type[] {DataBuffer.Type.FLOAT, DataBuffer.Type.DOUBLE,
                        DataBuffer.Type.FLOAT, DataBuffer.Type.DOUBLE};
        DataBuffer.AllocationMode[] modes =
                        new DataBuffer.AllocationMode[] {DataBuffer.AllocationMode.HEAP, DataBuffer.AllocationMode.HEAP,
                                        DataBuffer.AllocationMode.DIRECT, DataBuffer.AllocationMode.DIRECT};

        String factoryClassName = Nd4j.factory().getClass().toString().toLowerCase();
        if (factoryClassName.contains("jcublas") || factoryClassName.contains("cuda")) {
            //Only test direct for CUDA; test all for CPU
            types = new DataBuffer.Type[] {DataBuffer.Type.FLOAT, DataBuffer.Type.DOUBLE};
            modes = new DataBuffer.AllocationMode[] {DataBuffer.AllocationMode.DIRECT,
                            DataBuffer.AllocationMode.DIRECT};
        }

        DataBuffer.Type inititalType = Nd4j.dataType();
        for (int i = 0; i < types.length; i++) {
            DataBuffer.Type type = types[i];
            DataBuffer.AllocationMode mode = modes[i];

            DataTypeUtil.setDTypeForContext(type);
            Nd4j.alloc = mode;

            for (int m : miniBatches) {
                for (int d : depths) {
                    for (int h : inHeights) {
                        for (int w : inWidths) {
                            for (int sh : strideH) {
                                for (int sw : strideW) {
                                    for (int kh : sizeH) {
                                        for (int kw : sizeW) {
                                            for (int ph : padH) {
                                                for (int pw : padW) {
                                                    System.out.println("Before assertion");
                                                    if ((w - kw + 2 * pw) % sw != 0 || (h - kh + 2 * ph) % sh != 0)
                                                        continue; //(w-kp+2*pW)/sw + 1 is not an integer, i.e., number of outputs doesn't fit

                                                    INDArray in = Nd4j.rand(new int[] {m, d, h, w});
                                                    assertEquals(in.data().allocationMode(), mode);
                                                    assertEquals(in.data().dataType(), type);
                                                    INDArray im2col = Convolution.im2col(in, kh, kw, sh, sw, ph, pw,
                                                                    false); //Cheating, to get correct shape for input

                                                    INDArray imgOutOld =
                                                                    OldConvolution.col2im(im2col, sh, sw, ph, pw, h, w);
                                                    INDArray imgOutNew =
                                                                    Convolution.col2im(im2col, sh, sw, ph, pw, h, w);
                                                    System.out.println("F order test");
                                                    System.out.println(imgOutOld);
                                                    System.out.println(imgOutNew);
                                                    assertEquals(imgOutOld, imgOutNew);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        DataTypeUtil.setDTypeForContext(inititalType);
    }


    @Test
    public void testCol2Im() {
        int kh = 1;
        int kw = 1;
        int sy = 1;
        int sx = 1;
        int ph = 1;
        int pw = 1;
        INDArray linspaced = Nd4j.linspace(1, 64, 64).reshape(2, 2, 2, 2, 2, 2);
        INDArray newTest = Convolution.col2im(linspaced, sy, sx, ph, pw, 2, 2);
        INDArray assertion = OldConvolution.col2im(linspaced, sy, sx, ph, pw, 2, 2);

        System.out.println("Ordering of the result, new test: " + newTest.ordering());

        System.out.println("Assertion dimensions: " + Arrays.toString(assertion.shape()));
        assertEquals(assertion, newTest);
    }


    @Test
    public void testimcolim() {
        int nEx = 2;
        int depth = 3;
        int width = 7;
        int height = 7;
        int[] kernel = {3, 2};
        int[] stride = {2, 3};
        int[] padding = {1, 2};
        int prod = nEx * depth * width * height;

        INDArray in = Nd4j.linspace(1, prod, prod).reshape(nEx, depth, width, height);

        INDArray assertim2col = OldConvolution.im2col(in, kernel, stride, padding);
        INDArray im2col = Convolution.im2col(in, kernel, stride, padding);
        assertEquals(assertim2col, im2col);

        INDArray assertcol2im = OldConvolution.col2im(im2col, stride, padding, height, width);
        INDArray col2im = Convolution.col2im(im2col, stride, padding, height, width);
        assertEquals(assertcol2im, col2im);
    }

    @Test
    public void testIm2ColWithDilation() {
        int kH = 2;
        int kW = 2;
        int sH = 1;
        int sW = 1;
        int pH = 0;
        int pW = 0;
        int dH = 1;
        int dW = 2;
        boolean same = false;

        /*
        Input:
        [ 1,  2,  3
          4,  5,  6
          7,  8,  9 ]

        Im2col:
        [ 1,  3
          4,  6 ]

        [ 4,  6
          7,  9 ]
         */


        INDArray in = Nd4j.create(1, 1, 3, 3);
        in.get(point(0), point(0), all(), all()).assign(Nd4j.linspace(1, 9, 9).reshape('c', 3, 3));

        INDArray out = Nd4j.create(1, 1, 2, 2, 2, 1);    //minibatch, depth, kH, kW, outH, outW
        Convolution.im2col(in, kH, kW, sH, sW, pH, pW, dH, dW, same, out);

        INDArray act0 = out.get(point(0), point(0), all(), all(), point(0), point(0));
        INDArray act1 = out.get(point(0), point(0), all(), all(), point(1), point(0));

        INDArray exp0 = Nd4j.create(new double[][]{{1, 3}, {4, 6}});
        INDArray exp1 = Nd4j.create(new double[][]{{4, 6}, {7, 9}});

        assertEquals(exp0, act0);
        assertEquals(exp1, act1);
    }


    @Test
    public void testPoolingEdgeCases(){
        //Average pooling with same mode: should we include the padded values, when deciding what to divide by?
        ///*** Note: Mode 2 is the "DL4J always divide by kH*kW" approach ***

        /*
        Input:
        [ 1, 2, 3
          4, 5, 6
          7, 8, 9 ]


         Kernel 2, stride 1
         outH = 3, outW = 3 (i.e., ceil(in/stride)
         totalHPad = (outH-1) * strideH + kH - inH = (3-1)*1 + 2 - 3 = 1
         topPad = 0, bottomPad = 1
         leftPad = 0, rightPad = 1
         */

        for( char inputOrder : new char[]{'c', 'f'}) {
            for( char outputOrder : new char[]{'c', 'f'}) {

                INDArray input = Nd4j.create(1, 1, 3, 3);
                input.get(point(0), point(0), all(), all())
                        .assign(Nd4j.linspace(1, 9, 9).reshape('c', 3, 3))
                        .dup(inputOrder);

                input = input.dup('c');

                INDArray input2 = Nd4j.create(new double[]{1,2,3,4,5,6,7,8,9}, new int[]{1,1,3,3}, 'c');//.dup(inputOrder);
                assertEquals(input, input2);

                input = input2;

                for( int i=0; i<3; i++){
                    for( int j=0; j<3; j++ ){
                        System.out.print(input.getDouble(0,0,i,j) + ",");
                    }
                    System.out.println();
                }
                System.out.println();

                INDArray sums = Nd4j.create(new double[][]{
                        {(1 + 2 + 4 + 5), (2 + 3 + 5 + 6), (3 + 6)},
                        {(4 + 5 + 7 + 8), (5 + 6 + 8 + 9), (6 + 9)},
                        {(7 + 8), (8 + 9), (9)}
                });

                INDArray divEnabled = Nd4j.create(new double[][]{
                        {4, 4, 2},
                        {4, 4, 2},
                        {2, 2, 1}
                });

                INDArray expEnabled = sums.div(divEnabled);
                INDArray expDl4j = sums.div(4);

                //https://github.com/deeplearning4j/libnd4j/blob/master/include/ops/declarable/generic/convo/pooling/avgpool2d.cpp
                DynamicCustomOp op1 = DynamicCustomOp.builder("avgpool2d")
                        .addIntegerArguments(new int[]{2, 2, 1, 1, 0, 0, 1, 1, 1, 0, 0})   //ky, kx, sH, sW, py, px, dy, dx, isSameMode, ???, divisor, nchw
                        .addInputs(input)
                        .addOutputs(Nd4j.create(new int[]{1, 1, 3, 3}, outputOrder))
                        .build();

                DynamicCustomOp op2 = DynamicCustomOp.builder("avgpool2d")
                        .addIntegerArguments(new int[]{2, 2, 1, 1, 0, 0, 1, 1, 1, 1, 0})   //ky, kx, sH, sW, py, px, dy, dx, isSameMode, ???, divisor, nchw
                        .addInputs(input)
                        .addOutputs(Nd4j.create(new int[]{1, 1, 3, 3}, outputOrder))
                        .build();

                Nd4j.getExecutioner().exec(op1);
                Nd4j.getExecutioner().exec(op2);
                INDArray actEnabled = op1.getOutputArgument(0);
                INDArray actDl4j = op2.getOutputArgument(0);


                String msg = "inOrder=" + inputOrder + ", outOrder=" + outputOrder;
                assertEquals(msg, expDl4j, actDl4j.get(point(0), point(0), all(), all()));
                assertEquals(msg, expEnabled, actEnabled.get(point(0), point(0), all(), all()));
            }
        }
    }

    @Test
    public void testPooling1() {
        for( char outputOrder : new char[]{'c', 'f'}) {
            INDArray exp = Nd4j.create(new float[]{6.f, 7.f, 10.f, 11.f, 22.f, 23.f, 26.f, 27.f, 38.f, 39.f, 42.f, 43.f, 54.f, 55.f, 58.f, 59.f}, new int[]{2, 2, 2, 2}, 'c');

            int len = 2 * 4 * 4 * 2;
            INDArray x = Nd4j.linspace(1, len, len).reshape('c', 2, 4, 4, 2);

            DynamicCustomOp op = DynamicCustomOp.builder("avgpool2d")
                    .addIntegerArguments(new int[]{2, 2, 2, 2, 0, 0, 1, 1, 1, 1, 1})        //ky, kx, sH, sW, py, px, dy, dx, isSameMode, ???, divisor, nchw
                    .addInputs(x)
                    .addOutputs(Nd4j.create(new int[]{2, 2, 2, 2}, outputOrder))
                    .build();

            Nd4j.getExecutioner().exec(op);

            INDArray out = op.getOutputArgument(0);

            assertEquals("Output order: " + outputOrder, exp, out);

            /*
            k=2, s=2, p=0, d=1, same mode, divisor = 1


            //c order: strides are descending... i.e., last dimension changes quickest

            //Minibatch 0:
                //Depth 0
            [ 0,  1
              2,  3
              4,  5
              6,  7 ]

                //Depth 1
             [ 8,  9
              10, 11
              12, 13
              14, 15 ]

                //Depth 2
             [16, 17
              18, 19
              20, 21
              22, 23 ]

                //Depth 3
             [24, 25
              26, 27
              28, 29
              30, 31 ]



            //Minibatch 1:

             */


        }
    }


    @Test
    public void testPooling2() {
        for( char outputOrder : new char[]{'c', 'f'}) {
            INDArray exp = Nd4j.create(new float[]{6.f, 7.f,  10.f,  11.f,  22.f,  23.f,  26.f,  27.f,  38.f,  39.f,  42.f,  43.f,  54.f,  55.f,  58.f, 59.f}, new int[]{2, 2, 2, 2}, 'c');

            int len = 2 * 4 * 4 * 2;
            INDArray x = Nd4j.linspace(1, len, len).reshape('c', 2, 4, 4, 2);

            DynamicCustomOp op = DynamicCustomOp.builder("avgpool2d")
                    .addIntegerArguments(new int[]{2, 2, 2, 2, 0, 0, 1, 1, 0, 1, 1})
                    .addInputs(x)
                    .addOutputs(Nd4j.create(new int[]{2, 2, 2, 2}, outputOrder))
                    .build();

            Nd4j.getExecutioner().exec(op);

            INDArray out = op.getOutputArgument(0);

            assertEquals("Output order: " + outputOrder, exp, out);
        }
    }

    @Test
    public void testPooling3() {
        for( char outputOrder : new char[]{'c', 'f'}) {
            INDArray exp = Nd4j.create(new float[]{11.f,  12.f,  15.f,  16.f,  27.f,  28.f,  31.f,  32.f,  43.f,  44.f,  47.f,  48.f,  59.f,  60.f,  63.f, 64.f}, new int[]{2, 2, 2, 2}, 'c');

            int len = 2 * 4 * 4 * 2;
            INDArray x = Nd4j.linspace(1, len, len).reshape('c', 2, 4, 4, 2);

            DynamicCustomOp op = DynamicCustomOp.builder("maxpool2d")
                    .addIntegerArguments(new int[]{2, 2, 2, 2, 0, 0, 1, 1, 1, 1, 1})
                    .addInputs(x)
                    .addOutputs(Nd4j.create(new int[]{2, 2, 2, 2}, outputOrder))
                    .build();

            Nd4j.getExecutioner().exec(op);

            INDArray out = op.getOutputArgument(0);

            assertEquals("Output order: " + outputOrder, exp, out);
        }
    }


    @Test
    public void testPooling4() {
        for( char outputOrder : new char[]{'c', 'f'}) {
            INDArray exp = Nd4j.create(new float[]{11.f,  12.f,  15.f,  16.f,  27.f,  28.f,  31.f,  32.f,  43.f,  44.f,  47.f,  48.f,  59.f,  60.f,  63.f, 64.f}, new int[]{2, 2, 2, 2}, 'c');

            int len = 2 * 4 * 4 * 2;
            INDArray x = Nd4j.linspace(1, len, len).reshape('c', 2, 4, 4, 2);

            DynamicCustomOp op = DynamicCustomOp.builder("maxpool2d")
                    .addIntegerArguments(new int[]{2, 2, 2, 2, 0, 0, 1, 1, 0, 1, 1})
                    .addInputs(x)
                    .addOutputs(Nd4j.create(new int[]{2, 2, 2, 2}, outputOrder))
                    .build();

            Nd4j.getExecutioner().exec(op);

            INDArray out = op.getOutputArgument(0);

            assertEquals("Output order: " + outputOrder, exp, out);
        }
    }


    @Test
    public void testPooling5() {
        for( char outputOrder : new char[]{'c', 'f'}) {
            INDArray exp = Nd4j.create(new float[]{7.f,    8.f,   11.f,   12.f,   14.f,   15.f,   27.f,   28.f,   31.f,   32.f,   34.f,   35.f, 42.f,   43.f,   46.f,   47.f,   49.f,   50.f,   57.f,   58.f,   61.f,   62.f,   64.f,   65.f, 77.f,   78.f,   81.f,   82.f,   84.f,   85.f,   92.f,   93.f,   96.f,   97.f,   99.f,  100.f}, new int[]{2, 3, 3, 2}, 'c');

            int len = 2 * 5 * 5 * 2;
            INDArray x = Nd4j.linspace(1, len, len).reshape('c', 2, 5, 5, 2);

            DynamicCustomOp op = DynamicCustomOp.builder("avgpool2d")
                    .addIntegerArguments(new int[]{2, 2, 2, 2, 0, 0, 1, 1, 1, 0, 1})
                    .addInputs(x)
                    .addOutputs(Nd4j.create(new int[]{2, 3, 3, 2}, outputOrder))
                    .build();

            Nd4j.getExecutioner().exec(op);

            INDArray out = op.getOutputArgument(0);

            assertEquals("Output order: " + outputOrder, exp, out);
        }
    }


    @Test
    public void testPooling6() {
        for( char outputOrder : new char[]{'c', 'f'}) {
            INDArray exp = Nd4j.create(new float[]{7.f,   8.f,  11.f,  12.f,  27.f,  28.f,  31.f,  32.f,  57.f,  58.f,  61.f,  62.f,  77.f,  78.f,  81.f, 82.f}, new int[]{2, 2, 2, 2}, 'c');

            int len = 2 * 5 * 5 * 2;
            INDArray x = Nd4j.linspace(1, len, len).reshape('c', 2, 5, 5, 2);

            DynamicCustomOp op = DynamicCustomOp.builder("avgpool2d")
                    .addIntegerArguments(new int[]{2, 2, 2, 2, 0, 0, 1, 1, 0, 1, 1})
                    .addInputs(x)
                    .addOutputs(Nd4j.create(new int[]{2, 2, 2, 2}, outputOrder))
                    .build();

            Nd4j.getExecutioner().exec(op);

            INDArray out = op.getOutputArgument(0);

            assertEquals("Output order: " + outputOrder, exp, out);
        }
    }


    @Test
    public void testPooling7() {
        for( char outputOrder : new char[]{'c', 'f'}) {
            INDArray exp = Nd4j.create(new float[]{7.f, 9.f, 17.f, 19.f, 32.f, 34.f, 42.f, 44.f, 57.f, 59.f, 67.f, 69.f, 82.f, 84.f, 92.f, 94.f}, new int[]{2, 2, 2, 2}, 'c');

            int len = 2 * 2 * 5 * 5;
            INDArray x = Nd4j.linspace(1, len, len).reshape('c', 2, 2, 5, 5);

            DynamicCustomOp op = DynamicCustomOp.builder("maxpool2d")
                    .addIntegerArguments(new int[]{2, 2, 2, 2, 0, 0, 1, 1, 0, 1, 0})
                    .addInputs(x)
                    .addOutputs(Nd4j.create(new int[]{2, 2, 2, 2}, outputOrder))
                    .build();

            Nd4j.getExecutioner().exec(op);

            INDArray out = op.getOutputArgument(0);

            assertEquals("Output order: " + outputOrder, exp, out);
        }
    }

    @Test
    public void testPooling8() {
        for( char outputOrder : new char[]{'c', 'f'}) {
            INDArray exp = Nd4j.create(new float[]{1.f, 2.5f, 4.5f, 8.5f, 10.f, 12.f, 18.5f, 20.f, 22.f, 26.f, 27.5f, 29.5f, 33.5f, 35.f, 37.f, 43.5f, 45.f, 47.f,  51.f, 52.5f, 54.5f,  58.5f, 60.f, 62.f, 68.5f, 70.f, 72.f,  76.f, 77.5f, 79.5f, 83.5f, 85.f, 87.f,  93.5f, 95.f, 97.f}, new int[]{2, 2, 3, 3}, 'c');

            int len = 2 * 2 * 5 * 5;
            INDArray x = Nd4j.linspace(1, len, len).reshape('c', 2, 2, 5, 5);

            DynamicCustomOp op = DynamicCustomOp.builder("avgpool2d")
                    .addIntegerArguments(new int[]{2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0})
                    .addInputs(x)
                    .addOutputs(Nd4j.create(new int[]{2, 2, 3, 3}, outputOrder))
                    .build();

            Nd4j.getExecutioner().exec(op);

            INDArray out = op.getOutputArgument(0);

            assertEquals("Output order: " + outputOrder, exp, out);
        }
    }

    @Test
    public void testPooling9() {
        for( char outputOrder : new char[]{'c', 'f'}) {
            INDArray exp = Nd4j.create(new float[]{0.25f, 1.25f, 2.25f,  4.25f, 10.f, 12.f, 9.25f, 20.f, 22.f, 6.5f, 13.75f, 14.75f, 16.75f, 35.f, 37.f,  21.75f, 45.f, 47.f,  12.75f, 26.25f, 27.25f,  29.25f, 60.f, 62.f, 34.25f, 70.f, 72.f, 19.f, 38.75f, 39.75f, 41.75f, 85.f, 87.f, 46.75f, 95.f, 97.f}, new int[]{2, 2, 3, 3}, 'c');

            int len = 2 * 2 * 5 * 5;
            INDArray x = Nd4j.linspace(1, len, len).reshape('c', 2, 2, 5, 5);

            DynamicCustomOp op = DynamicCustomOp.builder("avgpool2d")
                    .addIntegerArguments(new int[]{2, 2, 2, 2, 1, 1, 1, 1, 0, 1, 0})
                    .addInputs(x)
                    .addOutputs(Nd4j.create(new int[]{2, 2, 3, 3}, outputOrder))
                    .build();

            Nd4j.getExecutioner().exec(op);

            INDArray out = op.getOutputArgument(0);

            assertEquals("Output order: " + outputOrder, exp, out);
        }
    }

    @Test
    public void testPooling10() {
        for( char outputOrder : new char[]{'c', 'f'}) {
            INDArray exp = Nd4j.create(new float[]{4.f, 6.f, 7.5f, 14.f, 16.f, 17.5f,  21.5f, 23.5f, 25.f, 29.f, 31.f, 32.5f, 39.f, 41.f, 42.5f, 46.5f, 48.5f, 50.f, 54.f, 56.f, 57.5f,  64.f, 66.f, 67.5f, 71.5f, 73.5f, 75.f, 79.f, 81.f, 82.5f, 89.f, 91.f, 92.5f,  96.5f, 98.5f, 100.f}, new int[]{2, 2, 3, 3}, 'c');

            int len = 2 * 2 * 5 * 5;
            INDArray x = Nd4j.linspace(1, len, len).reshape('c', 2, 2, 5, 5);

            DynamicCustomOp op = DynamicCustomOp.builder("avgpool2d")
                    .addIntegerArguments(new int[]{2, 2, 2, 2, 0, 0, 1, 1, 1, 0, 0})
                    .addInputs(x)
                    .addOutputs(Nd4j.create(new int[]{2, 2, 3, 3}, outputOrder))
                    .build();

            Nd4j.getExecutioner().exec(op);

            INDArray out = op.getOutputArgument(0);

            assertEquals("Output order: " + outputOrder, exp, out);
        }
    }

    @Test
    public void testPooling11() {
        for( char outputOrder : new char[]{'c', 'f'}) {
            INDArray exp = Nd4j.create(new float[]{3, 4, 6, 7}, new int[]{1, 1, 2, 2}, 'c');

            int len = 1 * 1 * 3 * 3;
            INDArray x = Nd4j.linspace(1, len, len).reshape('c', 1, 1, 3, 3);

            DynamicCustomOp op = DynamicCustomOp.builder("avgpool2d")
                    .addIntegerArguments(new int[]{2, 2, 1, 1, 0, 0, 1, 1, 0, 0, 0})
                    .addInputs(x)
                    .addOutputs(Nd4j.create(new int[]{1, 1, 2, 2}, outputOrder))
                    .build();

            Nd4j.getExecutioner().exec(op);

            INDArray out = op.getOutputArgument(0);

            assertEquals("Output order: " + outputOrder, exp, out);
        }
    }

    @Test
    public void testPooling12() {
        for( char outputOrder : new char[]{'c', 'f'}) {
            INDArray exp = Nd4j.create(new float[]{3.f, 4.f, 4.5f, 6.f, 7.f, 7.5f, 7.5f, 8.5f, 9.f}, new int[]{1, 1, 3, 3}, 'c');

            int len = 1 * 1 * 3 * 3;
            INDArray x = Nd4j.linspace(1, len, len).reshape('c', 1, 1, 3, 3);

            DynamicCustomOp op = DynamicCustomOp.builder("avgpool2d")
                    .addIntegerArguments(new int[]{2, 2, 1, 1, 0, 0, 1, 1, 1, 0, 0})
                    .addInputs(x)
                    .addOutputs(Nd4j.create(new int[]{1, 1, 3, 3}, outputOrder))
                    .build();

            Nd4j.getExecutioner().exec(op);

            INDArray out = op.getOutputArgument(0);

            assertEquals("Output order: " + outputOrder, exp, out);
        }
    }


    @Test
    public void testPooling13() {
        for( char outputOrder : new char[]{'c'}) {
            INDArray exp = Nd4j.create(new float[]{3.f, 4.f, 4.5f, 6.f, 7.f, 7.5f, 7.5f, 8.5f, 9.f}, new int[]{1, 1, 3, 3}, 'c');

            int len = 1 * 1 * 3 * 3;
            INDArray x = Nd4j.linspace(1, len, len).reshape('c', 1, 1, 3, 3);

            DynamicCustomOp op = DynamicCustomOp.builder("avgpool2d")
                    .addIntegerArguments(new int[]{2, 2, 1, 1, 0, 0, 1, 1, 1, 0, 0})
                    .addInputs(x)
                    .addOutputs(Nd4j.create(new int[]{1, 1, 3, 3}, outputOrder))
                    .build();

            Nd4j.getExecutioner().exec(op);

            INDArray out = op.getOutputArgument(0);

            assertEquals("Output order: " + outputOrder, exp, out);
        }
    }



    @Test
    public void testPoolingDilation(){

        int[] inputShape = {1, 1, 4, 5};
        int outH = inputShape[2];
        int outW = inputShape[3];

        int[] kernel = {2, 2};
        int[] strides = {1, 1};
        int[] pad = {1, 1};         //From same mode
        int[] dilation = {2, 2};
        boolean same = true;

        /*
        Input:
        [ 1,  2,  3,  4,  5
          6,  7,  8,  9, 10
         11, 12, 13, 14, 15
         16, 17, 18, 19, 20 ]

        Input with SAME padding:
        [ 0,  0,  0,  0,  0,  0,  0
          0,  1,  2,  3,  4,  5,  0
          0,  6,  7,  8,  9, 10,  0
          0, 11, 12, 13, 14, 15,  0
          0, 16, 17, 18, 19, 20,  0
          0,  0,  0,  0,  0,  0,  0]

         4x5 in
         Same mode, stride 1, dilation 2, kernel 2
         kHEffective = (2 + (2-1)*(2-1)) = 3
         oH = ceil(iH/sH) = 4
         oW = ceil(iW/sW) = 5
         totalPadH = (oH-1)*sH + kH - inH = (4-1)*1 + 3 - 4 = 2
         padTop = 1, padBottom = 1

         totalPadW = (oW-1)*sW + kW - inW = (5-1)*1 + 3 - 5 = 2
         padLeft = 1, padRight = 1

        [ 0,  0]    [ 0,  0]    [ 0,  0]    [ 0,  0]    [ 0,  0]
        [ 0,  7]    [ 6,  8]    [ 7,  9]    [ 8, 10]    [ 9,  0]

        [ 0   2]    [ 1,  3]    [ 2,  4]    [ 3,  5]    [ 4,  0]
        [ 0, 12]    [11, 13]    [12, 14]    [13, 15]    [14,  0]

        [ 0,  7]    [ 6,  8]    [ 7,  9]    [ 8, 10]    [ 9,  0]
        [ 0, 17]    [16, 18]    [17, 19]    [18, 20]    [19,  0]

        [ 0, 12]    [11, 13]    [12, 14]    [13, 15]    [14,  0]
        [ 0,  0],   [ 0,  0]    [ 0,  0]    [ 0,  0]    [ 0,  0]
         */

        INDArray origInput = Nd4j.create(inputShape);
        origInput.get(point(0), point(0), all(), all()).assign(
                Nd4j.linspace(1,20,20).reshape('c',4,5));


        INDArray expMax = Nd4j.create(1,1,4,5);
        expMax.get(point(0), point(0), all(), all()).assign(
                Nd4j.create(new double[][]{
                        { 7,  8,  9, 10,  9},
                        {12, 13, 14, 15, 14},
                        {17, 18, 19, 20, 19},
                        {12, 13, 14, 15, 14}}));

        INDArray sum = Nd4j.create(1,1,4,5);
        sum.get(point(0), point(0), all(), all()).assign(
                Nd4j.create(new double[][]{
                        { 7,     (6+8),       (7+9),       (8+10),       9},
                        {(2+12), (1+3+11+13), (2+4+12+14), (3+5+13+15),  (4+14)},
                        {(7+17), (6+8+16+18), (7+9+17+19), (8+10+18+20), (9+19)},
                        {12,     (11+13),     (12+14),     (13+15),      14}}));
        INDArray expAvgExclude = sum.dup();
        expAvgExclude.get(point(0), point(0), all(), all()).divi(
                Nd4j.create(new double[][]{
                        { 1,  2,  2,  2,  1},
                        { 2,  4,  4,  4,  2},
                        { 2,  4,  4,  4,  2},
                        { 1,  2,  2,  2,  1}}));

        INDArray expAvgInclude = sum.div(4.0);


        int testNum = 0;
        for( int i=0; i<3; i++ ){


            List<Pair<INDArray, String>> inputs = NDArrayCreationUtil.getAll4dTestArraysWithShape(12345, inputShape);

            for(Pair<INDArray,String> pIn : inputs){
                INDArray input = pIn.getFirst().assign(origInput);

                INDArray out = Nd4j.create(input.shape(), 'c');

                //TODO Test on weird strides also (i.e., remove the dup here)
                input = input.dup('c');

                INDArray exp;
                String mode;
                switch (i){
                    case 0: //Max
                        Convolution.pooling2D(input, kernel[0], kernel[1], strides[0], strides[1], pad[0], pad[1], dilation[0], dilation[1],
                                same, Pooling2D.Pooling2DType.MAX, Pooling2D.Divisor.INCLUDE_PADDING,
                                0.0, outH, outW, out);
                        exp = expMax;
                        mode = "max";
                        break;
                    case 1: //Avg + mode 0 (exclude padding)
                        Convolution.pooling2D(input, kernel[0], kernel[1], strides[0], strides[1], pad[0], pad[1], dilation[0], dilation[1],
                                same, Pooling2D.Pooling2DType.AVG, Pooling2D.Divisor.EXCLUDE_PADDING,
                                0.0, outH, outW, out);
                        exp = expAvgExclude;
                        mode = "avg_0";
                        break;
                    case 2: //Avg + mode 1 (include padding)
                        Convolution.pooling2D(input, kernel[0], kernel[1], strides[0], strides[1], pad[0], pad[1], dilation[0], dilation[1],
                                same, Pooling2D.Pooling2DType.AVG, Pooling2D.Divisor.INCLUDE_PADDING,
                                0.0, outH, outW, out);
                        exp = expAvgInclude;
                        mode = "avg_2";
                        break;
                    default:
                        throw new RuntimeException();
                }

                String msg = "TestNum=" + testNum + ", Mode: " + mode + ", " + pIn.getSecond();
                assertEquals(msg, exp, out);
                testNum++;
            }
        }
    }


    @Override
    public char ordering() {
        return 'f';
    }
}
