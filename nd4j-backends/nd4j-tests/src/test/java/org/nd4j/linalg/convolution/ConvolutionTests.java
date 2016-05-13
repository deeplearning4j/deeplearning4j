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
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;


import java.util.Arrays;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/**
 * Created by agibsonccc on 9/6/14.
 */
@RunWith(Parameterized.class)
public  class ConvolutionTests extends BaseNd4jTest {

    public ConvolutionTests(Nd4jBackend backend) {
        super(backend);
    }


    @Test
    public void testIm2ColKnownValues(){
        //Input: w=3, h=3, depth=2, minibatch = 2
        //kh=2, kw=2
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
        INDArray input = Nd4j.create(new int[]{miniBatch,depth,height,width},'c');
        input.put(new INDArrayIndex[]{NDArrayIndex.point(0), NDArrayIndex.point(0),NDArrayIndex.all(), NDArrayIndex.all()}, Nd4j.create(new double[][]{{0,1,2},{3,4,5},{6,7,8}}));
        input.put(new INDArrayIndex[]{NDArrayIndex.point(0), NDArrayIndex.point(1),NDArrayIndex.all(), NDArrayIndex.all()}, Nd4j.create(new double[][]{{9,10,11},{12,13,14},{15,16,17}}));
        input.put(new INDArrayIndex[]{NDArrayIndex.point(1), NDArrayIndex.point(0),NDArrayIndex.all(), NDArrayIndex.all()}, Nd4j.create(new double[][]{{18,19,20},{21,22,23},{24,25,26}}));
        input.put(new INDArrayIndex[]{NDArrayIndex.point(1), NDArrayIndex.point(1),NDArrayIndex.all(), NDArrayIndex.all()}, Nd4j.create(new double[][]{{27,28,29},{30,31,32},{33,34,35}}));

        //Expected data:
        INDArray expected = Nd4j.create(new int[]{miniBatch,depth,kH,kW,outH,outW},'c');

            //Example 0
                //depth 0
        expected.put(new INDArrayIndex[]{NDArrayIndex.point(0),NDArrayIndex.point(0),NDArrayIndex.all(),NDArrayIndex.all(), NDArrayIndex.point(0), NDArrayIndex.point(0)}, Nd4j.create(new double[][]{{0,1},{3,4}}));
        expected.put(new INDArrayIndex[]{NDArrayIndex.point(0),NDArrayIndex.point(0),NDArrayIndex.all(),NDArrayIndex.all(), NDArrayIndex.point(0), NDArrayIndex.point(1)}, Nd4j.create(new double[][]{{1,2},{4,5}}));
        expected.put(new INDArrayIndex[]{NDArrayIndex.point(0),NDArrayIndex.point(0),NDArrayIndex.all(),NDArrayIndex.all(), NDArrayIndex.point(1), NDArrayIndex.point(0)}, Nd4j.create(new double[][]{{3,4},{6,7}}));
        expected.put(new INDArrayIndex[]{NDArrayIndex.point(0),NDArrayIndex.point(0),NDArrayIndex.all(),NDArrayIndex.all(), NDArrayIndex.point(1), NDArrayIndex.point(1)}, Nd4j.create(new double[][]{{4,5},{7,8}}));
                //depth 1
        expected.put(new INDArrayIndex[]{NDArrayIndex.point(0),NDArrayIndex.point(1),NDArrayIndex.all(),NDArrayIndex.all(), NDArrayIndex.point(0), NDArrayIndex.point(0)}, Nd4j.create(new double[][]{{9,10},{12,13}}));
        expected.put(new INDArrayIndex[]{NDArrayIndex.point(0),NDArrayIndex.point(1),NDArrayIndex.all(),NDArrayIndex.all(), NDArrayIndex.point(0), NDArrayIndex.point(1)}, Nd4j.create(new double[][]{{10,11},{13,14}}));
        expected.put(new INDArrayIndex[]{NDArrayIndex.point(0),NDArrayIndex.point(1),NDArrayIndex.all(),NDArrayIndex.all(), NDArrayIndex.point(1), NDArrayIndex.point(0)}, Nd4j.create(new double[][]{{12,13},{15,16}}));
        expected.put(new INDArrayIndex[]{NDArrayIndex.point(0),NDArrayIndex.point(1),NDArrayIndex.all(),NDArrayIndex.all(), NDArrayIndex.point(1), NDArrayIndex.point(1)}, Nd4j.create(new double[][]{{13,14},{16,17}}));

        //Example 1
        //depth 0
        expected.put(new INDArrayIndex[]{NDArrayIndex.point(1),NDArrayIndex.point(0),NDArrayIndex.all(),NDArrayIndex.all(), NDArrayIndex.point(0), NDArrayIndex.point(0)}, Nd4j.create(new double[][]{{18,19},{21,22}}));
        expected.put(new INDArrayIndex[]{NDArrayIndex.point(1),NDArrayIndex.point(0),NDArrayIndex.all(),NDArrayIndex.all(), NDArrayIndex.point(0), NDArrayIndex.point(1)}, Nd4j.create(new double[][]{{19,20},{22,23}}));
        expected.put(new INDArrayIndex[]{NDArrayIndex.point(1),NDArrayIndex.point(0),NDArrayIndex.all(),NDArrayIndex.all(), NDArrayIndex.point(1), NDArrayIndex.point(0)}, Nd4j.create(new double[][]{{21,22},{24,25}}));
        expected.put(new INDArrayIndex[]{NDArrayIndex.point(1),NDArrayIndex.point(0),NDArrayIndex.all(),NDArrayIndex.all(), NDArrayIndex.point(1), NDArrayIndex.point(1)}, Nd4j.create(new double[][]{{22,23},{25,26}}));
        //depth 1
        expected.put(new INDArrayIndex[]{NDArrayIndex.point(1),NDArrayIndex.point(1),NDArrayIndex.all(),NDArrayIndex.all(), NDArrayIndex.point(0), NDArrayIndex.point(0)}, Nd4j.create(new double[][]{{27,28},{30,31}}));
        expected.put(new INDArrayIndex[]{NDArrayIndex.point(1),NDArrayIndex.point(1),NDArrayIndex.all(),NDArrayIndex.all(), NDArrayIndex.point(0), NDArrayIndex.point(1)}, Nd4j.create(new double[][]{{28,29},{31,32}}));
        expected.put(new INDArrayIndex[]{NDArrayIndex.point(1),NDArrayIndex.point(1),NDArrayIndex.all(),NDArrayIndex.all(), NDArrayIndex.point(1), NDArrayIndex.point(0)}, Nd4j.create(new double[][]{{30,31},{33,34}}));
        expected.put(new INDArrayIndex[]{NDArrayIndex.point(1),NDArrayIndex.point(1),NDArrayIndex.all(),NDArrayIndex.all(), NDArrayIndex.point(1), NDArrayIndex.point(1)}, Nd4j.create(new double[][]{{31,32},{34,35}}));

        INDArray out = Convolution.im2col(input,kH,kW,sY,sX,pY,pX,false);
        assertEquals(expected, out);

        //Now: test with a provided results array, where the results array has weird strides
        INDArray out2 = Nd4j.create(new int[]{miniBatch,depth,outH,outW,kH,kW},'c');
        INDArray out2p = out2.permute(0,1,4,5,2,3);
        Convolution.im2col(input,kH,kW,sY,sX,pY,pX,false,out2p);
        assertEquals(expected,out2p);

        INDArray out3 = Nd4j.create(new int[]{miniBatch,outH,outW,depth,kH,kW},'c');
        INDArray out3p = out3.permute(0,3,4,5,1,2);
        Convolution.im2col(input,kH,kW,sY,sX,pY,pX,false,out3p);
        assertEquals(expected,out3p);
    }

    @Test
    public void testIm2ColKnownValuesMiniBatch3(){
        //Input: w=3, h=3, depth=2, minibatch = 3
        //kh=2, kw=2
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
        INDArray input = Nd4j.create(new int[]{miniBatch,depth,height,width},'c');
        input.put(new INDArrayIndex[]{NDArrayIndex.point(0), NDArrayIndex.point(0),NDArrayIndex.all(), NDArrayIndex.all()}, Nd4j.create(new double[][]{{0,1,2},{3,4,5},{6,7,8}}));
        input.put(new INDArrayIndex[]{NDArrayIndex.point(0), NDArrayIndex.point(1),NDArrayIndex.all(), NDArrayIndex.all()}, Nd4j.create(new double[][]{{9,10,11},{12,13,14},{15,16,17}}));
        input.put(new INDArrayIndex[]{NDArrayIndex.point(1), NDArrayIndex.point(0),NDArrayIndex.all(), NDArrayIndex.all()}, Nd4j.create(new double[][]{{18,19,20},{21,22,23},{24,25,26}}));
        input.put(new INDArrayIndex[]{NDArrayIndex.point(1), NDArrayIndex.point(1),NDArrayIndex.all(), NDArrayIndex.all()}, Nd4j.create(new double[][]{{27,28,29},{30,31,32},{33,34,35}}));
        input.put(new INDArrayIndex[]{NDArrayIndex.point(2), NDArrayIndex.point(0),NDArrayIndex.all(), NDArrayIndex.all()}, Nd4j.create(new double[][]{{36,37,38},{39,40,41},{42,43,44}}));
        input.put(new INDArrayIndex[]{NDArrayIndex.point(2), NDArrayIndex.point(1),NDArrayIndex.all(), NDArrayIndex.all()}, Nd4j.create(new double[][]{{45,46,47},{48,49,50},{51,52,53}}));

        //Expected data:
        INDArray expected = Nd4j.create(new int[]{miniBatch,depth,kH,kW,outH,outW},'c');

        //Example 0
        //depth 0
        expected.put(new INDArrayIndex[]{NDArrayIndex.point(0),NDArrayIndex.point(0),NDArrayIndex.all(),NDArrayIndex.all(), NDArrayIndex.point(0), NDArrayIndex.point(0)}, Nd4j.create(new double[][]{{0,1},{3,4}}));
        expected.put(new INDArrayIndex[]{NDArrayIndex.point(0),NDArrayIndex.point(0),NDArrayIndex.all(),NDArrayIndex.all(), NDArrayIndex.point(0), NDArrayIndex.point(1)}, Nd4j.create(new double[][]{{1,2},{4,5}}));
        expected.put(new INDArrayIndex[]{NDArrayIndex.point(0),NDArrayIndex.point(0),NDArrayIndex.all(),NDArrayIndex.all(), NDArrayIndex.point(1), NDArrayIndex.point(0)}, Nd4j.create(new double[][]{{3,4},{6,7}}));
        expected.put(new INDArrayIndex[]{NDArrayIndex.point(0),NDArrayIndex.point(0),NDArrayIndex.all(),NDArrayIndex.all(), NDArrayIndex.point(1), NDArrayIndex.point(1)}, Nd4j.create(new double[][]{{4,5},{7,8}}));
        //depth 1
        expected.put(new INDArrayIndex[]{NDArrayIndex.point(0),NDArrayIndex.point(1),NDArrayIndex.all(),NDArrayIndex.all(), NDArrayIndex.point(0), NDArrayIndex.point(0)}, Nd4j.create(new double[][]{{9,10},{12,13}}));
        expected.put(new INDArrayIndex[]{NDArrayIndex.point(0),NDArrayIndex.point(1),NDArrayIndex.all(),NDArrayIndex.all(), NDArrayIndex.point(0), NDArrayIndex.point(1)}, Nd4j.create(new double[][]{{10,11},{13,14}}));
        expected.put(new INDArrayIndex[]{NDArrayIndex.point(0),NDArrayIndex.point(1),NDArrayIndex.all(),NDArrayIndex.all(), NDArrayIndex.point(1), NDArrayIndex.point(0)}, Nd4j.create(new double[][]{{12,13},{15,16}}));
        expected.put(new INDArrayIndex[]{NDArrayIndex.point(0),NDArrayIndex.point(1),NDArrayIndex.all(),NDArrayIndex.all(), NDArrayIndex.point(1), NDArrayIndex.point(1)}, Nd4j.create(new double[][]{{13,14},{16,17}}));

        //Example 1
        //depth 0
        expected.put(new INDArrayIndex[]{NDArrayIndex.point(1),NDArrayIndex.point(0),NDArrayIndex.all(),NDArrayIndex.all(), NDArrayIndex.point(0), NDArrayIndex.point(0)}, Nd4j.create(new double[][]{{18,19},{21,22}}));
        expected.put(new INDArrayIndex[]{NDArrayIndex.point(1),NDArrayIndex.point(0),NDArrayIndex.all(),NDArrayIndex.all(), NDArrayIndex.point(0), NDArrayIndex.point(1)}, Nd4j.create(new double[][]{{19,20},{22,23}}));
        expected.put(new INDArrayIndex[]{NDArrayIndex.point(1),NDArrayIndex.point(0),NDArrayIndex.all(),NDArrayIndex.all(), NDArrayIndex.point(1), NDArrayIndex.point(0)}, Nd4j.create(new double[][]{{21,22},{24,25}}));
        expected.put(new INDArrayIndex[]{NDArrayIndex.point(1),NDArrayIndex.point(0),NDArrayIndex.all(),NDArrayIndex.all(), NDArrayIndex.point(1), NDArrayIndex.point(1)}, Nd4j.create(new double[][]{{22,23},{25,26}}));
        //depth 1
        expected.put(new INDArrayIndex[]{NDArrayIndex.point(1),NDArrayIndex.point(1),NDArrayIndex.all(),NDArrayIndex.all(), NDArrayIndex.point(0), NDArrayIndex.point(0)}, Nd4j.create(new double[][]{{27,28},{30,31}}));
        expected.put(new INDArrayIndex[]{NDArrayIndex.point(1),NDArrayIndex.point(1),NDArrayIndex.all(),NDArrayIndex.all(), NDArrayIndex.point(0), NDArrayIndex.point(1)}, Nd4j.create(new double[][]{{28,29},{31,32}}));
        expected.put(new INDArrayIndex[]{NDArrayIndex.point(1),NDArrayIndex.point(1),NDArrayIndex.all(),NDArrayIndex.all(), NDArrayIndex.point(1), NDArrayIndex.point(0)}, Nd4j.create(new double[][]{{30,31},{33,34}}));
        expected.put(new INDArrayIndex[]{NDArrayIndex.point(1),NDArrayIndex.point(1),NDArrayIndex.all(),NDArrayIndex.all(), NDArrayIndex.point(1), NDArrayIndex.point(1)}, Nd4j.create(new double[][]{{31,32},{34,35}}));

        //Example 2
        //depth 0
        expected.put(new INDArrayIndex[]{NDArrayIndex.point(2),NDArrayIndex.point(0),NDArrayIndex.all(),NDArrayIndex.all(), NDArrayIndex.point(0), NDArrayIndex.point(0)}, Nd4j.create(new double[][]{{36,37},{39,40}}));
        expected.put(new INDArrayIndex[]{NDArrayIndex.point(2),NDArrayIndex.point(0),NDArrayIndex.all(),NDArrayIndex.all(), NDArrayIndex.point(0), NDArrayIndex.point(1)}, Nd4j.create(new double[][]{{37,38},{40,41}}));
        expected.put(new INDArrayIndex[]{NDArrayIndex.point(2),NDArrayIndex.point(0),NDArrayIndex.all(),NDArrayIndex.all(), NDArrayIndex.point(1), NDArrayIndex.point(0)}, Nd4j.create(new double[][]{{39,40},{42,43}}));
        expected.put(new INDArrayIndex[]{NDArrayIndex.point(2),NDArrayIndex.point(0),NDArrayIndex.all(),NDArrayIndex.all(), NDArrayIndex.point(1), NDArrayIndex.point(1)}, Nd4j.create(new double[][]{{40,41},{43,44}}));
        //depth 1
        expected.put(new INDArrayIndex[]{NDArrayIndex.point(2),NDArrayIndex.point(1),NDArrayIndex.all(),NDArrayIndex.all(), NDArrayIndex.point(0), NDArrayIndex.point(0)}, Nd4j.create(new double[][]{{45,46},{48,49}}));
        expected.put(new INDArrayIndex[]{NDArrayIndex.point(2),NDArrayIndex.point(1),NDArrayIndex.all(),NDArrayIndex.all(), NDArrayIndex.point(0), NDArrayIndex.point(1)}, Nd4j.create(new double[][]{{46,47},{49,50}}));
        expected.put(new INDArrayIndex[]{NDArrayIndex.point(2),NDArrayIndex.point(1),NDArrayIndex.all(),NDArrayIndex.all(), NDArrayIndex.point(1), NDArrayIndex.point(0)}, Nd4j.create(new double[][]{{48,49},{51,52}}));
        expected.put(new INDArrayIndex[]{NDArrayIndex.point(2),NDArrayIndex.point(1),NDArrayIndex.all(),NDArrayIndex.all(), NDArrayIndex.point(1), NDArrayIndex.point(1)}, Nd4j.create(new double[][]{{49,50},{52,53}}));

        INDArray out = Convolution.im2col(input,kH,kW,sY,sX,pY,pX,false);
        assertEquals(expected, out);

        //Now: test with a provided results array, where the results array has weird strides
        INDArray out2 = Nd4j.create(new int[]{miniBatch,depth,outH,outW,kH,kW},'c');
        INDArray out2p = out2.permute(0,1,4,5,2,3);
        Convolution.im2col(input,kH,kW,sY,sX,pY,pX,false,out2p);
        assertEquals(expected,out2p);

        INDArray out3 = Nd4j.create(new int[]{miniBatch,outH,outW,depth,kH,kW},'c');
        INDArray out3p = out3.permute(0,3,4,5,1,2);
        Convolution.im2col(input,kH,kW,sY,sX,pY,pX,false,out3p);
        assertEquals(expected,out3p);
    }


    @Test
    public void testConvOutWidthAndHeight() {
        int outSize = Convolution.outSize(2,1,1,2,false);
        assertEquals(6,outSize);
    }

    @Test
    public void testIm2Col() {
        INDArray linspaced = Nd4j.linspace(1,16,16).reshape(2,2,2,2);
        INDArray ret = Convolution.im2col(linspaced, 1, 1, 1, 1, 2, 2, 0, false);
        System.out.println(ret);
    }




    @Test
    @Ignore
    public void testCompareIm2ColImpl() {

        int[] miniBatches = {1, 3, 5};
        int[] depths = {1, 3, 5};
        int[] inHeights = {5,21};
        int[] inWidths = {5,21};
        int[] strideH = {1,2};
        int[] strideW = {1,2};
        int[] sizeW = {1,2,3};
        int[] sizeH = {1,2,3};
        int[] padH = {0,1,2};
        int[] padW = {0,1,2};
        boolean[] coverall = {false,true};

        DataBuffer.Type[] types = new DataBuffer.Type[]{DataBuffer.Type.FLOAT, DataBuffer.Type.DOUBLE, DataBuffer.Type.FLOAT, DataBuffer.Type.DOUBLE};
        DataBuffer.AllocationMode[] modes = new DataBuffer.AllocationMode[]{DataBuffer.AllocationMode.HEAP, DataBuffer.AllocationMode.HEAP,
                DataBuffer.AllocationMode.DIRECT, DataBuffer.AllocationMode.DIRECT};

        String factoryClassName = Nd4j.factory().getClass().toString().toLowerCase();
        if( factoryClassName.contains("jcublas") || factoryClassName.contains("cuda") ){
            //Only test direct for CUDA; test all for CPU
            types = new DataBuffer.Type[]{DataBuffer.Type.FLOAT, DataBuffer.Type.DOUBLE};
            modes = new DataBuffer.AllocationMode[]{DataBuffer.AllocationMode.DIRECT, DataBuffer.AllocationMode.DIRECT};
        }

        for( int i=0; i<types.length; i++ ) {
            DataBuffer.Type type = types[i];
            DataBuffer.AllocationMode mode = modes[i];

            Nd4j.factory().setDType(type);
            Nd4j.dtype = type;
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
                                                        continue;   //(w-kp+2*pw)/sw + 1 is not an integer,  i.e., number of outputs doesn't fit

                                                    System.out.println("Running " + m + " " + d + " " + h + " " + w);
                                                    for( boolean cAll : coverall ) {

                                                        INDArray in = Nd4j.rand(new int[]{m, d, h, w});
                                                        //assertEquals(in.data().allocationMode(), mode);
                                                        //assertEquals(in.data().dataType(), type);

                                                        INDArray outOrig = OldConvolution.im2col(in, kh, kw, sh, sw, ph, pw, -1, cAll); //Old implementation
                                                        INDArray outNew = Convolution.im2col(in, kh, kw, sh, sw, ph, pw, cAll);         //Current implementation

                                                        assertArrayEquals(outOrig.data().asFloat(), outNew.data().asFloat(), 0.01f);
                                                        assertEquals(outOrig,outNew);
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
    }

    @Test
    @Ignore
    public void testCompareIm2Col() throws Exception {

        int[] miniBatches = {1, 3, 5};
        int[] depths = {1, 3, 5};
        int[] inHeights = {5,21};
        int[] inWidths = {5,21};
        int[] strideH = {1,2};
        int[] strideW = {1,2};
        int[] sizeW = {1,2,3};
        int[] sizeH = {1,2,3};
        int[] padH = {0,1,2};
        int[] padW = {0,1,2};

        DataBuffer.Type[] types = new DataBuffer.Type[]{DataBuffer.Type.FLOAT, DataBuffer.Type.DOUBLE, DataBuffer.Type.FLOAT, DataBuffer.Type.DOUBLE};
        DataBuffer.AllocationMode[] modes = new DataBuffer.AllocationMode[]{DataBuffer.AllocationMode.HEAP, DataBuffer.AllocationMode.HEAP,
                DataBuffer.AllocationMode.DIRECT, DataBuffer.AllocationMode.DIRECT};

        String factoryClassName = Nd4j.factory().getClass().toString().toLowerCase();
        if( factoryClassName.contains("jcublas") || factoryClassName.contains("cuda") ){
            //Only test direct for CUDA; test all for CPU
            types = new DataBuffer.Type[]{DataBuffer.Type.FLOAT, DataBuffer.Type.DOUBLE};
            modes = new DataBuffer.AllocationMode[]{DataBuffer.AllocationMode.DIRECT, DataBuffer.AllocationMode.DIRECT};
        }

        for( int i=0; i<types.length; i++ ) {
            DataBuffer.Type type = types[i];
            DataBuffer.AllocationMode mode = modes[i];

            Nd4j.factory().setDType(type);
            Nd4j.dtype = type;
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
                                                        continue;   //(w-kp+2*pw)/sw + 1 is not an integer, i.e., number of outputs doesn't fit

                                                    INDArray in = Nd4j.rand(new int[]{m, d, h, w});
                                                    assertEquals(in.data().allocationMode(), mode);
                                                    assertEquals(in.data().dataType(), type);
                                                    INDArray im2col = Convolution.im2col(in, kh, kw, sh, sw, ph, pw, false);    //Cheating, to get correct shape for input

                                                    INDArray imgOutOld = OldConvolution.col2im(im2col, sh, sw, ph, pw, h, w);
                                                    INDArray imgOutNew = Convolution.col2im(im2col, sh, sw, ph, pw, h, w);
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
    }


    @Test
    public void testCol2Im() {
        int kh = 1;
        int kw = 1;
        int sy = 1;
        int sx = 1;
        int ph = 1;
        int pw = 1;
        INDArray linspaced = Nd4j.linspace(1,64,64).reshape(2,2,2,2,2,2);
        INDArray newTest = Convolution.col2im(linspaced,sy,sx,ph,pw,2,2);
        INDArray assertion = OldConvolution.col2im(linspaced,sy,sx,ph,pw,2,2);

	System.out.println("Ordering of the result, new test: " + newTest.ordering());

        System.out.println("Assertion dimensions: " + Arrays.toString(assertion.shape()));
        assertEquals(assertion,newTest);
    }


	@Test
    public void testimcolim() {
		int nEx = 2;
		int depth = 3;
		int width = 7;
		int height = 7;
		int [] kernel = {3,2} ;
		int [] stride = {2,3} ;
		int [] padding = {1,2} ;
		int prod = nEx*depth*width*height;

		INDArray in = Nd4j.linspace(1,prod,prod).reshape(nEx,depth,width,height);

		INDArray assertim2col = OldConvolution.im2col(in, kernel, stride, padding);
		INDArray im2col = Convolution.im2col(in, kernel, stride, padding);
		assertEquals(assertim2col,im2col);

		INDArray assertcol2im = OldConvolution.col2im(im2col,stride,padding,height,width);
		INDArray col2im = Convolution.col2im(im2col,stride,padding,height,width);
		assertEquals(assertcol2im,col2im);
	}


    @Override
    public char ordering() {
        return 'f';
    }
}
