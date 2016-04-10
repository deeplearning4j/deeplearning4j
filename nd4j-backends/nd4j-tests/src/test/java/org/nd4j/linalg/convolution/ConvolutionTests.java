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


import java.util.Arrays;

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
                                                    if ((w - kw + 2 * pw) % sw != 0 || (h - kh + 2 * ph) % sh != 0)
                                                        continue;   //(w-kp+2*pw)/sw + 1 is not an integer, i.e., number of outputs doesn't fit

                                                    INDArray in = Nd4j.rand(new int[]{m, d, h, w});
                                                    assertEquals(in.data().allocationMode(), mode);
                                                    assertEquals(in.data().dataType(), type);
                                                    INDArray im2col = Convolution.im2col(in, kh, kw, sh, sw, ph, pw, false);    //Cheating, to get correct shape for input

                                                    INDArray imgOutOld = OldConvolution.col2im(im2col, sh, sw, ph, pw, h, w);
                                                    INDArray imgOutNew = Convolution.col2im(im2col, sh, sw, ph, pw, h, w);
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

        System.out.println("Assertion dimensions: " + Arrays.toString(assertion.shape()));
        assertEquals(assertion,newTest);
    }

    @Override
    public char ordering() {
        return 'f';
    }
}
