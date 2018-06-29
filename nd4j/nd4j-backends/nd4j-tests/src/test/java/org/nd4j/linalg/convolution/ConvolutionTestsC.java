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

import lombok.extern.slf4j.Slf4j;
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
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.primitives.Pair;

import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.nd4j.linalg.checkutil.NDArrayCreationUtil.getAll4dTestArraysWithShape;

/**
 * Created by agibsonccc on 9/6/14.
 */
@Slf4j
@RunWith(Parameterized.class)
public class ConvolutionTestsC extends BaseNd4jTest {

    public ConvolutionTestsC(Nd4jBackend backend) {
        super(backend);
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
        INDArray im2colAssertion = Nd4j.create(new double[] {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        5.0, 6.0, 0.0, 0.0, 0.0, 0.0, 7.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.0, 10.0,
                        0.0, 0.0, 0.0, 0.0, 11.0, 12.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 13.0, 14.0, 0.0, 0.0,
                        0.0, 0.0, 15.0, 16.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                        new int[] {2, 2, 1, 1, 6, 6});
        assertEquals(im2colAssertion, ret);
        INDArray col2ImAssertion = Nd4j.create(new double[] {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0,
                        12.0, 13.0, 14.0, 15.0, 16.0

        }, new int[] {2, 2, 2, 2});

        INDArray otherConv = Convolution.col2im(ret, 1, 1, 2, 2, 2, 2);
        assertEquals(col2ImAssertion, otherConv);

    }

    @Test
    public void testIm2Col2() {
        int kh = 2;
        int kw = 2;
        int ph = 0;
        int pw = 0;
        int sy = 2;
        int sx = 2;
        int depth = 2;
        INDArray assertion = Nd4j.create(new double[] {1, 1, 1, 1, 3, 3, 3, 3, 1, 1, 1, 1, 3, 3, 3, 3, 1, 1, 1, 1, 3, 3,
                        3, 3, 1, 1, 1, 1, 3, 3, 3, 3, 2, 2, 2, 2, 4, 4, 4, 4, 2, 2, 2, 2, 4, 4, 4, 4, 2, 2, 2, 2, 4, 4,
                        4, 4, 2, 2, 2, 2, 4, 4, 4, 4}, new int[] {1, 1, 2, 2, 4, 4});
        INDArray ret = Nd4j.create(new double[] {1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
                        4, 4, 4, 4, 4, 4, 4, 4, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
                        4, 4, 4, 4, 4, 4, 4, 4}, new int[] {1, 1, 8, 8});

        INDArray test = Convolution.im2col(ret, kh, kw, sy, sx, ph, pw, 0, false);
        assertEquals(assertion, test);

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
    public void testPooling2D_Same() {
        int[] miniBatches = {1, 3, 5};
        int[] depths = {1, 3, 5};
        int[] inHeights = {5, 21};
        int[] inWidths = {5, 21};
        int[] strideH = {1, 2};
        int[] strideW = {1, 2};
        int[] sizeW = {1, 2, 3};
        int[] sizeH = {1, 2, 3};
        int[] padH = {0};
        int[] padW = {0};
        Pooling2D.Pooling2DType[] types = new Pooling2D.Pooling2DType[]{Pooling2D.Pooling2DType.PNORM, Pooling2D.Pooling2DType.AVG, Pooling2D.Pooling2DType.MAX};

        int cnt = 0;

        for (Pooling2D.Pooling2DType type : types) {
            log.info("Trying pooling type: [{}]", type);
            for (int m : miniBatches) {
                for (int d : depths) {
                    for (int h : inHeights) {
                        for (int w : inWidths) {
                            for (int sh : strideH) {
                                for (int sw : strideW) {
                                    for (int kh : sizeH) {
                                        for (int kw : sizeW) {

                                            INDArray in = Nd4j.rand(new int[]{m, d, h, w});

                                            int[] outSize = getOutputSize(in, new int[]{kh, kw}, new int[]{sh, sw}, null, true);

                                            //Calculate padding for same mode:
                                            int pHTotal = (outSize[0]-1)*sh + kh - h;
                                            int pWTotal = (outSize[1]-1)*sw + kw - w;
                                            int padTop = pHTotal / 2;
                                            int padLeft = pWTotal / 2;

                                            INDArray col = Nd4j.create(new int[]{m, d, outSize[0], outSize[1], kh, kw}, 'c');
                                            INDArray col2 = col.permute(0, 1, 4, 5, 2, 3);
                                            //INDArray col = Nd4j.createUninitialized(new int[]{m, d, kH, kW, outSize[0], outSize[1]}, 'c');
                                            //INDArray col2 = col;

                                            Convolution.im2col(in, kh, kw, sh, sw, padTop, padLeft, true, col2);

                                            INDArray col2d = col.reshape('c', m * d * outSize[0] * outSize[1], kh * kw);

                                            INDArray output = Nd4j.create(m, d, outSize[0], outSize[1]);



                                            INDArray reduced = null;
                                            switch (type) {
                                                case PNORM:
                                                    int pnorm = 3;

                                                    Transforms.abs(col2d, false);
                                                    Transforms.pow(col2d, pnorm, false);
                                                    reduced = col2d.sum(1);
                                                    Transforms.pow(reduced, (1.0 / pnorm), false);

                                                    Convolution.pooling2D(in, kh, kw, sh, sw, padTop, padLeft, 1, 1,
                                                            true, Pooling2D.Pooling2DType.PNORM, Pooling2D.Divisor.INCLUDE_PADDING,
                                                            (double) pnorm, outSize[0], outSize[1], output);

                                                    break;
                                                case MAX:
                                                    Convolution.pooling2D(in, kh, kw, sh, sw, padTop, padLeft, 1, 1,
                                                            true, Pooling2D.Pooling2DType.MAX, Pooling2D.Divisor.INCLUDE_PADDING,
                                                            0.0, outSize[0], outSize[1], output);

                                                    reduced = col2d.max(1);
                                                    break;
                                                case AVG:

                                                    Convolution.pooling2D(in, kh, kw, sh, sw, padTop, padLeft, 1, 1,
                                                            true, Pooling2D.Pooling2DType.AVG, Pooling2D.Divisor.INCLUDE_PADDING,
                                                            0.0, outSize[0], outSize[1], output);

                                                    reduced = col2d.mean(1);
                                                    break;
                                            }

                                            reduced = reduced.reshape('c',m,d, outSize[0], outSize[1]);

                                            assertEquals("Failed opType: " + type, reduced, output);
                                            log.info("-----------");
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
    public void testMoreIm2Col2() {
        int kh = 2;
        int kw = 2;
        int ph = 0;
        int pw = 0;
        int sy = 2;
        int sx = 2;

        INDArray ret = Nd4j.create(new double[] {1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
                        4, 4, 4, 4, 4, 4, 4, 4, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
                        4, 4, 4, 4, 4, 4, 4, 4}, new int[] {1, 1, 8, 8});

        INDArray assertion = Nd4j.create(new double[] {1, 1, 1, 1, 3, 3, 3, 3, 1, 1, 1, 1, 3, 3, 3, 3, 1, 1, 1, 1, 3, 3,
                        3, 3, 1, 1, 1, 1, 3, 3, 3, 3, 2, 2, 2, 2, 4, 4, 4, 4, 2, 2, 2, 2, 4, 4, 4, 4, 2, 2, 2, 2, 4, 4,
                        4, 4, 2, 2, 2, 2, 4, 4, 4, 4}, new int[] {1, 1, 2, 2, 4, 4});
        INDArray im2colTest = Convolution.im2col(ret, kh, kw, sy, sx, ph, pw, 0, false);
        assertEquals(assertion, im2colTest);

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
    @Ignore
    public void testMaxPoolBackprop(){
        Nd4j.getRandom().setSeed(12345);

        for( int i=0; i<5; i++ ) {

            int[] inputShape = {1, 1, 4, 3};

            int[] kernel = {2, 2};
            int[] strides = {1, 1};
            int[] pad = {0, 0};
            int[] dilation = {1, 1};        //TODO non 1-1 dilation
            boolean same = true;


            String fn = "maxpool2d_bp";
            int nIArgs = 11;

            int[] a = new int[nIArgs];
            a[0] = kernel[0];
            a[1] = kernel[1];
            a[2] = strides[0];
            a[3] = strides[1];
            a[4] = pad[0];
            a[5] = pad[1];
            a[6] = dilation[0];
            a[7] = dilation[1];
            a[8] = same ? 1 : 0;
            //a[9]: Not used with max pooling
            a[10] = 0;  //For NCHW

            List<Pair<INDArray, String>> inputs = NDArrayCreationUtil.getAll4dTestArraysWithShape(12345, inputShape);

            for(Pair<INDArray,String> pIn : inputs){
                INDArray input = pIn.getFirst();
                int[] outShapeHW = getOutputSize(input, kernel, strides, pad, same);
                List<Pair<INDArray, String>> eps = NDArrayCreationUtil.getAll4dTestArraysWithShape(12345, inputShape[0], inputShape[1], outShapeHW[0], outShapeHW[1]);
                for(Pair<INDArray,String> pEps : eps){
                    INDArray epsilon = pEps.getFirst();
                    INDArray epsNext = Nd4j.create(inputShape, 'c');

                    //Runs fine with dups:
//                    input = input.dup('c');
                    epsilon = epsilon.dup('c');

                    DynamicCustomOp op = DynamicCustomOp.builder(fn)
                            .addInputs(input, epsilon)
                            .addOutputs(epsNext)
                            .addIntegerArguments(a)
                            .build();

                    Nd4j.getExecutioner().exec(op);

                    INDArray expEpsNext = expGradMaxPoolBackPropSame(input, epsilon, kernel, strides, same);

                    String msg = "input=" + pIn.getSecond() + ", eps=" + pEps.getSecond();
                    assertEquals(msg, expEpsNext, epsNext);
                }
            }
        }
    }

    public static INDArray expGradMaxPoolBackPropSame(INDArray input, INDArray gradient, int[] k, int[] s, boolean same){
        input = input.dup();
        if(!same){
            throw new UnsupportedOperationException("non-Same mode not yet supported here");
        }

        int outH = (int)Math.ceil(input.size(2)/(double)s[0]);
        int outW = (int)Math.ceil(input.size(3)/(double)s[1]);

        // FIXME: int cast
        int totalPadH = (outH-1)*s[0] + k[0] - (int) input.size(2);
        int totalPadW = (outW-1)*s[1] + k[1] - (int) input.size(3);

        int topPad = totalPadH/2;
        int bottomPad = totalPadH - topPad;
        int leftPad = totalPadW/2;
        int rightPad = totalPadW - leftPad;

        INDArray outGrad = Nd4j.create(input.shape());

        for( int m=0; m<input.size(0); m++ ){
            for( int d=0; d<input.size(1); d++ ){
                for( int y=0; y<outH; y++ ){
                    for( int x=0; x<outW; x++){

                        //First: work out the *original* position for this kernel...
                        int kTLy = y*s[0] - topPad;
                        int kTLx = x*s[1] - leftPad;

                        int[] maxPos = {kTLy,kTLx};
                        double max = -Double.MAX_VALUE;
                        for( int kY=0; kY<k[0]; kY++){
                            for( int kX=0; kX<k[1]; kX++){
                                if(kTLy + kY < 0 || kTLy + kY >= input.size(2) || kTLx + kX < 0 || kTLx + kX >= input.size(3)){
                                    //Is padding
                                    continue;
                                }
                                double v = input.getDouble(m, d, kTLy + kY, kTLx + kX);
                                if(v > max){
                                    max = v;
                                    maxPos = new int[]{kTLy + kY, kTLx + kX};
                                }
                            }
                        }
                        if(max == -Double.MAX_VALUE){
                            //All input values are padding, so can skip this input (should rarely happen)
                            continue;
                        }

                        //Now that we know *where* the max is from: add the gradient
                        double v = outGrad.getDouble(m, d, maxPos[0], maxPos[1]);
                        double toAdd = gradient.getDouble(m,d,y,x);
                        outGrad.putScalar(m, d, maxPos[0], maxPos[1], v + toAdd);
                    }
                }
            }
        }

        return outGrad;
    }



    protected static int[] getOutputSize(INDArray inputData, int[] kernel, int[] strides, int[] padding, boolean convolutionModeSame) {

        //FIXME: int cast
        int inH = (int) inputData.size(2);
        int inW = (int) inputData.size(3);

        if (convolutionModeSame != true && (kernel[0] <= 0 || kernel[0] > inH + 2 * padding[0])) {
            throw new ND4JIllegalStateException();
        }

        if (convolutionModeSame != true && (kernel[1] <= 0 || kernel[1] > inW + 2 * padding[1])) {
            throw new ND4JIllegalStateException();
        }

        if (convolutionModeSame != true) {
            if ((inH - kernel[0] + 2 * padding[0]) % strides[0] != 0) {
                double d = (inH - kernel[0] + 2 * padding[0]) / ((double) strides[0]) + 1.0;
                String str = String.format("%.2f", d);
                int truncated = (int) d;
                int sameSize = (int) Math.ceil(inH / ((double) strides[0]));
                throw new ND4JIllegalStateException();
            }

            if ((inW - kernel[1] + 2 * padding[1]) % strides[1] != 0) {
                double d = (inW - kernel[1] + 2 * padding[1]) / ((double) strides[1]) + 1.0;
                String str = String.format("%.2f", d);
                int truncated = (int) d;
                int sameSize = (int) Math.ceil(inW / ((double) strides[1]));
                throw new ND4JIllegalStateException();
            }
        } else if (convolutionModeSame) {
            //'Same' padding mode:
            //outH = ceil(inHeight / strideH)           decimal division
            //outW = ceil(inWidth / strideW)            decimal division

            //padHeightSum = ((outH - 1) * strideH + kH - inHeight)
            //padTop = padHeightSum / 2                 integer division
            //padBottom = padHeghtSum - padTop

            //padWidthSum = ((outW - 1) * strideW + kW - inWidth)
            //padLeft = padWidthSum / 2                 integer division
            //padRight = padWidthSum - padLeft

            int outH = (int) Math.ceil(inH / ((double) strides[0]));
            int outW = (int) Math.ceil(inW / ((double) strides[1]));

            return new int[] {outH, outW};
        }

        int hOut = (inH - kernel[0] + 2 * padding[0]) / strides[0] + 1;
        int wOut = (inW - kernel[1] + 2 * padding[1]) / strides[1] + 1;

        return new int[] {hOut, wOut};
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
