/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.nd4j.autodiff.opvalidation;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.junit.Test;
import org.nd4j.OpValidationSuite;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.validation.OpValidation;
import org.nd4j.autodiff.validation.TestCase;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.iter.NdIndexIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.layers.convolution.AvgPooling2D;
import org.nd4j.linalg.api.ops.impl.layers.convolution.Pooling2D;
import org.nd4j.linalg.api.ops.impl.layers.convolution.Pooling2DDerivative;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.*;
import org.nd4j.linalg.api.ops.impl.transforms.custom.Standardize;
import org.nd4j.linalg.api.ops.impl.transforms.custom.LayerNorm;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;

@Slf4j
public class LayerOpValidation extends BaseOpValidation {
    public LayerOpValidation(Nd4jBackend backend) {
        super(backend);
    }

    @Test
    public void testXwPlusB() {
        Nd4j.getRandom().setSeed(12345);

        SameDiff sameDiff = SameDiff.create();
        INDArray input = Nd4j.rand(new long[]{2, 3});
        INDArray weights = Nd4j.rand(new long[]{3, 4});
        INDArray b = Nd4j.rand(new long[]{4});

        SDVariable sdInput = sameDiff.var("input", input);
        SDVariable sdWeights = sameDiff.var("weights", weights);
        SDVariable sdBias = sameDiff.var("bias", b);

        SDVariable res = sameDiff.nn().linear(sdInput, sdWeights, sdBias);
        SDVariable loss = sameDiff.standardDeviation(res, true);

        INDArray exp = input.mmul(weights).addiRowVector(b);

        TestCase tc = new TestCase(sameDiff)
                .gradientCheck(true)
                .expectedOutput(res.getVarName(), exp);

        System.out.println(sameDiff.summary());
        System.out.println("============================");
        sameDiff.createGradFunction();
        System.out.println(sameDiff.getFunction("grad").summary());


        String err = OpValidation.validate(tc);
        assertNull(err);
    }

    @Test
    public void testReluLayer() {
        Nd4j.getRandom().setSeed(12345);

        SameDiff sameDiff = SameDiff.create();
        INDArray input = Nd4j.rand(new long[]{2, 3});
        INDArray weights = Nd4j.rand(new long[]{3, 4});
        INDArray b = Nd4j.rand(new long[]{4});

        SDVariable sdInput = sameDiff.var("input", input);
        SDVariable sdWeights = sameDiff.var("weights", weights);
        SDVariable sdBias = sameDiff.var("bias", b);

        SDVariable res = sameDiff.nn().reluLayer(sdInput, sdWeights, sdBias);
        SDVariable loss = sameDiff.standardDeviation(res, true);

        INDArray exp = input.mmul(weights).addiRowVector(b);
        Transforms.relu(exp, false);

        TestCase tc = new TestCase(sameDiff)
                .gradientCheck(true)
                .expectedOutput(res.getVarName(), exp);


        String err = OpValidation.validate(tc);
        assertNull(err);
    }

    @Test
    public void testBiasAdd() {
        Nd4j.getRandom().setSeed(12345);

        for (boolean rank1Bias : new boolean[]{false, true}) {

            SameDiff sameDiff = SameDiff.create();
            INDArray input = Nd4j.linspace(1, 8, 8, DataType.DOUBLE).reshape(new long[]{2, 4});
            INDArray b = Nd4j.linspace(1, 4, 4, DataType.DOUBLE).reshape(rank1Bias ? new long[]{4} : new long[]{1, 4}).divi(4);

            SDVariable sdInput = sameDiff.var("input", input);
            SDVariable sdBias = sameDiff.var("bias", b);

            SDVariable res = sameDiff.nn().biasAdd(sdInput, sdBias);
            SDVariable loss = sameDiff.standardDeviation(res, true);

            INDArray exp = input.addRowVector(b);

            TestCase tc = new TestCase(sameDiff)
                    .gradientCheck(true)
                    .expectedOutput(res.getVarName(), exp);

            String err = OpValidation.validate(tc);
            assertNull(err);
        }
    }

    @Test
    public void testConv2d() {
        //avg pool, batch norm, conv2d, max pool 2d, pooling2d, upsampling
        //Tested elsewhere: deconv2d, depthwise2d, LRN, sconv2d

        Nd4j.getRandom().setSeed(12345);

        int[][] inputSizes = new int[][]{{1, 3, 8, 8}}; //, {3, 6, 12, 12}};

        List<String> failed = new ArrayList<>();

        for (int i = 0; i < 8; i++) {
            for (int[] inSizeNCHW : inputSizes) {

                SameDiff sd = SameDiff.create();
                SDVariable in = null;

                int[] inSize;

                SDVariable out;
                String msg;
                switch (i) {
                    case 0:
                        //Conv2d, with bias, NCHW, same
                        msg = "0 - conv2d+bias, nchw - input " + Arrays.toString(inSizeNCHW);
                        inSize = inSizeNCHW;
                        in = sd.var("in", inSize);
                        SDVariable w0 = sd.var("w0", Nd4j.rand(new int[]{3, 3, inSizeNCHW[1], 3}).muli(10));  //kH,kW,iC,oC
                        SDVariable b0 = sd.var("b0", Nd4j.rand(new long[]{3}).muli(10));
                        out = sd.cnn().conv2d(in, w0, b0, Conv2DConfig.builder()
                                .dataFormat(Conv2DConfig.NCHW)
                                .isSameMode(true)
                                .kH(3).kW(3)
                                .sH(1).sW(1)
                                .build());
                        break;
                    case 1:
                        //Conv2d, with bias, NHWC, no same
                        msg = "1 - conv2d+bias, nhwc - input " + Arrays.toString(inSizeNCHW);
                        inSize = nchwToNhwc(inSizeNCHW);
                        in = sd.var("in", inSize);
                        SDVariable w1 = sd.var("w1", Nd4j.rand(new int[]{2, 4, inSizeNCHW[1], 3}).muli(10));  //kH,kW,nIn,nOut
                        SDVariable b1 = sd.var("b1", Nd4j.rand(new long[]{3}).muli(10));
                        out = sd.cnn().conv2d(in, w1, b1, Conv2DConfig.builder()
                                .dataFormat(Conv2DConfig.NHWC)
                                .isSameMode(false)
                                .kH(2).kW(4)
                                .sH(2).sW(2)
                                .build());
                        break;
                    case 2:
                        //Conv2d, no bias, NCHW
                        msg = "2 - conv2d, no bias, nchw - input " + Arrays.toString(inSizeNCHW);
                        inSize = inSizeNCHW;
                        in = sd.var("in", inSize);
                        SDVariable w2 = sd.var("w0", Nd4j.rand(new int[]{1, 3, inSizeNCHW[1], 3}).muli(10));  ////kH,kW,iC,oC
                        out = sd.cnn().conv2d(in, w2, Conv2DConfig.builder()
                                .dataFormat(Conv2DConfig.NCHW)
                                .isSameMode(true)
                                .kH(1).kW(3)
                                .sH(1).sW(2)
                                .build());
                        break;
                    case 3:
                        //Avg pool, NCHW
                        msg = "3 - avg pool, NCHW, same - input " + Arrays.toString(inSizeNCHW);
                        inSize = inSizeNCHW;
                        in = sd.var("in", inSize);
                        out = sd.cnn().avgPooling2d(in, Pooling2DConfig.builder()
                                .isNHWC(true)
                                .isSameMode(true)
                                .kH(2).kW(2)
                                .sH(1).sW(1)
                                .build());
                        break;
                    case 4:
                        //Avg pool, NHWC, not same
                        msg = "3 - avg pool, NHWC, not same - input " + Arrays.toString(inSizeNCHW);
                        inSize = nchwToNhwc(inSizeNCHW);
                        in = sd.var("in", inSize);
                        out = sd.cnn().avgPooling2d(in, Pooling2DConfig.builder()
                                .isNHWC(true)
                                .isSameMode(false)
                                .kH(3).kW(2)
                                .sH(2).sW(2)
                                .build());
                        break;
                    case 5:
                        //Avg pool, NCHW
                        msg = "5 - avg pool, NCHW, same - input " + Arrays.toString(inSizeNCHW);
                        inSize = inSizeNCHW;
                        in = sd.var("in", inSize);
                        out = sd.cnn().maxPooling2d(in, Pooling2DConfig.builder()
                                .isNHWC(false)
                                .isSameMode(true)
                                .kH(2).kW(2)
                                .sH(1).sW(1)
                                .build());
                        break;
                    case 6:
                        //Max pool, NHWC, not same
                        msg = "6 - avg pool, NHWC, not same - input " + Arrays.toString(inSizeNCHW);
                        inSize = inSizeNCHW;
                        in = sd.var("in", inSize);
                        out = sd.cnn().maxPooling2d(in, Pooling2DConfig.builder()
                                .isNHWC(true)
                                .isSameMode(false)
                                .kH(3).kW(2)
                                .sH(2).sW(2)
                                .build());
                        break;
                    case 7:
                        //Upsampling
                        msg = "7 - upsampling2d, NCHW, 2x2 - " + Arrays.toString(inSizeNCHW);
                        inSize = inSizeNCHW;
                        in = sd.var("in", inSize);
                        out = sd.cnn().upsampling2d(in, true, 2, 2);
                        break;
                    default:
                        throw new RuntimeException();

                }

                INDArray inArr = Nd4j.rand(inSize).muli(10);
                in.setArray(inArr);
                SDVariable loss = sd.standardDeviation("loss", out, true);

                log.info("Starting test: " + msg);
                TestCase tc = new TestCase(sd);
                String error = OpValidation.validate(tc);
                if (error != null) {
                    failed.add(msg);
                }

            }
        }

        assertEquals(failed.toString(), 0, failed.size());
    }

    @Test
    public void testLrn2d() {
        Nd4j.getRandom().setSeed(12345);

        int[][] inputSizes = new int[][]{{1, 3, 8, 8}, {3, 6, 12, 12}};

        List<String> failed = new ArrayList<>();

        for (int[] inSizeNCHW : inputSizes) {

            SameDiff sd = SameDiff.create();
            SDVariable in = null;

            int[] inSize;

            //LRN
            String msg = "LRN with NCHW - input" + Arrays.toString(inSizeNCHW);
            inSize = inSizeNCHW;
            in = sd.var("in", inSize);
            SDVariable out = sd.cnn().localResponseNormalization(in, LocalResponseNormalizationConfig.builder()
                    .depth(3)
                    .bias(1)
                    .alpha(1)
                    .beta(0.5)
                    .build());

            INDArray inArr = Nd4j.rand(inSize).muli(10);
            in.setArray(inArr);
            SDVariable loss = sd.mean("loss", out);

            log.info("Starting test: " + msg);
            TestCase tc = new TestCase(sd);
            String error = OpValidation.validate(tc);
            if (error != null) {
                failed.add(msg);
            }

        }
        assertEquals(failed.toString(), 0, failed.size());
    }

    @Test
    public void testIm2Col() {
        OpValidationSuite.ignoreFailing();      //TEMPORARY DUE TO JVM CRASH: https://github.com/deeplearning4j/deeplearning4j/issues/6873
        Nd4j.getRandom().setSeed(12345);

        int[][] inputSizes = new int[][]{{1, 3, 8, 8}, {3, 6, 12, 12}};

        List<String> failed = new ArrayList<>();

        for (int[] inSizeNCHW : inputSizes) {

            SameDiff sd = SameDiff.create();
            SDVariable var = sd.var("in", Nd4j.rand(DataType.DOUBLE, inSizeNCHW));
            SDVariable im2col = sd.cnn().im2Col(var, Conv2DConfig.builder()
                    .kH(2).kW(2)
                    .sH(1).sW(1)
                    .isSameMode(true)
                    .build());

            SDVariable loss = sd.standardDeviation("loss", im2col, true);

            String msg = Arrays.toString(inSizeNCHW);

            TestCase tc = new TestCase(sd).testName(msg);
            String error = OpValidation.validate(tc);
            if (error != null) {
                failed.add(msg);
            }
        }

        assertEquals(failed.toString(), 0, failed.size());
    }


    private static int[] nchwToNhwc(int[] in) {
        return new int[]{in[0], in[2], in[3], in[1]};
    }


    @Test
    public void testOutputShape() {
        long[] inSize = {1, 8, 8, 3};

        SameDiff sd = SameDiff.create();
        SDVariable in = sd.var("in", inSize);
//        SDVariable out = sd.avgPooling2d(in, );

//        Pooling2DConfig conf = Pooling2DConfig.builder()
//                .isNHWC(false)
//                .isSameMode(false)
//                .kH(2).kW(2)
//                .sW(1).sH(1)
//                .build();

        Pooling2DConfig conf = Pooling2DConfig.builder()
                .isNHWC(true)   //***NHWC
                .isSameMode(false)
                .kH(3).kW(2)
                .sH(2).sW(2)
                .build();

        INDArray input = Nd4j.create(inSize);
        AvgPooling2D avgPooling2D = AvgPooling2D.builder()
                .arrayInput(input)
                .config(conf)
                .build();

        val outSizes = Nd4j.getExecutioner().calculateOutputShape(avgPooling2D);

        assertEquals(1, outSizes.size());

        //NO SAME: out = (in - k + 2*p)/s + 1;
        int outH = (8 - 3) / 2 + 1;
        int outW = (8 - 2) / 2 + 1;
        long[] exp = new long[]{1, outH, outW, 3};    //NHWC

        assertEquals(1, outSizes.size());
        assertArrayEquals(exp, outSizes.get(0).getShape());

        INDArray grad = Nd4j.create(exp);


        //Test backprop:
        Pooling2DDerivative avg2dDeriv = Pooling2DDerivative.derivativeBuilder()
                .arrayInputs(new INDArray[]{input, grad})
                .config(conf)
                .build();

        val outSizesBP = Nd4j.getExecutioner().calculateOutputShape(avg2dDeriv);
        assertEquals(1, outSizesBP.size());

        assertArrayEquals(inSize, outSizesBP.get(0).getShape());
    }


    @Test
    public void testAvgPool() {
        long[] inSize = {1, 8, 8, 3};  //NHWC

        Pooling2DConfig conf = Pooling2DConfig.builder()
                .isNHWC(true)   //***NHWC
                .isSameMode(false)
                .kH(3).kW(2)
                .sH(2).sW(2)
                .type(Pooling2D.Pooling2DType.AVG)
                .build();

        INDArray input = Nd4j.create(inSize);
        AvgPooling2D avgPooling2D = AvgPooling2D.builder()
                .arrayInput(input)
                .config(conf)
                .build();

        val outSizes = Nd4j.getExecutioner().calculateOutputShape(avgPooling2D);
        assertEquals(1, outSizes.size());

        //NO SAME: out = (in - k + 2*p)/s + 1;
        int outH = (8 - 3) / 2 + 1;
        int outW = (8 - 2) / 2 + 1;
        long[] exp = new long[]{1, outH, outW, 3};    //NHWC

        assertEquals(1, outSizes.size());
        assertArrayEquals(exp, outSizes.get(0).getShape());

        INDArray grad = Nd4j.create(exp);

        //Test backprop:
        Pooling2DDerivative avg2dDeriv = Pooling2DDerivative.derivativeBuilder()
                .arrayInputs(new INDArray[]{input, grad})           //Original input, and output gradient (eps - same shape as output)
                .arrayOutputs(new INDArray[]{Nd4j.create(inSize)})  //Output for BP: same shape as original input
                .config(conf)
                .build();

        val outSizesBP = Nd4j.getExecutioner().calculateOutputShape(avg2dDeriv);
        assertEquals(1, outSizesBP.size());
        assertArrayEquals(inSize, outSizesBP.get(0).getShape());

        Nd4j.getExecutioner().execAndReturn(avg2dDeriv);
    }


    private static int[] ncdhwToNdhwc(int[] in) {
        return new int[]{in[0], in[2], in[3], in[4], in[1]};
    }

    @Test
    public void testConv3d() {
        //Pooling3d, Conv3D, batch norm
        Nd4j.getRandom().setSeed(12345);

        //NCDHW format
        int[][] inputSizes = new int[][]{{2, 3, 4, 5, 5}};

        List<String> failed = new ArrayList<>();

        for (int[] inSizeNCDHW : inputSizes) {
            for (boolean ncdhw : new boolean[]{true, false}) {
                int nIn = inSizeNCDHW[1];
                int[] shape = (ncdhw ? inSizeNCDHW : ncdhwToNdhwc(inSizeNCDHW));

                for (int i = 0; i < 5; i++) {
                    SameDiff sd = SameDiff.create();
                    SDVariable in = sd.var("in", shape);

                    SDVariable out;
                    String msg;
                    switch (i) {
                        case 0:
                            //Conv3d, with bias, same
                            msg = "0 - conv3d+bias+same, ncdhw=" + ncdhw + " - input " + Arrays.toString(shape);
                            SDVariable w0 = sd.var("w0", Nd4j.rand(new int[]{2, 2, 2, nIn, 3}).muli(10));  //[kD, kH, kW, iC, oC]
                            SDVariable b0 = sd.var("b0", Nd4j.rand(new long[]{3}).muli(10));
                            out = sd.cnn().conv3d(in, w0, b0, Conv3DConfig.builder()
                                    .dataFormat(ncdhw ? Conv3DConfig.NCDHW : Conv3DConfig.NDHWC)
                                    .isSameMode(true)
                                    .kH(2).kW(2).kD(2)
                                    .sD(1).sH(1).sW(1)
                                    .build());
                            break;
                        case 1:
                            //Conv3d, no bias, no same
                            msg = "1 - conv3d+no bias+no same, ncdhw=" + ncdhw + " - input " + Arrays.toString(shape);
                            SDVariable w1 = sd.var("w1", Nd4j.rand(new int[]{2, 2, 2, nIn, 3}).muli(10));  //[kD, kH, kW, iC, oC]
                            out = sd.cnn().conv3d(in, w1, Conv3DConfig.builder()
                                    .dataFormat(ncdhw ? Conv3DConfig.NCDHW : Conv3DConfig.NDHWC)
                                    .isSameMode(false)
                                    .kH(2).kW(2).kD(2)
                                    .sD(1).sH(1).sW(1)
                                    .build());
                            break;
                        case 2:
                            //pooling3d - average, same
                            msg = "2 - pooling 3d, average, same";
                            out = sd.cnn().avgPooling3d(in, Pooling3DConfig.builder()
                                    .kH(2).kW(2).kD(2)
                                    .sH(1).sW(1).sD(1)
                                    .isSameMode(false)
                                    .build());
                            break;
                        case 3:
                            //pooling 3d - max, no same
                            msg = "3 - pooling 3d, max, no same";
                            out = sd.cnn().avgPooling3d(in, Pooling3DConfig.builder()
                                    .kH(2).kW(2).kD(2)
                                    .sH(1).sW(1).sD(1)
                                    .isSameMode(true)
                                    .build());
                            break;
                        case 4:
                            //Deconv3d
                            msg = "4 - deconv3d, ncdhw=" + ncdhw;
                            SDVariable wDeconv = sd.var(Nd4j.rand(new int[]{2, 2, 2, 3, nIn}));  //[kD, kH, kW, oC, iC]
                            SDVariable bDeconv = sd.var(Nd4j.rand(new int[]{3}));
                            out = sd.cnn().deconv3d("Deconv3d", in, wDeconv, bDeconv, DeConv3DConfig.builder()
                                    .kD(2).kH(2).kW(2)
                                    .isSameMode(true)
                                    .dataFormat(ncdhw ? DeConv3DConfig.NCDHW : DeConv3DConfig.NDHWC)
                                    .build());
                            break;
                        case 5:
                            //Batch norm - 3d input
                            throw new RuntimeException("Batch norm test not yet implemented");
                        default:
                            throw new RuntimeException();
                    }

                    INDArray inArr = Nd4j.rand(shape).muli(10);
                    in.setArray(inArr);
                    SDVariable loss = sd.standardDeviation("loss", out, true);

                    log.info("Starting test: " + msg);
                    TestCase tc = new TestCase(sd);
                    tc.testName(msg);
                    String error = OpValidation.validate(tc);
                    if (error != null) {
                        failed.add(name);
                    }
                }
            }
        }

        assertEquals(failed.toString(), 0, failed.size());
    }


    @Test
    public void testDepthWiseConv2dBasic() {
        int nIn = 3;
        int depthWise = 4;
        int kH = 2;
        int kW = 2;

        int mb = 3;
        int imgH = 28;
        int imgW = 28;


        SameDiff sd = SameDiff.create();
        INDArray depthWeightArr = Nd4j.create(kH, kW, nIn, depthWise);

        INDArray bArr = Nd4j.create(1, depthWise * nIn);
        INDArray inArr = Nd4j.create(mb, nIn, imgH, imgW);

        SDVariable in = sd.var("in", inArr);
        SDVariable dW = sd.var("dW", depthWeightArr);
        SDVariable b = sd.var("b", bArr);

        SDVariable[] vars = new SDVariable[]{in, dW, b};

        Conv2DConfig c = Conv2DConfig.builder()
                .kH(kH).kW(kW)
                .pH(0).pW(0)
                .sH(1).sW(1)
                .dH(1).dW(1)
                .isSameMode(false)
                .build();

        SDVariable out = sd.cnn().sconv2d(vars, c);
        out = sd.nn().tanh("out", out);

        INDArray outArr = sd.execAndEndResult();
        //Expected output size: out = (in - k + 2*p)/s + 1 = (28-2+0)/1+1 = 27
        val outShape = outArr.shape();
        assertArrayEquals(new long[]{mb, depthWise * nIn, 27, 27}, outShape);
    }

    @Test
    public void testSeparableConv2dBasic() {
        Nd4j.getRandom().setSeed(12345);
        int nIn = 2;
        int nOut = 3;
        int kH = 2;
        int kW = 2;

        int mb = 2;
        int imgH = 8;
        int imgW = 8;

        int depthWise = 3;

        SameDiff sd = SameDiff.create();
        INDArray depthWeightArr = Nd4j.rand(new int[]{kH, kW, nIn, depthWise});
        INDArray pointWeightArr = Nd4j.rand(new int[]{1, 1, nIn * depthWise, nOut});

        INDArray bArr = Nd4j.rand(new int[]{nOut});
        INDArray inArr = Nd4j.rand(new int[]{mb, nIn, imgH, imgW});

        SDVariable in = sd.var("in", inArr);
        SDVariable dW = sd.var("dW", depthWeightArr);
        SDVariable pW = sd.var("pW", pointWeightArr);
        SDVariable b = sd.var("b", bArr);

        SDVariable[] vars = new SDVariable[]{in, dW, pW, b};

        Conv2DConfig c = Conv2DConfig.builder()
                .kH(kH).kW(kW)
                .pH(0).pW(0)
                .sH(1).sW(1)
                .dH(1).dW(1)
                .isSameMode(false)
                .dataFormat(Conv2DConfig.NCHW)
                .build();

        SDVariable out = sd.cnn().sconv2d(vars, c);
        out = sd.nn().tanh("out", out);

        INDArray outArr = sd.execAndEndResult();
        //Expected output size: out = (in - k + 2*p)/s + 1 = (8-2+0)/1+1 = 7
        val outShape = outArr.shape();
        assertArrayEquals(new long[]{mb, nOut, 7, 7}, outShape);

        SDVariable loss = out.std(true);

//        System.out.println(sd.summary());
//        System.out.println("--------------------------");
//        sd.createGradFunction();
//        System.out.println(sd.getFunction("grad").summary());

        //Gradient check:
        TestCase tc = new TestCase(sd);
        String err = OpValidation.validate(tc);
        assertNull(err);
    }

    @Test
    public void testDeconv2dBasic() {
        int nIn = 2;
        int nOut = 3;
        int kH = 2;
        int kW = 2;

        int mb = 2;
        int imgH = 8;
        int imgW = 8;

        SameDiff sd = SameDiff.create();
        INDArray wArr = Nd4j.rand(new int[]{kH, kW, nOut, nIn}); //Libnd4j expected weights format: [kH, kW, cOut, cIn]
        INDArray bArr = Nd4j.rand(new long[]{nOut});
        INDArray inArr = Nd4j.rand(new long[]{mb, nIn, imgH, imgW});

        SDVariable in = sd.var("in", inArr);
        SDVariable w = sd.var("W", wArr);
        SDVariable b = sd.var("b", bArr);

        SDVariable[] vars = new SDVariable[]{in, w, b};

        DeConv2DConfig deconv = DeConv2DConfig.builder()
                .kH(kH).kW(kW)
                .pH(0).pW(0)
                .sH(1).sW(1)
                .dH(1).dW(1)
                .isSameMode(false)
                .build();

        SDVariable out = sd.cnn().deconv2d(vars, deconv);
        out = sd.nn().tanh("out", out);

        INDArray outArr = sd.execAndEndResult();
        //Expected output size: out = (in + k + 2*p)/ s - 1 = (8 + 2+0)/1 - 1 = 9
        val outShape = outArr.shape();
        assertArrayEquals(new long[]{mb, nOut, 9, 9}, outShape);

        SDVariable loss = out.std(true);
        //Gradient check:
        TestCase tc = new TestCase(sd);
        String err = OpValidation.validate(tc);
        assertNull(err);
    }


    @Test
    public void testConv2dBasic() {
        int nIn = 3;
        int nOut = 4;
        int kH = 2;
        int kW = 2;

        int mb = 3;
        int imgH = 28;
        int imgW = 28;

        SameDiff sd = SameDiff.create();
        INDArray wArr = Nd4j.create(kH, kW, nIn, nOut);
        INDArray bArr = Nd4j.create(1, nOut);
        INDArray inArr = Nd4j.create(mb, nIn, imgH, imgW);

        SDVariable in = sd.var("in", inArr);
        SDVariable w = sd.var("W", wArr);
        SDVariable b = sd.var("b", bArr);

        //Order: https://github.com/deeplearning4j/libnd4j/blob/6c41ea5528bb1f454e92a9da971de87b93ff521f/include/ops/declarable/generic/convo/conv2d.cpp#L20-L22
        //in, w, b - bias is optional
        SDVariable[] vars = new SDVariable[]{in, w, b};

        Conv2DConfig c = Conv2DConfig.builder()
                .kH(kH).kW(kW)
                .pH(0).pW(0)
                .sH(1).sW(1)
                .dH(1).dW(1)
                .isSameMode(false)
                .build();

        SDVariable out = sd.cnn().conv2d(vars, c);
        out = sd.nn().tanh("out", out);

        INDArray outArr = sd.execAndEndResult();
        //Expected output size: out = (in - k + 2*p)/s + 1 = (28-2+0)/1+1 = 27
        val outShape = outArr.shape();
        assertArrayEquals(new long[]{mb, nOut, 27, 27}, outShape);
        // sd.execBackwards(); // TODO: test failing here
    }

    @Test
    public void testMaxPooling2dBasic() {
        Nd4j.getRandom().setSeed(12345);
        int nIn = 3;
        int kH = 2;
        int kW = 2;

        int mb = 3;
        int imgH = 8;
        int imgW = 8;

        SameDiff sd = SameDiff.create();
        INDArray inArr = Nd4j.rand(new int[]{mb, nIn, imgH, imgW});

        SDVariable in = sd.var("in", inArr);

        Pooling2DConfig pooling2DConfig = Pooling2DConfig.builder()
                .kH(kH).kW(kW)
                .pH(0).pW(0)
                .sH(1).sW(1)
                .dH(1).dW(1)
                .isSameMode(false)
                .build();

        SDVariable outPool = sd.cnn().maxPooling2d(in, pooling2DConfig);
        SDVariable out = sd.nn().tanh("out", outPool);

        INDArray outArr = sd.execAndEndResult();
        val outShape = outArr.shape();
        // oH = (iH - (kH + (kH-1)*(dH-1)) + 2*pH)/sH + 1;
        assertArrayEquals(new long[]{mb, nIn, 7, 7}, outShape);

        SDVariable loss = out.std(true);

        INDArray exp = Nd4j.create(mb, nIn, 7, 7);
        NdIndexIterator iter = new NdIndexIterator(mb, nIn, 7, 7);
        while (iter.hasNext()) {
            long[] next = iter.next();
            double max = max(inArr.getDouble(next),
                    inArr.getDouble(next[0], next[1], next[2] + 1, next[3]),
                    inArr.getDouble(next[0], next[1], next[2], next[3] + 1),
                    inArr.getDouble(next[0], next[1], next[2] + 1, next[3] + 1));
            exp.putScalar(next, max);
        }

        assertNull(OpValidation.validate(new TestCase(sd)
                .expected(outPool, exp)));
    }

    private double max(double... in) {
        double max = -Double.MAX_VALUE;
        for (double d : in) {
            if (d > max)
                max = d;
        }
        return max;
    }

    @Test
    public void testAvgPooling2dBasic() {
        Nd4j.getRandom().setSeed(12345);
        int nIn = 3;
        int kH = 2;
        int kW = 2;

        int mb = 3;
        int imgH = 8;
        int imgW = 8;

        SameDiff sd = SameDiff.create();
        INDArray inArr = Nd4j.rand(new int[]{mb, nIn, imgH, imgW});

        SDVariable in = sd.var("in", inArr);

        Pooling2DConfig pooling2DConfig = Pooling2DConfig.builder()
                .kH(kH).kW(kW)
                .pH(0).pW(0)
                .sH(1).sW(1)
                .dH(1).dW(1)
                .isSameMode(false)
                .build();

        SDVariable outPool = sd.cnn().avgPooling2d(in, pooling2DConfig);
        SDVariable out = sd.nn().tanh("out", outPool);

        INDArray outArr = sd.execAndEndResult();
        val outShape = outArr.shape();
        // oH = (iH - (kH + (kH-1)*(dH-1)) + 2*pH)/sH + 1;
        assertArrayEquals(new long[]{mb, nIn, 7, 7}, outShape);

        SDVariable loss = out.std(true);

        INDArray exp = Nd4j.create(mb, nIn, 7, 7);
        NdIndexIterator iter = new NdIndexIterator(mb, nIn, 7, 7);
        while (iter.hasNext()) {
            long[] next = iter.next();
            double avg = (inArr.getDouble(next) + inArr.getDouble(next[0], next[1], next[2] + 1, next[3])
                    + inArr.getDouble(next[0], next[1], next[2], next[3] + 1)
                    + inArr.getDouble(next[0], next[1], next[2] + 1, next[3] + 1)) / 4.0;
            exp.putScalar(next, avg);
        }

        assertNull(OpValidation.validate(new TestCase(sd)
                .expected(outPool, exp)));

    }

    @Test
    public void testAvgPooling3dBasic() {
        int nIn = 3;
        int kH = 2;
        int kW = 2;
        int kD = 2;

        int mb = 3;
        int imgH = 8;
        int imgW = 8;
        int imgD = 8;

        SameDiff sd = SameDiff.create();
        INDArray inArr = Nd4j.rand(new long[]{mb, nIn, imgD, imgH, imgW});

        SDVariable in = sd.var("in", inArr);

        Pooling3DConfig pooling3DConfig = Pooling3DConfig.builder()
                .kH(kH).kW(kW).kD(kD)
                .pH(0).pW(0).pD(0)
                .sH(1).sW(1).sD(1)
                .dH(1).dW(1).dD(1)
                .isSameMode(false)
                .isNCDHW(true)
                .build();

        SDVariable out = sd.cnn().avgPooling3d(in, pooling3DConfig);
        out = sd.nn().tanh("out", out);

        INDArray outArr = sd.execAndEndResult();
        val outShape = outArr.shape();
        // oH = (iH - (kH + (kH-1)*(dH-1)) + 2*pH)/sH + 1;
        assertArrayEquals(new long[]{mb, nIn, 7, 7, 7}, outShape);

        SDVariable loss = out.std(true);
        //Gradient check:
        TestCase tc = new TestCase(sd);
        String err = OpValidation.validate(tc);
        assertNull(err);
    }

    @Test
    public void testMaxPooling3dBasic() {
        int nIn = 3;
        int kH = 2;
        int kW = 2;
        int kD = 2;

        int mb = 3;
        int imgH = 28;
        int imgW = 28;
        int imgD = 28;

        SameDiff sd = SameDiff.create();
        INDArray inArr = Nd4j.create(mb, nIn, imgD, imgH, imgW);

        SDVariable in = sd.var("in", inArr);

        Pooling3DConfig pooling3DConfig = Pooling3DConfig.builder()
                .kH(kH).kW(kW).kD(kD)
                .pH(0).pW(0).pD(0)
                .sH(1).sW(1).sD(1)
                .dH(1).dW(1).dD(1)
                .isSameMode(false)
                .build();

        SDVariable out = sd.cnn().maxPooling3d(in, pooling3DConfig);
        out = sd.nn().tanh("out", out);

        INDArray outArr = sd.execAndEndResult();
        val outShape = outArr.shape();
        // oH = (iH - (kH + (kH-1)*(dH-1)) + 2*pH)/sH + 1;
        assertArrayEquals(new long[]{mb, nIn, 27, 27, 27}, outShape);
    }

    @Test
    public void testConv1dBasic() {
        int nIn = 3;
        int nOut = 4;
        int k = 2;
        int mb = 3;
        int img = 28;

        SameDiff sd = SameDiff.create();
        INDArray wArr = Nd4j.create(k, nIn, nOut);
        INDArray inArr = Nd4j.create(mb, nIn, img);

        SDVariable in = sd.var("in", inArr);
        SDVariable w = sd.var("W", wArr);

        SDVariable[] vars = new SDVariable[]{in, w};

        Conv1DConfig conv1DConfig = Conv1DConfig.builder()
                .k(k).p(0).s(1)
                .isSameMode(false)
                .build();

        SDVariable out = sd.cnn().conv1d(in, w, conv1DConfig);
        out = sd.nn().tanh("out", out);

        INDArray outArr = sd.execAndEndResult();
        INDArray iOut = out.getArr();
        //Expected output size: out = (in - k + 2*p)/s + 1 = (28-2+0)/1+1 = 27
        val outShape = outArr.shape();
        assertArrayEquals(new long[]{mb, nOut, 27}, outShape);
    }


    @Test
    public void testConv3dBasic() {
        int nIn = 3;
        int nOut = 4;
        int kH = 2;
        int kW = 2;
        int kD = 2;

        int mb = 3;
        int imgH = 8;
        int imgW = 8;
        int imgT = 8;

        SameDiff sd = SameDiff.create();
        INDArray wArr = Nd4j.rand(new int[]{kD, kH, kW, nIn, nOut});
        INDArray bArr = Nd4j.rand(1, nOut);
        INDArray inArr = Nd4j.rand(new int[]{mb, nIn, imgT, imgH, imgW});

        SDVariable in = sd.var("in", inArr);
        SDVariable w = sd.var("W", wArr);
        SDVariable b = sd.var("b", bArr);

        Conv3DConfig conv3DConfig = Conv3DConfig.builder()
                .kH(kH).kW(kW).kD(kD)
                .sD(1).sH(1).sW(1)
                .dH(1).dW(1).dD(1)
                .isSameMode(true)
                .biasUsed(false)
                .dataFormat(Conv3DConfig.NCDHW)
                .build();

        SDVariable out = sd.cnn().conv3d(in, w, b, conv3DConfig);
        out = sd.nn().tanh("out", out);

        INDArray outArr = sd.execAndEndResult();
        //Expected output size, NOT same mode: out = (in - k)/d + 1 = (28-2+0)/1+1 = 27
        //Expected output size, WITH same mode: out = in/stride
        val outShape = outArr.shape();
        assertArrayEquals(new long[]{mb, nOut, 8, 8, 8}, outShape);

        SDVariable loss = out.std(true);
        //Gradient check:
        TestCase tc = new TestCase(sd);
        String err = OpValidation.validate(tc);
        assertNull(err);
    }

    @Test
    public void testLayerNorm() {
        final INDArray random = Nd4j.rand(new int[]{10, 4});
        final INDArray standardized = random.ulike();
        Nd4j.getExecutioner().exec(new Standardize(random, standardized, 1));

        final INDArray gain = Nd4j.rand(new int[]{1, 4});
        final INDArray bias = Nd4j.rand(new int[]{1, 4});
        final INDArray res = standardized.mulRowVector(gain).addRowVector(bias);
        final INDArray expOut = res.norm1();

        final int[] axis = new int[]{1};
        SameDiff sd = SameDiff.create();
        SDVariable sdInput = sd.var("input", standardized);
        SDVariable sdGain = sd.var("gain", gain);
        SDVariable sdBias = sd.var("bias", bias);
        SDVariable out = sd.nn.layerNorm(sdInput, sdGain, sdBias, axis);
        out.norm1("out");

        String err = OpValidation.validate(new TestCase(sd)
                .expectedOutput("out", expOut)
                .gradientCheck(true));
        assertNull(err, err);
    }

    @Test
    public void testLayerNormOP() {
        final INDArray random = Nd4j.rand(new int[]{10, 4});
        final INDArray standardized = random.ulike();
        Nd4j.getExecutioner().exec(new Standardize(random, standardized, 1));

        final INDArray gain = Nd4j.rand(new int[]{1, 4});
        final INDArray bias = Nd4j.rand(new int[]{1, 4});
        final INDArray res = standardized.mulRowVector(gain).addRowVector(bias);

        final INDArray output = Nd4j.zerosLike(res);
        Nd4j.getExecutioner().exec(new LayerNorm(standardized, gain, bias, output, 1));

        assertEquals(res, output);
    }

    @Test
    public void testLayerNormNoBias() {
        final INDArray random = Nd4j.rand(new int[]{10, 4});
        final INDArray standardized = random.ulike();
        Nd4j.getExecutioner().exec(new Standardize(random, standardized, 1));

        final INDArray gain = Nd4j.rand(new int[]{1, 4});
        final INDArray res = standardized.mulRowVector(gain);
        final INDArray expOut = res.norm1();

        final int[] axis = new int[]{1};
        SameDiff sd = SameDiff.create();
        SDVariable sdInput = sd.var("input", standardized);
        SDVariable sdGain = sd.var("gain", gain);
        SDVariable out = sd.nn.layerNorm(sdInput, sdGain, axis);
        out.norm1("out");

        String err = OpValidation.validate(new TestCase(sd)
                .expectedOutput("out", expOut)
                .gradientCheck(true));
        assertNull(err, err);
    }

    @Test
    public void testLayerNormOPNoBias() {
        final INDArray random = Nd4j.rand(new int[]{10, 4});
        final INDArray standardized = random.ulike();
        Nd4j.getExecutioner().exec(new Standardize(random, standardized, 1));

        final INDArray gain = Nd4j.rand(new int[]{1, 4});
        final INDArray res = standardized.mulRowVector(gain);

        final INDArray output = Nd4j.zerosLike(res);
        Nd4j.getExecutioner().exec(new LayerNorm(standardized, gain, output, 1));

        assertEquals(res, output);
    }

    @Test
    public void testLayerNormNoDeviation() {
        final INDArray random = Nd4j.rand(new int[]{10, 4});
        for (int i = 0; i < 4; i++) {
            random.putScalar(1,i, 7);
        }

        final INDArray standardized = random.ulike();
        Nd4j.getExecutioner().exec(new Standardize(random, standardized, 1));

        final INDArray gain = Nd4j.rand(new int[]{1, 4});
        final INDArray bias = Nd4j.rand(new int[]{1, 4});
        final INDArray res = standardized.mulRowVector(gain).addRowVector(bias);
        final INDArray expOut = res.norm1();

        final int[] axis = new int[]{1};
        SameDiff sd = SameDiff.create();
        SDVariable sdInput = sd.var("input", standardized);
        SDVariable sdGain = sd.var("gain", gain);
        SDVariable sdBias = sd.var("bias", bias);
        SDVariable out = sd.nn.layerNorm(sdInput, sdGain, sdBias, axis);
        out.norm1("out");

        String err = OpValidation.validate(new TestCase(sd)
                .expectedOutput("out", expOut)
                .gradCheckMask(Collections.singletonMap("input", random.neq(7)))
                .gradientCheck(true));
        assertNull(err, err);
    }

    @Test(expected = IllegalStateException.class)
    public void exceptionThrown_WhenConv1DConfigInvalid() {
        int nIn = 3;
        int nOut = 4;
        int k = 2;
        int mb = 3;
        int img = 28;

        SameDiff sd = SameDiff.create();
        INDArray wArr = Nd4j.create(k, nIn, nOut);
        INDArray inArr = Nd4j.create(mb, nIn, img);

        SDVariable in = sd.var("in", inArr);
        SDVariable w = sd.var("W", wArr);

        SDVariable[] vars = new SDVariable[]{in, w};

        Conv1DConfig conv1DConfig = Conv1DConfig.builder()
                .k(k).p(-1).s(0)
                .isSameMode(false)
                .build();

        SDVariable out = sd.cnn().conv1d(in, w, conv1DConfig);

    }

    @Test(expected = IllegalStateException.class)
    public void exceptionThrown_WhenConv2DConfigInvalid() {

        Nd4j.getRandom().setSeed(12345);

        SameDiff sd = SameDiff.create();
        SDVariable in = null;

        int[] inSizeNCHW = {1, 3, 8, 8};

        String msg = "0 - conv2d+bias, nchw - input " + Arrays.toString(inSizeNCHW);
        SDVariable w0 = sd.var("w0", Nd4j.rand(new int[]{3, 3, inSizeNCHW[1], 3}).muli(10));  //kH,kW,iC,oC
        SDVariable b0 = sd.var("b0", Nd4j.rand(new long[]{3}).muli(10));
        SDVariable out = sd.cnn().conv2d(in, w0, b0, Conv2DConfig.builder()
                .dataFormat(Conv2DConfig.NCHW)
                .isSameMode(true)
                .kH(3).kW(-3)
                .sH(1).sW(0)
                .build());
    }

    @Test(expected = IllegalStateException.class)
    public void exceptionThrown_WhenConf3DInvalid() {
        Nd4j.getRandom().setSeed(12345);

        //NCDHW format
        int[] inSizeNCDHW = {2, 3, 4, 5, 5};

        List<String> failed = new ArrayList<>();

        for (boolean ncdhw : new boolean[]{true, false}) {
                int nIn = inSizeNCDHW[1];
                int[] shape = (ncdhw ? inSizeNCDHW : ncdhwToNdhwc(inSizeNCDHW));

                SameDiff sd = SameDiff.create();
                SDVariable in = sd.var("in", shape);

                SDVariable out;
                String msg = "0 - conv3d+bias+same, ncdhw=" + ncdhw + " - input " + Arrays.toString(shape);

                SDVariable w0 = sd.var("w0", Nd4j.rand(new int[]{2, 2, 2, nIn, 3}).muli(10));  //[kD, kH, kW, iC, oC]
                SDVariable b0 = sd.var("b0", Nd4j.rand(new long[]{3}).muli(10));
                out = sd.cnn().conv3d(in, w0, b0, Conv3DConfig.builder()
                        .dataFormat(ncdhw ? Conv3DConfig.NCDHW : Conv3DConfig.NDHWC)
                        .isSameMode(true)
                        .kH(2).kW(2).kD(2)
                        .sD(1).sH(1).sW(-1).dW(-1)
                        .build());
        }
    }

    @Test
    public void testLayerNormMixedOrders(){
        Nd4j.getRandom().setSeed(12345);
        INDArray input = Nd4j.rand(DataType.DOUBLE, 3, 8).dup('f');
        INDArray gain = Nd4j.rand(DataType.DOUBLE, 1, 8).dup('f');
        INDArray bias = Nd4j.rand(DataType.DOUBLE, 1, 8).dup('f');

        INDArray outFF = Nd4j.create(DataType.DOUBLE, new long[]{3,8}, 'f');
        INDArray outCC = Nd4j.create(DataType.DOUBLE, new long[]{3,8}, 'c');
        INDArray outFC = Nd4j.create(DataType.DOUBLE, new long[]{3,8}, 'c');
        INDArray outCF = Nd4j.create(DataType.DOUBLE, new long[]{3,8}, 'f');

        //F in, F out case
        Nd4j.exec(DynamicCustomOp.builder("layer_norm")
                .addInputs(input, gain, bias)
                .addOutputs(outFF)
                .addIntegerArguments(1) //Axis
                .build());

        //C in, C out case
        Nd4j.exec(DynamicCustomOp.builder("layer_norm")
                .addInputs(input.dup('c'), gain.dup('c'), bias.dup('c'))
                .addOutputs(outCC)
                .addIntegerArguments(1) //Axis
                .build());

        assertEquals(outFF, outCC);       //OK

        //C in, F out case
        outFF.assign(0);
        Nd4j.exec(DynamicCustomOp.builder("layer_norm")
                .addInputs(input.dup('c'), gain.dup('c'), bias.dup('c'))
                .addOutputs(outCF)
                .addIntegerArguments(1) //Axis
                .build());
        assertEquals(outCC, outCF);       //Fails here

        //F in, C out case
        outFF.assign(0);
        Nd4j.exec(DynamicCustomOp.builder("layer_norm")
                .addInputs(input, gain, bias)
                .addOutputs(outFC)
                .addIntegerArguments(1) //Axis
                .build());
        assertEquals(outCC, outFC);       //Fails here
    }
}