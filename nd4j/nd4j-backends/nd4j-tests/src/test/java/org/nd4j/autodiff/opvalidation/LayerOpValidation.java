/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.autodiff.opvalidation;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInfo;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.validation.OpValidation;
import org.nd4j.autodiff.validation.TestCase;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.iter.NdIndexIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.layers.convolution.AvgPooling2D;
import org.nd4j.linalg.api.ops.impl.layers.convolution.DepthwiseConv2D;
import org.nd4j.linalg.api.ops.impl.layers.convolution.Pooling2D;
import org.nd4j.linalg.api.ops.impl.layers.convolution.Pooling2DDerivative;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv1DConfig;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv2DConfig;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv3DConfig;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.DeConv2DConfig;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.DeConv3DConfig;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.LocalResponseNormalizationConfig;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.PaddingMode;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Pooling2DConfig;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Pooling3DConfig;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.GRU;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.config.LSTMActivations;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.config.LSTMDataFormat;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.config.LSTMDirectionMode;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.config.LSTMLayerConfig;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.outputs.LSTMLayerOutputs;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.weights.LSTMLayerWeights;
import org.nd4j.linalg.api.ops.impl.transforms.custom.LayerNorm;
import org.nd4j.linalg.api.ops.impl.transforms.custom.Standardize;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.ops.transforms.Transforms;

import static org.junit.jupiter.api.Assertions.*;

@Slf4j
@NativeTag
@Tag(TagNames.SAMEDIFF)
public class LayerOpValidation extends BaseOpValidation {

    @Override
    public long getTimeoutMilliseconds() {
        return 90000L;
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testXwPlusB(Nd4jBackend backend) {
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
                .expectedOutput(res.name(), exp);

//        System.out.println(sameDiff.summary());
//        System.out.println("============================");
        sameDiff.summary();
        sameDiff.createGradFunction();
//        System.out.println(sameDiff.getFunction("grad").summary());
        sameDiff.getFunction("grad").summary();


        String err = OpValidation.validate(tc);
        assertNull(err);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testReluLayer(Nd4jBackend backend) {
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
                .expectedOutput(res.name(), exp);


        String err = OpValidation.validate(tc);
        assertNull(err);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testBiasAdd(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(12345);

        SameDiff sameDiff = SameDiff.create();
        INDArray input = Nd4j.linspace(1, 8, 8, DataType.DOUBLE).reshape(new long[]{2, 4});
        INDArray b = Nd4j.linspace(1, 4, 4, DataType.DOUBLE).divi(4);

        SDVariable sdInput = sameDiff.var("input", input);
        SDVariable sdBias = sameDiff.var("bias", b);

        SDVariable res = sameDiff.nn().biasAdd(sdInput, sdBias, true);
        SDVariable loss = sameDiff.standardDeviation(res, true);

        INDArray exp = input.addRowVector(b);

        TestCase tc = new TestCase(sameDiff)
                .gradientCheck(true)
                .expectedOutput(res.name(), exp);

        String err = OpValidation.validate(tc);
        assertNull(err);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testConv2d(Nd4jBackend backend) {
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
                        out = sd.cnn().upsampling2d(in,  2, 2, true);
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

        assertEquals(0, failed.size(),failed.toString());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testLrn2d(Nd4jBackend backend) {
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
            TestCase tc = new TestCase(sd).gradientCheck(true);
            String error = OpValidation.validate(tc);
            if (error != null) {
                failed.add(msg);
            }

        }
        assertEquals(0, failed.size(),failed.toString());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testIm2Col(Nd4jBackend backend) {
        //OpValidationSuite.ignoreFailing();      //TEMPORARY DUE TO JVM CRASH: https://github.com/eclipse/deeplearning4j/issues/6873
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

            TestCase tc = new TestCase(sd).gradientCheck(true).testName(msg);
            String error = OpValidation.validate(tc);
            if (error != null) {
                failed.add(msg);
            }
        }

        assertEquals( 0, failed.size(),failed.toString());
    }


    private static int[] nchwToNhwc(int[] in) {
        return new int[]{in[0], in[2], in[3], in[1]};
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testOutputShape(Nd4jBackend backend) {
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
        AvgPooling2D avgPooling2D = new AvgPooling2D(input, null, conf);

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
        Pooling2DDerivative avg2dDeriv = new Pooling2DDerivative(input, grad, null, conf);

        val outSizesBP = Nd4j.getExecutioner().calculateOutputShape(avg2dDeriv);
        assertEquals(1, outSizesBP.size());

        assertArrayEquals(inSize, outSizesBP.get(0).getShape());
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testAvgPool(Nd4jBackend backend) {
        long[] inSize = {1, 8, 8, 3};  //NHWC

        Pooling2DConfig conf = Pooling2DConfig.builder()
                .isNHWC(true)   //***NHWC
                .isSameMode(false)
                .kH(3).kW(2)
                .sH(2).sW(2)
                .type(Pooling2D.Pooling2DType.AVG)
                .build();

        INDArray input = Nd4j.create(inSize);
        AvgPooling2D avgPooling2D = new AvgPooling2D(input, null, conf);

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
        Pooling2DDerivative avg2dDeriv = new Pooling2DDerivative(input, grad, Nd4j.create(inSize), conf);

        val outSizesBP = Nd4j.getExecutioner().calculateOutputShape(avg2dDeriv);
        assertEquals(1, outSizesBP.size());
        assertArrayEquals(inSize, outSizesBP.get(0).getShape());

        Nd4j.getExecutioner().execAndReturn(avg2dDeriv);
    }


    private static int[] ncdhwToNdhwc(int[] in) {
        return new int[]{in[0], in[2], in[3], in[4], in[1]};
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testConv3d(Nd4jBackend backend, TestInfo testInfo) {
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
                            //pooling3d - average, no same
                            msg = "2 - pooling 3d, average, same";
                            out = sd.cnn().avgPooling3d(in, Pooling3DConfig.builder()
                                    .kH(2).kW(2).kD(2)
                                    .sH(1).sW(1).sD(1)
                                    .isSameMode(false)
                                    .isNCDHW(ncdhw)
                                    .build());
                            break;
                        case 3:
                            //pooling 3d - max, no same
                            msg = "3 - pooling 3d, max, same";
                            out = sd.cnn().maxPooling3d(in, Pooling3DConfig.builder()
                                    .kH(2).kW(2).kD(2)
                                    .sH(1).sW(1).sD(1)
                                    .isSameMode(true)
                                    .isNCDHW(ncdhw)
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
                    TestCase tc = new TestCase(sd).gradientCheck(true);
                    tc.testName(msg);
                    String error = OpValidation.validate(tc);
                    if (error != null) {
                        failed.add(testInfo.getTestMethod().get().getName());
                    }
                }
            }
        }

        assertEquals( 0, failed.size(),failed.toString());
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testDepthWiseConv2dBasic(Nd4jBackend backend) {
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

        Conv2DConfig c = Conv2DConfig.builder()
                .kH(kH).kW(kW)
                .pH(0).pW(0)
                .sH(1).sW(1)
                .dH(1).dW(1)
                .isSameMode(false)
                .build();

        SDVariable out = sd.cnn().separableConv2d(in, dW, null, b, c);
        out = sd.nn().tanh("out", out);

        INDArray outArr = out.eval();
        //Expected output size: out = (in - k + 2*p)/s + 1 = (28-2+0)/1+1 = 27
        val outShape = outArr.shape();
        assertArrayEquals(new long[]{mb, depthWise * nIn, 27, 27}, outShape);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSeparableConv2dBasic(Nd4jBackend backend) {
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

        Conv2DConfig c = Conv2DConfig.builder()
                .kH(kH).kW(kW)
                .pH(0).pW(0)
                .sH(1).sW(1)
                .dH(1).dW(1)
                .isSameMode(false)
                .dataFormat(Conv2DConfig.NCHW)
                .build();

        SDVariable out = sd.cnn().separableConv2d(in, dW, pW, b, c);
        out = sd.nn().tanh("out", out);

        INDArray outArr = out.eval();
        //Expected output size: out = (in - k + 2*p)/s + 1 = (8-2+0)/1+1 = 7
        val outShape = outArr.shape();
        assertArrayEquals(new long[]{mb, nOut, 7, 7}, outShape);

        SDVariable loss = out.std(true);

//        System.out.println(sd.summary());
//        System.out.println("--------------------------");
//        sd.createGradFunction();
//        System.out.println(sd.getFunction("grad").summary());

        //Gradient check:
        TestCase tc = new TestCase(sd).gradientCheck(true);
        String err = OpValidation.validate(tc);
        assertNull(err);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testDeconv2dBasic(Nd4jBackend backend) {
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

        DeConv2DConfig deconv = DeConv2DConfig.builder()
                .kH(kH).kW(kW)
                .pH(0).pW(0)
                .sH(1).sW(1)
                .dH(1).dW(1)
                .isSameMode(false)
                .build();

        SDVariable out = sd.cnn().deconv2d(in, w, b, deconv);
        out = sd.nn().tanh("out", out);

        INDArray outArr = out.eval();
        //Expected output size: out = (in + k + 2*p)/ s - 1 = (8 + 2+0)/1 - 1 = 9
        val outShape = outArr.shape();
        assertArrayEquals(new long[]{mb, nOut, 9, 9}, outShape);

        SDVariable loss = out.std(true);
        //Gradient check:
        TestCase tc = new TestCase(sd).gradientCheck(true);
        String err = OpValidation.validate(tc);
        assertNull(err);
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testConv2dBasic(Nd4jBackend backend) {
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

        Conv2DConfig c = Conv2DConfig.builder()
                .kH(kH).kW(kW)
                .pH(0).pW(0)
                .sH(1).sW(1)
                .dH(1).dW(1)
                .isSameMode(false)
                .build();

        SDVariable out = sd.cnn().conv2d("conv", in, w, b, c);
        out = sd.nn().tanh("out", out);

        INDArray outArr = out.eval();
        //Expected output size: out = (in - k + 2*p)/s + 1 = (28-2+0)/1+1 = 27
        val outShape = outArr.shape();
        assertArrayEquals(new long[]{mb, nOut, 27, 27}, outShape);
        // sd.execBackwards(); // TODO: test failing here
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMaxPoolingArgMax(Nd4jBackend backend) {
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
                .isSameMode(true)
                .build();

        SDVariable[] results = sd.cnn().maxPoolWithArgmax(new String[]{"out", "idx"}, in, pooling2DConfig);
        assertArrayEquals(inArr.shape(), results[0].eval().shape());
        assertArrayEquals(inArr.shape(), results[1].eval().shape());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMaxPooling2dBasic(Nd4jBackend backend) {
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

        INDArray outArr = out.eval();
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

        assertNull(OpValidation.validate(new TestCase(sd).gradientCheck(true)
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

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testAvgPooling2dBasic(Nd4jBackend backend) {
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

        INDArray outArr = out.eval();
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
                .expected(outPool, exp).gradientCheck(true)));

    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testAvgPooling3dBasic(Nd4jBackend backend) {
        int nIn = 3;
        int kH = 2;
        int kW = 2;
        int kD = 2;

        int mb = 3;
        int imgH = 5;
        int imgW = 5;
        int imgD = 5;

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
        out = sd.nn().tanh("loss", out).shape().rename("out");

        // oH = (iH - (kH + (kH-1)*(dH-1)) + 2*pH)/sH + 1;
        INDArray outArr = Nd4j.createFromArray(mb, nIn, 4, 4, 4L);

        TestCase tc = new TestCase(sd).expectedOutput("out", outArr).gradientCheck(false);
        String err = OpValidation.validate(tc);
        assertNull(err);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMaxPooling3dBasic(Nd4jBackend backend) {
        int nIn = 3;
        int kH = 2;
        int kW = 2;
        int kD = 2;

        int mb = 3;
        int imgH = 5;
        int imgW = 5;
        int imgD = 5;

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
        out = sd.nn().tanh("loss", out).shape().rename("out");

        sd.setLossVariables("loss");

        // oH = (iH - (kH + (kH-1)*(dH-1)) + 2*pH)/sH + 1;
        INDArray outArr = Nd4j.createFromArray(mb, nIn, 4, 4, 4L);

        TestCase tc = new TestCase(sd).expectedOutput("out", outArr).gradientCheck(false);
        String err = OpValidation.validate(tc);
        assertNull(err);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testConv1dBasic(Nd4jBackend backend) {
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
                .paddingMode(PaddingMode.VALID)
                .build();

        SDVariable out = sd.cnn().conv1d(in, w, conv1DConfig);
        out = sd.nn().tanh("loss", out).shape().rename("out");

        sd.setLossVariables("loss");

        //Expected output size: out = (in - k + 2*p)/s + 1 = (28-2+0)/1+1 = 27
        INDArray outArr = Nd4j.createFromArray(mb, nOut, 27L);
        TestCase tc = new TestCase(sd).expectedOutput("out", outArr).gradientCheck(false);
        String err = OpValidation
                .validate(tc);
        assertNull(err);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testConv1dCausal(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(12345);
        int nIn = 3;
        int nOut = 4;
        int mb = 2;

        for (int k : new int[]{2, 3}) {
            for (int sz : new int[]{3, 4, 5}) {
                for (int s : new int[]{1, 2}) {
                    for (int d : new int[]{1, 2}) {
                        for (boolean ncw : new boolean[]{true, false}) {

                            SameDiff sd = SameDiff.create();
                            INDArray wArr = Nd4j.rand(DataType.DOUBLE, k, nIn, nOut);
                            INDArray inArr = Nd4j.rand(DataType.DOUBLE, (ncw ? new long[]{mb, nIn, sz} : new long[]{mb, sz, nIn}));
                            INDArray bArr = Nd4j.rand(DataType.DOUBLE, nOut);

                            SDVariable in = sd.var("in", inArr);
                            SDVariable w = sd.var("W", wArr);
                            SDVariable b = sd.var("b", bArr);

                            Conv1DConfig conv1DConfig = Conv1DConfig.builder()
                                    .dataFormat(ncw ? Conv1DConfig.NCW : Conv1DConfig.NWC)
                                    .k(k).p(0).s(s).d(d)
                                    .paddingMode(PaddingMode.CAUSAL)
                                    .build();

                            SDVariable out = sd.cnn().conv1d(in, w, b, conv1DConfig);
                            SDVariable loss = sd.nn().tanh(out).std(true).rename("loss");

                            sd.setLossVariables("loss");

                            String name = "k=" + k + ", sz=" + sz + ", ncw=" + ncw;

                            System.out.println(name);

                            TestCase tc = new TestCase(sd).testName(name).gradientCheck(true);
                            String err = OpValidation
                                    .validate(tc);
                            assertNull(err);
                        }
                    }
                }
            }
        }
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testConv1dForward(Nd4jBackend backend) {
        int nIn = 2;
        int nOut = 1;
        int kernel = 3;
        int batchSize = 10;
        int sequenceSize = 5;

        SameDiff sd = SameDiff.create();

        INDArray inArr = Nd4j.linspace(0, nIn * batchSize * sequenceSize, nIn * batchSize * sequenceSize)
                .reshape(batchSize, nIn, sequenceSize);

        INDArray wArr = Nd4j.linspace(0, kernel * nIn * nOut, kernel * nIn * nOut)
                .reshape(kernel, nIn, nOut);

        SDVariable in = sd.var("in", inArr);
        SDVariable w = sd.var("w", wArr);

        SDVariable res = sd.cnn.conv1d(in, w, Conv1DConfig.builder().k(kernel).paddingMode(PaddingMode.VALID).build());

        INDArray expected = Nd4j.createFromArray(
                new double[][][]{
                        {{82.42424f, 100.60606f, 118.78788f}},
                        {{264.2424f, 282.4242f, 300.6061f}},
                        {{446.0606f, 464.2424f, 482.424f}},
                        {{627.8788f, 646.0606f, 664.2424f}},
                        {{809.6970f, 827.8788f, 846.0606f}},
                        {{991.5152f, 1009.69696f, 1027.8788f}},
                        {{1173.3333f, 1191.5152f, 1209.6970f}},
                        {{1355.1515f, 1373.3333f, 1391.5153f}},
                        {{1536.9697f, 1555.1515f, 1573.3333f}},
                        {{1718.7878f, 1736.9697f, 1755.1515f}}
                }
        );

        TestCase tc = new TestCase(sd).gradientCheck(false).expectedOutput(res.name(), expected);
        String err = OpValidation.validate(tc);

        assertNull(err);
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testConv3dBasic(Nd4jBackend backend) {
        int nIn = 3;
        int nOut = 4;
        int kH = 2;
        int kW = 2;
        int kD = 2;

        int mb = 3;
        int imgH = 5;
        int imgW = 5;
        int imgT = 5;

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
        out = sd.nn().tanh("loss", out).shape().rename("out");

        sd.setLossVariables("loss");

        //Expected output size, NOT same mode: out = (in - k)/d + 1 = (28-2+0)/1+1 = 27
        //Expected output size, WITH same mode: out = in/stride
        INDArray outArr = Nd4j.createFromArray(mb, nOut, 5, 5, 5L);

        TestCase tc = new TestCase(sd).expectedOutput("out", outArr).gradientCheck(true);
        String err = OpValidation
                .validate(tc);
        assertNull(err);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testDeConv3dBasic(Nd4jBackend backend) {
        int nIn = 4;
        int nOut = 3;
        int kH = 2;
        int kW = 2;
        int kD = 2;

        int mb = 3;
        int imgH = 5;
        int imgW = 5;
        int imgT = 5;

        SameDiff sd = SameDiff.create();
        INDArray inArr = Nd4j.rand(new long[]{mb, nIn, 5, 5, 5});
        INDArray wArr = Nd4j.rand(kD, kH, kW, nOut, nIn);

        SDVariable in = sd.var("in", inArr);
        SDVariable w = sd.var("W", wArr);

        DeConv3DConfig conv3DConfig = DeConv3DConfig.builder()
                .kH(kH).kW(kW).kD(kD)
                .sD(1).sH(1).sW(1)
                .dH(1).dW(1).dD(1)
                .isSameMode(true)
                .dataFormat(DeConv3DConfig.NCDHW)
                .build();

        SDVariable out = sd.cnn().deconv3d(in, w, conv3DConfig);
        out = sd.nn().tanh("loss", out).shape().rename("out");

        sd.setLossVariables("loss");

        //Expected conv3d size, NOT same mode: out = (in - k)/d + 1 = (28-2+0)/1+1 = 27
        //Expected conv3d size, WITH same mode: out = in/stride
        // reversed this for deconv3d
        INDArray outArr = Nd4j.createFromArray(new long[]{mb, nOut, imgT, imgH, imgW});

        TestCase tc = new TestCase(sd)
                .expectedOutput("out", outArr)
                .gradientCheck(true);
        String err = OpValidation.validate(tc);
        assertNull(err);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testLayerNorm(Nd4jBackend backend) {
        final INDArray random = Nd4j.rand(DataType.DOUBLE, 10, 4);
        final INDArray standardized = random.ulike();
        Nd4j.getExecutioner().exec(new Standardize(random, standardized, 1));

        final INDArray gain = Nd4j.rand(DataType.DOUBLE, 4);
        final INDArray bias = Nd4j.rand(DataType.DOUBLE, 4);
        final INDArray res = standardized.mulRowVector(gain).addRowVector(bias);
        final INDArray expOut = res.norm1();

        final int[] axis = new int[]{1};
        SameDiff sd = SameDiff.create();
        SDVariable sdInput = sd.var("input", standardized);
        SDVariable sdGain = sd.var("gain", gain);
        SDVariable sdBias = sd.var("bias", bias);
        SDVariable out = sd.nn.layerNorm(sdInput, sdGain, sdBias, true, axis);
        out.norm1("out");

        String err = OpValidation.validate(new TestCase(sd)
                .expectedOutput("out", expOut)
                .gradientCheck(true));
        assertNull(err);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testLayerNorm4d(Nd4jBackend backend) {
        int mb = 3;
        int ch = 4;
        for (boolean nchw : new boolean[]{true, false}) {
            double eps = 0.0;
            INDArray x = Nd4j.rand(DataType.DOUBLE, nchw ? new long[]{mb, ch, 8, 8} : new long[]{mb, 8, 8, ch});
            INDArray gain4d = Nd4j.rand(DataType.DOUBLE, nchw ? new long[]{1, ch, 1, 1} : new long[]{1, 1, 1, ch});
            INDArray bias4d = Nd4j.rand(DataType.DOUBLE, nchw ? new long[]{1, ch, 1, 1} : new long[]{1, 1, 1, ch});
            INDArray mean = x.mean(true, 1, 2, 3);
            INDArray std = Transforms.sqrt(x.var(false, 1, 2, 3).addi(eps)).reshape(mb, 1, 1, 1);

            INDArray standardized = x.sub(mean).div(std);
            INDArray exp = standardized.mul(gain4d).add(bias4d);

            final int[] axis = new int[]{1, 2, 3};
            SameDiff sd = SameDiff.create();
            SDVariable sdInput = sd.var("input", x);
            SDVariable sdGain = sd.var("gain", gain4d.reshape(ch));
            SDVariable sdBias = sd.var("bias", bias4d.reshape(ch));
            SDVariable out = sd.nn.layerNorm("layernorm", sdInput, sdGain, sdBias, nchw, axis);

            SDVariable loss = sd.loss.l2Loss(out);

            String err = OpValidation.validate(new TestCase(sd)
                    .expectedOutput("layernorm", exp)
                    .gradientCheck(true));
            assertNull(err);
        }
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testLayerNormOP(Nd4jBackend backend) {
        final INDArray random = Nd4j.rand(DataType.DOUBLE, 10, 4);
        final INDArray standardized = random.ulike();
        Nd4j.getExecutioner().exec(new Standardize(random, standardized, 1));

        final INDArray gain = Nd4j.rand(DataType.DOUBLE, 4);
        final INDArray bias = Nd4j.rand(DataType.DOUBLE, 4);
        final INDArray res = standardized.mulRowVector(gain).addRowVector(bias);

        final INDArray output = Nd4j.zerosLike(res);
        Nd4j.getExecutioner().exec(new LayerNorm(standardized, gain, bias, output, true, 1));

        assertEquals(res, output);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testLayerNormNoBias(Nd4jBackend backend) {
        final INDArray random = Nd4j.rand(DataType.DOUBLE, 10, 4);
        final INDArray standardized = random.ulike();
        Nd4j.getExecutioner().exec(new Standardize(random, standardized, 1));

        final INDArray gain = Nd4j.rand(DataType.DOUBLE, 4);
        final INDArray res = standardized.mulRowVector(gain);
        final INDArray expOut = res.norm1();

        final int[] axis = new int[]{1};
        SameDiff sd = SameDiff.create();
        SDVariable sdInput = sd.var("input", standardized);
        SDVariable sdGain = sd.var("gain", gain);
        SDVariable out = sd.nn.layerNorm(sdInput, sdGain, true, axis);
        out.norm1("out");

        String err = OpValidation.validate(new TestCase(sd)
                .expectedOutput("out", expOut)
                .gradientCheck(true));
        assertNull(err, err);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testLayerNormOPNoBias(Nd4jBackend backend) {
        final INDArray random = Nd4j.rand(DataType.DOUBLE, 10, 4);
        final INDArray standardized = random.ulike();
        Nd4j.getExecutioner().exec(new Standardize(random, standardized, 1));

        final INDArray gain = Nd4j.rand(DataType.DOUBLE, 4);
        final INDArray res = standardized.mulRowVector(gain);

        final INDArray output = Nd4j.zerosLike(res);
        Nd4j.getExecutioner().exec(new LayerNorm(standardized, gain, output, true, 1));

        assertEquals(res, output);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testLayerNormNoDeviation(Nd4jBackend backend) {
        final INDArray random = Nd4j.rand(DataType.DOUBLE, 10, 4);
        for (int i = 0; i < 4; i++) {
            random.putScalar(1, i, 7);
        }

        final INDArray standardized = random.ulike();
        Nd4j.getExecutioner().exec(new Standardize(random, standardized, 1));

        final INDArray gain = Nd4j.rand(DataType.DOUBLE, 4);
        final INDArray bias = Nd4j.rand(DataType.DOUBLE, 4);
        final INDArray res = standardized.mulRowVector(gain).addRowVector(bias);
        final INDArray expOut = res.norm1();

        final int[] axis = new int[]{1};
        SameDiff sd = SameDiff.create();
        SDVariable sdInput = sd.var("input", standardized);
        SDVariable sdGain = sd.var("gain", gain);
        SDVariable sdBias = sd.var("bias", bias);
        SDVariable out = sd.nn.layerNorm(sdInput, sdGain, sdBias, true, axis);
        out.norm1("out");

        String err = OpValidation.validate(new TestCase(sd)
                .expectedOutput("out", expOut)
                .gradCheckMask(Collections.singletonMap("input", random.neq(7)))
                .gradientCheck(true));
        assertNull(err, err);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void exceptionThrown_WhenConv1DConfigInvalid(Nd4jBackend backend) {
        assertThrows(IllegalArgumentException.class,() -> {
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
                    .paddingMode(PaddingMode.VALID)
                    .build();

            SDVariable out = sd.cnn().conv1d(in, w, conv1DConfig);

        });

    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void exceptionThrown_WhenConv2DConfigInvalid(Nd4jBackend backend) {
        assertThrows(IllegalArgumentException.class,() -> {
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
        });

    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void exceptionThrown_WhenConf3DInvalid(Nd4jBackend backend) {
        assertThrows(IllegalArgumentException.class,() -> {
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
        });

    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testLayerNormMixedOrders(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(12345);
        INDArray input = Nd4j.rand(DataType.DOUBLE, 3, 8).dup('f');
        INDArray gain = Nd4j.rand(DataType.DOUBLE, 8).dup('f');
        INDArray bias = Nd4j.rand(DataType.DOUBLE, 8).dup('f');

        INDArray outFF = Nd4j.create(DataType.DOUBLE, new long[]{3, 8}, 'f');
        INDArray outCC = Nd4j.create(DataType.DOUBLE, new long[]{3, 8}, 'c');
        INDArray outFC = Nd4j.create(DataType.DOUBLE, new long[]{3, 8}, 'c');
        INDArray outCF = Nd4j.create(DataType.DOUBLE, new long[]{3, 8}, 'f');

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

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testBiasAdd_nchw_nhwc(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(12345);

        for (boolean nchw : new boolean[]{true, false}) {
            log.info("Starting test: {}", nchw ? "nchw" : "nhwc");
            SameDiff sameDiff = SameDiff.create();

            SDVariable in = sameDiff.var("input", Nd4j.rand(DataType.DOUBLE, nchw ? new long[]{2, 4, 3, 3} : new long[]{2, 3, 3, 4}));
            SDVariable b = sameDiff.var("bias", Nd4j.rand(DataType.DOUBLE, new long[]{4}));

            SDVariable bAdd = sameDiff.nn.biasAdd(in, b, nchw);
            SDVariable loss = bAdd.std(true);


            INDArray exp = in.getArr().dup();
            if (nchw) {
                exp.addi(b.getArr().reshape(1, 4, 1, 1));
            } else {
                exp.addi(b.getArr().reshape(1, 1, 1, 4));
            }

            TestCase tc = new TestCase(sameDiff)
                    .gradientCheck(true)
                    .expectedOutput(bAdd.name(), exp);

            String err = OpValidation.validate(tc);
            assertNull(err);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testDepthwiseConv2D(){

        int bS = 10;

        int kernelHeight = 2;
        int kernelWidth = 2;
        int strideHeight = 2;
        int strideWidth = 2;
        int inChannels = 2;
        int outChannels = 3;
        Nd4j.getRandom().setSeed(12345);
        SameDiff sd = SameDiff.create();
        SDVariable in = sd.var("in", Nd4j.rand(bS, inChannels, 5,5));
        SDVariable weights = sd.var("weights", Nd4j.rand(DataType.DOUBLE, kernelHeight, kernelWidth, inChannels, outChannels));
        SDVariable bias = sd.var("bias", Nd4j.rand(DataType.DOUBLE, inChannels*outChannels));
        Conv2DConfig config = Conv2DConfig.builder()
                .kH(kernelHeight)
                .kW(kernelWidth)
                .sH(strideHeight)
                .sW(strideWidth)
                .dataFormat("NCHW")
                .build();

        SDVariable out = sd.cnn.depthWiseConv2d(in, weights, bias, config);
        SDVariable loss = sd.standardDeviation("loss", out, true);
        loss.markAsLoss();

        String err = OpValidation.validate(new TestCase(sd)
                .gradientCheck(true)
        );
        assertNull(err);



    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void LSTMLayerTestCase1(Nd4jBackend backend) {

        int bS = 5;
        int nIn = 3;
        int numUnits = 7;
        int sL = 3; //small just for test


        // notations:
        // bS - batch size, numExamples
        // sL - sequence length, number of time steps, timeLength
        // nIn - input size, inOutSize

        //  TNS: shape [timeLength, numExamples, inOutSize] - sometimes referred to as "time major"<br>
        //  NST: shape [numExamples, inOutSize, timeLength]<br>
        //  NTS: shape [numExamples, timeLength, inOutSize]<br>
        //  for bidirectional:
        //  T2NS: 3 = [timeLength, 2, numExamples, inOutSize] (for ONNX)


        for (boolean useCLast : new boolean[]{false, true}) {
            for (boolean useYLast : new boolean[]{false, true}) {

                SameDiff sd = SameDiff.create();
                SDVariable in = sd.var("in", Nd4j.randn(DataType.DOUBLE, bS, nIn, sL));


                SDVariable cLast = useCLast ? sd.var("cLast", Nd4j.zeros(DataType.DOUBLE, bS, numUnits)) : null;
                SDVariable yLast = useYLast ? sd.var("yLast", Nd4j.zeros(DataType.DOUBLE, bS, numUnits)) : null;


                LSTMLayerConfig c = LSTMLayerConfig.builder()
                        .lstmdataformat(LSTMDataFormat.NST)
                        .directionMode(LSTMDirectionMode.FWD)
                        .gateAct(LSTMActivations.SIGMOID)
                        .cellAct(LSTMActivations.TANH)
                        .outAct(LSTMActivations.TANH)
                        .retFullSequence(true)
                        .retLastC(true)
                        .retLastH(true)
                        .build();

                LSTMLayerOutputs outputs = new LSTMLayerOutputs(sd.rnn.lstmLayer(
                        in, cLast, yLast, null,
                        LSTMLayerWeights.builder()
                                .weights(sd.var("weights", Nd4j.randn(DataType.DOUBLE, nIn, 4 * numUnits)))
                                .rWeights(sd.var("rWeights", Nd4j.randn(DataType.DOUBLE, numUnits, 4 * numUnits)))
                                .peepholeWeights(sd.var("inputPeepholeWeights", Nd4j.randn(DataType.DOUBLE, 3 * numUnits)))
                                .bias(sd.var("bias", Nd4j.rand(DataType.DOUBLE, 4 * numUnits))).build(),
                        c), c);

                long[] out = new long[]{bS, numUnits, sL};
                long[] hL = new long[]{bS, numUnits};
                long[] cL = new long[]{bS, numUnits};

                assertArrayEquals(out, outputs.getOutput().eval().shape());
                assertArrayEquals(hL, outputs.getLastOutput().eval().shape());
                assertArrayEquals(cL, outputs.getLastState().eval().shape());

                sd.setLossVariables(outputs.getOutput(), outputs.getLastTimeStepOutput(), outputs.getTimeSeriesOutput());

                String err = OpValidation.validate(new TestCase(sd)
                        .gradientCheck(true)
                        .testName("cLast=" + cLast + ", yLast=" + yLast)
                );

                assertNull(err);
            }
        }


    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void LSTMLayerTestCase2(Nd4jBackend backend) {
        int bS = 5;
        int nIn = 3;
        int numUnits = 7;
        int sL = 3; //small just for test

        SameDiff sd = SameDiff.create();

        // notations:
        // bS - batch size, numExamples
        // sL - sequence length, number of time steps, timeLength
        // nIn - input size, inOutSize

        //  TNS: shape [timeLength, numExamples, inOutSize] - sometimes referred to as "time major"<br>
        //  NST: shape [numExamples, inOutSize, timeLength]<br>
        //  NTS: shape [numExamples, timeLength, inOutSize]<br>
        //  for bidirectional:
        //  T2NS: 3 = [timeLength, 2, numExamples, inOutSize] (for ONNX)
        SDVariable in = sd.var("in", Nd4j.rand(DataType.DOUBLE, sL, bS, nIn));


        SDVariable cLast = sd.var("cLast", Nd4j.zeros(DataType.DOUBLE, bS, numUnits));
        SDVariable yLast = sd.var("yLast", Nd4j.zeros(DataType.DOUBLE, bS, numUnits));

        LSTMLayerConfig c = LSTMLayerConfig.builder()
                .lstmdataformat(LSTMDataFormat.TNS)
                .directionMode(LSTMDirectionMode.FWD)
                .gateAct(LSTMActivations.SIGMOID)
                .cellAct(LSTMActivations.TANH)
                .outAct(LSTMActivations.TANH)
                .retFullSequence(true)
                .retLastC(false)
                .retLastH(false)
                .build();

        LSTMLayerOutputs outputs = new LSTMLayerOutputs(sd.rnn.lstmLayer(
                in, cLast, yLast, null,
                LSTMLayerWeights.builder()
                        .weights(sd.var("weights", Nd4j.rand(DataType.DOUBLE, nIn, 4 * numUnits)))
                        .rWeights(sd.var("rWeights", Nd4j.rand(DataType.DOUBLE, numUnits, 4 * numUnits)))
                        .build(),
                c), c);


        long[] out = new long[]{sL, bS, numUnits};
        assertArrayEquals(out, outputs.getOutput().eval().shape());

        sd.setLossVariables(outputs.getOutput());

        String err = OpValidation.validate(new TestCase(sd)
                .gradientCheck(true)
        );

        assertNull(err);

    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void LSTMLayerTestCase3(Nd4jBackend backend) {
        int bS = 5;
        int nIn = 3;
        int numUnits = 7;
        int sL = 3; //small just for test

        SameDiff sd = SameDiff.create();

        // notations:
        // bS - batch size, numExamples
        // sL - sequence length, number of time steps, timeLength
        // nIn - input size, inOutSize

        //  TNS: shape [timeLength, numExamples, inOutSize] - sometimes referred to as "time major"<br>
        //  NST: shape [numExamples, inOutSize, timeLength]<br>
        //  NTS: shape [numExamples, timeLength, inOutSize]<br>
        //  for bidirectional:
        //  T2NS: 3 = [timeLength, 2, numExamples, inOutSize] (for ONNX)
        SDVariable in = sd.var("in", Nd4j.rand(DataType.DOUBLE, bS, sL, nIn));


        // when directionMode >= 2 (BIDIR_CONCAT=3)
        // Wx, Wr [2, nIn, 4*nOut]
        // hI, cI [2, bS, nOut]
        SDVariable cLast = sd.var("cLast", Nd4j.zeros(DataType.DOUBLE, 2, bS, numUnits));
        SDVariable yLast = sd.var("yLast", Nd4j.zeros(DataType.DOUBLE, 2, bS, numUnits));

        LSTMLayerConfig c = LSTMLayerConfig.builder()
                .lstmdataformat(LSTMDataFormat.NTS)
                .directionMode(LSTMDirectionMode.BIDIR_CONCAT)
                .gateAct(LSTMActivations.SIGMOID)
                .cellAct(LSTMActivations.SOFTPLUS)
                .outAct(LSTMActivations.SOFTPLUS)
                .retFullSequence(true)
                .retLastC(false)
                .retLastH(false)
                .build();

        LSTMLayerOutputs outputs = new LSTMLayerOutputs(sd.rnn.lstmLayer(new String[]{"out"},
                in, cLast, yLast, null,
                LSTMLayerWeights.builder()
                        .weights(sd.var("weights", Nd4j.rand(DataType.DOUBLE, 2, nIn, 4 * numUnits)))
                        .rWeights(sd.var("rWeights", Nd4j.rand(DataType.DOUBLE, 2, numUnits, 4 * numUnits)))
                        .build(),
                c), c);


        long[] out = new long[]{bS, sL, 2 * numUnits};

        assertArrayEquals(out, outputs.getOutput().eval().shape());

        sd.setLossVariables(outputs.getOutput());

        String err = OpValidation.validate(new TestCase(sd)
                .gradientCheck(true)
        );

        assertNull(err);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void GRUTestCase(Nd4jBackend backend) {
        int bS = 5;
        int nIn = 4;
        int nOut = 6;
        int time = 2;

        SameDiff sd = SameDiff.create();

        SDVariable in = sd.var("in", Nd4j.randn(DataType.DOUBLE, time, bS, nIn).muli(10));
        SDVariable hLast = sd.var("cLast", Nd4j.zeros(DataType.DOUBLE, bS, nOut));
        SDVariable Wx = sd.var("Wx", Nd4j.randn(DataType.DOUBLE, nIn, 3*nOut));
        SDVariable Wh = sd.var("Wh", Nd4j.randn(DataType.DOUBLE, nOut, 3*nOut));
        SDVariable biases = sd.var("bias", Nd4j.randn(DataType.DOUBLE, 3*nOut));

        SDVariable out = new GRU(sd, in, hLast, Wx, Wh,biases).outputVariable();

        long[] outShapes = new long[]{time,bS, nOut};
        assertArrayEquals(new long[]{time,bS, nOut}, out.eval().shape());

        sd.setLossVariables(out.std(true));
        String err = OpValidation.validate(new TestCase(sd)
                .gradientCheck(true)
        );

        assertNull(err);

    }




}