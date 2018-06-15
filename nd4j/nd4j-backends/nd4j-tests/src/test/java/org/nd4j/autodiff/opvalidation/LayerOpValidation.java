package org.nd4j.autodiff.opvalidation;

import org.junit.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.validation.OpValidation;
import org.nd4j.autodiff.validation.TestCase;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.layers.convolution.AvgPooling2D;
import org.nd4j.linalg.api.ops.impl.layers.convolution.Pooling2D;
import org.nd4j.linalg.api.ops.impl.layers.convolution.Pooling2DDerivative;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.*;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.fail;

public class LayerOpValidation extends BaseOpValidation {
    public LayerOpValidation(Nd4jBackend backend) {
        super(backend);
    }

    @Test
    public void testXwPlusB() {
        Nd4j.getRandom().setSeed(12345);

        for (boolean rank1Bias : new boolean[]{false, true}) {

            SameDiff sameDiff = SameDiff.create();
            INDArray input = Nd4j.rand(new long[]{2, 3});
            INDArray weights = Nd4j.rand(new long[]{3, 4});
            INDArray b = Nd4j.rand(rank1Bias ? new long[]{4} : new long[]{1, 4});

            SDVariable sdInput = sameDiff.var("input", input);
            SDVariable sdWeights = sameDiff.var("weights", weights);
            SDVariable sdBias = sameDiff.var("bias", b);

            SDVariable res = sameDiff.xwPlusB(sdInput, sdWeights, sdBias);
            SDVariable loss = sameDiff.standardDeviation(res, true);

            INDArray exp = input.mmul(weights).addiRowVector(b);

            TestCase tc = new TestCase(sameDiff)
                    .gradientCheck(true)
                    .expectedOutput(res.getVarName(), exp);


            String err = OpValidation.validate(tc);
            assertNull(err);
        }
    }

    @Test
    public void testReluLayer() {
        Nd4j.getRandom().setSeed(12345);

        for (boolean rank1Bias : new boolean[]{false, true}) {

            SameDiff sameDiff = SameDiff.create();
            INDArray input = Nd4j.rand(new long[]{2, 3});
            INDArray weights = Nd4j.rand(new long[]{3, 4});
            INDArray b = Nd4j.rand(rank1Bias ? new long[]{4} : new long[]{1, 4});

            SDVariable sdInput = sameDiff.var("input", input);
            SDVariable sdWeights = sameDiff.var("weights", weights);
            SDVariable sdBias = sameDiff.var("bias", b);

            SDVariable res = sameDiff.reluLayer(sdInput, sdWeights, sdBias);
            SDVariable loss = sameDiff.standardDeviation(res, true);

            INDArray exp = input.mmul(weights).addiRowVector(b);
            Transforms.relu(exp, false);

            TestCase tc = new TestCase(sameDiff)
                    .gradientCheck(true)
                    .expectedOutput(res.getVarName(), exp);


            String err = OpValidation.validate(tc);
            assertNull(err);
        }
    }

    @Test
    public void testBiasAdd() {
        Nd4j.getRandom().setSeed(12345);

        for (boolean rank1Bias : new boolean[]{false, true}) {

            SameDiff sameDiff = SameDiff.create();
            INDArray input = Nd4j.rand(new long[]{2, 4});
            INDArray b = Nd4j.rand(rank1Bias ? new long[]{4} : new long[]{1, 4});

            SDVariable sdInput = sameDiff.var("input", input);
            SDVariable sdBias = sameDiff.var("bias", b);

            SDVariable res = sameDiff.biasAdd(sdInput, sdBias);
            SDVariable loss = sameDiff.standardDeviation(res, true);

            INDArray exp = input.addiRowVector(b);

            TestCase tc = new TestCase(sameDiff)
                    .gradientCheck(true)
                    .expectedOutput(res.getVarName(), exp);

            String err = OpValidation.validate(tc);
            assertNull(err);
        }
    }


    @Test
    public void testLinear() {

        fail();
    }

    @Test
    public void testConv2d() {
        //avg pool, batch norm, conv2d, deconv2d, depthwise2d, LRN, max pool 2d, pooling2d, sconv2d, upsamilpng

        Nd4j.getRandom().setSeed(12345);

        int[][] inputSizes = new int[][]{{1, 3, 8, 8}, {3, 6, 12, 12}};

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
                        SDVariable w0 = sd.var("w0", Nd4j.rand(new int[]{3, inSizeNCHW[1], 3, 3}).muli(10));  //NCHW: nOut,nIn,kH,kW
                        SDVariable b0 = sd.var("b0", Nd4j.rand(new long[]{3}).muli(10));
                        out = sd.conv2d(in, w0, b0, Conv2DConfig.builder()
                                .isNHWC(false)
                                .isSameMode(true)
                                .kh(3).kw(3)
                                .sx(1).sy(1)
                                .build());
                        break;
                    case 1:
                        //Conv2d, with bias, NHWC, no same
                        msg = "1 - conv2d+bias, nhwc - input " + Arrays.toString(inSizeNCHW);
                        inSize = nchwToNhwc(inSizeNCHW);
                        in = sd.var("in", inSize);
                        SDVariable w1 = sd.var("w1", Nd4j.rand(new int[]{2, 4, inSizeNCHW[1], 3}).muli(10));  //NHWC: kH,kW,nIn,nOut
                        SDVariable b1 = sd.var("b1", Nd4j.rand(new long[]{3}).muli(10));
                        out = sd.conv2d(in, w1, b1, Conv2DConfig.builder()
                                .isNHWC(true)
                                .isSameMode(false)
                                .kh(2).kw(4)
                                .sx(2).sy(2)
                                .build());
                        break;
                    case 2:
                        //Conv2d, no bias, NCHW
                        msg = "2 - conv2d, no bias, nchw - input " + Arrays.toString(inSizeNCHW);
                        inSize = inSizeNCHW;
                        in = sd.var("in", inSize);
                        SDVariable w2 = sd.var("w0", Nd4j.rand(new int[]{3, inSizeNCHW[1], 1, 3}).muli(10));  //NCHW: nOut,nIn,kH,kW
                        out = sd.conv2d(in, w2, Conv2DConfig.builder()
                                .isNHWC(false)
                                .isSameMode(true)
                                .kh(1).kw(3)
                                .sx(1).sy(2)
                                .build());
                        break;
                    case 3:
                        //Avg pool, NCHW
                        msg = "3 - avg pool, NCHW, same - input " + Arrays.toString(inSizeNCHW);
                        inSize = inSizeNCHW;
                        in = sd.var("in", inSize);
                        out = sd.avgPooling2d(in, Pooling2DConfig.builder()
                                .isNHWC(false)
                                .isSameMode(true)
                                .kh(2).kw(2)
                                .sx(1).sy(1)
                                .build());
                        break;
                    case 4:
                        //Avg pool, NHWC, not same
                        msg = "3 - avg pool, NHWC, not same - input " + Arrays.toString(inSizeNCHW);
                        inSize = nchwToNhwc(inSizeNCHW);
                        in = sd.var("in", inSize);
                        out = sd.avgPooling2d(in, Pooling2DConfig.builder()
                                .isNHWC(true)
                                .isSameMode(false)
                                .kh(3).kw(2)
                                .sx(2).sy(2)
                                .build());
                        break;
                    case 5:
                        //Avg pool, NCHW
                        msg = "5 - avg pool, NCHW, same - input " + Arrays.toString(inSizeNCHW);
                        inSize = inSizeNCHW;
                        in = sd.var("in", inSize);
                        out = sd.maxPooling2d(in, Pooling2DConfig.builder()
                                .isNHWC(false)
                                .isSameMode(true)
                                .kh(2).kw(2)
                                .sx(1).sy(1)
                                .build());
                        break;
                    case 6:
                        //Max pool, NHWC, not same
                        msg = "6 - avg pool, NHWC, not same - input " + Arrays.toString(inSizeNCHW);
                        inSize = inSizeNCHW;
                        in = sd.var("in", inSize);
                        out = sd.maxPooling2d(in, Pooling2DConfig.builder()
                                .isNHWC(true)
                                .isSameMode(false)
                                .kh(3).kw(2)
                                .sx(2).sy(2)
                                .build());
                        break;
                    case 7:
                        //LRN
                        msg = "LRN";
                        inSize = inSizeNCHW;
                        in = sd.var("in", inSize);
                        out = sd.localResponseNormalization(in, LocalResponseNormalizationConfig.builder()
                                .depth(3)
                                .bias(1)
                                .alpha(1)
                                .beta(0.5)
                                .build());
                        break;
                    default:
                        throw new RuntimeException();

                }

                INDArray inArr = Nd4j.rand(inSize).muli(10);
                in.setArray(inArr);
                SDVariable loss = sd.standardDeviation("loss", out, true);

                TestCase tc = new TestCase(sd);
                String error = OpValidation.validate(tc);
                if (error != null) {
                    failed.add(name);
                }

            }
        }

        assertEquals(failed.toString(), 0, failed.size());
    }

    @Test
    public void testConv2d2() {
        //avg pool, batch norm, conv2d, deconv2d, depthwise2d, LRN, max pool 2d, pooling2d, sconv2d, upsamilpng

        Nd4j.getRandom().setSeed(12345);

        int[][] inputSizes = new int[][]{{1, 3, 8, 8}, {3, 6, 12, 12}};

        List<String> failed = new ArrayList<>();

        for (int i = 4; i <= 4; i++) {
            for (int[] inSizeNCHW : inputSizes) {

                SameDiff sd = SameDiff.create();
                SDVariable in = null;

                int[] inSize;

                SDVariable out;
                String msg;
                switch (i) {
                    case 4:
                        //Avg pool, NHWC, not same
                        msg = "3 - avg pool, NHWC, not same - input " + Arrays.toString(inSizeNCHW);
                        inSize = nchwToNhwc(inSizeNCHW);
                        in = sd.var("in", inSize);
                        out = sd.avgPooling2d(in, Pooling2DConfig.builder()
                                .isNHWC(true)
                                .isSameMode(false)
                                .kh(3).kw(2)
                                .sx(2).sy(2)
                                .build());
                        break;
                    default:
                        throw new RuntimeException();

                }

                INDArray inArr = Nd4j.rand(inSize).muli(10);
                in.setArray(inArr);
                SDVariable loss = sd.standardDeviation("loss", out, true);

                TestCase tc = new TestCase(sd);
                String error = OpValidation.validate(tc);
                if (error != null) {
                    failed.add(name);
                }

            }
        }

        assertEquals(failed.toString(), 0, failed.size());
    }

    @Test
    public void testIm2Col() {

        Nd4j.getRandom().setSeed(12345);

        int[][] inputSizes = new int[][]{{1, 3, 8, 8}, {3, 6, 12, 12}};

        List<String> failed = new ArrayList<>();

        for (int[] inSizeNCHW : inputSizes) {

            SameDiff sd = SameDiff.create();
            SDVariable var = sd.var("in", Nd4j.create(inSizeNCHW));
            SDVariable im2col = sd.im2Col(var, Conv2DConfig.builder()
                    .kh(2).kw(2)
                    .sx(1).sy(1)
                    .isSameMode(true)
                    .build());

            SDVariable loss = sd.standardDeviation("loss", im2col, true);

            String msg = Arrays.toString(inSizeNCHW);

            TestCase tc = new TestCase(sd);
            String error = OpValidation.validate(tc);
            if (error != null) {
                failed.add(name);
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
                .kh(3).kw(2)
                .sx(2).sy(2)
                .build();

        INDArray input = Nd4j.create(inSize);
        AvgPooling2D avgPooling2D = AvgPooling2D.builder()
                .arrayInput(input)
                .config(conf)
                .build();

        List<long[]> outSizes = Nd4j.getExecutioner().calculateOutputShape(avgPooling2D);

        assertEquals(1, outSizes.size());

        //NO SAME: out = (in - k + 2*p)/s + 1;
        int outH = (8 - 3) / 2 + 1;
        int outW = (8 - 2) / 2 + 1;
        long[] exp = new long[]{1, outH, outW, 3};    //NHWC

        assertEquals(1, outSizes.size());
        assertArrayEquals(exp, outSizes.get(0));

        INDArray grad = Nd4j.create(exp);


        //Test backprop:
        Pooling2DDerivative avg2dDeriv = Pooling2DDerivative.derivativeBuilder()
                .arrayInputs(new INDArray[]{input, grad})
                .config(conf)
                .build();

        List<long[]> outSizesBP = Nd4j.getExecutioner().calculateOutputShape(avg2dDeriv);
        assertEquals(1, outSizesBP.size());

        assertArrayEquals(inSize, outSizesBP.get(0));
    }


    @Test
    public void testAvgPool() {
        long[] inSize = {1, 8, 8, 3};  //NHWC

        Pooling2DConfig conf = Pooling2DConfig.builder()
                .isNHWC(true)   //***NHWC
                .isSameMode(false)
                .kh(3).kw(2)
                .sx(2).sy(2)
                .type(Pooling2D.Pooling2DType.AVG)
                .build();

        INDArray input = Nd4j.create(inSize);
        AvgPooling2D avgPooling2D = AvgPooling2D.builder()
                .arrayInput(input)
                .config(conf)
                .build();

        List<long[]> outSizes = Nd4j.getExecutioner().calculateOutputShape(avgPooling2D);
        assertEquals(1, outSizes.size());

        //NO SAME: out = (in - k + 2*p)/s + 1;
        int outH = (8 - 3) / 2 + 1;
        int outW = (8 - 2) / 2 + 1;
        long[] exp = new long[]{1, outH, outW, 3};    //NHWC

        assertEquals(1, outSizes.size());
        assertArrayEquals(exp, outSizes.get(0));

        INDArray grad = Nd4j.create(exp);

        //Test backprop:
        Pooling2DDerivative avg2dDeriv = Pooling2DDerivative.derivativeBuilder()
                .arrayInputs(new INDArray[]{input, grad})           //Original input, and output gradient (eps - same shape as output)
                .arrayOutputs(new INDArray[]{Nd4j.create(inSize)})  //Output for BP: same shape as original input
                .config(conf)
                .build();

        List<long[]> outSizesBP = Nd4j.getExecutioner().calculateOutputShape(avg2dDeriv);
        assertEquals(1, outSizesBP.size());
        assertArrayEquals(inSize, outSizesBP.get(0));

        Nd4j.getExecutioner().exec(avg2dDeriv);
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
                int[] shape = (ncdhw ? inSizeNCDHW : ncdhwToNdhwc(inSizeNCDHW));

                for( int i=0; i<4; i++ ) {
                    SameDiff sd = SameDiff.create();
                    SDVariable in = sd.var("in", shape);

                    SDVariable out;
                    String msg;
                    switch (i) {
                        case 0:
                            //Conv3d, with bias, same
                            msg = "0 - conv3d+bias+same, ncdhw=" + ncdhw + " - input " + Arrays.toString(shape);
                            SDVariable w0;
                            if(ncdhw){
                                w0 = sd.var("w0", Nd4j.rand(new int[]{3, shape[1], 2, 2, 2}).muli(10));  //NCDHW: [oC, iC, kD, kH, kW]
                            } else {
                                w0 = sd.var("w0", Nd4j.rand(new int[]{2, 2, 2, 3, shape[1]}).muli(10));  //NDHWC: [kD, kH, kW, iC, oC]
                            }
                            SDVariable b0 = sd.var("b0", Nd4j.rand(new long[]{3}).muli(10));
                            out = sd.conv3d(in, w0, b0, Conv3DConfig.builder()
                                    .isNCDHW(ncdhw)
                                    .isValidMode(false)
                                    .kH(2).kW(2).kT(2)
                                    //strides
                                    .dT(1).dH(1).dW(1)
                                    .build());
                            break;
                        case 1:
                            //Conv3d, no bias, no same
                            msg = "1 - conv3d+no bias+no same, ncdhw=" + ncdhw + " - input " + Arrays.toString(shape);
                            SDVariable w1;
                            if(ncdhw){
                                w1 = sd.var("w1", Nd4j.rand(new int[]{3, shape[1], 2, 2, 2}).muli(10));  //NCDHW: [oC, iC, kD, kH, kW]
                            } else {
                                w1 = sd.var("w1", Nd4j.rand(new int[]{2, 2, 2, 3, shape[1]}).muli(10));  //NDHWC: [kD, kH, kW, iC, oC]
                            }
                            out = sd.conv3d(in, w1, Conv3DConfig.builder()
                                    .isNCDHW(ncdhw)
                                    .isValidMode(true)
                                    .kH(2).kW(2).kT(2)
                                    //strides
                                    .dT(1).dH(1).dW(1)
                                    .build());
                            break;
                        case 2:
                            //pooling3d - average, same
                            msg = "2 - pooling 3d, average, same";
                            out = sd.avgPooling3d(in, Pooling3DConfig.builder()
                                    .kH(2).kW(2).kT(2)
                                    .sH(1).sW(1).sT(1)
                                    .ceilingMode(false)
                                    .build());
                            break;
                        case 3:
                            //pooling 3d - max, no same
                            msg = "3 - pooling 3d, max, no same";
                            out = sd.avgPooling3d(in, Pooling3DConfig.builder()
                                    .kH(2).kW(2).kT(2)
                                    .sH(1).sW(1).sT(1)
                                    .ceilingMode(true)
                                    .build());
                            break;
                        case 4:
                            //Batch norm - 3d input
                            throw new RuntimeException("Batch norm test not yet implemented");
                        default:
                            throw new RuntimeException();

                    }

                    INDArray inArr = Nd4j.rand(shape).muli(10);
                    in.setArray(inArr);
                    SDVariable loss = sd.standardDeviation("loss", out, true);

                    TestCase tc = new TestCase(sd);
                    String error = OpValidation.validate(tc);
                    if (error != null) {
                        failed.add(name);
                    }
                }
            }
        }

        assertEquals(failed.toString(), 0, failed.size());
    }


}
