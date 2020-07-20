/* ******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
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
package org.deeplearning4j.cuda.convolution;

import lombok.*;
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.cuda.CuDNNTestUtils;
import org.deeplearning4j.cuda.TestUtils;
import org.deeplearning4j.nn.api.MaskState;
import org.deeplearning4j.nn.conf.CNN2DFormat;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.layers.convolutional.Cropping2D;
import org.deeplearning4j.nn.conf.preprocessor.CnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.ComposableInputPreProcessor;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.common.primitives.Pair;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;

@RunWith(Parameterized.class)
public class ConvDataFormatTests extends BaseDL4JTest {

    private final DataType dataType;

    public ConvDataFormatTests(DataType dataType){
        this.dataType = dataType;
    }

    @Parameterized.Parameters(name = "{0}")
    public static Object[] params(){
        return new DataType[]{DataType.FLOAT, DataType.DOUBLE};
    }

    @Test
    public void testConv2d() {
        try {
            for (boolean helpers : new boolean[]{false, true}) {
                for (ConvolutionMode cm : new ConvolutionMode[]{ConvolutionMode.Truncate, ConvolutionMode.Same}) {
                    Nd4j.getRandom().setSeed(12345);
                    Nd4j.getEnvironment().allowHelpers(helpers);
                    String msg = helpers ? "With helpers (" + cm + ")" : "No helpers (" + cm + ")";
                    System.out.println(" --- " + msg + " ---");

                    INDArray inNCHW = Nd4j.rand(this.dataType, 2, 3, 12, 12);
                    INDArray labels = TestUtils.randomOneHot(2, 10);

                    TestCase tc = TestCase.builder()
                            .msg(msg)
                            .net1(getConv2dNet(CNN2DFormat.NCHW, true, cm))
                            .net2(getConv2dNet(CNN2DFormat.NCHW, false, cm))
                            .net3(getConv2dNet(CNN2DFormat.NHWC, true, cm))
                            .net4(getConv2dNet(CNN2DFormat.NHWC, false, cm))
                            .inNCHW(inNCHW)
                            .labelsNCHW(labels)
                            .labelsNHWC(labels)
                            .testLayerIdx(1)
                            .helpers(helpers)
                            .build();

                    testHelper(tc);
                }
            }
        } finally {
            Nd4j.getEnvironment().allowHelpers(true);
        }
    }

    @Test
    public void testSubsampling2d() {
        try {
            for (boolean helpers : new boolean[]{false, true}) {
                for (ConvolutionMode cm : new ConvolutionMode[]{ConvolutionMode.Truncate, ConvolutionMode.Same}) {
                    Nd4j.getRandom().setSeed(12345);
                    Nd4j.getEnvironment().allowHelpers(helpers);
                    String msg = helpers ? "With helpers (" + cm + ")" : "No helpers (" + cm + ")";
                    System.out.println(" --- " + msg + " ---");

                    INDArray inNCHW = Nd4j.rand(this.dataType, 2, 3, 12, 12);
                    INDArray labels = TestUtils.randomOneHot(2, 10);

                    TestCase tc = TestCase.builder()
                            .msg(msg)
                            .net1(getSubsampling2dNet(CNN2DFormat.NCHW, true, cm))
                            .net2(getSubsampling2dNet(CNN2DFormat.NCHW, false, cm))
                            .net3(getSubsampling2dNet(CNN2DFormat.NHWC, true, cm))
                            .net4(getSubsampling2dNet(CNN2DFormat.NHWC, false, cm))
                            .inNCHW(inNCHW)
                            .labelsNCHW(labels)
                            .labelsNHWC(labels)
                            .testLayerIdx(1)
                            .helpers(helpers)
                            .build();

                    testHelper(tc);
                }
            }
        } finally {
            Nd4j.getEnvironment().allowHelpers(true);
        }
    }

    @Test
    public void testDepthwiseConv2d() {
        try {
            for (boolean helpers : new boolean[]{false, true}) {
                for (ConvolutionMode cm : new ConvolutionMode[]{ConvolutionMode.Truncate, ConvolutionMode.Same}) {
                    Nd4j.getRandom().setSeed(12345);
                    Nd4j.getEnvironment().allowHelpers(helpers);
                    String msg = helpers ? "With helpers (" + cm + ")" : "No helpers (" + cm + ")";
                    System.out.println(" --- " + msg + " ---");

                    INDArray inNCHW = Nd4j.rand(this.dataType, 2, 3, 12, 12);
                    INDArray labels = TestUtils.randomOneHot(2, 10);

                    TestCase tc = TestCase.builder()
                            .msg(msg)
                            .net1(getDepthwiseConv2dNet(CNN2DFormat.NCHW, true, cm))
                            .net2(getDepthwiseConv2dNet(CNN2DFormat.NCHW, false, cm))
                            .net3(getDepthwiseConv2dNet(CNN2DFormat.NHWC, true, cm))
                            .net4(getDepthwiseConv2dNet(CNN2DFormat.NHWC, false, cm))
                            .inNCHW(inNCHW)
                            .labelsNCHW(labels)
                            .labelsNHWC(labels)
                            .testLayerIdx(1)
                            .helpers(helpers)
                            .build();

                    testHelper(tc);
                }
            }
        } finally {
            Nd4j.getEnvironment().allowHelpers(true);
        }
    }

    @Test
    public void testSeparableConv2d() {
        try {
            for (boolean helpers : new boolean[]{false, true}) {
                for (ConvolutionMode cm : new ConvolutionMode[]{ConvolutionMode.Truncate, ConvolutionMode.Same}) {
                    Nd4j.getRandom().setSeed(12345);
                    Nd4j.getEnvironment().allowHelpers(helpers);
                    String msg = helpers ? "With helpers (" + cm + ")" : "No helpers (" + cm + ")";
                    System.out.println(" --- " + msg + " ---");

                    INDArray inNCHW = Nd4j.rand(this.dataType, 2, 3, 12, 12);
                    INDArray labels = TestUtils.randomOneHot(2, 10);

                    TestCase tc = TestCase.builder()
                            .msg(msg)
                            .net1(getSeparableConv2dNet(CNN2DFormat.NCHW, true, cm))
                            .net2(getSeparableConv2dNet(CNN2DFormat.NCHW, false, cm))
                            .net3(getSeparableConv2dNet(CNN2DFormat.NHWC, true, cm))
                            .net4(getSeparableConv2dNet(CNN2DFormat.NHWC, false, cm))
                            .inNCHW(inNCHW)
                            .labelsNCHW(labels)
                            .labelsNHWC(labels)
                            .testLayerIdx(1)
                            .build();

                    testHelper(tc);
                }
            }
        } finally {
            Nd4j.getEnvironment().allowHelpers(true);
        }
    }

    @Test
    public void testDeconv2d() {
        try {
            for (boolean helpers : new boolean[]{false, true}) {
                for (ConvolutionMode cm : new ConvolutionMode[]{ConvolutionMode.Truncate, ConvolutionMode.Same}) {
                    Nd4j.getRandom().setSeed(12345);
                    Nd4j.getEnvironment().allowHelpers(helpers);
                    String msg = helpers ? "With helpers (" + cm + ")" : "No helpers (" + cm + ")";
                    System.out.println(" --- " + msg + " ---");

                    INDArray inNCHW = Nd4j.rand(this.dataType, 2, 3, 12, 12);
                    INDArray labels = TestUtils.randomOneHot(2, 10);

                    TestCase tc = TestCase.builder()
                            .msg(msg)
                            .net1(getDeconv2DNet2dNet(CNN2DFormat.NCHW, true, cm))
                            .net2(getDeconv2DNet2dNet(CNN2DFormat.NCHW, false, cm))
                            .net3(getDeconv2DNet2dNet(CNN2DFormat.NHWC, true, cm))
                            .net4(getDeconv2DNet2dNet(CNN2DFormat.NHWC, false, cm))
                            .inNCHW(inNCHW)
                            .labelsNCHW(labels)
                            .labelsNHWC(labels)
                            .testLayerIdx(1)
                            .helpers(helpers)
                            .build();

                    testHelper(tc);
                }
            }
        } finally {
            Nd4j.getEnvironment().allowHelpers(true);
        }
    }

    @Test
    public void testLRN() {
        try {
            for (boolean helpers : new boolean[]{false, true}) {
                for (ConvolutionMode cm : new ConvolutionMode[]{ConvolutionMode.Truncate, ConvolutionMode.Same}) {
                    Nd4j.getRandom().setSeed(12345);
                    Nd4j.getEnvironment().allowHelpers(helpers);
                    String msg = helpers ? "With helpers (" + cm + ")" : "No helpers (" + cm + ")";
                    System.out.println(" --- " + msg + " ---");

                    INDArray inNCHW = Nd4j.rand(this.dataType, 2, 3, 12, 12);
                    INDArray labels = TestUtils.randomOneHot(2, 10);

                    TestCase tc = TestCase.builder()
                            .msg(msg)
                            .net1(getLrnLayer(CNN2DFormat.NCHW, true, cm))
                            .net2(getLrnLayer(CNN2DFormat.NCHW, false, cm))
                            .net3(getLrnLayer(CNN2DFormat.NHWC, true, cm))
                            .net4(getLrnLayer(CNN2DFormat.NHWC, false, cm))
                            .inNCHW(inNCHW)
                            .labelsNCHW(labels)
                            .labelsNHWC(labels)
                            .testLayerIdx(1)
                            .helpers(helpers)
                            .build();

                    testHelper(tc);
                }
            }
        } finally {
            Nd4j.getEnvironment().allowHelpers(true);
        }
    }

    @Test
    public void testZeroPaddingLayer(){
        try {
            for (boolean helpers : new boolean[]{false, true}) {
                Nd4j.getRandom().setSeed(12345);
                Nd4j.getEnvironment().allowHelpers(helpers);
                String msg = helpers ? "With helpers" : "No helpers";
                System.out.println(" --- " + msg + " ---");

                INDArray inNCHW = Nd4j.rand(this.dataType, 2, 3, 12, 12);
                INDArray labels = TestUtils.randomOneHot(2, 10);

                TestCase tc = TestCase.builder()
                        .msg(msg)
                        .net1(getZeroPaddingNet(CNN2DFormat.NCHW, true))
                        .net2(getZeroPaddingNet(CNN2DFormat.NCHW, false))
                        .net3(getZeroPaddingNet(CNN2DFormat.NHWC, true))
                        .net4(getZeroPaddingNet(CNN2DFormat.NHWC, false))
                        .inNCHW(inNCHW)
                        .labelsNCHW(labels)
                        .labelsNHWC(labels)
                        .testLayerIdx(1)
                        .helpers(helpers)
                        .build();

                testHelper(tc);
            }
        } finally {
            Nd4j.getEnvironment().allowHelpers(true);
        }
    }

    @Test
    public void testCropping2DLayer(){
        try {
            for (boolean helpers : new boolean[]{false, true}) {
                Nd4j.getRandom().setSeed(12345);
                Nd4j.getEnvironment().allowHelpers(helpers);
                String msg = helpers ? "With helpers" : "No helpers";
                System.out.println(" --- " + msg + " ---");

                INDArray inNCHW = Nd4j.rand(this.dataType, 2, 3, 12, 12);
                INDArray labels = TestUtils.randomOneHot(2, 10);

                TestCase tc = TestCase.builder()
                        .msg(msg)
                        .net1(getCropping2dNet(CNN2DFormat.NCHW, true))
                        .net2(getCropping2dNet(CNN2DFormat.NCHW, false))
                        .net3(getCropping2dNet(CNN2DFormat.NHWC, true))
                        .net4(getCropping2dNet(CNN2DFormat.NHWC, false))
                        .inNCHW(inNCHW)
                        .labelsNCHW(labels)
                        .labelsNHWC(labels)
                        .testLayerIdx(1)
                        .helpers(helpers)
                        .build();

                testHelper(tc);
            }
        } finally {
            Nd4j.getEnvironment().allowHelpers(true);
        }
    }

    @Test
    public void testUpsampling2d(){
        try {
            for (boolean helpers : new boolean[]{false, true}) {
                Nd4j.getRandom().setSeed(12345);
                Nd4j.getEnvironment().allowHelpers(helpers);
                String msg = helpers ? "With helpers" : "No helpers";
                System.out.println(" --- " + msg + " ---");

                INDArray inNCHW = Nd4j.rand(this.dataType, 2, 3, 12, 12);
                INDArray labels = TestUtils.randomOneHot(2, 10);

                TestCase tc = TestCase.builder()
                        .msg(msg)
                        .net1(getUpsamplingNet(CNN2DFormat.NCHW, true))
                        .net2(getUpsamplingNet(CNN2DFormat.NCHW, false))
                        .net3(getUpsamplingNet(CNN2DFormat.NHWC, true))
                        .net4(getUpsamplingNet(CNN2DFormat.NHWC, false))
                        .inNCHW(inNCHW)
                        .labelsNCHW(labels)
                        .labelsNHWC(labels)
                        .testLayerIdx(1)
                        .helpers(helpers)
                        .build();

                testHelper(tc);
            }
        } finally {
            Nd4j.getEnvironment().allowHelpers(true);
        }
    }

    @Test
    public void testBatchNormNet(){
        try {
            for(boolean useLogStd : new boolean[]{true, false}) {
                for (boolean helpers : new boolean[]{false, true}) {
                    Nd4j.getRandom().setSeed(12345);
                    Nd4j.getEnvironment().allowHelpers(helpers);
                    String msg = (helpers ? "With helpers" : "No helpers") + " - " + (useLogStd ? "logstd" : "std");
                    System.out.println(" --- " + msg + " ---");

                    INDArray inNCHW = Nd4j.rand(this.dataType, 2, 3, 12, 12);
                    INDArray labels = TestUtils.randomOneHot(2, 10);

                    TestCase tc = TestCase.builder()
                            .msg(msg)
                            .net1(getBatchNormNet(useLogStd, CNN2DFormat.NCHW, true))
                            .net2(getBatchNormNet(useLogStd, CNN2DFormat.NCHW, false))
                            .net3(getBatchNormNet(useLogStd, CNN2DFormat.NHWC, true))
                            .net4(getBatchNormNet(useLogStd, CNN2DFormat.NHWC, false))
                            .inNCHW(inNCHW)
                            .labelsNCHW(labels)
                            .labelsNHWC(labels)
                            .testLayerIdx(1)
                            .helpers(helpers)
                            .build();

                    testHelper(tc);
                }
            }
        } finally {
            Nd4j.getEnvironment().allowHelpers(true);
        }
    }

    @Test
    public void testCnnLossLayer() {
        try {
            for (boolean helpers : new boolean[]{false, true}) {
                Nd4j.getRandom().setSeed(12345);
                Nd4j.getEnvironment().allowHelpers(helpers);
                String msg = helpers ? "With helpers" : "No helpers";
                System.out.println(" --- " + msg + " ---");

                INDArray inNCHW = Nd4j.rand(this.dataType, 2, 3, 12, 12);
                INDArray labelsNHWC = TestUtils.randomOneHot(this.dataType,2*6*6, 3);
                labelsNHWC = labelsNHWC.reshape(2,6,6,3);
                INDArray labelsNCHW = labelsNHWC.permute(0,3,1,2).dup();



                TestCase tc = TestCase.builder()
                        .msg(msg)
                        .net1(getCnnLossNet(CNN2DFormat.NCHW, true, ConvolutionMode.Same))
                        .net2(getCnnLossNet(CNN2DFormat.NCHW, false, ConvolutionMode.Same))
                        .net3(getCnnLossNet(CNN2DFormat.NHWC, true, ConvolutionMode.Same))
                        .net4(getCnnLossNet(CNN2DFormat.NHWC, false, ConvolutionMode.Same))
                        .inNCHW(inNCHW)
                        .labelsNCHW(labelsNCHW)
                        .labelsNHWC(labelsNHWC)
                        .testLayerIdx(1)
                        .nhwcOutput(true)
                        .helpers(helpers)
                        .build();

                testHelper(tc);
            }
        } finally {
            Nd4j.getEnvironment().allowHelpers(true);
        }
    }

    @Test
    public void testSpaceToDepthNet(){
        try {
            for (boolean helpers : new boolean[]{false, true}) {
                Nd4j.getRandom().setSeed(12345);
                Nd4j.getEnvironment().allowHelpers(helpers);
                String msg = helpers ? "With helpers" : "No helpers";
                System.out.println(" --- " + msg + " ---");

                INDArray inNCHW = Nd4j.rand(this.dataType, 2, 3, 12, 12);
                INDArray labels = TestUtils.randomOneHot(2, 10);

                TestCase tc = TestCase.builder()
                        .msg(msg)
                        .net1(getSpaceToDepthNet(CNN2DFormat.NCHW, true))
                        .net2(getSpaceToDepthNet(CNN2DFormat.NCHW, false))
                        .net3(getSpaceToDepthNet(CNN2DFormat.NHWC, true))
                        .net4(getSpaceToDepthNet(CNN2DFormat.NHWC, false))
                        .inNCHW(inNCHW)
                        .labelsNCHW(labels)
                        .labelsNHWC(labels)
                        .testLayerIdx(1)
                        .helpers(helpers)
                        .build();

                testHelper(tc);
            }
        } finally {
            Nd4j.getEnvironment().allowHelpers(true);
        }
    }

    @Test
    public void testSpaceToBatchNet(){
        try {
            for (boolean helpers : new boolean[]{false, true}) {
                Nd4j.getRandom().setSeed(12345);
                Nd4j.getEnvironment().allowHelpers(helpers);
                String msg = helpers ? "With helpers" : "No helpers";
                System.out.println(" --- " + msg + " ---");

                INDArray inNCHW = Nd4j.rand(this.dataType, 2, 3, 16, 16);
                INDArray labels = TestUtils.randomOneHot(8, 10);

                TestCase tc = TestCase.builder()
                        .msg(msg)
                        .net1(getSpaceToBatchNet(CNN2DFormat.NCHW, true))
                        .net2(getSpaceToBatchNet(CNN2DFormat.NCHW, false))
                        .net3(getSpaceToBatchNet(CNN2DFormat.NHWC, true))
                        .net4(getSpaceToBatchNet(CNN2DFormat.NHWC, false))
                        .inNCHW(inNCHW)
                        .labelsNCHW(labels)
                        .labelsNHWC(labels)
                        .testLayerIdx(1)
                        .helpers(helpers)
                        .build();

                testHelper(tc);
            }
        } finally {
            Nd4j.getEnvironment().allowHelpers(true);
        }
    }

    @Test
    public void testLocallyConnected() {
        try {
            for (boolean helpers : new boolean[]{false, true}) {
                for (ConvolutionMode cm : new ConvolutionMode[]{ConvolutionMode.Truncate, ConvolutionMode.Same}) {
                    Nd4j.getRandom().setSeed(12345);
                    Nd4j.getEnvironment().allowHelpers(helpers);
                    String msg = helpers ? "With helpers (" + cm + ")" : "No helpers (" + cm + ")";
                    System.out.println(" --- " + msg + " ---");

                    INDArray inNCHW = Nd4j.rand(this.dataType, 2, 3, 12, 12);
                    INDArray labels = TestUtils.randomOneHot(2, 10);

                    TestCase tc = TestCase.builder()
                            .msg(msg)
                            .net1(getLocallyConnectedNet(CNN2DFormat.NCHW, true, cm))
                            .net2(getLocallyConnectedNet(CNN2DFormat.NCHW, false, cm))
                            .net3(getLocallyConnectedNet(CNN2DFormat.NHWC, true, cm))
                            .net4(getLocallyConnectedNet(CNN2DFormat.NHWC, false, cm))
                            .inNCHW(inNCHW)
                            .labelsNCHW(labels)
                            .labelsNHWC(labels)
                            .testLayerIdx(1)
                            .helpers(helpers)
                            .build();

                    testHelper(tc);
                }
            }
        } finally {
            Nd4j.getEnvironment().allowHelpers(true);
        }
    }

    @Test
    public void testGlobalPooling() {
        try {
            for (boolean helpers : new boolean[]{false, true}) {
                for (PoolingType pt : PoolingType.values()) {
                    Nd4j.getRandom().setSeed(12345);
                    Nd4j.getEnvironment().allowHelpers(helpers);
                    String msg = helpers ? "With helpers (" + pt + ")" : "No helpers (" + pt + ")";
                    System.out.println(" --- " + msg + " ---");

                    INDArray inNCHW = Nd4j.rand(this.dataType, 2, 3, 12, 12);
                    INDArray labels = TestUtils.randomOneHot(2, 10);

                    TestCase tc = TestCase.builder()
                            .msg(msg)
                            .net1(getGlobalPoolingNet(CNN2DFormat.NCHW, pt, true))
                            .net2(getGlobalPoolingNet(CNN2DFormat.NCHW, pt, false))
                            .net3(getGlobalPoolingNet(CNN2DFormat.NHWC, pt, true))
                            .net4(getGlobalPoolingNet(CNN2DFormat.NHWC, pt, false))
                            .inNCHW(inNCHW)
                            .labelsNCHW(labels)
                            .labelsNHWC(labels)
                            .testLayerIdx(1)
                            .build();

                    testHelper(tc);
                }
            }
        } finally {
            Nd4j.getEnvironment().allowHelpers(true);
        }
    }

    private MultiLayerNetwork getConv2dNet(CNN2DFormat format, boolean setOnLayerAlso, ConvolutionMode cm) {
        if (setOnLayerAlso) {
            return getNetWithLayer(new ConvolutionLayer.Builder()
                    .kernelSize(3, 3)
                    .stride(2, 2)
                    .activation(Activation.TANH)
                    .dataFormat(format)
                    .nOut(3)
                    .helperAllowFallback(false)
                    .build(), format, cm, null);
        } else {
            return getNetWithLayer(new ConvolutionLayer.Builder()
                    .kernelSize(3, 3)
                    .stride(2, 2)
                    .activation(Activation.TANH)
                    .nOut(3)
                    .helperAllowFallback(false)
                    .build(), format, cm, null);
        }
    }

    private MultiLayerNetwork getSubsampling2dNet(CNN2DFormat format, boolean setOnLayerAlso, ConvolutionMode cm) {
        if (setOnLayerAlso) {
            return getNetWithLayer(new SubsamplingLayer.Builder()
                            .kernelSize(2, 2)
                            .stride(1, 1)
                            .dataFormat(format)
                            .helperAllowFallback(false)
                            .build(), format, cm, null);
        } else {
            return getNetWithLayer(new SubsamplingLayer.Builder()
                            .kernelSize(2, 2)
                            .stride(1, 1)
                            .helperAllowFallback(false)
                            .build(), format, cm, null);
        }
    }

    private MultiLayerNetwork getSeparableConv2dNet(CNN2DFormat format, boolean setOnLayerAlso, ConvolutionMode cm) {
        if (setOnLayerAlso) {
            return getNetWithLayer(new SeparableConvolution2D.Builder()
                    .kernelSize(3, 3)
                    .stride(2, 2)
                    .activation(Activation.TANH)
                    .dataFormat(format)
                    .nOut(3)
                    .helperAllowFallback(false)
                    .build(), format, cm, null);
        } else {
            return getNetWithLayer(new SeparableConvolution2D.Builder()
                    .kernelSize(3, 3)
                    .stride(2, 2)
                    .activation(Activation.TANH)
                    .nOut(3)
                    .helperAllowFallback(false)
                    .build(), format, cm, null);
        }
    }

    private MultiLayerNetwork getDepthwiseConv2dNet(CNN2DFormat format, boolean setOnLayerAlso, ConvolutionMode cm) {
        if (setOnLayerAlso) {
            return getNetWithLayer(new DepthwiseConvolution2D.Builder()
                    .depthMultiplier(2)
                    .kernelSize(3, 3)
                    .stride(2, 2)
                    .activation(Activation.TANH)
                    .dataFormat(format)
                    .nOut(3)
                    .helperAllowFallback(false)
                    .build(), format, cm, null);
        } else {
            return getNetWithLayer(new DepthwiseConvolution2D.Builder()
                    .depthMultiplier(2)
                    .kernelSize(3, 3)
                    .stride(2, 2)
                    .activation(Activation.TANH)
                    .nOut(3)
                    .helperAllowFallback(false)
                    .build(), format, cm, null);
        }
    }

    private MultiLayerNetwork getLrnLayer(CNN2DFormat format, boolean setOnLayerAlso, ConvolutionMode cm) {
        if (setOnLayerAlso) {
            return getNetWithLayer(new LocalResponseNormalization.Builder()
                    .dataFormat(format)
                    .helperAllowFallback(false)
                    .build(), format, cm, null);
        } else {
            return getNetWithLayer(new LocalResponseNormalization.Builder()
                    .helperAllowFallback(false)
                    .build(), format, cm, null);
        }
    }

    private MultiLayerNetwork getZeroPaddingNet(CNN2DFormat format, boolean setOnLayerAlso) {
        if (setOnLayerAlso) {
            return getNetWithLayer(new ZeroPaddingLayer.Builder(2,2)
                            .dataFormat(format).build(), format, ConvolutionMode.Same, null);
        } else {
            return getNetWithLayer(new ZeroPaddingLayer.Builder(2,2).build(),
                    format, ConvolutionMode.Same, null);
        }
    }

    private MultiLayerNetwork getCropping2dNet(CNN2DFormat format, boolean setOnLayerAlso) {
        if (setOnLayerAlso) {
           return getNetWithLayer(new Cropping2D.Builder(2,2)
                            .dataFormat(format).build(), format, ConvolutionMode.Same, null);
        } else {
            return getNetWithLayer(new Cropping2D.Builder(2,2)
                    .build(), format, ConvolutionMode.Same, null);
        }
    }

    private MultiLayerNetwork getUpsamplingNet(CNN2DFormat format, boolean setOnLayerAlso) {
        if (setOnLayerAlso) {
            return getNetWithLayer(new Upsampling2D.Builder(2)
                    .dataFormat(format).build(), format, ConvolutionMode.Same, null);
        } else {
            return getNetWithLayer(new Upsampling2D.Builder(2)
                    .build(), format, ConvolutionMode.Same, null);
        }
    }

    private MultiLayerNetwork getDeconv2DNet2dNet(CNN2DFormat format, boolean setOnLayerAlso, ConvolutionMode cm) {
        if (setOnLayerAlso) {
            return getNetWithLayer(new Deconvolution2D.Builder().nOut(2)
                    .activation(Activation.TANH)
                    .kernelSize(2,2)
                    .stride(2,2)
                    .build(), format, cm, null);
        } else {
            return getNetWithLayer(new Deconvolution2D.Builder().nOut(2)
                    .activation(Activation.TANH)
                    .kernelSize(2,2)
                    .stride(2,2)
                    .build(), format, cm, null);
        }
    }

    private MultiLayerNetwork getBatchNormNet(boolean logStdev, CNN2DFormat format, boolean setOnLayerAlso) {
        if (setOnLayerAlso) {
            return getNetWithLayer(new BatchNormalization.Builder()
                    .useLogStd(logStdev)
                    .dataFormat(format)
                    .helperAllowFallback(false)
                    .nOut(3).build(), format, ConvolutionMode.Same, null);
        } else {
            return getNetWithLayer(new BatchNormalization.Builder()
                    .useLogStd(logStdev)
                    .helperAllowFallback(false)
                    .nOut(3).build(), format, ConvolutionMode.Same, null);
        }
    }

    private MultiLayerNetwork getSpaceToDepthNet(CNN2DFormat format, boolean setOnLayerAlso) {
        if (setOnLayerAlso) {
            return getNetWithLayer(new SpaceToDepthLayer.Builder()
                    .blocks(2)
                    .dataFormat(format)
                    .build(), format, ConvolutionMode.Same, null);
        } else {
            return getNetWithLayer(new SpaceToDepthLayer.Builder()
                    .blocks(2)
                    .build(), format, ConvolutionMode.Same, null);
        }
    }

    private MultiLayerNetwork getSpaceToBatchNet(CNN2DFormat format, boolean setOnLayerAlso) {
        if (setOnLayerAlso) {
            return getNetWithLayer(new SpaceToBatchLayer.Builder()
                    .blocks(2, 2)
                    .dataFormat(format)
                    .build(), format, ConvolutionMode.Same, InputType.convolutional(16, 16, 3, format));
        } else {
            return getNetWithLayer(new SpaceToBatchLayer.Builder()
                    .blocks(2, 2)
                    .build(), format, ConvolutionMode.Same, InputType.convolutional(16, 16, 3, format));
        }
    }

    private MultiLayerNetwork getLocallyConnectedNet(CNN2DFormat format, boolean setOnLayerAlso, ConvolutionMode cm) {
        if (setOnLayerAlso) {
            return getNetWithLayer(new LocallyConnected2D.Builder()
                    .kernelSize(3, 3)
                    .stride(2, 2)
                    .activation(Activation.TANH)
                    .dataFormat(format)
                    .nOut(3)
                    .build(), format, cm, null);
        } else {
            return getNetWithLayer(new LocallyConnected2D.Builder()
                    .kernelSize(3, 3)
                    .stride(2, 2)
                    .activation(Activation.TANH)
                    .nOut(3)
                    .build(), format, cm, null);
        }
    }

    private MultiLayerNetwork getGlobalPoolingNet(CNN2DFormat format, PoolingType pt, boolean setOnLayerAlso) {
        if (setOnLayerAlso) {
            return getNetWithLayer(new GlobalPoolingLayer.Builder(pt)
                    .poolingDimensions(format == CNN2DFormat.NCHW ? new int[]{2,3} : new int[]{1,2})
                    .build(), format, ConvolutionMode.Same, null);
        } else {
            return getNetWithLayer(new GlobalPoolingLayer.Builder(pt)
                    .build(), format, ConvolutionMode.Same, null);
        }
    }

    private MultiLayerNetwork getCnnLossNet(CNN2DFormat format, boolean setOnLayerAlso, ConvolutionMode cm){
        NeuralNetConfiguration.ListBuilder builder = new NeuralNetConfiguration.Builder()
                .seed(12345)
                .convolutionMode(cm)
                .list()
                .layer(new ConvolutionLayer.Builder()
                        .kernelSize(3, 3)
                        .stride(2, 2)
                        .activation(Activation.TANH)
                        .dataFormat(format)
                        .nOut(3)
                        .helperAllowFallback(false)
                        .build());
        if(setOnLayerAlso){
            builder.layer(new CnnLossLayer.Builder().format(format).activation(Activation.SOFTMAX).build());
        } else {
            builder.layer(new CnnLossLayer.Builder().activation(Activation.SOFTMAX).build());
        }

        builder.setInputType(InputType.convolutional(12, 12, 3, format));

        MultiLayerNetwork net = new MultiLayerNetwork(builder.build());
        net.init();
        return net;
    }

    private MultiLayerNetwork getNetWithLayer(Layer layer, CNN2DFormat format, ConvolutionMode cm, InputType inputType) {
        NeuralNetConfiguration.ListBuilder builder = new NeuralNetConfiguration.Builder()
                .dataType(this.dataType)
                .seed(12345)
                .convolutionMode(cm)
                .list()
                .layer(new ConvolutionLayer.Builder()
                        .kernelSize(3, 3)
                        .stride(2, 2)
                        .activation(Activation.TANH)
                        .dataFormat(format)
                        .nOut(3)
                        .helperAllowFallback(false)
                        .build())
                .layer(layer)
                .layer(new OutputLayer.Builder().activation(Activation.SOFTMAX).nOut(10).build())
                .setInputType(inputType != null ? inputType : InputType.convolutional(12, 12, 3, format));

        if(format == CNN2DFormat.NHWC && !(layer instanceof GlobalPoolingLayer)){
            //Add a preprocessor due to the differences in how NHWC and NCHW activations are flattened
            //DL4J's flattening behaviour matches Keras (hence TF) for import compatibility
            builder.inputPreProcessor(2, new ComposableInputPreProcessor(new NHWCToNCHWPreprocessor(), new CnnToFeedForwardPreProcessor()));
        }

        MultiLayerNetwork net = new MultiLayerNetwork(builder.build());
        net.init();
        return net;
    }

    @AllArgsConstructor
    @Data
    @NoArgsConstructor
    @Builder
    private static class TestCase {
        private String msg;
        private MultiLayerNetwork net1;
        private MultiLayerNetwork net2;
        private MultiLayerNetwork net3;
        private MultiLayerNetwork net4;
        private INDArray inNCHW;
        private INDArray labelsNCHW;
        private INDArray labelsNHWC;
        private int testLayerIdx;
        private boolean nhwcOutput;
        private boolean helpers;
    }

    public static void testHelper(TestCase tc) {

        if(!tc.helpers){
            try {
                CuDNNTestUtils.removeHelpers(tc.net1.getLayers());
                CuDNNTestUtils.removeHelpers(tc.net2.getLayers());
                CuDNNTestUtils.removeHelpers(tc.net3.getLayers());
                CuDNNTestUtils.removeHelpers(tc.net4.getLayers());
            } catch (Throwable t){
                throw new RuntimeException(t);
            }
        }


        tc.net2.params().assign(tc.net1.params());
        tc.net3.params().assign(tc.net1.params());
        tc.net4.params().assign(tc.net1.params());

        //Test forward pass:
        INDArray inNCHW = tc.inNCHW;
        INDArray inNHWC = tc.inNCHW.permute(0, 2, 3, 1).dup();

        INDArray l0_1 = tc.net1.feedForward(inNCHW).get(tc.testLayerIdx + 1);
        INDArray l0_2 = tc.net2.feedForward(inNCHW).get(tc.testLayerIdx + 1);
        INDArray l0_3 = tc.net3.feedForward(inNHWC).get(tc.testLayerIdx + 1);
        INDArray l0_4 = tc.net4.feedForward(inNHWC).get(tc.testLayerIdx + 1);

        assertEquals(tc.msg, l0_1, l0_2);
        if(l0_1.rank() == 4) {
            assertEquals(tc.msg, l0_1, l0_3.permute(0, 3, 1, 2));
            assertEquals(tc.msg, l0_1, l0_4.permute(0, 3, 1, 2));
        } else {
            assertEquals(tc.msg, l0_1, l0_3);
            assertEquals(tc.msg, l0_1, l0_4);
        }


        INDArray out1 = tc.net1.output(inNCHW);
        INDArray out2 = tc.net2.output(inNCHW);
        INDArray out3 = tc.net3.output(inNHWC);
        INDArray out4 = tc.net4.output(inNHWC);

        assertEquals(tc.msg, out1, out2);
        if(!tc.nhwcOutput) {
            assertEquals(tc.msg, out1, out3);
            assertEquals(tc.msg, out1, out4);
        } else {
            assertEquals(tc.msg, out1, out3.permute(0,3,1,2));      //NHWC to NCHW
            assertEquals(tc.msg, out1, out4.permute(0,3,1,2));
        }

        //Test backprop
        Pair<Gradient, INDArray> p1 = tc.net1.calculateGradients(inNCHW, tc.labelsNCHW, null, null);
        Pair<Gradient, INDArray> p2 = tc.net2.calculateGradients(inNCHW, tc.labelsNCHW, null, null);
        Pair<Gradient, INDArray> p3 = tc.net3.calculateGradients(inNHWC, tc.labelsNHWC, null, null);
        Pair<Gradient, INDArray> p4 = tc.net4.calculateGradients(inNHWC, tc.labelsNHWC, null, null);

            //Inpput gradients
        assertEquals(tc.msg, p1.getSecond(), p2.getSecond());
        assertEquals(tc.msg, p1.getSecond(), p3.getSecond().permute(0,3,1,2));  //Input gradients for NHWC input are also in NHWC format
        assertEquals(tc.msg, p1.getSecond(), p4.getSecond().permute(0,3,1,2));

        List<String> diff12 = differentGrads(p1.getFirst(), p2.getFirst());
        List<String> diff13 = differentGrads(p1.getFirst(), p3.getFirst());
        List<String> diff14 = differentGrads(p1.getFirst(), p4.getFirst());
        assertEquals(tc.msg + " " + diff12, 0, diff12.size());
        assertEquals(tc.msg + " " + diff13, 0, diff13.size());
        assertEquals(tc.msg + " " + diff14, 0, diff14.size());

        assertEquals(tc.msg, p1.getFirst().gradientForVariable(), p2.getFirst().gradientForVariable());
        assertEquals(tc.msg, p1.getFirst().gradientForVariable(), p3.getFirst().gradientForVariable());
        assertEquals(tc.msg, p1.getFirst().gradientForVariable(), p4.getFirst().gradientForVariable());

        tc.net1.fit(inNCHW, tc.labelsNCHW);
        tc.net2.fit(inNCHW, tc.labelsNCHW);
        tc.net3.fit(inNHWC, tc.labelsNHWC);
        tc.net4.fit(inNHWC, tc.labelsNHWC);

        assertEquals(tc.msg, tc.net1.params(), tc.net2.params());
        assertEquals(tc.msg, tc.net1.params(), tc.net3.params());
        assertEquals(tc.msg, tc.net1.params(), tc.net4.params());

        //Test serialization
        MultiLayerNetwork net1a = TestUtils.testModelSerialization(tc.net1);
        MultiLayerNetwork net2a = TestUtils.testModelSerialization(tc.net2);
        MultiLayerNetwork net3a = TestUtils.testModelSerialization(tc.net3);
        MultiLayerNetwork net4a = TestUtils.testModelSerialization(tc.net4);

        if(!tc.helpers){
            try {
                CuDNNTestUtils.removeHelpers(net1a.getLayers());
                CuDNNTestUtils.removeHelpers(net2a.getLayers());
                CuDNNTestUtils.removeHelpers(net3a.getLayers());
                CuDNNTestUtils.removeHelpers(net4a.getLayers());
            } catch (Throwable t){
                throw new RuntimeException(t);
            }
        }

        out1 = tc.net1.output(inNCHW);
        assertEquals(tc.msg, out1, net1a.output(inNCHW));
        assertEquals(tc.msg, out1, net2a.output(inNCHW));
        if(!tc.nhwcOutput) {
            assertEquals(tc.msg, out1, net3a.output(inNHWC));
            assertEquals(tc.msg, out1, net4a.output(inNHWC));
        } else {
            assertEquals(tc.msg, out1, net3a.output(inNHWC).permute(0,3,1,2));   //NHWC to NCHW
            assertEquals(tc.msg, out1, net4a.output(inNHWC).permute(0,3,1,2));
        }

    }

    private static List<String> differentGrads(Gradient g1, Gradient g2){
        List<String> differs = new ArrayList<>();
        Map<String,INDArray> m1 = g1.gradientForVariable();
        Map<String,INDArray> m2 = g2.gradientForVariable();
        for(String s : m1.keySet()){
            INDArray a1 = m1.get(s);
            INDArray a2 = m2.get(s);
            if(!a1.equals(a2)){
                differs.add(s);
            }
        }
        return differs;
    }

    //Converts NHWC to NCHW activations
    @EqualsAndHashCode
    private static class NHWCToNCHWPreprocessor implements InputPreProcessor {

        @Override
        public INDArray preProcess(INDArray input, int miniBatchSize, LayerWorkspaceMgr workspaceMgr) {
            return workspaceMgr.leverageTo(ArrayType.ACTIVATIONS, input.permute(0,3,1,2));
        }

        @Override
        public INDArray backprop(INDArray output, int miniBatchSize, LayerWorkspaceMgr workspaceMgr) {
            return workspaceMgr.leverageTo(ArrayType.ACTIVATION_GRAD, output.permute(0,2,3,1));
        }

        @Override
        public InputPreProcessor clone() {
            return this;
        }

        @Override
        public InputType getOutputType(InputType inputType) {
            InputType.InputTypeConvolutional c = (InputType.InputTypeConvolutional) inputType;
            return InputType.convolutional(c.getHeight(), c.getWidth(), c.getChannels(), CNN2DFormat.NCHW);
        }

        @Override
        public Pair<INDArray, MaskState> feedForwardMaskArray(INDArray maskArray, MaskState currentMaskState, int minibatchSize) {
            return null;
        }
    }
}
