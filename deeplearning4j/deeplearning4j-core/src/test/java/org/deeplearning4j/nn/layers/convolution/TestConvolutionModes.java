/*******************************************************************************
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

package org.deeplearning4j.nn.layers.convolution;

import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.exception.DL4JException;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.LayerVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ConvolutionUtils;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.*;

/**
 * Created by Alex on 15/11/2016.
 */
public class TestConvolutionModes extends BaseDL4JTest {

    @Test
    public void testStrictTruncateConvolutionModeOutput() {

        //Idea: with convolution mode == Truncate, input size shouldn't matter (within the bounds of truncated edge),
        // and edge data shouldn't affect the output

        //Use: 9x9, kernel 3, stride 3, padding 0
        // Should get same output for 10x10 and 11x11...

        Nd4j.getRandom().setSeed(12345);
        int[] minibatches = {1, 3};
        int[] inDepths = {1, 3};
        int[] inSizes = {9, 10, 11};

        for (boolean isSubsampling : new boolean[] {false, true}) {
            for (int minibatch : minibatches) {
                for (int inDepth : inDepths) {

                    INDArray origData = Nd4j.rand(new int[] {minibatch, inDepth, 9, 9});

                    for (int inSize : inSizes) {

                        for (ConvolutionMode cm : new ConvolutionMode[] {ConvolutionMode.Strict,
                                        ConvolutionMode.Truncate}) {

                            INDArray inputData = Nd4j.rand(new int[] {minibatch, inDepth, inSize, inSize});
                            inputData.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.interval(0, 9),
                                            NDArrayIndex.interval(0, 9)).assign(origData);

                            Layer layer;
                            if (isSubsampling) {
                                layer = new SubsamplingLayer.Builder().kernelSize(3, 3).stride(3, 3).padding(0, 0)
                                                .build();
                            } else {
                                layer = new ConvolutionLayer.Builder().kernelSize(3, 3).stride(3, 3).padding(0, 0)
                                                .nOut(3).build();
                            }

                            MultiLayerNetwork net = null;
                            try {
                                MultiLayerConfiguration conf =
                                                new NeuralNetConfiguration.Builder().weightInit(WeightInit.XAVIER)
                                                                .convolutionMode(cm).list()
                                                                .layer(0, layer).layer(1,
                                                                                new OutputLayer.Builder()
                                                                                        .activation(Activation.SOFTMAX)
                                                                                        .lossFunction(LossFunctions.LossFunction.MCXENT)
                                                                                        .nOut(3).build())
                                                                .setInputType(InputType.convolutional(inSize, inSize,
                                                                                inDepth))
                                                                .build();

                                net = new MultiLayerNetwork(conf);
                                net.init();
                                if (inSize > 9 && cm == ConvolutionMode.Strict) {
                                    fail("Expected exception");
                                }
                            } catch (DL4JException e) {
                                if (inSize == 9 || cm != ConvolutionMode.Strict) {
                                    e.printStackTrace();
                                    fail("Unexpected exception");
                                }
                                continue; //Expected exception
                            } catch (Exception e) {
                                e.printStackTrace();
                                fail("Unexpected exception");
                            }

                            INDArray out = net.output(origData);
                            INDArray out2 = net.output(inputData);

                            assertEquals(out, out2);
                        }
                    }
                }
            }
        }
    }

    @Test
    public void testStrictTruncateConvolutionModeCompGraph() {

        //Idea: with convolution mode == Truncate, input size shouldn't matter (within the bounds of truncated edge),
        // and edge data shouldn't affect the output

        //Use: 9x9, kernel 3, stride 3, padding 0
        // Should get same output for 10x10 and 11x11...

        Nd4j.getRandom().setSeed(12345);
        int[] minibatches = {1, 3};
        int[] inDepths = {1, 3};
        int[] inSizes = {9, 10, 11};

        for (boolean isSubsampling : new boolean[] {false, true}) {
            for (int minibatch : minibatches) {
                for (int inDepth : inDepths) {

                    INDArray origData = Nd4j.rand(new int[] {minibatch, inDepth, 9, 9});

                    for (int inSize : inSizes) {

                        for (ConvolutionMode cm : new ConvolutionMode[] {ConvolutionMode.Strict,
                                        ConvolutionMode.Truncate}) {

                            INDArray inputData = Nd4j.rand(new int[] {minibatch, inDepth, inSize, inSize});
                            inputData.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.interval(0, 9),
                                            NDArrayIndex.interval(0, 9)).assign(origData);

                            Layer layer;
                            if (isSubsampling) {
                                layer = new SubsamplingLayer.Builder().kernelSize(3, 3).stride(3, 3).padding(0, 0)
                                                .build();
                            } else {
                                layer = new ConvolutionLayer.Builder().kernelSize(3, 3).stride(3, 3).padding(0, 0)
                                                .nOut(3).build();
                            }

                            ComputationGraph net = null;
                            try {
                                ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
                                                .weightInit(WeightInit.XAVIER).convolutionMode(cm).graphBuilder()
                                                .addInputs("in").addLayer("0", layer, "in")
                                                .addLayer("1", new OutputLayer.Builder()
                                                                .activation(Activation.SOFTMAX)
                                                                .lossFunction(LossFunctions.LossFunction.MCXENT).nOut(3)
                                                                .build(), "0")
                                                .setOutputs("1")
                                                .setInputTypes(InputType.convolutional(inSize, inSize, inDepth))
                                                .build();

                                net = new ComputationGraph(conf);
                                net.init();
                                if (inSize > 9 && cm == ConvolutionMode.Strict) {
                                    fail("Expected exception");
                                }
                            } catch (DL4JException e) {
                                if (inSize == 9 || cm != ConvolutionMode.Strict) {
                                    e.printStackTrace();
                                    fail("Unexpected exception");
                                }
                                continue; //Expected exception
                            } catch (Exception e) {
                                e.printStackTrace();
                                fail("Unexpected exception");
                            }

                            INDArray out = net.outputSingle(origData);
                            INDArray out2 = net.outputSingle(inputData);

                            assertEquals(out, out2);
                        }
                    }
                }
            }
        }
    }

    @Test
    public void testGlobalLocalConfig() {
        for (ConvolutionMode cm : new ConvolutionMode[] {ConvolutionMode.Strict, ConvolutionMode.Truncate}) {
            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().weightInit(WeightInit.XAVIER)
                            .convolutionMode(cm).list()
                            .layer(0, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(3, 3).padding(0, 0)
                                            .nIn(3).nOut(
                                                            3)
                                            .build())
                            .layer(1, new ConvolutionLayer.Builder().convolutionMode(ConvolutionMode.Strict)
                                            .kernelSize(3, 3).stride(3, 3).padding(0, 0)
                                            .nIn(3).nOut(
                                                            3)
                                            .build())
                            .layer(2, new ConvolutionLayer.Builder().convolutionMode(ConvolutionMode.Truncate)
                                            .kernelSize(3, 3).stride(3, 3).padding(0, 0)
                                            .nIn(3).nOut(
                                                            3)
                                            .build())
                            .layer(3, new ConvolutionLayer.Builder().convolutionMode(ConvolutionMode.Same)
                                            .kernelSize(3, 3).stride(3, 3).padding(0, 0).nIn(3).nOut(3).build())
                            .layer(4, new SubsamplingLayer.Builder().kernelSize(3, 3).stride(3, 3).padding(0, 0)
                                            .build())
                            .layer(5, new SubsamplingLayer.Builder().convolutionMode(ConvolutionMode.Strict)
                                            .kernelSize(3, 3).stride(3, 3).padding(0, 0).build())
                            .layer(6, new SubsamplingLayer.Builder().convolutionMode(ConvolutionMode.Truncate)
                                            .kernelSize(3, 3).stride(3, 3).padding(0, 0).build())
                            .layer(7, new SubsamplingLayer.Builder().convolutionMode(ConvolutionMode.Same)
                                            .kernelSize(3, 3).stride(3, 3).padding(0, 0).build())
                            .layer(8, new OutputLayer.Builder().lossFunction(LossFunctions.LossFunction.MCXENT).nOut(3)
                                    .activation(Activation.SOFTMAX).build())
                            .build();

            assertEquals(cm, ((ConvolutionLayer) conf.getConf(0).getLayer()).getConvolutionMode());
            assertEquals(ConvolutionMode.Strict, ((ConvolutionLayer) conf.getConf(1).getLayer()).getConvolutionMode());
            assertEquals(ConvolutionMode.Truncate,
                            ((ConvolutionLayer) conf.getConf(2).getLayer()).getConvolutionMode());
            assertEquals(ConvolutionMode.Same, ((ConvolutionLayer) conf.getConf(3).getLayer()).getConvolutionMode());

            assertEquals(cm, ((SubsamplingLayer) conf.getConf(4).getLayer()).getConvolutionMode());
            assertEquals(ConvolutionMode.Strict, ((SubsamplingLayer) conf.getConf(5).getLayer()).getConvolutionMode());
            assertEquals(ConvolutionMode.Truncate,
                            ((SubsamplingLayer) conf.getConf(6).getLayer()).getConvolutionMode());
            assertEquals(ConvolutionMode.Same, ((SubsamplingLayer) conf.getConf(7).getLayer()).getConvolutionMode());
        }
    }

    @Test
    public void testGlobalLocalConfigCompGraph() {
        for (ConvolutionMode cm : new ConvolutionMode[] {ConvolutionMode.Strict, ConvolutionMode.Truncate,
                        ConvolutionMode.Same}) {
            ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder().weightInit(WeightInit.XAVIER)
                            .convolutionMode(cm).graphBuilder().addInputs("in")
                            .addLayer("0", new ConvolutionLayer.Builder().kernelSize(3, 3).stride(3, 3).padding(0, 0)
                                            .nIn(3).nOut(
                                                            3)
                                            .build(), "in")
                            .addLayer("1", new ConvolutionLayer.Builder().convolutionMode(ConvolutionMode.Strict)
                                            .kernelSize(3, 3).stride(3, 3).padding(0, 0)
                                            .nIn(3).nOut(
                                                            3)
                                            .build(), "0")
                            .addLayer("2", new ConvolutionLayer.Builder().convolutionMode(ConvolutionMode.Truncate)
                                            .kernelSize(3, 3).stride(3, 3).padding(0, 0)
                                            .nIn(3).nOut(
                                                            3)
                                            .build(), "1")
                            .addLayer("3", new ConvolutionLayer.Builder().convolutionMode(ConvolutionMode.Same)
                                            .kernelSize(3, 3).stride(3, 3).padding(0, 0).nIn(3).nOut(3).build(), "2")
                            .addLayer("4", new SubsamplingLayer.Builder().kernelSize(3, 3).stride(3, 3).padding(0, 0)
                                            .build(), "3")
                            .addLayer("5", new SubsamplingLayer.Builder().convolutionMode(ConvolutionMode.Strict)
                                            .kernelSize(3, 3).stride(3, 3).padding(0, 0).build(), "4")
                            .addLayer("6", new SubsamplingLayer.Builder().convolutionMode(ConvolutionMode.Truncate)
                                            .kernelSize(3, 3).stride(3, 3).padding(0, 0).build(), "5")
                            .addLayer("7", new SubsamplingLayer.Builder().convolutionMode(ConvolutionMode.Same)
                                            .kernelSize(3, 3).stride(3, 3).padding(0, 0).build(), "6")
                            .addLayer("8", new OutputLayer.Builder().lossFunction(LossFunctions.LossFunction.MCXENT)
                                    .activation(Activation.SOFTMAX).nOut(3).build(), "7")
                            .setOutputs("8").build();

            assertEquals(cm, ((ConvolutionLayer) ((LayerVertex) conf.getVertices().get("0")).getLayerConf().getLayer())
                            .getConvolutionMode());
            assertEquals(ConvolutionMode.Strict,
                            ((ConvolutionLayer) ((LayerVertex) conf.getVertices().get("1")).getLayerConf().getLayer())
                                            .getConvolutionMode());
            assertEquals(ConvolutionMode.Truncate,
                            ((ConvolutionLayer) ((LayerVertex) conf.getVertices().get("2")).getLayerConf().getLayer())
                                            .getConvolutionMode());
            assertEquals(ConvolutionMode.Same,
                            ((ConvolutionLayer) ((LayerVertex) conf.getVertices().get("3")).getLayerConf().getLayer())
                                            .getConvolutionMode());

            assertEquals(cm, ((SubsamplingLayer) ((LayerVertex) conf.getVertices().get("4")).getLayerConf().getLayer())
                            .getConvolutionMode());
            assertEquals(ConvolutionMode.Strict,
                            ((SubsamplingLayer) ((LayerVertex) conf.getVertices().get("5")).getLayerConf().getLayer())
                                            .getConvolutionMode());
            assertEquals(ConvolutionMode.Truncate,
                            ((SubsamplingLayer) ((LayerVertex) conf.getVertices().get("6")).getLayerConf().getLayer())
                                            .getConvolutionMode());
            assertEquals(ConvolutionMode.Same,
                            ((SubsamplingLayer) ((LayerVertex) conf.getVertices().get("7")).getLayerConf().getLayer())
                                            .getConvolutionMode());
        }
    }


    @Test
    public void testConvolutionModeInputTypes() {
        //Test 1: input 3x3, stride 1, kernel 2

        int inH = 3;
        int inW = 3;
        int kH = 2;
        int kW = 2;
        int sH = 1;
        int sW = 1;
        int pH = 0;
        int pW = 0;

        int minibatch = 3;
        int dIn = 5;
        int dOut = 7;
        int[] kernel = {kH, kW};
        int[] stride = {sH, sW};
        int[] padding = {pH, pW};
        int[] dilation = {1,1};

        INDArray inData = Nd4j.create(minibatch, dIn, inH, inW);
        InputType inputType = InputType.convolutional(inH, inW, dIn);

        //Strict mode: expect 2x2 out -> (inH - kernel + 2*padding)/stride + 1 = (3-2+0)/1+1 = 2
        InputType.InputTypeConvolutional it =
                        (InputType.InputTypeConvolutional) InputTypeUtil.getOutputTypeCnnLayers(inputType, kernel,
                                        stride, padding, dilation, ConvolutionMode.Strict, dOut, -1, "layerName",
                                        ConvolutionLayer.class);
        assertEquals(2, it.getHeight());
        assertEquals(2, it.getWidth());
        assertEquals(dOut, it.getChannels());
        int[] outSize = ConvolutionUtils.getOutputSize(inData, kernel, stride, padding, ConvolutionMode.Strict);
        assertEquals(2, outSize[0]);
        assertEquals(2, outSize[1]);


        //Truncate: same as strict here
        it = (InputType.InputTypeConvolutional) InputTypeUtil.getOutputTypeCnnLayers(inputType, kernel, stride, padding,
                dilation, ConvolutionMode.Truncate, dOut, -1, "layerName", ConvolutionLayer.class);
        assertEquals(2, it.getHeight());
        assertEquals(2, it.getWidth());
        assertEquals(dOut, it.getChannels());
        outSize = ConvolutionUtils.getOutputSize(inData, kernel, stride, padding, ConvolutionMode.Truncate);
        assertEquals(2, outSize[0]);
        assertEquals(2, outSize[1]);

        //Same mode: ceil(in / stride) = 3
        it = (InputType.InputTypeConvolutional) InputTypeUtil.getOutputTypeCnnLayers(inputType, kernel, stride, null,
                dilation, ConvolutionMode.Same, dOut, -1, "layerName", ConvolutionLayer.class);
        assertEquals(3, it.getHeight());
        assertEquals(3, it.getWidth());
        assertEquals(dOut, it.getChannels());
        outSize = ConvolutionUtils.getOutputSize(inData, kernel, stride, null, ConvolutionMode.Same);
        assertEquals(3, outSize[0]);
        assertEquals(3, outSize[1]);



        //Test 2: input 3x4, stride 2, kernel 3
        inH = 3;
        inW = 4;
        kH = 3;
        kW = 3;
        sH = 2;
        sW = 2;

        kernel = new int[] {kH, kW};
        stride = new int[] {sH, sW};
        padding = new int[] {pH, pW};

        inData = Nd4j.create(minibatch, dIn, inH, inW);
        inputType = InputType.convolutional(inH, inW, dIn);

        //Strict mode: (4-3+0)/2+1 is not an integer -> exception
        try {
            InputTypeUtil.getOutputTypeCnnLayers(inputType, kernel, stride, padding, dilation, ConvolutionMode.Strict, dOut, -1,
                            "layerName", ConvolutionLayer.class);
            fail("Expected exception");
        } catch (DL4JException e) {
            System.out.println(e.getMessage());
        }
        try {
            outSize = ConvolutionUtils.getOutputSize(inData, kernel, stride, padding, ConvolutionMode.Strict);
            fail("Exception expected");
        } catch (DL4JException e) {
            System.out.println(e.getMessage());
        }

        //Truncate: (3-3+0)/2+1 = 1 in height dim; (4-3+0)/2+1 = 1 in width dim
        it = (InputType.InputTypeConvolutional) InputTypeUtil.getOutputTypeCnnLayers(inputType, kernel, stride, padding,
                dilation, ConvolutionMode.Truncate, dOut, -1, "layerName", ConvolutionLayer.class);
        assertEquals(1, it.getHeight());
        assertEquals(1, it.getWidth());
        assertEquals(dOut, it.getChannels());
        outSize = ConvolutionUtils.getOutputSize(inData, kernel, stride, padding, ConvolutionMode.Truncate);
        assertEquals(1, outSize[0]);
        assertEquals(1, outSize[1]);

        //Same mode: ceil(3/2) = 2 in height dim; ceil(4/2) = 2 in width dimension
        it = (InputType.InputTypeConvolutional) InputTypeUtil.getOutputTypeCnnLayers(inputType, kernel, stride, null,
                dilation, ConvolutionMode.Same, dOut, -1, "layerName", ConvolutionLayer.class);
        assertEquals(2, it.getHeight());
        assertEquals(2, it.getWidth());
        assertEquals(dOut, it.getChannels());
        outSize = ConvolutionUtils.getOutputSize(inData, kernel, stride, null, ConvolutionMode.Same);
        assertEquals(2, outSize[0]);
        assertEquals(2, outSize[1]);
    }

    @Test
    public void testSameModeActivationSizes() {
        int inH = 3;
        int inW = 4;
        int inDepth = 3;
        int minibatch = 5;

        int sH = 2;
        int sW = 2;
        int kH = 3;
        int kW = 3;

        Layer[] l = new Layer[2];
        l[0] = new ConvolutionLayer.Builder().nOut(4).kernelSize(kH, kW).stride(sH, sW).build();
        l[1] = new SubsamplingLayer.Builder().kernelSize(kH, kW).stride(sH, sW).build();

        for (int i = 0; i < l.length; i++) {

            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().convolutionMode(ConvolutionMode.Same)
                            .list().layer(0, l[i]).layer(1, new OutputLayer.Builder().nOut(3).activation(Activation.SOFTMAX).build())
                            .setInputType(InputType.convolutional(inH, inW, inDepth)).build();

            MultiLayerNetwork net = new MultiLayerNetwork(conf);
            net.init();

            INDArray inData = Nd4j.create(minibatch, inDepth, inH, inW);
            List<INDArray> activations = net.feedForward(inData);
            INDArray actL0 = activations.get(1);

            int outH = (int) Math.ceil(inH / ((double) sH));
            int outW = (int) Math.ceil(inW / ((double) sW));

            System.out.println(Arrays.toString(actL0.shape()));
            assertArrayEquals(new long[] {minibatch, (i == 0 ? 4 : inDepth), outH, outW}, actL0.shape());
        }
    }
}
