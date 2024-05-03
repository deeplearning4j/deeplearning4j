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
package org.eclipse.deeplearning4j.dl4jcore.nn.layers.convolution;

import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.gradientcheck.GradientCheckUtil;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.eclipse.deeplearning4j.dl4jcore.TestUtils;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.OpContext;
import org.nd4j.linalg.api.ops.executioner.DefaultOpExecutioner;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.learning.config.NoOp;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Map;

import org.junit.jupiter.api.DisplayName;

import static org.junit.jupiter.api.Assertions.*;

/**
 * @author Max Pumperla
 */
@DisplayName("Locally Connected Layer Test")
@NativeTag
@Tag(TagNames.DL4J_OLD_API)
class LocallyConnectedLayerTest extends BaseDL4JTest {

    @BeforeEach
    void before() {
        DataTypeUtil.setDTypeForContext(DataType.DOUBLE);
        Nd4j.factory().setDType(DataType.DOUBLE);
        Nd4j.EPS_THRESHOLD = 1e-4;
    }

    @Test
    @DisplayName("Test 2 d Forward")
    void test2dForward() {
        ListBuilder builder = new NeuralNetConfiguration.Builder().seed(123).optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).l2(2e-4).updater(new Nesterovs(0.9)).dropOut(0.5).list().layer(new LocallyConnected2D.Builder().kernelSize(8, 8).nIn(3).stride(4, 4).nOut(16).dropOut(0.5).convolutionMode(ConvolutionMode.Strict).setInputSize(28, 28).activation(Activation.RELU).weightInit(WeightInit.XAVIER).build()).layer(// output layer
                new OutputLayer.Builder(LossFunctions.LossFunction.SQUARED_LOSS).nOut(10).weightInit(WeightInit.XAVIER).activation(Activation.SOFTMAX).build()).setInputType(InputType.convolutionalFlat(28, 28, 3));
        MultiLayerConfiguration conf = builder.build();
        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.init();
        INDArray input = Nd4j.ones(10, 3, 28, 28);
        INDArray output = network.output(input, false);
        assertArrayEquals(new long[] { 10, 10 }, output.shape());
    }

    @Test
    @DisplayName("Test 1 d Forward")
    void test1dForward() {
        ListBuilder builder = new NeuralNetConfiguration.Builder().seed(123).optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).l2(2e-4).updater(new Nesterovs(0.9)).dropOut(0.5).list().layer(new LocallyConnected1D.Builder().kernelSize(4).nIn(3).stride(1).nOut(16).dropOut(0.5).convolutionMode(ConvolutionMode.Strict).setInputSize(28).activation(Activation.RELU).weightInit(WeightInit.XAVIER).build()).layer(// output layer
                new OutputLayer.Builder(LossFunctions.LossFunction.SQUARED_LOSS).nOut(10).weightInit(WeightInit.XAVIER).activation(Activation.SOFTMAX).build()).setInputType(InputType.recurrent(3, 8));
        MultiLayerConfiguration conf = builder.build();
        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.init();
        INDArray input = Nd4j.ones(10, 3, 8);
        INDArray output = network.output(input, false);
        ;
        for (int i = 0; i < 100; i++) {
            // TODO: this falls flat for 1000 iterations on my machine
            output = network.output(input, false);
        }
        assertArrayEquals(new long[] { (8 - 4 + 1) * 10, 10 }, output.shape());
        network.fit(input, output);
    }

    @Test
    public void dummyTestRecreation() {
        INDArray arr = Nd4j.create(2);
        OpExecutioner executioner = Nd4j.getExecutioner();
        OpContext opContext = executioner.buildContext();
        opContext.addIntermediateResult(arr);
        assertEquals(1, opContext.numIntermediateResults());
        INDArray arr2 = opContext.getIntermediateResult(0);
        assertEquals(arr, arr2);
    }

    @Test
    @DisplayName("Test Locally Connected")
    void testLocallyConnected() {
        Nd4j.getRandom().setSeed(12345);
        for (DataType globalDtype : new DataType[] { DataType.DOUBLE }) {
            Nd4j.setDefaultDataTypes(globalDtype, globalDtype);
            for (DataType networkDtype : new DataType[] { DataType.DOUBLE, DataType.FLOAT, DataType.HALF }) {
                assertEquals(globalDtype, Nd4j.dataType());
                assertEquals(globalDtype, Nd4j.defaultFloatingPointType());
                for (CNN2DFormat format : new CNN2DFormat[] {CNN2DFormat.NHWC}) {
                    for (int test = 1; test < 2; test++) {
                        String msg = "Global dtype: " + globalDtype + ", network dtype: " + networkDtype + ", format: " + format + ", test=" + test;
                        ComputationGraph net = null;
                        INDArray[] in;
                        INDArray label;

                        ComputationGraphConfiguration.GraphBuilder b = new NeuralNetConfiguration.Builder()
                                .dataType(networkDtype)
                                .updater(new NoOp())
                                .seed(12345)
                                .weightInit(WeightInit.XAVIER)
                                .convolutionMode(ConvolutionMode.Same)
                                .graphBuilder();

                        switch (test) {
                            case 0: {
                                System.out.println("Test case 0:");
                                INDArray lstmWeights = Nd4j.linspace(1, 4 * 5 * 5, 4 * 5 * 5).reshape(4, 5, 5).castTo(networkDtype);
                                INDArray lstmBias = Nd4j.linspace(1, 4 * 5, 4 * 5).reshape(4, 5).castTo(networkDtype);
                                b.addInputs("in")
                                        .addLayer("1", new LSTM.Builder().nOut(5).weightInit(WeightInit.ZERO).biasInit(0).build(), "in")
                                        .addLayer("2", new LocallyConnected1D.Builder().kernelSize(2).nOut(4).build(), "1")
                                        .addLayer("out", new RnnOutputLayer.Builder().nOut(10).build(), "2")
                                        .setOutputs("out")
                                        .setInputTypes(InputType.recurrent(5, 4));
                                net = new ComputationGraph(b.build());
                                net.init();
                                net.getLayer("1").setParam("W", lstmWeights);
                                net.getLayer("1").setParam("b", lstmBias);
                                in = new INDArray[]{Nd4j.linspace(0, 1, 2 * 5 * 4).reshape(2, 5, 4).castTo(networkDtype)};
                                label = Nd4j.linspace(0, 9, 2 * 4 * 10, DataType.INT32).reshape(2, 4, 10).castTo(networkDtype);
                            }
                            break;
                            case 1: {
                                System.out.println("Test case 1: PID: " + ProcessHandle.current().pid());
                                INDArray convWeights;
                                long[] inputShape;
                                switch (format) {
                                    case NCHW:
                                        convWeights = Nd4j.linspace(1, 5 * 1 * 2 * 2, 5 * 1 * 2 * 2).reshape(2, 2, 1, 5).castTo(networkDtype);
                                        inputShape = new long[]{2, 1, 8, 8};
                                        break;
                                    case NHWC:
                                        convWeights = Nd4j.linspace(1, 1 * 5 * 2 * 2, 1 * 5 * 2 * 2).reshape(5, 1, 2, 2).castTo(networkDtype);
                                        inputShape = new long[]{2, 8, 8, 1};
                                        break;
                                    default:
                                        throw new IllegalStateException("Unknown format: " + format);
                                }
                                b.addInputs("in")
                                        .addLayer("1", new ConvolutionLayer.Builder()
                                                .kernelSize(2, 2).nOut(5)
                                                .convolutionMode(ConvolutionMode.Same)
                                                .weightInit(WeightInit.ZERO)
                                                .dataFormat(format)
                                                .build(), "in")
                                        .addLayer("2", new LocallyConnected2D.Builder()
                                                .hasBias(false)
                                                .kernelSize(2, 2).nOut(5).build(), "1")
                                        .addLayer("out", new OutputLayer.Builder().nOut(10).build(), "2")
                                        .setOutputs("out")
                                        .setInputTypes(InputType.convolutional(8, 8, 1, format));
                                net = new ComputationGraph(b.build());
                                net.init();
                                net.getLayer("1").setParam("W", convWeights);
                                in = new INDArray[]{Nd4j.linspace(0, 1, 2 * 1 * 8 * 8).reshape(inputShape).castTo(networkDtype)};
                                label = Nd4j.linspace(0, 9, 2 * 10, DataType.INT32).reshape(2, 10).castTo(networkDtype);
                            }
                                break;
                            default : {
                                throw new RuntimeException();
                            }
                        }

                        INDArray out = net.outputSingle(in);
                        assertEquals(networkDtype, out.dataType(), msg);
                        net.setInputs(in);
                        net.setLabels(label);

                        boolean gradOK = GradientCheckUtil.checkGradients(new GradientCheckUtil.GraphConfig()
                                .excludeParams(new HashSet<>(Arrays.asList("1_W", "1_b")))
                                .net(net).inputs(in).labels(new INDArray[]{label}));
                        assertTrue(gradOK);
                        TestUtils.testModelSerialization(net);
                    }
                }
            }
        }
    }
}
