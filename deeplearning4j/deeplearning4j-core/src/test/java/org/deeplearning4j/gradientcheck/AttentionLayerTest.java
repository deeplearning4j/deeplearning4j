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

package org.deeplearning4j.gradientcheck;

import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.TestUtils;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.AttentionVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.layers.recurrent.SimpleRnn;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.NoOp;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.Random;

import static org.junit.Assert.assertTrue;

public class AttentionLayerTest extends BaseDL4JTest {
    @Rule
    public ExpectedException exceptionRule = ExpectedException.none();

    private static final boolean PRINT_RESULTS = true;
    private static final boolean RETURN_ON_FIRST_FAILURE = false;
    private static final double DEFAULT_EPS = 1e-6;
    private static final double DEFAULT_MAX_REL_ERROR = 1e-3;
    private static final double DEFAULT_MIN_ABS_ERROR = 1e-8;

    @Test
    public void testSelfAttentionLayer() {
        int nIn = 3;
        int nOut = 2;
        int tsLength = 4;
        int layerSize = 4;

        for (int mb : new int[]{1, 3}) {
            for (boolean inputMask : new boolean[]{false, true}) {
                for (boolean projectInput : new boolean[]{false, true}) {
                    INDArray in = Nd4j.rand(DataType.DOUBLE, new int[]{mb, nIn, tsLength});
                    INDArray labels = TestUtils.randomOneHot(mb, nOut);
                    String maskType = (inputMask ? "inputMask" : "none");

                    INDArray inMask = null;
                    if (inputMask) {
                        inMask = Nd4j.ones(mb, tsLength);
                        for (int i = 0; i < mb; i++) {
                            int firstMaskedStep = tsLength - 1 - i;
                            if (firstMaskedStep == 0) {
                                firstMaskedStep = tsLength;
                            }
                            for (int j = firstMaskedStep; j < tsLength; j++) {
                                inMask.putScalar(i, j, 0.0);
                            }
                        }
                    }

                    String name = "testSelfAttentionLayer() - mb=" + mb + ", tsLength = " + tsLength + ", maskType=" + maskType + ", projectInput = " + projectInput;
                    System.out.println("Starting test: " + name);


                    MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                            .dataType(DataType.DOUBLE)
                            .activation(Activation.TANH)
                            .updater(new NoOp())
                            .weightInit(WeightInit.XAVIER)
                            .list()
                            .layer(new LSTM.Builder().nOut(layerSize).build())
                            .layer( projectInput ?
                                            new SelfAttentionLayer.Builder().nOut(4).nHeads(2).projectInput(true).build()
                                            : new SelfAttentionLayer.Builder().nHeads(1).projectInput(false).build()
                                    )
                            .layer(new GlobalPoolingLayer.Builder().poolingType(PoolingType.MAX).build())
                            .layer(new OutputLayer.Builder().nOut(nOut).activation(Activation.SOFTMAX)
                                    .lossFunction(LossFunctions.LossFunction.MCXENT).build())
                            .setInputType(InputType.recurrent(nIn))
                            .build();

                    MultiLayerNetwork net = new MultiLayerNetwork(conf);
                    net.init();

                    boolean gradOK = GradientCheckUtil.checkGradients(net, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
                            DEFAULT_MIN_ABS_ERROR, PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, in, labels, inMask, null, true, 100);
                    assertTrue(name, gradOK);
                }
            }
        }
    }

    @Test
    public void testLearnedSelfAttentionLayer() {
        int nIn = 3;
        int nOut = 2;
        int tsLength = 4;
        int layerSize = 4;
        int numQueries = 3;

        for (boolean inputMask : new boolean[]{false, true}) {
            for (int mb : new int[]{3, 1}) {
                for (boolean projectInput : new boolean[]{false, true}) {
                    INDArray in = Nd4j.rand(DataType.DOUBLE, new int[]{mb, nIn, tsLength});
                    INDArray labels = TestUtils.randomOneHot(mb, nOut);
                    String maskType = (inputMask ? "inputMask" : "none");

                    INDArray inMask = null;
                    if (inputMask) {
                        inMask = Nd4j.ones(mb, tsLength);
                        for (int i = 0; i < mb; i++) {
                            int firstMaskedStep = tsLength - 1 - i;
                            if (firstMaskedStep == 0) {
                                firstMaskedStep = tsLength;
                            }
                            for (int j = firstMaskedStep; j < tsLength; j++) {
                                inMask.putScalar(i, j, 0.0);
                            }
                        }
                    }

                    String name = "testLearnedSelfAttentionLayer() - mb=" + mb + ", tsLength = " + tsLength + ", maskType=" + maskType + ", projectInput = " + projectInput;
                    System.out.println("Starting test: " + name);


                    MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                            .dataType(DataType.DOUBLE)
                            .activation(Activation.TANH)
                            .updater(new NoOp())
                            .weightInit(WeightInit.XAVIER)
                            .list()
                            .layer(new LSTM.Builder().nOut(layerSize).build())
                            .layer( projectInput ?
                                    new LearnedSelfAttentionLayer.Builder().nOut(4).nHeads(2).nQueries(numQueries).projectInput(true).build()
                                    : new LearnedSelfAttentionLayer.Builder().nHeads(1).nQueries(numQueries).projectInput(false).build()
                            )
                            .layer(new GlobalPoolingLayer.Builder().poolingType(PoolingType.MAX).build())
                            .layer(new OutputLayer.Builder().nOut(nOut).activation(Activation.SOFTMAX)
                                    .lossFunction(LossFunctions.LossFunction.MCXENT).build())
                            .setInputType(InputType.recurrent(nIn))
                            .build();

                    MultiLayerNetwork net = new MultiLayerNetwork(conf);
                    net.init();

                    boolean gradOK = GradientCheckUtil.checkGradients(net, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
                            DEFAULT_MIN_ABS_ERROR, PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, in, labels, inMask, null, true, 100);
                    assertTrue(name, gradOK);
                }
            }
        }
    }

    @Test
    public void testLearnedSelfAttentionLayer_differentMiniBatchSizes() {
        int nIn = 3;
        int nOut = 2;
        int tsLength = 4;
        int layerSize = 4;
        int numQueries = 3;

        Random r = new Random(12345);
        for (boolean inputMask : new boolean[]{false, true}) {
            for (boolean projectInput : new boolean[]{false, true}) {

                MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                    .dataType(DataType.DOUBLE)
                    .activation(Activation.TANH)
                    .updater(new NoOp())
                    .weightInit(WeightInit.XAVIER)
                    .list()
                    .layer(new LSTM.Builder().nOut(layerSize).build())
                    .layer( projectInput ?
                            new LearnedSelfAttentionLayer.Builder().nOut(4).nHeads(2).nQueries(numQueries).projectInput(true).build()
                            : new LearnedSelfAttentionLayer.Builder().nHeads(1).nQueries(numQueries).projectInput(false).build()
                    )
                    .layer(new GlobalPoolingLayer.Builder().poolingType(PoolingType.MAX).build())
                    .layer(new OutputLayer.Builder().nOut(nOut).activation(Activation.SOFTMAX)
                            .lossFunction(LossFunctions.LossFunction.MCXENT).build())
                    .setInputType(InputType.recurrent(nIn))
                    .build();

            MultiLayerNetwork net = new MultiLayerNetwork(conf);
            net.init();
            for (int mb : new int[]{3, 1}) {
                    INDArray in = Nd4j.rand(DataType.DOUBLE, new int[]{mb, nIn, tsLength});
                    INDArray labels = TestUtils.randomOneHot(mb, nOut);
                    String maskType = (inputMask ? "inputMask" : "none");

                    INDArray inMask = null;
                    if (inputMask) {
                        inMask = Nd4j.ones(DataType.INT, mb, tsLength);
                        for (int i = 0; i < mb; i++) {
                            int firstMaskedStep = tsLength - 1 - i;
                            if (firstMaskedStep == 0) {
                                firstMaskedStep = tsLength;
                            }
                            for (int j = firstMaskedStep; j < tsLength; j++) {
                                inMask.putScalar(i, j, 0.0);
                            }
                        }
                    }

                    String name = "testLearnedSelfAttentionLayer() - mb=" + mb + ", tsLength = " + tsLength + ", maskType=" + maskType + ", projectInput = " + projectInput;
                    System.out.println("Starting test: " + name);

                    boolean gradOK = GradientCheckUtil.checkGradients(net, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
                            DEFAULT_MIN_ABS_ERROR, PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, in, labels, inMask, null, true, 100);
                    assertTrue(name, gradOK);
                }
            }
        }
    }

    @Test
    public void testRecurrentAttentionLayer_differingTimeSteps(){
        int nIn = 9;
        int nOut = 5;
        int layerSize = 8;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .dataType(DataType.DOUBLE)
                .activation(Activation.IDENTITY)
                .updater(new NoOp())
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(new LSTM.Builder().nOut(layerSize).build())
                .layer(new RecurrentAttentionLayer.Builder().nIn(layerSize).nOut(layerSize).nHeads(1).projectInput(false).hasBias(false).build())
                .layer(new GlobalPoolingLayer.Builder().poolingType(PoolingType.AVG).build())
                .layer(new OutputLayer.Builder().nOut(nOut).activation(Activation.SOFTMAX)
                        .lossFunction(LossFunctions.LossFunction.MCXENT).build())
                .setInputType(InputType.recurrent(nIn))
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        final INDArray initialInput = Nd4j.rand(new int[]{8, nIn, 7});
        final INDArray goodNextInput = Nd4j.rand(new int[]{8, nIn, 7});
        final INDArray badNextInput = Nd4j.rand(new int[]{8, nIn, 12});

        final INDArray labels = Nd4j.rand(new int[]{8, nOut});

        net.fit(initialInput, labels);
        net.fit(goodNextInput, labels);

        exceptionRule.expect(IllegalArgumentException.class);
        exceptionRule.expectMessage("This layer only supports fixed length mini-batches. Expected 7 time steps but got 12.");
        net.fit(badNextInput, labels);
    }

    @Test
    public void testRecurrentAttentionLayer() {
        int nIn = 4;
        int nOut = 2;
        int tsLength = 3;
        int layerSize = 3;

        for (int mb : new int[]{3, 1}) {
            for (boolean inputMask : new boolean[]{true, false}) {
                INDArray in = Nd4j.rand(DataType.DOUBLE, new int[]{mb, nIn, tsLength});
                INDArray labels = TestUtils.randomOneHot(mb, nOut);
                String maskType = (inputMask ? "inputMask" : "none");

                INDArray inMask = null;
                if (inputMask) {
                    inMask = Nd4j.ones(mb, tsLength);
                    for (int i = 0; i < mb; i++) {
                        int firstMaskedStep = tsLength - 1 - i;
                        if (firstMaskedStep == 0) {
                            firstMaskedStep = tsLength;
                        }
                        for (int j = firstMaskedStep; j < tsLength; j++) {
                            inMask.putScalar(i, j, 0.0);
                        }
                    }
                }

                String name = "testRecurrentAttentionLayer() - mb=" + mb + ", tsLength = " + tsLength + ", maskType=" + maskType;
                System.out.println("Starting test: " + name);


                MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                        .dataType(DataType.DOUBLE)
                        .activation(Activation.IDENTITY)
                        .updater(new NoOp())
                        .weightInit(WeightInit.XAVIER)
                        .list()
                        .layer(new LSTM.Builder().nOut(layerSize).build())
                        .layer(new RecurrentAttentionLayer.Builder().nIn(layerSize).nOut(layerSize).nHeads(1).projectInput(false).hasBias(false).build())
                        .layer(new GlobalPoolingLayer.Builder().poolingType(PoolingType.AVG).build())
                        .layer(new OutputLayer.Builder().nOut(nOut).activation(Activation.SOFTMAX)
                                .lossFunction(LossFunctions.LossFunction.MCXENT).build())
                        .setInputType(InputType.recurrent(nIn))
                        .build();

                MultiLayerNetwork net = new MultiLayerNetwork(conf);
                net.init();

                //System.out.println("Original");
                boolean gradOK = GradientCheckUtil.checkGradients(net, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
                        DEFAULT_MIN_ABS_ERROR, PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, in, labels, inMask, null, true, 100, null);
                assertTrue(name, gradOK);
            }
        }
    }

    @Test
    public void testAttentionVertex() {
        int nIn = 3;
        int nOut = 2;
        int tsLength = 3;
        int layerSize = 3;

        Random r = new Random(12345);
        for (boolean inputMask : new boolean[]{false, true}) {
            for (int mb : new int[]{3, 1}) {
                for (boolean projectInput : new boolean[]{false, true}) {
                    INDArray in = Nd4j.rand(DataType.DOUBLE, new int[]{mb, nIn, tsLength});
                    INDArray labels = TestUtils.randomOneHot(mb, nOut);
                    String maskType = (inputMask ? "inputMask" : "none");

                    INDArray inMask = null;
                    if (inputMask) {
                        inMask = Nd4j.ones(mb, tsLength);
                        for (int i = 0; i < mb; i++) {
                            int firstMaskedStep = tsLength - 1 - i;
                            if (firstMaskedStep == 0) {
                                firstMaskedStep = tsLength;
                            }
                            for (int j = firstMaskedStep; j < tsLength; j++) {
                                inMask.putScalar(i, j, 0.0);
                            }
                        }
                    }

                    String name = "testAttentionVertex() - mb=" + mb + ", tsLength = " + tsLength + ", maskType=" + maskType + ", projectInput = " + projectInput;
                    System.out.println("Starting test: " + name);


                    ComputationGraphConfiguration graph = new NeuralNetConfiguration.Builder()
                            .dataType(DataType.DOUBLE)
                            .activation(Activation.TANH)
                            .updater(new NoOp())
                            .weightInit(WeightInit.XAVIER)
                            .graphBuilder()
                            .addInputs("input")
                            .addLayer("rnnKeys", new SimpleRnn.Builder().nOut(layerSize).build(), "input")
                            .addLayer("rnnQueries", new SimpleRnn.Builder().nOut(layerSize).build(), "input")
                            .addLayer("rnnValues", new SimpleRnn.Builder().nOut(layerSize).build(), "input")
                            .addVertex("attention",
                                    projectInput ?
                                    new AttentionVertex.Builder().nOut(4).nHeads(2).projectInput(true).nInQueries(layerSize).nInKeys(layerSize).nInValues(layerSize).build()
                                            :  new AttentionVertex.Builder().nOut(3).nHeads(1).projectInput(false).nInQueries(layerSize).nInKeys(layerSize).nInValues(layerSize).build(), "rnnQueries", "rnnKeys", "rnnValues")
                            .addLayer("pooling", new GlobalPoolingLayer.Builder().poolingType(PoolingType.MAX).build(), "attention")
                            .addLayer("output", new OutputLayer.Builder().nOut(nOut).activation(Activation.SOFTMAX).lossFunction(LossFunctions.LossFunction.MCXENT).build(), "pooling")
                            .setOutputs("output")
                            .setInputTypes(InputType.recurrent(nIn))
                            .build();

                    ComputationGraph net = new ComputationGraph(graph);
                    net.init();

                    boolean gradOK = GradientCheckUtil.checkGradients(net, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
                            DEFAULT_MIN_ABS_ERROR, PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, new INDArray[]{in}, new INDArray[]{labels}, inMask != null ? new INDArray[]{inMask} : null, null);
                    assertTrue(name, gradOK);
                }
            }
        }
    }

    @Test
    public void testAttentionVertexSameInput() {
        int nIn = 3;
        int nOut = 2;
        int tsLength = 4;
        int layerSize = 4;

        Random r = new Random(12345);
        for (boolean inputMask : new boolean[]{false, true}) {
            for (int mb : new int[]{3, 1}) {
                for (boolean projectInput : new boolean[]{false, true}) {
                    INDArray in = Nd4j.rand(new int[]{mb, nIn, tsLength});
                    INDArray labels = TestUtils.randomOneHot(mb, nOut);
                    String maskType = (inputMask ? "inputMask" : "none");

                    INDArray inMask = null;
                    if (inputMask) {
                        inMask = Nd4j.ones(mb, tsLength);
                        for (int i = 0; i < mb; i++) {
                            int firstMaskedStep = tsLength - 1 - i;
                            if (firstMaskedStep == 0) {
                                firstMaskedStep = tsLength;
                            }
                            for (int j = firstMaskedStep; j < tsLength; j++) {
                                inMask.putScalar(i, j, 0.0);
                            }
                        }
                    }

                    String name = "testAttentionVertex() - mb=" + mb + ", tsLength = " + tsLength + ", maskType=" + maskType + ", projectInput = " + projectInput;
                    System.out.println("Starting test: " + name);


                    ComputationGraphConfiguration graph = new NeuralNetConfiguration.Builder()
                            .dataType(DataType.DOUBLE)
                            .activation(Activation.TANH)
                            .updater(new NoOp())
                            .weightInit(WeightInit.XAVIER)
                            .graphBuilder()
                            .addInputs("input")
                            .addLayer("rnn", new SimpleRnn.Builder().activation(Activation.TANH).nOut(layerSize).build(), "input")
                            .addVertex("attention",
                                    projectInput ?
                                            new AttentionVertex.Builder().nOut(4).nHeads(2).projectInput(true).nInQueries(layerSize).nInKeys(layerSize).nInValues(layerSize).build()
                                            :  new AttentionVertex.Builder().nOut(4).nHeads(1).projectInput(false).nInQueries(layerSize).nInKeys(layerSize).nInValues(layerSize).build(), "rnn", "rnn", "rnn")
                            .addLayer("pooling", new GlobalPoolingLayer.Builder().poolingType(PoolingType.MAX).build(), "attention")
                            .addLayer("output", new OutputLayer.Builder().nOut(nOut).activation(Activation.SOFTMAX).lossFunction(LossFunctions.LossFunction.MCXENT).build(), "pooling")
                            .setOutputs("output")
                            .setInputTypes(InputType.recurrent(nIn))
                            .build();

                    ComputationGraph net = new ComputationGraph(graph);
                    net.init();

                    boolean gradOK = GradientCheckUtil.checkGradients(net, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
                            DEFAULT_MIN_ABS_ERROR, PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, new INDArray[]{in},
                            new INDArray[]{labels}, inMask != null ? new INDArray[]{inMask} : null, null);
                    assertTrue(name, gradOK);
                }
            }
        }
    }
}
