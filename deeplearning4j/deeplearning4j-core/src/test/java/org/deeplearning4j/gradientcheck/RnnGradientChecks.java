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

package org.deeplearning4j.gradientcheck;

import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.TestUtils;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.conf.layers.recurrent.Bidirectional;
import org.deeplearning4j.nn.conf.layers.recurrent.LastTimeStep;
import org.deeplearning4j.nn.conf.layers.recurrent.SimpleRnn;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.NoOp;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.Random;

import static org.junit.Assert.assertTrue;

public class RnnGradientChecks extends BaseDL4JTest {

    private static final boolean PRINT_RESULTS = true;
    private static final boolean RETURN_ON_FIRST_FAILURE = false;
    private static final double DEFAULT_EPS = 1e-6;
    private static final double DEFAULT_MAX_REL_ERROR = 1e-3;
    private static final double DEFAULT_MIN_ABS_ERROR = 1e-8;

    static {
        Nd4j.setDataType(DataBuffer.Type.DOUBLE);
    }

    @Test
    public void testBidirectionalWrapper() {

        int nIn = 3;
        int nOut = 5;
        int tsLength = 4;

        Bidirectional.Mode[] modes = new Bidirectional.Mode[]{Bidirectional.Mode.CONCAT, Bidirectional.Mode.ADD,
                Bidirectional.Mode.AVERAGE, Bidirectional.Mode.MUL};

        Random r = new Random(12345);
        for (int mb : new int[]{1, 3}) {
            for (boolean inputMask : new boolean[]{false, true}) {
                for (boolean simple : new boolean[]{false, true}) {

                    INDArray in = Nd4j.rand(new int[]{mb, nIn, tsLength});
                    INDArray labels = Nd4j.create(mb, nOut, tsLength);
                    for (int i = 0; i < mb; i++) {
                        for (int j = 0; j < tsLength; j++) {
                            labels.putScalar(i, r.nextInt(nOut), j, 1.0);
                        }
                    }
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
                                inMask.putScalar(i, j, 1.0);
                            }
                        }
                    }

                    for (Bidirectional.Mode m : modes) {
                        String name = "mb=" + mb + ", maskType=" + maskType + ", mode=" + m + ", rnnType="
                                + (simple ? "SimpleRnn" : "LSTM");

                        System.out.println("Starting test: " + name);

                        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                                .updater(new NoOp())
                                .weightInit(WeightInit.XAVIER)
                                .list()
                                .layer(new LSTM.Builder().nIn(nIn).nOut(3).build())
                                .layer(new Bidirectional(m,
                                        (simple ?
                                                new SimpleRnn.Builder().nIn(3).nOut(3).build() :
                                                new LSTM.Builder().nIn(3).nOut(3).build())))
                                .layer(new RnnOutputLayer.Builder().nOut(nOut).build())
                                .build();


                        MultiLayerNetwork net = new MultiLayerNetwork(conf);
                        net.init();


                        boolean gradOK = GradientCheckUtil.checkGradients(net, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
                                DEFAULT_MIN_ABS_ERROR, PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, in, labels, inMask, null);
                        assertTrue(gradOK);


                        TestUtils.testModelSerialization(net);
                    }
                }
            }
        }
    }

    @Test
    public void testSimpleRnn() {
        int nOut = 5;

        double[] l1s = new double[]{0.0, 0.4};
        double[] l2s = new double[]{0.0, 0.6};

        Random r = new Random(12345);
        for (int mb : new int[]{1, 3}) {
            for (int tsLength : new int[]{1, 4}) {
                for (int nIn : new int[]{3, 1}) {
                    for (int layerSize : new int[]{4, 1}) {
                        for (boolean inputMask : new boolean[]{false, true}) {
                            for (int l = 0; l < l1s.length; l++) {

                                INDArray in = Nd4j.rand(new int[]{mb, nIn, tsLength});
                                INDArray labels = Nd4j.create(mb, nOut, tsLength);
                                for (int i = 0; i < mb; i++) {
                                    for (int j = 0; j < tsLength; j++) {
                                        labels.putScalar(i, r.nextInt(nOut), j, 1.0);
                                    }
                                }
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

                                String name = "testSimpleRnn() - mb=" + mb + ", tsLength = " + tsLength + ", maskType=" +
                                        maskType + ", l1=" + l1s[l] + ", l2=" + l2s[l];

                                System.out.println("Starting test: " + name);

                                MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                                        .updater(new NoOp())
                                        .weightInit(WeightInit.XAVIER)
                                        .activation(Activation.TANH)
                                        .l1(l1s[l])
                                        .l2(l2s[l])
                                        .list()
                                        .layer(new SimpleRnn.Builder().nIn(nIn).nOut(layerSize).build())
                                        .layer(new SimpleRnn.Builder().nIn(layerSize).nOut(layerSize).build())
                                        .layer(new RnnOutputLayer.Builder().nIn(layerSize).nOut(nOut)
                                                .activation(Activation.SOFTMAX).lossFunction(LossFunctions.LossFunction.MCXENT)
                                                .build())
                                        .build();

                                MultiLayerNetwork net = new MultiLayerNetwork(conf);
                                net.init();


                                boolean gradOK = GradientCheckUtil.checkGradients(net, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
                                        DEFAULT_MIN_ABS_ERROR, PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, in, labels, inMask, null);
                                assertTrue(gradOK);
                                TestUtils.testModelSerialization(net);
                            }
                        }
                    }
                }
            }
        }
    }

    @Test
    public void testLastTimeStepLayer(){
        int nIn = 3;
        int nOut = 5;
        int tsLength = 4;
        int layerSize = 8;

        Bidirectional.Mode[] modes = new Bidirectional.Mode[]{Bidirectional.Mode.CONCAT, Bidirectional.Mode.ADD,
                Bidirectional.Mode.AVERAGE, Bidirectional.Mode.MUL};

        Random r = new Random(12345);
        for (int mb : new int[]{1, 3}) {
            for (boolean inputMask : new boolean[]{false, true}) {
                for (boolean simple : new boolean[]{false, true}) {

                    INDArray in = Nd4j.rand(new int[]{mb, nIn, tsLength});
                    INDArray labels = Nd4j.create(mb, nOut);
                    for (int i = 0; i < mb; i++) {
                        labels.putScalar(i, r.nextInt(nOut), 1.0);
                    }
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

                    String name = "testLastTimeStepLayer() - mb=" + mb + ", tsLength = " + tsLength + ", maskType=" + maskType
                            + ", rnnType=" + (simple ? "SimpleRnn" : "LSTM");
                    if(PRINT_RESULTS){
                        System.out.println("Starting test: " + name);
                    }

                    MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                            .activation(Activation.TANH)
                            .updater(new NoOp())
                            .weightInit(WeightInit.XAVIER)
                            .list()
                            .layer(simple ? new SimpleRnn.Builder().nOut(layerSize).build() :
                                    new LSTM.Builder().nOut(layerSize).build())
                            .layer(new LastTimeStep(simple ? new SimpleRnn.Builder().nOut(layerSize).build() :
                                    new LSTM.Builder().nOut(layerSize).build()))
                            .layer(new OutputLayer.Builder().nOut(nOut).activation(Activation.SOFTMAX)
                                    .lossFunction(LossFunctions.LossFunction.MCXENT).build())
                            .setInputType(InputType.recurrent(nIn))
                            .build();

                    MultiLayerNetwork net = new MultiLayerNetwork(conf);
                    net.init();

                    boolean gradOK = GradientCheckUtil.checkGradients(net, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
                            DEFAULT_MIN_ABS_ERROR, PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, in, labels, inMask, null);
                    assertTrue(name, gradOK);
                    TestUtils.testModelSerialization(net);
                }
            }
        }
    }
}
