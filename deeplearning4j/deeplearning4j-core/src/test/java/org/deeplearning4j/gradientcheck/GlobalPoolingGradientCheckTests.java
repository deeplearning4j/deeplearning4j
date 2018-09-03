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
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.NoOp;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.Random;

import static org.junit.Assert.assertTrue;

/**
 * Created by Alex on 17/01/2017.
 */
public class GlobalPoolingGradientCheckTests extends BaseDL4JTest {

    static {
        Nd4j.setDataType(DataBuffer.Type.DOUBLE);
    }

    private static final boolean PRINT_RESULTS = true;
    private static final boolean RETURN_ON_FIRST_FAILURE = false;
    private static final double DEFAULT_EPS = 1e-6;
    private static final double DEFAULT_MAX_REL_ERROR = 1e-3;
    private static final double DEFAULT_MIN_ABS_ERROR = 1e-8;

    @Test
    public void testLSTMGlobalPoolingBasicMultiLayer() {
        //Basic test of global pooling w/ LSTM
        Nd4j.getRandom().setSeed(12345L);

        int timeSeriesLength = 10;
        int nIn = 5;
        int layerSize = 4;
        int nOut = 2;

        int[] minibatchSizes = new int[] {1, 3};
        PoolingType[] poolingTypes =
                        new PoolingType[] {PoolingType.AVG, PoolingType.SUM, PoolingType.MAX, PoolingType.PNORM};

        for (int miniBatchSize : minibatchSizes) {
            for (PoolingType pt : poolingTypes) {

                MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                                .updater(new NoOp()).weightInit(WeightInit.DISTRIBUTION)
                                .dist(new NormalDistribution(0, 1.0)).seed(12345L).list()
                                .layer(0, new GravesLSTM.Builder().nIn(nIn).nOut(layerSize).activation(Activation.TANH)
                                                .build())
                                .layer(1, new GlobalPoolingLayer.Builder().poolingType(pt).build())
                                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                                                .activation(Activation.SOFTMAX).nIn(layerSize).nOut(nOut).build())
                                .build();

                MultiLayerNetwork mln = new MultiLayerNetwork(conf);
                mln.init();

                Random r = new Random(12345L);
                INDArray input = Nd4j.zeros(miniBatchSize, nIn, timeSeriesLength);
                for (int i = 0; i < miniBatchSize; i++) {
                    for (int j = 0; j < nIn; j++) {
                        for (int k = 0; k < timeSeriesLength; k++) {
                            input.putScalar(new int[] {i, j, k}, r.nextDouble() - 0.5);
                        }
                    }
                }

                INDArray labels = Nd4j.zeros(miniBatchSize, nOut);
                for (int i = 0; i < miniBatchSize; i++) {
                    int idx = r.nextInt(nOut);
                    labels.putScalar(i, idx, 1.0);
                }

                if (PRINT_RESULTS) {
                    System.out.println("testLSTMGlobalPoolingBasicMultiLayer() - " + pt + ", minibatch = "
                                    + miniBatchSize);
                    for (int j = 0; j < mln.getnLayers(); j++)
                        System.out.println("Layer " + j + " # params: " + mln.getLayer(j).numParams());
                }

                boolean gradOK = GradientCheckUtil.checkGradients(mln, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
                                DEFAULT_MIN_ABS_ERROR, PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, input, labels);

                assertTrue(gradOK);
                TestUtils.testModelSerialization(mln);
            }
        }
    }

    @Test
    public void testCnnGlobalPoolingBasicMultiLayer() {
        //Basic test of global pooling w/ CNN
        Nd4j.getRandom().setSeed(12345L);

        int inputDepth = 3;
        int inputH = 5;
        int inputW = 4;
        int layerDepth = 4;
        int nOut = 2;

        int[] minibatchSizes = new int[] {1, 3};
        PoolingType[] poolingTypes =
                        new PoolingType[] {PoolingType.AVG, PoolingType.SUM, PoolingType.MAX, PoolingType.PNORM};

        for (int miniBatchSize : minibatchSizes) {
            for (PoolingType pt : poolingTypes) {

                MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                                .updater(new NoOp()).weightInit(WeightInit.DISTRIBUTION)
                                .dist(new NormalDistribution(0, 1.0)).seed(12345L).list()
                                .layer(0, new ConvolutionLayer.Builder().kernelSize(2, 2).stride(1, 1).nOut(layerDepth)
                                                .build())
                                .layer(1, new GlobalPoolingLayer.Builder().poolingType(pt).build())
                                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                                                .activation(Activation.SOFTMAX).nOut(nOut).build())

                                .setInputType(InputType.convolutional(inputH, inputW, inputDepth)).build();

                MultiLayerNetwork mln = new MultiLayerNetwork(conf);
                mln.init();

                Random r = new Random(12345L);
                INDArray input = Nd4j.rand(new int[] {miniBatchSize, inputDepth, inputH, inputW}).subi(0.5);

                INDArray labels = Nd4j.zeros(miniBatchSize, nOut);
                for (int i = 0; i < miniBatchSize; i++) {
                    int idx = r.nextInt(nOut);
                    labels.putScalar(i, idx, 1.0);
                }

                if (PRINT_RESULTS) {
                    System.out.println(
                                    "testCnnGlobalPoolingBasicMultiLayer() - " + pt + ", minibatch = " + miniBatchSize);
                    for (int j = 0; j < mln.getnLayers(); j++)
                        System.out.println("Layer " + j + " # params: " + mln.getLayer(j).numParams());
                }

                boolean gradOK = GradientCheckUtil.checkGradients(mln, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
                                DEFAULT_MIN_ABS_ERROR, PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, input, labels);

                assertTrue(gradOK);
                TestUtils.testModelSerialization(mln);
            }
        }
    }

    @Test
    public void testLSTMWithMasking() {
        //Basic test of GravesLSTM layer
        Nd4j.getRandom().setSeed(12345L);

        int timeSeriesLength = 10;
        int nIn = 5;
        int layerSize = 4;
        int nOut = 2;

        int miniBatchSize = 3;
        PoolingType[] poolingTypes =
                        new PoolingType[] {PoolingType.AVG, PoolingType.SUM, PoolingType.MAX, PoolingType.PNORM};

        for (PoolingType pt : poolingTypes) {

            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                            .updater(new NoOp()).weightInit(WeightInit.DISTRIBUTION)
                            .dist(new NormalDistribution(0, 1.0)).seed(12345L).list()
                            .layer(0, new GravesLSTM.Builder().nIn(nIn).nOut(layerSize).activation(Activation.TANH)
                                            .build())
                            .layer(1, new GlobalPoolingLayer.Builder().poolingType(pt).build())
                            .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                                            .activation(Activation.SOFTMAX).nIn(layerSize).nOut(nOut).build())
                            .build();

            MultiLayerNetwork mln = new MultiLayerNetwork(conf);
            mln.init();

            Random r = new Random(12345L);
            INDArray input = Nd4j.zeros(miniBatchSize, nIn, timeSeriesLength);
            for (int i = 0; i < miniBatchSize; i++) {
                for (int j = 0; j < nIn; j++) {
                    for (int k = 0; k < timeSeriesLength; k++) {
                        input.putScalar(new int[] {i, j, k}, r.nextDouble() - 0.5);
                    }
                }
            }

            INDArray featuresMask = Nd4j.create(miniBatchSize, timeSeriesLength);
            for (int i = 0; i < miniBatchSize; i++) {
                int to = timeSeriesLength - i;
                for (int j = 0; j < to; j++) {
                    featuresMask.putScalar(i, j, 1.0);
                }
            }

            INDArray labels = Nd4j.zeros(miniBatchSize, nOut);
            for (int i = 0; i < miniBatchSize; i++) {
                int idx = r.nextInt(nOut);
                labels.putScalar(i, idx, 1.0);
            }

            mln.setLayerMaskArrays(featuresMask, null);

            if (PRINT_RESULTS) {
                System.out.println("testLSTMGlobalPoolingBasicMultiLayer() - " + pt + ", minibatch = " + miniBatchSize);
                for (int j = 0; j < mln.getnLayers(); j++)
                    System.out.println("Layer " + j + " # params: " + mln.getLayer(j).numParams());
            }

            boolean gradOK = GradientCheckUtil.checkGradients(mln, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
                            DEFAULT_MIN_ABS_ERROR, PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, input, labels, featuresMask, null);

            assertTrue(gradOK);
            TestUtils.testModelSerialization(mln);
        }
    }


    @Test
    public void testCnnGlobalPoolingMasking() {
        //Global pooling w/ CNN + masking, where mask is along dimension 2, then separately test along dimension 3
        Nd4j.getRandom().setSeed(12345L);

        int inputDepth = 2;
        int inputH = 5;
        int inputW = 5;
        int layerDepth = 3;
        int nOut = 2;

        for (int maskDim = 2; maskDim <= 3; maskDim++) {

            int[] minibatchSizes = new int[] {1, 3};
            PoolingType[] poolingTypes =
                            new PoolingType[] {PoolingType.AVG, PoolingType.SUM, PoolingType.MAX, PoolingType.PNORM};

            for (int miniBatchSize : minibatchSizes) {
                for (PoolingType pt : poolingTypes) {

                    int[] kernel;
                    int[] stride;
                    if (maskDim == 2) {
                        //"time" (variable length) dimension is dimension 2
                        kernel = new int[] {2, inputW};
                        stride = new int[] {1, inputW};
                    } else {
                        kernel = new int[] {inputH, 2};
                        stride = new int[] {inputH, 1};
                    }

                    MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                                    .updater(new NoOp()).weightInit(WeightInit.DISTRIBUTION)
                                    .dist(new NormalDistribution(0, 1.0)).convolutionMode(ConvolutionMode.Same)
                                    .seed(12345L).list()
                                    .layer(0, new ConvolutionLayer.Builder().kernelSize(kernel).stride(stride)
                                                    .nOut(layerDepth).build())
                                    .layer(1, new GlobalPoolingLayer.Builder().poolingType(pt).build())
                                    .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                                                    .activation(Activation.SOFTMAX).nOut(nOut).build())

                                    .setInputType(InputType.convolutional(inputH, inputW, inputDepth)).build();

                    MultiLayerNetwork mln = new MultiLayerNetwork(conf);
                    mln.init();

                    Random r = new Random(12345L);
                    INDArray input = Nd4j.rand(new int[] {miniBatchSize, inputDepth, inputH, inputW}).subi(0.5);

                    INDArray inputMask;
                    if (miniBatchSize == 1) {
                        inputMask = Nd4j.create(new double[] {1, 1, 1, 1, 0}).reshape(1,1,(maskDim == 2 ? inputH : 1), (maskDim == 3 ? inputW : 1));
                    } else if (miniBatchSize == 3) {
                        inputMask = Nd4j.create(new double[][] {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 0}, {1, 1, 1, 0, 0}})
                                .reshape(miniBatchSize,1,(maskDim == 2 ? inputH : 1), (maskDim == 3 ? inputW : 1));
                    } else {
                        throw new RuntimeException();
                    }


                    INDArray labels = Nd4j.zeros(miniBatchSize, nOut);
                    for (int i = 0; i < miniBatchSize; i++) {
                        int idx = r.nextInt(nOut);
                        labels.putScalar(i, idx, 1.0);
                    }

                    if (PRINT_RESULTS) {
                        System.out.println("testCnnGlobalPoolingBasicMultiLayer() - " + pt + ", minibatch = "
                                        + miniBatchSize);
                        for (int j = 0; j < mln.getnLayers(); j++)
                            System.out.println("Layer " + j + " # params: " + mln.getLayer(j).numParams());
                    }

                    boolean gradOK = GradientCheckUtil.checkGradients(mln, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
                                    DEFAULT_MIN_ABS_ERROR, PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, input, labels, inputMask, null);

                    assertTrue(gradOK);
                    TestUtils.testModelSerialization(mln);
                }
            }
        }
    }
}
