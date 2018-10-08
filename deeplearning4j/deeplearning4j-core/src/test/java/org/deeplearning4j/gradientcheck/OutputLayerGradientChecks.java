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
import org.deeplearning4j.nn.conf.layers.CnnLossLayer;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnLossLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.random.impl.BernoulliDistribution;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.NoOp;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.lossfunctions.impl.LossMCXENT;
import org.nd4j.linalg.lossfunctions.impl.LossMSE;

import java.util.Random;

import static org.junit.Assert.assertTrue;

public class OutputLayerGradientChecks extends BaseDL4JTest {

    private static final boolean PRINT_RESULTS = true;
    private static final boolean RETURN_ON_FIRST_FAILURE = false;
    private static final double DEFAULT_EPS = 1e-6;
    private static final double DEFAULT_MAX_REL_ERROR = 1e-3;
    private static final double DEFAULT_MIN_ABS_ERROR = 1e-8;

    static {
        Nd4j.setDataType(DataBuffer.Type.DOUBLE);
    }

    @Test
    public void testRnnLossLayer() {
        Nd4j.getRandom().setSeed(12345L);

        int timeSeriesLength = 4;
        int nIn = 2;
        int layerSize = 2;
        int nOut = 2;
        int miniBatchSize = 3;

        ILossFunction[] lfs = new ILossFunction[]{new LossMSE(), new LossMCXENT()};

        for (int maskType = 0; maskType < 3; maskType++) {

            Random r = new Random(12345L);
            INDArray input = Nd4j.rand(new int[]{miniBatchSize, nIn, timeSeriesLength});
            INDArray labels = Nd4j.zeros(miniBatchSize, nOut, timeSeriesLength);
            for (int i = 0; i < miniBatchSize; i++) {
                for (int j = 0; j < timeSeriesLength; j++) {
                    int idx = r.nextInt(nOut);
                    labels.putScalar(new int[]{i, idx, j}, 1.0);
                }
            }

            INDArray labelMask;
            String mt;
            switch (maskType) {
                case 0:
                    //No masking
                    labelMask = null;
                    mt = "none";
                    break;
                case 1:
                    //Per time step masking
                    labelMask = Nd4j.createUninitialized(miniBatchSize, timeSeriesLength);
                    Nd4j.getExecutioner().exec(new BernoulliDistribution(labelMask, 0.5));
                    mt = "PerTimeStep";
                    break;
                case 2:
                    //Per output masking:
                    labelMask = Nd4j.createUninitialized(new int[]{miniBatchSize, nOut, timeSeriesLength});
                    Nd4j.getExecutioner().exec(new BernoulliDistribution(labelMask, 0.5));
                    mt = "PerOutput";
                    break;
                default:
                    throw new RuntimeException();
            }

            for (ILossFunction lf : lfs) {

                Activation oa = maskType == 2 ? Activation.SIGMOID : Activation.SOFTMAX;

                MultiLayerConfiguration conf =
                        new NeuralNetConfiguration.Builder().seed(12345L)
                                .updater(new NoOp())
                                .list()
                                .layer(new LSTM.Builder().nIn(nIn).nOut(layerSize).activation(Activation.TANH)
                                        .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0, 1.0))
                                        .updater(new NoOp()).build())
                                .layer(new RnnLossLayer.Builder(lf)
                                        .activation(oa)
                                        .build())
                                .validateOutputLayerConfig(false).build();

                MultiLayerNetwork mln = new MultiLayerNetwork(conf);
                mln.init();

                String testName = "testRnnLossLayer(lf=" + lf + ", maskType=" + mt + ", outputActivation = " + oa + ")";
                if (PRINT_RESULTS) {
                    System.out.println(testName);
                    for (int j = 0; j < mln.getnLayers(); j++)
                        System.out.println("Layer " + j + " # params: " + mln.getLayer(j).numParams());
                }

                System.out.println("Starting test: " + testName);
                boolean gradOK = GradientCheckUtil.checkGradients(mln, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
                        DEFAULT_MIN_ABS_ERROR, PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, input, labels, null, labelMask);

                assertTrue(testName, gradOK);
                TestUtils.testModelSerialization(mln);
            }
        }
    }


    @Test
    public void testCnnLossLayer() {
        Nd4j.getRandom().setSeed(12345L);

        int dIn = 2;
        int layerSize = 2;
        int dOut = 2;
        int miniBatchSize = 3;

        ILossFunction[] lfs = new ILossFunction[]{new LossMSE(), new LossMCXENT()};

        int[] heights = {4, 4, 5};
        int[] widths = {4, 5, 6};

        for (int i = 0; i < heights.length; i++) {
            int h = heights[i];
            int w = widths[i];

            for (int maskType = 0; maskType < 4; maskType++) {

                Random r = new Random(12345L);
                INDArray input = Nd4j.rand(new int[]{miniBatchSize, dIn, h, w});

                INDArray labelMask;
                String mt;
                switch (maskType) {
                    case 0:
                        //No masking
                        labelMask = null;
                        mt = "none";
                        break;
                    case 1:
                        //Per example masking (2d mask, shape [minibatch, 1]
                        labelMask = Nd4j.createUninitialized(miniBatchSize, 1);
                        Nd4j.getExecutioner().exec(new BernoulliDistribution(labelMask, 0.5));
                        mt = "PerTimeStep";
                        break;
                    case 2:
                        //Per x/y masking (3d mask, shape [minibatch, h, w])
                        labelMask = Nd4j.createUninitialized(new int[]{miniBatchSize, h, w});
                        Nd4j.getExecutioner().exec(new BernoulliDistribution(labelMask, 0.5));
                        mt = "PerXY";
                        break;
                    case 3:
                        //Per output masking (4d mask, same shape as output [minibatch, c, h, w])
                        labelMask = Nd4j.createUninitialized(new int[]{miniBatchSize, dOut, h, w});
                        Nd4j.getExecutioner().exec(new BernoulliDistribution(labelMask, 0.5));
                        mt = "PerOutput";
                        break;
                    default:
                        throw new RuntimeException();
                }

                for (ILossFunction lf : lfs) {

                    INDArray labels;
                    if (lf instanceof LossMSE) {
                        labels = Nd4j.rand(new int[]{miniBatchSize, dOut, h, w});
                    } else {
                        labels = Nd4j.zeros(miniBatchSize, dOut, h, w);
                        for (int mb = 0; mb < miniBatchSize; mb++) {
                            for (int x = 0; x < w; x++) {
                                for (int y = 0; y < h; y++) {
                                    int idx = r.nextInt(dOut);
                                    labels.putScalar(new int[]{mb, idx, y, x}, 1.0);
                                }
                            }
                        }
                    }

                    Activation oa = maskType == 3 ? Activation.SIGMOID : Activation.SOFTMAX;

                    MultiLayerConfiguration conf =
                            new NeuralNetConfiguration.Builder().seed(12345L)
                                    .updater(new NoOp())
                                    .convolutionMode(ConvolutionMode.Same)
                                    .list()
                                    .layer(new ConvolutionLayer.Builder().nIn(dIn).nOut(dOut).activation(Activation.TANH)
                                            .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0, 1.0))
                                            .updater(new NoOp()).build())
                                    .layer(new CnnLossLayer.Builder(lf)
                                            .activation(oa)
                                            .build())
                                    .validateOutputLayerConfig(false).build();

                    MultiLayerNetwork mln = new MultiLayerNetwork(conf);
                    mln.init();

                    String testName = "testCnnLossLayer(lf=" + lf + ", maskType=" + mt + ", outputActivation = " + oa + ")";
                    if (PRINT_RESULTS) {
                        System.out.println(testName);
                        for (int j = 0; j < mln.getnLayers(); j++)
                            System.out.println("Layer " + j + " # params: " + mln.getLayer(j).numParams());
                    }

                    System.out.println("Starting test: " + testName);
                    boolean gradOK = GradientCheckUtil.checkGradients(mln, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
                            DEFAULT_MIN_ABS_ERROR, PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, input, labels, null, labelMask);

                    assertTrue(testName, gradOK);
                    TestUtils.testModelSerialization(mln);
                }
            }
        }
    }
}
