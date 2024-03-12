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

package org.eclipse.deeplearning4j.dl4jcore.gradientcheck;

import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.nn.conf.ListBuilder;
import org.eclipse.deeplearning4j.dl4jcore.TestUtils;
import org.deeplearning4j.gradientcheck.GradientCheckUtil;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.preprocessor.RnnToCnnPreProcessor;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.NoOp;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertTrue;

@Tag(TagNames.NDARRAY_ETL)
@Tag(TagNames.TRAINING)
@Tag(TagNames.DL4J_OLD_API)
@NativeTag
public class LSTMGradientCheckTests extends BaseDL4JTest {

    private static final boolean PRINT_RESULTS = true;
    private static final boolean RETURN_ON_FIRST_FAILURE = false;
    private static final double DEFAULT_EPS = 1e-6;
    private static final double DEFAULT_MAX_REL_ERROR = 1e-3;
    private static final double DEFAULT_MIN_ABS_ERROR = 1e-8;

    static {
        Nd4j.setDataType(DataType.DOUBLE);
    }

    @Override
    public long getTimeoutMilliseconds() {
        return 90000L;
    }

    @Test
    public void testLSTMBasicMultiLayer() {
        //Basic test of LSTM layer
        Nd4j.getRandom().setSeed(12345L);

        int timeSeriesLength = 4;
        int nIn = 2;
        int layerSize = 2;
        int nOut = 2;
        int miniBatchSize = 5;

        boolean[] LSTM = new boolean[] {true, false};

        for (boolean graves : LSTM) {

            Layer l0;
            Layer l1;
            if (graves) {
                l0 = new LSTM.Builder().nIn(nIn).nOut(layerSize).activation(Activation.SIGMOID)
                                .dist(new NormalDistribution(0, 1.0))
                                .updater(new NoOp()).build();
                l1 = new LSTM.Builder().nIn(layerSize).nOut(layerSize).activation(Activation.SIGMOID)
                                .dist(new NormalDistribution(0, 1.0))
                                .updater(new NoOp()).build();
            } else {
                l0 = new LSTM.Builder().nIn(nIn).nOut(layerSize).activation(Activation.SIGMOID)
                                .dist(new NormalDistribution(0, 1.0))
                                .updater(new NoOp()).build();
                l1 = new LSTM.Builder().nIn(layerSize).nOut(layerSize).activation(Activation.SIGMOID)
                                .dist(new NormalDistribution(0, 1.0))
                                .updater(new NoOp()).build();
            }

            MultiLayerConfiguration conf =
                            new NeuralNetConfiguration.Builder().seed(12345L)
                                            .dataType(DataType.DOUBLE)
                                            .list()
                                            .layer(0, l0).layer(1,
                                                            l1)
                                            .layer(2, new RnnOutputLayer.Builder(LossFunction.MCXENT)
                                                            .activation(Activation.SOFTMAX).nIn(layerSize).nOut(nOut)

                                                            .dist(new NormalDistribution(0, 1.0)).updater(new NoOp())
                                                            .build())
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

            INDArray labels = Nd4j.zeros(miniBatchSize, nOut, timeSeriesLength);
            for (int i = 0; i < miniBatchSize; i++) {
                for (int j = 0; j < timeSeriesLength; j++) {
                    int idx = r.nextInt(nOut);
                    labels.putScalar(new int[] {i, idx, j}, 1.0);
                }
            }

            String testName = "testLSTMBasic(" + (graves ? "LSTM" : "LSTM") + ")";
            if (PRINT_RESULTS) {
                System.out.println(testName);
//                for (int j = 0; j < mln.getnLayers(); j++)
//                    System.out.println("Layer " + j + " # params: " + mln.getLayer(j).numParams());
            }

            boolean gradOK = GradientCheckUtil.checkGradients(mln, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
                            DEFAULT_MIN_ABS_ERROR, PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, input, labels);

            assertTrue(gradOK, testName);
            TestUtils.testModelSerialization(mln);
        }
    }

    @Test
    public void testGradientLSTMFull() {

        int timeSeriesLength = 4;
        int nIn = 3;
        int layerSize = 4;
        int nOut = 2;
        int miniBatchSize = 2;

        boolean[] LSTM = new boolean[] {true, false};

        for (boolean graves : LSTM) {

            Random r = new Random(12345L);
            INDArray input = Nd4j.rand(DataType.DOUBLE,'f',new long[]{miniBatchSize, nIn, timeSeriesLength}).subi(0.5);

            INDArray labels = Nd4j.zeros(miniBatchSize, nOut, timeSeriesLength);
            for (int i = 0; i < miniBatchSize; i++) {
                for (int j = 0; j < timeSeriesLength; j++) {
                    int idx = r.nextInt(nOut);
                    labels.putScalar(new int[] {i, idx, j}, 1.0f);
                }
            }


            //use l2vals[i] with l1vals[i]
            double[] l2vals = {0.4, 0.0};
            double[] l1vals = {0.0, 0.5};
            double[] biasL2 = {0.3, 0.0};
            double[] biasL1 = {0.0, 0.6};
            Activation[] activFns = {Activation.TANH, Activation.SOFTSIGN};
            LossFunction[] lossFunctions = {LossFunction.MCXENT, LossFunction.MSE};
            Activation[] outputActivations = {Activation.SOFTMAX, Activation.TANH};

            for (int i = 0; i < l2vals.length; i++) {

                LossFunction lf = lossFunctions[i];
                Activation outputActivation = outputActivations[i];
                double l2 = l2vals[i];
                double l1 = l1vals[i];
                Activation afn = activFns[i];

                NeuralNetConfiguration.Builder conf =
                        new NeuralNetConfiguration.Builder()
                                .dataType(DataType.DOUBLE)
                                .seed(12345L)
                                .dist(new NormalDistribution(0, 1)).updater(new NoOp());

                if (l1 > 0.0)
                    conf.l1(l1);
                if (l2 > 0.0)
                    conf.l2(l2);
                if (biasL2[i] > 0)
                    conf.l2Bias(biasL2[i]);
                if (biasL1[i] > 0)
                    conf.l1Bias(biasL1[i]);

                Layer layer;
                if (graves) {
                    layer = new LSTM.Builder().nIn(nIn).nOut(layerSize).activation(afn).build();
                } else {
                    layer = new LSTM.Builder().nIn(nIn).nOut(layerSize).activation(afn).build();
                }

                ListBuilder conf2 = conf.list().layer(0, layer)
                        .layer(1, new RnnOutputLayer.Builder(lf).activation(outputActivation)
                                .nIn(layerSize).nOut(nOut).build())
                        ;

                MultiLayerNetwork mln = new MultiLayerNetwork(conf2.build());
                mln.init();

                String testName = "testGradientLSTMFull(" + (graves ? "LSTM" : "LSTM")
                        + " - activationFn=" + afn + ", lossFn=" + lf + ", outputActivation="
                        + outputActivation + ", l2=" + l2 + ", l1=" + l1;
                if (PRINT_RESULTS) {
                    System.out.println(testName);
                }

                boolean gradOK = GradientCheckUtil.checkGradients(new GradientCheckUtil.MLNConfig().net(mln).input(input)
                        .labels(labels).subset(true).maxPerParam(128));

                assertTrue(gradOK, testName);
                TestUtils.testModelSerialization(mln);
            }
        }
    }


    @Test
    public void testGradientLSTMEdgeCases() {
        Nd4j.getExecutioner().enableVerboseMode(true);
        Nd4j.getExecutioner().enableDebugMode(true);
        //Edge cases: T=1, miniBatchSize=1, both
        int[] timeSeriesLength = {1, 5, 1};
        int[] miniBatchSize = {7, 1, 1};

        Nd4j.getRandom().setSeed(42);

        int nIn = 3;
        int layerSize = 4;
        int nOut = 2;

        boolean[] LSTM = new boolean[] {true, false};

        for (boolean graves : LSTM) {

            for (int i = 0; i < timeSeriesLength.length; i++) {

                INDArray input = Nd4j.rand(DataType.DOUBLE, miniBatchSize[i], nIn, timeSeriesLength[i]);

                INDArray labels = TestUtils.randomOneHotTimeSeries(miniBatchSize[i], nOut, timeSeriesLength[i],42);

                Layer layer;
                if (graves) {
                    layer = new LSTM.Builder().nIn(nIn).nOut(layerSize).activation(Activation.TANH).build();
                } else {
                    layer = new LSTM.Builder().nIn(nIn).nOut(layerSize).activation(Activation.TANH).build();
                }

                MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(12345L)
                                .dataType(DataType.DOUBLE)
                                .dist(new NormalDistribution(0, 1))
                                .updater(new NoOp()).list().layer(0, layer)
                                .layer(1, new RnnOutputLayer.Builder(LossFunction.MCXENT).activation(Activation.SOFTMAX)
                                                .nIn(layerSize).nOut(nOut).build())
                                .build();
                MultiLayerNetwork mln = new MultiLayerNetwork(conf);
                mln.init();

                String msg = "testGradientLSTMEdgeCases(" + (graves ? "LSTM" : "LSTM") + " - timeSeriesLength="
                                + timeSeriesLength[i] + ", miniBatchSize=" + miniBatchSize[i];
                System.out.println(msg);
                boolean gradOK = GradientCheckUtil.checkGradients(mln, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
                                DEFAULT_MIN_ABS_ERROR, PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, input, labels);
                assertTrue(gradOK, msg);
                TestUtils.testModelSerialization(mln);
            }
        }
    }



    @Test
    public void testGradientCnnFfRnn() {
        //Test gradients with CNN -> FF -> LSTM -> RnnOutputLayer
        //time series input/output (i.e., video classification or similar)

        int nChannelsIn = 2;
        int inputSize = 6 * 6 * nChannelsIn; //10px x 10px x 3 channels
        int miniBatchSize = 2;
        int timeSeriesLength = 4;
        int nClasses = 2;

        //Generate
        Nd4j.getRandom().setSeed(12345);
        INDArray input = Nd4j.rand(new int[] {miniBatchSize, inputSize, timeSeriesLength});
        INDArray labels = Nd4j.zeros(miniBatchSize, nClasses, timeSeriesLength);
        Random r = new Random(12345);
        for (int i = 0; i < miniBatchSize; i++) {
            for (int j = 0; j < timeSeriesLength; j++) {
                int idx = r.nextInt(nClasses);
                labels.putScalar(new int[] {i, idx, j}, 1.0);
            }
        }


        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().updater(new NoOp()).seed(12345)
                        .dataType(DataType.DOUBLE)
                        .dist(new UniformDistribution(-2, 2)).list()
                        .layer(0, new ConvolutionLayer.Builder(3, 3).nIn(2).nOut(3).stride(1, 1)
                                        .activation(Activation.TANH).build()) //Out: (10-5)/1+1 = 6 -> 6x6x5
                        .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2)
                                        .stride(1, 1).build()) //Out: (6-2)/1+1 = 5 -> 5x5x5
                        .layer(2, new DenseLayer.Builder().nIn(27).nOut(4).activation(Activation.TANH).build())
                        .layer(3, new LSTM.Builder().nIn(4).nOut(3).activation(Activation.TANH).build())
                        .layer(4, new RnnOutputLayer.Builder().lossFunction(LossFunction.MCXENT).nIn(3).nOut(nClasses)
                                        .activation(Activation.SOFTMAX).build())
                        .setInputType(InputType.convolutional(6, 6, 2)).build();

        //Here: ConvolutionLayerSetup in config builder doesn't know that we are expecting time series input, not standard FF input -> override it here
        conf.getInputPreProcessors().put(0, new RnnToCnnPreProcessor(6, 6, 2));

        MultiLayerNetwork mln = new MultiLayerNetwork(conf);
        mln.init();

        System.out.println("Params per layer:");
        for (int i = 0; i < mln.getnLayers(); i++) {
            System.out.println("layer " + i + "\t" + mln.getLayer(i).numParams());
        }

        boolean gradOK = GradientCheckUtil.checkGradients(new GradientCheckUtil.MLNConfig().net(mln).input(input)
                .labels(labels).subset(true).maxPerParam(32));
        assertTrue(gradOK);
        TestUtils.testModelSerialization(mln);
    }
}
