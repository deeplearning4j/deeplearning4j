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

import lombok.val;
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.TestUtils;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.NoOp;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.lossfunctions.impl.*;

import java.util.Arrays;
import java.util.Random;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.nd4j.linalg.indexing.NDArrayIndex.all;
import static org.nd4j.linalg.indexing.NDArrayIndex.point;

/**Gradient checking tests with masking (i.e., variable length time series inputs, one-to-many and many-to-one etc)
 */
public class GradientCheckTestsMasking extends BaseDL4JTest {

    private static final boolean PRINT_RESULTS = true;
    private static final boolean RETURN_ON_FIRST_FAILURE = false;
    private static final double DEFAULT_EPS = 1e-6;
    private static final double DEFAULT_MAX_REL_ERROR = 1e-3;
    private static final double DEFAULT_MIN_ABS_ERROR = 1e-7;

    static {
        Nd4j.setDataType(DataBuffer.Type.DOUBLE);
    }

    private static class GradientCheckSimpleScenario {
        private final ILossFunction lf;
        private final Activation act;
        private final int nOut;
        private final int labelWidth;

        GradientCheckSimpleScenario(ILossFunction lf, Activation act, int nOut, int labelWidth) {
            this.lf = lf;
            this.act = act;
            this.nOut = nOut;
            this.labelWidth = labelWidth;
        }

    }

    @Test
    public void gradientCheckMaskingOutputSimple() {

        int timeSeriesLength = 5;
        boolean[][] mask = new boolean[5][0];
        mask[0] = new boolean[] {true, true, true, true, true}; //No masking
        mask[1] = new boolean[] {false, true, true, true, true}; //mask first output time step
        mask[2] = new boolean[] {false, false, false, false, true}; //time series classification: mask all but last
        mask[3] = new boolean[] {false, false, true, false, true}; //time series classification w/ variable length TS
        mask[4] = new boolean[] {true, true, true, false, true}; //variable length TS

        int nIn = 4;
        int layerSize = 3;

        GradientCheckSimpleScenario[] scenarios = new GradientCheckSimpleScenario[] {
                        new GradientCheckSimpleScenario(LossFunctions.LossFunction.MCXENT.getILossFunction(),
                                        Activation.SOFTMAX, 2, 2),
                        new GradientCheckSimpleScenario(LossMixtureDensity.builder().gaussians(2).labelWidth(3).build(),
                                        Activation.TANH, 10, 3),
                        new GradientCheckSimpleScenario(LossMixtureDensity.builder().gaussians(2).labelWidth(4).build(),
                                        Activation.IDENTITY, 12, 4),
                        new GradientCheckSimpleScenario(LossFunctions.LossFunction.L2.getILossFunction(),
                                        Activation.SOFTMAX, 2, 2)};

        for (GradientCheckSimpleScenario s : scenarios) {

            Random r = new Random(12345L);
            INDArray input = Nd4j.zeros(1, nIn, timeSeriesLength);
            for (int m = 0; m < 1; m++) {
                for (int j = 0; j < nIn; j++) {
                    for (int k = 0; k < timeSeriesLength; k++) {
                        input.putScalar(new int[] {m, j, k}, r.nextDouble() - 0.5);
                    }
                }
            }

            INDArray labels = Nd4j.zeros(1, s.labelWidth, timeSeriesLength);
            for (int m = 0; m < 1; m++) {
                for (int j = 0; j < timeSeriesLength; j++) {
                    int idx = r.nextInt(s.labelWidth);
                    labels.putScalar(new int[] {m, idx, j}, 1.0f);
                }
            }

            for (int i = 0; i < mask.length; i++) {

                //Create mask array:
                INDArray maskArr = Nd4j.create(1, timeSeriesLength);
                for (int j = 0; j < mask[i].length; j++) {
                    maskArr.putScalar(new int[] {0, j}, mask[i][j] ? 1.0 : 0.0);
                }

                MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(12345L)
                                .list()
                                .layer(0, new GravesLSTM.Builder().nIn(nIn).nOut(layerSize)
                                                .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0, 1))
                                                .updater(new NoOp()).build())
                                .layer(1, new RnnOutputLayer.Builder(s.lf).activation(s.act).nIn(layerSize).nOut(s.nOut)
                                                .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0, 1))
                                                .updater(new NoOp()).build())
                                .build();
                MultiLayerNetwork mln = new MultiLayerNetwork(conf);
                mln.init();

                boolean gradOK = GradientCheckUtil.checkGradients(mln, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
                                DEFAULT_MIN_ABS_ERROR, PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, input, labels, null, maskArr);

                String msg = "gradientCheckMaskingOutputSimple() - timeSeriesLength=" + timeSeriesLength
                                + ", miniBatchSize=" + 1;
                assertTrue(msg, gradOK);
                TestUtils.testModelSerialization(mln);
            }
        }
    }

    @Test
    public void testBidirectionalLSTMMasking() {
        Nd4j.getRandom().setSeed(12345L);

        int timeSeriesLength = 5;
        int nIn = 5;
        int layerSize = 4;
        int nOut = 3;

        int miniBatchSize = 3;

        INDArray[] masks = new INDArray[] {null,
                        Nd4j.create(new double[][] {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}),
                        Nd4j.create(new double[][] {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 0}, {1, 1, 1, 0, 0}}),
                        Nd4j.create(new double[][] {{1, 1, 1, 1, 1}, {0, 1, 1, 1, 1}, {0, 0, 1, 1, 1}})};

        int testNum = 0;
        for (INDArray mask : masks) {

            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                            .updater(new NoOp()).weightInit(WeightInit.DISTRIBUTION)
                            .dist(new NormalDistribution(0, 1.0)).seed(12345L).list()
                            .layer(0, new GravesBidirectionalLSTM.Builder().nIn(nIn).nOut(layerSize)
                                            .activation(Activation.TANH).build())
                            .layer(1, new GravesBidirectionalLSTM.Builder().nIn(layerSize).nOut(layerSize)
                                            .activation(Activation.TANH).build())
                            .layer(2, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                                            .activation(Activation.SOFTMAX).nIn(layerSize).nOut(nOut).build())
                            .build();

            MultiLayerNetwork mln = new MultiLayerNetwork(conf);
            mln.init();

            Random r = new Random(12345L);
            INDArray input = Nd4j.rand(new int[]{miniBatchSize, nIn, timeSeriesLength}, 'f').subi(0.5);

            INDArray labels = Nd4j.zeros(miniBatchSize, nOut, timeSeriesLength);
            for (int i = 0; i < miniBatchSize; i++) {
                for (int j = 0; j < nIn; j++) {
                    labels.putScalar(i, r.nextInt(nOut), j, 1.0);
                }
            }

            if (PRINT_RESULTS) {
                System.out.println("testBidirectionalLSTMMasking() - testNum = " + testNum++);
                for (int j = 0; j < mln.getnLayers(); j++)
                    System.out.println("Layer " + j + " # params: " + mln.getLayer(j).numParams());
            }

            boolean gradOK = GradientCheckUtil.checkGradients(mln, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
                            DEFAULT_MIN_ABS_ERROR, PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, input, labels, mask, mask);

            assertTrue(gradOK);
            TestUtils.testModelSerialization(mln);
        }
    }


    @Test
    public void testPerOutputMaskingMLP() {
        int nIn = 6;
        int layerSize = 4;

        INDArray mask1 = Nd4j.create(new double[] {1, 0, 0, 1, 0});
        INDArray mask3 = Nd4j.create(new double[][] {{1, 1, 1, 1, 1}, {0, 1, 0, 1, 0}, {1, 0, 0, 1, 1}});
        INDArray[] labelMasks = new INDArray[] {mask1, mask3};

        ILossFunction[] lossFunctions = new ILossFunction[] {new LossBinaryXENT(),
                        //                new LossCosineProximity(),    //Doesn't support per-output masking, as it doesn't make sense for cosine proximity
                        new LossHinge(), new LossKLD(), new LossKLD(), new LossL1(), new LossL2(), new LossMAE(),
                        new LossMAE(), new LossMAPE(), new LossMAPE(),
                        //                new LossMCXENT(),             //Per output masking on MCXENT+Softmax: not yet supported
                        new LossMCXENT(), new LossMSE(), new LossMSE(), new LossMSLE(), new LossMSLE(),
                        new LossNegativeLogLikelihood(), new LossPoisson(), new LossSquaredHinge()};

        Activation[] act = new Activation[] {Activation.SIGMOID, //XENT
                        //                Activation.TANH,
                        Activation.TANH, //Hinge
                        Activation.SIGMOID, //KLD
                        Activation.SOFTMAX, //KLD + softmax
                        Activation.TANH, //L1
                        Activation.TANH, //L2
                        Activation.TANH, //MAE
                        Activation.SOFTMAX, //MAE + softmax
                        Activation.TANH, //MAPE
                        Activation.SOFTMAX, //MAPE + softmax
                        //                Activation.SOFTMAX, //MCXENT + softmax: see comment above
                        Activation.SIGMOID, //MCXENT + sigmoid
                        Activation.TANH, //MSE
                        Activation.SOFTMAX, //MSE + softmax
                        Activation.SIGMOID, //MSLE - needs positive labels/activations (due to log)
                        Activation.SOFTMAX, //MSLE + softmax
                        Activation.SIGMOID, //NLL
                        Activation.SIGMOID, //Poisson
                        Activation.TANH //Squared hinge
        };

        for (INDArray labelMask : labelMasks) {

            val minibatch = labelMask.size(0);
            val nOut = labelMask.size(1);

            for (int i = 0; i < lossFunctions.length; i++) {
                ILossFunction lf = lossFunctions[i];
                Activation a = act[i];


                MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().updater(new NoOp())
                                .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0, 1)).seed(12345)
                                .list()
                                .layer(0, new DenseLayer.Builder().nIn(nIn).nOut(layerSize).activation(Activation.TANH)
                                                .build())
                                .layer(1, new OutputLayer.Builder().nIn(layerSize).nOut(nOut).lossFunction(lf)
                                                .activation(a).build())
                                .validateOutputLayerConfig(false)
                                .build();

                MultiLayerNetwork net = new MultiLayerNetwork(conf);
                net.init();

                INDArray[] fl = LossFunctionGradientCheck.getFeaturesAndLabels(lf, minibatch, nIn, nOut, 12345);
                INDArray features = fl[0];
                INDArray labels = fl[1];

                String msg = "testPerOutputMaskingMLP(): maskShape = " + Arrays.toString(labelMask.shape())
                                + ", loss function = " + lf + ", activation = " + a;

                System.out.println(msg);

                boolean gradOK = GradientCheckUtil.checkGradients(net, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
                                DEFAULT_MIN_ABS_ERROR, PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, features, labels, null, labelMask);

                assertTrue(msg, gradOK);
                TestUtils.testModelSerialization(net);
            }
        }
    }

    @Test
    public void testPerOutputMaskingRnn() {
        //For RNNs: per-output masking uses 3d masks (same shape as output/labels), as compared to the standard
        // 2d masks (used for per *example* masking)

        int nIn = 4;
        int layerSize = 4;
        int nOut = 4;

        //1 example, TS length 3
        INDArray mask1 = Nd4j.create(new double[] {1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0}, new int[] {1, nOut, 3}, 'f');
        //1 example, TS length 1
        INDArray mask2 = Nd4j.create(new double[] {1, 1, 0, 1}, new int[] {1, nOut, 1}, 'f');
        //3 examples, TS length 3
        INDArray mask3 = Nd4j.create(new double[] {
                        //With fortran order: dimension 0 (example) changes quickest, followed by dimension 1 (value within time
                        // step) followed by time index (least frequently)
                        1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0,

                        0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1,

                        1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0}, new int[] {3, nOut, 3}, 'f');
        INDArray[] labelMasks = new INDArray[] {mask1, mask2, mask3};

        ILossFunction[] lossFunctions = new ILossFunction[] {new LossBinaryXENT(),
                        //                new LossCosineProximity(),    //Doesn't support per-output masking, as it doesn't make sense for cosine proximity
                        new LossHinge(), new LossKLD(), new LossKLD(), new LossL1(), new LossL2(), new LossMAE(),
                        new LossMAE(), new LossMAPE(), new LossMAPE(),
                        //                new LossMCXENT(),             //Per output masking on MCXENT+Softmax: not yet supported
                        new LossMCXENT(), new LossMSE(), new LossMSE(), new LossMSLE(), new LossMSLE(),
                        new LossNegativeLogLikelihood(), new LossPoisson(), new LossSquaredHinge()};

        Activation[] act = new Activation[] {Activation.SIGMOID, //XENT
                        //                Activation.TANH,
                        Activation.TANH, //Hinge
                        Activation.SIGMOID, //KLD
                        Activation.SOFTMAX, //KLD + softmax
                        Activation.TANH, //L1
                        Activation.TANH, //L2
                        Activation.TANH, //MAE
                        Activation.SOFTMAX, //MAE + softmax
                        Activation.TANH, //MAPE
                        Activation.SOFTMAX, //MAPE + softmax
                        //                Activation.SOFTMAX, //MCXENT + softmax: see comment above
                        Activation.SIGMOID, //MCXENT + sigmoid
                        Activation.TANH, //MSE
                        Activation.SOFTMAX, //MSE + softmax
                        Activation.SIGMOID, //MSLE - needs positive labels/activations (due to log)
                        Activation.SOFTMAX, //MSLE + softmax
                        Activation.SIGMOID, //NLL
                        Activation.SIGMOID, //Poisson
                        Activation.TANH //Squared hinge
        };

        for (INDArray labelMask : labelMasks) {

            val minibatch = labelMask.size(0);
            val tsLength = labelMask.size(2);

            for (int i = 0; i < lossFunctions.length; i++) {
                ILossFunction lf = lossFunctions[i];
                Activation a = act[i];

                Nd4j.getRandom().setSeed(12345);
                MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().updater(new NoOp())
                                .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0, 1)).seed(12345)
                                .list()
                                .layer(0, new GravesLSTM.Builder().nIn(nIn).nOut(layerSize).activation(Activation.TANH)
                                                .build())
                                .layer(1, new RnnOutputLayer.Builder().nIn(layerSize).nOut(nOut).lossFunction(lf)
                                                .activation(a).build())
                                .validateOutputLayerConfig(false)
                                .build();

                MultiLayerNetwork net = new MultiLayerNetwork(conf);
                net.init();

                INDArray[] fl = LossFunctionGradientCheck.getFeaturesAndLabels(lf, new long[] {minibatch, nIn, tsLength},
                                new long[] {minibatch, nOut, tsLength}, 12345);
                INDArray features = fl[0];
                INDArray labels = fl[1];

                String msg = "testPerOutputMaskingRnn(): maskShape = " + Arrays.toString(labelMask.shape())
                                + ", loss function = " + lf + ", activation = " + a;

                System.out.println(msg);

                boolean gradOK = GradientCheckUtil.checkGradients(net, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
                                DEFAULT_MIN_ABS_ERROR, PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, features, labels, null, labelMask);

                assertTrue(msg, gradOK);


                //Check the equivalent compgraph:
                Nd4j.getRandom().setSeed(12345);
                ComputationGraphConfiguration cg = new NeuralNetConfiguration.Builder().updater(new NoOp())
                                .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0, 2)).seed(12345)
                                .graphBuilder().addInputs("in")
                                .addLayer("0", new GravesLSTM.Builder().nIn(nIn).nOut(layerSize)
                                                .activation(Activation.TANH).build(), "in")
                                .addLayer("1", new RnnOutputLayer.Builder().nIn(layerSize).nOut(nOut).lossFunction(lf)
                                                .activation(a).build(), "0")
                                .setOutputs("1").validateOutputLayerConfig(false).build();

                ComputationGraph graph = new ComputationGraph(cg);
                graph.init();

                gradOK = GradientCheckUtil.checkGradients(graph, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
                                DEFAULT_MIN_ABS_ERROR, PRINT_RESULTS, RETURN_ON_FIRST_FAILURE,
                                new INDArray[] {features}, new INDArray[] {labels}, null, new INDArray[]{labelMask}, null);

                assertTrue(msg + " (compgraph)", gradOK);
                TestUtils.testModelSerialization(graph);
            }
        }
    }


    @Test
    public void testOutputLayerMasking(){
        Nd4j.getRandom().setSeed(12345);
        //Idea: RNN input, global pooling, OutputLayer - with "per example" mask arrays

        int mb = 10;
        int tsLength = 5;
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .weightInit(new NormalDistribution(0,2))
                .updater(new NoOp())
                .list()
                .layer(new LSTM.Builder().nIn(10).nOut(10).build())
                .layer(new GlobalPoolingLayer.Builder().poolingType(PoolingType.AVG).build())
                .layer(new OutputLayer.Builder().nIn(10).nOut(10).activation(Activation.SOFTMAX).build())
                .setInputType(InputType.recurrent(10))
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        INDArray f = Nd4j.rand(new int[]{mb, 10, tsLength});
        INDArray l = TestUtils.randomOneHot(mb, 10);
        INDArray lm = TestUtils.randomBernoulli(mb, 1);

        assertTrue(lm.sumNumber().intValue() > 0);

        boolean gradOK = GradientCheckUtil.checkGradients(net, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
                DEFAULT_MIN_ABS_ERROR, PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, f, l, null, lm);
        assertTrue(gradOK);

        //Also ensure score doesn't depend on masked feature or label values
        double score = net.score(new DataSet(f,l,null,lm));

        for( int i=0; i<mb; i++ ){
            if(lm.getDouble(i) != 0.0){
                continue;
            }

            INDArray fView = f.get(point(i), all(),all());
            fView.assign(Nd4j.rand(fView.shape()));

            INDArray lView = l.get(point(i), all());
            lView.assign(TestUtils.randomOneHot(1, lView.size(1)));

            double score2 = net.score(new DataSet(f,l,null,lm));

            assertEquals(String.valueOf(i), score, score2, 1e-8);
        }
    }

    @Test
    public void testOutputLayerMaskingCG(){
        Nd4j.getRandom().setSeed(12345);
        //Idea: RNN input, global pooling, OutputLayer - with "per example" mask arrays

        int mb = 10;
        int tsLength = 5;
        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
                .weightInit(new NormalDistribution(0,2))
                .updater(new NoOp())
                .graphBuilder()
                .addInputs("in")
                .layer("0", new LSTM.Builder().nIn(10).nOut(10).build(), "in")
                .layer("1", new GlobalPoolingLayer.Builder().poolingType(PoolingType.AVG).build(), "0")
                .layer("out", new OutputLayer.Builder().nIn(10).nOut(10).activation(Activation.SOFTMAX).build(), "1")
                .setOutputs("out")
                .setInputTypes(InputType.recurrent(10))
                .build();

        ComputationGraph net = new ComputationGraph(conf);
        net.init();

        INDArray f = Nd4j.rand(new int[]{mb, 10, tsLength});
        INDArray l = TestUtils.randomOneHot(mb, 10);
        INDArray lm = TestUtils.randomBernoulli(mb, 1);

        assertTrue(lm.sumNumber().intValue() > 0);

        boolean gradOK = GradientCheckUtil.checkGradients(net, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
                DEFAULT_MIN_ABS_ERROR, PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, new INDArray[]{f}, new INDArray[]{l},
                null, new INDArray[]{lm});
        assertTrue(gradOK);

        //Also ensure score doesn't depend on masked feature or label values
        double score = net.score(new DataSet(f,l,null,lm));

        for( int i=0; i<mb; i++ ){
            if(lm.getDouble(i) != 0.0){
                continue;
            }

            INDArray fView = f.get(point(i), all(),all());
            fView.assign(Nd4j.rand(fView.shape()));

            INDArray lView = l.get(point(i), all());
            lView.assign(TestUtils.randomOneHot(1, lView.size(1)));

            double score2 = net.score(new DataSet(f,l,null,lm));

            assertEquals(String.valueOf(i), score, score2, 1e-8);
        }
    }
}
