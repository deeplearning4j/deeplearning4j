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
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.variational.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.impl.ActivationIdentity;
import org.nd4j.linalg.activations.impl.ActivationTanH;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.random.impl.BernoulliDistribution;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.learning.config.NoOp;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.nd4j.linalg.lossfunctions.impl.LossMAE;
import org.nd4j.linalg.lossfunctions.impl.LossMSE;

import java.util.Arrays;

import static org.junit.Assert.assertTrue;

/**
 * @author Alex Black
 */
public class VaeGradientCheckTests extends BaseDL4JTest {

    private static final boolean PRINT_RESULTS = true;
    private static final boolean RETURN_ON_FIRST_FAILURE = false;
    private static final double DEFAULT_EPS = 1e-6;
    private static final double DEFAULT_MAX_REL_ERROR = 1e-3;
    private static final double DEFAULT_MIN_ABS_ERROR = 1e-8;

    static {
        Nd4j.setDataType(DataBuffer.Type.DOUBLE);
    }

    @Test
    public void testVaeAsMLP() {
        //Post pre-training: a VAE can be used as a MLP, by taking the mean value from p(z|x) as the output
        //This gradient check tests this part

        Activation[] activFns = {Activation.IDENTITY, Activation.TANH, Activation.IDENTITY, Activation.TANH, Activation.IDENTITY, Activation.TANH};

        LossFunction[] lossFunctions = {LossFunction.MCXENT, LossFunction.MCXENT, LossFunction.MSE, LossFunction.MSE, LossFunction.MCXENT, LossFunction.MSE};
        Activation[] outputActivations = {Activation.SOFTMAX, Activation.SOFTMAX, Activation.TANH, Activation.TANH, Activation.SOFTMAX, Activation.TANH};

        //use l2vals[i] with l1vals[i]
        double[] l2vals = {0.4, 0.0, 0.4, 0.4, 0.0, 0.0};
        double[] l1vals = {0.0, 0.0, 0.5, 0.0, 0.0, 0.5};
        double[] biasL2 = {0.0, 0.0, 0.0, 0.2, 0.0, 0.4};
        double[] biasL1 = {0.0, 0.0, 0.6, 0.0, 0.0, 0.0};

        int[][] encoderLayerSizes = new int[][] {{5}, {5}, {5, 6}, {5, 6}, {5}, {5, 6}};
        int[][] decoderLayerSizes = new int[][] {{6}, {7, 8}, {6}, {7, 8}, {6}, {7, 8}};

        int[] minibatches = new int[]{1,5,4,3,1,4};

        Nd4j.getRandom().setSeed(12345);
        for( int i=0; i<activFns.length; i++ ){
            LossFunction lf = lossFunctions[i];
            Activation outputActivation = outputActivations[i];
            double l2 = l2vals[i];
            double l1 = l1vals[i];
            int[] encoderSizes = encoderLayerSizes[i];
            int[] decoderSizes = decoderLayerSizes[i];
            int minibatch = minibatches[i];
            INDArray input = Nd4j.rand(minibatch, 4);
            INDArray labels = Nd4j.create(minibatch, 3);
            for (int j = 0; j < minibatch; j++) {
                labels.putScalar(j, j % 3, 1.0);
            }
            Activation afn = activFns[i];

            MultiLayerConfiguration conf =
                    new NeuralNetConfiguration.Builder().l2(l2).l1(l1)
                            .updater(new NoOp())
                            .l2Bias(biasL2[i]).l1Bias(biasL1[i])
                            .updater(new NoOp()).seed(12345L).list()
                            .layer(0, new VariationalAutoencoder.Builder().nIn(4)
                                    .nOut(3).encoderLayerSizes(encoderSizes)
                                    .decoderLayerSizes(decoderSizes)
                                    .weightInit(WeightInit.DISTRIBUTION)
                                    .dist(new NormalDistribution(0, 1))
                                    .activation(afn)
                                    .build())
                            .layer(1, new OutputLayer.Builder(lf)
                                    .activation(outputActivation).nIn(3).nOut(3)
                                    .weightInit(WeightInit.DISTRIBUTION)
                                    .dist(new NormalDistribution(0, 1))
                                    .build())
                            .build();

            MultiLayerNetwork mln = new MultiLayerNetwork(conf);
            mln.init();

            String msg = "testVaeAsMLP() - activationFn=" + afn + ", lossFn=" + lf
                    + ", outputActivation=" + outputActivation + ", encLayerSizes = "
                    + Arrays.toString(encoderSizes) + ", decLayerSizes = "
                    + Arrays.toString(decoderSizes) + ", l2=" + l2 + ", l1=" + l1;
            if (PRINT_RESULTS) {
                System.out.println(msg);
                for (int j = 0; j < mln.getnLayers(); j++)
                    System.out.println("Layer " + j + " # params: " + mln.getLayer(j).numParams());
            }

            boolean gradOK = GradientCheckUtil.checkGradients(mln, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
                    DEFAULT_MIN_ABS_ERROR, PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, input,
                    labels);
            assertTrue(msg, gradOK);
            TestUtils.testModelSerialization(mln);
        }
    }


    @Test
    public void testVaePretrain() {
        Nd4j.getRandom().setSeed(12345);
        Activation[] activFns = {Activation.IDENTITY, Activation.TANH, Activation.IDENTITY, Activation.TANH};

        LossFunction[] lossFunctions = {LossFunction.MCXENT, LossFunction.MCXENT, LossFunction.MSE, LossFunction.MSE};
        Activation[] outputActivations = {Activation.SOFTMAX, Activation.SOFTMAX, Activation.TANH, Activation.TANH};
        Activation[] pzxAfns = {Activation.IDENTITY, Activation.TANH, Activation.IDENTITY, Activation.TANH};
        Activation[] pxzAfns = {Activation.TANH, Activation.IDENTITY, Activation.TANH, Activation.TANH};

        //use l2vals[i] with l1vals[i]
        double[] l2vals = {0.4, 0.0, 0.4, 0.4};
        double[] l1vals = {0.0, 0.0, 0.5, 0.0};
        double[] biasL2 = {0.0, 0.0, 0.0, 0.2};
        double[] biasL1 = {0.0, 0.0, 0.6, 0.0};

        int[][] encoderLayerSizes = new int[][] {{5}, {5}, {5, 6}, {5, 6}};
        int[][] decoderLayerSizes = new int[][] {{6}, {7, 8}, {6}, {7, 8}};

        int[] minibatches = new int[]{1,5,4,3};

        Nd4j.getRandom().setSeed(12345);
        for( int i=0; i<activFns.length; i++ ){
            LossFunction lf = lossFunctions[i];
            Activation outputActivation = outputActivations[i];
            double l2 = l2vals[i];
            double l1 = l1vals[i];
            int[] encoderSizes = encoderLayerSizes[i];
            int[] decoderSizes = decoderLayerSizes[i];
            int minibatch = minibatches[i];
            INDArray input = Nd4j.rand(minibatch, 4);
            INDArray labels = Nd4j.create(minibatch, 3);
            for (int j = 0; j < minibatch; j++) {
                labels.putScalar(j, j % 3, 1.0);
            }
            Activation afn = activFns[i];
            Activation pzxAfn = pzxAfns[i];
            Activation pxzAfn = pxzAfns[i];

            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().l2(l2)
                    .l1(l1).l2Bias(biasL2[i]).l1Bias(biasL1[i]).updater(new NoOp())
                    .seed(12345L).weightInit(WeightInit.XAVIER).list()
                    .layer(0, new VariationalAutoencoder.Builder().nIn(4).nOut(3)
                            .encoderLayerSizes(encoderSizes).decoderLayerSizes(decoderSizes)
                            .pzxActivationFunction(pzxAfn)
                            .reconstructionDistribution(
                                    new GaussianReconstructionDistribution(pxzAfn))
                            .activation(afn).build())
                    .pretrain(true).backprop(false).build();

            MultiLayerNetwork mln = new MultiLayerNetwork(conf);
            mln.init();
            mln.initGradientsView();

            org.deeplearning4j.nn.api.Layer layer = mln.getLayer(0);

            String msg = "testVaePretrain() - activationFn=" + afn + ", p(z|x) afn = " + pzxAfn
                    + ", p(x|z) afn = " + pxzAfn + ", encLayerSizes = " + Arrays.toString(encoderSizes)
                    + ", decLayerSizes = " + Arrays.toString(decoderSizes) + ", l2=" + l2 + ", l1="
                    + l1;
            if (PRINT_RESULTS) {
                System.out.println(msg);
                for (int l = 0; l < mln.getnLayers(); l++)
                    System.out.println("Layer " + l + " # params: " + mln.getLayer(l).numParams());
            }

            boolean gradOK = GradientCheckUtil.checkGradientsPretrainLayer(layer, DEFAULT_EPS,
                    DEFAULT_MAX_REL_ERROR, DEFAULT_MIN_ABS_ERROR, PRINT_RESULTS,
                    RETURN_ON_FIRST_FAILURE, input, 12345);

            assertTrue(msg, gradOK);
            TestUtils.testModelSerialization(mln);
        }
    }

    @Test
    public void testVaePretrainReconstructionDistributions() {

        int inOutSize = 6;

        ReconstructionDistribution[] reconstructionDistributions =
                new ReconstructionDistribution[]{new GaussianReconstructionDistribution(Activation.IDENTITY),
                        new GaussianReconstructionDistribution(Activation.TANH),
                        new BernoulliReconstructionDistribution(Activation.SIGMOID),
                        new CompositeReconstructionDistribution.Builder()
                                .addDistribution(2,
                                        new GaussianReconstructionDistribution(
                                                Activation.IDENTITY))
                                .addDistribution(2, new BernoulliReconstructionDistribution())
                                .addDistribution(2,
                                        new GaussianReconstructionDistribution(
                                                Activation.TANH))
                                .build(),
                        new ExponentialReconstructionDistribution(Activation.TANH),
                        new LossFunctionWrapper(new ActivationTanH(), new LossMSE())};

        Nd4j.getRandom().setSeed(12345);
        for (int i = 0; i < reconstructionDistributions.length; i++) {
            int minibatch = (i % 2 == 0 ? 1 : 3);

            INDArray data;
            switch (i) {
                case 0: //Gaussian + identity
                case 1: //Gaussian + tanh
                    data = Nd4j.rand(minibatch, inOutSize);
                    break;
                case 2: //Bernoulli
                    data = Nd4j.create(minibatch, inOutSize);
                    Nd4j.getExecutioner().exec(new BernoulliDistribution(data, 0.5), Nd4j.getRandom());
                    break;
                case 3: //Composite
                    data = Nd4j.create(minibatch, inOutSize);
                    data.get(NDArrayIndex.all(), NDArrayIndex.interval(0, 2)).assign(Nd4j.rand(minibatch, 2));
                    Nd4j.getExecutioner()
                            .exec(new BernoulliDistribution(
                                            data.get(NDArrayIndex.all(), NDArrayIndex.interval(2, 4)), 0.5),
                                    Nd4j.getRandom());
                    data.get(NDArrayIndex.all(), NDArrayIndex.interval(4, 6)).assign(Nd4j.rand(minibatch, 2));
                    break;
                case 4:
                case 5:
                    data = Nd4j.rand(minibatch, inOutSize);
                    break;
                default:
                    throw new RuntimeException();
            }

            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().l2(0.2).l1(0.3)
                    .updater(new NoOp())
                    .seed(12345L).weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0, 1))
                    .list().layer(0,
                            new VariationalAutoencoder.Builder().nIn(inOutSize).nOut(3)
                                    .encoderLayerSizes(5).decoderLayerSizes(6)
                                    .pzxActivationFunction(Activation.TANH)
                                    .reconstructionDistribution(
                                            reconstructionDistributions[i])
                                    .activation(Activation.TANH)
                                    .build())
                    .pretrain(true).backprop(false).build();

            MultiLayerNetwork mln = new MultiLayerNetwork(conf);
            mln.init();
            mln.initGradientsView();

            org.deeplearning4j.nn.api.Layer layer = mln.getLayer(0);

            String msg = "testVaePretrainReconstructionDistributions() - " + reconstructionDistributions[i];
            if (PRINT_RESULTS) {
                System.out.println(msg);
                for (int j = 0; j < mln.getnLayers(); j++)
                    System.out.println("Layer " + j + " # params: " + mln.getLayer(j).numParams());
            }

            boolean gradOK = GradientCheckUtil.checkGradientsPretrainLayer(layer, DEFAULT_EPS,
                    DEFAULT_MAX_REL_ERROR, DEFAULT_MIN_ABS_ERROR, PRINT_RESULTS, RETURN_ON_FIRST_FAILURE,
                    data, 12345);

            assertTrue(msg, gradOK);
            TestUtils.testModelSerialization(mln);
        }
    }

    @Test
    public void testVaePretrainMultipleSamples() {

        int minibatch = 2;
        Nd4j.getRandom().setSeed(12345);
        for (int numSamples : new int[]{1, 3}) {

            //            for (int numSamples : new int[]{10}) {
            INDArray features = Nd4j.rand(minibatch, 4);

            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().l2(0.2).l1(0.3)
                    .updater(new NoOp())
                    .seed(12345L).weightInit(WeightInit.XAVIER).list()
                    .layer(0, new VariationalAutoencoder.Builder().nIn(4).nOut(3).encoderLayerSizes(5, 6)
                            .decoderLayerSizes(7, 8).pzxActivationFunction(Activation.TANH)
                            .reconstructionDistribution(
                                    new GaussianReconstructionDistribution(Activation.TANH))
                            .numSamples(numSamples).activation(Activation.TANH)
                            .build())
                    .pretrain(true).backprop(false).build();

            MultiLayerNetwork mln = new MultiLayerNetwork(conf);
            mln.init();
            mln.initGradientsView();

            org.deeplearning4j.nn.api.Layer layer = mln.getLayer(0);

            String msg = "testVaePretrainMultipleSamples() - numSamples = " + numSamples;
            if (PRINT_RESULTS) {
                System.out.println(msg);
                for (int j = 0; j < mln.getnLayers(); j++)
                    System.out.println("Layer " + j + " # params: " + mln.getLayer(j).numParams());
            }

            boolean gradOK = GradientCheckUtil.checkGradientsPretrainLayer(layer, DEFAULT_EPS,
                    DEFAULT_MAX_REL_ERROR, DEFAULT_MIN_ABS_ERROR, PRINT_RESULTS, RETURN_ON_FIRST_FAILURE,
                    features, 12345);

            assertTrue(msg, gradOK);
            TestUtils.testModelSerialization(mln);
        }
    }
}
