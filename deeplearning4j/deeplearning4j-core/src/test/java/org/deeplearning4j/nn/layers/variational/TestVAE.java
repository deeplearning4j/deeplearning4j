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

package org.deeplearning4j.nn.layers.variational;

import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.variational.*;
import org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.impl.ActivationTanH;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.random.impl.BernoulliDistribution;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.lossfunctions.impl.LossMAE;
import org.nd4j.linalg.lossfunctions.impl.LossMSE;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;

import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.*;

/**
 * Created by Alex on 26/11/2016.
 */
public class TestVAE extends BaseDL4JTest {

    @Test
    public void testInitialization() {

        MultiLayerConfiguration mlc =
                        new NeuralNetConfiguration.Builder().list()
                                        .layer(0, new org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder.Builder()
                                                        .nIn(10).nOut(5).encoderLayerSizes(12).decoderLayerSizes(13)
                                                        .build())
                                        .build();

        NeuralNetConfiguration c = mlc.getConf(0);
        org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder vae =
                        (org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder) c.getLayer();

        long allParams = vae.initializer().numParams(c);

        //                  Encoder         Encoder -> p(z|x)       Decoder         //p(x|z)
        int expNumParams = (10 * 12 + 12) + (12 * (2 * 5) + (2 * 5)) + (5 * 13 + 13) + (13 * (2 * 10) + (2 * 10));
        assertEquals(expNumParams, allParams);

        MultiLayerNetwork net = new MultiLayerNetwork(mlc);
        net.init();

        System.out.println("Exp num params: " + expNumParams);
        assertEquals(expNumParams, net.getLayer(0).params().length());
        Map<String, INDArray> paramTable = net.getLayer(0).paramTable();
        int count = 0;
        for (INDArray arr : paramTable.values()) {
            count += arr.length();
        }
        assertEquals(expNumParams, count);

        assertEquals(expNumParams, net.getLayer(0).numParams());
    }

    @Test
    public void testForwardPass() {

        int[][] encLayerSizes = new int[][] {{12}, {12, 13}, {12, 13, 14}};
        for (int i = 0; i < encLayerSizes.length; i++) {

            MultiLayerConfiguration mlc = new NeuralNetConfiguration.Builder().list().layer(0,
                            new org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder.Builder().nIn(10)
                                            .nOut(5).encoderLayerSizes(encLayerSizes[i]).decoderLayerSizes(13).build())
                            .build();

            NeuralNetConfiguration c = mlc.getConf(0);
            org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder vae =
                            (org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder) c.getLayer();

            MultiLayerNetwork net = new MultiLayerNetwork(mlc);
            net.init();

            INDArray in = Nd4j.rand(1, 10);

            //        net.output(in);
            List<INDArray> out = net.feedForward(in);
            assertArrayEquals(new long[] {1, 10}, out.get(0).shape());
            assertArrayEquals(new long[] {1, 5}, out.get(1).shape());
        }
    }

    @Test
    public void testPretrainSimple() {

        int inputSize = 3;

        MultiLayerConfiguration mlc = new NeuralNetConfiguration.Builder().list()
                        .layer(0, new org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder.Builder()
                                        .nIn(inputSize).nOut(4).encoderLayerSizes(5).decoderLayerSizes(6).build())
                        .pretrain(true).backprop(false).build();

        NeuralNetConfiguration c = mlc.getConf(0);
        org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder vae =
                        (org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder) c.getLayer();

        long allParams = vae.initializer().numParams(c);

        MultiLayerNetwork net = new MultiLayerNetwork(mlc);
        net.init();
        net.initGradientsView(); //TODO this should happen automatically

        Map<String, INDArray> paramTable = net.getLayer(0).paramTable();
        Map<String, INDArray> gradTable =
                        ((org.deeplearning4j.nn.layers.variational.VariationalAutoencoder) net.getLayer(0))
                                        .getGradientViews();

        assertEquals(paramTable.keySet(), gradTable.keySet());
        for (String s : paramTable.keySet()) {
            assertEquals(paramTable.get(s).length(), gradTable.get(s).length());
            assertArrayEquals(paramTable.get(s).shape(), gradTable.get(s).shape());
        }

        System.out.println("Num params: " + net.numParams());

        INDArray data = Nd4j.rand(1, inputSize);


        net.pretrainLayer(0, data);
    }


    @Test
    public void testParamGradientOrderAndViews() {
        Nd4j.getRandom().setSeed(12345);
        MultiLayerConfiguration mlc = new NeuralNetConfiguration.Builder().list()
                        .layer(0, new org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder.Builder()
                                        .nIn(10).nOut(5).encoderLayerSizes(12, 13).decoderLayerSizes(14, 15).build())
                        .build();

        NeuralNetConfiguration c = mlc.getConf(0);
        org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder vae =
                        (org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder) c.getLayer();

        MultiLayerNetwork net = new MultiLayerNetwork(mlc);
        net.init();

        net.initGradientsView();

        org.deeplearning4j.nn.layers.variational.VariationalAutoencoder layer =
                        (org.deeplearning4j.nn.layers.variational.VariationalAutoencoder) net.getLayer(0);

        Map<String, INDArray> layerParams = layer.paramTable();
        Map<String, INDArray> layerGradViews = layer.getGradientViews();

        layer.setInput(Nd4j.rand(3, 10), LayerWorkspaceMgr.noWorkspaces());
        layer.computeGradientAndScore(LayerWorkspaceMgr.noWorkspaces());
        Gradient g = layer.gradient();
        Map<String, INDArray> grads = g.gradientForVariable();

        assertEquals(layerParams.size(), layerGradViews.size());
        assertEquals(layerParams.size(), grads.size());

        //Iteration order should be consistent due to linked hashmaps
        Iterator<String> pIter = layerParams.keySet().iterator();
        Iterator<String> gvIter = layerGradViews.keySet().iterator();
        Iterator<String> gIter = grads.keySet().iterator();

        while (pIter.hasNext()) {
            String p = pIter.next();
            String gv = gvIter.next();
            String gr = gIter.next();

            //            System.out.println(p + "\t" + gv + "\t" + gr);

            assertEquals(p, gv);
            assertEquals(p, gr);

            INDArray pArr = layerParams.get(p);
            INDArray gvArr = layerGradViews.get(p);
            INDArray gArr = grads.get(p);

            assertArrayEquals(pArr.shape(), gvArr.shape());
            assertTrue(gvArr == gArr); //Should be the exact same object due to view mechanics
        }
    }


    @Test
    public void testPretrainParamsDuringBackprop() {
        //Idea: pretrain-specific parameters shouldn't change during backprop

        Nd4j.getRandom().setSeed(12345);
        MultiLayerConfiguration mlc = new NeuralNetConfiguration.Builder().seed(12345).list()
                        .layer(0, new org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder.Builder()
                                        .nIn(10).nOut(5).encoderLayerSizes(12, 13).decoderLayerSizes(14, 15).build())
                        .layer(1, new OutputLayer.Builder().lossFunction(LossFunctions.LossFunction.MSE).nIn(5).nOut(6)
                                        .activation(new ActivationTanH()).build())
                        .pretrain(true).backprop(true).build();

        NeuralNetConfiguration c = mlc.getConf(0);
        org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder vae =
                        (org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder) c.getLayer();

        MultiLayerNetwork net = new MultiLayerNetwork(mlc);
        net.init();

        net.initGradientsView();

        org.deeplearning4j.nn.layers.variational.VariationalAutoencoder layer =
                        (org.deeplearning4j.nn.layers.variational.VariationalAutoencoder) net.getLayer(0);

        INDArray input = Nd4j.rand(3, 10);
        net.pretrainLayer(0, input);

        //Get a snapshot of the pretrain params after fitting:
        Map<String, INDArray> layerParams = layer.paramTable();
        Map<String, INDArray> pretrainParamsBefore = new HashMap<>();
        for (String s : layerParams.keySet()) {
            if (layer.isPretrainParam(s)) {
                pretrainParamsBefore.put(s, layerParams.get(s).dup());
            }
        }


        INDArray features = Nd4j.rand(3, 10);
        INDArray labels = Nd4j.rand(3, 6);

        net.getLayerWiseConfigurations().setPretrain(false);

        for (int i = 0; i < 3; i++) {
            net.fit(features, labels);
        }

        Map<String, INDArray> layerParamsAfter = layer.paramTable();

        for (String s : pretrainParamsBefore.keySet()) {
            INDArray before = pretrainParamsBefore.get(s);
            INDArray after = layerParamsAfter.get(s);
            assertEquals(before, after);
        }
    }


    @Test
    public void testJsonYaml() {

        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder().seed(12345).list()
                        .layer(0, new org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder.Builder()
                                        .reconstructionDistribution(new GaussianReconstructionDistribution(Activation.IDENTITY))
                                        .nIn(3).nOut(4).encoderLayerSizes(5).decoderLayerSizes(6).build())
                        .layer(1, new org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder.Builder()
                                        .reconstructionDistribution(new GaussianReconstructionDistribution(Activation.TANH))
                                        .nIn(7).nOut(8).encoderLayerSizes(9).decoderLayerSizes(10).build())
                        .layer(2, new org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder.Builder()
                                        .reconstructionDistribution(new BernoulliReconstructionDistribution()).nIn(11)
                                        .nOut(12).encoderLayerSizes(13).decoderLayerSizes(14).build())
                        .layer(3, new org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder.Builder()
                                        .reconstructionDistribution(new ExponentialReconstructionDistribution(Activation.TANH))
                                        .nIn(11).nOut(12).encoderLayerSizes(13).decoderLayerSizes(14).build())
                        .layer(4, new org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder.Builder()
                                        .lossFunction(new ActivationTanH(), LossFunctions.LossFunction.MSE).nIn(11)
                                        .nOut(12).encoderLayerSizes(13).decoderLayerSizes(14).build())
                        .layer(5, new org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder.Builder()
                                        .reconstructionDistribution(new CompositeReconstructionDistribution.Builder()
                                                        .addDistribution(5, new GaussianReconstructionDistribution())
                                                        .addDistribution(5,
                                                                        new GaussianReconstructionDistribution(Activation.TANH))
                                                        .addDistribution(5, new BernoulliReconstructionDistribution())
                                                        .build())
                                        .nIn(15).nOut(16).encoderLayerSizes(17).decoderLayerSizes(18).build())
                        .layer(1, new OutputLayer.Builder().lossFunction(LossFunctions.LossFunction.MSE).nIn(18)
                                        .nOut(19).activation(new ActivationTanH()).build())
                        .pretrain(true).backprop(true).build();

        String asJson = config.toJson();
        String asYaml = config.toYaml();

        MultiLayerConfiguration fromJson = MultiLayerConfiguration.fromJson(asJson);
        MultiLayerConfiguration fromYaml = MultiLayerConfiguration.fromYaml(asYaml);

        assertEquals(config, fromJson);
        assertEquals(config, fromYaml);
    }

    @Test
    public void testReconstructionDistributionsSimple() {

        int inOutSize = 6;

        ReconstructionDistribution[] reconstructionDistributions =
                        new ReconstructionDistribution[] {new GaussianReconstructionDistribution(Activation.IDENTITY),
                                        new GaussianReconstructionDistribution(Activation.TANH),
                                        new BernoulliReconstructionDistribution(Activation.SIGMOID),
                                        new CompositeReconstructionDistribution.Builder()
                                                        .addDistribution(2,
                                                                        new GaussianReconstructionDistribution(
                                                                                        Activation.IDENTITY))
                                                        .addDistribution(2, new BernoulliReconstructionDistribution())
                                                        .addDistribution(2, new GaussianReconstructionDistribution(
                                                                        Activation.TANH))
                                                        .build()};

        Nd4j.getRandom().setSeed(12345);
        for (int minibatch : new int[] {1, 5}) {
            for (int i = 0; i < reconstructionDistributions.length; i++) {
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
                    default:
                        throw new RuntimeException();
                }

                MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().l2(0.2).l1(0.3)
                                .updater(new Sgd(1.0))
                                .seed(12345L).weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0, 1))
                                .list().layer(0,
                                                new VariationalAutoencoder.Builder().nIn(inOutSize).nOut(3)
                                                                .encoderLayerSizes(5).decoderLayerSizes(6)
                                                                .pzxActivationFunction(Activation.TANH)
                                                                .reconstructionDistribution(
                                                                                reconstructionDistributions[i])
                                                                .activation(new ActivationTanH())
                                                                .build())
                                .pretrain(true).backprop(false).build();

                MultiLayerNetwork mln = new MultiLayerNetwork(conf);
                mln.init();
                mln.initGradientsView();
                mln.pretrainLayer(0, data);

                org.deeplearning4j.nn.layers.variational.VariationalAutoencoder layer =
                                (org.deeplearning4j.nn.layers.variational.VariationalAutoencoder) mln.getLayer(0);
                assertFalse(layer.hasLossFunction());

                Nd4j.getRandom().setSeed(12345);
                INDArray reconstructionProb = layer.reconstructionProbability(data, 50);
                assertArrayEquals(new long[] {minibatch, 1}, reconstructionProb.shape());

                Nd4j.getRandom().setSeed(12345);
                INDArray reconstructionLogProb = layer.reconstructionLogProbability(data, 50);
                assertArrayEquals(new long[] {minibatch, 1}, reconstructionLogProb.shape());

                //                System.out.println(reconstructionDistributions[i]);
                for (int j = 0; j < minibatch; j++) {
                    double p = reconstructionProb.getDouble(j);
                    double logp = reconstructionLogProb.getDouble(j);
                    assertTrue(p >= 0.0 && p <= 1.0);
                    assertTrue(logp <= 0.0);

                    double pFromLogP = Math.exp(logp);
                    assertEquals(p, pFromLogP, 1e-6);
                }
            }
        }
    }


    @Test
    public void testReconstructionErrorSimple() {

        int inOutSize = 6;

        ReconstructionDistribution[] reconstructionDistributions =
                        new ReconstructionDistribution[] {new LossFunctionWrapper(Activation.TANH, new LossMSE()),
                                        new LossFunctionWrapper(Activation.IDENTITY, new LossMAE()),
                                        new CompositeReconstructionDistribution.Builder()
                                                        .addDistribution(3,
                                                                        new LossFunctionWrapper(Activation.TANH,
                                                                                        new LossMSE()))
                                                        .addDistribution(3, new LossFunctionWrapper(Activation.IDENTITY,
                                                                        new LossMAE()))
                                                        .build()};

        Nd4j.getRandom().setSeed(12345);
        for (int minibatch : new int[] {1, 5}) {
            for (int i = 0; i < reconstructionDistributions.length; i++) {
                INDArray data = Nd4j.rand(minibatch, inOutSize).muli(2).subi(1);

                MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().l2(0.2).l1(0.3)
                                .updater(new Sgd(1.0))
                                .seed(12345L).weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0, 1))
                                .list().layer(0,
                                                new VariationalAutoencoder.Builder().nIn(inOutSize).nOut(3)
                                                                .encoderLayerSizes(5).decoderLayerSizes(6)
                                                                .pzxActivationFunction(Activation.TANH)
                                                                .reconstructionDistribution(
                                                                                reconstructionDistributions[i])
                                                                .activation(new ActivationTanH())
                                                                .build())
                                .pretrain(true).backprop(false).build();

                MultiLayerNetwork mln = new MultiLayerNetwork(conf);
                mln.init();
                mln.initGradientsView();
                mln.pretrainLayer(0, data);

                org.deeplearning4j.nn.layers.variational.VariationalAutoencoder layer =
                                (org.deeplearning4j.nn.layers.variational.VariationalAutoencoder) mln.getLayer(0);
                assertTrue(layer.hasLossFunction());

                Nd4j.getRandom().setSeed(12345);
                INDArray reconstructionError = layer.reconstructionError(data);
                assertArrayEquals(new long[] {minibatch, 1}, reconstructionError.shape());

                for (int j = 0; j < minibatch; j++) {
                    double re = reconstructionError.getDouble(j);
                    assertTrue(re >= 0.0);
                }
            }
        }
    }
}
