/*-
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.deeplearning4j.nn.conf;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.stepfunctions.DefaultStepFunction;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.ConvexOptimizer;
import org.deeplearning4j.optimize.solvers.StochasticGradientDescent;
import org.deeplearning4j.optimize.stepfunctions.NegativeDefaultStepFunction;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.LeakyReLU;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.HashMap;

import static org.junit.Assert.*;

/**
 * Created by agibsonccc on 11/27/14.
 */
public class NeuralNetConfigurationTest {

    final DataSet trainingSet = createData();

    public DataSet createData() {
        int numFeatures = 40;

        INDArray input = Nd4j.create(2, numFeatures); // have to be at least two or else output layer gradient is a scalar and cause exception
        INDArray labels = Nd4j.create(2, 2);

        INDArray row0 = Nd4j.create(1, numFeatures);
        row0.assign(0.1);
        input.putRow(0, row0);
        labels.put(0, 1, 1); // set the 4th column

        INDArray row1 = Nd4j.create(1, numFeatures);
        row1.assign(0.2);

        input.putRow(1, row1);
        labels.put(1, 0, 1); // set the 2nd column

        return new DataSet(input, labels);
    }


    @Test
    public void testJson() {
        NeuralNetConfiguration conf = getRBMConfig(1, 1, WeightInit.XAVIER, true);
        String json = conf.toJson();
        NeuralNetConfiguration read = NeuralNetConfiguration.fromJson(json);

        assertEquals(conf, read);
    }


    @Test
    public void testYaml() {
        NeuralNetConfiguration conf = getRBMConfig(1, 1, WeightInit.XAVIER, true);
        String json = conf.toYaml();
        NeuralNetConfiguration read = NeuralNetConfiguration.fromYaml(json);

        assertEquals(conf, read);
    }

    @Test
    public void testClone() {
        NeuralNetConfiguration conf = getRBMConfig(1, 1, WeightInit.UNIFORM, true);
        BaseLayer bl = (BaseLayer) conf.getLayer();
        bl.setMomentumSchedule(new HashMap<Integer, Double>());
        conf.setStepFunction(new DefaultStepFunction());

        NeuralNetConfiguration conf2 = conf.clone();

        assertEquals(conf, conf2);
        assertNotSame(conf, conf2);
        assertNotSame(bl.getMomentumSchedule(), ((BaseLayer) conf2.getLayer()).getMomentumSchedule());
        assertNotSame(conf.getLayer(), conf2.getLayer());
        assertNotSame(bl.getDist(), ((BaseLayer) conf2.getLayer()).getDist());
        assertNotSame(conf.getStepFunction(), conf2.getStepFunction());
    }

    @Test
    public void testRNG() {
        RBM layer = new RBM.Builder().nIn(trainingSet.numInputs()).nOut(trainingSet.numOutcomes())
                        .weightInit(WeightInit.UNIFORM).visibleUnit(RBM.VisibleUnit.GAUSSIAN)
                        .hiddenUnit(RBM.HiddenUnit.RECTIFIED).activation(Activation.TANH)
                        .lossFunction(LossFunctions.LossFunction.RMSE_XENT).build();

        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder().seed(123).iterations(3)
                        .optimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT).layer(layer).build();

        int numParams = conf.getLayer().initializer().numParams(conf);
        INDArray params = Nd4j.create(1, numParams);
        Layer model = conf.getLayer().instantiate(conf, null, 0, params, true);
        INDArray modelWeights = model.getParam(DefaultParamInitializer.WEIGHT_KEY);


        RBM layer2 = new RBM.Builder().nIn(trainingSet.numInputs()).nOut(trainingSet.numOutcomes())
                        .weightInit(WeightInit.UNIFORM).visibleUnit(RBM.VisibleUnit.GAUSSIAN)
                        .hiddenUnit(RBM.HiddenUnit.RECTIFIED).activation(Activation.TANH)
                        .lossFunction(LossFunctions.LossFunction.RMSE_XENT).build();
        NeuralNetConfiguration conf2 = new NeuralNetConfiguration.Builder().seed(123).iterations(3)
                        .optimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT).layer(layer2).build();

        int numParams2 = conf2.getLayer().initializer().numParams(conf);
        INDArray params2 = Nd4j.create(1, numParams);
        Layer model2 = conf2.getLayer().instantiate(conf2, null, 0, params2, true);
        INDArray modelWeights2 = model2.getParam(DefaultParamInitializer.WEIGHT_KEY);

        assertEquals(modelWeights, modelWeights2);
    }

    @Test
    public void testSetSeedSize() {
        Nd4j.getRandom().setSeed(123);

        Layer model = getRBMLayer(trainingSet.numInputs(), trainingSet.numOutcomes(), WeightInit.XAVIER, true);
        INDArray modelWeights = model.getParam(DefaultParamInitializer.WEIGHT_KEY);
        Nd4j.getRandom().setSeed(123);

        Layer model2 = getRBMLayer(trainingSet.numInputs(), trainingSet.numOutcomes(), WeightInit.XAVIER, true);
        INDArray modelWeights2 = model2.getParam(DefaultParamInitializer.WEIGHT_KEY);
        assertEquals(modelWeights, modelWeights2);
    }


    @Test
    public void testSetSeedNormalized() {
        Nd4j.getRandom().setSeed(123);

        Layer model = getRBMLayer(trainingSet.numInputs(), trainingSet.numOutcomes(), WeightInit.XAVIER, true);
        INDArray modelWeights = model.getParam(DefaultParamInitializer.WEIGHT_KEY);
        Nd4j.getRandom().setSeed(123);

        Layer model2 = getRBMLayer(trainingSet.numInputs(), trainingSet.numOutcomes(), WeightInit.XAVIER, true);
        INDArray modelWeights2 = model2.getParam(DefaultParamInitializer.WEIGHT_KEY);
        assertEquals(modelWeights, modelWeights2);
    }

    @Test
    public void testSetSeedXavier() {
        Nd4j.getRandom().setSeed(123);

        Layer model = getRBMLayer(trainingSet.numInputs(), trainingSet.numOutcomes(), WeightInit.UNIFORM, true);
        INDArray modelWeights = model.getParam(DefaultParamInitializer.WEIGHT_KEY);
        Nd4j.getRandom().setSeed(123);

        Layer model2 = getRBMLayer(trainingSet.numInputs(), trainingSet.numOutcomes(), WeightInit.UNIFORM, true);
        INDArray modelWeights2 = model2.getParam(DefaultParamInitializer.WEIGHT_KEY);

        assertEquals(modelWeights, modelWeights2);
    }

    @Test
    public void testSetSeedDistribution() {
        Nd4j.getRandom().setSeed(123);

        Layer model = getRBMLayer(trainingSet.numInputs(), trainingSet.numOutcomes(), WeightInit.DISTRIBUTION, true);
        INDArray modelWeights = model.getParam(DefaultParamInitializer.WEIGHT_KEY);
        Nd4j.getRandom().setSeed(123);

        Layer model2 = getRBMLayer(trainingSet.numInputs(), trainingSet.numOutcomes(), WeightInit.DISTRIBUTION, true);
        INDArray modelWeights2 = model2.getParam(DefaultParamInitializer.WEIGHT_KEY);

        assertEquals(modelWeights, modelWeights2);
    }

    @Test
    public void testPretrain() {
        Layer model = getRBMLayer(trainingSet.numInputs(), trainingSet.numOutcomes(), WeightInit.UNIFORM, true);

        Layer model2 = getRBMLayer(trainingSet.numInputs(), trainingSet.numOutcomes(), WeightInit.UNIFORM, false);

        assertNotEquals(model.conf().isPretrain(), model2.conf().isPretrain());
    }


    private static NeuralNetConfiguration getRBMConfig(int nIn, int nOut, WeightInit weightInit, boolean pretrain) {
        RBM layer = new RBM.Builder().nIn(nIn).nOut(nOut).weightInit(weightInit).dist(new NormalDistribution(1, 1))
                        .visibleUnit(RBM.VisibleUnit.GAUSSIAN).hiddenUnit(RBM.HiddenUnit.RECTIFIED)
                        .activation(Activation.TANH).lossFunction(LossFunctions.LossFunction.KL_DIVERGENCE).build();

        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder().iterations(3)
                        .optimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT).regularization(false).layer(layer)
                        .build();
        conf.setPretrain(pretrain);
        return conf;
    }

    private static Layer getRBMLayer(int nIn, int nOut, WeightInit weightInit, boolean preTrain) {
        NeuralNetConfiguration conf = getRBMConfig(nIn, nOut, weightInit, preTrain);
        int numParams = conf.getLayer().initializer().numParams(conf);
        INDArray params = Nd4j.create(1, numParams);
        return conf.getLayer().instantiate(conf, null, 0, params, true);
    }


    @Test
    public void testLearningRateByParam() {
        double lr = 0.01;
        double biasLr = 0.02;
        int[] nIns = {4, 3, 3};
        int[] nOuts = {3, 3, 3};
        int oldScore = 1;
        int newScore = 1;
        int iteration = 3;
        INDArray gradientW = Nd4j.ones(nIns[0], nOuts[0]);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().learningRate(0.3).list()
                        .layer(0, new DenseLayer.Builder().nIn(nIns[0]).nOut(nOuts[0])
                                        .updater(org.deeplearning4j.nn.conf.Updater.SGD).learningRate(lr)
                                        .biasLearningRate(biasLr).build())
                        .layer(1, new BatchNormalization.Builder().nIn(nIns[1]).nOut(nOuts[1]).learningRate(0.7)
                                        .build())
                        .layer(2, new OutputLayer.Builder().nIn(nIns[2]).nOut(nOuts[2])
                                        .updater(org.deeplearning4j.nn.conf.Updater.SGD).build())
                        .backprop(true).pretrain(false).build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        ConvexOptimizer opt = new StochasticGradientDescent(net.getDefaultConfiguration(),
                        new NegativeDefaultStepFunction(), null, net);
        opt.checkTerminalConditions(gradientW, oldScore, newScore, iteration);
        assertEquals(lr, ((Sgd)net.getLayer(0).conf().getLayer().getIUpdaterByParam("W")).getLearningRate(), 1e-4);
        assertEquals(biasLr, ((Sgd)net.getLayer(0).conf().getLayer().getIUpdaterByParam("b")).getLearningRate(), 1e-4);
        assertEquals(0.7, ((Sgd)net.getLayer(1).conf().getLayer().getIUpdaterByParam("gamma")).getLearningRate(), 1e-4);
        assertEquals(0.3, ((Sgd)net.getLayer(2).conf().getLayer().getIUpdaterByParam("W")).getLearningRate(), 1e-4); //From global LR
        assertEquals(0.3, ((Sgd)net.getLayer(2).conf().getLayer().getIUpdaterByParam("W")).getLearningRate(), 1e-4); //From global LR
    }

    @Test
    public void testLeakyreluAlpha() {
        //FIXME: Make more generic to use neuralnetconfs
        int sizeX = 4;
        int scaleX = 10;
        System.out.println("Here is a leaky vector..");
        INDArray leakyVector = Nd4j.linspace(-1, 1, sizeX);
        leakyVector = leakyVector.mul(scaleX);
        System.out.println(leakyVector);


        double myAlpha = 0.5;
        System.out.println("======================");
        System.out.println("Exec and Return: Leaky Relu transformation with alpha = 0.5 ..");
        System.out.println("======================");
        INDArray outDef = Nd4j.getExecutioner().execAndReturn(new LeakyReLU(leakyVector.dup(), myAlpha));
        System.out.println(outDef);

        String confActivation = "leakyrelu";
        Object[] confExtra = {myAlpha};
        INDArray outMine = Nd4j.getExecutioner().execAndReturn(
                        Nd4j.getOpFactory().createTransform(confActivation, leakyVector.dup(), confExtra));
        System.out.println("======================");
        System.out.println("Exec and Return: Leaky Relu transformation with a value via getOpFactory");
        System.out.println("======================");
        System.out.println(outMine);

        //Test equality for ndarray elementwise
        //assertArrayEquals(..)
    }

    @Test
    public void testL1L2ByParam() {
        double l1 = 0.01;
        double l2 = 0.07;
        int[] nIns = {4, 3, 3};
        int[] nOuts = {3, 3, 3};
        int oldScore = 1;
        int newScore = 1;
        int iteration = 3;
        INDArray gradientW = Nd4j.ones(nIns[0], nOuts[0]);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().learningRate(8).regularization(true).l1(l1)
                        .l2(l2).list()
                        .layer(0, new DenseLayer.Builder().nIn(nIns[0]).nOut(nOuts[0])
                                        .updater(org.deeplearning4j.nn.conf.Updater.SGD).build())
                        .layer(1, new BatchNormalization.Builder().nIn(nIns[1]).nOut(nOuts[1]).l2(0.5).build())
                        .layer(2, new OutputLayer.Builder().nIn(nIns[2]).nOut(nOuts[2])
                                        .updater(org.deeplearning4j.nn.conf.Updater.SGD).build())
                        .backprop(true).pretrain(false).build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        ConvexOptimizer opt = new StochasticGradientDescent(net.getDefaultConfiguration(),
                        new NegativeDefaultStepFunction(), null, net);
        opt.checkTerminalConditions(gradientW, oldScore, newScore, iteration);
        assertEquals(l1, net.getLayer(0).conf().getL1ByParam("W"), 1e-4);
        assertEquals(0.0, net.getLayer(0).conf().getL1ByParam("b"), 0.0);
        assertEquals(0.0, net.getLayer(1).conf().getL2ByParam("beta"), 0.0);
        assertEquals(0.0, net.getLayer(1).conf().getL2ByParam("gamma"), 0.0);
        assertEquals(0.0, net.getLayer(1).conf().getL2ByParam("mean"), 0.0);
        assertEquals(0.0, net.getLayer(1).conf().getL2ByParam("var"), 0.0);
        assertEquals(l2, net.getLayer(2).conf().getL2ByParam("W"), 1e-4);
        assertEquals(0.0, net.getLayer(2).conf().getL2ByParam("b"), 0.0);
    }

    @Test
    public void testLayerPretrainConfig() {
        boolean pretrain = true;

        org.deeplearning4j.nn.conf.layers.RBM layer =
                        new org.deeplearning4j.nn.conf.layers.RBM.Builder(RBM.HiddenUnit.BINARY, RBM.VisibleUnit.BINARY)
                                        .nIn(10).nOut(5).learningRate(1e-1f)
                                        .lossFunction(LossFunctions.LossFunction.KL_DIVERGENCE).build();

        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder().iterations(1).seed(42).layer(layer).build();

        assertFalse(conf.isPretrain());
        conf.setPretrain(pretrain);
        assertTrue(conf.isPretrain());
    }

}
