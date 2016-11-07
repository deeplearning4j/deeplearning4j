/*
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

package org.deeplearning4j.nn.layers.feedforward.rbm;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.datasets.fetchers.IrisDataFetcher;
import org.deeplearning4j.datasets.fetchers.MnistDataFetcher;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.RBM.HiddenUnit;
import org.deeplearning4j.nn.conf.layers.RBM.VisibleUnit;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.params.PretrainParamInitializer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;


/**
 * Created by agibsonccc on 8/27/14.
 * @Author agibsoncc & nyghtowl
 */
public class RBMTests {
    private static final Logger log = LoggerFactory.getLogger(RBMTests.class);


    @Test
    public void testRBMBiasInit() {
        org.deeplearning4j.nn.conf.layers.RBM cnn = new org.deeplearning4j.nn.conf.layers.RBM.Builder()
                .nIn(1)
                .nOut(3)
                .biasInit(1)
                .build();

        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .layer(cnn)
                .build();

        int numParams = conf.getLayer().initializer().numParams(conf);
        INDArray params = Nd4j.create(1, numParams);
        Layer layer = conf.getLayer().instantiate(conf, null, 0, params, true);

        assertEquals(1, layer.getParam("b").size(0));
    }

    @Test
    public void testLfw() throws Exception {
        DataSet d = new MnistDataSetIterator(10, true, 12345).next();

        int nOut = 600;

        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .layer(new org.deeplearning4j.nn.conf.layers.RBM.Builder(org.deeplearning4j.nn.conf.layers.RBM.HiddenUnit.RECTIFIED, org.deeplearning4j.nn.conf.layers.RBM.VisibleUnit.GAUSSIAN)
                        .nIn(d.numInputs()).nOut(nOut)
                        .weightInit(WeightInit.XAVIER)
                        .lossFunction(LossFunctions.LossFunction.KL_DIVERGENCE)
                        .build())
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(1e-3f)
                .build();

        int numParams = conf.getLayer().initializer().numParams(conf);
        INDArray params = Nd4j.create(1, numParams);
        RBM rbm = (RBM) conf.getLayer().instantiate(conf, Arrays.<IterationListener>asList(new ScoreIterationListener(1)), 0, params, true);

        rbm.fit(d.getFeatureMatrix());
    }

    @Test
    public void testIrisGaussianHidden() {
        IrisDataFetcher fetcher = new IrisDataFetcher();
        fetcher.fetch(150);
        DataSet d = fetcher.next();
        d.normalizeZeroMeanZeroUnitVariance();

        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .learningRate(1e-1f)
                .layer(new org.deeplearning4j.nn.conf.layers.RBM.Builder(
                        org.deeplearning4j.nn.conf.layers.RBM.HiddenUnit.GAUSSIAN, org.deeplearning4j.nn.conf.layers.RBM.VisibleUnit.GAUSSIAN)
                        .nIn(d.numInputs()).nOut(3)
                        .lossFunction(LossFunctions.LossFunction.KL_DIVERGENCE).build())
                .build();

        int numParams = conf.getLayer().initializer().numParams(conf);
        INDArray params = Nd4j.create(1, numParams);
        RBM r = (RBM) conf.getLayer().instantiate(conf, null, 0, params, true);
        r.fit(d.getFeatureMatrix());

    }


    @Test
    public void testIris() {
        IrisDataFetcher fetcher = new IrisDataFetcher();
        fetcher.fetch(150);
        DataSet d = fetcher.next();
        d.normalizeZeroMeanZeroUnitVariance();

        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .learningRate(1e-1f)
                .layer(new org.deeplearning4j.nn.conf.layers.RBM.Builder(org.deeplearning4j.nn.conf.layers.RBM.HiddenUnit.RECTIFIED, org.deeplearning4j.nn.conf.layers.RBM.VisibleUnit.GAUSSIAN)
                        .nIn(d.numInputs()).nOut(3)
                        .lossFunction(LossFunctions.LossFunction.KL_DIVERGENCE).build())
                .build();

        int numParams = conf.getLayer().initializer().numParams(conf);
        INDArray params = Nd4j.create(1, numParams);
        RBM r = (RBM) conf.getLayer().instantiate(conf, null, 0, params, true);
        r.fit(d.getFeatureMatrix());

    }


    @Test
    public void testBasic() {
        float[][] data = new float[][]
                {
                        {1, 1, 1, 0, 0, 0},
                        {1, 0, 1, 0, 0, 0},
                        {1, 1, 1, 0, 0, 0},
                        {0, 0, 1, 1, 1, 0},
                        {0, 0, 1, 1, 0, 0},
                        {0, 0, 1, 1, 1, 0},
                        {0, 0, 1, 1, 1, 0}
                };

        INDArray input = Nd4j.create(data);

        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .learningRate(1e-1f)
                .layer(new org.deeplearning4j.nn.conf.layers.RBM.Builder()
                        .nIn(6).nOut(4)
                        .lossFunction(LossFunctions.LossFunction.KL_DIVERGENCE).build())
                .build();

        int numParams = conf.getLayer().initializer().numParams(conf);
        INDArray params = Nd4j.create(1, numParams);
        RBM rbm = (RBM) conf.getLayer().instantiate(conf, null, 0, params, true);
        rbm.fit(input);

        assertEquals(24, rbm.gradient().getGradientFor("W").length());
    }

    @Test
    public void testMnist() throws Exception {
        MnistDataFetcher fetcher = new MnistDataFetcher(true);
        Nd4j.ENFORCE_NUMERICAL_STABILITY = true;

        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .iterations(30)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(1e-1f)
                .layer(new org.deeplearning4j.nn.conf.layers.RBM.Builder()
                        .nIn(784).nOut(600)
                        .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(1, 1e-5))
                        .lossFunction(LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY)
                        .build())
                .build();

        org.deeplearning4j.nn.conf.layers.RBM layerConf =
                (org.deeplearning4j.nn.conf.layers.RBM) conf.getLayer();

        fetcher.fetch(10);
        DataSet d2 = fetcher.next();

        org.nd4j.linalg.api.rng.distribution.Distribution dist = Nd4j.getDistributions().createNormal(1, 1e-5);
        System.out.println(dist.sample(new int[]{layerConf.getNIn(), layerConf.getNOut()}));

        INDArray input = d2.getFeatureMatrix();

        int numParams = conf.getLayer().initializer().numParams(conf);
        INDArray params = Nd4j.create(1, numParams);
        RBM rbm = (RBM) conf.getLayer().instantiate(conf, null, 0, params, true);
        rbm.fit(input);

    }

    @Test
    public void testSetGetParams() {
        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .learningRate(1e-1f)
                .layer(new org.deeplearning4j.nn.conf.layers.RBM.Builder()
                        .nIn(6).nOut(4)
                        .lossFunction(LossFunctions.LossFunction.KL_DIVERGENCE).build())
                .build();

        int numParams = conf.getLayer().initializer().numParams(conf);
        INDArray params = Nd4j.create(1, numParams);
        RBM rbm = (RBM) conf.getLayer().instantiate(conf, null, 0, params, true);
        INDArray rand2 = Nd4j.rand(new int[]{1, rbm.numParams()});
        rbm.setParams(rand2);
        rbm.setInput(Nd4j.zeros(6));
        rbm.computeGradientAndScore();
        INDArray getParams = rbm.params();
        assertEquals(rand2, getParams);
    }

    @Test
    public void testCg() {
        float[][] data = new float[][]
                {
                        {1, 1, 1, 0, 0, 0},
                        {1, 0, 1, 0, 0, 0},
                        {1, 1, 1, 0, 0, 0},
                        {0, 0, 1, 1, 1, 0},
                        {0, 0, 1, 1, 0, 0},
                        {0, 0, 1, 1, 1, 0},
                        {0, 0, 1, 1, 1, 0}
                };

        INDArray input = Nd4j.create(data);

        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .learningRate(1e-1f)
                .layer(new org.deeplearning4j.nn.conf.layers.RBM.Builder()
                        .nIn(6).nOut(4)
                        .lossFunction(LossFunctions.LossFunction.KL_DIVERGENCE).build())
                .build();

        int numParams = conf.getLayer().initializer().numParams(conf);
        INDArray params = Nd4j.create(1, numParams);
        RBM rbm = (RBM) conf.getLayer().instantiate(conf, null, 0, params, true);
        double value = rbm.score();
        rbm.fit(input);
        value = rbm.score();

    }


    @Test
    public void testGradient() {
        float[][] data = new float[][]
                {
                        {1, 1, 1, 0, 0, 0},
                        {1, 0, 1, 0, 0, 0},
                        {1, 1, 1, 0, 0, 0},
                        {0, 0, 1, 1, 1, 0},
                        {0, 0, 1, 1, 0, 0},
                        {0, 0, 1, 1, 1, 0},
                        {0, 0, 1, 1, 1, 0}
                };


        INDArray input = Nd4j.create(data);

        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .learningRate(1e-1f)
                .layer(new org.deeplearning4j.nn.conf.layers.RBM.Builder()
                        .nIn(6).nOut(4)
                        .lossFunction(LossFunctions.LossFunction.KL_DIVERGENCE).build())
                .build();

        int numParams = conf.getLayer().initializer().numParams(conf);
        INDArray params = Nd4j.create(1, numParams);
        RBM rbm = (RBM) conf.getLayer().instantiate(conf, null, 0, params, true);

        rbm.fit(input);
        double value = rbm.score();

        Gradient grad2 = rbm.gradient();

    }

    @Test
    public void testPropUpDownBinary() {
        INDArray input = Nd4j.linspace(1, 10, 10);
        INDArray params = getLinParams(10, 5);

        RBM rbm = getRBMLayer(10, 5, HiddenUnit.BINARY, VisibleUnit.BINARY, params, true, false);

        INDArray actualParams = rbm.params();
        assertEquals(params, actualParams);

        // propUp
        INDArray expectedHOut = Transforms.sigmoid(Nd4j.create(new double[]{
                386., 936., 1486., 2036., 2586.
        }));

        INDArray actualHOut = rbm.propUp(input);
        assertEquals(expectedHOut, actualHOut);

        // propDown
        INDArray expectedVOut = Transforms.sigmoid(Nd4j.create(new double[]{
                106., 111., 116., 121., 126., 131., 136., 141., 146., 151.
        }));

        INDArray actualVOut = rbm.propDown(actualHOut);
        assertEquals(expectedVOut, actualVOut);

    }

    @Test
    public void testActivate(){
        INDArray input = Nd4j.linspace(1, 10, 10);
        INDArray expectedActivations = Nd4j.ones(5);
        INDArray params = getStandardParams(10, 5);
        RBM rbm = getRBMLayer(10, 5, HiddenUnit.BINARY, VisibleUnit.BINARY, params, true, true);
        rbm.setInput(input);
        INDArray actualActivations = rbm.activate();

        assertEquals(expectedActivations, actualActivations);
    }

    @Test
    public void testComputeGradientAndScore(){
        INDArray input = Nd4j.linspace(1, 10, 10);
        INDArray params = getStandardParams(10, 5);

        RBM rbm = getRBMLayer(10, 5, HiddenUnit.BINARY, VisibleUnit.BINARY, params, true, false);
        rbm.setInput(input);
        rbm.computeGradientAndScore();
        Pair<Gradient, Double> pair = rbm.gradientAndScore();

        INDArray wGradient = Nd4j.ones(1, 50, 10);
        INDArray hBiasGradient = Nd4j.zeros(1, 5, 10);
        INDArray vBiasGradient = Nd4j.zeros(1, 10, 10);

        Gradient expectedGradient = new DefaultGradient();
        expectedGradient.gradientForVariable().put(PretrainParamInitializer.WEIGHT_KEY, wGradient);
        expectedGradient.gradientForVariable().put(PretrainParamInitializer.BIAS_KEY,hBiasGradient);
        expectedGradient.gradientForVariable().put(PretrainParamInitializer.VISIBLE_BIAS_KEY,vBiasGradient);


        INDArray params3 = Nd4j.create(new double[] {0.50, 0.50, -0.50, -0.50, -0.50});
        INDArray params2 = Nd4j.create(new double[] {
                0.00, 1.00, 3.00, 4.00, 5.00, 5.00, 6.00, 7.00, 8.00, 9.00});
        INDArray params4 = Nd4j.create(new double[] {
                0.50, 0.50, -0.50, -0.50, -0.50, 1.50, 1.50, -0.50, -0.50, -0.50,
                3.00, 3.00, 0.00, 0.00, 0.00, 4.00, 4.00, 0.00, 0.00, 0.00,
                5.00, 5.00, 0.00, 0.00, 0.00, 5.50, 5.50, -0.50, -0.50, -0.50,
                6.50, 6.50, -0.50, -0.50, -0.50, 7.50, 7.50, -0.50, -0.50, -0.50,
                8.50, 8.50, -0.50, -0.50, -0.50, 9.50, 9.50, -0.50, -0.50, -0.50
        }).reshape(10,5);

        double expectedScore = 0.0;

        assertEquals(expectedGradient, pair.getFirst());
        assertEquals(expectedScore, pair.getSecond(), 1e-8);

    }

    @Test
    public void testBackprop() throws Exception{
        int numSamples = 10;
        int batchSize = 10;
        DataSetIterator mnistIter = new MnistDataSetIterator(batchSize, numSamples, true);
        DataSet input = mnistIter.next();
        INDArray features = input.getFeatures();
        mnistIter.reset();

        MultiLayerNetwork rbm = getRBMMLNNet(true, true, features);
        rbm.fit(mnistIter);

        MultiLayerNetwork rbm2 = getRBMMLNNet(true, true, features);
        rbm2.fit(mnistIter);

        DataSet test = mnistIter.next();

        Evaluation eval = new Evaluation();
        INDArray output = rbm.output(test.getFeatureMatrix());
        eval.eval(test.getLabels(), output);
        double f1Score = eval.f1();

        Evaluation eval2 = new Evaluation();
        INDArray output2 = rbm2.output(test.getFeatureMatrix());
        eval2.eval(test.getLabels(), output2);
        double f1Score2 = eval2.f1();

        assertEquals(f1Score, f1Score2, 1e-4);

    }

    @Test
    public void testRBM() {
//        Original test from @Treo
        INDArray features = Nd4j.rand(new int[]{100, 10});

        System.out.println("Training RBM network, initialized with Xavier");
        MultiLayerNetwork rbm = getRBMMLNNet(true, true, features);
        rbm.fit(features);
        double v = rbm.score();

        System.out.println("Training RBM network, initialized with correct solution");
        MultiLayerNetwork rbm2 = getRBMMLNNet(true, true, features);

        rbm2.setParam("0_W", Nd4j.diag(Nd4j.onesLike(Nd4j.diag(rbm2.getParam("0_W")))));
        rbm2.setParam("1_W", Nd4j.diag(Nd4j.onesLike(Nd4j.diag(rbm2.getParam("1_W")))));

        rbm2.fit(features);
        double x = rbm2.score();
    }

    //////////////////////////////////////////////////////////////////////////////////


    private static INDArray getStandardParams(int nIn, int nOut){
        return Nd4j.hstack(Nd4j.ones(nIn & nOut), Nd4j.zeros(nOut + nIn));

    }

    private static INDArray getLinParams(int nIn, int nOut){
        return Nd4j.hstack(
                Nd4j.linspace(1, nIn * nOut, nIn * nOut), Nd4j.ones(nOut + nIn));
    }


//    TODO run through all options
//        List<HiddenUnit> hiddenUnits = Arrays.asList(HiddenUnit.BINARY, HiddenUnit.GAUSSIAN, HiddenUnit.SOFTMAX, HiddenUnit.RECTIFIED);
//        List<VisibleUnit> visibleUnits = Arrays.asList(VisibleUnit.BINARY, VisibleUnit.GAUSSIAN, VisibleUnit.SOFTMAX, VisibleUnit.LINEAR);

    private static RBM getRBMLayer(int nIn, int nOut, HiddenUnit hiddenUnit, VisibleUnit visibleUnit, INDArray params, boolean pretrain, boolean initialize) {
        org.deeplearning4j.nn.conf.layers.RBM layer = new org.deeplearning4j.nn.conf.layers.RBM.Builder(hiddenUnit, visibleUnit)
                .nIn(nIn)
                .nOut(nOut)
                .learningRate(1e-1f)
                .lossFunction(LossFunctions.LossFunction.KL_DIVERGENCE)
                .build();

        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .iterations(1)
                .seed(42)
                .layer(layer)
                .build();
        conf.setPretrain(pretrain);

        return (RBM) conf.getLayer().instantiate(conf, null, 0, params, initialize);
    }

    private static MultiLayerNetwork getRBMMLNNet(boolean backprop, boolean pretrain, INDArray input) {
        MultiLayerConfiguration rbm = new NeuralNetConfiguration.Builder()
                .seed(0xDEADBEEF)
                .iterations(1000)
                .biasInit(0)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.NONE)
                .epsilon(1)
                .weightInit(WeightInit.UNIFORM)
                .list(
                        new org.deeplearning4j.nn.conf.layers.RBM.Builder()
                                .lossFunction(LossFunctions.LossFunction.COSINE_PROXIMITY)
                                .activation("identity")
                                .nOut(input.columns()).build(),
                        new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(LossFunctions.LossFunction.COSINE_PROXIMITY)
                                .activation("identity")
                                .nOut(input.columns()).build()
                )
                .pretrain(pretrain)
                .backprop(backprop)
                .setInputType(InputType.feedForward(input.columns()))
                .build();
        MultiLayerNetwork network = new MultiLayerNetwork(rbm);
        network.init();
        network.setListeners(new ScoreIterationListener(100));

        return network;

    }
}
