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
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.params.PretrainParamInitializer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.impl.ActivationSigmoid;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.distribution.Distribution;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import static org.nd4j.linalg.ops.transforms.Transforms.sigmoid;

import java.util.ArrayList;
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
    public void testGradientFlattening() {
        INDArray features = Nd4j.create(new double[][] {{0, 0, 0, 0, 0, 0}});
        INDArray params = Nd4j.create(
                        new double[] {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
        INDArray expectedParams = params.dup();
        RBM rbm = getRBMLayer(6, 3, HiddenUnit.BINARY, VisibleUnit.BINARY, params, true, false, 1,
                        LossFunctions.LossFunction.SQUARED_LOSS, 1);
        //        INDArray expectedStepParams = Nd4j.create(new double[] {-0.25,-0.25,-0.25,-0.25,-0.25,-0.25,-0.25,-0.25,-0.25,-0.25,-0.25,-0.25,-0.25,-0.25,-0.25,-0.25,-0.25,-0.25,0.0,0.0,0.0,-0.5,-0.5,-0.5,-0.5,-0.5,-0.5});
        rbm.setBackpropGradientsViewArray(Nd4j.create(1, params.length()));
        rbm.fit(features);
        Gradient g = rbm.gradient();

        List<INDArray> grList = new ArrayList();
        grList.add(g.getGradientFor("W"));
        grList.add(g.getGradientFor("b"));
        grList.add(g.getGradientFor("vb"));
        INDArray expectedGradient = Nd4j.toFlattened('f', grList);

        assertEquals(expectedParams.subi(expectedGradient), rbm.params());
    }

    @Test
    public void testIrisGaussianHidden() {
        IrisDataFetcher fetcher = new IrisDataFetcher();
        fetcher.fetch(150);
        DataNormalization norm = new NormalizerStandardize();
        DataSet d = fetcher.next();
        norm.fit(d);
        norm.transform(d);

        INDArray params = Nd4j.create(1, 4 * 3 + 4 + 3);
        RBM rbm = getRBMLayer(4, 3, HiddenUnit.GAUSSIAN, VisibleUnit.GAUSSIAN, params, true, false, 1,
                        LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD);
        rbm.setBackpropGradientsViewArray(Nd4j.create(1, params.length()));

        rbm.fit(d.getFeatureMatrix());
    }

    @Test
    public void testIrisRectifiedHidden() {
        IrisDataFetcher fetcher = new IrisDataFetcher();
        fetcher.fetch(150);
        DataNormalization norm = new NormalizerStandardize();
        DataSet d = fetcher.next();
        norm.fit(d);
        norm.transform(d);

        INDArray params = Nd4j.create(1, 4 * 3 + 4 + 3);
        RBM rbm = getRBMLayer(4, 3, HiddenUnit.RECTIFIED, VisibleUnit.LINEAR, params, true, false, 1,
                        LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD);
        rbm.setBackpropGradientsViewArray(Nd4j.create(1, params.length()));

        rbm.fit(d.getFeatureMatrix());

    }

    @Test
    public void testMnist() throws Exception {
        MnistDataFetcher fetcher = new MnistDataFetcher(true);
        Nd4j.ENFORCE_NUMERICAL_STABILITY = true;

        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder().iterations(30)
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).learningRate(1e-1f)
                        .layer(new org.deeplearning4j.nn.conf.layers.RBM.Builder().nIn(784).nOut(600)
                                        .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(1, 1e-5))
                                        .lossFunction(LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY).build())
                        .build();

        conf.setPretrain(true);
        org.deeplearning4j.nn.conf.layers.RBM layerConf = (org.deeplearning4j.nn.conf.layers.RBM) conf.getLayer();

        fetcher.fetch(10);
        DataSet d2 = fetcher.next();

        //        org.nd4j.linalg.api.rng.distribution.Distribution dist = Nd4j.getDistributions().createNormal(1, 1e-5);
        //        System.out.println(dist.sample(new int[] {layerConf.getNIn(), layerConf.getNOut()}));

        INDArray input = d2.getFeatureMatrix();

        int numParams = conf.getLayer().initializer().numParams(conf);
        INDArray params = Nd4j.create(1, numParams);
        RBM rbm = (RBM) conf.getLayer().instantiate(conf, null, 0, params, true);
        rbm.setBackpropGradientsViewArray(Nd4j.create(params.shape()));
        rbm.fit(input);

    }

    @Test
    public void testSetGetParams() {
        INDArray params = Nd4j.create(1, 6 * 4 + 6 + 4);
        RBM rbm = getRBMLayer(6, 4, HiddenUnit.BINARY, VisibleUnit.BINARY, params, true, false, 1,
                        LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD);
        rbm.setBackpropGradientsViewArray(Nd4j.create(1, rbm.numParams()));
        INDArray rand2 = Nd4j.rand(new int[] {1, rbm.numParams()});
        rbm.setParams(rand2);
        rbm.setInput(Nd4j.zeros(6));
        rbm.computeGradientAndScore();
        INDArray getParams = rbm.params();
        assertEquals(rand2, getParams);
    }


    @Test
    public void testGradient() {
        float[][] data = new float[][] {{1, 1, 1, 0, 0, 0}, {1, 0, 1, 0, 0, 0}, {1, 1, 1, 0, 0, 0}, {0, 0, 1, 1, 1, 0},
                        {0, 0, 1, 1, 0, 0}, {0, 0, 1, 1, 1, 0}, {0, 0, 1, 1, 1, 0}};


        INDArray input = Nd4j.create(data);
        INDArray params = Nd4j.create(1, 6 * 4 + 6 + 4);
        RBM rbm = getRBMLayer(6, 4, HiddenUnit.BINARY, VisibleUnit.BINARY, params, true, false, 1,
                        LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD);
        rbm.setBackpropGradientsViewArray(Nd4j.create(1, params.length()));

        rbm.fit(input);
        double value = rbm.score();
        Gradient grad2 = rbm.gradient();
        assertEquals(24, grad2.getGradientFor("W").length());

    }

    @Test
    public void testPropUpDownBinary() {
        INDArray input = Nd4j.linspace(1, 10, 10);
        INDArray params = getLinParams(10, 5);

        RBM rbm = getRBMLayer(10, 5, HiddenUnit.BINARY, VisibleUnit.BINARY, params, true, false, 1,
                        LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD);

        INDArray actualParams = rbm.params();
        assertEquals(params, actualParams);

        // propUp
        INDArray expectedHOut = Transforms.sigmoid(Nd4j.create(new double[] {386., 936., 1486., 2036., 2586.}));

        INDArray actualHOut = rbm.propUp(input);
        assertEquals(expectedHOut, actualHOut);

        // propDown
        INDArray expectedVOut = Transforms.sigmoid(
                        Nd4j.create(new double[] {106., 111., 116., 121., 126., 131., 136., 141., 146., 151.}));

        INDArray actualVOut = rbm.propDown(actualHOut);
        assertEquals(expectedVOut, actualVOut);

    }

    @Test
    public void testComputeGradientAndScore() {
        INDArray input = Nd4j.linspace(1, 10, 10);
        INDArray params = getStandardParams(10, 5);

        RBM rbm = getRBMLayer(10, 5, HiddenUnit.BINARY, VisibleUnit.BINARY, params, true, false, 1,
                        LossFunctions.LossFunction.MSE);
        rbm.setInput(input);
        rbm.setBackpropGradientsViewArray(Nd4j.create(1, rbm.numParams()));
        rbm.computeGradientAndScore();
        Pair<Gradient, Double> pair = rbm.gradientAndScore();

        INDArray hprob = sigmoid(input.mmul(rbm.getParam(PretrainParamInitializer.WEIGHT_KEY))
                        .addiRowVector(rbm.getParam(PretrainParamInitializer.BIAS_KEY)));
        INDArray vprob = sigmoid(hprob.mmul(rbm.getParam(PretrainParamInitializer.WEIGHT_KEY).transpose())
                        .addiRowVector(rbm.getParam(PretrainParamInitializer.VISIBLE_BIAS_KEY)));
        Distribution dist = Nd4j.getDistributions().createBinomial(1, vprob);
        dist.reseedRandomGenerator(42);
        INDArray vSample = dist.sample(vprob.shape());

        //double expectedScore = LossFunctions.LossFunction.MSE.getILossFunction().computeScore(input, vSample, "sigmoid", null, false);
        double expectedScore = LossFunctions.LossFunction.MSE.getILossFunction().computeScore(input, vSample,
                        new ActivationSigmoid(), null, false);

        assertEquals(expectedScore, pair.getSecond(), 1e-8);
    }

    @Test
    public void testBackprop() throws Exception {
        int numSamples = 10;
        int batchSize = 10;
        DataSetIterator mnistIter = new MnistDataSetIterator(batchSize, numSamples, true);
        DataSet input = mnistIter.next();
        INDArray features = input.getFeatures();
        mnistIter.reset();

        MultiLayerNetwork rbm = getRBMMLNNet(true, true, features, 50, 10, WeightInit.UNIFORM);
        rbm.fit(mnistIter);

        MultiLayerNetwork rbm2 = getRBMMLNNet(true, true, features, 50, 10, WeightInit.UNIFORM);
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

    // Use to verify the model decreases in value
    @Test
    public void testRBMLayer() {
        INDArray features = Nd4j.create(new double[] {1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1,
                        1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0}).reshape(6, 6);

        INDArray params = getStandardParams(6, 2);

        RBM rbm = getRBMLayer(6, 2, HiddenUnit.IDENTITY, VisibleUnit.IDENTITY, params, true, true, 500,
                        LossFunctions.LossFunction.MSE);
        //        RBM rbm = getRBMLayer(6, 2, HiddenUnit.BINARY, VisibleUnit.BINARY, params, true, true, 500, LossFunctions.LossFunction.MSE);
        //        RBM rbm = getRBMLayer(6, 2, HiddenUnit.GAUSSIAN, VisibleUnit.BINARY, params, true, true, 500, LossFunctions.LossFunction.COSINE_PROXIMITY);
        //        RBM rbm = getRBMLayer(6, 2, HiddenUnit.BINARY, VisibleUnit.GAUSSIAN, params, true, true, 500, LossFunctions.LossFunction.KL_DIVERGENCE);
        //        RBM rbm = getRBMLayer(6, 2, HiddenUnit.RECTIFIED, VisibleUnit.BINARY, params, true, true, 500, LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD);
        //        RBM rbm = getRBMLayer(6, 2, HiddenUnit.SOFTMAX, VisibleUnit.LINEAR, params, true, true, 500, LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD);
        //        RBM rbm = getRBMLayer(6, 2, HiddenUnit.SOFTMAX, VisibleUnit.SOFTMAX, params, true, true, 500, LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD);
        rbm.setListeners(new ScoreIterationListener(10));
        rbm.setBackpropGradientsViewArray(Nd4j.create(1, params.length()));
        rbm.fit(features);
    }

    // Use to verify the model decreases in value

    @Ignore
    @Test
    public void testRBMMLN() {
        INDArray features = Nd4j.rand(new int[] {100, 10});

        System.out.println("Training RBM network, initialized with Xavier");
        MultiLayerNetwork rbm = getRBMMLNNet(true, true, features, 10, 10, WeightInit.XAVIER);
        rbm.setListeners(new ScoreIterationListener(1));
        rbm.fit(features);

        System.out.println("Training RBM network, initialized with correct solution");
        MultiLayerNetwork rbm2 = getRBMMLNNet(true, true, features, 10, 10, WeightInit.UNIFORM);
        rbm.setListeners(new ScoreIterationListener(1));

        rbm2.setParam("0_W", Nd4j.diag(Nd4j.onesLike(Nd4j.diag(rbm2.getParam("0_W")))));
        rbm2.setParam("1_W", Nd4j.diag(Nd4j.onesLike(Nd4j.diag(rbm2.getParam("1_W")))));

        rbm2.fit(features, features);
    }

    @Ignore
    @Test
    public void testMultiRBM() {
        INDArray features = Nd4j.rand(new int[] {100, 10});

        MultiLayerNetwork rbm = getMultiLayerRBMNet(true, true, features, 10, 5, 10, WeightInit.XAVIER);

        rbm.setListeners(new ScoreIterationListener(10));
        rbm.fit(features, features);

    }
    //////////////////////////////////////////////////////////////////////////////////


    private static INDArray getStandardParams(int nIn, int nOut) {
        return Nd4j.hstack(Nd4j.ones(nIn * nOut), Nd4j.zeros(nOut + nIn));

    }

    private static INDArray getLinParams(int nIn, int nOut) {
        return Nd4j.hstack(Nd4j.linspace(1, nIn * nOut, nIn * nOut), Nd4j.ones(nOut + nIn));
    }

    private List<HiddenUnit> getHiddenUnits() {
        return Arrays.asList(HiddenUnit.IDENTITY, HiddenUnit.BINARY, HiddenUnit.GAUSSIAN, HiddenUnit.RECTIFIED,
                        HiddenUnit.SOFTMAX);
    }

    private List<VisibleUnit> getVisibleUnits() {
        return Arrays.asList(VisibleUnit.IDENTITY, VisibleUnit.BINARY, VisibleUnit.GAUSSIAN, VisibleUnit.LINEAR,
                        VisibleUnit.SOFTMAX);
    }


    private static RBM getRBMLayer(int nIn, int nOut, HiddenUnit hiddenUnit, VisibleUnit visibleUnit, INDArray params,
                    boolean pretrain, boolean initialize, int iterations, LossFunctions.LossFunction lossFunctions,
                    int learningRate) {
        org.deeplearning4j.nn.conf.layers.RBM layer =
                        new org.deeplearning4j.nn.conf.layers.RBM.Builder(hiddenUnit, visibleUnit).nIn(nIn).nOut(nOut)
                                        .learningRate(learningRate).lossFunction(lossFunctions).build();

        NeuralNetConfiguration conf =
                        new NeuralNetConfiguration.Builder().iterations(iterations).seed(42).layer(layer).build();
        conf.setPretrain(pretrain);

        return (RBM) conf.getLayer().instantiate(conf, null, 0, params, initialize);
    }

    private static RBM getRBMLayer(int nIn, int nOut, HiddenUnit hiddenUnit, VisibleUnit visibleUnit, INDArray params,
                    boolean pretrain, boolean initialize, int iterations, LossFunctions.LossFunction lossFunctions) {
        org.deeplearning4j.nn.conf.layers.RBM layer =
                        new org.deeplearning4j.nn.conf.layers.RBM.Builder(hiddenUnit, visibleUnit).nIn(nIn).nOut(nOut)
                                        .learningRate(1e-1f).lossFunction(lossFunctions).build();

        NeuralNetConfiguration conf =
                        new NeuralNetConfiguration.Builder().iterations(iterations).seed(42).layer(layer).build();
        conf.setPretrain(pretrain);

        return (RBM) conf.getLayer().instantiate(conf, null, 0, params, initialize);
    }

    private static MultiLayerNetwork getRBMMLNNet(boolean backprop, boolean pretrain, INDArray input, int nOut1,
                    int nOut2, WeightInit weightInit) {
        MultiLayerConfiguration rbm = new NeuralNetConfiguration.Builder().seed(0xDEADBEEF).iterations(1000).biasInit(0)
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).updater(Updater.NONE)
                        .epsilon(1).weightInit(weightInit)
                        .list(new org.deeplearning4j.nn.conf.layers.RBM.Builder(HiddenUnit.BINARY, VisibleUnit.BINARY)
                                        .lossFunction(LossFunctions.LossFunction.MSE).nOut(nOut1).build(),
                                        new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(
                                                        LossFunctions.LossFunction.MSE).nOut(nOut2).build())
                        .pretrain(pretrain).backprop(backprop).setInputType(InputType.feedForward(input.columns()))
                        .build();
        MultiLayerNetwork network = new MultiLayerNetwork(rbm);
        network.init();

        return network;

    }

    private static MultiLayerNetwork getMultiLayerRBMNet(boolean backprop, boolean pretrain, INDArray input, int nOut1,
                    int nOut2, int nOut3, WeightInit weightInit) {
        MultiLayerConfiguration rbm = new NeuralNetConfiguration.Builder().seed(0xDEADBEEF).iterations(1000).biasInit(0)
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).updater(Updater.NONE)
                        .epsilon(1).weightInit(weightInit)
                        .list(new org.deeplearning4j.nn.conf.layers.RBM.Builder()
                                        .lossFunction(LossFunctions.LossFunction.KL_DIVERGENCE).nOut(nOut1).build(),
                                        new org.deeplearning4j.nn.conf.layers.RBM.Builder()
                                                        .lossFunction(LossFunctions.LossFunction.KL_DIVERGENCE)
                                                        .nOut(nOut2).build(),
                                        new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(
                                                        LossFunctions.LossFunction.MSE).activation(Activation.RELU)
                                                                        .nOut(nOut3).build())
                        .pretrain(pretrain).backprop(backprop).setInputType(InputType.feedForward(input.columns()))
                        .build();
        MultiLayerNetwork network = new MultiLayerNetwork(rbm);
        network.init();

        return network;

    }


}
