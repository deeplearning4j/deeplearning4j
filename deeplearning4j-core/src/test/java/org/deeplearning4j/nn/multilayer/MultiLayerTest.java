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

package org.deeplearning4j.nn.multilayer;

import com.google.common.collect.Lists;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.LFWDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.api.Updater;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.layers.BatchNormalization;
import org.deeplearning4j.nn.conf.preprocessor.CnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToCnnPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.ReshapePreProcessor;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.BaseOutputLayer;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.deeplearning4j.nn.layers.normalization.*;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.nn.params.PretrainParamInitializer;
import org.deeplearning4j.nn.updater.UpdaterCreator;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.api.StepFunction;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.optimize.stepfunctions.NegativeGradientStepFunction;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.heartbeat.Heartbeat;
import org.nd4j.linalg.heartbeat.reports.Environment;
import org.nd4j.linalg.heartbeat.reports.Event;
import org.nd4j.linalg.heartbeat.reports.Task;
import org.nd4j.linalg.heartbeat.utils.EnvironmentUtils;
import org.nd4j.linalg.heartbeat.utils.TaskUtils;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;

import static org.junit.Assert.*;

/**
 * Created by agibsonccc on 12/27/14.
 */
public class MultiLayerTest {

    private static final Logger log = LoggerFactory.getLogger(MultiLayerTest.class);

    @Test
    public void testSetParams() {
        Nd4j.MAX_ELEMENTS_PER_SLICE = Integer.MAX_VALUE;
        Nd4j.MAX_SLICES_TO_PRINT = Integer.MAX_VALUE;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .list(2)
                .layer(0, new RBM.Builder(RBM.HiddenUnit.RECTIFIED, RBM.VisibleUnit.GAUSSIAN)
                        .nIn(4).nOut(3)
                        .activation("tanh")
                        .build())
                .layer(1,new RBM.Builder(RBM.HiddenUnit.GAUSSIAN, RBM.VisibleUnit.GAUSSIAN).nIn(3).nOut(2)
                        .build())
                .build();

        MultiLayerNetwork network3 = new MultiLayerNetwork(conf);
        network3.init();

        INDArray params = network3.params();
        INDArray weights = network3.getLayer(0).getParam(DefaultParamInitializer.WEIGHT_KEY).dup();
        INDArray bias = network3.getLayer(0).getParam(DefaultParamInitializer.BIAS_KEY).dup();
        network3.setParameters(params);
        assertEquals(weights, network3.getLayer(0).getParam(DefaultParamInitializer.WEIGHT_KEY));
        assertEquals(bias,network3.getLayer(0).getParam(DefaultParamInitializer.BIAS_KEY));
        INDArray params4 = network3.params();
        assertEquals(params, params4);
    }

    @Test
    public void testReDistribute() {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .list(2)
                .layer(0, new RBM.Builder(RBM.HiddenUnit.RECTIFIED, RBM.VisibleUnit.GAUSSIAN)
                        .nIn(4).nOut(3)
                        .activation("tanh")
                        .build())
                .layer(1, new RBM.Builder(RBM.HiddenUnit.GAUSSIAN, RBM.VisibleUnit.GAUSSIAN).nIn(3).nOut(2)
                        .build())
                .build();


        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.init();
        INDArray params = network.params(true);
        network.reDistributeParams(true);
        INDArray params2 = network.params(true);
        assertEquals(params,params2);
    }



    @Test
    public void testBatchNorm() {
        Nd4j.getRandom().setSeed(123);
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.LINE_GRADIENT_DESCENT)
                .iterations(5)
                .seed(123)
                .list(4)
                .layer(0, new DenseLayer.Builder().nIn(4).nOut(3).weightInit(WeightInit.XAVIER).activation("tanh").build())
                .layer(1, new DenseLayer.Builder().nIn(3).nOut(2).weightInit(WeightInit.XAVIER).activation("tanh").build())
                .layer(2, new BatchNormalization.Builder().build())
                .layer(3, new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .weightInit(WeightInit.XAVIER)
                        .activation("softmax")
                        .nIn(2).nOut(3).build())
                .backprop(true).pretrain(false)
                .inputPreProcessor(2,new FeedForwardToCnnPreProcessor(1,2,1))
                .inputPreProcessor(3,new CnnToFeedForwardPreProcessor(1,2,1))
                .build();


        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.init();
        network.setListeners(new ScoreIterationListener(1));

        DataSetIterator iter = new IrisDataSetIterator(150, 150);

        DataSet next = iter.next();
        next.normalizeZeroMeanZeroUnitVariance();
        SplitTestAndTrain trainTest = next.splitTestAndTrain(110);
        network.setLabels(trainTest.getTrain().getLabels());
        network.init();
        network.setInput(trainTest.getTrain().getFeatureMatrix());
        network.setLabels(trainTest.getTrain().getLabels());
        network.computeGradientAndScore();
        INDArray grad = network.gradient().gradient();
        
        assertTrue("Gradient contains invalid numbers", !Nd4j.hasInvalidNumber(grad));
        
        int numParams = network.numParams();
        network.fit(trainTest.getTrain());

    }

    @Test
    public void testBackProp() {
        Nd4j.getRandom().setSeed(123);
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.LINE_GRADIENT_DESCENT)
                .iterations(5)
                .seed(123)
                .list(3)
                .layer(0, new DenseLayer.Builder().nIn(4).nOut(3).weightInit(WeightInit.XAVIER).activation("tanh").build())
                .layer(1, new DenseLayer.Builder().nIn(3).nOut(2).weightInit(WeightInit.XAVIER).activation("tanh").build())
                .layer(2, new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .weightInit(WeightInit.XAVIER)
                        .activation("softmax")
                        .nIn(2).nOut(3).build())
                .backprop(true).pretrain(false).build();


        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.init();
        network.setListeners(Lists.<IterationListener>newArrayList(new ScoreIterationListener(1)));

        DataSetIterator iter = new IrisDataSetIterator(150, 150);

        DataSet next = iter.next();
        next.normalizeZeroMeanZeroUnitVariance();
        SplitTestAndTrain trainTest = next.splitTestAndTrain(110);
        network.setInput(trainTest.getTrain().getFeatureMatrix());
        network.setLabels(trainTest.getTrain().getLabels());
        network.init();
        network.fit(trainTest.getTrain());

        DataSet test = trainTest.getTest();
        Evaluation eval = new Evaluation();
        INDArray output = network.output(test.getFeatureMatrix());
        eval.eval(test.getLabels(), output);
        log.info("Score " + eval.stats());

    }

    @Test
    public void testDbn() throws Exception {
        Nd4j.MAX_SLICES_TO_PRINT = -1;
        Nd4j.MAX_ELEMENTS_PER_SLICE = -1;
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .iterations(100)
                .momentum(0.9)
                .optimizationAlgo(OptimizationAlgorithm.LBFGS)
                .constrainGradientToUnitNorm(true)
                .regularization(true)
                .l2(2e-4)
                .list(2)
                .layer(0, new RBM.Builder(RBM.HiddenUnit.GAUSSIAN, RBM.VisibleUnit.GAUSSIAN)
                        .nIn(4).nOut(3)
                        .weightInit(WeightInit.DISTRIBUTION).dist(new UniformDistribution(0, 1))
                        .activation("tanh")
                        .lossFunction(LossFunctions.LossFunction.RMSE_XENT).build())
                .layer(1, new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .nIn(3).nOut(3)
                        .weightInit(WeightInit.DISTRIBUTION).dist(new UniformDistribution(0, 1))
                        .activation("softmax").build())
                .build();


        MultiLayerNetwork d = new MultiLayerNetwork(conf);

        DataSetIterator iter = new IrisDataSetIterator(150, 150);

        DataSet next = iter.next();

        Nd4j.writeTxt(next.getFeatureMatrix(), "iris.txt", "\t");

        next.normalizeZeroMeanZeroUnitVariance();

        SplitTestAndTrain testAndTrain = next.splitTestAndTrain(110);
        DataSet train = testAndTrain.getTrain();

        d.fit(train);

        DataSet test = testAndTrain.getTest();

        Evaluation eval = new Evaluation();
        INDArray output = d.output(test.getFeatureMatrix());
        eval.eval(test.getLabels(), output);
        log.info("Score " + eval.stats());

    }



    @Test
    public void testGradientWithAsList(){
        MultiLayerNetwork net1 = new MultiLayerNetwork(getConf());
        MultiLayerNetwork net2 = new MultiLayerNetwork(getConf());
        net1.init();
        net2.init();

        DataSet x1 = new IrisDataSetIterator(1,150).next();
        DataSet all = new IrisDataSetIterator(150,150).next();
        DataSet x2 = all.asList().get(0);

        //x1 and x2 contain identical data
        assertArrayEquals(asFloat(x1.getFeatureMatrix()), asFloat(x2.getFeatureMatrix()), 0.0f);
        assertArrayEquals(asFloat(x1.getLabels()), asFloat(x2.getLabels()), 0.0f);
        assertEquals(x1, x2);

        //Set inputs/outputs so gradient can be calculated:
        net1.feedForward(x1.getFeatureMatrix());
        net2.feedForward(x2.getFeatureMatrix());
        ((BaseOutputLayer)net1.getLayer(1)).setLabels(x1.getLabels());
        ((BaseOutputLayer)net2.getLayer(1)).setLabels(x2.getLabels());

        net1.gradient();
        net2.gradient();
    }


    @Test
    public void testFeedForwardActivationsAndDerivatives(){
        MultiLayerNetwork network = new MultiLayerNetwork(getConf());
        network.init();
        DataSet data = new IrisDataSetIterator(1,150).next();
        network.fit(data);
        Pair result = network.feedForwardActivationsAndDerivatives(false);
        List<INDArray> first = (List) result.getFirst();
        List<INDArray> second = (List) result.getSecond();
        assertEquals(first.size(), second.size());
    }

    /**
     *  This test intended only to test activateSelectedLayers method, it does not involves fully-working AutoEncoder.
     */
    @Test
    public void testSelectedActivations() {
        // Train DeepAutoEncoder on very limited trainset
        final int numRows = 28;
        final int numColumns = 28;
        int seed = 123;
        int numSamples = 3;
        int iterations = 1;
        int listenerFreq = iterations/5;

        log.info("Load data....");

        float[][] trainingData = new float[numSamples][numColumns * numRows];
        Arrays.fill(trainingData[0],0.95f);
        Arrays.fill(trainingData[1],0.5f);
        Arrays.fill(trainingData[2], 0.05f);



        log.info("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .optimizationAlgo(OptimizationAlgorithm.LINE_GRADIENT_DESCENT)
                .list(10)
                .layer(0, new RBM.Builder().nIn(numRows * numColumns).nOut(1000).lossFunction(LossFunctions.LossFunction.RMSE_XENT).build())
                .layer(1, new RBM.Builder().nIn(1000).nOut(500).lossFunction(LossFunctions.LossFunction.RMSE_XENT).build())
                .layer(2, new RBM.Builder().nIn(500).nOut(250).lossFunction(LossFunctions.LossFunction.RMSE_XENT).build())
                .layer(3, new RBM.Builder().nIn(250).nOut(100).lossFunction(LossFunctions.LossFunction.RMSE_XENT).build())
                .layer(4, new RBM.Builder().nIn(100).nOut(30).lossFunction(LossFunctions.LossFunction.RMSE_XENT).build()) //encoding stops
                .layer(5, new RBM.Builder().nIn(30).nOut(100).lossFunction(LossFunctions.LossFunction.RMSE_XENT).build()) //decoding starts
                .layer(6, new RBM.Builder().nIn(100).nOut(250).lossFunction(LossFunctions.LossFunction.RMSE_XENT).build())
                .layer(7, new RBM.Builder().nIn(250).nOut(500).lossFunction(LossFunctions.LossFunction.RMSE_XENT).build())
                .layer(8, new RBM.Builder().nIn(500).nOut(1000).lossFunction(LossFunctions.LossFunction.RMSE_XENT).build())
                .layer(9, new OutputLayer.Builder(LossFunctions.LossFunction.RMSE_XENT).nIn(1000).nOut(numRows*numColumns).build())
                .pretrain(true).backprop(true)
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        model.setListeners(Arrays.asList((IterationListener) new ScoreIterationListener(listenerFreq)));

        log.info("Train model....");
        int cnt = 0;
        while(cnt < numSamples) {
            INDArray input = Nd4j.create(trainingData[cnt]);
            model.fit(new DataSet(input, input));
            cnt++;
        }
        // Make two separate selective calls

        log.info("Testing full cycle...");

        List<INDArray> comparableResult = model.feedForward(Nd4j.create(trainingData[0]));

        INDArray encodeResult = model.activateSelectedLayers(0,4, Nd4j.create(trainingData[0]));

        log.info("Compare feedForward results with selectedActivation");

        assertEquals(comparableResult.get(5), encodeResult);

        INDArray decodeResults = model.activateSelectedLayers(5,9, encodeResult);


        log.info("Decode results: " + decodeResults.columns() + " " + decodeResults);
        log.info("Comparable  results: " + comparableResult.get(10).columns() + " " + comparableResult.get(10));

        assertEquals(comparableResult.get(10), decodeResults);
    }

    private static MultiLayerConfiguration getConf(){
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(12345L)
                .list(2)
                .layer(0, new RBM.Builder(RBM.HiddenUnit.RECTIFIED, RBM.VisibleUnit.GAUSSIAN)
                        .nIn(4).nOut(3)
                        .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0, 1))
                        .build())
                .layer(1, new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation("softmax")
                        .nIn(3).nOut(3)
                        .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0, 1))
                        .build())
                .build();
        return conf;
    }

    public static float[] asFloat( INDArray arr ){
        int len = arr.length();
        float[] f = new float[len];
        for( int i=0; i<len; i++ ) f[i] = arr.getFloat(i);
        return f;
    }

    @Test
    public void testFeedForwardToLayer(){

        int nIn = 30;
        int nOut = 25;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT)
                .iterations(5).learningRate(1e-3)
                .list(4)
                .layer(0, new RBM.Builder(RBM.HiddenUnit.RECTIFIED, RBM.VisibleUnit.GAUSSIAN)
                        .nIn(nIn).nOut(600)
                        .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0, 1e-5))
                        .lossFunction(LossFunctions.LossFunction.RMSE_XENT).build())
                .layer(1, new RBM.Builder(RBM.HiddenUnit.RECTIFIED, RBM.VisibleUnit.GAUSSIAN)
                        .nIn(600).nOut(250)
                        .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0, 1e-5))
                        .lossFunction(LossFunctions.LossFunction.RMSE_XENT).build())
                .layer(2, new RBM.Builder(RBM.HiddenUnit.RECTIFIED, RBM.VisibleUnit.GAUSSIAN)
                        .nIn(250).nOut(100)
                        .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0, 1e-5))
                        .lossFunction(LossFunctions.LossFunction.RMSE_XENT).build())
                .layer(3, new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .nIn(100).nOut(25)
                        .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0, 1e-5)).build())
                .build();

        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.init();


        INDArray input = Nd4j.rand(5,nIn);

        List<INDArray> activations = network.feedForward(input);
        assertEquals(5,activations.size());     //4 layers + input

        List<INDArray> activationsAll = network.feedForwardToLayer(3,input);
        assertEquals(activations,activationsAll);

        for( int i=3; i>=0; i-- ){
            List<INDArray> activationsPartial = network.feedForwardToLayer(i,input);
            assertEquals(i+2,activationsPartial.size());    //i+2: for layer 3: input + activations of {0,1,2,3} -> 5 total = 3+2
            for( int j=0; j<=i; j++ ){
                INDArray exp = activationsAll.get(j);
                INDArray act = activationsPartial.get(j);
                assertEquals(exp,act);
            }
        }
    }


    @Test
    public void testBackpropGradient(){
        //Testing: MultiLayerNetwork.backpropGradient()
        //i.e., specifically without an output layer

        int nIn = 10;
        int nOut = 40;
        int miniBatch = 5;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .updater(org.deeplearning4j.nn.conf.Updater.SGD)
                .learningRate(0.1)
                .list(3)
                .layer(0, new DenseLayer.Builder().nIn(nIn).nOut(20).activation("relu").weightInit(WeightInit.XAVIER).build())
                .layer(1, new DenseLayer.Builder().nIn(20).nOut(30).activation("relu").weightInit(WeightInit.XAVIER).build())
                .layer(2, new DenseLayer.Builder().nIn(30).nOut(nOut).activation("relu").weightInit(WeightInit.XAVIER).build())
                .build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        Nd4j.getRandom().setSeed(12345);
        INDArray eps = Nd4j.rand(miniBatch,nOut);
        INDArray input = Nd4j.rand(miniBatch, nIn);

        net.feedForward(input); //Need to feed forward before backprop

        Pair<Gradient,INDArray> pair = net.backpropGradient(eps);
        INDArray epsOut = pair.getSecond();
        assertNotNull(epsOut);
        assertArrayEquals(new int[]{miniBatch,nIn},epsOut.shape());

        Gradient g = pair.getFirst();
        Map<String,INDArray> gradMap = g.gradientForVariable();
        assertEquals(6, gradMap.size());    //3 layers, weight + bias gradients for each

        String[] expKeys = {"0_"+DefaultParamInitializer.WEIGHT_KEY,"0_"+DefaultParamInitializer.BIAS_KEY,
                "1_"+DefaultParamInitializer.WEIGHT_KEY,"2_"+DefaultParamInitializer.BIAS_KEY,
                "2_"+DefaultParamInitializer.WEIGHT_KEY,"2_"+DefaultParamInitializer.BIAS_KEY};
        Set<String> keys = gradMap.keySet();
        for( String s : expKeys ){
            assertTrue(keys.contains(s));
        }

        /*
        System.out.println(pair);

        //Use updater to go from raw gradients -> updates
        //Apply learning rate, gradient clipping, adagrad/momentum/rmsprop etc
        Updater updater = UpdaterCreator.getUpdater(net);
        updater.update(net, g, 0, miniBatch);

        StepFunction stepFunction = new NegativeGradientStepFunction();
        INDArray params = net.params();
        System.out.println(Arrays.toString(params.get(NDArrayIndex.all(), NDArrayIndex.interval(0, 10)).dup().data().asFloat()));
        stepFunction.step(params, g.gradient());
        net.setParams(params);    //params() may not be in-place
        System.out.println(Arrays.toString(params.get(NDArrayIndex.all(), NDArrayIndex.interval(0, 10)).dup().data().asFloat()));
        */
    }

    @Test
    public void testLayerNames(){
        int nIn = 10;
        int nOut = 40;

        List<String> layerNameList = new ArrayList<>();
        layerNameList.add("dnn1");
        layerNameList.add("dnn2");
        layerNameList.add("dnn3");

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .updater(org.deeplearning4j.nn.conf.Updater.SGD)
                .learningRate(0.1)
                .list(3)
                .layer(0, new DenseLayer.Builder().name("dnn1").nIn(nIn).nOut(20).activation("relu").weightInit(WeightInit.XAVIER).build())
                .layer(1, new DenseLayer.Builder().name("dnn2").nIn(20).nOut(30).activation("relu").weightInit(WeightInit.XAVIER).build())
                .layer(2, new DenseLayer.Builder().name("dnn3").nIn(30).nOut(nOut).activation("softmax").weightInit(WeightInit.XAVIER).build())
                .build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        assertEquals(layerNameList.get(0), net.getLayer(0).conf().getLayer().getLayerName());
        assertEquals(layerNameList, net.getLayerNames());
        assertEquals("softmax", net.getLayer(layerNameList.get(2)).conf().getLayer().getActivationFunction());
    }

    @Test
    public void testTranspose(){
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .iterations(100)
                .momentum(0.9)
                .regularization(true)
                .l2(2e-4)
                .list(2)
                .layer(0, new RBM.Builder(RBM.HiddenUnit.GAUSSIAN, RBM.VisibleUnit.GAUSSIAN)
                        .nIn(4).nOut(3)
                        .weightInit(WeightInit.DISTRIBUTION).dist(new UniformDistribution(0, 1))
                        .activation("tanh")
                        .lossFunction(LossFunctions.LossFunction.RMSE_XENT).build())
                .layer(1, new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .nIn(3).nOut(3)
                        .weightInit(WeightInit.DISTRIBUTION).dist(new UniformDistribution(0, 1))
                        .activation("softmax").build())
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        Layer layer = net.getLayer(0);
        Layer transposed = layer.transpose();

        assertArrayEquals(new int[]{4,3},layer.getParam(DefaultParamInitializer.WEIGHT_KEY).shape());
        assertArrayEquals(new int[]{1,3},layer.getParam(DefaultParamInitializer.BIAS_KEY).shape());
        assertArrayEquals(new int[]{1,4},layer.getParam(PretrainParamInitializer.VISIBLE_BIAS_KEY).shape());

        assertArrayEquals(new int[]{3, 4}, transposed.getParam(DefaultParamInitializer.WEIGHT_KEY).shape());
        assertArrayEquals(new int[]{1, 4}, transposed.getParam(DefaultParamInitializer.BIAS_KEY).shape());
        assertArrayEquals(new int[]{1, 3}, transposed.getParam(PretrainParamInitializer.VISIBLE_BIAS_KEY).shape());


        INDArray origWeights = layer.getParam(DefaultParamInitializer.WEIGHT_KEY);
        INDArray transposedWeights = transposed.getParam(DefaultParamInitializer.WEIGHT_KEY);
        assertEquals(origWeights.transpose(), transposedWeights);
        assertEquals(layer.getParam(PretrainParamInitializer.VISIBLE_BIAS_KEY), transposed.getParam(DefaultParamInitializer.BIAS_KEY));
        assertEquals(layer.getParam(DefaultParamInitializer.BIAS_KEY),transposed.getParam(PretrainParamInitializer.VISIBLE_BIAS_KEY));

        assertEquals(3,((FeedForwardLayer)transposed.conf().getLayer()).getNIn());
        assertEquals(4,((FeedForwardLayer)transposed.conf().getLayer()).getNOut());
    }


    @Test
    public void testScoreExamples(){
        Nd4j.getRandom().setSeed(12345);
        int nIn = 5;
        int nOut = 6;
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(12345)
                .regularization(true).l1(0.01).l2(0.01)
                .learningRate(0.1).activation("tanh").weightInit(WeightInit.XAVIER)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(nIn).nOut(20).build())
                .layer(1, new DenseLayer.Builder().nIn(20).nOut(30).build())
                .layer(2, new OutputLayer.Builder().lossFunction(LossFunctions.LossFunction.MSE).nIn(30).nOut(nOut).build())
                .build();

        MultiLayerConfiguration confNoReg = new NeuralNetConfiguration.Builder()
                .seed(12345)
                .regularization(false)
                .learningRate(0.1).activation("tanh").weightInit(WeightInit.XAVIER)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(nIn).nOut(20).build())
                .layer(1, new DenseLayer.Builder().nIn(20).nOut(30).build())
                .layer(2, new OutputLayer.Builder().lossFunction(LossFunctions.LossFunction.MSE).nIn(30).nOut(nOut).build())
                .build();


        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        MultiLayerNetwork netNoReg = new MultiLayerNetwork(confNoReg);
        netNoReg.init();
        netNoReg.setParameters(net.params().dup());

        //Score single example, and compare to scoreExamples:
        INDArray input = Nd4j.rand(3,nIn);
        INDArray output = Nd4j.rand(3,nOut);
        DataSet ds = new DataSet(input,output);

        INDArray scoresWithRegularization = net.scoreExamples(ds,true);
        INDArray scoresNoRegularization = net.scoreExamples(ds,false);

        assertArrayEquals(new int[]{3,1},scoresWithRegularization.shape());
        assertArrayEquals(new int[]{3,1},scoresNoRegularization.shape());

        for( int i=0; i<3; i++ ){
            DataSet singleEx = new DataSet(input.getRow(i),output.getRow(i));
            double score = net.score(singleEx);
            double scoreNoReg = netNoReg.score(singleEx);

            double scoreUsingScoreExamples = scoresWithRegularization.getDouble(i);
            double scoreUsingScoreExamplesNoReg = scoresNoRegularization.getDouble(i);
            assertEquals(score,scoreUsingScoreExamples,1e-4);
            assertEquals(scoreNoReg,scoreUsingScoreExamplesNoReg,1e-4);
            assertTrue(scoreUsingScoreExamples > scoreUsingScoreExamplesNoReg);     //Regularization term increases score

//            System.out.println(score + "\t" + scoreUsingScoreExamples + "\t|\t" + scoreNoReg + "\t" + scoreUsingScoreExamplesNoReg);
        }
    }

    @Test
    @Ignore
    public void testCid() throws Exception {
        System.out.println(EnvironmentUtils.buildCId());

        Environment environment = EnvironmentUtils.buildEnvironment();
        environment.setSerialVersionID(EnvironmentUtils.buildCId());

        Task task = TaskUtils.buildTask(Nd4j.create(new double[]{1,2,3,4,5,6}));

        Heartbeat.getInstance().reportEvent(Event.STANDALONE, environment, task);

        Thread.sleep(25000);
    }
}