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
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.layers.BatchNormalization;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.RBM;
import org.deeplearning4j.nn.conf.preprocessor.CnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToCnnPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.ReshapePreProcessor;
import org.deeplearning4j.nn.layers.BaseOutputLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

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
        network3.setParameters(params);
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
                .layer(1,new RBM.Builder(RBM.HiddenUnit.GAUSSIAN, RBM.VisibleUnit.GAUSSIAN).nIn(3).nOut(2)
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
    public void testDbnFaces() {
        DataSetIterator iter = new LFWDataSetIterator(28,28);

        DataSet next = iter.next();
        next.normalizeZeroMeanZeroUnitVariance();

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT)
                .iterations(5).learningRate(1e-3)
                .list(4)
                .layer(0, new RBM.Builder(RBM.HiddenUnit.RECTIFIED, RBM.VisibleUnit.GAUSSIAN)
                        .nIn(next.numInputs()).nOut(600)
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
                        .nIn(100).nOut(iter.totalOutcomes())
                        .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0, 1e-5)).build())
                .build();

        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.init();
        network.setListeners(Arrays.<IterationListener>asList(new ScoreIterationListener(4)));
        network.fit(next);

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
                .layer(2, BatchNormalization.builder().shape(new int[]{1,2}).build())
                .layer(3, new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .weightInit(WeightInit.XAVIER)
                        .activation("softmax")
                        .nIn(2).nOut(3).build())
                .backprop(true).pretrain(false).inputPreProcessor(2,new FeedForwardToCnnPreProcessor(1,2,1))
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
        network.fit(trainTest.getTrain());

        DataSet test = trainTest.getTest();
        Evaluation eval = new Evaluation();
        INDArray output = network.output(test.getFeatureMatrix());
        eval.eval(test.getLabels(), output);
        log.info("Score " + eval.stats());

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

        Nd4j.writeTxt(next.getFeatureMatrix(),"iris.txt","\t");

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
        assertArrayEquals(asFloat(x1.getFeatureMatrix()), asFloat(x2.getFeatureMatrix()), 0.0f);  //OK
        assertArrayEquals(asFloat(x1.getLabels()), asFloat(x2.getLabels()), 0.0f);                //OK
        assertEquals(x1,x2);    //Fails, DataSet doesn't override Object.equals()

        //Set inputs/outputs so gradient can be calculated:
        net1.feedForward(x1.getFeatureMatrix());
        net2.feedForward(x2.getFeatureMatrix());
        ((BaseOutputLayer)net1.getLayer(1)).setLabels(x1.getLabels());
        ((BaseOutputLayer)net2.getLayer(1)).setLabels(x2.getLabels());

        net1.gradient();        //OK
        net2.gradient();        //IllegalArgumentException: Buffers must fill up specified length 29
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


}
