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

package org.deeplearning4j.nn.layers;

import java.util.Arrays;

import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;

import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static org.junit.Assert.*;

/**
 * Created by agibsonccc on 9/1/14.
 */
public class OutputLayerTest {
    private static final Logger log = LoggerFactory.getLogger(OutputLayerTest.class);


    @Test
    public void testIris2() {
        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .iterations(10)
                .learningRate(1e-1)
                .layer(new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder()
                        .nIn(4).nOut(3)
                        .weightInit(WeightInit.XAVIER)
                        .activation("softmax")
                        .lossFunction(LossFunctions.LossFunction.MCXENT).build())
                .build();

        OutputLayer l = LayerFactories.getFactory(conf.getLayer()).create(conf, Arrays.<IterationListener>asList(new ScoreIterationListener(1)),0);
        DataSetIterator iter = new IrisDataSetIterator(150, 150);


        DataSet next = iter.next();
        next.shuffle();
        SplitTestAndTrain trainTest = next.splitTestAndTrain(110);
        trainTest.getTrain().normalizeZeroMeanZeroUnitVariance();
        l.fit(trainTest.getTrain());


        DataSet test = trainTest.getTest();
        test.normalizeZeroMeanZeroUnitVariance();
        Evaluation eval = new Evaluation();
        INDArray output = l.output(test.getFeatureMatrix());
        eval.eval(test.getLabels(),output);
        log.info("Score " +eval.stats());


    }
    @Test
    public void test3() {

        org.nd4j.linalg.dataset.api.iterator.DataSetIterator iter = new IrisDataSetIterator(150,150);
        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder().iterations(3)
                .miniBatch(false).optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .layer(new org.deeplearning4j.nn.conf.layers.OutputLayer
                        .Builder(LossFunctions.LossFunction.MCXENT)
                        .nIn(4).nOut(3)
                        .activation("softmax")
                        .weightInit(WeightInit.XAVIER).build()).build();
        org.deeplearning4j.nn.layers.OutputLayer layer = LayerFactories.getFactory(conf).create(conf);
        DataSet next = iter.next();
        next.normalizeZeroMeanZeroUnitVariance();
        layer.setListeners(new ScoreIterationListener(1));

        layer.fit(next);


    }

    @Test
    public void testWeightsDifferent() {
        Nd4j.MAX_ELEMENTS_PER_SLICE = Integer.MAX_VALUE;
        Nd4j.MAX_SLICES_TO_PRINT = Integer.MAX_VALUE;

        NeuralNetConfiguration neuralNetConfiguration = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .constrainGradientToUnitNorm(true)
                .seed(123)
                .iterations(1000)
                .learningRate(1e-1)
                .layer(new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder()
                        .nIn(4).nOut(3)
                        .weightInit(WeightInit.XAVIER)
                        .updater(Updater.ADAGRAD)
                        .lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .activation("softmax").build())
                .build();

        OutputLayer o = LayerFactories.getFactory(neuralNetConfiguration).create(neuralNetConfiguration);

        int numSamples = 150;
        int batchSize = 150;


        DataSetIterator iter = new IrisDataSetIterator(batchSize, numSamples);
        DataSet iris = iter.next(); // Loads data into generator and format consumable for NN
        iris.normalizeZeroMeanZeroUnitVariance();
        o.setListeners(new ScoreIterationListener(1));
        SplitTestAndTrain t = iris.splitTestAndTrain(0.8);
        o.fit(t.getTrain());
        log.info("Evaluate model....");
        Evaluation eval = new Evaluation(3);
        eval.eval(t.getTest().getLabels(),o.output(t.getTest().getFeatureMatrix(), true));
        log.info(eval.stats());

    }


    @Test
    public void testBinary() {

        Nd4j.MAX_ELEMENTS_PER_SLICE = Integer.MAX_VALUE;
        Nd4j.MAX_SLICES_TO_PRINT = Integer.MAX_VALUE;
        Nd4j.dtype = DataBuffer.Type.DOUBLE;
        INDArray data = Nd4j.create(new double[][]
                {{1,1,1,0,0,0},
                        {1,0,1,0,0,0},
                        {1,1,1,0,0,0},
                        {0,0,1,1,1,0},
                        {0,0,1,1,0,0},
                        {0,0,1,1,1,0}});

        INDArray data2 = Nd4j.create(new double[][]
                {{1, 0},
                        {1, 0},
                        {1, 0},
                        {0, 1},
                        {0, 1},
                        {0, 1}});

        DataSet dataset = new DataSet(data,data2);
        NeuralNetConfiguration neuralNetConfiguration = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .seed(123)
                .iterations(200)
                .learningRate(1e-2)
                .layer(new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder()
                        .nIn(6).nOut(2)
                        .weightInit(WeightInit.ZERO)
                        .updater(Updater.SGD)
                        .activation("softmax")
                        .lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .build())
                .build();

        OutputLayer o = LayerFactories.getFactory(neuralNetConfiguration).create(neuralNetConfiguration);
        o.setListeners(new ScoreIterationListener(1));
        o.fit(dataset);


    }


    @Test
    public void testIris() {
        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.LINE_GRADIENT_DESCENT)
                .iterations(5)
                .learningRate(1e-1)
                .layer(new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder()
                        .nIn(4).nOut(3)
                        .weightInit(WeightInit.XAVIER)
                        .activation("softmax")
                        .lossFunction(LossFunctions.LossFunction.MCXENT).build())
                .build();

        OutputLayer l = LayerFactories.getFactory(conf.getLayer()).create(conf, Arrays.<IterationListener>asList(new ScoreIterationListener(1)),0);
        DataSetIterator iter = new IrisDataSetIterator(150, 150);


        DataSet next = iter.next();
        next.shuffle();
        SplitTestAndTrain trainTest = next.splitTestAndTrain(110);
        trainTest.getTrain().normalizeZeroMeanZeroUnitVariance();
        l.fit(trainTest.getTrain());


        DataSet test = trainTest.getTest();
        test.normalizeZeroMeanZeroUnitVariance();
        Evaluation eval = new Evaluation();
        INDArray output = l.output(test.getFeatureMatrix());
        eval.eval(test.getLabels(),output);
        log.info("Score " +eval.stats());


    }

    @Test
    public void testSetParams() {
        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.LINE_GRADIENT_DESCENT)
                .iterations(100)
                .learningRate(1e-1)
                .layer(new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder()
                        .nIn(4).nOut(3)
                        .weightInit(WeightInit.ZERO)
                        .activation("softmax")
                        .lossFunction(LossFunctions.LossFunction.MCXENT).build())
                .build();

        OutputLayer l = LayerFactories.getFactory(conf.getLayer()).create(conf, Arrays.<IterationListener>asList(new ScoreIterationListener(1)),0);
        INDArray params = l.params();
        l.setParams(params);
        assertEquals(params,l.params());
    }


}
