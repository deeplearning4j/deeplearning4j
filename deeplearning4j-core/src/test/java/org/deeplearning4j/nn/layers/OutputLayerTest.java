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

package org.deeplearning4j.nn.layers;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.TestUtils;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToRnnPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.RnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.layers.recurrent.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.AdaGrad;
import org.nd4j.linalg.learning.config.NoOp;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Collections;
import java.util.Random;

import static org.junit.Assert.*;

/**
 * Created by agibsonccc on 9/1/14.
 */
@Slf4j
public class OutputLayerTest extends BaseDL4JTest {

    @Test
    public void testIris2() {
        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                        .updater(new Sgd(1e-1))
                        .layer(new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder().nIn(4).nOut(3)
                                        .weightInit(WeightInit.XAVIER).activation(Activation.SOFTMAX)
                                        .lossFunction(LossFunctions.LossFunction.MCXENT).build())
                        .build();

        int numParams = conf.getLayer().initializer().numParams(conf);
        INDArray params = Nd4j.create(1, numParams);
        OutputLayer l = (OutputLayer) conf.getLayer().instantiate(conf,
                        Collections.<IterationListener>singletonList(new ScoreIterationListener(1)), 0, params, true);
        l.setBackpropGradientsViewArray(Nd4j.create(1, params.length()));
        DataSetIterator iter = new IrisDataSetIterator(150, 150);


        DataSet next = iter.next();
        next.shuffle();
        SplitTestAndTrain trainTest = next.splitTestAndTrain(110);
        trainTest.getTrain().normalizeZeroMeanZeroUnitVariance();
        for( int i=0; i<10; i++ ) {
            l.fit(trainTest.getTrain());
        }


        DataSet test = trainTest.getTest();
        test.normalizeZeroMeanZeroUnitVariance();
        Evaluation eval = new Evaluation();
        INDArray output = l.output(test.getFeatureMatrix());
        eval.eval(test.getLabels(), output);
        log.info("Score " + eval.stats());


    }

    @Test
    public void test3() {

        org.nd4j.linalg.dataset.api.iterator.DataSetIterator iter = new IrisDataSetIterator(150, 150);
        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder().miniBatch(false)
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                        .layer(new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(
                                        LossFunctions.LossFunction.MCXENT).nIn(4).nOut(3).activation(Activation.SOFTMAX)
                                                        .weightInit(WeightInit.XAVIER).build())
                        .build();

        int numParams = conf.getLayer().initializer().numParams(conf);
        INDArray params = Nd4j.create(1, numParams);
        org.deeplearning4j.nn.layers.OutputLayer layer = (org.deeplearning4j.nn.layers.OutputLayer) conf.getLayer()
                        .instantiate(conf, null, 0, params, true);
        layer.setBackpropGradientsViewArray(Nd4j.create(1, params.length()));

        DataSet next = iter.next();
        next.normalizeZeroMeanZeroUnitVariance();
        layer.setListeners(new ScoreIterationListener(1));

        for( int i=0; i<3; i++ ) {
            layer.fit(next);
        }


    }

    @Test
    public void testWeightsDifferent() {
        Nd4j.MAX_ELEMENTS_PER_SLICE = Integer.MAX_VALUE;
        Nd4j.MAX_SLICES_TO_PRINT = Integer.MAX_VALUE;

        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                        .miniBatch(false).seed(123)
                        .updater(new AdaGrad(1e-1))
                        .layer(new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder().nIn(4).nOut(3)
                                        .weightInit(WeightInit.XAVIER)
                                        .lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                                        .activation(Activation.SOFTMAX).build())
                        .build();

        int numParams = conf.getLayer().initializer().numParams(conf);
        INDArray params = Nd4j.create(1, numParams);
        OutputLayer o = (OutputLayer) conf.getLayer().instantiate(conf, null, 0, params, true);
        o.setBackpropGradientsViewArray(Nd4j.create(1, params.length()));


        int numSamples = 150;
        int batchSize = 150;


        DataSetIterator iter = new IrisDataSetIterator(batchSize, numSamples);
        DataSet iris = iter.next(); // Loads data into generator and format consumable for NN
        iris.normalizeZeroMeanZeroUnitVariance();
        o.setListeners(new ScoreIterationListener(1));
        SplitTestAndTrain t = iris.splitTestAndTrain(0.8);
        for( int i=0; i<1000; i++ ){
            o.fit(t.getTrain());
        }
        log.info("Evaluate model....");
        Evaluation eval = new Evaluation(3);
        eval.eval(t.getTest().getLabels(), o.output(t.getTest().getFeatureMatrix(), true));
        log.info(eval.stats());

    }


    @Test
    public void testBinary() {

        Nd4j.MAX_ELEMENTS_PER_SLICE = Integer.MAX_VALUE;
        Nd4j.MAX_SLICES_TO_PRINT = Integer.MAX_VALUE;
        DataBuffer.Type initialType = Nd4j.dataType();
        DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE);
        INDArray data = Nd4j.create(new double[][] {{1, 1, 1, 0, 0, 0}, {1, 0, 1, 0, 0, 0}, {1, 1, 1, 0, 0, 0},
                        {0, 0, 1, 1, 1, 0}, {0, 0, 1, 1, 0, 0}, {0, 0, 1, 1, 1, 0}});

        INDArray data2 = Nd4j.create(new double[][] {{1, 0}, {1, 0}, {1, 0}, {0, 1}, {0, 1}, {0, 1}});

        DataSet dataset = new DataSet(data, data2);
        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                        .seed(123)
                        .updater(new Sgd(1e-2))
                        .layer(new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder().nIn(6).nOut(2)
                                        .weightInit(WeightInit.ZERO).activation(Activation.SOFTMAX)
                                        .lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).build())
                        .build();

        int numParams = conf.getLayer().initializer().numParams(conf);
        INDArray params = Nd4j.create(1, numParams);
        OutputLayer o = (OutputLayer) conf.getLayer().instantiate(conf, null, 0, params, true);
        o.setBackpropGradientsViewArray(Nd4j.create(1, params.length()));

        o.setListeners(new ScoreIterationListener(1));
        for( int i=0; i<200; i++ ) {
            o.fit(dataset);
        }

        DataTypeUtil.setDTypeForContext(initialType);
    }


    @Test
    public void testIris() {
        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                        .optimizationAlgo(OptimizationAlgorithm.LINE_GRADIENT_DESCENT).updater(new Sgd(1e-1))
                        .layer(new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder().nIn(4).nOut(3)
                                        .weightInit(WeightInit.XAVIER).activation(Activation.SOFTMAX)
                                        .lossFunction(LossFunctions.LossFunction.MCXENT).build())
                        .build();

        int numParams = conf.getLayer().initializer().numParams(conf);
        INDArray params = Nd4j.create(1, numParams);
        OutputLayer l = (OutputLayer) conf.getLayer().instantiate(conf,
                        Collections.<IterationListener>singletonList(new ScoreIterationListener(1)), 0, params, true);
        l.setBackpropGradientsViewArray(Nd4j.create(1, params.length()));
        DataSetIterator iter = new IrisDataSetIterator(150, 150);


        DataSet next = iter.next();
        next.shuffle();
        SplitTestAndTrain trainTest = next.splitTestAndTrain(110);
        trainTest.getTrain().normalizeZeroMeanZeroUnitVariance();
        for( int i=0; i<5; i++ ) {
            l.fit(trainTest.getTrain());
        }


        DataSet test = trainTest.getTest();
        test.normalizeZeroMeanZeroUnitVariance();
        Evaluation eval = new Evaluation();
        INDArray output = l.output(test.getFeatureMatrix());
        eval.eval(test.getLabels(), output);
        log.info("Score " + eval.stats());


    }

    @Test
    public void testSetParams() {
        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                        .optimizationAlgo(OptimizationAlgorithm.LINE_GRADIENT_DESCENT)
                        .updater(new Sgd(1e-1))
                        .layer(new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder().nIn(4).nOut(3)
                                        .weightInit(WeightInit.ZERO).activation(Activation.SOFTMAX)
                                        .lossFunction(LossFunctions.LossFunction.MCXENT).build())
                        .build();

        int numParams = conf.getLayer().initializer().numParams(conf);
        INDArray params = Nd4j.create(1, numParams);
        OutputLayer l = (OutputLayer) conf.getLayer().instantiate(conf,
                        Collections.<IterationListener>singletonList(new ScoreIterationListener(1)), 0, params, true);
        params = l.params();
        l.setParams(params);
        assertEquals(params, l.params());
    }

    @Test
    public void testOutputLayersRnnForwardPass() {
        //Test output layer with RNNs (
        //Expect all outputs etc. to be 2d
        int nIn = 2;
        int nOut = 5;
        int layerSize = 4;
        int timeSeriesLength = 6;
        int miniBatchSize = 3;

        Random r = new Random(12345L);
        INDArray input = Nd4j.zeros(miniBatchSize, nIn, timeSeriesLength);
        for (int i = 0; i < miniBatchSize; i++) {
            for (int j = 0; j < nIn; j++) {
                for (int k = 0; k < timeSeriesLength; k++) {
                    input.putScalar(new int[] {i, j, k}, r.nextDouble() - 0.5);
                }
            }
        }

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(12345L).list()
                        .layer(0, new GravesLSTM.Builder().nIn(nIn).nOut(layerSize).weightInit(WeightInit.DISTRIBUTION)
                                        .dist(new NormalDistribution(0, 1)).activation(Activation.TANH)
                                        .updater(new NoOp()).build())
                        .layer(1, new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(LossFunction.MCXENT)
                                        .activation(Activation.SOFTMAX).nIn(layerSize).nOut(nOut)
                                        .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0, 1))
                                        .updater(new NoOp()).build())
                        .inputPreProcessor(1, new RnnToFeedForwardPreProcessor()).build();

        MultiLayerNetwork mln = new MultiLayerNetwork(conf);
        mln.init();

        INDArray out2d = mln.feedForward(input).get(2);
        assertArrayEquals(out2d.shape(), new int[] {miniBatchSize * timeSeriesLength, nOut});

        INDArray out = mln.output(input);
        assertArrayEquals(out.shape(), new int[] {miniBatchSize * timeSeriesLength, nOut});

        INDArray preout = mln.preOutput(input);
        assertArrayEquals(preout.shape(), new int[] {miniBatchSize * timeSeriesLength, nOut});

        //As above, but for RnnOutputLayer. Expect all activations etc. to be 3d

        MultiLayerConfiguration confRnn = new NeuralNetConfiguration.Builder().seed(12345L).list()
                        .layer(0, new GravesLSTM.Builder().nIn(nIn).nOut(layerSize).weightInit(WeightInit.DISTRIBUTION)
                                        .dist(new NormalDistribution(0, 1)).activation(Activation.TANH)
                                        .updater(new NoOp()).build())
                        .layer(1, new org.deeplearning4j.nn.conf.layers.RnnOutputLayer.Builder(LossFunction.MCXENT)
                                        .activation(Activation.SOFTMAX).nIn(layerSize).nOut(nOut)
                                        .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0, 1))
                                        .updater(new NoOp()).build())
                        .build();

        MultiLayerNetwork mlnRnn = new MultiLayerNetwork(confRnn);
        mln.init();

        INDArray out3d = mlnRnn.feedForward(input).get(2);
        assertArrayEquals(out3d.shape(), new int[] {miniBatchSize, nOut, timeSeriesLength});

        INDArray outRnn = mlnRnn.output(input);
        assertArrayEquals(outRnn.shape(), new int[] {miniBatchSize, nOut, timeSeriesLength});

        INDArray preoutRnn = mlnRnn.preOutput(input);
        assertArrayEquals(preoutRnn.shape(), new int[] {miniBatchSize, nOut, timeSeriesLength});
    }

    @Test
    public void testRnnOutputLayerIncEdgeCases() {
        //Basic test + test edge cases: timeSeriesLength==1, miniBatchSize==1, both
        int[] tsLength = {5, 1, 5, 1};
        int[] miniBatch = {7, 7, 1, 1};
        int nIn = 3;
        int nOut = 6;
        int layerSize = 4;

        FeedForwardToRnnPreProcessor proc = new FeedForwardToRnnPreProcessor();

        for (int t = 0; t < tsLength.length; t++) {
            Nd4j.getRandom().setSeed(12345);
            int timeSeriesLength = tsLength[t];
            int miniBatchSize = miniBatch[t];

            Random r = new Random(12345L);
            INDArray input = Nd4j.zeros(miniBatchSize, nIn, timeSeriesLength);
            for (int i = 0; i < miniBatchSize; i++) {
                for (int j = 0; j < nIn; j++) {
                    for (int k = 0; k < timeSeriesLength; k++) {
                        input.putScalar(new int[] {i, j, k}, r.nextDouble() - 0.5);
                    }
                }
            }
            INDArray labels3d = Nd4j.zeros(miniBatchSize, nOut, timeSeriesLength);
            for (int i = 0; i < miniBatchSize; i++) {
                for (int j = 0; j < timeSeriesLength; j++) {
                    int idx = r.nextInt(nOut);
                    labels3d.putScalar(new int[] {i, idx, j}, 1.0f);
                }
            }
            INDArray labels2d = proc.backprop(labels3d, miniBatchSize);

            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(12345L).list()
                            .layer(0, new GravesLSTM.Builder().nIn(nIn).nOut(layerSize)
                                            .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0, 1))
                                            .activation(Activation.TANH).updater(new NoOp()).build())
                            .layer(1, new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(LossFunction.MCXENT)
                                            .activation(Activation.SOFTMAX).nIn(layerSize).nOut(nOut)
                                            .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0, 1))
                                            .updater(new NoOp()).build())
                            .inputPreProcessor(1, new RnnToFeedForwardPreProcessor()).pretrain(false).backprop(true)
                            .build();

            MultiLayerNetwork mln = new MultiLayerNetwork(conf);
            mln.init();

            INDArray out2d = mln.feedForward(input).get(2);
            INDArray out3d = proc.preProcess(out2d, miniBatchSize);

            MultiLayerConfiguration confRnn = new NeuralNetConfiguration.Builder().seed(12345L).list()
                            .layer(0, new GravesLSTM.Builder().nIn(nIn).nOut(layerSize)
                                            .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0, 1))
                                            .activation(Activation.TANH).updater(new NoOp()).build())
                            .layer(1, new org.deeplearning4j.nn.conf.layers.RnnOutputLayer.Builder(LossFunction.MCXENT)
                                            .activation(Activation.SOFTMAX).nIn(layerSize).nOut(nOut)
                                            .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0, 1))
                                            .updater(new NoOp()).build())
                            .pretrain(false).backprop(true).build();

            MultiLayerNetwork mlnRnn = new MultiLayerNetwork(confRnn);
            mlnRnn.init();

            INDArray outRnn = mlnRnn.feedForward(input).get(2);

            mln.setLabels(labels2d);
            mlnRnn.setLabels(labels3d);


            mln.computeGradientAndScore();
            mlnRnn.computeGradientAndScore();

            //score is average over all examples.
            //However: OutputLayer version has miniBatch*timeSeriesLength "examples" (after reshaping)
            //RnnOutputLayer has miniBatch examples
            //Hence: expect difference in scores by factor of timeSeriesLength
            double score = mln.score() * timeSeriesLength;
            double scoreRNN = mlnRnn.score();

            assertTrue(!Double.isNaN(score));
            assertTrue(!Double.isNaN(scoreRNN));

            double relError = Math.abs(score - scoreRNN) / (Math.abs(score) + Math.abs(scoreRNN));
            System.out.println(relError);
            assertTrue(relError < 1e-6);

            //Check labels and inputs for output layer:
            OutputLayer ol = (OutputLayer) mln.getOutputLayer();
            assertArrayEquals(ol.getInput().shape(), new int[] {miniBatchSize * timeSeriesLength, layerSize});
            assertArrayEquals(ol.getLabels().shape(), new int[] {miniBatchSize * timeSeriesLength, nOut});

            RnnOutputLayer rnnol = (RnnOutputLayer) mlnRnn.getOutputLayer();
            //assertArrayEquals(rnnol.getInput().shape(),new int[]{miniBatchSize,layerSize,timeSeriesLength});
            //Input may be set by BaseLayer methods. Thus input may end up as reshaped 2d version instead of original 3d version.
            //Not ideal, but everything else works.
            assertArrayEquals(rnnol.getLabels().shape(), new int[] {miniBatchSize, nOut, timeSeriesLength});

            //Check shapes of output for both:
            assertArrayEquals(out2d.shape(), new int[] {miniBatchSize * timeSeriesLength, nOut});

            INDArray out = mln.output(input);
            assertArrayEquals(out.shape(), new int[] {miniBatchSize * timeSeriesLength, nOut});

            INDArray preout = mln.preOutput(input);
            assertArrayEquals(preout.shape(), new int[] {miniBatchSize * timeSeriesLength, nOut});


            INDArray outFFRnn = mlnRnn.feedForward(input).get(2);
            assertArrayEquals(outFFRnn.shape(), new int[] {miniBatchSize, nOut, timeSeriesLength});

            INDArray outRnn2 = mlnRnn.output(input);
            assertArrayEquals(outRnn2.shape(), new int[] {miniBatchSize, nOut, timeSeriesLength});

            INDArray preoutRnn = mlnRnn.preOutput(input);
            assertArrayEquals(preoutRnn.shape(), new int[] {miniBatchSize, nOut, timeSeriesLength});
        }
    }


    @Test
    public void testCompareRnnOutputRnnLoss(){
        Nd4j.getRandom().setSeed(12345);

        int timeSeriesLength = 4;
        int nIn = 5;
        int layerSize = 6;
        int nOut = 6;
        int miniBatchSize = 3;

        MultiLayerConfiguration conf1 =
                new NeuralNetConfiguration.Builder().seed(12345L)
                        .updater(new NoOp())
                        .list()
                        .layer(new LSTM.Builder().nIn(nIn).nOut(layerSize).activation(Activation.TANH)
                                .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0, 1.0))
                                .updater(new NoOp()).build())
                        .layer(new DenseLayer.Builder().nIn(layerSize).nOut(nOut).activation(Activation.IDENTITY).build())
                        .layer(new RnnLossLayer.Builder(LossFunction.MCXENT)
                                .activation(Activation.SOFTMAX)
                                .build())
                        .pretrain(false).backprop(true).build();

        MultiLayerNetwork mln = new MultiLayerNetwork(conf1);
        mln.init();


        MultiLayerConfiguration conf2 =
                new NeuralNetConfiguration.Builder().seed(12345L)
                        .updater(new NoOp())
                        .list()
                        .layer(new LSTM.Builder().nIn(nIn).nOut(layerSize).activation(Activation.TANH)
                                .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0, 1.0))
                                .updater(new NoOp()).build())
                        .layer(new org.deeplearning4j.nn.conf.layers.RnnOutputLayer.Builder(LossFunction.MCXENT)
                                .activation(Activation.SOFTMAX)
                                .nIn(layerSize).nOut(nOut)
                                .build())
                        .pretrain(false).backprop(true).build();

        MultiLayerNetwork mln2 = new MultiLayerNetwork(conf2);
        mln2.init();

        mln2.setParams(mln.params());

        INDArray in = Nd4j.rand(new int[]{miniBatchSize, nIn, timeSeriesLength});

        INDArray out1 = mln.output(in);
        INDArray out2 = mln.output(in);

        assertEquals(out1, out2);

        Random r = new Random(12345);
        INDArray labels = Nd4j.create(miniBatchSize, nOut, timeSeriesLength);
        for( int i=0; i<miniBatchSize; i++ ){
            for( int j=0; j<timeSeriesLength; j++ ){
                labels.putScalar(i, r.nextInt(nOut), j, 1.0);
            }
        }

        mln.setInput(in);
        mln.setLabels(labels);

        mln2.setInput(in);
        mln2.setLabels(labels);

        mln.computeGradientAndScore();
        mln2.computeGradientAndScore();

        assertEquals(mln.gradient().gradient(), mln2.gradient().gradient());
        assertEquals(mln.score(), mln2.score(), 1e-6);

        TestUtils.testModelSerialization(mln);
    }



    @Test
    public void testCnnOutputLayer(){

        for(Activation a : new Activation[]{Activation.TANH, Activation.SELU}) {
            //Check that (A+identity) is equal to (identity+A), for activation A
            //i.e., should get same output and weight gradients for both

            MultiLayerConfiguration conf1 =
                    new NeuralNetConfiguration.Builder().seed(12345L)
                            .updater(new NoOp())
                            .convolutionMode(ConvolutionMode.Same)
                            .list()
                            .layer(new ConvolutionLayer.Builder().nIn(3).nOut(4).activation(Activation.IDENTITY)
                                    .kernelSize(2,2).stride(1,1)
                                    .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0, 1.0))
                                    .updater(new NoOp()).build())
                            .layer(new CnnLossLayer.Builder(LossFunction.MSE)
                                    .activation(a)
                                    .build())
                            .build();

            MultiLayerConfiguration conf2 =
                    new NeuralNetConfiguration.Builder().seed(12345L)
                            .updater(new NoOp())
                            .convolutionMode(ConvolutionMode.Same)
                            .list()
                            .layer(new ConvolutionLayer.Builder().nIn(3).nOut(4).activation(a)
                                    .kernelSize(2,2).stride(1,1)
                                    .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0, 1.0))
                                    .updater(new NoOp()).build())
                            .layer(new CnnLossLayer.Builder(LossFunction.MSE)
                                    .activation(Activation.IDENTITY)
                                    .build())
                            .build();

            MultiLayerNetwork mln = new MultiLayerNetwork(conf1);
            mln.init();

            MultiLayerNetwork mln2 = new MultiLayerNetwork(conf2);
            mln2.init();


            mln2.setParams(mln.params());


            INDArray in = Nd4j.rand(new int[]{3,3,5,5});

            INDArray out1 = mln.output(in);
            INDArray out2 = mln2.output(in);

            assertEquals(out1, out2);

            INDArray labels = Nd4j.rand(out1.shape());

            mln.setInput(in);
            mln.setLabels(labels);

            mln2.setInput(in);
            mln2.setLabels(labels);

            mln.computeGradientAndScore();
            mln2.computeGradientAndScore();

            assertEquals(mln.score(), mln2.score(), 1e-6);
            assertEquals(mln.gradient().gradient(), mln2.gradient().gradient());

            //Also check computeScoreForExamples
            INDArray in2a = Nd4j.rand(new int[]{1,3,5,5});
            INDArray labels2a = Nd4j.rand(new int[]{1,4,5,5});

            INDArray in2 = Nd4j.concat(0, in2a, in2a);
            INDArray labels2 = Nd4j.concat(0, labels2a, labels2a);

            INDArray s = mln.scoreExamples(new DataSet(in2, labels2), false);
            assertArrayEquals(new int[]{2,1}, s.shape());
            assertEquals(s.getDouble(0), s.getDouble(1), 1e-6);

            TestUtils.testModelSerialization(mln);
        }
    }

    @Test
    public void testCnnOutputLayerSoftmax(){
        //Check that softmax is applied depth-wise

        MultiLayerConfiguration conf =
                new NeuralNetConfiguration.Builder().seed(12345L)
                        .updater(new NoOp())
                        .convolutionMode(ConvolutionMode.Same)
                        .list()
                        .layer(new ConvolutionLayer.Builder().nIn(3).nOut(4).activation(Activation.IDENTITY)
                                .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0, 1.0))
                                .updater(new NoOp()).build())
                        .layer(new CnnLossLayer.Builder(LossFunction.MSE)
                                .activation(Activation.SOFTMAX)
                                .build())
                        .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        INDArray in = Nd4j.rand(new int[]{2,3,4,5});
        INDArray out = net.output(in);

        double min = out.minNumber().doubleValue();
        double max = out.maxNumber().doubleValue();

        assertTrue(min >= 0 && max <= 1.0);

        INDArray sum = out.sum(1);
        assertEquals(Nd4j.ones(2,4,5), sum);

    }
}
