/*
 * Copyright 2015 Skymind,Inc.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

package org.deeplearning4j.models.layers;

import java.util.Arrays;

import org.deeplearning4j.datasets.fetchers.MnistDataFetcher;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.override.ConfOverride;
import org.deeplearning4j.nn.layers.OutputLayer;
import org.deeplearning4j.nn.layers.convolution.ConvolutionDownSampleLayer;
import org.deeplearning4j.nn.layers.convolution.preprocessor.ConvolutionInputPreProcessor;
import org.deeplearning4j.nn.layers.convolution.preprocessor.ConvolutionPostProcessor;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Created by agibsonccc on 9/7/14.
 */
public class ConvolutionDownSampleLayerTest {
    private static final Logger log = LoggerFactory.getLogger(ConvolutionDownSampleLayerTest.class);

    @Test
    public void testConvolution() throws Exception {
        boolean switched = false;
        if(Nd4j.dtype == DataBuffer.Type.FLOAT) {
            Nd4j.dtype = DataBuffer.Type.DOUBLE;
            switched = true;
        }
        MnistDataFetcher data = new MnistDataFetcher(true);
        data.fetch(2);
        DataSet d = data.next();

        d.setFeatures(d.getFeatureMatrix().reshape(2, 1, 28, 28));
        NeuralNetConfiguration n = new NeuralNetConfiguration.Builder()
                .filterSize(2, 1, 2, 2).layer(new org.deeplearning4j.nn.conf.layers.ConvolutionDownSampleLayer()).build();

        ConvolutionDownSampleLayer c = LayerFactories.getFactory(n.getLayer()).create(n);

        if(switched) {
            Nd4j.dtype = DataBuffer.Type.FLOAT;
        }

    }

    @Test
    public void testMultiLayer() {

        int batchSize = 110;
        /**
         *
         */
        Nd4j.MAX_ELEMENTS_PER_SLICE = Integer.MAX_VALUE;
        Nd4j.MAX_ELEMENTS_PER_SLICE = Integer.MAX_VALUE;
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.ITERATION_GRADIENT_DESCENT).momentum(0.9)
                .dist(new UniformDistribution(1e-5, 1e-1)).constrainGradientToUnitNorm(true)
                .iterations(1000).convolutionType(org.deeplearning4j.nn.conf.layers.ConvolutionDownSampleLayer.ConvolutionType.NONE)
                .activationFunction("tanh").filterSize(1, 1, 2, 2)
                .nIn(4).nOut(3).batchSize(batchSize)
                .layer(new org.deeplearning4j.nn.conf.layers.ConvolutionDownSampleLayer())
                .list(3)
                .preProcessor(0, new ConvolutionPostProcessor()).inputPreProcessor(0, new ConvolutionInputPreProcessor(2, 2))
                .preProcessor(1, new ConvolutionPostProcessor()).inputPreProcessor(1, new ConvolutionInputPreProcessor(3, 3))
                .hiddenLayerSizes(new int[]{4, 16})
                .override(0, new ConfOverride() {

                    @Override
                    public void overrideLayer(int i, NeuralNetConfiguration.Builder builder) {
                        if (i == 0)
                            builder.filterSize(1, 1, 2, 2);

                    }
                })    .override(1, new ConfOverride() {

                    @Override
                    public void overrideLayer(int i, NeuralNetConfiguration.Builder builder) {


                    }
                })
                .override(2, new ConfOverride() {

                    @Override
                    public void overrideLayer(int i, NeuralNetConfiguration.Builder builder) {
                        if (i == 2) {
                            builder.activationFunction("softmax");
                            builder.weightInit(WeightInit.ZERO);
                            builder.layer(new org.deeplearning4j.nn.conf.layers.OutputLayer());
                            builder.lossFunction(LossFunctions.LossFunction.MCXENT);
                        }
                    }
                }).build();

        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.setListeners(Arrays.<IterationListener>asList(new ScoreIterationListener(1)));
        DataSetIterator iter = new IrisDataSetIterator(150, 150);


        org.nd4j.linalg.dataset.DataSet next = iter.next();
        next.normalizeZeroMeanZeroUnitVariance();
        SplitTestAndTrain trainTest = next.splitTestAndTrain(110);
        /**
         * Likely cause: shape[0] mis match on the filter size and the input batch size.
         * Likely need to make a little more general.
         */
        network.fit(trainTest.getTrain());


        //org.nd4j.linalg.dataset.DataSet test = trainTest.getTest();
        Evaluation eval = new Evaluation();
        INDArray output = network.output(trainTest.getTest().getFeatureMatrix());
        eval.eval(trainTest.getTest().getLabels(),output);
        log.info("Score " +eval.stats());

    }


}
