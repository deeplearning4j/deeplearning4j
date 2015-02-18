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

package org.deeplearning4j.models.featuredetectors.rbm;

import static org.junit.Assert.*;

import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.datasets.fetchers.IrisDataFetcher;
import org.deeplearning4j.datasets.fetchers.MnistDataFetcher;
import org.deeplearning4j.datasets.iterator.impl.LFWDataSetIterator;
import org.deeplearning4j.distributions.Distributions;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.LayerFactory;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.factory.DefaultLayerFactory;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ComposableIterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.plot.NeuralNetPlotter;
import org.deeplearning4j.plot.iterationlistener.NeuralNetPlotterIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


/**
 * Created by agibsonccc on 8/27/14.
 */
public class RBMTests {
    private static final Logger log = LoggerFactory.getLogger(RBMTests.class);

    @Test
    public void testLfw() {
        LFWDataSetIterator iter = new LFWDataSetIterator(10,10,28,28);
        DataSet d = iter.next();

        d.normalizeZeroMeanZeroUnitVariance();

        int nOut = 600;
        RandomGenerator rng = new MersenneTwister(123);
        LayerFactory layerFactory = LayerFactories.getFactory(RBM.class);

        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .hiddenUnit(RBM.HiddenUnit.RECTIFIED).weightInit(WeightInit.VI)
                .iterationListener(new ScoreIterationListener(1))
                .visibleUnit(RBM.VisibleUnit.GAUSSIAN).layerFactory(layerFactory)
                .optimizationAlgo(OptimizationAlgorithm.ITERATION_GRADIENT_DESCENT)
                .lossFunction(LossFunctions.LossFunction.RMSE_XENT).rng(rng)
                .learningRate(1e-3f)
                .nIn(d.numInputs()).nOut(nOut).build();

        RBM rbm = layerFactory.create(conf);

        rbm.fit(d.getFeatureMatrix());




    }

    @Test
    public void testIris() {
        IrisDataFetcher fetcher = new IrisDataFetcher();
        fetcher.fetch(150);
        DataSet d = fetcher.next();
        d.normalizeZeroMeanZeroUnitVariance();
        RandomGenerator g = new MersenneTwister(123);

        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder().lossFunction(LossFunctions.LossFunction.RMSE_XENT)
                .visibleUnit(RBM.VisibleUnit.GAUSSIAN).hiddenUnit(RBM.HiddenUnit.RECTIFIED).learningRate(1e-1f)
                .nIn(d.numInputs()).rng(g).
                        nOut(3).build();


        LayerFactory layerFactory = LayerFactories.getFactory(RBM.class);
        RBM r = layerFactory.create(conf);
        r.fit(d.getFeatureMatrix());

    }



    @Test
    public void testBasic() {
        float[][] data = new float[][]
                {
                        {1,1,1,0,0,0},
                        {1,0,1,0,0,0},
                        {1,1,1,0,0,0},
                        {0,0,1,1,1,0},
                        {0,0,1,1,0,0},
                        {0,0,1,1,1,0},
                        {0,0,1,1,1,0}
                };


        INDArray input = Nd4j.create(data);
        LayerFactory layerFactory = LayerFactories.getFactory(RBM.class);

        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .lossFunction(LossFunctions.LossFunction.RMSE_XENT)
                .learningRate(1e-1f).nIn(6).nOut(4).build();
        RBM rbm = layerFactory.create(conf);
        rbm.fit(input);



    }

    @Test
    public void testMnist() throws Exception {
        MnistDataFetcher fetcher = new MnistDataFetcher(true);
        RandomGenerator gen = new MersenneTwister(123);
        Nd4j.ENFORCE_NUMERICAL_STABILITY = true;
        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .iterations(30).constrainGradientToUnitNorm(true).weightInit(WeightInit.DISTRIBUTION).dist(Distributions.normal(gen,1e-5))
                .optimizationAlgo(OptimizationAlgorithm.ITERATION_GRADIENT_DESCENT)
                .iterationListener(new ComposableIterationListener(new NeuralNetPlotterIterationListener(10),new ScoreIterationListener(5)))
                        .lossFunction(LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY).rng(gen)
                        .learningRate(1e-1f).nIn(784).nOut(600).build();

        fetcher.fetch(10);
        DataSet d2 = fetcher.next();

        INDArray input = d2.getFeatureMatrix();

        LayerFactory layerFactory = LayerFactories.getFactory(RBM.class);
        RBM rbm = layerFactory.create(conf);

        rbm.fit(input);






    }


    @Test
    public void testSetGetParams() {
        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .lossFunction(LossFunctions.LossFunction.RMSE_XENT)
                .learningRate(1e-1f).nIn(6).nOut(4).build();
        LayerFactory layerFactory = new DefaultLayerFactory(RBM.class);
        RBM rbm = layerFactory.create(conf);
        INDArray rand2 = Nd4j.rand(new int[]{1, rbm.numParams()});
        rbm.setParams(rand2);
        INDArray getParams = rbm.params();
        assertEquals(rand2,getParams);
    }





    @Test
    public void testCg() {
        float[][] data = new float[][]
                {
                        {1,1,1,0,0,0},
                        {1,0,1,0,0,0},
                        {1,1,1,0,0,0},
                        {0,0,1,1,1,0},
                        {0,0,1,1,0,0},
                        {0,0,1,1,1,0},
                        {0,0,1,1,1,0}
                };


        INDArray input = Nd4j.create(data);

        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .lossFunction(LossFunctions.LossFunction.RMSE_XENT)
                .learningRate(1e-1f).nIn(6).nOut(4).build();
        LayerFactory layerFactory = new DefaultLayerFactory(RBM.class);
        RBM rbm = layerFactory.create(conf);
        rbm.setInput(input);
        double value = rbm.score();
        rbm.contrastiveDivergence();
        value = rbm.score();



    }

    @Test
    public void testGradient() {
        float[][] data = new float[][]
                {
                        {1,1,1,0,0,0},
                        {1,0,1,0,0,0},
                        {1,1,1,0,0,0},
                        {0,0,1,1,1,0},
                        {0,0,1,1,0,0},
                        {0,0,1,1,1,0},
                        {0,0,1,1,1,0}
                };


        INDArray input = Nd4j.create(data);

        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .lossFunction(LossFunctions.LossFunction.RMSE_XENT)
                .learningRate(1e-1f).nIn(6).nOut(4).build();
        LayerFactory layerFactory = new DefaultLayerFactory(RBM.class);
        RBM rbm = layerFactory.create(conf);
        rbm.setInput(input);
        double value = rbm.score();


        Gradient grad2 = rbm.gradient();
        rbm.fit(input);

    }



}
