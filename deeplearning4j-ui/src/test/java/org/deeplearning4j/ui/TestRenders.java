package org.deeplearning4j.ui;

import org.deeplearning4j.datasets.fetchers.MnistDataFetcher;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.deeplearning4j.nn.layers.feedforward.autoencoder.AutoEncoder;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.params.PretrainParamInitializer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.plot.PlotFilters;
import org.deeplearning4j.ui.activation.UpdateActivationIterationListener;
import org.deeplearning4j.ui.renders.UpdateFilterIterationListener;
import org.deeplearning4j.ui.weights.HistogramIterationListener;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.Arrays;
import java.util.Collections;


/**
 * @author Adam Gibson
 */
@Ignore
public class TestRenders extends BaseUiServerTest {
    @Test
    public void renderSetup() throws Exception {
        MnistDataFetcher fetcher = new MnistDataFetcher(true);
        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder().momentum(0.9f)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .iterations(100)
                .learningRate(1e-1f)
                .layer(new org.deeplearning4j.nn.conf.layers.AutoEncoder.Builder()
                        .nIn(784).nOut(600)
                        .corruptionLevel(0.6)
                        .lossFunction(LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY).build())
                .build();


        fetcher.fetch(100);
        DataSet d2 = fetcher.next();

        INDArray input = d2.getFeatureMatrix();
        PlotFilters filters = new PlotFilters(input,new int[]{10,10},new int[]{0,0},new int[]{28,28});
        AutoEncoder da = LayerFactories.getFactory(conf.getLayer()).create(conf, Arrays.<IterationListener>asList(new ScoreIterationListener(1)
                ,new UpdateFilterIterationListener(filters,Collections.singletonList(PretrainParamInitializer.WEIGHT_KEY),1)),0);
        da.setParams(da.params());
        da.fit(input);
    }

    @Test
    public void renderActivation() throws Exception {
        MnistDataFetcher fetcher = new MnistDataFetcher(true);
        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder().momentum(0.9f)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .iterations(100)
                .learningRate(1e-1f)
                .layer(new org.deeplearning4j.nn.conf.layers.AutoEncoder.Builder()
                        .nIn(784).nOut(600)
                        .corruptionLevel(0.6)
                        .lossFunction(LossFunctions.LossFunction.RMSE_XENT).build())
                .build();


        fetcher.fetch(100);
        DataSet d2 = fetcher.next();

        INDArray input = d2.getFeatureMatrix();
        AutoEncoder da = LayerFactories.getFactory(conf.getLayer()).create(conf, Arrays.asList(new ScoreIterationListener(1),new UpdateActivationIterationListener(1)),0);
        da.setParams(da.params());
        da.fit(input);
    }

    @Test
    public void renderHistogram() throws Exception {
        MnistDataFetcher fetcher = new MnistDataFetcher(true);
        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder().momentum(0.9f)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .iterations(100)
                .learningRate(1e-1f)
                .layer(new org.deeplearning4j.nn.conf.layers.AutoEncoder.Builder()
                        .nIn(784).nOut(600)
                        .corruptionLevel(0.6)
                        .weightInit(WeightInit.XAVIER)
                        .lossFunction(LossFunctions.LossFunction.RMSE_XENT).build())
                .build();


        fetcher.fetch(100);
        DataSet d2 = fetcher.next();

        INDArray input = d2.getFeatureMatrix();
        AutoEncoder da = LayerFactories.getFactory(conf.getLayer()).create(conf);
        da.setListeners(new ScoreIterationListener(1),new HistogramIterationListener(5));
        da.setParams(da.params());
        da.fit(input);
    }

    @Test
    public void renderHistogram2() throws Exception {
        MnistDataFetcher fetcher = new MnistDataFetcher(true);
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().momentum(0.9f)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .iterations(1000)
                .learningRate(1e-1f)
                .list(2)
                .layer(0, new org.deeplearning4j.nn.conf.layers.DenseLayer.Builder()
                        .nIn(784).nOut(100)
                        .weightInit(WeightInit.XAVIER).build())
                .layer(1, new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder()
                        .lossFunction(LossFunctions.LossFunction.MCXENT)
                        .nIn(100).nOut(10).build())
                .pretrain(false).backprop(true)
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        net.setListeners(Arrays.<IterationListener>asList(new ScoreIterationListener(1),new HistogramIterationListener(1,true,"myweightpath")));

        fetcher.fetch(100);
        DataSet d2 = fetcher.next();
        net.fit(d2);
    }

}
