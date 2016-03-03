package org.deeplearning4j.ui;

import org.deeplearning4j.datasets.fetchers.MnistDataFetcher;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.preprocessor.CnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToCnnPreProcessor;
import org.deeplearning4j.nn.graph.ComputationGraph;
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
                .list()
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

    @Test
    public void testHistogramComputationGraph() throws Exception {
        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .graphBuilder()
                .addInputs("input")
                .addLayer("cnn1", new ConvolutionLayer.Builder(2,2).stride(2, 2).nIn(1).nOut(3).build(), "input")
                .addLayer("cnn2", new ConvolutionLayer.Builder(4, 4).stride(2, 2).padding(1, 1).nIn(1).nOut(3).build(), "input")
                .addLayer("max1", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2).build(), "cnn1", "cnn2")
                .addLayer("output", new OutputLayer.Builder().nIn(7 * 7 * 6).nOut(10).build(), "max1")
                .setOutputs("output")
                .inputPreProcessor("cnn1", new FeedForwardToCnnPreProcessor(28, 28, 1))
                .inputPreProcessor("cnn2", new FeedForwardToCnnPreProcessor(28, 28, 1))
                .inputPreProcessor("output", new CnnToFeedForwardPreProcessor(7, 7, 6))
                .pretrain(false).backprop(true)
                .build();

        ComputationGraph graph = new ComputationGraph(conf);
        graph.init();

        graph.setListeners(new HistogramIterationListener(1), new ScoreIterationListener(1));

        DataSetIterator mnist = new MnistDataSetIterator(32,640,false,true,false,12345);

        graph.fit(mnist);
    }

    @Test
    public void testHistogramComputationGraphUnderscoresInName() throws Exception {
        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .graphBuilder()
                .addInputs("input")
                .setInputTypes(InputType.convolutional(1,28,28))
                .addLayer("cnn_1", new ConvolutionLayer.Builder(2,2).stride(2, 2).nIn(1).nOut(3).build(), "input")
                .addLayer("cnn_2", new ConvolutionLayer.Builder(4,4).stride(2,2).padding(1,1).nIn(1).nOut(3).build(), "input")
                .addLayer("max_1", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2).build(), "cnn_1", "cnn_2")
                .addLayer("output", new OutputLayer.Builder().nIn(7 * 7 * 6).nOut(10).build(), "max_1")
                .setOutputs("output")
                .pretrain(false).backprop(true)
                .build();

        ComputationGraph graph = new ComputationGraph(conf);
        graph.init();

        graph.setListeners(new HistogramIterationListener(1), new ScoreIterationListener(1));

        DataSetIterator mnist = new MnistDataSetIterator(32,640,false,true,false,12345);

        graph.fit(mnist);
    }

}
