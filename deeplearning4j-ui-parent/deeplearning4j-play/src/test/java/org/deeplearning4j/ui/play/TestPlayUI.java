package org.deeplearning4j.ui.play;

import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.variational.GaussianReconstructionDistribution;
import org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import static org.junit.Assert.assertEquals;

/**
 * Created by Alex on 08/10/2016.
 */
@Ignore
public class TestPlayUI {

    @Test
    @Ignore
    public void testUI() throws Exception {

        StatsStorage ss = new InMemoryStatsStorage();

        PlayUIServer uiServer = (PlayUIServer) UIServer.getInstance();
        assertEquals(9000, uiServer.getPort());
        uiServer.stop();
        PlayUIServer playUIServer = new PlayUIServer();
        playUIServer.runMain(new String[] {"--uiPort", "9100", "-r", "true"});

        assertEquals(9100, playUIServer.getPort());
        playUIServer.stop();


        //        uiServer.attach(ss);
        //
        //        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
        //                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
        //                .list()
        //                .layer(0, new DenseLayer.Builder().activation(Activation.TANH).nIn(4).nOut(4).build())
        //                .layer(1, new OutputLayer.Builder().lossFunction(LossFunctions.LossFunction.MCXENT).activation(Activation.SOFTMAX).nIn(4).nOut(3).build())
        //                .pretrain(false).backprop(true).build();
        //
        //        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        //        net.init();
        //        net.setListeners(new StatsListener(ss, 3), new ScoreIterationListener(1));
        //
        //        DataSetIterator iter = new IrisDataSetIterator(150, 150);
        //
        //        for (int i = 0; i < 500; i++) {
        //            net.fit(iter);
        ////            Thread.sleep(100);
        //            Thread.sleep(100);
        //        }
        //
        ////        uiServer.stop();

        Thread.sleep(100000);
    }

    @Test
    @Ignore
    public void testUI_VAE() throws Exception {
        //Variational autoencoder - for unsupervised layerwise pretraining

        StatsStorage ss = new InMemoryStatsStorage();

        UIServer uiServer = UIServer.getInstance();
        uiServer.attach(ss);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                        .updater(new Sgd(1e-5))
                        .list().layer(0,
                                        new VariationalAutoencoder.Builder().nIn(4).nOut(3).encoderLayerSizes(10, 11)
                                                        .decoderLayerSizes(12, 13).weightInit(WeightInit.XAVIER)
                                                        .pzxActivationFunction(Activation.IDENTITY)
                                                        .reconstructionDistribution(
                                                                        new GaussianReconstructionDistribution())
                                                        .activation(Activation.LEAKYRELU).build())
                        .layer(1, new VariationalAutoencoder.Builder().nIn(3).nOut(3).encoderLayerSizes(7)
                                        .decoderLayerSizes(8).weightInit(WeightInit.XAVIER)
                                        .pzxActivationFunction(Activation.IDENTITY)
                                        .reconstructionDistribution(new GaussianReconstructionDistribution())
                                        .activation(Activation.LEAKYRELU).build())
                        .layer(2, new OutputLayer.Builder().nIn(3).nOut(3).build()).pretrain(true).backprop(true)
                        .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.setListeners(new StatsListener(ss), new ScoreIterationListener(1));

        DataSetIterator iter = new IrisDataSetIterator(150, 150);

        for (int i = 0; i < 50; i++) {
            net.fit(iter);
            Thread.sleep(100);
        }


        Thread.sleep(100000);
    }


    @Test
    @Ignore
    public void testUIMultipleSessions() throws Exception {

        for (int session = 0; session < 3; session++) {

            StatsStorage ss = new InMemoryStatsStorage();

            UIServer uiServer = UIServer.getInstance();
            uiServer.attach(ss);

            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).list()
                            .layer(0, new DenseLayer.Builder().activation(Activation.TANH).nIn(4).nOut(4).build())
                            .layer(1, new OutputLayer.Builder().lossFunction(LossFunctions.LossFunction.MCXENT)
                                            .activation(Activation.SOFTMAX).nIn(4).nOut(3).build())
                            .pretrain(false).backprop(true).build();

            MultiLayerNetwork net = new MultiLayerNetwork(conf);
            net.init();
            net.setListeners(new StatsListener(ss), new ScoreIterationListener(1));

            DataSetIterator iter = new IrisDataSetIterator(150, 150);

            for (int i = 0; i < 20; i++) {
                net.fit(iter);
                Thread.sleep(100);
            }
        }


        Thread.sleep(1000000);
    }

    @Test
    @Ignore
    public void testUICompGraph() throws Exception {

        StatsStorage ss = new InMemoryStatsStorage();

        UIServer uiServer = UIServer.getInstance();
        uiServer.attach(ss);

        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder().graphBuilder().addInputs("in")
                        .addLayer("L0", new DenseLayer.Builder().activation(Activation.TANH).nIn(4).nOut(4).build(),
                                        "in")
                        .addLayer("L1", new OutputLayer.Builder().lossFunction(LossFunctions.LossFunction.MCXENT)
                                        .activation(Activation.SOFTMAX).nIn(4).nOut(3).build(), "L0")
                        .pretrain(false).backprop(true).setOutputs("L1").build();

        ComputationGraph net = new ComputationGraph(conf);
        net.init();

        net.setListeners(new StatsListener(ss), new ScoreIterationListener(1));

        DataSetIterator iter = new IrisDataSetIterator(150, 150);

        for (int i = 0; i < 100; i++) {
            net.fit(iter);
            Thread.sleep(100);
        }

        Thread.sleep(100000);
    }

}
