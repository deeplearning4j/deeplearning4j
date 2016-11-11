package org.deeplearning4j.ui.play;

import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.flow.beans.LayerInfo;
import org.deeplearning4j.ui.flow.beans.ModelInfo;
import org.deeplearning4j.ui.module.train.TrainModuleUtils;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.ui.storage.mapdb.MapDBStatsStorage;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.List;

/**
 * Created by Alex on 08/10/2016.
 */
@Ignore
public class TestPlayUI {

    @Test @Ignore
    public void testUI() throws Exception {

        StatsStorage ss = new InMemoryStatsStorage();

        UIServer uiServer = UIServer.getInstance();
        uiServer.attach(ss);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
                .list()
                .layer(0, new DenseLayer.Builder().activation("tanh").nIn(4).nOut(4).build())
                .layer(1, new OutputLayer.Builder().lossFunction(LossFunctions.LossFunction.MCXENT).activation("softmax").nIn(4).nOut(3).build())
                .pretrain(false).backprop(true).build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.setListeners(new StatsListener(ss), new ScoreIterationListener(1));

        DataSetIterator iter = new IrisDataSetIterator(150,150);

        for( int i=0; i<500; i++ ){
            net.fit(iter);
//            Thread.sleep(100);
            Thread.sleep(100);
        }


        Thread.sleep(100000);
    }


    @Test @Ignore
    public void testUIMultipleSessions() throws Exception {

        for( int session=0; session<3; session++ ) {

            StatsStorage ss = new InMemoryStatsStorage();

            UIServer uiServer = UIServer.getInstance();
            uiServer.attach(ss);

            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
                    .list()
                    .layer(0, new DenseLayer.Builder().activation("tanh").nIn(4).nOut(4).build())
                    .layer(1, new OutputLayer.Builder().lossFunction(LossFunctions.LossFunction.MCXENT).activation("softmax").nIn(4).nOut(3).build())
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

    @Test @Ignore
    public void testUICompGraph() throws Exception {

        StatsStorage ss = new InMemoryStatsStorage();

        UIServer uiServer = UIServer.getInstance();
        uiServer.attach(ss);

        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
                .graphBuilder()
                .addInputs("in")
                .addLayer("L0", new DenseLayer.Builder().activation("tanh").nIn(4).nOut(4).build(), "in")
                .addLayer("L1", new OutputLayer.Builder().lossFunction(LossFunctions.LossFunction.MCXENT).activation("softmax").nIn(4).nOut(3).build(), "L0")
                .pretrain(false).backprop(true)
                .setOutputs("L1")
                .build();

        ComputationGraph net = new ComputationGraph(conf);
        net.init();

        net.setListeners(new StatsListener(ss), new ScoreIterationListener(1));

        DataSetIterator iter = new IrisDataSetIterator(150,150);

        for( int i=0; i<100; i++ ){
            net.fit(iter);
            Thread.sleep(100);
        }

        Thread.sleep(100000);
    }

}
