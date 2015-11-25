package org.deeplearning4j.nn.updater;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.api.Updater;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.junit.Test;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

public class TestUpdaterAggregation {

    @Test
    public void test(){

        Updater[] updaters = new Updater[]{
                new AdaDeltaUpdater(),
                new AdaGradUpdater(),
                new AdamUpdater(),
                new NesterovsUpdater(),
                new NoOpUpdater(),
                new RmsPropUpdater(),
                new SgdUpdater(),
        };

        org.deeplearning4j.nn.conf.Updater[] arr = new org.deeplearning4j.nn.conf.Updater[]{
                org.deeplearning4j.nn.conf.Updater.ADADELTA,
                org.deeplearning4j.nn.conf.Updater.ADAGRAD,
                org.deeplearning4j.nn.conf.Updater.ADAM,
                org.deeplearning4j.nn.conf.Updater.NESTEROVS,
                org.deeplearning4j.nn.conf.Updater.NONE,
                org.deeplearning4j.nn.conf.Updater.RMSPROP,
                org.deeplearning4j.nn.conf.Updater.SGD
        };

        DataSet dsTemp = new DataSet(Nd4j.rand(5,10), Nd4j.rand(5, 10));

        for(int i=0; i<updaters.length; i++ ){

            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                    .iterations(1)
                    .updater(arr[i])
                    .list(2)
                    .layer(0,new DenseLayer.Builder().nIn(10).nOut(10).build())
                    .layer(1,new OutputLayer.Builder().nIn(10).nOut(10).build())
                    .backprop(true).pretrain(false).build();

            MultiLayerNetwork net = new MultiLayerNetwork(conf);
            net.init();

            net.fit(dsTemp);

            Updater updater = net.getUpdater();

            System.out.println(i);
            assertNotNull(updater);
            assertTrue(updater instanceof MultiLayerUpdater);

        }


    }


}
