package org.deeplearning4j.ui.play;

import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.ui.storage.mapdb.MapDBStatsStorage;
import org.junit.Test;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.Properties;

/**
 * Created by Alex on 08/10/2016.
 */
public class TestPlayUI {

    @Test
    public void testUI() throws Exception {

        Properties p = Nd4j.getExecutioner().getEnvironmentInformation();

        for(Object o : p.keySet()){
            System.out.println(o + "\t" + p.get(o));
        }
//
//        StatsStorage ss = new MapDBStatsStorage();  //In-memory
//
//        UIServer uiServer = UIServer.getInstance();
//        uiServer.attach(ss);
//
////        System.out.println("TITLE: " + Messages.get("home.title"));
////        System.out.println("TITLE EN: " + Messages.get(new Lang(Lang.forCode("en")),"home.title"));
////        System.out.println("TITLE JP: " + Messages.get(new Lang(Lang.forCode("jp")),"home.title"));
//
//        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
//                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
//                .list()
//                .layer(0, new OutputLayer.Builder().lossFunction(LossFunctions.LossFunction.MCXENT).activation("softmax").nIn(4).nOut(3).build())
//                .pretrain(false).backprop(true).build();
//
//        MultiLayerNetwork net = new MultiLayerNetwork(conf);
//        net.init();
//        net.setListeners(new StatsListener(ss), new ScoreIterationListener(1));
//
//        DataSetIterator iter = new IrisDataSetIterator(150,150);
//
//        for( int i=0; i<10; i++ ){
//            net.fit(iter);
//            Thread.sleep(1000);
//        }
//
//
//
//
//        Thread.sleep(100000);


    }

}
