package org.deeplearning4j.spark.ui;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.deeplearning4j.api.storage.Persistable;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.spark.BaseSparkTest;
import org.deeplearning4j.spark.api.TrainingMaster;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.ui.storage.mapdb.MapDBStatsStorage;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.Collections;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Created by Alex on 07/11/2016.
 */
@Ignore
public class TestSparkUI extends BaseSparkTest {


    @Test
    public void testSparkUI() throws Exception {

        JavaSparkContext sc = getContext();

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(123)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .iterations(1)
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(4).nOut(100)
                        .weightInit(WeightInit.XAVIER)
                        .activation("relu")
                        .build())
                .layer(1, new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .nIn(100).nOut(3)
                        .activation("softmax")
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .pretrain(false).backprop(true)
                .build();



        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.init();


        TrainingMaster tm = new ParameterAveragingTrainingMaster.Builder(1)
                .batchSizePerWorker(5)
                .averagingFrequency(4)
                .build();

        SparkDl4jMultiLayer net = new SparkDl4jMultiLayer(sc,conf,tm);
        StatsStorage ss = new InMemoryStatsStorage();

        net.setListeners(ss, Collections.singletonList(new StatsListener(null)));

        UIServer.getInstance().attach(ss);

        List<DataSet> list = new IrisDataSetIterator(150,150).next().asList();
        //120 examples, 4 executors, 30 examples per executor -> 6 updates of size 5 per executor

        JavaRDD<DataSet> rdd = sc.parallelize(list);

        int nEpochs = 5;
        for(int i=0; i<nEpochs; i++ ) {
            net.fit(rdd);
            Thread.sleep(1000);
        }



    }


}
